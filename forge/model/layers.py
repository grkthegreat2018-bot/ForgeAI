"""Layer modules: RMSNorm, RoPE, GQA, DoubleGatedConv, SwiGLU, ModularBlock."""
import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.config import ModelConfig
from forge.keys._tensor_utils import _repeat_kv as _repeat_kv_shared
from forge.keys._tensor_utils import _rotate_half

logger = logging.getLogger(__name__)

from .attention_ops import _causal_mask, flash_attention, varlen_attention
from .kv_cache import KVCache, PreAllocatedKVCache


class RMSNorm(nn.Module):
    """RMSNorm — faster than LayerNorm, no mean subtraction or bias.

    Uses torch.nn.functional.rms_norm (available in PyTorch 2.4+) for a single
    fused kernel with fp32 internal accumulation. This replaces the manual
    x.float().pow(2).mean() → rsqrt → x.float() * norm → .to(dtype) * weight
    chain that launched 5+ kernels and allocated fp32 temporaries per call.
    With 113 norm calls per forward (57 ln + 56 qk_norm), this saves ~200
    kernel launches and ~50MB of temporary allocations per forward pass.

    R&D round 14: when use_triton_kernels=True in config, uses a custom
    Triton fused RMSNorm kernel (Liger-Kernel-style) tuned for SM120.
    Falls back to F.rms_norm on CPU or when Triton is unavailable.
    """

    def __init__(self, d_model, eps=1e-6, use_triton: bool = False,
                 zero_centered: bool = False):
        super().__init__()
        # Zero-centered reparam (Qwen3-Next, d2): the stored parameter is
        # gamma-1 and the forward adds 1 back — identical function, but
        # weight decay now pulls toward the IDENTITY scale instead of
        # zeroing the channel gains during hone.
        self._zero_centered = zero_centered
        init = torch.zeros(d_model) if zero_centered else torch.ones(d_model)
        self.weight = nn.Parameter(init)
        self.eps = eps
        self.normalized_shape = [d_model]
        self._use_triton = use_triton

    def _gamma(self):
        return (1 + self.weight) if self._zero_centered else self.weight

    def forward(self, x):
        w = self._gamma()
        if self._use_triton:
            from forge.decoding.triton_train_kernels import triton_rms_norm
            return triton_rms_norm(x, w, self.eps)
        return F.rms_norm(x, self.normalized_shape, w, self.eps)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) for query/key tensors.

    Supports optional YaRN scaling (Peng et al. 2023) for context extension.
    YaRN non-uniformly interpolates RoPE frequencies: high-freq bands extrapolate
    unchanged, low-freq bands are linearly interpolated, and a smooth ramp
    (controlled by beta_fast/beta_slow) blends the two zones.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0, rope_scaling=None,
                 partial_rotary_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_scaling = rope_scaling
        # LFM3 zero-tail partial rotation: the first
        # partial_rotary_factor*dim//2 frequency entries keep the full-dim
        # progression and the tail is zeroed → cos=1/sin=0 there, leaving
        # those head dims unrotated (NOT a shrunk rotary dimension).
        self.partial_rotary_factor = partial_rotary_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if partial_rotary_factor < 1.0:
            rope_angles = int(partial_rotary_factor * dim) // 2
            inv_freq[rope_angles:] = 0.0

        if rope_scaling:
            rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
            if rope_type == "yarn":
                inv_freq = self._yarn_inv_freq(inv_freq, rope_scaling, max_seq_len)
            elif rope_type in ("llama3", "llama3.1"):
                inv_freq = self._llama3_inv_freq(inv_freq, rope_scaling)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        # Pre-compute bf16 versions to avoid .to(x.dtype) per forward call.
        # 56 calls/forward (28 layers × Q+K), each saving a dtype conversion kernel.
        self.register_buffer("cos_cached_bf16", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached_bf16", emb.sin().to(torch.bfloat16), persistent=False)

    @staticmethod
    def _yarn_inv_freq(inv_freq, cfg, max_seq_len):
        """Apply YaRN wavelength-aware scaling to inv_freq."""
        factor = cfg.get("factor", 4.0)
        orig_len = cfg.get("original_max_position_embeddings", max_seq_len // factor)
        beta_fast = cfg.get("beta_fast", 32.0)
        beta_slow = cfg.get("beta_slow", 1.0)

        def _gamma_fn(x):
            # Smooth ramp from 0 to 1 using tanh.
            return 1.0 - torch.tanh(x * math.pi / 2.0)

        # Wavelengths for each freq band: lambda_i = 2*pi / inv_freq_i
        # YaRN defines low/high freq w.r.t. the original context length.
        low_freq_wavelen = orig_len / beta_fast
        high_freq_wavelen = orig_len * 2.0 / beta_slow
        wavelens = 2.0 * math.pi / inv_freq

        # Bands: extrapolate (high freq), interpolate (low freq), ramp (middle).
        # inv_freq_scaled = inv_freq / scale, where scale=1 for high freq, =factor for low.
        scale = torch.ones_like(inv_freq)
        # Smooth ramp factor across the middle zone.
        x = (inv_freq * low_freq_wavelen - 1.0) / (high_freq_wavelen / low_freq_wavelen - 1.0)
        x = x.clamp(0.0, 1.0)
        ramp = _gamma_fn(x)
        scale = ramp * factor + (1.0 - ramp) * 1.0
        # Bands fully above high_freq_wavelen get full interpolation.
        scale = torch.where(wavelens > high_freq_wavelen, torch.full_like(scale, factor), scale)
        # Bands fully below low_freq_wavelen extrapolate (scale=1).
        scale = torch.where(wavelens < low_freq_wavelen, torch.ones_like(scale), scale)
        return inv_freq / scale

    @staticmethod
    def _llama3_inv_freq(inv_freq, cfg):
        """Apply Llama-3/3.1 wavelength-aware RoPE scaling (HF _compute_llama3).

        Low-frequency bands (long wavelengths) are divided by `factor`,
        high-frequency bands pass through unchanged, and a linear ramp blends
        the boundary — matching transformers.models.llama.modeling_llama.
        """
        factor = cfg.get("factor", 8.0)
        low_f = cfg.get("low_freq_factor", 1.0)
        high_f = cfg.get("high_freq_factor", 4.0)
        orig_len = cfg.get("original_max_position_embeddings", 8192)

        low_wl = orig_len / low_f
        high_wl = orig_len / high_f
        wavelens = 2.0 * math.pi / inv_freq

        scaled = torch.where(wavelens > low_wl, inv_freq / factor, inv_freq)
        smooth = (orig_len / wavelens - low_f) / (high_f - low_f)
        smoothed = (1.0 - smooth) * inv_freq / factor + smooth * inv_freq
        mid = (wavelens <= low_wl) & (wavelens >= high_wl)
        return torch.where(mid, smoothed, scaled)

    _rotate_half = staticmethod(_rotate_half)

    def forward(self, x: torch.Tensor, offset: int = 0,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        # x shape: (B, n_heads, seq_len, head_dim)
        seq_len = x.shape[-2]
        if position_ids is not None:
            # Per-sequence positions (for batched left-padded generation).
            # position_ids: (B, seq_len) — each sequence has its own position indices.
            if x.dtype == torch.bfloat16:
                cos = self.cos_cached_bf16[position_ids]  # (B, seq_len, dim)
                sin = self.sin_cached_bf16[position_ids]
            else:
                cos = self.cos_cached[position_ids].to(x.dtype)
                sin = self.sin_cached[position_ids].to(x.dtype)
            # Broadcast: (B, seq_len, dim) -> (B, 1, seq_len, dim) for x (B, n_heads, seq_len, dim)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            return (x * cos) + (self._rotate_half(x) * sin)
        # Scalar offset (original path — single sequence or uniform batch).
        if x.dtype == torch.bfloat16:
            cos = self.cos_cached_bf16[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached_bf16[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0).to(x.dtype)
            sin = self.sin_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0).to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    """GQA: n_heads query heads share n_kv_heads KV heads (saves KV cache memory).

    With use_qk_norm=True, applies RMSNorm on Q and K per-head (head_dim)
    after projection, before RoPE — matching LFM2's q_layernorm/k_layernorm.
    Identity init (all 1.0) = no-op, lossless at start.
    """

    def __init__(self, d_model=768, n_heads=12, n_kv_heads=None, max_seq_len=2048, base=10000.0, rope_scaling=None,
                 use_qk_norm=False, attn_scale=None, attn_bias=False, qk_norm_eps=1e-6,
                 use_rope=True, head_dim: int | None = None,
                 sliding_window: int | None = None,
                 use_out_gate: bool = False,
                 rope_partial_factor: float = 1.0,
                 zero_centered_norm: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads  # default to MHA
        # Explicit head_dim supports archs that decouple it from d_model //
        # n_heads (Qwen3-4B: 2560/32=80 vs real head_dim 128).
        self.head_dim = head_dim or (d_model // n_heads)
        self.n_rep = n_heads // self.n_kv_heads
        self._qk_norm_eps = qk_norm_eps
        self.use_rope = use_rope
        # LFM3: sliding-window attention (token sees itself + last W-1 keys)
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=attn_bias)
        # o_proj input is n_heads*head_dim (== d_model only when head_dim
        # divides evenly); never has bias (Qwen2 convention).
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        # LFM3 gated attention: per-head sigmoid gate on the attention
        # output before out_proj (gate_proj: d_model -> n_heads).
        self.gate_proj = (nn.Linear(d_model, n_heads, bias=False)
                          if use_out_gate else None)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=base, rope_scaling=rope_scaling,
                                    partial_rotary_factor=rope_partial_factor) if use_rope else None

        # QK-norm: RMSNorm on Q and K before RoPE (LFM2 / Gemma3 / Qwen3 style).
        # When weights are identity (all 1.0), skip — it's a no-op.
        self.use_qk_norm = use_qk_norm
        self._qk_norm_identity = True  # assume identity until weights loaded
        if use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, eps=getattr(self, '_qk_norm_eps', 1e-6),
                zero_centered=zero_centered_norm)
            self.k_norm = RMSNorm(
                self.head_dim, eps=getattr(self, '_qk_norm_eps', 1e-6),
                zero_centered=zero_centered_norm)

        # ValueResidual: V_0 from layer 0 is injected by the parent model.
        # When set, v = v + gate * v0 (applied before attention).
        self._v0_residual: torch.Tensor | None = None
        self._v0_gate: torch.Tensor | None = None
        self._v0_capture: torch.Tensor | None = None  # set by parent for layer 0

        # LearnedSink (GPT-OSS): per-head attention sink bias.
        # init=0 → lossless (no bias added). Training learns sink values.
        self.sinks: nn.Parameter | None = None

        # QK-Clip monitor (Kimi K2 MuonClip): set by QKClipMonitor.attach().
        # None = disabled (zero overhead).
        self._qk_clip_monitor = None
        self._qk_clip_layer_idx = 0

    def _repeat_kv(self, x):
        """Repeat KV heads to match query heads."""
        return _repeat_kv_shared(x, self.n_rep)

    def forward(self, x, past_key_value=None, use_cache=False,
                preallocated_cache: Optional["PreAllocatedKVCache"] = None, layer_idx: int = 0,
                attention_bias: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                cu_seqlens: torch.Tensor | None = None):
        B, T, _C = x.shape
        # Fused path: one GEMM emits q/k/v (+ gate logits for LFM3 out-
        # gate). Rows are packed from the loaded weights by the loader.
        gate_logits = None
        if getattr(self, '_fused_qkv', False):
            parts = self.qkv_proj(x)
            if self.qkv_proj.g_dim:
                q_raw, k_raw, v_raw, gate_logits = parts
            else:
                q_raw, k_raw, v_raw = parts
        else:
            q_raw = self.q_proj(x)
            k_raw = self.k_proj(x)
            v_raw = self.v_proj(x)
        q = q_raw.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_raw.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v_raw.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ValueResidual (ResFormer): add V_0 from layer 0 to this layer's V.
        # gate=0 at init → v unchanged (lossless). Training opens the gate.
        if self._v0_residual is not None and self._v0_gate is not None:
            gate_val = self._v0_gate
            if gate_val.item() != 0.0:
                # V_0 is (B, n_kv_heads, T, head_dim) — same shape as v.
                v = v + gate_val * self._v0_residual
        # Capture V_0 from layer 0 (set by parent model for the first layer).
        if self._v0_capture is not None:
            self._v0_capture = v.detach()

        # Fused QK-Norm + RoPE path (opt-in via FORGE_FUSED_ROPE_QKNORM=1).
        # Fuses RMSNorm and RoPE into a single Triton kernel, halving HBM
        # traffic for Q/K preprocessing. Only used when QK-norm is active
        # (non-identity weights) and position_ids is None (offset path).
        _use_fused = (
            os.environ.get("FORGE_FUSED_ROPE_QKNORM", "0") == "1"
            and self.use_qk_norm and not self._qk_norm_identity
            and position_ids is None and q.is_cuda
            and self.use_rope
        )
        if _use_fused:
            from forge.decoding.fused_rope_qknorm import fused_qk_norm_rope
            if preallocated_cache is not None:
                past_len = preallocated_cache.position
            else:
                past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
            # Slice cos/sin tables to the current sequence positions
            cos_slice = self.rope.cos_cached[past_len:past_len + T, :].to(q.dtype)
            sin_slice = self.rope.sin_cached[past_len:past_len + T, :].to(q.dtype)
            q, k = fused_qk_norm_rope(
                q, k, self.q_norm.weight, self.k_norm.weight,
                cos_slice, sin_slice, eps=self.q_norm.eps)
        else:
            # QK-norm: normalize Q and K per-head before RoPE (LFM2 style).
            # When weights are identity (all 1.0), skip — no-op.
            if self.use_qk_norm and not self._qk_norm_identity:
                q = self.q_norm(q)
                k = self.k_norm(k)

            # RoPE: skip when use_rope=False (Jamba attention has no RoPE)
            if self.use_rope:
                # Pre-allocated cache path: O(1) append, no torch.cat.
                if preallocated_cache is not None:
                    past_len = preallocated_cache.position
                    q = self.rope(q, offset=past_len, position_ids=position_ids)
                    k = self.rope(k, offset=past_len, position_ids=position_ids)
                else:
                    past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
                    q = self.rope(q, offset=past_len, position_ids=position_ids)
                    k = self.rope(k, offset=past_len, position_ids=position_ids)

        # Cache append + KV retrieval (same for both paths)
        if preallocated_cache is not None:
            if preallocated_cache.pos_t is not None and T == 1:
                # Graph-safe: device-indexed write + full-buffer read.
                # Future slots stay masked by attention_bias — required
                # for CUDA-graph decode capture (see PreAllocatedKVCache).
                preallocated_cache.append_t(layer_idx, k, v)
                k, v = preallocated_cache.kv_view(layer_idx)
            else:
                ring_c = preallocated_cache.ring_cap(layer_idx)
                if ring_c is not None:
                    # This forward needs the last (C-1)+T keys — more than
                    # the C-slot buffer. Read the held window in position
                    # order, then cat the new (just-computed) keys; the
                    # append below retains only the last C for later steps.
                    n_past = min(past_len, ring_c)
                    if n_past and past_len >= ring_c:
                        s = past_len % ring_c
                        pk = torch.cat([preallocated_cache.k_caches[layer_idx][:, :, s:],
                                        preallocated_cache.k_caches[layer_idx][:, :, :s]], dim=2)
                        pv = torch.cat([preallocated_cache.v_caches[layer_idx][:, :, s:],
                                        preallocated_cache.v_caches[layer_idx][:, :, :s]], dim=2)
                    elif n_past:
                        pk = preallocated_cache.k_caches[layer_idx][:, :, :n_past]
                        pv = preallocated_cache.v_caches[layer_idx][:, :, :n_past]
                    else:
                        pk = pv = None
                    preallocated_cache.append(layer_idx, k, v)
                    if pk is not None:
                        k = torch.cat([pk, k], dim=2)
                        v = torch.cat([pv, v], dim=2)
                else:
                    preallocated_cache.append(layer_idx, k, v)
                    k = preallocated_cache.k_caches[layer_idx][:, :, :past_len + T]
                    v = preallocated_cache.v_caches[layer_idx][:, :, :past_len + T]
        elif past_key_value is not None and not _use_fused:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        new_kv = (k, v) if use_cache else None

        # Repeat KV heads to match Q heads.
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # QK-Clip monitoring (training-only, opt-in): record per-head max |logit|.
        if self._qk_clip_monitor is not None and self.training:
            self._qk_clip_monitor.observe(self._qk_clip_layer_idx, q, k)

        # Varlen attention path (R&D round 14): for packed sequences with
        # cu_seqlens, use varlen attention to prevent cross-example
        # contamination. Only during training (no KV cache, no attention_bias).
        if cu_seqlens is not None and not use_cache and attention_bias is None:
            out = varlen_attention(q, k, v, cu_seqlens)
            return self._finish_attn(x, out, gate_logits), new_kv

        # Single-token decode with cached KV: no causal mask needed (all keys are valid)
        total_len = k.shape[-2]
        # Cache length before this forward's tokens (for SWA window mask).
        if preallocated_cache is not None:
            _past_len = preallocated_cache.position
        elif past_key_value is not None:
            _past_len = past_key_value[0].shape[-2]
        else:
            _past_len = total_len - T
        # LFM3 sliding-window attention: key j is visible to query i iff
        # j <= i and i - j < sliding_window (self + W-1 previous keys).
        # Combines additively with any incoming attention_bias (pad masks
        # and pos_t future-slot masks keep working).
        if self.sliding_window is not None:
            ring_c = (preallocated_cache.ring_cap(layer_idx)
                      if preallocated_cache is not None else None)
            pos_t_path = (preallocated_cache is not None
                          and preallocated_cache.pos_t is not None and T == 1)
            if pos_t_path:
                # pos_t (CUDA-graph decode): position lives on-device —
                # bake the mask from the live tensor so each replay sees
                # the CURRENT position, not the capture-time int.
                q_pos = preallocated_cache.pos_t.reshape(1)
            else:
                q_pos = _past_len + torch.arange(T, device=q.device)
            if pos_t_path and ring_c is not None:
                # Ring buffer: every resident key is in-window by
                # construction. Slots > q_pos are stale only before the
                # ring fills (slot == position while pos < C).
                k_pos = torch.arange(total_len, device=q.device)
                allow = k_pos[None, :] <= q_pos.clamp(max=ring_c - 1)[:, None]
                # The shared bucket bias encodes POSITION-based validity —
                # wrong (and wrong-sized) for a slot-indexed ring buffer.
                attention_bias = None
            else:
                # Position-ordered read: the returned keys span positions
                # P-total_len..P-1 for both linear buffers (P=total_len)
                # and wrapped rings (last-total_len positions).
                k_pos = torch.arange(_past_len + T - total_len,
                                     _past_len + T, device=q.device)
                allow = (k_pos[None, :] <= q_pos[:, None]) & \
                        ((q_pos[:, None] - k_pos[None, :]) < self.sliding_window)
            swa = torch.zeros(T, total_len, device=q.device, dtype=q.dtype)
            swa = swa.masked_fill(~allow, float('-inf')).view(1, 1, T, total_len)
            if attention_bias is not None:
                if attention_bias.shape[-1] != total_len:
                    # Position-ordered ring read ↔ last-total_len columns of
                    # a positional bias.
                    attention_bias = attention_bias[..., -total_len:]
                swa = swa + attention_bias
            if self.sinks is not None and self.sinks.abs().max().item() != 0.0:
                swa = swa + self.sinks.view(1, self.n_heads, 1, 1).to(swa.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=swa)
            return self._finish_attn(x, out, gate_logits), new_kv
        # CSA (Compressed Sparse Attention): for long sequences, use top-k
        # position selection to reduce O(S^2) → O(S*k). Short sequences
        # use full attention (lossless when S <= top_k).
        csa_top_k = getattr(self, '_csa_top_k', 0)
        if getattr(self, '_csa_enabled', False) and csa_top_k > 0 and total_len > csa_top_k and T > 1:
            from forge.keys.attention.csa_key import CSAAttention
            csa = CSAAttention(
                self.d_model,
                self.n_heads, top_k=csa_top_k,
                head_dim=self.head_dim)
            out = csa(q, k, v, is_causal=True)
            return self._finish_attn(x, out, gate_logits), new_kv
        # LearnedSink: add per-head sink bias to attention scores.
        # sink is (n_heads,) → broadcast to (1, n_heads, 1, total_len).
        # init=0 → no-op (lossless). Training learns positive sink values.
        sink_bias = None
        if self.sinks is not None and self.sinks.abs().max().item() != 0.0:
            sink_bias = self.sinks.view(1, self.n_heads, 1, 1).expand(1, self.n_heads, 1, total_len)
        if attention_bias is not None:
            if sink_bias is not None:
                attention_bias = attention_bias + sink_bias.to(attention_bias.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        elif sink_bias is not None:
            # Build a combined bias: causal + sink
            if T == 1 and total_len > 1:
                bias = sink_bias.to(q.dtype).expand(B, self.n_heads, 1, total_len)
            else:
                causal = _causal_mask(T, total_len, total_len - T,
                                      q.device, q.dtype)
                bias = causal + sink_bias.to(q.dtype).expand(B, self.n_heads, T, total_len)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        elif T == 1 and total_len > 1:
            out = flash_attention(q, k, v, is_causal=False)
        elif total_len == T:
            out = flash_attention(q, k, v, is_causal=True)
        else:
            # Continuation forward (T>1 over a cached prefix): SDPA's
            # is_causal anchors the mask TOP-LEFT on non-square shapes,
            # which would hide the entire cached prefix — anchor the
            # causal mask bottom-right via an explicit mask instead.
            mask = _causal_mask(T, total_len, total_len - T,
                                q.device, q.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self._finish_attn(x, out, gate_logits), new_kv

    def _finish_attn(self, x: torch.Tensor, out: torch.Tensor,
                     gate_logits: torch.Tensor | None = None) -> torch.Tensor:
        """(B,H,T,Dh) attn output -> optional per-head sigmoid gate ->
        (B,T,H*Dh) -> out_proj.

        LFM3 gated attention: out = out_proj(attn * sigmoid(gate_proj(x)))
        with one scalar gate per query head, gate read off the block input.
        """
        out = out.transpose(1, 2)                        # (B,T,H,Dh)
        if self.gate_proj is not None:
            lg = gate_logits if gate_logits is not None else self.gate_proj(x)
            out = out * torch.sigmoid(lg).unsqueeze(-1)  # (B,T,H,1)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        return self.out_proj(out)


class DoubleGatedConvLayer(nn.Module):
    """LFM2-style double-gated short convolution layer.

    Matches LiquidAI LFM2 architecture exactly:
      BCx = in_proj(x)       # Linear(d → 3d), split into B, C, x
      Bx = B * x             # input gate (raw multiply, NO sigmoid)
      conv_out = conv(Bx)    # short depthwise causal conv
      out = C * conv_out     # output gate (raw multiply, NO sigmoid)
      out = out_proj(out)    # Linear(d → d)

    Key difference from our earlier (broken) version:
      - NO sigmoid gates (sigmoid squashes to [0,1], killing residual stream norm)
      - Conv applied to GATED input (B*x), not raw input
      - in_proj projects to 3*d_model (more capacity, matches LFM2)

    The conv state is a fixed-size buffer (kernel_size-1 past tokens) enabling
    O(1) per-token generation — no growing KV cache for conv layers.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, bias: bool = False,
                 use_bitnet: bool = False, bitnet_config=None):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        if use_bitnet and bitnet_config is not None:
            from forge.keys.quantization.bitnet_b158_key import build_bitnet_conv1d, build_bitnet_linear
            self.in_proj = build_bitnet_linear(
                bitnet_config, d_model, 3 * d_model, bias=bias)
            self.conv = build_bitnet_conv1d(
                bitnet_config, d_model, d_model, kernel_size,
                groups=d_model, bias=bias)
            self.out_proj = build_bitnet_linear(
                bitnet_config, d_model, d_model, bias=bias)
        else:
            # Input projection: d_model → 3*d_model (splits into B, C, x)
            self.in_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
            # Depthwise causal conv: (d_model, 1, kernel_size), groups=d_model
            self.conv = nn.Conv1d(
                d_model, d_model, kernel_size,
                groups=d_model, bias=bias, padding=0,
            )
            # Output projection (like attention out_proj)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        # Conv state buffer for incremental generation: (B, d_model, kernel_size-1)
        self._conv_state = None
        # Owned backing buffer sized to the largest batch seen — conv
        # state for batch B is the [:B] view. CUDA-graph decode captures
        # read/write THIS storage: the model-level reset only re-points
        # the alias, so replays never chase a reallocated tensor (the old
        # single-buffer scheme freed and realloc'd on every position-0
        # reset — stale reads → OOV garbage, worst at B>8). One shared
        # buffer also keeps torch.compile guards happy — a per-B dict of
        # the same tensor showed up as "duplicate tensor" guard churn.
        self._conv_state_buf = None

    # Max batch slice pre-allocated: covers the static-graph B<=32 cap.
    _CONV_STATE_MAX_B = 32

    def _init_conv_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        """Initialize conv state buffer for incremental decoding.

        Slice [:batch] of the owned backing buffer — CUDA-graph decode
        captures read/write this storage, so re-allocating orphans the
        captured graph's pointer (stale reads → logit drift / OOV tokens
        at replay)."""
        buf = self._conv_state_buf
        if (buf is None or buf.shape[0] < batch
                or buf.device != device or buf.dtype != dtype):
            # Allocate OUTSIDE inference mode: a buffer created under
            # inference_mode is an inference tensor and cannot be
            # mutated by copy_() under plain no_grad (mixed-mode callers
            # exist — engine prefill runs inference_mode, some callers
            # decode under no_grad). Normal tensors mutate fine in both.
            with torch.inference_mode(False):
                self._conv_state_buf = torch.zeros(
                    max(batch, self._CONV_STATE_MAX_B), self.d_model,
                    self.kernel_size - 1, device=device, dtype=dtype)
        else:
            buf[:batch].zero_()
        self._conv_state = self._conv_state_buf[:batch]

    def _causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv via left-padding + Conv1d.

        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        # Transpose to (B, d_model, T) for Conv1d
        x_t = x.transpose(1, 2)  # (B, d_model, T)
        # Left-pad with (kernel_size - 1) zeros for causal conv
        pad = (self.kernel_size - 1, 0)
        x_padded = F.pad(x_t, pad)  # (B, d_model, T + k - 1)
        out = self.conv(x_padded)   # (B, d_model, T)
        return out.transpose(1, 2)  # (B, T, d_model)

    def _incremental_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Single-token conv using state buffer (O(1) per token).

        Args:
            x: (B, 1, d_model) — one token at a time
        Returns:
            (B, 1, d_model)
        """
        B, T, _D = x.shape
        assert T == 1, f"incremental_conv expects T=1, got T={T}"
        x_t = x.transpose(1, 2)  # (B, d_model, 1)

        if (self._conv_state is None
                or self._conv_state.shape[0] != B):
            self._init_conv_state(B, x.device, x.dtype)

        # Concatenate state + new token: (B, d_model, kernel_size)
        window = torch.cat([self._conv_state, x_t], dim=-1)
        # Apply conv (no padding needed — window is exactly kernel_size)
        out = self.conv(window)  # (B, d_model, 1)
        # Update state in place: a fixed buffer survives CUDA-graph
        # capture — reassigning the attr would freeze a stale storage
        # pointer into the replay (and a B-mismatch cats [1,D,k]+[8,D,1]).
        self._conv_state.copy_(window[:, :, 1:])

        return out.transpose(1, 2)  # (B, 1, d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        pad_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """LFM2-style forward: in_proj → gate → conv → gate → out_proj.

        For prefill (T > 1): full causal conv.
        For decode (T == 1): incremental conv with state buffer.
        """
        B, T, D = x.shape

        # Project to 3*d_model and split into B (input gate), C (output gate), x_proj
        BCx = self.in_proj(x)  # (B, T, 3*D)
        B_gate, C_gate, x_proj = BCx.chunk(3, dim=-1)  # each (B, T, D)

        # Input gate: raw multiply (NO sigmoid — LFM2 uses multiplicative gates)
        Bx = B_gate * x_proj  # (B, T, D)

        # LFM3 padding semantics: the HF model zeroes the gated conv input
        # at padding positions (apply_mask_to_padding_states). The mask
        # arrives as an explicit kwarg from ModularBlock (consumed at
        # block level so gradient-checkpoint recomputation sees the same
        # ops); the _pad_mask attr fallback covers direct callers.
        if pad_mask is None:
            pad_mask = getattr(self, '_pad_mask', None)
            self._pad_mask = None
        if pad_mask is not None:
            Bx = Bx * pad_mask.unsqueeze(-1).to(Bx.dtype)

        # Conv state is reset at the MODEL level when starting a new sequence
        # (past_key_values is None at the model level). Per-layer reset based
        # on past_key_value is incorrect because conv layers ALWAYS get
        # past_key_value=None (they don't have KV cache entries), which would
        # wipe the state during every decode step.
        #
        # The _conv_state_reset flag is set by the parent model's forward when
        # starting a new sequence.
        if getattr(self, '_conv_state_reset', False):
            self._conv_state = None
            self._conv_state_reset = False

        # A stored prefix boundary (_conv_state_prefix, from a prefix/session
        # KV cache hit) restores the conv context.  For T==1 it becomes the
        # live state so incremental decode sees the right left context.
        # Restores into the banked buffer (copy_, not reassignment) so any
        # captured CUDA graph keeps reading the same storage.
        prefix_ctx = getattr(self, '_conv_state_prefix', None)
        self._conv_state_prefix = None
        if prefix_ctx is not None and T == 1:
            _buf = self._conv_state_buf
            if (_buf is not None and _buf.shape[0] >= Bx.shape[0]
                    and _buf[0].shape == prefix_ctx.shape[1:]
                    and _buf.dtype == prefix_ctx.dtype):
                _buf[:Bx.shape[0]].copy_(prefix_ctx.to(Bx.device))
                self._conv_state = _buf[:Bx.shape[0]]
            else:
                self._conv_state = prefix_ctx.to(
                    device=Bx.device, dtype=Bx.dtype)

        if T == 1 and self._conv_state is not None:
            conv_out = self._incremental_conv(Bx)
        else:
            # Prefill or no state: full causal conv. When a prefix/session KV
            # hit supplies _conv_state_prefix (the stored gated-input context
            # at the reuse boundary), use it as the left context instead of
            # zero-padding — otherwise the first kernel_size-1 suffix
            # positions get corrupted conv outputs.  A LIVE _conv_state serves
            # the same role for mid-sequence T>1 forwards (chunked prefill,
            # speculative verify): it holds the last kernel_size-1 gated
            # inputs of the committed span, which is exactly the left context
            # a continuation chunk needs.  _conv_state is None at a fresh
            # prefill (the model-level reset flag cleared it), so this only
            # engages on genuine continuations.
            left_ctx = prefix_ctx
            if (left_ctx is None and self._conv_state is not None
                    and self._conv_state.shape[0] == B):
                # Same-batch continuation only — a stale state from a
                # different batch size (switched without model reset)
                # would cat-mismatch; treat as a fresh left context.
                left_ctx = self._conv_state
            if left_ctx is not None and T > 1:
                Bx_t = Bx.transpose(1, 2)  # (B, D, T)
                full = torch.cat(
                    [left_ctx.to(device=Bx_t.device, dtype=Bx_t.dtype),
                     Bx_t], dim=-1)
                conv_out = self.conv(full).transpose(1, 2)  # (B, T, D)
            else:
                conv_out = self._causal_conv(Bx)
            # Initialize state from last (kernel_size - 1) tokens of GATED input.
            # Write in place when a same-shaped buffer exists — reassignment
            # breaks the static-buffer contract CUDA-graph decode replays on.
            if use_cache:
                self._init_conv_state(B, x.device, x.dtype)
                if self.kernel_size - 1 <= T:
                    new_state = Bx[:, -(self.kernel_size - 1):, :].transpose(1, 2)
                else:
                    pad_len = self.kernel_size - 1 - T
                    pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
                    new_state = torch.cat([pad, Bx], dim=1).transpose(1, 2)
                if self._conv_state.shape == new_state.shape:
                    self._conv_state.copy_(new_state)
                else:
                    self._conv_state = new_state.clone()

        # Output gate: raw multiply (NO sigmoid)
        gated = C_gate * conv_out  # (B, T, D)

        out = self.out_proj(gated)
        # Conv layers don't produce KV cache entries
        return out, None


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network.

    With use_clamp=True, uses GPT-OSS clamped SwiGLU: gate clamped to
    [None, limit], up clamped to [-limit, limit], scaled sigmoid (α=1.702),
    and +1 residual on the linear path. Prevents outlier activations.

    R&D round 14: when use_triton=True, uses a custom Triton fused SwiGLU
    activation kernel (Liger-Kernel-style) that fuses silu(gate) * up into
    a single kernel, saving one intermediate tensor read/write. The linear
    projections (w_gate, w_up, w_down) stay as cuBLAS GEMMs (already optimal).
    """

    def __init__(self, d_model: int = 768, hidden_dim: int | None = None,
                 use_clamp: bool = False, clamp_alpha: float = 1.702,
                 clamp_limit: float = 7.0, use_triton: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.use_clamp = use_clamp
        self.clamp_alpha = clamp_alpha
        self.clamp_limit = clamp_limit
        self._use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, '_fused_gate_up', False):
            gate, up = self.gate_up_proj(x)
        else:
            gate = self.w_gate(x)
            up = self.w_up(x)
        if self.use_clamp:
            # GPT-OSS clamped SwiGLU: scaled sigmoid + clamp + (up+1) residual
            gate = gate.clamp(min=None, max=self.clamp_limit)
            up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
            glu = gate * torch.sigmoid(self.clamp_alpha * gate)
            return self.w_down((up + 1) * glu)
        if self._use_triton:
            from forge.decoding.triton_train_kernels import triton_swiglu_act
            act = triton_swiglu_act(gate, up)
            return self.w_down(act)
        return self.w_down(F.silu(gate) * up)


class ModularBlock(nn.Module):
    """One transformer block with swappable attention and FFN types."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        norm = RMSNorm if getattr(config, 'norm_type', 'layernorm') == 'rmsnorm' else nn.LayerNorm
        _use_triton = getattr(config, 'use_triton_kernels', False)
        # AdaLN-zero conditioning (DiT): when cond_dim is set, use AdaLNZero
        # instead of standard norm. Zero-init modulation → identity at start.
        _cond_dim = getattr(config, 'cond_dim', None)
        self._use_adaln = _cond_dim is not None
        _norm_type = getattr(config, 'norm_type', 'rmsnorm')
        _norm_eps = getattr(config, 'norm_eps', 1e-6)
        _zc = getattr(config, 'zero_centered_norm', False)
        if self._use_adaln:
            from forge.training.losses.adaln_zero import AdaLNZero
            self.ln1 = AdaLNZero(config.d_model, _cond_dim,
                                 norm_type=_norm_type, eps=_norm_eps)
        elif norm is RMSNorm:
            self.ln1 = RMSNorm(config.d_model, eps=_norm_eps,
                              use_triton=_use_triton, zero_centered=_zc)
        else:
            self.ln1 = norm(config.d_model)
        # Determine layer type: conv, mamba, or attention
        layer_types = getattr(config, 'layer_types', None)
        if layer_types is not None and layer_idx < len(layer_types):
            ltype = layer_types[layer_idx].lower()
        else:
            ltype = "attention"
        self.layer_type = ltype
        if ltype in ("conv", "liquid"):
            ksize = getattr(config, 'conv_kernel_size', 3)
            self.attn = DoubleGatedConvLayer(
                config.d_model, kernel_size=ksize,
                use_bitnet=getattr(config, 'use_bitnet', False),
                bitnet_config=config)
        elif ltype in ("mamba", "ssm"):
            # Mamba-3: complex-valued SSM states (R37-1). Selected via
            # config.ssm_type == "mamba3" or config.use_mamba3 == True.
            # Default is "mamba2" (backward compatible).
            _ssm_type = getattr(config, 'ssm_type', 'mamba2')
            if _ssm_type == "mamba3" or getattr(config, 'use_mamba3', False):
                from forge.engine.mamba3 import Mamba3Block
                self.attn = Mamba3Block(
                    d_model=config.d_model,
                    d_state=getattr(config, 'mamba3_d_state',
                                    getattr(config, 'mamba_d_state', 16)),
                    d_conv=getattr(config, 'mamba_d_conv', 4),
                    expand=getattr(config, 'mamba_expand', 2),
                    dt_rank=getattr(config, 'mamba_dt_rank', "auto"),
                    bias=getattr(config, 'mamba_bias', False),
                    conv_bias=getattr(config, 'mamba_conv_bias', True),
                    layer_idx=layer_idx,
                    norm_eps=getattr(config, 'norm_eps', 1e-6),
                    n_inputs=getattr(config, 'mamba3_n_inputs', 1),
                    n_outputs=getattr(config, 'mamba3_n_outputs', 1),
                )
            else:
                from forge.keys.architecture.mamba_probe import MambaLayer
                self.attn = MambaLayer(
                    d_model=config.d_model,
                    d_state=getattr(config, 'mamba_d_state', 16),
                    d_conv=getattr(config, 'mamba_d_conv', 4),
                    expand=getattr(config, 'mamba_expand', 2),
                    dt_rank=getattr(config, 'mamba_dt_rank', "auto"),
                    bias=getattr(config, 'mamba_bias', False),
                    conv_bias=getattr(config, 'mamba_conv_bias', True),
                    layer_idx=layer_idx,
                    norm_eps=getattr(config, 'norm_eps', 1e-6),
                    use_jamba_norms=True,
                )
        else:
            from .builders import build_attention  # lazy: circular dep
            self.attn = build_attention(config, layer_idx=layer_idx)
        if self._use_adaln:
            from forge.training.losses.adaln_zero import AdaLNZero
            self.ln2 = AdaLNZero(config.d_model, _cond_dim,
                                 norm_type=_norm_type, eps=_norm_eps)
        elif norm is RMSNorm:
            self.ln2 = RMSNorm(config.d_model, eps=_norm_eps,
                              use_triton=_use_triton, zero_centered=_zc)
        else:
            self.ln2 = norm(config.d_model)
        from .builders import build_ffn  # lazy: circular dep
        self.ffn = build_ffn(config)
        # Cache whether attn supports pre-allocated KV cache (avoids inspect.signature per forward).
        self._supports_prealloc_cache = isinstance(self.attn, GroupedQueryAttention) or \
            type(self.attn).__name__ in (
                "DifferentialAttention",
                "GroupedTiedAttention",
                "GroupedLatentAttention")
        self._is_conv = isinstance(self.attn, DoubleGatedConvLayer)
        # LFM3 conv pad mask — set by the parent model per forward,
        # consumed HERE (block level) so checkpoint recomputation
        # replays identical ops (see llm.py _forward_impl).
        self._pad_mask = None
        self._is_mamba = type(self.attn).__name__ in ("MambaLayer", "Mamba3Block")
        self._gradient_checkpointing = False

        # V12 ForgeHybrid: zero-init SSM path alongside attention (lossless
        # warm start). The SSM output is zero at init, so the block output
        # is identical to pure attention. Router = sink norm signal.
        self._forge_hybrid_ssm = None
        if getattr(config, 'use_forge_hybrid', False) and not self._is_mamba:
            from forge.keys.architecture.mamba_probe import MambaLayer
            d_state = getattr(config, 'forge_hybrid_d_state',
                               getattr(config, 'mamba_d_state', 16))
            self._forge_hybrid_ssm = MambaLayer(
                d_model=config.d_model,
                d_state=d_state,
                d_conv=getattr(config, 'mamba_d_conv', 4),
                expand=getattr(config, 'mamba_expand', 2),
                dt_rank=getattr(config, 'mamba_dt_rank', "auto"),
                bias=getattr(config, 'mamba_bias', False),
                conv_bias=getattr(config, 'mamba_conv_bias', True),
                layer_idx=layer_idx,
                norm_eps=getattr(config, 'norm_eps', 1e-6),
                use_jamba_norms=True,
            )
            # Zero-init the output projection → SSM contributes nothing at start.
            if hasattr(self._forge_hybrid_ssm, 'out_proj') and hasattr(
                    self._forge_hybrid_ssm.out_proj, 'weight'):
                self._forge_hybrid_ssm.out_proj.weight.detach().zero_()
            self._forge_hybrid_sink_threshold = getattr(
                config, 'forge_hybrid_sink_threshold', float('inf'))

        # R49-2 KDA: gated delta-attention side-path (gate=0 → lossless at
        # start). Applies to every block type — attention, mamba, and conv —
        # as an additive branch on the pre-block residual.
        self._kda = None
        self._kda_gate_zero: bool | None = None  # cached eval gate state
        if getattr(config, 'use_kda', False):
            from forge.keys.attention.kda_key import KDALayer
            kda_heads = getattr(config, 'kda_n_heads', 0) or config.n_heads
            self._kda = KDALayer(
                config.d_model, kda_heads,
                head_dim=getattr(config, 'kda_head_dim', 0) or None,
                beta_gt1=getattr(config, 'kda_beta_gt1', False),
                decay_floor=getattr(config, 'kda_decay_floor', 0.0),
                layer_idx=layer_idx)

        # R60-1 GDN2: decoupled erase/write gated delta side-path (strict
        # KDA superset; gate=0 → lossless). Mutually exclusive with use_kda.
        self._gdn2 = None
        self._gdn2_gate_zero: bool | None = None  # cached eval gate state
        _gdn2_subset = getattr(config, 'gdn2_layers', None)
        if getattr(config, 'use_gdn2', False) and (
                _gdn2_subset is None or layer_idx in _gdn2_subset):
            from forge.keys.attention.gdn2_key import GDN2Layer
            gdn2_heads = getattr(config, 'gdn2_n_heads', 0) or config.n_heads
            self._gdn2 = GDN2Layer(
                config.d_model, gdn2_heads,
                head_dim=getattr(config, 'gdn2_head_dim', 0) or None,
                decay_floor=getattr(config, 'gdn2_decay_floor', 0.0),
                layer_idx=layer_idx)

        # Selective checkpoint strategy: "all" (full block), "ffn" (recompute
        # only FFN — biggest activation consumer, ~2-4x VRAM savings on
        # intermediates with minimal compute penalty), "attn" (recompute only
        # attention), "none" (no recomputation).
        self._gradient_checkpointing_strategy = getattr(
            config, 'selective_gradient_checkpointing', 'all')

        # FFN-SkipLLM: skip FFN on saturated layers during eval.
        # Disabled by default — calibration shows no saturation in V3 (16 layers).
        # See docs/FFN_RESEARCH.md. Infrastructure kept for future 32+ layer models.
        self._ffn_skip_threshold = getattr(config, 'ffn_skip_threshold', 0.0)
        self._ffn_skip_count = 0
        self._static_skip_layers: set[int] = set()  # populated when threshold > 0

        # SandwichNorm: post-sublayer RMSNorm (identity init = lossless).
        # "post_residual" (default): applied to the stream AFTER the add —
        # stabilizes MoE training by bounding activations.
        # "pre_residual" (LFM2/3): applied to the operator/FFN output BEFORE
        # the add — the branch-norm topology HF Lfm2/Lfm3 uses.
        self._use_sandwich = getattr(config, 'use_sandwich_norm', False)
        self._sandwich_pre = (
            getattr(config, 'sandwich_norm_mode', 'post_residual')
            == 'pre_residual')
        if self._use_sandwich:
            if norm is RMSNorm:
                self.post_attn_norm = RMSNorm(config.d_model, eps=_norm_eps,
                                              use_triton=_use_triton,
                                              zero_centered=_zc)
                self.post_ffn_norm = RMSNorm(config.d_model, eps=_norm_eps,
                                             use_triton=_use_triton,
                                             zero_centered=_zc)
            else:
                self.post_attn_norm = norm(config.d_model)
                self.post_ffn_norm = norm(config.d_model)

        # TITAN neural memory + MoD token router (zero-init => lossless at
        # start; the ported checkpoint loads and behaves identically).
        self._memory = None
        self._mod = None
        self._mhc = None
        self._mem_gate_zero: bool | None = None  # cached eval gate state
        _titan_subset = getattr(config, 'titan_layers', None)
        if getattr(config, 'use_titan_memory', False) and (
                _titan_subset is None or layer_idx in _titan_subset):
            from forge.keys.architecture.titan_memory_key import TitanMemory
            self._memory = TitanMemory(
                config.d_model,
                rank=getattr(config, 'titan_memory_rank', 0),
                update_lr=getattr(config, 'titan_update_lr', 0.5),
                forget=getattr(config, 'titan_forget', 0.99),
                momentum=getattr(config, 'titan_momentum', 0.9),
                surprise_cap=getattr(config, 'titan_surprise_cap', 4.0),
                solver=getattr(config, 'titan_solver', 'gd'),
                ridge=getattr(config, 'titan_ridge', 1.0),
                cg_iters=getattr(config, 'titan_cg_iters', 8))
        if getattr(config, 'use_mod', False):
            from forge.keys.architecture.mod_router_key import ModRouter
            self._mod = ModRouter(
                config.d_model,
                keep_fraction=getattr(config, 'mod_keep_fraction', 1.0))
        # MHC: Manifold Hyper-Connections (gate=0 → lossless at start).
        if getattr(config, 'use_mhc', False):
            from forge.keys.architecture.mhc_key import MHCModule
            mhc_rank = getattr(config, 'mhc_rank', 0)
            # rank=0 means "auto" (d_model // 4); MHCModule needs None for auto.
            self._mhc = MHCModule(
                config.d_model,
                rank=mhc_rank if mhc_rank > 0 else None)
            self._mhc_gate_zero: bool | None = None

    def _norm1(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply ln1, passing cond when AdaLN-zero is active."""
        if self._use_adaln:
            return self.ln1(x, cond)
        return self.ln1(x)

    def _norm2(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply ln2, passing cond when AdaLN-zero is active."""
        if self._use_adaln:
            return self.ln2(x, cond)
        return self.ln2(x)

    def _add_attn(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """Residual add for the operator branch, honoring sandwich_norm_mode.

        pre_residual (LFM3): x + post_attn_norm(attn_out)
        post_residual:       post_attn_norm(x + attn_out)   (no-op if off)
        """
        if self._use_sandwich and self._sandwich_pre:
            return x + self.post_attn_norm(attn_out)
        x = x + attn_out
        if self._use_sandwich:
            x = self.post_attn_norm(x)
        return x

    def _add_ffn(self, x: torch.Tensor, ffn_out: torch.Tensor) -> torch.Tensor:
        """Residual add for the FFN branch (same mode rules as _add_attn)."""
        if self._use_sandwich and self._sandwich_pre:
            return x + self.post_ffn_norm(ffn_out)
        x = x + ffn_out
        if self._use_sandwich:
            x = self.post_ffn_norm(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        preallocated_cache: Optional["PreAllocatedKVCache"] = None,
        layer_idx: int = 0,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        # DiffusionBlocks: AdaLN modulation (shift_msa, scale_msa, gate_msa,
        # shift_mlp, scale_mlp, gate_mlp) — 6 * d_model values
        modulation: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        # AdaLN-zero conditioning (DiT): cond embedding for adaptive layer norm.
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        x0 = x  # pre-update residual (for TITAN read + MoD gating)

        # LFM3 conv pad mask: consumed at block level (NOT inside the
        # checkpointed region) so torch.utils.checkpoint recompute
        # replays identical ops — a consumed-attr inside the checkpoint
        # changed the saved-tensor count and crashed training.
        pad_mask = self._pad_mask
        self._pad_mask = None
        # Conv-only kwarg — attention modules don't accept pad_mask.
        _pm_kw = {"pad_mask": pad_mask} if (
            self._is_conv and pad_mask is not None) else {}

        # TRUE Mixture-of-Depths: in training (no KV cache, no attention
        # mask) router-skipped tokens genuinely BYPASS attention + FFN —
        # the block only computes the top-k kept tokens, so FLOPs scale
        # with keep_fraction instead of T. (Inference keeps all tokens to
        # preserve KV cache position alignment.)
        if (self.training and not use_cache and self._mod is not None
                and self._mod.keep_fraction < 1.0
                and attention_bias is None and position_ids is None):
            return self._forward_mod_skip(
                x, layer_idx, attention_bias, position_ids, cond), None

        # Activation checkpointing: recompute forward during backward to save VRAM.
        # Only applies during training (use_cache=False); inference materializes normally.
        if self.training and not use_cache and self._gradient_checkpointing:
            strategy = self._gradient_checkpointing_strategy
            if strategy == "all":
                def custom_forward(x_inner):
                    attn_out, present = self.attn(
                        self._norm1(x_inner, cond),
                        past_key_value=past_key_value, use_cache=False,
                        preallocated_cache=(
                            preallocated_cache
                            if self._supports_prealloc_cache else None),
                        layer_idx=layer_idx, attention_bias=attention_bias,
                        position_ids=position_ids, cu_seqlens=cu_seqlens,
                        **_pm_kw)
                    x_inner = self._add_attn(x_inner, attn_out)
                    ffn_out = self.ffn(self._norm2(x_inner, cond))
                    aux = None
                    if isinstance(ffn_out, tuple):
                        aux = ffn_out[1]
                        ffn_out = ffn_out[0]
                    x_inner = self._add_ffn(x_inner, ffn_out)
                    return x_inner, present, aux
                x, present, aux = torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
                self._last_aux_loss = aux
            elif strategy == "ffn":
                # Keep attention activations (cheap: ~2*d_model per token),
                # recompute only the FFN (hidden_dim >> d_model) during backward.
                if preallocated_cache is not None and self._supports_prealloc_cache:
                    attn_out, present = self.attn(
                        self._norm1(x, cond), past_key_value=past_key_value, use_cache=False,
                        preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                        attention_bias=attention_bias, position_ids=position_ids,
                        cu_seqlens=cu_seqlens, **_pm_kw,
                    )
                else:
                    attn_out, present = self.attn(
                        self._norm1(x, cond), past_key_value=past_key_value, use_cache=False,
                        attention_bias=attention_bias, position_ids=position_ids,
                        cu_seqlens=cu_seqlens, **_pm_kw,
                    )
                x = self._add_attn(x, attn_out)
                ffn_out = torch.utils.checkpoint.checkpoint(
                    self.ffn, self._norm2(x, cond), use_reentrant=False)
                self._last_aux_loss = None
                if isinstance(ffn_out, tuple):
                    self._last_aux_loss = ffn_out[1]
                    ffn_out = ffn_out[0]
                x = self._add_ffn(x, ffn_out)
            elif strategy == "attn":
                # Recompute only attention; FFN activations stay materialized.
                def _attn_forward(x_inner):
                    return self.attn(
                        x_inner, past_key_value=past_key_value, use_cache=False,
                        preallocated_cache=preallocated_cache if self._supports_prealloc_cache else None,
                        layer_idx=layer_idx, attention_bias=attention_bias,
                        position_ids=position_ids, cu_seqlens=cu_seqlens,
                        **_pm_kw)
                attn_out, present = torch.utils.checkpoint.checkpoint(
                    _attn_forward, self._norm1(x, cond), use_reentrant=False)
                x = self._add_attn(x, attn_out)
                ffn_out = self.ffn(self._norm2(x, cond))
                self._last_aux_loss = None
                if isinstance(ffn_out, tuple):
                    self._last_aux_loss = ffn_out[1]
                    ffn_out = ffn_out[0]
                x = self._add_ffn(x, ffn_out)
            else:  # "none" — no recomputation this block
                if preallocated_cache is not None and self._supports_prealloc_cache:
                    attn_out, present = self.attn(
                        self._norm1(x, cond), past_key_value=past_key_value, use_cache=use_cache,
                        preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                        attention_bias=attention_bias, position_ids=position_ids,
                        cu_seqlens=cu_seqlens, **_pm_kw,
                    )
                else:
                    attn_out, present = self.attn(
                        self._norm1(x, cond), past_key_value=past_key_value, use_cache=use_cache,
                        attention_bias=attention_bias, position_ids=position_ids,
                        cu_seqlens=cu_seqlens, **_pm_kw,
                    )
                x = self._add_attn(x, attn_out)
                ffn_out = self.ffn(self._norm2(x, cond))
                self._last_aux_loss = None
                if isinstance(ffn_out, tuple):
                    self._last_aux_loss = ffn_out[1]
                    ffn_out = ffn_out[0]
                x = self._add_ffn(x, ffn_out)
        else:
            # DiffusionBlocks: extract AdaLN modulation (4 * d_model)
            # No gates — shift/scale only (zero-init = identity, gradients flow)
            shift_msa = scale_msa = shift_mlp = scale_mlp = None
            if modulation is not None:
                # Cast modulation to x's dtype (AdaLN may be float32, x may be bf16)
                mod = modulation.to(x.dtype)
                chunks = mod.chunk(4, dim=-1)
                shift_msa, scale_msa, shift_mlp, scale_mlp = chunks

            # Attention path
            attn_in = self._norm1(x, cond)
            if shift_msa is not None:
                attn_in = attn_in * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            # Inference path: wrappers (SeqSplit, FusedQKNormRopeCache, etc.)
            # patch attn.forward with fixed signatures that don't accept
            # cu_seqlens. cu_seqlens is only meaningful when not use_cache
            # (training varlen); at inference it is always None. Only pass
            # it when set, so patched forwards never see the unexpected kwarg.
            _cu_kw = {"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}
            if preallocated_cache is not None and self._supports_prealloc_cache:
                attn_out, present = self.attn(
                    attn_in, past_key_value=past_key_value, use_cache=use_cache,
                    preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                    attention_bias=attention_bias, position_ids=position_ids,
                    **_cu_kw, **_pm_kw,
                )
            else:
                attn_out, present = self.attn(
                    attn_in, past_key_value=past_key_value, use_cache=use_cache,
                    attention_bias=attention_bias, position_ids=position_ids,
                    **_cu_kw, **_pm_kw,
                )
            x = self._add_attn(x, attn_out)
            # FFN-SkipLLM: skip FFN on saturated layers during eval.
            # Uses a static skip set (calibrated via cosine similarity).
            # Safe with KV cache: FFN doesn't touch KV state, only attention.
            # Disabled by default — V3 has no saturation region (see docs/FFN_RESEARCH.md).
            if (self._ffn_skip_threshold > 0.0 and not self.training
                    and self.layer_idx in self._static_skip_layers):
                self._ffn_skip_count += 1
                ffn_out = torch.zeros_like(x)
                self._last_aux_loss = None
            else:
                ffn_in = self._norm2(x, cond)
                if shift_mlp is not None:
                    ffn_in = ffn_in * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
                ffn_out = self.ffn(ffn_in)
                self._last_aux_loss = None
                if isinstance(ffn_out, tuple):
                    self._last_aux_loss = ffn_out[1]
                    ffn_out = ffn_out[0]
            x = self._add_ffn(x, ffn_out)

        # TITAN memory read + MoD token gating (zero-init => lossless).
        # Applied on the combined block update for every branch above.
        if self._memory is not None or self._mod is not None:
            # Fast lossless path: TITAN gate closed + MoD keep-all => no-op,
            # return x untouched (bit-exact vs. a plain block).
            # The gate .item() check is a GPU->CPU sync; the gate value is
            # static (zero-init, only changed by optimizer steps), so it is
            # cached after the first check (16 syncs/token otherwise — the
            # biggest decode overhead on V3). Caching applies to both training
            # and eval: the gate parameter only changes via optimizer weight
            # updates, not during forward passes.
            if self._memory is None:
                mem_noop = True
            elif self._mem_gate_zero is None:
                self._mem_gate_zero = (self._memory.gate.item() == 0.0)
                mem_noop = self._mem_gate_zero
            else:
                mem_noop = self._mem_gate_zero
            mod_noop = (self._mod is None
                        or self._mod.token_mask(x0) is None)
            if not (mem_noop and mod_noop):
                update = x - x0
                if self._memory is not None and not mem_noop:
                    update = update + self._memory(x0)
                if self._mod is not None and not mod_noop:
                    update = self._mod.apply(x0, update)
                x = x0 + update

        # MHC: Manifold Hyper-Connection (gate=0 → lossless at start).
        # Wraps the full block update: x = x0 + update + gate * proj(update).
        # At gate=0, this is x = x0 + update (standard residual, bit-exact).
        if self._mhc is not None:
            if self._mhc_gate_zero is None:
                self._mhc_gate_zero = (self._mhc.gate.item() == 0.0)
            if not self._mhc_gate_zero:
                update = x - x0
                x = self._mhc(x0, update)

        # V12 ForgeHybrid: zero-init SSM path alongside attention.
        # At warm start the SSM output is zero (zero-init out_proj), so this
        # is a no-op. Training opens the path via the SSM weights.
        if self._forge_hybrid_ssm is not None:
            ssm_out = self._forge_hybrid_ssm(x0)
            if isinstance(ssm_out, tuple):
                ssm_out = ssm_out[0]
            x = x + ssm_out

        # R49-2 KDA: gated delta-attention branch on the pre-block residual.
        # gate=0 → skip entirely (bit-exact vs. baseline + no decode overhead).
        # The gate is a scalar parameter that only changes via optimizer steps,
        # so its zero-check is cached after the first .item() sync.
        if self._kda is not None:
            if self._kda_gate_zero is None:
                self._kda_gate_zero = (self._kda.gate.item() == 0.0)
            if not self._kda_gate_zero:
                x = x + self._kda(x0, use_cache=use_cache)

        # R60-1 GDN2: decoupled erase/write delta-rule branch (same
        # gate-zero skip convention as KDA).
        if self._gdn2 is not None:
            if self._gdn2_gate_zero is None:
                self._gdn2_gate_zero = (self._gdn2.gate.item() == 0.0)
            if not self._gdn2_gate_zero:
                x = x + self._gdn2(x0, use_cache=use_cache)

        return x, present

    def _forward_mod_skip(self, x: torch.Tensor, layer_idx: int,
                          attention_bias: torch.Tensor | None,
                          position_ids: torch.Tensor | None,
                          cond: torch.Tensor | None = None,
                          ) -> torch.Tensor:
        """Run the block only on router-kept tokens (per batch row).

        Matches the MoD paper: at this depth, skipped tokens are absent from
        attention entirely (causal within the kept subsequence). Kept tokens
        still get the full attention + FFN + TITAN update. The hard top-k
        selection is non-differentiable, so the router is trained by the
        aux loss attached to `_last_aux_loss` (see ModRouter.aux_loss).
        """
        mask = self._mod.token_mask(x)
        out = x.clone()
        for b in range(x.shape[0]):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            x_k = x[b][idx].unsqueeze(0)  # (1, T_k, D)
            attn_out, _ = self.attn(
                self._norm1(x_k, cond), past_key_value=None, use_cache=False,
                preallocated_cache=None, layer_idx=layer_idx,
                attention_bias=attention_bias, position_ids=position_ids)
            h = self._add_attn(x_k, attn_out)
            ffn_out = self.ffn(self._norm2(h, cond))
            self._last_aux_loss = None
            if isinstance(ffn_out, tuple):
                self._last_aux_loss = ffn_out[1]
                ffn_out = ffn_out[0]
            h = self._add_ffn(h, ffn_out)
            if self._memory is not None:
                h = h + self._memory(x_k)
            out[b][idx] = h[0]
        if self._last_aux_loss is None:
            self._last_aux_loss = self._mod.aux_loss(x, mask)
        return out


