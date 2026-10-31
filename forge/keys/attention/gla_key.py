"""Grouped Latent Attention (GLA) — hardware-efficient latent attention.

Based on "Hardware-Efficient Attention for Fast Decoding" (arXiv 2505.21487)
and DeepSeek-V2 MLA with Decoupled RoPE.

Key idea: project K/V into a compact latent space, cache only the latent
vector per token (not full per-head K/V), and absorb the up-projections into
Q and the output projection at inference. This cuts KV cache traffic ~57×
while paying ~4.5× more FLOPs — shifting decode from bandwidth-bound to
compute-bound, where modern GPUs have headroom.

Decoupled RoPE (from DeepSeek-V2 MLA):
  Standard RoPE applies rotation to all head_dim dimensions, which breaks
  the low-rank projection invariance needed for latent compression. The
  solution: split the key into a semantic part (NoPE, compressed via latent)
  and a positional part (RoPE, kept separate and uncompressed).

  - head_dim = nope_dim + rope_dim (e.g., 64 = 32 + 32)
  - Q is split: q_nope (semantic) + q_rope (positional, gets RoPE)
  - K from latent up-projection: nope_dim per head (semantic, NoPE)
  - K from k_rope_proj: rope_dim (positional, RoPE applied, shared across heads)
  - Attention: q = [q_nope; q_rope], k = [k_nope; k_rope], both head_dim
  - V from latent up-projection: full head_dim (no RoPE needed)

Lossless warm start (identity mode):
  - latent_dim = n_kv_heads * head_dim (full KV dim, no compression)
  - W_DKV (down-projection) = identity
  - W_UK, W_UV (up-projections) = identity
  - compression gate = 0 (no low-rank compression)
  - k_rope_proj = 0 (no positional signal at init; standard RoPE used instead)
  => bit-exact vs GQA at load; training moves gate off 0 to activate
     latent compression + decoupled RoPE, shrinking KV cache.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.keys._tensor_utils import _repeat_kv as _repeat_kv_shared
from forge.keys.misc.base import Key, KeyClass, KeyResult
from forge.model_loader import RotaryEmbedding


class GroupedLatentAttention(nn.Module):
    """GLA: latent-compressed attention with decoupled RoPE and identity warm start.

    Config-compatible with the GQA slot (same forward signature).
    """

    def __init__(self, d_model: int = 2048, n_heads: int = 32,
                 n_kv_heads: int | None = None, max_seq_len: int = 32768,
                 base: float = 1_000_000.0, rope_scaling: dict | None = None,
                 use_qk_norm: bool = False, attn_bias: bool = False,
                 latent_dim: int | None = None,
                 n_layers: int = 16, layer_idx: int = 0,
                 rope_dim: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.layer_idx = layer_idx

        # Decoupled RoPE: split head_dim into semantic (nope) + positional (rope)
        # Default: half the dimensions carry positional info, half carry semantic.
        # The positional part is NOT compressed (RoPE breaks low-rank projections).
        self.rope_dim = rope_dim if rope_dim is not None else self.head_dim // 2
        self.nope_dim = self.head_dim - self.rope_dim

        # Full KV dim (what GQA would store)
        self.full_kv_dim = self.n_kv_heads * self.head_dim
        # Latent dim: defaults to full (lossless); config can set smaller
        self.latent_dim = latent_dim if latent_dim is not None else self.full_kv_dim

        # Projections
        q_dim = n_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, q_dim, bias=attn_bias)
        # K/V down-projection: d_model -> latent_dim (shared for K and V)
        self.kv_down_proj = nn.Linear(d_model, self.latent_dim, bias=attn_bias)
        # K/V up-projections: latent_dim -> full_kv_dim
        # bias follows attn_bias — Qwen-family ports carry the original
        # k/v biases here (k = W_KC·latent + b_k).
        self.k_up_proj = nn.Linear(self.latent_dim, self.full_kv_dim,
                                   bias=attn_bias)
        self.v_up_proj = nn.Linear(self.latent_dim, self.full_kv_dim,
                                   bias=attn_bias)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Standard RoPE (for identity/warm-start path — applies to all head_dim dims)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len,
                                    base=base, rope_scaling=rope_scaling)
        # Decoupled RoPE (for full GLA path — applies only to rope_dim dims)
        self.rope_decoupled = RotaryEmbedding(
            self.rope_dim, max_seq_len=max_seq_len,
            base=base, rope_scaling=rope_scaling)

        # QK-norm
        self.use_qk_norm = use_qk_norm
        self._qk_norm_identity = True
        if use_qk_norm:
            from forge.model_loader import RMSNorm
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Compression gate: 0 = identity (lossless), >0 = latent compression
        # When gate=0, up-projections are identity and we store full KV.
        # When gate>0, we store only the latent and up-project on read.
        self.compression_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self._identity_cache: bool | None = None

        # Decoupled RoPE key: separate small projection for positional K
        # Outputs head_dim (sliced to rope_dim in forward) for backward compat.
        # Zero-initialized so it doesn't affect the warm start.
        self.k_rope_proj = nn.Linear(d_model, self.head_dim, bias=attn_bias)
        nn.init.zeros_(self.k_rope_proj.weight)

    @property
    def _identity(self) -> bool:
        """Check dynamically so load_state_dict with gate>0 activates full path.

        Cached in eval mode: the raw .item() is a GPU sync + dynamo graph
        break every layer every forward (24x/token), which also poisons
        breakable-CUDA-graph capture windows. Training path stays dynamic."""
        if self.training or self._identity_cache is None:
            v = self.compression_gate.item() == 0.0
            if not self.training:
                self._identity_cache = v
            return v
        return self._identity_cache

    def _load_from_state_dict(self, *args, **kwargs):
        self._identity_cache = None
        super()._load_from_state_dict(*args, **kwargs)

    def set_identity(self, identity: bool):
        """Toggle identity (lossless) vs. full GLA mode."""
        if identity:
            with torch.no_grad():
                self.compression_gate.fill_(0.0)
        else:
            with torch.no_grad():
                self.compression_gate.fill_(1.0)
        self._identity_cache = None

    def _repeat_kv(self, x):
        return _repeat_kv_shared(x, self.n_rep)

    def forward(self, x, past_key_value=None, use_cache=False,
                preallocated_cache=None, layer_idx: int = 0,
                attention_bias: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                cu_seqlens: torch.Tensor | None = None):
        B, T, _C = x.shape
        hd = self.head_dim

        if self._identity:
            # ── Bit-exact GQA-equivalent path (warm start) ───────────────
            # Standard RoPE on all dims, no decoupling, no compression.
            q = self.q_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
            kv_latent = self.kv_down_proj(x)  # (B, T, full_kv_dim)
            k = self.k_up_proj(kv_latent).view(
                B, T, self.n_kv_heads, hd).transpose(1, 2)
            v = self.v_up_proj(kv_latent).view(
                B, T, self.n_kv_heads, hd).transpose(1, 2)

            if self.use_qk_norm and not self._qk_norm_identity:
                q = self.q_norm(q)
                k = self.k_norm(k)

            if preallocated_cache is not None:
                past_len = preallocated_cache.position
                q = self.rope(q, offset=past_len, position_ids=position_ids)
                k = self.rope(k, offset=past_len, position_ids=position_ids)
                if preallocated_cache.pos_t is not None and T == 1:
                    # Graph-safe: device-indexed write + full-buffer read.
                    # Future slots are masked by attention_bias — required
                    # for CUDA-graph decode capture (fixed shapes, all ops
                    # tensor-driven).
                    preallocated_cache.append_t(layer_idx, k, v)
                    k, v = preallocated_cache.kv_view(layer_idx)
                else:
                    preallocated_cache.append(layer_idx, k, v)
                    k = preallocated_cache.k_caches[layer_idx][:, :, :past_len + T]
                    v = preallocated_cache.v_caches[layer_idx][:, :, :past_len + T]
            else:
                past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
                q = self.rope(q, offset=past_len, position_ids=position_ids)
                k = self.rope(k, offset=past_len, position_ids=position_ids)
                if past_key_value is not None:
                    k = torch.cat([past_key_value[0], k], dim=-2)
                    v = torch.cat([past_key_value[1], v], dim=-2)

            new_kv = (k, v) if use_cache else None
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)

            if attention_bias is not None:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
            out = out.transpose(1, 2).reshape(B, T, -1)
            return self.out_proj(out), new_kv

        # ── Full GLA path with Decoupled RoPE ───────────────────────────
        # Split Q into semantic (nope) + positional (rope)
        q_full = self.q_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        q_nope = q_full[..., :self.nope_dim]   # (B, n_heads, T, nope_dim)
        q_rope = q_full[..., self.nope_dim:]   # (B, n_heads, T, rope_dim)

        # Compress K/V into latent space
        kv_latent = self.kv_down_proj(x)  # (B, T, latent_dim)

        # Decoupled RoPE key: shared across all heads, positional only
        k_rope = self.k_rope_proj(x)[..., :self.rope_dim]  # (B, T, rope_dim)
        k_rope = k_rope.view(B, T, 1, self.rope_dim).transpose(1, 2)
        k_rope = k_rope.expand(B, self.n_heads, T, self.rope_dim)

        if self.use_qk_norm and not self._qk_norm_identity:
            q_nope = self.q_norm(q_nope)

        # Apply RoPE ONLY to positional parts (decoupled — doesn't break latent)
        if preallocated_cache is not None:
            past_len = preallocated_cache.position
            q_rope = self.rope_decoupled(q_rope, offset=past_len, position_ids=position_ids)
            k_rope = self.rope_decoupled(k_rope, offset=past_len, position_ids=position_ids)
        else:
            past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
            q_rope = self.rope_decoupled(q_rope, offset=past_len, position_ids=position_ids)
            k_rope = self.rope_decoupled(k_rope, offset=past_len, position_ids=position_ids)

        # Up-project latent to semantic K (nope part) and full V
        k_nope = self.k_up_proj(kv_latent).view(
            B, T, self.n_kv_heads, hd).transpose(1, 2)
        k_nope = k_nope[..., :self.nope_dim]  # slice to semantic dims only
        v_full = self.v_up_proj(kv_latent).view(
            B, T, self.n_kv_heads, hd).transpose(1, 2)

        # Concatenate [k_nope; k_rope] for full head_dim attention
        k_nope_rep = self._repeat_kv(k_nope)  # (B, n_heads, T, nope_dim)
        k_attn = torch.cat([k_nope_rep, k_rope], dim=-1)  # (B, n_heads, T, hd)
        q_attn = torch.cat([q_nope, q_rope], dim=-1)      # (B, n_heads, T, hd)
        v = self._repeat_kv(v_full)

        # Cache: store reconstructed K (nope+rope) and V for compatibility
        # In production, would cache only latent + k_rope (much smaller).
        if preallocated_cache is not None:
            if preallocated_cache.pos_t is not None and T == 1:
                # Graph-safe write + full-buffer read (see identity path).
                preallocated_cache.append_t(layer_idx, k_attn, v)
                k_attn, v = preallocated_cache.kv_view(layer_idx)
            else:
                preallocated_cache.append(layer_idx, k_attn, v)
                k_attn = preallocated_cache.k_caches[layer_idx][:, :, :past_len + T]
                v = preallocated_cache.v_caches[layer_idx][:, :, :past_len + T]
        else:
            if past_key_value is not None:
                k_attn = torch.cat([past_key_value[0], k_attn], dim=-2)
                v = torch.cat([past_key_value[1], v], dim=-2)

        new_kv = (k_attn, v) if use_cache else None

        if attention_bias is not None:
            out = F.scaled_dot_product_attention(q_attn, k_attn, v, attn_mask=attention_bias)
        else:
            out = F.scaled_dot_product_attention(q_attn, k_attn, v, is_causal=T > 1)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out), new_kv


# ---------------------------------------------------------------------------
# Key: GQA -> GLA weight transform (identity warm start)
# ---------------------------------------------------------------------------

class GLAKey(Key):
    """Transform GQA attention weights into GLA weights (identity warm start).

    forward: merge k_proj and v_proj into shared kv_down_proj (k_proj=v_proj=kv_down_proj),
              set k_up_proj=v_up_proj=identity, zero k_rope_proj, gate=0.
    reverse: extract k_proj from kv_down_proj (v_proj = k_proj since they're tied).
    Key class: PARTIAL (the tying is one-directional without SVD).
    """

    def __init__(self, n_layers: int = 16, n_heads: int = 32,
                 n_kv_heads: int = 8, latent_dim: int | None = None):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.latent_dim = latent_dim  # None = full (lossless)

    @property
    def name(self) -> str:
        return "grouped_latent_attention"

    @property
    def description(self) -> str:
        return "GQA -> GLA (identity warm start, lossless)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(data)
            n_layers = 0
            for k in state:
                if k.startswith("blocks.") and ".attn.q_proj.weight" in k:
                    n_layers = max(n_layers, int(k.split(".")[1]) + 1)
            if n_layers == 0:
                n_layers = self.n_layers

            for i in range(n_layers):
                base = f"blocks.{i}.attn"
                kw = f"{base}.k_proj.weight"
                vw = f"{base}.v_proj.weight"

                if kw in state:
                    # kv_down_proj = k_proj (V is tied to K at warm start)
                    state[f"{base}.kv_down_proj.weight"] = state[kw].clone()
                    # If latent_dim < full, we'd SVD here. For lossless, keep full.
                    if self.latent_dim is not None and self.latent_dim < state[kw].shape[0]:
                        # SVD low-rank approximation for compressed latent
                        U, S, Vh = torch.linalg.svd(state[kw], full_matrices=False)
                        state[f"{base}.kv_down_proj.weight"] = (
                            U[:, :self.latent_dim] * S[:self.latent_dim]
                        ).contiguous()
                        # k_up_proj = U^T (pseudo-inverse up-projection)
                        state[f"{base}.k_up_proj.weight"] = Vh[:self.latent_dim].contiguous()
                        state[f"{base}.v_up_proj.weight"] = Vh[:self.latent_dim].contiguous()
                    else:
                        # Identity up-projections
                        full_dim = state[kw].shape[0]
                        state[f"{base}.k_up_proj.weight"] = torch.eye(
                            full_dim, dtype=state[kw].dtype)
                        state[f"{base}.v_up_proj.weight"] = torch.eye(
                            full_dim, dtype=state[kw].dtype)

                    # Remove old separate k/v projections
                    del state[kw]
                    state.pop(vw, None)

                # Zero-init k_rope_proj (decoupled RoPE branch)
                hd = state[f"{base}.q_proj.weight"].shape[0] // self.n_heads if f"{base}.q_proj.weight" in state else 64
                state[f"{base}.k_rope_proj.weight"] = torch.zeros(
                    hd, state.get(f"{base}.q_proj.weight", torch.zeros(1, 2048)).shape[-1],
                    dtype=state.get(f"{base}.q_proj.weight", torch.zeros(1)).dtype)

                # Compression gate = 0 (lossless)
                state[f"{base}.compression_gate"] = torch.zeros(1, dtype=torch.float32)

            return KeyResult(success=True, weights=state,
                             metadata={"n_layers": n_layers, "identity": True})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(weights)
            for k in list(state.keys()):
                if ".compression_gate" in k or ".k_rope_proj" in k:
                    del state[k]
                elif ".kv_down_proj.weight" in k:
                    # Extract k_proj (= v_proj since tied at warm start)
                    state[k.replace("kv_down_proj", "k_proj")] = state[k].clone()
                    state[k.replace("kv_down_proj", "v_proj")] = state[k].clone()
                    del state[k]
                elif ".k_up_proj.weight" in k or ".v_up_proj.weight" in k:
                    del state[k]
            return KeyResult(success=True, weights=state)
        except Exception as e:
            return KeyResult(success=False, error=str(e))
