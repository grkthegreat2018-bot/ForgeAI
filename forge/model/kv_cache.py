"""Pre-allocated KV cache and output unpacking for ConfigurableResearchLLM."""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]

__all__ = [
    "KVCache",
    "PreAllocatedKVCache",
    "create_kv_cache",
    "unpack_output_with_kv",
]



class PreAllocatedKVCache:
    """Pre-allocated KV cache — O(1) append instead of O(n) torch.cat per token.

    Allocates max_seq_len slots upfront. Each attention layer writes new k/v
    into the buffer by index, then reads a view of the filled portion.
    This eliminates the O(n²) tensor growth from torch.cat in generation loops.

    With quantize="int8": stores K/V as int8 with per-token scale (q8_0 style),
    halving cache memory. Dequantization happens on read in get_layer().
    This matches llama.cpp's q8_0 KV cache approach.

    Usage:
        cache = PreAllocatedKVCache(n_layers, batch, n_kv_heads, max_seq_len, head_dim, dtype, device)
        # In generation loop:
        cache.advance()  # increment position
        # Pass cache.get_layer(i) as past_key_value to attention
        # Attention writes new k/v via cache.append(i, k_new, v_new)
    """

    def __init__(self, n_layers: int, batch: int, n_kv_heads: int,
                 max_seq_len: int, head_dim: int, dtype: torch.dtype,
                 device: torch.device, n_kv_heads_per_layer: list | None = None,
                 quantize: str = "none",
                 ring_windows: list[int | None] | None = None):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.position = 0  # current fill position (shared across layers)
        self.quantize = quantize  # "none", "int8"
        self._dtype = dtype

        # Support per-layer head counts (e.g. GQA with different n_kv_heads)
        if n_kv_heads_per_layer is None:
            n_kv_heads_per_layer = [n_kv_heads] * n_layers

        # Skip cache allocation for layers with 0 KV heads (conv layers)
        self.n_kv_heads_per_layer = n_kv_heads_per_layer

        # SWA ring KV (d2 strict-lossless): a sliding-window layer only ever
        # reads the last W keys, so its buffer is capped at min(max_seq_len, W)
        # and position p writes to slot p % W. Attention is permutation-
        # invariant over keys, so wrapped storage is mathematically identical
        # to the masked full buffer. Unquantized path only — the int8 scales
        # buffers would need the same wrap bookkeeping.
        if ring_windows is None or quantize != "none":
            ring_windows = [None] * n_layers
        self.ring_caps = [
            min(max_seq_len, w) if w else None for w in ring_windows]

        if quantize == "int8":
            # INT8 quantized cache: store int8 values + fp16 per-token scale
            # Memory: 1 byte/element + 2 bytes/scale per (B, n_kv, T) = ~50% of fp16
            cache_dtype = torch.int8
            scale_dtype = torch.float16
        else:
            cache_dtype = dtype
            scale_dtype = dtype

        self.k_caches = []
        self.v_caches = []
        self.k_scales = []
        self.v_scales = []
        for i in range(n_layers):
            nkvh = n_kv_heads_per_layer[i]
            if nkvh == 0:
                # Conv layer — no KV cache needed
                self.k_caches.append(None)
                self.v_caches.append(None)
                self.k_scales.append(None)
                self.v_scales.append(None)
                continue
            cap = self.ring_caps[i] or max_seq_len
            k = torch.zeros(batch, nkvh, cap, head_dim, dtype=cache_dtype, device=device)
            v = torch.zeros(batch, nkvh, cap, head_dim, dtype=cache_dtype, device=device)
            self.k_caches.append(k)
            self.v_caches.append(v)
            if quantize == "int8":
                # Per-token scale: (B, n_kv, max_seq_len, 1)
                ks = torch.zeros(batch, nkvh, max_seq_len, 1, dtype=scale_dtype, device=device)
                vs = torch.zeros(batch, nkvh, max_seq_len, 1, dtype=scale_dtype, device=device)
                self.k_scales.append(ks)
                self.v_scales.append(vs)
            else:
                self.k_scales.append(None)
                self.v_scales.append(None)

    def ring_cap(self, layer_idx: int) -> int | None:
        """Ring-buffer capacity for a sliding-window layer, else None."""
        return self.ring_caps[layer_idx]

    def get_layer(self, layer_idx: int) -> KVCache | None:
        """Get the (k, v) view for a layer, or None if at position 0.

        Ring layers return the last min(pos, C) keys in POSITION ORDER —
        a cat of the two buffer slices when the ring has wrapped (slot
        p % C holds position p). Small copy once pos > C; W is tiny.
        """
        if self.position == 0 or self.k_caches[layer_idx] is None:
            return None
        pos = self.position
        ring_c = self.ring_caps[layer_idx]
        if ring_c is not None and pos > ring_c:
            s = pos % ring_c
            k = torch.cat([self.k_caches[layer_idx][:, :, s:],
                           self.k_caches[layer_idx][:, :, :s]], dim=2)
            v = torch.cat([self.v_caches[layer_idx][:, :, s:],
                           self.v_caches[layer_idx][:, :, :s]], dim=2)
            return (k, v)
        if ring_c is not None:
            pos = min(pos, ring_c)
        if self.quantize == "int8":
            # Dequantize: int8 * scale → original dtype
            k = self.k_caches[layer_idx][:, :, :pos].to(self._dtype) * \
                self.k_scales[layer_idx][:, :, :pos].to(self._dtype)
            v = self.v_caches[layer_idx][:, :, :pos].to(self._dtype) * \
                self.v_scales[layer_idx][:, :, :pos].to(self._dtype)
            return (k, v)
        k = self.k_caches[layer_idx][:, :, :pos]
        v = self.v_caches[layer_idx][:, :, :pos]
        return (k, v)

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Write new k/v at the current position and advance (per-layer)."""
        if self.k_caches[layer_idx] is None:
            return  # conv layer, no cache
        T = k_new.shape[-2]
        pos = self.position
        ring_c = self.ring_caps[layer_idx]
        if ring_c is not None:
            # Ring write: position p lands in slot p % C. Only the last C
            # tokens of a >C append are in-window — earlier writes would
            # be overwritten anyway (dup index order is unspecified).
            n = min(T, ring_c)
            idx = torch.arange(pos + T - n, pos + T,
                               device=k_new.device) % ring_c
            self.k_caches[layer_idx].index_copy_(2, idx, k_new[..., -n:, :])
            self.v_caches[layer_idx].index_copy_(2, idx, v_new[..., -n:, :])
            return
        if pos + T > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: pos={pos} + T={T} > max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len or use a paged/evicting KV cache strategy."
            )
        if self.quantize == "int8":
            # Quantize: scale = max(abs(x)) / 127, q = round(x / scale)
            k_scale = k_new.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            v_scale = v_new.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            k_q = torch.clamp(torch.round(k_new / k_scale), -128, 127).to(torch.int8)
            v_q = torch.clamp(torch.round(v_new / v_scale), -128, 127).to(torch.int8)
            self.k_caches[layer_idx][:, :, pos:pos + T] = k_q
            self.v_caches[layer_idx][:, :, pos:pos + T] = v_q
            self.k_scales[layer_idx][:, :, pos:pos + T] = k_scale.to(torch.float16)
            self.v_scales[layer_idx][:, :, pos:pos + T] = v_scale.to(torch.float16)
        else:
            self.k_caches[layer_idx][:, :, pos:pos + T] = k_new
            self.v_caches[layer_idx][:, :, pos:pos + T] = v_new

    def advance(self, n: int = 1):
        """Advance the fill position by n tokens."""
        self.position += n

    def reset(self):
        """Reset to empty (start of new sequence)."""
        self.position = 0
        if self.pos_t is not None:
            self.pos_t.zero_()

    # ── Graph-safe tensor-position mode ──────────────────────────────────
    # For CUDA-graph decode capture: python-int `position` and `[:pos]`
    # slices bake into a graph at capture time, freezing the context
    # length. With pos_t set (device scalar), append_t writes via
    # index_copy_ and callers read the FULL buffer, masking future slots
    # via attention_bias — every op stays tensor-driven and replayable.

    pos_t: torch.Tensor | None = None

    # Read horizon for the pos_t path: attention reads k_caches[:, :, :view_len].
    # None = full buffer.  Set per CUDA-graph capture so each bucket's graph
    # replays attention over a fixed-length VIEW of the same storage —
    # switching buckets at decode time costs zero data movement.
    view_len: int | None = None

    def enable_tensor_pos(self):
        """Switch to device-tensor position mode for graph capture."""
        dev = next((k.device for k in self.k_caches if k is not None),
                   torch.device('cuda'))
        self.pos_t = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos_t.fill_(self.position)

    def sync_tensor_pos(self):
        """Copy python position into pos_t (call before each replay)."""
        if self.pos_t is not None:
            self.pos_t.fill_(self.position)

    def append_t(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Device-indexed write at pos_t — capturable inside a CUDA graph."""
        if self.k_caches[layer_idx] is None or self.pos_t is None:
            return
        ring_c = self.ring_caps[layer_idx]
        if ring_c is not None:
            # Ring write stays tensor-driven: slot = pos_t % C.
            idx = self.pos_t % ring_c
        else:
            idx = self.pos_t
        self.k_caches[layer_idx].index_copy_(2, idx, k_new)
        self.v_caches[layer_idx].index_copy_(2, idx, v_new)

    def kv_view(self, layer_idx: int):
        """k/v read view honoring view_len (pos_t path only)."""
        k = self.k_caches[layer_idx]
        v = self.v_caches[layer_idx]
        if k is None:
            return None, None
        if self.view_len is not None:
            return k[:, :, :self.view_len], v[:, :, :self.view_len]
        return k, v

    @property
    def filled(self) -> int:
        return self.position

    def cache_memory_mb(self) -> float:
        """Estimate current KV cache memory usage in MB."""
        total = 0
        for i in range(self.n_layers):
            if self.k_caches[i] is None:
                continue
            k_bytes = self.k_caches[i].element_size() * self.k_caches[i].numel()
            v_bytes = self.v_caches[i].element_size() * self.v_caches[i].numel()
            total += k_bytes + v_bytes
            if self.quantize == "int8":
                total += self.k_scales[i].element_size() * self.k_scales[i].numel()
                total += self.v_scales[i].element_size() * self.v_scales[i].numel()
        return total / 1e6


def create_kv_cache(model: nn.Module, max_total: int, batch: int = 1,
                    device: torch.device | None = None) -> PreAllocatedKVCache:
    """Build a PreAllocatedKVCache sized for *model* with per-layer head counts.

    Handles hybrid conv/attention architectures (e.g. LFM2.5): conv layers get
    0 KV heads to avoid wasting VRAM.  This replaces the open-coded cache
    construction that was duplicated across self_play and inference modules.
    """
    cfg = model.config
    n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
    head_dim = getattr(cfg, 'head_dim', None) or (cfg.d_model // cfg.n_heads)
    if getattr(cfg, 'use_diff_attn', False) or cfg.attn_type == "diff":
        # Differential attention stores two q/k groups per head slot.
        head_dim *= 2
    dtype = next(model.parameters()).dtype
    if device is None:
        device = next(model.parameters()).device

    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is not None:
        n_kv_heads_per_layer = [
            n_kv_heads if (i < len(layer_types)
                           and ("attention" in str(layer_types[i])
                                or "attn" in str(layer_types[i])))
            else 0
            for i in range(cfg.n_layers)
        ]
    else:
        n_kv_heads_per_layer = None

    # SWA ring KV: sliding-window layers cap their buffer at the window —
    # decode length is then bounded only by the full-attention layers.
    ring_windows = None
    if getattr(cfg, "swa_ring_kv", False) and layer_types is not None:
        win = getattr(cfg, "sliding_window", None)
        if win:
            swa = ("attention_swa", "sliding_attention", "swa", "attn_swa")
            ring_windows = [
                win if (i < len(layer_types)
                        and str(layer_types[i]) in swa) else None
                for i in range(cfg.n_layers)
            ]

    return PreAllocatedKVCache(
        n_layers=cfg.n_layers, batch=batch, n_kv_heads=n_kv_heads,
        max_seq_len=min(cfg.max_seq_len, max_total),
        head_dim=head_dim, dtype=dtype, device=device,
        n_kv_heads_per_layer=n_kv_heads_per_layer,
        ring_windows=ring_windows,
    )


def unpack_output_with_kv(out) -> tuple[torch.Tensor, KVCache | None]:
    """Unpack a model forward output into (logits, past_kv).

    Handles:
      - (logits, loss, presents) and (logits, presents) tuple shapes
        emitted by ConfigurableResearchLLM.
      - HuggingFace ModelOutput dataclasses (CausalLMOutputWithPast, etc.)
        which expose .logits and .past_key_values attributes.
    Replaces the 3-4 line ``if isinstance(out, tuple): ...`` block repeated
    across inference paths.
    """
    # HuggingFace ModelOutput (CausalLMOutputWithPast, BaseModelOutputWithPast, etc.)
    if hasattr(out, "logits"):
        logits = out.logits
        past_kv = getattr(out, "past_key_values", None)
        return logits, past_kv
    # Some HF base-model outputs use last_hidden_state instead of logits
    if hasattr(out, "last_hidden_state") and not hasattr(out, "logits"):
        return out.last_hidden_state, getattr(out, "past_key_values", None)
    if isinstance(out, tuple):
        logits = out[0]
        past_kv = out[2] if len(out) > 2 else out[1]
        return logits, past_kv
    return out, None


