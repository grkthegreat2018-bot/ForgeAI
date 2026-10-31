"""FlexDecoding: PyTorch's fused FlashDecoding backend for decode-phase attention.

PyTorch 2.5+ introduced FlexAttention with a dedicated decode backend
(FlexDecoding). When torch.compile lowers flex_attention with a short query
(q_len=1) and long KV cache, it generates a fused FlashDecoding kernel that:

  1. Splits the KV cache across SMs for parallelism (low-head-count decode
     underutilizes SMs without splitting — the key bottleneck on small models)
  2. Fuses the entire attention computation (QK^T → softmax → AV) into one
     kernel with no HBM round-trips for intermediate scores
  3. Supports GQA, paged KV, and custom score modifications (RoPE, etc.)

For our 1.2B model on RTX 5070 (192 SMs, 8 KV heads):
  - Standard SDPA decode: 8 KV heads × 1 SM = 8 SMs active (4% utilization)
  - FlexDecoding: splits KV across all 192 SMs → 100% utilization
  - Expected: 1.5-3× decode speedup for batch=1

Usage:
    from forge.engine.attention.flex_decoding import FlexDecodingAttention
    # Replace SDPA in GroupedQueryAttention.forward with FlexDecodingAttention
    # Or use as a drop-in via ForgeEngine.activate(acceleration="flex_decoding")
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Check if flex_attention is available (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: F401
    _FLEX_AVAILABLE = True
except ImportError:
    _FLEX_AVAILABLE = False


def flex_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    score_mod=None,
    block_mask=None,
) -> torch.Tensor:
    """Decode-phase attention using FlexDecoding backend.

    When q_len=1 (decode step), flex_attention automatically switches to the
    FlexDecoding kernel which splits the KV cache across SMs for parallelism.

    Args:
        q: (B, n_heads, 1, head_dim) — single token query
        k: (B, n_kv_heads, S, head_dim) — full KV cache keys
        v: (B, n_kv_heads, S, head_dim) — full KV cache values
        score_mod: optional score modification function (e.g., for RoPE)
        block_mask: optional block mask for sparse attention

    Returns:
        out: (B, n_heads, 1, head_dim) — attention output
    """
    if not _FLEX_AVAILABLE:
        # Fallback to SDPA if flex_attention not available
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Repeat KV heads to match Q heads (GQA)
    n_heads = q.shape[1]
    n_kv = k.shape[1]
    if n_kv != n_heads:
        n_rep = n_heads // n_kv
        k = k[:, :, None, :, :].expand(q.shape[0], n_kv, n_rep, k.shape[2], k.shape[3])
        k = k.reshape(q.shape[0], n_heads, k.shape[2], k.shape[3])
        v = v[:, :, None, :, :].expand(q.shape[0], n_kv, n_rep, v.shape[2], v.shape[3])
        v = v.reshape(q.shape[0], n_heads, v.shape[2], v.shape[3])

    # flex_attention with compile — auto-switches to FlexDecoding for q_len=1
    compiled_flex = torch.compile(flex_attention, dynamic=False)
    out = compiled_flex(q, k, v, score_mod=score_mod, block_mask=block_mask)
    return out


class FlexDecodingWrapper:
    """Wraps a model's attention to use FlexDecoding for decode steps.

    Patches GroupedQueryAttention.forward to use flex_attention when q_len=1.
    Prefill (q_len > 1) still uses standard SDPA/flash_attention.
    """

    def __init__(self):
        self._active = False
        self._original_forwards = {}

    def apply(self, model: torch.nn.Module):
        """Patch all attention layers in the model to use FlexDecoding."""
        if not _FLEX_AVAILABLE:
            logger.warning("  [FlexDecoding] flex_attention not available (need PyTorch 2.5+)")
            return False

        from forge.model_loader import GroupedQueryAttention
        for name, module in model.named_modules():
            if isinstance(module, GroupedQueryAttention) or type(module).__name__ in ("GroupedTiedAttention", "GroupedLatentAttention"):
                self._patch_attention(module, name)

        self._active = True
        logger.info(f"  [FlexDecoding] Patched {len(self._original_forwards)} attention layers")
        return True

    def _patch_attention(self, attn_module, name: str):
        """Patch a single attention module to use FlexDecoding for decode."""
        original_forward = attn_module.forward
        self._original_forwards[name] = original_forward

        def flex_forward(self, x, past_key_value=None, use_cache=False,
                         preallocated_cache=None, layer_idx=0,
                         attention_bias=None, position_ids=None):
            _B, T, _C = x.shape
            # Only use FlexDecoding for single-token decode (T=1)
            if T == 1 and preallocated_cache is not None and attention_bias is None:
                return self._flex_decode_forward(
                    x, preallocated_cache, layer_idx, position_ids, use_cache)
            # Prefill or masked attention: use original forward
            return original_forward(
                x, past_key_value=past_key_value, use_cache=use_cache,
                preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                attention_bias=attention_bias, position_ids=position_ids)

        # Bind the flex_decode_forward method
        attn_module._flex_decode_forward = self._make_flex_decode_forward(attn_module)
        attn_module.forward = flex_forward.__get__(attn_module, type(attn_module))

    def _make_flex_decode_forward(self, attn_module):
        """Create a flex_decode forward method for a specific attention module."""
        def _flex_decode_forward(x, preallocated_cache, layer_idx, position_ids, use_cache):
            B, T, _C = x.shape
            hd = attn_module.head_dim

            # Project Q/K/V (same as original)
            q = attn_module.q_proj(x).view(B, T, attn_module.n_heads, hd).transpose(1, 2)
            k = attn_module.k_proj(x).view(B, T, attn_module.n_kv_heads, hd).transpose(1, 2)

            # Handle V (GTA: V=K in identity mode)
            if hasattr(attn_module, '_identity') and attn_module._identity:
                v = k
            else:
                v = attn_module.v_proj(x).view(B, T, attn_module.n_kv_heads, hd).transpose(1, 2)

            # QK-norm
            if attn_module.use_qk_norm and not getattr(attn_module, '_qk_norm_identity', True):
                q = attn_module.q_norm(q)
                k = attn_module.k_norm(k)

            # RoPE
            past_len = preallocated_cache.position
            q = attn_module.rope(q, offset=past_len, position_ids=position_ids)
            k = attn_module.rope(k, offset=past_len, position_ids=position_ids)

            # Append to cache
            preallocated_cache.append(layer_idx, k, v)
            k_cache = preallocated_cache.k_caches[layer_idx][:, :, :past_len + T]
            v_cache = preallocated_cache.v_caches[layer_idx][:, :, :past_len + T]

            new_kv = (k_cache, v_cache) if use_cache else None

            # FlexDecoding: use flex_attention for the decode step
            out = flex_decode(q, k_cache, v_cache)
            if getattr(attn_module, '_finish_attn', None) is not None:
                return attn_module._finish_attn(x, out), new_kv
            out = out.transpose(1, 2).reshape(B, T, -1)
            return attn_module.out_proj(out), new_kv

        return _flex_decode_forward

    def revert(self, model: torch.nn.Module):
        """Revert all attention layers to their original forward methods."""
        for name, module in model.named_modules():
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
        self._original_forwards.clear()
        self._active = False
