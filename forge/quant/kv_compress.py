"""KVQuant + H2O — KV cache compression for long context on 12GB VRAM.

Two complementary techniques:
1. KVQuant: quantize KV cache to 2-3 bits (per-channel keys, per-token values)
2. H2O (Heavy-Hitter Oracle): evict non-essential tokens, keep heavy hitters

Combined: 4-8x KV cache compression → 32K+ context on 12GB GPU.

Usage:
    from forge.quant.kv_compress import KVQuantCache, H2OCache, CompressedKVCache

    # In your attention forward:
    cache = CompressedKVCache(max_tokens=32768, n_heads=16, head_dim=64,
                              kv_bits=2, h2o_keep_ratio=0.2)
    # During generation:
    cache.append(new_k, new_v, token_positions)
    k, v = cache.get()  # returns compressed, evicted KV for attention
"""
from collections import deque

import torch


class KVQuantCache:
    """Quantize KV cache to low-bit integers.

    Keys: per-channel quantization (each channel has its own scale).
    Values: per-token quantization (each token has its own scale).

    Args:
        kv_bits: 2, 3, or 4 (bits per KV element)
    """

    def __init__(self, kv_bits=2):
        self.kv_bits = kv_bits
        self.qmax = 2 ** (kv_bits - 1) - 1  # e.g. 1 for 2-bit, 3 for 3-bit
        self.qmin = -(2 ** (kv_bits - 1))  # e.g. -2 for 2-bit, -4 for 3-bit

    def quantize_keys(self, k):
        """Per-channel key quantization.

        k: (B, n_heads, T, head_dim) → quantized (int8 storage), scales (B, n_heads, 1, head_dim)
        """
        # Per-channel scale: along time dim.
        abs_max = k.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
        scale = abs_max / self.qmax
        k_quant = torch.clamp(torch.round(k / scale), self.qmin, self.qmax).to(torch.int8)
        return k_quant, scale

    def quantize_values(self, v):
        """Per-token value quantization.

        v: (B, n_heads, T, head_dim) → quantized (int8), scales (B, n_heads, T, 1)
        """
        # Per-token scale: along head_dim.
        abs_max = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = abs_max / self.qmax
        v_quant = torch.clamp(torch.round(v / scale), self.qmin, self.qmax).to(torch.int8)
        return v_quant, scale

    def dequantize(self, kv_quant, scale):
        """Dequantize back to float for attention computation.

        Uses bf16 instead of fp32 to save VRAM — the quantized values are
        int8 (max 127), so bf16 has more than enough precision.
        """
        return kv_quant.to(torch.bfloat16) * scale.to(torch.bfloat16)


class H2OCache:
    """H2O (Heavy-Hitter Oracle) token eviction.

    Keeps the most "important" tokens based on cumulative attention scores.
    Evicts least-important tokens when cache exceeds capacity.

    Args:
        max_tokens: maximum tokens to keep in cache
        keep_ratio: fraction of tokens to always keep (heavy hitters)
    """

    def __init__(self, max_tokens=4096, keep_ratio=0.2):
        self.max_tokens = max_tokens
        self.keep_ratio = keep_ratio
        # Per-token importance scores (updated as attention is computed).
        self.scores = deque(maxlen=max_tokens * 2)  # (token_idx, score)
        self.evicted = set()

    def update_scores(self, attention_weights):
        """Update token importance from attention weights.

        attention_weights: (B, n_heads, T_q, T_k) softmaxed attention.
        Sum over heads and query positions → per-key-token importance.
        """
        # (T_k,) — how much attention each key token received.
        importance = attention_weights.sum(dim=(0, 1, 2))
        for i, score in enumerate(importance.tolist()):
            if i < len(self.scores):
                self.scores[i] = (self.scores[i][0], self.scores[i][1] + score)
            else:
                self.scores.append((i, score))

    def get_eviction_mask(self, current_length):
        """Return boolean mask: True = keep, False = evict.

        Evicts tokens with lowest cumulative scores, keeping top keep_ratio.
        """
        if current_length <= self.max_tokens:
            return torch.ones(current_length, dtype=torch.bool)

        int(self.max_tokens * self.keep_ratio)
        n_evict = current_length - self.max_tokens

        # Sort by score, evict lowest n_evict.
        sorted_scores = sorted(range(current_length),
                               key=lambda i: self.scores[i][1] if i < len(self.scores) else 0)
        evict_indices = set(sorted_scores[:n_evict])
        mask = torch.ones(current_length, dtype=torch.bool)
        for idx in evict_indices:
            mask[idx] = False
            self.evicted.add(idx)
        return mask


class CompressedKVCache:
    """Combined KVQuant + H2O cache.

    Drop-in replacement for the standard KV cache in attention layers.

    Args:
        max_tokens: max tokens after eviction (H2O)
        n_heads: number of attention heads
        head_dim: dimension per head
        kv_bits: quantization bits (2, 3, or 4)
        h2o_keep_ratio: fraction of heavy hitters to always keep
    """

    def __init__(self, max_tokens=4096, n_heads=16, head_dim=64,
                 kv_bits=2, h2o_keep_ratio=0.2, device="cuda"):
        self.max_tokens = max_tokens
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device

        self.quantizer = KVQuantCache(kv_bits=kv_bits)
        self.h2o = H2OCache(max_tokens=max_tokens, keep_ratio=h2o_keep_ratio)

        # Storage for quantized KV (grows as tokens are added).
        self.k_quant: torch.Tensor | None = None  # (1, n_heads, T, head_dim) int8
        self.v_quant: torch.Tensor | None = None
        self.k_scale: torch.Tensor | None = None
        self.v_scale: torch.Tensor | None = None
        self.current_length = 0

    def append(self, new_k, new_v):
        """Append new K, V tensors and apply compression.

        new_k, new_v: (B, n_heads, T_new, head_dim) float
        """
        _B, H, T_new, D = new_k.shape
        assert self.n_heads == H and self.head_dim == D

        # Quantize new tokens.
        new_k_quant, new_k_scale = self.quantizer.quantize_keys(new_k)
        new_v_quant, new_v_scale = self.quantizer.quantize_values(new_v)

        if self.k_quant is None:
            self.k_quant = new_k_quant
            self.v_quant = new_v_quant
            self.k_scale = new_k_scale  # (B, H, 1, D) per-channel
            self.v_scale = new_v_scale  # (B, H, T, 1) per-token
        else:
            # Append along time dim.
            self.k_quant = torch.cat([self.k_quant, new_k_quant], dim=2)
            self.v_quant = torch.cat([self.v_quant, new_v_quant], dim=2)
            # Key scale is per-channel (B, H, 1, D) — keep the max across all tokens.
            self.k_scale = torch.maximum(self.k_scale, new_k_scale)
            # Value scale is per-token (B, H, T, 1) — concatenate along time.
            self.v_scale = torch.cat([self.v_scale, new_v_scale], dim=2)

        self.current_length += T_new

        # Evict if over capacity.
        if self.current_length > self.max_tokens:
            self._evict()

    def _evict(self):
        """Apply H2O eviction to reduce cache to max_tokens."""
        mask = self.h2o.get_eviction_mask(self.current_length)
        # Apply mask to KV storage (time dim = 2).
        self.k_quant = self.k_quant[:, :, mask, :]
        self.v_quant = self.v_quant[:, :, mask, :]
        # Key scale is per-channel (B, H, 1, D) — no masking needed.
        # Value scale is per-token (B, H, T, 1) — apply mask.
        self.v_scale = self.v_scale[:, :, mask, :]
        self.current_length = mask.sum().item()

    def get(self):
        """Return dequantized K, V for attention computation.

        Returns:
            k: (1, n_heads, T, head_dim) float32
            v: (1, n_heads, T, head_dim) float32
        """
        if self.k_quant is None:
            return None, None
        k = self.quantizer.dequantize(self.k_quant, self.k_scale)
        v = self.quantizer.dequantize(self.v_quant, self.v_scale)
        return k.to(self.device), v.to(self.device)

    def memory_usage_mb(self):
        """Estimate current memory usage in MB."""
        if self.k_quant is None:
            return 0.0
        # int8 storage + float32 scales.
        kv_bytes = self.k_quant.numel() + self.v_quant.numel()  # int8 = 1 byte
        scale_bytes = (self.k_scale.numel() + self.v_scale.numel()) * 4  # float32
        return (kv_bytes + scale_bytes) / (1024 * 1024)

    def compression_ratio(self):
        """Compression ratio vs FP16 KV cache."""
        if self.k_quant is None:
            return 1.0
        fp16_size = self.current_length * self.n_heads * self.head_dim * 2 * 2  # K+V, fp16
        actual_size = (self.k_quant.numel() + self.v_quant.numel() +
                       (self.k_scale.numel() + self.v_scale.numel()) * 4)
        return fp16_size / actual_size
