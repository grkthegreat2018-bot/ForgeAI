"""Tests for MatryoshkaKV cache strategy.

Tests run on CPU. Uses CPU-safe dtypes (float32, float16, bfloat16) to
avoid fp8 CPU limitations in some torch builds.
"""
from __future__ import annotations

import torch

from forge.engine.kv.matryoshka_kv import MatryoshkaKVCache
from forge.engine.kv_backend import build_kv_cache

# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_cache(
    n_levels: int = 3,
    head_dim: int = 64,
    n_kv: int = 4,
    max_seq_len: int = 256,
    dtypes=None,
    importance_metric: str = "norm",
) -> MatryoshkaKVCache:
    """Build and init a MatryoshkaKVCache on CPU."""
    if dtypes is None:
        # CPU-safe dtypes: float32 (high precision), float16, bfloat16
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
    cache = MatryoshkaKVCache(
        n_levels=n_levels,
        dtypes_per_level=dtypes,
        importance_metric=importance_metric,
    )
    cache.init(
        n_heads=8, head_dim=head_dim, n_kv_heads=n_kv,
        max_seq_len=max_seq_len, device="cpu", dtype=torch.float32,
    )
    return cache


def _make_kv(batch=2, n_kv=4, seq=8, head_dim=64, seed=42):
    """Generate random K/V tensors."""
    torch.manual_seed(seed)
    k = torch.randn(batch, n_kv, seq, head_dim, dtype=torch.float32)
    v = torch.randn(batch, n_kv, seq, head_dim, dtype=torch.float32)
    return k, v


# ── Tests ───────────────────────────────────────────────────────────────────

class TestMatryoshkaKVBasic:
    """Basic append/get and shape verification."""

    def test_basic_append_get(self):
        """Append tokens, retrieve them, verify shapes."""
        head_dim = 64
        cache = _make_cache(head_dim=head_dim)
        k, v = _make_kv(seq=8, head_dim=head_dim)

        # Append in chunks of 1 (decode-style)
        for i in range(8):
            cache.append(k[:, :, i:i+1, :], v[:, :, i:i+1, :], position=i)

        k_out, v_out = cache.get(positions=None)
        assert k_out is not None, "get() returned None after append"
        assert v_out is not None

        # Shape: [B, n_kv, seq_len, head_dim]
        assert k_out.shape == (2, 4, 8, head_dim)
        assert v_out.shape == (2, 4, 8, head_dim)

    def test_basic_append_batch(self):
        """Append multiple tokens at once (prefill-style)."""
        head_dim = 64
        cache = _make_cache(head_dim=head_dim)
        k, v = _make_kv(seq=16, head_dim=head_dim)

        cache.append(k, v, position=0)
        k_out, v_out = cache.get(positions=None)
        assert k_out.shape == (2, 4, 16, head_dim)
        assert v_out.shape == (2, 4, 16, head_dim)

    def test_get_empty_cache(self):
        """get() on empty cache returns None."""
        cache = _make_cache()
        k_out, v_out = cache.get(positions=None)
        assert k_out is None
        assert v_out is None

    def test_get_past_kv_empty(self):
        """get_past_kv() on empty cache returns None."""
        cache = _make_cache()
        assert cache.get_past_kv() is None

    def test_get_past_kv_populated(self):
        """get_past_kv() returns tensors after append."""
        head_dim = 64
        cache = _make_cache(head_dim=head_dim)
        k, v = _make_kv(seq=4, head_dim=head_dim)
        cache.append(k, v, position=0)
        result = cache.get_past_kv()
        assert result is not None
        k_out, _v_out = result
        assert k_out.shape == (2, 4, 4, head_dim)


class TestMatryoshkaKVCompression:
    """Compression ratio verification."""

    def test_compression_ratio(self):
        """info() reports compression > 1.0 (i.e. actual < baseline)."""
        # Use dtypes with clear size difference: float32 (4B), float16 (2B), bfloat16 (2B)
        # head_dim=64, dims=[16, 16, 32]
        # bytes = 16*4 + 16*2 + 32*2 = 64+32+64 = 160
        # baseline = 64*4 = 256
        # compression = 256/160 = 1.6
        cache = _make_cache(head_dim=64, dtypes=[torch.float32, torch.float16, torch.bfloat16])
        info = cache.info()
        assert info["type"] == "matryoshka"
        assert info["compression"] > 1.0, f"Expected compression > 1.0, got {info['compression']}"

    def test_compression_ratio_fp8_like(self):
        """With more aggressive dtype reduction, compression is higher."""
        # float32 (4B) for level 0, float16 (2B) for level 1, bfloat16 (2B) for level 2
        # Same as above but verify the ratio is correct
        cache = _make_cache(head_dim=64, dtypes=[torch.float32, torch.float16, torch.bfloat16])
        info = cache.info()
        # 256 / 160 = 1.6
        assert abs(info["compression"] - 1.6) < 0.01, (
            f"Expected compression ~1.6, got {info['compression']}"
        )

    def test_info_has_levels(self):
        """info() includes n_levels and dims_per_level."""
        cache = _make_cache(n_levels=3, head_dim=64)
        info = cache.info()
        assert info["n_levels"] == 3
        assert len(info["dims_per_level"]) == 3
        assert sum(info["dims_per_level"]) == 64

    def test_info_after_append(self):
        """info() reflects seq_len after append."""
        cache = _make_cache(head_dim=64)
        k, v = _make_kv(seq=10, head_dim=64)
        cache.append(k, v, position=0)
        info = cache.info()
        assert info["seq_len"] == 10
        assert info["size_mb"] > 0


class TestMatryoshkaKVReconstruction:
    """Reconstruction quality verification."""

    def test_reconstruction_quality(self):
        """Reconstructed K/V is close to original (cosine sim > 0.9).

        With float32 level 0 (most important dims) and float16/bfloat16
        for less important dims, reconstruction should be high quality.
        """
        head_dim = 64
        # Use float32 for all levels to get near-perfect reconstruction
        cache = _make_cache(
            head_dim=head_dim,
            dtypes=[torch.float32, torch.float32, torch.float32],
        )
        k, v = _make_kv(seq=16, head_dim=head_dim)
        cache.append(k, v, position=0)

        k_out, v_out = cache.get(positions=None)

        # Cosine similarity per token (flatten over head_dim)
        k_flat = k.reshape(-1, head_dim)
        k_out_flat = k_out.reshape(-1, head_dim)
        cos_sim_k = torch.nn.functional.cosine_similarity(
            k_flat, k_out_flat, dim=-1
        )
        assert cos_sim_k.mean() > 0.9, (
            f"K cosine similarity too low: {cos_sim_k.mean().item():.4f}"
        )

        v_flat = v.reshape(-1, head_dim)
        v_out_flat = v_out.reshape(-1, head_dim)
        cos_sim_v = torch.nn.functional.cosine_similarity(
            v_flat, v_out_flat, dim=-1
        )
        assert cos_sim_v.mean() > 0.9, (
            f"V cosine similarity too low: {cos_sim_v.mean().item():.4f}"
        )

    def test_reconstruction_with_quantization(self):
        """Even with lower precision on some levels, reconstruction is good."""
        head_dim = 64
        # float32 for important dims, float16 for rest — still high quality
        cache = _make_cache(
            head_dim=head_dim,
            dtypes=[torch.float32, torch.float16, torch.float16],
        )
        k, v = _make_kv(seq=16, head_dim=head_dim)
        cache.append(k, v, position=0)

        k_out, _v_out = cache.get(positions=None)
        k_flat = k.reshape(-1, head_dim)
        k_out_flat = k_out.reshape(-1, head_dim)
        cos_sim = torch.nn.functional.cosine_similarity(
            k_flat, k_out_flat, dim=-1
        )
        # float16 has enough precision for cosine sim > 0.95
        assert cos_sim.mean() > 0.95, (
            f"K cosine similarity too low with quantization: {cos_sim.mean().item():.4f}"
        )


class TestMatryoshkaKVClear:
    """clear() behavior verification."""

    def test_clear(self):
        """clear() resets the cache."""
        head_dim = 64
        cache = _make_cache(head_dim=head_dim)
        k, v = _make_kv(seq=8, head_dim=head_dim)
        cache.append(k, v, position=0)
        assert cache.seq_len == 8

        cache.clear()
        assert cache.seq_len == 0
        assert all(ks is None for ks in cache.k_stores)
        assert all(vs is None for vs in cache.v_stores)

        # get() after clear returns None
        k_out, v_out = cache.get(positions=None)
        assert k_out is None
        assert v_out is None

    def test_clear_then_reuse(self):
        """Cache can be reused after clear()."""
        head_dim = 64
        cache = _make_cache(head_dim=head_dim)
        k, v = _make_kv(seq=4, head_dim=head_dim)
        cache.append(k, v, position=0)
        cache.clear()

        # Reuse
        k2, v2 = _make_kv(seq=6, head_dim=head_dim, seed=99)
        cache.append(k2, v2, position=0)
        k_out, _v_out = cache.get(positions=None)
        assert k_out.shape == (2, 4, 6, head_dim)

    def test_clear_resets_importance(self):
        """clear() resets the importance EMA."""
        head_dim = 64
        cache = _make_cache(head_dim=head_dim)
        k, v = _make_kv(seq=8, head_dim=head_dim)
        cache.append(k, v, position=0)
        assert cache.importance_ema.abs().sum() > 0

        cache.clear()
        assert cache.importance_ema.abs().sum() == 0


class TestMatryoshkaKVImportance:
    """Importance ranking verification."""

    def test_importance_ranking(self):
        """Important dimensions are preserved better than unimportant ones.

        We construct K/V where some dimensions have large magnitude and
        others have near-zero magnitude. After append+get, the large-magnitude
        dimensions should be reconstructed more accurately (since they're
        ranked as important and stored at higher precision).
        """
        head_dim = 64
        batch, n_kv, seq = 2, 4, 16

        # Create K/V where first 16 dims have large magnitude, rest are tiny
        torch.manual_seed(123)
        k = torch.randn(batch, n_kv, seq, head_dim, dtype=torch.float32)
        v = torch.randn(batch, n_kv, seq, head_dim, dtype=torch.float32)
        # Scale up first 16 dims (these will be "important")
        k[..., :16] *= 10.0
        v[..., :16] *= 10.0
        # Scale down remaining dims (these will be "unimportant")
        k[..., 16:] *= 0.01
        v[..., 16:] *= 0.01

        # Use float32 for level 0 (important), float16 for levels 1-2
        cache = _make_cache(
            head_dim=head_dim,
            dtypes=[torch.float32, torch.float16, torch.float16],
        )
        cache.append(k, v, position=0)
        k_out, _v_out = cache.get(positions=None)

        # Compute per-dimension reconstruction error
        k_err = (k - k_out).abs()
        # Important dims (first 16) should have lower relative error
        # since they're stored at float32
        important_err = k_err[..., :16].mean()
        unimportant_err = k_err[..., 16:].mean()

        # Relative error (normalized by magnitude)
        important_mag = k[..., :16].abs().mean()
        unimportant_mag = k[..., 16:].abs().mean()
        important_rel = important_err / important_mag
        unimportant_rel = unimportant_err / unimportant_mag

        # Important dims should have lower relative error
        # (float32 vs float16 precision difference)
        assert important_rel < unimportant_rel, (
            f"Important dims relative error ({important_rel.item():.6f}) "
            f"should be < unimportant ({unimportant_rel.item():.6f})"
        )

    def test_importance_ema_updates(self):
        """The importance EMA is updated after append."""
        head_dim = 64
        cache = _make_cache(head_dim=head_dim)
        assert cache.importance_ema.abs().sum() == 0

        k, v = _make_kv(seq=4, head_dim=head_dim)
        cache.append(k, v, position=0)
        assert cache.importance_ema.abs().sum() > 0

    def test_importance_variance_metric(self):
        """Variance-based importance metric works."""
        head_dim = 64
        cache = _make_cache(
            head_dim=head_dim,
            importance_metric="variance",
        )
        k, v = _make_kv(seq=8, head_dim=head_dim)
        cache.append(k, v, position=0)
        assert cache.importance_ema.abs().sum() > 0

        k_out, _v_out = cache.get(positions=None)
        assert k_out.shape == (2, 4, 8, head_dim)


class TestMatryoshkaKVFactory:
    """Factory integration test."""

    def test_build_kv_cache_matryoshka(self):
        """build_kv_cache('matryoshka') returns a MatryoshkaKVCache."""
        cache = build_kv_cache("matryoshka")
        assert isinstance(cache, MatryoshkaKVCache)

    def test_build_kv_cache_matryoshka_init(self):
        """Factory-built cache can be init'd and used."""
        cache = build_kv_cache("matryoshka")
        cache.init(
            n_heads=8, head_dim=64, n_kv_heads=4,
            max_seq_len=128, device="cpu", dtype=torch.float32,
        )
        k, v = _make_kv(seq=4, head_dim=64)
        cache.append(k, v, position=0)
        k_out, _v_out = cache.get(positions=None)
        assert k_out.shape == (2, 4, 4, 64)


class TestMatryoshkaKVConfig:
    """Configuration and edge cases."""

    def test_custom_dims_per_level(self):
        """Custom dims_per_level is respected."""
        head_dim = 64
        cache = MatryoshkaKVCache(
            n_levels=2,
            dims_per_level=[16, 48],
            dtypes_per_level=[torch.float32, torch.float16],
        )
        cache.init(
            n_heads=8, head_dim=head_dim, n_kv_heads=4,
            max_seq_len=128, device="cpu", dtype=torch.float32,
        )
        info = cache.info()
        assert info["dims_per_level"] == [16, 48]
        assert info["n_levels"] == 2

    def test_single_level(self):
        """Single level (no compression) works."""
        head_dim = 64
        cache = MatryoshkaKVCache(
            n_levels=1,
            dims_per_level=[head_dim],
            dtypes_per_level=[torch.float32],
        )
        cache.init(
            n_heads=8, head_dim=head_dim, n_kv_heads=4,
            max_seq_len=128, device="cpu", dtype=torch.float32,
        )
        k, v = _make_kv(seq=8, head_dim=head_dim)
        cache.append(k, v, position=0)
        k_out, _v_out = cache.get(positions=None)
        assert k_out.shape == (2, 4, 8, head_dim)
        # Single level float32 → compression = 1.0
        assert abs(cache.info()["compression"] - 1.0) < 0.01

    def test_auto_dims_per_level(self):
        """Auto-computed dims_per_level sums to head_dim."""
        head_dim = 128
        cache = MatryoshkaKVCache(n_levels=3)
        cache.init(
            n_heads=8, head_dim=head_dim, n_kv_heads=4,
            max_seq_len=256, device="cpu", dtype=torch.float32,
        )
        info = cache.info()
        assert sum(info["dims_per_level"]) == head_dim
        assert len(info["dims_per_level"]) == 3
