"""Tests for TurboQuant+ extensions on RotorQuantCache.

Covers:
  - Asymmetric K/V bits (K precision dominates; V tolerates lower bits)
  - Boundary-V (first/last layers keep higher-precision V)
  - Sparse-V dequant skip (positions with attn < threshold skip dequant)
  - RotorQuantKVCache strategy wiring (kv_backend) + honest info()

CPU-only, small dims.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from forge.engine.kv_backend import RotorQuantKVCache, build_kv_cache
from forge.quant.rotorquant import RotorQuantCache


def _mk(bits=4, **kw):
    return RotorQuantCache(
        d_model=64, n_heads=4, head_dim=16, bits=bits, max_seq_len=64, n_kv_heads=2, device="cpu", **kw
    )


def test_symmetric_bits_unchanged():
    """Default: bits_v == bits — identical to legacy behavior."""
    c = _mk(bits=4)
    assert c.bits_v == 4
    assert c.codebook_v is c.codebook_k  # shared table, no extra cost
    torch.manual_seed(0)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    c.append(k, v)
    c.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    k2, v2 = c.get()
    assert k2.shape == (1, 2, 9, 16) and v2.shape == (1, 2, 9, 16)


def test_asymmetric_v_bits():
    """bits_v=3 gives V its own coarser codebook."""
    c = _mk(bits=4, bits_v=3)
    assert c.bits_v == 3
    assert c.codebook_v.n_levels == 8
    assert c.codebook_k.n_levels == 16
    torch.manual_seed(0)
    c.append(torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16))
    c.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    assert c._v_idx.max().item() < 8  # 3-bit indices
    assert c._k_idx.max().item() < 16  # 4-bit indices


def test_boundary_v_layers():
    """First/last ``boundary_layers`` layers get boundary_v_bits for V."""
    c_edge = _mk(bits=4, bits_v=3, layer_idx=0, n_layers=8, boundary_v_bits=8)
    assert c_edge.bits_v == 8  # boundary layer → high-precision V
    c_mid = _mk(bits=4, bits_v=3, layer_idx=4, n_layers=8, boundary_v_bits=8)
    assert c_mid.bits_v == 3  # interior layer → coarse V
    c_last = _mk(bits=4, bits_v=3, layer_idx=7, n_layers=8, boundary_v_bits=8)
    assert c_last.bits_v == 8


def test_boundary_v_disabled_without_layer_info():
    """No layer_idx/n_layers → boundary_v_bits is inert, not silently
    applied to every layer."""
    c = _mk(bits=4, bits_v=3, boundary_v_bits=8)
    assert c.bits_v == 3


def test_sparse_v_skips_low_attn_positions():
    """Positions with attn weight < threshold keep zero V rows and are
    not dequantized."""
    c = _mk(bits=4)
    torch.manual_seed(0)
    v = torch.randn(1, 2, 10, 16)
    c.append(torch.randn(1, 2, 10, 16), v)
    c.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    T = 11
    attn = torch.full((1, 2, T), 1e-9)  # (B, Hkv, T)
    attn[..., 3] = 0.9  # only position 3 matters
    attn[..., 7] = 0.05
    v_hat, keep = c.get_sparse_v(attn, threshold=1e-6)
    assert keep.shape == (T,)
    assert bool(keep[3]) and bool(keep[7])
    assert int(keep.sum()) == 2
    assert v_hat[:, :, ~keep].abs().max() == 0
    # attn @ v_hat ≈ attn @ v (skipped positions contribute <1e-6 * |v|)
    v_hat = v_hat.float()
    out_sparse = torch.einsum("bht,bhtd->bhd", attn, v_hat)
    _k2, v2 = c.get()
    out_full = torch.einsum("bht,bhtd->bhd", attn, v2.float())
    assert torch.allclose(out_sparse, out_full, atol=1e-3)
    assert c._sparse_skipped == T - 2
    assert c._sparse_kept == 2


def test_sparse_v_all_kept_matches_get():
    c = _mk(bits=4)
    torch.manual_seed(0)
    c.append(torch.randn(1, 2, 6, 16), torch.randn(1, 2, 6, 16))
    c.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    attn = torch.ones(1, 2, 7)
    v_hat, _keep = c.get_sparse_v(attn)
    _, v2 = c.get()
    assert torch.equal(v_hat, v2)


def test_asymmetric_compression_ratio_honest():
    """Compression reflects bits + bits_v, not 2x bits."""
    c = _mk(bits=4, bits_v=3)
    torch.manual_seed(0)
    c.append(torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16))
    c.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    sym = _mk(bits=4)
    torch.manual_seed(0)
    sym.append(torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16))
    sym.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    assert c.compression_ratio() > sym.compression_ratio()


def test_strategy_wires_bits_and_reports_effective():
    """build_kv_cache kwargs reach the cache; info() reports real bits."""
    cache = build_kv_cache("rotorquant", bits=4, bits_v=3)
    assert isinstance(cache, RotorQuantKVCache)
    cache.init(n_heads=4, head_dim=16, n_kv_heads=2, max_seq_len=32, device="cpu", dtype=torch.float32)
    info = cache.info()
    assert info["bits_k"] == 4 and info["bits_v"] == 3
    # append/get roundtrip through the strategy interface
    torch.manual_seed(0)
    cache.append(torch.randn(1, 2, 4, 16), torch.randn(1, 2, 4, 16), 0)
    cache.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), 4)
    k, v = cache.get(None)
    assert k.shape[2] == 5 and v.shape[2] == 5


def test_strategy_unknown_kwarg_fails_loudly():
    """kwargs the strategy doesn't accept raise — not silently dropped."""
    try:
        build_kv_cache("standard", bogus_kwarg=1)
    except TypeError:
        return
    raise AssertionError("expected TypeError for unknown kwarg")


def test_clear_resets():
    c = _mk(bits=4)
    torch.manual_seed(0)
    c.append(torch.randn(1, 2, 4, 16), torch.randn(1, 2, 4, 16))
    c.append(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16))
    assert len(c) == 5
    c.clear()
    assert len(c) == 0
    assert c.get() == (None, None)
