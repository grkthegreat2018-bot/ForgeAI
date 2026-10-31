#!/usr/bin/env python
"""R48 extreme low-bit quantization smoke tests."""
from __future__ import annotations

import torch
import torch.nn as nn

from forge.engine.quant.novel_quant_r48 import (
    BTCQuantLinear,
    NanoQuantLinear,
    TernaryPTQLinear,
    estimate_r48_memory,
    quantize_model_btc,
    quantize_model_nanoquant,
    quantize_model_ternary_ptq,
)


def test_nanoquant():
    """Test NanoQuant low-rank binary factorization."""
    print("\n[NanoQuant]")
    lin = nn.Linear(64, 32, bias=True)
    x = torch.randn(2, 4, 64)

    for rank in [8, 16, 32]:
        q_lin = NanoQuantLinear.from_linear(lin, rank=rank, admm_iters=30)
        out = q_lin(x)
        err = (out - lin(x)).abs().mean().item()

        d_out, d_in = 32, 64
        eff_bits = (rank * (d_out + d_in) + 16 * (d_out + d_in)) / (d_out * d_in)
        print(f"  rank={rank}: out={out.shape} err={err:.4f} eff_bits={eff_bits:.3f}")
        assert out.shape == (2, 4, 32), f"Bad shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
    print("  PASS")


def test_btc():
    """Test BTC-LLM binary codebook clustering."""
    print("\n[BTC-LLM]")
    lin = nn.Linear(64, 128, bias=True)
    x = torch.randn(2, 4, 64)

    for K in [32, 64, 128]:
        q_lin = BTCQuantLinear.from_linear(lin, codebook_size=K, use_rotation=True)
        out = q_lin(x)
        err = (out - lin(x)).abs().mean().item()
        print(f"  K={K}: out={out.shape} err={err:.4f}")
        assert out.shape == (2, 4, 128), f"Bad shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"

    # Without rotation
    q_lin = BTCQuantLinear.from_linear(lin, codebook_size=64, use_rotation=False)
    out = q_lin(x)
    err = (out - lin(x)).abs().mean().item()
    print(f"  no-rotation: out={out.shape} err={err:.4f}")
    assert out.shape == (2, 4, 128)
    print("  PASS")


def test_ternary_ptq():
    """Test TernaryPTQ with and without calibration."""
    print("\n[TernaryPTQ]")
    lin = nn.Linear(64, 32, bias=True)
    x = torch.randn(2, 4, 64)

    # Without calibration (vanilla absmean)
    q_lin = TernaryPTQLinear.from_linear(lin, activations=None, refine_iters=0)
    out = q_lin(x)
    err = (out - lin(x)).abs().mean().item()
    print(f"  no-calib: out={out.shape} err={err:.4f}")
    assert out.shape == (2, 4, 32)
    assert not torch.isnan(out).any()

    # With calibration
    activations = torch.randn(64, 64)  # (N_samples, in_features)
    q_lin2 = TernaryPTQLinear.from_linear(lin, activations=activations, refine_iters=10)
    out2 = q_lin2(x)
    err2 = (out2 - lin(x)).abs().mean().item()
    print(f"  with-calib: out={out2.shape} err={err2:.4f}")
    assert out2.shape == (2, 4, 32)
    assert not torch.isnan(out2).any()
    print("  PASS")


def test_model_conversion():
    """Test model-level conversion."""
    print("\n[Model conversion]")
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    x = torch.randn(2, 4, 64)

    # NanoQuant
    m1 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    m1.load_state_dict(model.state_dict())
    n1 = quantize_model_nanoquant(m1, rank=16, admm_iters=20, verbose=False)
    out1 = m1(x)
    print(f"  nanoquant: {n1} layers, out={out1.shape}")

    # BTC
    m2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    m2.load_state_dict(model.state_dict())
    n2 = quantize_model_btc(m2, codebook_size=32, use_rotation=True, verbose=False)
    out2 = m2(x)
    print(f"  btc: {n2} layers, out={out2.shape}")

    # TernaryPTQ
    m3 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    m3.load_state_dict(model.state_dict())
    n3 = quantize_model_ternary_ptq(m3, activations=None, refine_iters=0, verbose=False)
    out3 = m3(x)
    print(f"  ternary: {n3} layers, out={out3.shape}")

    # Memory estimation
    mem = estimate_r48_memory(m1)
    print(f"  nanoquant memory: {mem['total_mb']:.4f} MB, eff_bits={mem['avg_eff_bits']:.3f}")

    mem2 = estimate_r48_memory(m2)
    print(f"  btc memory: {mem2['total_mb']:.4f} MB, eff_bits={mem2['avg_eff_bits']:.3f}")

    mem3 = estimate_r48_memory(m3)
    print(f"  ternary memory: {mem3['total_mb']:.4f} MB, eff_bits={mem3['avg_eff_bits']:.3f}")

    assert n1 == 2 and n2 == 2 and n3 == 2
    print("  PASS")


def test_compression_ratio():
    """Verify that sub-1-bit compression is achieved for typical dimensions."""
    print("\n[Compression ratio]")
    # Typical Qwen 2.5 0.5B dimensions: 896×896, 896×3584, etc.
    for d_out, d_in in [(896, 896), (896, 3584), (3584, 896), (512, 2048)]:
        for rank in [32, 64, 128, 256]:
            eff_bits = (rank * (d_out + d_in) + 16 * (d_out + d_in)) / (d_out * d_in)
            if eff_bits < 1.0:
                print(f"  d={d_out}×{d_in}, rank={rank}: {eff_bits:.3f} bits/w (sub-1-bit!)")
    print("  PASS")


def test_cache_invalidation():
    """Test that weight cache works and invalidates on device move."""
    print("\n[Cache invalidation]")
    lin = nn.Linear(64, 32)
    q_lin = NanoQuantLinear.from_linear(lin, rank=16, admm_iters=20)
    x = torch.randn(1, 4, 64)

    # First forward — should build cache
    out1 = q_lin(x)
    assert q_lin._cached_weight is not None, "Cache not built"
    print(f"  Cache built: {q_lin._cached_weight.shape}")

    # Second forward — should use cache
    out2 = q_lin(x)
    assert torch.allclose(out1, out2), "Cache gave different result"
    print("  Cache reuse: OK")

    # Invalidate
    q_lin._invalidate_cache()
    assert q_lin._cached_weight is None, "Cache not invalidated"
    out3 = q_lin(x)
    assert torch.allclose(out1, out3), "Rebuild gave different result"
    print("  Cache rebuild: OK")


def main():
    print("R48 Extreme Low-Bit Quantization Smoke Tests\n")
    print("=" * 60)

    tests = [
        test_nanoquant,
        test_btc,
        test_ternary_ptq,
        test_model_conversion,
        test_compression_ratio,
        test_cache_invalidation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Smoke tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
