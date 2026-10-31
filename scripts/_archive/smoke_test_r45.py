#!/usr/bin/env python
"""R45 Novel Quantization Smoke Test — CPU, fast.

Tests that all R45 algorithms:
  - Produce correct output shapes
  - Have reasonable reconstruction error
  - Can convert a small model
  - Memory estimation works
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from forge.engine.quant.novel_quant_r45 import (
    LloydMaxRotatedKVQuantizer,
    SchurABFP4Linear,
    SVDLiftBinaryLinear,
    WaveletLiftLinear,
    _haar_matrix,
    estimate_r45_memory,
    quantize_model_schur_ab_fp4,
    quantize_model_svd_lift_binary,
    quantize_model_wavelet_lift,
    requant_refine_model,
)


def test_haar_orthonormal():
    """Haar matrix should be orthonormal: H @ H^T = I."""
    H = _haar_matrix(8, torch.device('cpu'), torch.float32)
    I = H @ H.T
    assert torch.allclose(I, torch.eye(8), atol=1e-5), f"Haar not orthonormal: {I}"
    print("  Haar 8x8 orthonormal: OK")


def test_haar_frequency_separation():
    """Haar should separate low-freq (first rows) from high-freq (later rows)."""
    # Create a low-frequency signal (constant + slow ramp)
    x = torch.ones(1, 8) * 5.0  # constant signal → all energy in first coefficient
    H = _haar_matrix(8, torch.device('cpu'), torch.float32)
    x_rot = x @ H.T  # forward transform: project onto rows of H (basis vectors)
    # For a constant signal, only the first coefficient (average) should be nonzero
    assert x_rot[0, 0].abs() > 1.0, f"Average coefficient should dominate: {x_rot}"
    assert x_rot[0, 1:].abs().max() < 1e-5, \
        f"High-freq should be ~0 for constant signal: {x_rot[0,1:]}"
    print(f"  Haar frequency separation: OK (avg={x_rot[0,0]:.3f}, high_max={x_rot[0,1:].abs().max():.6f})")


def test_wavelet_lift():
    """WaveletLift: shape + reconstruction error."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=True)
    layer = WaveletLiftLinear.from_linear(lin, rank=32)

    x = torch.randn(2, 4, 128)
    out = layer(x)
    assert out.shape == (2, 4, 64), f"Bad shape: {out.shape}"

    # Reconstruction error
    w_orig = lin.weight.data.float()
    w_recon = layer._dequantize_weight(torch.float32)
    err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
    print(f"  WaveletLift rank=32: out={out.shape} err={err:.4f}")
    assert err < 1.0, f"Reconstruction error too high: {err}"


def test_schur_ab_fp4():
    """SchurAB-FP4: shape + reconstruction error."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=True)
    layer = SchurABFP4Linear.from_linear(lin, block_size=32, schur_iters=3)

    x = torch.randn(2, 4, 128)
    out = layer(x)
    assert out.shape == (2, 4, 64), f"Bad shape: {out.shape}"

    w_orig = lin.weight.data.float()
    w_recon = layer._dequantize_weight(torch.float32)
    err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
    print(f"  SchurAB-FP4: out={out.shape} err={err:.4f}")
    assert err < 0.8, f"Reconstruction error too high: {err}"


def test_svd_lift_binary():
    """SVDLiftBinary: shape + reconstruction error."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=True)
    layer = SVDLiftBinaryLinear.from_linear(lin, rank=32)

    x = torch.randn(2, 4, 128)
    out = layer(x)
    assert out.shape == (2, 4, 64), f"Bad shape: {out.shape}"

    w_orig = lin.weight.data.float()
    w_recon = layer._dequantize_weight(torch.float32)
    err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
    print(f"  SVDLiftBinary rank=32: out={out.shape} err={err:.4f}")
    assert err < 1.0, f"Reconstruction error too high: {err}"


def test_svd_lift_rank_vs_error():
    """Higher rank should give lower reconstruction error."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=False)

    errors = {}
    for rank in [8, 16, 32, 64]:
        layer = SVDLiftBinaryLinear.from_linear(lin, rank=rank)
        w_orig = lin.weight.data.float()
        w_recon = layer._dequantize_weight(torch.float32)
        err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
        errors[rank] = err
        print(f"  SVDLift rank={rank}: err={err:.4f}")

    # Higher rank should generally give lower error
    assert errors[64] < errors[8], \
        f"Higher rank should have lower error: {errors}"


def test_lloyd_max_kv():
    """LloydMaxRotatedKV: quantize + dequantize round-trip."""
    torch.manual_seed(42)
    quantizer = LloydMaxRotatedKVQuantizer(bits=2, head_dim=128, use_qjl=True)

    # Simulate KV cache: (batch=1, heads=2, seq=16, dim=128)
    kv = torch.randn(1, 2, 16, 128, dtype=torch.float32)
    packed = quantizer.quantize(kv)

    # Check compression
    orig_bytes = kv.numel() * 2  # fp16
    packed_bytes = packed['packed'].numel() + packed['scales'].numel() * 2
    if packed['qjl_packed'] is not None:
        packed_bytes += packed['qjl_packed'].numel()
    ratio = orig_bytes / packed_bytes
    print(f"  LloydMax KV: orig={orig_bytes}B, packed={packed_bytes}B, ratio={ratio:.1f}x")

    # Dequantize and check shape
    kv_recon = quantizer.dequantize(packed)
    assert kv_recon.shape == kv.shape, f"Shape mismatch: {kv_recon.shape} vs {kv.shape}"

    # Check reconstruction error
    err = (kv - kv_recon).norm().item() / kv.norm().item()
    print(f"  LloydMax KV: recon err={err:.4f}")
    assert err < 0.5, f"KV reconstruction error too high: {err}"


def test_lloyd_max_3bit():
    """LloydMax 3-bit should have lower error than 2-bit."""
    torch.manual_seed(42)
    kv = torch.randn(1, 2, 16, 128, dtype=torch.float32)

    q2 = LloydMaxRotatedKVQuantizer(bits=2, head_dim=128, use_qjl=True)
    q3 = LloydMaxRotatedKVQuantizer(bits=3, head_dim=128, use_qjl=True)

    p2 = q2.quantize(kv)
    p3 = q3.quantize(kv)

    r2 = q2.dequantize(p2)
    r3 = q3.dequantize(p3)

    err2 = (kv - r2).norm().item() / kv.norm().item()
    err3 = (kv - r3).norm().item() / kv.norm().item()
    print(f"  LloydMax 2-bit err={err2:.4f}, 3-bit err={err3:.4f}")
    assert err3 < err2, "3-bit should have lower error than 2-bit"


def test_model_conversion():
    """Test model-level conversion functions."""
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )

    # Test each method
    for name, fn, kwargs in [
        ("wavelet", quantize_model_wavelet_lift, {"rank": 16}),
        ("schur_ab_fp4", quantize_model_schur_ab_fp4, {"block_size": 32}),
        ("svd_lift", quantize_model_svd_lift_binary, {"rank": 16}),
    ]:
        import copy
        m = copy.deepcopy(model)
        n = fn(m, verbose=False, **kwargs)
        assert n > 0, f"{name}: no layers quantized"

        # Forward pass
        x = torch.randn(2, 4, 64)
        out = m(x)
        assert out.shape == (2, 4, 64), f"{name}: bad output shape {out.shape}"
        print(f"  {name}: {n} layers, out={out.shape}")

    # Test memory estimation
    m = copy.deepcopy(model)
    quantize_model_schur_ab_fp4(m, verbose=False)
    mem = estimate_r45_memory(m)
    assert mem['total_bytes'] > 0, "Memory estimation returned 0"
    print(f"  Memory estimation: {mem['total_mb']:.2f} MB, eff_bits={mem['avg_eff_bits']:.2f}")


def test_requant_refine():
    """Test ReQuant refinement on a quantized model."""
    import copy

    from forge.engine.quant.novel_quant_r44 import (
        quantize_model_adaptive_block_fp4,
    )

    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )

    orig = copy.deepcopy(model)
    quant = copy.deepcopy(model)
    quantize_model_adaptive_block_fp4(quant, verbose=False)

    # Compute error before refinement
    def compute_err(m1, m2):
        errs = []
        for (_n1, mod1), (_n2, mod2) in zip(m1.named_modules(), m2.named_modules()):
            if isinstance(mod1, nn.Linear) and hasattr(mod2, '_dequantize_weight'):
                w1 = mod1.weight.data.float()
                w2 = mod2._dequantize_weight(torch.float32)
                if w1.shape == w2.shape:
                    errs.append((w1 - w2).norm().item() / w1.norm().item())
        return sum(errs) / max(len(errs), 1)

    err_before = compute_err(orig, quant)
    n = requant_refine_model(quant, orig, verbose=False)
    err_after = compute_err(orig, quant)
    print(f"  ReQuant: {n} layers, err before={err_before:.4f}, after={err_after:.4f}")


if __name__ == "__main__":
    print("R45 Novel Quantization Smoke Tests\n")

    tests = [
        ("Haar orthonormal", test_haar_orthonormal),
        ("Haar frequency separation", test_haar_frequency_separation),
        ("WaveletLift", test_wavelet_lift),
        ("SchurAB-FP4", test_schur_ab_fp4),
        ("SVDLiftBinary", test_svd_lift_binary),
        ("SVDLift rank vs error", test_svd_lift_rank_vs_error),
        ("LloydMax KV", test_lloyd_max_kv),
        ("LloydMax 3-bit", test_lloyd_max_3bit),
        ("Model conversion", test_model_conversion),
        ("ReQuant refine", test_requant_refine),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n[{name}]")
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Smoke tests: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All smoke tests passed!")
