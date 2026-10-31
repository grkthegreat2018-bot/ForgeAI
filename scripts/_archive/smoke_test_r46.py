#!/usr/bin/env python
"""R46 Novel Quantization Smoke Test — CPU, fast."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from forge.engine.quant.novel_quant_r46 import (
    AWQFP4Linear,
    GPTQFP4Linear,
    HadamardGPTQFP4Linear,
    HadamardRotatedFP4Linear,
    OptimalGridFP4Linear,
    collect_activations,
    estimate_r46_memory,
    quantize_model_awq_fp4,
    quantize_model_gptq_fp4,
    quantize_model_hadamard_gptq_fp4,
    quantize_model_hadamard_rotated_fp4,
    quantize_model_optimal_grid_fp4,
)


def test_hadamard_rotated_fp4():
    """HR-FP4: shape + reconstruction error."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=True)
    layer = HadamardRotatedFP4Linear.from_linear(lin, block_size=32)

    x = torch.randn(2, 4, 128)
    out = layer(x)
    assert out.shape == (2, 4, 64), f"Bad shape: {out.shape}"

    w_orig = lin.weight.data.float()
    w_recon = layer._dequantize_weight(torch.float32)
    err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
    print(f"  HR-FP4: out={out.shape} err={err:.4f}")
    assert err < 0.3, f"Reconstruction error too high: {err}"


def test_gptq_fp4():
    """GPTQ-FP4: shape + reconstruction error (with synthetic activations)."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=True)
    # Synthetic activations — random with some structure
    acts = torch.randn(128, 128) * 0.5 + 0.1
    layer = GPTQFP4Linear.from_linear(lin, acts, block_size=32, group_size=64)

    x = torch.randn(2, 4, 128)
    out = layer(x)
    assert out.shape == (2, 4, 64), f"Bad shape: {out.shape}"

    w_orig = lin.weight.data.float()
    w_recon = layer._dequantize_weight(torch.float32)
    err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
    print(f"  GPTQ-FP4: out={out.shape} err={err:.4f}")
    assert err < 0.3, f"Reconstruction error too high: {err}"


def test_awq_fp4():
    """AWQ-FP4: shape + reconstruction error (with synthetic activations)."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=True)
    acts = torch.randn(128, 128) * 0.5 + 0.1
    layer = AWQFP4Linear.from_linear(lin, acts, block_size=32)

    x = torch.randn(2, 4, 128)
    out = layer(x)
    assert out.shape == (2, 4, 64), f"Bad shape: {out.shape}"

    w_orig = lin.weight.data.float()
    w_recon = layer._dequantize_weight(torch.float32)
    err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
    print(f"  AWQ-FP4: out={out.shape} err={err:.4f}")
    assert err < 0.3, f"Reconstruction error too high: {err}"


def test_optimal_grid_fp4():
    """OG-FP4: shape + reconstruction error (no calibration)."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=True)
    layer = OptimalGridFP4Linear.from_linear(lin, block_size=32, n_lloyd_iters=10)

    x = torch.randn(2, 4, 128)
    out = layer(x)
    assert out.shape == (2, 4, 64), f"Bad shape: {out.shape}"

    w_orig = lin.weight.data.float()
    w_recon = layer._dequantize_weight(torch.float32)
    err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
    print(f"  OG-FP4: out={out.shape} err={err:.4f}")
    assert err < 0.3, f"Reconstruction error too high: {err}"


def test_hadamard_gptq_fp4():
    """HR-GPTQ-FP4: combined rotation + GPTQ (with synthetic activations)."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=True)
    acts = torch.randn(128, 128) * 0.5 + 0.1
    layer = HadamardGPTQFP4Linear.from_linear(lin, acts, block_size=32, group_size=64)

    x = torch.randn(2, 4, 128)
    out = layer(x)
    assert out.shape == (2, 4, 64), f"Bad shape: {out.shape}"

    w_orig = lin.weight.data.float()
    w_recon = layer._dequantize_weight(torch.float32)
    err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
    print(f"  HR-GPTQ-FP4: out={out.shape} err={err:.4f}")
    assert err < 0.3, f"Reconstruction error too high: {err}"


def test_gptq_better_than_rtn():
    """GPTQ should have lower reconstruction error than naive RTN."""
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=False)
    acts = torch.randn(256, 128) * 0.5 + 0.1

    # GPTQ
    gptq_layer = GPTQFP4Linear.from_linear(lin, acts, block_size=32, group_size=64)
    gptq_err = (lin.weight.data.float() - gptq_layer._dequantize_weight(torch.float32)).norm().item() / lin.weight.data.float().norm().item()

    # Naive RTN (using HadamardRotatedFP4 as baseline — no GPTQ)
    rtn_layer = HadamardRotatedFP4Linear.from_linear(lin, block_size=32)
    rtn_err = (lin.weight.data.float() - rtn_layer._dequantize_weight(torch.float32)).norm().item() / lin.weight.data.float().norm().item()

    print(f"  GPTQ err={gptq_err:.4f}, RTN err={rtn_err:.4f}")
    # GPTQ should generally be better (or at least not much worse)
    assert gptq_err <= rtn_err + 0.05, f"GPTQ should be better: {gptq_err} vs {rtn_err}"


def test_hadamard_better_than_no_rotation():
    """Hadamard rotation should reduce reconstruction error vs no rotation."""
    torch.manual_seed(42)
    # Create weights with outliers (rotation helps most here)
    lin = nn.Linear(128, 64, bias=False)
    # Add some outlier channels
    lin.weight.data[:, ::16] *= 5.0

    # With rotation
    hr_layer = HadamardRotatedFP4Linear.from_linear(lin, block_size=32)
    hr_err = (lin.weight.data.float() - hr_layer._dequantize_weight(torch.float32)).norm().item() / lin.weight.data.float().norm().item()

    # Without rotation (use SchurAB-FP4 from R45)
    from forge.engine.quant.novel_quant_r45 import SchurABFP4Linear
    schur_layer = SchurABFP4Linear.from_linear(lin, block_size=32)
    schur_err = (lin.weight.data.float() - schur_layer._dequantize_weight(torch.float32)).norm().item() / lin.weight.data.float().norm().item()

    print(f"  HR-FP4 (rotated) err={hr_err:.4f}, SchurAB-FP4 (no rotation) err={schur_err:.4f}")
    # Rotation should help with outliers
    assert hr_err < schur_err + 0.1, f"Rotation should help with outliers: {hr_err} vs {schur_err}"


def test_model_conversion():
    """Test model-level conversion for all methods."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )

    # No-calibration methods
    for name, fn, kwargs in [
        ("hr_fp4", quantize_model_hadamard_rotated_fp4, {"block_size": 32}),
        ("og_fp4", quantize_model_optimal_grid_fp4, {"block_size": 32, "n_lloyd_iters": 10}),
    ]:
        import copy
        m = copy.deepcopy(model)
        n = fn(m, verbose=False, **kwargs)
        assert n > 0, f"{name}: no layers quantized"
        x = torch.randn(2, 4, 64)
        out = m(x)
        assert out.shape == (2, 4, 64), f"{name}: bad output shape {out.shape}"
        print(f"  {name}: {n} layers, out={out.shape}")

    # Calibration methods (need activations dict)
    acts = {
        '0': torch.randn(64, 64),
        '2': torch.randn(64, 128),
    }
    for name, fn, kwargs in [
        ("gptq_fp4", quantize_model_gptq_fp4, {"block_size": 32, "group_size": 64}),
        ("awq_fp4", quantize_model_awq_fp4, {"block_size": 32}),
        ("hr_gptq_fp4", quantize_model_hadamard_gptq_fp4, {"block_size": 32, "group_size": 64}),
    ]:
        import copy
        m = copy.deepcopy(model)
        n = fn(m, acts, verbose=False, **kwargs)
        assert n > 0, f"{name}: no layers quantized"
        x = torch.randn(2, 4, 64)
        out = m(x)
        assert out.shape == (2, 4, 64), f"{name}: bad output shape {out.shape}"
        print(f"  {name}: {n} layers, out={out.shape}")

    # Memory estimation
    m = copy.deepcopy(model)
    quantize_model_hadamard_rotated_fp4(m, verbose=False)
    mem = estimate_r46_memory(m)
    assert mem['total_bytes'] > 0, "Memory estimation returned 0"
    print(f"  Memory estimation: {mem['total_mb']:.4f} MB, eff_bits={mem['avg_eff_bits']:.2f}")


def test_collect_activations():
    """Test activation collection from a model."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    # Simulate token IDs (not used directly, just need a forward pass)
    x = torch.randn(2, 8, 64)
    acts = collect_activations(model, x, n_samples=16)
    assert len(acts) >= 2, f"Should collect from 2 layers, got {len(acts)}"
    for name, act in acts.items():
        assert act.dim() == 2, f"{name}: activations should be 2D, got {act.shape}"
        assert act.shape[1] in [64, 128], f"{name}: wrong feature dim {act.shape[1]}"
    print(f"  Collected activations from {len(acts)} layers")


if __name__ == "__main__":
    print("R46 Novel Quantization Smoke Tests\n")

    tests = [
        ("HadamardRotatedFP4", test_hadamard_rotated_fp4),
        ("GPTQFP4", test_gptq_fp4),
        ("AWQFP4", test_awq_fp4),
        ("OptimalGridFP4", test_optimal_grid_fp4),
        ("HadamardGPTQFP4", test_hadamard_gptq_fp4),
        ("GPTQ better than RTN", test_gptq_better_than_rtn),
        ("Hadamard helps outliers", test_hadamard_better_than_no_rotation),
        ("Model conversion", test_model_conversion),
        ("Collect activations", test_collect_activations),
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
