"""Smoke test for GRINQH quantization."""
from __future__ import annotations

import torch
import torch.nn as nn

from forge.engine.quant.grinqh import (
    GRINQHLinear,
    GRINQHQuantizer,
    _assign_precision_tiers,
    quantize_model_grinqh,
)


def test_precision_tiers():
    w = torch.randn(100, 256)
    for target in [2.0, 2.5, 3.0, 3.5, 4.0]:
        tiers = _assign_precision_tiers(w, target)
        avg = tiers.float().mean().item()
        assert abs(avg - target) < 0.3, f"target={target}, got avg={avg}"
        assert tiers.min() >= 2 and tiers.max() <= 4
    print("  [test_precision_tiers] OK")


def test_quantize_dequantize():
    q = GRINQHQuantizer(128, 2.5)
    w = torch.randn(64, 256)
    packed = q.quantize(w)
    print(f"  effective bits: {packed['effective_bits']:.3f}")
    w_dq = q.dequantize(packed, dtype=torch.float32)
    rel_err = (w - w_dq).norm() / w.norm()
    print(f"  rel err: {rel_err.item():.4f}")
    assert rel_err.item() < 0.8, f"rel err too high: {rel_err.item()}"
    print("  [test_quantize_dequantize] OK")


def test_grinqh_linear():
    lin = nn.Linear(256, 64, bias=True)
    gl = GRINQHLinear.from_linear(lin, group_size=128, target_effective_bits=2.5)
    x = torch.randn(4, 256)
    y_orig = lin(x)
    y_quant = gl(x)
    rel_err = (y_orig - y_quant).norm() / y_orig.norm()
    print(f"  output rel err: {rel_err.item():.4f}")
    print(f"  effective bits: {gl.effective_bits:.3f}")
    assert rel_err.item() < 0.5, f"output rel err too high: {rel_err.item()}"
    print("  [test_grinqh_linear] OK")


def test_model_replacement():
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )
    n = quantize_model_grinqh(model, group_size=128, target_effective_bits=2.5,
                              verbose=False)
    assert n == 2, f"expected 2 layers, got {n}"
    assert isinstance(model[0], GRINQHLinear)
    assert isinstance(model[2], GRINQHLinear)
    x = torch.randn(2, 128)
    out = model(x)
    assert out.shape == (2, 32)
    print("  [test_model_replacement] OK")


def test_non_multiple_group_size():
    lin = nn.Linear(300, 50, bias=False)
    gl = GRINQHLinear.from_linear(lin, group_size=128, target_effective_bits=3.0)
    x = torch.randn(4, 300)
    y_orig = lin(x)
    y_quant = gl(x)
    rel_err = (y_orig - y_quant).norm() / y_orig.norm()
    print(f"  non-multiple group_size rel err: {rel_err.item():.4f}")
    assert rel_err.item() < 0.5
    print("  [test_non_multiple_group_size] OK")


if __name__ == "__main__":
    test_precision_tiers()
    test_quantize_dequantize()
    test_grinqh_linear()
    test_model_replacement()
    test_non_multiple_group_size()
    print("\nAll GRINQH tests passed.")
