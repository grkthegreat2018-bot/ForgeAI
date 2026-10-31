"""Quick smoke test for R44 novel quant algorithms."""
import torch

from forge.engine.quant.novel_quant_r44 import (
    AdaptiveBlockFP4Linear,
    HadamardLiftLinear,
    SparseResidualINT3Linear,
    TernaryLiftLinear,
)

lin = torch.nn.Linear(64, 128, bias=True)
x = torch.randn(2, 4, 64)
y_ref = lin(x)

for cls, name in [
    (HadamardLiftLinear, "HLQ-2bit"),
    (AdaptiveBlockFP4Linear, "AB-FP4"),
    (SparseResidualINT3Linear, "SR-INT3"),
    (TernaryLiftLinear, "TL-1.5x"),
]:
    ql = cls.from_linear(lin)
    y = ql(x)
    err = (y_ref - y).norm().item() / max(y_ref.norm().item(), 1e-8)
    print(f"{name}: out={y.shape} err={err:.4f} repr={ql}")

# Test with different sizes (power-of-2 and non-power-of-2)
for in_f, out_f in [(96, 64), (100, 200), (256, 512)]:
    lin2 = torch.nn.Linear(in_f, out_f, bias=False)
    x2 = torch.randn(1, 3, in_f)
    y_ref2 = lin2(x2)
    for cls, name in [
        (HadamardLiftLinear, "HLQ"),
        (TernaryLiftLinear, "TL"),
    ]:
        ql = cls.from_linear(lin2)
        y = ql(x2)
        err = (y_ref2 - y).norm().item() / max(y_ref2.norm().item(), 1e-8)
        print(f"  {name} {in_f}x{out_f}: err={err:.4f}")

print("\nAll smoke tests passed!")
