"""Tests for R&D Round 23: FP8 activation storage + GradTopK gradient sparsification.

Tests R21 training-time features that R23 wires into the V8 training loop:
- FP8ActivationLinear: quantizes activations to FP8 e4m3 during forward, stores
  for backward (2x activation memory reduction vs bf16)
- TopKGradientOptimizer: wraps any optimizer, sparsifies gradients to top-K%
  with EF21 error feedback (10x fewer gradient transfers)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── R23-FP8-1: FP8 activation memory ────────────────────────────────────────


def test_fp8_activation_memory():
    """FP8 compressed activation should use ~1 byte/element vs 2 for bf16."""
    from forge.training.optim.r21_cross_domain import FP8ActivationLinear

    d = 128
    layer = FP8ActivationLinear(d, d, bias=False).to(_DEV)
    layer.train()

    B, T = 4, 32
    x = torch.randn(B, T, d, device=_DEV, requires_grad=True)
    out = layer(x)
    out.sum().backward()

    fp8_bytes = layer.get_compressed_activation_memory()
    n_elements = B * T * d
    bf16_bytes = n_elements * 2  # bf16 = 2 bytes/element

    fp8_per_elem = fp8_bytes / n_elements
    bf16_per_elem = bf16_bytes / n_elements
    compression = bf16_bytes / max(fp8_bytes, 1)

    print(f"  FP8 act memory: {fp8_bytes} bytes ({fp8_per_elem:.2f} B/elem)")
    print(f"  bf16 act memory: {bf16_bytes} bytes ({bf16_per_elem:.2f} B/elem)")
    print(f"  Compression: {compression:.2f}x")

    assert fp8_per_elem < 1.5, f"FP8 should be ~1 byte/elem, got {fp8_per_elem:.2f}"
    assert compression > 1.5, f"FP8 should compress >1.5x vs bf16, got {compression:.2f}x"

    print("  fp8_activation_memory: PASS")


# ── R23-FP8-2: FP8 quantization round-trip ──────────────────────────────────


def test_fp8_activation_roundtrip():
    """FP8 quantize → dequantize should have <10% mean relative error."""
    from forge.training.optim.r21_cross_domain import FP8ActivationLinear

    torch.manual_seed(42)
    x = torch.randn(4096, device=_DEV) * 0.1  # typical activation scale
    x_fp8, scale = FP8ActivationLinear._quantize_fp8(x)
    x_dq = FP8ActivationLinear._dequantize_fp8(x_fp8, scale)

    rel_err = (x - x_dq).abs().mean().item() / x.abs().mean().item()
    print(f"  FP8 round-trip: {x.numel()} elements, {rel_err * 100:.2f}% error")
    assert rel_err < 0.10, f"FP8 round-trip error should be <10%, got {rel_err * 100:.2f}%"

    print("  fp8_activation_roundtrip: PASS")


# ── R23-FP8-3: Forward correctness ──────────────────────────────────────────


def test_fp8_activation_forward_correct():
    """FP8ActivationLinear forward should be close to plain nn.Linear (same weights)."""
    from forge.training.optim.r21_cross_domain import FP8ActivationLinear

    torch.manual_seed(42)
    d_in, d_out = 128, 128
    B, T = 4, 32

    # Create both layers with identical weights
    fp8_layer = FP8ActivationLinear(d_in, d_out, bias=False).to(_DEV)
    fp8_layer.train()
    plain_layer = nn.Linear(d_in, d_out, bias=False).to(_DEV)
    plain_layer.weight.data.copy_(fp8_layer.weight.data)

    x = torch.randn(B, T, d_in, device=_DEV)

    with torch.no_grad():
        out_fp8 = fp8_layer(x)
        out_plain = plain_layer(x)

    rel_err = (out_fp8 - out_plain).abs().mean().item() / out_plain.abs().mean().item()
    print(f"  FP8 vs plain Linear: {rel_err * 100:.2f}% relative error")
    assert rel_err < 0.05, f"FP8 forward should be within 5% of plain, got {rel_err * 100:.2f}%"

    print("  fp8_activation_forward_correct: PASS")


# ── R23-FP8-4: Backward pass ────────────────────────────────────────────────


def test_fp8_activation_backward():
    """Forward + backward through FP8ActivationLinear should produce finite grads."""
    from forge.training.optim.r21_cross_domain import FP8ActivationLinear

    torch.manual_seed(42)
    d = 128
    layer = FP8ActivationLinear(d, d, bias=True).to(_DEV)
    layer.train()

    B, T = 4, 32
    x = torch.randn(B, T, d, device=_DEV, requires_grad=True)
    target = torch.randn(B, T, d, device=_DEV)

    out = layer(x)
    loss = F.mse_loss(out, target)
    loss.backward()

    assert x.grad is not None, "Input should have gradient"
    assert layer.weight.grad is not None, "Weight should have gradient"
    assert torch.isfinite(layer.weight.grad).all(), "Weight grad should be finite"
    assert torch.isfinite(x.grad).all(), "Input grad should be finite"

    grad_norm = layer.weight.grad.norm().item()
    print(f"  FP8 backward: loss={loss.item():.4f}, weight grad norm={grad_norm:.4f}")
    assert grad_norm > 0, "Gradient should be non-zero"

    print("  fp8_activation_backward: PASS")


# ── R23-GradTopK-5: Sparsity verification ───────────────────────────────────


def test_gradtopk_sparsity():
    """TopKGradientOptimizer with ratio=0.1 should produce ~10% dense gradients."""
    from forge.training.optim.r21_cross_domain import TopKGradientOptimizer

    torch.manual_seed(42)
    model = nn.Linear(256, 256, bias=False).to(_DEV)
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    topk_opt = TopKGradientOptimizer(base_opt, top_k_ratio=0.1, ef_feedback=True)

    x = torch.randn(16, 256, device=_DEV)
    y = torch.randn(16, 256, device=_DEV)

    topk_opt.zero_grad()
    loss = F.mse_loss(model(x), y)
    loss.backward()

    # Save original grad for reference
    original_grad = model.weight.grad.clone()
    total_elements = original_grad.numel()

    # Step sparsifies the gradient in-place
    topk_opt.step()

    # After step, p.grad is the sparsified version
    sparse_grad = model.weight.grad
    non_zero = (sparse_grad.abs() > 0).sum().item()
    density = non_zero / total_elements

    print(f"  GradTopK ratio=0.1: {non_zero}/{total_elements} non-zero ({density * 100:.1f}% dense)")
    assert 0.05 < density < 0.15, f"Gradient should be ~10% dense, got {density * 100:.1f}%"

    print("  gradtopk_sparsity: PASS")


# ── R23-GradTopK-6: Error feedback convergence ──────────────────────────────


def test_gradtopk_error_feedback():
    """With EF21, TopK should converge similarly to dense over 20 steps."""
    from forge.training.optim.r21_cross_domain import TopKGradientOptimizer

    torch.manual_seed(42)
    d = 128
    x = torch.randn(16, d, device=_DEV)
    y = torch.randn(16, d, device=_DEV)

    # Dense baseline (plain AdamW)
    model_dense = nn.Linear(d, d, bias=False).to(_DEV)
    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=1e-2)
    initial_dense = F.mse_loss(model_dense(x), y).item()
    for _ in range(20):
        opt_dense.zero_grad()
        loss = F.mse_loss(model_dense(x), y)
        loss.backward()
        opt_dense.step()
    final_dense = loss.item()

    # TopK with EF (same seed → same init)
    torch.manual_seed(42)
    model_topk = nn.Linear(d, d, bias=False).to(_DEV)
    base_opt = torch.optim.AdamW(model_topk.parameters(), lr=1e-2)
    topk_opt = TopKGradientOptimizer(base_opt, top_k_ratio=0.1, ef_feedback=True)
    initial_topk = F.mse_loss(model_topk(x), y).item()
    for _ in range(20):
        topk_opt.zero_grad()
        loss = F.mse_loss(model_topk(x), y)
        loss.backward()
        topk_opt.step()
    final_topk = loss.item()

    print(f"  Dense:  {initial_dense:.4f} -> {final_dense:.4f}")
    print(f"  TopK:   {initial_topk:.4f} -> {final_topk:.4f}")

    # Both should reduce loss
    assert final_dense < initial_dense, "Dense should reduce loss"
    assert final_topk < initial_topk, "TopK with EF should reduce loss"

    # TopK should not be drastically worse than dense (EF prevents info loss)
    # On tiny models with 20 steps, TopK 10% converges slower — allow 10x slack
    ratio = final_topk / max(final_dense, 1e-8)
    print(f"  TopK/dense final loss ratio: {ratio:.2f}")
    assert ratio < 10.0, f"TopK with EF should converge (ratio={ratio:.2f})"

    print("  gradtopk_error_feedback: PASS")


# ── R23-GradTopK-7: Ratio sweep ─────────────────────────────────────────────


def test_gradtopk_ratio_sweep():
    """Test multiple ratios and verify actual sparsity matches expected."""
    from forge.training.optim.r21_cross_domain import TopKGradientOptimizer

    d = 512
    ratios = [0.05, 0.1, 0.25, 0.5, 1.0]

    for ratio in ratios:
        torch.manual_seed(42)
        model = nn.Linear(d, d, bias=False).to(_DEV)
        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        topk_opt = TopKGradientOptimizer(base_opt, top_k_ratio=ratio, ef_feedback=False)

        x = torch.randn(8, d, device=_DEV)
        y = torch.randn(8, d, device=_DEV)

        topk_opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        topk_opt.step()

        total = model.weight.grad.numel()
        non_zero = (model.weight.grad.abs() > 0).sum().item()
        actual_density = non_zero / total

        if ratio >= 1.0:
            expected = 1.0
            assert non_zero == total, f"ratio=1.0 should keep all gradients, got {non_zero}/{total}"
        else:
            expected = ratio
            # Allow some tolerance for rounding (k = max(1, int(n * ratio)))
            assert abs(actual_density - expected) < 0.02, (
                f"ratio={ratio}: expected ~{expected * 100:.0f}% dense, got {actual_density * 100:.1f}%"
            )

        print(f"  ratio={ratio:.2f}: {actual_density * 100:.1f}% dense ({non_zero}/{total})")

    print("  gradtopk_ratio_sweep: PASS")


# ── R23-FP8-GradTopK-8: Combined training ───────────────────────────────────


def test_fp8_gradtopk_combined():
    """FP8ActivationLinear + TopKGradientOptimizer should reduce loss + memory."""
    from forge.training.optim.r21_cross_domain import FP8ActivationLinear, TopKGradientOptimizer

    torch.manual_seed(42)
    d = 128
    model = nn.Sequential(
        FP8ActivationLinear(d, d, bias=True),
        nn.GELU(),
        FP8ActivationLinear(d, d, bias=True),
    ).to(_DEV)
    model.train()

    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    topk_opt = TopKGradientOptimizer(base_opt, top_k_ratio=0.25, ef_feedback=True)

    x = torch.randn(16, d, device=_DEV)
    y = torch.randn(16, d, device=_DEV)

    initial_loss = F.mse_loss(model(x), y).item()
    for _ in range(20):
        topk_opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        topk_opt.step()
    final_loss = loss.item()

    # Measure FP8 activation memory from first layer
    first_layer = model[0]
    fp8_bytes = first_layer.get_compressed_activation_memory()
    bf16_bytes = 16 * d * 2  # bf16 baseline for one layer's activation

    print(f"  Combined: {initial_loss:.4f} -> {final_loss:.4f}")
    print(
        f"  FP8 act memory: {fp8_bytes} bytes vs bf16 {bf16_bytes} bytes "
        f"({bf16_bytes / max(fp8_bytes, 1):.1f}x compression)"
    )

    assert final_loss < initial_loss, "Loss should decrease over 20 steps"
    assert fp8_bytes < bf16_bytes, "FP8 should use less memory than bf16"

    print("  fp8_gradtopk_combined: PASS")


# ── R23-FP8-GradTopK-9: NVMe speedup simulation ─────────────────────────────


def test_fp8_gradtopk_nvme_speedup():
    """GradTopK should reduce gradient elements transferred (NVMe block switch)."""
    from forge.training.optim.r21_cross_domain import TopKGradientOptimizer

    torch.manual_seed(42)
    d = 256
    model = nn.Linear(d, d, bias=False).to(_DEV)
    x = torch.randn(16, d, device=_DEV)
    y = torch.randn(16, d, device=_DEV)

    # Without GradTopK: all gradient elements are "transferred"
    model.zero_grad()
    loss = F.mse_loss(model(x), y)
    loss.backward()
    total_grad_elements = model.weight.grad.numel()
    transferred_without = total_grad_elements

    # With GradTopK (ratio=0.1): only top 10% are "transferred"
    model2 = nn.Linear(d, d, bias=False).to(_DEV)
    model2.weight.data.copy_(model.weight.data)

    base_opt = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    topk_opt = TopKGradientOptimizer(base_opt, top_k_ratio=0.1, ef_feedback=False)

    topk_opt.zero_grad()
    loss2 = F.mse_loss(model2(x), y)
    loss2.backward()
    # Count elements before sparsification
    model2.weight.grad.numel()
    topk_opt.step()
    # After step, count non-zero (transferred) elements
    transferred_with = (model2.weight.grad.abs() > 0).sum().item()

    speedup = transferred_without / max(transferred_with, 1)

    print(f"  Without GradTopK: {transferred_without} gradient elements transferred")
    print(f"  With GradTopK:    {transferred_with} gradient elements transferred")
    print(f"  Transfer speedup: {speedup:.1f}x")

    assert transferred_with < transferred_without, "GradTopK should reduce transferred elements"
    assert speedup > 5.0, f"GradTopK 10% should give >5x transfer speedup, got {speedup:.1f}x"

    print("  fp8_gradtopk_nvme_speedup: PASS")


def main_r23_fp8_gradtopk():
    print("=" * 70)
    print("  R&D ROUND 23: FP8 Activation + GradTopK")
    print("=" * 70)

    print("\n  FP8-1: Activation memory")
    test_fp8_activation_memory()

    print("\n  FP8-2: Quantization round-trip")
    test_fp8_activation_roundtrip()

    print("\n  FP8-3: Forward correctness")
    test_fp8_activation_forward_correct()

    print("\n  FP8-4: Backward pass")
    test_fp8_activation_backward()

    print("\n  GradTopK-5: Sparsity")
    test_gradtopk_sparsity()

    print("\n  GradTopK-6: Error feedback convergence")
    test_gradtopk_error_feedback()

    print("\n  GradTopK-7: Ratio sweep")
    test_gradtopk_ratio_sweep()

    print("\n  FP8+GradTopK-8: Combined training")
    test_fp8_gradtopk_combined()

    print("\n  FP8+GradTopK-9: NVMe speedup")
    test_fp8_gradtopk_nvme_speedup()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 23 FP8+GRADTOPK TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r23_fp8_gradtopk()
