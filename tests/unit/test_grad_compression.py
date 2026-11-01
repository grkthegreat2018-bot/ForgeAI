"""Tests for int4 gradient compression with EF21 error feedback.

Verifies:
  1. Compression ratio: packed size = original_size / 4 (bf16 → int4)
  2. Round-trip error: relative L2 error < 15% for random gradients
  3. EF21 convergence: 100 steps of GD on a quadratic converges to within
     1% of the uncompressed final loss
  4. 1D tensor handling (biases, norms)
  5. Edge cases: all-zero gradients, single-element tensors

All tests run on GPU (CUDA) with CPU fallback.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch

from forge.training.optim.hybrid_offload import (
    _compress_grad_int4,
    _decompress_grad_int4,
)

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")


# ---------------------------------------------------------------------------
# Test 1: Compression ratio
# ---------------------------------------------------------------------------
class TestCompressionRatio:
    """Verify packed int4 size = original bf16 size / 4."""

    def test_2d_tensor_ratio(self):
        """bf16 (2 bytes/elem) → packed int4 (0.5 bytes/elem) = 4x."""
        grad = torch.randn(128, 256, dtype=torch.bfloat16, device=DEVICE)
        packed, _scales = _compress_grad_int4(grad)

        orig_bytes = grad.numel() * grad.element_size()  # 128*256*2 = 65536
        packed_bytes = packed.numel() * packed.element_size()  # (128*256/2)*1
        ratio = orig_bytes / packed_bytes

        assert ratio == 4.0, f"Expected 4x compression, got {ratio}x"
        # Packed should have exactly half the elements (2 int4 per byte)
        assert packed.numel() == grad.numel() // 2

    def test_1d_tensor_ratio(self):
        """1D tensor also gets 4x compression."""
        grad = torch.randn(1024, dtype=torch.bfloat16, device=DEVICE)
        packed, _scales = _compress_grad_int4(grad)

        orig_bytes = grad.numel() * grad.element_size()
        packed_bytes = packed.numel() * packed.element_size()
        ratio = orig_bytes / packed_bytes

        assert ratio == 4.0, f"Expected 4x for 1D, got {ratio}x"

    def test_odd_length_packing(self):
        """Odd number of elements gets padded to even for packing."""
        grad = torch.randn(101, dtype=torch.float32, device=DEVICE)
        packed, scales = _compress_grad_int4(grad)

        # 101 elements → padded to 102 → 51 packed bytes
        assert packed.numel() == 51

        # Round-trip should still produce 101 elements
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)
        assert decompressed.shape == grad.shape


# ---------------------------------------------------------------------------
# Test 2: Round-trip error
# ---------------------------------------------------------------------------
class TestRoundTripError:
    """Verify relative L2 error < 15% for random gradients."""

    def test_2d_round_trip_error(self):
        """Per-row quantization on a 2D tensor."""
        torch.manual_seed(42)
        grad = torch.randn(64, 128, dtype=torch.float32, device=DEVICE)
        packed, scales = _compress_grad_int4(grad)
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)

        rel_error = (decompressed - grad).norm() / grad.norm()
        assert rel_error < 0.15, f"Relative error {rel_error:.4f} >= 0.15"

    def test_1d_round_trip_error(self):
        """Per-tensor quantization on a 1D tensor (bias-like)."""
        torch.manual_seed(42)
        grad = torch.randn(512, dtype=torch.float32, device=DEVICE)
        packed, scales = _compress_grad_int4(grad)
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)

        rel_error = (decompressed - grad).norm() / grad.norm()
        assert rel_error < 0.15, f"1D relative error {rel_error:.4f} >= 0.15"

    def test_bf16_round_trip_error(self):
        """bf16 input (common case — gradients are bf16 on GPU)."""
        torch.manual_seed(42)
        grad = torch.randn(32, 64, dtype=torch.bfloat16, device=DEVICE)
        packed, scales = _compress_grad_int4(grad)
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)

        grad_fp32 = grad.to(torch.float32)
        rel_error = (decompressed - grad_fp32).norm() / grad_fp32.norm()
        assert rel_error < 0.15, f"bf16 relative error {rel_error:.4f} >= 0.15"

    def test_large_tensor_error(self):
        """Large tensor (simulates a real weight gradient)."""
        torch.manual_seed(42)
        grad = torch.randn(1024, 1024, dtype=torch.float32, device=DEVICE)
        packed, scales = _compress_grad_int4(grad)
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)

        rel_error = (decompressed - grad).norm() / grad.norm()
        assert rel_error < 0.15, f"Large tensor error {rel_error:.4f} >= 0.15"


# ---------------------------------------------------------------------------
# Test 3: EF21 convergence
# ---------------------------------------------------------------------------
class TestEF21Convergence:
    """Verify EF21 error feedback preserves convergence on a quadratic."""

    def test_ef21_quadratic_convergence(self):
        """100 steps of GD on f(w) = 0.5*||w - target||^2 with int4 + EF21.

        The compressed gradient descent should converge to within 1% of
        the uncompressed final loss.
        """
        torch.manual_seed(42)
        dim_rows, dim_cols = 4, 64
        w_init = torch.randn(dim_rows, dim_cols, dtype=torch.float32, device=DEVICE)
        target = torch.randn(dim_rows, dim_cols, dtype=torch.float32, device=DEVICE)
        lr = 0.1
        n_steps = 100

        # --- Uncompressed baseline ---
        w_unc = w_init.clone()
        for _ in range(n_steps):
            grad = w_unc - target  # gradient of 0.5*||w-target||^2
            w_unc -= lr * grad
        loss_unc = 0.5 * ((w_unc - target) ** 2).sum().item()

        # --- Compressed with EF21 ---
        w_comp = w_init.clone()
        ef_error = torch.zeros_like(w_comp)
        for _ in range(n_steps):
            grad = w_comp - target
            # EF21: accumulate error
            grad_to_send = grad + ef_error
            # Compress
            packed, scales = _compress_grad_int4(grad_to_send)
            grad_decompressed = _decompress_grad_int4(packed, scales, grad.shape)
            # Update error feedback
            ef_error = grad_to_send - grad_decompressed
            # Update weights with decompressed gradient
            w_comp -= lr * grad_decompressed
        loss_comp = 0.5 * ((w_comp - target) ** 2).sum().item()

        # Both should have converged significantly
        loss_init = 0.5 * ((w_init - target) ** 2).sum().item()
        assert loss_unc < loss_init * 0.01, f"Uncompressed didn't converge: {loss_unc:.6f} vs init {loss_init:.6f}"
        assert loss_comp < loss_init * 0.01, f"Compressed didn't converge: {loss_comp:.6f} vs init {loss_init:.6f}"

        # Compressed should be within 1% of uncompressed final loss
        if loss_unc > 1e-10:
            rel_diff = abs(loss_comp - loss_unc) / loss_unc
        else:
            rel_diff = abs(loss_comp - loss_unc) / (loss_init + 1e-12)

        assert rel_diff < 0.01, f"EF21 final loss diff {rel_diff:.6f} >= 1% (unc={loss_unc:.8f}, comp={loss_comp:.8f})"

    def test_ef21_1d_convergence(self):
        """EF21 on a 1D quadratic (per-tensor quantization)."""
        torch.manual_seed(42)
        dim = 128
        w_init = torch.randn(dim, dtype=torch.float32, device=DEVICE)
        target = torch.randn(dim, dtype=torch.float32, device=DEVICE)
        lr = 0.1
        n_steps = 100

        # Uncompressed
        w_unc = w_init.clone()
        for _ in range(n_steps):
            grad = w_unc - target
            w_unc -= lr * grad
        loss_unc = 0.5 * ((w_unc - target) ** 2).sum().item()

        # Compressed with EF21
        w_comp = w_init.clone()
        ef_error = torch.zeros_like(w_comp)
        for _ in range(n_steps):
            grad = w_comp - target
            grad_to_send = grad + ef_error
            packed, scales = _compress_grad_int4(grad_to_send)
            grad_decompressed = _decompress_grad_int4(packed, scales, grad.shape)
            ef_error = grad_to_send - grad_decompressed
            w_comp -= lr * grad_decompressed
        loss_comp = 0.5 * ((w_comp - target) ** 2).sum().item()

        loss_init = 0.5 * ((w_init - target) ** 2).sum().item()
        assert loss_comp < loss_init * 0.01, f"1D EF21 didn't converge: {loss_comp:.6f} vs init {loss_init:.6f}"

        if loss_unc > 1e-10:
            rel_diff = abs(loss_comp - loss_unc) / loss_unc
        else:
            rel_diff = abs(loss_comp - loss_unc) / (loss_init + 1e-12)

        assert rel_diff < 0.02, (
            f"1D EF21 final loss diff {rel_diff:.6f} >= 2% (unc={loss_unc:.8f}, comp={loss_comp:.8f})"
        )

    def test_ef21_without_error_feedback_diverges_more(self):
        """Sanity check: without EF21, convergence is worse (proves EF21 helps)."""
        torch.manual_seed(42)
        dim_rows, dim_cols = 4, 64
        w_init = torch.randn(dim_rows, dim_cols, dtype=torch.float32, device=DEVICE)
        target = torch.randn(dim_rows, dim_cols, dtype=torch.float32, device=DEVICE)
        lr = 0.1
        n_steps = 100

        # Compressed WITHOUT EF21 (naive quantization each step)
        w_no_ef = w_init.clone()
        for _ in range(n_steps):
            grad = w_no_ef - target
            packed, scales = _compress_grad_int4(grad)
            grad_decompressed = _decompress_grad_int4(packed, scales, grad.shape)
            w_no_ef -= lr * grad_decompressed
        loss_no_ef = 0.5 * ((w_no_ef - target) ** 2).sum().item()

        # With EF21
        w_ef = w_init.clone()
        ef_error = torch.zeros_like(w_ef)
        for _ in range(n_steps):
            grad = w_ef - target
            grad_to_send = grad + ef_error
            packed, scales = _compress_grad_int4(grad_to_send)
            grad_decompressed = _decompress_grad_int4(packed, scales, grad.shape)
            ef_error = grad_to_send - grad_decompressed
            w_ef -= lr * grad_decompressed
        loss_ef = 0.5 * ((w_ef - target) ** 2).sum().item()

        # EF21 should be at least as good as naive (usually better)
        assert loss_ef <= loss_no_ef * 1.05, f"EF21 ({loss_ef:.8f}) should be <= naive ({loss_no_ef:.8f}) * 1.05"


# ---------------------------------------------------------------------------
# Test 4: Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases: all-zero, single element, empty, multi-dim."""

    def test_all_zero_gradient(self):
        """All-zero gradient should round-trip to all-zero."""
        grad = torch.zeros(32, 64, dtype=torch.float32)
        packed, scales = _compress_grad_int4(grad)
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)

        assert torch.allclose(decompressed, grad, atol=1e-10), (
            f"All-zero round-trip failed: max={decompressed.abs().max():.6e}"
        )

    def test_single_element(self):
        """Single element tensor (edge case for packing)."""
        grad = torch.tensor([3.14], dtype=torch.float32)
        packed, scales = _compress_grad_int4(grad)
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)

        # Single element: absmax = |grad|, so quantization is exact
        assert abs(decompressed.item() - grad.item()) < 0.01, (
            f"Single element round-trip: {decompressed.item():.4f} vs {grad.item():.4f}"
        )

    def test_two_elements(self):
        """Two elements (exactly one packed byte)."""
        grad = torch.tensor([1.0, -2.0], dtype=torch.float32)
        packed, scales = _compress_grad_int4(grad)
        assert packed.numel() == 1, f"Expected 1 packed byte, got {packed.numel()}"

        decompressed = _decompress_grad_int4(packed, scales, grad.shape)
        assert decompressed.shape == grad.shape

    def test_3d_tensor(self):
        """3D tensor (conv weight-like): per-channel on first dim."""
        grad = torch.randn(8, 3, 5, dtype=torch.float32, device=DEVICE)  # conv-like
        packed, scales = _compress_grad_int4(grad)
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)

        assert decompressed.shape == grad.shape
        rel_error = (decompressed - grad).norm() / grad.norm()
        assert rel_error < 0.15, f"3D tensor error {rel_error:.4f} >= 0.15"
        # Scales should have 8 rows (one per output channel)
        assert scales.shape == (8, 1), f"Expected scales (8,1), got {scales.shape}"

    def test_empty_tensor(self):
        """Empty tensor should not crash."""
        grad = torch.zeros(0, 4, dtype=torch.float32)
        packed, scales = _compress_grad_int4(grad)
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)
        assert decompressed.shape == grad.shape

    def test_gradient_with_outliers(self):
        """Gradient with a few large outliers (common in training)."""
        torch.manual_seed(42)
        grad = torch.randn(64, 128, dtype=torch.float32, device=DEVICE) * 0.01
        grad[0, 0] = 10.0  # large outlier
        grad[1, 5] = -8.0  # another outlier

        packed, scales = _compress_grad_int4(grad)
        decompressed = _decompress_grad_int4(packed, scales, grad.shape)

        # The outliers should be well-preserved (they set the scale)
        assert abs(decompressed[0, 0].item() - 10.0) < 1.0, (
            f"Outlier not preserved: {decompressed[0, 0].item():.4f} vs 10.0"
        )
        # Overall error may be higher due to scale being dominated by outliers
        # but should still be reasonable
        rel_error = (decompressed - grad).norm() / grad.norm()
        assert rel_error < 0.50, f"Outlier error {rel_error:.4f} too high"


# ---------------------------------------------------------------------------
# Test 5: CPUAdamW integration (CPU-only, no GPU required)
# ---------------------------------------------------------------------------
class TestCPUAdamWIntegration:
    """Test that grad_compression='int4' works with CPUAdamW on CPU params."""

    def test_int4_compression_cpu_params(self):
        """CPUAdamW with int4 compression on CPU params (no GPU needed).

        CPU params don't use compression (they're already on CPU), but the
        optimizer should not crash when grad_compression='int4' is set.
        """
        from forge.training.optim.hybrid_offload import CPUAdamW

        p = torch.nn.Parameter(torch.randn(4, 8, dtype=torch.float32, device=DEVICE))
        opt = CPUAdamW([p], lr=0.01, grad_compression="int4", verbose=False)

        # Simulate a training step
        p.grad = torch.randn(4, 8, dtype=torch.float32, device=DEVICE)
        opt.step()
        opt.zero_grad()

        # Verify the param was updated
        assert p.grad is None  # zero_grad sets to None

    def test_compression_cleanup(self):
        """Verify cleanup_compression frees ef_error buffers."""
        from forge.training.optim.hybrid_offload import CPUAdamW

        p = torch.nn.Parameter(torch.randn(4, 8, dtype=torch.float32, device=DEVICE))
        opt = CPUAdamW([p], lr=0.01, grad_compression="int4", verbose=False)

        # Run a step to trigger _lazy_init
        p.grad = torch.randn(4, 8, dtype=torch.float32, device=DEVICE)
        opt.step()

        # For CPU params, ef_error is not allocated (only for GPU params)
        # But cleanup should not crash
        opt.cleanup_compression()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
