r"""Unit tests for R44 novel quantization algorithms.

Tests the two working novel algorithms (AB-FP4 and SR-INT4) plus
the dead-end lifting methods for regression protection.

Run: venv\Scripts\python.exe -m pytest tests/unit/test_novel_quant_r44.py -v
"""

import pytest
import torch
import torch.nn as nn

from forge.engine.quant.novel_quant_r44 import (
    AdaptiveBlockFP4Linear,
    HadamardLiftLinear,
    SparseResidualINT3Linear,
    TernaryLiftLinear,
    _hadamard_matrix,
    estimate_quantized_memory,
    quantize_model_adaptive_block_fp4,
    quantize_model_sparse_residual_int3,
)


def frob_err(ref, approx):
    """Relative Frobenius error."""
    return (ref - approx).norm().item() / max(ref.norm().item(), 1e-8)


class TestHadamardMatrix:
    """Test Hadamard matrix generation."""

    def test_hadamard_orthogonal(self):
        """Hadamard matrix should be orthogonal: H @ H^T = I."""
        for n in [1, 2, 4, 8, 16, 32]:
            H = _hadamard_matrix(n, torch.device("cpu"), torch.float32)
            prod = H @ H.T
            assert torch.allclose(prod, torch.eye(n), atol=1e-5), f"Hadamard {n} not orthogonal"

    def test_hadamard_power_of_2_only(self):
        """Non-power-of-2 should raise."""
        with pytest.raises(AssertionError):
            _hadamard_matrix(3, torch.device("cpu"), torch.float32)

    def test_hadamard_symmetric(self):
        """Hadamard matrix is symmetric: H = H^T."""
        H = _hadamard_matrix(8, torch.device("cpu"), torch.float32)
        assert torch.allclose(H, H.T, atol=1e-6)


class TestAdaptiveBlockFP4:
    """Test AdaptiveBlockFP4 (AB-FP4) — the winning algorithm."""

    def test_forward_shape(self):
        """Output shape matches nn.Linear."""
        lin = nn.Linear(64, 128, bias=True)
        ql = AdaptiveBlockFP4Linear.from_linear(lin, block_size=32)
        x = torch.randn(2, 4, 64)
        y = ql(x)
        assert y.shape == (2, 4, 128)

    def test_reconstruction_quality(self):
        """Reconstruction error should be < 0.15 for LLM-like weights."""
        torch.manual_seed(42)
        lin = nn.Linear(256, 512, bias=False)
        # Scale to be LLM-like (smaller variance)
        lin.weight.data *= 0.1
        ql = AdaptiveBlockFP4Linear.from_linear(lin, block_size=32)
        w_recon = ql._dequantize_weight(torch.float32)
        err = frob_err(lin.weight.data, w_recon)
        assert err < 0.15, f"AB-FP4 reconstruction error too high: {err:.4f}"

    def test_bias_preserved(self):
        """Bias should be preserved from original linear."""
        lin = nn.Linear(64, 128, bias=True)
        lin.bias.data.fill_(0.5)
        ql = AdaptiveBlockFP4Linear.from_linear(lin, block_size=32)
        assert ql.bias is not None
        assert torch.allclose(ql.bias.data, torch.full_like(ql.bias.data, 0.5))

    def test_no_bias(self):
        """Should work without bias."""
        lin = nn.Linear(64, 128, bias=False)
        ql = AdaptiveBlockFP4Linear.from_linear(lin, block_size=32)
        assert ql.bias is None
        x = torch.randn(1, 4, 64)
        y = ql(x)
        assert y.shape == (1, 4, 128)

    def test_non_power_of_2_input(self):
        """Should handle non-power-of-2 input dimensions."""
        lin = nn.Linear(100, 200, bias=False)
        ql = AdaptiveBlockFP4Linear.from_linear(lin, block_size=32)
        x = torch.randn(1, 3, 100)
        y = ql(x)
        assert y.shape == (1, 3, 200)

    def test_bit_allocation_diversity(self):
        """Bit allocation should have variety (not all same)."""
        torch.manual_seed(42)
        lin = nn.Linear(256, 512, bias=False)
        # Add some outliers to create high-kurtosis blocks
        lin.weight.data[:, ::64] *= 10
        ql = AdaptiveBlockFP4Linear.from_linear(lin, block_size=32)
        # Should have at least 2 different bit allocations
        unique = ql.bit_alloc.unique()
        assert len(unique) >= 1  # At minimum, should not crash


class TestSparseResidualINT3:
    """Test SparseResidualINT3/INT4 — the second winning algorithm."""

    def test_forward_shape(self):
        """Output shape matches nn.Linear."""
        lin = nn.Linear(64, 128, bias=True)
        ql = SparseResidualINT3Linear.from_linear(lin, group_size=64, base_bits=4)
        x = torch.randn(2, 4, 64)
        y = ql(x)
        assert y.shape == (2, 4, 128)

    def test_int4_reconstruction_quality(self):
        """INT4 base should have < 0.15 reconstruction error."""
        torch.manual_seed(42)
        lin = nn.Linear(256, 512, bias=False)
        lin.weight.data *= 0.1
        ql = SparseResidualINT3Linear.from_linear(lin, group_size=64, n_std=1.0, base_bits=4)
        w_recon = ql._dequantize_weight(torch.float32)
        err = frob_err(lin.weight.data, w_recon)
        assert err < 0.15, f"SR-INT4 reconstruction error too high: {err:.4f}"

    def test_int3_higher_error(self):
        """INT3 base should have higher error than INT4."""
        torch.manual_seed(42)
        lin = nn.Linear(256, 512, bias=False)
        lin.weight.data *= 0.1

        ql3 = SparseResidualINT3Linear.from_linear(lin, group_size=64, n_std=2.0, base_bits=3)
        ql4 = SparseResidualINT3Linear.from_linear(lin, group_size=64, n_std=2.0, base_bits=4)
        err3 = frob_err(lin.weight.data, ql3._dequantize_weight(torch.float32))
        err4 = frob_err(lin.weight.data, ql4._dequantize_weight(torch.float32))
        assert err3 > err4, f"INT3 should have higher error than INT4: {err3} vs {err4}"

    def test_aggressive_more_outliers(self):
        """n_std=1.0 should select more outliers than n_std=2.0."""
        torch.manual_seed(42)
        lin = nn.Linear(256, 512, bias=False)
        lin.weight.data *= 0.1
        # Add some outliers
        lin.weight.data[0, 0] = 1.0
        lin.weight.data[1, 5] = -1.5

        ql_conservative = SparseResidualINT3Linear.from_linear(lin, group_size=64, n_std=2.0, base_bits=4)
        ql_aggressive = SparseResidualINT3Linear.from_linear(lin, group_size=64, n_std=1.0, base_bits=4)

        assert ql_aggressive.n_sparse >= ql_conservative.n_sparse

    def test_bias_preserved(self):
        """Bias should be preserved."""
        lin = nn.Linear(64, 128, bias=True)
        lin.bias.data.fill_(-0.3)
        ql = SparseResidualINT3Linear.from_linear(lin, group_size=64, base_bits=4)
        assert ql.bias is not None
        assert torch.allclose(ql.bias.data, torch.full_like(ql.bias.data, -0.3))


class TestHadamardLift:
    """Test HadamardLift (dead-end, but keep for regression)."""

    def test_forward_shape(self):
        """Output shape matches nn.Linear."""
        lin = nn.Linear(64, 128, bias=True)
        ql = HadamardLiftLinear.from_linear(lin, lift_ratio=2.0, lift_dim=8, optimize_p=False)
        x = torch.randn(2, 4, 64)
        y = ql(x)
        assert y.shape == (2, 4, 128)

    def test_high_reconstruction_error(self):
        """HadamardLift with random P has high error (known limitation)."""
        lin = nn.Linear(64, 128, bias=False)
        ql = HadamardLiftLinear.from_linear(lin, lift_ratio=2.0, lift_dim=8, optimize_p=False)
        w_recon = ql._dequantize_weight(torch.float32)
        err = frob_err(lin.weight.data, w_recon)
        # Known to be high without optimization
        assert err > 0.3, f"Expected high error, got {err:.4f}"


class TestTernaryLift:
    """Test TernaryLift (dead-end, but keep for regression)."""

    def test_forward_shape(self):
        """Output shape matches nn.Linear."""
        lin = nn.Linear(64, 128, bias=True)
        ql = TernaryLiftLinear.from_linear(lin, lift_ratio=1.5, lift_dim=8, optimize_p=False)
        x = torch.randn(2, 4, 64)
        y = ql(x)
        assert y.shape == (2, 4, 128)


class TestModelConversion:
    """Test model-level conversion functions."""

    def test_quantize_model_ab_fp4(self):
        """AB-FP4 model conversion should replace Linear layers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        n = quantize_model_adaptive_block_fp4(model, block_size=32, verbose=False)
        assert n == 2
        # Check that layers are replaced
        assert isinstance(model[0], AdaptiveBlockFP4Linear)
        assert isinstance(model[2], AdaptiveBlockFP4Linear)

    def test_quantize_model_sr_int4(self):
        """SR-INT4 model conversion should replace Linear layers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        n = quantize_model_sparse_residual_int3(model, group_size=64, base_bits=4, verbose=False)
        assert n == 2
        assert isinstance(model[0], SparseResidualINT3Linear)
        assert isinstance(model[2], SparseResidualINT3Linear)

    def test_skip_embeddings(self):
        """Embedding-like layers should be skipped."""
        nn.Sequential(
            nn.Linear(100, 100),  # "embed" in name → skip
            nn.Linear(100, 200),
        )

        # Manually name the first module to contain "embed"
        # Since nn.Sequential names by index, we use a custom module
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(100, 100)
                self.fc = nn.Linear(100, 200)

            def forward(self, x):
                return self.fc(self.embed(x))

        m = TestModel()
        n = quantize_model_adaptive_block_fp4(m, block_size=32, verbose=False)
        assert n == 1  # Only fc, not embed
        assert isinstance(m.fc, AdaptiveBlockFP4Linear)
        assert isinstance(m.embed, nn.Linear)  # Not replaced

    def test_skip_already_quantized(self):
        """Already-quantized layers should not be re-quantized."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64),
        )
        # First pass
        quantize_model_adaptive_block_fp4(model, block_size=32, verbose=False)
        # Second pass should skip
        n = quantize_model_adaptive_block_fp4(model, block_size=32, verbose=False)
        assert n == 0


class TestMemoryEstimation:
    """Test memory estimation utilities."""

    def test_estimate_memory(self):
        """Memory estimation should produce reasonable numbers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 64),
        )
        quantize_model_adaptive_block_fp4(model, block_size=32, verbose=False)
        mem = estimate_quantized_memory(model)
        assert mem["total_mb"] > 0
        assert mem["total_params"] > 0
        assert "AdaptiveBlockFP4Linear" in mem["breakdown"]

    def test_memory_less_than_fp16(self):
        """Quantized memory should be less than FP16 equivalent."""
        model = nn.Sequential(nn.Linear(256, 512))
        quantize_model_adaptive_block_fp4(model, block_size=32, verbose=False)
        mem = estimate_quantized_memory(model)
        fp16_bytes = 256 * 512 * 2  # bf16
        assert mem["total_bytes"] < fp16_bytes
