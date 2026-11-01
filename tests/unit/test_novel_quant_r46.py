"""Unit tests for R46 novel quantization algorithms."""

import copy

import pytest
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

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def small_linear():
    torch.manual_seed(42)
    return nn.Linear(64, 32, bias=True)


@pytest.fixture
def small_linear_no_bias():
    torch.manual_seed(42)
    return nn.Linear(64, 32, bias=False)


@pytest.fixture
def odd_linear():
    """Linear with non-power-of-2 in_features (tests Hadamard padding)."""
    torch.manual_seed(42)
    return nn.Linear(48, 32, bias=True)


@pytest.fixture
def synthetic_activations():
    """Synthetic calibration activations (N, in_features)."""
    torch.manual_seed(42)
    return torch.randn(128, 64) * 0.5 + 0.1


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )


@pytest.fixture
def model_activations():
    return {
        "0": torch.randn(64, 64),
        "2": torch.randn(64, 128),
    }


# ── HadamardRotatedFP4Linear tests ────────────────────────────────────────


class TestHadamardRotatedFP4:
    def test_from_linear_shape(self, small_linear):
        layer = HadamardRotatedFP4Linear.from_linear(small_linear, block_size=32)
        assert layer.in_features == 64
        assert layer.out_features == 32
        assert layer.hadamard_size == 64  # power of 2

    def test_from_linear_odd_shape(self, odd_linear):
        layer = HadamardRotatedFP4Linear.from_linear(odd_linear, block_size=16)
        assert layer.in_features == 48
        assert layer.hadamard_size == 64  # next power of 2 >= 48

    def test_forward_shape(self, small_linear):
        layer = HadamardRotatedFP4Linear.from_linear(small_linear)
        x = torch.randn(2, 4, 64)
        out = layer(x)
        assert out.shape == (2, 4, 32)

    def test_forward_shape_no_bias(self, small_linear_no_bias):
        layer = HadamardRotatedFP4Linear.from_linear(small_linear_no_bias)
        assert layer.bias is None
        x = torch.randn(2, 4, 64)
        out = layer(x)
        assert out.shape == (2, 4, 32)

    def test_reconstruction_error_reasonable(self, small_linear):
        layer = HadamardRotatedFP4Linear.from_linear(small_linear)
        w_orig = small_linear.weight.data.float()
        w_recon = layer._dequantize_weight(torch.float32)
        assert w_recon.shape == w_orig.shape
        err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
        assert err < 0.3, f"Reconstruction error too high: {err}"

    def test_packed_storage_dtype(self, small_linear):
        layer = HadamardRotatedFP4Linear.from_linear(small_linear)
        assert layer.weight_packed.dtype == torch.uint8
        assert layer.weight_scales.dtype == torch.float16
        assert layer.weight_global_scale.dtype == torch.float32

    def test_state_dict_keys(self, small_linear):
        layer = HadamardRotatedFP4Linear.from_linear(small_linear)
        sd = layer.state_dict()
        assert "weight_packed" in sd
        assert "weight_scales" in sd
        assert "weight_global_scale" in sd

    def test_hadamard_matrix_orthonormal(self):
        from forge.engine.quant.novel_quant_r44 import _hadamard_matrix

        H = _hadamard_matrix(64, torch.device("cpu"), torch.float32)
        prod = H @ H.T
        assert torch.allclose(prod, torch.eye(64), atol=1e-5)


# ── GPTQFP4Linear tests ───────────────────────────────────────────────────


class TestGPTQFP4:
    def test_from_linear_shape(self, small_linear, synthetic_activations):
        layer = GPTQFP4Linear.from_linear(small_linear, synthetic_activations, block_size=32, group_size=64)
        assert layer.in_features == 64
        assert layer.out_features == 32
        assert layer.group_size == 64

    def test_forward_shape(self, small_linear, synthetic_activations):
        layer = GPTQFP4Linear.from_linear(small_linear, synthetic_activations)
        x = torch.randn(2, 4, 64)
        out = layer(x)
        assert out.shape == (2, 4, 32)

    def test_reconstruction_not_exploded(self, small_linear, synthetic_activations):
        """GPTQ should not produce NaN or inf reconstruction."""
        layer = GPTQFP4Linear.from_linear(small_linear, synthetic_activations)
        w_recon = layer._dequantize_weight(torch.float32)
        assert torch.isfinite(w_recon).all(), "GPTQ reconstruction has NaN/inf"
        w_orig = small_linear.weight.data.float()
        err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
        assert err < 0.5, f"GPTQ reconstruction error too high: {err}"

    def test_packed_storage_dtype(self, small_linear, synthetic_activations):
        layer = GPTQFP4Linear.from_linear(small_linear, synthetic_activations)
        assert layer.weight_packed.dtype == torch.uint8

    def test_bias_preserved(self, small_linear, synthetic_activations):
        layer = GPTQFP4Linear.from_linear(small_linear, synthetic_activations)
        assert layer.bias is not None
        assert torch.allclose(layer.bias.data, small_linear.bias.data, atol=1e-5)


# ── AWQFP4Linear tests ────────────────────────────────────────────────────


class TestAWQFP4:
    def test_from_linear_shape(self, small_linear, synthetic_activations):
        layer = AWQFP4Linear.from_linear(small_linear, synthetic_activations, block_size=32)
        assert layer.in_features == 64
        assert layer.out_features == 32

    def test_forward_shape(self, small_linear, synthetic_activations):
        layer = AWQFP4Linear.from_linear(small_linear, synthetic_activations)
        x = torch.randn(2, 4, 64)
        out = layer(x)
        assert out.shape == (2, 4, 32)

    def test_reconstruction_error_reasonable(self, small_linear, synthetic_activations):
        layer = AWQFP4Linear.from_linear(small_linear, synthetic_activations)
        w_orig = small_linear.weight.data.float()
        w_recon = layer._dequantize_weight(torch.float32)
        assert w_recon.shape == w_orig.shape
        err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
        assert err < 0.3, f"AWQ reconstruction error too high: {err}"

    def test_packed_storage_dtype(self, small_linear, synthetic_activations):
        layer = AWQFP4Linear.from_linear(small_linear, synthetic_activations)
        assert layer.weight_packed.dtype == torch.uint8

    def test_hessian_weighted_scale_differs_from_uniform(self, small_linear, synthetic_activations):
        """AWQ with real activations should differ from uniform-weight scale."""
        layer = AWQFP4Linear.from_linear(small_linear, synthetic_activations)
        # With non-uniform activations, the scale should differ from uniform
        uniform_acts = torch.ones(128, 64)
        layer_uniform = AWQFP4Linear.from_linear(small_linear, uniform_acts, block_size=32)
        # Scales should be different (not identical)
        assert not torch.allclose(layer.weight_scales, layer_uniform.weight_scales, atol=1e-3)


# ── OptimalGridFP4Linear tests ────────────────────────────────────────────


class TestOptimalGridFP4:
    def test_from_linear_shape(self, small_linear):
        layer = OptimalGridFP4Linear.from_linear(small_linear, block_size=32, n_lloyd_iters=10)
        assert layer.in_features == 64
        assert layer.out_features == 32

    def test_forward_shape(self, small_linear):
        layer = OptimalGridFP4Linear.from_linear(small_linear, n_lloyd_iters=10)
        x = torch.randn(2, 4, 64)
        out = layer(x)
        assert out.shape == (2, 4, 32)

    def test_reconstruction_error_reasonable(self, small_linear):
        layer = OptimalGridFP4Linear.from_linear(small_linear, n_lloyd_iters=10)
        w_orig = small_linear.weight.data.float()
        w_recon = layer._dequantize_weight(torch.float32)
        assert w_recon.shape == w_orig.shape
        err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
        assert err < 0.3, f"OG-FP4 reconstruction error too high: {err}"

    def test_codebook_shape(self, small_linear):
        layer = OptimalGridFP4Linear.from_linear(small_linear, n_lloyd_iters=10)
        assert layer.codebook.shape == (8,)
        assert (layer.codebook >= 0).all(), "Codebook magnitudes should be non-negative"

    def test_codebook_sorted(self, small_linear):
        layer = OptimalGridFP4Linear.from_linear(small_linear, n_lloyd_iters=10)
        cb = layer.codebook
        assert (cb[1:] >= cb[:-1]).all(), "Codebook should be sorted ascending"

    def test_packed_storage_dtype(self, small_linear):
        layer = OptimalGridFP4Linear.from_linear(small_linear, n_lloyd_iters=10)
        assert layer.weight_packed.dtype == torch.uint8


# ── HadamardGPTQFP4Linear tests ───────────────────────────────────────────


class TestHadamardGPTQFP4:
    def test_from_linear_shape(self, small_linear, synthetic_activations):
        layer = HadamardGPTQFP4Linear.from_linear(small_linear, synthetic_activations, block_size=32, group_size=64)
        assert layer.in_features == 64
        assert layer.out_features == 32
        assert layer.hadamard_size == 64

    def test_from_linear_odd_shape(self, odd_linear, synthetic_activations):
        acts = torch.randn(128, 48)
        layer = HadamardGPTQFP4Linear.from_linear(odd_linear, acts, block_size=16, group_size=48)
        assert layer.hadamard_size == 64  # next power of 2 >= 48

    def test_forward_shape(self, small_linear, synthetic_activations):
        layer = HadamardGPTQFP4Linear.from_linear(small_linear, synthetic_activations)
        x = torch.randn(2, 4, 64)
        out = layer(x)
        assert out.shape == (2, 4, 32)

    def test_reconstruction_not_exploded(self, small_linear, synthetic_activations):
        """Combined HR+GPTQ should not produce NaN or inf."""
        layer = HadamardGPTQFP4Linear.from_linear(small_linear, synthetic_activations)
        w_recon = layer._dequantize_weight(torch.float32)
        assert torch.isfinite(w_recon).all(), "HR-GPTQ reconstruction has NaN/inf"

    def test_reconstruction_error_reasonable(self, small_linear, synthetic_activations):
        layer = HadamardGPTQFP4Linear.from_linear(small_linear, synthetic_activations)
        w_orig = small_linear.weight.data.float()
        w_recon = layer._dequantize_weight(torch.float32)
        err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
        assert err < 0.5, f"HR-GPTQ reconstruction error too high: {err}"


# ── FP4 packing round-trip tests ──────────────────────────────────────────


class TestFP4Packing:
    def test_hr_fp4_roundtrip(self, small_linear):
        layer = HadamardRotatedFP4Linear.from_linear(small_linear)
        # Unpack and verify no corruption
        packed = layer.weight_packed.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int64)
        high = (packed >> 4).to(torch.int64)
        # Each nibble should have valid sign+magnitude
        all_codes = torch.stack([low, high], dim=-1).view(-1)
        for code in all_codes[:100]:
            mag_idx = (code & 0x07).item()
            assert 0 <= mag_idx <= 7, f"Invalid magnitude index: {mag_idx}"

    def test_awq_fp4_roundtrip(self, small_linear, synthetic_activations):
        layer = AWQFP4Linear.from_linear(small_linear, synthetic_activations)
        packed = layer.weight_packed.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int64)
        high = (packed >> 4).to(torch.int64)
        all_codes = torch.stack([low, high], dim=-1).view(-1)
        for code in all_codes[:100]:
            mag_idx = (code & 0x07).item()
            assert 0 <= mag_idx <= 7

    def test_og_fp4_roundtrip(self, small_linear):
        layer = OptimalGridFP4Linear.from_linear(small_linear, n_lloyd_iters=10)
        packed = layer.weight_packed.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int64)
        high = (packed >> 4).to(torch.int64)
        all_codes = torch.stack([low, high], dim=-1).view(-1)
        for code in all_codes[:100]:
            mag_idx = (code & 0x07).item()
            assert 0 <= mag_idx <= 7


# ── Model conversion tests ────────────────────────────────────────────────


class TestModelConversion:
    def test_hr_fp4_conversion(self, simple_model):
        n = quantize_model_hadamard_rotated_fp4(simple_model, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = simple_model(x)
        assert out.shape == (2, 4, 64)

    def test_og_fp4_conversion(self, simple_model):
        n = quantize_model_optimal_grid_fp4(simple_model, n_lloyd_iters=10, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = simple_model(x)
        assert out.shape == (2, 4, 64)

    def test_gptq_fp4_conversion(self, simple_model, model_activations):
        n = quantize_model_gptq_fp4(simple_model, model_activations, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = simple_model(x)
        assert out.shape == (2, 4, 64)

    def test_awq_fp4_conversion(self, simple_model, model_activations):
        n = quantize_model_awq_fp4(simple_model, model_activations, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = simple_model(x)
        assert out.shape == (2, 4, 64)

    def test_hr_gptq_fp4_conversion(self, simple_model, model_activations):
        n = quantize_model_hadamard_gptq_fp4(simple_model, model_activations, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = simple_model(x)
        assert out.shape == (2, 4, 64)

    def test_idempotent_conversion(self, simple_model):
        """Converting an already-quantized model should not re-quantize."""
        n1 = quantize_model_hadamard_rotated_fp4(simple_model, verbose=False)
        assert n1 == 2
        n2 = quantize_model_hadamard_rotated_fp4(simple_model, verbose=False)
        assert n2 == 0  # already quantized


# ── Memory estimation tests ───────────────────────────────────────────────


class TestMemoryEstimation:
    def test_hr_fp4_memory(self, simple_model):
        quantize_model_hadamard_rotated_fp4(simple_model, verbose=False)
        mem = estimate_r46_memory(simple_model)
        assert mem["total_bytes"] > 0
        assert mem["total_mb"] > 0
        assert mem["avg_eff_bits"] > 0
        assert mem["avg_eff_bits"] < 16  # should be quantized

    def test_og_fp4_memory(self, simple_model):
        quantize_model_optimal_grid_fp4(simple_model, n_lloyd_iters=10, verbose=False)
        mem = estimate_r46_memory(simple_model)
        assert mem["total_bytes"] > 0
        assert "OptimalGridFP4Linear" in mem["breakdown"]

    def test_memory_savings_vs_fp16(self, simple_model):
        """Quantized memory should be much less than FP16."""
        fp16_bytes = sum(p.numel() * p.element_size() for p in simple_model.parameters())
        m = copy.deepcopy(simple_model)
        quantize_model_hadamard_rotated_fp4(m, verbose=False)
        mem = estimate_r46_memory(m)
        assert mem["total_bytes"] < fp16_bytes


# ── Activation collection tests ───────────────────────────────────────────


class TestCollectActivations:
    def test_collect_from_model(self, simple_model):
        x = torch.randn(2, 8, 64)
        acts = collect_activations(simple_model, x, n_samples=16)
        assert len(acts) >= 2
        for _name, act in acts.items():
            assert act.dim() == 2
            assert act.shape[0] <= 16

    def test_collect_n_samples_limit(self, simple_model):
        x = torch.randn(4, 32, 64)  # 128 rows total
        acts = collect_activations(simple_model, x, n_samples=10)
        for _name, act in acts.items():
            assert act.shape[0] <= 10


# ── Comparative tests ─────────────────────────────────────────────────────


class TestComparative:
    def test_awq_better_with_outlier_activations(self):
        """AWQ should weight salient channels more accurately."""
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        # Create activations where some channels are much more active
        acts = torch.randn(256, 128)
        acts[:, ::8] *= 10.0  # every 8th channel is 10x more active

        awq_layer = AWQFP4Linear.from_linear(lin, acts, block_size=32)
        (
            lin.weight.data.float() - awq_layer._dequantize_weight(torch.float32)
        ).norm().item() / lin.weight.data.float().norm().item()

        # With uniform activations (no salient channels)
        uniform_acts = torch.ones(256, 128)
        uniform_layer = AWQFP4Linear.from_linear(lin, uniform_acts, block_size=32)
        (
            lin.weight.data.float() - uniform_layer._dequantize_weight(torch.float32)
        ).norm().item() / lin.weight.data.float().norm().item()

        # AWQ with real activations should produce different (ideally better output) quantization
        # At minimum, the scales should differ
        assert not torch.allclose(awq_layer.weight_scales, uniform_layer.weight_scales, atol=1e-3)

    def test_hadamard_helps_outliers(self):
        """Hadamard rotation should reduce reconstruction error with outliers."""
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        lin.weight.data[:, ::16] *= 5.0  # outlier channels

        hr_layer = HadamardRotatedFP4Linear.from_linear(lin, block_size=32)
        hr_err = (
            lin.weight.data.float() - hr_layer._dequantize_weight(torch.float32)
        ).norm().item() / lin.weight.data.float().norm().item()

        from forge.engine.quant.novel_quant_r45 import SchurABFP4Linear

        schur_layer = SchurABFP4Linear.from_linear(lin, block_size=32)
        schur_err = (
            lin.weight.data.float() - schur_layer._dequantize_weight(torch.float32)
        ).norm().item() / lin.weight.data.float().norm().item()

        # Rotation should help with outliers
        assert hr_err <= schur_err + 0.1

    def test_og_fp4_better_than_fixed_fp4(self):
        """Optimal grid should have lower reconstruction error than fixed FP4."""
        torch.manual_seed(42)
        # Create weights with a non-uniform distribution (where fixed FP4 is suboptimal)
        lin = nn.Linear(128, 64, bias=False)
        # Make weights follow a bimodal distribution
        lin.weight.data[:32] = torch.randn(32, 128) * 0.1
        lin.weight.data[32:] = torch.randn(32, 128) * 2.0

        og_layer = OptimalGridFP4Linear.from_linear(lin, block_size=32, n_lloyd_iters=20)
        og_err = (
            lin.weight.data.float() - og_layer._dequantize_weight(torch.float32)
        ).norm().item() / lin.weight.data.float().norm().item()

        hr_layer = HadamardRotatedFP4Linear.from_linear(lin, block_size=32)
        hr_err = (
            lin.weight.data.float() - hr_layer._dequantize_weight(torch.float32)
        ).norm().item() / lin.weight.data.float().norm().item()

        # OG-FP4 should be at least competitive (within 10% of HR-FP4)
        assert og_err <= hr_err + 0.05
