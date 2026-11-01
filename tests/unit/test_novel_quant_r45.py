"""Unit tests for R45 novel quantization algorithms."""

import copy

import pytest
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

# ──────────────────────────────────────────────────────────────────────────
# Haar wavelet matrix tests
# ──────────────────────────────────────────────────────────────────────────


class TestHaarMatrix:
    def test_orthonormal_8(self):
        H = _haar_matrix(8, torch.device("cpu"), torch.float32)
        prod = H @ H.T
        assert torch.allclose(prod, torch.eye(8), atol=1e-5)

    def test_orthonormal_16(self):
        H = _haar_matrix(16, torch.device("cpu"), torch.float32)
        prod = H @ H.T
        assert torch.allclose(prod, torch.eye(16), atol=1e-5)

    def test_orthonormal_4(self):
        H = _haar_matrix(4, torch.device("cpu"), torch.float32)
        prod = H @ H.T
        assert torch.allclose(prod, torch.eye(4), atol=1e-5)

    def test_constant_signal_only_dc(self):
        """Constant signal should only have DC (first) coefficient."""
        x = torch.ones(1, 8) * 5.0
        H = _haar_matrix(8, torch.device("cpu"), torch.float32)
        coeffs = x @ H.T
        assert coeffs[0, 0].abs() > 1.0
        assert coeffs[0, 1:].abs().max() < 1e-5

    def test_inverse_reconstruction(self):
        """Forward then inverse should reconstruct original."""
        torch.manual_seed(42)
        x = torch.randn(3, 8)
        H = _haar_matrix(8, torch.device("cpu"), torch.float32)
        coeffs = x @ H.T  # forward
        recon = coeffs @ H  # inverse (H is orthonormal)
        assert torch.allclose(recon, x, atol=1e-5)

    def test_power_of_2_required(self):
        with pytest.raises(AssertionError):
            _haar_matrix(7, torch.device("cpu"), torch.float32)


# ──────────────────────────────────────────────────────────────────────────
# WaveletLift tests
# ──────────────────────────────────────────────────────────────────────────


class TestWaveletLift:
    def test_shape_preservation(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=True)
        layer = WaveletLiftLinear.from_linear(lin, rank=32)
        x = torch.randn(2, 4, 128)
        out = layer(x)
        assert out.shape == (2, 4, 64)

    def test_bias_preserved(self):
        lin = nn.Linear(128, 64, bias=True)
        layer = WaveletLiftLinear.from_linear(lin, rank=32)
        assert layer.bias is not None
        assert torch.allclose(layer.bias.data, lin.bias.data)

    def test_no_bias(self):
        lin = nn.Linear(128, 64, bias=False)
        layer = WaveletLiftLinear.from_linear(lin, rank=32)
        assert layer.bias is None

    def test_reconstruction_error_reasonable(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        layer = WaveletLiftLinear.from_linear(lin, rank=32)
        w_orig = lin.weight.data.float()
        w_recon = layer._dequantize_weight(torch.float32)
        err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
        assert err < 1.0

    def test_higher_rank_lower_error(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        errors = {}
        for rank in [8, 16, 32, 64]:
            layer = WaveletLiftLinear.from_linear(lin, rank=rank)
            w_orig = lin.weight.data.float()
            w_recon = layer._dequantize_weight(torch.float32)
            err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
            errors[rank] = err
        assert errors[64] < errors[8]

    def test_state_dict_contents(self):
        """State dict should contain all expected buffers with correct shapes."""
        lin = nn.Linear(128, 64, bias=True)
        layer = WaveletLiftLinear.from_linear(lin, rank=32)
        sd = layer.state_dict()
        assert "u_binary" in sd and sd["u_binary"].shape == (64, 32)
        assert "v_binary" in sd and sd["v_binary"].shape == (128, 32)
        assert "row_scales" in sd and sd["row_scales"].shape == (64,)
        assert "col_scales" in sd and sd["col_scales"].shape == (128,)
        assert "rank_scales" in sd and sd["rank_scales"].shape == (32,)
        assert "bias" in sd


# ──────────────────────────────────────────────────────────────────────────
# SchurAB-FP4 tests
# ──────────────────────────────────────────────────────────────────────────


class TestSchurABFP4:
    def test_shape_preservation(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=True)
        layer = SchurABFP4Linear.from_linear(lin, block_size=32, schur_iters=3)
        x = torch.randn(2, 4, 128)
        out = layer(x)
        assert out.shape == (2, 4, 64)

    def test_bias_preserved(self):
        lin = nn.Linear(128, 64, bias=True)
        layer = SchurABFP4Linear.from_linear(lin, block_size=32)
        assert layer.bias is not None
        assert torch.allclose(layer.bias.data, lin.bias.data)

    def test_reconstruction_error_low(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        layer = SchurABFP4Linear.from_linear(lin, block_size=32, schur_iters=3)
        w_orig = lin.weight.data.float()
        w_recon = layer._dequantize_weight(torch.float32)
        err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
        assert err < 0.2, f"Reconstruction error too high: {err}"

    def test_schur_refinement_helps(self):
        """Schur refinement should not increase error (greedy improvement)."""
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        # Without Schur refinement
        layer_no = SchurABFP4Linear.from_linear(lin, block_size=32, schur_iters=0)
        # With Schur refinement
        layer_yes = SchurABFP4Linear.from_linear(lin, block_size=32, schur_iters=5)
        w_orig = lin.weight.data.float()
        err_no = (w_orig - layer_no._dequantize_weight(torch.float32)).norm().item() / w_orig.norm().item()
        err_yes = (w_orig - layer_yes._dequantize_weight(torch.float32)).norm().item() / w_orig.norm().item()
        assert err_yes <= err_no + 1e-6, f"Schur refinement increased error: {err_no} → {err_yes}"

    def test_packed_storage_size(self):
        """Packed weights should be half the size of unpacked (2 per byte)."""
        lin = nn.Linear(128, 64, bias=False)
        layer = SchurABFP4Linear.from_linear(lin, block_size=32)
        # 128 padded to 128, 64 output → 64*128 = 8192 elements
        # Packed: 8192 / 2 = 4096 bytes
        assert layer.weight_packed.shape[1] == 64  # 128 / 2

    def test_state_dict_contents(self):
        """State dict should contain all expected buffers with correct shapes."""
        lin = nn.Linear(128, 64, bias=True)
        layer = SchurABFP4Linear.from_linear(lin, block_size=32)
        sd = layer.state_dict()
        assert "weight_packed" in sd
        assert "weight_scales" in sd
        assert "weight_global_scale" in sd
        assert "bit_alloc" in sd
        assert "bias" in sd


# ──────────────────────────────────────────────────────────────────────────
# SVDLiftBinary tests
# ──────────────────────────────────────────────────────────────────────────


class TestSVDLiftBinary:
    def test_shape_preservation(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=True)
        layer = SVDLiftBinaryLinear.from_linear(lin, rank=32)
        x = torch.randn(2, 4, 128)
        out = layer(x)
        assert out.shape == (2, 4, 64)

    def test_bias_preserved(self):
        lin = nn.Linear(128, 64, bias=True)
        layer = SVDLiftBinaryLinear.from_linear(lin, rank=32)
        assert layer.bias is not None
        assert torch.allclose(layer.bias.data, lin.bias.data)

    def test_reconstruction_error_reasonable(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        layer = SVDLiftBinaryLinear.from_linear(lin, rank=32)
        w_orig = lin.weight.data.float()
        w_recon = layer._dequantize_weight(torch.float32)
        err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
        assert err < 1.0

    def test_higher_rank_lower_error(self):
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        errors = {}
        for rank in [8, 16, 32, 64]:
            layer = SVDLiftBinaryLinear.from_linear(lin, rank=rank)
            w_orig = lin.weight.data.float()
            w_recon = layer._dequantize_weight(torch.float32)
            err = (w_orig - w_recon).norm().item() / w_orig.norm().item()
            errors[rank] = err
        assert errors[64] < errors[8]

    def test_binary_factors_are_pm1(self):
        lin = nn.Linear(128, 64, bias=False)
        layer = SVDLiftBinaryLinear.from_linear(lin, rank=32)
        assert set(layer.a_binary.unique().tolist()) <= {-1, 1}
        assert set(layer.b_binary.unique().tolist()) <= {-1, 1}

    def test_state_dict_contents(self):
        """State dict should contain all expected buffers with correct shapes."""
        lin = nn.Linear(128, 64, bias=True)
        layer = SVDLiftBinaryLinear.from_linear(lin, rank=32)
        sd = layer.state_dict()
        assert "a_binary" in sd and sd["a_binary"].shape == (64, 32)
        assert "b_binary" in sd and sd["b_binary"].shape == (128, 32)
        assert "row_scales" in sd and sd["row_scales"].shape == (64,)
        assert "col_scales" in sd and sd["col_scales"].shape == (128,)
        assert "rank_scales" in sd and sd["rank_scales"].shape == (32,)
        assert "bias" in sd


# ──────────────────────────────────────────────────────────────────────────
# LloydMaxRotatedKV tests
# ──────────────────────────────────────────────────────────────────────────


class TestLloydMaxKV:
    def test_2bit_shape_preservation(self):
        torch.manual_seed(42)
        q = LloydMaxRotatedKVQuantizer(bits=2, head_dim=128, use_qjl=True)
        kv = torch.randn(1, 2, 16, 128)
        packed = q.quantize(kv)
        recon = q.dequantize(packed)
        assert recon.shape == kv.shape

    def test_3bit_shape_preservation(self):
        torch.manual_seed(42)
        q = LloydMaxRotatedKVQuantizer(bits=3, head_dim=128, use_qjl=True)
        kv = torch.randn(1, 2, 16, 128)
        packed = q.quantize(kv)
        recon = q.dequantize(packed)
        assert recon.shape == kv.shape

    def test_2bit_reconstruction_error(self):
        torch.manual_seed(42)
        q = LloydMaxRotatedKVQuantizer(bits=2, head_dim=128, use_qjl=True)
        kv = torch.randn(1, 2, 16, 128)
        packed = q.quantize(kv)
        recon = q.dequantize(packed)
        err = (kv - recon).norm().item() / kv.norm().item()
        assert err < 0.5

    def test_3bit_better_than_2bit(self):
        torch.manual_seed(42)
        kv = torch.randn(1, 2, 16, 128)
        q2 = LloydMaxRotatedKVQuantizer(bits=2, head_dim=128, use_qjl=True)
        q3 = LloydMaxRotatedKVQuantizer(bits=3, head_dim=128, use_qjl=True)
        r2 = q2.dequantize(q2.quantize(kv))
        r3 = q3.dequantize(q3.quantize(kv))
        err2 = (kv - r2).norm().item() / kv.norm().item()
        err3 = (kv - r3).norm().item() / kv.norm().item()
        assert err3 < err2

    def test_compression_ratio_2bit(self):
        q = LloydMaxRotatedKVQuantizer(bits=2, head_dim=128, use_qjl=True)
        ratio = q.compression_ratio()
        # 2 bits + 1 bit QJL + 16/128 bits scale = 3.125 bits per element
        # 16 / 3.125 = 5.12x
        assert 4.0 < ratio < 6.0

    def test_compression_ratio_3bit(self):
        q = LloydMaxRotatedKVQuantizer(bits=3, head_dim=128, use_qjl=True)
        ratio = q.compression_ratio()
        # 3 bits + 1 bit QJL + 16/128 bits scale = 4.125 bits per element
        # 16 / 4.125 = 3.88x
        assert 3.0 < ratio < 5.0

    def test_2d_input(self):
        torch.manual_seed(42)
        q = LloydMaxRotatedKVQuantizer(bits=2, head_dim=64, use_qjl=True)
        kv = torch.randn(16, 64)
        packed = q.quantize(kv)
        recon = q.dequantize(packed)
        assert recon.shape == kv.shape

    def test_no_qjl(self):
        torch.manual_seed(42)
        q = LloydMaxRotatedKVQuantizer(bits=2, head_dim=128, use_qjl=False)
        kv = torch.randn(1, 2, 16, 128)
        packed = q.quantize(kv)
        assert packed["qjl_packed"] is None
        recon = q.dequantize(packed)
        assert recon.shape == kv.shape

    def test_padding_when_head_dim_not_power_of_2(self):
        """head_dim=96 should pad to 128 internally."""
        torch.manual_seed(42)
        q = LloydMaxRotatedKVQuantizer(bits=2, head_dim=96, use_qjl=True)
        kv = torch.randn(1, 2, 16, 96)
        packed = q.quantize(kv)
        recon = q.dequantize(packed)
        assert recon.shape == kv.shape


# ──────────────────────────────────────────────────────────────────────────
# Model conversion tests
# ──────────────────────────────────────────────────────────────────────────


class TestModelConversion:
    def _make_model(self):
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def test_wavelet_lift_conversion(self):
        model = self._make_model()
        n = quantize_model_wavelet_lift(model, rank=16, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = model(x)
        assert out.shape == (2, 4, 64)

    def test_schur_ab_fp4_conversion(self):
        model = self._make_model()
        n = quantize_model_schur_ab_fp4(model, block_size=32, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = model(x)
        assert out.shape == (2, 4, 64)

    def test_svd_lift_conversion(self):
        model = self._make_model()
        n = quantize_model_svd_lift_binary(model, rank=16, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = model(x)
        assert out.shape == (2, 4, 64)

    def test_skips_embedding_layers(self):
        """Conversion should skip layers with 'embed' in name."""
        model = nn.Sequential(
            nn.Linear(64, 128),  # '0' — should be replaced
            nn.ReLU(),
            nn.Linear(128, 64),  # '2' — should be replaced
        )
        # Manually add an "embed" named layer
        model.embed = nn.Linear(64, 64)
        n = quantize_model_schur_ab_fp4(model, verbose=False)
        assert n == 2  # only the two in Sequential
        assert isinstance(model.embed, nn.Linear)  # not replaced

    def test_memory_estimation(self):
        model = self._make_model()
        quantize_model_schur_ab_fp4(model, verbose=False)
        mem = estimate_r45_memory(model)
        assert mem["total_bytes"] > 0
        assert mem["total_mb"] > 0
        assert "SchurABFP4Linear" in mem["breakdown"]

    def test_double_conversion_skips_existing(self):
        """Converting twice should not replace already-quantized layers."""
        model = self._make_model()
        n1 = quantize_model_schur_ab_fp4(model, verbose=False)
        n2 = quantize_model_schur_ab_fp4(model, verbose=False)
        assert n1 == 2
        assert n2 == 0  # already quantized


# ──────────────────────────────────────────────────────────────────────────
# ReQuant refinement tests
# ──────────────────────────────────────────────────────────────────────────


class TestReQuantRefine:
    def test_requant_runs_without_error(self):
        from forge.engine.quant.novel_quant_r44 import (
            quantize_model_adaptive_block_fp4,
        )

        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        orig = copy.deepcopy(model)
        quantize_model_adaptive_block_fp4(model, verbose=False)
        n = requant_refine_model(model, orig, verbose=False)
        assert n > 0

    def test_requant_does_not_increase_error(self):
        from forge.engine.quant.novel_quant_r44 import (
            quantize_model_adaptive_block_fp4,
        )

        def compute_err(m1, m2):
            errs = []
            for (_n1, mod1), (_n2, mod2) in zip(m1.named_modules(), m2.named_modules(), strict=False):
                if isinstance(mod1, nn.Linear) and hasattr(mod2, "_dequantize_weight"):
                    w1 = mod1.weight.data.float()
                    w2 = mod2._dequantize_weight(torch.float32)
                    if w1.shape == w2.shape:
                        errs.append((w1 - w2).norm().item() / w1.norm().item())
            return sum(errs) / max(len(errs), 1)

        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        orig = copy.deepcopy(model)
        quantize_model_adaptive_block_fp4(model, verbose=False)
        err_before = compute_err(orig, model)
        requant_refine_model(model, orig, verbose=False)
        err_after = compute_err(orig, model)
        assert err_after <= err_before + 1e-6
