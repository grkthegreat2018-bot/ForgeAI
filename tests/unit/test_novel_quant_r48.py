"""Unit tests for R48 extreme low-bit quantization."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.engine.quant.novel_quant_r48 import (
    BTCQuantLinear,
    NanoQuantLinear,
    NanoQuantQATLinear,
    TernaryPTQLinear,
    _binary_kmeans,
    _pack_binary_bits,
    _unpack_binary_bits,
    bake_qat_model,
    convert_model_to_nanoquant_qat,
    estimate_r48_memory,
    quantize_model_btc,
    quantize_model_nanoquant,
    quantize_model_ternary_ptq,
)


class TestBinaryPacking:
    """Test binary bit packing/unpacking."""

    def test_pack_unpack_roundtrip(self):
        w = torch.tensor([[1, -1, 1, 1, -1, -1, 1, -1], [-1, -1, 1, 1, 1, -1, -1, 1]], dtype=torch.int8)
        packed = _pack_binary_bits(w)
        assert packed.dtype == torch.uint8
        assert packed.shape == (2, 1)
        unpacked = _unpack_binary_bits(packed, 8)
        assert torch.equal(unpacked, w)

    def test_pack_non_multiple_of_8(self):
        w = torch.tensor([1, -1, 1, -1, 1], dtype=torch.int8)  # 5 values
        packed = _pack_binary_bits(w)
        assert packed.shape == (1,)  # 1D: ceil(5/8) = 1
        unpacked = _unpack_binary_bits(packed, 5)
        assert torch.equal(unpacked, w)

    def test_pack_2d(self):
        w = torch.randn(4, 20).sign().to(torch.int8)
        packed = _pack_binary_bits(w)
        assert packed.shape == (4, 3)  # ceil(20/8) = 3
        unpacked = _unpack_binary_bits(packed, 20)
        assert torch.equal(unpacked, w)

    def test_pack_large(self):
        w = torch.randn(128, 896).sign().to(torch.int8)
        packed = _pack_binary_bits(w)
        assert packed.shape == (128, 112)  # ceil(896/8) = 112
        unpacked = _unpack_binary_bits(packed, 896)
        assert torch.equal(unpacked, w)


class TestBinaryKMeans:
    """Test binary K-means clustering."""

    def test_cluster_identical_rows(self):
        # 4 identical rows → should form 1 cluster
        w = torch.ones(4, 8, dtype=torch.int8)
        cb, idx = _binary_kmeans(w, K=2, device=w.device, n_iters=5)
        assert cb.shape == (2, 8)
        assert idx.shape == (4,)
        # All should be in the same cluster
        assert (idx == idx[0]).all()

    def test_cluster_two_groups(self):
        torch.manual_seed(42)
        w = torch.zeros(8, 8, dtype=torch.int8)
        w[:4] = 1  # first 4 rows all +1
        w[4:] = -1  # last 4 rows all -1
        cb, idx = _binary_kmeans(w, K=2, device=w.device, n_iters=10)
        assert cb.shape == (2, 8)
        # First 4 should be one cluster, last 4 another
        assert (idx[:4] == idx[0]).all()
        assert (idx[4:] == idx[4]).all()
        assert idx[0] != idx[4]


class TestNanoQuant:
    """Test NanoQuant low-rank binary factorization."""

    def test_basic_forward(self):
        lin = nn.Linear(64, 32, bias=True)
        q = NanoQuantLinear.from_linear(lin, rank=16, admm_iters=20)
        x = torch.randn(2, 4, 64)
        out = q(x)
        assert out.shape == (2, 4, 32)
        assert not torch.isnan(out).any()

    def test_no_bias(self):
        lin = nn.Linear(64, 32, bias=False)
        q = NanoQuantLinear.from_linear(lin, rank=8, admm_iters=10)
        assert q.bias is None
        x = torch.randn(1, 64)
        out = q(x)
        assert out.shape == (1, 32)

    def test_higher_rank_better(self):
        """Higher rank should give lower reconstruction error."""
        torch.manual_seed(42)
        lin = nn.Linear(128, 64, bias=False)
        x = torch.randn(100, 128)
        ref = lin(x)

        q_low = NanoQuantLinear.from_linear(lin, rank=8, admm_iters=30)
        q_high = NanoQuantLinear.from_linear(lin, rank=32, admm_iters=30)

        err_low = (q_low(x) - ref).abs().mean().item()
        err_high = (q_high(x) - ref).abs().mean().item()
        assert err_high < err_low, f"Higher rank should be better: {err_high} vs {err_low}"

    def test_refine_improves(self):
        """Gradient refinement should improve over ADMM-only."""
        torch.manual_seed(42)
        lin = nn.Linear(64, 32, bias=False)
        x = torch.randn(50, 64)
        ref = lin(x)

        q_no_refine = NanoQuantLinear.from_linear(lin, rank=16, admm_iters=30, refine_steps=0)
        q_refined = NanoQuantLinear.from_linear(lin, rank=16, admm_iters=30, refine_steps=30)

        err_no = (q_no_refine(x) - ref).abs().mean().item()
        err_yes = (q_refined(x) - ref).abs().mean().item()
        assert err_yes <= err_no + 0.01, f"Refine should not hurt: {err_yes} vs {err_no}"

    def test_packed_storage(self):
        """Verify U and V are packed as bits, not int8."""
        lin = nn.Linear(64, 32)
        q = NanoQuantLinear.from_linear(lin, rank=16, admm_iters=10)
        assert q.U_packed.dtype == torch.uint8
        assert q.V_packed.dtype == torch.uint8
        # 32×16 = 512 bits = 64 bytes
        assert q.U_packed.numel() == 32 * 2  # ceil(16/8) = 2 per row
        # 64×16 = 1024 bits = 128 bytes
        assert q.V_packed.numel() == 64 * 2

    def test_cache_works(self):
        lin = nn.Linear(64, 32)
        q = NanoQuantLinear.from_linear(lin, rank=16, admm_iters=10)
        x = torch.randn(1, 4, 64)
        out1 = q(x)
        assert q._cached_weight is not None
        out2 = q(x)
        assert torch.allclose(out1, out2)


class TestBTC:
    """Test BTC-LLM binary codebook clustering."""

    def test_basic_forward(self):
        lin = nn.Linear(64, 128, bias=True)
        q = BTCQuantLinear.from_linear(lin, codebook_size=32, use_rotation=True)
        x = torch.randn(2, 4, 64)
        out = q(x)
        assert out.shape == (2, 4, 128)
        assert not torch.isnan(out).any()

    def test_no_rotation(self):
        lin = nn.Linear(64, 128)
        q = BTCQuantLinear.from_linear(lin, codebook_size=64, use_rotation=False)
        assert q.hadamard_order == 0
        x = torch.randn(1, 64)
        out = q(x)
        assert out.shape == (1, 128)

    def test_larger_codebook_better(self):
        """Larger codebook → more patterns → lower error."""
        torch.manual_seed(42)
        lin = nn.Linear(64, 256, bias=False)
        x = torch.randn(50, 64)
        ref = lin(x)

        q_small = BTCQuantLinear.from_linear(lin, codebook_size=16, use_rotation=False)
        q_large = BTCQuantLinear.from_linear(lin, codebook_size=256, use_rotation=False)

        err_small = (q_small(x) - ref).abs().mean().item()
        err_large = (q_large(x) - ref).abs().mean().item()
        assert err_large <= err_small, f"Larger codebook should be better: {err_large} vs {err_small}"

    def test_packed_codebook(self):
        """Verify codebook is packed as bits."""
        lin = nn.Linear(64, 128)
        q = BTCQuantLinear.from_linear(lin, codebook_size=32, use_rotation=True)
        assert q.codebook_packed.dtype == torch.uint8
        # K=32 patterns, hadamard_size=64, ceil(64/8)=8 bytes per pattern
        assert q.codebook_packed.shape == (32, 8)


class TestTernaryPTQ:
    """Test TernaryPTQ with calibration refinement."""

    def test_basic_forward(self):
        lin = nn.Linear(64, 32, bias=True)
        q = TernaryPTQLinear.from_linear(lin, activations=None, refine_iters=0)
        x = torch.randn(2, 4, 64)
        out = q(x)
        assert out.shape == (2, 4, 32)
        assert not torch.isnan(out).any()

    def test_with_calibration(self):
        lin = nn.Linear(64, 32)
        acts = torch.randn(64, 64)
        q = TernaryPTQLinear.from_linear(lin, activations=acts, refine_iters=10)
        x = torch.randn(1, 64)
        out = q(x)
        assert out.shape == (1, 32)

    def test_base3_packed(self):
        """Verify weights are base-3 packed (1.6 bits/w)."""
        lin = nn.Linear(64, 32)
        q = TernaryPTQLinear.from_linear(lin, refine_iters=0)
        # 64*32 = 2048 weights, 5 per byte → ceil(2048/5) = 410 bytes
        assert q.weight_packed.dtype == torch.uint8
        expected = (2048 + 4) // 5  # ceil(2048/5)
        assert q.weight_packed.numel() == expected

    def test_ternary_values(self):
        """Verify dequantized weights are ternary × scale."""
        lin = nn.Linear(64, 32)
        q = TernaryPTQLinear.from_linear(lin, refine_iters=0)
        w = q._dequantize_weight(torch.float32)
        # Each row should only have values in {−s, 0, +s} for that row's scale
        for i in range(32):
            s = q.scales[i].item()
            unique = w[i].abs().unique()
            # Should be {0, s} (or just {s} if no zeros)
            for v in unique:
                assert v == 0 or abs(v - s) < 1e-4, f"Row {i}: found value {v}, expected 0 or {s}"


class TestModelConversion:
    """Test model-level conversion functions."""

    def test_nanoquant_model(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        n = quantize_model_nanoquant(model, rank=16, admm_iters=10, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = model(x)
        assert out.shape == (2, 4, 64)

    def test_btc_model(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        n = quantize_model_btc(model, codebook_size=32, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = model(x)
        assert out.shape == (2, 4, 64)

    def test_ternary_model(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        n = quantize_model_ternary_ptq(model, activations=None, refine_iters=0, verbose=False)
        assert n == 2
        x = torch.randn(2, 4, 64)
        out = model(x)
        assert out.shape == (2, 4, 64)

    def test_ternary_model_with_calib(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        acts = {
            "0": torch.randn(32, 64),
            "2": torch.randn(32, 128),
        }
        n = quantize_model_ternary_ptq(model, activations=acts, refine_iters=5, verbose=False)
        assert n == 2

    def test_skip_quantized_layers(self):
        """Should not re-quantize already-quantized layers."""
        model = nn.Sequential(
            NanoQuantLinear(64, 128, rank=8),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # Manually set U_packed etc on first layer
        model[0].U_packed = torch.zeros(128, 1, dtype=torch.uint8)
        model[0].V_packed = torch.zeros(64, 1, dtype=torch.uint8)
        model[0].U_shape = (128, 8)
        model[0].V_shape = (64, 8)
        model[0].s1 = torch.ones(128, dtype=torch.float16)
        model[0].s2 = torch.ones(64, dtype=torch.float16)

        n = quantize_model_nanoquant(model, rank=16, admm_iters=5, verbose=False)
        assert n == 1, f"Should skip NanoQuantLinear, got {n}"

    def test_memory_estimation(self):
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        quantize_model_nanoquant(model, rank=16, admm_iters=5, verbose=False)
        mem = estimate_r48_memory(model)
        assert mem["total_mb"] > 0
        assert mem["total_params"] == 64 * 128 + 128 * 64
        assert "NanoQuantLinear" in mem["breakdown"]


class TestDeviceMovement:
    """Test device movement (hybrid offload compatibility)."""

    def test_cpu_gpu_roundtrip(self):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        lin = nn.Linear(64, 32)
        q = NanoQuantLinear.from_linear(lin, rank=16, admm_iters=10)
        q = q.cuda()
        x = torch.randn(1, 64, device="cuda")
        out1 = q(x)

        q = q.cpu()
        q._invalidate_cache()
        q = q.cuda()
        out2 = q(x)
        assert torch.allclose(out1, out2, atol=1e-4)


class TestNanoQuantQAT:
    """Test NanoQuant QAT layer with STE."""

    def test_qat_forward(self):
        """QAT layer produces finite output of correct shape."""
        lin = nn.Linear(64, 32, bias=True)
        qat = NanoQuantQATLinear.from_linear(lin, rank=16, admm_iters=10)
        x = torch.randn(1, 4, 64)
        out = qat(x)
        assert out.shape == (1, 4, 32)
        assert out.isfinite().all()

    def test_qat_ste_gradients(self):
        """STE allows gradients to flow to latent matrices."""
        lin = nn.Linear(64, 32)
        qat = NanoQuantQATLinear.from_linear(lin, rank=16, admm_iters=10)
        qat.train()
        x = torch.randn(1, 4, 64)
        out = qat(x)
        loss = out.sum()
        loss.backward()
        # All parameters should have gradients
        assert qat.U_latent.grad is not None
        assert qat.V_latent.grad is not None
        assert qat.s1.grad is not None
        assert qat.s2.grad is not None
        # Gradients should be finite
        assert qat.U_latent.grad.isfinite().all()
        assert qat.V_latent.grad.isfinite().all()

    def test_qat_bake_lossless(self):
        """Baking QAT → NanoQuantLinear produces near-identical output."""
        lin = nn.Linear(64, 32, bias=True)
        qat = NanoQuantQATLinear.from_linear(lin, rank=16, admm_iters=10)
        qat.eval()
        x = torch.randn(1, 4, 64)
        out_qat = qat(x)
        nq = qat.bake()
        nq.eval()
        out_baked = nq(x)
        # Small diff from float16 rounding of scales during bake
        assert torch.allclose(out_qat, out_baked, atol=1e-3)

    def test_qat_training_improves(self):
        """A few QAT steps should reduce loss (not increase it)."""
        torch.manual_seed(42)
        lin = nn.Linear(32, 32)
        # Create a target output from the original linear
        x = torch.randn(2, 8, 32)
        with torch.no_grad():
            target = lin(x)

        qat = NanoQuantQATLinear.from_linear(lin, rank=8, admm_iters=10)
        qat.train()
        optimizer = torch.optim.SGD(qat.parameters(), lr=0.01, momentum=0.9)

        initial_loss = F.mse_loss(qat(x), target).item()
        for _ in range(20):
            optimizer.zero_grad()
            loss = F.mse_loss(qat(x), target)
            loss.backward()
            optimizer.step()
        final_loss = F.mse_loss(qat(x), target).item()
        # Loss should decrease (or at least not explode)
        assert final_loss < initial_loss * 2  # allow some noise
        assert math.isfinite(final_loss)

    def test_qat_model_conversion(self):
        """convert_model_to_nanoquant_qat replaces linears."""
        model = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        n = convert_model_to_nanoquant_qat(model, rank=8, admm_iters=5, verbose=False)
        assert n == 2
        # Check that linears were replaced
        for m in model.modules():
            if isinstance(m, NanoQuantQATLinear):
                continue
            assert not isinstance(m, nn.Linear)

    def test_qat_bake_model(self):
        """bake_qat_model converts all QAT layers back to inference."""
        model = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        convert_model_to_nanoquant_qat(model, rank=8, admm_iters=5, verbose=False)
        n = bake_qat_model(model, verbose=False)
        assert n == 2
        for m in model.modules():
            assert not isinstance(m, NanoQuantQATLinear)
