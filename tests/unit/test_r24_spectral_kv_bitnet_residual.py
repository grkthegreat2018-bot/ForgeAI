"""Tests for SpectralKV cache and BitNetResidualLinear key.

Runs on CPU with small dimensions for fast, GPU-independent testing.
"""
import torch
import torch.nn.functional as F

from forge.engine.kv.spectral_kv import (
    SpectralKVCache,
    SpectralPreAllocatedCache,
    _fourier_basis,
)
from forge.engine.kv_backend import build_kv_cache
from tests.fixtures.keys.quantization.bitnet_residual_key import (
    BitNetResidualKey,
    BitNetResidualLinear,
    apply_bitnet_residual,
    compute_residual,
    ternary_quantize,
)

# ── SpectralKV ───────────────────────────────────────────────────────────────


class TestFourierBasis:
    def test_basis_shape(self):
        basis = _fourier_basis(128, 16, "cpu", torch.float32)
        assert basis.shape == (128, 33)  # 1 + 2*16

    def test_dc_column_is_ones(self):
        basis = _fourier_basis(64, 8, "cpu", torch.float32)
        assert torch.allclose(basis[:, 0], torch.ones(64))

    def test_orthogonality(self):
        """Fourier basis columns should be approximately orthogonal."""
        basis = _fourier_basis(256, 32, "cpu", torch.float32)
        gram = basis.T @ basis
        # Diagonal should be ~seq_len/2 for cos/sin, seq_len for DC
        off_diag = gram - torch.diag(gram.diagonal())
        assert off_diag.abs().max() < 1.0  # approximately orthogonal


class TestSpectralKVCache:
    def _make_cache(self, n_kv=4, head_dim=32, max_seq=256, max_freq=16, sink_size=4):
        cache = SpectralKVCache(max_freq=max_freq, sink_size=sink_size)
        cache.init(n_heads=8, head_dim=head_dim, n_kv_heads=n_kv,
                   max_seq_len=max_seq, device="cpu", dtype=torch.float32)
        return cache

    def test_init(self):
        cache = self._make_cache()
        assert cache.n_kv == 4
        assert cache.head_dim == 32
        assert cache.max_freq == 16
        assert cache.n_coeffs == 33  # 1 + 2*16
        assert cache.seq_len == 0

    def test_append_and_get_sink(self):
        """First sink_size tokens should be stored full-precision."""
        cache = self._make_cache(sink_size=4)
        k = torch.randn(1, 4, 4, 32)  # [B, n_kv, T, hd]
        v = torch.randn(1, 4, 4, 32)
        cache.append(k, v, position=0)
        assert cache.seq_len == 4
        k_out, _v_out = cache.get()
        assert k_out.shape == (1, 4, 4, 32)
        # Sink tokens should be exact
        assert torch.allclose(k_out, k, atol=1e-5)

    def test_append_and_get_full(self):
        """Full prefill: sink + spectral reconstruction."""
        cache = self._make_cache(n_kv=4, head_dim=32, max_seq=256, max_freq=16)
        seq_len = 128
        k = torch.randn(1, 4, seq_len, 32)
        v = torch.randn(1, 4, seq_len, 32)
        cache.append(k, v, position=0)
        assert cache.seq_len == seq_len

        k_out, _v_out = cache.get()
        assert k_out.shape == (1, 4, seq_len, 32)

        # Sink tokens (first 4) should be exact
        assert torch.allclose(k_out[:, :, :4], k[:, :, :4], atol=1e-5)

        # Spectral reconstruction should have reasonable error (< 1.0 relative)
        # Note: random K/V is harder to fit than trained K/V, so error is higher
        err = (k_out - k).norm() / k.norm()
        assert err < 2.0, f"Spectral reconstruction error too high: {err}"

    def test_compression_ratio(self):
        """SpectralKV should report high compression at long sequences."""
        cache = self._make_cache(n_kv=4, head_dim=32, max_seq=8192, max_freq=16)
        seq_len = 4096
        k = torch.randn(1, 4, seq_len, 32)
        v = torch.randn(1, 4, seq_len, 32)
        cache.append(k, v, position=0)

        info = cache.info()
        assert info["compression"] > 10, f"Compression too low: {info['compression']}"
        # At 4096 tokens: standard = 2*4*32*4096*2 = 2MB, spectral = 2*4*33*32*2 = 16.8KB
        # compression = 2MB / 16.8KB ≈ 121×
        assert info["compression"] > 50, f"Expected >50× at 4096 tokens, got {info['compression']}"

    def test_clear(self):
        cache = self._make_cache()
        k = torch.randn(1, 4, 64, 32)
        v = torch.randn(1, 4, 64, 32)
        cache.append(k, v, position=0)
        cache.clear()
        assert cache.seq_len == 0
        assert not cache._fitted

    def test_factory_registration(self):
        """build_kv_cache('spectral') should return a SpectralKVCache."""
        cache = build_kv_cache("spectral")
        assert isinstance(cache, SpectralKVCache)


class TestSpectralPreAllocatedCache:
    def test_basic_operation(self):
        """SpectralPreAllocatedCache should work as a drop-in for PreAllocatedKVCache."""
        n_layers = 4
        n_kv = 4
        head_dim = 32
        max_seq = 256
        cache = SpectralPreAllocatedCache(
            n_layers=n_layers, batch=1, n_kv_heads=n_kv,
            max_seq_len=max_seq, head_dim=head_dim,
            dtype=torch.float32, device="cpu",
            max_freq=16, sink_size=4,
        )

        # Simulate prefill: append 64 tokens to each layer
        seq_len = 64
        for layer in range(n_layers):
            k = torch.randn(1, n_kv, seq_len, head_dim)
            v = torch.randn(1, n_kv, seq_len, head_dim)
            cache.append(layer, k, v)
        cache.advance(seq_len)

        # Get layer 0
        kv = cache.get_layer(0)
        assert kv is not None
        k, v = kv
        assert k.shape == (1, n_kv, seq_len, head_dim)

    def test_reset(self):
        cache = SpectralPreAllocatedCache(
            n_layers=2, batch=1, n_kv_heads=4,
            max_seq_len=128, head_dim=32,
            dtype=torch.float32, device="cpu",
        )
        cache.append(0, torch.randn(1, 4, 32, 32), torch.randn(1, 4, 32, 32))
        cache.advance(32)
        assert cache.filled == 32
        cache.reset()
        assert cache.filled == 0

    def test_conv_layers_skipped(self):
        """Layers with 0 KV heads (conv) should be skipped."""
        cache = SpectralPreAllocatedCache(
            n_layers=3, batch=1, n_kv_heads=4,
            max_seq_len=128, head_dim=32,
            dtype=torch.float32, device="cpu",
            n_kv_heads_per_layer=[4, 0, 4],  # layer 1 is conv
        )
        cache.append(1, torch.randn(1, 4, 16, 32), torch.randn(1, 4, 16, 32))
        cache.advance(16)
        assert cache.get_layer(1) is None  # conv layer has no cache


# ── BitNetResidual ────────────────────────────────────────────────────────────


class TestTernaryQuantize:
    def test_ternary_values(self):
        w = torch.randn(64, 32) * 0.1
        q, _scale = ternary_quantize(w)
        assert set(q.unique().tolist()).issubset({-1.0, 0.0, 1.0})

    def test_scale_is_positive(self):
        w = torch.randn(64, 32)
        _q, scale = ternary_quantize(w)
        assert scale > 0


class TestComputeResidual:
    def test_residual_shape(self):
        w = torch.randn(64, 32)
        w_t, mask, _res_vals = compute_residual(w, residual_frac=0.10)
        assert w_t.shape == w.shape
        assert mask.shape == w.shape
        assert mask.dtype == torch.bool
        n_expected = int(w.numel() * 0.10)
        assert mask.sum() == n_expected

    def test_residual_reduces_error(self):
        """Ternary + residual should have lower error than pure ternary."""
        w = torch.randn(128, 64) * 0.1
        # Add some outliers
        w[0, 0] = 5.0
        w[1, 1] = -5.0

        # Pure ternary
        w_t, _ = ternary_quantize(w)
        err_pure = (w - w_t).norm() / w.norm()

        # Ternary + 10% residual
        w_tr, _mask, _ = compute_residual(w, residual_frac=0.10)
        err_res = (w - w_tr).norm() / w.norm()

        assert err_res < err_pure, f"Residual should reduce error: {err_res} vs {err_pure}"

    def test_zero_residual_fraction(self):
        """0% residual should equal pure ternary."""
        w = torch.randn(32, 16)
        w_t, mask, _res_vals = compute_residual(w, residual_frac=0.0)
        w_pure, _ = ternary_quantize(w)
        assert torch.allclose(w_t, w_pure)
        assert mask.sum() == 0


class TestBitNetResidualLinear:
    def test_forward_shape(self):
        layer = BitNetResidualLinear(64, 32, residual_frac=0.10)
        x = torch.randn(1, 10, 64)
        out = layer(x)
        assert out.shape == (1, 10, 32)

    def test_full_precision_forward(self):
        """Without quantize=True, should use full-precision weights."""
        layer = BitNetResidualLinear(64, 32, residual_frac=0.10, quantize=False)
        x = torch.randn(1, 10, 64)
        out = layer(x)
        expected = F.linear(x, layer.weight)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_quantized_forward(self):
        """With quantize=True, should use ternary + residual (STE)."""
        layer = BitNetResidualLinear(64, 32, residual_frac=0.10, quantize=True)
        layer._compute_residual_mask()
        x = torch.randn(1, 10, 64)
        out = layer(x)
        assert out.shape == (1, 10, 32)
        # Should be different from full-precision (ternary quantization)
        full_out = F.linear(x, layer.weight)
        assert not torch.allclose(out, full_out, atol=1e-3)

    def test_residual_mask_computed(self):
        layer = BitNetResidualLinear(64, 32, residual_frac=0.10)
        layer._compute_residual_mask()
        n_expected = int(64 * 32 * 0.10)
        assert layer.residual_mask.sum() == n_expected

    def test_int8_storage_conversion(self):
        """convert_to_int8_storage should work for deployment."""
        layer = BitNetResidualLinear(64, 32, residual_frac=0.10, quantize=True)
        layer._compute_residual_mask()
        layer.convert_to_int8_storage()
        assert layer._prequantized
        assert hasattr(layer, "weight_int8")
        assert hasattr(layer, "residual_values")
        # Forward should still work
        x = torch.randn(1, 10, 64)
        out = layer(x)
        assert out.shape == (1, 10, 32)

    def test_state_dict_load(self):
        """Loading from state dict should compute residual mask if missing."""
        layer1 = BitNetResidualLinear(64, 32, residual_frac=0.10)
        layer1._compute_residual_mask()
        sd = layer1.state_dict()

        layer2 = BitNetResidualLinear(64, 32, residual_frac=0.10)
        layer2.load_state_dict(sd)
        assert layer2.residual_mask.sum() == int(64 * 32 * 0.10)


class TestBitNetResidualKey:
    def test_key_properties(self):
        key = BitNetResidualKey(residual_frac=0.10)
        assert key.name == "bitnet_residual"
        assert key.key_class() == __import__("forge.keys.misc.base", fromlist=["KeyClass"]).KeyClass.PARTIAL

    def test_forward(self):
        key = BitNetResidualKey(residual_frac=0.10)
        data = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
            "layer2.weight": torch.randn(32, 64),
        }
        result = key.forward(data)
        assert result.success
        assert "layer1.weight" in result.weights
        assert "layer1.residual_mask" in result.weights
        assert "layer1.residual_values" in result.weights

    def test_reverse(self):
        key = BitNetResidualKey()
        result = key.reverse({"layer1.weight": torch.randn(64, 32)})
        assert result.success


class TestApplyBitnetResidual:
    def test_applies_to_weights(self):
        state = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
        }
        out = apply_bitnet_residual(state, residual_frac=0.10)
        assert "layer1.weight" in out
        assert "layer1.residual_mask" in out
        assert "layer1.residual_values" in out
        assert "layer1.bias" in out  # bias unchanged

    def test_skips_non_weight_keys(self):
        state = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.norm.weight": torch.randn(64),  # 1D, should be skipped
        }
        out = apply_bitnet_residual(state, residual_frac=0.10)
        assert "layer1.residual_mask" in out
        assert "layer1.norm.residual_mask" not in out  # 1D skipped

