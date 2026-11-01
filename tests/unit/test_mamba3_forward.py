"""Tests for Mamba-3 forward pass module (complex states, MIMO, exp-trapezoidal)."""

from __future__ import annotations

import pytest
import torch

from forge.engine.mamba3 import (
    Mamba3Block,
    Mamba3Cache,
    exp_trapezoidal_discretize,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_block():
    """A small Mamba-3 block for fast CPU tests."""
    torch.manual_seed(42)
    return Mamba3Block(
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        dt_rank=4,
        n_inputs=1,
        n_outputs=1,
    )


@pytest.fixture
def mimo_block():
    """A MIMO Mamba-3 block (2 inputs -> 3 outputs)."""
    torch.manual_seed(42)
    return Mamba3Block(
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        dt_rank=4,
        n_inputs=2,
        n_outputs=3,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMamba3ForwardShape:
    """Verify output shape matches input."""

    def test_forward_shape_seq(self, small_block):
        x = torch.randn(2, 16, 32)
        out, _present = small_block(x, use_cache=False)
        assert out.shape == (2, 16, 32), f"Expected (2,16,32), got {out.shape}"

    def test_forward_shape_single_step(self, small_block):
        x = torch.randn(1, 1, 32)
        out, _present = small_block(x, use_cache=True)
        assert out.shape == (1, 1, 32)

    def test_forward_shape_batch(self, small_block):
        x = torch.randn(4, 8, 32)
        out, _ = small_block(x)
        assert out.shape == (4, 8, 32)


class TestMamba3ComplexState:
    """Verify state is complex-valued."""

    def test_A_log_is_complex_storage(self, small_block):
        # A_log stored as (n_ssm_units, d_state, 2) [real, imag]
        assert small_block.A_log.shape[-1] == 2, "A_log must have trailing 2 for complex"
        assert small_block.A_log.shape == (small_block.d_inner, small_block.d_state, 2)

    def test_A_complex_recovery(self, small_block):
        A = small_block._get_A_complex()
        assert A.is_complex(), "Recovered A must be complex"
        assert A.shape == (small_block.n_ssm_units, small_block.d_state)

    def test_imag_zero_at_init(self, small_block):
        # At init, imaginary part of A_log is zero (lossless warm start)
        assert torch.all(small_block.A_log[..., 1] == 0), "Imaginary part must be zero at init"

    def test_norms_are_complex(self, small_block):
        for norm_name in ["dt_norm", "A_norm", "B_norm", "C_norm"]:
            norm = getattr(small_block, norm_name)
            assert norm.shape[-1] == 2, f"{norm_name} must have trailing 2 for complex"

    def test_scan_produces_complex_state(self, small_block):
        x = torch.randn(1, 8, 32)
        _out, present = small_block(x, use_cache=True)
        # present should contain real and imag parts of the complex state
        assert "ssm_state_real" in present
        assert "ssm_state_imag" in present
        assert present["ssm_state_real"].shape == (1, small_block.n_ssm_units, small_block.d_state)
        assert present["ssm_state_imag"].shape == (1, small_block.n_ssm_units, small_block.d_state)


class TestMamba3RecurrentMode:
    """Verify inference cache works for incremental decoding."""

    def test_cache_class(self):
        cache = Mamba3Cache.create(batch_size=2, n_ssm_units=4, d_state=8, d_inner=8, d_conv=4)
        assert cache.ssm_state_real.shape == (2, 4, 8)
        assert cache.ssm_state_imag.shape == (2, 4, 8)
        assert cache.conv_state.shape == (2, 8, 3)
        assert not cache.is_empty

    def test_cache_reset(self):
        cache = Mamba3Cache.create(1, 4, 8, 8, 4)
        cache.reset()
        assert cache.is_empty

    def test_cache_to_complex(self):
        cache = Mamba3Cache.create(1, 4, 8, 8, 4)
        h = cache.to_complex_state()
        assert h.is_complex()
        assert h.shape == (1, 4, 8)

    def test_incremental_matches_full(self, small_block):
        """Incremental decoding (T=1 step) should match full-sequence forward.

        Process first N-1 tokens in full mode (with cache), then feed the Nth
        token incrementally using the cache. Compare with a full N-token forward.
        """
        small_block.eval()
        torch.manual_seed(0)
        x = torch.randn(1, 6, 32)

        # Full sequence forward (all 6 tokens)
        with torch.no_grad():
            out_full, _ = small_block(x, use_cache=False)

        # Process first 5 tokens in full mode (with cache)
        with torch.no_grad():
            _out_5, present = small_block(x[:, :5, :], use_cache=True)

        # Feed the 6th token (index 5) incrementally using the cache
        with torch.no_grad():
            out_inc, _ = small_block(x[:, 5:6, :], past_key_value=present, use_cache=True)

        # The incremental output for token 5 should match the full-sequence output
        assert out_inc.shape == (1, 1, 32)
        diff = (out_inc - out_full[:, 5:6, :]).abs().max()
        assert diff < 1e-4, f"Incremental vs full mismatch at token 5: max diff {diff:.6e}"

    def test_cache_from_dict(self, small_block):
        x = torch.randn(1, 4, 32)
        _out, present = small_block(x, use_cache=True)
        cache = Mamba3Cache.from_dict(present)
        assert cache is not None
        assert cache.ssm_state_real is not None
        assert cache.conv_state is not None


class TestMamba3VsMamba2Init:
    """Verify that with identity-like init, Mamba-3 output is similar to Mamba-2."""

    def test_imag_zero_matches_mamba2(self):
        """With imaginary parts zeroed, Mamba-3 should behave like Mamba-2
        (same recurrence, just with complex arithmetic that has zero imaginary)."""
        from forge.keys.architecture.mamba_probe import MambaLayer

        torch.manual_seed(123)
        d_model, d_state, d_conv, expand = 32, 8, 4, 2

        # Mamba-2 block
        m2 = MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=4, use_jamba_norms=True)
        m2.eval()

        # Mamba-3 block with same dimensions
        torch.manual_seed(123)
        m3 = Mamba3Block(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=4, n_inputs=1, n_outputs=1
        )
        m3.eval()

        # Copy shared weights from Mamba-2 to Mamba-3 (real parts)
        # in_proj, conv1d, dt_proj, out_proj, D are identical
        with torch.no_grad():
            m3.in_proj.weight.copy_(m2.in_proj.weight)
            if m2.in_proj.bias is not None:
                m3.in_proj.bias.copy_(m2.in_proj.bias)
            m3.conv1d.weight.copy_(m2.conv1d.weight)
            m3.conv1d.bias.copy_(m2.conv1d.bias)
            # x_proj: Mamba-3 has complex B/C (2x the B/C rows vs Mamba-2).
            # Copy dt rows (identical) and real-part rows of B/C from Mamba-2.
            dt_rank = m2.dt_rank if hasattr(m2, "dt_rank") else 4
            m2_xp = m2.x_proj.weight  # (dt_rank + 2*d_state, d_inner)
            m3_xp = m3.x_proj.weight  # (dt_rank + 4*d_state, d_inner)
            # dt rows are the same
            m3_xp[:dt_rank].copy_(m2_xp[:dt_rank])
            # Mamba-2 B_real rows -> Mamba-3 B_real rows
            d_state = m2.A_log.shape[1]
            m2_b_start = dt_rank
            m3_b_real_start = dt_rank
            dt_rank + d_state
            m3_xp[m3_b_real_start : m3_b_real_start + d_state].copy_(m2_xp[m2_b_start : m2_b_start + d_state])
            # Mamba-2 C_real rows -> Mamba-3 C_real rows
            m2_c_start = dt_rank + d_state
            m3_c_real_start = dt_rank + 2 * d_state
            dt_rank + 3 * d_state
            m3_xp[m3_c_real_start : m3_c_real_start + d_state].copy_(m2_xp[m2_c_start : m2_c_start + d_state])
            m3.dt_proj.weight.copy_(m2.dt_proj.weight)
            m3.dt_proj.bias.copy_(m2.dt_proj.bias)
            m3.out_proj.weight.copy_(m2.out_proj.weight)
            if m2.out_proj.bias is not None:
                m3.out_proj.bias.copy_(m2.out_proj.bias)
            m3.D.copy_(m2.D)
            # A_log: Mamba-2 is (d_inner, d_state), Mamba-3 is (d_inner, d_state, 2)
            m3.A_log[..., 0].copy_(m2.A_log)
            m3.A_log[..., 1].zero_()  # imag = 0
            # Norms: Mamba-2 uses real weights, Mamba-3 uses [..., 2]
            # dt_norm: Mamba-2's dt_layernorm is (dt_rank,) applied BEFORE dt_proj;
            # Mamba-3's dt_norm is (d_inner, 2) applied AFTER dt_proj. Different
            # positions/dims — leave Mamba-3's dt_norm at identity (ones, imag=0).
            # B/C norms: both are (d_state,) -> (d_state, 2), same position.
            m3.B_norm[..., 0].copy_(m2.b_layernorm)
            m3.B_norm[..., 1].zero_()
            m3.C_norm[..., 0].copy_(m2.c_layernorm)
            m3.C_norm[..., 1].zero_()
            # A_norm: Mamba-2 doesn't have A_norm (Jamba only has dt/b/c).
            # Mamba-3 has A_norm init to ones (identity). Keep as-is.

        x = torch.randn(2, 8, d_model)
        with torch.no_grad():
            out2, _ = m2(x)
            out3, _ = m3(x)

        # With imag=0 and identity A_norm, the complex recurrence's real part
        # should match the real Mamba-2 recurrence. The exp-trapezoidal
        # discretization differs from ZOH, so we check approximate similarity
        # (same order of magnitude, not exact match).
        assert out2.shape == out3.shape
        # The outputs should be in the same ballpark (both are valid SSM outputs)
        assert not torch.isnan(out3).any(), "Mamba-3 output has NaN"
        assert not torch.isinf(out3).any(), "Mamba-3 output has Inf"
        # Correlation should be positive (same underlying computation, but
        # exp-trapezoidal discretization and A_norm differ from Mamba-2's ZOH)
        flat2 = out2.flatten()
        flat3 = out3.flatten()
        corr = torch.corrcoef(torch.stack([flat2, flat3]))[0, 1]
        assert corr.item() > 0.3, f"Mamba-3 vs Mamba-2 correlation too low: {corr.item():.4f}"


class TestMamba3GradientFlow:
    """Verify gradients flow through the block."""

    def test_gradients_flow(self, small_block):
        x = torch.randn(2, 8, 32, requires_grad=False)
        out, _ = small_block(x)
        loss = out.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in small_block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not param.grad.isnan().any(), f"NaN gradient for {name}"

    def test_gradient_flow_through_cache(self, small_block):
        x = torch.randn(1, 4, 32)
        out, _present = small_block(x, use_cache=True)
        loss = out.sum()
        loss.backward()
        for name, param in small_block.named_parameters():
            assert param.grad is not None, f"No gradient for {name} (cache mode)"


class TestMamba3MIMO:
    """Verify multi-input multi-output works."""

    def test_mimo_output_shape(self, mimo_block):
        x = torch.randn(2, 8, 32)
        out, _ = mimo_block(x)
        assert out.shape == (2, 8, 32), f"MIMO output shape: {out.shape}"

    def test_mimo_n_ssm_units(self, mimo_block):
        # d_inner = 2*32 = 64, n_inputs=2 -> n_ssm_units = 32
        assert mimo_block.n_ssm_units == 32
        # d_inner_out = 32 * 3 = 96
        assert mimo_block.d_inner_out == 96

    def test_mimo_x_proj_size(self, mimo_block):
        # x_proj: dt_rank + 2*d_state*n_inputs + 2*d_state*n_outputs
        # = 4 + 2*8*2 + 2*8*3 = 4 + 32 + 48 = 84
        expected = 4 + 2 * 8 * 2 + 2 * 8 * 3
        assert mimo_block.x_proj.weight.shape[0] == expected

    def test_mimo_gradients(self, mimo_block):
        x = torch.randn(1, 4, 32)
        out, _ = mimo_block(x)
        out.sum().backward()
        for name, param in mimo_block.named_parameters():
            assert param.grad is not None, f"No gradient for {name} (MIMO)"

    def test_mimo_A_log_shape(self, mimo_block):
        # A_log: (n_ssm_units, d_state, 2) = (32, 8, 2)
        assert mimo_block.A_log.shape == (32, 8, 2)


class TestMamba3ExponentialTrapezoidal:
    """Verify the exponential-trapezoidal discretization formula."""

    def test_A_bar_formula(self):
        """A_bar = exp(delta * A)."""
        # A: (d_state,) complex, delta: (1,) real, B: (d_state, 1) complex
        A = torch.complex(torch.tensor([-1.0, -2.0]), torch.tensor([0.0, 0.0]))
        delta = torch.tensor([0.5]).unsqueeze(-1)  # (1, 1)
        B = torch.complex(torch.ones(2, 1), torch.zeros(2, 1))
        A_bar, _B_bar = exp_trapezoidal_discretize(A.unsqueeze(0), B.unsqueeze(0), delta)
        # A_bar = exp(0.5 * [-1, -2]) = [exp(-0.5), exp(-1.0)]
        expected = torch.exp(torch.tensor([-0.5, -1.0]))
        assert A_bar.shape == (1, 2)
        assert torch.allclose(A_bar[0].real, expected, atol=1e-6)
        assert torch.allclose(A_bar[0].imag, torch.zeros(2), atol=1e-6)

    def test_B_bar_zoh_limit(self):
        """As delta*A -> 0, B_bar -> delta * B (ZOH limit)."""
        A = torch.complex(torch.tensor([-1.0]), torch.tensor([0.0]))  # (1,)
        delta = torch.tensor([1e-8]).unsqueeze(-1)  # (1, 1)
        B = torch.complex(torch.tensor([[2.0]]), torch.tensor([[0.0]]))  # (1, 1)
        _A_bar, B_bar = exp_trapezoidal_discretize(A.unsqueeze(0), B.unsqueeze(0), delta)
        # B_bar ≈ delta * B = 1e-8 * 2 = 2e-8
        expected = torch.tensor([2e-8])
        assert torch.allclose(B_bar[0, :, 0].real, expected, atol=1e-10)

    def test_B_bar_trapezoidal_correction(self):
        """B_bar = (exp(da)-1)/da * delta * B (trapezoidal, not ZOH)."""
        A = torch.complex(torch.tensor([-1.0]), torch.tensor([0.0]))  # (1,)
        delta = torch.tensor([1.0]).unsqueeze(-1)  # (1, 1)
        B = torch.complex(torch.tensor([[1.0]]), torch.tensor([[0.0]]))  # (1, 1)
        _A_bar, B_bar = exp_trapezoidal_discretize(A.unsqueeze(0), B.unsqueeze(0), delta)
        da = -1.0  # delta * A = 1.0 * -1.0
        correction = (torch.exp(torch.tensor(da)) - 1) / da  # (exp(-1)-1)/(-1)
        expected = correction * 1.0 * 1.0  # correction * delta * B
        assert torch.allclose(B_bar[0, 0, 0].real, expected.reshape(1), atol=1e-6)

    def test_B_bar_differs_from_zoh(self):
        """For finite delta*A, trapezoidal B_bar != ZOH B_bar (= delta*B)."""
        A = torch.complex(torch.tensor([-2.0]), torch.tensor([0.0]))  # (1,)
        delta = torch.tensor([1.0]).unsqueeze(-1)  # (1, 1)
        B = torch.complex(torch.tensor([[1.0]]), torch.tensor([[0.0]]))  # (1, 1)
        _A_bar, B_bar = exp_trapezoidal_discretize(A.unsqueeze(0), B.unsqueeze(0), delta)
        zoh_B_bar = torch.tensor([1.0])  # ZOH: delta * B = 1.0 * 1.0
        assert not torch.allclose(B_bar[0, 0, 0].real, zoh_B_bar, atol=1e-3), (
            "Trapezoidal B_bar should differ from ZOH for finite delta*A"
        )

    def test_complex_A_bar(self):
        """A_bar with complex A produces complex result."""
        A = torch.complex(torch.tensor([-1.0]), torch.tensor([0.5]))  # (1,)
        delta = torch.tensor([0.5]).unsqueeze(-1)  # (1, 1)
        B = torch.complex(torch.tensor([[1.0]]), torch.tensor([[0.0]]))  # (1, 1)
        A_bar, _B_bar = exp_trapezoidal_discretize(A.unsqueeze(0), B.unsqueeze(0), delta)
        assert A_bar.is_complex()
        # exp(0.5 * (-1 + 0.5j)) should have nonzero imaginary part
        assert A_bar[0].imag.abs().item() > 1e-6

    def test_discretize_stability(self):
        """Discretization should not produce NaN/Inf for typical values."""
        A = torch.complex(
            torch.randn(10, 16) * 0.1 - 1.0,  # negative real part
            torch.randn(10, 16) * 0.1,
        )
        delta = torch.rand(10, 1) * 2.0  # [0, 2]
        B = torch.complex(torch.randn(10, 16, 4), torch.randn(10, 16, 4) * 0.1)
        A_bar, B_bar = exp_trapezoidal_discretize(A, B, delta)
        assert not A_bar.isnan().any()
        assert not A_bar.isinf().any()
        assert not B_bar.isnan().any()
        assert not B_bar.isinf().any()


class TestMamba3CacheIntegration:
    """Integration tests for Mamba3Cache with the block."""

    def test_cache_create_and_use(self, small_block):
        x = torch.randn(1, 4, 32)
        _out, present = small_block(x, use_cache=True)

        # Build cache from present
        cache = Mamba3Cache.from_dict(present)
        assert cache is not None

        # Use cache for next step
        x_next = torch.randn(1, 1, 32)
        out_next, present_next = small_block(x_next, past_key_value=cache.to_dict(), use_cache=True)
        assert out_next.shape == (1, 1, 32)
        assert present_next["ssm_state_real"] is not None

    def test_recurrent_state_persistence(self, small_block):
        """State should persist across multiple incremental steps."""
        small_block.eval()
        x = torch.randn(1, 1, 32)

        # Step 1
        with torch.no_grad():
            _out1, present1 = small_block(x, use_cache=True)
        # Step 2 (using state from step 1)
        with torch.no_grad():
            _out2, present2 = small_block(x, past_key_value=present1, use_cache=True)

        # State should have evolved (not identical)
        assert not torch.equal(present1["ssm_state_real"], present2["ssm_state_real"]), (
            "SSM state should evolve between steps"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q", "--tb=short"])
