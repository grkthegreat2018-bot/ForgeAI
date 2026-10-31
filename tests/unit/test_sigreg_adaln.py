"""Tests for SIGReg (Spectral Implicit Geometry Regularization) and AdaLN-zero.

Covers:
  - SIGReg loss: positive when spectral norm < threshold, zero when above.
  - SIGReg gradient flow through the loss.
  - AdaLN-zero identity at initialization (scale=1, shift=0).
  - AdaLN-zero forward with conditioning changes the output.
  - AdaLN-zero backward compatibility: model works without cond when cond_dim=None.
"""
from __future__ import annotations

import torch

from forge.config import ModelConfig
from forge.model_loader import ConfigurableResearchLLM
from forge.training.losses.adaln_zero import AdaLNZero, AdaLNZeroModulation
from forge.training.losses.sigreg import SIGRegLoss

# ── SIGReg tests ──────────────────────────────────────────────────────────────

class TestSIGReg:
    """SIGReg spectral regularization loss tests."""

    def test_sigreg_basic(self):
        """SIGReg loss is positive when spectral norm < threshold, zero when above."""
        sigreg = SIGRegLoss(threshold=10.0)
        # Small hidden state → low spectral norm → penalty > 0.
        h_small = torch.randn(2, 4, 8) * 0.01
        loss_small = sigreg([h_small])
        assert loss_small.item() > 0.0, "Loss should be positive when sigma < threshold"

        # Large hidden state → high spectral norm → penalty = 0.
        h_large = torch.randn(2, 4, 8) * 100.0
        loss_large = sigreg([h_large])
        assert loss_large.item() == 0.0, "Loss should be zero when sigma >= threshold"

    def test_sigreg_basic_power_backend(self):
        """Same test with power-iteration backend."""
        sigreg = SIGRegLoss(threshold=10.0, backend="power", power_iters=5)
        h_small = torch.randn(2, 4, 8) * 0.01
        loss_small = sigreg([h_small])
        assert loss_small.item() > 0.0

        h_large = torch.randn(2, 4, 8) * 100.0
        loss_large = sigreg([h_large])
        assert loss_large.item() == 0.0

    def test_sigreg_gradient(self):
        """Gradients flow through the SIGReg loss."""
        sigreg = SIGRegLoss(threshold=5.0)
        h = torch.randn(2, 4, 8, requires_grad=True)
        loss = sigreg([h])
        loss.backward()
        assert h.grad is not None, "Gradient should flow through SIGReg loss"
        assert torch.isfinite(h.grad).all(), "Gradients should be finite"

    def test_sigreg_gradient_power_backend(self):
        """Gradients flow through the power-iteration backend."""
        sigreg = SIGRegLoss(threshold=5.0, backend="power", power_iters=3)
        h = torch.randn(2, 4, 8, requires_grad=True)
        loss = sigreg([h])
        loss.backward()
        assert h.grad is not None
        assert torch.isfinite(h.grad).all()

    def test_sigreg_multiple_layers(self):
        """SIGReg sums penalties across multiple layers."""
        sigreg = SIGRegLoss(threshold=10.0, reduction="sum")
        h1 = torch.randn(2, 4, 8) * 0.01  # small → penalized
        h2 = torch.randn(2, 4, 8) * 100.0  # large → not penalized
        loss = sigreg([h1, h2])
        # Only h1 contributes.
        sigreg_single = SIGRegLoss(threshold=10.0, reduction="sum")
        loss_single = sigreg_single([h1])
        assert torch.allclose(loss, loss_single, atol=1e-5)

    def test_sigreg_mean_reduction(self):
        """Mean reduction divides by number of layers."""
        sigreg_mean = SIGRegLoss(threshold=10.0, reduction="mean")
        sigreg_sum = SIGRegLoss(threshold=10.0, reduction="sum")
        h1 = torch.randn(2, 4, 8) * 0.01
        h2 = torch.randn(2, 4, 8) * 0.01
        loss_mean = sigreg_mean([h1, h2])
        loss_sum = sigreg_sum([h1, h2])
        assert torch.allclose(loss_mean, loss_sum / 2.0, atol=1e-5)


# ── AdaLN-zero tests ──────────────────────────────────────────────────────────

class TestAdaLNZero:
    """AdaLN-zero conditioning tests."""

    def test_adaln_zero_init(self):
        """At initialization, AdaLNZero produces identity (scale=1, shift=0)."""
        dim, cond_dim = 64, 32
        mod = AdaLNZeroModulation(dim, cond_dim)
        cond = torch.randn(2, cond_dim)
        scale, shift = mod(cond)
        # Zero-init linear → gamma=0, beta=0 → scale=0+1=1, shift=0.
        assert torch.allclose(scale, torch.ones_like(scale), atol=1e-6), \
            "Scale should be 1 at init (zero-init + 1)"
        assert torch.allclose(shift, torch.zeros_like(shift), atol=1e-6), \
            "Shift should be 0 at init"

    def test_adaln_zero_init_full_module(self):
        """AdaLNZero full module is identity w.r.t. normalization at init."""
        dim, cond_dim = 64, 32
        adaln = AdaLNZero(dim, cond_dim, norm_type="rmsnorm")
        x = torch.randn(2, 8, dim)
        cond = torch.randn(2, cond_dim)
        out_cond = adaln(x, cond)
        out_nocond = adaln(x, None)
        # At init, modulation is identity → output == plain normalization.
        assert torch.allclose(out_cond, out_nocond, atol=1e-6), \
            "At init, AdaLNZero with cond should equal normalization without cond"

    def test_adaln_zero_forward(self):
        """Forward pass with conditioning changes the output after perturbing weights."""
        dim, cond_dim = 64, 32
        adaln = AdaLNZero(dim, cond_dim, norm_type="rmsnorm")
        x = torch.randn(2, 8, dim)
        cond = torch.randn(2, cond_dim)

        # At init: identity.
        out_init = adaln(x, cond)
        out_base = adaln(x, None)
        assert torch.allclose(out_init, out_base, atol=1e-6)

        # Perturb modulation weights → output changes.
        with torch.no_grad():
            adaln.modulation.linear.weight.normal_(std=0.1)
            adaln.modulation.linear.bias.normal_(std=0.1)

        out_perturbed = adaln(x, cond)
        assert not torch.allclose(out_perturbed, out_base, atol=1e-4), \
            "After perturbing weights, conditioned output should differ from base"

    def test_adaln_zero_gradient(self):
        """Gradients flow through AdaLNZero modulation."""
        dim, cond_dim = 64, 32
        adaln = AdaLNZero(dim, cond_dim, norm_type="rmsnorm")
        x = torch.randn(2, 8, dim, requires_grad=True)
        cond = torch.randn(2, cond_dim, requires_grad=True)

        # Perturb weights so there's a non-trivial gradient.
        with torch.no_grad():
            adaln.modulation.linear.weight.normal_(std=0.1)

        out = adaln(x, cond)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow to x"
        assert cond.grad is not None, "Gradient should flow to cond"
        assert adaln.modulation.linear.weight.grad is not None, \
            "Gradient should flow to modulation weights"


# ── Model integration tests ───────────────────────────────────────────────────

class TestModelIntegration:
    """Model-level integration tests for AdaLN-zero and SIGReg."""

    def test_adaln_zero_backward_compat(self):
        """Model works without cond argument when cond_dim=None (backward compat)."""
        cfg = ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=128,
            max_seq_len=32,
            use_qk_norm=False,
            layer_types=["attention", "attention"],
            cond_dim=None,  # No AdaLN conditioning → standard norms.
        )
        model = ConfigurableResearchLLM(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            out = model(idx)
        # Standard output: (logits, loss).
        assert isinstance(out, tuple)
        logits = out[0]
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_model_with_adaln_zero_cond(self):
        """Model with cond_dim set accepts cond and produces correct output."""
        cond_dim = 16
        cfg = ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=128,
            max_seq_len=32,
            use_qk_norm=False,
            layer_types=["attention", "attention"],
            cond_dim=cond_dim,
        )
        model = ConfigurableResearchLLM(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        cond = torch.randn(2, cond_dim)

        # At init, AdaLN-zero is identity → output with cond == output without.
        with torch.no_grad():
            out_with_cond = model(idx, cond=cond)
            out_without_cond = model(idx)
        logits_cond = out_with_cond[0]
        logits_nocond = out_without_cond[0]
        assert logits_cond.shape == (2, 8, cfg.vocab_size)
        # At init, should be (nearly) identical.
        assert torch.allclose(logits_cond, logits_nocond, atol=1e-5), \
            "At init, AdaLN-zero should be identity (cond has no effect)"

    def test_model_return_hidden_states(self):
        """Model returns per-layer hidden states when return_hidden_states=True."""
        cfg = ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=3,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=128,
            max_seq_len=32,
            use_qk_norm=False,
            layer_types=["attention", "attention", "attention"],
            cond_dim=None,
        )
        model = ConfigurableResearchLLM(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            out = model(idx, return_hidden_states=True)
        # Output: (logits, loss, hidden_states_list).
        assert isinstance(out, tuple)
        assert len(out) == 3
        hidden_states_list = out[2]
        assert isinstance(hidden_states_list, list)
        assert len(hidden_states_list) == 3  # one per layer
        for h in hidden_states_list:
            assert h.shape == (2, 8, 64)

    def test_sigreg_with_model_hidden_states(self):
        """End-to-end: collect hidden states from model and compute SIGReg loss."""
        cfg = ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=128,
            max_seq_len=32,
            use_qk_norm=False,
            layer_types=["attention", "attention"],
            cond_dim=None,
        )
        model = ConfigurableResearchLLM(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            out = model(idx, return_hidden_states=True)
        hidden_states_list = out[2]

        sigreg = SIGRegLoss(threshold=0.1)
        loss = sigreg(hidden_states_list)
        # With random init, spectral norms are likely > 0.1 for d_model=64.
        # Just verify it's a finite scalar.
        assert loss.dim() == 0, "SIGReg loss should be scalar"
        assert torch.isfinite(loss)
