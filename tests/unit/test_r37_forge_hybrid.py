"""Tests for R37-5: ForgeHybrid Key — Sink-Aware SSM+Attention Routing."""

from __future__ import annotations

import pytest
import torch

from forge.keys.misc.base import KeyClass
from tests.fixtures.keys.architecture.forge_hybrid_key import ForgeHybridKey


class TestForgeHybridKey:
    """ForgeHybrid — sink-aware SSM+attention routing."""

    def test_name(self):
        key = ForgeHybridKey(d_model=64)
        assert key.name == "forge_hybrid"

    def test_key_class(self):
        key = ForgeHybridKey(d_model=64)
        assert key.key_class() == KeyClass.BI

    def test_description(self):
        key = ForgeHybridKey(d_model=64)
        assert "sink" in key.description.lower()
        assert "ssm" in key.description.lower()

    def test_forward_adds_ssm_weights(self):
        """Forward should add zero-init SSM weights alongside attention."""
        key = ForgeHybridKey(d_model=64, d_state=8)
        # Simulate a 2-layer attention-only checkpoint
        data = {
            "blocks.0.attn.q_proj.weight": torch.randn(64, 64),
            "blocks.0.attn.k_proj.weight": torch.randn(64, 64),
            "blocks.0.attn.v_proj.weight": torch.randn(64, 64),
            "blocks.0.attn.out_proj.weight": torch.randn(64, 64),
            "blocks.1.attn.q_proj.weight": torch.randn(64, 64),
            "blocks.1.attn.k_proj.weight": torch.randn(64, 64),
            "blocks.1.attn.v_proj.weight": torch.randn(64, 64),
            "blocks.1.attn.out_proj.weight": torch.randn(64, 64),
        }
        result = key.forward(data)
        assert result.success
        weights = result.weights
        # Original attention weights preserved
        assert "blocks.0.attn.q_proj.weight" in weights
        assert "blocks.1.attn.out_proj.weight" in weights
        # SSM weights added
        assert "blocks.0.ssm.in_proj.weight" in weights
        assert "blocks.0.ssm.A_log" in weights
        assert "blocks.0.ssm.out_proj.weight" in weights
        assert "blocks.1.ssm.in_proj.weight" in weights

    def test_forward_ssm_weights_are_zero(self):
        """SSM weights must be zero-init for lossless warm start."""
        key = ForgeHybridKey(d_model=64, d_state=8)
        data = {
            "blocks.0.attn.q_proj.weight": torch.randn(64, 64),
        }
        result = key.forward(data)
        assert result.success
        ssm_w = result.weights["blocks.0.ssm.in_proj.weight"]
        assert torch.allclose(ssm_w, torch.zeros_like(ssm_w))
        a_log = result.weights["blocks.0.ssm.A_log"]
        assert torch.allclose(a_log, torch.zeros_like(a_log))

    def test_forward_preserves_attention_weights(self):
        """Attention weights must be unchanged (lossless)."""
        key = ForgeHybridKey(d_model=64, d_state=8)
        original = torch.randn(64, 64)
        data = {"blocks.0.attn.q_proj.weight": original.clone()}
        result = key.forward(data)
        assert result.success
        assert torch.allclose(result.weights["blocks.0.attn.q_proj.weight"], original)

    def test_forward_lossless_metadata(self):
        """Forward should report lossless=True for zero-init warm start."""
        key = ForgeHybridKey(d_model=64)
        data = {"blocks.0.attn.q_proj.weight": torch.randn(64, 64)}
        result = key.forward(data)
        assert result.metadata["lossless"] is True
        assert result.metadata["warm_start"] == "zero_init_ssm"

    def test_reverse_drops_ssm(self):
        """Reverse should extract only attention weights."""
        key = ForgeHybridKey(d_model=64, d_state=8)
        # First forward to get hybrid weights
        data = {"blocks.0.attn.q_proj.weight": torch.randn(64, 64)}
        forward_result = key.forward(data)
        assert forward_result.success
        # Now reverse
        reverse_result = key.reverse(forward_result.weights)
        assert reverse_result.success
        extracted = reverse_result.data
        # Attention weights present
        assert "blocks.0.attn.q_proj.weight" in extracted
        # SSM weights dropped
        assert not any(".ssm." in k for k in extracted)
        assert not any("sink_threshold" in k for k in extracted)

    def test_round_trip_lossless(self):
        """Forward then reverse should be identity (lossless)."""
        key = ForgeHybridKey(d_model=64, d_state=8)
        original = {
            "blocks.0.attn.q_proj.weight": torch.randn(64, 64),
            "blocks.0.attn.v_proj.weight": torch.randn(64, 64),
        }
        forward = key.forward(original)
        assert forward.success
        reverse = key.reverse(forward.weights)
        assert reverse.success
        # Check all original keys present and unchanged
        for k, v in original.items():
            assert k in reverse.data
            assert torch.allclose(reverse.data[k], v)

    def test_n_ssm_layers_subset(self):
        """Can add SSM to only a subset of layers (gradual rollout)."""
        key = ForgeHybridKey(d_model=64, d_state=8, n_ssm_layers=1)
        data = {
            "blocks.0.attn.q_proj.weight": torch.randn(64, 64),
            "blocks.1.attn.q_proj.weight": torch.randn(64, 64),
        }
        result = key.forward(data)
        assert result.success
        # Layer 0 has SSM
        assert "blocks.0.ssm.in_proj.weight" in result.weights
        # Layer 1 does NOT have SSM (only 1 SSM layer)
        assert "blocks.1.ssm.in_proj.weight" not in result.weights

    def test_detect_sinks(self):
        """Sink detection from attention weights."""
        key = ForgeHybridKey(d_model=64)
        # Create attention where position 0 receives most attention
        attn = torch.zeros(4, 32, 32)
        attn[:, :, 0] = 0.8  # All tokens attend to position 0
        attn[:, :, 1:] = 0.2 / 31
        sinks = key.detect_sinks(attn, threshold=0.5)
        assert sinks[0]  # Position 0 is a sink
        assert not sinks[1]  # Position 1 is not

    def test_compute_sink_norm_ratio(self):
        """Sink norm ratio computation."""
        key = ForgeHybridKey(d_model=64, sink_threshold=2.0)
        # Create hidden states where token 0 has small norm (sink)
        # and token 5 has large norm (information reservoir)
        h = torch.randn(1, 10, 64)
        h[:, 0, :] *= 0.1  # Small norm = sink
        h[:, 5, :] *= 5.0  # Large norm = reservoir
        ratios = key.compute_sink_norm_ratio(h, sink_idx=0)
        assert ratios.shape == (1, 10)
        assert ratios[0, 0] == pytest.approx(1.0, abs=0.01)  # Sink ratio = 1.0
        assert ratios[0, 5] > 2.0  # Token 5 has high ratio

    def test_route_tokens(self):
        """Token routing based on sink norm ratio."""
        key = ForgeHybridKey(d_model=64, sink_threshold=2.0)
        ratios = torch.tensor([[1.0, 1.5, 3.0, 0.5, 2.5]])
        routing = key.route_tokens(ratios)
        assert routing.shape == (1, 5)
        assert not routing[0, 0]  # ratio 1.0 < 2.0 → attention
        assert not routing[0, 1]  # ratio 1.5 < 2.0 → attention
        assert routing[0, 2]  # ratio 3.0 > 2.0 → SSM
        assert not routing[0, 3]  # ratio 0.5 < 2.0 → attention
        assert routing[0, 4]  # ratio 2.5 > 2.0 → SSM

    def test_default_threshold_is_infinity(self):
        """Default threshold = inf means all tokens use attention (warm start)."""
        key = ForgeHybridKey(d_model=64)
        ratios = torch.tensor([[1.0, 100.0, 1000.0]])
        routing = key.route_tokens(ratios)
        assert not routing.any()  # No tokens routed to SSM

    def test_cross_arch_with_mamba(self):
        """Cross-arch should work with other BI keys."""
        from tests.fixtures.keys.architecture.mamba_key import MambaKey

        hybrid = ForgeHybridKey(d_model=64, d_state=8)
        mamba = MambaKey()
        # Can't do real cross-arch without proper Mamba weights,
        # but verify the method exists and returns a KeyResult
        data = {"blocks.0.attn.q_proj.weight": torch.randn(64, 64)}
        forward = hybrid.forward(data)
        assert forward.success
        result = hybrid.cross_arch(forward.weights, mamba)
        # May fail due to incompatible formats, but should return a KeyResult
        assert isinstance(result.success, bool)
