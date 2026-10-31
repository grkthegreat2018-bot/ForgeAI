"""Tests for R38-6: ForgeAdapter — Entropy-Guided Dynamic Adapter Fusion."""
from __future__ import annotations

import torch
import torch.nn as nn

from forge.training.forge_adapter import (
    ForgeAdapter,
    ForgeAdapterForLinear,
    apply_forge_adapter_to_model,
)


class TestForgeAdapter:
    """Multi-rank LoRA with entropy-guided fusion."""

    def test_construction(self):
        adapter = ForgeAdapter(64, 128, ranks=[4, 8, 16, 32], alpha=16)
        assert adapter.in_features == 64
        assert adapter.out_features == 128
        assert adapter.ranks == [4, 8, 16, 32]
        assert adapter.n_adapters == 4

    def test_adapter_parameters_shapes(self):
        adapter = ForgeAdapter(64, 128, ranks=[4, 8, 16])
        for i, r in enumerate([4, 8, 16]):
            assert adapter.adapters_A[i].shape == (64, r)
            assert adapter.adapters_B[i].shape == (r, 128)

    def test_b_zero_init(self):
        """B matrices are zero-init (standard LoRA no-op start)."""
        adapter = ForgeAdapter(64, 128, ranks=[4, 8])
        for B in adapter.adapters_B:
            assert torch.all(B == 0)

    def test_a_kaiming_init(self):
        """A matrices are kaiming-init (non-zero)."""
        adapter = ForgeAdapter(64, 128, ranks=[4, 8])
        for A in adapter.adapters_A:
            assert not torch.all(A == 0)

    def test_forward_no_entropy_uses_highest_rank(self):
        """Without entropy, uses the highest-rank adapter."""
        adapter = ForgeAdapter(64, 128, ranks=[4, 8, 16], alpha=16)
        x = torch.randn(2, 10, 64)
        out = adapter(x, entropy=None)
        assert out.shape == (2, 10, 128)

    def test_forward_with_entropy(self):
        """Forward with entropy produces correct shape."""
        adapter = ForgeAdapter(64, 128, ranks=[4, 8, 16], alpha=16, temperature=1.0)
        x = torch.randn(2, 10, 64)
        entropy = torch.rand(2, 10) * 5  # random entropy 0-5
        out = adapter(x, entropy=entropy)
        assert out.shape == (2, 10, 128)

    def test_forward_no_entropy_matches_single_adapter(self):
        """Without entropy, output = base + highest-rank adapter."""
        torch.manual_seed(42)
        adapter = ForgeAdapter(64, 128, ranks=[4, 8], alpha=16)
        torch.manual_seed(42)
        x = torch.randn(2, 5, 64)
        out = adapter(x, entropy=None)
        # Manually compute
        base = torch.nn.functional.linear(x, adapter.base_weight)
        A = adapter.adapters_A[-1]  # highest rank
        B = adapter.adapters_B[-1]
        scale = adapter.alpha / adapter.ranks[-1]
        lora = torch.nn.functional.linear(
            torch.nn.functional.linear(x, A.T), B.T) * scale
        expected = base + lora
        assert torch.allclose(out, expected, atol=1e-5)

    def test_merge_specific_adapter(self):
        """Merge a specific adapter into base weight."""
        adapter = ForgeAdapter(64, 128, ranks=[4, 8], alpha=16)
        merged = adapter.merge(0)  # merge rank-4 adapter
        assert merged.shape == (128, 64)

    def test_merge_all(self):
        """Merge all adapters."""
        adapter = ForgeAdapter(64, 128, ranks=[4, 8], alpha=16)
        all_merged = adapter.merge_all()
        assert len(all_merged) == 2
        for m in all_merged:
            assert m.shape == (128, 64)

    def test_entropy_routing(self):
        """Entropy routing produces valid distribution."""
        adapter = ForgeAdapter(64, 128, ranks=[4, 8, 16, 32], temperature=1.0)
        entropy = torch.tensor([[0.5, 2.0, 5.0, 8.0]])
        routing = adapter.get_entropy_routing(entropy)
        assert routing.shape == (1, 4, 4)  # (batch, seq, n_adapters)
        # Each token's routing weights sum to 1 (softmax)
        assert torch.allclose(routing.sum(dim=-1), torch.ones(1, 4), atol=1e-5)

    def test_hard_route(self):
        """Hard routing assigns each token to nearest bin."""
        adapter = ForgeAdapter(64, 128, ranks=[4, 8, 16, 32])
        # entropy_bins = linspace(0, 10, 5) = [0, 2.5, 5, 7.5, 10]
        # bin_centers = [1.25, 3.75, 6.25, 8.75]
        entropy = torch.tensor([[0.5, 3.0, 6.0, 9.0]])
        routes = adapter.hard_route(entropy)
        # 0.5 → closest to 1.25 → adapter 0
        # 3.0 → closest to 3.75 → adapter 1
        # 6.0 → closest to 6.25 → adapter 2
        # 9.0 → closest to 8.75 → adapter 3
        assert routes[0, 0] == 0
        assert routes[0, 1] == 1
        assert routes[0, 2] == 2
        assert routes[0, 3] == 3

    def test_low_entropy_routes_to_low_rank(self):
        """Low entropy tokens should weight more on low-rank adapters."""
        adapter = ForgeAdapter(64, 128, ranks=[4, 8, 16, 32], temperature=1.0)
        low_ent = torch.tensor([[1.0]])  # close to bin 0 (center 1.25)
        high_ent = torch.tensor([[9.0]])  # close to bin 3 (center 8.75)
        low_routing = adapter.get_entropy_routing(low_ent)
        high_routing = adapter.get_entropy_routing(high_ent)
        # Low entropy → more weight on adapter 0
        assert low_routing[0, 0, 0] > low_routing[0, 0, 3]
        # High entropy → more weight on adapter 3
        assert high_routing[0, 0, 3] > high_routing[0, 0, 0]


class TestForgeAdapterForLinear:
    """Wrapper for nn.Linear."""

    def test_construction(self):
        linear = nn.Linear(64, 128, bias=True)
        wrapped = ForgeAdapterForLinear(linear, ranks=[4, 8])
        assert wrapped.adapter.in_features == 64
        assert wrapped.adapter.out_features == 128
        assert wrapped.has_bias

    def test_weight_copied(self):
        """Base weight should match original linear."""
        linear = nn.Linear(64, 128)
        wrapped = ForgeAdapterForLinear(linear, ranks=[4, 8])
        assert torch.allclose(wrapped.adapter.base_weight, linear.weight)

    def test_forward_no_entropy(self):
        linear = nn.Linear(64, 128)
        wrapped = ForgeAdapterForLinear(linear, ranks=[4, 8])
        x = torch.randn(2, 10, 64)
        out = wrapped(x, entropy=None)
        assert out.shape == (2, 10, 128)

    def test_forward_with_bias(self):
        linear = nn.Linear(64, 128, bias=True)
        wrapped = ForgeAdapterForLinear(linear, ranks=[4, 8])
        x = torch.randn(2, 5, 64)
        out = wrapped(x, entropy=torch.rand(2, 5))
        assert out.shape == (2, 5, 128)


class TestApplyForgeAdapter:
    """Model-level application."""

    def test_apply_to_model(self):
        """Apply ForgeAdapter to all Linear layers in a model."""
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 64)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = SmallModel()
        n = apply_forge_adapter_to_model(model, ranks=[4, 8])
        assert n == 2
        assert isinstance(model.fc1, ForgeAdapterForLinear)
        assert isinstance(model.fc2, ForgeAdapterForLinear)

    def test_apply_with_target_modules(self):
        """Apply only to specific modules."""
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.out_proj = nn.Linear(64, 64)

            def forward(self, x):
                return self.out_proj(self.v_proj(x) + self.q_proj(x))

        model = SmallModel()
        n = apply_forge_adapter_to_model(model, ranks=[4], target_modules=["q_proj", "v_proj"])
        assert n == 2
        assert isinstance(model.q_proj, ForgeAdapterForLinear)
        assert isinstance(model.v_proj, ForgeAdapterForLinear)
        assert not isinstance(model.out_proj, ForgeAdapterForLinear)  # not wrapped

    def test_forward_after_apply(self):
        """Model forward works after applying ForgeAdapter."""
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def forward(self, x):
                return self.fc(x)

        model = SmallModel()
        apply_forge_adapter_to_model(model, ranks=[4, 8])
        x = torch.randn(2, 10, 64)
        out = model(x)
        assert out.shape == (2, 10, 64)
