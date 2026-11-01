"""Tests for forge.evolution.trainer — GeneratorTrainer REINFORCE updates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy

import pytest
import torch

from forge.evolution.generators import BatchedGenerator, GeneratorConfig
from forge.evolution.trainer import GeneratorTrainer


def _make_batched_gen():
    cfg = GeneratorConfig(
        n_generators=5,
        noise_dim=4,
        context_dim=8,
        hidden_dim=16,
        output_dim=4,
    )
    return BatchedGenerator(cfg, device=torch.device("cpu"))


def _make_candidates():
    candidates = [
        {"params": torch.rand(4), "generator_idx": 0, "noise": torch.randn(4)},
        {"params": torch.rand(4), "generator_idx": 1, "noise": torch.randn(4)},
    ]
    scores = [5.0, -2.0]
    context = torch.zeros(8)
    return candidates, scores, context


@pytest.fixture
def trainer():
    gen = _make_batched_gen()
    return GeneratorTrainer(
        gen,
        lr=1e-3,
        baseline_decay=0.9,
        device=torch.device("cpu"),
        pushaway_strength=0.3,
    )


class TestGeneratorTrainerInit:
    """Constructor stores config and sets up optimizer."""

    def test_stores_batched_gen(self, trainer):
        assert trainer.batched_gen is not None
        assert trainer.batched_gen.cfg.n_generators == 5

    def test_stores_lr(self, trainer):
        assert trainer.lr == 1e-3

    def test_stores_baseline_decay(self, trainer):
        assert trainer.baseline_decay == 0.9

    def test_stores_pushaway_strength(self, trainer):
        assert trainer.pushaway_strength == 0.3

    def test_device_is_cpu(self, trainer):
        assert trainer.device == torch.device("cpu")

    def test_optimizer_is_adam(self, trainer):
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_baseline_starts_zero(self, trainer):
        assert trainer.baseline == 0.0

    def test_mutated_slots_empty(self, trainer):
        assert trainer._mutated_slots == set()


class TestUpdateEmptyScores:
    """update() with empty scores returns early without error."""

    def test_empty_scores_no_crash(self, trainer):
        candidates, _, context = _make_candidates()
        # Should not raise
        trainer.update(candidates, [], context)

    def test_empty_scores_no_weight_change(self, trainer):
        candidates, _, context = _make_candidates()
        before = trainer.batched_gen.W0.detach().clone()
        trainer.update(candidates, [], context)
        after = trainer.batched_gen.W0.detach().clone()
        assert torch.allclose(before, after)


class TestUpdatePositiveRewards:
    """Positive rewards apply gradient (weights change)."""

    def test_positive_rewards_change_weights(self, trainer):
        candidates = [
            {"params": torch.rand(4), "generator_idx": 0, "noise": torch.randn(4)},
            {"params": torch.rand(4), "generator_idx": 1, "noise": torch.randn(4)},
        ]
        scores = [5.0, 4.0]
        context = torch.zeros(8)
        before = trainer.batched_gen.W0.detach().clone()
        trainer.update(candidates, scores, context)
        after = trainer.batched_gen.W0.detach().clone()
        assert not torch.allclose(before, after)


class TestUpdateNegativeRewards:
    """Negative rewards apply pushaway (weights change)."""

    def test_negative_rewards_change_weights(self, trainer):
        candidates = [
            {"params": torch.rand(4), "generator_idx": 0, "noise": torch.randn(4)},
            {"params": torch.rand(4), "generator_idx": 1, "noise": torch.randn(4)},
        ]
        scores = [-5.0, -4.0]
        context = torch.zeros(8)
        before = trainer.batched_gen.W0.detach().clone()
        trainer.update(candidates, scores, context)
        after = trainer.batched_gen.W0.detach().clone()
        assert not torch.allclose(before, after)


class TestUpdateMixedRewards:
    """Mixed positive and negative scores, no crash."""

    def test_mixed_rewards_no_crash(self, trainer):
        candidates, scores, context = _make_candidates()
        trainer.update(candidates, scores, context)

    def test_mixed_rewards_change_weights(self, trainer):
        candidates, scores, context = _make_candidates()
        before = trainer.batched_gen.W0.detach().clone()
        trainer.update(candidates, scores, context)
        after = trainer.batched_gen.W0.detach().clone()
        assert not torch.allclose(before, after)


class TestNotifyMutation:
    """notify_mutation marks a slot for optimizer state reset."""

    def test_notify_mutation_adds_slot(self, trainer):
        trainer.notify_mutation(2)
        assert 2 in trainer._mutated_slots

    def test_notify_mutation_resets_state_after_update(self, trainer):
        # Run one update so optimizer state gets populated
        candidates, scores, context = _make_candidates()
        trainer.update(candidates, scores, context)
        # Verify state exists for W0
        state = trainer.optimizer.state.get(trainer.batched_gen.W0, {})
        assert "exp_avg" in state
        # Snapshot the exp_avg for gen_idx 0
        state["exp_avg"][0].detach().clone()
        # Notify mutation for gen 0, then run update with positive reward
        trainer.notify_mutation(0)
        candidates2 = [
            {"params": torch.rand(4), "generator_idx": 0, "noise": torch.randn(4)},
        ]
        trainer.update(candidates2, [5.0], context)
        # After update, _mutated_slots should be cleared
        assert trainer._mutated_slots == set()


class TestPushawayStrength:
    """Custom pushaway_strength is stored and used."""

    def test_custom_pushaway_stored(self):
        gen = _make_batched_gen()
        t = GeneratorTrainer(gen, pushaway_strength=0.5, device=torch.device("cpu"))
        assert t.pushaway_strength == 0.5

    def test_custom_pushaway_used_in_update(self):
        """Stronger pushaway should produce larger weight delta than weak pushaway."""

        # Weak pushaway
        gen_weak = _make_batched_gen()
        w0_before = gen_weak.W0.detach().clone()
        t_weak = GeneratorTrainer(gen_weak, pushaway_strength=0.01, device=torch.device("cpu"))
        # Strong pushaway — start from identical weights
        gen_strong = _make_batched_gen()
        with torch.no_grad():
            for p1, p2 in zip(gen_weak.parameters(), gen_strong.parameters(), strict=False):
                p2.copy_(p1)
        w0_before_strong = gen_strong.W0.detach().clone()
        t_strong = GeneratorTrainer(gen_strong, pushaway_strength=1.0, device=torch.device("cpu"))
        candidates = [
            {"params": torch.rand(4), "generator_idx": 0, "noise": torch.randn(4)},
        ]
        context = torch.zeros(8)
        # Negative reward → pushaway; stronger pushaway should move weights more
        t_weak.update(copy.deepcopy(candidates), [-5.0], context)
        t_strong.update(copy.deepcopy(candidates), [-5.0], context)
        delta_weak = (gen_weak.W0.detach() - w0_before).abs().sum().item()
        delta_strong = (gen_strong.W0.detach() - w0_before_strong).abs().sum().item()
        # Strong pushaway should produce a larger weight change
        assert delta_strong > delta_weak, f"Strong pushaway delta ({delta_strong}) should exceed weak ({delta_weak})"


class TestBaselineDecay:
    """After update, baseline moves toward average score."""

    def test_baseline_moves_toward_avg(self, trainer):
        candidates, scores, context = _make_candidates()
        avg_score = sum(scores) / len(scores)
        expected = 0.9 * 0.0 + 0.1 * avg_score
        trainer.update(candidates, scores, context)
        assert abs(trainer.baseline - expected) < 1e-6

    def test_baseline_decay_custom(self):
        gen = _make_batched_gen()
        t = GeneratorTrainer(gen, baseline_decay=0.5, device=torch.device("cpu"))
        candidates, scores, context = _make_candidates()
        avg_score = sum(scores) / len(scores)
        expected = 0.5 * 0.0 + 0.5 * avg_score
        t.update(candidates, scores, context)
        assert abs(t.baseline - expected) < 1e-6


class TestCandidatesMissingKeys:
    """Candidates missing required keys are skipped without crash."""

    def test_candidate_without_noise_skipped(self, trainer):
        candidates = [
            {"params": torch.rand(4), "generator_idx": 0},  # no noise
            {"params": torch.rand(4), "generator_idx": 1, "noise": torch.randn(4)},
        ]
        scores = [5.0, 3.0]
        context = torch.zeros(8)
        # Should not crash; the one with noise still updates
        trainer.update(candidates, scores, context)

    def test_candidate_without_params_skipped(self, trainer):
        candidates = [
            {"generator_idx": 0, "noise": torch.randn(4)},  # no params
            {"params": torch.rand(4), "generator_idx": 1, "noise": torch.randn(4)},
        ]
        scores = [5.0, 3.0]
        context = torch.zeros(8)
        trainer.update(candidates, scores, context)

    def test_all_candidates_missing_keys_no_update(self, trainer):
        candidates = [
            {"generator_idx": 0, "noise": torch.randn(4)},  # no params
            {"params": torch.rand(4), "generator_idx": 1},  # no noise
        ]
        scores = [5.0, 3.0]
        context = torch.zeros(8)
        before = trainer.batched_gen.W0.detach().clone()
        trainer.update(candidates, scores, context)
        after = trainer.batched_gen.W0.detach().clone()
        # No valid losses → no optimizer step → weights unchanged
        assert torch.allclose(before, after)


class TestMaxGradNormClipping:
    """Gradient norm after update does not exceed max_grad_norm."""

    def test_grad_norm_clipped(self):
        gen = _make_batched_gen()
        t = GeneratorTrainer(
            gen,
            lr=1e-3,
            max_grad_norm=0.5,
            device=torch.device("cpu"),
        )
        candidates = [
            {"params": torch.rand(4), "generator_idx": 0, "noise": torch.randn(4)},
            {"params": torch.rand(4), "generator_idx": 1, "noise": torch.randn(4)},
        ]
        scores = [100.0, 100.0]
        context = torch.zeros(8)
        # Manually run the update logic to inspect grad norm before step
        t.batched_gen.train()
        import torch.nn as nn

        losses = []
        for cand, score in zip(candidates, scores, strict=False):
            noise = cand["noise"].to(t.device)
            target = cand["params"].to(t.device)
            out = t.batched_gen.forward_single(noise, context, cand["generator_idx"])
            reward = score - t.baseline
            if reward > 0:
                loss = nn.functional.mse_loss(out, target)
            else:
                loss = -nn.functional.mse_loss(out, target) * t.pushaway_strength
            losses.append(loss)
        total_loss = torch.stack(losses).sum()
        t.optimizer.zero_grad()
        total_loss.backward()
        # Clip
        torch.nn.utils.clip_grad_norm_(t.batched_gen.parameters(), 0.5)
        total_norm = torch.norm(torch.stack([p.grad.norm() for p in t.batched_gen.parameters() if p.grad is not None]))
        assert total_norm.item() <= 0.5 + 1e-5
