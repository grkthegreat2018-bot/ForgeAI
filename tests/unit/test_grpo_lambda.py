"""Tests for GRPO-λ dynamic length penalty and GoalScorer minimalism toggle.

GRPO-λ (arXiv 2509.01155): dynamically adjust reward strategy based on
group correctness ratio. When correctness is low, disable length penalty
to prioritize reasoning. When high, activate penalty for efficiency.

Covers:
  - GRPOTrainer._group_correctness_ratio
  - GRPOTrainer._apply_grpo_lambda_penalty (low ratio = no penalty, high = penalty)
  - GRPOTrainer warmup (no penalty during early steps)
  - GoalScorer minimalism toggle (set_minimalism_active)
  - GoalScorer weight redistribution when minimalism disabled
"""

import torch

from forge.evaluation.goal_scorer import GoalScorer
from forge.self_play.grpo_trainer import GRPOConfig, GRPOTrainer


def _make_trainer(config=None):
    """Create a minimal GRPOTrainer for unit testing (no real model needed)."""
    model = torch.nn.Linear(10, 10)
    ref_model = torch.nn.Linear(10, 10)
    tokenizer = type(
        "MockTok",
        (),
        {"__call__": lambda self, text, **kw: type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3]])})()},
    )()
    return GRPOTrainer(model, tokenizer, ref_model, device="cpu", config=config or GRPOConfig())


# ── GRPO-λ: correctness ratio ───────────────────────────────────────────────


class TestGroupCorrectnessRatio:
    def test_all_correct(self):
        trainer = _make_trainer()
        cr = trainer._group_correctness_ratio([1.0, 1.0, 1.0, 1.0])
        assert cr == 1.0

    def test_all_incorrect(self):
        trainer = _make_trainer()
        cr = trainer._group_correctness_ratio([0.0, 0.0, 0.0, 0.0])
        assert cr == 0.0

    def test_mixed(self):
        trainer = _make_trainer()
        cr = trainer._group_correctness_ratio([1.0, 0.0, 1.0, 0.0])
        assert cr == 0.5

    def test_empty(self):
        trainer = _make_trainer()
        cr = trainer._group_correctness_ratio([])
        assert cr == 0.0

    def test_threshold_uses_099(self):
        """Rewards >= 0.99 count as correct (handles continuous tool-use rewards)."""
        trainer = _make_trainer()
        cr = trainer._group_correctness_ratio([0.99, 0.98, 1.0, 0.5])
        assert cr == 0.5  # only 0.99 and 1.0 pass


# ── GRPO-λ: dynamic penalty ─────────────────────────────────────────────────


class TestGrpoLambdaPenalty:
    def test_disabled_by_default(self):
        """GRPO-λ is off by default — rewards unchanged."""
        config = GRPOConfig()
        assert not config.use_grpo_lambda
        trainer = _make_trainer(config)
        rewards = [1.0, 0.0, 1.0, 0.0]
        comps = ["a" * 100, "b" * 50, "c" * 200, "d" * 10]
        out = trainer._apply_grpo_lambda_penalty(rewards, comps, 0.5)
        assert out == rewards  # unchanged

    def test_low_correctness_no_penalty(self):
        """When correctness ratio < threshold, no penalty applied."""
        config = GRPOConfig(use_grpo_lambda=True, length_penalty_threshold=0.6)
        trainer = _make_trainer(config)
        rewards = [1.0, 0.0, 0.0, 0.0]  # cr = 0.25 < 0.6
        comps = ["a" * 100, "b" * 50, "c" * 200, "d" * 10]
        out = trainer._apply_grpo_lambda_penalty(rewards, comps, 0.25)
        assert out == rewards  # no penalty

    def test_high_correctness_applies_penalty(self):
        """When correctness ratio >= threshold, penalty applied to correct solutions."""
        config = GRPOConfig(use_grpo_lambda=True, length_penalty_coeff=0.01, length_penalty_threshold=0.6)
        trainer = _make_trainer(config)
        rewards = [1.0, 1.0, 0.0, 1.0]  # cr = 0.75 >= 0.6
        comps = ["a" * 100, "b" * 50, "c" * 200, "d" * 10]
        out = trainer._apply_grpo_lambda_penalty(rewards, comps, 0.75)

        # Correct solutions get penalized by -λ * n_tokens
        # "a"*100 → 100 chars → ~25 tokens → penalty = 0.01 * 25 = 0.25
        assert out[0] < 1.0  # penalized
        assert out[1] < 1.0  # penalized
        assert out[2] == 0.0  # incorrect, unchanged
        assert out[3] < 1.0  # penalized

    def test_penalty_proportional_to_length(self):
        """Longer correct solutions get more penalty."""
        config = GRPOConfig(use_grpo_lambda=True, length_penalty_coeff=0.01, length_penalty_threshold=0.5)
        trainer = _make_trainer(config)
        rewards = [1.0, 1.0]
        comps = ["a" * 400, "b" * 40]  # 100 tokens vs 10 tokens
        out = trainer._apply_grpo_lambda_penalty(rewards, comps, 1.0)

        # Longer solution should have more penalty (lower adjusted reward)
        assert out[0] < out[1]

    def test_warmup_blocks_penalty(self):
        """During warmup steps, no penalty even if correctness is high."""
        config = GRPOConfig(
            use_grpo_lambda=True, length_penalty_coeff=0.01, length_penalty_threshold=0.6, length_penalty_warmup=10
        )
        trainer = _make_trainer(config)
        # total_steps is 0, warmup is 10 → penalty blocked
        rewards = [1.0, 1.0, 1.0, 1.0]  # cr = 1.0
        comps = ["a" * 100, "b" * 50, "c" * 200, "d" * 10]
        out = trainer._apply_grpo_lambda_penalty(rewards, comps, 1.0)
        assert out == rewards  # no penalty during warmup

    def test_penalty_tracks_activation_count(self):
        """Stats should track how many groups had penalty active."""
        config = GRPOConfig(use_grpo_lambda=True, length_penalty_threshold=0.5)
        trainer = _make_trainer(config)
        assert trainer.stats.length_penalty_active_count == 0

        # High correctness → penalty active
        trainer._apply_grpo_lambda_penalty([1.0, 1.0], ["aa", "bb"], 1.0)
        assert trainer.stats.length_penalty_active_count == 1

        # Low correctness → penalty inactive
        trainer._apply_grpo_lambda_penalty([0.0, 0.0], ["aa", "bb"], 0.0)
        assert trainer.stats.length_penalty_active_count == 1  # unchanged

    def test_incorrect_solutions_not_penalized(self):
        """Length penalty only applies to correct solutions (reward >= 0.99)."""
        config = GRPOConfig(use_grpo_lambda=True, length_penalty_coeff=0.01, length_penalty_threshold=0.5)
        trainer = _make_trainer(config)
        rewards = [1.0, 0.0]
        comps = ["a" * 400, "b" * 400]  # both long
        out = trainer._apply_grpo_lambda_penalty(rewards, comps, 0.5)

        assert out[0] < 1.0  # correct → penalized
        assert out[1] == 0.0  # incorrect → unchanged


# ── GoalScorer minimalism toggle ────────────────────────────────────────────


class TestGoalScorerMinimalismToggle:
    def test_default_minimalism_active(self):
        scorer = GoalScorer()
        assert scorer.minimalism_active is True

    def test_can_disable_minimalism(self):
        scorer = GoalScorer(minimalism_active=False)
        assert scorer.minimalism_active is False

    def test_toggle_at_runtime(self):
        scorer = GoalScorer()
        assert scorer.minimalism_active is True
        scorer.set_minimalism_active(False)
        assert scorer.minimalism_active is False
        scorer.set_minimalism_active(True)
        assert scorer.minimalism_active is True

    def test_minimalism_disabled_changes_quality(self):
        """With minimalism disabled, long correct solutions should score
        higher than with minimalism enabled (no length penalty)."""
        # Long, correct, efficient solution
        long_code = "def solve(x):\n" + "    y = 0\n" * 60 + "    return y\n"
        # Short, correct, efficient solution
        short_code = "def solve(x):\n    return x\n"

        # With minimalism: short > long (conciseness rewarded)
        scorer_on = GoalScorer(minimalism_active=True)
        r_short_on = scorer_on.score(short_code, correct=True, exec_time_ms=5.0, mean_logprob=-0.5, goal_id="g1")
        r_long_on = scorer_on.score(long_code, correct=True, exec_time_ms=5.0, mean_logprob=-0.5, goal_id="g1")
        # Short should have higher minimalism score
        assert r_short_on.scores["minimalism"] > r_long_on.scores["minimalism"]

        # Without minimalism: long and short should have similar quality
        # (minimalism weight = 0, redistributed to other dims)
        scorer_off = GoalScorer(minimalism_active=False)
        r_short_off = scorer_off.score(short_code, correct=True, exec_time_ms=5.0, mean_logprob=-0.5, goal_id="g2")
        r_long_off = scorer_off.score(long_code, correct=True, exec_time_ms=5.0, mean_logprob=-0.5, goal_id="g2")
        # Minimalism score is still computed but weight is 0
        assert r_short_off.scores["minimalism"] > r_long_off.scores["minimalism"]
        # But quality difference should be smaller (minimalism weight = 0)
        diff_on = abs(r_short_on.quality - r_long_on.quality)
        diff_off = abs(r_short_off.quality - r_long_off.quality)
        assert diff_off < diff_on  # less difference without minimalism

    def test_weights_redistribute_when_disabled(self):
        """When minimalism is disabled, its weight should be redistributed."""
        scorer_on = GoalScorer(minimalism_active=True)
        scorer_off = GoalScorer(minimalism_active=False)

        w_on = scorer_on._active_weights
        w_off = scorer_off._active_weights

        assert w_on["minimalism"] == 0.35
        assert w_off["minimalism"] == 0.0

        # Redistributed weights should sum to 1.0
        assert abs(sum(w_off.values()) - 1.0) < 1e-6
        # Other weights should be higher when minimalism is off
        assert w_off["efficiency"] > w_on["efficiency"]
