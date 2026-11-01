"""Unit tests for DecisionScorer (forge/engine/decision_head.py).

Tier-1 calibrated decision head: a linear verifier probe on last-token
hidden states, trained with group-softmax cross-entropy (proper scoring
rule).  Tests cover the module mechanics (score/probs/temperature/
persistence), the ECE helper, and decide() integration — a loaded
scorer routes through _head_scores instead of raw LM probabilities.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forge.engine.decide import SystemOneEvaluator
from forge.engine.decision_head import (
    DecisionScorer,
    expected_calibration_error,
    fit_decision_scorer,
)


def _tiny_model():
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM

    cfg = get_config("gen_model_tiny")
    cfg.device = "cpu"
    cfg.dtype = "float32"
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from research.tokenizer_cache import get_tokenizer

        return get_tokenizer()
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


# ── DecisionScorer module ──────────────────────────────────────────────


class TestScorerModule:
    def test_score_shape(self):
        s = DecisionScorer(64)
        h = torch.randn(5, 64)
        assert s.score(h).shape == (5,)

    def test_probs_sum_to_one(self):
        s = DecisionScorer(64)
        p = s.probs([1.0, 0.0, -1.0])
        assert len(p) == 3
        assert sum(p) == pytest.approx(1.0)
        assert p[0] > p[1] > p[2]

    def test_temperature_softens(self):
        s = DecisionScorer(64, temperature=4.0)
        hot = s.probs([2.0, 0.0])
        s.temperature = 1.0
        cold = s.probs([2.0, 0.0])
        assert hot[0] < cold[0]  # higher T → flatter

    def test_mlp_variant(self):
        s = DecisionScorer(64, hidden_dim=32)
        assert s.score(torch.randn(3, 64)).shape == (3,)

    def test_save_load_roundtrip(self, tmp_path):
        s = DecisionScorer(64, temperature=1.7)
        with torch.no_grad():
            for p in s.parameters():
                p.copy_(torch.randn_like(p))
        path = str(tmp_path / "scorer.pt")
        s.save(path, meta={"n": 42})
        s2 = DecisionScorer.load(path)
        assert s2.temperature == pytest.approx(1.7)
        h = torch.randn(2, 64)
        assert torch.allclose(s.score(h), s2.score(h), atol=1e-6)

    def test_save_load_mlp(self, tmp_path):
        s = DecisionScorer(64, hidden_dim=16)
        path = str(tmp_path / "mlp.pt")
        s.save(path)
        s2 = DecisionScorer.load(path)
        assert isinstance(s2.net, torch.nn.Sequential)


# ── ECE ────────────────────────────────────────────────────────────────


class TestECE:
    def test_perfect_is_zero(self):
        p = torch.tensor([0.95, 0.95, 0.05, 0.05])
        y = torch.tensor([1.0, 1.0, 0.0, 0.0])
        # confident and right/wrong-aligned → ECE ≈ 0.05
        assert expected_calibration_error(p, y) < 0.1

    def test_worst_is_high(self):
        p = torch.tensor([0.9] * 10)
        y = torch.tensor([0.0] * 10)
        assert expected_calibration_error(p, y) > 0.8


# ── linear separability (module-level learning) ───────────────────────


class TestLinearLearning:
    def test_learns_separable_direction(self):
        """A planted direction in hidden space must be learnable."""
        torch.manual_seed(0)
        d = 256
        w_true = torch.randn(d)
        X = torch.randn(400, d)
        (X @ w_true + 0.5 * torch.randn(400) > 0).long()
        # group into fake questions of 2 candidates (pos/neg style):
        # build pairs where exactly one row satisfies y=1 — approximate
        # by grouping consecutive rows and marking argmax of true logit
        groups, labels = [], []
        for i in range(0, 400, 2):
            groups.append(X[i : i + 2])
            labels.append(int((X[i : i + 2] @ w_true).argmax()))
        scorer = DecisionScorer(d)
        opt = torch.optim.AdamW(scorer.parameters(), lr=3e-3)
        for _ in range(5):
            for g, lab in zip(groups, labels, strict=False):
                opt.zero_grad()
                loss = F.cross_entropy(scorer.score(g).unsqueeze(0), torch.tensor([lab]))
                loss.backward()
                opt.step()
        correct = sum(int(scorer.score(g).argmax() == lab) for g, lab in zip(groups, labels, strict=False))
        assert correct / len(groups) > 0.8


# ── decide() integration ───────────────────────────────────────────────

_STATE = "Acme Corp reported record Q3 revenue."


@pytest.fixture(scope="module")
def evaluator(tokenizer):
    model = _tiny_model()
    return SystemOneEvaluator(model, tokenizer, torch.device("cpu"))


class TestScorerIntegration:
    @staticmethod
    def _zero(scorer):
        with torch.no_grad():
            for p in scorer.parameters():
                p.zero_()
        return scorer

    def test_decide_uses_head_probs(self, evaluator):
        """A stub scorer with known outputs must drive the answer."""
        scorer = DecisionScorer(evaluator.model.config.d_model)
        # constant scores → softmax deterministic: 0.622/0.378 for 2 cands
        scorer.net.weight.data.zero_()
        scorer.net.bias.data.fill_(0.5)
        evaluator.scorer = scorer
        try:
            out = evaluator.evaluate(_STATE, {"q": {"type": "noul", "instructions": "x?"}})
            p = out["answers"]["q"]["noul"]
            # both candidates score 0.5 → uniform 0.5
            assert p == pytest.approx(0.5)
        finally:
            evaluator.scorer = None

    def test_decide_falls_back_without_scorer(self, evaluator):
        evaluator.scorer = None
        out = evaluator.evaluate(_STATE, {"q": {"type": "noul", "instructions": "x?"}})
        assert 0.0 <= out["answers"]["q"]["noul"] <= 1.0

    def test_head_path_all_types(self, evaluator):
        scorer = DecisionScorer(evaluator.model.config.d_model)
        evaluator.scorer = scorer
        try:
            out = evaluator.evaluate(
                _STATE,
                {
                    "n": {"type": "noul", "instructions": "ok?"},
                    "c": {"type": "choice", "criteria": {"a": None, "b": None}},
                    "s": {"type": "score", "criteria": ["lo", "hi"]},
                },
            )
            assert out["answers"]["n"]["type"] == "noul"
            assert out["answers"]["c"]["type"] == "choice"
            assert out["answers"]["s"]["type"] == "score"
            for ans in out["answers"].values():
                probs = ans.get("probabilities")
                if probs:
                    assert sum(probs.values()) == pytest.approx(1.0)
        finally:
            evaluator.scorer = None

    def test_scorer_temperature_applies(self, evaluator):
        scorer = self._zero(DecisionScorer(evaluator.model.config.d_model, temperature=3.0))
        evaluator.scorer = scorer
        try:
            out = evaluator.evaluate(_STATE, {"c": {"type": "choice", "criteria": {"a": None, "b": None}}})
            probs = out["answers"]["c"]["probabilities"]
            # zero-init scorer → logits 0 → uniform regardless
            assert probs["a"] == pytest.approx(probs["b"])
        finally:
            evaluator.scorer = None


# ── fit_decision_scorer smoke ─────────────────────────────────────────


class TestFit:
    def test_fit_runs_and_returns_metrics(self, tokenizer):
        """End-to-end harness check on the tiny model (mechanics only —
        random weights can't learn real semantics)."""
        model = _tiny_model()
        dataset = [
            ("state a", {"type": "noul", "instructions": "q1?"}, 0),
            ("state b", {"type": "noul", "instructions": "q2?"}, 1),
            ("state c", {"type": "noul", "instructions": "q3?"}, 0),
            ("state d", {"type": "noul", "instructions": "q4?"}, 1),
            ("state e", {"type": "noul", "instructions": "q5?"}, 0),
        ]
        scorer, metrics = fit_decision_scorer(model, tokenizer, "cpu", dataset, epochs=2, val_frac=0.2, progress=False)
        assert isinstance(scorer, DecisionScorer)
        assert metrics["n_examples"] == 5
        assert metrics["forwards"] == 10  # 5 questions x 2 candidates
        assert "harvest_s" in metrics
