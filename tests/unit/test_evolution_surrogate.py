"""Tests for forge.evolution.surrogate — SurrogateModel ensemble predictor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from forge.evolution.surrogate import SurrogateModel


@pytest.fixture
def surrogate():
    """Small-dim SurrogateModel for fast CPU tests."""
    return SurrogateModel(
        input_dim=8,
        hidden_dim=32,
        mode="mlp",
        n_ensemble=2,
        device=torch.device("cpu"),
    )


class TestSurrogateInit:
    """SurrogateModel construction and defaults."""

    def test_init_defaults(self):
        s = SurrogateModel(
            input_dim=8,
            hidden_dim=64,
            mode="mlp",
            n_ensemble=3,
            device=torch.device("cpu"),
        )
        assert s.input_dim == 8
        assert s.mode == "mlp"
        assert s.n_ensemble == 3
        assert len(s.nets) == 3
        assert len(s.optimizers) == 3
        assert s.n_trained == 0

    def test_init_device_cpu(self, surrogate):
        assert surrogate.device == torch.device("cpu")

    def test_init_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown surrogate mode"):
            SurrogateModel(mode="bogus", device=torch.device("cpu"))


class TestPredictFewSamples:
    """predict with < 10 training samples returns random scores."""

    def test_predict_returns_random_when_untrained(self, surrogate):
        cands = torch.rand(20, 8)
        scores = surrogate.predict(cands)
        assert scores.shape == (20,)

    def test_predict_random_values_in_range(self, surrogate):
        cands = torch.rand(50, 8)
        scores = surrogate.predict(cands)
        # torch.rand returns values in [0, 1)
        assert scores.min() >= 0.0
        assert scores.max() < 1.0 + 1e-5

    def test_predict_shape(self, surrogate):
        cands = torch.rand(13, 8)
        scores = surrogate.predict(cands)
        assert scores.shape == (13,)
        assert scores.ndim == 1


class TestTrainPredict:
    """train + predict pipeline with enough samples."""

    def test_train_then_predict_shape(self, surrogate):
        cands = torch.rand(15, 8)
        scores = torch.randn(15)
        surrogate.train(cands, scores)
        test_cands = torch.rand(25, 8)
        preds = surrogate.predict(test_cands)
        assert preds.shape == (25,)

    def test_train_increases_n_trained(self, surrogate):
        assert surrogate.n_trained == 0
        cands = torch.rand(12, 8)
        scores = torch.randn(12)
        surrogate.train(cands, scores)
        assert surrogate.n_trained == 12

    def test_train_multiple_batches_accumulates(self, surrogate):
        cands1 = torch.rand(8, 8)
        scores1 = torch.randn(8)
        surrogate.train(cands1, scores1)
        assert surrogate.n_trained == 8
        cands2 = torch.rand(7, 8)
        scores2 = torch.randn(7)
        surrogate.train(cands2, scores2)
        assert surrogate.n_trained == 15

    def test_predict_after_train_is_deterministic(self, surrogate):
        cands = torch.rand(15, 8)
        scores = torch.randn(15)
        surrogate.train(cands, scores)
        test_cands = torch.rand(10, 8)
        p1 = surrogate.predict(test_cands)
        p2 = surrogate.predict(test_cands)
        assert torch.allclose(p1, p2)


class TestFilterTopK:
    """filter_top_k returns valid, unique indices."""

    def test_filter_top_k_count(self, surrogate):
        cands = torch.rand(20, 8)
        idx = surrogate.filter_top_k(cands, k=5)
        assert idx.shape == (5,)

    def test_filter_top_k_indices_in_range(self, surrogate):
        cands = torch.rand(30, 8)
        idx = surrogate.filter_top_k(cands, k=5)
        assert (idx >= 0).all()
        assert (idx < 30).all()

    def test_filter_top_k_indices_unique(self, surrogate):
        cands = torch.rand(40, 8)
        idx = surrogate.filter_top_k(cands, k=8)
        assert len(idx.unique()) == 8

    def test_filter_top_k_k_larger_than_n(self, surrogate):
        cands = torch.rand(5, 8)
        idx = surrogate.filter_top_k(cands, k=10)
        # min(k, len) => 5
        assert idx.shape == (5,)

    def test_filter_top_k_after_train(self, surrogate):
        train_cands = torch.rand(15, 8)
        train_scores = torch.randn(15)
        surrogate.train(train_cands, train_scores)
        cands = torch.rand(50, 8)
        idx = surrogate.filter_top_k(cands, k=5)
        assert idx.shape == (5,)
        assert len(idx.unique()) == 5
        assert (idx >= 0).all()
        assert (idx < 50).all()


class TestExplorationBonus:
    """With n_trained < 10, predictions are random (high variance)."""

    def test_untrained_predictions_are_random(self, surrogate):
        cands = torch.rand(100, 8)
        p1 = surrogate.predict(cands)
        p2 = surrogate.predict(cands)
        # Random predictions differ across calls
        assert not torch.allclose(p1, p2)

    def test_untrained_predictions_have_variance(self, surrogate):
        cands = torch.rand(100, 8)
        scores = surrogate.predict(cands)
        # Random scores should have some spread
        assert scores.std() > 0.0


class TestGPMode:
    """GP mode surrogate."""

    def test_gp_predict_few_samples_returns_random(self):
        s = SurrogateModel(
            input_dim=8,
            hidden_dim=32,
            mode="gp",
            device=torch.device("cpu"),
        )
        cands = torch.rand(10, 8)
        scores = s.predict(cands)
        assert scores.shape == (10,)

    def test_gp_train_then_predict(self):
        s = SurrogateModel(
            input_dim=8,
            hidden_dim=32,
            mode="gp",
            device=torch.device("cpu"),
        )
        cands = torch.rand(6, 8)
        scores = torch.randn(6)
        s.train(cands, scores)
        assert s.n_trained == 6
        test_cands = torch.rand(8, 8)
        preds = s.predict(test_cands)
        assert preds.shape == (8,)

    def test_gp_train_accumulates(self):
        s = SurrogateModel(
            input_dim=8,
            hidden_dim=32,
            mode="gp",
            device=torch.device("cpu"),
        )
        s.train(torch.rand(3, 8), torch.randn(3))
        assert s.n_trained == 3
        s.train(torch.rand(4, 8), torch.randn(4))
        assert s.n_trained == 7
