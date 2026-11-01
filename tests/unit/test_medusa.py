"""Tests for Medusa speculative decoding heads."""

import pytest
import torch
import torch.nn as nn

from forge.decoding.medusa import MedusaHeads, MedusaTrainer, medusa_generate

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


class TinyModel(nn.Module):
    """Minimal model returning hidden states (d_model) with an lm_head."""

    def __init__(self, d_model=32, vocab_size=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        h = self.embed(input_ids)  # (B, T, d_model)
        h = self.linear(h)
        return h  # returns hidden states, not logits


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    return TinyModel(d_model=32, vocab_size=100)


@pytest.fixture
def medusa_heads():
    torch.manual_seed(42)
    return MedusaHeads(d_model=32, vocab_size=100, n_heads=4)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMedusaHeadsShape:
    def test_forward_shape(self, medusa_heads):
        h = torch.randn(2, 8, 32)
        out = medusa_heads(h)
        assert isinstance(out, list)
        assert len(out) == 4
        for logits in out:
            assert logits.shape == (2, 8, 100)

    def test_predict_candidates_shape(self, medusa_heads):
        h = torch.randn(1, 5, 32)
        tokens, probs = medusa_heads.predict_candidates(h, top_k=5)
        assert tokens.shape == (1, 4, 5)
        assert probs.shape == (1, 4, 5)

    def test_single_head(self):
        heads = MedusaHeads(d_model=16, vocab_size=50, n_heads=1)
        h = torch.randn(1, 3, 16)
        out = heads(h)
        assert len(out) == 1
        assert out[0].shape == (1, 3, 50)


class TestMedusaHeadsParams:
    def test_n_heads_param(self):
        heads = MedusaHeads(d_model=32, vocab_size=100, n_heads=3)
        assert len(heads.heads) == 3

    def test_custom_hidden_dim(self):
        heads = MedusaHeads(d_model=32, vocab_size=100, n_heads=2, hidden_dim=64)
        # First layer of first head should map d_model → hidden_dim
        assert heads.heads[0][0].in_features == 32
        assert heads.heads[0][0].out_features == 64

    def test_share_embedding(self):
        emb = nn.Linear(32, 100, bias=False)
        heads = MedusaHeads(d_model=32, vocab_size=100, n_heads=2, share_embedding=emb)
        # All heads should share the same weight as emb
        for head in heads.heads:
            assert head[-1].weight is emb.weight

    def test_gradient_flow(self, medusa_heads):
        h = torch.randn(1, 4, 32, requires_grad=True)
        out = medusa_heads(h)
        loss = sum(o.sum() for o in out)
        loss.backward()
        for p in medusa_heads.parameters():
            assert p.grad is not None


class TestMedusaTrainer:
    def test_compute_loss(self, tiny_model, medusa_heads):
        trainer = MedusaTrainer(tiny_model, medusa_heads, lr=1e-4)
        input_ids = torch.randint(0, 100, (2, 16))
        loss, stats = trainer.compute_loss(input_ids)
        assert loss.item() > 0
        assert "head_losses" in stats
        assert len(stats["head_losses"]) == 4
        # Main model should be frozen
        for p in tiny_model.parameters():
            assert not p.requires_grad

    def test_train_step_decreases_loss(self, tiny_model, medusa_heads):
        trainer = MedusaTrainer(tiny_model, medusa_heads, lr=1e-3)
        input_ids = torch.randint(0, 100, (2, 16))
        _, stats1 = trainer.compute_loss(input_ids)
        for _ in range(5):
            trainer.train_step(input_ids)
        _, stats2 = trainer.compute_loss(input_ids)
        assert stats2["avg_loss"] < stats1["avg_loss"]


class TestMedusaGenerate:
    def test_generate_greedy(self, tiny_model, medusa_heads):
        input_ids = torch.randint(0, 100, (1, 8))
        out = medusa_generate(tiny_model, medusa_heads, input_ids, max_new_tokens=10, temperature=0.0, device="cpu")
        assert out.shape[0] == 1
        assert out.shape[1] >= 8  # at least the prompt
        assert out.shape[1] <= 18  # prompt + max_new_tokens

    def test_generate_sampling(self, tiny_model, medusa_heads):
        input_ids = torch.randint(0, 100, (1, 8))
        out = medusa_generate(tiny_model, medusa_heads, input_ids, max_new_tokens=10, temperature=0.8, device="cpu")
        assert out.shape[0] == 1
        assert out.shape[1] >= 8

    def test_generate_no_new_tokens(self, tiny_model, medusa_heads):
        input_ids = torch.randint(0, 100, (1, 8))
        out = medusa_generate(tiny_model, medusa_heads, input_ids, max_new_tokens=0, device="cpu")
        # Should return just the prompt (or prompt + 1 from first step)
        assert out.shape[1] >= 8
