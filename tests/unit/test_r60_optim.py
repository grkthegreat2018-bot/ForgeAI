"""Tests for the R60 training-side research additions:
SAMuon-lite optimizer (arXiv 2608.25990) and acceptance-aware draft-head
losses (arXiv 2609.24150): eal_loss + wtv_loss.

All CPU-runnable.
"""
import os
import sys

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from forge.training.optim.sf_spectral_optimizers import (
    SAMuonLite, SAMuonLiteWithAuxAdam, _newton_schulz_5, _top_singular_pair,
    build_samuon_lite)
from forge.decoding.acceptance_losses import (
    eal_loss, expected_accepted_length, wtv_loss)


# ═══════════════════════════════════════════════════════════════════════════
# SAMuon-lite
# ═══════════════════════════════════════════════════════════════════════════

def test_samuon_bulk_gain_1_reduces_to_plain_muon():
    """bulk_gain=1.0 must reproduce the plain Muon update:
    p ← p − lr · NS(mom) · max(1, m/n)^0.5  (up to bf16 rounding)."""
    torch.manual_seed(0)
    m, n = 32, 16
    p = nn.Parameter(torch.randn(m, n))
    grad = torch.randn(m, n)
    momentum, lr = 0.95, 0.05

    opt = SAMuonLite([p], lr=lr, momentum=momentum, bulk_gain=1.0,
                     power_iters=3, weight_decay=0.0)
    p.grad = grad
    p_init = p.detach().clone()
    opt.step()

    mom = (1 - momentum) * grad  # first step: mom = (1-β)·g
    expected = _newton_schulz_5(mom, 5).float()
    expected = expected * max(1, m / n) ** 0.5
    delta = p.detach() - p_init
    assert torch.allclose(delta, -lr * expected, rtol=0.05, atol=2e-2), (
        f"bulk_gain=1.0 should equal plain Muon; max diff "
        f"{(delta + lr * expected).abs().max().item():.4f}")


def test_samuon_bulk_gain_amplifies_bulk():
    """bulk_gain>1 must produce a larger update than bulk_gain=1 whenever
    the momentum buffer is not purely rank-1."""
    torch.manual_seed(1)
    m, n = 32, 16
    grad = torch.randn(m, n)
    updates = {}
    for gain in (1.0, 2.0):
        p = nn.Parameter(torch.zeros(m, n))
        opt = SAMuonLite([p], lr=0.1, momentum=0.95, bulk_gain=gain,
                         weight_decay=0.0)
        p.grad = grad
        opt.step()
        updates[gain] = p.detach().clone()
    assert updates[2.0].norm() > updates[1.0].norm()


def test_samuon_converges_toy_matrix_problem():
    """SAMuon-lite should reduce loss on ||W − W*||²_F; compare against a
    plain-Muon baseline (bulk_gain=1.0) run on the same problem."""
    torch.manual_seed(2)
    m, n = 32, 16
    W_star = torch.randn(m, n)

    def run(bulk_gain):
        torch.manual_seed(3)
        W = nn.Parameter(torch.randn(m, n))
        opt = SAMuonLite([W], lr=0.1, momentum=0.9, bulk_gain=bulk_gain,
                         weight_decay=0.0)
        losses = []
        for _ in range(60):
            loss = ((W - W_star) ** 2).sum()
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        return losses

    samuon = run(1.5)
    plain = run(1.0)
    assert samuon[-1] < 0.5 * samuon[0], \
        f"SAMuon-lite did not converge: {samuon[0]:.3f} → {samuon[-1]:.3f}"
    assert plain[-1] < 0.5 * plain[0], \
        f"plain Muon baseline did not converge: {plain[0]:.3f} → {plain[-1]:.3f}"
    # Spectral split should not be catastrophically worse than plain Muon.
    assert samuon[-1] < 5.0 * plain[-1]


def test_samuon_state_and_non2d_fallback():
    """Momentum state shape must match the param; non-2D params take the
    plain-momentum path (no orthogonalization)."""
    p2d = nn.Parameter(torch.randn(8, 4))
    p1d = nn.Parameter(torch.randn(8))
    p1d_init = p1d.detach().clone()
    opt = SAMuonLite([p2d, p1d], lr=0.02, momentum=0.9, bulk_gain=1.5)
    p2d.grad = torch.randn(8, 4)
    g1d = torch.randn(8)
    p1d.grad = g1d.clone()
    opt.step()
    assert opt.state[p2d]['momentum'].shape == p2d.shape
    assert opt.state[p1d]['momentum'].shape == p1d.shape
    # 1-D fallback: mom = (1-β)·g, p ← p − lr·mom.
    assert torch.allclose(opt.state[p1d]['momentum'], 0.1 * g1d)
    assert torch.allclose(p1d.detach(), p1d_init - 0.02 * 0.1 * g1d)


def test_samuon_top_singular_pair_recovers_rank1():
    """Power iteration on a rank-1 matrix must find the true u1, v1."""
    torch.manual_seed(4)
    u_true = torch.randn(24)
    u_true = u_true / u_true.norm()
    v_true = torch.randn(12)
    v_true = v_true / v_true.norm()
    G = 3.7 * torch.outer(u_true, v_true)
    u, v = _top_singular_pair(G, iters=5)
    assert abs(float(u.float() @ u_true)) > 0.99
    assert abs(float(v.float() @ v_true)) > 0.99


def test_samuon_builder_and_factory():
    """build_samuon_lite: hidden 2D → SAMuon-lite, embed/head/scalars → AdamW."""
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)      # embed → AdamW
            self.fc = nn.Linear(8, 8)             # hidden 2D → SAMuon
            self.norm = nn.LayerNorm(8)           # 1-D → AdamW
            self.head = nn.Linear(8, 16)          # head → AdamW

    model = Tiny()
    opt = build_samuon_lite(model, max_lr=0.003, weight_decay=0.01,
                            bulk_gain=1.5)
    assert isinstance(opt, SAMuonLiteWithAuxAdam)
    # Exposed param_groups = SAMuon groups only (hidden matrices).
    flat = [p for g in opt.param_groups for p in g['params']]
    assert len(flat) == 1 and flat[0].shape == (8, 8)
    assert opt.param_groups[0]['bulk_gain'] == 1.5

    # One step: every trainable param with nonzero grad should move.
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    x = torch.randint(0, 16, (2, 4))
    h = model.norm(torch.relu(model.fc(model.embed(x))))
    loss = model.head(h).sum()
    loss.backward()
    opt.step()
    moved = [n for n, p in model.named_parameters()
             if not torch.equal(p.detach(), before[n])]
    assert len(moved) == len(before), f"params not updated: {set(before) - set(moved)}"

    # Factory wiring.
    from forge.training.training_utils import configure_optimizer
    opt2 = configure_optimizer(model, 0.003, 0.01, optimizer_name="samuon_lite")
    assert isinstance(opt2, SAMuonLiteWithAuxAdam)


# ═══════════════════════════════════════════════════════════════════════════
# eal_loss — Expected Accepted Length (greedy verification)
# ═══════════════════════════════════════════════════════════════════════════

def _one_hot_logits(idx: int, vocab: int, scale: float = 30.0) -> torch.Tensor:
    """Logits that softmax ≈ one-hot at idx."""
    t = torch.zeros(vocab)
    t[idx] = scale
    return t


def test_eal_zero_when_draft_matches_target_greedy():
    W, V = 4, 8
    target = torch.stack([_one_hot_logits(i % V, V) for i in range(W)])
    draft = target.clone()
    loss = eal_loss(draft, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)
    # EAL itself should be ≈ W.
    assert expected_accepted_length(draft, target).item() == pytest.approx(W, abs=1e-3)


def test_eal_positive_and_monotonic_in_mismatch():
    W, V = 4, 8
    target = torch.stack([_one_hot_logits(0, V) for _ in range(W)])
    # Draft puts varying mass on the target's pick (token 0).
    near = torch.full((W, V), 0.0)
    near[:, 0] = 3.0   # strong but imperfect match
    mid = torch.full((W, V), 0.0)
    mid[:, 0] = 1.0    # weaker match
    far = torch.full((W, V), 0.0)
    far[:, 1] = 3.0    # peak on the WRONG token
    l_near = eal_loss(near, target).item()
    l_mid = eal_loss(mid, target).item()
    l_far = eal_loss(far, target).item()
    assert l_near > 0 and l_mid > 0 and l_far > 0
    assert l_near < l_mid < l_far


def test_eal_shapes_and_gradient():
    """(W, V) and (B, W, V) both work; gradient flows to draft logits."""
    B, W, V = 3, 5, 8
    draft = torch.randn(B, W, V, requires_grad=True)
    target = torch.randn(B, W, V)
    loss = eal_loss(draft, target)
    loss.backward()
    assert loss.dim() == 0
    assert draft.grad is not None and draft.grad.abs().sum() > 0
    # 2-D form returns a scalar too.
    loss2 = eal_loss(draft[0].detach(), target[0])
    assert loss2.dim() == 0


def test_eal_gradient_pushes_mass_to_target_argmax():
    """One step of GD on eal_loss must raise p_match at every position."""
    torch.manual_seed(5)
    W, V = 4, 8
    draft = torch.randn(W, V, requires_grad=True)
    target = torch.stack([_one_hot_logits(2, V) for _ in range(W)])
    before = expected_accepted_length(draft.detach(), target).item()
    loss = eal_loss(draft, target)
    loss.backward()
    with torch.no_grad():
        draft -= 1.0 * draft.grad
    after = expected_accepted_length(draft.detach(), target).item()
    assert after > before


# ═══════════════════════════════════════════════════════════════════════════
# wtv_loss — Window Total-Variation (sampling verification)
# ═══════════════════════════════════════════════════════════════════════════

def test_wtv_zero_at_identical_distributions():
    torch.manual_seed(6)
    W, V = 5, 7
    logits = torch.randn(W, V)
    assert wtv_loss(logits, logits.clone()).item() == pytest.approx(0.0, abs=1e-6)
    batched = logits.unsqueeze(0).repeat(2, 1, 1)
    assert wtv_loss(batched, batched.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_wtv_hand_computed_3class():
    """W=2, V=3, one-hot logits:
      pos1: q = p = [1,0,0]          → TV₁ = 0, w₁ = 1 → contrib 0
      pos2: q = [1,0,0], p = [0,1,0] → TV₂ = 1, w₂ = 1 → contrib 1
    Expected loss = 0·1 + 1·1 = 1.0."""
    W, V = 2, 3
    draft = torch.stack([_one_hot_logits(0, V), _one_hot_logits(0, V)])
    target = torch.stack([_one_hot_logits(0, V), _one_hot_logits(1, V)])
    assert wtv_loss(draft, target).item() == pytest.approx(1.0, abs=1e-3)


def test_wtv_sequential_weighting():
    """Once a position is fully rejected (TV=1), later positions get zero
    weight — the window is unreachable past that point."""
    V = 3
    # TV pattern [1.0, x]: loss must equal 1.0 regardless of x.
    draft = torch.stack([_one_hot_logits(0, V), _one_hot_logits(0, V)])
    tgt_same2 = torch.stack([_one_hot_logits(1, V), _one_hot_logits(0, V)])
    tgt_diff2 = torch.stack([_one_hot_logits(1, V), _one_hot_logits(2, V)])
    l_a = wtv_loss(draft, tgt_same2).item()
    l_b = wtv_loss(draft, tgt_diff2).item()
    assert l_a == pytest.approx(1.0, abs=1e-3)
    assert l_b == pytest.approx(1.0, abs=1e-3)

    # TV pattern [0.5, 0.5]: L = 0.5·1 + 0.5·0.5 = 0.75.
    # Use log-prob logits so softmax gives exact distributions.
    p_d = torch.tensor([0.5, 0.5, 1e-9])
    p_t = torch.tensor([0.5, 1e-9, 0.5])
    draft2 = torch.log(p_d).unsqueeze(0).repeat(2, 1)
    target2 = torch.log(p_t).unsqueeze(0).repeat(2, 1)
    # TV = 0.5·(|0|+|0.5|+|−0.5|) = 0.5 at each position.
    assert wtv_loss(draft2, target2).item() == pytest.approx(0.75, abs=1e-3)


def test_wtv_temperature_softens():
    """Higher temperature flattens both distributions → smaller TV."""
    torch.manual_seed(7)
    W, V = 3, 5
    draft = torch.randn(W, V)
    target = torch.randn(W, V)
    hot = wtv_loss(draft, target, temperature=4.0).item()
    cold = wtv_loss(draft, target, temperature=0.5).item()
    assert hot < cold


def test_wtv_gradient_flows():
    B, W, V = 2, 4, 6
    draft = torch.randn(B, W, V, requires_grad=True)
    target = torch.randn(B, W, V)
    loss = wtv_loss(draft, target)
    loss.backward()
    assert draft.grad is not None and draft.grad.abs().sum() > 0


# ═══════════════════════════════════════════════════════════════════════════
# DSpark trainer seam (loss_type routing)
# ═══════════════════════════════════════════════════════════════════════════

class _StubLLM(nn.Module):
    """Minimal main-model stub: returns hidden states via return_hidden,
    exposes .head for target logits (d_model != vocab on purpose)."""

    def __init__(self, vocab=16, d=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, return_hidden=False):
        h = self.embed_tokens(input_ids)
        logits = self.head(h)
        if return_hidden:
            return (logits, None, h)
        return (logits,)


def test_dspark_trainer_loss_type_routing():
    from forge.decoding.dspark import DSparkHead, DSparkTrainer

    torch.manual_seed(8)
    for loss_type in ("ce", "eal", "wtv"):
        model = _StubLLM()
        head = DSparkHead(d_model=8, vocab_size=16, n_predict=3,
                          n_layers=1, seq_mode="markov")
        trainer = DSparkTrainer(model, head, lr=1e-3, loss_type=loss_type)
        ids = torch.randint(0, 16, (2, 10))
        loss, stats = trainer.compute_loss(ids)
        assert torch.isfinite(loss)
        assert stats["loss_type"] == loss_type
        assert stats["ce_loss"] >= 0.0


def test_dspark_trainer_rejects_bad_loss_type():
    from forge.decoding.dspark import DSparkHead, DSparkTrainer
    model = _StubLLM()
    head = DSparkHead(d_model=8, vocab_size=16, n_predict=2, n_layers=1)
    with pytest.raises(ValueError):
        DSparkTrainer(model, head, loss_type="bogus")
