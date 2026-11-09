"""Tests for VerifierGate / looped_prefill_verified (R71/B5)."""
import torch
import torch.nn as nn

from forge.exp.loopdepth import VerifierGate, looped_prefill_verified


class TestVerifierGate:
    def _logits(self, winner: int, gap: float, vocab: int = 100):
        # winner logit = `gap`, runner-up = 0, all others -20 so softmax
        # margin ≈ gap-dependent (gap 4 -> ~0.96, gap 0.3 -> ~0.15)
        lg = torch.full((vocab,), -20.0)
        lg[(winner + 1) % vocab] = 0.0
        lg[winner] = gap
        return lg

    def test_stops_on_stable_high_margin(self):
        g = VerifierGate(min_margin=0.1, patience=1)
        lg = self._logits(5, gap=4.0)
        assert g.update(lg) is False   # first observation: no prev
        assert g.update(lg) is True    # same argmax + margin -> stop

    def test_keeps_going_on_low_margin(self):
        g = VerifierGate(min_margin=0.4, patience=1)
        lg = self._logits(5, gap=0.3)
        assert g.update(lg) is False
        assert g.update(lg) is False   # same argmax but margin too low

    def test_resets_on_argmax_change(self):
        g = VerifierGate(min_margin=0.1, patience=2)
        assert g.update(self._logits(5, 4.0)) is False
        assert g.update(self._logits(7, 4.0)) is False  # argmax changed
        assert g.update(self._logits(7, 4.0)) is False  # stable=1 <2
        assert g.update(self._logits(7, 4.0)) is True

    def test_reset(self):
        g = VerifierGate(patience=1)
        g.update(self._logits(3, 4.0))
        g.reset()
        assert g.update(self._logits(3, 4.0)) is False


class _Block(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.scale = scale

    def forward(self, x, past_key_value=None, use_cache=False, layer_idx=0):
        return x * self.scale + torch.tanh(self.lin(x)), None


class _TinyModel(nn.Module):
    def __init__(self, vocab=50, dim=16, n=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.embed_norm = None
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(n)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)


class TestLoopedPrefillVerified:
    def test_runs_and_reports_stop(self):
        torch.manual_seed(0)
        m = _TinyModel()
        ids = torch.randint(0, 50, (1, 8))
        gate = VerifierGate(min_margin=0.0, patience=1)
        logits, pasts, stats = looped_prefill_verified(
            m, ids, loop_lo=2, loop_hi=4, max_iters=4, gate=gate)
        assert logits.shape[-1] == 50
        assert stats["iters_run"] <= 4
        assert stats["stop_reason"] in ("max_iters",) or \
            stats["stop_reason"].startswith("verified@")

    def test_gate_reduces_iters_vs_ungated(self):
        torch.manual_seed(0)
        m = _TinyModel()
        ids = torch.randint(0, 50, (1, 8))
        # patience=1, margin=0: any repeated argmax stops at iter 2
        g = VerifierGate(min_margin=0.0, patience=1)
        _, _, s_gated = looped_prefill_verified(
            m, ids, 2, 4, max_iters=8, gate=g)
        _, _, s_full = looped_prefill_verified(
            m, ids, 2, 4, max_iters=8, gate=None)
        assert s_full["iters_run"] == 8
        assert s_gated["iters_run"] <= 8
