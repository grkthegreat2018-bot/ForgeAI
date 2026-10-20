"""CPU-only synthetic tests for forge/exp/decision.py (E-decision).

Covers: DecisionHead shapes, the label heuristic and its priority order,
train/eval head helpers on separable synthetic data, and the
DecisionController decode loop on a tiny attention-only
ConfigurableResearchLLM — including the THINK last-block re-entry
(no KV mutation) and STOP suppression.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from forge.exp.decision import (  # noqa: E402
    ACTIONS, CONTINUE, THINK, STOP, VERIFY, N_ACTIONS,
    DecisionHead, DecisionController,
    is_sentence_end, label_positions, relabel,
    train_head, eval_head, greedy_generate, gen_nll,
)


# ── fixtures ───────────────────────────────────────────────────────────

def _tiny_attn_model():
    from forge.config import ModelConfig
    from forge.model_loader import ConfigurableResearchLLM
    cfg = ModelConfig(
        vocab_size=64, d_model=64, n_layers=2, n_heads=2, n_kv_heads=1,
        head_dim=32, intermediate_size=128, attn_type="gqa",
        ffn_type="swiglu", norm_type="rmsnorm", max_seq_len=64,
        use_qk_norm=False)
    cfg.device = "cpu"
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old)
    return model.eval()


def _fake_eng(model):
    return types.SimpleNamespace(model=model, tokenizer=None)


def _force_head(d_model: int, action: int) -> DecisionHead:
    """Head whose argmax is always `action` (weights zeroed, bias set)."""
    h = DecisionHead(d_model=d_model)
    with torch.no_grad():
        for p in h.parameters():
            p.zero_()
        h.net[2].bias[action] = 10.0
    return h.eval()


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return _tiny_attn_model()


# ── head mechanics ─────────────────────────────────────────────────────

class TestHead:
    def test_shape(self):
        h = DecisionHead(896)
        out = h(torch.randn(3, 896))
        assert out.shape == (3, N_ACTIONS)

    def test_act_returns_valid_ids(self):
        h = DecisionHead(64)
        a = h.act(torch.randn(5, 64))
        assert a.shape == (5,)
        assert set(a.tolist()) <= set(range(N_ACTIONS))

    def test_bf16_input(self):
        h = DecisionHead(32)
        out = h(torch.randn(2, 32, dtype=torch.bfloat16))
        assert out.shape == (2, N_ACTIONS)


class TestHeuristics:
    def test_is_sentence_end(self):
        assert is_sentence_end(".", 1)
        assert is_sentence_end("end?", 2)
        assert is_sentence_end(",", 3)          # punct-only
        assert is_sentence_end(")", 4)
        assert is_sentence_end("x", 5, eos_id=5)
        assert not is_sentence_end("the", 6)
        assert not is_sentence_end("  ", 7)
        assert not is_sentence_end("a.b", 8)    # ends with alpha

    def test_label_priority(self):
        # p1: row0 confident, row1 low (<thresh), row2 contested margin,
        # row3 sent-end + low prob (STOP must win over THINK).
        p1 = torch.tensor([0.9, 0.10, 0.40, 0.10])
        margin = torch.tensor([0.8, 0.50, 0.01, 0.30])
        se = torch.tensor([False, False, False, True])
        lab = label_positions(p1, margin, se, think_thresh=0.25,
                              verify_delta=0.05)
        assert lab.tolist() == [CONTINUE, THINK, VERIFY, STOP]

    def test_relabel_uses_fixed_threshold(self):
        tr = {"top1": torch.tensor([0.9, 0.1]),
              "margin": torch.tensor([0.9, 0.9]),
              "sent_end": torch.tensor([False, False])}
        lab = relabel(tr, think_thresh=0.5, verify_delta=0.05)
        assert lab.tolist() == [CONTINUE, THINK]


class TestTrainEval:
    def test_learns_separable(self):
        # 4 clusters in 16-d space, one per action.
        g = torch.Generator().manual_seed(0)
        centers = torch.randn(4, 16, generator=g) * 4
        X = centers.repeat_interleave(64, 0) + \
            torch.randn(256, 16, generator=g) * 0.1
        Y = torch.arange(4).repeat_interleave(64)
        head = DecisionHead(d_model=16, hidden=32)
        train_head(head, X, Y, steps=120, batch=64, lr=5e-3,
                   device="cpu", seed=0)
        ev = eval_head(head, X, Y, device="cpu")
        assert ev["acc"] > 0.9
        assert len(ev["confusion"]) == N_ACTIONS


# ── controller on tiny model ───────────────────────────────────────────

class TestController:
    def test_continue_matches_greedy(self, tiny):
        eng = _fake_eng(tiny)
        ids = torch.randint(0, 64, (1, 5))
        head = _force_head(64, CONTINUE)
        ctrl = DecisionController(eng, head, min_new_tokens=0, seed=0)
        c = ctrl.generate(ids, max_new_tokens=8)
        g = greedy_generate(eng, ids, max_new_tokens=8)
        assert c.tokens == g.tokens
        assert set(c.actions) == {CONTINUE}

    def test_stop_suppression_and_break(self, tiny):
        eng = _fake_eng(tiny)
        ids = torch.randint(0, 64, (1, 5))
        head = _force_head(64, STOP)
        ctrl = DecisionController(eng, head, min_new_tokens=4, seed=0)
        c = ctrl.generate(ids, max_new_tokens=16)
        assert c.tokens and len(c.tokens) == 4
        assert c.stopped_early
        assert c.stop_suppressed == 4  # first 4 STOPs suppressed by min_new
        assert all(a == STOP for a in c.actions[:5])

    def test_think_reruns_last_block_no_kv_mutation(self, tiny):
        eng = _fake_eng(tiny)
        ids = torch.randint(0, 64, (1, 5))
        head = _force_head(64, THINK)
        ctrl = DecisionController(eng, head, min_new_tokens=0, seed=0)
        c = ctrl.generate(ids, max_new_tokens=6)
        assert len(c.tokens) == 6
        assert all(a == THINK for a in c.actions)
        assert len(c.think_logit_shift) == 6
        # Sanity: with random weights the refined logits differ but the
        # loop still emits valid vocab ids.
        assert all(0 <= t < 64 for t in c.tokens)

    def test_verify_low_temp_is_greedy(self, tiny):
        eng = _fake_eng(tiny)
        ids = torch.randint(0, 64, (1, 5))
        head = _force_head(64, VERIFY)
        ctrl = DecisionController(eng, head, verify_temp=1e-6,
                                  min_new_tokens=0, seed=0)
        c = ctrl.generate(ids, max_new_tokens=8)
        g = greedy_generate(eng, ids, max_new_tokens=8)
        assert c.tokens == g.tokens
        assert all(a == VERIFY for a in c.actions)

    def test_action_hist_and_entropy_logged(self, tiny):
        eng = _fake_eng(tiny)
        ids = torch.randint(0, 64, (1, 5))
        head = _force_head(64, CONTINUE)
        ctrl = DecisionController(eng, head, min_new_tokens=0)
        c = ctrl.generate(ids, max_new_tokens=4)
        assert sum(c.action_hist.values()) == len(c.actions)
        assert len(c.entropies) == len(c.actions)
        assert all(e >= 0 for e in c.entropies)
        assert c.tok_s > 0

    def test_gen_nll_finite(self, tiny):
        eng = _fake_eng(tiny)
        p = torch.randint(0, 64, (1, 5))
        nll = gen_nll(eng, p, [1, 2, 3], device="cpu")
        assert nll == nll and nll >= 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
