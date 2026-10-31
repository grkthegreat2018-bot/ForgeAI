"""R60: SpecBound adaptive bounded self-speculation + ExGRPO replay.

Part A — SpecBoundDecoding (forge/engine/decoding.py):
  Early-exit drafting via ``layer_indices`` partial forward + ONE full
  verify forward per round.  Per-exit-layer acceptance EMA picks the
  deepest confident exit; draft_len adapts within [min_draft, max_draft].
  Under greedy (temperature=0) the emitted sequence MUST equal
  StandardDecoding's — the verify step resamples at the first mismatch.

Part B — ExGRPO replay (forge/self_play):
  ExGRPOReplayBuffer stores rollout groups valued by
  correctness_flag * mean-token-entropy; the self-play loop mixes
  top-valued entries into each GRPO update batch (opt-in, flag-gated).
"""
import math
import os
import random
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from forge.engine.decoding import SpecBoundDecoding, StandardDecoding, build_decoding
from forge.self_play.grpo_trainer import ExGRPOReplayBuffer, GRPOConfig, GRPOTrainer
from forge.self_play.infinite_loop import InfiniteSelfPlayLoop, LoopConfig

# ═══════════════════════════════════════════════════════════════════════════
# Stubs
# ═══════════════════════════════════════════════════════════════════════════

class _BigramStub(nn.Module):
    """Deterministic next-token LM: logits[b, t, :] = TABLE[idx[b, t]].

    ``full_table`` is the "true" model; ``draft_table`` emulates the
    early-exit approximation and is returned whenever the forward gets a
    ``layer_indices`` kwarg (the SpecBound partial-forward path).  Only
    ``kv_layer`` carries a KV entry — the rest are None, mirroring a
    hybrid model where conv layers produce no presents.  The seq dim of
    the fake (k, v) tensors tracks the committed length so KV-rollback
    slicing works exactly like the real model.
    """

    def __init__(self, vocab=40, n_layers=4, kv_layer=2,
                 draft_mismatches=None):
        super().__init__()
        self.vocab = vocab
        self.n_layers = n_layers
        self.kv_layer = kv_layer
        self.blocks = [nn.Identity() for _ in range(n_layers)]
        self.eos_token_id = vocab + 999  # outside range → never emitted
        # full table: i -> 10 + ((i-10+3) % 20) for i in 10..29 — a
        # deterministic cycle that never touches the default EOS ids.
        t = torch.full((vocab, vocab), -10.0)
        for i in range(vocab):
            if 10 <= i < 30:
                nxt = 10 + ((i - 10 + 3) % 20)
            else:
                nxt = (i + 5) % vocab
                while nxt in (2, 7):
                    nxt = (nxt + 1) % vocab
            t[i, nxt] = 10.0  # unique argmax — no ties
        self.full_table = t
        d = t.clone()
        for i, nxt in (draft_mismatches or {}).items():
            d[i, :] = -10.0
            d[i, nxt] = 10.0
        self.draft_table = d

    def forward(self, idx, targets=None, past_key_values=None,
                use_cache=False, layer_indices=None,
                logits_last_only=False, **kw):
        table = (self.draft_table if layer_indices is not None
                 else self.full_table)
        logits = table[idx]
        if logits_last_only:
            logits = logits[:, -1:, :]
        if use_cache:
            prev = 0
            if past_key_values is not None:
                for e in past_key_values:
                    if e is not None:
                        prev = e[0].shape[-2]
                        break
            new_len = prev + idx.shape[1]
            presents = []
            for i in range(self.n_layers):
                if i == self.kv_layer:
                    presents.append((torch.zeros(idx.shape[0], 1,
                                                 new_len, 4),
                                     torch.zeros(idx.shape[0], 1,
                                                 new_len, 4)))
                else:
                    presents.append(None)
            return logits, None, presents
        return logits, None


def _build_tiny(vocab=64):
    """forgelm_tiny on CPU, fp32 — real hybrid conv+attention model."""
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM
    cfg = get_config("forgelm_tiny")
    cfg.vocab_size = vocab
    cfg.device = "cpu"
    cfg.dtype = "float32"
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("cpu"):
            model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# SpecBound — registry + construction
# ═══════════════════════════════════════════════════════════════════════════

def test_build_decoding_specbound():
    d = build_decoding("specbound")
    assert isinstance(d, SpecBoundDecoding)
    d2 = build_decoding("specbound", exit_layer=3, conf_threshold=0.5,
                        draft_len=6, min_draft=1, max_draft=9)
    assert d2.exit_layer == 3 and d2.conf_threshold == 0.5
    assert d2._cur_draft == 6 and d2.min_draft == 1 and d2.max_draft == 9


def test_specbound_draft_len_clamped_to_bounds():
    d = SpecBoundDecoding(draft_len=99, min_draft=2, max_draft=5)
    assert d._cur_draft == 5
    d = SpecBoundDecoding(draft_len=0, min_draft=2, max_draft=5)
    assert d._cur_draft == 2


# ═══════════════════════════════════════════════════════════════════════════
# SpecBound — greedy equivalence with StandardDecoding (THE guarantee)
# ═══════════════════════════════════════════════════════════════════════════

def test_specbound_greedy_equals_standard_stub():
    """Bit-exact greedy output on a deterministic stub, exercising the
    reject+correct path (draft table differs at token 19)."""
    stub = _BigramStub(vocab=40, draft_mismatches={19: 30})
    prompt = torch.tensor([[10]], dtype=torch.long)
    kw = {"max_new_tokens": 15, "temperature": 0.0, "top_p": 1.0,
              "top_k": 0, "repetition_penalty": 1.0}
    std = StandardDecoding().generate(stub, prompt, **kw)
    d = SpecBoundDecoding(draft_len=4)
    spec = d.generate(stub, prompt, **kw)
    assert torch.equal(std, spec)
    # Chain must have actually emitted tokens (not vacuously equal).
    assert std.shape[1] > prompt.shape[1]
    # The stub forced a draft mismatch at token 19 → verify rejected at
    # least once (acceptance stats recorded).
    assert d._total_drafts > 0


def test_specbound_greedy_equals_standard_stub_all_accept():
    """draft == full → every draft accepted; still identical output."""
    stub = _BigramStub(vocab=40)  # no mismatches
    prompt = torch.tensor([[10]], dtype=torch.long)
    kw = {"max_new_tokens": 12, "temperature": 0.0, "top_p": 1.0,
              "top_k": 0, "repetition_penalty": 1.0}
    std = StandardDecoding().generate(stub, prompt, **kw)
    d = SpecBoundDecoding(draft_len=3)
    spec = d.generate(stub, prompt, **kw)
    assert torch.equal(std, spec)
    assert d.acceptance_rate > 0.99


def test_specbound_greedy_equals_standard_tiny_model():
    """Real forgelm_tiny hybrid (conv+attention) — conv-state repair path
    keeps greedy output identical to standard decode.  exit_layer=1 pins
    the draft to layers 0-1 (conv,conv) so early-exit logits diverge
    from the full model → verify rejections exercise the recurrent-state
    snapshot/restore + committed-prefix replay on a real hybrid."""
    torch.manual_seed(0)
    model = _build_tiny(vocab=64)
    prompt = torch.tensor([[3, 5, 8]], dtype=torch.long)
    kw = {"max_new_tokens": 14, "temperature": 0.0, "top_p": 1.0,
              "top_k": 0, "repetition_penalty": 1.0}
    std = StandardDecoding().generate(model, prompt, **kw)
    spec = SpecBoundDecoding(exit_layer=1, draft_len=4)
    out = spec.generate(model, prompt, **kw)
    assert torch.equal(std, out)
    assert spec._total_drafts > 0


# ═══════════════════════════════════════════════════════════════════════════
# SpecBound — calibration + adaptive bound
# ═══════════════════════════════════════════════════════════════════════════

def test_specbound_layer_selection_by_ema():
    d = SpecBoundDecoding(conf_threshold=0.6)
    cands = [1, 2, 3]
    d._layer_ema = {1: 0.9, 2: 0.7, 3: 0.5}
    assert d._select_exit_layer(cands) == 2  # deepest >= threshold
    d._layer_ema[3] = 0.8
    assert d._select_exit_layer(cands) == 3
    d._layer_ema = {1: 0.1, 2: 0.2, 3: 0.3}
    assert d._select_exit_layer(cands) == 3  # none eligible → deepest
    # Fixed exit_layer override wins.
    d2 = SpecBoundDecoding(exit_layer=1, conf_threshold=0.6)
    assert d2._select_exit_layer([1, 2, 3]) == 1


def test_specbound_draft_bound_grows_on_full_accept():
    stub = _BigramStub(vocab=40)  # draft == full → all accepted
    d = SpecBoundDecoding(draft_len=2, min_draft=1, max_draft=4)
    d.generate(stub, torch.tensor([[10]]), max_new_tokens=8,
               temperature=0.0, top_k=0, top_p=1.0,
               repetition_penalty=1.0)
    assert d._cur_draft > 2
    assert d._cur_draft <= 4
    # Per-layer EMA recorded for the used exit layer.
    assert d._cur_exit in d._layer_ema
    assert 0.0 <= d._layer_ema[d._cur_exit] <= 1.0


def test_specbound_draft_bound_shrinks_on_reject():
    # Every draft proposes a wrong token → n_accepted == 0 each round.
    bad = {}
    for i in range(10, 30):
        bad[i] = 10 + ((i - 10 + 7) % 20)  # != full's +3 step
    stub = _BigramStub(vocab=40, draft_mismatches=bad)
    d = SpecBoundDecoding(draft_len=5, min_draft=1, max_draft=6)
    d.generate(stub, torch.tensor([[10]]), max_new_tokens=8,
               temperature=0.0, top_k=0, top_p=1.0,
               repetition_penalty=1.0)
    assert d._cur_draft < 5
    assert d._cur_draft >= 1
    assert d._total_drafts > 0
    assert d._total_accepted == 0


def test_specbound_sampling_path_runs():
    """temperature>0 smoke: runs, returns prompt + appended tokens."""
    stub = _BigramStub(vocab=40, draft_mismatches={19: 30})
    torch.manual_seed(0)
    out = SpecBoundDecoding(draft_len=3).generate(
        stub, torch.tensor([[10]]), max_new_tokens=8,
        temperature=0.8, top_p=1.0, top_k=0, repetition_penalty=1.0)
    assert out.shape[1] >= 1
    assert out[0, 0].item() == 10


# ═══════════════════════════════════════════════════════════════════════════
# ExGRPO — replay buffer
# ═══════════════════════════════════════════════════════════════════════════

def test_exgrpo_buffer_priority_ordering():
    """value = correctness_flag * entropy — top() returns desc order."""
    buf = ExGRPOReplayBuffer(capacity=8)
    buf.add("p1", ["a", "b"], [1.0, 0.0], entropy=0.5)   # value 0.5
    buf.add("p2", ["a", "b"], [1.0, 0.0], entropy=0.9)   # value 0.9
    buf.add("p3", ["a", "b"], [0.0, 0.0], entropy=9.9)   # incorrect → 0
    top = buf.top(2)
    assert [e["prompt"] for e in top] == ["p2", "p1"]
    assert top[0]["value"] == pytest.approx(0.9)
    assert top[0]["correctness"] == 1.0


def test_exgrpo_buffer_capacity_eviction():
    """Over capacity → lowest-value entry evicted; worse entries dropped."""
    buf = ExGRPOReplayBuffer(capacity=2)
    buf.add("low", ["a", "b"], [1.0, 0.0], entropy=0.1)
    buf.add("mid", ["a", "b"], [1.0, 0.0], entropy=0.5)
    buf.add("high", ["a", "b"], [1.0, 0.0], entropy=0.9)  # evicts "low"
    assert len(buf) == 2
    assert {e["prompt"] for e in buf.entries} == {"mid", "high"}
    # A new entry worse than the min is dropped.
    added = buf.add("worse", ["a", "b"], [0.0, 0.0], entropy=0.0)
    assert added is None and len(buf) == 2


def test_exgrpo_buffer_dedup_by_prompt():
    buf = ExGRPOReplayBuffer(capacity=4)
    buf.add("p", ["a", "b"], [1.0, 0.0], entropy=0.5)
    buf.add("p", ["a", "b"], [1.0, 0.0], entropy=0.2)  # worse → rejected
    assert len(buf) == 1
    assert buf.entries[0]["entropy"] == pytest.approx(0.5)
    buf.add("p", ["a", "b"], [1.0, 0.0], entropy=0.8)  # better → replaces
    assert len(buf) == 1
    assert buf.entries[0]["entropy"] == pytest.approx(0.8)
    assert buf.entries[0]["value"] == pytest.approx(0.8)


def test_exgrpo_sample_weighted_without_replacement():
    buf = ExGRPOReplayBuffer(capacity=8)
    buf.add("a", ["x", "y"], [1.0, 0.0], entropy=1.0)
    buf.add("b", ["x", "y"], [1.0, 0.0], entropy=3.0)
    picks = buf.sample(2, rng=random.Random(0))
    assert len(picks) == 2
    assert {e["prompt"] for e in picks} == {"a", "b"}
    # Value-weighted: 'b' should win the first slot most of the time.
    firsts = [buf.sample(1, rng=random.Random(s))[0]["prompt"]
              for s in range(50)]
    assert firsts.count("b") > firsts.count("a")
    # sample(0) and oversampling are safe.
    assert buf.sample(0, rng=random.Random(0)) == []
    assert len(buf.sample(10, rng=random.Random(0))) == 2


def test_exgrpo_flag_off_defaults():
    """Opt-in: every surface defaults OFF — zero behavior change."""
    assert LoopConfig().use_exgrpo is False
    assert LoopConfig().exgrpo_replay_size == 512
    assert LoopConfig().exgrpo_replay_fraction == pytest.approx(0.25)
    assert GRPOConfig().use_exgrpo is False
    # Fresh loop carries no buffer.
    loop = InfiniteSelfPlayLoop("ckpt", LoopConfig())
    assert loop._exgrpo is None
    # Invalid fraction is rejected by config validation.
    with pytest.raises(ValueError):
        GRPOConfig(exgrpo_replay_fraction=1.5)
    with pytest.raises(ValueError):
        GRPOConfig(exgrpo_replay_size=0)


# ═══════════════════════════════════════════════════════════════════════════
# ExGRPO — mean-token-entropy scoring
# ═══════════════════════════════════════════════════════════════════════════

class _CharTokenizer:
    """Minimal HF-style tokenizer stub for the entropy helper."""
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=None,
                 max_length=None, add_special_tokens=None):
        ids = [ord(c) % 32 for c in text] or [0]
        return SimpleNamespace(input_ids=torch.tensor([ids]))


class _FixedLogitModel(nn.Module):
    """Returns fixed logits: uniform → entropy ln(V); peaked → ~0."""

    def __init__(self, vocab=32, peaked=False):
        super().__init__()
        self.vocab = vocab
        self.peaked = peaked

    def forward(self, idx, **kw):
        B, T = idx.shape
        logits = torch.zeros(B, T, self.vocab)
        if self.peaked:
            logits[..., 3] = 20.0
        return logits, None


def test_exgrpo_group_entropy_values():
    tok = _CharTokenizer()
    m_uni = _FixedLogitModel(vocab=32, peaked=False)
    tr = GRPOTrainer(m_uni, tok, m_uni, device="cpu", config=GRPOConfig())
    ent = tr.exgrpo_group_entropy("ab", ["cd", "ef"])
    assert ent == pytest.approx(math.log(32), rel=1e-3)

    m_peak = _FixedLogitModel(vocab=32, peaked=True)
    tr2 = GRPOTrainer(m_peak, tok, m_peak, device="cpu",
                      config=GRPOConfig())
    ent2 = tr2.exgrpo_group_entropy("ab", ["cd"])
    assert ent2 < 0.01


def test_exgrpo_entropy_feeds_buffer_value():
    """End-to-end: entropy * correctness drives buffer ordering."""
    buf = ExGRPOReplayBuffer(capacity=8)
    buf.add("correct", ["a", "b"], [1.0, 0.0], entropy=2.0)  # value 2.0
    buf.add("wrong", ["a", "b"], [0.0, 0.0], entropy=5.0)    # value 0.0
    assert buf.top(1)[0]["prompt"] == "correct"
    assert buf.entries[1]["value"] == 0.0
