"""Teacher-channel tests - stubbed TeacherChannel (no HF download).

Contract under test:
- ``_tch_lift`` applies  w_rel * (log p_tch - log p_uni)  clipped at
  4*lift_clip, and abstains when the channel weight is ~0 or the
  teacher is unwarmed.
- Bulk ingest routes teacher PROBABILITIES into real_r - a stub that
  predicts the stream well must raise w_tch above init share.
- The lazy teacher object never enters a Flux snapshot.
"""

import numpy as np
import pytest
import torch

from forge.model.flux import FluxConfig, FluxLM


class _StubTeacher:
    """Minimal TeacherChannel stand-in: fixed per-token prob."""

    def __init__(self, vocab: int, p_hit: float = 0.9):
        self.vocab = vocab
        self.p_hit = p_hit
        self.pos = 0
        self._next = torch.zeros(vocab, dtype=torch.float32)

    def sync_to(self, count, ring, ep_start=0):
        self.pos = count

    def bulk_logprobs(self, ids, chunk=8192):
        self.pos += len(ids)
        return torch.full((len(ids),), self.p_hit, dtype=torch.float32)

    def logp_row(self):
        return torch.log_softmax(self._next, -1)

    def prob(self, x):
        return float(torch.softmax(self._next, -1)[x])


def _cfg(**kw):
    base = {
        "vocab_size": 64,
        "orders": (1, 2),
        "use_knn": False,
        "sparse_taps": (),
        "class_orders": (),
        "word_orders": (),
        "hybrid_taps": (),
        "teacher_channel": True,
        "teacher_bulk_rewards": True,
        "bulk_channel_rewards": True,
    }
    base.update(kw)
    return FluxConfig(**base)


def test_tch_in_channels():
    m = FluxLM(_cfg())
    assert "tch" in m._chix


def test_tch_lift_math():
    m = FluxLM(_cfg())
    m._tch_obj = _StubTeacher(m.vocab_size)
    m.ingest([1, 2, 3, 1, 2, 3, 1, 2], tag="t")
    i = m._chix["tch"]
    row = torch.full((m.vocab_size,), -5.0)
    lift = m._tch_lift(row, 0)
    assert lift is not None
    wt = float(m._w[i])
    lp = torch.log_softmax(torch.zeros(m.vocab_size), -1)
    # row-blend: lift = wt*(lp - row); row+lift lerps toward lp
    exp = wt * (lp - row)
    assert torch.allclose(lift, exp, atol=1e-6)
    assert torch.allclose(row + lift, row.lerp(lp, wt), atol=1e-6)


def test_tch_lift_abstains_on_zero_weight():
    m = FluxLM(_cfg())
    m._tch_obj = _StubTeacher(m.vocab_size)
    m._w[m._chix["tch"]] = 0.0
    row = torch.full((m.vocab_size,), -5.0)
    assert m._tch_lift(row, 0) is None


def test_tch_lift_yields_to_confident_cell():
    """Arbitration: _dconf_g=1 (verbatim context) zeroes the teacher
    lift so memory owns the position outright."""
    m = FluxLM(_cfg())
    m._tch_obj = _StubTeacher(m.vocab_size)
    m.ingest([1, 2, 3] * 5, tag="t")
    row = torch.full((m.vocab_size,), -5.0)
    m._dconf_g = torch.ones(1, dtype=torch.int64)
    assert m._tch_lift(row, 0) is None


def test_cpu_ingest_teacher_abstains():
    """Host-side ingest must NOT invoke the teacher (a HF forward per
    position is unusably slow on CPU; tch sits in _pidx's abstain
    set and its weight must not blow up)."""
    m = FluxLM(_cfg())
    stub = _StubTeacher(m.vocab_size, p_hit=0.9)
    m._tch_obj = stub
    w0 = float(m._w[m._chix["tch"]])
    m.ingest([1, 2, 3, 4] * 40, tag="t")
    assert stub.pos == 0
    assert float(m._w[m._chix["tch"]]) <= w0 + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_bulk_rewards_use_teacher_probs():
    """GPU bulk ingest: a teacher predicting p=0.9 on every observed
    token must gain weight share (rewards are probs, not logs -
    log-probs would go negative and decay the channel instead)."""
    m = FluxLM(_cfg(device="cuda", cuda_primary=True))
    stub = _StubTeacher(m.vocab_size, p_hit=0.9)
    m._tch_obj = stub
    w0 = float(m._w[m._chix["tch"]])
    rng = np.random.RandomState(0)
    ids = rng.randint(0, m.vocab_size, size=2000).tolist()
    m.ingest(ids, tag="t")
    assert stub.pos == 2000
    assert float(m._w[m._chix["tch"]]) > w0


def test_truncate_keeps_tail_not_prefix():
    """Regression: DynamicCache.crop() keeps the FIRST tokens - the
    window must be sliced off the tail (prefix eviction), else the
    cache = oldest context + new tokens = garbage (~4 nats lost)."""
    from forge.model.flux.teacher import TeacherChannel

    class _Layer:
        def __init__(self, L):
            self.keys = torch.arange(L).reshape(1, 1, L, 1).float()
            self.values = torch.arange(L).reshape(1, 1, L, 1).float()

        def get_seq_length(self):
            return self.keys.shape[2]

    class _Past:
        layers = None

    t = TeacherChannel.__new__(TeacherChannel)
    t.window = 8
    t.sink = 0
    t._has_sink = False  # mid-stream cache: no pinned sinks
    t.past = _Past()
    t.past.layers = [_Layer(20)]
    TeacherChannel._truncate(t)
    k = t.past.layers[0].keys.flatten().tolist()
    assert k == list(range(12, 20))

    # with a live sink prefix: first `sink` positions pinned
    t2 = TeacherChannel.__new__(TeacherChannel)
    t2.window = 8
    t2.sink = 3
    t2._has_sink = True
    t2.past = _Past()
    t2.past.layers = [_Layer(20)]
    TeacherChannel._truncate(t2)
    k2 = t2.past.layers[0].keys.flatten().tolist()
    assert k2 == [0, 1, 2, *list(range(15, 20))]


def test_teacher_w0_is_persistent_floor():
    """teacher_w0 is a guaranteed share, not just an init value.  On a
    stream memory memorizes perfectly the hedge would drive tch to ~0
    (the 15M collapse: w_tch hit 0.001 -> mixture degenerated to a
    bigram model).  The floor must keep it at ~w0 on every update
    path."""
    m = FluxLM(_cfg(teacher_w0=0.6))
    stub = _StubTeacher(m.vocab_size, p_hit=0.9)
    m._tch_obj = stub
    ti = m._chix["tch"]
    m.ingest([i % 7 for i in range(2000)], tag="t")
    assert float(m._w[ti]) >= 0.59
    assert abs(float(m._w.sum()) - 1.0) < 1e-5
    # above floor -> untouched
    m._w[ti] = 0.9
    m._teacher_floor_(m._w)
    assert abs(float(m._w[ti]) - 0.9) < 1e-5
    # floor=0 -> no enforcement (legacy behaviour)
    m._w[ti] = 0.01
    m2 = FluxLM(_cfg(teacher_w0=0.0))
    m2._teacher_floor_(m._w)
    assert float(m._w[ti]) == 0.01


def test_teacher_excluded_from_snapshot():
    m = FluxLM(_cfg())
    m._tch_obj = _StubTeacher(m.vocab_size)
    m.ingest([1, 2, 3] * 10, tag="t")
    st = m._snapshot_state()
    assert "_tch_obj" not in st
    assert not any("teacher" in k and "teacher_channel" not in k for k in st)
    bad = [k for k, v in st.items() if isinstance(v, torch.nn.Module)]
    assert not bad
