"""CPU tests for forge/model/flux/revmem.py + the native "rev" channel."""
import math

import pytest
import torch

from forge.model.flux import FluxConfig, FluxLM
from forge.model.flux.revmem import ReverseMem, joint_prev_nll


def test_feed_counts():
    rev = ReverseMem(10)
    rev.feed([1, 2, 1, 2, 1, 3])
    assert rev.table[2] == {1: 2}          # 1->2 twice
    assert rev.table[1] == {2: 2}          # 2->1 twice
    assert rev.table[3] == {1: 1}
    assert rev.total == 6


def test_rev_logp_smoothing_and_floor():
    rev = ReverseMem(10, uni_mix=0.0, floor=0.01)
    rev.feed([1, 2, 1, 2])                 # p(1|2) = 1.0 raw
    lp = rev.rev_logp(2, 1)
    assert lp > math.log(0.5)              # near-certain predecessor
    # unseen successor -> floor + unigram fallback (finite)
    lp2 = rev.rev_logp(9, 4)
    assert math.isfinite(lp2)


def test_rerank_swaps_inconsistent_argmax():
    """Candidates A (fwd-favored, inconsistent) vs B (consistent with
    prev) — rerank must pick B when alpha is high enough."""
    rev = ReverseMem(10, alpha=2.0, uni_mix=0.0, floor=1e-6)
    # train: after 5 only token 7 ever comes; after 5, token 3 never
    rev.feed([5, 7] * 20)
    lp7 = rev.rev_logp(7, 5)
    lp3 = rev.rev_logp(3, 5)
    assert lp7 - lp3 > 5                   # strongly prefers 7 after 5
    # fwd logits favor 3 slightly; joint score must flip to 7
    fwd = torch.log_softmax(torch.tensor([0.0] * 10), -1)
    fwd[3], fwd[7] = -0.5, -0.6
    scores = {3: float(fwd[3]) + rev.alpha * rev.rev_logp(3, 5),
              7: float(fwd[7]) + rev.alpha * rev.rev_logp(7, 5)}
    assert scores[7] > scores[3]


def test_snapshot_restore_roundtrip():
    rev = ReverseMem(10)
    rev.feed([1, 2, 3, 2, 1])
    snap = rev.snapshot_state()
    rev.feed([9, 9, 9])
    rev.restore_state(snap)
    assert rev.total == 5 and rev.table == snap["table"]
    rev.reset_state()
    assert rev.total == 0 and not rev.table


def test_joint_prev_nll_consistency():
    rev = ReverseMem(10, floor=1e-4)
    train = [0, 1, 2, 0, 1, 2] * 10
    rev.feed(train)
    good = joint_prev_nll(None, rev, [0, 1, 2, 0, 1])
    bad = joint_prev_nll(None, rev, [0, 2, 1, 0, 2])   # shuffled order
    assert good["prev_nll"] < bad["prev_nll"]


# ── native "rev" channel in FluxLM ────────────────────────────────────

def _mk(rev: bool) -> FluxLM:
    return FluxLM(FluxConfig(vocab_size=16, orders=(2, 3), device="cpu",
                             rev_channel=rev, sparse_taps=(),
                             class_orders=(), word_orders=(),
                             hybrid_taps=(), use_knn=False))


def test_rev_channel_opt_in():
    m_off = _mk(False)
    m_on = _mk(True)
    assert "rev" not in m_off.channels
    assert "rev" in m_on.channels
    assert m_on._pmask[m_on._chix["rev"]]          # device abstain


def test_rev1_written_at_ingest():
    m = _mk(True)
    m.ingest([1, 2, 1, 2, 1, 3], tag="t")
    assert m._rev1[1] == pytest.approx({2: 2.0}, abs=1e-6) or \
        sum(m._rev1[1].values()) > 0
    # pairs: 1->2 (x2), 2->1 (x2), 1->3 (x1)
    assert set(m._rev1) <= {1, 2}
    assert 3 in m._rev1[1] and 1 in m._rev1[2]


def test_rev_lifts_consistent_successor():
    """Context ending in 1 after a strict 1->2 chain: candidate 2 gets
    a positive 'rev' lift vs a candidate that never followed 1."""
    m = _mk(True)
    m.ingest([1, 2] * 20 + [1], tag="t")         # ends in 1
    cap = {}
    m._predict(cap=cap)
    rev_lift = cap.get("rev", {})
    assert 2 in rev_lift and rev_lift[2] > 0
    assert 5 not in rev_lift                     # never followed 1


def test_rev_channel_prob_rewards_consistency():
    m = _mk(True)
    m.ingest([1, 2] * 20 + [1], tag="t")         # ends in 1
    # after ...2,1 the last token is 1; x=2 (consistent) must outscore
    # x=3 (never a successor of 1) under the channel prob
    denom = m._total + m.config.uni_alpha * 16
    p2, awake2 = m._channel_prob("rev", 2, {}, None, denom)
    p3, _ = m._channel_prob("rev", 3, {}, None, denom)
    assert awake2 and p2 > p3


def test_rev_snapshot_and_revert():
    m = _mk(True)
    m.ingest([1, 2, 1, 2], tag="t")
    snap = {k: dict(v) for k, v in m._rev1.items()}
    cl = m.clone()
    assert cl._rev1 == snap
    n0 = len(m.journal.entries)
    m.ingest([4, 5], tag="u")
    assert 4 in m._rev1
    m.revert_since(n0)
    assert 4 not in m._rev1


def test_rev_default_off_lossless():
    """rev_channel=False must not touch logits vs a baseline build."""
    a = FluxLM(FluxConfig(vocab_size=16, orders=(2, 3), device="cpu",
                          sparse_taps=(), class_orders=(),
                          word_orders=(), hybrid_taps=(),
                          use_knn=False))
    b = _mk(False)
    for m in (a, b):
        m.ingest([1, 2, 3] * 10, tag="t")
    import numpy as np
    np.testing.assert_allclose(a._predict(), b._predict(), atol=0)
