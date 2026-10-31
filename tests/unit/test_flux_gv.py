"""CPU tests for the FluxLM generalized-variant channel "gv".

The channel's contract: read-time-only generalization over the LIVE
order tables - same-class 1-edit suffix variants probed via rolling-
hash linearity.  It owns no state, so removed/changed weight
information can never persist through it.
"""
import numpy as np
import pytest

from forge.model.flux import FluxConfig, FluxLM

# byte classes: 'a','b','c' = class 2 (lower), 'A','B','C' = class 1
# (upper), '5' = class 0 (digit).  a=97 b=98 c=99 A=65 B=66 C=67 5=53


def _mk(**kw) -> FluxLM:
    kw.setdefault("orders", (2, 3))
    kw.setdefault("vocab_size", 256)
    kw.setdefault("device", "cpu")
    kw.setdefault("sparse_taps", ())
    kw.setdefault("class_orders", ())
    kw.setdefault("word_orders", ())
    kw.setdefault("hybrid_taps", ())
    kw.setdefault("use_knn", False)
    return FluxLM(FluxConfig(**kw))


def _ctx(m, ids):
    """Park the context at ``ids`` without learning (ring only)."""
    for t in ids:
        m._push(int(t))


def test_gv_opt_in():
    assert "gv" not in _mk().channels
    m = _mk(gen_variants=True)
    assert "gv" in m.channels
    assert m._pmask[m._chix["gv"]]              # device abstain


def test_gv_fires_on_same_class_variant():
    """(a,B)->c trained; context (a,C) - same class at the varied
    position - must surface c's votes."""
    m = _mk(gen_variants=True, gen_fix_tail=0, orders=(2,))
    m.ingest([97, 66, 99] * 10, tag="t")        # a B c
    _ctx(m, [97, 67])                            # a C (unseen pair)
    votes = m._gen_votes(m._count - 1, m._keys_at(m._count - 1))
    assert votes.get(99, 0.0) > 0               # c via (a,B) variant


def test_gv_sleeps_when_exact_cell_rich():
    m = _mk(gen_variants=True, gen_fix_tail=0, orders=(2,),
            gen_min_tot=2.0)
    m.ingest([97, 66, 99] * 10, tag="t")
    _ctx(m, [97, 66])                            # exact (a,B) seen 10x
    assert m._gen_votes(m._count - 1,
                        m._keys_at(m._count - 1)) == {}


def test_gv_no_persistence_after_delete():
    """The contract: delete the underlying cell and the generalized
    votes must vanish - nothing of the weight lives in the channel."""
    m = _mk(gen_variants=True, gen_fix_tail=0, orders=(2,))
    m.ingest([97, 66, 99] * 10, tag="t")
    _ctx(m, [97, 67])
    end = m._count - 1
    assert m._gen_votes(end, m._keys_at(end)).get(99, 0.0) > 0
    m._tables[2].clear()                        # all order-2 cells gone
    m._totals[2].clear()
    votes = m._gen_votes(end, m._keys_at(end))
    assert votes == {}                          # removed info is gone


def test_gv_deletion_via_revert():
    """Partial removal via journaled writes: reverting the writes
    shrinks variant votes by exactly that evidence."""
    m = _mk(gen_variants=True, gen_fix_tail=0, orders=(2,))
    for _i in range(10):                         # journaled writes
        for t in (97, 66, 99):
            m._learn_token(t)
    n0 = len(m.journal.entries)
    for _i in range(5):
        for t in (97, 66, 99):
            m._learn_token(t)
    _ctx(m, [97, 67])
    end = m._count - 1
    before = m._gen_votes(end, m._keys_at(end))[99]
    m.revert_since(n0)
    after = m._gen_votes(end, m._keys_at(end))[99]
    assert 0 < after < before                   # shrank, not vanished


def test_gv_snapshot_clone_deterministic():
    m = _mk(gen_variants=True, gen_fix_tail=0, orders=(2,))
    m.ingest([97, 66, 99] * 10 + [65, 98, 97] * 3, tag="t")
    _ctx(m, [97, 67])
    end = m._count - 1
    cl = m.clone()
    end_c = cl._count - 1
    assert cl._gen_votes(end_c, cl._keys_at(end_c)) == \
        m._gen_votes(end, m._keys_at(end))


def test_gv_channel_prob_semantics():
    m = _mk(gen_variants=True, gen_fix_tail=0, orders=(2,))
    m.ingest([97, 66, 99] * 10, tag="t")
    _ctx(m, [97, 67])
    denom = m._total + m.config.uni_alpha * 256
    p_c, awake = m._channel_prob("gv", 99, m._keys_at(m._count - 1),
                               None, denom)
    p_z, _ = m._channel_prob("gv", 90, m._keys_at(m._count - 1),
                             None, denom)
    assert awake and p_c > p_z


def test_gv_default_off_lossless():
    a = _mk()
    b = _mk(gen_variants=False)
    for m in (a, b):
        m.ingest([97, 66, 99] * 10, tag="t")
        _ctx(m, [97, 67])
    np.testing.assert_allclose(a._predict(), b._predict(), atol=0)


def test_gv_affects_mixture_when_enabled():
    on = _mk(gen_variants=True, gen_fix_tail=0, orders=(2,))
    off = _mk()
    for m in (on, off):
        m.ingest([97, 66, 99] * 10, tag="t")
        _ctx(m, [97, 67])
    # gv votes 'c' => its logit must be higher with the channel on
    assert float(on._predict()[99]) > float(off._predict()[99])


def _mk_cuda(**kw) -> FluxLM:
    kw["device"] = "cuda"
    kw["cuda_primary"] = True
    return _mk(**kw)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(),
                    reason="no cuda")
def test_gv_cuda_lift_and_no_persistence():
    """Device path: variant votes read _gt directly - lift appears on
    unseen same-class contexts and vanishes when the source cells do."""
    on = _mk_cuda(gen_variants=True, gen_fix_tail=0, orders=(2,))
    off = _mk_cuda(orders=(2,))
    for m in (on, off):
        m.ingest([97, 66, 99] * 20, tag="t")
        _ctx(m, [97, 67])
    lg_on = on._predict().cpu()
    lg_off = off._predict().cpu()
    assert float(lg_on[99]) > float(lg_off[99])      # lift landed
    # persistence contract on device: clear the GPU order-2 row and
    # the generalized lift must die with it
    on._gt.ttoks[0, :] = -1    # orders=(2,) -> order-2 table is row 0
    on._gt.tcnt[0, :] = 0
    on._gt.ttot[0, :] = 0
    on._gt.tkeys[0, :] = 0
    lg2 = on._predict().cpu()
    assert abs(float(lg2[99]) - float(lg_off[99])) < 0.25
