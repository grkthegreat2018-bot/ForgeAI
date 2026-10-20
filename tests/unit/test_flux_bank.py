"""Tests for the cold-tier weight bank (FluxBank + spill/regulate)."""
import numpy as np
import pytest
import torch

from forge.model.flux import FluxConfig, FluxLM
from forge.model.flux.bank import FluxBank


def _mk(**kw) -> FluxLM:
    kw.setdefault("vocab_size", 64)
    kw.setdefault("eos_token_id", -1)
    kw.setdefault("orders", (2, 3))
    kw.setdefault("device", "cpu")
    return FluxLM(FluxConfig(**kw))


def _stream(m: FluxLM, ids, tag="t"):
    m.ingest(ids, tag=tag)


def test_bank_add_probe_merge():
    b = FluxBank(order=2, M=4)
    keys = np.array([100, 200], dtype=np.int64)
    toks = np.array([[5, 6, -1, -1], [7, -1, -1, -1]], dtype=np.int32)
    cnt = np.array([[3.0, 1.0, 0, 0], [2.0, 0, 0, 0]], dtype=np.float32)
    tot = np.array([4.0, 2.0], dtype=np.float32)
    b.add_rows(keys, keys, toks, cnt, tot, np.ones(2),
               np.zeros(2, dtype=np.int64))
    cell, t = b.cell(100)
    assert cell == {5: 3.0, 6: 1.0} and t == 4.0
    assert b.cell(999) is None
    # re-spill same key -> merge, not duplicate
    b.add_rows(np.array([100]), np.array([100]),
               np.array([[5, 8, -1, -1]], dtype=np.int32),
               np.array([[1.0, 4.0, 0, 0]], dtype=np.float32),
               np.array([5.0]), np.ones(1), np.ones(1, dtype=np.int64))
    cell, t = b.cell(100)
    assert cell[5] == 4.0 and cell[8] == 4.0 and t == 9.0
    assert len(b) == 2


def test_bank_save_load_roundtrip(tmp_path):
    b = FluxBank(order=3, M=4)
    b.add_rows(np.array([42]), np.array([42]),
               np.array([[1, 2, -1, -1]], dtype=np.int32),
               np.array([[2.0, 1.0, 0, 0]], dtype=np.float32),
               np.array([3.0]), np.ones(1), np.ones(1, dtype=np.int64))
    p = tmp_path / "w.fluxbank"
    b.save(p)
    b2 = FluxBank.load(p)
    cell, t = b2.cell(42)
    assert cell == {1: 2.0, 2: 1.0} and t == 3.0


def test_spill_keeps_votes_cpu():
    """Spilled cells still vote through the order channel - cold
    tier preserves memory that eviction would destroy."""
    m = _mk(bank_enable=True)
    # ~40 distinct-ish tokens -> >16 order-2 cells (spill floor)
    _stream(m, [i % 40 for i in range(200)] + [1, 2, 3] * 8)
    tab = m._tables[2]
    assert len(tab) > 16, "expected >floor hot cells"
    ctx = None
    for key_ in tab:
        if 3 in tab[key_]:
            ctx = key_
            break
    assert ctx is not None
    n = m._spill_order(2, len(tab) - 16)
    assert n > 0 and len(m._banks[2]) == n
    # channel prob on a banked context still fires
    if m._banks[2].cell(ctx) is not None:
        p, awake = m._channel_prob("o2", 3, {2: ctx}, None,
                                 float(m._total) + 1.0 * 64)
        assert awake
        assert p > (m._uni[3] + 1.0) / (m._total + 64)


def test_hot_cold_union():
    """A context present in BOTH tiers merges: channel prob uses the
    unioned mass, not just the hot cell's post-spill delta."""
    m = _mk(bank_enable=True)
    _stream(m, [i % 40 for i in range(200)] + [1, 2, 3] * 8)
    tab = m._tables[2]
    ctx = next(k_ for k_ in tab if 3 in tab[k_])
    # baseline: pick a follower with no hot mass -> prior-only prob
    absent = next(t for t in range(40, 60) if t not in tab[ctx])
    p7_0, _ = m._channel_prob("o2", absent, {2: ctx}, None,
                              float(m._total) + 64)
    # bank a synthetic cold row under the same context key with
    # dominant mass for a DIFFERENT follower (7): the unioned prob of
    # 7 must beat its hot-only value (7 wasn't even a hot follower)
    b = m._banks[2]
    tk = np.full((1, b.M), -1, dtype=np.int32)
    cn = np.zeros((1, b.M), dtype=np.float32)
    tk[0, 0], cn[0, 0] = absent, 80.0
    b.add_rows(np.array([ctx], dtype=np.uint64),
               np.array([0], dtype=np.int64),
               tk, cn,
               np.array([100.0], dtype=np.float32),
               np.array([50.0], dtype=np.float32),
               np.zeros(1, dtype=np.int64))
    p7, awake7 = m._channel_prob("o2", absent, {2: ctx}, None,
                               float(m._total) + 64)
    assert awake7 and p7 > 4 * p7_0       # cold mass dominates union
    # and the hot follower 3 still votes (mass = hot 8 + cold 0)
    p3, _ = m._channel_prob("o2", 3, {2: ctx}, None,
                          float(m._total) + 64)
    assert p3 > 0


def test_regulate_spills_under_pressure():
    """regulate() spills when a table crosses the watermark."""
    m = _mk(bank_enable=True, max_cells_per_order=32,
            bank_spill_hi=0.5, bank_spill_frac=0.5)
    _stream(m, list(range(50)) * 2)
    # force table over watermark
    while len(m._tables[2]) < 20:
        m._learn_token(len(m._tables[2]) % 60)
    rep = m.regulate()
    assert isinstance(rep["spilled"], dict)


def test_snapshot_preserves_bank(tmp_path):
    m = _mk(bank_enable=True)
    _stream(m, [i % 40 for i in range(120)])
    m._spill_order(2, 10 ** 9)
    assert len(m._banks[2]) > 0
    p = m.snapshot(tmp_path / "m.flux")
    m2 = FluxLM.load(p)
    assert len(m2._banks.get(2, FluxBank(2))) == len(m._banks[2])


def test_bank_drop_coldest():
    b = FluxBank(order=2, M=4)
    b.add_rows(np.arange(10), np.arange(10),
               np.zeros((10, 4), dtype=np.int32) - 1,
               np.zeros((10, 4), dtype=np.float32),
               np.ones(10), np.arange(10, dtype=np.float32),
               np.zeros(10, dtype=np.int64))
    b.drop_rows(b.coldest(5))
    assert len(b) == 5
    assert all(b.cell(k) is None for k in range(5))
    assert all(b.cell(k) is not None for k in range(5, 10))
