"""Tests for FluxLM domain lanes (domain-mixed order keys)."""

from forge.model.flux import FluxConfig, FluxLM


def _mk(**kw) -> FluxLM:
    kw.setdefault("vocab_size", 64)
    kw.setdefault("eos_token_id", -1)
    kw.setdefault("orders", (2, 3))
    kw.setdefault("device", "cpu")
    kw.setdefault("domain_route_every", 10**9)  # manual routing
    return FluxLM(FluxConfig(**kw))


def _seq(m, ids, reps=4):
    for _ in range(reps):
        for x in ids:
            m._learn_token(x)


def test_domain_lanes_isolate_cells():
    """Same context, two lanes: code-(a,b)->7 and prose-(a,b)->9 live
    in different cells of the SAME order table."""
    m = _mk(domains=("base", "code", "prose"))
    m._set_domain(1)
    _seq(m, [1, 2, 7])
    m._set_domain(2)
    _seq(m, [1, 2, 9])
    # land the ring on context (1,2) under each lane's read mask
    m._learn_token(1)
    m._learn_token(2)
    end = m._count - 1
    m._set_domain(1)
    k_code = m._keys_at(end)[2]  # lane-1 masked key
    m._set_domain(2)
    k_prose = m._keys_at(end)[2]
    assert k_code != k_prose
    denom = float(m._total) + 64
    p_code, aw_c = m._channel_prob("o2", 7, {2: k_code}, None, denom)
    p_prose, aw_p = m._channel_prob("o2", 9, {2: k_prose}, None, denom)
    p_cross, _ = m._channel_prob("o2", 9, {2: k_code}, None, denom)
    assert aw_c and aw_p
    assert p_code > 0.8 and p_prose > 0.8
    assert p_cross < p_prose  # lane isolation holds


def test_domain_default_off_is_identical():
    """domains=() -> mask is zero: keys identical to pre-domain."""
    m = _mk()
    _seq(m, [1, 2, 3])
    k = m._keys_at(m._count - 1)
    raw = m._suffix_key(2, m._count - 1)
    assert k[2] == raw  # no mask applied


def test_domain_router_follows_evidence():
    """The router EMA picks the lane whose cells keep hitting."""
    m = _mk(domains=("base", "code", "prose"), domain_route_every=2)
    m._set_domain(1)
    _seq(m, [1, 2, 7])
    m._set_domain(2)
    _seq(m, [5, 6, 9])
    # shared lane: feed a code-only 3-gram - router must find lane 1
    m._set_domain(0)
    m._learn_token(2)  # ctx (...,7,1,2)-ish ring
    m._learn_token(7)
    m._learn_token(1)
    m._learn_token(2)  # o3 ctx (7,1,2): code only
    assert m._dom_id == 1


def test_ingest_domain_param_and_tag():
    """ingest(domain=) and ingest(tag=<lane>) both select the lane."""
    m = _mk(domains=("base", "code", "prose"))
    m.ingest([1, 2, 7] * 3, tag="code")  # tag == lane name -> lane 1
    assert m._dom_id == 1
    m.ingest([5, 6, 9] * 3, domain="prose")
    assert m._dom_id == 2
    m.ingest([3, 4, 5] * 3)  # no domain -> stays lane 2
    assert m._dom_id == 2


def test_domain_masked_key_survives_spill():
    """Masked keys spilled to the bank still vote - the cold tier is
    transparent to domain lanes (pk IS the masked key)."""
    m = _mk(domains=("base", "code", "prose"), bank_enable=True, max_cells_per_order=200)
    m._set_domain(1)
    _seq(m, [1, 2, 7])
    _seq(m, list(range(10, 50)) * 3)  # pad -> >16 cells
    n = m._spill_order(2, len(m._tables[2]) - 16)
    assert n > 0
    m._learn_token(1)
    m._learn_token(2)
    end = m._count - 1
    k_code = m._keys_at(end)[2]
    if m._banks[2].cell(k_code) is not None:
        p, awake = m._channel_prob("o2", 7, {2: k_code}, None, float(m._total) + 64)
        assert awake and p > 0
    else:  # still hot -> hot path
        assert m._tables[2].get(k_code)


def test_laned_unigram_prior():
    """Prior dilution check: lane-d writes lift lane-d's prior for
    that token, not the shared pool's."""
    m = _mk(domains=("shared", "code", "prose"))
    m._set_domain(1)
    _seq(m, [1, 2, 7])
    m._set_domain(2)
    _seq(m, [5, 6, 9])
    p_lane = m._uni_p(9, float(m._total) + 64)
    # lane-2 prior for 9 >> lane-1's (9 never written under code)
    m._set_domain(1)
    p_code = m._uni_p(9, float(m._total) + 64)
    assert p_lane > p_code
    # shared lane reports the global table
    m._set_domain(0)
    den = float(m._total) + m.config.uni_alpha * 64
    p_shared = m._uni_p(9, den)
    assert p_shared == (m._uni[9] + m.config.uni_alpha) / den


def test_domain_snapshot_roundtrip(tmp_path):
    m = _mk(domains=("base", "code", "prose"))
    m._set_domain(1)
    _seq(m, [1, 2, 7])
    m._set_domain(2)
    _seq(m, [1, 2, 9])
    p = m.snapshot(tmp_path / "d.flux")
    m2 = FluxLM.load(p)
    # lane keys still resolve after load
    m2._set_domain(1)
    m2._learn_token(1)
    m2._learn_token(2)
    k_code = m2._keys_at(m2._count - 1)[2]
    cell = m2._tables[2].get(k_code)
    assert cell is not None and 7 in cell
