"""Tests for the Scout lightweight discovery loop (forge/evolution/scout.py).

Uses the deterministic `synthetic` domain (Rastrigin landscape, fixed
optimum) so verification is exact — no GPU, sub-ms evals.
"""
from __future__ import annotations

import numpy as np
import pytest

from forge.evolution.domain_spec import load_spec
from forge.evolution.scout import (
    Scout,
    ScoutConfig,
    _apply_prior,
    _DomainState,
    _Welford,
)


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "scout_test.db")


def _scout(db_path, **kw):
    cfg = ScoutConfig(db_path=db_path, domains=["synthetic"],
                      verbose=False, **kw)
    return Scout(cfg)


def test_welford():
    w = _Welford()
    for v in [1.0, 2.0, 3.0, 4.0]:
        w.add(v)
    assert w.n == 4
    assert abs(w.mean - 2.5) < 1e-9
    assert abs(w.var - 5 / 3) < 1e-9  # sample variance of 1..4


def test_domain_state_propose_bounds(db_path):
    st = _DomainState("synthetic", "cpu")
    rng = np.random.default_rng(0)
    for _ in range(30):
        x = rng.random(st.dim)
        st.add(x, float(-np.sum(x ** 2)))
    X = st.propose(8, rng)
    assert X.shape == (8, st.dim)
    assert np.all((X >= 0.0) & (X <= 1.0))


def test_random_mode(db_path):
    st = _DomainState("synthetic", "cpu")
    rng = np.random.default_rng(0)
    X = st.propose(8, rng, mode="random")
    assert X.shape == (8, st.dim)


def test_apply_prior_biases_samples(db_path):
    rng = np.random.default_rng(0)
    pts = rng.random((64, 2))
    # prior says param 0 lives in the last bin with certainty
    dens = np.zeros(8); dens[-1] = 1.0
    out = _apply_prior(pts.copy(), {0: dens}, beta=1.0, rng=rng)
    assert np.all(out[:, 0] >= 7 / 8)
    # untouched dim stays uniform-ish
    assert out[:, 1].std() > 0.1


def test_run_finds_improvement(db_path):
    """Synthetic Rastrigin: a short run must beat a random-only baseline
    score and record at least one verified discovery."""
    sc = _scout(db_path, cycles=6, batch=16, verify_reps=2,
                patience=6, patience_dom=6)
    res = sc.run()
    st = sc.states["synthetic"]
    assert res["evals"] > 0
    assert np.isfinite(st.best_score)
    # Rastrigin upper bound ~ +66 at optimum; random draws land far lower.
    # A few TPE cycles should already beat a pure-random best materially.
    assert st.best_score > -80
    assert len(res["discoveries"]) >= 1
    # first discovery is baseline establishment (delta=0); at least one
    # later discovery must be a real improvement
    assert any(d["delta"] > 0 for d in res["discoveries"])


def test_stagnation_stops_early(db_path):
    """patience=3 with an impossible goal → loop exits via stagnation."""
    sc = _scout(db_path, cycles=50, batch=8, verify_reps=1,
                patience=3, patience_dom=99, require="raw_score>=1e9")
    res = sc.run()
    assert "stagnation" in res["stop_reason"]
    assert res["cycles"] < 50


def test_require_gate_blocks_promotion(db_path):
    """require='raw_score<=-1e9' can never be met → zero discoveries."""
    sc = _scout(db_path, cycles=4, batch=8, verify_reps=1,
                patience=4, require="raw_score<=-1e9")
    res = sc.run()
    assert len(res["discoveries"]) == 0


def test_cross_domain_stats_written(db_path):
    """synthetic is transferable=false → its evals must NOT pollute the
    global prior table (non-transferable domains don't leak)."""
    sc = _scout(db_path, cycles=3, batch=8, verify_reps=1)
    sc.run()
    n = sc.db.conn.execute("SELECT COUNT(*) FROM scout_param_stats").fetchone()[0]
    assert n == 0
    # but domain stats ARE written
    n = sc.db.conn.execute("SELECT COUNT(*) FROM scout_domain_stats").fetchone()[0]
    assert n == 1


def test_prior_used_for_transferable(db_path):
    """Feed param_stats for a shared name and check _prior_fns emits a
    density for a transferable domain param."""
    spec = load_spec("hadamard_kv")
    if spec is None:
        pytest.skip("hadamard_kv spec missing")
    cfg = ScoutConfig(db_path=db_path, domains=["hadamard_kv"],
                      verbose=False, cycles=1, batch=4)
    sc = Scout(cfg)
    st = sc.states.get("hadamard_kv")
    if st is None:
        pytest.skip("hadamard_kv not in light set")
    for b in range(8):
        sc._param_stats[("quant_bits", b)] = _Welford(
            10, 1.0 if b == 7 else -1.0, 0.1)
    fns = sc._prior_fns(st)
    assert fns, "expected prior density for shared param name"
    dens = next(iter(fns.values()))
    assert dens[-1] == dens.max()  # best bin favoured


def test_evaluations_persisted(db_path):
    sc = _scout(db_path, cycles=2, batch=8, verify_reps=1)
    sc.run()
    n = sc.db.conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    assert n == 16
    runs = sc.db.conn.execute(
        "SELECT total_evals FROM runs WHERE run_id=?", (sc.run_id,)).fetchone()
    assert runs[0] == 16
