"""Tests for the harness-genome spike (forge/evolution/harness_genome.py).

Pure CPU — dataclass mutation bounds, archive admission gating, and
serialization round-trips. No model, no GPU.
"""

from __future__ import annotations

import random
import types

import numpy as np
import pytest

from forge.evolution.harness_genome import (
    _BOUNDS,
    HarnessArchive,
    HarnessGenome,
    default_fitness,
)


# ── mutate() bounds + lineage ────────────────────────────────────────

@pytest.mark.parametrize("rng", [
    random.Random(0),
    np.random.default_rng(0),
], ids=["random.Random", "np.Generator"])
def test_mutate_stays_in_bounds(rng):
    g = HarnessGenome(
        temperature=1.29, top_p=0.99, top_k=250,
        max_gen_tokens=1500, think_budget=1500, n_insights=19,
        hint_idx=8, stop_suffixes=("\n#",),
        mut_op_w={"sort_reuse": 3.9})
    for _ in range(300):
        g = g.mutate(rng, p=0.9)          # high p: hit every field often
        for f, (lo, hi) in _BOUNDS.items():
            v = getattr(g, f)
            assert lo <= v <= hi, f"{f}={v} out of [{lo},{hi}]"
        assert isinstance(g.top_k, int)
        assert isinstance(g.max_gen_tokens, int)
        assert isinstance(g.think_budget, int)
        assert isinstance(g.n_insights, int)
        assert isinstance(g.hint_idx, int)
        assert len(g.stop_suffixes) <= 2
        for op, w in g.mut_op_w.items():
            assert 0.1 <= w <= 4.0, f"mut_op_w[{op}]={w}"


def test_mutate_lineage_and_isolation():
    rng = random.Random(1)
    parent = HarnessGenome(stop_suffixes=("\n#",), mut_op_w={"literal": 2.0})
    child = parent.mutate(rng, p=1.0)
    assert child.genome_id != parent.genome_id
    assert child.parents == (parent.genome_id,)
    assert child.gen == parent.gen + 1
    # containers are copies — mutating the child can't reach the parent
    child.mut_op_w["new_op"] = 9.9
    assert "new_op" not in parent.mut_op_w


def test_mutate_p0_is_identical_except_lineage():
    rng = random.Random(2)
    g = HarnessGenome(temperature=0.7, hint_idx=3)
    c = g.mutate(rng, p=0.0)
    for f in _BOUNDS:
        assert getattr(c, f) == getattr(g, f)
    assert c.stop_suffixes == g.stop_suffixes
    assert c.mut_op_w == g.mut_op_w


# ── apply seams ──────────────────────────────────────────────────────

def test_apply_cfg_sets_only_real_fields():
    cfg = types.SimpleNamespace(
        temperature=0.8, top_p=0.95, top_k=80, max_gen_tokens=640,
        think_budget=512, n_insights=5, unrelated="keep")
    g = HarnessGenome(temperature=1.1, top_k=40, n_insights=0)
    applied = set(g.apply_cfg(cfg))
    assert applied == {"temperature", "top_p", "top_k",
                       "max_gen_tokens", "think_budget", "n_insights"}
    assert cfg.temperature == 1.1
    assert cfg.top_k == 40
    assert cfg.n_insights == 0
    assert cfg.unrelated == "keep"


def test_apply_mutator_scales_base_weights():
    from forge.evolution.mutator import Mutator
    mut = Mutator(seed=0, db=None)
    base = dict(Mutator._BASE_W)
    g = HarnessGenome(mut_op_w={"sort_reuse": 2.0, "literal": 0.5})
    g.apply_mutator(mut)
    assert mut._BASE_W["sort_reuse"] == pytest.approx(
        base["sort_reuse"] * 2.0)
    assert mut._BASE_W["literal"] == pytest.approx(base["literal"] * 0.5)
    assert mut._BASE_W["crossover"] == pytest.approx(base["crossover"])
    assert Mutator._BASE_W == base          # class dict untouched


def test_apply_mutator_noop_without_weights():
    from forge.evolution.mutator import Mutator
    mut = Mutator(seed=0, db=None)
    HarnessGenome().apply_mutator(mut)      # empty mut_op_w
    assert mut._BASE_W == Mutator._BASE_W


# ── archive admission ────────────────────────────────────────────────

def test_archive_admission_gates(tmp_path):
    arch = HarnessArchive(tmp_path / "forge_evolve.harness.json", top_k=3)
    assert arch.best() is None
    # not full -> always admits
    for i, fit in enumerate([0.5, 0.9, 0.2]):
        assert arch.admit(HarnessGenome(temperature=0.5 + i * 0.1),
                          fit)
    assert len(arch.entries) == 3
    assert arch.min_fitness == pytest.approx(0.2)
    # full: at/below min rejected, above min evicts the weakest
    assert not arch.admit(HarnessGenome(), 0.2)
    assert not arch.admit(HarnessGenome(), 0.1)
    assert arch.admit(HarnessGenome(), 0.7)
    assert [e["fitness"] for e in arch.entries] == [0.9, 0.7, 0.5]
    assert arch.best().temperature == pytest.approx(0.6)


def test_archive_persistence_round_trip(tmp_path):
    p = tmp_path / "forge_evolve.harness.json"
    arch = HarnessArchive(p, top_k=4)
    g = HarnessGenome(temperature=1.05, hint_idx=2,
                      stop_suffixes=("\n#", "</code>"),
                      mut_op_w={"hoist_ctor": 1.5})
    arch.record(g, 0.77)
    arch2 = HarnessArchive(p, top_k=4)      # reload from disk
    assert len(arch2.entries) == 1
    e = arch2.entries[0]
    assert e["fitness"] == pytest.approx(0.77)
    rg = e["genome"]
    assert rg.genome_id == g.genome_id
    assert rg.temperature == pytest.approx(1.05)
    assert rg.hint_idx == 2
    assert rg.stop_suffixes == ("\n#", "</code>")
    assert rg.mut_op_w == {"hoist_ctor": 1.5}


def test_archive_missing_or_corrupt_file(tmp_path):
    arch = HarnessArchive(tmp_path / "nope.harness.json")
    assert arch.entries == []
    p = tmp_path / "bad.harness.json"
    p.write_text("{not json", encoding="utf-8")
    assert HarnessArchive(p).entries == []


def test_sample_defaults_and_lineage(tmp_path):
    rng = random.Random(3)
    arch = HarnessArchive(tmp_path / "a.harness.json", top_k=4)
    g0 = arch.sample(rng)                   # empty -> frozen default
    d0, ddef = g0.to_dict(), HarnessGenome().to_dict()
    d0.pop("genome_id"); ddef.pop("genome_id")   # fresh uuid each time
    assert d0 == ddef
    elite = HarnessGenome(temperature=0.9)
    arch.admit(elite, 0.9)
    arch.admit(HarnessGenome(temperature=0.4), 0.3)
    got = arch.sample(rng, explore_p=0.0)   # exploit -> the best elite
    assert got.genome_id == elite.genome_id
    for _ in range(20):                     # extend -> mutated children
        child = arch.sample(rng, explore_p=1.0)
        assert child.genome_id not in {e["genome"].genome_id
                                       for e in arch.entries}
        assert child.parents and child.gen >= 1


# ── serialization + fitness hook ─────────────────────────────────────

def test_genome_dict_round_trip():
    g = HarnessGenome(temperature=1.2, top_p=0.7, top_k=32,
                      max_gen_tokens=256, think_budget=0, n_insights=9,
                      hint_idx=4, stop_suffixes=("</code>",),
                      mut_op_w={"while_for": 0.5},
                      parents=("hg_deadbeef",), gen=3)
    r = HarnessGenome.from_dict(g.to_dict())
    assert r.to_dict() == g.to_dict()


def test_from_dict_clamps_corrupt_values():
    g = HarnessGenome.from_dict({"temperature": 99.0, "top_k": -5,
                                 "n_insights": 1000, "bogus_field": 1})
    assert g.temperature == 1.3
    assert g.top_k == 0
    assert g.n_insights == 20
    assert not hasattr(g, "bogus_field")


def test_default_fitness():
    assert default_fitness({"raws": 10, "gated": 8, "evals": 5,
                            "promos": 1}) == pytest.approx(0.8 + 0.2)
    assert default_fitness({}) == pytest.approx(0.0)


# ── CodeEvolve wiring (stubbed — no target, no evals) ────────────────

def test_code_evolve_harness_wiring(tmp_path):
    from forge.evolution.code_evolve import CodeEvolve, CodeEvolveConfig
    cfg = CodeEvolveConfig(
        target="dummy", db_path=str(tmp_path / "d.db"),
        harness_evolve=True, harness_evolve_every=4, verbose=False)
    ev = CodeEvolve.__new__(CodeEvolve)     # skip __init__/target load
    ev.cfg = cfg
    ev.rng = np.random.default_rng(0)
    ev._proposer = ev._basin_llm = ev._mutator = None
    ev._eval_n = ev._gen_n = ev._gen_ok = ev._promos = 0
    ev._harness_setup()
    assert ev._harness_archive is not None
    assert ev._genome is not None
    assert (tmp_path / "d.harness.json") == ev._harness_archive.path
    # window too small -> no admission yet
    ev._eval_n, ev._gen_n, ev._gen_ok = 3, 3, 2
    ev._harness_maybe()
    assert len(ev._harness_archive.entries) == 0
    # window elapsed -> scored + archived + resampled
    ev._eval_n, ev._gen_n, ev._gen_ok, ev._promos = 10, 10, 6, 1
    ev._harness_maybe()
    assert len(ev._harness_archive.entries) == 1
    assert ev._harness_archive.entries[0]["fitness"] == pytest.approx(
        0.6 + 0.1)
    assert 0.2 <= cfg.temperature <= 1.3   # genome re-applied to cfg


def test_code_evolve_harness_disabled_is_noop(tmp_path):
    from forge.evolution.code_evolve import CodeEvolve, CodeEvolveConfig
    cfg = CodeEvolveConfig(target="dummy",
                           db_path=str(tmp_path / "d.db"))
    ev = CodeEvolve.__new__(CodeEvolve)
    ev.cfg = cfg
    ev._harness_setup()
    assert ev._harness_archive is None
    assert ev._genome is None
    ev._harness_maybe(force=True)           # safe no-op
    assert not (tmp_path / "d.harness.json").exists()
