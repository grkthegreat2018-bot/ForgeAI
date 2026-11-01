"""Unit tests for forge/evolution/auto_targets — the @self_improve
harvester that turns opted-in production functions into target specs.

Covers: decorator registration, spec generation (seed source lifted
live, deps/globals/imports spliced, _gen_cases rename), the generated
spec passing its own test/bench oracle, mutation safety, and prune.
CPU-only.
"""

import importlib.util
import textwrap

from forge.evolution import auto_targets


def _load_spec(path):
    spec = importlib.util.spec_from_file_location("_auto_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_decorator_registers():
    def gen():
        return [(1,), (2,)]

    @auto_targets.self_improve(gen_input=gen, name="ut_dummy")
    def _dummy(x):
        return x * 2

    reg = auto_targets.registered()
    assert "ut_dummy" in reg
    assert reg["ut_dummy"]["fn"] is _dummy
    auto_targets._registry.pop("ut_dummy", None)


def test_collect_writes_spec(tmp_path):
    mod_src = textwrap.dedent("""
        from forge.evolution.auto_targets import self_improve


        def _evolve_inputs_ut():
            return [(3,), ("ab",)]


        @self_improve(gen_input=_evolve_inputs_ut, name="ut_emit",
                      evals=42, tags=("test",))
        def _ut_emit(x):
            return x * 10
    """)
    f = tmp_path / "prod_mod.py"
    f.write_text(mod_src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("prod_mod", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    out = tmp_path / "targets"
    # _registry is process-global — other test files' @self_improve
    # imports leak in under a whole-suite run. Keep only this module's
    # registration (the decorator ran at exec_module above) so the emit
    # set is deterministic.
    saved = dict(auto_targets._registry)
    auto_targets._registry.clear()
    auto_targets._registry["ut_emit"] = saved["ut_emit"]
    try:
        emitted = auto_targets.collect_targets(out_dir=out, modules=())
    finally:
        auto_targets._registry.update(saved)
    assert emitted == {"auto_ut_emit": 42}
    p = out / "auto_ut_emit.py"
    assert p.exists()
    spec_mod = _load_spec(p)
    assert spec_mod.NAME == "auto_ut_emit"
    assert spec_mod.REQUIRED == ["_ut_emit"]
    ns = {}
    exec(spec_mod.SEED_CODE, ns)
    assert ns["_ut_emit"](7) == 70
    t = spec_mod.test(ns)
    assert t.get("ok"), t
    b = spec_mod.bench(ns)
    assert "ms_per_call" in b
    auto_targets._registry.pop("ut_emit", None)


def test_generated_specs_pass_own_gate():
    """Every harvested production target's seed must pass its own
    test+bench (the oracle contract — fake wins otherwise)."""
    emitted = auto_targets.collect_targets()
    assert len(emitted) >= 10, emitted
    from forge.evolution.code_evolve import Target

    for name in emitted:
        spec_mod = Target(Target.resolve(name)).mod
        ns = {}
        exec(spec_mod.SEED_CODE, ns)
        t = spec_mod.test(ns)
        ok = t.get("ok") if isinstance(t, dict) else bool(t)
        assert ok, f"{name}: {t}"
        b = spec_mod.bench(ns)
        assert spec_mod.METRIC in b, f"{name}: {b}"


def test_rejected_candidate_fails_gate(tmp_path):
    """A wrong-semantics candidate must NOT pass the oracle."""
    from forge.evolution.code_evolve import Target

    auto_targets.collect_targets()
    spec_mod = Target(Target.resolve("auto_ngram_lookup")).mod
    bad = "def _ngram_lookup(seq, ngram_size, draft_length):\n    return []"
    ns = {}
    exec(bad, ns)
    t = spec_mod.test(ns)
    ok = t.get("ok") if isinstance(t, dict) else bool(t)
    assert not ok
