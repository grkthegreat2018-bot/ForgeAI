"""Integrity canaries for the eval pool (COMPETITIVE_AUDIT A13).

exec'd candidates share the worker's module state — a monkeypatched
math.sqrt/time.perf_counter would poison the gate+bench of later evals,
and a write to the target spec rewrites the gate future workers load.
Pure-python spec, single worker, CPU-only.
"""

from __future__ import annotations

import textwrap

SPEC = textwrap.dedent("""\
    NAME = "fast_sum"
    GOAL = "min"
    METRIC = "ms_per_call"
    TAGS = ["test"]
    BUDGETS = {}
    REQUIRED = ["fast_sum"]
    SEED_CODE = "def fast_sum(xs):\\n    return sum(xs)\\n"

    def test(ns):
        fn = ns.get("fast_sum")
        if not callable(fn):
            return {"ok": False, "detail": "no fast_sum"}
        if fn([1, 2, 3]) != 6:
            return {"ok": False, "detail": "wrong"}
        return {"ok": True}

    def bench(ns):
        import time
        fn = ns["fast_sum"]
        t0 = time.perf_counter()
        for _ in range(200):
            fn([1, 2, 3])
        return {"ms_per_call": (time.perf_counter() - t0)}
    """)

GOOD = "def fast_sum(xs):\n    return sum(xs)\n"


def test_canary_math_patch_poisons(tmp_path):
    """Monkeypatching math.sqrt must flag the patching job stage='poison'
    (worker exits); the respawned worker evals the follow-up honestly."""
    from forge.evolution.eval_pool import EvalPool

    p = tmp_path / "fast_sum.py"
    p.write_text(SPEC, encoding="utf-8")
    pool = EvalPool(str(p), n_workers=1, timeout_s=60)
    try:
        patcher = (
            "import math\n"
            "math.sqrt = lambda x: 0.0\n"
            "def fast_sum(xs):\n    return sum(xs)\n"
        )
        res = pool.eval_sync(patcher)
        assert not res["ok"] and res["stage"] == "poison"
        assert "math.sqrt" in res["detail"]
        good = pool.eval_sync(GOOD)
        assert good["ok"] and good["stage"] == "done"
    finally:
        pool.close()


def test_target_file_rewrite_flagged_and_restored(tmp_path):
    """A candidate writing the target spec mid-eval: the parent detects
    the sha drift on the ~1s poll tick, restores the init bytes, flags
    the in-flight job poison, and respawns the untrusted worker."""
    from forge.evolution.eval_pool import EvalPool

    p = tmp_path / "fast_sum.py"
    p.write_text(SPEC, encoding="utf-8")
    original = p.read_bytes()
    pool = EvalPool(str(p), n_workers=1, timeout_s=60)
    try:
        patcher = (
            "import pathlib, time\n"
            f"pathlib.Path({str(p)!r})"
            ".write_text('tampered', encoding='utf-8')\n"
            "time.sleep(4)\n"
            "def fast_sum(xs):\n    return sum(xs)\n"
        )
        res = pool.eval_sync(patcher, timeout_s=30)
        assert not res["ok"] and res["stage"] == "poison"
        assert p.read_bytes() == original
        good = pool.eval_sync(GOOD)
        assert good["ok"] and good["stage"] == "done"
    finally:
        pool.close()


def test_clean_evals_unaffected(tmp_path):
    """A clean two-job sequence still returns stage='done' — canary and
    integrity checks must not false-positive on honest code."""
    from forge.evolution.eval_pool import EvalPool

    p = tmp_path / "fast_sum.py"
    p.write_text(SPEC, encoding="utf-8")
    pool = EvalPool(str(p), n_workers=1, timeout_s=60)
    try:
        r1 = pool.eval_sync(GOOD)
        r2 = pool.eval_sync("def fast_sum(xs):\n    return sum(list(xs))\n")
        assert r1["ok"] and r1["stage"] == "done"
        assert r2["ok"] and r2["stage"] == "done"
    finally:
        pool.close()
