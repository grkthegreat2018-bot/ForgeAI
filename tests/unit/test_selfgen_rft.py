"""Selfgen (codebase-grounded) RFT helpers — function harvest eligibility,
fuzz inputs, output comparison, novelty gating, Goldilocks filter, and
the sandbox test harness."""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "gen_sft3_data", REPO / "scripts" / "gen_sft3_data.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

SAMPLE = '''"""Sample module."""
import math


def two_smallest_sum(xs: list[int]) -> int:
    """Return the sum of the two smallest distinct values in a list."""
    u = sorted(set(xs))
    if len(u) < 2:
        return u[0]
    return u[0] + u[1]


def floor_mean(xs: list[float]) -> float:
    """Return the floor of the arithmetic mean of the values."""
    return math.floor(sum(xs) / max(1, len(xs)))


def _private(x: int) -> int:
    """Private functions are not eligible tasks."""
    return x


def uses_random(x: int) -> int:
    """Non-deterministic functions are not eligible."""
    import random
    return random.randint(0, x)


def uses_module_global(x: int) -> int:
    """Module-level dependencies are not eligible."""
    return x + GLOBAL_CONST


def no_docstring(x: int) -> int:
    y = x * 2
    return y


def too_many_params(a: int, b: int, c: int, d: int, e: int) -> int:
    """Five params is over the fuzzing budget."""
    return a + b + c + d + e


def unannotated(x) -> int:
    y = x * 2
    return y
'''


@pytest.fixture
def sample_tree(tmp_path):
    (tmp_path / "sample.py").write_text(SAMPLE, encoding="utf-8")
    return tmp_path


def test_harvest_eligibility(sample_tree):
    tasks = G.harvest_functions(roots=(str(sample_tree),))
    names = {t["name"] for t in tasks}
    # Documented + undocumented/self-contained qualify (the latter get a
    # describe-first pass); random/global/5-param/private are rejected.
    assert names == {"two_smallest_sum", "floor_mean", "no_docstring",
                     "unannotated"}
    t = next(t for t in tasks if t["name"] == "two_smallest_sum")
    assert t["types"] == ["list[int]"]
    assert "two smallest distinct" in t["desc"]
    assert t["signature"] == "(xs: list[int])"
    # Unannotated params get name-heuristic types (validated by execution).
    u = next(t for t in tasks if t["name"] == "unannotated")
    assert u["types"] == ["float"]  # "x" maps to float by the name rules
    assert u["desc"] == ""  # describe-first pass fills this in


def test_valid_description():
    assert G.valid_description("This function returns the sum of two "
                               "smallest distinct values.") == \
        "Returns the sum of two smallest distinct values."
    assert G.valid_description("short") == ""
    assert G.valid_description("```python\ndef x(): pass\n```") == ""
    assert G.valid_description("Compute the moving average over a window "
                               "of k items, dropping partial windows.")


def test_fuzz_inputs_literals():
    import ast
    tests = G.fuzz_inputs(["list[int]", "int"], n=4)
    assert len(tests) == 4
    for a in tests:
        ast.literal_eval(f"({a},)")  # every arg string is a valid literal
    assert tests[0] == "[], 0"
    assert tests[2] == "[1, 2, 3], -3"


def test_approx_eq_float_tolerance():
    assert G._approx_eq("0.30000000000000004", "0.3")
    assert not G._approx_eq("0.31", "0.3")
    assert G._approx_eq("[1, 2.0000000001, 'x']", "[1, 2.0, 'x']")
    assert not G._approx_eq("[1, 2]", "[1, 2, 3]")
    assert G._approx_eq("'abc'", "'abc'")


def test_novelty_gate():
    seen = G.novelty_ngrams("Return the sum of the two smallest distinct "
                            "values in a list of integers")
    assert not G.is_novel(
        "Return the sum of the two smallest distinct values in a list "
        "of integers", seen)
    assert G.is_novel(
        "Compute the longest alternating parity subsequence in an array", seen)
    assert not G.is_novel("", seen)


def test_goldilocks_filter():
    assert not G.goldilocks_ok(0.0)     # never solved: broken/too hard
    assert G.goldilocks_ok(0.25)
    assert G.goldilocks_ok(0.75)
    assert not G.goldilocks_ok(1.0)     # always solved: no signal


def test_sandbox_harness_executes_reference(sample_tree):
    """End-to-end: harvested references run in the real sandbox (stdlib
    imports injected), wrong solvers mismatch, custom names work."""
    from research.persistent_executor import PersistentCodeExecutor
    tasks = G.harvest_functions(roots=(str(sample_tree),))
    ex = PersistentCodeExecutor(timeout_s=15.0, memory_limit_mb=512)
    try:
        t = next(t for t in tasks if t["name"] == "two_smallest_sum")
        tests = G.fuzz_inputs(t["types"], n=4)
        outs, err = G.run_tests(ex, t["code"], tests, t["name"])
        assert err is None
        # [] crashes the reference (out-of-domain fuzz input) — per-input
        # isolation keeps the other inputs usable as ground truth.
        valid = [i for i, o in enumerate(outs)
                 if not o.startswith("__ERROR__")]
        assert len(valid) >= 2
        assert len({outs[i] for i in valid}) >= 2
        bad = "def two_smallest_sum(xs):\n    return 0"
        outs2, _ = G.run_tests(ex, bad, tests, t["name"])
        assert not all(G._approx_eq(outs2[i], outs[i]) for i in valid)
        # floor_mean uses math without importing it in the source segment —
        # the harness must inject stdlib imports.
        t2 = next(t for t in tasks if t["name"] == "floor_mean")
        outs3, err3 = G.run_tests(ex, t2["code"],
                                  G.fuzz_inputs(t2["types"], n=4), t2["name"])
        assert err3 is None
    finally:
        ex.close()


def test_solver_prompt_names_function(sample_tree):
    tasks = G.harvest_functions(roots=(str(sample_tree),))
    t = next(t for t in tasks if t["name"] == "two_smallest_sum")
    sp = G.codebase_solver_prompt(t)
    assert "`two_smallest_sum`" in sp
    assert "two smallest distinct" in sp
    assert "(xs: list[int])" in sp


def test_real_codebase_harvest_pool():
    """The production pool must be large and diverse (anti-fixed-pool)."""
    tasks = G.harvest_functions(limit=400)
    assert len(tasks) >= 20, f"pool too small: {len(tasks)}"
    # Undocumented tasks (desc="") get a describe-first pass at run time.
    assert all(t["desc"] == "" or len(t["desc"]) >= 15 for t in tasks)
    assert len({t["key"] for t in tasks}) == len(tasks)
