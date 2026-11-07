"""Selfgen (AZR) RFT helpers — proposal parsing, output comparison,
novelty gating, Goldilocks filter, and the sandbox test harness."""
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

GOOD = """# Task: Return the sum of the two smallest distinct values in a list.
```python
def solve(xs):
    u = sorted(set(xs))
    if len(u) < 2:
        return u[0]
    return u[0] + u[1]
# solve([3, 1, 1, 2])
# solve([5])
# solve([-1, -2, 0])
# solve([10, 9, 9, 8, 7])
```
"""


def test_parse_proposal_good():
    p = G.parse_proposal(GOOD)
    assert p is not None
    assert "two smallest distinct" in p["desc"]
    assert len(p["tests"]) == 4
    assert "def solve" in p["code"]


def test_parse_proposal_rejects_missing_task():
    assert G.parse_proposal("```python\ndef solve(x):\n    return x\n"
                            "# solve(1)\n# solve(2)\n# solve(3)\n```") is None


def test_parse_proposal_rejects_too_few_tests():
    raw = """# Task: Double the input integer value.
```python
def solve(x):
    return x * 2
# solve(1)
```
"""
    assert G.parse_proposal(raw) is None


def test_parse_proposal_rejects_nonliteral_args():
    raw = """# Task: Double the input integer value.
```python
def solve(x):
    return x * 2
# solve(1)
# solve(2)
# solve(some_variable)
# solve(4)
```
"""
    assert G.parse_proposal(raw) is None


def test_parse_proposal_rejects_dangerous_import():
    raw = """# Task: List files in the current directory.
```python
import os
def solve(x):
    return os.listdir(x)
# solve(".")
# solve("..")
# solve("a")
```
"""
    assert G.parse_proposal(raw) is None


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


def test_sandbox_harness_executes_reference():
    """End-to-end: reference implementation runs in the real sandbox and
    produces ground-truth repr outputs."""
    from research.persistent_executor import PersistentCodeExecutor
    ex = PersistentCodeExecutor(timeout_s=15.0, memory_limit_mb=512)
    try:
        p = G.parse_proposal(GOOD)
        outs, err = G.run_tests(ex, p["code"], p["tests"])
        assert err is None
        assert outs == ["3", "5", "-3", "15"]
        # A wrong solver must mismatch the ground truth.
        bad = "def solve(xs):\n    return 0"
        outs2, _ = G.run_tests(ex, bad, p["tests"])
        assert outs2 is not None
        assert not all(G._approx_eq(o, e) for o, e in zip(outs2, outs))
    finally:
        ex.close()


def test_solver_prompt_names_solve():
    p = G.parse_proposal(GOOD)
    sp = G.selfgen_solver_prompt(p["desc"], p["code"])
    assert "def solve(xs)" in sp
    assert "two smallest distinct" in sp
