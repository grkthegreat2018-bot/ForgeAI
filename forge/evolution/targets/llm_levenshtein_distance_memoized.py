"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'levenshtein_distance_memoized'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['levenshtein_distance_memoized']
PROMPT = 'Implement the Levenshtein distance algorithm with memoization to avoid redundant computations. The function must correctly compute the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into another. Ensure the memoization cache is used to avoid recalculating subproblems, and the solution must be more efficient than the naive recursive approach.'

SEED_CODE = 'def levenshtein_distance_memoized(s1: str, s2: str) -> int:\n    if len(s1) < len(s2):\n        s1, s2 = s2, s1\n    if len(s2) == 0:\n        return len(s1)\n    previous_row = range(len(s2) + 1)\n    for i, c1 in enumerate(s1):\n        current_row = [i + 1]\n        for j, c2 in enumerate(s2):\n            insertions = previous_row[j + 1] + 1\n            deletions = current_row[j] + 1\n            substitutions = previous_row[j] + (c1 != c2)\n            current_row.append(min(insertions, deletions, substitutions))\n        previous_row = current_row\n    return previous_row[-1]'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

def _gen_cases(seed):
    import random
    random.seed(seed)
    cases = [
        ("kitten", "sitting"),
        ("", ""),
        ("abc", "def"),
        ("longerstring", "short"),
        ("python", "java"),
        ("optimization", "optimisation"),
        ("memoization", "memorization"),
        ("algorithm", "algorithmic"),
        ("benchmark", "benchmarking"),
        ("recursion", "iteration"),
        ("cache", "cash"),
        ("levenshtein", "levenshtein"),
        ("distance", "difference"),
        ("computation", "calculation"),
        ("subproblem", "subtask")
    ]
    return cases


def _ref():
    ns: dict = {}
    exec(SEED_CODE, ns)
    return ns[REQUIRED[0]]


def _same(a, b, atol=1e-5, rtol=1e-4):
    import math as _m
    ta, tb = torch.is_tensor(a), torch.is_tensor(b)
    if ta or tb:
        if not (ta and tb) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        if torch.equal(a, b):
            return True
        if a.is_complex() or b.is_complex():
            fa = torch.isfinite(a.real) & torch.isfinite(a.imag)
            fb = torch.isfinite(b.real) & torch.isfinite(b.imag)
        else:
            fa, fb = torch.isfinite(a), torch.isfinite(b)
        if not torch.equal(fa, fb):
            return False
        return bool(torch.allclose(a[fa], b[fb], atol=atol, rtol=rtol))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _same(x, y, atol, rtol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _same(a[k], b[k], atol, rtol) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return False
    if isinstance(a, dict) or isinstance(b, dict):
        return False
    if isinstance(a, float) or isinstance(b, float):
        try:
            return _m.isclose(float(a), float(b),
                              abs_tol=atol, rel_tol=rtol)
        except (TypeError, ValueError):
            return a == b
    return a == b


def _call(fn, case):
    return fn(*case) if isinstance(case, tuple) else fn(case)


def _any_cuda(obj, depth=0):
    if depth > 4:
        return False
    if torch.is_tensor(obj):
        return bool(obj.is_cuda)
    if isinstance(obj, (list, tuple)):
        return any(_any_cuda(x, depth + 1) for x in obj)
    if isinstance(obj, dict):
        return any(_any_cuda(x, depth + 1) for x in obj.values())
    return False


def test(ns):
    fn = ns.get(REQUIRED[0])
    if not callable(fn):
        return {"ok": False, "detail": f"no {REQUIRED[0]}"}
    ref = _ref()
    _s = random.Random().getrandbits(63)
    cases_a = _gen_cases(_s)
    cases_b = _gen_cases(_s)          # fresh inputs — mutation-safe
    for i, (ca, cb) in enumerate(zip(cases_a, cases_b)):
        try:
            y = _call(fn, ca)
            y0 = _call(ref, cb)
        except Exception as e:
            return {"ok": False,
                    "detail": f"case {i} raised {type(e).__name__}: {e}"}
        try:
            if not _same(y, y0):
                return {"ok": False, "detail": f"case {i} mismatch"}
        except Exception as e:
            return {"ok": False, "detail": f"case {i} compare {e}"}
    return {"ok": True}


def bench(ns):
    fn = ns[REQUIRED[0]]
    cases = _gen_cases(random.Random().getrandbits(63))
    for c in cases:
        _call(fn, c)
    on_cuda = _any_cuda(cases)
    if on_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(BENCH_REPS):
        for c in cases:
            _call(fn, c)
            n += 1
    if on_cuda:
        torch.cuda.synchronize()
    return {"ms_per_call": (time.perf_counter() - t0) / max(n, 1) * 1000}
