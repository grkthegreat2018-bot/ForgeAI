"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'frequent_set_intersections'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['frequent_set_intersections']
PROMPT = "Optimize the function to find the most frequent element in a list of sets, where frequency is defined as the number of sets in which the element appears. The function must return the element with the highest frequency, or the smallest element in case of a tie. The solution must use only Python's standard library and must correctly handle empty lists or sets."

SEED_CODE = 'def frequent_set_intersections(sets):\n    if not sets:\n        return None\n    \n    element_counts = {}\n    for s in sets:\n        for elem in s:\n            if elem in element_counts:\n                element_counts[elem] += 1\n            else:\n                element_counts[elem] = 1\n    \n    max_count = max(element_counts.values(), default=0)\n    candidates = [k for k, v in element_counts.items() if v == max_count]\n    \n    return min(candidates) if candidates else None'

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
        # Edge cases
        [],
        [set()],
        [set(range(10)), set(range(5, 15))],
        # Random cases
        [set(random.sample(range(100), k=random.randint(1, 20))) for _ in range(10)],
        [set(random.sample(range(5), k=random.randint(0, 5))) for _ in range(20)],
        # Large case
        [set(random.sample(range(1000), k=random.randint(1, 50))) for _ in range(50)],
        # Overlapping case
        [set(range(i, i+10)) for i in range(0, 100, 5)],
        # Duplicate elements
        [set([1, 2, 3]), set([1, 2, 3]), set([1, 2, 3])],
        # Single element
        [set([42]) for _ in range(100)],
        # Mixed sizes
        [set([1]), set([1, 2]), set([1, 2, 3]), set([1, 2, 3, 4])]
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
