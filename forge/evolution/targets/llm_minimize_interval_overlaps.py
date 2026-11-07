"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'minimize_interval_overlaps'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['minimize_interval_overlaps']
PROMPT = 'Given a list of intervals, find the minimum number of intervals to remove so that no two intervals overlap. The solution must be correct and efficiently minimize the count, using numpy for array operations where helpful. The intervals are represented as tuples (start, end), and the function must return the count of intervals to remove.'

SEED_CODE = 'import numpy as np\ndef minimize_interval_overlaps(intervals):\n    intervals = sorted(intervals, key=lambda x: x[1])\n    count = 0\n    last_end = -np.inf\n    for start, end in intervals:\n        if start < last_end:\n            count += 1\n        else:\n            last_end = end\n    return count'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

import numpy as np
def _gen_cases(seed):
    np.random.seed(seed % (2 ** 32))
    cases = [
        [(1, 3), (2, 4), (5, 6)],
        [(1, 2), (2, 3), (3, 4), (4, 5)],
        [(1, 10), (2, 3), (4, 5), (6, 7), (8, 9)],
        [(1, 5), (2, 6), (3, 7), (4, 8), (5, 9)],
        [(1, 100), (1, 100), (1, 100)],
        [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)],
        [(1, 5), (2, 3), (4, 6), (7, 8), (9, 10)],
        [(1, 10), (11, 20), (21, 30), (22, 31), (23, 32)],
        [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50)],
        [(1, 2), (1, 3), (2, 3), (3, 4), (3, 5)],
        [(1, 10), (5, 15), (10, 20), (15, 25), (20, 30)],
        [(1, 100), (2, 101), (3, 102), (4, 103), (5, 104)],
        [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2)],
        [(1, 10), (11, 20), (11, 20), (21, 30), (21, 30)],
        [(1, 10), (2, 9), (3, 8), (4, 7), (5, 6)]
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
