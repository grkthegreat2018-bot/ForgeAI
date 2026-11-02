"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'rolling_window_median'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['rolling_window_median']
PROMPT = 'Implement a function that computes the median of a sliding window over a list of numbers. The window size is fixed and the function must return a list of medians for each window position. Ensure the implementation is optimized to avoid O(n^2) behavior, and handle edge cases like window size larger than the input list or empty input.'

SEED_CODE = 'def rolling_window_median(numbers, window_size):\n    if window_size <= 0 or window_size > len(numbers):\n        return []\n    medians = []\n    for i in range(len(numbers) - window_size + 1):\n        window = numbers[i:i + window_size]\n        sorted_window = sorted(window)\n        mid = window_size // 2\n        if window_size % 2 == 1:\n            medians.append(sorted_window[mid])\n        else:\n            medians.append((sorted_window[mid - 1] + sorted_window[mid]) / 2)\n    return medians'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

def _gen_cases():
    cases = [
        ([1, 2, 3, 4, 5], 3),
        ([5, 4, 3, 2, 1], 2),
        ([1, 1, 2, 2, 3, 3], 4),
        ([10, 20, 30, 40, 50, 60], 5),
        ([], 3),
        ([1, 2, 3], 0),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1),
        ([1, 1, 1, 1, 1], 5),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3),
        ([1.5, 2.5, 3.5, 4.5, 5.5], 2),
        ([10, 20, 30, 40, 50], 4),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 6),
        ([1, 3, 2, 4, 5, 7, 6, 8, 9, 10], 5),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 7)
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
    cases_a = _gen_cases()
    cases_b = _gen_cases()          # fresh inputs — mutation-safe
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
    cases = _gen_cases()
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
