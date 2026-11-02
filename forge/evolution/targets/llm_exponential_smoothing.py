"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'exponential_smoothing'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['exponential_smoothing']
PROMPT = 'Implement an exponential smoothing filter to reduce noise in a signal. The function must correctly compute the smoothed values using the provided alpha parameter, but should be optimized to avoid repeated calculations and unnecessary list operations. The output must match the reference implementation exactly for all inputs.'

SEED_CODE = 'def exponential_smoothing(signal, alpha):\n    smoothed = []\n    prev = signal[0]\n    for value in signal:\n        smoothed_value = alpha * value + (1 - alpha) * prev\n        smoothed.append(smoothed_value)\n        prev = smoothed_value\n    return smoothed'

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
        ([1.0, 2.0, 3.0, 4.0, 5.0], 0.5),
        ([10.0, 12.0, 15.0, 13.0, 14.0], 0.3),
        ([0.0, 0.1, 0.2, 0.3, 0.4], 0.8),
        ([5.0, 5.0, 5.0, 5.0, 5.0], 0.1),
        ([1.0, 3.0, 5.0, 7.0, 9.0], 0.9),
        ([2.0, 4.0, 6.0, 8.0, 10.0], 0.2),
        ([0.5, 1.5, 2.5, 3.5, 4.5], 0.6),
        ([1.0, 0.5, 1.0, 0.5, 1.0], 0.4),
        ([100.0, 200.0, 300.0, 400.0, 500.0], 0.7),
        ([-1.0, -2.0, -3.0, -4.0, -5.0], 0.5),
        ([1.0, 1.0, 1.0, 1.0, 1.0], 0.9),
        ([1.0, 2.0, 1.0, 2.0, 1.0], 0.3),
        ([0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
        ([1.0, 2.0, 4.0, 8.0, 16.0], 0.1),
        ([1.0, 1.1, 1.2, 1.3, 1.4], 0.8)
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
