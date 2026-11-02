"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'complex_magnitude_scaling'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['complex_magnitude_scaling']
PROMPT = 'Optimize a function that computes the scaled magnitude of complex numbers in a 2D array, where the scaling factors are derived from the average magnitude of each row. Ensure the output matches the exact same values as the unoptimized version, but improve performance by reducing redundant computations and minimizing memory allocations.'

SEED_CODE = 'import math\n\ndef complex_magnitude_scaling(complex_numbers: list[tuple[float, float]], scale_factors: list[float]) -> list[tuple[float, float]]: \n    """Scale each complex number in the input list by its corresponding scale factor, where the scale factor is derived from the average magnitude of the row.\n\n    Args:\n        complex_numbers: List of complex numbers as (real, imag) tuples.\n        scale_factors: List of scaling factors, one per row.\n\n    Returns:\n        List of scaled complex numbers as (real, imag) tuples.\n    """\n    result = []\n    for i, (real, imag) in enumerate(complex_numbers):\n        magnitude = math.sqrt(real**2 + imag**2)\n        avg_magnitude = magnitude / len(complex_numbers)\n        scale = scale_factors[i] * avg_magnitude\n        result.append((real * scale, imag * scale))\n    return result'

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
        # Small case
        ([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], [0.5, 1.0, 1.5]),
        # Medium case
        ([(0.0, 1.0), (1.0, 0.0), (-1.0, -1.0), (0.5, 0.5)], [1.0, 0.5, 1.0, 0.5]),
        # Large case
        ([(i, i) for i in range(100)], [0.1 * i for i in range(100)]),
        # Edge case: zero magnitude
        ([(0.0, 0.0), (1.0, 1.0)], [1.0, 1.0]),
        # Edge case: uniform scale factors
        ([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)], [1.0, 1.0, 1.0]),
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
