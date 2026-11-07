"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'compute_vertical_bands'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['compute_vertical_bands']
PROMPT = 'Implement a function that identifies vertical bands in a 2D matrix by computing the column importance as the sum of squared values in each column, then selecting the top-k columns with the highest importance. The function must be optimized for large matrices, avoiding unnecessary computations or copies. Ensure correctness by maintaining the same output as the naive implementation.'

SEED_CODE = 'def compute_vertical_bands(matrix, k):\n    """Identify the top-k columns with highest sum of squared values.\n    Args:\n        matrix: 2D list of numbers (rows x columns)\n        k: number of top columns to return\n    Returns:\n        list of column indices sorted by importance (descending)\n    """\n    if not matrix or k <= 0:\n        return []\n    n_cols = len(matrix[0])\n    col_importance = [0.0] * n_cols\n    for row in matrix:\n        for col_idx in range(n_cols):\n            val = row[col_idx]\n            col_importance[col_idx] += val * val\n    # Sort columns by importance (descending)\n    sorted_cols = sorted(\n        ((i, imp) for i, imp in enumerate(col_importance)),\n        key=lambda x: -x[1]\n    )\n    return [col[0] for col in sorted_cols[:k]]'

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
    cases = [
        # Small matrix
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2),
        # Uniform values
        ([[2, 2, 2], [2, 2, 2]], 3),
        # Single row
        ([[1, 3, 2]], 1),
        # Large matrix (5x10)
        ([[i + j for j in range(10)] for i in range(5)], 4),
        # Edge case: k larger than columns
        ([[1, 2], [3, 4], [5, 6]], 3),
        # Edge case: empty matrix
        ([], 2),
        # Edge case: k=0
        ([[1, 2, 3], [4, 5, 6]], 0),
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
