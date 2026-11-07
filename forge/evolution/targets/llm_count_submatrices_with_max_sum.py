"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'count_submatrices_with_max_sum'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['count_submatrices_with_max_sum']
PROMPT = 'Count the number of submatrices in a given 2D array where the sum of elements is exactly equal to the maximum possible sum of any submatrix. The solution must be optimized to avoid O(n^4) time complexity and should use numpy for efficient operations. Ensure correctness by verifying the result against a brute-force approach for small matrices.'

SEED_CODE = 'import numpy as np\n\ndef count_submatrices_with_max_sum(matrix):\n    n, m = matrix.shape\n    max_sum = -np.inf\n    count = 0\n    \n    # Compute prefix sums for efficient submatrix sum calculation\n    prefix = np.zeros((n+1, m+1))\n    for i in range(1, n+1):\n        for j in range(1, m+1):\n            prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]\n    \n    # Find the maximum submatrix sum\n    for i1 in range(n):\n        for j1 in range(m):\n            for i2 in range(i1, n):\n                for j2 in range(j1, m):\n                    current_sum = prefix[i2+1][j2+1] - prefix[i1][j2+1] - prefix[i2+1][j1] + prefix[i1][j1]\n                    if current_sum > max_sum:\n                        max_sum = current_sum\n    \n    # Count submatrices with max_sum\n    for i1 in range(n):\n        for j1 in range(m):\n            for i2 in range(i1, n):\n                for j2 in range(j1, m):\n                    current_sum = prefix[i2+1][j2+1] - prefix[i1][j2+1] - prefix[i2+1][j1] + prefix[i1][j1]\n                    if current_sum == max_sum:\n                        count += 1\n    \n    return count'

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
    np.random.seed(seed)
    cases = [
        # Small matrices
        (np.array([[1, 2], [3, 4]]),),
        (np.array([[1, -1], [-1, 1]]),),
        # Medium matrices
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),),
        (np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]]),),
        # Large matrix (edge case)
        (np.random.randint(-10, 10, size=(10, 10)),),
        # All positive
        (np.random.randint(1, 10, size=(5, 5)),),
        # All negative
        (np.random.randint(-10, -1, size=(5, 5)),),
        # Mixed with zeros
        (np.array([[0, 1, -1], [2, 0, 3], [-2, 1, 0]]),),
        # Single row
        (np.array([[1, 2, 3, 4, 5]]),),
        # Single column
        (np.array([[1], [2], [3], [4], [5]]),)
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
