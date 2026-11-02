"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'matrix_diagonal_scaling'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['matrix_diagonal_scaling']
PROMPT = 'Optimize the function to scale each diagonal of a matrix by a corresponding factor, ensuring the output is correct and the implementation is more efficient than the provided naive version. The function must handle both square and non-square matrices, scaling only the existing diagonals without modifying other elements.'

SEED_CODE = 'import torch\n\ndef matrix_diagonal_scaling(matrix: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:\n    """Scale each diagonal of the matrix by the corresponding factor in `factors`.\n    Args:\n        matrix: Input matrix of shape (m, n).\n        factors: Tensor of shape (k,) where k is the number of diagonals.\n    Returns:\n        Matrix with diagonals scaled by `factors`.\n    """\n    m, n = matrix.shape\n    result = matrix.clone()\n    for i in range(min(m, n)):\n        if i < len(factors):\n            result.diagonal(i)[:] *= factors[i]\n    for i in range(1, min(m, n)):\n        if i < len(factors):\n            result.diagonal(-i)[:] *= factors[i]\n    return result'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

import torch

def _gen_cases():
    cases = [
        (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), torch.tensor([2, 3])),
        (torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([1, 2, 3])),
        (torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), torch.tensor([1, 2])),
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([1, 2, 3])),
        (torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), torch.tensor([1, 2, 3, 4])),
        (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]), torch.tensor([1, 2, 3])),
        (torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), torch.tensor([1, 2, 3, 4])),
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
