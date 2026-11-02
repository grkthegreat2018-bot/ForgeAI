"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'optimized_binary_search_with_torch'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['optimized_binary_search_with_torch']
PROMPT = "Implement a binary search function that leverages PyTorch's vectorized operations to search for a target value in a sorted 1D tensor. The function must return the index of the target if found, or -1 if not found. The correctness constraint is that the function must return the same result as a standard binary search but must be significantly faster for large tensors by avoiding Python loops."

SEED_CODE = 'import torch\n\ndef optimized_binary_search_with_torch(sorted_tensor, target):\n    left, right = 0, len(sorted_tensor) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if sorted_tensor[mid] == target:\n            return mid\n        elif sorted_tensor[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1'

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
import random

def _gen_cases():
    random.seed(42)
    cases = [
        (torch.tensor([1, 3, 5, 7, 9]), 5),
        (torch.tensor([-10, -5, 0, 5, 10]), 0),
        (torch.tensor([i for i in range(1000)]), 999),
        (torch.tensor([i for i in range(1000)]), -1),
        (torch.tensor([i for i in range(10000)]), 5000),
        (torch.tensor([i for i in range(10000)]), 10000),
        (torch.tensor([random.randint(0, 10000) for _ in range(1000)]), 42),
        (torch.tensor([random.randint(0, 10000) for _ in range(1000)]), -1),
        (torch.tensor([i * 2 for i in range(5000)]), 9998),
        (torch.tensor([i * 2 for i in range(5000)]), 9999)
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
