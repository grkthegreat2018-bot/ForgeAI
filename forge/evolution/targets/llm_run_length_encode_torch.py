"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'run_length_encode_torch'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['run_length_encode_torch']
PROMPT = 'Optimize this function to perform run-length encoding on a 1D tensor using PyTorch operations. The function must correctly return a tuple of (values, counts) where values is a tensor of unique consecutive elements and counts is a tensor of their run lengths. Ensure the output matches the naive implementation but execute faster for large inputs.'

SEED_CODE = 'import torch\n\ndef run_length_encode_torch(tensor):\n    if len(tensor) == 0:\n        return torch.tensor([]), torch.tensor([])\n    \n    values = [tensor[0].item()]\n    counts = [1]\n    \n    for i in range(1, len(tensor)):\n        if tensor[i] == tensor[i-1]:\n            counts[-1] += 1\n        else:\n            values.append(tensor[i].item())\n            counts.append(1)\n    \n    return torch.tensor(values), torch.tensor(counts)'

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
        torch.tensor([], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1, 1, 1], dtype=torch.int32),
        torch.tensor([1, 2, 2, 3, 3, 3], dtype=torch.int32),
        torch.tensor([1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int32),
        torch.tensor([i % 5 for i in range(100)], dtype=torch.int32),
        torch.tensor([i // 10 for i in range(1000)], dtype=torch.int32),
        torch.tensor([random.randint(0, 2) for _ in range(500)], dtype=torch.int32),
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
