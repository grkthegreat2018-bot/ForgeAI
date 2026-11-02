"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'quantized_tensor_merge'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['quantized_tensor_merge']
PROMPT = 'Optimize a function that merges two quantized tensors by packing their values and scales into a single output tensor. The function must ensure correct quantization and merging while minimizing redundant computations and memory copies. The correctness constraint is that the merged tensor must exactly match the result of sequentially processing each element, but the implementation must avoid per-element loops where possible.'

SEED_CODE = 'import torch\n\ndef quantized_tensor_merge(tensor_a, scales_a, tensor_b, scales_b, dtype=torch.bfloat16):\n    # Ensure tensors are contiguous and have the same shape\n    tensor_a = tensor_a.contiguous().to(dtype)\n    tensor_b = tensor_b.contiguous().to(dtype)\n    scales_a = scales_a.contiguous().to(dtype)\n    scales_b = scales_b.contiguous().to(dtype)\n    \n    # Initialize output tensors\n    merged_values = torch.empty_like(tensor_a)\n    merged_scales = torch.empty_like(scales_a)\n    \n    # Iterate through each element and merge\n    for i in range(tensor_a.shape[0]):\n        for j in range(tensor_a.shape[1]):\n            for k in range(tensor_a.shape[2]):\n                # Determine which value to keep based on scale (smaller scale means higher precision)\n                if scales_a[i, j, k] < scales_b[i, j, k]:\n                    merged_values[i, j, k] = tensor_a[i, j, k]\n                    merged_scales[i, j, k] = scales_a[i, j, k]\n                else:\n                    merged_values[i, j, k] = tensor_b[i, j, k]\n                    merged_scales[i, j, k] = scales_b[i, j, k]\n    \n    return merged_values, merged_scales'

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
    torch.manual_seed(42)
    cases = []
    for _ in range(5):
        batch_size = random.randint(1, 4)
        seq_len = random.randint(1, 10)
        head_dim = random.randint(8, 32)
        tensor_a = torch.randn(batch_size, seq_len, head_dim)
        tensor_b = torch.randn(batch_size, seq_len, head_dim)
        scales_a = torch.rand(batch_size, seq_len, head_dim) * 0.1
        scales_b = torch.rand(batch_size, seq_len, head_dim) * 0.1
        cases.append((tensor_a, scales_a, tensor_b, scales_b))
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
