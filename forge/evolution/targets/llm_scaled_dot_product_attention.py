"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'scaled_dot_product_attention'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['scaled_dot_product_attention']
PROMPT = 'Optimize the scaled dot-product attention computation to minimize memory usage and maximize throughput. The function must compute attention weights and apply them to values, using the formula: softmax((Q @ K.T) / sqrt(d_k)) @ V. The correctness constraint is that the output must match the naive implementation within floating-point tolerance, but the optimized version must avoid redundant computations and memory allocations.'

SEED_CODE = "import torch\nimport torch.nn.functional as F\ndef scaled_dot_product_attention(Q, K, V, mask=None):\n    d_k = Q.size(-1)\n    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, float('-inf'))\n    p_attn = F.softmax(scores, dim=-1)\n    return torch.matmul(p_attn, V)"

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
    torch.manual_seed(42)
    cases = [
        (torch.randn(3, 4, 16), torch.randn(3, 4, 16), torch.randn(3, 4, 16)),
        (torch.randn(1, 8, 64), torch.randn(1, 8, 64), torch.randn(1, 8, 64)),
        (torch.randn(2, 16, 32), torch.randn(2, 16, 32), torch.randn(2, 16, 32)),
        (torch.randn(4, 4, 64), torch.randn(4, 4, 64), torch.randn(4, 4, 64)),
        (torch.randn(1, 1, 128), torch.randn(1, 1, 128), torch.randn(1, 1, 128)),
        (torch.randn(3, 4, 16), torch.randn(3, 4, 16), torch.randn(3, 4, 16), torch.ones(3, 4, 4) > 0.5),
        (torch.randn(2, 16, 32), torch.randn(2, 16, 32), torch.randn(2, 16, 32), torch.tril(torch.ones(16, 16)) > 0)
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
