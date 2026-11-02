"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'packed_int4_to_signed'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['packed_int4_to_signed']
PROMPT = 'Optimize a function that converts packed 4-bit integers into signed 8-bit integers, where the packed data is stored in a byte array with two integers per byte. The function must correctly unpack the integers and convert them to signed values in the range [-8, 7]. The optimization should focus on minimizing memory operations and improving the throughput of the unpacking process.'

SEED_CODE = 'def packed_int4_to_signed(packed_bytes: bytes, out_features: int, in_padded: int) -> list[int]:\n    """Unpack packed 4-bit integers into signed 8-bit integers."""\n    result = []\n    for i in range(out_features):\n        for j in range(in_padded):\n            byte_index = (i * in_padded + j) // 2\n            byte = packed_bytes[byte_index]\n            if j % 2 == 0:\n                value = byte & 0x0F\n            else:\n                value = (byte >> 4) & 0x0F\n            result.append(value - 8)\n    return result'

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
        (bytes([0x00, 0x11, 0x22, 0x33]), 2, 4),
        (bytes([0x44, 0x55, 0x66, 0x77]), 2, 4),
        (bytes([0x88, 0x99, 0xAA, 0xBB]), 2, 4),
        (bytes([0xCC, 0xDD, 0xEE, 0xFF]), 2, 4),
        (bytes([0x0F, 0xF0, 0x0F, 0xF0]), 2, 4),
        (bytes([0x12, 0x34, 0x56, 0x78]), 2, 4),
        (bytes([0x80, 0x08, 0x80, 0x08]), 2, 4),
        (bytes([0xFF, 0xFF, 0xFF, 0xFF]), 2, 4),
        (bytes([0x00, 0x00, 0x00, 0x00]), 2, 4),
        (bytes([0xAA, 0xAA, 0xAA, 0xAA]), 2, 4),
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
