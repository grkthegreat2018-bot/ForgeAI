"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'packed_bitwise_operations'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['packed_bitwise_operations']
PROMPT = 'Optimize the function to perform bitwise operations on packed bits efficiently. The function must correctly process packed bits into a tensor of integers, apply a series of bitwise operations (AND, OR, XOR), and unpack the results back into a tensor of integers. The correctness constraint is that the output tensor must match the result of the unoptimized version, but the implementation must minimize redundant operations and maximize vectorized computations using PyTorch.'

SEED_CODE = 'import torch\n\ndef packed_bitwise_operations(packed_bits: torch.Tensor, operations: list, bit_width: int) -> torch.Tensor:\n    """Process packed bits with bitwise operations.\n\n    Args:\n        packed_bits: Tensor of packed bits (dtype=torch.uint8).\n        operations: List of bitwise operations (\'AND\', \'OR\', \'XOR\').\n        bit_width: Number of bits per packed value.\n\n    Returns:\n        Tensor of unpacked integers after applying operations.\n    """\n    # Unpack the packed bits into individual integers\n    unpacked = torch.zeros(packed_bits.numel() * bit_width, dtype=torch.uint8)\n    for i in range(packed_bits.numel()):\n        for j in range(bit_width):\n            unpacked[i * bit_width + j] = (packed_bits[i] >> (bit_width - 1 - j)) & 1\n\n    # Apply each operation sequentially\n    for op in operations:\n        if op == \'AND\':\n            unpacked = unpacked & 1\n        elif op == \'OR\':\n            unpacked = unpacked | 1\n        elif op == \'XOR\':\n            unpacked = unpacked ^ 1\n\n    # Repack the bits into integers\n    result = torch.zeros(packed_bits.numel(), dtype=torch.uint8)\n    for i in range(packed_bits.numel()):\n        for j in range(bit_width):\n            result[i] |= (unpacked[i * bit_width + j] << (bit_width - 1 - j))\n\n    return result'

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
        (torch.tensor([0b10101010], dtype=torch.uint8), ['AND'], 8),
        (torch.tensor([0b11110000, 0b00001111], dtype=torch.uint8), ['OR'], 8),
        (torch.tensor([0b10101010, 0b01010101], dtype=torch.uint8), ['XOR'], 8),
        (torch.tensor([0b11001100, 0b00110011], dtype=torch.uint8), ['AND', 'OR'], 8),
        (torch.tensor([0b10010110, 0b01101001], dtype=torch.uint8), ['XOR', 'AND'], 8),
        (torch.tensor([0b11111111], dtype=torch.uint8), ['AND', 'OR', 'XOR'], 8),
        (torch.tensor([0b00000000, 0b11111111], dtype=torch.uint8), ['OR', 'XOR'], 8),
        (torch.tensor([0b10101010, 0b10101010, 0b10101010], dtype=torch.uint8), ['AND', 'XOR'], 8),
        (torch.tensor([0b01010101, 0b01010101], dtype=torch.uint8), ['OR', 'AND'], 8),
        (torch.tensor([0b11111111, 0b00000000, 0b11111111], dtype=torch.uint8), ['XOR', 'OR'], 8)
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
