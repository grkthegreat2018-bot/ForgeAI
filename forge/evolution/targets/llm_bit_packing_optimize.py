"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'bit_packing_optimize'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['bit_packing_optimize']
PROMPT = 'Optimize the function to efficiently pack and unpack variable-length bit sequences into a compact byte array. The function must correctly handle sequences of bits (1-8 bits per element) and pack them into bytes, minimizing padding and ensuring the original sequence can be perfectly reconstructed. The correctness constraint is that the unpacked sequence must exactly match the input sequence.'

SEED_CODE = 'def bit_packing_optimize(bit_sequences):\n    """Pack a list of bit sequences (each 1-8 bits) into a byte array, then unpack it to verify correctness."""\n    packed = bytearray()\n    remaining_bits = 0\n    current_byte = 0\n\n    for bits in bit_sequences:\n        bit_count = len(bits)\n        if bit_count > 8 or bit_count < 1:\n            raise ValueError("Bit sequence must be 1-8 bits")\n        \n        # Convert bits to integer\n        value = int(\'\'.join(map(str, bits)), 2)\n        \n        # Pack into bytes\n        while bit_count > 0:\n            if remaining_bits == 0:\n                current_byte = 0\n            \n            bits_to_take = min(8 - remaining_bits, bit_count)\n            shifted_value = (value >> (bit_count - bits_to_take)) & ((1 << bits_to_take) - 1)\n            current_byte |= shifted_value << (8 - remaining_bits - bits_to_take)\n            remaining_bits += bits_to_take\n            bit_count -= bits_to_take\n\n            if remaining_bits == 8:\n                packed.append(current_byte)\n                remaining_bits = 0\n\n    if remaining_bits > 0:\n        packed.append(current_byte)\n\n    # Unpack to verify correctness\n    unpacked = []\n    for byte in packed:\n        for i in range(8):\n            bit = (byte >> (7 - i)) & 1\n            unpacked.append(bit)\n\n    # Reconstruct sequences from unpacked bits\n    result = []\n    idx = 0\n    for seq_len in map(len, bit_sequences):\n        result.append(unpacked[idx:idx + seq_len])\n        idx += seq_len\n\n    return result\n'

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
    import random
    random.seed(42)
    cases = [
        # Simple cases
        ([[1, 0], [1, 1, 0], [1]],),
        ([[1], [0], [1, 0, 1, 0, 1, 0, 1, 0]],),
        # Edge cases
        ([[1, 0, 1, 0, 1, 0, 1, 0]],),
        ([[1], [0, 1], [1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0, 1]],),
        # Random cases
        ([random.choices([0, 1], k=random.randint(1, 8)) for _ in range(10)],),
        ([random.choices([0, 1], k=random.randint(1, 8)) for _ in range(15)],),
        # Large sequences
        ([[1, 0, 1, 0, 1, 0, 1, 0]] * 10,),
        ([[1, 1, 1, 1, 1, 1, 1, 1]] * 5,)
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
