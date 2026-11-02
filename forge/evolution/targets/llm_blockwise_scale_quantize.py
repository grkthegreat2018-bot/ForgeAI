"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'blockwise_scale_quantize'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['blockwise_scale_quantize']
PROMPT = 'Optimize a function that quantizes a 2D array by scaling and rounding blocks of elements. The function must process the input in fixed-size blocks, compute per-block scales, and apply quantization. The output must match the original values when rescaled, with minimal error. The optimization should focus on reducing redundant computations and memory copies, particularly in the block processing loop.'

SEED_CODE = 'def blockwise_scale_quantize(data: list[list[float]], block_size: int) -> tuple[list[list[int]], list[list[float]]]:\n    """Quantize a 2D array by scaling and rounding blocks of elements.\n\n    Args:\n        data: 2D array of floats to quantize.\n        block_size: size of the square blocks to process.\n    \n    Returns:\n        A tuple of (quantized_data, scales), where:\n        - quantized_data: 2D array of integers (0-255) representing quantized values.\n        - scales: 2D array of floats representing the per-block scaling factors.\n    """\n    rows = len(data)\n    cols = len(data[0]) if rows > 0 else 0\n    quantized = [[0 for _ in range(cols)] for _ in range(rows)]\n    scales = [[0.0 for _ in range(cols)] for _ in range(rows)]\n\n    for i in range(0, rows, block_size):\n        for j in range(0, cols, block_size):\n            # Extract block\n            block = [\n                [data[x][y] for y in range(j, min(j + block_size, cols))]\n                for x in range(i, min(i + block_size, rows))\n            ]\n\n            # Compute block statistics\n            min_val = min(min(row) for row in block)\n            max_val = max(max(row) for row in block)\n            scale = max_val - min_val if max_val != min_val else 1.0\n\n            # Quantize block\n            for x in range(len(block)):\n                for y in range(len(block[0])):\n                    scaled = (block[x][y] - min_val) / scale\n                    quantized_val = int(round(scaled * 255))\n                    quantized[i + x][j + y] = quantized_val\n                    scales[i + x][j + y] = scale\n\n    return quantized, scales'

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
        # Small uniform block
        ([[1.0, 2.0], [3.0, 4.0]], 2),
        # Larger block with varying values
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], 3),
        # Block with single value (edge case)
        ([[5.0, 5.0], [5.0, 5.0]], 2),
        # Non-square block
        ([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], 2),
        # Block with negative values
        ([[-1.0, 0.0], [1.0, 2.0]], 2),
        # Large block with random values
        ([[i + j * 0.1 for j in range(10)] for i in range(10)], 4),
        # Edge case: block_size larger than data
        ([[1.0, 2.0], [3.0, 4.0]], 4)
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
