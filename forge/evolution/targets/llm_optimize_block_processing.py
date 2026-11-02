"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'optimize_block_processing'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['optimize_block_processing']
PROMPT = 'Optimize the processing of blocks in a matrix where each block is independently processed and stored. The function must correctly process all blocks in the matrix, but should minimize redundant computations and memory allocations. The correctness constraint is that the output matrix must match the block-wise processing of the input matrix.'

SEED_CODE = 'def optimize_block_processing(matrix: list[list[int]], block_size: int) -> list[list[int]]:\n    """Process each block of the matrix independently and return a new matrix with processed blocks.\n\n    Args:\n        matrix: A 2D list of integers (rows x cols).\n        block_size: The size of each square block to process.\n\n    Returns:\n        A new matrix where each block is processed (e.g., summed).\n    """\n    rows = len(matrix)\n    cols = len(matrix[0]) if rows > 0 else 0\n    result = [[0 for _ in range(cols)] for _ in range(rows)]\n\n    for i in range(0, rows, block_size):\n        for j in range(0, cols, block_size):\n            # Extract the current block\n            block = [\n                row[j:j+block_size]\n                for row in matrix[i:i+block_size]\n            ]\n            \n            # Process the block (example: sum all elements)\n            processed_block = sum(sum(row) for row in block)\n            \n            # Place the processed block back into the result\n            for bi in range(len(block)):\n                for bj in range(len(block[0])):\n                    result[i + bi][j + bj] = processed_block\n\n    return result'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

def _gen_cases() -> list[tuple[list[list[int]], int]]:
    cases = [
        # Small matrix, small block size
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1),
        # Small matrix, large block size
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3),
        # Large matrix, small block size
        ([[i + j for j in range(10)] for i in range(10)], 1),
        # Large matrix, large block size
        ([[i + j for j in range(10)] for i in range(10)], 5),
        # Non-square matrix
        ([[1, 2, 3, 4], [5, 6, 7, 8]], 2),
        # Edge case: empty matrix
        ([], 2),
        # Edge case: single element matrix
        ([[42]], 1)
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
