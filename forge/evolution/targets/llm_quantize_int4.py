"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'quantize_int4'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['quantize_int4']
PROMPT = 'Implement a symmetric INT4 quantization function that processes a 2D input tensor in groups, computes per-group scales and zero-points, and returns quantized codes. The function must handle padding to ensure group alignment and must preserve the original tensor shape after quantization. Optimize the function to minimize redundant computations and memory allocations, particularly focusing on the scale and zero-point calculations.'

SEED_CODE = 'import math\n\ndef quantize_int4(input_tensor, group_size, zp=None):\n    """Quantize a 2D tensor to symmetric INT4 with per-group scaling and zero-point.\n\n    Args:\n        input_tensor: 2D list of floats (rows=output_features, cols=input_features)\n        group_size: int, size of each quantization group\n        zp: optional 2D list of zero-points (same shape as scales)\n\n    Returns:\n        tuple: (quantized_codes, scales, zero_points, pad, num_groups)\n    """\n    out_f, in_f = len(input_tensor), len(input_tensor[0])\n    pad = (group_size - in_f % group_size) % group_size\n    if pad > 0:\n        input_tensor = [row + [0.0] * pad for row in input_tensor]\n    in_padded = len(input_tensor[0])\n    n_groups = in_padded // group_size\n\n    if zp is None:\n        zp = [[0.0 for _ in range(n_groups)] for _ in range(out_f)]\n\n    scales = []\n    quantized_codes = []\n    for i in range(out_f):\n        row_scales = []\n        row_codes = []\n        for g in range(n_groups):\n            start = g * group_size\n            end = start + group_size\n            group = [input_tensor[i][j] - zp[i][g] for j in range(start, end)]\n            absmax = max(abs(x) for x in group) if group else 0.0\n            scale = absmax / 3.0 if absmax > 0 else 1.0\n            row_scales.append(scale)\n            row_codes.extend([round((x - zp[i][g]) / scale) for x in group])\n        scales.append(row_scales)\n        quantized_codes.append(row_codes)\n\n    return quantized_codes, scales, zp, pad, n_groups'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

def _gen_cases(seed):
    cases = []
    # Small case
    cases.append(([[1.0, -1.0, 2.0, -2.0]], 2))
    # Medium case
    cases.append(([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0]], 3))
    # Large case with padding
    cases.append(([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]], 5))
    # Case with zero-points
    cases.append(([[1.0, -1.0, 2.0, -2.0]], 2, [[0.5, -0.5]]))
    # Edge case: empty group
    cases.append(([[1.0, -1.0, 2.0, -2.0, 0.0, 0.0]], 3))
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
    _s = random.Random().getrandbits(63)
    cases_a = _gen_cases(_s)
    cases_b = _gen_cases(_s)          # fresh inputs — mutation-safe
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
    cases = _gen_cases(random.Random().getrandbits(63))
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
