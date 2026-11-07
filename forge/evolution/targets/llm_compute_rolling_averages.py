"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'compute_rolling_averages'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['compute_rolling_averages']
PROMPT = 'Compute the rolling average of a sequence of numbers with a fixed window size. The function must correctly handle edge cases where the window size is larger than the sequence length. The optimization goal is to reduce the time complexity from O(n*m) to O(n) where n is the sequence length and m is the window size.'

SEED_CODE = 'import numpy as np\ndef compute_rolling_averages(sequence, window_size):\n    """Compute rolling averages of a sequence with a given window size.\n    Args:\n        sequence: List or array of numbers.\n        window_size: Size of the rolling window.\n    Returns:\n        List of rolling averages."""\n    n = len(sequence)\n    if window_size <= 0 or n == 0:\n        return []\n    if window_size > n:\n        window_size = n\n    rolling_avgs = []\n    for i in range(n - window_size + 1):\n        window = sequence[i:i + window_size]\n        avg = sum(window) / window_size\n        rolling_avgs.append(avg)\n    return rolling_avgs'

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
    cases = [
        ([1, 2, 3, 4, 5], 3),
        ([10, 20, 30, 40, 50], 2),
        ([5, 5, 5, 5, 5], 5),
        ([], 3),
        ([1, 2, 3], 0),
        ([1, 2, 3, 4, 5], 1),
        ([1, 2, 3, 4, 5], 6),
        ([1.5, 2.5, 3.5, 4.5], 2),
        ([100, 200, 300, 400, 500], 3),
        ([1, 1, 1, 1, 1], 4),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10),
        ([1], 1),
        ([1, 2, 3, 4, 5], 4),
        ([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 7)
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
