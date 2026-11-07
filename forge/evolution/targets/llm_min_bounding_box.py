"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'min_bounding_box'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['min_bounding_box']
PROMPT = 'Given a set of 2D points (as a numpy array of shape (N, 2)), compute the smallest axis-aligned bounding box that contains all points. The bounding box is defined by its minimum and maximum coordinates (x_min, y_min, x_max, y_max). Optimize for speed, ensuring correctness by verifying that all points lie within the computed box. The input array is guaranteed to have at least one point.'

SEED_CODE = "import numpy as np\n\ndef min_bounding_box(points):\n    min_x = float('inf')\n    min_y = float('inf')\n    max_x = float('-inf')\n    max_y = float('-inf')\n    for x, y in points:\n        if x < min_x:\n            min_x = x\n        if y < min_y:\n            min_y = y\n        if x > max_x:\n            max_x = x\n        if y > max_y:\n            max_y = y\n    return (min_x, min_y, max_x, max_y)"

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

import numpy as np

def _gen_cases(seed):
    np.random.seed(seed % (2 ** 32))
    cases = [
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 1], [2, 2], [3, 3]]),
        np.array([[-1, -1], [1, 1], [0, 0]]),
        np.random.rand(100, 2) * 10 - 5,
        np.random.rand(1000, 2) * 100 - 50,
        np.array([[5, 5], [5, 5], [5, 5]]),
        np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
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
