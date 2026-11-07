"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'closest_pair_of_points'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['closest_pair_of_points']
PROMPT = 'Implement a function that finds the pair of points with the smallest Euclidean distance in a 2D space. The function must return the distance and the indices of the two points in the input array. The input is a 2D numpy array where each row represents a point (x, y). Optimize for speed, ensuring correctness for all valid inputs.'

SEED_CODE = "import numpy as np\n\ndef closest_pair_of_points(points):\n    min_dist = float('inf')\n    indices = (0, 1)\n    n = len(points)\n    for i in range(n):\n        for j in range(i + 1, n):\n            dist = np.linalg.norm(points[i] - points[j])\n            if dist < min_dist:\n                min_dist = dist\n                indices = (i, j)\n    return min_dist, indices"

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
    np.random.seed(seed)
    cases = [
        np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
        np.array([[1, 1], [1, 2], [2, 1], [2, 2]]),
        np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]]),
        np.random.rand(10, 2),
        np.random.rand(20, 2),
        np.array([[0, 0], [0, 0.1], [0.1, 0], [0.1, 0.1]]),
        np.array([[0, 0], [10, 0], [5, 5 * np.sqrt(3)]]),
        np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
        np.array([[0, 0], [0, 1], [0, 2], [0, 3]]),
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
        np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]),
        np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]),
        np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2]]),
        np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
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
