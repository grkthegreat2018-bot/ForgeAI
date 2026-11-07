"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'point_in_polygon'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['point_in_polygon']
PROMPT = 'Implement a function that checks if a point is inside a polygon using the ray-casting algorithm. The function should be optimized to handle large polygons efficiently while maintaining correctness. The polygon is represented as a list of vertices in order, and the point is a tuple of (x, y) coordinates. The function should return True if the point is inside the polygon, False otherwise.'

SEED_CODE = 'import numpy as np\n\ndef point_in_polygon(point, polygon):\n    x, y = point\n    n = len(polygon)\n    inside = False\n    for i in range(n):\n        x1, y1 = polygon[i]\n        x2, y2 = polygon[(i + 1) % n]\n        if y1 == y2:\n            continue\n        if min(y1, y2) <= y < max(y1, y2):\n            x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1\n            if x <= x_intersect:\n                inside = not inside\n    return inside'

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
        # Simple square
        ((0.5, 0.5), [(0, 0), (1, 0), (1, 1), (0, 1)]),
        # Point outside
        ((2, 2), [(0, 0), (1, 0), (1, 1), (0, 1)]),
        # Complex polygon
        ((0.5, 0.5), [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]),
        # Edge case: point on vertex
        ((0, 0), [(0, 0), (1, 0), (1, 1), (0, 1)]),
        # Large random polygon
        ((0.5, 0.5), np.random.rand(100, 2).tolist()),
        # Point on edge
        ((0.5, 0.5), [(0, 0), (1, 0), (1, 1), (0, 1)]),
        # Point inside hole
        ((1.5, 0.5), [(0, 0), (2, 0), (2, 2), (0, 2), (0, 1), (1, 1), (1, 0.5), (0, 0.5)]),
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
