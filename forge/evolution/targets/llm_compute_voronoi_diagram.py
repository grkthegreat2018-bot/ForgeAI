"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'compute_voronoi_diagram'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['compute_voronoi_diagram']
PROMPT = 'Implement a function to compute the Voronoi diagram for a set of points in 2D space. The function must return a list of edges representing the Voronoi diagram, where each edge is a tuple of two points (start, end). The correctness constraint is that the diagram must correctly partition the plane such that all points in a Voronoi cell are closer to the corresponding input point than to any other input point. Optimize the implementation to avoid redundant distance calculations and unnecessary comparisons.'

SEED_CODE = 'import math\n\ndef compute_voronoi_diagram(points):\n    edges = []\n    n = len(points)\n    for i in range(n):\n        for j in range(n):\n            if i == j:\n                continue\n            p1, p2 = points[i], points[j]\n            x1, y1 = p1\n            x2, y2 = p2\n            mid_x = (x1 + x2) / 2\n            mid_y = (y1 + y2) / 2\n            dx = x2 - x1\n            dy = y2 - y1\n            if dx == 0 and dy == 0:\n                continue\n            dist = math.sqrt(dx**2 + dy**2)\n            normal_x = -dy / dist\n            normal_y = dx / dist\n            edges.append(((mid_x, mid_y), (mid_x + normal_x, mid_y + normal_y)))\n    return edges'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

import random

def _gen_cases():
    random.seed(42)
    cases = [
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1)
        ],
        [
            (0, 0),
            (2, 0),
            (1, 1),
            (0, 2),
            (2, 2)
        ],
        [
            (random.uniform(-10, 10), random.uniform(-10, 10))
            for _ in range(5)
        ],
        [
            (random.uniform(-50, 50), random.uniform(-50, 50))
            for _ in range(10)
        ],
        [
            (i, i * 2) for i in range(15)
        ]
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
