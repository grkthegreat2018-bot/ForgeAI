"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'update_simulation_state'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['update_simulation_state']
PROMPT = 'Optimize a function that updates the state of a simulation with multiple entities. The function must correctly handle collisions between entities and update their positions and velocities. The correctness constraint is that after each update, no two entities can occupy the same position, and velocities must be updated based on collision rules (elastic collisions). The goal is to minimize the time complexity while maintaining correctness.'

SEED_CODE = 'def update_simulation_state(entities):\n    """\n    Update the state of entities in a simulation.\n\n    Args:\n        entities: List of tuples (x, y, vx, vy) representing position and velocity of each entity.\n\n    Returns:\n        List of updated tuples (x, y, vx, vy) after one time step.\n    """\n    updated_entities = []\n    for i, (x1, y1, vx1, vy1) in enumerate(entities):\n        new_x = x1 + vx1\n        new_y = y1 + vy1\n        new_vx, new_vy = vx1, vy1\n        for j, (x2, y2, vx2, vy2) in enumerate(entities):\n            if i != j and (new_x == x2 and new_y == y2):\n                # Elastic collision: swap velocities\n                new_vx, new_vy = vx2, vy2\n                break\n        updated_entities.append((new_x, new_y, new_vx, new_vy))\n    return updated_entities'

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
        # No collisions
        [
            (0, 0, 1, 0),
            (1, 1, 0, 1),
            (2, 2, -1, 0),
        ],
        # Single collision
        [
            (0, 0, 1, 0),
            (1, 0, -1, 0),
        ],
        # Multiple collisions
        [
            (0, 0, 1, 0),
            (1, 0, -1, 0),
            (0, 1, 0, -1),
            (0, -1, 0, 1),
        ],
        # Large set with no collisions
        [(i, i, 1, 0) for i in range(10)],
        # Large set with collisions
        [(i, 0, 1, 0) for i in range(10)] + [(i, 0, -1, 0) for i in range(10, 20)],
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
