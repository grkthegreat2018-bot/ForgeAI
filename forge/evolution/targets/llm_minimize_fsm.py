"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'minimize_fsm'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['minimize_fsm']
PROMPT = "Optimize this function to minimize a deterministic finite state machine (DFSM) by removing unreachable states. The function must preserve the machine's behavior for all reachable states, and the minimized machine must have no unreachable states. The input is a dictionary where keys are state names and values are tuples of (transitions, is_accepting), where transitions is a dict mapping input symbols to next states. The output should be a new DFSM with the same language but no unreachable states."

SEED_CODE = 'def minimize_fsm(fsm: dict) -> dict:\n    """Minimize a DFSM by removing unreachable states."""\n    # Naive implementation: repeatedly remove states with no incoming edges\n    states = set(fsm.keys())\n    reachable = set()\n    stack = [\'start\']  # Assume \'start\' is the initial state\n    \n    while stack:\n        state = stack.pop()\n        if state in reachable:\n            continue\n        reachable.add(state)\n        for symbol, next_state in fsm[state][0].items():\n            if next_state not in reachable:\n                stack.append(next_state)\n    \n    minimized = {\n        state: (transitions, is_accepting)\n        for state, (transitions, is_accepting) in fsm.items()\n        if state in reachable\n    }\n    return minimized'

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
    case1 = {
        'start': ({'a': 's1', 'b': 's2'}, False),
        's1': ({'a': 's1', 'b': 's3'}, True),
        's2': ({'a': 's3', 'b': 's2'}, False),
        's3': ({'a': 's3', 'b': 's1'}, True)
    }
    case2 = {
        'start': ({'a': 's1'}, False),
        's1': ({'a': 's2'}, True),
        's2': ({'a': 's1'}, False),
        's3': ({'a': 's1'}, False)  # Unreachable
    }
    case3 = {
        'start': ({'a': 's1', 'b': 's2'}, True),
        's1': ({'a': 's1', 'b': 's3'}, False),
        's2': ({'a': 's2', 'b': 's3'}, False),
        's3': ({'a': 's3', 'b': 's1'}, True)
    }
    return [case1, case2, case3]


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
