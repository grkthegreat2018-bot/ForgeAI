"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'merge_dicts'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['merge_dicts']
PROMPT = 'Optimize a function that merges multiple dictionaries into one, combining values for duplicate keys by concatenating them into lists. The function must handle dictionaries with arbitrary keys and values, and preserve the order of keys as they appear in the input dictionaries. The correctness constraint is that the output must be a single dictionary where each key maps to a list of all values from the input dictionaries, in the order they were encountered.'

SEED_CODE = 'def merge_dicts(*dicts):\n    result = {}\n    for d in dicts:\n        for key, value in d.items():\n            if key in result:\n                result[key].append(value)\n            else:\n                result[key] = [value]\n    return result'

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
    import random
    random.seed(seed)
    cases = [
        ({}, {}, {'a': 1}, {'a': 2, 'b': 3}),
        ({'x': 10}, {'x': 20, 'y': 30}, {'z': 40}),
        ({'a': 'foo'}, {'a': 'bar', 'b': 'baz'}, {'a': 'qux'}, {'b': 'quux'}),
        ({'k': [1, 2]}, {'k': [3, 4]}, {'k': [5, 6]}),
        ({'m': None}, {'m': True, 'n': False}, {'m': 1.0, 'n': 0.0}),
        ({'p': {'q': 1}}, {'p': {'r': 2}}, {'s': {'t': 3}}),
        ({'x': 1}, {'y': 2}, {'z': 3}, {'w': 4}, {'v': 5}),
        ({'a': 1, 'b': 2}, {'a': 3, 'c': 4}, {'b': 5, 'd': 6}),
        ({'k': 'v1'}, {'k': 'v2'}, {'k': 'v3'}, {'k': 'v4'}, {'k': 'v5'}),
        ({'first': 1}, {'second': 2}, {'third': 3}, {'fourth': 4}, {'fifth': 5}),
        ({'one': 1}, {'two': 2, 'one': 3}, {'three': 3, 'two': 4}),
        ({'a': 1}, {'b': 2}, {'c': 3}, {'d': 4}, {'e': 5}, {'f': 6}),
        ({'x': 100}, {'y': 200, 'x': 300}, {'z': 400, 'y': 500}),
        ({'k1': 'v1'}, {'k2': 'v2'}, {'k1': 'v3', 'k3': 'v4'}, {'k2': 'v5'}),
        ({'nested': {'a': 1}}, {'nested': {'b': 2}}, {'nested': {'c': 3}}),
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
