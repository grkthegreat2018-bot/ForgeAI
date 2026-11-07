"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'parse_and_tokenize_regex'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['parse_and_tokenize_regex']
PROMPT = 'Optimize this function to parse and tokenize a string using regex patterns, ensuring correctness for all valid inputs. The function must accurately split the input into tokens based on the given regex patterns, handling edge cases like empty strings, consecutive delimiters, and nested structures. The optimization should focus on reducing redundant operations and improving the efficiency of pattern matching.'

SEED_CODE = 'import re\n\ndef parse_and_tokenize_regex(s, patterns):\n    tokens = []\n    remaining = s\n    \n    for pattern in patterns:\n        matches = re.findall(pattern, remaining)\n        for match in matches:\n            if match in remaining:\n                parts = remaining.split(match, 1)\n                if parts[0]:\n                    tokens.append(parts[0])\n                tokens.append(match)\n                remaining = parts[1]\n    \n    if remaining:\n        tokens.append(remaining)\n    \n    return tokens'

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

def _gen_cases(seed):
    random.seed(seed)
    cases = [
        # Simple cases
        ('a,b,c', [r'\d+', r'\w+']),
        ('123abc456', [r'\d+', r'\w+']),
        # Edge cases
        ('', [r'\d+', r'\w+']),
        ('abc', [r'\d+', r'\w+']),
        ('123', [r'\d+', r'\w+']),
        # Complex cases
        ('a1b2c3', [r'\d+', r'\w+']),
        ('x.y.z', [r'\.', r'\w+']),
        ('a,,b', [r',', r'\w+']),
        ('123 456', [r'\s+', r'\d+']),
        ('(a(b)c)', [r'\(.*?\)', r'\w+']),
        # Random cases
        (''.join(random.choices('abc123 ,.', k=20)), [r'\d+', r'\w+']),
        (''.join(random.choices('abc123 ,.', k=30)), [r'\d+', r'\w+']),
        (''.join(random.choices('abc123 ,.', k=40)), [r'\d+', r'\w+'])
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
