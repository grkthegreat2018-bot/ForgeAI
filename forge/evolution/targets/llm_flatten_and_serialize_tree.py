"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'flatten_and_serialize_tree'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['flatten_and_serialize_tree']
PROMPT = "Implement a function that flattens a nested tree structure into a list of node values in depth-first order, then serializes the list into a compact byte string using PyTorch's serialization. The function must correctly handle arbitrary tree structures with any hashable node values. Optimize the implementation to minimize memory copies and redundant computations."

SEED_CODE = "import torch\n\ndef flatten_and_serialize_tree(root):\n    # Flatten the tree into a list of node values in depth-first order\n    flattened = []\n    stack = [root]\n    while stack:\n        node = stack.pop()\n        if node is None:\n            continue\n        if isinstance(node, tuple):\n            # Assuming binary tree for simplicity\n            left, right, value = node\n            stack.append(right)\n            stack.append(left)\n        else:\n            value = node\n        flattened.append(value)\n    \n    # Serialize the flattened list into bytes using torch.save\n    buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(bytes(str(flattened), 'utf-8')))\n    return buffer"

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
        # Simple binary tree
        ((None, None, 1),),
        # Larger tree
        (
            (
                ((None, None, 3), (None, None, 4), 2),
                ((None, None, 5), None, 6),
                1
            ),
        ),
        # Tree with None nodes
        (
            (
                (None, ((None, None, 2), None, 3), 1),
                None,
                0
            ),
        ),
        # Tree with non-integer values
        (
            (
                ((None, None, 'a'), (None, None, 'b'), 'c'),
                None,
                'root'
            ),
        ),
        # Deep tree
        (
            (
                (
                    (
                        (None, None, 10), (None, None, 11), 9
                    ),
                    (
                        (None, None, 12), None, 13
                    ),
                    8
                ),
                (
                    (
                        (None, None, 14), (None, None, 15), 16
                    ),
                    None,
                    7
                ),
                1
            ),
        )
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
