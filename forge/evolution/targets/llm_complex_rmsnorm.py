"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'complex_rmsnorm'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['complex_rmsnorm']
PROMPT = 'Implement a function that normalizes complex-valued tensors using root-mean-square (RMS) normalization, where the gain tensor may be in bfloat16 or float32. The function should handle complex64 inputs and produce complex64 outputs, with the normalization applied element-wise across the last dimension. Ensure numerical stability by adding a small epsilon value to the denominator. The implementation must avoid unnecessary copies and minimize redundant computations.'

SEED_CODE = 'import torch\n\ndef complex_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:\n    """Normalize complex-valued tensor x using RMS normalization with gain tensor w.\n\n    Args:\n        x: Complex64 tensor of shape (..., D).\n        w: Real-valued tensor of shape (..., D, 2) or (..., D, 2) in bfloat16/float32.\n        eps: Small value to avoid division by zero.\n\n    Returns:\n        Complex64 tensor of the same shape as x, normalized by RMS.\n    """\n    # Split into real and imaginary parts\n    x_real = x.real\n    x_imag = x.imag\n    \n    # Compute RMS denominator\n    rms_denom = torch.sqrt(torch.mean(x_real**2 + x_imag**2, dim=-1, keepdim=True) + eps)\n    \n    # Normalize real and imaginary parts\n    x_real_norm = x_real / rms_denom\n    x_imag_norm = x_imag / rms_denom\n    \n    # Apply gain\n    gain_real = w[..., 0]\n    gain_imag = w[..., 1]\n    \n    # Combine with gain\n    out_real = x_real_norm * gain_real\n    out_imag = x_imag_norm * gain_imag\n    \n    return torch.complex(out_real, out_imag)'

import hashlib
import math
import random
import struct
import time
import weakref
import numpy as np
import torch
import torch.nn.functional as F

import torch
def _gen_cases():
    cases = []
    g = torch.Generator().manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate cases with complex64 inputs and real gain tensors
    for shape in [(1, 8, 32), (2, 16, 64), (1, 64, 128), (4, 32, 96)]:
        re = torch.randn(*shape, generator=g).to(dev)
        im = torch.randn(*shape, generator=g).to(dev)
        x = torch.complex(re, im)
        w = torch.randn(*shape, 2, generator=g).to(dev)
        cases.append((x, w, 1e-6))
    
    # Add a case with bfloat16 gain tensor
    x = torch.complex(torch.randn(2, 8, 16, generator=g).to(dev),
                      torch.randn(2, 8, 16, generator=g).to(dev))
    w = torch.randn(2, 8, 16, 2, generator=g).to(dev).to(torch.bfloat16)
    cases.append((x, w, 1e-6))
    
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
