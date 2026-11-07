"""LLM-generated evolution target — gen_llm_targets.py."""

NAME = 'fsm_minimize'
GOAL = 'min'
METRIC = 'ms_per_call'
TAGS = ['llm-generated']
BUDGETS = {}
BENCH_REPS = 3
REQUIRED = ['fsm_minimize']
PROMPT = 'Implement a function to minimize a deterministic finite state machine (DFSM) using the DFA minimization algorithm. The function must return a minimized version of the input FSM, preserving the language it accepts. The correctness constraint is that the minimized FSM must accept exactly the same language as the input FSM. Optimize for speed, especially for large state spaces, by avoiding unnecessary computations and using efficient data structures.'

SEED_CODE = 'import numpy as np\n\ndef fsm_minimize(states, alphabet, transitions, initial_state, accept_states):\n    """\n    Minimize a deterministic finite state machine.\n\n    Args:\n        states: List of states (e.g., [0, 1, 2])\n        alphabet: List of input symbols (e.g., [\'a\', \'b\'])\n        transitions: Dict mapping (state, symbol) to next state (e.g., {(0, \'a\'): 1})\n        initial_state: Initial state (e.g., 0)\n        accept_states: Set of accept states (e.g., {2})\n\n    Returns:\n        Tuple of (minimized_states, minimized_alphabet, minimized_transitions, minimized_initial, minimized_accept)\n    """\n    # Step 1: Initialize partitions (non-accept and accept states)\n    partition = [set(accept_states), set(states) - set(accept_states)]\n    \n    # Step 2: Refine partitions until no further changes\n    changed = True\n    while changed:\n        changed = False\n        new_partition = []\n        for group in partition:\n            # Split each group based on transitions\n            subgroups = {}\n            for state in group:\n                key = tuple(transitions.get((state, symbol), None) for symbol in alphabet)\n                subgroups.setdefault(key, set()).add(state)\n            if len(subgroups) > 1:\n                changed = True\n            new_partition.extend(subgroups.values())\n        partition = new_partition\n    \n    # Step 3: Build minimized FSM\n    state_map = {state: i for i, group in enumerate(partition) for state in group}\n    minimized_states = list(range(len(partition)))\n    minimized_transitions = {\n        (state_map[s], sym): state_map[transitions.get((s, sym), s)]\n        for s in states for sym in alphabet\n    }\n    minimized_initial = state_map[initial_state]\n    minimized_accept = {state_map[s] for s in accept_states}\n    \n    return minimized_states, alphabet, minimized_transitions, minimized_initial, minimized_accept'

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
    cases = [
        # Simple FSM
        (
            [0, 1, 2], ['a', 'b'],
            {(0, 'a'): 1, (0, 'b'): 0, (1, 'a'): 2, (1, 'b'): 0, (2, 'a'): 2, (2, 'b'): 2},
            0, {2}
        ),
        # Larger FSM
        (
            [0, 1, 2, 3, 4], ['a', 'b'],
            {(0, 'a'): 1, (0, 'b'): 0, (1, 'a'): 2, (1, 'b'): 3, (2, 'a'): 4, (2, 'b'): 0, (3, 'a'): 4, (3, 'b'): 0, (4, 'a'): 4, (4, 'b'): 4},
            0, {4}
        ),
        # FSM with unreachable states
        (
            [0, 1, 2, 3], ['a', 'b'],
            {(0, 'a'): 1, (0, 'b'): 0, (1, 'a'): 2, (1, 'b'): 0, (2, 'a'): 2, (2, 'b'): 2, (3, 'a'): 3, (3, 'b'): 3},
            0, {2}
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
