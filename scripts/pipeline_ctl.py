"""Pause / resume / stop the distill + evolve pipeline (Windows).

Suspends whole process TREES via ntdll.NtSuspendProcess so the user
gets the machine back without losing any run state — gen batches,
evolve evals, and training all freeze in place and resume exactly
where they stopped. GPU compute frees instantly on pause; note VRAM
stays allocated while processes are suspended (use `stop` + each
script's --resume to reclaim memory too).

Usage:
    python scripts/pipeline_ctl.py status
    python scripts/pipeline_ctl.py pause
    python scripts/pipeline_ctl.py resume
    python scripts/pipeline_ctl.py stop          # terminate gracefully
    python scripts/pipeline_ctl.py pause --match "fn_evolve|sft_train"
"""
from __future__ import annotations

import argparse
import ctypes
import re
import sys
import time

import psutil

DEFAULT_MATCH = (r"python.*?(fn_evolve|evolve_api_sweep|"
                 r"gen_qwen_distill|gen_llm_targets|sft_train|"
                 r"finetune_proposer)")

_SUSPEND_RESUME = 0x0800          # PROCESS_SUSPEND_RESUME
_TERMINATE = 0x0001
_QUERY = 0x0400


def _nt(pid: int, fn_name: str) -> bool:
    k32 = ctypes.windll.kernel32
    nt = ctypes.windll.ntdll
    h = k32.OpenProcess(_SUSPEND_RESUME | _TERMINATE | _QUERY,
                        False, pid)
    if not h:
        return False
    try:
        fn = getattr(nt, fn_name)
        return fn(ctypes.c_void_p(h)) == 0
    finally:
        k32.CloseHandle(h)


def _match_procs(pat: str) -> list[psutil.Process]:
    rx = re.compile(pat)
    me = {psutil.Process().pid, *(
        p.pid for p in psutil.Process().parents())}
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.pid in me:
            continue
        try:
            cmd = " ".join(p.cmdline() or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if rx.search(cmd):
            out.append(p)
    return out


def _tree(procs: list[psutil.Process]) -> list[psutil.Process]:
    """Children first (suspend order); de-duped."""
    seen, order = set(), []
    def add(p):
        try:
            kids = p.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            kids = []
        for k in kids:
            if k.pid not in seen:
                seen.add(k.pid)
                order.append(k)
        if p.pid not in seen:
            seen.add(p.pid)
            order.append(p)
    for p in procs:
        add(p)
    return order


def _state(p: psutil.Process) -> str:
    try:
        return p.status()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "pause", "resume", "stop"])
    ap.add_argument("--match", default=DEFAULT_MATCH)
    args = ap.parse_args()

    roots = _match_procs(args.match)
    if args.cmd == "status":
        if not roots:
            print("[ctl] no pipeline processes")
            return 0
        for p in roots:
            try:
                kids = len(p.children(recursive=True))
                print(f"[ctl] pid={p.pid} status={_state(p)} "
                      f"children={kids} "
                      f"cmd={' '.join(p.cmdline() or [])[:110]}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return 0

    order = _tree(roots)
    if args.cmd == "pause":
        n = 0
        for p in order:                       # children first
            if _state(p) == "stopped":
                continue                      # already suspended
            if _nt(p.pid, "NtSuspendProcess"):
                n += 1
        print(f"[ctl] suspended {n} processes (state kept; VRAM held)")
        return 0

    if args.cmd == "resume":
        n = 0
        for p in reversed(order):             # parents first
            if _nt(p.pid, "NtResumeProcess"):
                n += 1
        print(f"[ctl] resumed {n} processes")
        return 0

    # stop: terminate roots first, wait, then force-kill leftovers
    for p in roots:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    deadline = time.time() + 20
    for p in order:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except (psutil.TimeoutExpired, psutil.NoSuchProcess):
            pass
    left = [p for p in order if p.is_running()]
    for p in left:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    print(f"[ctl] terminated {len(roots)} pipeline roots "
          f"({len(left)} force-killed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
