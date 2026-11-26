"""Supervised wrapper for the RSI self-play loop.

A CUDA device-side assert (OOB gather / index_select) poisons the whole
process context — every subsequent CUDA call fails, so an in-loop
try/except turns the remaining epochs into instant no-ops (observed
2026-11-24: epoch-1 assert -> 99 dead epochs, "LOOP COMPLETE" with zero
work done). The only recovery boundary is a fresh process.

This supervisor runs ``forge.self_play.infinite_loop`` in epoch chunks
inside a subprocess; a crashed chunk loses at most its in-flight epoch
(counted via the "PHASE 1: AZR SELF-PLAY (epoch N)" markers in the
child log). Promoted checkpoints are already durable on disk
(``checkpoint_dir/ForgeLM_<TAG>_SP*.safetensors``), so the next chunk
resumes from the newest durable checkpoint automatically.

Usage:
    python scripts/rsi_supervisor.py \
        --checkpoint research/checkpoints/lfm3_distill_v10_full \
        --config lfm3_1p7b --staging-dir D:\\forge_spool \
        --epochs 100 --chunk 10 -- <extra infinite_loop args>
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time

_EPOCH_MARK = re.compile(r"AZR SELF-PLAY \(epoch (\d+)\)")
# Only prints on a REAL epoch completion (success rate + examples) —
# crashed epochs never reach it, so this is the honest counter.
_EPOCH_DONE = re.compile(r"^\s*Epoch \d+: success=", re.MULTILINE)


def _newest_best(base_ckpt: str, ckpt_dir: str, cfg_tag: str) -> str:
    """Newest durable promoted checkpoint, else the run's base."""
    pats = glob.glob(os.path.join(
        ckpt_dir, f"ForgeLM_{cfg_tag}_SP*.safetensors"))
    if not pats:
        return base_ckpt
    return max(pats, key=os.path.getmtime)


def _completed_epochs(log_path: str, rc: int) -> int:
    """Epochs fully completed in a child run.

    Counts "Epoch N: success=" summaries — printed only when an epoch
    truly finished. A per-epoch exception (recoverable python error) or
    a poisoned-context run (every epoch instantly crashing to LOOP
    COMPLETE with rc=0) both yield 0. Phase markers print at epoch
    START and are kept only as a fallback for logs lacking summaries.
    """
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return 0
    done = len(_EPOCH_DONE.findall(txt))
    if done == 0 and rc != 0:
        # Fallback: crashed mid-first-epoch — count markers minus the
        # in-flight one.
        done = max(0, len(_EPOCH_MARK.findall(txt)) - 1)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=10,
                    help="epochs per subprocess (crash blast radius)")
    ap.add_argument("--staging-dir", default="")
    ap.add_argument("--checkpoint-dir", default="research/checkpoints")
    ap.add_argument("--log-dir", default=".devin/tmp",
                    help="per-chunk child logs land here")
    ap.add_argument("--max-consecutive-fails", type=int, default=3,
                    help="abort after N chunks that complete 0 epochs")
    ap.add_argument("loop_args", nargs=argparse.REMAINDER,
                    help="extra args forwarded to infinite_loop")
    args = ap.parse_args()

    cfg_tag = args.config.replace("forgelm_", "").upper() or "MODEL"
    os.makedirs(args.log_dir, exist_ok=True)
    done_total = 0
    zero_streak = 0
    chunk_i = 0

    while done_total < args.epochs:
        n = min(args.chunk, args.epochs - done_total)
        best = _newest_best(args.checkpoint, args.checkpoint_dir, cfg_tag)
        cmd = [sys.executable, "-u", "-m", "forge.self_play.infinite_loop",
               "--checkpoint", best, "--config", args.config,
               "--epochs", str(n)]
        if args.staging_dir:
            cmd += ["--staging-dir", args.staging_dir]
        cmd += [a for a in args.loop_args if a != "--"]
        chunk_i += 1
        log_path = os.path.join(
            args.log_dir, f"rsi_chunk{chunk_i:03d}.log")
        print(f"[supervisor] chunk {chunk_i}: {n} epochs "
              f"(target {done_total + n}/{args.epochs}) ckpt={best} "
              f"log={log_path}", flush=True)
        with open(log_path, "w", encoding="utf-8") as lf:
            rc = subprocess.call(
                cmd, stdout=lf, stderr=subprocess.STDOUT)
        done_here = _completed_epochs(log_path, rc)
        if rc == 0 and done_here == n:
            pass  # honest full completion
        done_total += done_here
        print(f"[supervisor] chunk {chunk_i} rc={rc}: "
              f"{done_here}/{n} epochs completed "
              f"({done_total}/{args.epochs} total)", flush=True)
        if done_here == 0:
            zero_streak += 1
            if zero_streak >= args.max_consecutive_fails:
                print("[supervisor] aborting: "
                      f"{zero_streak} chunks completed 0 epochs — "
                      "persistent crash, not a transient assert",
                      flush=True)
                return 2
        else:
            zero_streak = 0
        if rc != 0:
            print("[supervisor] respawning from newest durable "
                  "checkpoint", flush=True)
    print(f"[supervisor] done — {done_total} epochs completed",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
