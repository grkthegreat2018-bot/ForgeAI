"""End-to-end LoopThink smoke test through ForgeEngine.generate.

Runs the same prompts (a) standard and (b) FORGE_LOOPTHINK=gates —
verifies the wired _prefill path produces coherent text and reports
the LoopThink log lines (iters used / stop reason).

  set FORGE_LOOPTHINK=8,20,3,0.15,research/checkpoints/loopgates_v1.pt
  venv\\Scripts\\python.exe scripts/smoke_loopthink.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = "research/checkpoints/spark_x26_1p7b"
CFG = "spark_x26_1p7b"
PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "def reverse_words(s):\n    \"\"\"Reverse word order in s.\"\"\"",
    "Explain what a hash collision is in one paragraph.",
]


def run(engine, tag):
    outs = []
    for p in PROMPTS[:int(os.environ.get("SMOKE_N", len(PROMPTS)))]:
        t0 = time.time()
        text = engine.generate(
            p, max_new_tokens=int(os.environ.get("SMOKE_TOKENS", 64)),
            temperature=0.0)
        dt = time.time() - t0
        outs.append((p, text, dt))
        print(f"[{tag}] {dt:.1f}s | {text[:120].replace(chr(10),' / ')}")
    return outs


def main():
    from forge.engine.forge_engine import ForgeEngine

    # auto_activate=False: skip torch.compile/CUDA-graph activation —
    # inductor on Windows can compile for 30+ min or hang with 0% GPU
    # util (pytorch #128424/#181638). Eager engine isolates the
    # LoopThink seam itself.
    engine = ForgeEngine.from_checkpoint(
        checkpoint=CKPT, config_name=CFG, auto_activate=False)

    std = run(engine, "std")

    # Flip on LoopThink in-process: set env + clear the cached config.
    os.environ["FORGE_LOOPTHINK"] = (
        "8,20,3,0.15,research/checkpoints/loopgates_v1.pt")
    if hasattr(engine, "_lt_cfg"):
        del engine._lt_cfg
    lt = run(engine, "loopthink")

    for (p, s, _), (_, l, _) in zip(std, lt):
        print(f"\n=== {p[:50]}\n  std: {s[:100]!r}\n  lt : {l[:100]!r}")


if __name__ == "__main__":
    main()
