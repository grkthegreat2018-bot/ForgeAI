"""End-to-end StreamHead test: force airllm_streaming on a VRAM-fitting
checkpoint and verify the new pinned-ring streamer produces correct
output through ForgeEngine.generate (real weights, real decode loop).

  venv\\Scripts\\python.exe -X utf8 -u scripts/smoke_streamhead.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = "research/checkpoints/spark_x26_1p7b"
CFG = "spark_x26_1p7b"
PROMPT = "def reverse_words(s):\n    \"\"\"Reverse word order in s.\"\"\""


def main():
    from forge.engine.forge_engine import ForgeEngine

    engine = ForgeEngine.from_checkpoint(
        checkpoint=CKPT, config_name=CFG, auto_activate=False)

    # baseline
    t0 = time.time()
    ref = engine.generate(PROMPT, max_new_tokens=32, temperature=0.0,
                          finish_sentence=False)
    t_std = time.time() - t0
    print(f"[std    ] {t_std:5.1f}s | {ref[:100].replace(chr(10),' / ')}")

    # force REAL streaming setup: move model to CPU, load resident
    # shard, stream layers through the pinned ring (true e2e path)
    from forge.engine.airllm_streamer import AirLLMStreamer
    AirLLMStreamer._prepare_shards(engine)
    AirLLMStreamer._load_resident_shard(engine)
    t0 = time.time()
    out = engine.generate(PROMPT, max_new_tokens=32, temperature=0.0)
    t_str = time.time() - t0
    print(f"[stream ] {t_str:5.1f}s | {out[:100].replace(chr(10),' / ')}")
    print(f"\nmatch: {ref == out}  ({t_str/max(t_std,1e-9):.1f}x std time)")
    if ref != out:
        for i, (a, b) in enumerate(zip(ref, out)):
            if a != b:
                print(f"first diff @char {i}:")
                print("  ref:", repr(ref[max(0, i-30):i+30]))
                print("  out:", repr(out[max(0, i-30):i+30]))
                break
        else:
            print("prefix identical; length diff")
            print("  ref tail:", repr(ref[len(out):]))
            print("  out tail:", repr(out[len(ref):]))


if __name__ == "__main__":
    main()
