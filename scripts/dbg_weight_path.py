"""Replay-time comparison: int4-fused GEMM vs bf16-cached cuBLAS at B=16.
Flags must be set BEFORE batch-graph capture — a recorded graph keeps
whichever kernels were live at capture time. Two fresh engine loads."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_lfm3_evolve import _load, _prompts, _pad_to_b  # noqa

B = 16
ps = _pad_to_b(_prompts(B), B)


def run(fused: bool, cache_w: bool, label: str):
    eng = _load(static=True, compile_=False, cache_w=cache_w)
    model = eng.model
    qmods = [m for m in model.modules()
             if hasattr(m, "_fused_gemv") and hasattr(m, "_dequantize_weight")]
    for m in qmods:
        m._fused_gemv = fused
        m._cache_w = cache_w
    # warmup captures the batch graph with the CURRENT kernel flags
    eng.generate_batch(ps, max_new_tokens=8, temperature=0.0, top_k=1)
    torch.cuda.synchronize()

    bcg = model._bcg
    bs = eng._decode_state['batch'][B]
    tag = next(t for b in bs['buckets']
               if (t := f"{bs['prefix']}kv{b}") in bcg._decode_graphs)
    g = bcg._decode_graphs[tag]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(60):
        g.replay()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 60 * 1000

    # full-pipeline timing (prefill + decode)
    t0 = time.time()
    outs = eng.generate_batch(ps, max_new_tokens=64, temperature=0.0,
                              top_k=1)
    wall = time.time() - t0
    ntok = sum(len(o) for o in outs[1]) if isinstance(outs, tuple) \
        else sum(len(o) for o in outs)
    print(f"[{label}] replay {ms:.2f} ms/step  "
          f"batch {wall:.2f}s {ntok} tok -> {ntok/wall:.0f} tok/s  "
          f"vram={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
    del eng
    torch.cuda.empty_cache()


run(fused=False, cache_w=True, label="cached-bf16")
run(fused=True, cache_w=False, label="int4-fused")
