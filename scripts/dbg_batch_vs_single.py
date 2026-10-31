"""Does batched functional decode preserve per-row semantics vs single?
And does static batched match on left-padded rows?
Compares per-row outputs of generate_batch vs single generate() for
prompts that end up left-padded inside the batch."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_lfm3_evolve import _load, _prompts, _pad_to_b  # noqa

static = "--static" in sys.argv
eng = _load(static=static, compile_=False, cache_w=True)
label = "static" if static else "func"
ps8 = _pad_to_b(_prompts(8), 8)
outs_b = eng.generate_batch(ps8, max_new_tokens=64,
                            temperature=0.0, top_k=1)
print(f"[{label}-batched B=8] done")
for i in (0, 3, 5):
    o = eng.generate(ps8[i], max_new_tokens=64,
                     temperature=0.0, top_k=1)
    same = o == outs_b[i]
    n = 0
    while n < min(len(o), len(outs_b[i])) and o[n] == outs_b[i][n]:
        n += 1
    print(f"[{label}] row {i}: single-vs-batch "
          f"{'IDENTICAL' if same else f'diverge at char {n}'}")
    if not same:
        print(f"   single: {o[n:n+80]!r}")
        print(f"   batch : {outs_b[i][n:n+80]!r}")
