"""Profile the BatchedDecoding step: replay time vs python/sampling
overhead. Times graph replay alone on the live engine state, then the
full generate_batch, to attribute the per-step cost."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_lfm3_evolve import _load, _prompts, _pad_to_b  # noqa

B = 16
eng = _load(static=True, compile_="--compile" in sys.argv, cache_w=True)
ps = _pad_to_b(_prompts(B), B)
# warm capture
eng.generate_batch(ps, max_new_tokens=8, temperature=0.0, top_k=1)
torch.cuda.synchronize()

model = eng.model
bcg = model._bcg
bs = eng._decode_state['batch'][B]
cache, m_buf, prefix = bs['cache'], bs['mask'], bs['prefix']
graphs = bcg._decode_graphs
bufs = bcg._static_buffers
tag = next(t for b in bs['buckets']
           if (t := f"{prefix}kv{b}") in graphs)
g, buf = graphs[tag], bufs[tag]

# 1) replay alone
for _ in range(3):
    g.replay()
torch.cuda.synchronize()
t0 = time.time()
N = 100
for _ in range(N):
    g.replay()
torch.cuda.synchronize()
rep_ms = (time.time() - t0) / N * 1000
print(f"[replay B={B}] {rep_ms:.2f} ms/step")

# 2) sampling block alone (mimic loop: argmax on [B,V])
V = 128000
lg = torch.randn(B, V, device='cuda', dtype=torch.bfloat16)
temp = torch.full((B, 1), 0.9, device='cuda', dtype=lg.dtype)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    nl = lg / temp
    tv, ti = nl.topk(80, dim=-1)
    nl = torch.full_like(nl, float('-inf')).scatter_(-1, ti, tv)
    probs = torch.nn.functional.softmax(nl, -1).clamp(min=1e-10)
    probs = torch.nan_to_num(probs, nan=1.0 / V)
    nt = torch.multinomial(probs, num_samples=1).squeeze(-1)
    fl = nt == 519
    _ = fl.any().item()
torch.cuda.synchronize()
samp_ms = (time.time() - t0) / N * 1000
print(f"[sampling B={B} t0.9] {samp_ms:.2f} ms/step")

# 3) greedy sampling
t0 = time.time()
for _ in range(N):
    nt = lg.argmax(-1)
    _ = (nt == 519).any().item()
torch.cuda.synchronize()
gr_ms = (time.time() - t0) / N * 1000
print(f"[greedy+sync B={B}] {gr_ms:.2f} ms/step")

# 4) eos check sync cost alone
fl = torch.zeros(B, dtype=torch.bool, device='cuda')
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    _ = fl.any().item()
torch.cuda.synchronize()
print(f"[.any().item() sync] {(time.time()-t0)/N*1000:.3f} ms")

# 5) full generate_batch wall for comparison
t0 = time.time()
outs = eng.generate_batch(ps, max_new_tokens=96, temperature=0.8,
                          top_p=0.95, top_k=80,
                          seeds=[1000 + i for i in range(B)])
torch.cuda.synchronize()
dt = time.time() - t0
n = sum(len(eng.tokenizer(o, add_special_tokens=False).input_ids)
        for o in outs)
print(f"[generate_batch B={B}] {dt:.2f}s {n} tok "
      f"-> {n/dt:.0f} tok/s agg, ~{dt/(n/B)*1000:.1f} ms/step")
