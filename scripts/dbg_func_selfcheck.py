"""Is functional generate_batch greedy output self-consistent?
Establishes the nondeterminism noise floor for the static-vs-func gate."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_lfm3_evolve import _load, _prompts, _pad_to_b  # noqa

eng = _load(static=False, compile_=False, cache_w=True)
for B in (8, 16):
    ps = _pad_to_b(_prompts(B), B)
    a = eng.generate_batch(ps, max_new_tokens=96,
                           temperature=0.0, top_k=1)
    b = eng.generate_batch(ps, max_new_tokens=96,
                           temperature=0.0, top_k=1)
    same = sum(x == y for x, y in zip(a, b))
    print(f"[func-vs-func B={B}] {same}/{B} identical")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            n = 0
            while n < min(len(x), len(y)) and x[n] == y[n]:
                n += 1
            print(f"  row {i}: common prefix {n} chars "
                  f"| f[{n}:{n+40}]={x[n:n+40]!r} "
                  f"| s={y[n:n+40]!r}")
