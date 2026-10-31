"""Bench: sequential vs self-speculative decode throughput.

Trains a small FluxLM on a repetitive+natural mixed stream, then
times generate() at speculate=0 (pipelined) vs K in {4,8,12} — also
verifies bit-identical outputs.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

import torch

from forge.model.flux import FluxConfig, FluxLM

V = 4096


def main():
    # mixed corpus: periodic runs + a random-ish tail so accept rate
    # is realistic (not 100%)
    rng = torch.Generator().manual_seed(0)
    seq = (([7, 8, 9, 10] * 120)
           + torch.randint(0, V, (2400,), generator=rng).tolist()
           + ([7, 8, 9, 10] * 60))
    cfg = {"vocab_size": V, "device": "cuda", "cuda_primary": True,
               "orders": (1, 2, 3, 4, 6, 8), "use_knn": False,
               "sparse_taps": (), "class_orders": (), "word_orders": (),
               "hybrid_taps": ()}
    m = FluxLM(FluxConfig(**cfg))
    m.ingest(seq, tag="t")
    prompt = [7, 8, 9]

    for spec in (0, 4, 8, 12):
        # warm then timed
        m.generate(prompt, max_new=16, temperature=0.0,
                   speculate=spec)
        t0 = time.perf_counter()
        out = m.generate(prompt, max_new=128, temperature=0.0,
                         speculate=spec)
        dt = time.perf_counter() - t0
        print(f"spec={spec:2d}: {len(out)/dt:7.1f} tok/s "
              f"({dt*1e3/len(out):.2f} ms/tok)")
        if spec:
            ref = m.generate(prompt, max_new=128,
                             temperature=0.0)
            print(f"    identical: {out == ref}")


if __name__ == "__main__":
    main()
