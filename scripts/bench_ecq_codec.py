"""Bench ECQ codec on a real checkpoint (R71/B2).

Measures actual packed bits/param + SQNR for ECQ k=16 at several entropy
penalties vs the minmax-int4-g32 baseline, plus true lossless bf16 b/param.

Usage: venv/Scripts/python.exe scripts/bench_ecq_codec.py [ckpt_dir] [n_tensors]
"""
import sys, time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge.quant.ecq_codec import (ecq_codebook, ecq_pack, ecq_unpack,
                                   minmax_int4_baseline, pack_bf16_lossless,
                                   sqnr_db, unpack_bf16_lossless)


def main():
    ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else "research/checkpoints/spark_x26_1p7b")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    tensors = {}
    for sh in sorted(ckpt.glob("model-*.safetensors")) + sorted(ckpt.glob("*.safetensors")):
        with safe_open(str(sh), framework="pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                tensors[k] = (str(sh), sl.get_shape())
        if len(tensors) > 200:
            break

    big = sorted([(k, p, s) for k, (p, s) in tensors.items() if int(np.prod(s)) > 4_000_000],
                 key=lambda x: -int(np.prod(x[2])))[:limit]

    print(f"{'tensor':<54}{'params':>11} | {'i4 SQNR':>7} {'i4 b/p':>6} |"
          f" {'ECQ SQNR':>8} {'ECQ b/p':>7} | {'lam=.1':>6} {'lam=.3':>6} | {'lz b/p':>6}")
    for name, path, shape in big:
        with safe_open(path, framework="pt") as f:
            t = f.get_slice(name)[...]
            u16 = t.view(torch.uint16).numpy().ravel()
            w = t.float().numpy().ravel()
        w = w[:8_000_000]
        u16 = u16[:8_000_000]
        n = w.size
        sq_i4, bits_i4 = minmax_int4_baseline(w)

        row = f"{name[-54:]:<54}{n:>11,} | {sq_i4:>7.2f} {bits_i4:>6.2f} |"
        for lam in (0.0, 0.1, 0.3):
            blob = ecq_pack(w, k=16, lam=lam)
            if lam == 0.0:
                out = ecq_unpack(blob)
                sq = sqnr_db(w, out)
                bp = len(blob) * 8 / n
                row += f" {sq:>8.2f} {bp:>7.2f} |"
            else:
                bp = len(blob) * 8 / n
                row += f" {bp:>6.2f}"
        lzb = pack_bf16_lossless(u16)
        row += f" | {len(lzb)*8/n:>6.2f}"
        print(row, flush=True)

    print("\nLegend: i4 = g32 minmax int4 SQNR@bits | ECQ = k16 ECVQ+range-coded "
          "(SQNR shown at lam=0) | lz = lossless bf16 byte-plane coder, actual b/p")


if __name__ == "__main__":
    main()
