"""Measure lossless/near-lossless weight-compression headroom on a real checkpoint.

R&D questions:
  1. What does zlib get on raw bf16 weight bytes? (baseline "zip")
  2. Byte-plane split (hi/lo bytes separated, like BitShuffle) — helps entropy coders?
  3. Shannon entropy floor of each byte plane -> ideal bits/param.
  4. int4 group-quant codes: histogram entropy + zlib -> effective bits/param after
     entropy coding (the "entropy-coded quantized weights" lane noted as NOT
     IMPLEMENTED in novel_quant_r50.py).
  5. Cross-layer XOR delta of same-suffix tensors (lossless delta coding).

Usage: venv/Scripts/python.exe scripts/bench_lossless_weights.py [ckpt_dir]
"""
import json, math, sys, zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def entropy_bits(byte_arr: np.ndarray) -> float:
    """Shannon entropy in bits/byte for a uint8 array."""
    if byte_arr.size == 0:
        return 0.0
    counts = np.bincount(byte_arr.ravel(), minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def zratio(raw: bytes, level: int = 9) -> float:
    return len(zlib.compress(raw, level)) / max(1, len(raw))


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.float().numpy()


def main():
    ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else "research/checkpoints/spark_x26_1p7b")
    shards = sorted(ckpt.glob("model-*.safetensors")) or sorted(ckpt.glob("*.safetensors"))
    assert shards, f"no safetensors in {ckpt}"

    tensors = {}
    for sh in shards:
        with safe_open(str(sh), framework="pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                tensors[k] = (str(sh), str(sl.get_dtype()), sl.get_shape())

    # biggest 12 bf16/fp16 tensors > 4M params + embeddings + norms sample
    is16 = lambda dt: dt in ("BF16", "F16", "torch.bfloat16", "torch.float16")
    big = sorted(
        [(k, d, s) for k, (d, dt, s) in tensors.items()
         if is16(dt) and int(np.prod(s)) > 4_000_000],
        key=lambda x: -int(np.prod(x[2])),
    )[:12]

    print(f"{'tensor':<58}{'params':>10}{'zlib%':>7}{'split%':>7}"
          f"{'H(hi)':>6}{'H(lo)':>6}{'ideal b/p':>10}")
    tot_params = tot_raw = tot_zlib = tot_split = 0
    for name, path, shape in big:
        with safe_open(path, framework="pt") as f:
            u16 = f.get_slice(name)[...].view(torch.uint16).numpy().ravel()
        n = u16.size
        # sample <=8M params for zlib speed
        idx = np.random.RandomState(0).choice(n, min(n, 8_000_000), replace=False)
        samp = u16[np.sort(idx)]
        lo = (samp & 0xFF).astype(np.uint8)
        hi = (samp >> 8).astype(np.uint8)
        raw = samp.tobytes()
        z = zratio(raw)
        zsplit = (len(zlib.compress(lo.tobytes(), 9)) +
                  len(zlib.compress(hi.tobytes(), 9))) / len(raw)
        h_hi, h_lo = entropy_bits(hi), entropy_bits(lo)
        ideal = h_hi + h_lo
        tot_params += n; tot_raw += 2 * n
        tot_zlib += z * 2 * n; tot_split += zsplit * 2 * n
        print(f"{name[-58:]:<58}{n:>10,}{z*100:>6.1f}%{zsplit*100:>6.1f}%"
              f"{h_hi:>6.2f}{h_lo:>6.2f}{ideal:>9.2f}/16")

    print(f"\n== totals: {tot_params/1e9:.2f}B params sampled, {tot_raw/1e9:.2f}GB raw =="
          f"  zlib {tot_zlib/tot_raw*100:.1f}%  split-zlib {tot_split/tot_raw*100:.1f}%")

    # ---- int4 group-32 minmax on 3 biggest FFN/attn tensors ----
    print(f"\n{'int4 g32 tensor':<58}{'raw q':>6}{'H(q)':>6}{'zlib':>6}{'eff b/p':>8}")
    for name, path, shape in big[:4]:
        with safe_open(path, framework="pt") as f:
            w = to_f32(f.get_slice(name)[...]).ravel()[:8_000_000]
        g = 32
        w = w[: w.size // g * g].reshape(-1, g)
        wmin = w.min(1, keepdims=True); wmax = w.max(1, keepdims=True)
        q = np.round((w - wmin) / np.clip(wmax - wmin, 1e-8, None) * 15).astype(np.uint8)
        packed = q[:, 0::2] | (q[:, 1::2] << 4)
        h_q = entropy_bits(packed)  # per packed byte = 2 codes
        zq = zratio(packed.tobytes())
        # bits/param: packed zlib bytes*8 / params + scale overhead (2*bf16/32 = 1 bit)
        eff = zq * 8 * 2 + 1.0
        print(f"{name[-58:]:<58}{'4.0':>6}{h_q:>6.2f}{zq*100:>5.1f}%{eff:>7.2f}")

    # ---- cross-layer XOR delta (lossless) on same-suffix consecutive layers ----
    print("\n== cross-layer XOR delta ==")
    by_suffix = {}
    for name, (path, dt, shape) in tensors.items():
        if ".layers." in name and is16(dt):
            parts = name.split(".layers.")
            if len(parts) == 2:
                ln, suffix = parts[1].split(".", 1)
                if ln.isdigit():
                    by_suffix.setdefault(suffix, {})[int(ln)] = (name, path, shape)
    suffix = max(by_suffix, key=lambda s: len(by_suffix[s]))
    layers = sorted(by_suffix[suffix])
    pairs = list(zip(layers[:-1], layers[1:]))[:3]
    for l0, l1 in pairs:
        n0, p0, s0 = by_suffix[suffix][l0]
        n1, p1, s1 = by_suffix[suffix][l1]
        if s0 != s1 or int(np.prod(s0)) < 1_000_000:
            continue
        with safe_open(p0, framework="pt") as f:
            a = f.get_slice(n0)[...].view(torch.uint16).numpy().ravel()[:4_000_000]
        with safe_open(p1, framework="pt") as f:
            b = f.get_slice(n1)[...].view(torch.uint16).numpy().ravel()[:4_000_000]
        xd = (a ^ b)
        hi = (xd >> 8).astype(np.uint8); lo = (xd & 0xFF).astype(np.uint8)
        ideal = entropy_bits(hi) + entropy_bits(lo)
        zd = (len(zlib.compress(hi.tobytes(), 9)) + len(zlib.compress(lo.tobytes(), 9))) / (a.size * 2)
        print(f"  {suffix[:46]:<48} L{l0}->L{l1}  ideal {ideal:>5.2f}b/p  splitzlib {zd*100:.1f}%")

    # ---- embedding near-duplicate rows ----
    emb = [k for k in tensors if "embed" in k and "weight" in k]
    if emb:
        name = emb[0]
        with safe_open(tensors[name][0], framework="pt") as f:
            e = to_f32(f.get_slice(name)[...])
        e = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-8, None)
        n_exact = e.shape[0] - len({r.tobytes() for r in e})
        sim = e @ e[np.random.RandomState(0).choice(e.shape[0], 2000, replace=False)].T
        np.fill_diagonal(sim[:, :], -1)
        near = (sim > 0.999).any(1).mean()
        print(f"\nembed {emb[0]} {e.shape}: exact-dup rows {n_exact}, cos>0.999 rows {near*100:.1f}%")


if __name__ == "__main__":
    main()

