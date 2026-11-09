"""ECQ — entropy-constrained quantization + arithmetic-coded weight codec.

R71/B2 (docs/R71_STREAMED_INTELLIGENCE_DESIGN.md). Two codecs:

  1. ecq_pack / ecq_unpack — lossy: per-tensor ECVQ codebook (Lloyd-Max with
     optional entropy penalty) + order-0 range coding of the index stream.
     Motivation (measured on spark_x26_1p7b): group-minmax int4 codes are
     ~incompressible (H=3.86/4 bits — minmax spreads codes uniformly by
     construction), floor ~4.89 b/param. A learned non-uniform codebook makes
     the index histogram non-uniform -> range coding reaches ~3.2-3.7 b/param.

  2. pack_bf16_lossless / unpack_bf16_lossless — lossless "ZIP for weights":
     byte-plane split of bf16 (hi=sign+exp+mantissa[9:2] has H~2.7 bits;
     lo mantissa byte ~7.97, stored raw) + range coding of the hi plane.
     Measured ideal ~10.7 b/param -> ~1.45-1.5x on any bf16 checkpoint.

Range coder: Subbotin carry-less variant (byte renormalization, no carry
propagation). Pure Python ~1-3M symbols/s — fine for offline packing.
"""
from __future__ import annotations

import json
import struct
from bisect import bisect_right

import numpy as np

_TOP = 1 << 24
_BOT = 1 << 16
_U32 = 0xFFFFFFFF


# ---------------- range coder ----------------

def rc_encode(symbols: np.ndarray, counts: np.ndarray, total: int) -> bytes:
    """Order-0 range-encode uint16 symbols. counts must sum to `total`,
    every present symbol >= 1. Returns raw byte stream (no header)."""
    cum = np.zeros(len(counts), dtype=np.int64)
    cum[1:] = np.cumsum(counts[:-1])
    low = 0
    rng = _U32
    out = bytearray()
    for s in symbols:
        r = rng // total
        low = (low + int(cum[s]) * r) & _U32
        rng = r * int(counts[s])
        while True:
            if ((low ^ (low + rng)) & _U32) < _TOP:
                pass
            elif rng < _BOT:
                rng = (-low) & (_BOT - 1)
            else:
                break
            out.append(low >> 24)
            low = (low << 8) & _U32
            rng = (rng << 8) & _U32
    out += low.to_bytes(4, "big")
    return bytes(out)


def rc_decode(stream: bytes, n: int, counts: np.ndarray, total: int) -> np.ndarray:
    cum = np.zeros(len(counts), dtype=np.int64)
    cum[1:] = np.cumsum(counts[:-1])
    cum_list = cum.tolist()
    pos = 4
    code = int.from_bytes(stream[:4], "big")
    low = 0            # absolute interval base, mirrors encoder `low`
    rng = _U32
    out = np.empty(n, dtype=np.uint16)
    for i in range(n):
        r = rng // total
        cnt = ((code - low) & _U32) // r    # wrapped position inside interval
        if cnt >= total:
            cnt = total - 1
        s = bisect_right(cum_list, cnt) - 1
        low = (low + cum_list[s] * r) & _U32
        rng = r * counts[s]
        # renorm: identical decisions to encoder (same low,rng)
        while True:
            if ((low ^ (low + rng)) & _U32) < _TOP:
                pass
            elif rng < _BOT:
                rng = (-low) & (_BOT - 1)
            else:
                break
            low = (low << 8) & _U32
            rng = (rng << 8) & _U32
            code = ((code << 8) | stream[pos]) & _U32
            pos += 1
        out[i] = s
    return out


def scale_hist(symbols: np.ndarray, n_sym: int, total: int = 1 << 15) -> np.ndarray:
    """Scale a symbol histogram to `total`, every present symbol >= 1."""
    hist = np.bincount(symbols, minlength=n_sym).astype(np.float64)
    p = hist / hist.sum()
    scaled = np.maximum(1, np.round(p * total)).astype(np.int64)
    # fix drift
    diff = int(scaled.sum()) - total
    order = np.argsort(-scaled)
    for idx in order:
        if diff == 0:
            break
        step = min(diff, scaled[idx] - 1) if diff > 0 else None
        if diff > 0 and step > 0:
            scaled[idx] -= step
            diff -= step
        elif diff < 0:
            scaled[idx] += 1
            diff += 1
    return scaled


# ---------------- ECVQ codebook ----------------

def ecq_codebook(x: np.ndarray, k: int = 16, lam: float = 0.0,
                 iters: int = 40, seed: int = 0) -> np.ndarray:
    """Entropy-constrained Lloyd-Max scalar codebook.

    lam=0  -> plain Lloyd-Max (already non-uniform usage -> compressible).
    lam>0  -> assignment cost = (x-c)^2 + lam * scale * (-log2 p_j);
              merges tail cells, lowers index entropy, costs distortion.
    """
    x = np.asarray(x, dtype=np.float64)
    qs = np.quantile(x, (np.arange(k) + 0.5) / k)
    c = qs.copy()
    # normalize lam to distortion units so O(0.01-0.3) is meaningful
    scale = x.var() / max(1.0, np.log2(k))
    for _ in range(iters):
        d = (x[:, None] - c[None, :]) ** 2
        if lam > 0:
            idx = d.argmin(1)
            p = np.bincount(idx, minlength=k) / x.size
            p = np.clip(p, 1e-12, 1.0)
            d = d + lam * scale * (-np.log2(p))[None, :]
        idx = d.argmin(1)
        new_c = c.copy()
        for j in range(k):
            m = idx == j
            if m.any():
                new_c[j] = x[m].mean()
        if np.allclose(new_c, c, atol=1e-12):
            break
        c = new_c
    return c


def ecq_assign(x: np.ndarray, cb: np.ndarray, lam: float = 0.0,
               scale: float = 1.0) -> np.ndarray:
    d = (x[:, None] - cb[None, :]) ** 2
    if lam > 0:
        idx0 = d.argmin(1)
        p = np.clip(np.bincount(idx0, minlength=len(cb)) / x.size, 1e-12, 1.0)
        d = d + lam * scale * (-np.log2(p))[None, :]
    return d.argmin(1).astype(np.uint16)


# ---------------- pack format ----------------
# header json: {"kind","dtype","shape","k","total"} + codebook f32 + hist u32

def _pack_payload(header: dict, arrays: list[tuple[str, bytes]]) -> bytes:
    h = json.dumps(header).encode()
    out = bytearray(struct.pack("<I", len(h)) + h)
    for name, blob in arrays:
        hb = name.encode()
        out += struct.pack("<H", len(hb)) + hb + struct.pack("<Q", len(blob)) + blob
    return bytes(out)


def _unpack_payload(blob: bytes):
    hlen = struct.unpack("<I", blob[:4])[0]
    header = json.loads(blob[4:4 + hlen])
    pos = 4 + hlen
    arrays = {}
    while pos < len(blob):
        nl = struct.unpack("<H", blob[pos:pos + 2])[0]
        pos += 2
        name = blob[pos:pos + nl].decode()
        pos += nl
        bl = struct.unpack("<Q", blob[pos:pos + 8])[0]
        pos += 8
        arrays[name] = blob[pos:pos + bl]
        pos += bl
    return header, arrays


def ecq_pack(w: np.ndarray, k: int = 16, lam: float = 0.0,
             codebook: np.ndarray | None = None) -> bytes:
    """Lossy pack: ECVQ indices + range-coded index stream."""
    w = np.asarray(w)
    x = w.astype(np.float64).ravel()
    if codebook is None:
        codebook = ecq_codebook(x, k=k, lam=lam)
    scale = x.var() / max(1.0, np.log2(k))
    idx = ecq_assign(x, codebook, lam=lam, scale=scale)
    counts = scale_hist(idx, k)
    stream = rc_encode(idx, counts, int(counts.sum()))
    header = {"kind": "ecq", "shape": list(w.shape), "dtype": str(w.dtype),
              "k": int(k), "total": int(counts.sum()), "n": int(x.size)}
    return _pack_payload(header, [
        ("cb", np.asarray(codebook, dtype=np.float32).tobytes()),
        ("hist", counts.astype(np.uint32).tobytes()),
        ("idx", stream),
    ])


def ecq_unpack(blob: bytes, dtype: np.dtype | None = None) -> np.ndarray:
    header, arrays = _unpack_payload(blob)
    assert header["kind"] == "ecq"
    k = header["k"]
    cb = np.frombuffer(arrays["cb"], dtype=np.float32).astype(np.float64)
    counts = np.frombuffer(arrays["hist"], dtype=np.uint32).astype(np.int64)
    idx = rc_decode(arrays["idx"], header["n"], counts, header["total"])
    w = cb[idx].reshape(header["shape"])
    return w.astype(dtype) if dtype is not None else w


def pack_bf16_lossless(w16: np.ndarray) -> bytes:
    """Lossless pack of a uint16-viewed bf16/fp16 tensor."""
    u = np.asarray(w16, dtype=np.uint16).ravel()
    hi = (u >> 8).astype(np.uint16)
    lo = (u & 0xFF).astype(np.uint8)
    counts = scale_hist(hi, 256)
    stream = rc_encode(hi, counts, int(counts.sum()))
    header = {"kind": "bf16lz", "shape": list(w16.shape), "n": int(u.size),
              "total": int(counts.sum())}
    return _pack_payload(header, [
        ("hist", counts.astype(np.uint32).tobytes()),
        ("hi", stream),
        ("lo", lo.tobytes()),
    ])


def unpack_bf16_lossless(blob: bytes) -> np.ndarray:
    header, arrays = _unpack_payload(blob)
    assert header["kind"] == "bf16lz"
    counts = np.frombuffer(arrays["hist"], dtype=np.uint32).astype(np.int64)
    hi = rc_decode(arrays["hi"], header["n"], counts, header["total"])
    lo = np.frombuffer(arrays["lo"], dtype=np.uint8)
    u = (hi.astype(np.uint32) << 8) | lo.astype(np.uint32)
    return u.astype(np.uint16).reshape(header["shape"])


# ---------------- measurement helpers ----------------

def sqnr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    approx = np.asarray(approx, dtype=np.float64)
    err = ref - approx
    return float(10 * np.log10((ref * ref).sum() / max(1e-30, (err * err).sum())))


def bits_per_param(packed_len: int, n: int) -> float:
    return packed_len * 8.0 / max(1, n)


def minmax_int4_baseline(x: np.ndarray, g: int = 32):
    """Reference: g32 minmax int4 -> SQNR + stored bits/param (raw, no entropy)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size // g * g
    w = x[:n].reshape(-1, g)
    mn = w.min(1, keepdims=True)
    mx = w.max(1, keepdims=True)
    q = np.round((w - mn) / np.clip(mx - mn, 1e-12, None) * 15)
    deq = mn + q / 15.0 * (mx - mn)
    sq = sqnr_db(w, deq)
    bits = 4.0 + 32.0 / g  # 4b code + 2 f16 scales per group
    return sq, bits
