"""probe_weight_entropy.py — information floor of trained weights.

Any storage scheme claiming IDENTICAL output must preserve the weight
information content. Floor = empirical entropy of the tensors. Measures on a
real trained checkpoint (proposer_mla_moe ~1GB bf16):

  per tensor-class: byte entropy -> lossless ratio bound = 8/H
                    lzma-9 actual ratio on raw bytes (sampled)
                    bf16 field entropy: sign | exponent(8b) | mantissa(7b)
                    first-difference byte entropy along dim1 (delta coding)
                    fraction of exactly-zero weights (sparsity floor)
"""
import lzma
import numpy as np
import torch
from safetensors import safe_open

CKPT = r"forge/evolution/proposer_mla_moe/model.safetensors"
MAX_BYTES_LZMA = 4 << 20   # 4MB per class sample

def h_bits_per_byte(arr_bytes: np.ndarray) -> float:
    cnt = np.bincount(arr_bytes.ravel(), minlength=256).astype(np.float64)
    p = cnt[cnt > 0] / arr_bytes.size
    return float(-(p * np.log2(p)).sum())  # bits per byte

def class_of(name: str) -> str:
    for pat, cls in [("embed", "embed"), ("head", "head"),
                     ("q_proj", "attn"), ("k_proj", "attn"), ("v_proj", "attn"),
                     ("out_proj", "attn"), ("kv_down", "gla"), ("k_up", "gla"),
                     ("v_up", "gla"), ("k_rope", "gla"), ("router", "router"),
                     ("experts", "moe_ffn"), ("w1", "ffn"), ("w2", "ffn"),
                     ("w3", "ffn"), ("norm", "norm"), ("ln_", "norm")]:
        if pat in name:
            return cls
    return "other"

stats = {}
with safe_open(CKPT, framework="pt") as f:
    names = list(f.keys())
    total_bytes = 0
    for n in names:
        t = f.get_tensor(n)
        total_bytes += t.numel() * t.element_size()
        cls = class_of(n)
        st = stats.setdefault(cls, dict(bytes=[], lzma_in=0, lzma_out=0,
                                        zeros=0, count=0,
                                        sign=[], exp=[], man=[], diff=[]))
        u8 = t.detach().cpu().view(torch.uint8).numpy().ravel()
        st["bytes"].append(u8)
        st["zeros"] += int((u8 == 0).sum()); st["count"] += u8.size
        # bf16 fields (uint16 view: [s eeeeeee mmmmmmm])
        if t.element_size() == 2:
            u16 = t.detach().cpu().view(torch.uint16).numpy().astype(np.uint32)
            st["sign"].append((u16 >> 15).astype(np.uint8).ravel())
            st["exp"].append(((u16 >> 7) & 0xFF).astype(np.uint8).ravel())
            st["man"].append((u16 & 0x7F).astype(np.uint8).ravel())
            if t.ndim == 2:
                d = (u16[:, 1:].astype(np.int32) - u16[:, :-1].astype(np.int32)) & 0xFFFF
                st["diff"].append((d & 0xFF).astype(np.uint8).ravel())
        # lzma sample
        if st["lzma_in"] < MAX_BYTES_LZMA:
            st["lzma_in"] += u8.size
            st["lzma_out"] += len(lzma.compress(u8.tobytes(), preset=6))

print(f"checkpoint: {CKPT}  ({total_bytes/1e9:.2f} GB, {len(names)} tensors)\n")
hdr = f"{'class':9s} {'MB':>7s} {'H(byte)':>8s} {'bound':>6s} {'lzma':>6s} " \
      f"{'H(sign)':>7s} {'H(exp)':>7s} {'H(man)':>7s} {'H(diff)':>7s} {'zeros%':>6s}"
print(hdr); print("-" * len(hdr))
agg_bytes = 0.0; agg_bits = 0.0
for cls, st in sorted(stats.items()):
    b = np.concatenate(st["bytes"]); nbytes = b.size
    hb = h_bits_per_byte(b)
    agg_bytes += nbytes; agg_bits += hb * nbytes
    lz = st["lzma_in"] / max(st["lzma_out"], 1)
    hs = he = hm = hd = float("nan")
    if st["exp"]:
        s = np.concatenate(st["sign"]); ps = s.mean()
        hs = -(ps * np.log2(ps + 1e-12) + (1 - ps) * np.log2(1 - ps + 1e-12))
        he = h_bits_per_byte(np.concatenate(st["exp"]))
        hm = h_bits_per_byte(np.concatenate(st["man"]))  # bits, max=7
        if st["diff"]:
            hd = h_bits_per_byte(np.concatenate(st["diff"]))
    print(f"{cls:9s} {nbytes/1e6:7.1f} {hb:8.3f} {8/hb:6.3f}x {lz:6.3f}x "
          f"{hs:7.3f} {he:7.3f} {hm:7.3f}/7 {hd:7.3f} {100*st['zeros']/st['count']:6.2f}")
print("-" * len(hdr))
print(f"ALL      {agg_bytes/1e6:7.1f} {agg_bits/agg_bytes:8.3f} "
      f"{8/(agg_bits/agg_bytes):6.3f}x   <- global lossless floor (byte-wise)")
print("\nNote: byte-wise H ignores correlations -> true floor is LOWER bound;")
print("lzma exploits some structure -> measured ratio is achievable-today bound.")
