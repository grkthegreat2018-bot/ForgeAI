"""R&D Round 25, Algo 1v3: AdditiveFP4-scaled parameter sweep.

Sweep n_codes and n_codebooks to find the optimal configuration.
Focus on EXACT bpw matches against IRI-FP4 x2 (1.066) and x3 (1.600).

Key configs to test (bpw ≈ 0.53 + n_cb * (log2(n_codes)/8 + 1/32)):
  1cb(16)  → 1.061 bpw ← matches IRI x2 (1.066)
  2cb(16)  → 1.592 bpw ← matches IRI x3 (1.600)
  1cb(32)  → 1.186 bpw
  1cb(64)  → 1.311 bpw
  1cb(128) → 1.436 bpw
  2cb(8)   → 1.343 bpw
  2cb(32)  → 2.124 bpw
  3cb(8)   → 1.748 bpw
  4cb(8)   → 2.154 bpw

Also test: per-block scale strategy (absmax/6 vs L2-norm vs MSE-optimal).
"""
import math
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn.functional as F
from safetensors import safe_open

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
CKPT = r"D:\windsurf\ForgeAI\research\checkpoints\ForgeLM_V2_Light.safetensors"

from forge.engine.quant.novel_quant import _optimal_fp4_scale, quantize_iri_fp4
from forge.engine.quant.nvfp4_quant import _FP4_BOUNDARIES, _FP4_MAGNITUDES


def frob_err(ref, q):
    return (ref - q).norm().item() / ref.norm().clamp(min=1e-8).item()

def sqnr(ref, q):
    s = (ref ** 2).sum().item()
    n = ((ref - q) ** 2).sum().item()
    return 999.0 if n < 1e-30 else 10.0 * math.log10(s / n)


def kmeans_1d(data, n_codes, n_iters=20, seed=0):
    g = torch.Generator(device=data.device).manual_seed(seed)
    perm = torch.randperm(data.numel(), generator=g, device=data.device)[:n_codes]
    levels = data[perm].clone()
    for _ in range(n_iters):
        d = (data.unsqueeze(-1) - levels.unsqueeze(0)) ** 2
        a = d.argmin(dim=1)
        new = torch.zeros_like(levels)
        cnt = torch.zeros(n_codes, device=data.device, dtype=data.dtype)
        new.index_add_(0, a, data)
        cnt.index_add_(0, a, torch.ones_like(data))
        ne = cnt > 0
        new[ne] /= cnt[ne]
        levels = torch.where(ne, new, levels)
    return levels.sort().values


def quantize_additive_fp4_scaled(W, block_size=32, n_codes=8, n_codebooks=2,
                                  scale_mode="absmax6", kmeans_iters=25):
    """AdditiveFP4-scaled with configurable per-block scale strategy."""
    out_f, in_f = W.shape
    pad = (block_size - in_f % block_size) % block_size
    wp = F.pad(W, (0, pad)) if pad > 0 else W
    in_p = wp.shape[1]
    n_blocks = in_p // block_size

    wb = wp.view(out_f, n_blocks, block_size)
    scale = _optimal_fp4_scale(wb)
    s = scale.clamp(min=1e-12)
    w_norm = wb / s
    idx = torch.searchsorted(_FP4_BOUNDARIES.to(W.device), w_norm.abs()).clamp(0, 7)
    mag = _FP4_MAGNITUDES.to(W.device)[idx]
    w_dq = torch.sign(w_norm) * mag * s
    residual = wb - w_dq
    acc = w_dq.clone()

    for k in range(n_codebooks):
        res_absmax = residual.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        if scale_mode == "absmax6":
            res_scale = res_absmax / 6.0
        elif scale_mode == "absmax4":
            res_scale = res_absmax / 4.0
        elif scale_mode == "absmax3":
            res_scale = res_absmax / 3.0
        elif scale_mode == "l2norm":
            res_scale = residual.norm(dim=-1, keepdim=True).clamp(min=1e-12) / (math.sqrt(block_size) * 3.0)
        else:
            res_scale = res_absmax / 6.0
        res_norm = residual / res_scale

        res_flat = res_norm.reshape(-1).contiguous()
        N = res_flat.numel()
        max_km = 200_000
        if max_km < N:
            g = torch.Generator(device=W.device).manual_seed(42 + k)
            sub = res_flat[torch.randperm(N, generator=g, device=W.device)[:max_km]]
        else:
            sub = res_flat
        levels = kmeans_1d(sub, n_codes, kmeans_iters, seed=42 + k)

        d = (res_norm.unsqueeze(-1) - levels.reshape(1, 1, 1, -1)) ** 2
        assignments = d.argmin(dim=-1)
        res_q = levels[assignments] * res_scale
        acc = acc + res_q
        residual = residual - res_q

    w_dq_final = acc.view(out_f, in_p)[:, :in_f].contiguous()
    n = W.numel()
    bytes_fp4 = out_f * ((in_p + 1) // 2) + out_f * n_blocks * 1 + out_f * 4
    bits_per_cb = math.log2(n_codes)
    bytes_cb_data = n_codebooks * n * bits_per_cb / 8.0
    bytes_cb_scales = n_codebooks * out_f * n_blocks * 1
    bytes_tables = n_codebooks * n_codes * 4
    bpw = (bytes_fp4 + bytes_cb_data + bytes_cb_scales + bytes_tables) / n
    return w_dq_final, bpw


def benchmark_weight(label, W_key):
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        W = f.get_tensor(W_key).to(DEV).to(DTYPE)

    print(f"\n  {label}: {W.shape} std={W.std():.5f} kurt={((W-W.mean())**4).mean()/(W.std()**4):.2f}")

    configs = [
        # Exact bpw matches against IRI
        ("IRI-FP4 x2",              lambda W: (quantize_iri_fp4(W, 32, 2), 1.066)),
        ("IRI-FP4 x3",              lambda W: (quantize_iri_fp4(W, 32, 3), 1.600)),
        # Exact matches: 1cb(16)→1.061, 2cb(16)→1.592
        ("AddFP4 1cb(16) absmax6",  lambda W: quantize_additive_fp4_scaled(W, 32, 16, 1, "absmax6")),
        ("AddFP4 2cb(16) absmax6",  lambda W: quantize_additive_fp4_scaled(W, 32, 16, 2, "absmax6")),
        # Scale mode sweep at 2cb(8) (1.343 bpw)
        ("AddFP4 2cb(8) absmax6",   lambda W: quantize_additive_fp4_scaled(W, 32, 8, 2, "absmax6")),
        ("AddFP4 2cb(8) absmax4",   lambda W: quantize_additive_fp4_scaled(W, 32, 8, 2, "absmax4")),
        ("AddFP4 2cb(8) absmax3",   lambda W: quantize_additive_fp4_scaled(W, 32, 8, 2, "absmax3")),
        ("AddFP4 2cb(8) l2norm",    lambda W: quantize_additive_fp4_scaled(W, 32, 8, 2, "l2norm")),
        # n_codes sweep at 1 codebook
        ("AddFP4 1cb(8)",           lambda W: quantize_additive_fp4_scaled(W, 32, 8, 1)),
        ("AddFP4 1cb(32)",          lambda W: quantize_additive_fp4_scaled(W, 32, 32, 1)),
        ("AddFP4 1cb(64)",          lambda W: quantize_additive_fp4_scaled(W, 32, 64, 1)),
        ("AddFP4 1cb(128)",         lambda W: quantize_additive_fp4_scaled(W, 32, 128, 1)),
        # n_codebooks sweep at n_codes=8
        ("AddFP4 3cb(8)",           lambda W: quantize_additive_fp4_scaled(W, 32, 8, 3)),
        ("AddFP4 4cb(8)",           lambda W: quantize_additive_fp4_scaled(W, 32, 8, 4)),
        # Higher codebooks with fewer codes
        ("AddFP4 4cb(4)",           lambda W: quantize_additive_fp4_scaled(W, 32, 4, 4)),
        ("AddFP4 6cb(4)",           lambda W: quantize_additive_fp4_scaled(W, 32, 4, 6)),
    ]

    print(f"  {'Algorithm':<28} {'bpw':>6} {'SQNR(dB)':>10} {'delta':>8}")
    print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*8}")

    iri_x2_sq = None
    iri_x3_sq = None
    for name, fn in configs:
        try:
            W_q, bpw = fn(W)
            sq = sqnr(W, W_q)
            if "IRI-FP4 x2" in name: iri_x2_sq = sq
            if "IRI-FP4 x3" in name: iri_x3_sq = sq
            # Delta vs nearest IRI
            if iri_x2_sq and iri_x3_sq and "IRI" not in name:
                if bpw < 1.3:
                    delta = f"{sq - iri_x2_sq:+.2f}"
                else:
                    delta = f"{sq - iri_x3_sq:+.2f}"
            else:
                delta = ""
            print(f"  {name:<28} {bpw:>6.3f} {sq:>10.2f} {delta:>8}")
        except Exception as e:
            print(f"  {name:<28} FAILED: {str(e)[:50]}")

    if DEV.type == "cuda":
        torch.cuda.empty_cache()


def main():
    print("=" * 80)
    print("  R25 Algo 1v3: AdditiveFP4-scaled SWEEP on V9-1.2B")
    print("=" * 80)

    weights = [
        ("FFN gate L2",   "blocks.2.ffn.w_gate.weight"),
        ("FFN down L2",   "blocks.2.ffn.w_down.weight"),
        ("Attn Q L2",     "blocks.2.attn.q_proj.weight"),
        ("FFN gate L14",  "blocks.14.ffn.w_gate.weight"),
    ]
    for label, key in weights:
        try:
            benchmark_weight(label, key)
        except Exception as e:
            print(f"\n  SKIP {label}: {e}")


if __name__ == "__main__":
    main()
