"""R&D Round 25, Algo 1: AdditiveFP4 — AQLM-inspired additive codebook quantization.

NOVEL IDEA:
  AQLM represents weight blocks as a sum of codebook vector lookups:
    W[block] ≈ sum_{k=1}^{K} C_k[code_k]
  This captures weight CORRELATIONS within a block (vector quantization),
  which scalar quantizers (FP4, INT4, ternary) cannot.

  Our novel twist: use the EXISTING FP4 codebook as the FIRST codebook (free,
  already optimal for the dense mass per R14/R15), then learn ADDITIONAL
  codebooks on the RESIDUAL via k-means. This is cross-domain:
    AQLM (additive vector quantization) + FP4 (existing scalar base) + IRI-FP4
    (iterative residual, but with LEARNED codebooks instead of reusing FP4).

  Hypothesis: the FP4 quantization residual has a different distribution than
  the original weights (it's the rounding error, more uniform/structured).
  A codebook LEARNED for that residual distribution should compress it more
  efficiently than reusing the FP4 codebook (which IRI-FP4 does).

TWO VARIANTS:
  1. AdditiveFP4-scalar: FP4 base + K learned SCALAR codebooks on residual
     - Each codebook: n_codes levels, log2(n_codes) bits/element
     - Like IRI-FP4 but with k-means-optimized levels instead of FP4's {0..6}
  2. AdditiveFP4-vector: FP4 base + K learned VECTOR codebooks on residual blocks
     - Each codebook: (n_codes, block_size) vectors, log2(n_codes) bits/block
     - True AQLM-style: captures correlations in the residual

BIT BUDGET MATH (hand-computed per §F):
  FP4 base: 0.53 bytes/w (4.24 bits/w) — packed 4-bit + FP8 scales + global
  Scalar codebook (n=16, 4 bits): +0.5 bytes/w per codebook
  Vector codebook (n=256, block=8, 8 bits/block): +0.125 bytes/w per codebook

  Targets:
    FP4 + 1 scalar codebook (16):  0.53 + 0.50 = 1.03 bpw (vs IRI x2 1.07)
    FP4 + 2 scalar codebooks (16): 0.53 + 1.00 = 1.53 bpw (vs IRI x3 1.60)
    FP4 + 1 vector codebook (256, b=8): 0.53 + 0.125 = 0.655 bpw
    FP4 + 2 vector codebooks (256, b=8): 0.53 + 0.25 = 0.78 bpw

Runs on real V9 weights. Compares against IRI-FP4 at matched bpw.
"""
import math
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn.functional as F
from safetensors import safe_open

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
CKPT = r"D:\windsurf\ForgeAI\research\checkpoints\ForgeLM_V2_Light.safetensors"

from forge.engine.quant.novel_quant import (
    _optimal_fp4_scale,
    quantize_iri_fp4,
)
from forge.engine.quant.nvfp4_quant import (
    _FP4_BOUNDARIES,
    _FP4_MAGNITUDES,
)

# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def frob_err(ref, q):
    return (ref - q).norm().item() / ref.norm().clamp(min=1e-8).item()

def sqnr(ref, q):
    signal = (ref ** 2).sum().item()
    noise = ((ref - q) ** 2).sum().item()
    if noise < 1e-30:
        return 999.0
    return 10.0 * math.log10(signal / noise)

def output_err(W_ref, W_q, x):
    return frob_err(x @ W_ref.T, x @ W_q.T)


# ──────────────────────────────────────────────────────────────────────────
# K-means codebook learning (GPU-accelerated, Lloyd's algorithm)
# ──────────────────────────────────────────────────────────────────────────

def kmeans_codebook(data: torch.Tensor, n_codes: int, n_iters: int = 20,
                    seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Learn a codebook via k-means.

    Args:
        data: (N, D) float tensor — N data points of dimension D
        n_codes: number of codebook entries
        n_iters: k-means iterations

    Returns:
        codebook: (n_codes, D) float tensor
        assignments: (N,) long tensor — index of nearest codebook entry per point
    """
    N, _D = data.shape
    g = torch.Generator(device=data.device).manual_seed(seed)
    # Initialize: random subset of data points
    perm = torch.randperm(N, generator=g, device=data.device)[:n_codes]
    codebook = data[perm].clone()

    for _ in range(n_iters):
        # Assign: nearest codebook entry (L2)
        # dist[i,j] = ||data[i] - codebook[j]||^2
        # = ||data[i]||^2 - 2*data[i]@codebook[j].T + ||codebook[j]||^2
        d_sq = data @ codebook.T  # (N, n_codes)
        d_sq = d_sq * 2 - (data ** 2).sum(-1, keepdim=True) - (codebook ** 2).sum(-1)
        assignments = d_sq.argmin(dim=1)  # (N,)

        # Update: mean of assigned points
        new_codebook = torch.zeros_like(codebook)
        counts = torch.zeros(n_codes, device=data.device, dtype=data.dtype)
        new_codebook.index_add_(0, assignments, data)
        counts.index_add_(0, assignments, torch.ones(N, device=data.device, dtype=data.dtype))
        # Avoid empty clusters: keep old centroid for empty clusters
        nonempty = counts > 0
        new_codebook[nonempty] /= counts[nonempty].unsqueeze(-1)
        codebook = torch.where(nonempty.unsqueeze(-1), new_codebook, codebook)

    # Final assignment
    d_sq = data @ codebook.T * 2 - (data ** 2).sum(-1, keepdim=True) - (codebook ** 2).sum(-1)
    assignments = d_sq.argmin(dim=1)
    return codebook, assignments


# ──────────────────────────────────────────────────────────────────────────
# Variant 1: AdditiveFP4-scalar
# FP4 base + K learned SCALAR codebooks on residual
# ──────────────────────────────────────────────────────────────────────────

def quantize_additive_fp4_scalar(W: torch.Tensor, block_size: int = 32,
                                  n_codes: int = 16, n_codebooks: int = 2,
                                  kmeans_iters: int = 20) -> tuple[torch.Tensor, float]:
    """AdditiveFP4-scalar: FP4 base + learned scalar codebooks on residual.

    Round 0: FP4 quantize with AS-FP4 (MSE-optimal) scale
    Round k: Learn a scalar codebook (n_codes levels) on the residual via
             k-means, quantize the residual to nearest codebook level, subtract.

    Storage:
      FP4: 0.53 bytes/w
      Each scalar codebook: log2(n_codes) bits / 8 bytes/w + n_codes * 4 bytes
                            (codebook table, negligible for large weights)

    Returns: (dequantized_weight, bytes_per_weight)
    """
    out_f, in_f = W.shape
    pad = (block_size - in_f % block_size) % block_size
    wp = F.pad(W, (0, pad)) if pad > 0 else W
    in_p = wp.shape[1]
    n_blocks = in_p // block_size

    # Round 0: FP4 with AS-FP4 scale
    wb = wp.view(out_f, n_blocks, block_size)
    scale = _optimal_fp4_scale(wb)
    s = scale.clamp(min=1e-12)
    w_norm = wb / s
    abs_norm = w_norm.abs()
    idx = torch.searchsorted(_FP4_BOUNDARIES.to(W.device), abs_norm).clamp(0, 7)
    mag = _FP4_MAGNITUDES.to(W.device)[idx]
    w_q = torch.sign(w_norm) * mag
    w_dq = w_q * s  # dequantized FP4
    residual = wb - w_dq  # residual after FP4

    acc = w_dq.clone()

    # Rounds 1..K: learn scalar codebook on residual
    for k in range(n_codebooks):
        # Residual is (out_f, n_blocks, block_size) — flatten to (N, 1) for scalar k-means
        res_flat = residual.reshape(-1, 1).contiguous()
        # Subsample for k-means speed (k-means on 16M points is slow)
        N = res_flat.shape[0]
        max_kmeans_N = 200_000
        if max_kmeans_N < N:
            g = torch.Generator(device=W.device).manual_seed(42 + k)
            sub_idx = torch.randperm(N, generator=g, device=W.device)[:max_kmeans_N]
            res_sub = res_flat[sub_idx]
        else:
            res_sub = res_flat
        codebook, _ = kmeans_codebook(res_sub, n_codes, kmeans_iters, seed=42 + k)
        # Quantize ALL residuals to nearest codebook entry
        # dist: (N, n_codes)
        d_sq = res_flat ** 2 - 2 * res_flat @ codebook.T + codebook.T ** 2
        assignments = d_sq.argmin(dim=1)  # (N,)
        res_q = codebook[assignments].reshape_as(residual)
        acc = acc + res_q
        residual = residual - res_q

    w_dq_final = acc.view(out_f, in_p)[:, :in_f].contiguous()

    # Storage calculation
    n = W.numel()
    # FP4: packed (0.5 bytes/w) + FP8 scales + global scale
    bytes_fp4 = out_f * ((in_p + 1) // 2) + out_f * n_blocks * 1 + out_f * 4
    # Scalar codebooks: log2(n_codes) bits per element per codebook
    bits_per_codebook = math.log2(n_codes)
    bytes_codebooks = n_codebooks * n * bits_per_codebook / 8.0
    # Codebook tables: n_codes * 4 bytes * n_codebooks (negligible for large W)
    bytes_tables = n_codebooks * n_codes * 4
    bpw = (bytes_fp4 + bytes_codebooks + bytes_tables) / n

    return w_dq_final, bpw


# ──────────────────────────────────────────────────────────────────────────
# Variant 2: AdditiveFP4-vector (true AQLM-style)
# FP4 base + K learned VECTOR codebooks on residual blocks
# ──────────────────────────────────────────────────────────────────────────

def quantize_additive_fp4_vector(W: torch.Tensor, block_size: int = 32,
                                  vq_block: int = 8, n_codes: int = 256,
                                  n_codebooks: int = 2,
                                  kmeans_iters: int = 20) -> tuple[torch.Tensor, float]:
    """AdditiveFP4-vector: FP4 base + learned VECTOR codebooks on residual.

    Round 0: FP4 quantize with AS-FP4 scale (scalar, captures dense mass)
    Round k: Learn a VECTOR codebook (n_codes entries of dimension vq_block)
             on vq_block-sized residual blocks via k-means. Each block is
             quantized to the nearest codebook vector. This captures
             CORRELATIONS in the residual — the key AQLM advantage.

    Storage:
      FP4: 0.53 bytes/w
      Each vector codebook: log2(n_codes) bits per vq_block elements
                            = log2(n_codes) / vq_block bytes/w
                            + n_codes * vq_block * 4 bytes (table, negligible)

    Returns: (dequantized_weight, bytes_per_weight)
    """
    out_f, in_f = W.shape
    pad = (block_size - in_f % block_size) % block_size
    wp = F.pad(W, (0, pad)) if pad > 0 else W
    in_p = wp.shape[1]
    n_blocks = in_p // block_size

    # Round 0: FP4 with AS-FP4 scale
    wb = wp.view(out_f, n_blocks, block_size)
    scale = _optimal_fp4_scale(wb)
    s = scale.clamp(min=1e-12)
    w_norm = wb / s
    abs_norm = w_norm.abs()
    idx = torch.searchsorted(_FP4_BOUNDARIES.to(W.device), abs_norm).clamp(0, 7)
    mag = _FP4_MAGNITUDES.to(W.device)[idx]
    w_q = torch.sign(w_norm) * mag
    w_dq = w_q * s
    residual = wb - w_dq  # (out_f, n_blocks, block_size)

    acc = w_dq.clone()

    # Reshape residual into vq_block-sized vectors for vector quantization
    # residual: (out_f, n_blocks, block_size) → (out_f * n_blocks * block_size / vq_block, vq_block)
    assert block_size % vq_block == 0, f"block_size {block_size} must be divisible by vq_block {vq_block}"
    res_vecs = residual.reshape(-1, vq_block).contiguous()  # (N_vecs, vq_block)

    for k in range(n_codebooks):
        # Subsample for k-means speed
        N = res_vecs.shape[0]
        max_kmeans_N = 100_000
        if max_kmeans_N < N:
            g = torch.Generator(device=W.device).manual_seed(42 + k)
            sub_idx = torch.randperm(N, generator=g, device=W.device)[:max_kmeans_N]
            res_sub = res_vecs[sub_idx]
        else:
            res_sub = res_vecs
        codebook, _ = kmeans_codebook(res_sub, n_codes, kmeans_iters, seed=42 + k)
        # Quantize ALL residual vectors
        d_sq = (res_vecs ** 2).sum(-1, keepdim=True) - 2 * res_vecs @ codebook.T + (codebook ** 2).sum(-1)
        assignments = d_sq.argmin(dim=1)  # (N_vecs,)
        res_q = codebook[assignments]  # (N_vecs, vq_block)
        acc = acc + res_q.reshape_as(residual)
        res_vecs = res_vecs - res_q  # next residual

    w_dq_final = acc.view(out_f, in_p)[:, :in_f].contiguous()

    # Storage
    n = W.numel()
    bytes_fp4 = out_f * ((in_p + 1) // 2) + out_f * n_blocks * 1 + out_f * 4
    bits_per_vq = math.log2(n_codes)
    bytes_vq = n_codebooks * (n // vq_block) * bits_per_vq / 8.0
    bytes_tables = n_codebooks * n_codes * vq_block * 4
    bpw = (bytes_fp4 + bytes_vq + bytes_tables) / n

    return w_dq_final, bpw


# ──────────────────────────────────────────────────────────────────────────
# Benchmark on real V9 weights
# ──────────────────────────────────────────────────────────────────────────

def benchmark_weight(label, W_key, x_dim=None):
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        W = f.get_tensor(W_key).to(DEV).to(DTYPE)

    m, n = W.shape
    print(f"\n{'='*95}")
    print(f"  {label}: {W.shape} ({m*n:,} params)")
    print(f"  W: std={W.std():.5f}, kurt={((W-W.mean())**4).mean()/(W.std()**4):.2f}")
    print(f"{'='*95}")

    if x_dim is None:
        x_dim = n
    g = torch.Generator(device=DEV).manual_seed(42)
    x = torch.randn(1, 64, x_dim, generator=g, device=DEV, dtype=DTYPE) * 0.5

    configs = [
        # (name, fn, target_bpw)
        ("IRI-FP4 x2 (baseline)",     lambda W: (quantize_iri_fp4(W, 32, 2), 1.066), 1.07),
        ("IRI-FP4 x3 (baseline)",     lambda W: (quantize_iri_fp4(W, 32, 3), 1.600), 1.60),
        ("AddFP4-scalar 1cb(16)",     lambda W: quantize_additive_fp4_scalar(W, 32, 16, 1), 1.03),
        ("AddFP4-scalar 2cb(16)",     lambda W: quantize_additive_fp4_scalar(W, 32, 16, 2), 1.53),
        ("AddFP4-scalar 3cb(16)",     lambda W: quantize_additive_fp4_scalar(W, 32, 16, 3), 2.03),
        ("AddFP4-scalar 2cb(32)",     lambda W: quantize_additive_fp4_scalar(W, 32, 32, 2), 1.78),
        ("AddFP4-scalar 2cb(64)",     lambda W: quantize_additive_fp4_scalar(W, 32, 64, 2), 2.03),
        ("AddFP4-vector 1cb(256,b8)", lambda W: quantize_additive_fp4_vector(W, 32, 8, 256, 1), 0.66),
        ("AddFP4-vector 2cb(256,b8)", lambda W: quantize_additive_fp4_vector(W, 32, 8, 256, 2), 0.78),
        ("AddFP4-vector 1cb(256,b4)", lambda W: quantize_additive_fp4_vector(W, 32, 4, 256, 1), 0.78),
        ("AddFP4-vector 2cb(256,b4)", lambda W: quantize_additive_fp4_vector(W, 32, 4, 256, 2), 1.03),
        ("AddFP4-vector 3cb(256,b4)", lambda W: quantize_additive_fp4_vector(W, 32, 4, 256, 3), 1.28),
        ("AddFP4-vector 2cb(64,b8)",  lambda W: quantize_additive_fp4_vector(W, 32, 8, 64, 2), 0.72),
        ("AddFP4-vector 2cb(1024,b8)",lambda W: quantize_additive_fp4_vector(W, 32, 8, 1024, 2), 0.91),
    ]

    print(f"  {'Algorithm':<28} {'bpw':>6} {'frob_err':>10} {'SQNR(dB)':>10} {'out_err':>10} {'time':>7}")
    print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")

    for name, fn, _target_bpw in configs:
        try:
            t0 = time.time()
            result = fn(W)
            dt = time.time() - t0
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], float):
                W_q, bpw = result
            else:
                W_q, bpw = result
            fe = frob_err(W, W_q)
            sq = sqnr(W, W_q)
            oe = output_err(W, W_q, x)
            print(f"  {name:<28} {bpw:>6.3f} {fe:>10.4f} {sq:>10.2f} {oe:>10.4f} {dt:>6.1f}s")
        except Exception as e:
            print(f"  {name:<28} FAILED: {e}")

    if DEV.type == "cuda":
        torch.cuda.empty_cache()


def main():
    print("=" * 95)
    print("  R&D ROUND 25, Algo 1: AdditiveFP4 (AQLM-inspired) on ForgeLM V9-1.2B")
    print(f"  Device: {DEV}")
    print("=" * 95)

    weights = [
        ("FFN gate (layer 2)",   "blocks.2.ffn.w_gate.weight",  2048),
        ("FFN down (layer 2)",   "blocks.2.ffn.w_down.weight",  8192),
        ("Attn Q_proj (layer 2)","blocks.2.attn.q_proj.weight", 2048),
        ("FFN gate (layer 14)",  "blocks.14.ffn.w_gate.weight", 2048),
    ]

    for label, key, xd in weights:
        try:
            benchmark_weight(label, key, xd)
        except Exception as e:
            print(f"\n  SKIP {label}: {e}")

    print("\n  KEY: AdditiveFP4 must beat IRI-FP4 at matched bpw.")
    print("  Win condition: higher SQNR at same-or-lower bytes/weight.")


if __name__ == "__main__":
    main()
