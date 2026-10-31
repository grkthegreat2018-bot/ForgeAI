"""R&D Round 26v2: Ternary + Low-Rank Correction (TernLC) + bug fixes.

KEY INSIGHT from R26v1: ternary {-1,0,+1} is near-optimal for post-training
at sub-1.58 bits. Binary loses because the zero captures the Gaussian dense
mass near 0. To beat BitNet, the correction must add negligible bits.

NOVEL APPROACH: Ternary + Low-Rank Correction (TernLC)
  W ≈ T * scale + A @ B
  - T: ternary weights (0.2 bytes/w, base-3 packed) — the BitNet base
  - A @ B: low-rank correction (rank r, float16)
  - The correction captures the SYSTEMATIC error of ternary quantization
  - Ternary error is NOT random — it's correlated with weight value
  - A low-rank matrix can capture this structure efficiently

Storage math (hand-computed per §F):
  Ternary: 0.2 bytes/w (base-3 packed)
  Correction: (out*r + r*in) * 2 bytes / (out*in)
  For 8192x2048, r=16: (131072 + 32768) * 2 / 16M = 0.0196 bytes/w
  Total: 0.22 bytes/w — 10% above BitNet, but with MUCH better quality

  For r=8: 0.01 bytes/w overhead → total 0.21 bytes/w — 5% above BitNet
  For r=4: 0.005 bytes/w overhead → total 0.205 bytes/w — 2.5% above BitNet
  For r=32: 0.039 bytes/w overhead → total 0.239 bytes/w — 20% above BitNet

The correction is computed via SVD of the ternary error: E = W - T*scale.
A @ B = SVD_top_r(E). This is optimal low-rank approximation.

ALSO: Fixed TernPrep broadcasting bug + BinCB OOM (batched distance).

ALSO TESTS: TernaryPack with per-CHANNEL scale (vs per-tensor).
  Per-channel scale: out_f * 4 bytes overhead = 0.0005 bytes/w for 8192x2048.
  Quality: much better than per-tensor (each output channel gets its own scale).
  This is what BitNet QAT does — but we apply it post-training.
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

from tests.fixtures.keys.quantization.bitnet_residual_key import ternary_quantize


def frob_err(ref, q):
    return (ref - q).norm().item() / ref.norm().clamp(min=1e-8).item()

def sqnr(ref, q):
    s = (ref ** 2).sum().item()
    n = ((ref - q) ** 2).sum().item()
    return 999.0 if n < 1e-30 else 10.0 * math.log10(s / n)

def output_err(W_ref, W_q, x):
    return frob_err(x @ W_ref.T, x @ W_q.T)


# ════════════════════════════════════════════════════════════════════════════
# TernaryPack with per-channel scale (key improvement over per-tensor)
# ════════════════════════════════════════════════════════════════════════════

def quantize_ternary_per_channel(W: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Ternary with per-OUTPUT-CHANNEL scale (vs BitNet's per-tensor).

    BitNet uses a single absmean/0.7 scale for the entire weight matrix.
    Per-channel scale gives each output channel its own scale, adapting
    to per-channel variance. This is what BitNet QAT converges to, but
    we apply it post-training.

    Storage: ternary (0.2 bytes/w base-3 packed) + per-channel scales
    (out_f * 4 bytes = negligible for large weights).
    """
    out_f, _in_f = W.shape
    # Per-output-row scale
    scales = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8) / 0.7  # (out, 1)
    w_norm = W / scales
    w_t = torch.sign(w_norm) * (w_norm.abs() > 0.5).float()
    w_dq = w_t * scales

    n = W.numel()
    # Ternary: 0.2 bytes/w (base-3 packed) + scales: out_f * 4 bytes
    bytes_ternary = n * 0.2
    bytes_scales = out_f * 4
    bpw = (bytes_ternary + bytes_scales) / n
    return w_dq, bpw


def quantize_ternary_per_channel_int8(W: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Ternary with per-channel scale, int8 storage (like BitNet but per-channel)."""
    out_f, _in_f = W.shape
    scales = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8) / 0.7
    w_norm = W / scales
    w_t = torch.sign(w_norm) * (w_norm.abs() > 0.5).float()
    w_dq = w_t * scales
    n = W.numel()
    bpw = (n * 1 + out_f * 4) / n  # int8 + per-channel scale
    return w_dq, bpw


# ════════════════════════════════════════════════════════════════════════════
# TernLC: Ternary + Low-Rank Correction
# ════════════════════════════════════════════════════════════════════════════

def quantize_ternlc(W: torch.Tensor, rank: int = 16,
                    per_channel: bool = True) -> tuple[torch.Tensor, float]:
    """Ternary + Low-Rank Correction.

    W ≈ T * scale + A @ B
    - T: ternary weights (base-3 packed, 0.2 bytes/w)
    - A @ B: top-rank SVD of the ternary error (float16)

    Args:
        W: (out, in) weight tensor
        rank: rank of the low-rank correction
        per_channel: use per-channel ternary scale (better quality)
    """
    out_f, in_f = W.shape

    # Step 1: Ternary quantize
    if per_channel:
        scales = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8) / 0.7
    else:
        scales = W.abs().mean().clamp(min=1e-8) / 0.7
    w_norm = W / scales
    w_t = torch.sign(w_norm) * (w_norm.abs() > 0.5).float()
    w_ternary = w_t * scales  # dequantized ternary

    # Step 2: Compute error
    error = W - w_ternary  # (out, in)

    # Step 3: SVD of error → top-rank low-rank correction
    U, S, Vh = torch.linalg.svd(error.float(), full_matrices=False)
    r = min(rank, S.shape[0])
    A = (U[:, :r] * S[:r].unsqueeze(0)).to(torch.float16)  # (out, r)
    B = Vh[:r, :].to(torch.float16)  # (r, in)

    # Reconstruct
    w_dq = w_ternary + (A.to(torch.float32) @ B.to(torch.float32))

    # Storage
    n = W.numel()
    bytes_ternary = n * 0.2  # base-3 packed
    bytes_scales = out_f * 4 if per_channel else 4  # per-channel or per-tensor
    bytes_A = out_f * r * 2  # float16
    bytes_B = r * in_f * 2  # float16
    bpw = (bytes_ternary + bytes_scales + bytes_A + bytes_B) / n
    return w_dq, bpw


# ════════════════════════════════════════════════════════════════════════════
# TernLC with alternating refinement (optimize ternary + correction jointly)
# ════════════════════════════════════════════════════════════════════════════

def quantize_ternlc_refined(W: torch.Tensor, rank: int = 16,
                            n_iters: int = 5) -> tuple[torch.Tensor, float]:
    """TernLC with alternating refinement.

    Iterate:
    1. Ternary quantize W - A@B (residual after correction)
    2. SVD of (W - T*scale) → update A@B
    This jointly optimizes the ternary base and low-rank correction.
    """
    out_f, in_f = W.shape
    scales = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8) / 0.7

    # Initialize: ternary on W, correction = SVD of error
    w_norm = W / scales
    w_t = torch.sign(w_norm) * (w_norm.abs() > 0.5).float()
    w_ternary = w_t * scales
    error = W - w_ternary
    U, S, Vh = torch.linalg.svd(error.float(), full_matrices=False)
    r = min(rank, S.shape[0])
    A = (U[:, :r] * S[:r].unsqueeze(0))
    B = Vh[:r, :]

    for _ in range(n_iters):
        # Step 1: Re-ternarize the residual after correction
        residual = W - A @ B
        r_norm = residual / scales
        w_t = torch.sign(r_norm) * (r_norm.abs() > 0.5).float()
        w_ternary = w_t * scales

        # Step 2: Update correction = SVD of new error
        error = W - w_ternary
        U, S, Vh = torch.linalg.svd(error.float(), full_matrices=False)
        A = U[:, :r] * S[:r].unsqueeze(0)
        B = Vh[:r, :]

    w_dq = w_ternary + A @ B

    # Storage (same as basic TernLC)
    n = W.numel()
    bytes_ternary = n * 0.2
    bytes_scales = out_f * 4
    bytes_A = out_f * r * 2
    bytes_B = r * in_f * 2
    bpw = (bytes_ternary + bytes_scales + bytes_A + bytes_B) / n
    return w_dq, bpw


# ════════════════════════════════════════════════════════════════════════════
# Fixed TernPrep (broadcasting bug fix)
# ════════════════════════════════════════════════════════════════════════════

def quantize_ternary_preprocess_fixed(W: torch.Tensor, block_size: int = 128
                                      ) -> tuple[torch.Tensor, float]:
    """Ternary + per-block (scale, shift) preprocessing. Fixed broadcasting."""
    out_f, in_f = W.shape
    pad = (block_size - in_f % block_size) % block_size
    wp = F.pad(W, (0, pad)) if pad > 0 else W
    in_p = wp.shape[1]
    n_blocks = in_p // block_size

    blocks = wp.view(out_f, n_blocks, block_size)

    # Per-block mean and std
    block_mean = blocks.mean(dim=-1, keepdim=True)  # (out, n_blocks, 1)
    block_std = blocks.std(dim=-1, keepdim=True).clamp(min=1e-8)

    # Grid search over shift and scale
    shift_opts = torch.tensor([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
                              device=W.device, dtype=W.dtype)
    scale_opts = torch.tensor([0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3],
                              device=W.device, dtype=W.dtype)

    best_err = torch.full((out_f, n_blocks), float('inf'), device=W.device, dtype=W.dtype)
    best_shift = torch.zeros(out_f, n_blocks, 1, device=W.device, dtype=W.dtype)
    best_scale = torch.ones(out_f, n_blocks, 1, device=W.device, dtype=W.dtype)

    centered = blocks - block_mean
    absmax = centered.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    for si in range(7):
        shift = block_mean + shift_opts[si] * block_std * 0.5  # (out, n_blocks, 1)
        for ci in range(7):
            scale = (absmax / 0.7) * scale_opts[ci]  # (out, n_blocks, 1)
            centered_s = blocks - shift
            normalized = centered_s / scale.clamp(min=1e-12)
            w_t = torch.sign(normalized) * (normalized.abs() > 0.5).float()
            w_dq = w_t * scale + shift
            err = ((blocks - w_dq) ** 2).mean(dim=-1)  # (out, n_blocks)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_shift[better] = shift[better]
            best_scale[better] = scale[better]

    # Apply best
    centered_s = blocks - best_shift
    normalized = centered_s / best_scale.clamp(min=1e-12)
    w_t = torch.sign(normalized) * (normalized.abs() > 0.5).float()
    w_dq = w_t * best_scale + best_shift

    w_dq_final = w_dq.view(out_f, in_p)[:, :in_f].contiguous()

    n = W.numel()
    bytes_ternary = n * 0.2
    bytes_scales = out_f * n_blocks * 8  # 2 * float32 per block
    bpw = (bytes_ternary + bytes_scales) / n
    return w_dq_final, bpw


# ════════════════════════════════════════════════════════════════════════════
# Benchmark
# ════════════════════════════════════════════════════════════════════════════

def q_bitnet(W):
    w_t, scale = ternary_quantize(W)
    return w_t * scale, 0.25

def q_bitnet_per_channel(W):
    return quantize_ternary_per_channel_int8(W)

def benchmark_weight(label, W_key, x_dim=None):
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        W = f.get_tensor(W_key).to(DEV).to(DTYPE)

    _m, n = W.shape
    print(f"\n  {label}: {W.shape} std={W.std():.5f}")

    if x_dim is None:
        x_dim = n
    g = torch.Generator(device=DEV).manual_seed(42)
    x = torch.randn(1, 64, x_dim, generator=g, device=DEV, dtype=DTYPE) * 0.5

    configs = [
        # Baselines
        ("BitNet per-tensor",      q_bitnet),
        ("BitNet per-channel",     q_bitnet_per_channel),
        ("TernPack per-channel",   lambda W: quantize_ternary_per_channel(W)),
        # TernLC (the main novel approach)
        ("TernLC r=4",             lambda W: quantize_ternlc(W, 4, True)),
        ("TernLC r=8",             lambda W: quantize_ternlc(W, 8, True)),
        ("TernLC r=16",            lambda W: quantize_ternlc(W, 16, True)),
        ("TernLC r=32",            lambda W: quantize_ternlc(W, 32, True)),
        ("TernLC r=64",            lambda W: quantize_ternlc(W, 64, True)),
        ("TernLC r=128",           lambda W: quantize_ternlc(W, 128, True)),
        # TernLC refined (alternating optimization)
        ("TernLC-refined r=16",    lambda W: quantize_ternlc_refined(W, 16, 5)),
        ("TernLC-refined r=32",    lambda W: quantize_ternlc_refined(W, 32, 5)),
        ("TernLC-refined r=64",    lambda W: quantize_ternlc_refined(W, 64, 5)),
        # TernPrep (fixed)
        ("TernPrep b128",          lambda W: quantize_ternary_preprocess_fixed(W, 128)),
        ("TernPrep b64",           lambda W: quantize_ternary_preprocess_fixed(W, 64)),
        # TernLC per-tensor (for comparison)
        ("TernLC r=16 per-tensor", lambda W: quantize_ternlc(W, 16, False)),
    ]

    print(f"  {'Algorithm':<24} {'bpw':>6} {'bits/w':>7} {'SQNR(dB)':>10} {'out_err':>10} {'vs BitNet':>10}")
    print(f"  {'-'*24} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")

    bitnet_sq = None
    for name, fn in configs:
        try:
            t0 = time.time()
            W_q, bpw = fn(W)
            dt = time.time() - t0
            sq = sqnr(W, W_q)
            oe = output_err(W, W_q, x)
            bits = bpw * 8
            if name == "BitNet per-tensor":
                bitnet_sq = sq
            delta = f"{sq - bitnet_sq:+.2f}dB" if bitnet_sq is not None and name != "BitNet per-tensor" else ""
            print(f"  {name:<24} {bpw:>6.4f} {bits:>6.2f}b {sq:>10.2f} {oe:>10.4f} {delta:>10}  ({dt:.1f}s)")
        except Exception as e:
            print(f"  {name:<24} FAILED: {str(e)[:60]}")

    if DEV.type == "cuda":
        torch.cuda.empty_cache()


def main():
    print("=" * 100)
    print("  R&D ROUND 26v2: TernLC (Ternary + Low-Rank Correction) — SUB-BITNET")
    print("  Goal: near BitNet memory (0.2 bytes/w) with MUCH better quality")
    print("=" * 100)

    weights = [
        ("FFN gate L2",   "blocks.2.ffn.w_gate.weight",  2048),
        ("FFN down L2",   "blocks.2.ffn.w_down.weight",  8192),
        ("Attn Q L2",     "blocks.2.attn.q_proj.weight", 2048),
        ("FFN gate L14",  "blocks.14.ffn.w_gate.weight", 2048),
    ]

    for label, key, xd in weights:
        try:
            benchmark_weight(label, key, xd)
        except Exception as e:
            print(f"\n  SKIP {label}: {e}")

    print("\n  KEY: TernLC target = 0.2-0.25 bytes/w with SQNR >> 6.61 dB (BitNet post-training).")
    print("  Per-channel ternary alone should beat per-tensor BitNet by several dB.")
    print("  Low-rank correction adds ~0.02 bytes/w for +5-10 dB more.")


if __name__ == "__main__":
    main()
