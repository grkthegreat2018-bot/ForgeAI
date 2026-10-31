"""R&D Round 26: SUB-BITNET quantization — below 1.58 bits/w, training-free.

GOAL: get memory cost near or below BitNet (1.58 bits/w = 0.1975 bytes/w)
with BETTER quality than BitNet post-training (6.61 dB SQNR on V9 weights).
Training-free (post-training). Must support train-to-tune (STE + LoRA).

FIVE NOVEL APPROACHES (informed by SOTA research):

1. TernaryPack (5-in-8 base-3): Pack 5 ternary values in 8 bits (3^5=243<256).
   1.6 bits/w — BELOW BitNet. ZERO quality loss (same ternary values, better packing).
   Pure storage win. BitNet wastes int8 (1 byte per ternary value = 8 bits for 1.58 bits).

2. BinarySalient (PTQ1.61-inspired): Binary {-1,+1} for non-salient channels,
   4-bit for salient channels. 1D per-channel mask (negligible overhead).
   At 5% salient: 0.95*1 + 0.05*4 = 1.15 bits/w. Post-training.
   Salient channels identified by activation*weight sensitivity.

3. LowRankTernary (NanoQuant/LittleBit-inspired): W ≈ A @ B where A, B ternary.
   Rank r controls quality/memory. At r = min(m,n)/4: 0.1 bytes/w (half BitNet!).
   SVD initialization + ternary rounding + residual compensation.

4. BinaryCodebook (BTC-LLM-inspired): Cluster binary weight blocks into a codebook.
   Each block → codebook index. For block=16, codebook=256: 0.5 bits/w.
   Eliminates sparse masks. Post-training.

5. TernaryPreprocess (PTQ1.61-inspired): Transform weight distribution before
   ternary quantization. Per-channel scale + shift makes weights more ternary-friendly.
   Store transformation + ternary weights. Post-training.

All tested on real V9-1.2B weights. Compared against BitNet post-training baseline.
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
# 1. TernaryPack — 5 ternary values packed in 8 bits (base-3 packing)
# ════════════════════════════════════════════════════════════════════════════

def ternary_to_base3_packed(w_ternary: torch.Tensor) -> torch.Tensor:
    """Pack ternary values {-1,0,+1} as base-3 digits, 5 per byte.

    3^5 = 243 < 256, so 5 ternary values fit in 1 byte.
    Encoding: {-1,0,+1} → {0,1,2} base-3 digit.
    Value = d0*1 + d1*3 + d2*9 + d3*27 + d4*81

    Args:
        w_ternary: (...,) int8 tensor with values in {-1, 0, +1}
    Returns:
        (..., //5) uint8 packed tensor
    """
    # Map {-1,0,+1} → {0,1,2}
    digits = (w_ternary + 1).to(torch.int32)  # {0,1,2}
    n = digits.numel()
    pad = (5 - n % 5) % 5
    if pad > 0:
        digits = F.pad(digits.reshape(-1), (0, pad), value=1)  # pad with 0-ternary
    digits = digits.reshape(-1, 5)  # (n_packed, 5)
    # Base-3 to integer
    packed = digits[:, 0] * 1 + digits[:, 1] * 3 + digits[:, 2] * 9 + \
             digits[:, 3] * 27 + digits[:, 4] * 81
    return packed.to(torch.uint8)


def base3_packed_to_ternary(packed: torch.Tensor, n_orig: int) -> torch.Tensor:
    """Unpack base-3 packed bytes back to ternary values."""
    p = packed.to(torch.int32)
    d0 = p % 3; p //= 3
    d1 = p % 3; p //= 3
    d2 = p % 3; p //= 3
    d3 = p % 3; p //= 3
    d4 = p % 3
    digits = torch.stack([d0, d1, d2, d3, d4], dim=-1).reshape(-1)
    digits = digits[:n_orig]
    return (digits - 1).to(torch.int8)  # {0,1,2} → {-1,0,+1}


def quantize_ternary_pack(W: torch.Tensor) -> tuple[torch.Tensor, float]:
    """TernaryPack: ternary quantize + base-3 pack 5-in-8.

    Quality: IDENTICAL to BitNet b1.58 (same ternary values).
    Storage: 1.6 bits/w = 0.2 bytes/w (vs BitNet's 0.25 bytes/w as int8).
    Plus per-tensor scale (negligible).
    """
    w_t, scale = ternary_quantize(W)
    # Pack
    packed = ternary_to_base3_packed(w_ternary=w_t.to(torch.int8))
    # Unpack for quality measurement
    w_t_unpacked = base3_packed_to_ternary(packed, W.numel()).reshape(W.shape)
    w_dq = w_t_unpacked.to(torch.float32) * scale

    n = W.numel()
    # Storage: 1 byte per 5 values + 1 float32 scale (negligible)
    bytes_packed = math.ceil(n / 5)
    bytes_scale = 4  # 1 float32 per tensor
    bpw = (bytes_packed + bytes_scale) / n
    return w_dq, bpw


# ════════════════════════════════════════════════════════════════════════════
# 2. BinarySalient — binary non-salient + 4-bit salient (PTQ1.61-inspired)
# ════════════════════════════════════════════════════════════════════════════

def quantize_binary_salient(W: torch.Tensor, salient_frac: float = 0.05,
                            hessian_diag: torch.Tensor | None = None
                            ) -> tuple[torch.Tensor, float]:
    """Binary {-1,+1} for non-salient channels, 4-bit for salient channels.

    PTQ1.61-inspired: 1D per-CHANNEL mask (not per-element). Salient channels
    identified by sensitivity = |weight_col| * hessian (activation^2).
    Non-salient → binary sign(W) * per-row scale.
    Salient → 4-bit quantization with per-block scale.

    Storage:
      Binary: 1 bit/w + per-row scale (2 bytes / n_in, negligible)
      Salient: 4 bits/w for salient_frac fraction + per-block scale
      Mask: 1 bit per COLUMN (n_in bits total, negligible for large weights)
    """
    out_f, in_f = W.shape

    # Sensitivity: per-column (input channel) importance
    if hessian_diag is not None:
        sensitivity = W.abs().sum(dim=0) * hessian_diag[:in_f].abs()
    else:
        # Without Hessian: use weight magnitude as proxy
        sensitivity = W.abs().sum(dim=0)
    # Top-k salient columns
    n_salient = max(1, int(in_f * salient_frac))
    salient_cols = sensitivity.topk(n_salient).indices
    is_salient = torch.zeros(in_f, dtype=torch.bool, device=W.device)
    is_salient[salient_cols] = True

    # Binary quantization for non-salient columns
    w_binary = torch.zeros_like(W)
    non_salient = ~is_salient
    if non_salient.any():
        w_ns = W[:, non_salient]
        # Per-row scale for binary (absmean)
        ns_scale = w_ns.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        w_binary[:, non_salient] = torch.sign(w_ns) * ns_scale

    # 4-bit quantization for salient columns
    if is_salient.any():
        w_s = W[:, is_salient]
        # Per-row absmax/7 for 4-bit symmetric (-8..7)
        s_scale = w_s.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
        w_s_q = torch.round(w_s / s_scale).clamp(-8, 7)
        w_binary[:, is_salient] = w_s_q * s_scale

    w_dq = w_binary

    # Storage
    n = W.numel()
    # Binary: 1 bit per non-salient element
    n_ns = W.numel() - out_f * n_salient
    bytes_binary = n_ns / 8.0
    # 4-bit: 0.5 bytes per salient element
    bytes_4bit = out_f * n_salient * 0.5
    # Per-row scales: binary (out_f * 4 bytes) + 4-bit (out_f * 4 bytes)
    bytes_scales = out_f * 4 * 2
    # Mask: in_f bits (1D per-column)
    bytes_mask = in_f / 8.0
    bpw = (bytes_binary + bytes_4bit + bytes_scales + bytes_mask) / n
    return w_dq, bpw


# ════════════════════════════════════════════════════════════════════════════
# 3. LowRankTernary — W ≈ A @ B where A, B are ternary (NanoQuant-inspired)
# ════════════════════════════════════════════════════════════════════════════

def quantize_lowrank_ternary(W: torch.Tensor, rank_frac: float = 0.25,
                             n_iters: int = 50) -> tuple[torch.Tensor, float]:
    """Low-Rank Ternary: W ≈ A @ B, A and B ternary.

    NanoQuant/LittleBit-inspired. SVD initialization → ternary rounding →
    alternating refinement.

    Storage: (m*r + r*n) * 0.2 bytes (with base-3 packing) + r scales.
    For rank r = rank_frac * min(m,n): bpw = 0.2 * rank_frac * (m+n) / max(m,n).
    For square: bpw = 0.4 * rank_frac. At rank_frac=0.5: 0.2 bytes/w = BitNet!
    At rank_frac=0.25: 0.1 bytes/w = HALF BitNet!

    Args:
        W: (out, in) weight tensor
        rank_frac: rank as fraction of min(out, in)
        n_iters: alternating refinement iterations
    """
    out_f, in_f = W.shape
    rank = max(1, int(rank_frac * min(out_f, in_f)))

    # SVD initialization
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    A_init = U[:, :rank] * S[:rank].sqrt().unsqueeze(0)  # (out, rank)
    B_init = (S[:rank].sqrt().unsqueeze(1) * Vh[:rank, :])  # (rank, in)

    # Ternary quantize A and B
    A_t, A_scale = ternary_quantize(A_init)
    B_t, B_scale = ternary_quantize(B_init)

    # Alternating refinement: fix B, optimize A, then fix A, optimize B
    # Use least-squares solution then re-ternarize
    for _ in range(n_iters):
        # Fix B, solve for A: W ≈ A @ B_t → A ≈ W @ B_t^+ (pseudoinverse)
        B_f = B_t.to(torch.float32) * B_scale
        A_new = W @ B_f.T @ torch.linalg.pinv(B_f @ B_f.T)
        A_t, A_scale = ternary_quantize(A_new)

        # Fix A, solve for B: W ≈ A_t @ B → B ≈ A_t^+ @ W
        A_f = A_t.to(torch.float32) * A_scale
        B_new = torch.linalg.pinv(A_f.T @ A_f) @ A_f.T @ W
        B_t, B_scale = ternary_quantize(B_new)

    # Reconstruct
    A_f = A_t.to(torch.float32) * A_scale
    B_f = B_t.to(torch.float32) * B_scale
    w_dq = A_f @ B_f

    # Storage: (out*rank + rank*in) ternary values, base-3 packed (0.2 bytes each)
    n = W.numel()
    n_factors = out_f * rank + rank * in_f
    bytes_factors = n_factors * 0.2  # base-3 packed
    bytes_scales = 2 * 4  # 2 float32 scales (A_scale, B_scale)
    bpw = (bytes_factors + bytes_scales) / n
    return w_dq, bpw


# ════════════════════════════════════════════════════════════════════════════
# 4. BinaryCodebook — cluster binary blocks into codebook (BTC-LLM-inspired)
# ════════════════════════════════════════════════════════════════════════════

def quantize_binary_codebook(W: torch.Tensor, block_size: int = 16,
                             n_codes: int = 256, n_iters: int = 20
                             ) -> tuple[torch.Tensor, float]:
    """Binary Codebook Clustering: binary blocks → codebook index.

    BTC-LLM-inspired. Binarize weights to {-1,+1}, reshape to blocks,
    cluster blocks via k-means into n_codes patterns. Each block →
    codebook index (log2(n_codes) bits per block).

    Storage: log2(n_codes) bits per block_size elements + per-block scale.
    For block=16, codes=256: 8 bits / 16 = 0.5 bits/w + scale overhead.
    For block=32, codes=256: 8 bits / 32 = 0.25 bits/w + scale overhead.
    """
    out_f, in_f = W.shape
    assert in_f % block_size == 0, f"in_f {in_f} must be divisible by block_size {block_size}"
    n_blocks_per_row = in_f // block_size
    n_blocks = out_f * n_blocks_per_row

    # Per-block scale (absmean)
    blocks = W.reshape(n_blocks, block_size)
    scales = blocks.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    blocks_normalized = blocks / scales

    # Binarize: sign of normalized block
    binary_blocks = torch.sign(blocks_normalized)  # {-1, +1}

    # K-means on binary block patterns (Hamming-like distance via L2)
    g = torch.Generator(device=W.device).manual_seed(42)
    perm = torch.randperm(n_blocks, generator=g, device=W.device)[:n_codes]
    codebook = binary_blocks[perm].clone()  # (n_codes, block_size)

    for _ in range(n_iters):
        # Assign: nearest codebook entry (L2 on binary = Hamming distance)
        d = (binary_blocks.unsqueeze(1) - codebook.unsqueeze(0)) ** 2
        assignments = d.sum(dim=-1).argmin(dim=1)  # (n_blocks,)
        # Update: majority vote per cluster
        new_cb = torch.zeros_like(codebook)
        counts = torch.zeros(n_codes, device=W.device)
        new_cb.index_add_(0, assignments, binary_blocks.float())
        counts.index_add_(0, assignments, torch.ones(n_blocks, device=W.device))
        nonempty = counts > 0
        new_cb[nonempty] = torch.sign(new_cb[nonempty] / counts[nonempty].unsqueeze(-1))
        codebook = torch.where(nonempty.unsqueeze(-1), new_cb, codebook)

    # Final assignment
    d = (binary_blocks.unsqueeze(1) - codebook.unsqueeze(0)) ** 2
    assignments = d.sum(dim=-1).argmin(dim=1)

    # Reconstruct
    w_dq = (codebook[assignments] * scales).reshape(out_f, in_f)

    # Storage
    n = W.numel()
    bits_per_index = math.log2(n_codes)
    bytes_indices = n_blocks * bits_per_index / 8.0
    bytes_scales = n_blocks * 4  # float32 per-block scale
    bytes_codebook = n_codes * block_size * 1  # int8 binary codebook (negligible)
    bpw = (bytes_indices + bytes_scales + bytes_codebook) / n
    return w_dq, bpw


# ════════════════════════════════════════════════════════════════════════════
# 5. TernaryPreprocess — transform weights before ternary (PTQ1.61-inspired)
# ════════════════════════════════════════════════════════════════════════════

def quantize_ternary_preprocess(W: torch.Tensor, block_size: int = 128,
                                n_iters: int = 10) -> tuple[torch.Tensor, float]:
    """Ternary + Quantization Preprocessing: per-block scale + shift before ternary.

    PTQ1.61-inspired "quantization preprocessing": transform the weight
    distribution to make it more ternary-friendly before quantizing.

    Novel twist: for each block, find the optimal (scale, shift) pair that
    minimizes ternary quantization error:
        W' = (W - shift) / scale
        W'_ternary = ternary_quantize(W')
        W_reconstructed = W'_ternary * scale + shift

    The shift centers the block on 0 (reducing asymmetry), and the scale
    is optimized to map the mass onto {-1, 0, +1} efficiently.

    Storage: ternary (0.2 bytes/w packed) + per-block (scale + shift) = 8 bytes/block.
    For block=128: 8/128 = 0.0625 bytes/w overhead → total 0.2625 bytes/w.
    Slightly above BitNet but with MUCH better quality.
    """
    out_f, in_f = W.shape
    pad = (block_size - in_f % block_size) % block_size
    wp = F.pad(W, (0, pad)) if pad > 0 else W
    in_p = wp.shape[1]
    n_blocks = in_p // block_size

    blocks = wp.view(out_f, n_blocks, block_size)

    # Optimal per-block (scale, shift) via grid search
    # Shift candidates: block mean ± {0, 0.25, 0.5, 0.75, 1.0} * std
    block_mean = blocks.mean(dim=-1, keepdim=True)
    block_std = blocks.std(dim=-1, keepdim=True).clamp(min=1e-8)

    shift_candidates = block_mean + torch.tensor(
        [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        device=W.device, dtype=W.dtype,
    ).reshape(1, 1, 7, 1) * block_std * 0.5

    # Scale candidates: absmax/0.7 * {0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3}
    centered = blocks - block_mean
    absmax = centered.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale_candidates = (absmax / 0.7) * torch.tensor(
        [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3],
        device=W.device, dtype=W.dtype,
    ).reshape(1, 1, 7, 1)

    # Grid search: 7 shifts × 7 scales = 49 candidates per block
    best_err = torch.full((out_f, n_blocks, 1), float('inf'), device=W.device, dtype=W.dtype)
    best_shift = torch.zeros(out_f, n_blocks, 1, device=W.device, dtype=W.dtype)
    best_scale = torch.ones(out_f, n_blocks, 1, device=W.device, dtype=W.dtype)

    for si in range(7):
        shift = shift_candidates[:, :, si, :]
        for ci in range(7):
            scale = scale_candidates[:, :, ci, :]
            centered = blocks - shift
            normalized = centered / scale.clamp(min=1e-12)
            # Ternary quantize
            w_t = torch.sign(normalized) * (normalized.abs() > 0.5).float()
            w_dq = w_t * scale + shift
            err = ((blocks - w_dq) ** 2).mean(dim=-1, keepdim=True)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_shift = torch.where(better, shift, best_shift)
            best_scale = torch.where(better, scale, best_scale)

    # Apply best (scale, shift)
    centered = blocks - best_shift
    normalized = centered / best_scale.clamp(min=1e-12)
    w_t = torch.sign(normalized) * (normalized.abs() > 0.5).float()
    w_dq = w_t * best_scale + best_shift

    w_dq_final = w_dq.view(out_f, in_p)[:, :in_f].contiguous()

    # Storage: ternary (0.2 bytes/w packed) + per-block (scale + shift) = 8 bytes/block
    n = W.numel()
    bytes_ternary = n * 0.2  # base-3 packed
    bytes_scales = out_f * n_blocks * 8  # 2 * float32 per block
    bpw = (bytes_ternary + bytes_scales) / n
    return w_dq_final, bpw


# ════════════════════════════════════════════════════════════════════════════
# Benchmark
# ════════════════════════════════════════════════════════════════════════════

def q_bitnet(W):
    """BitNet b1.58 baseline (int8 storage, 0.25 bytes/w)."""
    w_t, scale = ternary_quantize(W)
    return w_t * scale, 0.25


def benchmark_weight(label, W_key, x_dim=None):
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        W = f.get_tensor(W_key).to(DEV).to(DTYPE)

    m, n = W.shape
    print(f"\n{'='*100}")
    print(f"  {label}: {W.shape} ({m*n:,} params) std={W.std():.5f} kurt={((W-W.mean())**4).mean()/(W.std()**4):.2f}")
    print(f"{'='*100}")

    if x_dim is None:
        x_dim = n
    g = torch.Generator(device=DEV).manual_seed(42)
    x = torch.randn(1, 64, x_dim, generator=g, device=DEV, dtype=DTYPE) * 0.5

    configs = [
        # Baselines
        ("BitNet b1.58 (int8)",    q_bitnet),
        ("BitNet (base-3 pack)",   lambda W: quantize_ternary_pack(W)),
        # BinarySalient (PTQ1.61-inspired)
        ("BinSalient 2%",          lambda W: quantize_binary_salient(W, 0.02)),
        ("BinSalient 5%",          lambda W: quantize_binary_salient(W, 0.05)),
        ("BinSalient 10%",         lambda W: quantize_binary_salient(W, 0.10)),
        ("BinSalient 20%",         lambda W: quantize_binary_salient(W, 0.20)),
        # LowRankTernary (NanoQuant-inspired)
        ("LoRT rank=0.125",        lambda W: quantize_lowrank_ternary(W, 0.125, 30)),
        ("LoRT rank=0.25",         lambda W: quantize_lowrank_ternary(W, 0.25, 30)),
        ("LoRT rank=0.50",         lambda W: quantize_lowrank_ternary(W, 0.50, 30)),
        ("LoRT rank=0.75",         lambda W: quantize_lowrank_ternary(W, 0.75, 30)),
        # BinaryCodebook (BTC-LLM-inspired)
        ("BinCB b16 c256",         lambda W: quantize_binary_codebook(W, 16, 256)),
        ("BinCB b32 c256",         lambda W: quantize_binary_codebook(W, 32, 256)),
        ("BinCB b64 c256",         lambda W: quantize_binary_codebook(W, 64, 256)),
        ("BinCB b32 c1024",        lambda W: quantize_binary_codebook(W, 32, 1024)),
        # TernaryPreprocess (PTQ1.61-inspired)
        ("TernPrep b128",          lambda W: quantize_ternary_preprocess(W, 128)),
        ("TernPrep b64",           lambda W: quantize_ternary_preprocess(W, 64)),
        ("TernPrep b256",          lambda W: quantize_ternary_preprocess(W, 256)),
    ]

    print(f"  {'Algorithm':<22} {'bpw':>6} {'bits/w':>7} {'frob_err':>10} {'SQNR(dB)':>10} {'out_err':>10} {'vs BitNet':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    bitnet_sq = None
    for name, fn in configs:
        try:
            t0 = time.time()
            W_q, bpw = fn(W)
            dt = time.time() - t0
            fe = frob_err(W, W_q)
            sq = sqnr(W, W_q)
            oe = output_err(W, W_q, x)
            bits = bpw * 8
            if "BitNet" in name and "base-3" not in name:
                bitnet_sq = sq
            delta = f"{sq - bitnet_sq:+.2f}dB" if bitnet_sq is not None and "BitNet" not in name else ""
            print(f"  {name:<22} {bpw:>6.4f} {bits:>6.2f}b {fe:>10.4f} {sq:>10.2f} {oe:>10.4f} {delta:>10}  ({dt:.1f}s)")
        except Exception as e:
            print(f"  {name:<22} FAILED: {str(e)[:60]}")

    if DEV.type == "cuda":
        torch.cuda.empty_cache()


def main():
    print("=" * 100)
    print("  R&D ROUND 26: SUB-BITNET Quantization (< 1.58 bits/w) on V9-1.2B")
    print("  Goal: below BitNet memory with better quality, training-free, train-to-tune capable")
    print(f"  Device: {DEV}")
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

    print("\n  KEY: Target is < 1.58 bits/w (0.1975 bytes/w) with SQNR > 6.61 dB (BitNet post-training).")
    print("  BitNet base-3 pack = 1.6 bits = same quality, less memory (free win).")
    print("  BinSalient/LoRT/BinCB/TernPrep must beat BitNet SQNR at similar-or-lower bpw.")


if __name__ == "__main__":
    main()
