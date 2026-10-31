"""R&D Round 25: Real-weight quantization benchmark on ForgeLM V9-1.2B.

Establishes gold-standard numbers for all existing quant formats on REAL
trained weights (not synthetic). Every R25 novel algorithm must beat these.

Measures:
  - Frobenius relative error (weight-space)
  - SQNR in dB (weight-space)
  - Output error (activation-weighted, with realistic activations)
  - Effective bytes/weight (storage budget)

Baselines tested:
  - BitNet b1.58 ternary (gold standard, needs QAT)
  - BitNetResidual (R24: ternary + 10% element residual)
  - NVFP4 (absmax scale)
  - AS-FP4 (MSE-optimal scale, R14)
  - SR-FP4 (stochastic rounding, R15 winner)
  - IRI-FP4 x2 (iterative residual, R15 winner)
  - TSDS-FP4 (threshold-split dual-scale, R15 winner)
  - HPR-FP4 (Hadamard + AS-FP4, R15)

Runs on CUDA. Loads real weights from V9 checkpoint.
"""
import math
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
from safetensors import safe_open

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
CKPT = r"D:\windsurf\ForgeAI\research\checkpoints\ForgeLM_V2_Light.safetensors"

from forge.engine.quant.novel_quant import (
    _quantize_to_fp4_adaptive,
    quantize_hpr_fp4,
    quantize_iri_fp4,
    quantize_sr_fp4,
    quantize_tsd_fp4,
)
from forge.engine.quant.nvfp4_quant import (
    _dequantize_fp4,
    _quantize_to_fp4,
)
from tests.fixtures.keys.quantization.bitnet_residual_key import ternary_quantize

# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def frob_err(ref, q):
    """Frobenius norm relative error."""
    return (ref - q).norm().item() / ref.norm().clamp(min=1e-8).item()


def sqnr(ref, q):
    """Signal-to-quantization-noise ratio in dB."""
    signal = (ref ** 2).sum().item()
    noise = ((ref - q) ** 2).sum().item()
    if noise < 1e-30:
        return 999.0
    return 10.0 * math.log10(signal / noise)


def output_err(W_ref, W_q, x):
    """Relative error of y = x @ W^T with quantized weights."""
    y_ref = x @ W_ref.T
    y_q = x @ W_q.T
    return frob_err(y_ref, y_q)


def bytes_per_weight(n_elem, total_bytes):
    return total_bytes / n_elem


# ──────────────────────────────────────────────────────────────────────────
# Baseline quantizers (return dequantized weight + bytes/weight)
# ──────────────────────────────────────────────────────────────────────────

def q_bitnet(W):
    """BitNet b1.58 ternary. Storage: 2 bits/w packed (int8 holds 4 ternary)."""
    w_t, scale = ternary_quantize(W)
    n = W.numel()
    # Ternary packed: 2 bits/elem → 0.25 bytes/elem + 1 scale (negligible)
    bpw = 0.25
    return w_t * scale, bpw


def q_bitnet_residual(W, frac=0.10):
    """BitNetResidual R24: ternary + element residual."""
    w_t, scale = ternary_quantize(W)
    err = (W - w_t * scale).abs()
    k = int(W.numel() * frac)
    if k == 0:
        return w_t * scale, 0.25
    flat_err = err.flatten()
    top_idx = flat_err.topk(k).indices
    w_q = (w_t * scale).clone()
    w_q.flatten()[top_idx] = W.flatten()[top_idx]
    # Storage: ternary 0.25 bytes/w + residual (frac * 2 bytes bf16 + frac * 4 bytes idx)
    # Idx shared via mask: 1 bit/elem mask + frac*2 bytes values
    bpw = 0.25 + 0.125 + frac * 2.0  # ternary + mask bits + residual values
    return w_q, bpw


def q_nvfp4(W, block_size=32):
    """NVFP4: absmax scale. 0.5 bytes/w + FP8 scale overhead."""
    packed, scales, gs = _quantize_to_fp4(W, block_size)
    w_dq = _dequantize_fp4(packed, scales, *W.shape, block_size, torch.float32,
                           global_scale=gs)
    n = W.numel()
    bytes_w = packed.numel()  # 0.5 bytes/w packed
    bytes_s = scales.numel() * 1  # FP8
    bytes_g = gs.numel() * 4
    bpw = (bytes_w + bytes_s + bytes_g) / n
    return w_dq, bpw


def q_asfp4(W, block_size=32):
    """AS-FP4: MSE-optimal scale. Same storage as NVFP4."""
    packed, scales, gs = _quantize_to_fp4_adaptive(W, block_size)
    w_dq = _dequantize_fp4(packed, scales, *W.shape, block_size, torch.float32,
                           global_scale=gs)
    n = W.numel()
    bpw = (packed.numel() + scales.numel() * 1 + gs.numel() * 4) / n
    return w_dq, bpw


def q_sr_fp4(W, block_size=32):
    """SR-FP4: stochastic rounding. Same storage as AS-FP4."""
    w_dq = quantize_sr_fp4(W, block_size, n_samples=8, seed=0)
    n = W.numel()
    # Same storage as AS-FP4 (packed + FP8 scales + global)
    n_blocks = (W.shape[1] + block_size - 1) // block_size
    bytes_w = W.shape[0] * ((W.shape[1] + 1) // 2)
    bytes_s = W.shape[0] * n_blocks * 1
    bytes_g = W.shape[0] * 4
    bpw = (bytes_w + bytes_s + bytes_g) / n
    return w_dq, bpw


def q_iri_fp4(W, block_size=32, n_rounds=2):
    """IRI-FP4: iterative residual. K * FP4 storage."""
    w_dq = quantize_iri_fp4(W, block_size, n_rounds)
    n = W.numel()
    n_blocks = (W.shape[1] + block_size - 1) // block_size
    bytes_w = n_rounds * W.shape[0] * ((W.shape[1] + 1) // 2)
    bytes_s = n_rounds * W.shape[0] * n_blocks * 1
    bytes_g = n_rounds * W.shape[0] * 4
    bpw = (bytes_w + bytes_s + bytes_g) / n
    return w_dq, bpw


def q_tsds_fp4(W, block_size=32):
    """TSDS-FP4: threshold-split dual-scale. 0.69 bytes/w."""
    w_dq = quantize_tsd_fp4(W, block_size, split_quantile=0.75)
    n = W.numel()
    # FP4 0.5 + mask 0.125 + inlier scale (FP16, n_blocks * 2 bytes)
    n_blocks = (W.shape[1] + block_size - 1) // block_size
    bytes_w = W.shape[0] * ((W.shape[1] + 1) // 2)
    bytes_mask = W.shape[0] * ((W.shape[1] + 7) // 8)
    bytes_inlier = W.shape[0] * n_blocks * 2
    bytes_s = W.shape[0] * n_blocks * 1  # FP8 outlier scale
    bytes_g = W.shape[0] * 4
    bpw = (bytes_w + bytes_mask + bytes_inlier + bytes_s + bytes_g) / n
    return w_dq, bpw


def q_hpr_fp4(W, block_size=32):
    """HPR-FP4: Hadamard + AS-FP4. Same storage as AS-FP4 (rotation is free)."""
    w_dq = quantize_hpr_fp4(W, block_size)
    n = W.numel()
    n_blocks = (W.shape[1] + block_size - 1) // block_size
    bytes_w = W.shape[0] * ((W.shape[1] + 1) // 2)
    bytes_s = W.shape[0] * n_blocks * 1
    bytes_g = W.shape[0] * 4
    bpw = (bytes_w + bytes_s + bytes_g) / n
    return w_dq, bpw


# ──────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────

BASELINES = [
    ("BitNet b1.58",       q_bitnet),
    ("BitNetResidual 10%", lambda W: q_bitnet_residual(W, 0.10)),
    ("BitNetResidual 5%",  lambda W: q_bitnet_residual(W, 0.05)),
    ("NVFP4",              q_nvfp4),
    ("AS-FP4",             q_asfp4),
    ("SR-FP4",             q_sr_fp4),
    ("IRI-FP4 x2",         lambda W: q_iri_fp4(W, 32, 2)),
    ("IRI-FP4 x3",         lambda W: q_iri_fp4(W, 32, 3)),
    ("TSDS-FP4",           q_tsds_fp4),
    ("HPR-FP4",            q_hpr_fp4),
]


def benchmark_weight(label, W_key, x_dim=None):
    """Benchmark all baselines on a single real weight tensor."""
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        W = f.get_tensor(W_key).to(DEV).to(DTYPE)

    m, n = W.shape
    print(f"\n{'='*90}")
    print(f"  {label}: {W.shape} ({m*n:,} params, {m*n*2/1024:.0f} KB bf16)")
    print(f"  W stats: mean={W.mean():.5f}, std={W.std():.5f}, "
          f"max|w|={W.abs().max():.4f}, kurt={((W-W.mean())**4).mean()/(W.std()**4):.2f}")
    print(f"{'='*90}")

    # Realistic activations for output error
    if x_dim is None:
        x_dim = n
    g = torch.Generator(device=DEV).manual_seed(42)
    x = torch.randn(1, 64, x_dim, generator=g, device=DEV, dtype=DTYPE) * 0.5

    # Header
    print(f"  {'Algorithm':<22} {'bpw':>6} {'frob_err':>10} {'SQNR(dB)':>10} "
          f"{'out_err':>10} {'compress':>9}")
    print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*9}")

    results = []
    for name, qfn in BASELINES:
        try:
            t0 = time.time()
            W_q, bpw = qfn(W)
            dt = time.time() - t0
            fe = frob_err(W, W_q)
            sq = sqnr(W, W_q)
            oe = output_err(W, W_q, x)
            comp = 2.0 / bpw  # vs bf16
            print(f"  {name:<22} {bpw:>6.3f} {fe:>10.4f} {sq:>10.2f} "
                  f"{oe:>10.4f} {comp:>8.1f}x  ({dt:.1f}s)")
            results.append((name, bpw, fe, sq, oe))
        except Exception as e:
            print(f"  {name:<22} FAILED: {e}")

    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return results


def main():
    print("=" * 90)
    print("  R&D ROUND 25: Real-Weight Quantization Baselines on ForgeLM V9-1.2B")
    print(f"  Device: {DEV}  Checkpoint: {CKPT}")
    print("=" * 90)

    # Test across layer types: conv-adjacent FFN, attention, deep FFN, embedding
    # V9 = LFM2.5 architecture: 10 conv + 6 GQA (layers 2,5,8,10,12,14)
    weights = [
        ("FFN gate (layer 2, GQA-adj)",  "blocks.2.ffn.w_gate.weight", 2048),
        ("FFN up (layer 2, GQA-adj)",    "blocks.2.ffn.w_up.weight",   2048),
        ("FFN down (layer 2, GQA-adj)",  "blocks.2.ffn.w_down.weight", 8192),
        ("Attn Q_proj (layer 2)",        "blocks.2.attn.q_proj.weight", 2048),
        ("Attn O_proj (layer 2)",        "blocks.2.attn.o_proj.weight", 2048),
        ("FFN gate (layer 14, deep)",    "blocks.14.ffn.w_gate.weight", 2048),
        ("FFN down (layer 14, deep)",    "blocks.14.ffn.w_down.weight", 8192),
    ]

    all_results = {}
    for label, key, xd in weights:
        try:
            all_results[label] = benchmark_weight(label, key, xd)
        except Exception as e:
            print(f"\n  SKIP {label}: {e} (key may not exist in V9 checkpoint)")

    # Summary: average SQNR per algorithm across all weights
    print(f"\n\n{'='*90}")
    print("  SUMMARY: Average SQNR (dB) across all tested weights")
    print(f"{'='*90}")
    if all_results:
        algo_names = [r[0] for r in next(iter(all_results.values()))]
        print(f"  {'Algorithm':<22} {'avg SQNR':>10} {'avg bpw':>10} {'avg out_err':>12}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12}")
        for i, algo in enumerate(algo_names):
            sqs = [res[i][3] for res in all_results.values() if len(res) > i]
            bpws = [res[i][1] for res in all_results.values() if len(res) > i]
            oes = [res[i][4] for res in all_results.values() if len(res) > i]
            avg_sq = sum(sqs) / len(sqs) if sqs else 0
            avg_bpw = sum(bpws) / len(bpws) if bpws else 0
            avg_oe = sum(oes) / len(oes) if oes else 0
            print(f"  {algo:<22} {avg_sq:>10.2f} {avg_bpw:>10.3f} {avg_oe:>12.4f}")

    print("\n  KEY: R25 novel algorithms (AdditiveFP4, IRI-Alloc, LatticeFP4)")
    print("  must beat these numbers on the SAME weights at <= 2.0 bpw.")


if __name__ == "__main__":
    main()
