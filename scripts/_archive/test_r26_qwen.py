"""R&D Round 26v3: Sub-BitNet quantization on Qwen 2.5 0.5B real weights.

Qwen 2.5 0.5B: 24 layers, d_model=896, FFN=4864, GQA (4 KV heads).
Real pretrained weights (not a port). Tests generalization of sub-BitNet methods.

Architecture:
  - FFN: gate_proj (4864x896), up_proj (4864x896), down_proj (896x4864) — SwiGLU
  - Attn: q_proj (896x896), k_proj (128x896), v_proj (128x896), o_proj (896x896)
  - 24 transformer layers

Tests ALL sub-BitNet approaches:
  1. BitNet per-tensor (baseline, 2.0 bits/w)
  2. TernPack per-channel (1.6 bits/w, free win)
  3. TernLC (ternary + low-rank correction, 1.7-2.2 bits/w)
  4. TernLC-refined (alternating optimization, better quality)
  5. BinarySalient (binary + 4-bit salient channels)
  6. LowRankTernary (W ≈ A@B ternary factors)
  7. Per-block ternary (block_size=32, 64, 128)
"""
import math
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import glob

import torch
import torch.nn.functional as F
from safetensors import safe_open

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
QWEN_PATH = glob.glob(os.path.join(
    r"C:\Users\tmk68\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots",
    "*", "model.safetensors"))[0]

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
# Quantization functions (from R26v2, proven on V9)
# ════════════════════════════════════════════════════════════════════════════

def q_bitnet_per_tensor(W):
    w_t, scale = ternary_quantize(W)
    return w_t * scale, 0.25  # int8 storage

def q_ternary_per_channel(W):
    """Ternary per-channel scale, base-3 packed (1.6 bits/w)."""
    out_f, _in_f = W.shape
    scales = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8) / 0.7
    w_norm = W / scales
    w_t = torch.sign(w_norm) * (w_norm.abs() > 0.5).float()
    w_dq = w_t * scales
    n = W.numel()
    bpw = (n * 0.2 + out_f * 4) / n  # base-3 packed + per-channel scale
    return w_dq, bpw

def q_ternary_per_block(W, block_size=32):
    """Ternary per-block scale, base-3 packed."""
    out_f, in_f = W.shape
    pad = (block_size - in_f % block_size) % block_size
    wp = F.pad(W, (0, pad)) if pad > 0 else W
    in_p = wp.shape[1]
    n_blocks = in_p // block_size
    blocks = wp.view(out_f, n_blocks, block_size)
    scales = blocks.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8) / 0.7
    w_norm = blocks / scales
    w_t = torch.sign(w_norm) * (w_norm.abs() > 0.5).float()
    w_dq = (w_t * scales).view(out_f, in_p)[:, :in_f].contiguous()
    n = W.numel()
    bpw = (n * 0.2 + out_f * n_blocks * 4) / n
    return w_dq, bpw

def q_ternlc(W, rank=16, per_channel=True):
    """Ternary + Low-Rank Correction (SVD of error)."""
    out_f, in_f = W.shape
    if per_channel:
        scales = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8) / 0.7
    else:
        scales = W.abs().mean().clamp(min=1e-8) / 0.7
    w_norm = W / scales
    w_t = torch.sign(w_norm) * (w_norm.abs() > 0.5).float()
    w_ternary = w_t * scales
    error = W - w_ternary
    U, S, Vh = torch.linalg.svd(error.float(), full_matrices=False)
    r = min(rank, S.shape[0])
    A = (U[:, :r] * S[:r].unsqueeze(0)).to(torch.float16)
    B = Vh[:r, :].to(torch.float16)
    w_dq = w_ternary + (A.to(torch.float32) @ B.to(torch.float32))
    n = W.numel()
    bytes_t = n * 0.2
    bytes_s = out_f * 4 if per_channel else 4
    bytes_ab = (out_f * r + r * in_f) * 2
    bpw = (bytes_t + bytes_s + bytes_ab) / n
    return w_dq, bpw

def q_ternlc_refined(W, rank=16, n_iters=5):
    """TernLC with alternating refinement."""
    out_f, in_f = W.shape
    scales = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8) / 0.7
    w_norm = W / scales
    w_t = torch.sign(w_norm) * (w_norm.abs() > 0.5).float()
    w_ternary = w_t * scales
    error = W - w_ternary
    U, S, Vh = torch.linalg.svd(error.float(), full_matrices=False)
    r = min(rank, S.shape[0])
    A = U[:, :r] * S[:r].unsqueeze(0)
    B = Vh[:r, :]
    for _ in range(n_iters):
        residual = W - A @ B
        r_norm = residual / scales
        w_t = torch.sign(r_norm) * (r_norm.abs() > 0.5).float()
        w_ternary = w_t * scales
        error = W - w_ternary
        U, S, Vh = torch.linalg.svd(error.float(), full_matrices=False)
        A = U[:, :r] * S[:r].unsqueeze(0)
        B = Vh[:r, :]
    w_dq = w_ternary + A @ B
    n = W.numel()
    bpw = (n * 0.2 + out_f * 4 + (out_f * r + r * in_f) * 2) / n
    return w_dq, bpw

def q_binary_salient(W, salient_frac=0.05):
    """Binary non-salient + 4-bit salient (PTQ1.61-inspired)."""
    out_f, in_f = W.shape
    sensitivity = W.abs().sum(dim=0)
    n_salient = max(1, int(in_f * salient_frac))
    salient_cols = sensitivity.topk(n_salient).indices
    is_salient = torch.zeros(in_f, dtype=torch.bool, device=W.device)
    is_salient[salient_cols] = True
    w_dq = torch.zeros_like(W)
    non_salient = ~is_salient
    if non_salient.any():
        w_ns = W[:, non_salient]
        ns_scale = w_ns.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
        w_dq[:, non_salient] = torch.sign(w_ns) * ns_scale
    if is_salient.any():
        w_s = W[:, is_salient]
        s_scale = w_s.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
        w_dq[:, is_salient] = torch.round(w_s / s_scale).clamp(-8, 7) * s_scale
    n = W.numel()
    n_ns = W.numel() - out_f * n_salient
    bpw = (n_ns / 8.0 + out_f * n_salient * 0.5 + out_f * 8 + in_f / 8.0) / n
    return w_dq, bpw

def q_lowrank_ternary(W, rank_frac=0.25, n_iters=30):
    """Low-Rank Ternary: W ≈ A@B, A and B ternary."""
    out_f, in_f = W.shape
    rank = max(1, int(rank_frac * min(out_f, in_f)))
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    A_init = U[:, :rank] * S[:rank].sqrt().unsqueeze(0)
    B_init = S[:rank].sqrt().unsqueeze(1) * Vh[:rank, :]
    A_t, A_scale = ternary_quantize(A_init)
    B_t, B_scale = ternary_quantize(B_init)
    for _ in range(n_iters):
        B_f = B_t.to(torch.float32) * B_scale
        A_new = W @ B_f.T @ torch.linalg.pinv(B_f @ B_f.T)
        A_t, A_scale = ternary_quantize(A_new)
        A_f = A_t.to(torch.float32) * A_scale
        B_new = torch.linalg.pinv(A_f.T @ A_f) @ A_f.T @ W
        B_t, B_scale = ternary_quantize(B_new)
    w_dq = (A_t.to(torch.float32) * A_scale) @ (B_t.to(torch.float32) * B_scale)
    n = W.numel()
    n_factors = out_f * rank + rank * in_f
    bpw = (n_factors * 0.2 + 8) / n
    return w_dq, bpw


# ════════════════════════════════════════════════════════════════════════════
# Benchmark on Qwen weights
# ════════════════════════════════════════════════════════════════════════════

def benchmark_qwen():
    print("=" * 110)
    print("  R&D ROUND 26v3: Sub-BitNet Quantization on Qwen 2.5 0.5B (real pretrained weights)")
    print("  Goal: < 1.58 bits/w with SQNR > BitNet post-training, training-free")
    print(f"  Device: {DEV}")
    print("=" * 110)

    # Sample layers from different positions
    test_keys = [
        ("FFN gate L0",   "model.layers.0.mlp.gate_proj.weight"),
        ("FFN down L0",   "model.layers.0.mlp.down_proj.weight"),
        ("FFN gate L12",  "model.layers.12.mlp.gate_proj.weight"),
        ("FFN down L12",  "model.layers.12.mlp.down_proj.weight"),
        ("FFN gate L23",  "model.layers.23.mlp.gate_proj.weight"),
        ("Attn Q L0",     "model.layers.0.self_attn.q_proj.weight"),
        ("Attn Q L12",    "model.layers.12.self_attn.q_proj.weight"),
        ("Attn K L12",    "model.layers.12.self_attn.k_proj.weight"),
        ("Attn V L12",    "model.layers.12.self_attn.v_proj.weight"),
        ("Attn O L12",    "model.layers.12.self_attn.o_proj.weight"),
    ]

    with safe_open(QWEN_PATH, framework="pt", device="cpu") as f:
        for label, key in test_keys:
            try:
                W = f.get_tensor(key).to(DEV).to(DTYPE)
            except Exception as e:
                print(f"\n  SKIP {label}: {e}")
                continue

            m, n = W.shape
            kurt = ((W - W.mean())**4).mean() / (W.std()**4)
            print(f"\n  {label}: {W.shape} ({m*n:,} params) std={W.std():.5f} kurt={kurt:.2f}")

            g = torch.Generator(device=DEV).manual_seed(42)
            x = torch.randn(1, 64, n, generator=g, device=DEV, dtype=DTYPE) * 0.5

            configs = [
                ("BitNet per-tensor",      q_bitnet_per_tensor),
                ("TernPack per-channel",   q_ternary_per_channel),
                ("TernBlock b32",          lambda W: q_ternary_per_block(W, 32)),
                ("TernBlock b64",          lambda W: q_ternary_per_block(W, 64)),
                ("TernLC r=8",             lambda W: q_ternlc(W, 8, True)),
                ("TernLC r=16",            lambda W: q_ternlc(W, 16, True)),
                ("TernLC r=32",            lambda W: q_ternlc(W, 32, True)),
                ("TernLC r=64",            lambda W: q_ternlc(W, 64, True)),
                ("TernLC-refined r=16",    lambda W: q_ternlc_refined(W, 16, 5)),
                ("TernLC-refined r=32",    lambda W: q_ternlc_refined(W, 32, 5)),
                ("TernLC-refined r=64",    lambda W: q_ternlc_refined(W, 64, 5)),
                ("BinSalient 5%",          lambda W: q_binary_salient(W, 0.05)),
                ("BinSalient 10%",         lambda W: q_binary_salient(W, 0.10)),
                ("BinSalient 20%",         lambda W: q_binary_salient(W, 0.20)),
                ("LoRT rank=0.25",         lambda W: q_lowrank_ternary(W, 0.25, 20)),
                ("LoRT rank=0.50",         lambda W: q_lowrank_ternary(W, 0.50, 20)),
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

    # Summary
    print(f"\n{'='*110}")
    print("  SUMMARY")
    print(f"{'='*110}")
    print("  BitNet per-tensor = 2.0 bits/w baseline (ternary, per-tensor absmean/0.7 scale)")
    print("  TernPack per-channel = 1.6 bits/w (base-3 packed + per-channel scale) — FREE WIN")
    print("  TernLC = ternary + low-rank SVD correction — adds ~0.02-0.08 bytes/w for +1-3 dB")
    print("  Win condition: bpw < 0.25 (2.0 bits) AND SQNR > BitNet")


if __name__ == "__main__":
    benchmark_qwen()
