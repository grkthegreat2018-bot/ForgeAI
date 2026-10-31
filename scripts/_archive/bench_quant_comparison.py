"""Benchmark new vs existing quantization methods on realistic LLM weights."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bench_int4(w, gs=128):
    n_groups = w.shape[1] // gs
    w_dq = torch.zeros_like(w)
    for g in range(n_groups):
        block = w[:, g*gs:(g+1)*gs]
        scale = block.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
        q = torch.round(block / scale).clamp(-8, 7)
        w_dq[:, g*gs:(g+1)*gs] = q * scale
    err = (w - w_dq).norm() / w.norm()
    bpw = 0.5 + 2/n_groups
    return err.item(), bpw


def bench_int8(w):
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale).clamp(-128, 127)
    w_dq = q * scale
    err = (w - w_dq).norm() / w.norm()
    bpw = 1.0 + 2/w.shape[0]
    return err.item(), bpw


def bench_nvfp4(w, gs=32):
    n_groups = (w.shape[1] + gs - 1) // gs
    pad = n_groups * gs - w.shape[1]
    wp = F.pad(w, (0, pad)) if pad > 0 else w
    wg = wp.reshape(w.shape[0], n_groups, gs)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 6.0
    wn = (wg / scale).clamp(-1, 1)
    fp4_mags = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=w.device)
    abs_n = wn.abs()
    thresholds = torch.tensor([0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 6.0], device=w.device)
    idx = torch.searchsorted(thresholds, abs_n).clamp(0, 7)
    wq = torch.sign(wn) * fp4_mags[idx]
    w_dq = (wq * scale).reshape(w.shape[0], -1)[:, :w.shape[1]]
    err = (w - w_dq).norm() / w.norm()
    bpw = 0.5 + 2/n_groups
    return err.item(), bpw


def bench_nf4(w, gs=64):
    from forge.training.bitnet_lora import NF4Linear
    nf4 = NF4Linear(w.shape[1], w.shape[0], bias=False, group_size=gs)
    nf4.load_from_weight(w)
    w_dq = nf4._dequantize_weight(torch.float32, cache=False)
    err = (w - w_dq).norm() / w.norm()
    bpw = 0.5 + 2/(w.shape[1]//gs)
    return err.item(), bpw


def bench_forge_quant(w, gs=128, sr=0.10):
    from forge.engine.quant.forge_quant import ForgeQuantLinear
    fq = ForgeQuantLinear(w.shape[1], w.shape[0], bias=False, group_size=gs, sparse_ratio=sr)
    fq.load_from_weight(w)
    w_dq = fq._dequantize_weight(torch.float32, cache=False)
    err = (w - w_dq).norm() / w.norm()
    bpw = 0.5 + sr * 1.0
    return err.item(), bpw


def bench_grinqh(w, gs=128, target=2.5):
    from forge.engine.quant.grinqh import GRINQHLinear
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
    lin.weight.data = w.clone()
    gl = GRINQHLinear.from_linear(lin, group_size=gs, target_effective_bits=target)
    w_dq = gl._dequantize_weight(torch.float32)
    err = (w - w_dq).norm() / w.norm()
    bpw = target / 8.0
    return err.item(), bpw


def bench_mixllm(w, gs=128, hf=0.10):
    from forge.engine.quant.mixllm import MixLLMLinear
    high_mask = torch.zeros(w.shape[0], dtype=torch.bool)
    norms = w.norm(dim=1)
    topk = norms.topk(int(w.shape[0] * hf)).indices
    high_mask[topk] = True
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
    lin.weight.data = w.clone()
    ml = MixLLMLinear.from_linear(lin, high_mask, group_size=gs)
    w_dq = ml._dequantize_weight(torch.float32)
    err = (w - w_dq).norm() / w.norm()
    bpw = 0.5 * (1 - hf) + 1.0 * hf
    return err.item(), bpw


def bench_acbq(w, gs=128):
    from forge.engine.quant.acbq import ACBQLinear
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
    lin.weight.data = w.clone()
    al = ACBQLinear.from_linear(lin, group_size=gs, bits=4)
    w_dq = al._dequantize_weight(torch.float32)
    err = (w - w_dq).norm() / w.norm()
    bpw = 0.5 + 2/(w.shape[1]//gs)
    return err.item(), bpw


if __name__ == "__main__":
    torch.manual_seed(42)
    # Simulate realistic LLM weights: mostly small, with outlier channels
    w = torch.randn(2048, 2048) * 0.02
    for i in range(0, 2048, 20):
        w[i] += torch.randn(2048) * 0.3
    print(f"Weight: {w.shape}, norm={w.norm():.2f}, max={w.abs().max():.4f}")
    print()

    methods = [
        ("INT4 (gs=128)", lambda: bench_int4(w, 128)),
        ("INT8 (per-ch)", lambda: bench_int8(w)),
        ("NVFP4 (gs=32)", lambda: bench_nvfp4(w, 32)),
        ("NF4 (gs=64)", lambda: bench_nf4(w, 64)),
        ("ForgeQuant (sr=0.10)", lambda: bench_forge_quant(w, 128, 0.10)),
        ("ForgeQuant (sr=0.15)", lambda: bench_forge_quant(w, 128, 0.15)),
        ("GRINQH (2.5 bit)", lambda: bench_grinqh(w, 128, 2.5)),
        ("GRINQH (3.0 bit)", lambda: bench_grinqh(w, 128, 3.0)),
        ("MixLLM (10% high)", lambda: bench_mixllm(w, 128, 0.10)),
        ("ACBQ (W4)", lambda: bench_acbq(w, 128)),
    ]

    print(f"{'Method':<25} {'Rel Error':>10} {'Bytes/w':>10}")
    print("-" * 50)
    results = []
    for name, fn in methods:
        try:
            err, bpw = fn()
            print(f"{name:<25} {err:>10.4f} {bpw:>10.4f}")
            results.append((name, err, bpw))
        except Exception as e:
            print(f"{name:<25} ERROR: {e}")

    print()
    print("=== Sorted by error (best first) ===")
    results.sort(key=lambda x: x[1])
    for name, err, bpw in results:
        print(f"  {name:<25} err={err:.4f}  bpw={bpw:.3f}")

    print()
    print("=== Pareto frontier (error vs bytes/w) ===")
    # Find Pareto-optimal: nothing has both lower error AND lower bpw
    pareto = []
    for name, err, bpw in results:
        dominated = any(e <= err and b <= bpw and (e < err or b < bpw) for _, e, b in results)
        if not dominated:
            pareto.append((name, err, bpw))
    pareto.sort(key=lambda x: x[1])
    for name, err, bpw in pareto:
        print(f"  {name:<25} err={err:.4f}  bpw={bpw:.3f}")
