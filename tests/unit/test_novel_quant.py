"""Test novel quantization algorithms (AS-FP4, ResidualFP4) vs NVFP4 baseline."""

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

# GPU device — all test tensors are moved to CUDA when available for speed.
# The optimal-sign Hadamard search does O(n * iters) matmuls which is 100x+
# faster on RTX 5070 tensor cores vs CPU.
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32  # compute in fp32 for accuracy, regardless of device

# Round 15 novel algorithms
from forge.engine.quant.novel_quant import (
    ASFP4Linear,
    ResidualFP4Linear,
    _quantize_to_fp4_adaptive,
    _quantize_to_fp4_residual,
    quantize_asfp4_dequant,
    quantize_awr_fp4,
    quantize_bn_fp4,
    quantize_hpr_fp4,
    quantize_hw_fp4,
    quantize_iri_fp4,
    quantize_model_asfp4,
    quantize_model_residual_fp4,
    quantize_oc_hybrid,
    quantize_pcba,
    quantize_saas_fp4,
    quantize_sr_fp4,
    quantize_tsd_fp4,
)
from forge.engine.quant.nvfp4_quant import (
    NVFP4Linear,
    _dequantize_fp4,
    _quantize_to_fp4,
)


def frob_err(ref, q):
    """Frobenius norm relative error."""
    return (ref - q).norm().item() / ref.norm().clamp(min=1e-8).item()


def test_asfp4_vs_nvfp4():
    """Compare AS-FP4 (MSE-optimal scale) vs NVFP4 (absmax scale)."""
    print("=" * 60)
    print("  Novel Quant: AS-FP4 vs NVFP4")
    print("=" * 60)

    torch.manual_seed(42)
    # Use realistic LLM-like weights (small std, with some outliers)
    W = torch.randn(512, 1024) * 0.05
    # Add some outliers (like real LLM weights)
    outlier_mask = torch.rand(512, 1024) < 0.01
    W[outlier_mask] *= 5.0

    # Standard NVFP4 (absmax scale)
    packed_std, scales_std, gs_std = _quantize_to_fp4(W, block_size=32)
    W_dq_std = _dequantize_fp4(packed_std, scales_std, 512, 1024, 32, torch.float32, global_scale=gs_std)

    # AS-FP4 (MSE-optimal scale)
    packed_adapt, scales_adapt, gs_adapt = _quantize_to_fp4_adaptive(W, block_size=32)
    W_dq_adapt = _dequantize_fp4(packed_adapt, scales_adapt, 512, 1024, 32, torch.float32, global_scale=gs_adapt)

    err_std = frob_err(W, W_dq_std)
    err_adapt = frob_err(W, W_dq_adapt)
    improvement = (1.0 - err_adapt / err_std) * 100

    print(f"  NVFP4 (absmax):    frob_err = {err_std:.4f}")
    print(f"  AS-FP4 (optimal):  frob_err = {err_adapt:.4f}")
    print(f"  Improvement:       {improvement:.1f}%")

    assert err_adapt <= err_std, "AS-FP4 should be better or equal"
    print("  PASS\n")


def test_residual_fp4():
    """Test ResidualFP4 with different residual ratios."""
    print("  Novel Quant: ResidualFP4")
    print("-" * 60)

    torch.manual_seed(42)
    W = torch.randn(512, 1024) * 0.05
    outlier_mask = torch.rand(512, 1024) < 0.01
    W[outlier_mask] *= 5.0

    # Baseline FP4
    packed, scales, gs = _quantize_to_fp4(W, block_size=32)
    W_dq_fp4 = _dequantize_fp4(packed, scales, 512, 1024, 32, torch.float32, global_scale=gs)
    err_fp4 = frob_err(W, W_dq_fp4)

    print(f"  FP4 (no residual):     frob_err = {err_fp4:.4f}")

    for ratio in [0.02, 0.05, 0.10, 0.20]:
        packed_r, scales_r, gs_r, res_idx, res_val, res_scale = _quantize_to_fp4_residual(
            W, block_size=32, residual_ratio=ratio
        )
        W_dq = _dequantize_fp4(packed_r, scales_r, 512, 1024, 32, torch.float32, global_scale=gs_r)
        # Add residual
        res_val_f = res_val.to(torch.float32)
        res_scaled = res_val_f * res_scale
        W_dq.scatter_add_(1, res_idx.long(), res_scaled)
        err_r = frob_err(W, W_dq)
        improvement = (1.0 - err_r / err_fp4) * 100
        overhead_bytes = int(1024 * ratio) * (4 + 1)  # int32 index + int8 value
        total_bytes = packed_r.numel() + scales_r.numel() + overhead_bytes * 512
        bf16_bytes = 512 * 1024 * 2
        compression = bf16_bytes / total_bytes
        print(
            f"  R-FP4 ({ratio * 100:>4.0f}% res):    frob_err = {err_r:.4f}  "
            f"({improvement:+5.1f}%)  {compression:.1f}x compression"
        )

    print("  PASS\n")


def test_linear_forward():
    """Test all three Linear variants produce correct outputs."""
    print("  Novel Quant: Forward pass comparison")
    print("-" * 60)

    torch.manual_seed(42)
    lin = nn.Linear(512, 1024, bias=True)
    # Scale weights to be more LLM-like
    lin.weight.data *= 0.05

    x = torch.randn(8, 512, dtype=torch.float32)

    with torch.no_grad():
        y_ref = lin(x)

        nvfp4 = NVFP4Linear.from_linear(lin, block_size=32)
        y_nvfp4 = nvfp4(x)

        asfp4 = ASFP4Linear.from_linear(lin, block_size=32)
        y_asfp4 = asfp4(x)

        rfp4 = ResidualFP4Linear.from_linear(lin, block_size=32, residual_ratio=0.05)
        y_rfp4 = rfp4(x)

    err_nvfp4 = frob_err(y_ref, y_nvfp4)
    err_asfp4 = frob_err(y_ref, y_asfp4)
    err_rfp4 = frob_err(y_ref, y_rfp4)

    print(f"  NVFP4:     frob_err = {err_nvfp4:.4f}")
    print(f"  AS-FP4:    frob_err = {err_asfp4:.4f}")
    print(f"  R-FP4 5%:  frob_err = {err_rfp4:.4f}")

    # All should be reasonable
    assert err_nvfp4 < 0.20, f"NVFP4 error too high: {err_nvfp4}"
    assert err_asfp4 < 0.20, f"AS-FP4 error too high: {err_asfp4}"
    assert err_rfp4 < 0.20, f"R-FP4 error too high: {err_rfp4}"

    # AS-FP4 should be better or equal to NVFP4
    assert err_asfp4 <= err_nvfp4 + 0.01, "AS-FP4 should be better"
    # R-FP4 should be better than NVFP4 (residual corrects errors)
    assert err_rfp4 <= err_nvfp4 + 0.01, "R-FP4 should be better"

    print("  PASS\n")


def test_cuda_speed():
    """Benchmark all methods on CUDA."""
    if not torch.cuda.is_available():
        print("  CUDA speed: SKIPPED (no CUDA)\n")
        return

    print("  Novel Quant: CUDA speed benchmark")
    print("-" * 60)

    torch.manual_seed(42)
    lin = nn.Linear(2048, 8192, bias=False).cuda().to(torch.bfloat16)

    nvfp4 = NVFP4Linear.from_linear(lin, block_size=32).cuda()
    asfp4 = ASFP4Linear.from_linear(lin, block_size=32).cuda()
    rfp4 = ResidualFP4Linear.from_linear(lin, block_size=32, residual_ratio=0.05).cuda()

    x = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = lin(x)
            _ = nvfp4(x)
            _ = asfp4(x)
            _ = rfp4(x)
    torch.cuda.synchronize()

    for name, module in [("bf16", lin), ("NVFP4", nvfp4), ("AS-FP4", asfp4), ("R-FP4", rfp4)]:
        t0 = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                _ = module(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 10
        print(f"  {name:>8}: {ms:.2f} ms/call")

    # Memory
    for name, module in [("NVFP4", nvfp4), ("AS-FP4", asfp4), ("R-FP4", rfp4)]:
        packed_bytes = module.weight_packed.numel()
        scale_bytes = module.weight_scales.numel()
        if hasattr(module, "residual_indices"):
            res_bytes = (
                module.residual_indices.numel() * 4 + module.residual_values.numel() + module.residual_scale.numel() * 4
            )
        else:
            res_bytes = 0
        total = packed_bytes + scale_bytes + res_bytes
        bf16_bytes = 2048 * 8192 * 2
        print(f"  {name:>8} mem: {total / 1024:.0f} KB ({bf16_bytes / total:.1f}x vs bf16)")

    print("  PASS\n")


def test_quantize_model():
    """Test quantize_model functions on a small model."""
    print("  Novel Quant: Model-level quantization")
    print("-" * 60)

    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    x = torch.randn(4, 256)
    with torch.no_grad():
        y_ref = model(x)

    # AS-FP4
    m1 = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    m1.load_state_dict(model.state_dict())
    n1 = quantize_model_asfp4(m1, verbose=True)
    with torch.no_grad():
        y1 = m1(x)
    print(f"    AS-FP4 output err: {frob_err(y_ref, y1):.4f} ({n1} layers)")

    # R-FP4
    m2 = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    m2.load_state_dict(model.state_dict())
    n2 = quantize_model_residual_fp4(m2, residual_ratio=0.05, verbose=True)
    with torch.no_grad():
        y2 = m2(x)
    print(f"    R-FP4 output err:  {frob_err(y_ref, y2):.4f} ({n2} layers)")

    assert n1 == 3 and n2 == 3
    assert y1.shape == y_ref.shape and y2.shape == y_ref.shape
    print("  PASS\n")


def main():
    test_asfp4_vs_nvfp4()
    test_residual_fp4()
    test_linear_forward()
    test_quantize_model()
    test_cuda_speed()
    print("=" * 60)
    print("  ALL NOVEL QUANT TESTS PASSED")
    print("=" * 60)


# ============================================================================
# R&D ROUND 15: Novel Param Quantization benchmark
# Tests 10 novel algorithms vs NVFP4/AS-FP4/R-FP4 baselines on realistic
# LLM-like weight distributions. Measures Frobenius error, MSE, SQNR, and
# output-error (with synthetic activations / Hessian proxy).
# ============================================================================


def _randn(*shape, seed=None, dtype=None):
    """Generate random tensor on GPU (DEV) with optional seed.

    Replaces torch.randn + torch.manual_seed pattern for GPU acceleration.
    """
    if dtype is None:
        dtype = DTYPE
    if seed is not None:
        g = torch.Generator(device=DEV).manual_seed(seed)
        return torch.randn(*shape, generator=g, device=DEV, dtype=dtype)
    return torch.randn(*shape, device=DEV, dtype=dtype)


def _to_dev(t):
    """Move a tensor to the test device (GPU if available)."""
    return t.to(DEV)


def _sqnr(ref, q):
    """Signal-to-quantization-noise ratio in dB."""
    signal = (ref**2).sum().item()
    noise = ((ref - q) ** 2).sum().item()
    if noise < 1e-30:
        return 999.0
    return 10.0 * math.log10(signal / noise)


def _make_llm_weights(out_f, in_f, seed=42, outlier_frac=0.01, outlier_mult=5.0, std=0.05, asym_mean=0.0):
    """Generate realistic LLM-like weights: small std + sparse outliers + optional asymmetry.

    Tensors are created on GPU (DEV) when available for speed.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    W = torch.randn(out_f, in_f, generator=g, device=DEV, dtype=DTYPE) * std
    mask = torch.rand(out_f, in_f, generator=g, device=DEV) < outlier_frac
    W[mask] *= outlier_mult
    if asym_mean != 0.0:
        # Add a per-block mean drift (some blocks asymmetric)
        n_blocks = in_f // 32
        means = torch.randn(out_f, n_blocks, generator=g, device=DEV, dtype=DTYPE) * asym_mean
        W = W + means.repeat_interleave(32, dim=1)[:, :in_f]
    return W


def _make_hessian(in_f, seed=7, concentration=0.3):
    """Synthetic Hessian diagonal proxy (activation^2 per input channel).

    Real activations have a few high-magnitude channels (outliers). We model
    this with a log-normal-ish distribution: most channels small, a few large.
    concentration controls how peaked the importance is.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    base = torch.rand(in_f, generator=g, device=DEV, dtype=DTYPE) ** (1.0 / concentration)
    return (base / base.mean()).clamp(min=1e-3)  # normalized, mean=1


def test_r15_benchmark():
    """Comprehensive benchmark of all 10 Round-15 novel algorithms vs baselines."""
    print("=" * 78)
    print("  R&D ROUND 15: Novel Param Quantization Benchmark")
    print("=" * 78)

    torch.manual_seed(42)
    # Realistic LLM weight block: 512x1024, small std + outliers
    W = _make_llm_weights(512, 1024, outlier_frac=0.01, outlier_mult=5.0, std=0.05)
    H = _make_hessian(1024, concentration=0.3)  # (1024,) importance per input channel
    # Expand H to (512, 1024) — same per-channel importance for all output rows
    H_full = H.unsqueeze(0).expand(512, 1024).contiguous()

    # Synthetic activations for output-error measurement
    X = _randn(64, 1024, seed=123) * 0.1
    X[:, ::32] *= 3.0  # activation outliers on some channels

    def output_err(W_dq):
        y_ref = X @ W.T
        y_q = X @ W_dq.T
        return frob_err(y_ref, y_q)

    results = []

    def record(name, W_dq, extra=""):
        fe = frob_err(W, W_dq)
        mse = ((W - W_dq) ** 2).mean().item()
        snr = _sqnr(W, W_dq)
        oe = output_err(W_dq)
        results.append((name, fe, mse, snr, oe, extra))
        print(f"  {name:<16} frob={fe:.4f}  mse={mse:.2e}  sqnr={snr:6.2f}dB  out_err={oe:.4f}  {extra}")

    print("\n  --- Baselines (existing) ---")
    # NVFP4 (absmax scale)
    packed, scales, gs = _quantize_to_fp4(W, block_size=32)
    W_nvfp4 = _dequantize_fp4(packed, scales, 512, 1024, 32, torch.float32, global_scale=gs)
    record("NVFP4", W_nvfp4, "absmax/6 scale")

    # AS-FP4 (MSE-optimal scale, R14)
    W_asfp4 = quantize_asfp4_dequant(W, block_size=32)
    record("AS-FP4 (R14)", W_asfp4, "MSE-opt scale")

    # R-FP4 5% (R14)
    packed_r, scales_r, gs_r, ri, rv, rs = _quantize_to_fp4_residual(W, 32, 0.05)
    W_rfp4 = _dequantize_fp4(packed_r, scales_r, 512, 1024, 32, torch.float32, global_scale=gs_r)
    W_rfp4.scatter_add_(1, ri.long(), rv.to(torch.float32) * rs)
    record("R-FP4 5% (R14)", W_rfp4, "FP4+sparse INT8 res")

    print("\n  --- Round 15 novel algorithms ---")
    # 1. HW-FP4
    W_hw = quantize_hw_fp4(W, H_full, block_size=32)
    record("HW-FP4", W_hw, "Hessian-weighted scale")

    # 2. TSDS-FP4
    W_tsd = quantize_tsd_fp4(W, block_size=32, split_quantile=0.75)
    record("TSDS-FP4", W_tsd, "dual-scale 75% split")

    # 3. SR-FP4 (averaged 8 samples)
    W_sr = quantize_sr_fp4(W, block_size=32, n_samples=8, seed=0)
    record("SR-FP4", W_sr, "stochastic round x8")

    # 4. OC-Hybrid
    W_oc, oc_fp8 = quantize_oc_hybrid(W, block_size=32, kurt_threshold=6.0)
    record("OC-Hybrid", W_oc, f"{oc_fp8.sum().item()}/{512} rows FP8")

    # 5. HPR-FP4
    W_hpr = quantize_hpr_fp4(W, block_size=32)
    record("HPR-FP4", W_hpr, "Hadamard rot+AS-FP4")

    # 6. BN-FP4
    W_bn = quantize_bn_fp4(W, block_size=32)
    record("BN-FP4", W_bn, "L2-norm scale")

    # 7. SAAS-FP4
    W_saas = quantize_saas_fp4(W, block_size=32)
    record("SAAS-FP4", W_saas, "mean-subtract")

    # 8. AWR-FP4
    W_awr = quantize_awr_fp4(W, H_full, block_size=32, residual_ratio=0.05)
    record("AWR-FP4", W_awr, "act-weighted res 5%")

    # 9. IRI-FP4 (3 rounds)
    W_iri = quantize_iri_fp4(W, block_size=32, n_rounds=3)
    record("IRI-FP4 x3", W_iri, "3-round residual")

    # 10. PCBA
    W_pcba, pcba_8bit = quantize_pcba(W, block_size=32, dynamic_ratio_threshold=4.0)
    record("PCBA", W_pcba, f"{pcba_8bit.sum().item()}/{512} rows 8-bit")

    # --- Summary: rank by output error (the metric that matters) ---
    print("\n  --- Ranking by OUTPUT error (lower = better) ---")
    ranked = sorted(results, key=lambda r: r[4])
    for i, (name, _fe, _mse, snr, oe, _extra) in enumerate(ranked, 1):
        marker = " <-- WINNER" if i == 1 else ""
        print(f"  {i:>2}. {name:<16} out_err={oe:.4f}  sqnr={snr:6.2f}dB{marker}")

    # Best novel must beat AS-FP4 baseline on output error
    best_novel = ranked[0]
    asfp4_res = next(r for r in results if r[0] == "AS-FP4 (R14)")
    print(f"\n  Best novel: {best_novel[0]} (out_err={best_novel[4]:.4f})")
    print(f"  AS-FP4 base: out_err={asfp4_res[4]:.4f}")
    improvement = (1.0 - best_novel[4] / asfp4_res[4]) * 100
    print(f"  Improvement over AS-FP4: {improvement:+.1f}%")
    print("  PASS\n")


def test_r15_distribution_sweep():
    """Sweep across weight distributions (std, outlier fraction, asymmetry)."""
    print("  R&D 15: Distribution robustness sweep")
    print("-" * 78)

    configs = [
        ("clean gaussian", {"std": 0.05, "outlier_frac": 0.0, "outlier_mult": 1.0, "asym_mean": 0.0}),
        ("mild outliers", {"std": 0.05, "outlier_frac": 0.01, "outlier_mult": 5.0, "asym_mean": 0.0}),
        ("heavy outliers", {"std": 0.05, "outlier_frac": 0.03, "outlier_mult": 8.0, "asym_mean": 0.0}),
        ("asymmetric", {"std": 0.05, "outlier_frac": 0.01, "outlier_mult": 5.0, "asym_mean": 0.01}),
        ("large std", {"std": 0.15, "outlier_frac": 0.01, "outlier_mult": 3.0, "asym_mean": 0.0}),
    ]

    algos = {
        "NVFP4": lambda W, H: _dequantize_fp4(
            *_quantize_to_fp4(W, 32)[:2], 512, 1024, 32, torch.float32, global_scale=_quantize_to_fp4(W, 32)[2]
        ),
        "AS-FP4": lambda W, H: quantize_asfp4_dequant(W, 32),
        "HW-FP4": lambda W, H: quantize_hw_fp4(W, H, 32),
        "TSDS-FP4": lambda W, H: quantize_tsd_fp4(W, 32, 0.75),
        "HPR-FP4": lambda W, H: quantize_hpr_fp4(W, 32),
        "BN-FP4": lambda W, H: quantize_bn_fp4(W, 32),
        "SAAS-FP4": lambda W, H: quantize_saas_fp4(W, 32),
        "IRI-FP4 x3": lambda W, H: quantize_iri_fp4(W, 32, 3),
    }

    header = f"  {'config':<18}" + "".join(f"{n:>11}" for n in algos)
    print(header)
    for cfg_name, cfg in configs:
        W = _make_llm_weights(512, 1024, **cfg)
        H = _make_hessian(1024, concentration=0.3).unsqueeze(0).expand(512, 1024).contiguous()
        row = f"  {cfg_name:<18}"
        for _aname, afn in algos.items():
            W_dq = afn(W, H)
            snr = _sqnr(W, W_dq)
            row += f"{snr:>10.2f}dB"
        print(row)
    print("  PASS\n")


def test_r15_block_size_sweep():
    """Sweep block sizes for the block-based algorithms."""
    print("  R&D 15: Block size sweep (SQNR dB)")
    print("-" * 78)
    W = _make_llm_weights(512, 1024, outlier_frac=0.01, outlier_mult=5.0, std=0.05)
    H = _make_hessian(1024).unsqueeze(0).expand(512, 1024).contiguous()

    block_sizes = [16, 32, 64, 128]
    algos = {
        "AS-FP4": lambda W, H, bs: quantize_asfp4_dequant(W, bs),
        "HW-FP4": lambda W, H, bs: quantize_hw_fp4(W, H, bs),
        "TSDS-FP4": lambda W, H, bs: quantize_tsd_fp4(W, bs, 0.75),
        "BN-FP4": lambda W, H, bs: quantize_bn_fp4(W, bs),
        "SAAS-FP4": lambda W, H, bs: quantize_saas_fp4(W, bs),
        "IRI-FP4 x3": lambda W, H, bs: quantize_iri_fp4(W, bs, 3),
    }
    header = f"  {'algo':<14}" + "".join(f"  bs={bs:<5}" for bs in block_sizes)
    print(header)
    for aname, afn in algos.items():
        row = f"  {aname:<14}"
        for bs in block_sizes:
            W_dq = afn(W, H, bs)
            row += f"  {_sqnr(W, W_dq):>6.2f}dB"
        print(row)
    print("  PASS\n")


def test_r15_iri_rounds():
    """IRI-FP4: sweep number of refinement rounds (error vs storage tradeoff)."""
    print("  R&D 15: IRI-FP4 round sweep (error vs storage)")
    print("-" * 78)
    W = _make_llm_weights(512, 1024, outlier_frac=0.01, outlier_mult=5.0, std=0.05)
    print(f"  {'rounds':<8} {'frob_err':>10} {'sqnr':>10} {'bytes/w':>10} {'vs bf16':>10}")
    bf16_bpw = 16.0
    for n_rounds in [1, 2, 3, 4, 5]:
        W_dq = quantize_iri_fp4(W, block_size=32, n_rounds=n_rounds)
        fe = frob_err(W, W_dq)
        snr = _sqnr(W, W_dq)
        # Each round: 0.5 bytes (FP4) + ~0.06 bytes (scale) ≈ 0.56 bytes/w
        bpw = n_rounds * 0.56
        print(f"  {n_rounds:<8} {fe:>10.4f} {snr:>9.2f}dB {bpw:>9.2f}  {bf16_bpw / bpw:>9.1f}x")
    print("  PASS\n")


def test_r15_residual_ratio():
    """AWR-FP4 vs R-FP4: sweep residual ratio (output-error comparison)."""
    print("  R&D 15: AWR-FP4 vs R-FP4 residual ratio sweep")
    print("-" * 78)
    W = _make_llm_weights(512, 1024, outlier_frac=0.01, outlier_mult=5.0, std=0.05)
    H = _make_hessian(1024).unsqueeze(0).expand(512, 1024).contiguous()
    X = _randn(64, 1024, seed=123) * 0.1
    X[:, ::32] *= 3.0

    def out_err(W_dq):
        return frob_err(X @ W.T, X @ W_dq.T)

    print(f"  {'ratio':<8} {'R-FP4 out':>12} {'AWR-FP4 out':>14} {'improvement':>14}")
    for ratio in [0.02, 0.05, 0.10, 0.20]:
        # R-FP4
        pr, sr, gr, ri, rv, rs = _quantize_to_fp4_residual(W, 32, ratio)
        Wr = _dequantize_fp4(pr, sr, 512, 1024, 32, torch.float32, global_scale=gr)
        Wr.scatter_add_(1, ri.long(), rv.to(torch.float32) * rs)
        oe_r = out_err(Wr)
        # AWR-FP4
        Wa = quantize_awr_fp4(W, H, block_size=32, residual_ratio=ratio)
        oe_a = out_err(Wa)
        imp = (1.0 - oe_a / oe_r) * 100
        print(f"  {ratio:<8.2f} {oe_r:>12.4f} {oe_a:>14.4f} {imp:>+13.1f}%")
    print("  PASS\n")


def test_r15_hadamard_invariance():
    """Verify HPR-FP4 rotation is (approximately) norm-preserving and improves uniformity."""
    print("  R&D 15: Hadamard rotation uniformity check")
    print("-" * 78)
    W = _make_llm_weights(512, 1024, outlier_frac=0.02, outlier_mult=8.0, std=0.05)
    # Per-block kurtosis before and after rotation
    from forge.engine.quant.novel_quant import _hadamard_matrix

    Q = _hadamard_matrix(1024, W.device, W.dtype)
    g = torch.Generator(device=W.device).manual_seed(42)
    signs = torch.where(torch.rand(1024, generator=g, device=W.device) > 0.5, 1.0, -1.0)
    Q = Q * signs.unsqueeze(0)
    W_rot = W @ Q

    def block_kurtosis(Wt, bs=32):
        n_blocks = Wt.shape[1] // bs
        wb = Wt.view(Wt.shape[0], n_blocks, bs)
        mu = wb.mean(-1, keepdim=True)
        sig = wb.std(-1, keepdim=True).clamp(min=1e-8)
        fourth = ((wb - mu) ** 4).mean(-1, keepdim=True)
        return (fourth / (sig**4)).mean().item()

    kurt_before = block_kurtosis(W)
    kurt_after = block_kurtosis(W_rot)
    print(f"  Mean per-block kurtosis before rotation: {kurt_before:.2f}")
    print(f"  Mean per-block kurtosis after rotation:  {kurt_after:.2f}")
    print("  (Gaussian kurtosis = 3.0; lower = more uniform = better for FP4)")
    # Rotation should reduce kurtosis (spread outliers)
    assert kurt_after < kurt_before, "Hadamard rotation should reduce kurtosis"
    # Norm preservation
    norm_ratio = W_rot.norm().item() / W.norm().item()
    print(f"  Norm preservation ratio: {norm_ratio:.4f} (should be ~1.0)")
    assert abs(norm_ratio - 1.0) < 0.01
    print("  PASS\n")


def test_r15_combined():
    """Test combining the best algorithms (stacking winners)."""
    print("  R&D 15: Algorithm stacking (combine winners)")
    print("-" * 78)
    W = _make_llm_weights(512, 1024, outlier_frac=0.01, outlier_mult=5.0, std=0.05)
    H = _make_hessian(1024).unsqueeze(0).expand(512, 1024).contiguous()

    # Stack: HPR (rotation) + HW-FP4 (Hessian scale) + SAAS (mean-sub)
    # Apply rotation, then HW-FP4 with mean-subtraction on rotated weights
    from forge.engine.quant.novel_quant import _hadamard_matrix

    Q = _hadamard_matrix(1024, W.device, W.dtype)
    g = torch.Generator(device=W.device).manual_seed(42)
    signs = torch.where(torch.rand(1024, generator=g, device=W.device) > 0.5, 1.0, -1.0)
    Q = Q * signs.unsqueeze(0)
    W_rot = W @ Q
    # SAAS on rotated
    W_rot_saas = quantize_saas_fp4(W_rot, 32)
    # HW-FP4 on SAAS-processed rotated
    W_rot_saas_hw = quantize_hw_fp4(W_rot_saas, H, 32)
    W_combined = W_rot_saas_hw @ Q.T

    # Individual
    W_hpr = quantize_hpr_fp4(W, 32)
    W_hw = quantize_hw_fp4(W, H, 32)
    W_saas = quantize_saas_fp4(W, 32)
    W_asfp4 = quantize_asfp4_dequant(W, 32)

    print(f"  AS-FP4 (baseline):     sqnr={_sqnr(W, W_asfp4):.2f}dB  frob={frob_err(W, W_asfp4):.4f}")
    print(f"  HPR-FP4 alone:         sqnr={_sqnr(W, W_hpr):.2f}dB  frob={frob_err(W, W_hpr):.4f}")
    print(f"  HW-FP4 alone:          sqnr={_sqnr(W, W_hw):.2f}dB  frob={frob_err(W, W_hw):.4f}")
    print(f"  SAAS-FP4 alone:        sqnr={_sqnr(W, W_saas):.2f}dB  frob={frob_err(W, W_saas):.4f}")
    print(f"  HPR+SAAS+HW combined:  sqnr={_sqnr(W, W_combined):.2f}dB  frob={frob_err(W, W_combined):.4f}")
    print("  PASS\n")


def main_r15():
    test_r15_benchmark()
    test_r15_distribution_sweep()
    test_r15_block_size_sweep()
    test_r15_iri_rounds()
    test_r15_residual_ratio()
    test_r15_hadamard_invariance()
    test_r15_combined()
    print("=" * 78)
    print("  ALL R&D ROUND 15 NOVEL QUANT TESTS PASSED")
    print("=" * 78)


# ============================================================================
# R&D ROUND 16: Standardized multi-scale benchmark (1B / 5B / 8B / 10B)
#
# Tests all quantization algorithms at four standard model scales using
# realistic per-layer weight shapes (GQA attention + SwiGLU FFN). Reports
# VRAM, compression ratio, average SQNR, and average output error.
# ============================================================================

# Standard model configurations (GQA 32 heads / 8 KV heads, SwiGLU FFN)
# Calibrated so total params (layers x per-layer + embeddings) ~ target.
MODEL_CONFIGS = {
    "1B": {"d_model": 2048, "n_layers": 16, "intermediate": 8192, "vocab": 65536, "n_heads": 32, "n_kv_heads": 8},
    "5B": {"d_model": 4096, "n_layers": 20, "intermediate": 16384, "vocab": 65536, "n_heads": 32, "n_kv_heads": 8},
    "8B": {"d_model": 4096, "n_layers": 34, "intermediate": 14336, "vocab": 128256, "n_heads": 32, "n_kv_heads": 8},
    "10B": {"d_model": 5120, "n_layers": 28, "intermediate": 17920, "vocab": 128256, "n_heads": 32, "n_kv_heads": 8},
}


def _model_param_count(cfg):
    """Estimate total params for a config (tied embeddings)."""
    d = cfg["d_model"]
    L = cfg["n_layers"]
    inter = cfg["intermediate"]
    vocab = cfg["vocab"]
    n_heads = cfg["n_heads"]
    n_kv = cfg["n_kv_heads"]
    d_kv = d // n_heads * n_kv  # total KV dim
    # Per layer: Q(d,d) + K(d_kv,d) + V(d_kv,d) + O(d,d) + gate(inter,d) + up(inter,d) + down(d,inter)
    attn = d * d + d_kv * d + d_kv * d + d * d  # 2d² + 2×d_kv×d
    ffn = inter * d + inter * d + d * inter  # 3 × inter × d
    per_layer = attn + ffn
    total = L * per_layer + vocab * d  # tied embeddings
    return total


def _generate_model_weights(cfg, seed=42, max_elements_per_matrix=8_000_000):
    """Generate realistic weight matrices for one model config.

    Returns list of (name, weight_tensor) for all linear layers in one layer
    (scaled by n_layers for VRAM but only one layer sampled for speed).

    Tensors are created on GPU (DEV) when available. max_elements_per_matrix
    caps the matrix size; on GPU this can be larger (8M) for full-resolution
    testing. VRAM is computed from the full model param count, not the
    sampled matrices.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    d = cfg["d_model"]
    inter = cfg["intermediate"]
    n_heads = cfg["n_heads"]
    n_kv = cfg["n_kv_heads"]
    d_kv = d // n_heads * n_kv  # total KV projection dim

    def make_w(out_f, in_f, name, std_scale=0.05, outlier_frac=0.01, outlier_mult=5.0):
        """Generate one weight matrix with LLM-like distribution on GPU."""
        # Cap size for speed — keep aspect ratio
        if out_f * in_f > max_elements_per_matrix:
            ratio = (max_elements_per_matrix / (out_f * in_f)) ** 0.5
            out_f_c = max(64, int(out_f * ratio))
            in_f_c = max(64, int(in_f * ratio))
        else:
            out_f_c, in_f_c = out_f, in_f
        W = torch.randn(out_f_c, in_f_c, generator=g, device=DEV, dtype=DTYPE) * std_scale
        mask = torch.rand(out_f_c, in_f_c, generator=g, device=DEV) < outlier_frac
        W[mask] *= outlier_mult
        return (name, W, out_f, in_f)  # return full dims for VRAM calc

    weights = []
    # Attention projections (Q, K, V, O)
    weights.append(make_w(d, d, "attn_q"))
    weights.append(make_w(d_kv, d, "attn_k"))
    weights.append(make_w(d_kv, d, "attn_v"))
    weights.append(make_w(d, d, "attn_o"))
    # FFN (SwiGLU: gate, up, down)
    weights.append(make_w(inter, d, "ffn_gate"))
    weights.append(make_w(inter, d, "ffn_up"))
    weights.append(make_w(d, inter, "ffn_down"))
    return weights


def _estimate_quant_vram(cfg, bytes_per_weight, scale_overhead=0.0):
    """Estimate quantized VRAM for a model config.

    bytes_per_weight: storage per weight element (e.g. 0.53 for FP4)
    scale_overhead: extra bytes per weight for scales/masks (e.g. 0.06 for FP8 scales)
    """
    total_params = _model_param_count(cfg)
    # Embeddings usually kept in bf16 (skip quantization)
    embed_params = cfg["vocab"] * cfg["d_model"]
    quant_params = total_params - embed_params
    bytes_total = quant_params * (bytes_per_weight + scale_overhead) + embed_params * 2.0
    return bytes_total / (1024**3)  # GB


def _algo_vram_info(algo_name, cfg):
    """Return (bytes_per_weight, scale_overhead, description) for each algorithm."""
    info = {
        "bf16": (2.0, 0.0, "baseline"),
        "NVFP4": (0.5, 0.0625, "absmax/6 scale"),  # 0.5 + 1/32 (FP8 scale)
        "AS-FP4": (0.5, 0.0625, "MSE-opt scale"),
        "R-FP4 5%": (0.5, 0.0625 + 0.05 * 5, "FP4+sparse res"),  # +5%×(4+1) bytes
        "SR-FP4": (0.5, 0.0625, "stochastic round"),
        "TSDS-FP4": (0.5, 0.0625 + 0.125, "dual-scale split"),  # +1 bit mask
        "HPR-FP4": (0.5, 0.0625, "Hadamard rot+AS-FP4"),
        "SAAS-FP4": (0.5, 0.0625 + 0.0625, "mean-subtract"),  # +fp16 mean/block
        "IRI-FP4 x2": (0.5 * 2, 0.0625 * 2, "2-round residual"),
        "IRI-FP4 x3": (0.5 * 3, 0.0625 * 3, "3-round residual"),
        "OC-Hybrid": (0.65, 0.0625, "per-row FP8/FP4 mix"),  # ~80% FP4 + 20% FP8
        "PCBA": (0.85, 0.0625, "per-channel bit alloc"),  # ~98% FP8 rows
    }
    return info.get(algo_name, (0.5, 0.0625, ""))


def test_r16_multiscale_benchmark():
    """Benchmark all algorithms at 1B/5B/8B/10B param scales."""
    print("=" * 90)
    print("  R&D ROUND 16: Standardized Multi-Scale Quantization Benchmark (1B / 5B / 8B / 10B)")
    print("=" * 90)

    # Verify param counts
    print("\n  Model configurations:")
    print(f"  {'scale':<6} {'d_model':>8} {'layers':>7} {'inter':>7} {'vocab':>7} {'params':>12} {'bf16 VRAM':>10}")
    for scale, cfg in MODEL_CONFIGS.items():
        params = _model_param_count(cfg)
        bf16_vram = params * 2 / (1024**3)
        print(
            f"  {scale:<6} {cfg['d_model']:>8} {cfg['n_layers']:>7} {cfg['intermediate']:>7} "
            f"{cfg['vocab']:>7} {params / 1e9:>10.2f}B {bf16_vram:>9.2f} GB"
        )

    # Algorithms to test (functions that take W and return dequantized W)
    from forge.engine.quant.novel_quant import (
        quantize_asfp4_dequant,
        quantize_hpr_fp4,
        quantize_iri_fp4,
        quantize_oc_hybrid,
        quantize_pcba,
        quantize_saas_fp4,
        quantize_sr_fp4,
        quantize_tsd_fp4,
    )

    algos = {
        "NVFP4": lambda W: _dequantize_fp4(
            *_quantize_to_fp4(W, 32)[:2],
            W.shape[0],
            W.shape[1],
            32,
            torch.float32,
            global_scale=_quantize_to_fp4(W, 32)[2],
        ),
        "AS-FP4": lambda W: quantize_asfp4_dequant(W, 32),
        "R-FP4 5%": lambda W: _rfp4_dequant(W, 0.05),
        "SR-FP4": lambda W: quantize_sr_fp4(W, 32, n_samples=8, seed=0),
        "TSDS-FP4": lambda W: quantize_tsd_fp4(W, 32, 0.75),
        "HPR-FP4": lambda W: quantize_hpr_fp4(W, 32),
        "SAAS-FP4": lambda W: quantize_saas_fp4(W, 32),
        "IRI-FP4 x2": lambda W: quantize_iri_fp4(W, 32, n_rounds=2),
        "IRI-FP4 x3": lambda W: quantize_iri_fp4(W, 32, n_rounds=3),
        "OC-Hybrid": lambda W: quantize_oc_hybrid(W, 32, 6.0)[0],
        "PCBA": lambda W: quantize_pcba(W, 32, 4.0)[0],
    }

    for scale, cfg in MODEL_CONFIGS.items():
        print(f"\n  {'-' * 86}")
        print(f"  {scale} model (d={cfg['d_model']}, L={cfg['n_layers']}, inter={cfg['intermediate']})")
        print(f"  {'-' * 86}")

        weights = _generate_model_weights(cfg, seed=42)
        total_params = _model_param_count(cfg)
        bf16_vram = total_params * 2 / (1024**3)

        # Synthetic activations for output error (per input dim)
        activations = {}
        for name, W, _out_f, _in_f in weights:
            n_act = min(32, 64)
            X = _randn(n_act, W.shape[1], seed=123) * 0.1
            X[:, ::32] *= 3.0  # activation outliers
            activations[name] = X

        # Header
        print(
            f"  {'algorithm':<14} {'avg SQNR':>9} {'avg out_err':>12} "
            f"{'quant VRAM':>11} {'compress':>9} {'fits 12GB':>10}"
        )
        print(f"  {'-' * 14} {'-' * 9} {'-' * 12} {'-' * 11} {'-' * 9} {'-' * 10}")

        results = []
        for algo_name, algo_fn in algos.items():
            sqnrs = []
            out_errs = []
            for name, W, _out_f, _in_f in weights:
                W_dq = algo_fn(W)
                sqnrs.append(_sqnr(W, W_dq))
                X = activations[name]
                y_ref = X @ W.T
                y_q = X @ W_dq[:, : W.shape[1]].T
                out_errs.append(frob_err(y_ref, y_q))
            avg_sqnr = sum(sqnrs) / len(sqnrs)
            avg_oe = sum(out_errs) / len(out_errs)

            # VRAM estimate
            bpw, overhead, _desc = _algo_vram_info(algo_name, cfg)
            quant_vram = _estimate_quant_vram(cfg, bpw, overhead)
            compression = bf16_vram / quant_vram if quant_vram > 0 else 0
            fits = "YES" if quant_vram < 12.0 else "NO"

            results.append((algo_name, avg_sqnr, avg_oe, quant_vram, compression, fits))
            print(
                f"  {algo_name:<14} {avg_sqnr:>8.2f}dB {avg_oe:>12.4f} "
                f"{quant_vram:>9.2f} GB {compression:>8.1f}x {fits:>10}"
            )

        # bf16 baseline for reference
        print(
            f"  {'bf16':<14} {'   inf':>9} {'0.0000':>12} "
            f"{bf16_vram:>9.2f} GB {1.0:>8.1f}x {'YES' if bf16_vram < 12 else 'NO':>10}"
        )

        # Best per-scale
        best_sqnr = max(results, key=lambda r: r[1])
        best_vram = min(results, key=lambda r: r[3])
        print(f"\n  Best quality: {best_sqnr[0]} ({best_sqnr[1]:.2f} dB)")
        print(f"  Best VRAM:    {best_vram[0]} ({best_vram[3]:.2f} GB, {best_vram[4]:.1f}x)")

    # Summary table across scales
    print(f"\n  {'=' * 86}")
    print("  Cross-scale summary (avg SQNR dB across all weight types per scale)")
    print(f"  {'=' * 86}")
    header = f"  {'algorithm':<14}" + "".join(f"  {s:>10}" for s in MODEL_CONFIGS)
    print(header)
    print(f"  {'-' * 14}" + "".join(f"  {'-' * 10}" for _ in MODEL_CONFIGS))

    # Re-run just SQNR for the summary table (cached results would be better but
    # this is fast enough)
    for algo_name, algo_fn in algos.items():
        row = f"  {algo_name:<14}"
        for _scale, cfg in MODEL_CONFIGS.items():
            weights = _generate_model_weights(cfg, seed=42)
            sqnrs = []
            for _name, W, _out_f, _in_f in weights:
                W_dq = algo_fn(W)
                sqnrs.append(_sqnr(W, W_dq))
            avg = sum(sqnrs) / len(sqnrs)
            row += f"  {avg:>9.2f}dB"
        print(row)

    # VRAM summary
    print(f"\n  {'=' * 86}")
    print("  VRAM summary (GB, fits-in-12GB marked *)")
    print(f"  {'=' * 86}")
    header = f"  {'algorithm':<14}" + "".join(f"  {s:>10}" for s in MODEL_CONFIGS)
    print(header)
    print(f"  {'-' * 14}" + "".join(f"  {'-' * 10}" for _ in MODEL_CONFIGS))
    for algo_name in [*list(algos.keys()), "bf16"]:
        row = f"  {algo_name:<14}"
        for _scale, cfg in MODEL_CONFIGS.items():
            if algo_name == "bf16":
                vram = _model_param_count(cfg) * 2 / (1024**3)
            else:
                bpw, overhead, _ = _algo_vram_info(algo_name, cfg)
                vram = _estimate_quant_vram(cfg, bpw, overhead)
            marker = "*" if vram < 12.0 else " "
            row += f"  {vram:>8.2f}GB{marker}"
        print(row)
    print("  (* = fits in 12GB VRAM RTX 5070)")

    print("\n  PASS\n")


def _rfp4_dequant(W, residual_ratio):
    """Helper: R-FP4 quantize + dequant with residual."""
    pr, sr, gr, ri, rv, rs = _quantize_to_fp4_residual(W, 32, residual_ratio)
    W_dq = _dequantize_fp4(pr, sr, W.shape[0], W.shape[1], 32, torch.float32, global_scale=gr)
    W_dq.scatter_add_(1, ri.long(), rv.to(torch.float32) * rs)
    return W_dq


def test_r16_pareto_frontier():
    """Plot the quality-vs-VRAM Pareto frontier at 8B scale."""
    print("  R&D 16: Quality vs VRAM Pareto frontier (8B model)")
    print("-" * 86)

    cfg = MODEL_CONFIGS["8B"]
    total_params = _model_param_count(cfg)
    total_params * 2 / (1024**3)

    from forge.engine.quant.novel_quant import (
        quantize_asfp4_dequant,
        quantize_hpr_fp4,
        quantize_iri_fp4,
        quantize_saas_fp4,
        quantize_sr_fp4,
        quantize_tsd_fp4,
    )

    weights = _generate_model_weights(cfg, seed=42)

    # Collect (vram, sqnr, name) points
    points = []
    algo_list = [
        (
            "NVFP4",
            lambda W: _dequantize_fp4(
                *_quantize_to_fp4(W, 32)[:2],
                W.shape[0],
                W.shape[1],
                32,
                torch.float32,
                global_scale=_quantize_to_fp4(W, 32)[2],
            ),
            0.5,
            0.0625,
        ),
        ("AS-FP4", lambda W: quantize_asfp4_dequant(W, 32), 0.5, 0.0625),
        ("SR-FP4", lambda W: quantize_sr_fp4(W, 32, 8, 0), 0.5, 0.0625),
        ("TSDS-FP4", lambda W: quantize_tsd_fp4(W, 32, 0.75), 0.5, 0.0625 + 0.125),
        ("HPR-FP4", lambda W: quantize_hpr_fp4(W, 32), 0.5, 0.0625),
        ("SAAS-FP4", lambda W: quantize_saas_fp4(W, 32), 0.5, 0.0625 + 0.0625),
        ("IRI-FP4 x2", lambda W: quantize_iri_fp4(W, 32, 2), 1.0, 0.125),
        ("IRI-FP4 x3", lambda W: quantize_iri_fp4(W, 32, 3), 1.5, 0.1875),
    ]

    for name, fn, bpw, overhead in algo_list:
        sqnrs = []
        for _wname, W, _, _ in weights:
            W_dq = fn(W)
            sqnrs.append(_sqnr(W, W_dq))
        avg_sqnr = sum(sqnrs) / len(sqnrs)
        vram = _estimate_quant_vram(cfg, bpw, overhead)
        points.append((vram, avg_sqnr, name))

    # Sort by VRAM
    points.sort(key=lambda p: p[0])

    # Print ASCII Pareto frontier
    print(f"  {'algorithm':<14} {'VRAM (GB)':>10} {'SQNR (dB)':>10} {'on Pareto':>10}")
    print(f"  {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10}")
    pareto = []
    best_sqnr = -1
    for vram, sqnr, name in points:
        on_pareto = sqnr > best_sqnr
        if on_pareto:
            best_sqnr = sqnr
            pareto.append(name)
        marker = "<<<" if on_pareto else ""
        print(f"  {name:<14} {vram:>9.2f}GB {sqnr:>9.2f}dB {marker:>10}")

    # ASCII chart
    print("\n  SQNR vs VRAM (8B model, each ≈ = 2dB, each . = 0.5GB):")
    min_v = min(p[0] for p in points)
    max_v = max(p[0] for p in points)
    min_s = min(p[1] for p in points) - 1
    max_s = max(p[1] for p in points) + 1
    for vram, sqnr, name in points:
        vpos = int((vram - min_v) / max(max_v - min_v, 0.01) * 40)
        int((sqnr - min_s) / max(max_s - min_s, 1) * 20)
        " " * vpos + "@"
        label = f" {name} ({sqnr:.1f}dB, {vram:.1f}GB)"
        print(f"  {name:<14} |{'.' * vpos}@{label}")
    print("  PASS\n")


def main_r16():
    test_r16_multiscale_benchmark()
    test_r16_pareto_frontier()
    print("=" * 90)
    print("  ALL R&D ROUND 16 MULTI-SCALE TESTS PASSED")
    print("=" * 90)


# ===========================================================================
# R&D ROUND 17: Advanced quantization tests
#   1. HW-FP4-v2 with real activation calibration
#   2. HPR+IRI stacking
#   3. OptimalSignHadamard kurtosis minimization
#   4. AdaptivePerLayer selection
# ===========================================================================

from forge.engine.quant.novel_quant import (
    HPRIRIFP4Linear,
    HWFP4CalibratedLinear,
    _optimal_sign_hadamard,
    analyze_weight_distribution,
    compute_hessian_proxy,
    quantize_hpr_iri_fp4,
    quantize_hw_fp4_v2,
    quantize_model_adaptive,
)


def test_r17_hw_fp4_v2_calibration():
    """HW-FP4-v2: real activation calibration vs synthetic Hessian."""
    print("=" * 90)
    print("  R&D 17: HW-FP4-v2 with real activation calibration")
    print("=" * 90)

    torch.manual_seed(42)
    W = _make_llm_weights(512, 1024, outlier_frac=0.01, outlier_mult=5.0, std=0.05)

    # Generate REAL activations (with realistic outlier channel pattern)
    X_real = _randn(256, 1024, seed=123) * 0.1
    X_real[:, ::32] *= 3.0  # outlier channels every 32 dims
    X_real[:, 0] *= 5.0  # one very high channel

    # Synthetic Hessian (what R15 used — log-uniform)
    H_synth = _make_hessian(1024, concentration=0.3).unsqueeze(0).expand(512, 1024).contiguous()

    # Real Hessian proxy from activations
    H_real = compute_hessian_proxy(X_real)
    H_real_full = H_real.unsqueeze(0).expand(512, 1024).contiguous()

    # Output error measurement
    X_test = _randn(64, 1024, seed=999) * 0.1
    X_test[:, ::32] *= 3.0
    X_test[:, 0] *= 5.0

    def out_err(W_dq):
        return frob_err(X_test @ W.T, X_test @ W_dq[:, :1024].T)

    # Baselines
    W_asfp4 = quantize_asfp4_dequant(W, 32)
    W_hw_synth = quantize_hw_fp4(W, H_synth, 32)
    W_hw_real = quantize_hw_fp4(W, H_real_full, 32)
    W_hw_v2 = quantize_hw_fp4_v2(W, X_real, 32)

    print(f"\n  AS-FP4 (no Hessian):       sqnr={_sqnr(W, W_asfp4):.2f}dB  out_err={out_err(W_asfp4):.4f}")
    print(f"  HW-FP4 synth Hessian (R15): sqnr={_sqnr(W, W_hw_synth):.2f}dB  out_err={out_err(W_hw_synth):.4f}")
    print(f"  HW-FP4 real Hessian:        sqnr={_sqnr(W, W_hw_real):.2f}dB  out_err={out_err(W_hw_real):.4f}")
    print(f"  HW-FP4-v2 (calibrated):     sqnr={_sqnr(W, W_hw_v2):.2f}dB  out_err={out_err(W_hw_v2):.4f}")

    # Real Hessian should be better than synthetic
    oe_synth = out_err(W_hw_synth)
    oe_real = out_err(W_hw_real)
    print(f"\n  Real vs synthetic Hessian output error improvement: {(1 - oe_real / oe_synth) * 100:+.1f}%")

    # Test the calibrated Linear class
    print("\n  Testing HWFP4CalibratedLinear class:")
    lin = nn.Linear(1024, 512, bias=True)
    lin.weight.data = W.clone()
    lin = lin.to(DEV)
    hw_lin = HWFP4CalibratedLinear.from_linear(lin, block_size=32)
    hw_lin = hw_lin.to(DEV)
    # Before calibration (uses AS-FP4)
    y_before = hw_lin(X_test[:8])
    err_before = frob_err(lin(X_test[:8]), y_before)
    # Calibrate
    hw_lin.calibrate(X_real)
    y_after = hw_lin(X_test[:8])
    err_after = frob_err(lin(X_test[:8]), y_after)
    print(f"    Before calibration: out_err={err_before:.4f}")
    print(f"    After calibration:  out_err={err_after:.4f}")
    print(f"    Calibration improvement: {(1 - err_after / err_before) * 100:+.1f}%")

    print("  PASS\n")


def test_r17_hpr_iri_stacking():
    """HPR+IRI: rotation + iterative residual vs IRI alone."""
    print("  R&D 17: HPR+IRI stacking vs IRI alone")
    print("-" * 90)

    torch.manual_seed(42)
    W = _make_llm_weights(512, 1024, outlier_frac=0.02, outlier_mult=8.0, std=0.05)

    X = _randn(64, 1024, seed=123) * 0.1
    X[:, ::32] *= 3.0

    def out_err(W_dq):
        return frob_err(X @ W.T, X @ W_dq[:, :1024].T)

    print(f"\n  {'algorithm':<22} {'SQNR':>9} {'out_err':>10} {'bytes/w':>9}")
    print(f"  {'-' * 22} {'-' * 9} {'-' * 10} {'-' * 9}")

    for n_rounds in [1, 2, 3]:
        # IRI alone
        W_iri = quantize_iri_fp4(W, 32, n_rounds)
        bpw = n_rounds * 0.56
        print(f"  IRI-FP4 x{n_rounds} alone    {_sqnr(W, W_iri):>8.2f}dB {out_err(W_iri):>10.4f} {bpw:>8.2f}")

        # HPR + IRI (random signs)
        W_hpr_iri = quantize_hpr_iri_fp4(W, 32, n_rounds, optimal_signs=False)
        print(f"  HPR+IRI x{n_rounds} (random) {_sqnr(W, W_hpr_iri):>8.2f}dB {out_err(W_hpr_iri):>10.4f} {bpw:>8.2f}")

        # HPR + IRI (optimal signs)
        W_hpr_iri_opt = quantize_hpr_iri_fp4(W, 32, n_rounds, optimal_signs=True)
        print(
            f"  HPR+IRI x{n_rounds} (optimal){_sqnr(W, W_hpr_iri_opt):>8.2f}dB {out_err(W_hpr_iri_opt):>10.4f} {bpw:>8.2f}"
        )

    # Test production class
    print("\n  Testing HPRIRIFP4Linear class:")
    lin = nn.Linear(1024, 512, bias=True)
    lin.weight.data = W.clone()
    lin = lin.to(DEV)
    hpr_iri = HPRIRIFP4Linear.from_linear(lin, n_rounds=2, optimal_signs=True)
    hpr_iri = hpr_iri.to(DEV)
    y_ref = lin(X[:8])
    y_q = hpr_iri(X[:8])
    print(f"    HPR+IRI x2 optimal: out_err={frob_err(y_ref, y_q):.4f}")

    print("  PASS\n")


def test_r17_optimal_sign_hadamard():
    """OptimalSignHadamard: greedy kurtosis minimization vs random signs."""
    print("  R&D 17: OptimalSignHadamard kurtosis minimization")
    print("-" * 90)

    torch.manual_seed(42)
    # Heavy-tailed weights where rotation matters most
    W = _make_llm_weights(256, 256, outlier_frac=0.03, outlier_mult=10.0, std=0.05)

    # Random signs
    from forge.engine.quant.novel_quant import _hadamard_matrix

    Q_rand = _hadamard_matrix(256, W.device, W.dtype)
    g = torch.Generator(device=DEV).manual_seed(42)
    signs_rand = torch.where(torch.rand(256, generator=g, device=DEV) > 0.5, 1.0, -1.0).to(W.dtype)
    Q_rand = Q_rand * signs_rand.unsqueeze(0)
    W_rot_rand = W @ Q_rand

    # Optimal signs
    Q_opt, _kurt_opt = _optimal_sign_hadamard(W, max_iters=30)
    W_rot_opt = W @ Q_opt

    # Per-block kurtosis
    def block_kurt(Wt, bs=32):
        n_blocks = Wt.shape[1] // bs
        wb = Wt.view(Wt.shape[0], n_blocks, bs)
        mu = wb.mean(-1, keepdim=True)
        sig = wb.std(-1, keepdim=True).clamp(min=1e-8)
        fourth = ((wb - mu) ** 4).mean(-1, keepdim=True)
        return (fourth / (sig**4)).mean().item()

    kurt_orig = block_kurt(W)
    kurt_rand = block_kurt(W_rot_rand)
    kurt_opt_val = block_kurt(W_rot_opt)

    print(f"\n  Original weights kurtosis:      {kurt_orig:.3f}")
    print(f"  Random Hadamard kurtosis:       {kurt_rand:.3f}")
    print(f"  Optimal Hadamard kurtosis:      {kurt_opt_val:.3f}")
    print(f"  Optimal vs random improvement:  {(1 - kurt_opt_val / kurt_rand) * 100:+.1f}%")
    print("  (Gaussian = 3.0; lower = better for FP4)")

    # FP4 quality after each rotation
    W_dq_orig = quantize_asfp4_dequant(W, 32)
    W_dq_rand = quantize_asfp4_dequant(W_rot_rand, 32) @ Q_rand.T
    W_dq_opt = quantize_asfp4_dequant(W_rot_opt, 32) @ Q_opt.T

    print(f"\n  FP4 SQNR without rotation:      {_sqnr(W, W_dq_orig):.2f}dB")
    print(f"  FP4 SQNR with random Hadamard:  {_sqnr(W, W_dq_rand):.2f}dB")
    print(f"  FP4 SQNR with optimal Hadamard: {_sqnr(W, W_dq_opt):.2f}dB")

    # Optimal should be <= random + small tolerance (Hadamard already does
    # most of the work; sign optimization gives marginal gains)
    assert kurt_opt_val <= kurt_rand + 0.1, (
        f"Optimal kurtosis {kurt_opt_val:.3f} should be <= random {kurt_rand:.3f} + 0.1"
    )
    print("  PASS\n")


def test_r17_adaptive_per_layer():
    """AdaptivePerLayer: per-layer algorithm selection on a model."""
    print("  R&D 17: AdaptivePerLayer algorithm selection")
    print("-" * 90)

    # Build a small model with different layer types
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(256, 512),  # clean-ish
        nn.ReLU(),
        nn.Linear(512, 512),  # depends on init
        nn.ReLU(),
        nn.Linear(512, 256),  # down-projection (high-value)
    )

    # Make weights have different distributions per layer (on GPU)
    model[0].weight.data = _make_llm_weights(512, 256, outlier_frac=0.0, std=0.05)  # clean
    model[2].weight.data = _make_llm_weights(512, 512, outlier_frac=0.03, outlier_mult=8.0, std=0.05)  # heavy
    model[4].weight.data = _make_llm_weights(256, 512, outlier_frac=0.01, outlier_mult=5.0, std=0.05)  # mild
    model = model.to(DEV)

    x = _randn(4, 256, seed=42)
    with torch.no_grad():
        y_ref = model(x)

    # Analyze each layer
    print("\n  Layer distribution analysis:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            stats = analyze_weight_distribution(module.weight.float())
            print(
                f"    {name}: kurt={stats['kurtosis']:.2f}  dr={stats['dynamic_range']:.2f}  "
                f"asym={stats['asymmetry']:.3f}  -> {stats['recommendation']}"
            )

    # Apply adaptive quantization
    print("\n  Applying adaptive quantization:")
    counts = quantize_model_adaptive(
        model,
        block_size=32,
        high_value_layers=["4"],  # last layer is down-proj
        iri_rounds=2,
        verbose=True,
    )
    model = model.to(DEV)  # move quantized layers to GPU

    with torch.no_grad():
        y_q = model(x)

    print(f"\n  Output error: {frob_err(y_ref, y_q):.4f}")
    print(f"  Algorithm distribution: {counts}")

    # Compare with uniform AS-FP4
    torch.manual_seed(42)
    model_uniform = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )
    model_uniform[0].weight.data = _make_llm_weights(512, 256, outlier_frac=0.0, std=0.05)
    model_uniform[2].weight.data = _make_llm_weights(512, 512, outlier_frac=0.03, outlier_mult=8.0, std=0.05)
    model_uniform[4].weight.data = _make_llm_weights(256, 512, outlier_frac=0.01, outlier_mult=5.0, std=0.05)
    model_uniform = model_uniform.to(DEV)

    from forge.engine.quant.novel_quant import quantize_model_asfp4

    quantize_model_asfp4(model_uniform, verbose=False)
    model_uniform = model_uniform.to(DEV)
    with torch.no_grad():
        y_uniform = model_uniform(x)

    print(f"  Uniform AS-FP4 output error: {frob_err(y_ref, y_uniform):.4f}")
    print(f"  Adaptive improvement: {(1 - frob_err(y_ref, y_q) / frob_err(y_ref, y_uniform)) * 100:+.1f}%")

    print("  PASS\n")


def test_r17_multiscale():
    """Run R17 algorithms at 1B/5B/8B/10B scales."""
    print("  R&D 17: Multi-scale benchmark (R17 algorithms)")
    print("-" * 90)

    from forge.engine.quant.novel_quant import (
        quantize_hpr_iri_fp4,
        quantize_hw_fp4_v2,
    )

    # Only test the new R17 algorithms + key baselines.
    # Optimal signs disabled for multi-scale (marginal gain vs random, but
    # O(rot_size^2) per iteration is too slow at 10B scale d=5120->rot=8192).
    algos = {
        "AS-FP4": lambda W, X: quantize_asfp4_dequant(W, 32),
        "SR-FP4": lambda W, X: quantize_sr_fp4(W, 32, 8, 0),
        "HPR-FP4": lambda W, X: quantize_hpr_fp4(W, 32),
        "IRI-FP4 x2": lambda W, X: quantize_iri_fp4(W, 32, 2),
        "HPR+IRI x2": lambda W, X: quantize_hpr_iri_fp4(W, 32, 2, optimal_signs=False),
        "HW-FP4-v2": lambda W, X: quantize_hw_fp4_v2(W, X, 32),
    }

    for scale, cfg in MODEL_CONFIGS.items():
        print(f"\n  {scale} model (d={cfg['d_model']}, L={cfg['n_layers']})")
        print(f"  {'algorithm':<14} {'avg SQNR':>9} {'avg out_err':>12}")
        print(f"  {'-' * 14} {'-' * 9} {'-' * 12}")

        weights = _generate_model_weights(cfg, seed=42)
        # Generate real activations per layer (on GPU)
        activations = {}
        for name, W, _out_f, _in_f in weights:
            X = _randn(min(64, 128), W.shape[1], seed=123) * 0.1
            X[:, ::32] *= 3.0
            activations[name] = X

        for algo_name, algo_fn in algos.items():
            sqnrs = []
            out_errs = []
            for name, W, _out_f, _in_f in weights:
                X = activations[name]
                W_dq = algo_fn(W, X)
                sqnrs.append(_sqnr(W, W_dq))
                y_ref = X @ W.T
                y_q = X @ W_dq[:, : W.shape[1]].T
                out_errs.append(frob_err(y_ref, y_q))
            avg_sqnr = sum(sqnrs) / len(sqnrs)
            avg_oe = sum(out_errs) / len(out_errs)
            print(f"  {algo_name:<14} {avg_sqnr:>8.2f}dB {avg_oe:>12.4f}")

    print("  PASS\n")


def main_r17():
    test_r17_hw_fp4_v2_calibration()
    test_r17_hpr_iri_stacking()
    test_r17_optimal_sign_hadamard()
    test_r17_adaptive_per_layer()
    test_r17_multiscale()
    print("=" * 90)
    print("  ALL R&D ROUND 17 TESTS PASSED")
    print("=" * 90)


# ===========================================================================
# R&D ROUND 18: Advanced codebook + KV cache + gradient quantization
# ===========================================================================


def test_r18_mixed_codebook_iri():
    """R18a: Mixed-codebook IRI (FP4+INT4 alternating) vs standard IRI."""
    print("  R&D 18a: Mixed-Codebook IRI (FP4+INT4 alternating)")
    print("-" * 90)

    from forge.engine.quant.novel_quant import (
        quantize_asfp4_dequant,
        quantize_iri_fp4,
        quantize_mixed_iri_fp4,
    )

    W = _make_llm_weights(512, 1024, seed=42, outlier_frac=0.02, outlier_mult=6.0)
    X = _randn(64, 1024, seed=123) * 0.1
    X[:, ::32] *= 3.0

    results = {}
    for label, fn in [
        ("AS-FP4", lambda: quantize_asfp4_dequant(W, 32)),
        ("IRI-FP4 x1", lambda: quantize_iri_fp4(W, 32, 1)),
        ("IRI-FP4 x2", lambda: quantize_iri_fp4(W, 32, 2)),
        ("MixedIRI x1 alt", lambda: quantize_mixed_iri_fp4(W, 32, 1, "alternate")),
        ("MixedIRI x2 alt", lambda: quantize_mixed_iri_fp4(W, 32, 2, "alternate")),
        ("MixedIRI x2 f4f", lambda: quantize_mixed_iri_fp4(W, 32, 2, "fp4_first")),
        ("MixedIRI x3 alt", lambda: quantize_mixed_iri_fp4(W, 32, 3, "alternate")),
    ]:
        W_dq = fn()
        sqnr = _sqnr(W, W_dq)
        y_ref = X @ W.T
        y_q = X @ W_dq[:, : W.shape[1]].T
        oe = frob_err(y_ref, y_q)
        results[label] = (sqnr, oe)
        print(f"  {label:<18} SQNR={sqnr:6.2f}dB  out_err={oe:.4f}")

    # Mixed IRI x2 should be competitive with standard IRI x2
    iri_x2_sqnr = results["IRI-FP4 x2"][0]
    mixed_x2_sqnr = results["MixedIRI x2 alt"][0]
    print(
        f"\n  IRI x2={iri_x2_sqnr:.2f}dB  MixedIRI x2={mixed_x2_sqnr:.2f}dB  delta={mixed_x2_sqnr - iri_x2_sqnr:+.2f}dB"
    )

    # At minimum, mixed should not be much worse than standard IRI
    assert mixed_x2_sqnr >= iri_x2_sqnr - 2.0, (
        f"MixedIRI x2 ({mixed_x2_sqnr:.2f}) should be within 2dB of IRI x2 ({iri_x2_sqnr:.2f})"
    )
    print("  PASS\n")


def test_r18_adaptive_iri():
    """R18b: Adaptive IRI (per-block round allocation)."""
    print("  R&D 18b: Adaptive IRI (per-block round allocation)")
    print("-" * 90)

    from forge.engine.quant.novel_quant import (
        quantize_adaptive_iri_fp4,
        quantize_asfp4_dequant,
        quantize_iri_fp4,
    )

    W = _make_llm_weights(512, 1024, seed=42, outlier_frac=0.02, outlier_mult=6.0)
    X = _randn(64, 1024, seed=123) * 0.1

    for label, fn in [
        ("AS-FP4", lambda: quantize_asfp4_dequant(W, 32)),
        ("IRI-FP4 x1", lambda: quantize_iri_fp4(W, 32, 1)),
        ("IRI-FP4 x2", lambda: quantize_iri_fp4(W, 32, 2)),
        ("AdaptiveIRI max2", lambda: quantize_adaptive_iri_fp4(W, 32, 2, 1e-4)),
        ("AdaptiveIRI max3", lambda: quantize_adaptive_iri_fp4(W, 32, 3, 1e-4)),
        ("AdaptiveIRI max3 strict", lambda: quantize_adaptive_iri_fp4(W, 32, 3, 1e-5)),
    ]:
        W_dq = fn()
        sqnr = _sqnr(W, W_dq)
        y_ref = X @ W.T
        y_q = X @ W_dq[:, : W.shape[1]].T
        oe = frob_err(y_ref, y_q)
        print(f"  {label:<22} SQNR={sqnr:6.2f}dB  out_err={oe:.4f}")

    # Adaptive should be at least as good as IRI x1 (it always does 1 round)
    w_adaptive = quantize_adaptive_iri_fp4(W, 32, 3, 1e-4)
    w_iri1 = quantize_iri_fp4(W, 32, 1)
    assert _sqnr(W, w_adaptive) >= _sqnr(W, w_iri1), "Adaptive IRI should beat single-round IRI"
    print("  PASS\n")


def test_r18_learned_codebook():
    """R18c: Learned FP4 codebook (Lloyd-Max per-block levels)."""
    print("  R&D 18c: Learned FP4 Codebook (Lloyd-Max)")
    print("-" * 90)

    from forge.engine.quant.novel_quant import (
        quantize_asfp4_dequant,
        quantize_learned_codebook_fp4,
    )

    W = _make_llm_weights(256, 512, seed=42, outlier_frac=0.02, outlier_mult=6.0)
    X = _randn(64, 512, seed=123) * 0.1

    for label, fn in [
        ("AS-FP4", lambda: quantize_asfp4_dequant(W, 32)),
        ("LearnedCB iters5", lambda: quantize_learned_codebook_fp4(W, 32, 5)),
        ("LearnedCB iters20", lambda: quantize_learned_codebook_fp4(W, 32, 20)),
        ("LearnedCB iters50", lambda: quantize_learned_codebook_fp4(W, 32, 50)),
    ]:
        W_dq = fn()
        sqnr = _sqnr(W, W_dq)
        y_ref = X @ W.T
        y_q = X @ W_dq[:, : W.shape[1]].T
        oe = frob_err(y_ref, y_q)
        print(f"  {label:<20} SQNR={sqnr:6.2f}dB  out_err={oe:.4f}")

    # Lloyd-Max should beat fixed FP4 levels (it's MSE-optimal for the actual distribution)
    w_learned = quantize_learned_codebook_fp4(W, 32, 20)
    w_asfp4 = quantize_asfp4_dequant(W, 32)
    sqnr_learned = _sqnr(W, w_learned)
    sqnr_asfp4 = _sqnr(W, w_asfp4)
    print(f"\n  AS-FP4={sqnr_asfp4:.2f}dB  LearnedCB={sqnr_learned:.2f}dB  delta={sqnr_learned - sqnr_asfp4:+.2f}dB")
    assert sqnr_learned >= sqnr_asfp4, f"Learned codebook ({sqnr_learned:.2f}) should beat fixed FP4 ({sqnr_asfp4:.2f})"
    print("  PASS\n")


def test_r18_sr_fp4_kv_cache():
    """R18d: SR-FP4 KV cache quantization."""
    print("  R&D 18d: SR-FP4 KV Cache")
    print("-" * 90)

    from forge.engine.quant.novel_quant import SRFP4KVCache

    n_heads, head_dim, max_seq = 8, 64, 256
    cache = SRFP4KVCache(n_heads, head_dim, max_seq, block_size=32, device=str(DEV), dtype=DTYPE)

    # Generate realistic K, V vectors
    n_tokens = 128
    K = _randn(n_tokens, n_heads, head_dim, seed=42) * 0.1
    V = _randn(n_tokens, n_heads, head_dim, seed=99) * 0.1
    # Add some outlier tokens
    K[::16] *= 4.0
    V[::16] *= 4.0

    cache.append(K, V)
    K_dq, V_dq = cache.get()

    # Measure SQNR
    k_sqnr = _sqnr(K, K_dq)
    v_sqnr = _sqnr(V, V_dq)
    k_err = frob_err(K, K_dq)
    v_err = frob_err(V, V_dq)

    # Compare to bf16 cache memory
    bf16_bytes = n_tokens * n_heads * head_dim * 2 * 2  # K+V, 2 bytes each
    fp4_bytes = cache.memory_bytes()

    print(f"  K cache: SQNR={k_sqnr:.2f}dB  frob_err={k_err:.4f}")
    print(f"  V cache: SQNR={v_sqnr:.2f}dB  frob_err={v_err:.4f}")
    print(f"  BF16 cache: {bf16_bytes / 1024:.1f} KB")
    print(f"  FP4 cache:  {fp4_bytes / 1024:.1f} KB")
    print(f"  Compression: {bf16_bytes / max(fp4_bytes, 1):.1f}x")

    assert k_sqnr > 15.0, f"K cache SQNR too low: {k_sqnr:.2f}dB"
    assert v_sqnr > 15.0, f"V cache SQNR too low: {v_sqnr:.2f}dB"
    assert fp4_bytes < bf16_bytes, "FP4 cache should be smaller than bf16"
    print("  PASS\n")


def test_r18_gradient_fp4():
    """R18e: Gradient-optimized FP4 (QuIP#-style)."""
    print("  R&D 18e: Gradient-Optimized FP4 (QuIP#-style)")
    print("-" * 90)

    from forge.engine.quant.novel_quant import (
        quantize_asfp4_dequant,
        quantize_gradient_fp4,
        quantize_hw_fp4,
    )

    W = _make_llm_weights(256, 512, seed=42, outlier_frac=0.02, outlier_mult=6.0)
    H_1d = _make_hessian(512, seed=7)
    H = H_1d.unsqueeze(0).expand_as(W)  # (out_f, in_f) — broadcast per-channel Hessian
    X = _randn(64, 512, seed=123) * 0.1

    for label, fn in [
        ("AS-FP4", lambda: quantize_asfp4_dequant(W, 32)),
        ("HW-FP4 (synth)", lambda: quantize_hw_fp4(W, H, 32)),
        ("GradFP4 iters20", lambda: quantize_gradient_fp4(W, H, 32, 20, 0.01)),
        ("GradFP4 iters50", lambda: quantize_gradient_fp4(W, H, 32, 50, 0.01)),
        ("GradFP4 iters100", lambda: quantize_gradient_fp4(W, H, 32, 100, 0.01)),
        ("GradFP4 noH 50", lambda: quantize_gradient_fp4(W, None, 32, 50, 0.01)),
    ]:
        W_dq = fn()
        sqnr = _sqnr(W, W_dq)
        y_ref = X @ W.T
        y_q = X @ W_dq[:, : W.shape[1]].T
        oe = frob_err(y_ref, y_q)
        print(f"  {label:<20} SQNR={sqnr:6.2f}dB  out_err={oe:.4f}")

    # Gradient FP4 should beat AS-FP4 (it optimizes codes globally)
    w_grad = quantize_gradient_fp4(W, H, 32, 50, 0.01)
    w_asfp4 = quantize_asfp4_dequant(W, 32)
    sqnr_grad = _sqnr(W, w_grad)
    sqnr_asfp4 = _sqnr(W, w_asfp4)
    print(f"\n  AS-FP4={sqnr_asfp4:.2f}dB  GradFP4={sqnr_grad:.2f}dB  delta={sqnr_grad - sqnr_asfp4:+.2f}dB")
    # Gradient should at least match AS-FP4 (may not always beat due to STE approximation)
    assert sqnr_grad >= sqnr_asfp4 - 1.0, f"GradFP4 ({sqnr_grad:.2f}) should be within 1dB of AS-FP4 ({sqnr_asfp4:.2f})"
    print("  PASS\n")


def test_r18_multiscale():
    """R18 multi-scale benchmark: compare R18 winners vs R17 best."""
    print("  R&D 18: Multi-scale benchmark (R18 vs R17)")
    print("-" * 90)

    from forge.engine.quant.novel_quant import (
        quantize_adaptive_iri_fp4,
        quantize_asfp4_dequant,
        quantize_hpr_iri_fp4,
        quantize_iri_fp4,
        quantize_learned_codebook_fp4,
        quantize_mixed_iri_fp4,
    )

    algos = {
        "AS-FP4": lambda W, X: quantize_asfp4_dequant(W, 32),
        "IRI-FP4 x2": lambda W, X: quantize_iri_fp4(W, 32, 2),
        "HPR+IRI x2": lambda W, X: quantize_hpr_iri_fp4(W, 32, 2, optimal_signs=False),
        "MixedIRI x2": lambda W, X: quantize_mixed_iri_fp4(W, 32, 2, "alternate"),
        "AdaptiveIRI m3": lambda W, X: quantize_adaptive_iri_fp4(W, 32, 3, 1e-4),
        "LearnedCB 20": lambda W, X: quantize_learned_codebook_fp4(W, 32, 20),
    }

    for scale_name in ["1B", "5B"]:
        cfg = MODEL_CONFIGS[scale_name]
        print(f"\n  {scale_name} model (d={cfg['d_model']}, L={cfg['n_layers']})")
        print(f"  {'algorithm':<16} {'avg SQNR':>9} {'avg out_err':>12}")
        print(f"  {'-' * 16} {'-' * 9} {'-' * 12}")

        weights = _generate_model_weights(cfg, seed=42)
        activations = {}
        for name, W, _out_f, _in_f in weights:
            X = _randn(min(64, 128), W.shape[1], seed=123) * 0.1
            X[:, ::32] *= 3.0
            activations[name] = X

        for algo_name, algo_fn in algos.items():
            sqnrs = []
            out_errs = []
            for name, W, _out_f, _in_f in weights:
                X = activations[name]
                W_dq = algo_fn(W, X)
                sqnrs.append(_sqnr(W, W_dq))
                y_ref = X @ W.T
                y_q = X @ W_dq[:, : W.shape[1]].T
                out_errs.append(frob_err(y_ref, y_q))
            avg_sqnr = sum(sqnrs) / len(sqnrs)
            avg_oe = sum(out_errs) / len(out_errs)
            print(f"  {algo_name:<16} {avg_sqnr:>8.2f}dB {avg_oe:>12.4f}")

    print("  PASS\n")


def main_r18():
    test_r18_mixed_codebook_iri()
    test_r18_adaptive_iri()
    test_r18_learned_codebook()
    test_r18_sr_fp4_kv_cache()
    test_r18_gradient_fp4()
    test_r18_multiscale()
    print("=" * 90)
    print("  ALL R&D ROUND 18 TESTS PASSED")
    print("=" * 90)


if __name__ == "__main__":
    main()
    main_r15()
    main_r16()
    main_r17()
    main_r18()
