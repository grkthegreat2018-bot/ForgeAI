"""Test NVFP4 quantization correctness and speed."""
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn

from forge.engine.quant.nvfp4_quant import (
    NVFP4Linear,
    _dequantize_fp4,
    _quantize_to_fp4,
    quantize_model_nvfp4,
)


def test_fp4_roundtrip():
    """Test FP4 quantize → dequantize roundtrip accuracy."""
    torch.manual_seed(42)
    W = torch.randn(128, 256) * 0.1  # small weights like in LLMs
    packed, scales, global_scale = _quantize_to_fp4(W, block_size=32)
    W_dq = _dequantize_fp4(packed, scales, 128, 256, block_size=32, dtype=torch.float32,
                           global_scale=global_scale)

    # Check shape
    assert W_dq.shape == W.shape, f"Shape mismatch: {W_dq.shape} vs {W.shape}"

    # Check relative error (FP4 has 8 levels, expect ~5-15% relative error)
    rel_err = ((W - W_dq).abs() / W.abs().clamp(min=1e-6)).mean().item()
    max_err = (W - W_dq).abs().max().item()
    print(f"Test 1 (FP4 roundtrip): mean_rel_err={rel_err:.4f}, max_err={max_err:.6f}")
    assert rel_err < 0.20, f"Relative error too high: {rel_err}"
    print("  PASS")


def test_nvfp4_linear():
    """Test NVFP4Linear forward pass matches nn.Linear closely."""
    torch.manual_seed(42)
    lin = nn.Linear(256, 128, bias=True)
    nvfp4 = NVFP4Linear.from_linear(lin, block_size=32)

    x = torch.randn(4, 256, dtype=torch.float32)
    y_ref = lin(x)
    y_q = nvfp4(x)

    # Check output shape
    assert y_q.shape == y_ref.shape, f"Output shape mismatch: {y_q.shape} vs {y_ref.shape}"

    # Check output error using Frobenius norm (standard quantization metric)
    # ||y_ref - y_q|| / ||y_ref||  — avoids division by near-zero elements
    frob_err = (y_ref - y_q).norm().item() / y_ref.norm().item()
    print(f"Test 2 (NVFP4Linear forward): frob_err={frob_err:.4f}")
    assert frob_err < 0.15, f"Output error too high: {frob_err}"
    print("  PASS")


def test_nvfp4_linear_cuda():
    """Test NVFP4Linear on CUDA (bf16)."""
    if not torch.cuda.is_available():
        print("Test 3 (CUDA): SKIPPED (no CUDA)")
        return

    torch.manual_seed(42)
    lin = nn.Linear(512, 1024, bias=True).cuda().to(torch.bfloat16)
    nvfp4 = NVFP4Linear.from_linear(lin, block_size=32).cuda()

    x = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        y_ref = lin(x)
        y_q = nvfp4(x)

    rel_err = (y_ref.float() - y_q.float()).norm().item() / y_ref.float().norm().item()
    print(f"Test 3 (CUDA bf16): frob_err={rel_err:.4f}")
    assert rel_err < 0.15, f"CUDA output error too high: {rel_err}"
    print("  PASS")


def test_memory_compression():
    """Test that NVFP4 actually compresses weights."""
    lin = nn.Linear(2048, 8192, bias=False)
    nvfp4 = NVFP4Linear.from_linear(lin, block_size=32)

    orig_bytes = lin.weight.numel() * 2  # bf16
    packed_bytes = nvfp4.weight_packed.numel()  # uint8, 2 FP4 per byte
    scale_bytes = nvfp4.weight_scales.numel()  # FP8, 1 byte each
    total_quant = packed_bytes + scale_bytes
    compression = orig_bytes / total_quant

    print(f"Test 4 (compression): orig={orig_bytes/1024:.0f}KB, "
          f"quant={total_quant/1024:.0f}KB, {compression:.1f}x")
    assert compression > 3.0, f"Compression too low: {compression:.1f}x"
    print("  PASS")


def test_quantize_model():
    """Test quantize_model_nvfp4 on a small model."""
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    n = quantize_model_nvfp4(model, verbose=True)
    assert n == 2, f"Expected 2 layers quantized, got {n}"

    # Check forward still works
    x = torch.randn(2, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 128), f"Output shape wrong: {y.shape}"
    print(f"Test 5 (quantize_model): {n} layers, output shape OK")
    print("  PASS")


def test_speed_cuda():
    """Benchmark NVFP4 vs bf16 on CUDA."""
    if not torch.cuda.is_available():
        print("Test 6 (speed): SKIPPED (no CUDA)")
        return

    torch.manual_seed(42)
    lin = nn.Linear(2048, 8192, bias=False).cuda().to(torch.bfloat16)
    nvfp4 = NVFP4Linear.from_linear(lin, block_size=32).cuda()

    x = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = lin(x)
            _ = nvfp4(x)
    torch.cuda.synchronize()

    # Benchmark bf16
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = lin(x)
    torch.cuda.synchronize()
    bf16_ms = (time.perf_counter() - t0) * 10  # ms per call

    # Benchmark NVFP4
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = nvfp4(x)
    torch.cuda.synchronize()
    fp4_ms = (time.perf_counter() - t0) * 10

    print(f"Test 6 (speed): bf16={bf16_ms:.2f}ms, NVFP4={fp4_ms:.2f}ms, "
          f"ratio={fp4_ms/bf16_ms:.2f}x")
    # NVFP4 dequant path may be slower for small matrices (overhead)
    # but should be competitive or faster for large ones
    print("  PASS (informational)")


def main():
    print("=" * 60)
    print("  NVFP4 Quantization Tests")
    print("=" * 60)
    test_fp4_roundtrip()
    test_nvfp4_linear()
    test_nvfp4_linear_cuda()
    test_memory_compression()
    test_quantize_model()
    test_speed_cuda()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
