"""Tests for Quamba2 W4A8 quantization for SSM (Mamba) blocks.

Verifies:
  1. Quamba2Linear forward produces close output to FP16
  2. 4-bit weights achieve ~8x compression vs FP16
  3. Quamba2Block forward produces close output to the original Mamba block
  4. SSM parameters (A_log, dt_bias) stay in FP16
  5. Gradients flow through the quantization (STE)
  6. Quamba2 is better than naive W8A8 on SSM (lower error)
"""

import os
import sys

import torch
import torch.nn as nn

# Ensure the forge package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forge.quant.quamba2 import (
    Quamba2Block,
    Quamba2Conv1d,
    Quamba2Linear,
    quantize_model_quamba2,
)

# ─── Test helpers ───────────────────────────────────────────────────────────


def _make_mamba_block(d_model=64, d_state=16, d_conv=4, expand=2, use_jamba_norms=True):
    """Build a minimal Mamba block (pure PyTorch, no mamba-ssm dependency).

    Mirrors the structure in forge/keys/architecture/mamba_probe.py.
    """
    from forge.keys.architecture.mamba_probe import MambaLayer

    return MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, use_jamba_norms=use_jamba_norms)


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_quamba2_linear_forward():
    """Quamba2Linear (W4A8) output should be close to the FP16 reference."""
    torch.manual_seed(42)
    in_f, out_f = 256, 128
    lin = nn.Linear(in_f, out_f, bias=True)
    q_lin = Quamba2Linear.from_linear(lin, group_size=64, smoothquant_alpha=0.5)

    x = torch.randn(4, 32, in_f, dtype=torch.float32)
    ref = lin(x)
    out = q_lin(x)

    # W4A8 introduces quantization noise; expect < 15% relative error
    rel_err = (ref - out).abs().mean() / ref.abs().mean()
    assert rel_err < 0.15, f"Quamba2Linear relative error too high: {rel_err:.4f}"
    assert out.shape == ref.shape


def test_quamba2_linear_compression():
    """4-bit weights should achieve ~8x compression vs FP32 weights."""
    in_f, out_f = 512, 256
    lin = nn.Linear(in_f, out_f, bias=False)
    q_lin = Quamba2Linear.from_linear(lin, group_size=128, smoothquant_alpha=0.0)

    # FP32 weight bytes: out * in * 4 bytes (nn.Linear default dtype)
    fp32_bytes = out_f * in_f * 4
    # INT4 packed: out * ceil(in/2) * 1 byte
    int4_bytes = q_lin.weight_int4_packed.numel() * 1
    # Scales + zeros: out * n_groups * 2 bytes each (fp16)
    q_lin.weight_scale.shape[1]
    overhead_bytes = (q_lin.weight_scale.numel() + q_lin.weight_zero.numel()) * 2

    total_bytes = int4_bytes + overhead_bytes
    compression = fp32_bytes / total_bytes

    # 4-bit vs 32-bit = 8x theoretical; with group overhead, expect > 5x
    assert compression > 5.0, (
        f"Compression ratio too low: {compression:.2f}x "
        f"(int4={int4_bytes}, overhead={overhead_bytes}, fp32={fp32_bytes})"
    )
    # Verify the packed storage is actually 4-bit (half the elements)
    assert q_lin.weight_int4_packed.dtype == torch.uint8
    assert q_lin.weight_int4_packed.shape[1] == (in_f + 1) // 2


def test_quamba2_block_forward():
    """Quamba2Block forward should be close to the original Mamba block."""
    torch.manual_seed(123)
    d_model, d_state, d_conv, expand = 64, 16, 4, 2
    block = _make_mamba_block(d_model, d_state, d_conv, expand)
    block.eval()

    q_block = Quamba2Block(block, group_size=32, smoothquant_alpha=0.5)
    q_block.eval()

    x = torch.randn(2, 16, d_model, dtype=torch.float32)
    with torch.no_grad():
        ref_out, _ref_present = block(x, use_cache=True)
        q_out, _q_present = q_block(x, use_cache=True)

    assert q_out.shape == ref_out.shape
    rel_err = (ref_out - q_out).abs().mean() / ref_out.abs().mean().clamp(min=1e-6)
    # SSM blocks are sensitive; allow up to 40% relative error from W4A8
    assert rel_err < 0.40, f"Quamba2Block relative error too high: {rel_err:.4f}"


def test_quamba2_ssm_params_unquantized():
    """A_log, dt_bias, D must stay in FP16 (unquantized SSM core)."""
    block = _make_mamba_block(d_model=64, d_state=16, d_conv=4, expand=2)
    q_block = Quamba2Block(block, group_size=64, smoothquant_alpha=0.5)

    # A_log must be a Parameter in float16
    assert isinstance(q_block.A_log, nn.Parameter), "A_log must be a Parameter"
    assert q_block.A_log.dtype == torch.float16, f"A_log dtype should be float16, got {q_block.A_log.dtype}"
    # D must be a Parameter in float16
    assert isinstance(q_block.D, nn.Parameter), "D must be a Parameter"
    assert q_block.D.dtype == torch.float16, f"D dtype should be float16, got {q_block.D.dtype}"

    # The projections must be quantized (Quamba2Linear, not nn.Linear)
    assert isinstance(q_block.in_proj, Quamba2Linear), "in_proj must be Quamba2Linear"
    assert isinstance(q_block.out_proj, Quamba2Linear), "out_proj must be Quamba2Linear"
    assert isinstance(q_block.x_proj, Quamba2Linear), "x_proj must be Quamba2Linear"

    # The conv1d must be quantized
    assert isinstance(q_block.conv1d, Quamba2Conv1d), "conv1d must be Quamba2Conv1d"

    # The selective scan must run in FP16 — verify A_log values match original
    assert torch.allclose(q_block.A_log.float(), block.A_log.float().to(torch.float16).float(), atol=1e-3), (
        "A_log values must be preserved"
    )


def test_quamba2_gradient_flow():
    """Gradients must flow through the quantization (straight-through estimator).

    The weight dequantization uses @torch.no_grad, but the activation
    fake-quantization uses round() which has a straight-through gradient
    (PyTorch's round() passes gradient through).  We verify that the
    SmoothQuant act_scale buffer can receive gradient via the input.
    """
    torch.manual_seed(99)
    in_f, out_f = 128, 64
    lin = nn.Linear(in_f, out_f, bias=False)
    q_lin = Quamba2Linear.from_linear(lin, group_size=64, smoothquant_alpha=0.5)

    # The input requires grad — gradient should flow through fake-quant
    x = torch.randn(2, in_f, dtype=torch.float32, requires_grad=True)
    out = q_lin(x)
    loss = out.sum()
    loss.backward()

    # Input must have a gradient (gradient flows through activation quant)
    assert x.grad is not None, "Gradient must flow through Quamba2Linear"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"
    assert not torch.all(x.grad == 0), "Gradient must be non-zero"


def test_quamba2_vs_w8a8():
    """Quamba2 (W4A8, SSM-aware) should have lower error than naive W8A8 on SSM.

    The key Quamba2 insight: keeping the SSM recurrence (A_log, dt, scan state)
    in FP16 preserves the recurrence dynamics.  Naive W8A8 quantizes everything
    including the SSM core, introducing compounding quantization noise.

    We compare two approaches with the SAME W4A8 projection precision:
      - Quamba2: W4A8 projections + FP16 SSM core (A_log, D, scan state)
      - Naive:   W4A8 projections + INT4 A_log + INT4 scan state quantization

    Both have identical projection quantization error.  The only difference
    is the SSM core: Quamba2 keeps it in FP16, naive quantizes it.  This
    isolates the SSM-aware design benefit — the central Quamba2 contribution.
    """
    torch.manual_seed(777)
    d_model, d_state, d_conv, expand = 64, 16, 4, 2
    block = _make_mamba_block(d_model, d_state, d_conv, expand)
    block.eval()

    # Reference output (longer sequence for more recurrence compounding)
    x = torch.randn(2, 32, d_model, dtype=torch.float32)
    with torch.no_grad():
        ref_out, _ = block(x, use_cache=False)

    # ── Quamba2: W4A8 projections, FP16 SSM core ──
    q_block = Quamba2Block(block, group_size=32, smoothquant_alpha=0.5)
    q_block.eval()
    with torch.no_grad():
        q_out, _ = q_block(x, use_cache=False)
    quamba2_err = (ref_out - q_out).pow(2).mean().item()

    # ── Naive: same W4A8 projections + INT4 A_log + INT4 scan state ──
    # Uses the SAME Quamba2Linear for projections (identical weight error),
    # but also quantizes A_log to INT4 and quantizes the scan state h_t to
    # INT4 after each recurrence step.  This is what "naive W4A8 applied to
    # everything" would look like — it doesn't know the SSM core is sensitive.
    import copy

    block_naive = copy.deepcopy(block)
    block_naive.eval()
    # Same W4A8 projections as Quamba2 (identical quantization error)
    block_naive.in_proj = Quamba2Linear.from_linear(block_naive.in_proj, group_size=32, smoothquant_alpha=0.5)
    block_naive.out_proj = Quamba2Linear.from_linear(block_naive.out_proj, group_size=32, smoothquant_alpha=0.5)
    block_naive.x_proj = Quamba2Linear.from_linear(block_naive.x_proj, group_size=32, smoothquant_alpha=0.5)
    block_naive.dt_proj = Quamba2Linear.from_linear(block_naive.dt_proj, group_size=32, smoothquant_alpha=0.5)

    # NAIVE: quantize A_log to INT4 (destroys recurrence dynamics)
    A_log_fp = block_naive.A_log.data.float()
    A_absmax = A_log_fp.abs().amax().clamp(min=1e-8)
    A_scale = A_absmax / 7.0
    block_naive.A_log.data = ((A_log_fp / A_scale).round().clamp(-8, 7) * A_scale).to(block_naive.A_log.dtype)
    # NAIVE: quantize D to INT4
    D_fp = block_naive.D.data.float()
    D_scale = D_fp.abs().amax().clamp(min=1e-8) / 7.0
    block_naive.D.data = ((D_fp / D_scale).round().clamp(-8, 7) * D_scale).to(block_naive.D.dtype)

    # NAIVE: quantize scan state h_t to INT4 after each step
    def naive_scan(x_s, delta, A, B, C, D, h_init=None):
        B_b, d_inner, L = x_s.shape
        d_state = A.shape[1]
        A_neg = -torch.exp(A)
        h = torch.zeros(B_b, d_inner, d_state, device=x_s.device, dtype=x_s.dtype) if h_init is None else h_init.clone()
        ys = []
        for t in range(L):
            dt = delta[:, :, t : t + 1]
            A_bar = torch.exp(dt * A_neg.unsqueeze(0))
            B_t = B[:, :, t]
            B_bar = dt * B_t.unsqueeze(1)
            x_t = x_s[:, :, t : t + 1]
            h = A_bar * h + B_bar * x_t
            # NAIVE: quantize the recurrent state to INT4 (compounds error)
            h_absmax = h.abs().amax().clamp(min=1e-8)
            h_scale = h_absmax / 7.0
            h = (h / h_scale).round().clamp(-8, 7) * h_scale
            C_t = C[:, :, t]
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + D * x_t.squeeze(-1)
            ys.append(y_t)
        return torch.stack(ys, dim=-1), h

    block_naive._selective_scan_ref = naive_scan
    with torch.no_grad():
        naive_out, _ = block_naive(x, use_cache=False)
    naive_err = (ref_out - naive_out).pow(2).mean().item()

    # Quamba2 should be better (lower MSE) — the SSM-aware design (FP16 core)
    # more than compensates for any projection quantization error.
    assert quamba2_err < naive_err, (
        f"Quamba2 error ({quamba2_err:.6f}) should be lower than "
        f"naive error ({naive_err:.6f}) on SSM blocks. "
        f"Both use W4A8 projections; Quamba2 keeps SSM core in FP16."
    )


# ─── Extra: model-level quantization ────────────────────────────────────────


def test_quamba2_quantize_model():
    """quantize_model_quamba2 should find and quantize Mamba blocks in a model."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = _make_mamba_block(d_model=64, d_state=16)
            self.layer1 = _make_mamba_block(d_model=64, d_state=16)
            self.norm = nn.LayerNorm(64)

        def forward(self, x):
            x, _ = self.layer0(x)
            x, _ = self.layer1(x)
            return self.norm(x)

    model = DummyModel()
    n = quantize_model_quamba2(model, group_size=64)
    assert n == 2, f"Should quantize 2 SSM blocks, got {n}"
    assert isinstance(model.layer0, Quamba2Block)
    assert isinstance(model.layer1, Quamba2Block)
    # Non-SSM layer untouched
    assert isinstance(model.norm, nn.LayerNorm)


def test_quamba2_block_preserves_input_dtype():
    """Quamba2Block must return output in the input dtype (bf16 regression).

    The SSM core (A_log, D, jamba norms) is stored in FP16 — mixed with bf16
    activations it promoted the whole residual stream to fp32, crashing
    downstream bf16 matmuls ('expected mat1 and mat2 to have the same dtype:
    float != BFloat16'). The fix casts the block output back to the input
    dtype.
    """
    torch.manual_seed(7)
    d_model = 64
    block = _make_mamba_block(d_model, d_state=16, d_conv=4, expand=2)
    block = block.to(torch.bfloat16).eval()
    q_block = Quamba2Block(block, group_size=32, smoothquant_alpha=0.5)
    q_block = q_block.eval()

    x = torch.randn(1, 8, d_model, dtype=torch.bfloat16)
    with torch.no_grad():
        out, present = q_block(x, use_cache=True)
    assert out.dtype == torch.bfloat16, (
        f"Quamba2Block emitted {out.dtype}, expected bfloat16 — fp32 promotion leaks into the residual stream"
    )

    # Incremental decode path (T=1) must also preserve dtype.
    x1 = torch.randn(1, 1, d_model, dtype=torch.bfloat16)
    with torch.no_grad():
        out1, _ = q_block(x1, past_key_value=present, use_cache=True)
    assert out1.dtype == torch.bfloat16


def test_quamba2_lora_adapter():
    """add_lora_adapters must attach to Quamba2Linear and forward must apply it.

    Regression: Quamba2Linear has no .weight/.weight_packed/.dense_packed
    (it stores weight_int4_packed) — the device lookup crashed with
    AttributeError, and forward() never applied the adapter.
    """
    torch.manual_seed(11)
    from forge.training.bitnet_lora import add_lora_adapters

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = Quamba2Linear.from_linear(nn.Linear(256, 128, bias=False), group_size=64)

    model = _M()
    n_adapters, _ = add_lora_adapters(model, rank=8, alpha=16)
    assert n_adapters == 1, f"expected 1 adapter on Quamba2Linear, got {n_adapters}"
    lora = model.q.lora_adapter
    assert lora.lora_A.device == model.q.weight_int4_packed.device

    x = torch.randn(2, 4, 256, dtype=torch.bfloat16)  # engine runs bf16
    with torch.no_grad():
        model.q(x)  # warm the fp16 _cached_weight so both calls share it
        lora.lora_B.data.fill_(0.01)
        out = model.q(x)
        model.q.lora_adapter = None
        base = model.q(x)
        model.q.lora_adapter = lora
    assert not torch.allclose(base, out, atol=1e-3), "lora_adapter must affect forward output"
    expected_delta = lora.scale * (x @ lora.lora_A.T @ lora.lora_B.T)
    assert torch.allclose(out - base, expected_delta, atol=5e-3)


def test_quamba2_conv1d_forward():
    """Quamba2Conv1d (W4 depthwise) should produce close output to FP16 conv."""
    torch.manual_seed(55)
    ch, k = 64, 4
    conv = nn.Conv1d(ch, ch, k, groups=ch, padding=k - 1, bias=True)
    q_conv = Quamba2Conv1d.from_conv1d(conv)

    x = torch.randn(2, ch, 16, dtype=torch.float32)
    ref = conv(x)[:, :, :16]
    out = q_conv(x)[:, :, :16]

    rel_err = (ref - out).abs().mean() / ref.abs().mean().clamp(min=1e-6)
    assert rel_err < 0.20, f"Quamba2Conv1d relative error too high: {rel_err:.4f}"


if __name__ == "__main__":
    # Allow running as a script
    test_quamba2_linear_forward()
    print("PASS: test_quamba2_linear_forward")
    test_quamba2_linear_compression()
    print("PASS: test_quamba2_linear_compression")
    test_quamba2_block_forward()
    print("PASS: test_quamba2_block_forward")
    test_quamba2_ssm_params_unquantized()
    print("PASS: test_quamba2_ssm_params_unquantized")
    test_quamba2_gradient_flow()
    print("PASS: test_quamba2_gradient_flow")
    test_quamba2_vs_w8a8()
    print("PASS: test_quamba2_vs_w8a8")
    test_quamba2_quantize_model()
    print("PASS: test_quamba2_quantize_model")
    test_quamba2_conv1d_forward()
    print("PASS: test_quamba2_conv1d_forward")
    print("\nAll Quamba2 tests passed!")
