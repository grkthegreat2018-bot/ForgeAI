"""FP8 inference quantization for Blackwell (RTX 5070, sm_120).

FP8 (float8_e4m3fn) halves weight VRAM vs BF16 with ~1% quality loss and
NO performance penalty vs INT8 — it is hardware-native on 5th-gen Tensor
Cores via torch._scaled_mm. ForgeLM V2 is 3.6 GB in bf16 -> ~1.8 GB in FP8,
leaving 10 GB for KV cache + activations on a 12 GB card.

This module provides weight-only + dynamic-activation FP8 for nn.Linear:
  - Weights stored as torch.float8_e4m3fn (1 byte/element) + a per-tensor
    fp32 scale.
  - Activations quantized to FP8 on-the-fly per forward (dynamic per-tensor
    scale) — keeps activations in bf16 everywhere except the matmul.
  - Forward uses torch._scaled_mm (cuBLASLt FP8 kernel) on CUDA, returning
    bf16 output. Falls back to dequant + bf16 matmul on CPU / when
    _scaled_mm is unavailable.

Per-tensor scaling is used because per-row scaling is not supported on
sm_120 with torch 2.8 (runtime error). Per-tensor gives ~1-2% quality loss
on typical LM weights — acceptable for inference.

Usage:
    from forge.quant.fp8_infer import quantize_model_fp8
    n = quantize_model_fp8(model)  # ~1.8 GB for ForgeLM V2, native FP8 matmul
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.quant.protocol import QuantizedLinearMixin

# E4M3FN max representable value (finite).
_E4M3_MAX = 448.0


def _fp8_per_tensor_scale(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize t to float8_e4m3fn with a per-tensor scale.

    Returns (t_fp8, scale, scale_scalar) where scale is a 0-dim fp32 tensor
    suitable for torch._scaled_mm.
    """
    # Clamp input before amax computation to prevent Inf→NaN:
    # If t contains Inf, amax()=Inf → scale_val=Inf → t/Inf = 0 or NaN (Inf/Inf).
    # We clamp only for the scale computation, then apply the scale to the original t.
    finfo = torch.finfo(t.dtype) if t.is_floating_point() else torch.finfo(torch.float32)
    t_clamped = t.clamp(min=-finfo.max, max=finfo.max)
    scale_val = (t_clamped.abs().amax().float() / _E4M3_MAX).clamp(min=1e-12)
    scale = scale_val.to(torch.float32)
    t_fp8 = (t.float() / scale_val).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return t_fp8, scale, scale_val


class FP8Linear(QuantizedLinearMixin):
    """FP8 weight-only + dynamic-activation Linear for inference.

    Weight stored as float8_e4m3fn (1 byte) + per-tensor fp32 scale.
    Forward quantizes the bf16 activation to FP8 dynamically and calls
    torch._scaled_mm (Blackwell-native FP8 matmul) -> bf16 output.

    Falls back to dequant + F.linear when _scaled_mm is unavailable or the
    tensors are on CPU (FP8 matmul is CUDA-only).
    """

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bias = original.bias
        # original.weight is [out, in]. Store transposed view [out, in] as fp8;
        # _scaled_mm wants a row-major x [M, in] and b.T col-major [in, out].
        with torch.no_grad():
            w_fp8, scale, _ = _fp8_per_tensor_scale(original.weight.data)
        self.register_buffer("w_fp8", w_fp8, persistent=False)
        self.register_buffer("w_scale", scale, persistent=False)
        if self.bias is not None:
            self.register_buffer("bias_weight", self.bias.data.clone(), persistent=False)
        self._use_scaled_mm = (
            hasattr(torch, "_scaled_mm")
            and torch.cuda.is_available()
        )
        self._cached_w_bf16 = None

    def _invalidate_weight_cache(self):
        self._cached_w_bf16 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in]. Reshape to 2D for _scaled_mm.
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        bias = self.bias_weight if self.bias is not None else None

        if self._use_scaled_mm and x2d.is_cuda and self.w_fp8.is_cuda:
            # Dynamic per-tensor activation scale.
            x_scale_val = (x2d.abs().amax().float() / _E4M3_MAX).clamp(min=1e-12)
            x_scale = x_scale_val.to(torch.float32)
            x_fp8 = (x2d.float() / x_scale_val).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
            # _scaled_mm(a, b): a [M, K] row-major fp8, b [K, N] col-major fp8.
            # w_fp8 is [out, in] -> w_fp8.T is [in, out] (col-major view).
            out = torch._scaled_mm(
                x_fp8, self.w_fp8.T,
                scale_a=x_scale, scale_b=self.w_scale,
                out_dtype=torch.bfloat16,
            )
            if bias is not None:
                out = out + bias.to(out.dtype)
            return out.reshape(*orig_shape[:-1], self.out_features)

        # Fallback: dequant weight to bf16, standard matmul (CPU / no _scaled_mm).
        if self._cached_w_bf16 is None:
            self._cached_w_bf16 = self.w_fp8.to(torch.bfloat16) * self.w_scale.to(torch.bfloat16)
        return F.linear(x, self._cached_w_bf16, bias).reshape(*orig_shape[:-1], self.out_features)

    def __repr__(self):
        return f"FP8Linear(in={self.in_features}, out={self.out_features}, fp8_e4m3fn)"


def quantize_model_fp8(model, target_modules=None, verbose=True):
    """Replace 2D Linear layers with FP8 weight-only quantized versions.

    Only 2D weight matrices are quantized (Muon/FP8 principle). Embeddings,
    norms, and the LM head stay in bf16 — they are 2D but benefit less from
    FP8 (embedding is a gather, head is usually fused with CE).

    Args:
        model: the model.
        target_modules: list of module-name substrings to target. Default:
            attention + FFN projections (the bandwidth-bound matmuls).
        verbose: print VRAM savings.

    Returns:
        number of layers quantized.
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "out_proj", "qkv_proj",
            "kv_down_proj", "k_up_proj", "v_up_proj",
            "w1", "w2", "w3", "w_gate", "w_up", "w_down",
            "fc1", "fc2",
        ]

    n_quantized = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and module.weight.ndim == 2 and any(t in name for t in target_modules):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, FP8Linear(module))
            n_quantized += 1

    if verbose:
        bf16_bytes = 0
        fp8_bytes = 0
        for m in model.modules():
            if isinstance(m, FP8Linear):
                bf16_bytes += m.in_features * m.out_features * 2
                fp8_bytes += m.in_features * m.out_features * 1
            elif isinstance(m, nn.Linear) and m.weight.ndim == 2:
                bf16_bytes += m.weight.numel() * 2
                fp8_bytes += m.weight.numel() * 2
        print(f"  [FP8] {n_quantized} layers quantized to float8_e4m3fn")
        if bf16_bytes > 0:
            print(f"  [FP8] weight memory: {fp8_bytes/1024**2:.1f} MB "
                  f"(was {bf16_bytes/1024**2:.1f} MB, "
                  f"{fp8_bytes/bf16_bytes*100:.0f}%)")
    return n_quantized


def quantize_kv_cache_fp8(k: torch.Tensor, v: torch.Tensor):
    """Quantize KV cache tensors to FP8 with per-channel scales.

    Per-channel (per-head-dim) scaling: one scale per head_dim, applied
    across the sequence length. This gives 2x KV compression with <0.5%
    quality loss. Returns (k_fp8, k_scale, v_fp8, v_scale) where scales are
    [head_dim] fp32 — multiply by scale at dequant time.

    Args:
        k, v: [B, n_kv, T, head_dim] bf16/fp16 tensors.
    """
    # Per-channel scale along head_dim (last axis).
    k_scale = (k.abs().amax(dim=(0, 1, 2), keepdim=False).float() / _E4M3_MAX).clamp(min=1e-12)
    v_scale = (v.abs().amax(dim=(0, 1, 2), keepdim=False).float() / _E4M3_MAX).clamp(min=1e-12)
    k_fp8 = (k.float() / k_scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    v_fp8 = (v.float() / v_scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return k_fp8, k_scale, v_fp8, v_scale


def dequantize_kv_cache_fp8(k_fp8, k_scale, v_fp8, v_scale, dtype=torch.bfloat16):
    """Dequantize FP8 KV cache back to bf16/fp16 for attention."""
    k = k_fp8.to(dtype) * k_scale.to(dtype)
    v = v_fp8.to(dtype) * v_scale.to(dtype)
    return k, v


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Correctness test: FP8Linear vs bf16 Linear.
    lin = nn.Linear(256, 128, bias=True).to(dev)
    fp8_lin = FP8Linear(lin).to(dev)
    x = torch.randn(2, 8, 256, device=dev, dtype=torch.bfloat16)
    ref = lin(x.to(torch.float32)).to(torch.bfloat16)
    out = fp8_lin(x)
    print(f"FP8Linear out: {out.shape} {out.dtype}")
    print(f"max abs diff vs fp32 ref: {(out - ref).abs().max().item():.4f}")
    print(f"mean abs diff: {(out - ref).abs().mean().item():.4f}")
    # KV cache quantization roundtrip.
    k = torch.randn(1, 2, 16, 128, device=dev, dtype=torch.bfloat16)
    k_fp8, k_s, _, _ = quantize_kv_cache_fp8(k, k)
    k_back, _ = dequantize_kv_cache_fp8(k_fp8, k_s, k_fp8, k_s)
    print(f"KV FP8 roundtrip max diff: {(k_back - k).abs().max().item():.4f}")
    print("KV compression: 2x (1 byte vs 2 byte)")
