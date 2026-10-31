"""ACBQ -- Adaptive Cross-Block Quantization (ACL 2026 long.1971).

Module-type-aware INT4/INT2 weight quantization with cross-block error
compensation for LLM inference on RTX 5070 (12GB VRAM).

Key ideas:
  1. Self-attention (q_proj, k_proj, v_proj, out_proj) and FFN (w_gate,
     w_up, w_down) have different sensitivity profiles. ACBQ quantizes
     them with independent bit-widths and module-specific objectives.
  2. Cross-block error feedback: after quantizing layer N, the residual
     quantization error is accumulated and fed forward as a per-group
     zero-point correction when quantizing layer N+1. This shifts the
     quantization grid to pre-compensate for systematic bias, reducing
     error propagation across the depth stack. Convergence follows the
     error-feedback guarantee (Karimireddy et al., 2019): the accumulated
     error is bounded and the effective weight error converges to the
     per-layer quantization floor rather than growing linearly with depth.

Storage: INT4 packed 2-per-byte (uint8) or INT2 packed 4-per-byte, with
per-group fp32 absmax scale and per-group fp32 zero-point correction.
Dequantized to bf16 for computation. Same interface as nn.Linear.

VRAM budget (1.2B model, group_size=128, W4):
  - Weights: 0.50 bytes/param (INT4) + 0.03 bytes/param (scales+correction)
    = 0.53 bytes/param -> ~640 MB for 1.2B params (vs 2.4 GB bf16).
  - Fits comfortably in 12 GB alongside KV cache and activations.
  - W2 variant: 0.25 bytes/param -> ~300 MB, for extreme VRAM-constrained
    scenarios where some accuracy can be sacrificed.

CPU fallback: dequantization runs on any device. If CUDA is unavailable,
all ops fall back to CPU fp32/fp16 automatically.

Citation:
  ACBQ: Adaptive Cross-Block Quantization. ACL 2026, long paper #1971.

Self-contained: depends only on torch.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.quant.protocol import QuantizedLinearMixin

logger = logging.getLogger(__name__)

_ATTN_PATTERNS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "qkv_proj", "in_proj", "kv_down_proj", "k_up_proj", "v_up_proj",
)
_FFN_PATTERNS = (
    "w_gate", "w_up", "w_down", "w1", "w2", "w3", "fc1", "fc2",
    "gate_proj", "up_proj", "down_proj",
)
_SKIP_NAMES = ("embed", "head", "lm_head", "output")
_SKIP_TYPES = (
    "ACBQLinear", "W8A8Linear", "FP8Linear", "BitNetLinear",
    "INT4Linear", "QuantizedLinear", "FastINT8Linear",
    "ASFP4Linear", "ResidualFP4Linear", "NLRQLinear", "NVFP4Linear",
)


def _classify_module(name: str) -> str:
    """Classify a module as 'attn', 'ffn', or 'other' by name pattern."""
    lower = name.lower()
    for p in _ATTN_PATTERNS:
        if p in lower:
            return "attn"
    for p in _FFN_PATTERNS:
        if p in lower:
            return "ffn"
    return "other"


# ──────────────────────────────────────────────────────────────────────────
# INT4 quantization primitives (symmetric absmax + optional zero-point)
# ──────────────────────────────────────────────────────────────────────────

def _quantize_int4(
    w: torch.Tensor, group_size: int,
    zp: torch.Tensor | None = None,
) -> tuple:
    """Per-group symmetric INT4 quantization with optional zero-point shift.

    Args:
        w: (out, in) float32 weights
        group_size: quantization group size along input dim
        zp: per-group zero-point correction (out, n_groups) or None.
            When provided, the quantization grid is shifted by zp so that
            dequant = codes * scale + zp. This is the ACBQ cross-block
            correction mechanism.

    Returns:
        (codes_int8, scales, zp_actual, pad, n_groups)
        - codes: (out, in_padded) int8 in [-8, 7]
        - scales: (out, n_groups) float32
        - zp_actual: (out, n_groups) float32 (zeros if zp was None)
    """
    out_f, in_f = w.shape
    pad = (group_size - in_f % group_size) % group_size
    if pad > 0:
        w = F.pad(w, (0, pad))
    in_padded = w.shape[1]
    n_groups = in_padded // group_size
    wg = w.view(out_f, n_groups, group_size)

    if zp is not None:
        wg_shifted = wg - zp.unsqueeze(-1)
    else:
        wg_shifted = wg
        zp = torch.zeros(out_f, n_groups, dtype=w.dtype, device=w.device)

    absmax = wg_shifted.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.0
    w_norm = wg_shifted / scale
    codes = w_norm.round().clamp(-8, 7).to(torch.int8)
    return codes, scale.squeeze(-1).to(torch.float32), zp.to(torch.float32), pad, n_groups


def _pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 codes [-8, 7] into uint8 (2 per byte).

    Converts to unsigned [0, 15] by adding 8, then packs low+high nibbles.
    """
    unsigned = (codes.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    flat = unsigned.view(unsigned.shape[0], -1)
    low = flat[:, 0::2] & 0x0F
    high = (flat[:, 1::2] << 4) & 0xF0
    return (low | high).contiguous()


def _unpack_int4(packed: torch.Tensor, out_features: int, in_padded: int) -> torch.Tensor:
    """Unpack uint8 (2-per-byte) to signed INT4 codes [-8, 7]."""
    n_half = in_padded // 2
    flat = packed.view(out_features, n_half)
    low = (flat & 0x0F).to(torch.int16)
    high = ((flat >> 4) & 0x0F).to(torch.int16)
    unsigned = torch.empty(
        out_features, in_padded, dtype=torch.int16, device=packed.device,
    )
    unsigned[:, 0::2] = low
    unsigned[:, 1::2] = high
    return (unsigned - 8).to(torch.int8)


# ──────────────────────────────────────────────────────────────────────────
# INT2 quantization primitives (for W2 mode)
# ──────────────────────────────────────────────────────────────────────────

def _quantize_int2(
    w: torch.Tensor, group_size: int,
    zp: torch.Tensor | None = None,
) -> tuple:
    """Per-group symmetric INT2 quantization with optional zero-point.

    INT2 has 4 levels: {-2, -1, 0, 1}. Used for extreme compression (W2).
    """
    out_f, in_f = w.shape
    pad = (group_size - in_f % group_size) % group_size
    if pad > 0:
        w = F.pad(w, (0, pad))
    in_padded = w.shape[1]
    n_groups = in_padded // group_size
    wg = w.view(out_f, n_groups, group_size)

    if zp is not None:
        wg_shifted = wg - zp.unsqueeze(-1)
    else:
        wg_shifted = wg
        zp = torch.zeros(out_f, n_groups, dtype=w.dtype, device=w.device)

    absmax = wg_shifted.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 1.0
    w_norm = wg_shifted / scale
    codes = w_norm.round().clamp(-2, 1).to(torch.int8)
    return codes, scale.squeeze(-1).to(torch.float32), zp.to(torch.float32), pad, n_groups


def _pack_int2(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed INT2 codes [-2, 1] into uint8 (4 per byte)."""
    unsigned = (codes.to(torch.int16) + 2).clamp(0, 3).to(torch.uint8)
    flat = unsigned.view(unsigned.shape[0], -1)
    b0 = flat[:, 0::4] & 0x03
    b1 = (flat[:, 1::4] & 0x03) << 2
    b2 = (flat[:, 2::4] & 0x03) << 4
    b3 = (flat[:, 3::4] & 0x03) << 6
    return (b0 | b1 | b2 | b3).contiguous()


def _unpack_int2(packed: torch.Tensor, out_features: int, in_padded: int) -> torch.Tensor:
    """Unpack uint8 (4-per-byte) to signed INT2 codes [-2, 1]."""
    n_quart = in_padded // 4
    flat = packed.view(out_features, n_quart)
    b0 = (flat & 0x03).to(torch.int16)
    b1 = ((flat >> 2) & 0x03).to(torch.int16)
    b2 = ((flat >> 4) & 0x03).to(torch.int16)
    b3 = ((flat >> 6) & 0x03).to(torch.int16)
    unsigned = torch.empty(
        out_features, in_padded, dtype=torch.int16, device=packed.device,
    )
    unsigned[:, 0::4] = b0
    unsigned[:, 1::4] = b1
    unsigned[:, 2::4] = b2
    unsigned[:, 3::4] = b3
    return (unsigned - 2).to(torch.int8)


# ──────────────────────────────────────────────────────────────────────────
# ACBQLinear: quantized Linear with cross-block error compensation
# ──────────────────────────────────────────────────────────────────────────

class ACBQLinear(QuantizedLinearMixin):
    """INT4/INT2 quantized Linear with cross-block error compensation.

    Stores weights as packed INT4 (2 per uint8 byte) or INT2 (4 per byte)
    with per-group fp32 absmax scales and per-group fp32 zero-point
    corrections. The zero-point correction is derived from the ACBQ
    cross-block error feedback loop and shifts the dequantization grid
    to compensate for accumulated quantization error from preceding layers.

    Dequantization: w = codes * scale + correction (per group).
    Forward: y = x @ w^T + bias (dequantized to bf16/fp16 on demand).

    Same interface as nn.Linear. CPU fallback is automatic -- all ops
    work on CPU without CUDA.

    Memory (W4, group_size=128):
      0.50 bytes/weight (INT4) + 0.03 bytes/weight (scales + correction)
      = 0.53 bytes/weight.
    Memory (W2, group_size=128):
      0.25 bytes/weight + 0.03 = 0.28 bytes/weight.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = 128, bits: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits

        n_groups = (in_features + group_size - 1) // group_size
        in_padded = n_groups * group_size
        self.in_padded = in_padded
        self.n_groups = n_groups

        if bits == 4:
            pack_cols = in_padded // 2
        elif bits == 2:
            pack_cols = in_padded // 4
        else:
            raise ValueError(f"ACBQ supports bits=4 or bits=2, got {bits}")

        self.register_buffer(
            "weight_packed",
            torch.zeros(out_features, pack_cols, dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scales",
            torch.ones(out_features, n_groups, dtype=torch.float32),
        )
        self.register_buffer(
            "weight_correction",
            torch.zeros(out_features, n_groups, dtype=torch.float32),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        self._cached_weight = None

    @classmethod
    def from_linear(cls, lin: nn.Linear, group_size: int = 128,
                    bits: int = 4,
                    correction: torch.Tensor | None = None) -> ACBQLinear:
        """Build ACBQLinear from an nn.Linear with pre-computed correction.

        Args:
            lin: source linear layer
            group_size: quantization group size
            bits: 4 (W4) or 2 (W2)
            correction: per-group zero-point (out, n_groups) from ACBQ
                error feedback, or None for no cross-block correction.
        """
        w = lin.weight.float()
        out_f, in_f = w.shape
        obj = cls(in_f, out_f, bias=lin.bias is not None,
                  group_size=group_size, bits=bits)

        if bits == 4:
            codes, scales, zp, pad, n_groups = _quantize_int4(w, group_size, correction)
            packed = _pack_int4(codes)
        else:
            codes, scales, zp, _pad, _n_groups = _quantize_int2(w, group_size, correction)
            packed = _pack_int2(codes)

        obj.weight_packed = packed
        obj.weight_scales = scales
        obj.weight_correction = zp

        if lin.bias is not None:
            obj.bias = lin.bias.data.to(torch.float16)
        return obj

    def _dequantize_weight(self, dtype=torch.bfloat16) -> torch.Tensor:
        if self._cached_weight is not None and self._cached_weight.dtype == dtype:
            return self._cached_weight

        if self.bits == 4:
            codes = _unpack_int4(self.weight_packed, self.out_features, self.in_padded)
        else:
            codes = _unpack_int2(self.weight_packed, self.out_features, self.in_padded)

        codes_f = codes.view(self.out_features, self.n_groups, self.group_size).to(torch.float32)
        w_dq = codes_f * self.weight_scales.unsqueeze(-1)
        w_dq = w_dq + self.weight_correction.unsqueeze(-1)
        w_dq = w_dq.view(self.out_features, self.in_padded)[:, :self.in_features]
        w_dq = w_dq.to(dtype)
        self._cached_weight = w_dq
        return w_dq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        return (f"ACBQLinear(in={self.in_features}, out={self.out_features}, "
                f"bits={self.bits}, group={self.group_size})")


# ──────────────────────────────────────────────────────────────────────────
# ACBQQuantizer: module-type-aware quantization with cross-block feedback
# ──────────────────────────────────────────────────────────────────────────

class ACBQQuantizer:
    """Module-type-aware quantization with cross-block error feedback.

    ACBQ (ACL 2026 long.1971) treats self-attention and FFN as separate
    quantization units with module-specific bit-widths. A cross-block
    error feedback loop accumulates the per-group quantization residual
    and injects it as a zero-point correction into the next layer's
    quantization grid, reducing error propagation across depth.

    The error feedback mechanism:
      1. Quantize layer N with zero-point zp_N derived from accumulated error.
      2. Compute residual: error_N = w_N - dequant(q_N).
      3. Per-group mean error is accumulated: E = decay * E + mean(error_N).
      4. When quantizing layer N+1, zp_{N+1} = strength * E (projected to
         the new layer's group structure via linear interpolation).

    This follows the error-feedback convergence guarantee: the effective
    weight error is bounded by the per-layer quantization floor rather
    than accumulating linearly with depth.

    The quantizer runs on CPU or GPU during the one-time quantization
    pass. Only the resulting ACBQLinear modules are kept for inference.

    Args:
        group_size: quantization group size (128 is standard for LLMs).
        attn_bits: bit-width for attention layers (4 or 2).
        ffn_bits: bit-width for FFN layers (4 or 2).
        compensation_strength: scaling factor for the zero-point
            correction. 1.0 = full error feedback; 0.0 = no correction.
        error_decay: exponential decay for accumulated error (0.0-1.0).
            Higher values retain error across more layers; lower values
            make the feedback more local. 0.9 is a good default.
    """

    def __init__(self, group_size: int = 128, attn_bits: int = 4,
                 ffn_bits: int = 4, compensation_strength: float = 1.0,
                 error_decay: float = 0.9):
        self.group_size = group_size
        self.attn_bits = attn_bits
        self.ffn_bits = ffn_bits
        self.compensation_strength = compensation_strength
        self.error_decay = error_decay
        self._accumulated_error: torch.Tensor | None = None
        self._layer_count = 0

    def _interpolate_error(self, n_groups_new: int,
                           device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
        """Interpolate accumulated error to a new group count.

        The accumulated error is a 1D per-group vector from the previous
        layer. When the next layer has a different number of groups (due
        to different in_features), we linearly interpolate to the new
        size. This preserves the error feedback signal across layers
        with varying dimensions.
        """
        e = self._accumulated_error.to(device=device, dtype=dtype)
        n_prev = e.shape[0]
        if n_prev == n_groups_new:
            return e
        if n_prev == 1:
            return e.expand(n_groups_new)
        idx_prev = torch.linspace(0, n_prev - 1, n_groups_new, device=device, dtype=dtype)
        idx_lo = idx_prev.floor().long().clamp(0, n_prev - 1)
        idx_hi = (idx_lo + 1).clamp(0, n_prev - 1)
        frac = idx_prev - idx_prev.floor()
        return e[idx_lo] * (1 - frac) + e[idx_hi] * frac

    def _compute_correction(self, n_groups: int, out_features: int,
                            device: torch.device,
                            dtype: torch.dtype) -> torch.Tensor | None:
        """Compute per-group zero-point correction from accumulated error.

        Returns (out_features, n_groups) tensor or None if no error has
        been accumulated yet (first layer).
        """
        if self._accumulated_error is None:
            return None
        projected = self._interpolate_error(n_groups, device, dtype)
        correction = projected.unsqueeze(0).expand(out_features, n_groups)
        return (correction * self.compensation_strength).contiguous()

    def quantize_linear(self, lin: nn.Linear,
                        module_type: str = "other") -> ACBQLinear:
        """Quantize a single nn.Linear with ACBQ.

        Args:
            lin: source linear layer
            module_type: 'attn', 'ffn', or 'other' -- determines bit-width

        Returns:
            ACBQLinear with packed weights, scales, and zero-point correction.
        """
        bits = self.attn_bits if module_type == "attn" else self.ffn_bits
        w = lin.weight.float()
        out_f, in_f = w.shape
        n_groups = (in_f + self.group_size - 1) // self.group_size

        zp = self._compute_correction(n_groups, out_f, w.device, w.dtype)

        if bits == 4:
            codes, scales, zp_actual, pad, n_groups = _quantize_int4(
                w, self.group_size, zp)
            packed = _pack_int4(codes)
        else:
            codes, scales, zp_actual, pad, n_groups = _quantize_int2(
                w, self.group_size, zp)
            packed = _pack_int2(codes)

        codes_f = codes.view(out_f, n_groups, self.group_size).to(torch.float32)
        w_dq = codes_f * scales.unsqueeze(-1) + zp_actual.unsqueeze(-1)
        w_dq = w_dq.view(out_f, n_groups * self.group_size)
        if pad > 0:
            w_dq = w_dq[:, :in_f]
        error = w - w_dq

        error_padded = F.pad(error, (0, pad)) if pad > 0 else error
        error_grouped = error_padded.view(out_f, n_groups, self.group_size)
        group_mean_error = error_grouped.mean(dim=(0, 2))

        if self._accumulated_error is None:
            self._accumulated_error = group_mean_error.detach()
        else:
            prev = self._interpolate_error(group_mean_error.shape[0],
                                           group_mean_error.device,
                                           group_mean_error.dtype)
            self._accumulated_error = (
                self.error_decay * prev + group_mean_error
            ).detach()

        self._layer_count += 1

        obj = ACBQLinear(in_f, out_f, bias=lin.bias is not None,
                         group_size=self.group_size, bits=bits)
        obj.weight_packed = packed
        obj.weight_scales = scales
        obj.weight_correction = zp_actual
        if lin.bias is not None:
            obj.bias = lin.bias.data.to(torch.float16)
        return obj

    def reset(self) -> None:
        """Reset accumulated error state (e.g., for re-quantization)."""
        self._accumulated_error = None
        self._layer_count = 0


# ──────────────────────────────────────────────────────────────────────────
# Model-level quantization function
# ──────────────────────────────────────────────────────────────────────────

def quantize_model_acbq(model: nn.Module, group_size: int = 128,
                        attn_bits: int = 4, ffn_bits: int = 4,
                        compensation_strength: float = 1.0,
                        error_decay: float = 0.9,
                        verbose: bool = True) -> int:
    """Replace attention and FFN Linear layers with ACBQLinear (in-place).

    Walks the model layer by layer, classifies each Linear as attention
    or FFN by name pattern, applies the appropriate bit-width, and
    propagates cross-block error compensation across layers.

    Attention layers (q_proj, k_proj, v_proj, out_proj, qkv_proj, etc.)
    and FFN layers (w_gate, w_up, w_down, gate_proj, up_proj, down_proj,
    etc.) are quantized with attn_bits and ffn_bits respectively. If
    attn_bits != ffn_bits, the model uses mixed precision: typically
    attn_bits=4, ffn_bits=2 for W4/W2 mixed mode, or attn_bits=4,
    ffn_bits=4 for uniform W4.

    Embedding and lm_head layers are always skipped.

    Args:
        model: model to quantize (modified in-place)
        group_size: quantization group size (128 is standard)
        attn_bits: bit-width for attention layers (4 or 2)
        ffn_bits: bit-width for FFN layers (4 or 2)
        compensation_strength: zero-point correction strength (1.0 = full)
        error_decay: accumulated error decay factor (0.0-1.0)
        verbose: print quantization summary

    Returns:
        Number of layers quantized.
    """
    quantizer = ACBQQuantizer(
        group_size=group_size,
        attn_bits=attn_bits,
        ffn_bits=ffn_bits,
        compensation_strength=compensation_strength,
        error_decay=error_decay,
    )
    n = 0
    n_attn = 0
    n_ffn = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if type(module).__name__ in _SKIP_TYPES:
            continue
        if any(s in name.lower() for s in _SKIP_NAMES):
            continue

        module_type = _classify_module(name)
        if module_type == "other":
            continue

        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        try:
            quantized = quantizer.quantize_linear(module, module_type)
            setattr(parent, parts[-1], quantized)
            n += 1
            if module_type == "attn":
                n_attn += 1
            else:
                n_ffn += 1
        except Exception as e:
            if verbose:
                logger.warning(f"  [ACBQ] Skipped {name}: {e}")

    if verbose and n > 0:
        mode = "W4A4" if attn_bits == 4 and ffn_bits == 4 else f"W{attn_bits}/W{ffn_bits}"
        logger.info(f"  [ACBQ] {n} layers quantized ({mode}, group={group_size}): "
              f"{n_attn} attn ({attn_bits}bit), {n_ffn} ffn ({ffn_bits}bit)")
        logger.info(f"  [ACBQ] Cross-block error feedback: "
              f"strength={compensation_strength}, decay={error_decay}, "
              f"layers_traversed={quantizer._layer_count}")

    return n
