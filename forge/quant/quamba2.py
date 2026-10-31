"""Quamba2: W4A8 quantization for SSM (Mamba) blocks.

From arXiv:2503.18737 (2025). Standard quantization (AQLM, GPTQ, SmoothQuant)
degrades SSM (State Space Model) blocks like Mamba because:

  1. The SSM recurrence is sensitive to quantization noise in the state update.
     Small errors in the discretized state h_t = A_bar * h_{t-1} + B_bar * x_t
     compound across the sequence (unlike attention, which is parallel).
  2. The discretization parameters A (log-space) and delta have special
     structure (A_log = log(-A), delta = softplus(dt_proj(...))) that must be
     preserved — quantizing them destroys the recurrence dynamics.
  3. The input-dependent gating (B, C, delta projections) has different
     sensitivity than attention QKV: the gating controls *which* state
     channels are read/written, so noise there corrupts the entire scan.

Quamba2 scheme:
  - **W4**: 4-bit weight quantization for in_proj, x_proj, out_proj, and the
    depthwise conv, with group-wise (per-group) asymmetric min-max scaling.
  - **A8**: 8-bit activation quantization for the SSM inputs/outputs
    (symmetric per-token absmax).
  - **Unquantized (FP16)**: A_log, dt_bias, D, and the SSM recurrence core
    stay in FP16 — these are small and sensitive.
  - **SmoothQuant**: per-channel scaling factor that migrates quantization
    difficulty from activations to weights (alpha=0.5 for SSMs, gentler than
    the 0.999 used for attention, since SSM activations are smoother).

The key insight: SSM blocks need DIFFERENT quantization than attention blocks.
The projections (in_proj, x_proj, out_proj) are standard GEMMs and tolerate
W4A8 well, but the SSM core (selective scan with A, delta, B, C) must stay
in FP16 to avoid compounding recurrence error.

References:
  - Quamba2: arXiv:2503.18737 (2025) — W4A8 for SSMs
  - Quamba:  arXiv:2406.03342 (2024) — INT8-only for SSMs (predecessor)
  - Mamba:   arXiv:2312.00752 (S6 selective state space model)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.quant.protocol import QuantizedLinearMixin

# ─── Quamba2Linear: W4 weight + A8 activation ─────────────────────────────

class Quamba2Linear(QuantizedLinearMixin):
    """4-bit weight, 8-bit activation linear layer with group-wise scaling.

    Weights are quantized to INT4 with per-group asymmetric (min-max) scaling
    along the input dimension.  Activations are quantized to INT8 with
    per-token symmetric (absmax) scaling, dynamically at inference time.

    SmoothQuant per-channel scaling (alpha) migrates outlier difficulty from
    activations to weights before quantization, reducing activation quant
    error.  For SSM blocks a gentler alpha (0.5) is used than for attention
    (0.999) because SSM activations are smoother and aggressive smoothing
    would inflate weight ranges past the 4-bit budget.

    Storage:
      - weight_int4_packed: (out, ceil(in/2)) uint8  — two 4-bit values/byte
      - weight_scale:       (out, n_groups) fp16      — per-group scale
      - weight_zero:        (out, n_groups) fp16      — per-group zero-point
      - act_scale:          (in,) fp32                — SmoothQuant channel scale
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = 128, smoothquant_alpha: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.smoothquant_alpha = smoothquant_alpha

        n_groups = (in_features + group_size - 1) // group_size
        # Packed INT4 weights: 2 values per uint8 byte
        self.register_buffer(
            "weight_int4_packed",
            torch.zeros(out_features, (in_features + 1) // 2, dtype=torch.uint8),
        )
        # Per-group scale and zero-point (asymmetric quantization)
        self.register_buffer(
            "weight_scale",
            torch.zeros(out_features, n_groups, dtype=torch.float16),
        )
        self.register_buffer(
            "weight_zero",
            torch.zeros(out_features, n_groups, dtype=torch.float16),
        )
        # SmoothQuant per-channel activation scale
        self.register_buffer(
            "act_scale",
            torch.ones(in_features, dtype=torch.float32),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        self._cached_weight: torch.Tensor | None = None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, lin: nn.Linear, group_size: int = 128,
                    smoothquant_alpha: float = 0.5,
                    calibration_activations: torch.Tensor | None = None,
                    ) -> Quamba2Linear:
        """Quantize an existing nn.Linear to Quamba2 W4A8 format.

        Args:
            lin: source nn.Linear (weight shape (out, in))
            group_size: number of input channels per quantization group
            smoothquant_alpha: activation smoothing (0=off, 0.5=gentle for SSM)
            calibration_activations: (N, in_features) sample activations for
                SmoothQuant scale computation. If None, uses weight-based heuristic.
        """
        w = lin.weight.float()  # (out, in)
        out_features, in_features = w.shape

        # ── SmoothQuant: migrate difficulty from activations to weights ──
        if smoothquant_alpha > 0:
            w_absmax = w.abs().amax(dim=0).clamp(min=1e-8)  # (in,)
            if calibration_activations is not None:
                a_absmax = calibration_activations.float().abs().amax(dim=0).clamp(min=1e-8)
            else:
                a_absmax = w_absmax * 0.5  # conservative heuristic
            act_scale = (a_absmax.pow(smoothquant_alpha) /
                         w_absmax.pow(1.0 - smoothquant_alpha)).clamp(min=1e-8)
            w = w * act_scale.unsqueeze(0)  # smooth weights
        else:
            act_scale = torch.ones(in_features, dtype=torch.float32)

        # ── Asymmetric per-group INT4 weight quantization ──
        gs = group_size
        n_groups = (in_features + gs - 1) // gs
        pad = n_groups * gs - in_features
        if pad > 0:
            w_padded = F.pad(w, (0, pad))
        else:
            w_padded = w
        w_grouped = w_padded.reshape(out_features, n_groups, gs)  # (out, n_groups, gs)

        # Asymmetric min-max: map [w_min, w_max] → [0, 15]
        w_min = w_grouped.amin(dim=-1, keepdim=True)
        w_max = w_grouped.amax(dim=-1, keepdim=True)
        scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)  # (out, n_groups, 1)
        zero = w_min  # zero-point = w_min (so 0 → w_min)
        # Quantize: q = round((w - zero) / scale), clamp to [0, 15]
        q = ((w_grouped - zero) / scale).round().clamp(0, 15).to(torch.uint8)
        # Flatten and pack two 4-bit values per byte
        q_flat = q.reshape(out_features, n_groups * gs)  # (out, padded_in)
        packed = q_flat[:, 0::2] | (q_flat[:, 1::2] << 4)  # (out, padded_in//2)

        obj = cls(in_features, out_features, bias=lin.bias is not None,
                  group_size=group_size, smoothquant_alpha=smoothquant_alpha)
        obj.weight_int4_packed = packed.contiguous()
        obj.weight_scale = scale.squeeze(-1).to(torch.float16)
        obj.weight_zero = zero.squeeze(-1).to(torch.float16)
        obj.act_scale = act_scale.to(torch.float32)
        if lin.bias is not None:
            obj.bias = lin.bias.to(torch.float16)
        obj._cached_weight = None
        return obj

    @torch.no_grad()
    def _dequantize_weight(self, dtype: torch.dtype,
                           cache: bool = False) -> torch.Tensor:
        """Dequantize INT4 packed weights to a full float tensor."""
        if self._cached_weight is not None:
            return self._cached_weight.to(dtype)
        packed = self.weight_int4_packed  # (out, packed_in)
        low = (packed & 0x0F).to(torch.float32)
        high = (packed >> 4).to(torch.float32)
        # Interleave: [low0, high0, low1, high1, ...]
        idx = torch.stack([low, high], dim=-1).reshape(self.out_features, -1)

        gs = self.group_size
        n_groups = self.weight_scale.shape[1]
        scale = self.weight_scale.to(torch.float32)  # (out, n_groups)
        zero = self.weight_zero.to(torch.float32)    # (out, n_groups)

        # Expand scales to per-element (idx includes padding from quantization)
        idx_grouped = idx[:, :n_groups * gs].reshape(self.out_features, n_groups, gs)
        w_deq = idx_grouped * scale.unsqueeze(-1) + zero.unsqueeze(-1)
        w = w_deq.reshape(self.out_features, -1)[:, :self.in_features]
        if cache:
            self._cached_weight = w.to(torch.float16).clone()
        return w.to(dtype)

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Dynamic per-token symmetric INT8 quantization (dequantized inline).

        Returns the dequantized activation (fake-quantization) so the GEMM
        runs in float.  This is the A8 path: the activation is quantized to
        INT8 and immediately dequantized, simulating the quantization noise.

        Uses a Straight-Through Estimator (STE) so gradients flow through
        the ``round()`` non-differentiability: forward uses the quantized
        value, backward passes the gradient through as if round were identity.
        """
        if self.smoothquant_alpha > 0:
            x = x / self.act_scale.to(x.dtype)
        x_flat = x.reshape(-1, self.in_features)
        absmax = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 127.0
        # Fake-quantize: round to int8 and back
        x_q = (x_flat / scale).round().clamp(-128, 127) * scale
        # STE: forward = x_q (quantized), backward = identity through x_flat
        x_q = x_flat + (x_q - x_flat).detach()
        return x_q.reshape(x.shape).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype, cache=True)
        x_q = self._quantize_activation(x)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x_q, w, bias)
        # LoRA adapter support — same convention as ForgeQuantLinear et al.
        lora = getattr(self, 'lora_adapter', None)
        if lora is not None:
            out = out + lora(x)
        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"group_size={self.group_size}, "
                f"smoothquant_alpha={self.smoothquant_alpha}, "
                f"quamba2_w4a8=True")


# ─── Quamba2Conv1d: W4 weight quantization for depthwise conv ──────────────

class Quamba2Conv1d(nn.Module):
    """4-bit weight quantized depthwise Conv1d for Quamba2.

    The depthwise causal conv in Mamba is a small but sensitive component.
    We quantize its weights to INT4 with per-channel asymmetric scaling
    (each output channel = each group is its own quantization group).
    Activations are kept in FP16 (the conv input is already post-SiLU and
    small; A8 here adds noise without benefit).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 groups: int = 1, bias: bool = True, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = padding
        assert groups == in_channels == out_channels, \
            "Quamba2Conv1d only supports depthwise (groups=in=out)"

        # Per-channel INT4: weight shape (out_ch, 1, kernel_size)
        self.register_buffer(
            "weight_int4_packed",
            torch.zeros(out_channels, (kernel_size + 1) // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(out_channels, dtype=torch.float16),
        )
        self.register_buffer(
            "weight_zero",
            torch.zeros(out_channels, dtype=torch.float16),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_channels, dtype=torch.float16))
        else:
            self.bias = None

        self._cached_weight: torch.Tensor | None = None

    @classmethod
    @torch.no_grad()
    def from_conv1d(cls, conv: nn.Conv1d) -> Quamba2Conv1d:
        """Quantize an nn.Conv1d (depthwise) to INT4 per-channel."""
        w = conv.weight.float()  # (out_ch, 1, k)
        out_ch = w.shape[0]
        k = w.shape[2]

        # Per-channel asymmetric quantization (each channel = one group)
        w_flat = w.reshape(out_ch, k)
        w_min = w_flat.amin(dim=-1, keepdim=True)
        w_max = w_flat.amax(dim=-1, keepdim=True)
        scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
        zero = w_min
        q = ((w_flat - zero) / scale).round().clamp(0, 15).to(torch.uint8)

        # Pack two per byte along kernel dimension
        pad = (2 - k % 2) % 2
        if pad > 0:
            q = F.pad(q, (0, pad), value=0)
        packed = q[:, 0::2] | (q[:, 1::2] << 4)

        obj = cls(conv.in_channels, conv.out_channels, conv.kernel_size[0]
                  if isinstance(conv.kernel_size, (tuple, list)) else conv.kernel_size,
                  groups=conv.groups if isinstance(conv.groups, int) else conv.groups[0],
                  bias=conv.bias is not None,
                  padding=conv.padding[0] if isinstance(conv.padding, (tuple, list)) else conv.padding)
        obj.weight_int4_packed = packed.contiguous()
        obj.weight_scale = scale.squeeze(-1).to(torch.float16)
        obj.weight_zero = zero.squeeze(-1).to(torch.float16)
        if conv.bias is not None:
            obj.bias = conv.bias.to(torch.float16)
        obj._cached_weight = None
        return obj

    @torch.no_grad()
    def _dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self._cached_weight is not None:
            return self._cached_weight.to(dtype)
        packed = self.weight_int4_packed  # (out_ch, packed_k)
        low = (packed & 0x0F).to(torch.float32)
        high = (packed >> 4).to(torch.float32)
        idx = torch.stack([low, high], dim=-1).reshape(self.out_channels, -1)
        idx = idx[:, :self.kernel_size]
        scale = self.weight_scale.to(torch.float32).unsqueeze(-1)
        zero = self.weight_zero.to(torch.float32).unsqueeze(-1)
        w = idx * scale + zero  # (out_ch, k)
        w = w.unsqueeze(1)  # (out_ch, 1, k) — depthwise
        self._cached_weight = w.to(torch.float16).clone()
        return w.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.conv1d(x, w, bias=bias, padding=self.padding,
                        groups=self.groups)


# ─── Quamba2Block: wraps a Mamba block with quantized projections ──────────

class Quamba2Block(nn.Module):
    """Wraps a Mamba/SSM block with W4A8 quantized projections.

    The linear projections (in_proj, x_proj, out_proj) and the depthwise conv
    are replaced with Quamba2Linear / Quamba2Conv1d (W4 weights, A8 activations).

    The SSM core stays in FP16:
      - A_log:  (d_inner, d_state) — log(-A), drives the recurrence dynamics
      - dt_bias / D: skip connection and discretization bias
      - The selective scan recurrence itself runs in FP16

    This is the central Quamba2 insight: quantize the GEMMs (which are
    noise-tolerant) but preserve the recurrence (which is noise-sensitive).
    """

    def __init__(self, mamba_block: nn.Module, group_size: int = 128,
                 smoothquant_alpha: float = 0.5):
        super().__init__()
        self.group_size = group_size
        self.smoothquant_alpha = smoothquant_alpha

        # Copy all attributes from the original block
        self.d_model = getattr(mamba_block, "d_model", 0)
        self.d_state = getattr(mamba_block, "d_state", 0)
        self.d_conv = getattr(mamba_block, "d_conv", 4)
        self.expand = getattr(mamba_block, "expand", 2)
        self.d_inner = getattr(mamba_block, "d_inner", self.expand * self.d_model)
        self.dt_rank = getattr(mamba_block, "dt_rank", max(1, self.d_model // 16))
        self.layer_idx = getattr(mamba_block, "layer_idx", 0)
        self.norm_eps = getattr(mamba_block, "norm_eps", 1e-6)

        # ── Quantized projections (W4A8) ──
        self.in_proj = Quamba2Linear.from_linear(
            mamba_block.in_proj, group_size=group_size,
            smoothquant_alpha=smoothquant_alpha)
        self.out_proj = Quamba2Linear.from_linear(
            mamba_block.out_proj, group_size=group_size,
            smoothquant_alpha=smoothquant_alpha)
        self.x_proj = Quamba2Linear.from_linear(
            mamba_block.x_proj, group_size=group_size,
            smoothquant_alpha=smoothquant_alpha)

        # dt_proj: quantize to W4A8 as well (it's a standard Linear)
        if hasattr(mamba_block, "dt_proj"):
            self.dt_proj = Quamba2Linear.from_linear(
                mamba_block.dt_proj, group_size=group_size,
                smoothquant_alpha=smoothquant_alpha)
        else:
            self.dt_proj = None

        # ── Quantized depthwise conv (W4) ──
        if hasattr(mamba_block, "conv1d"):
            self.conv1d = Quamba2Conv1d.from_conv1d(mamba_block.conv1d)
        else:
            self.conv1d = None

        # ── SSM core: kept in FP16 (unquantized) ──
        # A_log, D, dt_bias — these are small and sensitive
        if hasattr(mamba_block, "A_log"):
            self.A_log = nn.Parameter(mamba_block.A_log.data.clone().to(torch.float16))
        if hasattr(mamba_block, "D"):
            self.D = nn.Parameter(mamba_block.D.data.clone().to(torch.float16))
        if hasattr(mamba_block, "dt_bias"):
            # Some Mamba variants have a dt_bias (Mamba2)
            self.dt_bias = nn.Parameter(mamba_block.dt_bias.data.clone().to(torch.float16))
        else:
            self.dt_bias = None

        # Jamba RMSNorm layers (kept in FP16 — small, sensitive)
        for norm_attr in ("dt_layernorm", "b_layernorm", "c_layernorm"):
            if hasattr(mamba_block, norm_attr):
                val = getattr(mamba_block, norm_attr)
                if val is not None:
                    setattr(self, norm_attr,
                            nn.Parameter(val.data.clone().to(torch.float16)))
                else:
                    setattr(self, norm_attr, None)
            else:
                setattr(self, norm_attr, None)

        # SSM recurrent state (for incremental decoding)
        self._ssm_state = None
        self._conv_state = None

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        rms = xf.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        return weight.to(dtype) * (xf * rms).to(dtype)

    def reset_state(self):
        self._ssm_state = None
        self._conv_state = None

    def _selective_scan_ref(self, x, delta, A, B, C, D, h_init=None):
        """Reference selective scan — fp32 recurrence (unquantized SSM core),
        per-step output cast back to activation dtype like HF Jamba."""
        B_b, d_inner, L = x.shape
        d_state = A.shape[1]
        out_dtype = x.dtype
        delta_f = delta.float()
        B_f = B.float()
        x_f = x.float()
        A_neg = -torch.exp(A.float())  # (d_inner, d_state)
        if h_init is not None:
            h = h_init.float()
        else:
            h = torch.zeros(B_b, d_inner, d_state, device=x.device,
                            dtype=torch.float32)
        ys = []
        for t in range(L):
            dt = delta_f[:, :, t:t + 1]
            A_bar = torch.exp(dt * A_neg.unsqueeze(0))
            B_t = B_f[:, :, t]
            B_bar = dt * B_t.unsqueeze(1)
            x_t = x_f[:, :, t:t + 1]
            h = A_bar * h + B_bar * x_t
            C_t = C[:, :, t]
            y_t = (h.to(out_dtype) * C_t.unsqueeze(1)).sum(dim=-1) \
                + D.to(out_dtype) * x[:, :, t]
            ys.append(y_t)
        return torch.stack(ys, dim=-1), h

    def forward(self, x, past_key_value=None, use_cache=False, **kwargs):
        """Forward pass mirroring MambaLayer.forward but with quantized projections."""
        B, T, _D = x.shape
        # The SSM core (A_log, D, jamba norms) is stored in FP16 — mixed with
        # bf16 activations it promotes to fp32 mid-scan. Record the incoming
        # dtype so the block output is cast back before the residual stream
        # (fp32 output would break downstream bf16 matmuls).
        model_dtype = x.dtype

        # in_proj (W4A8) → split into x (ssm) and z (gate)
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Depthwise conv (W4)
        if T == 1 and past_key_value is not None and "conv_state" in past_key_value:
            conv_state = past_key_value["conv_state"]
            x_t = x.transpose(1, 2)
            full_input = torch.cat([conv_state, x_t], dim=-1)
            new_conv_state = torch.cat([conv_state[:, :, 1:], x_t], dim=-1)
            if self.conv1d is not None:
                w = self.conv1d._dequantize_weight(x.dtype).squeeze(1)
                conv_out = (w * full_input).sum(dim=-1, keepdim=True)
                if self.conv1d.bias is not None:
                    conv_out = conv_out + self.conv1d.bias.to(x.dtype).unsqueeze(0).unsqueeze(-1)
            else:
                conv_out = x_t
            x = conv_out.transpose(1, 2)
            present_conv_state = new_conv_state
        else:
            x = x.transpose(1, 2)  # (B, d_inner, T)
            if self.conv1d is not None:
                x = self.conv1d(x)[:, :, :T]
            x = x.transpose(1, 2)
            if use_cache:
                x_pre_conv, _ = xz.chunk(2, dim=-1)
                x_pre_conv_t = x_pre_conv.transpose(1, 2)
                if self.d_conv - 1 <= T:
                    present_conv_state = x_pre_conv_t[:, :, -(self.d_conv - 1):]
                else:
                    pad = self.d_conv - 1 - T
                    present_conv_state = torch.cat([
                        torch.zeros(B, self.d_inner, pad, device=x.device, dtype=x.dtype),
                        x_pre_conv_t,
                    ], dim=-1)
            else:
                present_conv_state = None

        x = F.silu(x)

        # x_proj (W4A8) → Δ, B, C
        x_proj_out = self.x_proj(x)
        delta, B, C = x_proj_out.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Jamba RMSNorms (FP16)
        if self.dt_layernorm is not None:
            delta = self._rmsnorm(delta, self.dt_layernorm)
        if self.b_layernorm is not None:
            B = self._rmsnorm(B, self.b_layernorm)
        if self.c_layernorm is not None:
            C = self._rmsnorm(C, self.c_layernorm)

        # dt_proj (W4A8) → Δ with softplus
        if self.dt_proj is not None:
            delta = self.dt_proj(delta)
        if self.dt_bias is not None:
            delta = delta + self.dt_bias.to(delta.dtype)
        delta = F.softplus(delta)

        # Selective scan (FP16 — unquantized SSM core)
        x_scan = x.transpose(1, 2)
        delta_scan = delta.transpose(1, 2)
        B_scan = B.transpose(1, 2)
        C_scan = C.transpose(1, 2)

        h_init = None
        if past_key_value is not None and "ssm_state" in past_key_value:
            h_init = past_key_value["ssm_state"]

        y, h_final = self._selective_scan_ref(
            x_scan, delta_scan, self.A_log.data, B_scan, C_scan, self.D,
            h_init=h_init)

        y = y.transpose(1, 2)  # (B, T, d_inner)
        # The FP16 SSM core promotes the scan output to fp32 — cast back to
        # the activation dtype before gating so out_proj (and any attached
        # lora_adapter) receive model-dtype input.
        y = y.to(z.dtype) * F.silu(z)

        # out_proj (W4A8)
        out = self.out_proj(y)
        # Restore the model dtype — internal fp16/fp32 promotion would
        # otherwise emit fp32 and break the next block's bf16 matmuls.
        out = out.to(model_dtype)

        present = None
        if use_cache:
            present = {"ssm_state": h_final, "conv_state": present_conv_state}
        return out, present


# ─── Quamba2Quantizer: quantizes a Mamba/SSM block to W4A8 ─────────────────

class Quamba2Quantizer:
    """Quantizes Mamba/SSM blocks to Quamba2 W4A8 format.

    Walks the model, finds Mamba/SSM blocks (detected by the presence of
    ``A_log`` and ``in_proj``), and replaces them with ``Quamba2Block``.
    Non-SSM Linear layers are left untouched (use a separate quantizer for
    those — Quamba2 is SSM-specific).

    Args:
        group_size: INT4 group size for weight quantization (default 128)
        smoothquant_alpha: SmoothQuant strength (0.5 = gentle, SSM-tuned)
    """

    def __init__(self, group_size: int = 128, smoothquant_alpha: float = 0.5):
        self.group_size = group_size
        self.smoothquant_alpha = smoothquant_alpha

    @staticmethod
    def _is_mamba_block(module: nn.Module) -> bool:
        """Detect a Mamba/SSM block by the presence of A_log + in_proj."""
        return (hasattr(module, "A_log") and hasattr(module, "in_proj")
                and hasattr(module, "out_proj"))

    def quantize_block(self, mamba_block: nn.Module) -> Quamba2Block:
        """Quantize a single Mamba/SSM block to W4A8.

        Replaces the block's linear layers (in_proj, x_proj, out_proj,
        dt_proj) and depthwise conv with quantized versions, while keeping
        A_log, dt_bias, D, and the SSM recurrence in FP16.

        Args:
            mamba_block: a Mamba/SSM block (must have A_log, in_proj, out_proj)

        Returns:
            Quamba2Block wrapping the quantized block
        """
        if not self._is_mamba_block(mamba_block):
            raise ValueError(
                f"Module {type(mamba_block).__name__} is not a Mamba/SSM block "
                f"(missing A_log/in_proj/out_proj)")
        return Quamba2Block(
            mamba_block, group_size=self.group_size,
            smoothquant_alpha=self.smoothquant_alpha)

    def quantize_model(self, model: nn.Module) -> int:
        """Replace all Mamba/SSM blocks in a model with Quamba2Block.

        Args:
            model: the model to quantize

        Returns:
            Number of SSM blocks quantized
        """
        n_quantized = 0

        def convert(module, prefix=""):
            nonlocal n_quantized
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                if self._is_mamba_block(child):
                    qblock = Quamba2Block(
                        child, group_size=self.group_size,
                        smoothquant_alpha=self.smoothquant_alpha)
                    qblock = qblock.to(next(child.parameters()).device
                                       if list(child.parameters()) else torch.device("cpu"))
                    setattr(module, name, qblock)
                    n_quantized += 1
                else:
                    convert(child, full_name)

        convert(model)
        return n_quantized


def quantize_model_quamba2(model: nn.Module, group_size: int = 128,
                           smoothquant_alpha: float = 0.5) -> int:
    """Replace all Mamba/SSM blocks in a model with Quamba2 W4A8 blocks.

    Non-SSM layers (attention, MLP, embeddings) are left untouched — Quamba2
    is SSM-specific.  Use a separate quantizer for non-SSM layers.

    Args:
        model: the model to quantize
        group_size: INT4 group size for weight quantization
        smoothquant_alpha: SmoothQuant strength (0.5 = SSM-tuned gentle)

    Returns:
        Number of SSM blocks quantized
    """
    quantizer = Quamba2Quantizer(group_size=group_size,
                                 smoothquant_alpha=smoothquant_alpha)
    return quantizer.quantize_model(model)
