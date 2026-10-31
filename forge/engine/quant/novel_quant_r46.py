"""Novel quantization algorithms (R&D Round 46, 2026-09-05).

Four techniques to close the quality gap between SchurAB-FP4 (PPL 43.75)
and FP16 (PPL 31.63) on Qwen 2.5 0.5B. Each addresses a different source
of quantization error, and they can be stacked:

1. HadamardRotatedFP4 (HR-FP4): QuaRot/SpinQuant-style Hadamard rotation
   pre-pass. Rotates weights (and activations at runtime) to reduce outliers
   and make the weight distribution more Gaussian. No calibration needed.
   The rotation is stored as a fixed matrix and applied at inference time.

2. GPTQFP4: GPTQ-style column-wise error compensation. Quantizes columns
   left-to-right, pushing the quantization error of each column into the
   remaining unquantized columns using the inverse Hessian. Needs calibration
   data to compute the Hessian. This is THE proven technique for closing
   the quality gap.

3. AWQFP4: Activation-aware weighting (AWQ-style). Weights the MSE objective
   by per-channel activation magnitude — salient channels get more accurate
   quantization. Simpler than GPTQ (no error compensation, just weighted
   scale search). Needs calibration data.

4. OptimalGridFP4 (OG-FP4): Data-dependent 4-bit grid search (OptIQ/AFQ-style).
   Instead of the fixed FP4 E2M1 magnitudes, search for per-layer optimal
   4-bit codebook via Lloyd-Max on the weight distribution. No calibration
   needed — uses weight statistics only.

Combined pipeline: Hadamard rotation → Optimal grid → GPTQ error compensation
                    (no cal)           (no cal)      (needs cal)

Sources (R46 research):
  - QuaRot: Ashkboos et al. ICLR 2025 (Hadamard rotation for quantization)
  - SpinQuant: Liu et al. ICLR 2025 (learned rotation, but Hadamard is free)
  - GPTQ: Frantar et al. ICLR 2023 (Hessian-based error compensation)
  - AWQ: Lin et al. MLSys 2024 (activation-aware salient channel protection)
  - OptIQ: Choi et al. 2025 (data-dependent optimal grid)
  - AFQ: Ma et al. 2025 (adaptive fractional quantization)
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse Hessian proxy computation from existing novel_quant
from forge.engine.quant.novel_quant import (
    _optimal_fp4_scale_hessian,
    compute_hessian_proxy,
)

# Reuse Hadamard matrix and skip-type list from R44
from forge.engine.quant.novel_quant_r44 import (
    _SKIP_NAMES,
    _SKIP_TYPES,
    _hadamard_matrix,
)

# Reuse FP4 primitives
from forge.engine.quant.nvfp4_quant import (
    _FP4_BOUNDARIES,
    _FP4_MAGNITUDES,
)
from forge.quant.protocol import QuantizedLinearMixin

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Weight caching mixin — eliminates redundant dequantization (speed fix)
# ──────────────────────────────────────────────────────────────────────────

class _CachedDequantMixin:
    """Mixin that caches dequantized weights for repeated forward passes.

    Without caching, every forward call re-dequantizes from packed FP4 →
    full precision, which is the #1 speed bottleneck (10 tok/s vs 47 tok/s
    for NVFP4 which caches). The cache is invalidated on device move or
    dtype change.
    """

    def _init_cache(self):
        self._cached_weight = None
        self._cached_dtype = None

    def _get_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Get dequantized weight, using cache if valid."""
        # Find a reference buffer to detect device changes
        ref = getattr(self, 'weight_packed', None)
        if ref is None:
            ref = getattr(self, 'U_binary', None)
        if ref is None:
            ref = getattr(self, 'codebook', None)
        if ref is None:
            ref = getattr(self, 'q_weights', None)
        ref_device = ref.device if ref is not None else None
        if (self._cached_weight is not None and
                self._cached_dtype == dtype and
                (ref_device is None or self._cached_weight.device == ref_device)):
            return self._cached_weight
        w = self._dequantize_weight(dtype)
        self._cached_weight = w
        self._cached_dtype = dtype
        return w

    def _invalidate_cache(self):
        self._cached_weight = None
        self._cached_dtype = None


# ──────────────────────────────────────────────────────────────────────────
# Algorithm 1: HadamardRotatedFP4 (HR-FP4) — QuaRot-style rotation
# ──────────────────────────────────────────────────────────────────────────

class HadamardRotatedFP4Linear(QuantizedLinearMixin, _CachedDequantMixin):
    """HadamardRotatedFP4: Hadamard rotation + FP4 quantization.

    Pipeline:
      1. Apply Hadamard rotation H to weight columns: W_rot = W @ H
      2. Quantize W_rot to FP4 with MSE-optimal per-block scale
      3. At inference: dequantize W_rot, then apply inverse rotation
         W = W_rot_dq @ H^T (H is orthonormal, H^{-1} = H^T)
      4. Forward: y = x @ W^T = x @ (W_rot_dq @ H^T)^T = x @ H @ W_rot_dq^T
         So we can either: (a) rotate x first, then use W_rot_dq, or
         (b) reconstruct W and use standard linear. We do (b) for simplicity.

    The rotation reduces outliers by mixing channels, making the weight
    distribution more Gaussian. This means the FP4 grid (designed for
    uniform-ish distributions) fits better, reducing quantization error.

    The rotation matrix H is NOT stored per-layer — it's recomputed from
    the deterministic Hadamard construction. Only the hadamard_size is stored.

    Memory overhead: zero (H is deterministic, not stored).
    Quality improvement: typically 10-30% reduction in reconstruction error.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 block_size: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Hadamard size (power of 2 >= in_features)
        h_size = 1
        while h_size < in_features:
            h_size *= 2
        self.hadamard_size = h_size

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # FP4 packed weights (2 per byte)
        self.register_buffer('weight_packed', torch.zeros(0, dtype=torch.uint8))
        self.register_buffer('weight_scales', torch.zeros(0, dtype=torch.float16))
        self.register_buffer('weight_global_scale', torch.zeros(0, dtype=torch.float32))

        self._init_cache()

    @classmethod
    def from_linear(cls, lin: nn.Linear, block_size: int = 32) -> HadamardRotatedFP4Linear:
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None, block_size=block_size)

        W = lin.weight.data.float()  # (out, in)
        device = W.device

        # Step 1: Pad to hadamard_size and apply Hadamard rotation
        pad = layer.hadamard_size - in_f
        if pad > 0:
            W = F.pad(W, (0, pad))
        H = _hadamard_matrix(layer.hadamard_size, device, W.dtype)
        W_rot = W @ H  # (out, hadamard_size)

        # Step 2: FP4 quantization with MSE-optimal scale (same as SchurAB-FP4)
        in_padded = W_rot.shape[1]
        pad2 = (block_size - in_padded % block_size) % block_size
        if pad2 > 0:
            W_rot = F.pad(W_rot, (0, pad2))
        in_padded = W_rot.shape[1]
        n_blocks = in_padded // block_size
        W_blocks = W_rot.view(out_f, n_blocks, block_size)

        # MSE-optimal scale search
        absmax = W_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        base_scale = absmax / 6.0
        candidates = torch.tensor(
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5],
            dtype=W.dtype, device=device,
        )
        scales_exp = (base_scale * candidates.unsqueeze(0).unsqueeze(0)).unsqueeze(-1)
        w_exp = W_blocks.unsqueeze(-2)
        w_norm = w_exp / scales_exp.clamp(min=1e-12)
        abs_norm = w_norm.abs()
        idx = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm)
        idx = idx.clamp(0, 7)
        magnitude = _FP4_MAGNITUDES.to(device)[idx]
        w_dq = torch.sign(w_norm) * magnitude * scales_exp
        mse = ((w_exp - w_dq) ** 2).mean(dim=-1)
        best_idx = mse.argmin(dim=-1)
        best_scale = base_scale.squeeze(-1) * candidates[best_idx]

        # Quantize with best scale
        w_norm_best = W_blocks / best_scale.unsqueeze(-1).clamp(min=1e-12)
        abs_norm_best = w_norm_best.abs()
        idx_best = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm_best).clamp(0, 7)

        # Pack: sign in bit 3, magnitude in bits 0-2
        sign_bit = (w_norm_best < 0).long() << 3
        fp4_code = (sign_bit | idx_best.long().clamp(0, 7)).to(torch.uint8)
        fp4_flat = fp4_code.view(out_f, -1)
        assert fp4_flat.shape[1] % 2 == 0
        low = fp4_flat[:, 0::2] & 0x0F
        high = (fp4_flat[:, 1::2] << 4) & 0xF0
        packed = (low | high).to(torch.uint8)

        # Two-level scaling
        global_scale = best_scale.amax(dim=1, keepdim=True).clamp(min=1e-12)
        block_scale_normalized = (best_scale / global_scale).to(torch.float16)

        layer.weight_packed = packed.contiguous().to(device)
        layer.weight_scales = block_scale_normalized.contiguous().to(device)
        layer.weight_global_scale = global_scale.squeeze(1).to(torch.float32).to(device)

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        device = self.weight_packed.device
        out_f = self.out_features
        in_f = self.in_features
        bs = self.block_size

        # Unpack FP4
        packed = self.weight_packed.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int64)
        high = (packed >> 4).to(torch.int64)
        fp4_flat = torch.stack([low, high], dim=-1).view(out_f, -1)

        sign = (fp4_flat >> 3).to(dtype) * -2 + 1
        mag_idx = (fp4_flat & 0x07).long()
        fp4_mag = _FP4_MAGNITUDES.to(device).to(dtype)
        n_blocks = fp4_flat.shape[1] // bs
        mag = fp4_mag[mag_idx.view(out_f, n_blocks, bs).clamp(0, 7)]

        # Two-level scaling
        global_s = self.weight_global_scale.to(dtype).unsqueeze(1)
        block_s = self.weight_scales.to(dtype)
        W_rot = sign.view(out_f, n_blocks, bs) * mag * block_s.unsqueeze(-1) * global_s.unsqueeze(-1)

        # Inverse Hadamard rotation: W = W_rot @ H^T
        W_rot_flat = W_rot.view(out_f, -1)
        H = _hadamard_matrix(self.hadamard_size, device, dtype)
        W = W_rot_flat @ H.T

        # Remove padding
        return W[:, :in_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        return (f"HadamardRotatedFP4Linear(in={self.in_features}, out={self.out_features}, "
                f"block={self.block_size}, h_size={self.hadamard_size})")


# ──────────────────────────────────────────────────────────────────────────
# Algorithm 2: GPTQFP4 — GPTQ error compensation
# ──────────────────────────────────────────────────────────────────────────

class GPTQFP4Linear(QuantizedLinearMixin, _CachedDequantMixin):
    """GPTQFP4: FP4 quantization with GPTQ-style error compensation.

    Pipeline:
      1. Compute Hessian H = E[x^T x] from calibration activations
      2. Quantize columns left-to-right (in blocks of `group_size`)
      3. For each column group:
         a. Quantize the group to FP4
         b. Compute the quantization error: e = W_q - W_orig
         c. Push error into remaining columns:
            W_remaining -= e @ H_remaining^{-1} @ H_cross
         (This is the GPTQ update: the error in quantized columns can be
          partially absorbed by adjusting unquantized columns, weighted by
          the Hessian which captures the output sensitivity.)
      4. Store the final FP4-quantized weights

    The GPTQ update formula (per column j):
      W[:, j+1:] -= (W_q[:, j] - W[:, j]) @ H^{-1}[j, j+1:] * (1 / H[j,j])

    We process columns in groups for efficiency (block GPTQ).

    Needs calibration data (activations) to compute the Hessian.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 block_size: int = 32, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.group_size = group_size

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.register_buffer('weight_packed', torch.zeros(0, dtype=torch.uint8))
        self.register_buffer('weight_scales', torch.zeros(0, dtype=torch.float16))
        self.register_buffer('weight_global_scale', torch.zeros(0, dtype=torch.float32))

        self._init_cache()

    @classmethod
    def from_linear(cls, lin: nn.Linear, activations: torch.Tensor,
                    block_size: int = 32, group_size: int = 128) -> GPTQFP4Linear:
        """Create GPTQ-quantized layer.

        Args:
            lin: original linear layer
            activations: (N, in_features) calibration activations
            block_size: FP4 block size for scale
            group_size: GPTQ column group size
        """
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None,
                    block_size=block_size, group_size=group_size)

        W = lin.weight.data.float().clone()  # (out, in) — will be modified
        device = W.device

        # Compute Hessian: H = X^T @ X / N (in_features x in_features)
        X = activations.float().to(device)  # (N, in_features)
        H = (X.T @ X / X.shape[0]).to(device)  # (in, in)
        # Add regularization for numerical stability
        H += 0.01 * torch.eye(in_f, device=device, dtype=H.dtype) * H.diag().mean().clamp(min=1e-6)

        # Pad in_features to multiple of block_size for FP4 storage
        pad = (block_size - in_f % block_size) % block_size
        if pad > 0:
            W = F.pad(W, (0, pad))
            H = F.pad(H, (0, pad, 0, pad))  # pad Hessian too
        in_padded = W.shape[1]
        n_blocks = in_padded // block_size

        # Compute inverse Hessian via Cholesky for numerical stability
        # GPTQ update requires H^{-1}, not H
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except Exception:
            # Fallback to direct inverse if Cholesky fails
            H_inv = torch.linalg.inv(H + 1e-2 * torch.eye(in_padded, device=device, dtype=H.dtype))

        # GPTQ: process columns left to right in groups
        # For each group, quantize then push error to remaining columns
        for g_start in range(0, in_padded, group_size):
            g_end = min(g_start + group_size, in_padded)

            # Get the current weights for this group (possibly updated by prior GPTQ)
            W_group = W[:, g_start:g_end].clone()

            # Quantize this group to FP4
            # Reshape into blocks within the group
            n_group_blocks = (g_end - g_start) // block_size
            W_group_blocks = W_group.view(out_f, n_group_blocks, block_size)

            # MSE-optimal scale per block
            absmax = W_group_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            base_scale = absmax / 6.0
            candidates = torch.tensor(
                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5],
                dtype=W.dtype, device=device,
            )
            scales_exp = (base_scale * candidates.unsqueeze(0).unsqueeze(0)).unsqueeze(-1)
            w_exp = W_group_blocks.unsqueeze(-2)
            w_norm = w_exp / scales_exp.clamp(min=1e-12)
            abs_norm = w_norm.abs()
            idx = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm).clamp(0, 7)
            magnitude = _FP4_MAGNITUDES.to(device)[idx]
            w_dq = torch.sign(w_norm) * magnitude * scales_exp
            mse = ((w_exp - w_dq) ** 2).mean(dim=-1)
            best_idx = mse.argmin(dim=-1)
            best_scale = base_scale.squeeze(-1) * candidates[best_idx]

            # Quantize
            w_norm_best = W_group_blocks / best_scale.unsqueeze(-1).clamp(min=1e-12)
            abs_norm_best = w_norm_best.abs()
            idx_best = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm_best).clamp(0, 7)
            mag_best = _FP4_MAGNITUDES.to(device)[idx_best]
            W_group_q = (torch.sign(w_norm_best) * mag_best * best_scale.unsqueeze(-1)).view(out_f, -1)

            # Error for this group
            err = W_group_q - W[:, g_start:g_end]  # (out, group_size)

            # GPTQ update: push error into remaining columns using INVERSE Hessian
            # W[:, remaining] -= err @ (H_inv[group, remaining] / diag(H_inv[group, group]))
            if g_end < in_padded:
                H_inv_cross = H_inv[g_start:g_end, g_end:]  # (group_size, remaining)
                H_inv_diag = H_inv[g_start:g_end, g_start:g_end].diag()
                H_inv_diag_inv = 1.0 / H_inv_diag.clamp(min=1e-8)
                # Batched GPTQ update: scale each column's contribution by 1/H_inv[j,j]
                update = err * H_inv_diag_inv.unsqueeze(0)  # (out, group_size)
                W[:, g_end:] -= update @ H_inv_cross  # (out, remaining)

            # Store quantized weights
            W[:, g_start:g_end] = W_group_q

        # Pack final weights into FP4 format
        # Re-quantize the full W (which now has GPTQ-corrected values)
        W_blocks = W.view(out_f, n_blocks, block_size)
        absmax = W_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        base_scale = absmax / 6.0
        candidates = torch.tensor(
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5],
            dtype=W.dtype, device=device,
        )
        scales_exp = (base_scale * candidates.unsqueeze(0).unsqueeze(0)).unsqueeze(-1)
        w_exp = W_blocks.unsqueeze(-2)
        w_norm = w_exp / scales_exp.clamp(min=1e-12)
        abs_norm = w_norm.abs()
        idx = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm).clamp(0, 7)
        magnitude = _FP4_MAGNITUDES.to(device)[idx]
        w_dq = torch.sign(w_norm) * magnitude * scales_exp
        mse = ((w_exp - w_dq) ** 2).mean(dim=-1)
        best_idx = mse.argmin(dim=-1)
        best_scale = base_scale.squeeze(-1) * candidates[best_idx]

        w_norm_final = W_blocks / best_scale.unsqueeze(-1).clamp(min=1e-12)
        abs_norm_final = w_norm_final.abs()
        idx_final = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm_final).clamp(0, 7)

        sign_bit = (w_norm_final < 0).long() << 3
        fp4_code = (sign_bit | idx_final.long().clamp(0, 7)).to(torch.uint8)
        fp4_flat = fp4_code.view(out_f, -1)
        assert fp4_flat.shape[1] % 2 == 0
        low = fp4_flat[:, 0::2] & 0x0F
        high = (fp4_flat[:, 1::2] << 4) & 0xF0
        packed = (low | high).to(torch.uint8)

        global_scale = best_scale.amax(dim=1, keepdim=True).clamp(min=1e-12)
        block_scale_normalized = (best_scale / global_scale).to(torch.float16)

        layer.weight_packed = packed.contiguous().to(device)
        layer.weight_scales = block_scale_normalized.contiguous().to(device)
        layer.weight_global_scale = global_scale.squeeze(1).to(torch.float32).to(device)

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        device = self.weight_packed.device
        out_f = self.out_features
        in_f = self.in_features
        bs = self.block_size

        packed = self.weight_packed.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int64)
        high = (packed >> 4).to(torch.int64)
        fp4_flat = torch.stack([low, high], dim=-1).view(out_f, -1)

        sign = (fp4_flat >> 3).to(dtype) * -2 + 1
        mag_idx = (fp4_flat & 0x07).long()
        fp4_mag = _FP4_MAGNITUDES.to(device).to(dtype)
        n_blocks = fp4_flat.shape[1] // bs
        mag = fp4_mag[mag_idx.view(out_f, n_blocks, bs).clamp(0, 7)]

        global_s = self.weight_global_scale.to(dtype).unsqueeze(1)
        block_s = self.weight_scales.to(dtype)
        W = sign.view(out_f, n_blocks, bs) * mag * block_s.unsqueeze(-1) * global_s.unsqueeze(-1)

        return W.view(out_f, -1)[:, :in_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        return (f"GPTQFP4Linear(in={self.in_features}, out={self.out_features}, "
                f"block={self.block_size}, group={self.group_size})")


# ──────────────────────────────────────────────────────────────────────────
# Algorithm 3: AWQFP4 — Activation-aware weighting
# ──────────────────────────────────────────────────────────────────────────

class AWQFP4Linear(QuantizedLinearMixin, _CachedDequantMixin):
    """AWQFP4: FP4 with activation-aware (Hessian-weighted) scale search.

    Pipeline:
      1. Compute per-channel Hessian proxy h_j = E[x_j^2] from activations
      2. For each FP4 block, find the MSE-optimal scale weighted by h_j
         (salient channels with high activation get more accurate quantization)
      3. Store FP4 weights with the activation-aware scale

    This is simpler than GPTQ (no error compensation) but captures the key
    AWQ insight: not all channels are equally important. The weighted scale
    search places FP4 levels where they reduce OUTPUT error, not WEIGHT error.

    Needs calibration data (activations) to compute the Hessian proxy.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 block_size: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.register_buffer('weight_packed', torch.zeros(0, dtype=torch.uint8))
        self.register_buffer('weight_scales', torch.zeros(0, dtype=torch.float16))
        self.register_buffer('weight_global_scale', torch.zeros(0, dtype=torch.float32))

        self._init_cache()

    @classmethod
    def from_linear(cls, lin: nn.Linear, activations: torch.Tensor,
                    block_size: int = 32) -> AWQFP4Linear:
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None, block_size=block_size)

        W = lin.weight.data.float()  # (out, in)
        device = W.device

        # Compute Hessian proxy from activations
        h = compute_hessian_proxy(activations.float().to(device))  # (in_features,)

        # Pad to block_size
        pad = (block_size - in_f % block_size) % block_size
        if pad > 0:
            W = F.pad(W, (0, pad))
            h = F.pad(h, (0, pad))
        in_padded = W.shape[1]
        n_blocks = in_padded // block_size

        W_blocks = W.view(out_f, n_blocks, block_size)
        # Reshape h to (n_blocks, block_size) then expand to (out_f, n_blocks, block_size)
        h_blocks = h.view(n_blocks, block_size).unsqueeze(0).expand(out_f, n_blocks, block_size).contiguous()

        # Hessian-weighted MSE-optimal scale
        best_scale = _optimal_fp4_scale_hessian(W_blocks, h_blocks)  # (out, n_blocks, 1)

        # Quantize
        w_norm = W_blocks / best_scale.clamp(min=1e-12)
        abs_norm = w_norm.abs()
        idx = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm).clamp(0, 7)

        # Pack
        sign_bit = (w_norm < 0).long() << 3
        fp4_code = (sign_bit | idx.long().clamp(0, 7)).to(torch.uint8)
        fp4_flat = fp4_code.view(out_f, -1)
        assert fp4_flat.shape[1] % 2 == 0
        low = fp4_flat[:, 0::2] & 0x0F
        high = (fp4_flat[:, 1::2] << 4) & 0xF0
        packed = (low | high).to(torch.uint8)

        global_scale = best_scale.amax(dim=1).clamp(min=1e-12)
        block_scale_normalized = (best_scale.squeeze(-1) / global_scale).to(torch.float16)
        global_scale = global_scale.squeeze(-1).to(torch.float32)

        layer.weight_packed = packed.contiguous().to(device)
        layer.weight_scales = block_scale_normalized.contiguous().to(device)
        layer.weight_global_scale = global_scale.to(device)

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        device = self.weight_packed.device
        out_f = self.out_features
        in_f = self.in_features
        bs = self.block_size

        packed = self.weight_packed.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int64)
        high = (packed >> 4).to(torch.int64)
        fp4_flat = torch.stack([low, high], dim=-1).view(out_f, -1)

        sign = (fp4_flat >> 3).to(dtype) * -2 + 1
        mag_idx = (fp4_flat & 0x07).long()
        fp4_mag = _FP4_MAGNITUDES.to(device).to(dtype)
        n_blocks = fp4_flat.shape[1] // bs
        mag = fp4_mag[mag_idx.view(out_f, n_blocks, bs).clamp(0, 7)]

        global_s = self.weight_global_scale.to(dtype).unsqueeze(1)
        block_s = self.weight_scales.to(dtype)
        W = sign.view(out_f, n_blocks, bs) * mag * block_s.unsqueeze(-1) * global_s.unsqueeze(-1)

        return W.view(out_f, -1)[:, :in_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        return (f"AWQFP4Linear(in={self.in_features}, out={self.out_features}, "
                f"block={self.block_size})")


# ──────────────────────────────────────────────────────────────────────────
# Algorithm 4: OptimalGridFP4 (OG-FP4) — data-dependent 4-bit codebook
# ──────────────────────────────────────────────────────────────────────────

class OptimalGridFP4Linear(QuantizedLinearMixin, _CachedDequantMixin):
    """OptimalGridFP4: Lloyd-Max optimal 4-bit codebook per layer.

    Instead of the fixed FP4 E2M1 magnitudes [0, 0.5, 0.75, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
    search for per-layer optimal 4-bit (16-level) codebook using Lloyd-Max
    iteration on the actual weight distribution.

    Pipeline:
      1. Collect weight statistics (or full weights) per layer
      2. Run Lloyd-Max iteration to find optimal 16-level codebook
      3. Quantize weights using the optimal codebook
      4. Store codebook + quantized indices

    The codebook has 16 levels (4 bits) including sign, so 8 positive magnitudes.
    We optimize the 8 magnitudes per layer.

    No calibration needed — uses weight statistics only.
    Memory overhead: 8 float values per layer (negligible).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 block_size: int = 32, n_lloyd_iters: int = 20):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.n_lloyd_iters = n_lloyd_iters

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.register_buffer('weight_packed', torch.zeros(0, dtype=torch.uint8))
        self.register_buffer('weight_scales', torch.zeros(0, dtype=torch.float16))
        self.register_buffer('weight_global_scale', torch.zeros(0, dtype=torch.float32))
        # Per-layer optimal codebook (8 magnitudes, symmetric ±)
        self.register_buffer('codebook', torch.zeros(8, dtype=torch.float32))

        self._init_cache()

    @classmethod
    def from_linear(cls, lin: nn.Linear, block_size: int = 32,
                    n_lloyd_iters: int = 20) -> OptimalGridFP4Linear:
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None,
                    block_size=block_size, n_lloyd_iters=n_lloyd_iters)

        W = lin.weight.data.float()  # (out, in)
        device = W.device

        # Pad to block_size
        pad = (block_size - in_f % block_size) % block_size
        if pad > 0:
            W = F.pad(W, (0, pad))
        in_padded = W.shape[1]
        n_blocks = in_padded // block_size

        # Step 1: Find optimal 8-magnitude codebook via Lloyd-Max
        # Use the absolute values of normalized weights
        # First, get a rough scale (per-block absmax / 6)
        W_blocks = W.view(out_f, n_blocks, block_size)
        absmax = W_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        base_scale = absmax / 6.0
        W_norm = W_blocks / base_scale.clamp(min=1e-12)

        # Collect all normalized absolute values for Lloyd-Max
        abs_vals = W_norm.abs().flatten()

        # Initialize codebook: evenly spaced in [0, max]
        max_val = abs_vals.max().item()
        codebook = torch.linspace(0, max_val, 8, dtype=torch.float32, device=device)
        # Ensure codebook is sorted and positive
        codebook = codebook.clamp(min=0)

        # Lloyd-Max iteration
        for _ in range(n_lloyd_iters):
            # Assign each value to nearest codebook entry
            dists = (abs_vals.unsqueeze(-1) - codebook.unsqueeze(0)).abs()
            assignments = dists.argmin(dim=-1)  # (n_vals,)
            # Update codebook: mean of assigned values
            for k in range(8):
                mask = (assignments == k)
                if mask.any():
                    codebook[k] = abs_vals[mask].mean()
            # Keep sorted
            codebook = codebook.sort().values

        # Step 2: Quantize using optimal codebook
        # For each block, find scale that minimizes MSE with the optimal codebook
        # Use the base_scale as starting point, search around it
        candidates = torch.tensor(
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5],
            dtype=W.dtype, device=device,
        )
        scales_exp = (base_scale * candidates.unsqueeze(0).unsqueeze(0)).unsqueeze(-1)
        w_exp = W_blocks.unsqueeze(-2)
        w_norm = w_exp / scales_exp.clamp(min=1e-12)
        abs_norm = w_norm.abs()

        # Find nearest codebook entry for each element
        cb = codebook.to(W.dtype)
        dists = (abs_norm.unsqueeze(-1) - cb.unsqueeze(0).unsqueeze(0).unsqueeze(0)).abs()
        idx = dists.argmin(dim=-1)  # (out, n_blocks, n_cand, block_size)
        mag = cb[idx]
        w_dq = torch.sign(w_norm) * mag * scales_exp
        mse = ((w_exp - w_dq) ** 2).mean(dim=-1)
        best_idx = mse.argmin(dim=-1)
        best_scale = base_scale.squeeze(-1) * candidates[best_idx]

        # Final quantization
        w_norm_final = W_blocks / best_scale.unsqueeze(-1).clamp(min=1e-12)
        abs_norm_final = w_norm_final.abs()
        dists_final = (abs_norm_final.unsqueeze(-1) - cb.unsqueeze(0).unsqueeze(0)).abs()
        idx_final = dists_final.argmin(dim=-1)

        # Pack: sign in bit 3, codebook index in bits 0-2
        sign_bit = (w_norm_final < 0).long() << 3
        fp4_code = (sign_bit | idx_final.long().clamp(0, 7)).to(torch.uint8)
        fp4_flat = fp4_code.view(out_f, -1)
        assert fp4_flat.shape[1] % 2 == 0
        low = fp4_flat[:, 0::2] & 0x0F
        high = (fp4_flat[:, 1::2] << 4) & 0xF0
        packed = (low | high).to(torch.uint8)

        global_scale = best_scale.amax(dim=1, keepdim=True).clamp(min=1e-12)
        block_scale_normalized = (best_scale / global_scale).to(torch.float16)

        layer.weight_packed = packed.contiguous().to(device)
        layer.weight_scales = block_scale_normalized.contiguous().to(device)
        layer.weight_global_scale = global_scale.squeeze(1).to(torch.float32).to(device)
        layer.codebook = codebook.contiguous().to(device)

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        device = self.weight_packed.device
        out_f = self.out_features
        in_f = self.in_features
        bs = self.block_size

        packed = self.weight_packed.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int64)
        high = (packed >> 4).to(torch.int64)
        fp4_flat = torch.stack([low, high], dim=-1).view(out_f, -1)

        sign = (fp4_flat >> 3).to(dtype) * -2 + 1
        mag_idx = (fp4_flat & 0x07).long()
        cb = self.codebook.to(device).to(dtype)
        n_blocks = fp4_flat.shape[1] // bs
        mag = cb[mag_idx.view(out_f, n_blocks, bs).clamp(0, 7)]

        global_s = self.weight_global_scale.to(dtype).unsqueeze(1)
        block_s = self.weight_scales.to(dtype)
        W = sign.view(out_f, n_blocks, bs) * mag * block_s.unsqueeze(-1) * global_s.unsqueeze(-1)

        return W.view(out_f, -1)[:, :in_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        return (f"OptimalGridFP4Linear(in={self.in_features}, out={self.out_features}, "
                f"block={self.block_size}, lloyd_iters={self.n_lloyd_iters})")


# ──────────────────────────────────────────────────────────────────────────
# Combined pipeline: Hadamard → OptimalGrid → GPTQ
# ──────────────────────────────────────────────────────────────────────────

class HadamardGPTQFP4Linear(QuantizedLinearMixin, _CachedDequantMixin):
    """HadamardGPTQFP4: Hadamard rotation + GPTQ error compensation.

    The strongest combination: rotation reduces outliers (making FP4 grid
    fit better), then GPTQ pushes remaining error into unquantized columns.

    Pipeline:
      1. Apply Hadamard rotation: W_rot = W @ H
      2. Compute Hessian of rotated activations: H_rot = H^T @ H_orig @ H
         (Since x_rot = x @ H, H_rot = E[x_rot^T x_rot] = H^T E[x^T x] H)
      3. GPTQ on W_rot with H_rot
      4. At inference: dequantize W_rot, apply inverse rotation

    Needs calibration data.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 block_size: int = 32, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.group_size = group_size

        h_size = 1
        while h_size < in_features:
            h_size *= 2
        self.hadamard_size = h_size

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.register_buffer('weight_packed', torch.zeros(0, dtype=torch.uint8))
        self.register_buffer('weight_scales', torch.zeros(0, dtype=torch.float16))
        self.register_buffer('weight_global_scale', torch.zeros(0, dtype=torch.float32))

        self._init_cache()

    @classmethod
    def from_linear(cls, lin: nn.Linear, activations: torch.Tensor,
                    block_size: int = 32, group_size: int = 128) -> HadamardGPTQFP4Linear:
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None,
                    block_size=block_size, group_size=group_size)

        W = lin.weight.data.float().clone()  # (out, in)
        device = W.device
        X = activations.float().to(device)  # (N, in_features)

        # Step 1: Pad and apply Hadamard rotation
        pad = layer.hadamard_size - in_f
        if pad > 0:
            W = F.pad(W, (0, pad))
            X = F.pad(X, (0, pad))
        H = _hadamard_matrix(layer.hadamard_size, device, W.dtype)

        W_rot = W @ H  # (out, hadamard_size)
        X_rot = X @ H  # (N, hadamard_size)

        # Step 2: Compute rotated Hessian and its inverse
        H_hess = (X_rot.T @ X_rot / X_rot.shape[0]).to(device)
        H_hess += 0.01 * torch.eye(layer.hadamard_size, device=device, dtype=H_hess.dtype) * H_hess.diag().mean().clamp(min=1e-6)

        # Step 3: GPTQ on rotated weights
        in_padded = W_rot.shape[1]
        pad2 = (block_size - in_padded % block_size) % block_size
        if pad2 > 0:
            W_rot = F.pad(W_rot, (0, pad2))
            H_hess = F.pad(H_hess, (0, pad2, 0, pad2))
        in_padded = W_rot.shape[1]
        n_blocks = in_padded // block_size

        # Compute inverse Hessian via Cholesky
        try:
            L = torch.linalg.cholesky(H_hess)
            H_inv = torch.cholesky_inverse(L)
        except Exception:
            H_inv = torch.linalg.inv(H_hess + 1e-2 * torch.eye(in_padded, device=device, dtype=H_hess.dtype))

        for g_start in range(0, in_padded, group_size):
            g_end = min(g_start + group_size, in_padded)

            W_group = W_rot[:, g_start:g_end].clone()
            n_group_blocks = (g_end - g_start) // block_size
            W_group_blocks = W_group.view(out_f, n_group_blocks, block_size)

            # MSE-optimal scale
            absmax = W_group_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            base_scale = absmax / 6.0
            candidates = torch.tensor(
                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5],
                dtype=W.dtype, device=device,
            )
            scales_exp = (base_scale * candidates.unsqueeze(0).unsqueeze(0)).unsqueeze(-1)
            w_exp = W_group_blocks.unsqueeze(-2)
            w_norm = w_exp / scales_exp.clamp(min=1e-12)
            abs_norm = w_norm.abs()
            idx = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm).clamp(0, 7)
            magnitude = _FP4_MAGNITUDES.to(device)[idx]
            w_dq = torch.sign(w_norm) * magnitude * scales_exp
            mse = ((w_exp - w_dq) ** 2).mean(dim=-1)
            best_idx = mse.argmin(dim=-1)
            best_scale = base_scale.squeeze(-1) * candidates[best_idx]

            w_norm_best = W_group_blocks / best_scale.unsqueeze(-1).clamp(min=1e-12)
            abs_norm_best = w_norm_best.abs()
            idx_best = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm_best).clamp(0, 7)
            mag_best = _FP4_MAGNITUDES.to(device)[idx_best]
            W_group_q = (torch.sign(w_norm_best) * mag_best * best_scale.unsqueeze(-1)).view(out_f, -1)

            err = W_group_q - W_rot[:, g_start:g_end]

            # GPTQ update with inverse Hessian
            if g_end < in_padded:
                H_inv_cross = H_inv[g_start:g_end, g_end:]
                H_inv_diag = H_inv[g_start:g_end, g_start:g_end].diag()
                H_inv_diag_inv = 1.0 / H_inv_diag.clamp(min=1e-8)
                update = err * H_inv_diag_inv.unsqueeze(0)
                W_rot[:, g_end:] -= update @ H_inv_cross

            W_rot[:, g_start:g_end] = W_group_q

        # Pack final weights
        W_blocks = W_rot.view(out_f, n_blocks, block_size)
        absmax = W_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        base_scale = absmax / 6.0
        candidates = torch.tensor(
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5],
            dtype=W.dtype, device=device,
        )
        scales_exp = (base_scale * candidates.unsqueeze(0).unsqueeze(0)).unsqueeze(-1)
        w_exp = W_blocks.unsqueeze(-2)
        w_norm = w_exp / scales_exp.clamp(min=1e-12)
        abs_norm = w_norm.abs()
        idx = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm).clamp(0, 7)
        magnitude = _FP4_MAGNITUDES.to(device)[idx]
        w_dq = torch.sign(w_norm) * magnitude * scales_exp
        mse = ((w_exp - w_dq) ** 2).mean(dim=-1)
        best_idx = mse.argmin(dim=-1)
        best_scale = base_scale.squeeze(-1) * candidates[best_idx]

        w_norm_final = W_blocks / best_scale.unsqueeze(-1).clamp(min=1e-12)
        abs_norm_final = w_norm_final.abs()
        idx_final = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm_final).clamp(0, 7)

        sign_bit = (w_norm_final < 0).long() << 3
        fp4_code = (sign_bit | idx_final.long().clamp(0, 7)).to(torch.uint8)
        fp4_flat = fp4_code.view(out_f, -1)
        assert fp4_flat.shape[1] % 2 == 0
        low = fp4_flat[:, 0::2] & 0x0F
        high = (fp4_flat[:, 1::2] << 4) & 0xF0
        packed = (low | high).to(torch.uint8)

        global_scale = best_scale.amax(dim=1, keepdim=True).clamp(min=1e-12)
        block_scale_normalized = (best_scale / global_scale).to(torch.float16)

        layer.weight_packed = packed.contiguous().to(device)
        layer.weight_scales = block_scale_normalized.contiguous().to(device)
        layer.weight_global_scale = global_scale.squeeze(1).to(torch.float32).to(device)

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        device = self.weight_packed.device
        out_f = self.out_features
        in_f = self.in_features
        bs = self.block_size

        packed = self.weight_packed.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int64)
        high = (packed >> 4).to(torch.int64)
        fp4_flat = torch.stack([low, high], dim=-1).view(out_f, -1)

        sign = (fp4_flat >> 3).to(dtype) * -2 + 1
        mag_idx = (fp4_flat & 0x07).long()
        fp4_mag = _FP4_MAGNITUDES.to(device).to(dtype)
        n_blocks = fp4_flat.shape[1] // bs
        mag = fp4_mag[mag_idx.view(out_f, n_blocks, bs).clamp(0, 7)]

        global_s = self.weight_global_scale.to(dtype).unsqueeze(1)
        block_s = self.weight_scales.to(dtype)
        W_rot = sign.view(out_f, n_blocks, bs) * mag * block_s.unsqueeze(-1) * global_s.unsqueeze(-1)

        # Inverse Hadamard
        W_rot_flat = W_rot.view(out_f, -1)
        H = _hadamard_matrix(self.hadamard_size, device, dtype)
        W = W_rot_flat @ H.T

        return W[:, :in_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        return (f"HadamardGPTQFP4Linear(in={self.in_features}, out={self.out_features}, "
                f"block={self.block_size}, group={self.group_size}, h_size={self.hadamard_size})")


# ──────────────────────────────────────────────────────────────────────────
# Algorithm 6: HadamardAWQFP4 — rotation + activation-aware weighting
# ──────────────────────────────────────────────────────────────────────────

class HadamardAWQFP4Linear(QuantizedLinearMixin, _CachedDequantMixin):
    """HadamardAWQFP4: Hadamard rotation + AWQ-style activation-aware scale.

    Combines the two best-performing R46 techniques:
      1. Hadamard rotation reduces outliers (QuaRot insight)
      2. AWQ-style Hessian-weighted scale search preserves salient channels

    Pipeline:
      1. Apply Hadamard rotation: W_rot = W @ H, x_rot = x @ H
      2. Compute rotated Hessian proxy: h_rot = E[x_rot^2]
      3. FP4 quantize W_rot with Hessian-weighted scale (AWQ on rotated weights)
      4. At inference: dequantize W_rot, apply inverse rotation

    Needs calibration data (for the Hessian proxy).
    Memory overhead: zero (H is deterministic).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 block_size: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        h_size = 1
        while h_size < in_features:
            h_size *= 2
        self.hadamard_size = h_size

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.register_buffer('weight_packed', torch.zeros(0, dtype=torch.uint8))
        self.register_buffer('weight_scales', torch.zeros(0, dtype=torch.float16))
        self.register_buffer('weight_global_scale', torch.zeros(0, dtype=torch.float32))

        self._init_cache()

    @classmethod
    def from_linear(cls, lin: nn.Linear, activations: torch.Tensor,
                    block_size: int = 32) -> HadamardAWQFP4Linear:
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None, block_size=block_size)

        W = lin.weight.data.float()  # (out, in)
        device = W.device
        X = activations.float().to(device)  # (N, in_features)

        # Step 1: Pad and apply Hadamard rotation
        pad = layer.hadamard_size - in_f
        if pad > 0:
            W = F.pad(W, (0, pad))
            X = F.pad(X, (0, pad))
        H = _hadamard_matrix(layer.hadamard_size, device, W.dtype)

        W_rot = W @ H  # (out, hadamard_size)
        X_rot = X @ H  # (N, hadamard_size)

        # Step 2: Compute rotated Hessian proxy
        h_rot = compute_hessian_proxy(X_rot)  # (hadamard_size,)

        # Step 3: Pad to block_size and quantize with AWQ-style weighted scale
        pad2 = (block_size - W_rot.shape[1] % block_size) % block_size
        if pad2 > 0:
            W_rot = F.pad(W_rot, (0, pad2))
            h_rot = F.pad(h_rot, (0, pad2))
        in_padded = W_rot.shape[1]
        n_blocks = in_padded // block_size

        W_blocks = W_rot.view(out_f, n_blocks, block_size)
        h_blocks = h_rot.view(n_blocks, block_size).unsqueeze(0).expand(
            out_f, n_blocks, block_size).contiguous()

        # Hessian-weighted MSE-optimal scale
        best_scale = _optimal_fp4_scale_hessian(W_blocks, h_blocks)  # (out, n_blocks, 1)

        # Quantize
        w_norm = W_blocks / best_scale.clamp(min=1e-12)
        abs_norm = w_norm.abs()
        idx = torch.searchsorted(_FP4_BOUNDARIES.to(device), abs_norm).clamp(0, 7)

        # Pack: sign in bit 3, magnitude in bits 0-2
        sign_bit = (w_norm < 0).long() << 3
        fp4_code = (sign_bit | idx.long().clamp(0, 7)).to(torch.uint8)
        fp4_flat = fp4_code.view(out_f, -1)
        assert fp4_flat.shape[1] % 2 == 0
        low = fp4_flat[:, 0::2] & 0x0F
        high = (fp4_flat[:, 1::2] << 4) & 0xF0
        packed = (low | high).to(torch.uint8)

        global_scale = best_scale.amax(dim=1).clamp(min=1e-12)
        block_scale_normalized = (best_scale.squeeze(-1) / global_scale).to(torch.float16)
        global_scale = global_scale.squeeze(-1).to(torch.float32)

        layer.weight_packed = packed.contiguous().to(device)
        layer.weight_scales = block_scale_normalized.contiguous().to(device)
        layer.weight_global_scale = global_scale.to(device)

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        device = self.weight_packed.device
        out_f = self.out_features
        in_f = self.in_features
        bs = self.block_size

        packed = self.weight_packed.to(torch.uint8)
        low = (packed & 0x0F).to(torch.int64)
        high = (packed >> 4).to(torch.int64)
        fp4_flat = torch.stack([low, high], dim=-1).view(out_f, -1)

        sign = (fp4_flat >> 3).to(dtype) * -2 + 1
        mag_idx = (fp4_flat & 0x07).long()
        fp4_mag = _FP4_MAGNITUDES.to(device).to(dtype)
        n_blocks = fp4_flat.shape[1] // bs
        mag = fp4_mag[mag_idx.view(out_f, n_blocks, bs).clamp(0, 7)]

        global_s = self.weight_global_scale.to(dtype).unsqueeze(1)
        block_s = self.weight_scales.to(dtype)
        W_rot = sign.view(out_f, n_blocks, bs) * mag * block_s.unsqueeze(-1) * global_s.unsqueeze(-1)

        # Inverse Hadamard rotation
        W_rot_flat = W_rot.view(out_f, -1)
        H = _hadamard_matrix(self.hadamard_size, device, dtype)
        W = W_rot_flat @ H.T

        return W[:, :in_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        return (f"HadamardAWQFP4Linear(in={self.in_features}, out={self.out_features}, "
                f"block={self.block_size}, h_size={self.hadamard_size})")


# ──────────────────────────────────────────────────────────────────────────
# Model-level conversion functions
# ──────────────────────────────────────────────────────────────────────────

_SKIP_TYPES_R46 = _SKIP_TYPES + (
    "HadamardRotatedFP4Linear", "GPTQFP4Linear", "AWQFP4Linear",
    "OptimalGridFP4Linear", "HadamardGPTQFP4Linear", "HadamardAWQFP4Linear",
    # R45 types
    "WaveletLiftLinear", "SchurABFP4Linear", "SVDLiftBinaryLinear",
)


def _replace_linears_r46(model: nn.Module, factory, verbose_name: str,
                         verbose: bool = True, **kwargs) -> int:
    """Generic nn.Linear replacement that skips R44/R45/R46 types."""
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R46:
            if any(s in name for s in _SKIP_NAMES):
                continue
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                setattr(parent, parts[-1], factory(module, **kwargs))
                n += 1
            except Exception as e:
                if verbose:
                    logger.warning(f"  [{verbose_name}] Skipped {name}: {e}")
    if verbose and n > 0:
        logger.info(f"  [{verbose_name}] {n} layers quantized")
    return n


def quantize_model_hadamard_rotated_fp4(model: nn.Module, block_size: int = 32,
                                        verbose: bool = True) -> int:
    """Replace all nn.Linear with HadamardRotatedFP4Linear (no calibration)."""
    return _replace_linears_r46(model, HadamardRotatedFP4Linear.from_linear,
                                "HR-FP4", verbose, block_size=block_size)


def quantize_model_gptq_fp4(model: nn.Module, activations: dict,
                            block_size: int = 32, group_size: int = 128,
                            verbose: bool = True) -> int:
    """Replace all nn.Linear with GPTQFP4Linear (needs calibration activations).

    Args:
        model: the model to quantize
        activations: dict mapping layer name -> (N, in_features) activation tensor
        block_size: FP4 block size
        group_size: GPTQ column group size
    """
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R46:
            if any(s in name for s in _SKIP_NAMES):
                continue
            if name not in activations:
                if verbose:
                    logger.info(f"  [GPTQ-FP4] No activations for {name}, using uniform")
                acts = torch.ones(64, module.in_features)
            else:
                acts = activations[name]
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                setattr(parent, parts[-1],
                        GPTQFP4Linear.from_linear(module, acts, block_size, group_size))
                n += 1
            except Exception as e:
                if verbose:
                    logger.warning(f"  [GPTQ-FP4] Skipped {name}: {e}")
    if verbose and n > 0:
        logger.info(f"  [GPTQ-FP4] {n} layers quantized")
    return n


def quantize_model_awq_fp4(model: nn.Module, activations: dict,
                           block_size: int = 32, verbose: bool = True) -> int:
    """Replace all nn.Linear with AWQFP4Linear (needs calibration activations)."""
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R46:
            if any(s in name for s in _SKIP_NAMES):
                continue
            if name not in activations:
                acts = torch.ones(64, module.in_features)
            else:
                acts = activations[name]
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                setattr(parent, parts[-1],
                        AWQFP4Linear.from_linear(module, acts, block_size))
                n += 1
            except Exception as e:
                if verbose:
                    logger.warning(f"  [AWQ-FP4] Skipped {name}: {e}")
    if verbose and n > 0:
        logger.info(f"  [AWQ-FP4] {n} layers quantized")
    return n


def quantize_model_optimal_grid_fp4(model: nn.Module, block_size: int = 32,
                                    n_lloyd_iters: int = 20,
                                    verbose: bool = True) -> int:
    """Replace all nn.Linear with OptimalGridFP4Linear (no calibration)."""
    return _replace_linears_r46(model, OptimalGridFP4Linear.from_linear,
                                "OG-FP4", verbose, block_size=block_size,
                                n_lloyd_iters=n_lloyd_iters)


def quantize_model_hadamard_gptq_fp4(model: nn.Module, activations: dict,
                                     block_size: int = 32, group_size: int = 128,
                                     verbose: bool = True) -> int:
    """Replace all nn.Linear with HadamardGPTQFP4Linear (needs calibration)."""
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R46:
            if any(s in name for s in _SKIP_NAMES):
                continue
            if name not in activations:
                acts = torch.ones(64, module.in_features)
            else:
                acts = activations[name]
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                setattr(parent, parts[-1],
                        HadamardGPTQFP4Linear.from_linear(module, acts, block_size, group_size))
                n += 1
            except Exception as e:
                if verbose:
                    logger.warning(f"  [HR-GPTQ-FP4] Skipped {name}: {e}")
    if verbose and n > 0:
        logger.info(f"  [HR-GPTQ-FP4] {n} layers quantized")
    return n


def quantize_model_hadamard_awq_fp4(model: nn.Module, activations: dict,
                                    block_size: int = 32,
                                    verbose: bool = True) -> int:
    """Replace all nn.Linear with HadamardAWQFP4Linear (needs calibration)."""
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R46:
            if any(s in name for s in _SKIP_NAMES):
                continue
            if name not in activations:
                acts = torch.ones(64, module.in_features)
            else:
                acts = activations[name]
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                setattr(parent, parts[-1],
                        HadamardAWQFP4Linear.from_linear(module, acts, block_size))
                n += 1
            except Exception as e:
                if verbose:
                    logger.warning(f"  [HR-AWQ-FP4] Skipped {name}: {e}")
    if verbose and n > 0:
        logger.info(f"  [HR-AWQ-FP4] {n} layers quantized")
    return n


# ──────────────────────────────────────────────────────────────────────────
# Calibration helper: collect activations from a model
# ──────────────────────────────────────────────────────────────────────────

def collect_activations(model: nn.Module, input_ids: torch.Tensor,
                        n_samples: int = 128, device: str = 'cpu') -> dict:
    """Run a forward pass and collect input activations for each nn.Linear.

    Args:
        model: the model (will be set to eval mode)
        input_ids: (batch, seq) token IDs
        n_samples: max activation rows to collect per layer
        device: device to run on

    Returns:
        dict mapping layer name -> (N, in_features) activation tensor
    """
    model.eval()
    activations = {}
    hooks = []

    def collect_hook(name):
        def hook(module, input, output):
            if len(activations.get(name, [])) < n_samples:
                x = input[0].detach()
                # Reshape to (batch*seq, features)
                x = x.reshape(-1, x.shape[-1])
                activations.setdefault(name, []).append(x[:n_samples])
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(collect_hook(name)))

    with torch.inference_mode():
        _ = model(input_ids.to(device))

    for h in hooks:
        h.remove()

    # Concatenate collected activations
    result = {}
    for name, acts_list in activations.items():
        if acts_list:
            result[name] = torch.cat(acts_list, dim=0)

    return result


# ──────────────────────────────────────────────────────────────────────────
# Memory estimation
# ──────────────────────────────────────────────────────────────────────────

def estimate_r46_memory(model: nn.Module) -> dict:
    """Estimate weight memory for R46 quantized layers."""
    breakdown = {}
    total_bytes = 0
    total_params = 0

    for _name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in ('HadamardRotatedFP4Linear', 'GPTQFP4Linear', 'AWQFP4Linear',
                        'OptimalGridFP4Linear', 'HadamardGPTQFP4Linear',
                        'HadamardAWQFP4Linear'):
            w_bytes = module.weight_packed.numel()
            s_bytes = module.weight_scales.numel() * 2
            g_bytes = module.weight_global_scale.numel() * 4
            cb_bytes = 0
            if hasattr(module, 'codebook') and module.codebook.numel() > 0:
                cb_bytes = module.codebook.numel() * 4
            w_bytes = w_bytes + s_bytes + g_bytes + cb_bytes
            params = module.out_features * module.in_features
            total_bits = w_bytes * 8
            eff_bits = total_bits / max(params, 1)
        else:
            continue

        if cls_name not in breakdown:
            breakdown[cls_name] = {'bytes': 0, 'params': 0, 'eff_bits': eff_bits}
        breakdown[cls_name]['bytes'] += w_bytes
        breakdown[cls_name]['params'] += params
        total_bytes += w_bytes
        total_params += params

    return {
        'breakdown': breakdown,
        'total_bytes': total_bytes,
        'total_mb': total_bytes / 1024**2,
        'total_params': total_params,
        'avg_eff_bits': (total_bytes * 8 / max(total_params, 1)),
    }
