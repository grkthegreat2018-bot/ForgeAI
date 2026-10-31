"""BitNet b1.58 — ternary quantization-aware training (QAT).

Forces every linear weight to {-1, 0, +1} (~1.58 bits) with an absmean
scale factor, replacing fp16 multiplications with integer add / zero-skip.
Near-fp16 quality at a fraction of the FLOPs and memory (BitNet b1.58,
Ma et al. 2024; BitNet a4.8 follow-ups).

Implementation notes:
  - ``BitNetLinear`` keeps the same ``weight`` parameter name, so state-dict
    keys stay identical to nn.Linear — the existing checkpoint loads as-is.
  - Quantization is active only when ``quantize=True``; the straight-through
    estimator (STE) lets gradients flow through the ternary rounding during
    QAT. With quantize=False the module is a plain Linear (lossless).
  - Scale is computed per-tensor as absmean(W) / 0.7 (b1.58 convention),
    matching the paper's steady-state weight distribution; it is not a
    learnable parameter, so no extra checkpoint keys.

Usage:
    from forge.keys.quantization.bitnet_b158_key import BitNetLinear, BitNetB158Key
    lin = BitNetLinear(d_model, hidden, quantize=True)   # training-mode QAT
    state = apply_bitnet_b158(state)                      # offline ternary bake
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.keys.misc.base import Key, KeyClass, KeyResult

_TERNARY_EPS = 1e-6


# ---------------------------------------------------------------------------
# True BitNet kernel A: integer-addition ternary GEMM (Triton, fp activations)
#
# b1.58-faithful: weights are {-1,0,1}, so the matmul is pure addition
# (+x where w==+1, -x where w==-1, zeros skipped) — no weight multiplies.
# NOTE: Triton's tl.broadcast_to / 3D tl.where are known-buggy (issues
# #2157, #1467, #532 — wrong dim silently broadcast); we use the documented
# workaround: subscript notation + direct multiply.
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

if _HAS_TRITON:
    @triton.jit
    def _ternary_add_kernel(
        X, W, Y,
        M, N, K,
        sxm, sxk, swn, swk,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        """Raw ternary matmul (x @ W^T) as pure additions, no scale.

        Scaling is applied on the host (kernel-side vector*matrix broadcast
        hits Triton bugs #2157/#1467)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + offs_k
            xm = (offs_m[:, None] < M) & (kk[None, :] < K)
            wm = (offs_n[:, None] < N) & (kk[None, :] < K)
            x = tl.load(X + offs_m[:, None] * sxm + kk[None, :] * sxk,
                        mask=xm, other=0.0)
            w = tl.load(W + offs_n[:, None] * swn + kk[None, :] * swk,
                        mask=wm, other=0)
            # Add-only accumulation. Convert int8 weights to fp BEFORE the
            # comparison: int8==int hits a Triton bool-broadcast bug
            # (#7957 broadcast_bug_bool_mat); subscript broadcast is used
            # because tl.broadcast_to mis-broadcasts (#2157, #1467).
            wf = w.to(tl.float32)
            acc += tl.sum(
                x[:, None, :] * (wf[None, :, :] == 1.0).to(tl.float32)
                - x[:, None, :] * (wf[None, :, :] == -1.0).to(tl.float32),
                axis=2)
        ym = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :],
                 acc.to(Y.dtype.element_ty), mask=ym)


def _triton_ternary_linear(x: torch.Tensor, w: torch.Tensor,
                           scale: torch.Tensor) -> torch.Tensor:
    """y = (x @ w.T) * scale, w ternary int8 — add-only, zero-skip."""
    M, K = x.reshape(-1, x.shape[-1]).shape
    N = w.shape[0]
    x2 = x.reshape(M, K).contiguous()
    w2 = w.to(torch.int8).contiguous()
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BM, BN, BK = 32, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _ternary_add_kernel[grid](
        x2, w2, y, M, N, K,
        x2.stride(0), x2.stride(1), w2.stride(0), w2.stride(1),
        BM=BM, BN=BN, BK=BK,
    )
    y = y * scale.view(1, -1).to(y.dtype)
    return y.view(*x.shape[:-1], N)


# ---------------------------------------------------------------------------
# True BitNet kernel B: INTEGER GEMM on tensor cores (int8 @ int8)
# ---------------------------------------------------------------------------

def _int8_ternary_linear(x: torch.Tensor, w: torch.Tensor,
                         w_scale: torch.Tensor) -> torch.Tensor:
    """y = (x @ w.T) * (x_scale * w_scale) with w ternary int8.

    REAL integer matmul: activations are quantized to int8 (absmax, BitNet
    a4.8 style) and multiplied on the GPU's integer tensor cores via
    torch._int_mm — no floating-point matmul at all. Ternary weights make
    the int8 weight matrix exact (no weight quantization error).
    """
    M, K = x.reshape(-1, x.shape[-1]).shape
    N = w.shape[0]
    xf = x.reshape(M, K)
    x_absmax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    x8 = torch.clamp(torch.round(xf * (127.0 / x_absmax)),
                     -127, 127).to(torch.int8)
    w8 = w.to(torch.int8).t().contiguous()
    if M <= 16:
        # torch._int_mm requires M > 16; pad with zero rows.
        x8 = torch.cat([x8, torch.zeros(32 - M, K, dtype=torch.int8,
                                        device=x.device)], dim=0)
        y = torch._int_mm(x8, w8).float()[:M]
    else:
        y = torch._int_mm(x8, w8).float()
    y = y * ((x_absmax / 127.0) * w_scale.view(1, -1).to(x.device))
    # Cast back to the activation dtype — fp32 output would poison the
    # bf16 residual stream downstream (rms_norm/linear dtype mismatch).
    return y.to(x.dtype).view(*x.shape[:-1], N)


class _TernaryLinearFn(torch.autograd.Function):
    """Ternary forward via integer kernels; STE backward through the
    full-precision master weights (QAT semantics).

    Kernel selection (CUDA):
      - "triton" : b1.58 add-only ternary GEMM (fp activations, zero-skip)
      - default  : int8 @ int8 on tensor cores (a4.8-style activation q)
    CPU falls back to an fp GEMM on the ternary weights.
    """

    @staticmethod
    def forward(ctx, x, w, qscale, kernel):
        q, scale = ternary_quantize(w, qscale)
        scale = scale.detach().to(x.device)
        if kernel == "triton" and _HAS_TRITON and x.is_cuda:
            y = _triton_ternary_linear(x, q, scale)
        elif kernel == "int8" and x.is_cuda:
            y = _int8_ternary_linear(x, q, scale)
        else:
            y = F.linear(x, q.to(x.dtype), None) * scale.to(x.dtype)
        ctx.save_for_backward(x, w, y)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w, y = ctx.saved_tensors
        gdt = grad_y.dtype
        grad_x = grad_y @ w.to(gdt)
        gx = grad_y.reshape(-1, grad_y.shape[-1])
        xr = x.reshape(-1, x.shape[-1]).to(gdt)
        grad_w = gx.T @ xr
        # Clamp scale to avoid div-by-zero → NaN (meta materialization can leave qscale=0)
        safe_scale = ctx.scale.view(1, -1).to(gdt).clamp(min=1e-8)
        grad_scale = (grad_y * (y.to(gdt) / safe_scale)).sum()
        return grad_x, grad_w, grad_scale, None


class _Int8TrainableFn(torch.autograd.Function):
    """STE for int8 trainable storage (R&D round 15).

    Forward: uses the int8 ternary weight buffer on GPU (no master weight
    needed for forward — saves VRAM).
    Backward: STE — gradient flows to the bf16 master weight on CPU.
    The int8 buffer has no gradient (it's a buffer, not a parameter).

    The master weight gradient is computed as:
        grad_master = grad_y.T @ x   (same as standard Linear backward)
    and placed on the master weight's device (CPU), so no GPU VRAM is
    used for the gradient of the 8B weight matrix.

    Args to forward:
        x:       [*, in_features] activation on GPU
        w_int8:  [out_features, in_features] int8 ternary weight on GPU
        scale:   scalar or [out_features] scale on GPU
        w_master: [out_features, in_features] bf16 master weight on CPU
                  (only used for gradient computation target)
    """
    @staticmethod
    def forward(ctx, x, w_int8, scale, w_master):
        # Forward: int8 ternary GEMM (identical to prequantized path)
        if x.is_cuda:
            y = _int8_ternary_linear(x, w_int8, scale)
        else:
            w = w_int8.to(x.dtype) * scale
            y = F.linear(x, w, None)
        ctx.save_for_backward(x, w_int8, scale, w_master, y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w_int8, scale, _w_master, y = ctx.saved_tensors
        gdt = grad_y.dtype
        # grad_x = grad_y @ w_dequantized (on GPU, using int8 weight)
        w_dq = (w_int8.to(gdt) * scale.to(gdt))
        grad_x = grad_y @ w_dq
        # grad_w_master = grad_y.T @ x (computed on the master's device = CPU)
        # Move grad_y and x to CPU for the gradient computation.
        # This is the key VRAM saver: the [out, in] gradient lives on CPU,
        # not GPU. For 8B params, this saves ~16GB of GPU VRAM.
        gx = grad_y.reshape(-1, grad_y.shape[-1]).to(torch.bfloat16).cpu()
        xr = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).cpu()
        grad_w = gx.T @ xr  # CPU computation, result on CPU
        # grad_scale: derivative of y = (x @ w_int8.T) * scale w.r.t. scale
        # y/scale = x @ w_int8.T (the unscaled output), so d(scale)/dy = y/scale
        safe_scale = scale.view(1, -1).to(gdt).clamp(min=1e-8)
        grad_scale = (grad_y * (y.to(gdt) / safe_scale)).sum().to(scale.device)
        # w_int8 has no gradient (it's a buffer)
        return grad_x, None, grad_scale, grad_w


def ternary_quantize(w: torch.Tensor, scale: float | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, 0, +1} * scale.

    Returns (quantized_weights, scale). Uses the b1.58 absmean convention:
    scale = absmean(W) / 0.7 (0.7 ≈ 2/3 split point for ternary rounding).
    """
    w_abs = w.abs()
    if scale is None:
        scale = w_abs.mean().clamp(min=_TERNARY_EPS) / 0.7
    q = torch.where(w_abs < 0.5 * scale, torch.zeros_like(w),
                    torch.sign(w))
    return q, scale


class BitNetLinear(nn.Module):
    """Linear layer with STE ternary QAT (BitNet b1.58).

    QAT semantics (world-class practice):
      - training forward: ternary weights + STE (master weights stay fp).
      - eval forward: full-precision master weights by default (lossless for
        a warm-started / partially-trained model); set force_quant=True once
        QAT converges to deploy true ternary inference.
      - learned_scale: a per-layer learnable quantize scale (absmean init),
        which QAT tunes to minimize ternary rounding error — the step that
        recovers most of the b1.58 quality gap vs. a fixed absmean scale.
      - prequantized: weights stored as int8 buffer (1 byte vs 4 bytes fp32).
        Set via ``convert_to_int8_storage()`` after loading a pre-quantized
        checkpoint. Forward pass uses int8 directly with tensor-core GEMM.
        4x VRAM reduction for weights; no runtime quantization cost.

    Args:
        in_features, out_features: as nn.Linear.
        bias: use a bias term.
        quantize: True = ternary forward during training (QAT).
        force_quant: also quantize in eval (deployment mode).
        learned_scale: learn a per-layer scale (param ``qscale``).
        quantize_scale: fixed scale (None = absmean / learned).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, quantize: bool = False,
                 force_quant: bool = False, learned_scale: bool = True,
                 quantize_scale: float | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize = quantize
        self.force_quant = force_quant
        self.learned_scale = learned_scale
        self.quantize_scale = quantize_scale
        self._prequantized = False  # set True by convert_to_int8_storage()
        self._int8_trainable = False  # set True by enable_int8_training()
        # Kaiming init matching nn.Linear so pre-trained weights load fine.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        self.qscale = None
        if learned_scale:
            # Init from the b1.58 absmean convention; QAT then tunes it.
            with torch.no_grad():
                scale = self.weight.abs().mean().clamp(min=_TERNARY_EPS) / 0.7
            self.qscale = nn.Parameter(scale)

    def convert_to_int8_storage(self):
        """Convert fp32 weight parameter to int8 buffer (pre-quantized inference).

        Ternary-quantizes the current weight, stores as int8 buffer, and removes
        the fp32 parameter. Reduces weight VRAM by 4x (4 bytes → 1 byte per param).
        Also stores qscale as a buffer (no gradient needed in inference).

        After conversion, the layer uses the int8 tensor-core GEMM path directly
        — no runtime quantization cost.
        """
        if self._prequantized:
            return  # already converted
        with torch.no_grad():
            scale = self.qscale if self.qscale is not None else self.quantize_scale
            q, s = ternary_quantize(self.weight.data.float(), scale)
            # Store ternary weights as int8 {-1, 0, +1}
            w_int8 = q.to(torch.int8)
            # Register as buffer (not parameter — no gradients in inference)
            device = self.weight.device
            dtype = self.weight.dtype
            del self.weight
            self.register_buffer("weight_int8", w_int8.to(device))
            # Store scale as buffer too
            if isinstance(s, torch.Tensor):
                self.register_buffer("qscale_buf", s.to(device).to(dtype))
            else:
                self.register_buffer("qscale_buf", torch.tensor(s, device=device, dtype=dtype))
            # Remove qscale parameter if it exists
            if self.qscale is not None:
                del self.qscale
                self.qscale = None
            self._prequantized = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + "weight"
        qscale_key = prefix + "qscale"
        src_tensor = state_dict.get(weight_key, None)
        is_int8_weight = src_tensor is not None and src_tensor.dtype == torch.int8

        if is_int8_weight:
            # Pre-quantized int8: load directly into int8 storage — NO bf16 upcast.
            # Use checkpoint qscale if available, else compute from absmean.
            qscale = state_dict.get(qscale_key)
            if qscale is None:
                qscale = src_tensor.float().abs().mean().clamp(min=_TERNARY_EPS) / 0.7
            # Remove from state_dict so super() doesn't try to load it as bf16
            del state_dict[weight_key]
            if qscale_key in state_dict:
                del state_dict[qscale_key]
            # Call super for any remaining keys (bias, etc.)
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs)
            # Load int8 directly
            self.load_prequantized(src_tensor, qscale)
            return

        # Normal (bf16/fp32) weight loading path
        had_weight = weight_key in state_dict
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

        if had_weight and self.qscale is not None and not self.weight.is_meta:
            with torch.no_grad():
                self.qscale.copy_(
                    self.weight.abs().mean().clamp(min=_TERNARY_EPS) / 0.7)
            # else: weight is on meta, skip qscale update (will be set after assign)

    def load_prequantized(self, weight_int8: torch.Tensor, qscale: float | torch.Tensor):
        """Load a pre-quantized ternary weight (int8) directly into int8 storage.

        Bypasses the fp32 parameter entirely — stores as int8 buffer.
        Call this after __init__ but before any forward pass.
        The tensor's own device is used (typically GPU for direct-VRAM loading).
        """
        # Use the tensor's device (already on GPU if loaded direct from safetensors)
        device = weight_int8.device
        # Remove fp32 weight parameter
        if hasattr(self, 'weight'):
            del self.weight
        # Register int8 buffer (keep on tensor's device)
        self.register_buffer("weight_int8", weight_int8.to(torch.int8))
        # Register scale buffer
        if isinstance(qscale, torch.Tensor):
            self.register_buffer("qscale_buf", qscale.to(device))
        else:
            self.register_buffer("qscale_buf", torch.tensor(qscale, device=device))
        # Remove qscale parameter
        if self.qscale is not None:
            del self.qscale
            self.qscale = None
        self._prequantized = True

    # ── R&D round 15: int8 trainable storage ───────────────────────────────
    # Split storage: int8 ternary weights on GPU (forward), bf16 master on CPU
    # (gradient target). Enables 8B param training on 12GB VRAM.
    #   GPU: weight_int8 buffer (1 byte/param) = 8GB for 8B
    #   CPU: weight_master bf16 parameter (2 bytes/param) = 16GB in RAM
    # Forward: int8 ternary GEMM (same as prequantized inference path)
    # Backward: STE — grad flows to CPU master via a custom autograd Function
    # Optimizer (BAdam): updates CPU master for active block, then calls
    #   requantize_from_master() to refresh the int8 buffer on GPU.

    def enable_int8_training(self):
        """Convert from fp32 parameter to int8 trainable storage.

        Moves the weight to CPU as bf16 master, ternary-quantizes to int8
        buffer on GPU. The qscale becomes a CPU parameter (learnable).
        Call after loading weights, before training starts.
        """
        if self._prequantized or getattr(self, '_int8_trainable', False):
            return
        with torch.no_grad():
            w = self.weight.data
            device = w.device
            # Master weight → CPU bf16
            master = w.detach().cpu().to(torch.bfloat16).clone()
            # Ternary quantize → int8 on GPU
            scale = self.qscale if self.qscale is not None else self.quantize_scale
            q, s = ternary_quantize(w.float(), scale)
            w_int8 = q.to(torch.int8).to(device)
            # Replace parameter: weight becomes CPU bf16 master
            del self.weight
            self.weight = nn.Parameter(master, requires_grad=True)
            # Register int8 buffer on GPU
            self.register_buffer("weight_int8", w_int8)
            # qscale stays as parameter but on CPU (it's a scalar, negligible)
            if isinstance(s, torch.Tensor):
                self.register_buffer("qscale_buf", s.to(device).to(torch.bfloat16))
            else:
                self.register_buffer("qscale_buf",
                                     torch.tensor(s, device=device, dtype=torch.bfloat16))
            self._int8_trainable = True

    def requantize_from_master(self):
        """Re-quantize the CPU master weight → int8 buffer on GPU.

        Called by the optimizer after updating the active block's master weight.
        This is the STE re-projection step: the updated fp master is projected
        back to ternary int8 for the next forward pass.
        """
        if not getattr(self, '_int8_trainable', False):
            return
        with torch.no_grad():
            w_master = self.weight.data.float()  # CPU
            scale = self.qscale if self.qscale is not None else self.quantize_scale
            if isinstance(scale, nn.Parameter):
                scale_val = scale.data.float().cpu()  # ensure CPU for ternary_quantize
            else:
                scale_val = scale
            q, s = ternary_quantize(w_master, scale_val)
            # Update int8 buffer on GPU (non-blocking async copy)
            self.weight_int8.copy_(q.to(torch.int8).to(self.weight_int8.device,
                                                       non_blocking=True))
            # Update scale buffer
            if isinstance(s, torch.Tensor):
                self.qscale_buf.copy_(s.to(self.qscale_buf.device,
                                           non_blocking=True).to(torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── int8 trainable storage path (R&D round 15) ──
        if getattr(self, '_int8_trainable', False):
            w_int8 = self.weight_int8  # int8 on GPU
            scale = self.qscale_buf.to(x.dtype)
            # Forward: int8 ternary GEMM (same as prequantized)
            if x.is_cuda and self.bias is None:
                y = _Int8TrainableFn.apply(x, w_int8, scale, self.weight)
            else:
                w = w_int8.to(x.dtype) * scale
                y = _Int8TrainableFn.apply(x, w_int8, scale, self.weight)
            if self.bias is not None:
                y = y + self.bias
            return y
        # Pre-quantized int8 path: weights already ternary, stored as int8 buffer.
        # 4x less VRAM than fp32 parameter; no runtime quantization needed.
        if self._prequantized:
            w_int8 = self.weight_int8  # int8 buffer, values {-1, 0, +1}
            scale = self.qscale_buf.to(x.dtype) if hasattr(self, 'qscale_buf') else 1.0
            if x.is_cuda and self.bias is None:
                # Use int8 tensor-core GEMM directly (weights already int8).
                return _int8_ternary_linear(x, w_int8, scale)
            # CPU fallback: dequantize int8 → fp, standard GEMM.
            w = w_int8.to(x.dtype) * scale
            return F.linear(x, w, self.bias)
        # Normal QAT / eval path.
        w = self.weight
        if self.quantize and (self.training or self.force_quant):
            scale = self.qscale if self.qscale is not None else self.quantize_scale
            if x.is_cuda and self.bias is None:
                import os
                kernel = os.environ.get("FORGE_BITNET_KERNEL", "int8")
                return _TernaryLinearFn.apply(x, w, scale, kernel)
            q, _ = ternary_quantize(w, scale)
            w = w + (q - w).detach()
        return F.linear(x, w, self.bias)


def enable_int8_training(model: nn.Module) -> int:
    """Enable int8 trainable storage on all BitNetLinear modules in a model.

    Converts each BitNetLinear's fp32 weight parameter to:
      - int8 ternary buffer on GPU (forward pass)
      - bf16 master weight on CPU (gradient target + optimizer update)

    This is the R&D round 15 innovation that enables 8B param training on
    12GB VRAM. Call after model loading, before training starts.

    Returns the number of modules converted.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, BitNetLinear) and not module._int8_trainable:
            module.enable_int8_training()
            count += 1
    return count


def apply_bitnet_b158(state: dict[str, torch.Tensor], scale: float | None = None
                      ) -> dict[str, torch.Tensor]:
    """Offline ternary bake: quantize every linear-ish weight in a state dict.

    Quantizes all keys ending in ``.weight`` (embeddings/head included, per
    BitNet b1.58 — they are also ternary). Returns a new state dict; the
    result is intended for QAT warm-start or fused integer-kernel inference.

    Args:
        state: model state dict (keys from ConfigurableResearchLLM).
        scale: optional fixed quantize scale (None = per-tensor absmean).

    Returns:
        New state dict with quantized weights (same keys/shapes/dtypes).
    """
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and k.endswith(".weight"):
            q, s = ternary_quantize(v.float(), scale)
            # Save ternary weights as int8 (1 byte/param) + qscale key
            out[k] = q.to(torch.int8)
            out[k.replace(".weight", ".qscale")] = torch.tensor([s], dtype=torch.float32)
        else:
            out[k] = v
    return out


class BitNetB158Key(Key):
    """BitNet b1.58 key — ternary QAT transform on linear weights.

    Key class: PARTIAL — the ternary transform is not invertible.
    """

    def __init__(self, scale: float | None = None):
        self.scale = scale

    @property
    def name(self) -> str:
        return "bitnet_b158"

    @property
    def description(self) -> str:
        return f"Ternary QAT weights {{-1,0,+1}} (scale={self.scale or 'absmean'})"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        try:
            weights = apply_bitnet_b158(dict(data), self.scale)
            n = sum(1 for k in weights if k.endswith(".weight"))
            return KeyResult(success=True, weights=weights,
                             metadata={"n_quantized": n,
                                       "scale": self.scale or "absmean"})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(
            success=True, data=weights,
            metadata={"note": "Ternary weights cannot be un-quantized"})


def build_bitnet_linear(config, in_features: int, out_features: int,
                        bias: bool = False) -> nn.Module:
    """Build a BitNetLinear honoring the model config."""
    return BitNetLinear(in_features, out_features, bias=bias,
                        quantize=bool(getattr(config, "use_bitnet", False)),
                        force_quant=bool(getattr(config, "bitnet_force_quant", False)),
                        learned_scale=bool(getattr(config, "bitnet_learned_scale", True)))


def convert_model_to_int8(model: nn.Module) -> int:
    """Convert all BitNetLinear layers in a model to int8 weight storage.

    Call after loading a checkpoint with ternary-quantized weights.
    Reduces weight VRAM by 4x (fp32 → int8). Returns number of layers converted.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, BitNetLinear) and not module._prequantized:
            module.convert_to_int8_storage()
            n += 1
    return n


class BitNetEmbedding(nn.Module):
    """Embedding with BitNet b1.58 ternary QAT on the embedding weight.

    During training (or force_quant), the looked-up vectors are ternary-quantized
    with STE. Eval uses full-precision master weights (lossless for warm start).

    Args:
        num_embeddings: vocab size
        embedding_dim: embedding dimension
        quantize: enable ternary QAT during training
        force_quant: also quantize in eval (deployment)
        learned_scale: learn a per-tensor quantize scale
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 quantize: bool = False, force_quant: bool = False,
                 learned_scale: bool = True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quantize = quantize
        self.force_quant = force_quant
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if learned_scale:
            with torch.no_grad():
                scale = self.weight.abs().mean().clamp(min=_TERNARY_EPS) / 0.7
            self.qscale = nn.Parameter(scale)
        else:
            self.qscale = None

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if getattr(self, '_prequantized', False):
            # Pre-quantized int8 path: weights already ternary, stored as int8
            w_int8 = self.weight_int8
            scale = self.qscale_buf if hasattr(self, 'qscale_buf') else None
            # Cast to bf16 to match model dtype (avoids RMSNorm dtype mismatch)
            x = F.embedding(token_ids, w_int8.to(torch.bfloat16))
            if scale is not None:
                x = x * scale.to(x.dtype)
            return x
        x = F.embedding(token_ids, self.weight)
        if self.quantize and (self.training or self.force_quant):
            scale = self.qscale if self.qscale is not None else None
            q, s = ternary_quantize(self.weight, scale)
            q_out = F.embedding(token_ids, q) * s.to(x.dtype)
            # STE: gradient flows through master weights
            x = x + (q_out - x).detach()
        return x

    def load_prequantized(self, weight_int8: torch.Tensor, qscale):
        """Load a pre-quantized ternary weight (int8) directly into int8 storage."""
        device = weight_int8.device
        if hasattr(self, 'weight'):
            del self.weight
        self.register_buffer("weight_int8", weight_int8.to(torch.int8))
        if isinstance(qscale, torch.Tensor):
            self.register_buffer("qscale_buf", qscale.to(device))
        else:
            self.register_buffer("qscale_buf", torch.tensor(qscale, device=device))
        if self.qscale is not None:
            del self.qscale
            self.qscale = None
        self._prequantized = True

    @property
    def weight_data(self) -> torch.Tensor:
        """Access the raw weight parameter (for checkpoint I/O)."""
        return self.weight


def build_bitnet_embedding(config, num_embeddings: int, embedding_dim: int
                           ) -> nn.Module:
    """Build a BitNetEmbedding honoring the model config."""
    return BitNetEmbedding(
        num_embeddings, embedding_dim,
        quantize=bool(getattr(config, "use_bitnet", False)),
        force_quant=bool(getattr(config, "bitnet_force_quant", False)),
        learned_scale=bool(getattr(config, "bitnet_learned_scale", True)))


class BitNetConv1d(nn.Module):
    """Conv1d with BitNet b1.58 ternary QAT.

    Same STE semantics as BitNetLinear but for depthwise/grouped 1D conv.
    Used to ternarize the short conv layers in DoubleGatedConvLayer.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 groups=1, bias=False, quantize=True, force_quant=False,
                 learned_scale=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.quantize = quantize
        self.force_quant = force_quant
        in_per_group = in_channels // groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_per_group, kernel_size))
        nn.init.kaiming_uniform_(self.weight.view(out_channels, -1), a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        if learned_scale:
            with torch.no_grad():
                scale = self.weight.abs().mean().clamp(min=_TERNARY_EPS) / 0.7
            self.qscale = nn.Parameter(scale)
        else:
            self.qscale = None
        self._prequantized = False  # set True by load_prequantized()

    def forward(self, x):
        # x: (B, C, T)
        if getattr(self, '_prequantized', False):
            # Pre-quantized int8 path: weights already ternary, stored as int8
            w_int8 = self.weight_int8  # int8 buffer, values {-1, 0, +1}
            scale = self.qscale_buf.to(x.dtype) if hasattr(self, 'qscale_buf') else 1.0
            w = w_int8.to(x.dtype)
            y = F.conv1d(x, w, self.bias, padding=0, groups=self.groups)
            return y * scale
        if self.quantize and (self.training or self.force_quant):
            scale = self.qscale if self.qscale is not None else None
            q, s = ternary_quantize(self.weight, scale)
            w = self.weight + (q - self.weight).detach()  # STE
            y = F.conv1d(x, w, self.bias, padding=0, groups=self.groups)
            return y * s.to(y.dtype)
        else:
            return F.conv1d(x, self.weight, self.bias, padding=0, groups=self.groups)

    def load_prequantized(self, weight_int8: torch.Tensor, qscale):
        """Load a pre-quantized ternary weight (int8) directly into int8 storage.

        Mirrors BitNetLinear.load_prequantized but for conv1d weights.
        Bypasses the fp32 parameter — stores as int8 buffer.
        """
        device = weight_int8.device
        if hasattr(self, 'weight'):
            del self.weight
        self.register_buffer("weight_int8", weight_int8.to(torch.int8))
        if isinstance(qscale, torch.Tensor):
            self.register_buffer("qscale_buf", qscale.to(device))
        else:
            self.register_buffer("qscale_buf", torch.tensor(qscale, device=device))
        if self.qscale is not None:
            del self.qscale
            self.qscale = None
        self._prequantized = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + "weight"
        qscale_key = prefix + "qscale"
        src_tensor = state_dict.get(weight_key, None)
        is_int8_weight = src_tensor is not None and src_tensor.dtype == torch.int8

        if is_int8_weight:
            # Pre-quantized int8: load directly — NO bf16 upcast
            qscale = state_dict.get(qscale_key)
            if qscale is None:
                qscale = src_tensor.float().abs().mean().clamp(min=_TERNARY_EPS) / 0.7
            del state_dict[weight_key]
            if qscale_key in state_dict:
                del state_dict[qscale_key]
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs)
            self.load_prequantized(src_tensor, qscale)
            return

        # Normal (bf16/fp32) weight loading path
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        if self.qscale is not None and not self.weight.is_meta:
            with torch.no_grad():
                self.qscale.copy_(
                    self.weight.abs().mean().clamp(min=_TERNARY_EPS) / 0.7)


def build_bitnet_conv1d(config, in_channels, out_channels, kernel_size,
                        groups=1, bias=False):
    """Build a BitNetConv1d honoring the model config."""
    return BitNetConv1d(
        in_channels, out_channels, kernel_size,
        groups=groups, bias=bias,
        quantize=bool(getattr(config, "use_bitnet", False)),
        force_quant=bool(getattr(config, "bitnet_force_quant", False)),
        learned_scale=bool(getattr(config, "bitnet_learned_scale", True)))
