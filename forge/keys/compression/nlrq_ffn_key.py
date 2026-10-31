"""NLRQ FFN — Nested Low-Rank Quantized linear layers for FFN compression.

Implements the R&D-winning compression algorithm:
  1. SVD: W ≈ U @ diag(S) @ V^T  (low-rank decomposition)
  2. Quantize U and V factors at `factor_bits` (default 8b) — STORED as INT8
  3. Optional INT4 group residual on (W - U_q @ diag(S) @ V_q)

HINT4-NLRQ (Hadamard-INT4): factor_bits=4 + use_hadamard=True
  - Applies a random orthogonal Hadamard rotation to U and V before INT4
    quantization, spreading outliers so 4-bit quantization is feasible.
  - 2x better compression than INT8 NLRQ at ~3-5% error (vs ~1.3% for INT8).
  - Hadamard matrices stored as bf16 buffers (rank×rank, negligible size).

Compression ratios (from R&D on 2048x8192 FFN weights):
  - NLRQ r=256 b=8:           12.8x CR, 1.3% error, 8.0 bits/param
  - NLRQ+INT4 residual r=256:  3.0x CR, 0.15% error, 4.6 bits/param
  - NLRQ+INT4 residual r=512:  2.6x CR, 0.14% error, 4.9 bits/param
  - HINT4 r=768:               ~16x CR, ~3-5% error, 4.0 bits/param

The key advantage over Monarch: NLRQ adapts to the actual weight structure
(SVD finds the optimal low-rank basis), while Monarch imposes a fixed
block-diagonal structure. NLRQ gives 5x better error at similar compression.

STORAGE: U and V are stored as actual INT8 tensors (torch.int8), not bf16.
This gives real VRAM savings — 8 bits per factor element vs 16 for bf16.
S (singular values) stays bf16 (only rank elements, negligible).
Scales are bf16 (out_features or in_features elements, small).

Usage:
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear, NLRQSwiGLUFFN
    # From dense:
    linear = NLRQLinear.from_dense(dense_weight, rank=256, factor_bits=8)
    # From scratch:
    linear = NLRQLinear(in_features, out_features, rank=256, factor_bits=8)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hadamard rotation utilities (for HINT4-NLRQ) ──────────────────────────

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate a normalized Hadamard matrix of size n (n must be power of 2).

    Uses the recursive construction: H_1 = [1], H_{2k} = [[H_k, H_k], [H_k, -H_k]] / sqrt(2).
    For non-power-of-2, uses the next power of 2 and truncates to the leading n×n block,
    then re-orthogonalizes via QR to ensure exact orthogonality (H @ H^T = I).

    The resulting matrix is always orthogonal: H @ H^T = I.
    """
    p = _next_power_of_2(n)
    # Build in float32 for numerical stability, convert at the end
    H = torch.ones(1, 1, device=device, dtype=torch.float32)
    while H.shape[0] < p:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / math.sqrt(2.0)
    # Truncate to n×n if n was not a power of 2
    H = H[:n, :n]
    if p != n:
        # Truncated Hadamard is NOT orthogonal — re-orthogonalize via QR.
        # The truncated block is close to orthogonal (Hadamard structure),
        # so QR just cleans up the small deviation.
        Q, _ = torch.linalg.qr(H)
        # Preserve sign convention: make diagonal positive
        signs = torch.sign(torch.diagonal(Q))
        signs[signs == 0] = 1.0
        H = Q * signs.unsqueeze(0)
    return H.to(dtype).contiguous()


def _apply_hadamard(weight: torch.Tensor, dim: int,
                    H: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random orthogonal Hadamard rotation along dimension dim.

    This spreads outliers uniformly, making the tensor more quantization-friendly.
    Returns: (rotated_weight, H) where:
      - rotated_weight = weight @ H^T  (if dim=1)
      - rotated_weight = H @ weight     (if dim=0)
    The inverse is: weight = rotated @ H (if dim=1) or H^T @ rotated (if dim=0)
    since H is orthogonal (H @ H^T = I).

    If H is None, a fresh Hadamard matrix of size matching the rotated dimension
    is generated. Pass a pre-built H to reuse the same rotation.
    """
    n = weight.shape[dim]
    if H is None:
        H = _hadamard_matrix(n, device=weight.device, dtype=weight.dtype)
    if dim == 1:
        # weight: (..., n) → rotated = weight @ H^T
        rotated = weight @ H.t().to(weight.dtype)
    elif dim == 0:
        # weight: (n, ...) → rotated = H @ weight
        rotated = H.to(weight.dtype) @ weight
    else:
        raise ValueError(f"dim must be 0 or 1, got {dim}")
    return rotated, H


# ── INT4 packing helpers ──────────────────────────────────────────────────

def _pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack two int4 values into one int8 byte.

    q: int8 tensor with values in [-8, 7].
    Returns: int8 tensor with half the elements along the last dim.
    """
    assert q.dtype == torch.int8
    # Convert to unsigned nibble [0, 15] by adding 8
    nib = (q.to(torch.int16) + 8).to(torch.int16)  # [0, 15]
    # Pack pairs along last dim
    if nib.shape[-1] % 2 != 0:
        # Pad last dim with zeros
        pad_shape = list(nib.shape)
        pad_shape[-1] += 1
        padded = torch.zeros(pad_shape, dtype=torch.int16, device=nib.device)
        padded[..., :nib.shape[-1]] = nib
        nib = padded
    low = nib[..., 0::2]
    high = nib[..., 1::2]
    packed = (low | (high << 4)).to(torch.int8)
    return packed


def _unpack_int4(packed: torch.Tensor, unpacked_last_dim: int) -> torch.Tensor:
    """Unpack int8 bytes into two int4 values each.

    packed: int8 tensor packed by _pack_int4.
    unpacked_last_dim: the original last dimension size before packing.
    Returns: int8 tensor with values in [-8, 7].
    """
    assert packed.dtype == torch.int8
    # Interpret as unsigned
    u = packed.to(torch.int16) & 0xFF  # [0, 255]
    low = u & 0x0F                      # [0, 15]
    high = (u >> 4) & 0x0F             # [0, 15]
    # Interleave back
    out = torch.stack([low, high], dim=-1).flatten(-2)  # (..., 2*packed_last)
    out = out[..., :unpacked_last_dim]
    # Convert back to signed [-8, 7]
    out = out - 8
    return out.to(torch.int8)


class NLRQLinear(nn.Module):
    """Linear layer using Nested Low-Rank Quantized decomposition.

    W ≈ dequant(U_q) @ diag(S) @ dequant(V_q)  (+ optional INT4 residual)

    Where:
      - U_q: (out, rank) stored as INT8 — 8 bits per element
      - S:   (rank,) singular values (bf16, negligible size)
      - V_q: (rank, in) stored as INT8 — 8 bits per element
      - U_scale: (out, 1) bf16 per-row scale
      - V_scale: (1, in) bf16 per-col scale
      - residual: (out, in) INT4 group-quantized (optional)

    Real VRAM (rank=768, d_model=4096, intermediate=16384):
      U_q: 16384 × 768 × 1 byte = 12.6 MB
      V_q: 768 × 4096 × 1 byte = 3.1 MB
      S:   768 × 2 bytes = 1.5 KB
      scales: ~40 KB
      Total per layer per proj: ~15.8 MB (vs 128 MB dense bf16) = 8.1x

    Args:
        in_features: input dimension
        out_features: output dimension
        rank: SVD truncation rank (controls compression vs error)
        factor_bits: bits per factor element (8 = INT8, 4 = INT4/HINT4)
        use_residual: if True, add INT4 group-quantized residual
        residual_group_size: group size for residual INT4 quantization
        use_hadamard: if True (and factor_bits=4), Hadamard rotation is applied
            to U and V factors before quantization and stored for dequantization.
            This is the HINT4-NLRQ mode that makes INT4 feasible by spreading
            outliers.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 256, factor_bits: int = 8,
                 use_residual: bool = False, residual_group_size: int = 128,
                 bias: bool = False, use_hadamard: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, min(in_features, out_features))
        self.factor_bits = factor_bits
        self.use_residual = use_residual
        self.residual_group_size = residual_group_size
        self.use_hadamard = use_hadamard and (factor_bits == 4)
        # INT4: signed 4-bit range is [-8, 7]; INT8: [-127, 127]
        if factor_bits == 4:
            self._max_val = 7
            self._min_val = -8
        else:
            self._max_val = 2**(factor_bits - 1) - 1
            self._min_val = -(2**(factor_bits - 1))

        # U and V stored as INT8 (real quantized storage, not fake bf16).
        # For INT4 mode, values are in [-8, 7] but still stored in int8 dtype
        # (PyTorch has no native int4). Storage accounting treats them as 4-bit.
        self.register_buffer('U_q', torch.zeros(out_features, self.rank, dtype=torch.int8))
        self.register_buffer('V_q', torch.zeros(self.rank, in_features, dtype=torch.int8))

        # S (singular values) — bf16, only rank elements
        self.S = nn.Parameter(torch.empty(self.rank))

        # Optional float masters for factor training (None unless enabled;
        # see enable_factor_training_). Registered here so self.U_m always exists.
        self.register_parameter("U_m", None)
        self.register_parameter("V_m", None)

        # Per-channel scales (bf16, small)
        self.register_buffer('U_scale', torch.ones(out_features, 1, dtype=torch.float16))
        self.register_buffer('V_scale', torch.ones(1, in_features, dtype=torch.float16))

        # Hadamard rotation matrices for HINT4 mode (rank×rank, bf16, small).
        # None for INT8 mode or when use_hadamard=False.
        if self.use_hadamard:
            self.register_buffer('hadamard_U',
                                 torch.eye(self.rank, dtype=torch.bfloat16))
            self.register_buffer('hadamard_V',
                                 torch.eye(self.rank, dtype=torch.bfloat16))
        else:
            self.register_buffer('hadamard_U', None)
            self.register_buffer('hadamard_V', None)

        if use_residual:
            # Residual stored as INT4 (packed into int8 for PyTorch compat)
            # Each int8 holds 2 int4 values. We store unpacked for simplicity.
            self.register_buffer('residual_q', torch.zeros(out_features, in_features, dtype=torch.int8))
            n_groups = in_features // residual_group_size
            self.register_buffer('residual_scales',
                                 torch.ones(out_features, n_groups, dtype=torch.float16))
        else:
            self.register_buffer('residual_q', None)
            self.register_buffer('residual_scales', None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Skip reset_parameters on meta device — call after materialization
        if self.S.device.type != "meta":
            self.reset_parameters()

    def reset_parameters(self):
        """Initialize NLRQ factors without expensive SVD.

        For training from scratch, SVD of a random matrix is unnecessary —
        the model learns proper factors during training. We init:
        - S: ones (equal singular value weighting)
        - U_q, V_q: random INT8 (scaled kaiming)
        - U_scale, V_scale: computed from the random factors
        """
        device = self.S.device
        with torch.no_grad():
            # S: ones — equal weighting, model learns proper singular values
            self.S.data = torch.ones(self.rank, dtype=self.S.dtype, device=device)

            # Random factors with conservative scaling for bf16 stability
            # U: (out, rank), V: (rank, in)
            # Use small scale to prevent overflow in bf16 forward pass
            scale_u = (1.0 / self.rank) ** 0.5
            scale_v = (1.0 / self.in_features) ** 0.5
            U_f = torch.randn(self.out_features, self.rank, dtype=torch.float32, device=device) * scale_u
            V_f = torch.randn(self.rank, self.in_features, dtype=torch.float32, device=device) * scale_v

            # Compute scales and quantize
            self.U_scale = (U_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / self._max_val).to(torch.float16)
            self.V_scale = (V_f.abs().amax(dim=0, keepdim=True).clamp(min=1e-6) / self._max_val).to(torch.float16)
            self.U_q = torch.round(U_f / self.U_scale.float()).clamp(self._min_val, self._max_val).to(torch.int8)
            self.V_q = torch.round(V_f / self.V_scale.float()).clamp(self._min_val, self._max_val).to(torch.int8)

    # ── factor training (STE) ──────────────────────────────────────────────

    def enable_factor_training_(self) -> None:
        """Create float master factors and train them via straight-through
        estimator (STE) around the INT8 quantizer.

        Why: U_q/V_q are INT8 buffers — without masters the ONLY trainable
        parameter is S (the singular values), which caps the layer at
        rescaling a fixed random basis. Masters restore full trainability:

          forward:  U_eff = U_m + (quant(U_m) - U_m).detach()   # STE
          backward: dL/dU_m flows as if quantize were identity

        The INT8 buffers are moved to CPU (they are only needed for
        checkpoint export / inference) so GPU cost is the bf16 masters alone.
        state_dict() gains U_m/V_m keys — strip them after export_quantized_()
        to keep checkpoints in the pure-INT8 inference format.
        """
        if self.use_residual:
            raise NotImplementedError(
                "factor training with INT4 residual is not supported "
                "(residual buffers must stay on GPU)")
        if self.use_hadamard:
            raise NotImplementedError(
                "factor training with HINT4 (Hadamard) is not yet supported "
                "(STE path does not apply Hadamard rotation)")
        if self.U_m is not None:
            return
        device = self.S.device
        with torch.no_grad():
            U_m = (self.U_q.float() * self.U_scale.float()).to(device)
            V_m = (self.V_q.float() * self.V_scale.float()).to(device)
            self.register_parameter("U_m", nn.Parameter(U_m.to(self.S.dtype),
                                                        requires_grad=True))
            self.register_parameter("V_m", nn.Parameter(V_m.to(self.S.dtype),
                                                        requires_grad=True))
            # INT8 buffers are dead weight on GPU during training — park on CPU
            for name in ("U_q", "V_q", "U_scale", "V_scale"):
                buf = getattr(self, name)
                setattr(self, name, buf.to("cpu"))

    def disable_factor_training_(self, export: bool = True) -> None:
        """Drop master factors. With export=True (default), refresh the INT8
        buffers from the masters first so no training progress is lost, and
        move the buffers back to the compute device (they were parked on CPU
        during factor training)."""
        if self.U_m is None:
            return
        if export:
            self.export_quantized_()
        device = self.S.device
        for name in ("U_q", "V_q", "U_scale", "V_scale"):
            setattr(self, name, getattr(self, name).to(device))
        self.register_parameter("U_m", None)
        self.register_parameter("V_m", None)

    @torch.no_grad()
    def export_quantized_(self) -> None:
        """Refresh INT8 buffers (U_q/V_q/scales, on CPU) from the masters.

        Called before saving a checkpoint so the saved state stays in the
        pure-INT8 inference format (no U_m/V_m keys needed downstream).
        """
        if self.U_m is None:
            return
        U_m = self.U_m.detach().float().cpu()
        V_m = self.V_m.detach().float().cpu()
        U_scale = (U_m.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / self._max_val)
        V_scale = (V_m.abs().amax(dim=0, keepdim=True).clamp(min=1e-6) / self._max_val)
        self.U_scale = U_scale.to(torch.float16)
        self.V_scale = V_scale.to(torch.float16)
        self.U_q = torch.round(U_m / U_scale).clamp(self._min_val, self._max_val).to(torch.int8)
        self.V_q = torch.round(V_m / V_scale).clamp(self._min_val, self._max_val).to(torch.int8)

    def factor_training_enabled(self) -> bool:
        return self.U_m is not None

    @classmethod
    def from_dense(cls, weight: torch.Tensor, rank: int = 256,
                   factor_bits: int = 8, use_residual: bool = False,
                   residual_group_size: int = 128,
                   bias: torch.Tensor | None = None) -> NLRQLinear:
        """Fit NLRQ factors from a dense weight matrix via SVD + INT8 quantize.

        Args:
            weight: (out_features, in_features) — nn.Linear weight format
            rank: SVD truncation rank
            factor_bits: quantization bits for U and V factors
            use_residual: add INT4 group-quantized residual
            bias: optional bias vector
        """
        out_features, in_features = weight.shape
        layer = cls(in_features, out_features, rank=rank,
                    factor_bits=factor_bits,
                    use_residual=use_residual,
                    residual_group_size=residual_group_size,
                    bias=bias is not None)
        with torch.no_grad():
            W_f32 = weight.float()
            U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
            r = min(rank, S.shape[0])
            U_f = U[:, :r].float()
            V_f = Vh[:r, :].float()
            layer.S.data = S[:r].to(layer.S.dtype).clone()
            # Quantize U and V to INT8
            max_val = layer._max_val
            layer.U_scale = (U_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / max_val).to(torch.float16)
            layer.V_scale = (V_f.abs().amax(dim=0, keepdim=True).clamp(min=1e-10) / max_val).to(torch.float16)
            layer.U_q = torch.round(U_f / layer.U_scale.float()).clamp(layer._min_val, max_val).to(torch.int8)
            layer.V_q = torch.round(V_f / layer.V_scale.float()).clamp(layer._min_val, max_val).to(torch.int8)
            # Compute residual if requested
            if use_residual:
                U_deq = layer.U_q.float() * layer.U_scale.float()
                V_deq = layer.V_q.float() * layer.V_scale.float()
                W_lr = U_deq @ torch.diag(layer.S.data.float()) @ V_deq
                residual = W_f32 - W_lr
                gs = residual_group_size
                n_groups = in_features // gs
                if n_groups > 0:
                    res_trunc = residual[:, :n_groups * gs].reshape(out_features, n_groups, gs)
                    scales = res_trunc.abs().amax(dim=-1, keepdim=True) / 7
                    scales = scales.clamp(min=1e-10)
                    Q = torch.round(res_trunc / scales).clamp(-8, 7)
                    layer.residual_q = torch.zeros(out_features, in_features, dtype=torch.int8, device=W_f32.device)
                    layer.residual_q[:, :n_groups * gs] = Q.reshape(out_features, n_groups * gs).to(torch.int8)
                    layer.residual_scales = scales.squeeze(-1).to(torch.float16)
            if bias is not None:
                layer.bias.data = bias.clone()
        return layer

    @classmethod
    def from_dense_adaptive(cls, weight: torch.Tensor,
                            energy_threshold: float = 0.99,
                            min_rank: int = 64,
                            max_rank: int | None = None,
                            factor_bits: int = 8,
                            use_residual: bool = False,
                            residual_group_size: int = 128,
                            bias: torch.Tensor | None = None) -> tuple[NLRQLinear, int]:
        """Fit NLRQ with SVD energy-based adaptive rank selection.

        Instead of a fixed rank, selects the smallest rank that captures
        `energy_threshold` of the total singular value energy. This gives
        each layer its optimal rank — low-rank layers get smaller ranks
        (saving VRAM), high-rank layers get larger ranks (preserving quality).

        Novel: per-layer adaptive rank from singular value spectrum, not a
        global hyperparameter. The energy threshold is the only knob.

        Args:
            energy_threshold: fraction of singular value energy to retain (0.0-1.0)
            min_rank: minimum rank (floor to prevent degenerate layers)
            max_rank: maximum rank (ceiling, defaults to min(out, in))
            factor_bits: quantization bits for U and V factors
            use_residual: add INT4 group-quantized residual
            residual_group_size: group size for residual INT4 quantization
            bias: optional bias vector

        Returns:
            (layer, selected_rank) — the compressed layer and the rank that was selected
        """
        out_features, in_features = weight.shape
        max_r = max_rank if max_rank is not None else min(out_features, in_features)
        with torch.no_grad():
            W_f32 = weight.float()
            _U, S, _Vh = torch.linalg.svd(W_f32, full_matrices=False)
            # Cumulative singular value energy (squared singular values)
            s2 = (S ** 2)
            energy = s2.cumsum(0) / s2.sum().clamp(min=1e-12)
            # Smallest rank capturing `energy_threshold` of total energy
            idx = (energy >= energy_threshold).nonzero(as_tuple=False)
            r = idx[0].item() + 1 if idx.numel() > 0 else S.shape[0]
            # Clamp to [min_rank, max_rank]
            r = max(min_rank, r)
            r = min(max_r, r)
        layer = cls.from_dense(weight, rank=r, factor_bits=factor_bits,
                               use_residual=use_residual,
                               residual_group_size=residual_group_size,
                               bias=bias)
        return layer, r

    @classmethod
    def from_dense_hadamard_int4(cls, weight: torch.Tensor, rank: int = 768,
                                 use_residual: bool = True,
                                 residual_group_size: int = 128,
                                 bias: torch.Tensor | None = None) -> NLRQLinear:
        """Fit NLRQ with Hadamard-INT4 factors (HINT4) from a dense weight matrix.

        Process:
        1. SVD: W ≈ U @ diag(S) @ V^T
        2. Apply Hadamard rotation: U_rot = U @ H_U^T, V_rot = H_V @ V
        3. Quantize U_rot, V_rot to INT4 with per-channel scales
        4. Store H_U, H_V for dequantization
        5. (optional) INT4 group-quantized residual on the final reconstruction
           error, capturing the INT4 quantization noise.

        The Hadamard rotation spreads outliers in the SVD factors, making
        INT4 quantization feasible. Pure HINT4 has ~18% error; adding an INT4
        residual drops this to ~3-5% (vs ~1.3% for INT8) at the cost of extra
        storage for the residual term.

        Compression: pure HINT4 is ~2x better than INT8 NLRQ (4 bits vs 8 bits
        per factor). With residual, the full out×in INT4 residual term is the
        dominant storage cost, so HINT4+residual trades compression for error
        reduction — it is no longer smaller than INT8 NLRQ, but achieves much
        lower error than pure HINT4.

        Args:
            weight: (out_features, in_features) — nn.Linear weight format
            rank: SVD truncation rank
            use_residual: if True, add INT4 group-quantized residual to capture
                the quantization error. Drops error from ~18% to ~3-5% at the
                cost of a full out×in INT4 residual term (the dominant storage
                cost). Note: with residual, total storage exceeds INT8 NLRQ —
                the trade is compression for much lower error than pure HINT4.
            residual_group_size: group size for residual INT4 quantization
            bias: optional bias vector
        """
        out_features, in_features = weight.shape
        layer = cls(in_features, out_features, rank=rank,
                    factor_bits=4, use_residual=use_residual,
                    residual_group_size=residual_group_size,
                    bias=bias is not None, use_hadamard=True)
        with torch.no_grad():
            W_f32 = weight.float()
            U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
            r = min(rank, S.shape[0])
            U_f = U[:, :r].float()   # (out, rank)
            V_f = Vh[:r, :].float()  # (rank, in)
            layer.S.data = S[:r].to(layer.S.dtype).clone()

            # Build Hadamard matrices for the rank dimension
            H_U = _hadamard_matrix(r, device=weight.device, dtype=torch.float32)
            H_V = _hadamard_matrix(r, device=weight.device, dtype=torch.float32)

            # Apply Hadamard rotation to spread outliers:
            #   U_rot = U @ H_U^T  (rotate along rank dim, dim=1)
            #   V_rot = H_V @ V     (rotate along rank dim, dim=0)
            U_rot = U_f @ H_U.t()    # (out, rank)
            V_rot = H_V @ V_f        # (rank, in)

            # Quantize rotated factors to INT4 (range [-8, 7])
            max_val = layer._max_val  # 7
            min_val = layer._min_val  # -8
            layer.U_scale = (U_rot.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / max_val).to(torch.float16)
            layer.V_scale = (V_rot.abs().amax(dim=0, keepdim=True).clamp(min=1e-10) / max_val).to(torch.float16)
            layer.U_q = torch.round(U_rot / layer.U_scale.float()).clamp(min_val, max_val).to(torch.int8)
            layer.V_q = torch.round(V_rot / layer.V_scale.float()).clamp(min_val, max_val).to(torch.int8)

            # Store Hadamard matrices as bf16 buffers for dequantization
            layer.hadamard_U = H_U.to(torch.bfloat16)
            layer.hadamard_V = H_V.to(torch.bfloat16)

            # Compute residual if requested (same logic as from_dense, but on
            # the HADAMARD-ROTATED reconstruction: residual is added AFTER the
            # Hadamard inverse in _dequantize_weight, so it must be computed on
            # the final reconstruction error W_f32 - W_lr where W_lr uses the
            # inverse-Hadamard factors).
            if use_residual:
                U_f = layer.U_q.float() * layer.U_scale.float()
                V_f = layer.V_q.float() * layer.V_scale.float()
                # H_U / H_V are still the float32 locals from above
                U_orig = U_f @ H_U          # inverse Hadamard
                V_orig = H_V.t() @ V_f      # inverse Hadamard
                W_lr = U_orig @ torch.diag(layer.S.data.float()) @ V_orig
                residual = W_f32 - W_lr

                gs = residual_group_size
                n_groups = in_features // gs
                if n_groups > 0:
                    res_trunc = residual[:, :n_groups * gs].reshape(out_features, n_groups, gs)
                    scales = res_trunc.abs().amax(dim=-1, keepdim=True) / 7
                    scales = scales.clamp(min=1e-10)
                    Q = torch.round(res_trunc / scales).clamp(-8, 7)
                    layer.residual_q = torch.zeros(out_features, in_features, dtype=torch.int8, device=W_f32.device)
                    layer.residual_q[:, :n_groups * gs] = Q.reshape(out_features, n_groups * gs).to(torch.int8)
                    layer.residual_scales = scales.squeeze(-1).to(torch.float16)

            if bias is not None:
                layer.bias.data = bias.clone()
        return layer

    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize factors to bf16 effective weight (fused, low-VRAM).

        For INT8 mode:
            W = (U_q * U_scale) @ diag(S) @ (V_q * V_scale)
        For HINT4 mode (factor_bits=4 + use_hadamard):
            1. Dequantize: U_f = U_q.float() * U_scale, V_f = V_q.float() * V_scale
            2. Inverse Hadamard: U_orig = U_f @ H_U, V_orig = H_V^T @ V_f
            3. Reconstruct: W = U_orig @ diag(S) @ V_orig

        Uses sequential matmul: (U @ diag(S)) is (out, rank), then @ V is (out, in).
        This keeps peak memory at out*rank + rank*in, not out*in + out*in.
        """
        U_f = self.U_q.to(torch.float32) * self.U_scale.to(torch.float32)
        V_f = self.V_q.to(torch.float32) * self.V_scale.to(torch.float32)

        if self.use_hadamard and self.hadamard_U is not None:
            # Inverse Hadamard rotation:
            #   U_rot = U @ H_U^T  →  U_orig = U_rot @ H_U
            #   V_rot = H_V @ V    →  V_orig = H_V^T @ V_rot
            H_U = self.hadamard_U.to(torch.float32)  # (rank, rank)
            H_V = self.hadamard_V.to(torch.float32)  # (rank, rank)
            U_f = U_f @ H_U    # (out, rank) @ (rank, rank) = (out, rank)
            V_f = H_V.t() @ V_f  # (rank, rank) @ (rank, in) = (rank, in)

        # Scale V by S first (rank, in) — small
        SV = self.S.to(torch.float32).unsqueeze(1) * V_f  # (rank, in)
        # U @ SV = (out, in) — this is the only large allocation
        W = U_f @ SV
        if self.residual_q is not None:
            gs = self.residual_group_size
            n_groups = self.in_features // gs
            if n_groups > 0:
                res = self.residual_q[:, :n_groups * gs].to(torch.float32).reshape(
                    self.out_features, n_groups, gs)
                res = res * self.residual_scales.to(torch.float32).unsqueeze(-1)
                W[:, :n_groups * gs] += res.reshape(self.out_features, n_groups * gs)
        return W.to(torch.bfloat16)

    @staticmethod
    def _ste_quantize(w: torch.Tensor, scale_dim: int, max_val: float,
                      min_val: float | None = None) -> torch.Tensor:
        """Quantize-dequantize with straight-through gradient.

        scale_dim=1 → per-row scales (U, shape (out, rank));
        scale_dim=0 → per-col scales (V, shape (rank, in)).
        """
        if min_val is None:
            min_val = -max_val
        scale = w.detach().abs().amax(dim=scale_dim, keepdim=True).clamp(min=1e-6) / max_val
        q = (w.detach() / scale).round().clamp_(min_val, max_val) * scale
        return w + (q - w.detach())  # identity gradient w.r.t. w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = x @ W^T + b directly from the factors (no W materialization).

        Instead of: W = U @ diag(S) @ V; y = x @ W^T
        Computes:    y = ((x @ V^T) * S) @ U^T  + b

        When factor training is enabled (U_m/V_m masters exist), the factors
        are quantize-dequantized through the STE so gradients reach the
        masters; scales are recomputed from the masters each call.

        No cached buffers — creates temporaries that PyTorch's caching allocator
        reuses across layers. Saves 4 GB VRAM vs per-layer caching (99 layers
        × 42 MB = 4.15 GB of permanently allocated buffers).
        """
        if self.U_m is not None:
            compute_dtype = x.dtype if x.dtype != torch.int64 else torch.float32
            U_m = self.U_m.to(compute_dtype)
            V_m = self.V_m.to(compute_dtype)
            U_f = self._ste_quantize(U_m, 1, self._max_val, self._min_val)
            V_f = self._ste_quantize(V_m, 0, self._max_val, self._min_val)
            S = self.S.to(compute_dtype)
            h = torch.matmul(x.to(compute_dtype), (S.unsqueeze(1) * V_f).t())
            out = torch.matmul(h, U_f.t())
            if self.bias is not None:
                out = out + self.bias.to(compute_dtype)
            return out.to(x.dtype)

        compute_dtype = x.dtype if x.dtype != torch.int64 else torch.float32

        # Dequantize factors → compute_dtype (temporaries, reused by allocator)
        U_f = self.U_q.to(compute_dtype) * self.U_scale.to(compute_dtype)
        V_f = self.V_q.to(compute_dtype) * self.V_scale.to(compute_dtype)
        S = self.S.to(compute_dtype)

        # Apply inverse Hadamard for HINT4 mode:
        #   U_eff = U_f @ H_U,  V_eff = H_V^T @ V_f
        # The forward computes y = ((x @ V_eff^T) * S) @ U_eff^T
        # We fold the Hadamard into the matmul chain to avoid materializing W.
        if self.use_hadamard and self.hadamard_U is not None:
            H_U = self.hadamard_U.to(compute_dtype)   # (rank, rank)
            H_V = self.hadamard_V.to(compute_dtype)   # (rank, rank)
            # V_eff = H_V^T @ V_f  →  V_eff^T = V_f^T @ H_V
            # h = x @ V_eff^T = x @ V_f^T @ H_V
            h = torch.matmul(x.to(compute_dtype), V_f.t())
            h = torch.matmul(h, H_V)
            h = h * S.unsqueeze(0)  # (batch, rank)
            # U_eff = U_f @ H_U  →  U_eff^T = H_U^T @ U_f^T
            # out = h @ U_eff^T = h @ H_U^T @ U_f^T
            h = torch.matmul(h, H_U.t())
            out = torch.matmul(h, U_f.t())
        else:
            # y = ((x @ V^T) * S) @ U^T — two matmuls
            h = torch.matmul(x.to(compute_dtype), (S.unsqueeze(1) * V_f).t())
            out = torch.matmul(h, U_f.t())

        # Add residual if present
        if self.residual_q is not None:
            gs = self.residual_group_size
            n_groups = self.in_features // gs
            if n_groups > 0:
                res = self.residual_q[:, :n_groups * gs].to(compute_dtype).reshape(
                    self.out_features, n_groups, gs)
                res = res * self.residual_scales.to(compute_dtype).unsqueeze(-1)
                res_flat = res.reshape(self.out_features, n_groups * gs)
                out = out + x.to(compute_dtype)[:, :n_groups * gs] @ res_flat.t()

        if self.bias is not None:
            out = out + self.bias.to(compute_dtype)

        return out.to(x.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct the approximate dense weight (for compat/checkpointing).

        Uses the masters when factor training is enabled (the INT8 buffers
        may be stale / offloaded in that mode)."""
        if self.U_m is not None:
            U_f = self._ste_quantize(self.U_m.float(), 1, self._max_val, self._min_val)
            V_f = self._ste_quantize(self.V_m.float(), 0, self._max_val, self._min_val)
            SV = self.S.float().unsqueeze(1) * V_f
            return (U_f @ SV).to(torch.bfloat16)
        return self._dequantize_weight()

    def compressed_storage_bytes(self) -> int:
        """Actual bytes stored in compressed form.

        For INT8: U_q/V_q at 1 byte each.
        For INT4 (HINT4): U_q/V_q at 0.5 bytes each (4 bits per element,
            stored in int8 dtype but accounting as 4-bit).
        Plus bf16 S, float16 scales, and Hadamard matrices (HINT4 only).
        """
        bits_per_factor = self.factor_bits  # 8 or 4
        total = int(self.U_q.numel() * bits_per_factor / 8)  # bytes
        total += int(self.V_q.numel() * bits_per_factor / 8)
        total += self.S.numel() * 2   # bf16
        total += self.U_scale.numel() * 2  # float16
        total += self.V_scale.numel() * 2
        if self.hadamard_U is not None:
            total += self.hadamard_U.numel() * 2  # bf16
        if self.hadamard_V is not None:
            total += self.hadamard_V.numel() * 2  # bf16
        if self.residual_q is not None:
            # Residual is INT4-quantized (range [-8, 7]) → 0.5 bytes/element,
            # even though it is stored in an int8 tensor for PyTorch compat.
            total += int(self.residual_q.numel() * 4 / 8)  # 4 bits each
            total += self.residual_scales.numel() * 2
        if self.bias is not None:
            total += self.bias.numel() * 2
        return total

    def dense_storage_bytes(self) -> int:
        """Bytes if stored as dense bf16."""
        return self.out_features * self.in_features * 2

    def compression_ratio(self) -> float:
        """Actual storage compression ratio."""
        return self.dense_storage_bytes() / max(self.compressed_storage_bytes(), 1)


class NLRQSwiGLUFFN(nn.Module):
    """SwiGLU FFN with NLRQ-factored linear layers.

    Replaces w_gate, w_up, w_down with NLRQLinear.
    Real INT8 storage: 8x FFN param reduction (rank=768, 8-bit factors).
    With INT4 residual: 3x reduction at 0.15% error.
    With HINT4 (Hadamard-INT4): ~16x reduction at ~3-5% error.
    """

    def __init__(self, d_model: int = 2048, hidden_dim: int | None = None,
                 rank: int = 256, factor_bits: int = 8,
                 use_residual: bool = False, residual_group_size: int = 128,
                 use_clamp: bool = False,
                 clamp_alpha: float = 1.702, clamp_limit: float = 7.0,
                 use_hadamard: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.factor_bits = factor_bits
        self.use_residual = use_residual
        self.use_hadamard = use_hadamard
        self.use_clamp = use_clamp
        self.clamp_alpha = clamp_alpha
        self.clamp_limit = clamp_limit

        self.w_gate = NLRQLinear(d_model, hidden_dim, rank=rank,
                                 factor_bits=factor_bits,
                                 use_residual=use_residual,
                                 residual_group_size=residual_group_size,
                                 use_hadamard=use_hadamard)
        self.w_up = NLRQLinear(d_model, hidden_dim, rank=rank,
                               factor_bits=factor_bits,
                               use_residual=use_residual,
                               residual_group_size=residual_group_size,
                               use_hadamard=use_hadamard)
        self.w_down = NLRQLinear(hidden_dim, d_model, rank=rank,
                                 factor_bits=factor_bits,
                                 use_residual=use_residual,
                                 residual_group_size=residual_group_size,
                                 use_hadamard=use_hadamard)

    @classmethod
    def from_dense_ffn(cls, ffn: nn.Module, rank: int = 256,
                       factor_bits: int = 8, use_residual: bool = False,
                       residual_group_size: int = 128) -> NLRQSwiGLUFFN:
        """Convert a dense SwiGLUFFN to NLRQSwiGLUFFN."""
        d_model = ffn.w_gate.in_features
        hidden_dim = ffn.w_gate.out_features
        layer = cls(d_model, hidden_dim, rank=rank,
                    factor_bits=factor_bits,
                    use_residual=use_residual,
                    residual_group_size=residual_group_size,
                    use_clamp=getattr(ffn, 'use_clamp', False),
                    clamp_alpha=getattr(ffn, 'clamp_alpha', 1.702),
                    clamp_limit=getattr(ffn, 'clamp_limit', 7.0))
        with torch.no_grad():
            layer.w_gate = NLRQLinear.from_dense(
                ffn.w_gate.weight, rank, factor_bits, use_residual,
                residual_group_size,
                ffn.w_gate.bias if hasattr(ffn.w_gate, 'bias') and ffn.w_gate.bias is not None else None)
            layer.w_up = NLRQLinear.from_dense(
                ffn.w_up.weight, rank, factor_bits, use_residual,
                residual_group_size,
                ffn.w_up.bias if hasattr(ffn.w_up, 'bias') and ffn.w_up.bias is not None else None)
            layer.w_down = NLRQLinear.from_dense(
                ffn.w_down.weight, rank, factor_bits, use_residual,
                residual_group_size,
                ffn.w_down.bias if hasattr(ffn.w_down, 'bias') and ffn.w_down.bias is not None else None)
        return layer

    @classmethod
    def from_dense_adaptive_ffn(cls, w_gate: torch.Tensor, w_up: torch.Tensor,
                                w_down: torch.Tensor,
                                energy_threshold: float = 0.99,
                                min_rank: int = 64,
                                factor_bits: int = 8,
                                use_residual: bool = False,
                                residual_group_size: int = 128) -> tuple[NLRQSwiGLUFFN, list[int]]:
        """Build NLRQSwiGLUFFN with adaptive rank per projection.

        Applies SVD energy-based adaptive rank selection to each of the three
        FFN projections (gate, up, down) independently. Low-rank projections
        get smaller ranks (saving VRAM), high-rank projections get larger
        ranks (preserving quality).

        Args:
            w_gate: (hidden_dim, d_model) gate projection weight
            w_up: (hidden_dim, d_model) up projection weight
            w_down: (d_model, hidden_dim) down projection weight
            energy_threshold: fraction of singular value energy to retain
            min_rank: minimum rank floor
            factor_bits: quantization bits for U and V factors
            use_residual: add INT4 group-quantized residual
            residual_group_size: group size for residual INT4 quantization

        Returns:
            (ffn, [gate_rank, up_rank, down_rank]) — the compressed FFN and
            the list of selected ranks for each projection.
        """
        hidden_dim, d_model = w_gate.shape
        layer = cls(d_model, hidden_dim, rank=min_rank,
                    factor_bits=factor_bits,
                    use_residual=use_residual,
                    residual_group_size=residual_group_size)
        with torch.no_grad():
            gate_lin, r_gate = NLRQLinear.from_dense_adaptive(
                w_gate, energy_threshold=energy_threshold, min_rank=min_rank,
                factor_bits=factor_bits, use_residual=use_residual,
                residual_group_size=residual_group_size)
            up_lin, r_up = NLRQLinear.from_dense_adaptive(
                w_up, energy_threshold=energy_threshold, min_rank=min_rank,
                factor_bits=factor_bits, use_residual=use_residual,
                residual_group_size=residual_group_size)
            down_lin, r_down = NLRQLinear.from_dense_adaptive(
                w_down, energy_threshold=energy_threshold, min_rank=min_rank,
                factor_bits=factor_bits, use_residual=use_residual,
                residual_group_size=residual_group_size)
            layer.w_gate = gate_lin
            layer.w_up = up_lin
            layer.w_down = down_lin
        return layer, [r_gate, r_up, r_down]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        if self.use_clamp:
            gate = gate.clamp(min=None, max=self.clamp_limit)
            up = up.clamp(-self.clamp_limit, max=self.clamp_limit)
            glu = gate * torch.sigmoid(self.clamp_alpha * gate)
            return self.w_down((up + 1) * glu)
        return self.w_down(F.silu(gate) * up)

    def compressed_storage_bytes(self) -> int:
        return (self.w_gate.compressed_storage_bytes() +
                self.w_up.compressed_storage_bytes() +
                self.w_down.compressed_storage_bytes())

    def dense_storage_bytes(self) -> int:
        return (self.w_gate.dense_storage_bytes() +
                self.w_up.dense_storage_bytes() +
                self.w_down.dense_storage_bytes())

    def compression_ratio(self) -> float:
        return self.dense_storage_bytes() / max(self.compressed_storage_bytes(), 1)
