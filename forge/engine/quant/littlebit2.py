"""LittleBit / LittleBit-2 sub-1-bit quantization via latent factorization.

Port of SamsungLabs/LittleBit (NeurIPS 2025) + LittleBit-2 (ICML 2026):
each dense weight W (out, in) is replaced by a scale-binary-scale-binary-scale
factorization

    W_hat = diag(u1) @ U_q @ diag(u2 * v1) @ V_q @ diag(v2)

where U_q (out, s) and V_q (s, in) are binarized ±1 latent factors of a
truncated-SVD decomposition, and u1/u2/v1/v2 are lightweight learnable
scales. Effective bit-width:

    eff_bit = (s*(in+out+16) + 16*(in+out)) / (in*out)           (primary only)
    eff_bit = (2s*(in+out+16) + 32*(in+out)) / (in*out)          (with residual)

so ``split_dim`` s directly controls bits-per-weight (0.1–1.0 BPW regime).

LittleBit-2's contribution over the R45 SVDLiftBinary baseline:
  - Internal Latent Rotation + Joint-ITQ: a shared orthogonal rotation R of
    the latent dimension (free in FP: U V^T = (UR)(R^T V^T)) is optimized by
    alternating minimization (B = sign(XR), R <- Procrustes(B^T X)) over the
    concatenated factors [U; V^T]. SVD singular vectors are high-coherence
    ("spiky") — the worst geometry for sign quantization; Joint-ITQ aligns
    them with the binary hypercube, cutting distortion below the Gaussian
    limit (~0.36) with zero inference overhead.
  - Dual-SVID scale init: scales come from a rank-1 SVD of |U| and |V|,
    not the least-squares rank_s trick used in R45.
  - QAT-ready: U, V, and all scales are nn.Parameters; STEBinary /
    SmoothSign provide gradients through the sign(). R45 confirmed binary
    PTQ without QAT is catastrophic (PPL 250K+) — this module is built to
    train.
  - Optional residual factorization: a second LittleBit factor pair fits
    W - W_hat (doubles the factor bit cost, ~0.2+ BPW regime).

Unlike R45's int8-buffer design, latent factors stay as Parameters so QAT
can update them; ``pack_weights()`` produces the true 1-bit-packed int32
form for export (``_binarized`` inference mode).

Sources:
  - LittleBit:   Lee et al. NeurIPS 2025, arXiv:2506.13771
  - LittleBit-2: Lee & Kim, ICML 2026, arXiv:2603.00042
  - Reference:   github.com/SamsungLabs/LittleBit (--use_itq flag)
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.quant.protocol import QuantizedLinearMixin

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Quantization autograd functions (sign + straight-through estimators)
# ──────────────────────────────────────────────────────────────────────────

class _STEBinary(torch.autograd.Function):
    """sign(x) with clipped straight-through gradient (|x|<1 passes grad)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * ((x > -1) & (x < 1))


class _SmoothSign(torch.autograd.Function):
    """sign(x) with tanh-derivative gradient: d/dx tanh(alpha*x), alpha=100.

    Smoother than STE near the decision boundary — the LittleBit paper's
    default for QAT.
    """

    @staticmethod
    def forward(ctx, x, alpha: float = 100.0):
        ctx.alpha = alpha
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        return grad_output * alpha * (1 - torch.tanh(alpha * x) ** 2), None


STEBinary = _STEBinary.apply
SmoothSign = _SmoothSign.apply


# ──────────────────────────────────────────────────────────────────────────
# 1-bit packing (row-major int32 words, lsb-first; +1 -> bit 0, -1 -> bit 1)
# ──────────────────────────────────────────────────────────────────────────

def binary_packer(tensor: torch.Tensor) -> torch.Tensor:
    """Pack a {-1,+1} int8 matrix (rows, cols) into (rows, ceil(cols/32)) int32."""
    if tensor.dtype != torch.int8:
        raise TypeError("Input tensor must be int8.")
    n_rows, n_cols = tensor.shape
    words_per_row = (n_cols + 31) // 32
    pad = words_per_row * 32 - n_cols
    if pad:
        tensor = torch.cat(
            [tensor, torch.ones(n_rows, pad, dtype=tensor.dtype,
                                device=tensor.device)], dim=1)
    bits = ((1 - tensor) // 2).int().reshape(n_rows, words_per_row, 32)
    wts = (2 ** torch.arange(32, dtype=torch.int32, device=tensor.device))
    return (bits * wts).sum(dim=2, dtype=torch.int32).contiguous()


def binary_unpacker(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Unpack (rows, words) int32 back to the {-1,+1} int8 matrix of ``shape``."""
    n_rows, n_cols = shape
    words_per_row = (n_cols + 31) // 32
    if packed.shape != (n_rows, words_per_row):
        raise ValueError(
            f"Packed shape {tuple(packed.shape)} != {(n_rows, words_per_row)}")
    bit_idx = torch.arange(32, device=packed.device)
    bits = ((packed.unsqueeze(-1) >> bit_idx) & 1).to(torch.int8)
    unpacked = bits.reshape(n_rows, words_per_row * 32)[:, :n_cols]
    return (1 - 2 * unpacked).to(torch.int8)


# ──────────────────────────────────────────────────────────────────────────
# LittleBitLinear — the factorized sub-1-bit linear layer
# ──────────────────────────────────────────────────────────────────────────

_SKIP_NAMES = ("embed", "head", "lm_head", "output")
_SKIP_TYPES = (
    "LittleBitLinear", "SVDLiftBinaryLinear", "WaveletLiftLinear",
    "NanoQuantLinear", "W8A8Linear", "FP8Linear", "BitNetLinear",
    "INT4Linear", "QuantizedLinear", "FastINT8Linear", "ASFP4Linear",
    "ResidualFP4Linear", "NLRQLinear", "NVFP4Linear", "ACBQLinear",
    "HadamardLiftLinear", "AdaptiveBlockFP4Linear", "TernaryLiftLinear",
    "SchurABFP4Linear", "SparseResidualINT3Linear",
)


def _estimate_split_dim(in_f: int, out_f: int, eff_bit: float | None,
                        residual: bool) -> float | None:
    """Invert the eff_bit formula to get the latent dim for a target BPW."""
    if eff_bit is None or in_f * out_f == 0:
        return None
    base = in_f + out_f + 16
    if residual:
        return (in_f * out_f * eff_bit - 32 * (in_f + out_f)) / (2 * base)
    return (in_f * out_f * eff_bit - 16 * (in_f + out_f)) / base


def _finalize_split_dim(split_float: float | None, split_default: int,
                        min_split_dim: int) -> int:
    """Round down to a multiple of 8 (packing-friendly), clamp to minimum."""
    cand = split_float if split_float is not None else split_default
    cand = (int(cand) // 8) * 8 or min_split_dim
    return max(cand, min_split_dim)


def _compute_eff_bits(in_f: int, out_f: int, s: int, residual: bool) -> float:
    """Actual bits-per-weight for a given latent dim (binary + fp16 scales)."""
    if in_f * out_f == 0:
        return float("inf")
    base = in_f + out_f + 16
    num = s * 2 * base + 32 * (in_f + out_f) if residual else s * base + 16 * (in_f + out_f)
    return num / (in_f * out_f)


def _svd_lowrank_safe(X: torch.Tensor, q: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncated SVD with sanity verification.

    torch.svd_lowrank is a randomized range-finder; on this box's MKL build
    it intermittently returns non-finite OR finite-but-wrong (giant)
    factors, so its output can't be trusted blindly. For moderate matrices
    (min dim <= 1024) use deterministic linalg.svd outright — the speed
    difference is negligible at init time. For large matrices, try
    svd_lowrank but verify the result (finite + S <= ||X||_F, since
    sigma_max <= ||X||_F always) and fall back to full SVD otherwise.
    """
    X = torch.nan_to_num(X.float(), nan=0.0, posinf=0.0, neginf=0.0)
    q = min(q, min(X.shape))
    if min(X.shape) > 1024:
        try:
            U, S, V = torch.svd_lowrank(X, q=q)
            if (torch.isfinite(U).all() and torch.isfinite(S).all()
                    and torch.isfinite(V).all()
                    and S.numel() > 0 and S.max() <= X.norm() * 1.05):
                return U, S, V
        except Exception:
            pass
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    return U[:, :q], S[:q], Vt.t()[:, :q]


def _rank_one_decompose(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-1 SVD of X (m, n) -> (u (1, m), v (1, n)) with X ~= u^T v."""
    U, S, V = _svd_lowrank_safe(X, q=1)
    sqrt_s = S[:1].clamp(min=0).sqrt()
    return (U[:, :1] * sqrt_s).t().contiguous(), (sqrt_s * V.t()[:1, :]).contiguous()


def _joint_itq_rotation(U: torch.Tensor, V: torch.Tensor,
                        n_iter: int = 50) -> torch.Tensor:
    """LittleBit-2 Joint-ITQ: shared latent rotation aligning [U; V^T] with {-1,+1}^s.

    Optimizes min_R ||B - X R||_F s.t. R orthogonal, B = sign(XR), over the
    CONCATENATED factors X = [U; V^T] (one rotation for both so U@V is
    preserved). Alternating minimization; each step is an orthogonal
    Procrustes solve R = Vt^T U^T from SVD(B^T X).

    Returns R (s, s) float32. Apply as U <- U @ R, V <- R^T @ V.
    """
    X = torch.cat([U.float(), V.float().t()], dim=0)  # (out+in, s)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    s = X.shape[1]
    device = X.device
    R = torch.empty(s, s, device=device, dtype=torch.float32)
    torch.nn.init.orthogonal_(R)
    eye = torch.eye(s, device=device)
    for _ in range(n_iter):
        B = torch.sign(X @ R)
        B[B == 0] = 1
        try:
            Up, _, Vtp = torch.linalg.svd(B.t() @ X, full_matrices=False)
        except Exception:
            break
        R_new = Vtp.t() @ Up.t()
        # A degenerate SVD draw can return non-finite/non-orthogonal
        # factors — a non-orthogonal R would blow up U@R magnitudes
        # downstream, so keep the last good rotation and stop.
        if not torch.isfinite(R_new).all():
            break
        if (R_new.t() @ R_new - eye).abs().max() > 1e-2:
            break
        R = R_new
    return R


class LittleBitLinear(QuantizedLinearMixin):
    """LittleBit-2 factorized sub-1-bit linear (Scale-Binary-Scale-Binary-Scale).

    Parameters (all learnable for QAT):
        U  (out, s): left latent factor, binarized through quant_func
        V  (s, in):  right latent factor, binarized through quant_func
        u1 (1, out), u2 (1, s): scales for the U side
        v1 (1, s), v2 (1, in): scales for the V side
        (optional _R twins for the residual factorization)

    Effective weight:  W_hat = diag(u1) U_q diag(u2*v1) V_q diag(v2)

    Forward avoids materializing W_hat — two skinny matmuls:
        y = ((((x * v2) @ V_q^T) * (v1*u2)) @ U_q^T) * u1
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 split_dim: int = 1024, residual: bool = False,
                 quant_func=SmoothSign):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.split_dim = split_dim
        self.residual = residual
        self.quant_func = quant_func
        self._binarized = False
        self.lora_adapter: nn.Module | None = None
        # from_linear overrides these so merge_lora() can re-factorize
        # with the same settings the layer was built with.
        self.use_itq = True
        self.itq_n_iter = 50

        self.U = nn.Parameter(torch.empty(out_features, split_dim))
        self.V = nn.Parameter(torch.empty(split_dim, in_features))
        self.u1 = nn.Parameter(torch.empty(1, out_features))
        self.u2 = nn.Parameter(torch.empty(1, split_dim))
        self.v1 = nn.Parameter(torch.empty(1, split_dim))
        self.v2 = nn.Parameter(torch.empty(1, in_features))

        if residual:
            self.U_R = nn.Parameter(torch.empty(out_features, split_dim))
            self.V_R = nn.Parameter(torch.empty(split_dim, in_features))
            self.u1_R = nn.Parameter(torch.empty(1, out_features))
            self.u2_R = nn.Parameter(torch.empty(1, split_dim))
            self.v1_R = nn.Parameter(torch.empty(1, split_dim))
            self.v2_R = nn.Parameter(torch.empty(1, in_features))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    # ── initialization ──────────────────────────────────────────────────

    @classmethod
    def from_linear(cls, lin: nn.Linear, split_dim: int | None = None,
                    eff_bit: float | None = None, residual: bool = False,
                    use_itq: bool = True, itq_n_iter: int = 50,
                    min_split_dim: int = 8, ratio_factor: float = 1.0,
                    quant_func=SmoothSign) -> "LittleBitLinear":
        """Dual-SVID (+ optional Joint-ITQ) init from a dense nn.Linear.

        Args:
            split_dim: latent rank s. If ``eff_bit`` is given, derived from it
                instead (rounded down to a multiple of 8).
            eff_bit: target bits-per-weight (0.1–1.0 regime).
            residual: factorize W - W_hat with a second LittleBit pair.
            use_itq: LittleBit-2 Joint-ITQ latent rotation (default ON —
                this is the LittleBit-2 improvement; False = LittleBit-1).
            itq_n_iter: alternating-minimization steps (~50 converges).
            ratio_factor: per-layer multiplier on the eff_bit-derived dim
                (e.g. smaller factors for k_proj/v_proj).
        """
        out_f, in_f = lin.weight.shape
        est = _estimate_split_dim(in_f, out_f, eff_bit, residual)
        if est is not None:
            est *= ratio_factor
        s = _finalize_split_dim(est, split_dim or 1024, min_split_dim)

        layer = cls(in_f, out_f, bias=lin.bias is not None, split_dim=s,
                    residual=residual, quant_func=quant_func)
        layer.use_itq = use_itq
        layer.itq_n_iter = itq_n_iter
        W = lin.weight.data.float()
        device = W.device

        U, V, scales = layer._dual_svid(W, s, use_itq=use_itq,
                                      itq_n_iter=itq_n_iter)
        layer._set_factors(U, V, scales)

        if residual:
            # W_approx must use the PRIMARY factors only — the residual
            # params are still uninitialized memory at this point.
            with torch.no_grad():
                Uq = layer._quantize(layer.U)
                Vq = layer._quantize(layer.V)
                W_approx = (layer.u1.t() * Uq * (layer.u2 * layer.v1)) @ \
                    (Vq * layer.v2)
            U_R, V_R, scales_R = layer._dual_svid(
                W - W_approx, s, use_itq=use_itq, itq_n_iter=itq_n_iter,
                suffix="_R")
            layer._set_factors(U_R, V_R, scales_R, suffix="_R")

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)
        return layer

    def _dual_svid(self, W: torch.Tensor, s: int, use_itq: bool,
                   itq_n_iter: int, suffix: str = ""):
        """Dual-SVID: truncated SVD -> optional Joint-ITQ -> rank-1 scale split.

        Returns (U, V, (u1, u2, v1, v2)) — U (out, s), V (s, in) with the
        singular values split symmetrically (sqrt(S) on both sides).
        """
        U_t, S_t, V_t = _svd_lowrank_safe(W, q=min(s, min(W.shape)))
        s_eff = min(s, S_t.numel())
        if s_eff < s:  # degenerate: pad with zeros so shapes stay consistent
            pad = s - s_eff
            U_t = F.pad(U_t, (0, pad))
            V_t = F.pad(V_t, (0, pad))
            S_t = F.pad(S_t, (0, pad))
        sqrt_S = S_t.clamp(min=0).sqrt()
        U = U_t @ torch.diag(sqrt_S)          # (out, s)
        V = torch.diag(sqrt_S) @ V_t.t()      # (s, in)

        if use_itq:
            R = _joint_itq_rotation(U, V, n_iter=itq_n_iter)
            # Magnitude sanity: an orthogonal R preserves norms, so any
            # large inflation means the rotation went bad — revert it.
            if (U @ R).abs().max() <= 100 * U.abs().max():
                U = U @ R          # latent rotation: lossless in FP
                V = R.t() @ V      # (UR)(R^T V) = UV
            else:
                logger.warning("Joint-ITQ produced an invalid rotation; "
                               "falling back to plain Dual-SVID")

        u1, u2 = _rank_one_decompose(U.abs())
        v1, v2 = _rank_one_decompose(V.abs())
        return U, V, (u1, u2, v1, v2)

    def _set_factors(self, U, V, scales, suffix: str = ""):
        u1, u2, v1, v2 = scales
        for name, val in (("U", U), ("V", V), ("u1", u1), ("u2", u2),
                          ("v1", v1), ("v2", v2)):
            # Guarantee finite params — a rare non-finite SVD/ITQ draw would
            # otherwise poison W_hat forever; a zeroed latent entry is
            # recoverable by QAT (sign(0) -> +1).
            # copy_ (not .data=) preserves device + CUDA-graph storage.
            getattr(self, name + suffix).data.copy_(torch.nan_to_num(
                val.contiguous(), nan=0.0, posinf=0.0, neginf=0.0))

    # ── inference ───────────────────────────────────────────────────────

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        return x if self._binarized else self.quant_func(x)

    def _factor_forward(self, x, V, U, v2, v1, u2, u1):
        dt = x.dtype
        Vq = self._quantize(V.to(dt))
        Uq = self._quantize(U.to(dt))
        # ((((x * v2) @ Vq^T) * (v1*u2)) @ Uq^T) * u1
        # Scales are fp32 Parameters — cast every operand to x.dtype so the
        # matmul dtype never silently promotes (bf16 x * fp32 v2 -> fp32,
        # then fp32 @ bf16 crashes).
        return ((((x * v2.to(dt)) @ Vq.t()) * (v1 * u2).to(dt))
                @ Uq.t()) * u1.to(dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        xf = x.reshape(-1, self.in_features)
        y = self._factor_forward(xf, self.V, self.U, self.v2, self.v1,
                                 self.u2, self.u1)
        if self.residual:
            y = y + self._factor_forward(xf, self.V_R, self.U_R, self.v2_R,
                                         self.v1_R, self.u2_R, self.u1_R)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        if self.lora_adapter is not None:
            y = y + self.lora_adapter(
                xf.to(self.lora_adapter.lora_A.dtype)).to(y.dtype)
        return y.reshape(*lead, self.out_features)

    @torch.no_grad()
    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Materialize W_hat (out, in) — eval/debug path, not the fast forward."""
        Uq = self._quantize(self.U).to(dtype)
        Vq = self._quantize(self.V).to(dtype)
        W = (self.u1.to(dtype).t() * Uq * (self.u2 * self.v1).to(dtype)) @ \
            (Vq * self.v2.to(dtype))
        if self.residual:
            Uq_R = self._quantize(self.U_R).to(dtype)
            Vq_R = self._quantize(self.V_R).to(dtype)
            W = W + (self.u1_R.to(dtype).t() * Uq_R *
                     (self.u2_R * self.v1_R).to(dtype)) @ \
                (Vq_R * self.v2_R.to(dtype))
        return W

    # ── QLoRA merge ───────────────────────────────────────────────────

    @torch.no_grad()
    def _refactorize(self, W: torch.Tensor) -> None:
        """Re-factorize a (possibly merged) dense weight via Dual-SVID + ITQ."""
        self._binarized = False   # merge produces fp factors again
        U, V, scales = self._dual_svid(W, self.split_dim,
                                     use_itq=self.use_itq,
                                     itq_n_iter=self.itq_n_iter)
        self._set_factors(U, V, scales)
        if self.residual:
            Uq = self._quantize(self.U)
            Vq = self._quantize(self.V)
            W_approx = (self.u1.t() * Uq * (self.u2 * self.v1)) @ \
                (Vq * self.v2)
            U_R, V_R, scales_R = self._dual_svid(
                W - W_approx, self.split_dim, use_itq=self.use_itq,
                itq_n_iter=self.itq_n_iter, suffix="_R")
            self._set_factors(U_R, V_R, scales_R, suffix="_R")

    @torch.no_grad()
    def merge_lora(self) -> bool:
        """QLoRA merge: W_hat += lora delta, then re-factorize.

        Unlike NF4 (re-quantize is near-lossless), re-factorizing the
        merged weight re-runs SVD+ITQ — the merge is lossy at <1 BPW.
        For repeated epoch promotion, prefer swapping adapters over
        merging into the factorized base.
        """
        adapter = self.lora_adapter
        if adapter is None:
            return False
        W = self._dequantize_weight(torch.float32)
        delta = adapter.scale * (adapter.lora_B @ adapter.lora_A)
        self._refactorize(W + delta.float())
        self.lora_adapter = None
        return True

    # ── bit accounting ──────────────────────────────────────────────────

    @property
    def eff_bit_actual(self) -> float:
        return _compute_eff_bits(self.in_features, self.out_features,
                                 self.split_dim, self.residual)

    @property
    def total_bit_usage(self) -> float:
        return self.eff_bit_actual * self.in_features * self.out_features

    # ── packing / frozen inference ──────────────────────────────────────

    @torch.no_grad()
    def pack_weights(self) -> dict[str, torch.Tensor]:
        """1-bit-pack the binarized latent factors for export/storage."""
        packed = {}
        names = ["U", "V"] + (["U_R", "V_R"] if self.residual else [])
        for name in names:
            p = getattr(self, name)
            p_bin = self._quantize(p.data).to(torch.int8)
            packed[f"{name}_packed"] = binary_packer(p_bin)
            packed[f"{name}_shape"] = torch.tensor(list(p.shape), dtype=torch.long)
        return packed

    @torch.no_grad()
    def freeze_binarized(self, dtype: torch.dtype | None = None) -> None:
        """Snap latent factors to ±1 int8 and skip the quant func in forward.

        Post-QAT inference mode: stores factors as int8 Parameters (no grad)
        so ``_quantize`` is a no-op cast. Pass ``dtype`` (e.g. bf16) to keep
        the snapped factors in the model's compute dtype — avoids a
        per-forward int8->bf16 materialization of every (s,in) factor.
        """
        for name in ["U", "V"] + (["U_R", "V_R"] if self.residual else []):
            p = getattr(self, name)
            p.requires_grad_(False)
            p.data = self.quant_func(p.data).to(
                dtype if dtype is not None else torch.int8)
        self._binarized = True

    def __repr__(self):
        return (f"LittleBitLinear(in={self.in_features}, "
                f"out={self.out_features}, s={self.split_dim}, "
                f"eff_bpw={self.eff_bit_actual:.3f}, residual={self.residual}, "
                f"binarized={self._binarized})")


# ──────────────────────────────────────────────────────────────────────────
# Model conversion
# ──────────────────────────────────────────────────────────────────────────

def quantize_model_littlebit2(model: nn.Module, split_dim: int | None = None,
                              eff_bit: float | None = 1.0,
                              residual: bool = False, use_itq: bool = True,
                              itq_n_iter: int = 50, min_split_dim: int = 8,
                              kv_ratio: float | None = None,
                              quant_func=SmoothSign,
                              verbose: bool = True) -> int:
    """Replace nn.Linear layers with LittleBitLinear (Dual-SVID + Joint-ITQ).

    Args:
        split_dim: fixed latent rank (used when ``eff_bit`` is None).
        eff_bit: target bits-per-weight; latent dim is derived per-layer.
        residual: add the residual factor pair (~doubles factor bits).
        use_itq: LittleBit-2 Joint-ITQ rotation (False = LittleBit-1 SVD init).
        kv_ratio: optional ratio_factor for k_proj/v_proj layers.
        quant_func: STE for QAT (SmoothSign default, STEBinary optional).

    Returns the number of layers converted.
    """
    n = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or type(module).__name__ in _SKIP_TYPES:
            continue
        if any(s in name for s in _SKIP_NAMES):
            continue
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        try:
            ratio = kv_ratio if (kv_ratio and ("k_proj" in name or "v_proj" in name)) else 1.0
            setattr(parent, parts[-1], LittleBitLinear.from_linear(
                module, split_dim=split_dim, eff_bit=eff_bit,
                residual=residual, use_itq=use_itq, itq_n_iter=itq_n_iter,
                min_split_dim=min_split_dim, ratio_factor=ratio,
                quant_func=quant_func))
            n += 1
        except Exception as e:
            if verbose:
                logger.warning(f"  [LittleBit2] Skipped {name}: {e}")
    if verbose and n > 0:
        logger.info(f"  [LittleBit2] {n} layers quantized "
                    f"(eff_bit={eff_bit}, residual={residual}, itq={use_itq})")
    return n


@torch.no_grad()
def refactorize_from_state_dict(model: nn.Module,
                                state: dict[str, torch.Tensor]) -> int:
    """Re-factorize LittleBitLinear modules from a DENSE fp state dict.

    Weight-swap paths use ``param.data.copy_(state[name])`` — under LB2
    there is no ``<mod>.weight`` parameter (params are U/V/scales), so the
    copy silently skips every factorized layer and the engine keeps
    generating with stale weights. Call this after the copy pass whenever
    the engine may be LB2-quantized.

    Returns the number of re-factorized layers.
    """
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LittleBitLinear):
            continue
        w = state.get(f"{name}.weight")
        if w is None:
            continue
        was_frozen = mod._binarized
        frozen_dtype = mod.U.dtype
        mod._refactorize(w.float())
        if was_frozen:
            mod.freeze_binarized(dtype=frozen_dtype)
        n += 1
    return n


@torch.no_grad()
def densify_model_littlebit2(model: nn.Module) -> int:
    """Replace every LittleBitLinear with an nn.Linear carrying W_hat.

    Keeps trained checkpoints in the standard dense format — the model
    builder/loader has no LittleBit key layout. Call AFTER
    ``merge_lora_adapters`` so the saved weights include merged deltas.
    Engines re-factorize at activation, so the fp checkpoint is
    losslessly interchangeable for the self-play loop.
    """
    n = 0
    for module in list(model.modules()):
        for cname, child in list(module.named_children()):
            if not isinstance(child, LittleBitLinear):
                continue
            W = child._dequantize_weight(torch.bfloat16)
            lin = nn.Linear(child.in_features, child.out_features,
                            bias=child.bias is not None,
                            device=W.device, dtype=W.dtype)
            lin.weight.data.copy_(W)
            if child.bias is not None:
                lin.bias.data.copy_(child.bias)
            setattr(module, cname, lin)
            n += 1
    return n


# ── Factorized-state sidecar (conversion cache) ─────────────────────────

@torch.no_grad()
def save_littlebit2_state(model: nn.Module, path: str) -> int:
    """Serialize factorized params + per-layer spec (safetensors + JSON spec).

    Conversion costs ~7min of SVD+ITQ on a 3B — the sidecar makes every
    later activation of the SAME checkpoint a tensor copy instead.
    """
    import json
    from safetensors.torch import save_file
    spec = {}
    tensors = {}
    for name, m in model.named_modules():
        if isinstance(m, LittleBitLinear):
            spec[name] = [m.split_dim, m.residual]
            for pn, p in m.named_parameters():
                tensors[f"{name}.{pn}"] = p.detach().cpu().contiguous()
    if not tensors:
        return 0
    save_file(tensors, path, metadata={"lb2_spec": json.dumps(spec)})
    return len(spec)


@torch.no_grad()
def load_littlebit2_state(model: nn.Module, path: str) -> int:
    """Rebuild LittleBitLinear shells from a sidecar and load factors.

    Replaces the nn.Linear at each spec'd path with a LittleBitLinear of
    the recorded split_dim/residual, then copies the stored factor params.
    """
    import json
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, framework="pt") as f:
        spec = json.loads(f.metadata()["lb2_spec"])
    tensors = load_file(path)
    n = 0
    for name, (s, residual) in spec.items():
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        lin = getattr(parent, parts[-1])
        if isinstance(lin, LittleBitLinear):
            continue
        layer = LittleBitLinear(lin.in_features, lin.out_features,
                                bias=lin.bias is not None,
                                split_dim=int(s), residual=bool(residual))
        for pn, p in layer.named_parameters():
            key = f"{name}.{pn}"
            if key in tensors:
                p.data = tensors[key].clone()
        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone()
        setattr(parent, parts[-1], layer)
        n += 1
    return n


def estimate_littlebit2_memory(model: nn.Module) -> dict:
    """Memory estimate: packed binary factors (1 bit) + fp16 scales."""
    total_bits = 0.0
    breakdown = {}
    for name, module in model.named_modules():
        if isinstance(module, LittleBitLinear):
            bits = module.total_bit_usage
            total_bits += bits
            breakdown[name] = {
                "bits": bits, "eff_bpw": module.eff_bit_actual,
                "split_dim": module.split_dim, "residual": module.residual,
            }
        elif isinstance(module, nn.Linear):
            w = module.weight
            total_bits += w.numel() * w.element_size() * 8
    return {
        "total_bits": total_bits,
        "total_bytes": total_bits / 8,
        "total_mb": total_bits / 8 / 1024 / 1024,
        "breakdown": breakdown,
    }
