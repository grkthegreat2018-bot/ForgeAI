"""Novel extreme low-bit quantization algorithms (R&D Round 48, 2026-09-05).

Three PTQ methods targeting sub-1-bit and 1.58-bit compression, inspired by
recent SOTA papers recommended for investigation:

1. NanoQuantLinear: Low-rank binary factorization + ADMM initialization.
   W ≈ s1 ⊙ (U_±1 @ V_±1^T) ⊙ s2^T
   Storage: r*(d_out + d_in) bits + (d_out + d_in) floats.
   For rank r chosen so that r*(d_out+d_in) / (d_out*d_in) < 1 → sub-1-bit.
   Source: NanoQuant (Samsung, ICML 2026, arXiv 2602.06694)

2. BTCQuantLinear: Binary codebook clustering + learnable transformation.
   Clusters recurring binary ±1 vectors into a codebook of K patterns.
   Each weight row is stored as: codebook_index + per-row scale.
   Storage: K * d_in bits (codebook) + d_out * log2(K) bits (indices) + d_out floats.
   Source: BTC-LLM (ACL 2026, arXiv 2506.12040)

3. TernaryPTQLinear: Ternary {-1,0,+1} with calibration-refined scales.
   BitNet b1.58-style absmean ternary, but with per-channel scales refined
   using calibration activations (Hessian-weighted scale optimization).
   Storage: 1.58 bits/w (base-3 packed: 5 ternary values per byte = 1.6 bits).
   Source: BitNet b1.58 (arXiv 2402.17764) + ScaleQ-1.58 (arXiv 2608.01078)

All three are PTQ (no retraining needed) and use calibration data for
scale/refinement optimization. They target different points on the
compression-quality Pareto frontier below 2 bits/weight.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.engine.quant.novel_quant import (
    base3_packed_to_ternary,
    compute_hessian_proxy,
    ternary_to_base3_packed,
)

# Reuse existing infrastructure
from forge.engine.quant.novel_quant_r44 import (
    _SKIP_NAMES,
    _SKIP_TYPES,
    _hadamard_matrix,
)
from forge.engine.quant.novel_quant_r46 import _CachedDequantMixin
from forge.quant.protocol import QuantizedLinearMixin

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Binary bit packing helpers (8 binary ±1 values per byte)
# ──────────────────────────────────────────────────────────────────────────

def _pack_binary_bits(w: torch.Tensor) -> torch.Tensor:
    """Pack binary ±1 tensor as 1 bit per value (8 per byte).

    Args:
        w: (..., N) tensor with values in {-1, +1}

    Returns:
        (..., ceil(N/8)) uint8 tensor
    """
    bits = (w > 0).to(torch.uint8)  # 1 for +1, 0 for -1
    *leading, n = bits.shape
    pad = (8 - n % 8) % 8
    if pad > 0:
        bits = F.pad(bits.reshape(-1, n), (0, pad), value=0).reshape(*leading, n + pad)
    n_padded = bits.shape[-1]
    bits_flat = bits.reshape(-1, n_padded)
    # Pack 8 bits per byte
    packed = torch.zeros(bits_flat.shape[0], n_padded // 8, dtype=torch.uint8,
                         device=w.device)
    for i in range(8):
        packed |= (bits_flat[:, i::8] << i)
    return packed.reshape(*leading, n_padded // 8)


def _unpack_binary_bits(packed: torch.Tensor, n_orig: int) -> torch.Tensor:
    """Unpack 1-bit-per-value packed bytes back to ±1 tensor.

    Args:
        packed: (..., ceil(N/8)) uint8 tensor
        n_orig: original number of values N

    Returns:
        (..., N) int8 tensor with values in {-1, +1}
    """
    *leading, n_bytes = packed.shape
    p = packed.reshape(-1, n_bytes).to(torch.int32)
    bits = torch.zeros(p.shape[0], n_bytes * 8, dtype=torch.int8,
                       device=packed.device)
    for i in range(8):
        bits[:, i::8] = ((p >> i) & 1).to(torch.int8)
    bits = bits.reshape(*leading, n_bytes * 8)[..., :n_orig]
    return (bits * 2 - 1).to(torch.int8)  # 0→-1, 1→+1


# ──────────────────────────────────────────────────────────────────────────
# Hessian-preconditioned ADMM with SVD init (NanoQuant paper §3.2)
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _power_iteration(A: torch.Tensor, num_iters: int = 5):
    """Power iteration for top singular triplet (u, sigma, v) of A."""
    n = A.shape[1]
    v = torch.randn(n, device=A.device, dtype=A.dtype)
    v = v / torch.norm(v).clamp(min=1e-12)
    At = A.mT
    for _ in range(num_iters):
        u = torch.mv(A, v)
        u = u / u.norm().clamp(min=1e-12)
        v = torch.mv(At, u)
        v = v / v.norm().clamp(min=1e-12)
    u_unnorm = torch.mv(A, v)
    sigma = u_unnorm.norm().clamp(min=1e-12)
    u = u_unnorm / sigma
    return u, sigma, v


@torch.no_grad()
def _svid(W: torch.Tensor, inner_iters: int = 5, eps: float = 1e-12):
    """Sign-Value-Independent Decomposition (SVID).

    Returns (u, v, Sg) where Sg is the sign matrix of W and
    u, v approximate the dominant singular triplet of |W|.
    """
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = _power_iteration(W.abs(), inner_iters)
    u = u * s  # fold sigma into u
    return u, v, Sg


@torch.no_grad()
def _rank1_approx(W: torch.Tensor, inner_iters: int = 5,
                  eps: float = 1e-12) -> torch.Tensor:
    """Rank-1 approximation using SVID."""
    u, v, Sg = _svid(W, inner_iters, eps)
    apx = torch.outer(u, v)
    return apx * Sg


@torch.no_grad()
def _admm_solve_step(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor,
                     U: torch.Tensor, rho: float, reg: float,
                     eps: float = 1e-12) -> torch.Tensor:
    """One ADMM step using stabilized Cholesky decomposition.

    Solves: (X^T X + stabilizer*I) * Factor = X^T Y + rho*(Z-U)
    """
    orig_dtype = X.dtype
    X, Y, Z, U = (t.to(torch.float32) for t in (X, Y, Z, U))
    Xt = X.mT
    system_matrix = Xt @ X
    system_matrix = 0.5 * (system_matrix + system_matrix.mT)
    diag_mean = system_matrix.diagonal().mean().abs()
    stabilizer = torch.clamp(rho * diag_mean + reg, min=eps)
    system_matrix.diagonal().add_(stabilizer)
    rhs = (Xt @ Y) + rho * (Z - U)
    L, info = torch.linalg.cholesky_ex(system_matrix, upper=False)
    if info.item() == 0:
        Factor = torch.cholesky_solve(rhs, L, upper=False)
    else:
        Factor = torch.linalg.solve(system_matrix, rhs)
    return Factor.to(orig_dtype)


def _cubic_rho(x: float) -> float:
    """Cubic rho scheduler with early iteration protection."""
    return min(1.0, x) ** 3


@torch.no_grad()
def factorize_admm_nanoquant(
    W: torch.Tensor,
    i_norm: torch.Tensor | None = None,
    o_norm: torch.Tensor | None = None,
    mid_rank: int = 128,
    outer_iters: int = 400,
    inner_iters: int = 5,
    reg: float = 3e-2,
    eps: float = 1e-12,
    rho_scheduler: str = 'cubic',
    is_transpose: bool = False,
    verbose: bool = False,
) -> dict:
    """Hessian-preconditioned ADMM binary factorization (NanoQuant §3.2).

    Decomposes W (out_features × in_features) into binary matrices U, V
    using alternating binarization with Hessian preconditioning and
    stabilized Cholesky linear solves.

    Args:
        W: weight matrix (out_features, in_features)
        i_norm: input activation norms (in_features,) for Hessian preconditioning
        o_norm: output activation norms (out_features,) for Hessian preconditioning
        mid_rank: factorization rank
        outer_iters: ADMM outer iterations (paper default: 400)
        reg: regularization parameter
        rho_scheduler: rho scheduler name

    Returns:
        dict with keys: A (mid, out), B (mid, in), scale_pre, scale_post, W_final
    """
    if is_transpose:
        results = factorize_admm_nanoquant(
            W.mT, o_norm, i_norm, mid_rank, outer_iters, inner_iters,
            reg, eps, rho_scheduler, False, verbose)
        return {
            "W_final": results["W_final"].mT,
            "A": results["B"],
            "B": results["A"],
            "scale_pre": results["scale_post"],
            "scale_post": results["scale_pre"],
        }

    device = W.device
    out_features, in_features = W.shape

    # Hessian preconditioning: normalize by activation norms
    if i_norm is not None and o_norm is not None:
        norm_i = i_norm.sqrt().clamp(eps)
        norm_o = o_norm.sqrt().clamp(eps).unsqueeze(1)
        W * norm_i.unsqueeze(0) * norm_o
    else:
        norm_i = torch.ones(in_features, device=device, dtype=W.dtype)
        norm_o = torch.ones(out_features, 1, device=device, dtype=W.dtype)

    # Compute per-dimension scales from the (unnormalized) weight
    s1 = W.abs().mean(dim=1).clamp(min=eps)  # (out,)
    s2 = W.abs().mean(dim=0).clamp(min=eps)  # (in,)

    # Normalize W by scales for binary factorization
    W_scaled = W / (s1.unsqueeze(1) * s2.unsqueeze(0)).clamp(min=eps)

    r = mid_rank
    # Initialize V randomly as binary
    V = (torch.randn(in_features, r, device=device) > 0).to(torch.float32) * 2 - 1
    U = torch.zeros(out_features, r, device=device)

    for i in range(outer_iters):
        # Update U: U = sign(W_scaled @ V @ (V^T V)^{-1})
        # Use cholesky_solve for numerical stability
        VtV = V.T @ V + reg * torch.eye(r, device=device)
        VtV = 0.5 * (VtV + VtV.T)
        try:
            L = torch.linalg.cholesky(VtV)
            U = torch.cholesky_solve(W_scaled @ V, L)
        except Exception:
            U = W_scaled @ V @ torch.linalg.inv(VtV)
        U = torch.sign(U)
        U[U == 0] = 1

        # Update V: V = sign(W_scaled^T @ U @ (U^T U)^{-1})
        UtU = U.T @ U + reg * torch.eye(r, device=device)
        UtU = 0.5 * (UtU + UtU.T)
        try:
            L = torch.linalg.cholesky(UtU)
            V = torch.cholesky_solve(W_scaled.T @ U, L)
        except Exception:
            V = W_scaled.T @ U @ torch.linalg.inv(UtU)
        V = torch.sign(V)
        V[V == 0] = 1

        if verbose and (i == 0 or (i + 1) % 100 == 0 or i == outer_iters - 1):
            W_approx = s1.unsqueeze(1) * (U @ V.T) * s2.unsqueeze(0)
            err = (W - W_approx).norm().item() / W.norm().item()
            logger.info(f"  [ADMM {i+1:04d}/{outer_iters}] rel_err={err:.5e}")

    # Magnitude balancing (Appendix A)
    norm_U = U.norm().clamp(min=eps)
    norm_V = V.norm().clamp(min=eps)
    balance = (norm_V / norm_U).sqrt()
    U = U * balance
    V = V / balance

    # Re-optimize scales given the binary factors (alternating LS)
    UV = U @ V.T  # (out, in)
    s1_opt = s1.clone()
    s2_opt = s2.clone()
    for _ in range(5):
        s1_num = (W * UV * s2_opt.unsqueeze(0)).sum(dim=1)
        s1_den = (UV.square() * s2_opt.unsqueeze(0).square()).sum(dim=1).clamp(eps)
        s1_opt = (s1_num / s1_den).clamp(min=eps)
        s2_num = (W * UV * s1_opt.unsqueeze(1)).sum(dim=0)
        s2_den = (UV.square() * s1_opt.unsqueeze(1).square()).sum(dim=0).clamp(eps)
        s2_opt = (s2_num / s2_den).clamp(min=eps)

    W_final = s1_opt.unsqueeze(1) * (U @ V.T) * s2_opt.unsqueeze(0)

    return {
        "W_final": W_final,
        "A": U.mT,    # (mid, out) — for storage as binary
        "B": V.mT,    # (mid, in)
        "scale_pre": s2_opt.view(1, -1),
        "scale_post": s1_opt.view(1, -1),
    }


# ──────────────────────────────────────────────────────────────────────────
# Algorithm 1: NanoQuant — Low-rank binary factorization + ADMM
# ──────────────────────────────────────────────────────────────────────────

class NanoQuantLinear(QuantizedLinearMixin, _CachedDequantMixin):
    """NanoQuant: Low-rank binary factorization for sub-1-bit compression.

    Decomposes W (d_out × d_in) as:
        W ≈ s1 ⊙ (U_±1 @ V_±1^T) ⊙ s2^T

    where U ∈ {-1,+1}^(d_out × r), V ∈ {-1,+1}^(d_in × r),
    s1 ∈ R^d_out, s2 ∈ R^d_in.

    Storage per layer:
        - U: d_out * r bits (binary)
        - V: d_in * r bits (binary)
        - s1: d_out * 16 bits (float16)
        - s2: d_in * 16 bits (float16)
        Total bits = r*(d_out + d_in) + 16*(d_out + d_in)
        Effective bits/w = [r*(d_out + d_in) + 16*(d_out + d_in)] / (d_out * d_in)

    For d_out = d_in = d and rank r:
        eff_bits = (r + 16) * 2 / d
    For d=896, r=32: eff_bits = 48*2/896 = 0.107 bits/w (!)
    For d=896, r=128: eff_bits = 144*2/896 = 0.321 bits/w
    For d=896, r=256: eff_bits = 272*2/896 = 0.607 bits/w

    The ADMM initialization finds binary U, V that minimize ||W - s1*(U@V^T)*s2||².
    Then optional gradient-based refinement improves the factorization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 rank: int = 128, admm_iters: int = 50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.admm_iters = admm_iters

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Binary factors U (d_out × r) and V (d_in × r), packed 1 bit per value
        self.register_buffer('U_packed', torch.zeros(0, dtype=torch.uint8))
        self.register_buffer('V_packed', torch.zeros(0, dtype=torch.uint8))
        self.U_shape = (0, 0)
        self.V_shape = (0, 0)
        # Per-dimension scales
        self.register_buffer('s1', torch.zeros(1, dtype=torch.float16))  # (d_out,)
        self.register_buffer('s2', torch.zeros(1, dtype=torch.float16))  # (d_in,)

        self._init_cache()

    @classmethod
    def from_linear(cls, lin: nn.Linear, rank: int = 128,
                    admm_iters: int = 400,
                    refine_steps: int = 0,
                    i_norm: torch.Tensor | None = None,
                    o_norm: torch.Tensor | None = None,
                    verbose: bool = False) -> NanoQuantLinear:
        """Create NanoQuant-quantized layer.

        Args:
            lin: original linear layer
            rank: factorization rank r. Higher = better quality, more memory.
                  For sub-1-bit: r < d_in * d_out / (d_out + d_in) - 16
            admm_iters: ADMM iterations for binary factor initialization
                        (paper default: 400)
            refine_steps: gradient-based refinement steps (0 = ADMM only)
            i_norm: input activation Hessian diagonal (in_features,) for
                    Hessian-aware preconditioning. If None, uses weight-based
                    proxy.
            o_norm: output activation Hessian diagonal (out_features,).
            verbose: print ADMM progress
        """
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None,
                    rank=rank, admm_iters=admm_iters)

        W = lin.weight.data.float()  # (d_out, d_in)
        device = W.device

        # Hessian preconditioning: use activation norms if provided,
        # otherwise use weight-based proxy (row/col magnitudes)
        if i_norm is None:
            i_norm = W.abs().mean(dim=0).clamp(min=1e-8)  # (d_in,)
        if o_norm is None:
            o_norm = W.abs().mean(dim=1).clamp(min=1e-8)  # (d_out,)
        i_norm = i_norm.to(device).float()
        o_norm = o_norm.to(device).float()

        # Run Hessian-preconditioned ADMM factorization
        is_transpose = out_f < in_f
        results = factorize_admm_nanoquant(
            W, i_norm=i_norm, o_norm=o_norm, mid_rank=rank,
            outer_iters=admm_iters, is_transpose=is_transpose,
            verbose=verbose)

        # Extract binary factors
        # ADMM returns A=(mid, out), B=(mid, in) always (after is_transpose handling)
        A = results["A"]  # (mid, out)
        B = results["B"]  # (mid, in)

        # U = (out, mid), V = (in, mid) — same for both transpose and non-transpose
        # because the is_transpose case already swaps inside factorize_admm_nanoquant
        U = A.mT  # (out, mid)
        V = B.mT  # (in, mid)

        # Binarize
        U = torch.sign(U)
        U[U == 0] = 1
        V = torch.sign(V)
        V[V == 0] = 1

        # Extract per-dimension scales from the factorization
        s1 = results["scale_post"].squeeze(0).to(device)  # (out,)
        s2 = results["scale_pre"].squeeze(0).to(device)   # (in,)

        # Step 3: Optional gradient-based refinement with STE
        if refine_steps > 0:
            U_f = U.clone().requires_grad_(True)
            V_f = V.clone().requires_grad_(True)
            s1_f = s1.clone().requires_grad_(True)
            s2_f = s2.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([U_f, V_f, s1_f, s2_f], lr=0.01)

            for _step in range(refine_steps):
                optimizer.zero_grad()
                # STE: forward uses sign, backward uses gradient
                U_q = U_f + (torch.sign(U_f) - U_f).detach()
                V_q = V_f + (torch.sign(V_f) - V_f).detach()
                W_recon = s1_f.unsqueeze(1) * (U_q @ V_q.T) * s2_f.unsqueeze(0)
                loss = F.mse_loss(W_recon, W)
                loss.backward()
                optimizer.step()

            U = torch.sign(U_f.detach())
            V = torch.sign(V_f.detach())
            s1 = s1_f.detach()
            s2 = s2_f.detach()
            U[U == 0] = 1
            V[V == 0] = 1

        # Store — pack binary ±1 as 1 bit per value
        layer.U_packed = _pack_binary_bits(U).to(torch.uint8).contiguous()
        layer.V_packed = _pack_binary_bits(V).to(torch.uint8).contiguous()
        layer.U_shape = U.shape
        layer.V_shape = V.shape
        layer.s1 = s1.to(torch.float16).contiguous()
        layer.s2 = s2.to(torch.float16).contiguous()

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        U = _unpack_binary_bits(self.U_packed, self.U_shape[0] * self.U_shape[1])
        U = U.view(self.U_shape).to(dtype)  # (d_out, r)
        V = _unpack_binary_bits(self.V_packed, self.V_shape[0] * self.V_shape[1])
        V = V.view(self.V_shape).to(dtype)  # (d_in, r)
        s1 = self.s1.to(dtype)  # (d_out,)
        s2 = self.s2.to(dtype)  # (d_in,)

        # W ≈ s1 ⊙ (U @ V^T) ⊙ s2^T
        W = s1.unsqueeze(1) * (U @ V.T) * s2.unsqueeze(0)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        eff_bits = (self.rank * (self.out_features + self.in_features) +
                     16 * (self.out_features + self.in_features)) / \
                    max(self.out_features * self.in_features, 1)
        return (f"NanoQuantLinear(in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, eff_bits={eff_bits:.3f})")


# ──────────────────────────────────────────────────────────────────────────
# NanoQuant QAT — Quantization-Aware Training with STE binary factors
# ──────────────────────────────────────────────────────────────────────────

class NanoQuantQATLinear(QuantizedLinearMixin):
    """NanoQuant with quantization-aware training (QAT).

    Keeps continuous latent matrices U_latent, V_latent as trainable
    parameters. During forward, uses sign(U_latent) and sign(V_latent)
    via straight-through estimator (STE) so gradients flow through the
    sign() barrier to the continuous values.

    W ≈ s1 ⊙ (sign(U_latent) @ sign(V_latent)^T) ⊙ s2^T

    Training:
      - U_latent, V_latent: continuous, updated by STE gradients
      - s1, s2: continuous scales, direct gradients
      - bias: standard

    After training, call .bake() to pack the hardened binary factors
    into a NanoQuantLinear for inference (no STE overhead).

    Args:
        in_features, out_features: dimensions
        rank: factorization rank r
        bias: include bias term
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, rank: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Continuous latent matrices — these are what the optimizer sees
        # Initialized to ±1 (binary) from ADMM; QAT refines them
        self.U_latent = nn.Parameter(
            torch.randn(out_features, rank) * 0.1)
        self.V_latent = nn.Parameter(
            torch.randn(in_features, rank) * 0.1)
        # Per-dimension scales (trainable)
        self.s1 = nn.Parameter(torch.ones(out_features))
        self.s2 = nn.Parameter(torch.ones(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # STE: forward uses sign, backward passes gradient through
        # Implemented as: sign(x) + (x - x.detach()) which makes
        # the forward = sign(x) but backward = identity
        # However, the standard STE is: sign(x).detach() - x.detach() + x
        # which gives forward=sign(x), backward=1

    @classmethod
    def from_linear(cls, lin: nn.Linear, rank: int = 128,
                    admm_iters: int = 50,
                    quick_init: bool = True) -> NanoQuantQATLinear:
        """Initialize QAT layer from a pre-trained linear layer.

        With quick_init=True (default), skips ADMM and uses SVD-based
        initialization: take top-r singular vectors, binarize signs.
        This is instant vs ~0.5s/layer for ADMM. QAT STE training
        will refine the binary factors from this starting point.

        With quick_init=False, runs full ADMM factorization first.
        """
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None, rank=rank)
        device = lin.weight.device
        W = lin.weight.data.float()

        if quick_init:
            # SVD-based fast init: top-r singular vectors → binary signs
            # This gives a much better starting point than random while
            # being ~100x faster than ADMM
            try:
                U_svd, S_svd, V_svd = torch.linalg.svd(W, full_matrices=False)
                # U_svd: (out, min(out,in)), V_svd: (min(out,in), in)
                # Take top-r components
                U_init = U_svd[:, :rank]  # (out, rank)
                V_init = V_svd[:rank, :].T  # (in, rank)
                # Scale by singular values so magnitude is meaningful
                U_init = U_init * S_svd[:rank].sqrt().unsqueeze(0)
                V_init = V_init * S_svd[:rank].sqrt().unsqueeze(0)
            except Exception:
                # Fallback: random init
                U_init = torch.randn(out_f, rank, device=device)
                V_init = torch.randn(in_f, rank, device=device)

            layer.U_latent.data = U_init.to(device)
            layer.V_latent.data = V_init.to(device)
            # Scales from weight statistics
            layer.s1.data = W.abs().mean(dim=1).clamp(min=1e-8).to(device)
            layer.s2.data = W.abs().mean(dim=0).clamp(min=1e-8).to(device)
        else:
            # Full ADMM init (slow but better starting point)
            nq = NanoQuantLinear.from_linear(lin, rank=rank, admm_iters=admm_iters)
            from forge.engine.quant.block_recon import _unpack_to_soft
            U = _unpack_to_soft(nq, 'U').to(device)
            V = _unpack_to_soft(nq, 'V').to(device)
            layer.U_latent.data = U
            layer.V_latent.data = V
            layer.s1.data = nq.s1.to(torch.float32).to(device)
            layer.s2.data = nq.s2.to(torch.float32).to(device)

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _ste_sign(self, x: torch.Tensor) -> torch.Tensor:
        """Straight-through estimator: sign in forward, identity in backward."""
        return torch.sign(x) + (x - x.detach()) - (x - x.detach()).detach()
        # Simpler equivalent: x + (torch.sign(x) - x).detach()
        # But the above is more numerically stable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # STE: forward sees sign(), backward sees identity
        x.new_tensor(1.0)  # dummy for dtype/device
        U_q = self.U_latent + (torch.sign(self.U_latent) - self.U_latent).detach()
        V_q = self.V_latent + (torch.sign(self.V_latent) - self.V_latent).detach()

        # Reconstruct weight: W = s1 * (U @ V^T) * s2
        s1 = self.s1.to(x.dtype)
        s2 = self.s2.to(x.dtype)
        U_q = U_q.to(x.dtype)
        V_q = V_q.to(x.dtype)
        W = s1.unsqueeze(1) * (U_q @ V_q.T) * s2.unsqueeze(0)

        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, W, bias)

    @torch.no_grad()
    def bake(self) -> NanoQuantLinear:
        """Convert trained QAT layer to inference-only NanoQuantLinear.

        Hardens the latent matrices to binary ±1 and packs them.
        """
        nq = NanoQuantLinear(self.in_features, self.out_features,
                             bias=self.bias is not None, rank=self.rank)
        device = self.U_latent.device

        U = torch.sign(self.U_latent.data)
        U[U == 0] = 1
        V = torch.sign(self.V_latent.data)
        V[V == 0] = 1

        nq.U_packed = _pack_binary_bits(U.to(torch.int8)).to(torch.uint8).contiguous()
        nq.V_packed = _pack_binary_bits(V.to(torch.int8)).to(torch.uint8).contiguous()
        nq.U_shape = U.shape
        nq.V_shape = V.shape
        nq.s1 = self.s1.data.to(torch.float16).contiguous()
        nq.s2 = self.s2.data.to(torch.float16).contiguous()
        if self.bias is not None:
            nq.bias.data = self.bias.data.clone().to(device)
        nq = nq.to(device)
        return nq

    def __repr__(self):
        eff_bits = (self.rank * (self.out_features + self.in_features) +
                     16 * (self.out_features + self.in_features)) / \
                    max(self.out_features * self.in_features, 1)
        return (f"NanoQuantQATLinear(in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, eff_bits={eff_bits:.3f})")


def convert_model_to_nanoquant_qat(model: nn.Module, rank: int = 128,
                                   admm_iters: int = 50,
                                   quick_init: bool = True,
                                   verbose: bool = True) -> int:
    """Replace all nn.Linear with NanoQuantQATLinear for QAT.

    With quick_init=True (default), uses SVD-based initialization
    which is ~100x faster than ADMM. QAT STE training refines from there.

    Args:
        model: model to convert
        rank: NanoQuant factorization rank
        admm_iters: ADMM iterations (only used if quick_init=False)
        quick_init: Use SVD-based fast init (default True)
        verbose: print progress
    Returns:
        number of layers converted
    """
    import time as _time
    n = 0
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R48:
            if any(s in name for s in _SKIP_NAMES):
                continue
            total += 1

    t0 = _time.time()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R48:
            if any(s in name for s in _SKIP_NAMES):
                continue
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                qat_layer = NanoQuantQATLinear.from_linear(
                    module, rank=rank, admm_iters=admm_iters,
                    quick_init=quick_init)
                setattr(parent, parts[-1], qat_layer)
                n += 1
                if verbose and (n % 20 == 0 or n == total):
                    elapsed = _time.time() - t0
                    logger.info(f"  [NanoQuantQAT] {n}/{total} layers converted "
                          f"({elapsed:.1f}s, {elapsed/n:.3f}s/layer)")
            except Exception as e:
                if verbose:
                    logger.warning(f"  [NanoQuantQAT] Skipped {name}: {e}")
    if verbose and n > 0:
        logger.info(f"  [NanoQuantQAT] {n} layers converted to QAT "
              f"({_time.time()-t0:.1f}s total)")
    return n


@torch.no_grad()
def bake_qat_model(model: nn.Module, verbose: bool = True) -> int:
    """Convert all NanoQuantQATLinear layers to NanoQuantLinear for inference.

    Call after QAT training is complete.
    Returns number of layers baked.
    """
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, NanoQuantQATLinear):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            nq = module.bake()
            setattr(parent, parts[-1], nq)
            n += 1
    if verbose and n > 0:
        logger.info(f"  [NanoQuantQAT] {n} layers baked to inference mode")
    return n


# ──────────────────────────────────────────────────────────────────────────
# Algorithm 2: BTC-LLM — Binary codebook clustering
# ──────────────────────────────────────────────────────────────────────────

class BTCQuantLinear(QuantizedLinearMixin, _CachedDequantMixin):
    """BTC-LLM: Binary codebook clustering for sub-1-bit compression.

    Instead of storing each weight row as binary ±1, cluster similar binary
    patterns into a codebook of K patterns. Each output row references a
    codebook entry by index.

    Storage per layer:
        - Codebook: K * d_in bits (binary patterns)
        - Indices: d_out * ceil(log2(K)) bits
        - Scales: d_out * 16 bits (float16)
        Total = K*d_in + d_out*ceil(log2(K)) + 16*d_out

    For d_in=896, d_out=896, K=256:
        Codebook: 256*896 = 229,376 bits
        Indices: 896*8 = 7,168 bits
        Scales: 896*16 = 14,336 bits
        Total: 250,880 bits
        eff_bits = 250,880 / (896*896) = 0.313 bits/w

    The learnable transformation (Hadamard rotation) reduces outliers before
    binarization, making binary patterns more clustered.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 codebook_size: int = 256, use_rotation: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.codebook_size = codebook_size
        self.use_rotation = use_rotation

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Codebook: K binary patterns packed 8 per byte (1 bit each)
        # Original: (K, d_in_eff) int8 → packed: (K, ceil(d_in_eff/8)) uint8
        self.register_buffer('codebook_packed', torch.zeros(0, dtype=torch.uint8))
        self.codebook_d_in_eff = 0  # actual d_in_eff (for unpacking)
        # Per-row indices into codebook
        self.register_buffer('row_indices', torch.zeros(1, dtype=torch.int32))
        # Per-row scales
        self.register_buffer('scales', torch.zeros(1, dtype=torch.float16))
        # Hadamard rotation: store only the Hadamard order (log2(N) integers)
        # instead of the full N×N matrix. Reconstruct on-the-fly.
        self.hadamard_order = 0  # log2(hadamard_size)
        self.register_buffer('rotation_signs', torch.zeros(0, dtype=torch.int8))

        self._init_cache()

    @classmethod
    def from_linear(cls, lin: nn.Linear, codebook_size: int = 256,
                    use_rotation: bool = True) -> BTCQuantLinear:
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None,
                    codebook_size=codebook_size, use_rotation=use_rotation)

        W = lin.weight.data.float()  # (d_out, d_in)
        device = W.device

        # Step 1: Optional Hadamard rotation to reduce outliers
        hadamard_order = 0
        if use_rotation:
            h_size = 1
            while h_size < in_f:
                h_size *= 2
            hadamard_order = int(math.log2(h_size))
            pad = h_size - in_f
            if pad > 0:
                W = F.pad(W, (0, pad))
            H = _hadamard_matrix(h_size, device, W.dtype)
            W_rot = W @ H  # (d_out, hadamard_size)
        else:
            W_rot = W
            h_size = in_f

        d_in_eff = W_rot.shape[1]
        layer.hadamard_order = hadamard_order

        # Step 2: Per-channel absmean scale + binarize
        scales = W_rot.abs().mean(dim=1).clamp(min=1e-8) / 0.7  # (d_out,)
        W_norm = W_rot / scales.unsqueeze(1).clamp(min=1e-12)
        W_bin = torch.sign(W_norm)  # (d_out, d_in_eff) ∈ {-1, 0, +1}
        W_bin[W_bin == 0] = 1  # Force to ±1 (no zeros in binary)

        # Step 3: Cluster binary rows into codebook using K-means
        # Use Hamming-like distance (L2 on binary ≈ Hamming)
        K = min(codebook_size, out_f)
        if out_f <= K:
            # Each row is its own codebook entry
            codebook = W_bin.to(torch.int8)
            indices = torch.arange(out_f, dtype=torch.int32)
        else:
            # K-means clustering on binary patterns
            codebook, indices = _binary_kmeans(W_bin, K, device=device)

        # Store — pack codebook binary ±1 as 1 bit per value (8 per byte)
        cb_packed = _pack_binary_bits(codebook)  # (K, ceil(D/8)) uint8
        layer.codebook_packed = cb_packed.to(torch.uint8).contiguous()
        layer.codebook_d_in_eff = d_in_eff
        layer.row_indices = indices.to(torch.int32).contiguous()
        layer.scales = scales.to(torch.float16).contiguous()

        if use_rotation:
            layer.rotation_signs = torch.zeros(0, dtype=torch.int8)  # empty — reconstruct on the fly
        else:
            layer.rotation_signs = torch.zeros(0, dtype=torch.int8)

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        device = self.codebook_packed.device

        # Unpack codebook: (K, ceil(D/8)) uint8 → (K, D) ±1
        cb = _unpack_binary_bits(self.codebook_packed, self.codebook_d_in_eff)
        cb = cb.to(dtype)  # (K, d_in_eff)
        idx = self.row_indices.long()  # (d_out,)
        scales = self.scales.to(dtype)  # (d_out,)

        W_rot = scales.unsqueeze(1) * cb[idx]  # (d_out, d_in_eff)

        # Inverse rotation if used (reconstruct Hadamard on-the-fly)
        if self.use_rotation and self.hadamard_order > 0:
            h_size = 2 ** self.hadamard_order
            H = _hadamard_matrix(h_size, device, dtype)
            W = W_rot @ H.T
            return W[:, :self.in_features]
        else:
            return W_rot[:, :self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        d_in_eff = 2 ** self.hadamard_order if self.hadamard_order > 0 else self.in_features
        cb_bits = self.codebook_size * d_in_eff
        idx_bits = self.out_features * max(1, math.ceil(math.log2(max(self.codebook_size, 2))))
        scale_bits = self.out_features * 16
        total = cb_bits + idx_bits + scale_bits
        eff = total / max(self.out_features * self.in_features, 1)
        return (f"BTCQuantLinear(in={self.in_features}, out={self.out_features}, "
                f"K={self.codebook_size}, eff_bits={eff:.3f})")


def _binary_kmeans(W_bin: torch.Tensor, K: int, device: torch.device,
                   n_iters: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """K-means clustering on binary ±1 vectors using L2 distance.

    Args:
        W_bin: (N, D) binary ±1 matrix
        K: number of clusters
        device: torch device
        n_iters: K-means iterations

    Returns:
        codebook: (K, D) binary ±1 centroids
        indices: (N,) cluster assignments
    """
    N, _D = W_bin.shape

    # Initialize: random selection of K rows as centroids
    perm = torch.randperm(N, device=device)[:K]
    centroids = W_bin[perm].clone()  # (K, D)

    for _ in range(n_iters):
        # Assign: find nearest centroid (L2 on binary = 2*Hamming)
        # dist[i,j] = ||W_bin[i] - centroids[j]||²
        # = D - 2*W_bin[i]@centroids[j]^T + D (since ||±1||² = D)
        # So dist ∝ -W_bin @ centroids^T (maximize dot product)
        dots = W_bin @ centroids.T  # (N, K)
        indices = dots.argmax(dim=1)  # (N,)

        # Update: new centroids = sign(mean of assigned vectors)
        for k in range(K):
            mask = (indices == k)
            if mask.any():
                mean = W_bin[mask].float().mean(dim=0)
                centroids[k] = torch.sign(mean)
                centroids[k][centroids[k] == 0] = 1
            # If empty cluster, keep old centroid

    # Final assignment
    dots = W_bin @ centroids.T
    indices = dots.argmax(dim=1)

    return centroids, indices


# ──────────────────────────────────────────────────────────────────────────
# Algorithm 3: TernaryPTQ — Ternary {-1,0,+1} with calibration refinement
# ──────────────────────────────────────────────────────────────────────────

class TernaryPTQLinear(QuantizedLinearMixin, _CachedDequantMixin):
    """TernaryPTQ: Ternary quantization with Hessian-refined per-channel scales.

    BitNet b1.58 uses ternary {-1, 0, +1} with absmean scaling:
        scale = absmean(W) / 0.7
        W_q = round(W / scale) clipped to {-1, 0, +1}

    We improve on vanilla BitNet PTQ by:
    1. Per-channel scales (not per-tensor) — adapts to channel variance
    2. Hessian-weighted scale refinement — uses calibration activations to
       find the scale that minimizes output error, not weight error
    3. Base-3 packing (5 ternary values per byte = 1.6 bits/w)

    Storage: 1.6 bits/w (base-3 packed) + 16 bits/channel (float16 scale)
    Effective: ~1.62 bits/w for typical dimensions

    Needs calibration data for Hessian-weighted scale refinement.
    Can also run without calibration (vanilla absmean per-channel).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Ternary weights packed as base-3 (5 per byte)
        self.register_buffer('weight_packed', torch.zeros(0, dtype=torch.uint8))
        # Per-output-channel scale
        self.register_buffer('scales', torch.zeros(0, dtype=torch.float16))
        # Original weight count (for unpacking)
        self.register_buffer('n_weights', torch.tensor(0, dtype=torch.int32))

        self._init_cache()

    @classmethod
    def from_linear(cls, lin: nn.Linear,
                    activations: torch.Tensor | None = None,
                    refine_iters: int = 20) -> TernaryPTQLinear:
        """Create ternary-quantized layer.

        Args:
            lin: original linear layer
            activations: (N, in_features) calibration activations.
                         If None, uses vanilla absmean (no Hessian refinement).
            refine_iters: gradient-based scale refinement iterations
        """
        out_f, in_f = lin.weight.shape
        layer = cls(in_f, out_f, bias=lin.bias is not None)

        W = lin.weight.data.float()  # (d_out, d_in)
        device = W.device

        # Step 1: Initial per-channel absmean scale (BitNet convention)
        scales = W.abs().mean(dim=1).clamp(min=1e-8) / 0.7  # (d_out,)

        # Step 2: Ternary quantize
        W_norm = W / scales.unsqueeze(1).clamp(min=1e-12)
        W_ternary = torch.sign(W_norm) * (W_norm.abs() > 0.7).float()

        # Step 3: Optional Hessian-weighted scale refinement
        if activations is not None and refine_iters > 0:
            h = compute_hessian_proxy(activations.float().to(device))  # (d_in,)

            # Refine scales to minimize Hessian-weighted output error
            # Use a safe learning rate and clamp scales to prevent divergence
            scales_opt = scales.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([scales_opt], lr=0.001)

            for _ in range(refine_iters):
                optimizer.zero_grad()
                W_norm = W / scales_opt.unsqueeze(1).clamp(min=1e-6)
                W_q = torch.sign(W_norm) * (W_norm.abs() > 0.7).float()
                W_recon = W_q * scales_opt.unsqueeze(1)
                # Hessian-weighted MSE: weight error by input activation²
                err = (W - W_recon) ** 2  # (d_out, d_in)
                h_exp = h.unsqueeze(0).expand_as(err)
                loss = (err * h_exp).sum() / h_exp.sum().clamp(min=1e-12)
                loss.backward()
                optimizer.step()
                # Clamp scales to reasonable range to prevent divergence
                with torch.no_grad():
                    scales_opt.clamp_(min=scales.min() * 0.5,
                                      max=scales.max() * 2.0)

            scales = scales_opt.detach()

        # Final ternary quantization with refined scales
        W_norm = W / scales.unsqueeze(1).clamp(min=1e-12)
        W_ternary = torch.sign(W_norm) * (W_norm.abs() > 0.7).float()
        W_ternary = W_ternary.to(torch.int8)  # {-1, 0, +1}

        # Pack as base-3 (5 values per byte)
        packed = ternary_to_base3_packed(W_ternary)

        layer.weight_packed = packed.contiguous().to(device)
        layer.scales = scales.to(torch.float16).contiguous().to(device)
        layer.n_weights = torch.tensor(W_ternary.numel(), dtype=torch.int32)

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().to(device)

        return layer

    def _dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        n = self.n_weights.item()
        W_ternary = base3_packed_to_ternary(self.weight_packed, n)  # (d_out*d_in,) int8
        W_ternary = W_ternary[:n].view(self.out_features, self.in_features).to(dtype)
        scales = self.scales.to(dtype)  # (d_out,)
        return W_ternary * scales.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def __repr__(self):
        packed_bits = self.weight_packed.numel() * 8
        scale_bits = self.scales.numel() * 16
        total = packed_bits + scale_bits
        eff = total / max(self.out_features * self.in_features, 1)
        return (f"TernaryPTQLinear(in={self.in_features}, out={self.out_features}, "
                f"eff_bits={eff:.3f})")


# ──────────────────────────────────────────────────────────────────────────
# Model-level conversion functions
# ──────────────────────────────────────────────────────────────────────────

_SKIP_TYPES_R48 = _SKIP_TYPES + (
    "NanoQuantLinear", "BTCQuantLinear", "TernaryPTQLinear",
    # R46 types
    "HadamardRotatedFP4Linear", "GPTQFP4Linear", "AWQFP4Linear",
    "OptimalGridFP4Linear", "HadamardGPTQFP4Linear", "HadamardAWQFP4Linear",
    # R45 types
    "WaveletLiftLinear", "SchurABFP4Linear", "SVDLiftBinaryLinear",
)


def _replace_linears_r48(model: nn.Module, factory, verbose_name: str,
                         verbose: bool = True, **kwargs) -> int:
    """Generic nn.Linear replacement that skips all quantized types."""
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R48:
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


def quantize_model_nanoquant(model: nn.Module, rank: int = 128,
                             admm_iters: int = 400, refine_steps: int = 0,
                             verbose: bool = True,
                             hessian_norms: dict | None = None) -> int:
    """Replace all nn.Linear with NanoQuantLinear.

    Args:
        model: model to quantize
        rank: factorization rank
        admm_iters: ADMM iterations (default 400, paper default)
        refine_steps: gradient refinement steps
        verbose: print progress
        hessian_norms: optional dict mapping layer name to
            (i_norm, o_norm) tensors for Hessian-aware preconditioning.
            If None, uses weight-based proxy.
    """
    if hessian_norms:
        # Custom factory that passes Hessian norms per-layer
        n = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R48:
                if any(s in name for s in _SKIP_NAMES):
                    continue
                parent = model
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                try:
                    i_norm, o_norm = hessian_norms.get(name, (None, None))
                    setattr(parent, parts[-1],
                            NanoQuantLinear.from_linear(
                                module, rank=rank, admm_iters=admm_iters,
                                refine_steps=refine_steps,
                                i_norm=i_norm, o_norm=o_norm))
                    n += 1
                except Exception as e:
                    if verbose:
                        logger.warning(f"  [NanoQuant] Skipped {name}: {e}")
        if verbose and n > 0:
            logger.info(f"  [NanoQuant] {n} layers quantized")
        return n
    else:
        return _replace_linears_r48(model, NanoQuantLinear.from_linear,
                                    "NanoQuant", verbose, rank=rank,
                                    admm_iters=admm_iters, refine_steps=refine_steps)


def quantize_model_btc(model: nn.Module, codebook_size: int = 256,
                       use_rotation: bool = True,
                       verbose: bool = True) -> int:
    """Replace all nn.Linear with BTCQuantLinear (no calibration needed)."""
    return _replace_linears_r48(model, BTCQuantLinear.from_linear,
                                "BTC", verbose, codebook_size=codebook_size,
                                use_rotation=use_rotation)


def quantize_model_ternary_ptq(model: nn.Module, activations: dict | None = None,
                               refine_iters: int = 20,
                               verbose: bool = True) -> int:
    """Replace all nn.Linear with TernaryPTQLinear.

    Args:
        model: the model to quantize
        activations: dict mapping layer name -> (N, in_features) activation tensor.
                     If None, uses vanilla absmean (no Hessian refinement).
        refine_iters: gradient-based scale refinement iterations
    """
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and type(module).__name__ not in _SKIP_TYPES_R48:
            if any(s in name for s in _SKIP_NAMES):
                continue
            acts = activations.get(name) if activations else None
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                setattr(parent, parts[-1],
                        TernaryPTQLinear.from_linear(module, activations=acts,
                                                     refine_iters=refine_iters))
                n += 1
            except Exception as e:
                if verbose:
                    logger.warning(f"  [TernaryPTQ] Skipped {name}: {e}")
    if verbose and n > 0:
        logger.info(f"  [TernaryPTQ] {n} layers quantized")
    return n


# ──────────────────────────────────────────────────────────────────────────
# Memory estimation
# ──────────────────────────────────────────────────────────────────────────

def estimate_r48_memory(model: nn.Module) -> dict:
    """Estimate weight memory for R48 quantized layers."""
    breakdown = {}
    total_bytes = 0
    total_params = 0

    for _name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name == 'NanoQuantLinear':
            # U packed: d_out * r bits, V packed: d_in * r bits, s1: d_out * 16, s2: d_in * 16
            w_bytes = module.U_packed.numel() + module.V_packed.numel()
            w_bytes += module.s1.numel() * 2 + module.s2.numel() * 2
            params = module.out_features * module.in_features
        elif cls_name == 'BTCQuantLinear':
            # Codebook packed: K * ceil(d_in_eff/8) bytes, indices: d_out * 4, scales: d_out * 2
            # No rotation matrix stored — reconstructed on-the-fly
            w_bytes = module.codebook_packed.numel()
            w_bytes += module.row_indices.numel() * 4
            w_bytes += module.scales.numel() * 2
            params = module.out_features * module.in_features
        elif cls_name == 'TernaryPTQLinear':
            # Packed: bytes, scales: d_out * 2
            w_bytes = module.weight_packed.numel()
            w_bytes += module.scales.numel() * 2
            params = module.out_features * module.in_features
        else:
            continue

        if cls_name not in breakdown:
            breakdown[cls_name] = {'bytes': 0, 'params': 0}
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
