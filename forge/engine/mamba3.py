"""Mamba-3 SSM forward pass module — complex-valued states, MIMO, exponential-trapezoidal discretization.

Mamba-3 (arXiv:2603.15569, March 2026) extends the Mamba selective SSM with
three key innovations over Mamba-2:

1. **Complex-valued states**: The SSM state h is complex (real + imaginary).
   A_log, B, C are all complex. This doubles effective state capacity per
   dimension without doubling parameters. The input projection (x_proj)
   produces complex B and C (extra rows for imaginary parts, zero-init for
   lossless warm start from Mamba-2).

2. **MIMO (Multiple Input Multiple Output)**: Instead of scalar SSM (one
   input -> one output per channel), Mamba-3 processes n_inputs inputs and
   produces n_outputs outputs simultaneously via matrix-valued B and C.
   B: (d_state, n_inputs) complex, C: (n_outputs, d_state) complex.
   When n_inputs = n_outputs = 1, this reduces to the standard scalar SSM
   (matches Mamba-2 exactly with imag=0).

3. **Exponential-trapezoidal discretization**: A new discretization scheme
   that combines exponential interpolation with trapezoidal integration:
       A_bar = exp(delta * A)
       B_bar = (exp(delta*A) - 1) / (delta*A) * delta * B
   The correction factor (exp(da)-1)/da -> 1 as da->0 (ZOH limit), giving
   better accuracy than standard zero-order hold for finite delta*A.

Core SSM recurrence (complex-valued):
    h_t = A_bar * h_{t-1} + B_bar @ x_t    (complex state update)
    y_t = Re(C_bar @ h_t) + D * x_t        (real output via Re())

Parameter shapes (matching mamba3_key.py for lossless warm start):
    in_proj.weight   (2*d_inner, d_model)        — unchanged from Mamba-2
    conv1d.weight    (d_inner, 1, d_conv)        — unchanged
    conv1d.bias      (d_inner,)                  — unchanged
    x_proj.weight    (dt_rank + 2*d_state*(n_inputs+n_outputs), d_inner)
                     — complex B/C; default n_inputs=n_outputs=1 gives
                       dt_rank + 4*d_state (matches mamba3_key.py)
    dt_proj.weight   (d_inner, dt_rank)          — stays real
    dt_proj.bias     (d_inner,)                  — unchanged
    A_log            (n_ssm_units, d_state, 2)   — complex [real, imag], imag=0
    D                (d_inner,)                  — unchanged (skip connection)
    out_proj.weight  (d_model, d_inner_out)      — d_inner_out = n_ssm_units*n_outputs
    dt_norm.weight   (d_inner, 2)                — complex RMSNorm, imag=0
    A_norm.weight    (d_state, 2)                — complex RMSNorm, imag=0
    B_norm.weight    (d_state, 2)                — complex RMSNorm, imag=0
    C_norm.weight    (d_state, 2)                — complex RMSNorm, imag=0

When n_inputs = n_outputs = 1 (default):
    n_ssm_units = d_inner, d_inner_out = d_inner
    A_log shape: (d_inner, d_state, 2)  — matches mamba3_key.py
    x_proj: (dt_rank + 4*d_state, d_inner)  — matches mamba3_key.py
    All other shapes identical to Mamba-2. Lossless warm start (imag=0).

Usage:
    block = Mamba3Block(d_model=64, d_state=16)
    out, present = block(x, use_cache=True)   # training / full sequence
    out, present = block(x_step, past_key_value=present, use_cache=True)  # inference
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.evolution.auto_targets import self_improve


def _to_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """torch.complex wrapper that handles BFloat16 by upcasting to float32.

    torch.complex only supports Half/Float/Double — BFloat16 inputs cause
    a RuntimeError. This helper upcasts to float32, constructs the complex
    tensor, and returns complex64 (which all downstream code handles).
    """
    if real.dtype == torch.bfloat16 or imag.dtype == torch.bfloat16:
        return torch.complex(real.float(), imag.float())
    return torch.complex(real, imag)


# ═══════════════════════════════════════════════════════════════════════════════
# Mamba3Cache — inference cache for the complex-valued recurrent state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Mamba3Cache:
    """Inference cache for Mamba-3 recurrent mode.

    Stores the complex SSM state as separate real and imaginary parts
    (for GPU compatibility — complex tensors on some backends need special
    handling) and the conv1d state for incremental decoding.

    Attributes:
        ssm_state_real: (B, n_ssm_units, d_state) real part of complex state h
        ssm_state_imag: (B, n_ssm_units, d_state) imaginary part of complex state h
        conv_state: (B, d_inner, d_conv-1) conv1d sliding window state
    """
    ssm_state_real: torch.Tensor | None = None
    ssm_state_imag: torch.Tensor | None = None
    conv_state: torch.Tensor | None = None

    @classmethod
    def create(
        cls,
        batch_size: int,
        n_ssm_units: int,
        d_state: int,
        d_inner: int,
        d_conv: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Mamba3Cache:
        """Create a zero-initialized cache."""
        return cls(
            ssm_state_real=torch.zeros(
                batch_size, n_ssm_units, d_state, device=device, dtype=dtype),
            ssm_state_imag=torch.zeros(
                batch_size, n_ssm_units, d_state, device=device, dtype=dtype),
            conv_state=torch.zeros(
                batch_size, d_inner, d_conv - 1, device=device, dtype=dtype),
        )

    @property
    def is_empty(self) -> bool:
        """True if the cache has no state (first step)."""
        return self.ssm_state_real is None

    def reset(self) -> None:
        """Clear all cached state."""
        self.ssm_state_real = None
        self.ssm_state_imag = None
        self.conv_state = None

    def to_complex_state(self) -> torch.Tensor:
        """Combine real + imag parts into a complex tensor (B, n_ssm_units, d_state)."""
        if self.ssm_state_real is None:
            raise RuntimeError("Cache is empty — cannot build complex state.")
        return _to_complex(self.ssm_state_real, self.ssm_state_imag)

    def from_complex_state(self, h: torch.Tensor) -> None:
        """Store a complex state tensor by splitting into real + imag parts."""
        self.ssm_state_real = h.real.contiguous()
        self.ssm_state_imag = h.imag.contiguous()

    def to_dict(self) -> dict:
        """Convert to a dict compatible with the past_key_value interface."""
        return {
            "ssm_state_real": self.ssm_state_real,
            "ssm_state_imag": self.ssm_state_imag,
            "conv_state": self.conv_state,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> Mamba3Cache | None:
        """Build a cache from a past_key_value dict (or None)."""
        if d is None:
            return None
        return cls(
            ssm_state_real=d.get("ssm_state_real"),
            ssm_state_imag=d.get("ssm_state_imag"),
            conv_state=d.get("conv_state"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Complex RMSNorm — operates on complex-valued inputs (stored as [..., 2])
# ═══════════════════════════════════════════════════════════════════════════════

def _evolve_inputs_complex_rmsnorm():
    """Representative cases for complex_rmsnorm: complex64 activations
    at mamba3 layer widths, real-pair gain tensors in fp32 and bf16."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    cases = []
    for shape in ((1, 8, 32), (2, 16, 64), (1, 64, 128), (4, 32, 96)):
        re = torch.randn(*shape, generator=g).to(dev)
        im = torch.randn(*shape, generator=g).to(dev)
        x = torch.complex(re, im)
        w = torch.randn(*shape, 2, generator=g).to(dev)
        cases.append((x, w, 1e-6))
    # bf16 gain path exercises the _to_complex upcast
    x = torch.complex(torch.randn(2, 8, 16, generator=g).to(dev),
                      torch.randn(2, 8, 16, generator=g).to(dev))
    w = torch.randn(2, 8, 16, 2, generator=g).to(dev).to(torch.bfloat16)
    cases.append((x, w, 1e-6))
    return cases


@self_improve(gen_input=_evolve_inputs_complex_rmsnorm, evals=80,
              deps=("_to_complex",),
              tags=("mamba3", "ssm", "norm", "kernel", "hot-path"),
              prompt=("Faster complex_rmsnorm(x, weight, eps): RMS-normal"
                      "ize a complex tensor then apply complex gain. "
                      "Seed materializes |x|^2, mean, rsqrt, complex mul "
                      "as separate kernels. Ideas: view_as_real to work "
                      "on (..., 2) floats end-to-end, fused complex mul "
                      "via real arithmetic, fewer temporaries. Complex64 "
                      "out, same values."))
def complex_rmsnorm(x: torch.Tensor, weight: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm for complex-valued tensors.

    Args:
        x: complex tensor (...,)
        weight: (..., 2) — complex gain [real, imag]
        eps: numerical stability epsilon
    Returns:
        normalized complex tensor (...,)
    """
    # Complex RMS: sqrt(E[|x|^2]) = sqrt(E[x * conj(x)])
    # |x|^2 = x.real^2 + x.imag^2
    x_mag_sq = x.real.pow(2) + x.imag.pow(2)  # (...,)
    rms_inv = x_mag_sq.mean(dim=-1, keepdim=True).add(eps).rsqrt()
    # Complex gain
    w = _to_complex(weight[..., 0], weight[..., 1])  # (...,)
    return x * rms_inv.to(x.dtype) * w


# ═══════════════════════════════════════════════════════════════════════════════
# Exponential-trapezoidal discretization
# ═══════════════════════════════════════════════════════════════════════════════

def _evolve_inputs_exp_discretize():
    """Representative cases for exp_trapezoidal_discretize: complex A/B
    at mamba3 dims (negative-real A), small positive deltas incl. the
    near-zero region where the trapezoidal correction hits its limit."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    cases = []
    for b, h, d_state, n_in in ((1, 4, 16, 8), (2, 8, 32, 16),
                                (1, 16, 64, 32)):
        A = torch.complex(
            -torch.rand(b, h, d_state, generator=g).to(dev) - 0.05,
            torch.randn(b, h, d_state, generator=g).to(dev))
        Bc = torch.complex(
            torch.randn(b, h, d_state, n_in, generator=g).to(dev),
            torch.randn(b, h, d_state, n_in, generator=g).to(dev))
        delta = (torch.rand(b, h, 1, generator=g).to(dev) * 0.1) + 1e-4
        cases.append((A, Bc, delta, 1e-12))
    # near-zero delta — correction factor must stay finite (L'Hopital)
    A = torch.complex(-torch.rand(1, 2, 8, generator=g).to(dev),
                      torch.randn(1, 2, 8, generator=g).to(dev))
    Bc = torch.complex(torch.randn(1, 2, 8, 4, generator=g).to(dev),
                       torch.randn(1, 2, 8, 4, generator=g).to(dev))
    cases.append((A, Bc,
                  torch.full((1, 2, 1), 1e-9, device=dev), 1e-12))
    return cases


@self_improve(gen_input=_evolve_inputs_exp_discretize, evals=80,
              tags=("mamba3", "ssm", "discretize", "kernel", "hot-path"),
              prompt=("Faster exp_trapezoidal_discretize(A, B, delta, "
                      "eps): A_bar=exp(dA); B_bar=expm1(dA)/(dA+eps)*"
                      "delta*B. Seed computes exp AND expm1 over complex "
                      "tensors plus several broadcasts. Ideas: reuse "
                      "exp(dA) to get expm1 (A_bar-1 — but watch the "
                      "small-|dA| accuracy the gate checks), "
                      "view_as_real arithmetic, precomputed broadcast "
                      "shapes, fuse into fewer kernels. Outputs must "
                      "match to 1e-5."))
def exp_trapezoidal_discretize(
    A: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponential-trapezoidal discretization for Mamba-3.

    Combines exponential interpolation with trapezoidal integration:
        A_bar = exp(delta * A)
        B_bar = (exp(delta*A) - 1) / (delta*A) * delta * B

    The correction factor (exp(da) - 1) / da approaches 1 as da -> 0
    (L'Hopital's rule), recovering the ZOH (zero-order hold) limit.
    For finite delta*A, this gives better accuracy than ZOH.

    Args:
        A: complex (..., d_state) — the SSM matrix A (negative real part)
        B: complex (..., d_state, n_inputs) — the input matrix B
        delta: real (...,) or (..., 1) — the timestep delta
        eps: small constant to avoid division by zero when delta*A ≈ 0
    Returns:
        A_bar: complex (..., d_state) — discretized A
        B_bar: complex (..., d_state, n_inputs) — discretized B
    """
    # delta * A — broadcast delta over d_state
    # delta: (..., 1), A: (..., d_state) -> da: (..., d_state)
    da = delta * A  # (..., d_state) complex
    A_bar = torch.exp(da)  # (..., d_state) complex

    # Trapezoidal correction: (exp(da) - 1) / da
    # Numerically stable: when |da| < eps, correction ≈ 1
    correction = (torch.expm1(da)) / (da + eps)  # (..., d_state) complex
    # expm1(da) = exp(da) - 1, more accurate for small da

    # B_bar = correction * delta * B
    # correction: (..., d_state), delta: (..., 1), B: (..., d_state, n_inputs)
    # Need to broadcast: correction -> (..., d_state, 1), delta -> (..., 1, 1)
    B_bar = correction.unsqueeze(-1) * delta.unsqueeze(-1) * B  # (..., d_state, n_inputs)

    return A_bar, B_bar


# ═══════════════════════════════════════════════════════════════════════════════
# Mamba3Block — the Mamba-3 SSM block
# ═══════════════════════════════════════════════════════════════════════════════

class Mamba3Block(nn.Module):
    """Mamba-3 SSM block with complex-valued states, MIMO, and exponential-trapezoidal discretization.

    Follows the same interface as the existing MambaLayer (Mamba-2) block:
        forward(x, past_key_value=None, use_cache=False, **kwargs) -> (out, present)

    The block supports:
    - Complex-valued SSM state (real + imaginary, stored separately for GPU compat)
    - MIMO: n_inputs -> n_outputs via matrix-valued B and C
    - Exponential-trapezoidal discretization (better than ZOH)
    - Training (full-sequence scan, differentiable) and inference (recurrent with cache)

    When n_inputs = n_outputs = 1 and all imaginary parts are zero, the block
    is identical to Mamba-2 (lossless warm start via mamba3_key.py).
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        bias: bool = False,
        conv_bias: bool = True,
        layer_idx: int = 0,
        norm_eps: float = 1e-6,
        n_inputs: int = 1,
        n_outputs: int = 1,
    ):
        """
        Args:
            d_model: model hidden dimension
            d_state: SSM state dimension (complex — effective capacity is 2*d_state)
            d_conv: causal conv1d kernel size
            expand: expansion factor (d_inner = expand * d_model)
            dt_rank: rank of delta projection ("auto" = ceil(d_model/16))
            bias: whether in_proj/out_proj have bias
            conv_bias: whether conv1d has bias
            layer_idx: layer index (for identification)
            norm_eps: RMSNorm epsilon
            n_inputs: MIMO input dimension (1 = scalar SSM, matches Mamba-2)
            n_outputs: MIMO output dimension (1 = scalar SSM, matches Mamba-2)
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.layer_idx = layer_idx
        self.norm_eps = norm_eps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # MIMO: d_inner channels are split into n_ssm_units parallel SSMs,
        # each processing n_inputs inputs and producing n_outputs outputs.
        assert self.d_inner % n_inputs == 0, (
            f"d_inner ({self.d_inner}) must be divisible by n_inputs ({n_inputs})")
        self.n_ssm_units = self.d_inner // n_inputs
        self.d_inner_out = self.n_ssm_units * n_outputs

        # dt_rank
        if dt_rank == "auto":
            self.dt_rank = max(1, (d_model + 15) // 16)
        else:
            self.dt_rank = int(dt_rank)

        # ── Projections (same structure as Mamba-2) ──
        # in_proj: d_model -> 2*d_inner (split into x [ssm] and z [gate])
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Depthwise causal conv1d
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            groups=self.d_inner, bias=conv_bias,
            padding=d_conv - 1,
        )

        # x_proj: d_inner -> dt_rank + 2*d_state*n_inputs + 2*d_state*n_outputs
        # (dt [real] + B [complex: d_state*n_inputs * 2] + C [complex: d_state*n_outputs * 2])
        # When n_inputs=n_outputs=1: dt_rank + 4*d_state (matches mamba3_key.py)
        x_proj_out = (self.dt_rank
                      + 2 * d_state * n_inputs   # B: real + imag
                      + 2 * d_state * n_outputs)  # C: real + imag
        self.x_proj = nn.Linear(self.d_inner, x_proj_out, bias=False)

        # dt_proj: dt_rank -> d_inner (stays real)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A_log: complex, stored as (n_ssm_units, d_state, 2) [real, imag]
        # S4D real init: A = -arange(1, d_state+1), A_log = log(-A) = log(arange)
        # Imaginary part is zero-init (lossless warm start from Mamba-2)
        A_real = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_real = A_real.repeat(self.n_ssm_units, 1)  # (n_ssm_units, d_state)
        A_imag = torch.zeros_like(A_real)
        self.A_log = nn.Parameter(torch.stack([A_real.log(), A_imag], dim=-1))  # (n_ssm_units, d_state, 2)

        # D: skip connection (d_inner,) — matches mamba3_key.py
        # For MIMO, D is used per-SSM-unit; we store (d_inner,) = (n_ssm_units * n_inputs,)
        # and reshape as needed.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # out_proj: d_inner_out -> d_model
        self.out_proj = nn.Linear(self.d_inner_out, d_model, bias=bias)

        # ── Complex RMSNorm weights (stored as [..., 2] [real, imag], imag=0) ──
        # dt_norm: (d_inner, 2) — on delta before dt_proj
        # A_norm: (d_state, 2) — on A before discretization
        # B_norm: (d_state, 2) — on B before scan
        # C_norm: (d_state, 2) — on C before scan
        # All init to real=1, imag=0 (identity = lossless warm start)
        self.dt_norm = nn.Parameter(torch.stack([
            torch.ones(self.d_inner), torch.zeros(self.d_inner)], dim=-1))
        self.A_norm = nn.Parameter(torch.stack([
            torch.ones(d_state), torch.zeros(d_state)], dim=-1))
        self.B_norm = nn.Parameter(torch.stack([
            torch.ones(d_state), torch.zeros(d_state)], dim=-1))
        self.C_norm = nn.Parameter(torch.stack([
            torch.ones(d_state), torch.zeros(d_state)], dim=-1))

        # Recurrent state (for incremental decoding, stored on the module)
        self._ssm_state_real: torch.Tensor | None = None
        self._ssm_state_imag: torch.Tensor | None = None
        self._conv_state: torch.Tensor | None = None
        self._conv_state_reset = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_A_complex(self) -> torch.Tensor:
        """Recover complex A from A_log parameter.

        A = -exp(A_log)  (S4D parameterization, ensures negative real part)
        A_log is stored as (n_ssm_units, d_state, 2) [real, imag].
        Returns: complex (n_ssm_units, d_state)
        """
        a_log_complex = _to_complex(self.A_log[..., 0], self.A_log[..., 1])
        return -torch.exp(a_log_complex)

    def reset_state(self) -> None:
        """Reset SSM and conv state (call at start of new generation)."""
        self._ssm_state_real = None
        self._ssm_state_imag = None
        self._conv_state = None

    # ── complex selective scan (reference, differentiable) ───────────────────

    def _complex_selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        h_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Complex-valued selective scan with exponential-trapezoidal discretization.

        Implements the Mamba-3 recurrence:
            h_t = A_bar * h_{t-1} + B_bar @ x_t    (complex)
            y_t = Re(C @ h_t) + D * x_t            (real output)

        This is a Python loop (differentiable, works for training and inference).
        For training, the full sequence is processed in one call (parallel scan
        is approximated by the differentiable loop — PyTorch autograd handles
        the unrolled computation graph).

        Args:
            x: (B, n_ssm_units, n_inputs, L) real — SSM input per unit
            delta: (B, n_ssm_units, L) real — timestep per unit
            A: (n_ssm_units, d_state) complex — SSM matrix A
            B: (B, n_ssm_units, d_state, n_inputs, L) complex — input matrix
            C: (B, n_ssm_units, n_outputs, d_state, L) complex — output matrix
            D: (n_ssm_units, n_inputs) real — skip connection
            h_init: (B, n_ssm_units, d_state) complex — initial state (for incremental)
        Returns:
            y: (B, n_ssm_units, n_outputs, L) real — SSM output
            h_final: (B, n_ssm_units, d_state) complex — final state
        """
        B_b, n_units, n_in, L = x.shape
        d_state = A.shape[-1]
        n_out = C.shape[2]

        # Initialize state
        if h_init is not None:
            h = h_init  # (B, n_ssm_units, d_state) complex
        else:
            # Use complex64 for all float types (smallest complex type in PyTorch;
            # float16/bfloat16 upcast to complex64, float64 uses complex128)
            _ctype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
            h = torch.zeros(B_b, n_units, d_state, device=x.device, dtype=_ctype)
        ys = []

        for t in range(L):
            dt = delta[:, :, t]  # (B, n_ssm_units)
            x_t = x[:, :, :, t]  # (B, n_ssm_units, n_inputs)
            B_t = B[:, :, :, :, t]  # (B, n_ssm_units, d_state, n_inputs)
            C_t = C[:, :, :, :, t]  # (B, n_ssm_units, n_outputs, d_state)

            # Exponential-trapezoidal discretization
            # A: (n_ssm_units, d_state), dt: (B, n_ssm_units)
            # da = dt * A -> (B, n_ssm_units, d_state)
            A_bar, B_bar = exp_trapezoidal_discretize(
                A.unsqueeze(0),  # (1, n_ssm_units, d_state)
                B_t,  # (B, n_ssm_units, d_state, n_inputs)
                dt.unsqueeze(-1),  # (B, n_ssm_units, 1)
            )  # A_bar: (B, n_ssm_units, d_state), B_bar: (B, n_ssm_units, d_state, n_inputs)

            # State update: h_t = A_bar * h_{t-1} + B_bar @ x_t
            # A_bar * h: (B, n_ssm_units, d_state) element-wise
            # B_bar @ x_t: (B, n_ssm_units, d_state, n_inputs) @ (B, n_ssm_units, n_inputs, 1)
            #            -> (B, n_ssm_units, d_state, 1) -> squeeze
            h = A_bar * h + (B_bar * x_t.unsqueeze(-2)).sum(dim=-1)
            # h: (B, n_ssm_units, d_state) complex

            # Output: y_t = Re(C_t @ h) + D * x_t
            # C_t @ h: (B, n_ssm_units, n_outputs, d_state) * (B, n_ssm_units, 1, d_state)
            #        -> sum over d_state -> (B, n_ssm_units, n_outputs)
            y_complex = (C_t * h.unsqueeze(-2)).sum(dim=-1)  # (B, n_ssm_units, n_outputs) complex
            # D: (n_ssm_units, n_inputs), x_t: (B, n_ssm_units, n_inputs)
            # When n_inputs == n_outputs, D * x_t matches y_complex.real directly.
            # When n_inputs != n_outputs, use a sum-projection for the skip.
            if n_in == n_out:
                y_t = y_complex.real + D.unsqueeze(0) * x_t  # (B, n_ssm_units, n_outputs)
            else:
                skip = (D.unsqueeze(0) * x_t).sum(dim=-1, keepdim=True)  # (B, n_ssm_units, 1)
                y_t = y_complex.real + skip.expand(-1, -1, n_out)  # (B, n_ssm_units, n_outputs)
            ys.append(y_t)

        y = torch.stack(ys, dim=-1)  # (B, n_ssm_units, n_outputs, L)
        return y, h

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: dict | None = None,
        use_cache: bool = False,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict | None]:
        """Forward pass.

        Args:
            x: (B, T, d_model)
            past_key_value: dict with 'ssm_state_real', 'ssm_state_imag', 'conv_state'
                            (for incremental decoding), or a Mamba3Cache
            use_cache: if True, return state for incremental decoding
            attention_bias: ignored (Mamba has no attention)
            position_ids: ignored (Mamba is position-agnostic)
        Returns:
            (out, present) — out is (B, T, d_model), present is state dict or None
        """
        B, T, _ = x.shape

        # in_proj -> split into x (ssm path) and z (gate)
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        # Cast to conv1d's dtype (ForgeQuant dequant may produce float32
        # while conv1d weights stay BFloat16 — mixed precision safety).
        xz = xz.to(self.conv1d.weight.dtype)
        x_ssm, z = xz.chunk(2, dim=-1)  # x first (ssm), z second (gate)

        # ── Conv1d (depthwise causal) ──
        if T == 1 and past_key_value is not None and "conv_state" in past_key_value:
            # Incremental decoding (T=1)
            conv_state = past_key_value["conv_state"]  # (B, d_inner, d_conv-1)
            x_t = x_ssm.transpose(1, 2)  # (B, d_inner, 1)
            full_input = torch.cat([conv_state, x_t], dim=-1)  # (B, d_inner, d_conv)
            new_conv_state = torch.cat([conv_state[:, :, 1:], x_t], dim=-1)
            w = self.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
            conv_out = (w * full_input).sum(dim=-1, keepdim=True)  # (B, d_inner, 1)
            if self.conv1d.bias is not None:
                conv_out = conv_out + self.conv1d.bias.unsqueeze(0).unsqueeze(-1)
            x = conv_out.transpose(1, 2)  # (B, 1, d_inner)
            present_conv_state = new_conv_state
        else:
            # Full sequence
            x = x_ssm.transpose(1, 2)  # (B, d_inner, T)
            cont = (past_key_value or {}).get("conv_state")
            if cont is not None and T > 1:
                # Continuation chunk: prepend committed conv context rather
                # than relying on the causal zero left-pad (same fix as
                # MambaLayer — chunked prefill / speculative verify).
                full = torch.cat([cont.to(x.device, x.dtype), x], dim=-1)
                x = F.conv1d(full, self.conv1d.weight, self.conv1d.bias,
                             groups=self.d_inner)
            else:
                x = self.conv1d(x)[:, :, :T]  # causal: trim right padding
            x = x.transpose(1, 2)  # (B, T, d_inner)
            if use_cache:
                # Save conv state (last d_conv-1 pre-conv inputs)
                x_pre_conv_t = x_ssm.transpose(1, 2)  # (B, d_inner, T)
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

        x = F.silu(x)  # (B, T, d_inner)

        # ── x_proj -> dt, B (complex), C (complex) ──
        x_proj_out = self.x_proj(x)  # (B, T, dt_rank + 2*d_state*(n_in+n_out))
        dt_dim = self.dt_rank
        b_dim = 2 * self.d_state * self.n_inputs   # B: real + imag
        c_dim = 2 * self.d_state * self.n_outputs   # C: real + imag
        delta, B_raw, C_raw = x_proj_out.split([dt_dim, b_dim, c_dim], dim=-1)

        # ── dt_proj -> delta (d_inner), then dt_norm, then softplus ──
        # dt_norm is (d_inner, 2) — applied AFTER dt_proj (Mamba-2 style).
        delta = self.dt_proj(delta)  # (B, T, d_inner)
        delta = self._rmsnorm_real(delta, self.dt_norm)  # (B, T, d_inner) real
        delta = F.softplus(delta)  # (B, T, d_inner)

        # ── B: reshape to complex (B, T, d_state, n_inputs) ──
        B_real = B_raw[..., :self.d_state * self.n_inputs].view(
            B, T, self.d_state, self.n_inputs)
        B_imag = B_raw[..., self.d_state * self.n_inputs:].view(
            B, T, self.d_state, self.n_inputs)
        B_c = _to_complex(B_real, B_imag)  # (B, T, d_state, n_inputs)
        # B_norm: complex RMSNorm over d_state (dim=-2 for this shape)
        B_c = self._complex_norm_dim(B_c, self.B_norm, dim=-2)

        # ── C: reshape to complex (B, T, n_outputs, d_state) ──
        C_real = C_raw[..., :self.d_state * self.n_outputs].view(
            B, T, self.n_outputs, self.d_state)
        C_imag = C_raw[..., self.d_state * self.n_outputs:].view(
            B, T, self.n_outputs, self.d_state)
        C_c = _to_complex(C_real, C_imag)  # (B, T, n_outputs, d_state)
        # C_norm: complex RMSNorm over d_state (dim=-1 for this shape)
        C_c = self._complex_norm_dim(C_c, self.C_norm, dim=-1)

        # ── A_norm and recover complex A ──
        A = self._get_A_complex()  # (n_ssm_units, d_state) complex
        # A_norm: complex RMSNorm on d_state
        a_weight = _to_complex(self.A_norm[..., 0], self.A_norm[..., 1])  # (d_state,)
        a_rms_inv = (A.real.pow(2) + A.imag.pow(2)).mean(dim=-1, keepdim=True).add(
            self.norm_eps).rsqrt()
        A = A * a_rms_inv.to(A.real.dtype) * a_weight

        # ── Prepare tensors for scan ──
        # x: (B, T, d_inner) -> (B, n_ssm_units, n_inputs, T)
        x_scan = x.view(B, T, self.n_ssm_units, self.n_inputs)
        x_scan = x_scan.permute(0, 2, 3, 1)  # (B, n_ssm_units, n_inputs, T)

        # delta: (B, T, d_inner) -> (B, n_ssm_units, T)
        delta_scan = delta.view(B, T, self.n_ssm_units, self.n_inputs).mean(dim=-1)
        delta_scan = delta_scan.permute(0, 2, 1)  # (B, n_ssm_units, T)

        # B: (B, T, d_state, n_inputs) -> (B, n_ssm_units, d_state, n_inputs, T)
        # B is shared across n_ssm_units (input-dependent, not per-unit)
        # Expand: (B, T, d_state, n_inputs) -> (B, 1, T, d_state, n_inputs) ->
        #         (B, n_ssm_units, d_state, n_inputs, T)
        B_scan = B_c.unsqueeze(1).expand(-1, self.n_ssm_units, T, self.d_state, self.n_inputs)
        B_scan = B_scan.permute(0, 1, 3, 4, 2)  # (B, n_ssm_units, d_state, n_inputs, T)

        # C: (B, T, n_outputs, d_state) -> (B, n_ssm_units, n_outputs, d_state, T)
        C_scan = C_c.unsqueeze(1).expand(-1, self.n_ssm_units, T, self.n_outputs, self.d_state)
        C_scan = C_scan.permute(0, 1, 3, 4, 2)  # (B, n_ssm_units, n_outputs, d_state, T)

        # D: (d_inner,) -> (n_ssm_units, n_inputs)
        D_scan = self.D.view(self.n_ssm_units, self.n_inputs)

        # ── Initial state (for incremental decoding) ──
        h_init = None
        if past_key_value is not None:
            sr = past_key_value.get("ssm_state_real")
            si = past_key_value.get("ssm_state_imag")
            if sr is not None and si is not None:
                h_init = _to_complex(sr, si)

        # ── Complex selective scan ──
        y, h_final = self._complex_selective_scan(
            x_scan, delta_scan, A, B_scan, C_scan, D_scan, h_init=h_init)
        # y: (B, n_ssm_units, n_outputs, T)

        # ── Reshape output ──
        y = y.permute(0, 3, 1, 2)  # (B, T, n_ssm_units, n_outputs)
        y = y.reshape(B, T, self.d_inner_out)  # (B, T, d_inner_out)

        # ── Gate with z ──
        # z: (B, T, d_inner). For MIMO, d_inner_out may != d_inner.
        # When n_inputs=n_outputs=1, d_inner_out=d_inner (matches Mamba-2).
        if self.d_inner_out == self.d_inner:
            y = y * F.silu(z)
        else:
            # MIMO: gate broadcasts — use mean of z gate per SSM unit
            z_gate = z.view(B, T, self.n_ssm_units, self.n_inputs).mean(dim=-1)
            z_gate = z_gate.unsqueeze(-1).expand(B, T, self.n_ssm_units, self.n_outputs)
            y = y * F.silu(z_gate.reshape(B, T, self.d_inner_out))

        # ── out_proj ──
        # The complex scan produces float32 (complex64 → real); cast back to
        # the projection's dtype so bf16/fp16 models don't hit a dtype
        # mismatch at out_proj.
        out = self.out_proj(y.to(self.out_proj.weight.dtype))  # (B, T, d_model)

        # ── Build present state ──
        present = None
        if use_cache:
            present = {
                "ssm_state_real": h_final.real.contiguous(),
                "ssm_state_imag": h_final.imag.contiguous(),
                "conv_state": present_conv_state,
            }

        return out, present

    # ── norm helpers ─────────────────────────────────────────────────────────

    def _rmsnorm_real(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm for a real-valued tensor with complex weight (use real part).

        Args:
            x: (..., D) real
            weight: (D, 2) — complex gain [real, imag]; we use real part
        Returns:
            (..., D) real
        """
        rms_inv = x.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        return x * rms_inv * weight[..., 0]

    def _complex_norm_dim(self, x: torch.Tensor, weight: torch.Tensor,
                          dim: int = -1) -> torch.Tensor:
        """Complex RMSNorm over a specified dimension.

        Args:
            x: (..., d_state, ...) complex
            weight: (d_state, 2) — complex gain [real, imag]
            dim: dimension to normalize over (must match weight's d_state)
        Returns:
            normalized complex tensor (same shape as x)
        """
        mag_sq = x.real.pow(2) + x.imag.pow(2)  # same shape as x
        rms_inv = mag_sq.mean(dim=dim, keepdim=True).add(self.norm_eps).rsqrt()
        w = _to_complex(weight[..., 0], weight[..., 1])  # (d_state,)
        # Reshape w to broadcast against the correct dim of x
        # w is (d_state,) — need to insert a singleton dim so it aligns with `dim`
        n_dims = x.dim()
        if dim < 0:
            dim = n_dims + dim
        w_shape = [1] * n_dims
        w_shape[dim] = w.shape[0]
        w = w.view(w_shape)
        return x * rms_inv.to(x.real.dtype) * w
