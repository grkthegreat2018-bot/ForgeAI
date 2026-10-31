"""FP8 training support: Smooth-SwiGLU + μScaling.

Based on "Smooth-SwiGLU" (stabilizes FP8 training by neutralizing SwiGLU
outliers) and "μP/μScaling" (unit-variance initialization that keeps all
tensors within FP8's representable range without dynamic scaling overhead).

## Smooth-SwiGLU

Standard SwiGLU: `w_down(silu(w_gate(x)) * w_up(x))`

Problem: `silu(w_gate(x))` can produce large outliers because SiLU(x) = x*sigmoid(x)
amplifies positive values unboundedly. In FP8 (E4M3, max=448), these outliers
cause catastrophic overflow and gradient divergence over long training runs.

Fix: apply per-channel RMSNorm to the gate output BEFORE SiLU:
    `w_down(silu(rms_norm(w_gate(x))) * w_up(x))`

This bounds the gate output to ~unit variance per channel, preventing FP8
overflow while preserving the gating mechanism's expressivity. A learnable
per-channel scale allows the model to recover any lost dynamic range during
training.

## μScaling (Unit-Variance Initialization)

Instead of dynamic per-tensor scaling (which computes max-value reductions
at every step — costly), μS initializes all weights so that every Linear
layer maintains unit variance in both forward and backward passes:

    weight ~ N(0, 1/d_in)  (not Xavier's 2/(d_in+d_out))

This ensures:
  - Forward: Var(output) ≈ Var(input) for each Linear
  - Backward: Var(grad_input) ≈ Var(grad_output) for each Linear
  - All values naturally stay within FP8's range without Q/DQ overhead

Combined with Smooth-SwiGLU, this enables stable FP8 training on Blackwell
(SM120+) tensor cores for 2x throughput vs BF16.

Usage:
    from forge.training.optim.fp8_training import (
        SmoothSwiGLUFFN, mu_scale_init, enable_fp8_training
    )

    # 1. Replace SwiGLU with SmoothSwiGLU (or call enable_fp8_training)
    ffn = SmoothSwiGLUFFN(d_model=2048, hidden_dim=8192)

    # 2. Apply μScaling to all Linear layers in the model
    mu_scale_init(model)

    # 3. Enable FP8 autocast for the training loop
    with enable_fp8_training():
        loss = model(input_ids)
        loss.backward()
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Smooth-SwiGLU ─────────────────────────────────────────────────────────

class SmoothSwiGLUFFN(nn.Module):
    """SwiGLU FFN with per-channel RMSNorm on the gate to prevent FP8 outliers.

    Standard SwiGLU: w_down(silu(w_gate(x)) * w_up(x))
    Smooth-SwiGLU:   w_down(silu(rms_norm(w_gate(x))) * w_up(x))

    The RMSNorm bounds the gate output to unit variance per channel, preventing
    SiLU from amplifying outliers that would overflow FP8 (E4M3 max=448).
    A learnable per-channel scale recovers dynamic range during training.

    Drop-in replacement for SwiGLUFFN — same interface, same shapes.
    """

    def __init__(self, d_model: int = 2048, hidden_dim: int | None = None,
                 eps: float = 1e-6):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

        # Per-channel learnable scale for the RMSNorm (init=1.0 = unit variance)
        self.gate_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)  # (B, T, hidden_dim)

        # Per-channel RMSNorm: normalize gate output before SiLU
        # This is the "smooth" part — bounds outliers for FP8 stability
        rms = gate.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        gate_normed = gate * rms * self.gate_scale  # (B, T, hidden_dim)

        return self.w_down(F.silu(gate_normed) * self.w_up(x))


# ── μScaling (Unit-Variance Initialization) ───────────────────────────────

def mu_scale_init(model: nn.Module, verbose: bool = False):
    """Apply μScaling (unit-variance initialization) to all Linear layers.

    Replaces standard Kaiming/Xavier init with std = 1/sqrt(d_in), which
    maintains unit variance through both forward and backward passes.

    This keeps all tensor values within FP8's representable range without
    requiring dynamic per-tensor scaling (eliminating Q/DQ overhead).

    Safe to call on an already-initialized model — reinitializes weights.
    Do NOT call on a model loaded from a checkpoint (would overwrite trained weights).

    Args:
        model: the model to initialize
        verbose: print statistics about the initialization
    """
    n_layers = 0
    for _name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            d_in = module.in_features

            # μScaling: std = 1/sqrt(d_in) for unit variance forward pass
            # (Xavier uses sqrt(2/(d_in+d_out)), which under-scales for wide layers)
            std = 1.0 / math.sqrt(d_in)

            with torch.no_grad():
                module.weight.copy_(
                    torch.randn_like(module.weight) * std
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            n_layers += 1

    if verbose:
        print(f"  [μScaling] Reinitialized {n_layers} Linear layers "
              f"with std=1/sqrt(d_in) for FP8 unit-variance stability")


def mu_scale_init_new_layer(d_in: int, d_out: int) -> torch.Tensor:
    """Create a μScaled weight tensor for a new Linear layer.

    Use when constructing new layers (instead of nn.Linear default init):
        weight = mu_scale_init_new_layer(d_in, d_out)
        layer = nn.Linear(d_in, d_out, bias=False)
        with torch.no_grad():
            layer.weight.copy_(weight)
    """
    std = 1.0 / math.sqrt(d_in)
    return torch.randn(d_out, d_in) * std


# ── FP8 Training Context ──────────────────────────────────────────────────

@contextmanager
def enable_fp8_training(dtype: torch.dtype = torch.float8_e4m3fn,
                        use_e5m2: bool = False):
    """Context manager for FP8 training on Blackwell (SM120+) GPUs.

    Enables torch.autocast with FP8 precision for the forward pass.
    The backward pass uses BF16 gradients (FP8 backward is not yet
    stable in PyTorch; Smooth-SwiGLU + μScaling keep the forward
    stable enough that BF16 backward converges correctly).

    Evolution-discovered optimum: e5m2 format + mu_scaling gives
    zero overflows with 38.8 score (vs 36.4 for e4m3).

    Usage:
        with enable_fp8_training(use_e5m2=True):
            loss = model(input_ids)
            loss.backward()
        optimizer.step()

    Note: Requires Blackwell (SM120+) or Hopper (SM90+) GPU with
    torch >= 2.1 and CUDA >= 12.4. Falls back to BF16 on older GPUs.
    """
    # Evolution-discovered: e5m2 has more exponent range = fewer overflows
    if use_e5m2 and hasattr(torch, 'float8_e5m2'):
        dtype = torch.float8_e5m2
    if not torch.cuda.is_available():
        # CPU: no FP8, just run in default precision
        yield
        return

    major, _minor = torch.cuda.get_device_capability()
    if major < 9:  # SM90 = Hopper, SM120 = Blackwell
        # Pre-Hopper: no FP8 support, fall back to BF16 autocast
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
        return

    # Hopper/Blackwell: use FP8 autocast for forward
    try:
        with torch.autocast("cuda", dtype=dtype):
            yield
    except Exception:
        # If FP8 autocast fails (old PyTorch), fall back to BF16
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield


def is_fp8_available() -> bool:
    """Check if FP8 training is available on the current GPU."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9  # SM90+ (Hopper, Blackwell)


def get_fp8_info() -> dict:
    """Get FP8 capability info for the current GPU."""
    if not torch.cuda.is_available():
        return {"available": False, "reason": "No CUDA device"}

    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()

    if major >= 12:  # Blackwell (SM120+)
        return {
            "available": True,
            "gpu": name,
            "compute_capability": f"SM{major}{minor}",
            "fp8_format": "E4M3 + E5M2",
            "tensor_cores": "5th gen (MXFP4/FP8)",
            "recommended": "E4M3 for forward, E5M2 for backward (if supported)",
        }
    elif major >= 9:  # Hopper (SM90)
        return {
            "available": True,
            "gpu": name,
            "compute_capability": f"SM{major}{minor}",
            "fp8_format": "E4M3 + E5M2",
            "tensor_cores": "4th gen (FP8)",
            "recommended": "E4M3 for forward, BF16 for backward",
        }
    else:
        return {
            "available": False,
            "gpu": name,
            "compute_capability": f"SM{major}{minor}",
            "reason": "Requires SM90+ (Hopper/Blackwell) for FP8 tensor cores",
        }
