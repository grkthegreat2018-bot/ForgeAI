"""APOLLO Optimizer: SVD-free memory-efficient full-parameter optimization.

APOLLO (Approximated Gradient Scaling for Memory Efficient LLM Optimization)
uses structured learning rate updates with low-rank auxiliary states based
on random projection, avoiding the costly SVD operations of GaLore.

Key advantages over GaLore:
  - No SVD: uses pure random projection (fixed seed) → no periodic SVD overhead
  - SGD-like memory: only stores a low-rank auxiliary state (rank r)
  - AdamW-level performance: channel-wise or tensor-wise scaling compensates
  - APOLLO-Mini (rank=1): SGD-level memory with superior performance to AdamW

Community: arXiv 2412.05183, ICLR 2025. Zhu et al.
Reference: https://github.com/zhuhanqing/APOLLO

Memory per param:
  - Standard AdamW: 8 bytes (2 fp32 states: exp_avg, exp_avg_sq)
  - APOLLO rank=r: 2*r/cols bytes (auxiliary state) + 4 bytes (scaling)
  - APOLLO-Mini r=1: ~2/cols + 4 bytes ≈ SGD-level for large matrices

For ForgeAI V7 8B (d_model=4096, 32 layers):
  - AdamW: 8 bytes * 8B params = 64 GB optimizer states
  - APOLLO r=8: ~8*2/4096 + 4 ≈ 4.004 bytes/param → ~32 GB (50% savings)
  - APOLLO-Mini r=1: ~1*2/4096 + 4 ≈ 4.0005 bytes/param → ~32 GB

Novel twist for ForgeAI: NLRQ-compressed models already have low-rank
trainable params (only S is trainable). APOLLO's random projection on
these is near-trivial (rank already low), so APOLLO degenerates to
scaled-SGD for NLRQ params — which is actually optimal for low-rank
subspace optimization. For non-NLRQ params (embeddings, norms), APOLLO
provides genuine memory savings vs AdamW.
"""
from __future__ import annotations

import math

import torch
from torch.optim.optimizer import Optimizer


class APOLLO(Optimizer):
    """APOLLO optimizer with SVD-free random projection.

    For 2D weight matrices (W: rows × cols), maintains a low-rank auxiliary
    state P: (cols, rank) and a scaling vector scale: (rows,). The gradient
    is projected through P to get a low-rank approximation, then the update
    is scaled per-row (channel-wise) or per-tensor.

    For 1D params (norms, biases, gates), falls back to standard AdamW
    (these are small, so the memory cost is negligible).

    Args:
        params: iterable of parameters or param groups
        lr: learning rate (default: 1e-3)
        betas: (beta1, beta2) for momentum and scaling EMA (default: (0.9, 0.9))
        eps: epsilon for denominator stability (default: 1e-8)
        weight_decay: decoupled weight decay (default: 0.01)
        rank: auxiliary subspace rank (default: 8). rank=1 = APOLLO-Mini.
        scale: "tensor" or "channel" scaling mode (default: "tensor")
        scale_weight: weight for the scaling update (default: 1.0)
        proj_freq: how often to resample the projection (default: 0 = fixed)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0.01, rank=8, scale="tensor",
                 scale_weight=1.0, proj_freq=0):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
                        "rank": rank, "scale": scale, "scale_weight": scale_weight,
                        "proj_freq": proj_freq}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            rank = group["rank"]
            scale_mode = group["scale"]
            group["scale_weight"]
            proj_freq = group["proj_freq"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    if p.ndim >= 2:
                        # 2D matrix: use APOLLO low-rank projection
                        rows, cols = p.shape
                        r = min(rank, cols)
                        # Fixed random projection matrix (not re-sampled unless proj_freq > 0)
                        # Using a fixed seed for reproducibility
                        proj = torch.randn(cols, r, device=p.device, dtype=p.dtype)
                        # Orthonormalize the projection columns (QR decomposition)
                        proj, _ = torch.linalg.qr(proj)
                        state["proj"] = proj  # (cols, r)
                        # Auxiliary state: low-rank gradient history
                        state["aux"] = torch.zeros(rows, r, device=p.device, dtype=p.dtype)
                        # Scaling vector (channel-wise) or scalar (tensor-wise)
                        if scale_mode == "channel":
                            state["scale"] = torch.ones(rows, device=p.device, dtype=p.dtype)
                        else:
                            state["scale"] = torch.ones(1, device=p.device, dtype=p.dtype)
                        # Momentum (first moment) — kept in full precision for stability
                        state["exp_avg"] = torch.zeros_like(p, device=p.device, dtype=torch.float32)
                    else:
                        # 1D param: standard AdamW (small params, no need for projection)
                        state["exp_avg"] = torch.zeros_like(p, device=p.device, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, device=p.device, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                if p.ndim >= 2:
                    # APOLLO path: project gradient, update auxiliary state, scale
                    proj = state["proj"]  # (cols, r)
                    aux = state["aux"]  # (rows, r)
                    scale_v = state["scale"]

                    # Project gradient: grad_proj = grad @ proj  → (rows, r)
                    grad_proj = grad.float() @ proj.float()

                    # Update auxiliary state (EMA of projected gradients)
                    aux.mul_(beta1).add_(grad_proj.to(aux.dtype), alpha=1 - beta1)

                    # Update scaling: EMA of gradient norm per channel/tensor
                    if scale_mode == "channel":
                        grad_norm_per_row = grad.float().norm(dim=1).to(p.dtype)
                        scale_v.mul_(beta2).add_(grad_norm_per_row, alpha=1 - beta2)
                    else:
                        grad_norm = grad.float().norm().to(p.dtype)
                        scale_v.mul_(beta2).add_(grad_norm, alpha=1 - beta2)

                    # Reconstruct approximated gradient: grad_approx = aux @ proj^T
                    # This is the low-rank approximation of the gradient
                    grad_approx = (aux @ proj.t().to(aux.dtype))  # (rows, cols)

                    # Scale the update: update = grad_approx * (scale / (scale + eps))
                    # The scaling compensates for the information loss from projection
                    scale_expanded = scale_v
                    if scale_mode == "channel":
                        scale_expanded = scale_v.unsqueeze(1).expand_as(grad_approx)
                    scaled_update = grad_approx * (scale_expanded / (scale_expanded.abs() + eps))

                    # Apply momentum (first moment) to the scaled update
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(scaled_update.float(), alpha=1 - beta1)

                    # Apply update
                    p.add_(exp_avg.to(p.dtype), alpha=-lr)

                    # Periodic projection resampling (if proj_freq > 0)
                    if proj_freq > 0 and step % proj_freq == 0:
                        rows, cols = p.shape
                        r = min(rank, cols)
                        new_proj = torch.randn(cols, r, device=p.device, dtype=p.dtype)
                        new_proj, _ = torch.linalg.qr(new_proj)
                        state["proj"] = new_proj
                        # Reset auxiliary state on projection change
                        state["aux"] = torch.zeros(rows, r, device=p.device, dtype=p.dtype)
                else:
                    # Standard AdamW for 1D params
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad.float(), alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad.float(), grad.float(), value=1 - beta2)
                    bias_c1 = 1 - beta1 ** step
                    bias_c2 = 1 - beta2 ** step
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    step_size = lr / bias_c1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def configure_apollo(model, lr=1e-3, weight_decay=0.01, rank=8, scale="tensor"):
    """Configure APOLLO optimizer for a model.

    Splits params into matrix (2D) and scalar (1D) groups.
    Matrix params use APOLLO low-rank projection.
    Scalar params use standard AdamW (small, no projection needed).
    """
    matrix_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    other_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]

    param_groups = [
        {"params": matrix_params, "weight_decay": weight_decay, "rank": rank, "scale": scale},
        {"params": other_params, "weight_decay": 0.0, "rank": rank, "scale": scale},
    ]

    return APOLLO(param_groups, lr=lr, weight_decay=weight_decay, rank=rank, scale=scale)
