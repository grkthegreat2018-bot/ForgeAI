"""Fira-NLRQ: Fira-style optimizer for models with NLRQ factorized layers.

Original intent (Fira, NeurIPS 2025 — https://arxiv.org/abs/2410.01623): run a
low-rank optimizer in a projection subspace and rescale to get full-rank
training quality at low-rank memory cost.

Why Fira degenerates to AdamW here: NLRQLinear's trainable parameters are
already low-rank — only `S` (the (rank,) singular values) is an nn.Parameter.
`U_q`/`V_q` are INT8 buffers and `U_scale`/`V_scale` are fp16 buffers, none of
which receive gradients. There is no full-rank weight to project, so the Fira
projection basis (originally V_q) and norm-based rescaling have nothing to
operate on. An earlier revision of this file allocated projection states
(`exp_avg_proj`, `grad_norm_ema`, `V_basis`) that were never read — that dead
machinery has been removed.

What remains: a two-group AdamW —
  - group 1: NLRQ params (S vectors + optional biases)
  - group 2: everything else (embeddings, norms, attention, head)
with decoupled weight decay, which is the correct low-memory optimizer for
NLRQ models given their current trainability surface.

If full-rank training through the INT8 factors is added later (e.g. STE
gradients to U_q/V_q master weights), this is the place to reintroduce Fira's
norm-based rescaling.

Usage:
    from forge.training.optim.fira_nlrq import configure_fira_nlrq
    optimizer = configure_fira_nlrq(model, lr=3e-4, weight_decay=0.01)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class FiraNLRQ(Optimizer):
    """Two-group AdamW for NLRQ models: NLRQ params vs everything else.

    Args:
        model: the model (NLRQLinear layers detected automatically)
        lr: learning rate
        betas: Adam beta1, beta2
        eps: Adam epsilon
        weight_decay: decoupled weight decay
        verbose: print group stats
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        verbose: bool = True,
    ):
        from forge.keys.compression.nlrq_ffn_key import NLRQLinear

        nlrq_param_ids: set[int] = set()
        n_nlrq = 0
        n_nlrq_layers = 0
        for module in model.modules():
            if isinstance(module, NLRQLinear):
                n_nlrq_layers += 1
                for p in module.parameters():
                    if p.requires_grad:
                        nlrq_param_ids.add(id(p))
                        n_nlrq += p.numel()

        nlrq_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) in nlrq_param_ids]
        other_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in nlrq_param_ids]
        n_other = sum(p.numel() for p in other_params)

        param_groups = [
            {"params": nlrq_params, "use_fira": True},
            {"params": other_params, "use_fira": False},
        ]
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(param_groups, defaults)

        if verbose:
            print(f"FiraNLRQ: {n_nlrq_layers} NLRQ layers, "
                  f"{n_nlrq/1e6:.1f}M NLRQ params, {n_other/1e6:.1f}M other params "
                  f"(Fira projection inactive — NLRQ params are already low-rank)")

    @torch.no_grad()
    def step(self, closure=None):
        """One AdamW step over both groups."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                grad = p.grad

                if wd > 0:
                    p.mul_(1 - lr * wd)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_c1 = 1 - beta1 ** step
                bias_c2 = 1 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                step_size = lr / bias_c1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def configure_fira_nlrq(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
) -> FiraNLRQ:
    """Configure the Fira-NLRQ optimizer for a ForgeAI model with NLRQ layers."""
    return FiraNLRQ(model, lr=lr, weight_decay=weight_decay, verbose=True)
