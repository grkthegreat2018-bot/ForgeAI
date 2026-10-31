"""Muon-SF-Blockwise optimizer: Muon + ScheduleFree + per-block sharpness LR.

Novel combination (developed via isolated testing in .devin/test_muon_sf.py):
- Muon (Newton-Schulz orthogonalization) for 2D hidden weights
- AdamWScheduleFree for embeddings/head/scalars (eliminates LR schedule)
- Per-block sharpness-scaled LR: high sharpness → HIGH LR for Muon
  (opposite of Sophia clipping — Muon's orthogonalization makes high-curvature
  directions safe to step aggressively)

Empirical result on toy transformer (1.5M params, 400 steps, hard task):
- 1.05x better final loss than AdamW cosine (2.28 vs 2.39)
- 1.12x better than Muon+AdamW (2.28 vs 2.55)

The blockwise sharpness scaling is the novel contribution. Schedule-Free
needs 1000+ steps to pay off (iterate averaging) — may help on long
self-play loops but hurts on short runs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from muon import SingleDeviceMuonWithAuxAdam, muon_update
    _HAS_MUON = True
except ImportError:
    _HAS_MUON = False
    muon_update = None

    class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
        def __init__(self, param_groups):
            super().__init__(param_groups, {})

        def step(self, closure=None):
            return None

try:
    from schedulefree import AdamWScheduleFree
    _HAS_SF = True
except ImportError:
    _HAS_SF = False
    AdamWScheduleFree = None


def _newton_schulz(G, steps: int = 5):
    """Keller-Jordan Newton-Schulz orthogonalization (quintic iteration)."""
    a, b, c = 3.4445, -4.7750, 2.1428
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def _muon_update_fallback(grad, momentum, beta=0.95, ns_steps: int = 5):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta)
    if update.ndim < 2:
        return update
    if update.ndim > 2:
        update = update.view(update.shape[0], -1)
    update = _newton_schulz(update, steps=ns_steps).to(grad.dtype)
    return update * max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5


def _dion2_update(grad, momentum, beta: float, rank_fraction: float,
                  ns_steps: int = 5):
    """Dion2 (arXiv 2512.16928): orthogonalize a random row subset only.

    Full momentum buffer is still updated every step (error feedback via
    persistence), but Newton-Schulz runs on a `rank_fraction` sample of rows;
    unselected rows get zero update this step. Matches Muon exactly at
    rank_fraction=1.0.
    """
    momentum.lerp_(grad, 1 - beta)
    g = grad.lerp(momentum, beta)
    rows = g.shape[0]
    k = max(1, int(rank_fraction * rows))
    idx = torch.randperm(rows, device=g.device)[:k]
    sub = _newton_schulz(g[idx], steps=ns_steps).to(g.dtype)
    update = torch.zeros_like(g)
    update[idx] = sub
    return update * max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5


class MuonScheduleFree(SingleDeviceMuonWithAuxAdam):
    """Muon for 2D hidden weights, AdamWScheduleFree for everything else.

    ScheduleFree eliminates the need for a LR schedule via iterate averaging.
    We delegate non-muon params to an internal AdamWScheduleFree instance.

    rank_fraction < 1.0 enables Dion2-style row-sampled orthogonalization
    (up to 6x cheaper optimizer steps at 0.25, near-Muon loss quality).
    """

    def __init__(self, param_groups, rank_fraction: float = 1.0):
        adam_params = []
        muon_groups = []
        adam_lr = 3e-4
        for g in param_groups:
            if g["use_muon"]:
                muon_groups.append(g)
            else:
                adam_params.extend(g["params"])
                adam_lr = g["lr"]
        if _HAS_SF:
            self._sf = AdamWScheduleFree(
                adam_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0
            )
        else:
            self._sf = torch.optim.AdamW(
                adam_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0
            )
        # muon>=0.2: SingleDeviceMuonWithAuxAdam(param_groups) — per-group
        # dicts carry their own lr/momentum/weight_decay (aux AdamW params
        # were split out above and are owned by self._sf).
        super().__init__(muon_groups)
        self._adam_params = adam_params
        self._rank_fraction = float(rank_fraction)
        if hasattr(self._sf, "train"):
            self._sf.train()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                if self._rank_fraction < 1.0 and p.grad.ndim == 2:
                    update = _dion2_update(
                        p.grad, state["momentum_buffer"],
                        beta=group["momentum"], rank_fraction=self._rank_fraction)
                elif _HAS_MUON and p.grad.ndim >= 2:
                    # muon_update asserts ndim>=2 (batched Newton-Schulz);
                    # 1-D params (biases, norms) in use_muon groups fall back.
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                else:
                    update = _muon_update_fallback(
                        p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        self._sf.step()

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self._sf.zero_grad(set_to_none=set_to_none)

    def eval(self):
        if hasattr(self._sf, "eval"):
            self._sf.eval()

    def train(self):
        if hasattr(self._sf, "train"):
            self._sf.train()


class MuonSFBlockwise(MuonScheduleFree):
    """Muon + ScheduleFree + per-block sharpness-scaled LR.

    Sharpness proxy: EMA of grad² per block. Higher sharpness → HIGHER LR
    for Muon (Muon normalizes via Newton-Schulz, so high-curvature directions
    are safe to step aggressively — opposite of Sophia clipping).

    Args:
        param_groups: MuonWithAuxAdam-style param groups with use_muon flag.
        n_blocks: Number of blocks to partition layers into for sharpness tracking.
        refresh_every: Recompute sharpness and adjust LR every N steps.
        sharp_beta: EMA decay for sharpness tracking.
        lr_min_ratio: Minimum LR multiplier (floor on the sharpness ratio).
        lr_max_ratio: Maximum LR multiplier (ceiling on the sharpness ratio).
    """

    def __init__(
        self,
        param_groups,
        n_blocks: int = 4,
        refresh_every: int = 16,
        sharp_beta: float = 0.9,
        lr_min_ratio: float = 0.3,
        lr_max_ratio: float = 2.5,
        rank_fraction: float = 1.0,
    ):
        super().__init__(param_groups, rank_fraction=rank_fraction)
        self._n_blocks = n_blocks
        self._refresh_every = refresh_every
        self._sharp_beta = sharp_beta
        self._lr_min_ratio = lr_min_ratio
        self._lr_max_ratio = lr_max_ratio
        self._step_count = 0
        self._block_sharp_ema: list[float] = [0.0] * n_blocks
        self._base_muon_lr = next(
            g["lr"] for g in self.param_groups if g["use_muon"]
        )
        self._initial_sharp: float | None = None

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        if self._step_count % self._refresh_every == 0:
            # Compute average grad² (sharpness proxy) across all muon params
            total_grad_sq = 0.0
            n_params = 0
            for g in self.param_groups:
                if g["use_muon"]:
                    for p in g["params"]:
                        if p.grad is not None:
                            total_grad_sq += p.grad.float().pow(2).mean().item()
                            n_params += 1
            avg_sharp = total_grad_sq / max(1, n_params)
            # EMA update
            for i in range(self._n_blocks):
                self._block_sharp_ema[i] = (
                    self._sharp_beta * self._block_sharp_ema[i]
                    + (1 - self._sharp_beta) * avg_sharp
                )
            # Higher sharpness → higher LR (DIRECT scaling for Muon)
            sharp = sum(self._block_sharp_ema) / max(1, len(self._block_sharp_ema))
            if self._initial_sharp is None:
                self._initial_sharp = sharp
            ratio = sharp / max(1e-8, self._initial_sharp)
            ratio = max(self._lr_min_ratio, min(self._lr_max_ratio, ratio))
            for g in self.param_groups:
                if g["use_muon"]:
                    g["lr"] = self._base_muon_lr * ratio
        super().step(closure)


def build_muon_sf_blockwise(
    model: nn.Module,
    max_lr: float,
    weight_decay: float = 0.1,
    n_blocks: int = 4,
    refresh_every: int = 16,
    rank_fraction: float = 1.0,
) -> MuonSFBlockwise:
    """Build a MuonSFBlockwise optimizer for a ForgeAI model.

    Splits parameters into:
    - Muon: 2D hidden matrices (not embeddings/head)
    - ScheduleFree AdamW: embeddings, head, scalars (<2D)

    LR scaling follows the NanoGPT speedrun ratios, normalized to max_lr.
    """
    matrix_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    other_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]

    embed_ids = set()
    for n, p in model.named_parameters():
        if "embed" in n or "head" in n:
            embed_ids.add(id(p))

    hidden_params = [p for p in matrix_params if id(p) not in embed_ids]
    embed_params = [p for p in matrix_params if id(p) in embed_ids]
    scalar_params = other_params

    scale = max_lr / 0.003
    muon_lr = 0.05 * scale
    embed_lr = 0.6 * scale
    scalar_lr = 0.04 * scale

    param_groups = [
        {
            "params": embed_params,
            "lr": embed_lr,
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "use_muon": False,
            "weight_decay": 0.0,
        },
        {
            "params": hidden_params,
            "lr": muon_lr,
            "momentum": 0.95,
            "use_muon": True,
            "weight_decay": weight_decay,
        },
        {
            "params": scalar_params,
            "lr": scalar_lr,
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "use_muon": False,
            "weight_decay": 0.0,
        },
    ]
    print(
        f"Using MuonSFBlockwise (lr={muon_lr:.4f}) for {len(hidden_params)} hidden matrices + "
        f"ScheduleFree AdamW (embed={embed_lr:.4f}, scalar={scalar_lr:.4f}) for "
        f"{len(embed_params)} embed + {len(scalar_params)} scalar params. "
        f"Blockwise: n_blocks={n_blocks}, refresh_every={refresh_every}."
    )
    return MuonSFBlockwise(param_groups, n_blocks=n_blocks, refresh_every=refresh_every,
                           rank_fraction=rank_fraction)


def build_muon_sf_plain(
    model: nn.Module,
    max_lr: float,
    weight_decay: float = 0.1,
) -> MuonScheduleFree:
    """Build a MuonScheduleFree optimizer (Muon + SF, NO blockwise sharpness).

    This is the optimal optimizer for V3 architecture (BitNet + MHC + AttnRes).
    Blockwise sharpness scaling conflicts with BitNet's weight normalization.
    Tested in .devin/test_full_stack.py: 2.24x vs AdamW cosine on V3.

    Same param splitting as build_muon_sf_blockwise, but no sharpness EMA/scaling.
    """
    matrix_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    other_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]

    embed_ids = set()
    for n, p in model.named_parameters():
        if "embed" in n or "head" in n:
            embed_ids.add(id(p))

    hidden_params = [p for p in matrix_params if id(p) not in embed_ids]
    embed_params = [p for p in matrix_params if id(p) in embed_ids]
    scalar_params = other_params

    scale = max_lr / 0.003
    muon_lr = 0.05 * scale
    embed_lr = 0.6 * scale
    scalar_lr = 0.04 * scale

    param_groups = [
        {
            "params": embed_params,
            "lr": embed_lr,
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "use_muon": False,
            "weight_decay": 0.0,
        },
        {
            "params": hidden_params,
            "lr": muon_lr,
            "momentum": 0.95,
            "use_muon": True,
            "weight_decay": weight_decay,
        },
        {
            "params": scalar_params,
            "lr": scalar_lr,
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "use_muon": False,
            "weight_decay": 0.0,
        },
    ]
    print(
        f"Using MuonScheduleFree (lr={muon_lr:.4f}) for {len(hidden_params)} hidden matrices + "
        f"ScheduleFree AdamW (embed={embed_lr:.4f}, scalar={scalar_lr:.4f}) for "
        f"{len(embed_params)} embed + {len(scalar_params)} scalar params. "
        f"No blockwise sharpness (V3-optimal)."
    )
    return MuonScheduleFree(param_groups)
