"""Schedule-Free / spectral optimizers: SF-NorMuon, AMUSE, MONA, SAMuon-lite.

Based on four 2026 papers:
  1. SF-NorMuon (arXiv 2605.23061): schedule-free spectral optimizer with
     per-neuron normalization. Matches tuned AdamW across 1-8× Chinchilla horizons.
  2. AMUSE (arXiv 2605.22432): Anytime Muon + Stable gradient Evaluation.
     Integrates Muon's rapid bulk progress with Schedule-Free averaging.
     Time-varying interpolation coefficient: fast Muon → stable averaged.
  3. MONA (arXiv 2605.26842): Muon + Nesterov Acceleration.
     Adds acceleration term from EMA of gradient differences.
     Escapes sharp minima while preserving spectral-norm regularization.
     SOTA on MoE pretraining 1B-68B.
  4. SAMuon-lite (arXiv 2608.25990): Spectral-Aware Muon. Splits the
     orthogonalized update into the momentum buffer's top singular
     direction ("head") and the residual "bulk", then amplifies the bulk
     by bulk_gain ≈ 1.5. Reports 13–24% fewer tokens to the same loss.

The first three eliminate learning-rate schedules (anytime training):
  - No warmup-stable-decay needed
  - Checkpoint quality is high at ANY point during training
  - 31% faster to target loss at 1000 TPP (ScheduleFree+)

For our 1.2B model:
  - Current: AdamW with cosine schedule (path-dependent, re-tuning needed)
  - SF-NorMuon: no schedule, matches tuned AdamW, anytime checkpoints
  - AMUSE: Muon speed + Schedule-Free stability, no schedule
  - MONA: Muon + Nesterov, escapes sharp minima, SOTA on MoE
  - SAMuon-lite: drop-in Muon variant, 13–24% token savings via bulk gain
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


def _newton_schulz_5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration for matrix orthogonalization (Muon core).

    Computes G @ (G^T G)^(-1/2) ≈ orthogonalized G.
    5 steps is sufficient for good convergence.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Normalize so spectral norm ≤ 1
    X = X / (X.norm() + 1e-7)
    # Transpose if tall (more rows than cols) for efficiency
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.bfloat16()


def _per_neuron_normalize(G: torch.Tensor) -> torch.Tensor:
    """Per-neuron (per-row) normalization (NorMuon).

    Normalizes each row of the gradient to unit norm before orthogonalization.
    This improves conditioning and convergence.
    """
    if G.ndim != 2:
        return G
    norms = G.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return G / norms


def _top_singular_pair(G: torch.Tensor, iters: int = 3):
    """Top singular vectors (u1, v1) of a 2D matrix via power iteration.

    Alternating iteration u ← G v / ‖G v‖, v ← Gᵀ u / ‖Gᵀ u‖ converges to
    the leading singular pair. 3 iterations suffice for SAMuon-lite's
    spectral split — the directions only select which components of the
    orthogonalized update are reweighted, so a rough pair is enough.
    Runs in bf16 like _newton_schulz_5.

    Returns (u1, v1): unit vectors of shape (m,) and (n,), bf16.
    """
    assert G.ndim == 2
    Gb = G.bfloat16()
    v = torch.randn(G.shape[1], device=G.device, dtype=torch.bfloat16)
    v = v / (v.norm() + 1e-8)
    u = torch.zeros(G.shape[0], device=G.device, dtype=torch.bfloat16)
    for _ in range(iters):
        u = Gb @ v
        u = u / (u.norm() + 1e-8)
        v = Gb.T @ u
        v = v / (v.norm() + 1e-8)
    return u, v


class SFNorMuon(Optimizer):
    """SF-NorMuon: Schedule-Free NorMuon optimizer.

    Combines:
      - Muon: matrix orthogonalization for hidden params
      - NorMuon: per-neuron normalization
      - Schedule-Free: iterate averaging (no LR schedule)
      - Weight decay at fast iterate Z (not Y)

    Three iterates:
      - X: slow averaged iterate (output/eval)
      - Y: fast iterate (gradient evaluation point)
      - Z: fastest iterate (Muon update target)

    Update: Z = Muon_update(Y, grad)
            Y = (1-c)*Y + c*Z  (interpolation)
            X = (1-β)*X + β*Y  (averaging)

    Args:
        params: model parameters
        lr: learning rate (can be fixed — no schedule needed)
        momentum: Muon momentum
        beta: averaging decay for X
        weight_decay: applied to Z (fast iterate)
        ns_steps: Newton-Schulz iterations
    """

    def __init__(self, params: Iterable, lr=0.02, momentum=0.95,
                 beta=0.999, weight_decay=0.01, ns_steps=5):
        defaults = {'lr': lr, 'momentum': momentum, 'beta': beta,
                        'weight_decay': weight_decay, 'ns_steps': ns_steps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            beta = group['beta']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    state['X'] = p.clone()  # slow averaged
                    state['Y'] = p.clone()  # fast eval point
                    state['Z'] = p.clone()  # fastest update target
                    state['step'] = 0

                state['step'] += 1
                mom = state['momentum']
                X, Y, Z = state['X'], state['Y'], state['Z']

                # Muon update (for 2D matrix params)
                if grad.ndim == 2 and grad.shape[0] > 1 and grad.shape[1] > 1:
                    # Per-neuron normalize (NorMuon)
                    grad_norm = _per_neuron_normalize(grad)
                    # Momentum
                    mom.mul_(momentum).add_(grad_norm, alpha=1 - momentum)
                    # Orthogonalize
                    update = _newton_schulz_5(mom, ns_steps)
                    update *= max(1, grad.shape[0] / grad.shape[1]) ** 0.5
                else:
                    # AdamW for non-matrix params (vectors, scalars)
                    mom.mul_(momentum).add_(grad, alpha=1 - momentum)
                    update = mom

                # Weight decay at Z (fast iterate)
                if wd > 0:
                    Z.mul_(1 - lr * wd)

                # Update Z
                Z.add_(update, alpha=-lr)

                # Interpolation coefficient (time-varying)
                c = 1.0 / max(state['step'], 1) ** 0.5  # decreases over time
                # Update Y: interpolate toward Z
                Y.mul_(1 - c).add_(Z, alpha=c)

                # Update X: averaging
                X.mul_(beta).add_(Y, alpha=1 - beta)

                # Set parameter to Y (eval point)
                p.copy_(Y)

        return loss

    def get_eval_params(self):
        """Return the averaged (X) parameters for evaluation."""
        eval_params = {}
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    eval_params[id(p)] = self.state[p]['X'].clone()
        return eval_params


class AMUSE(Optimizer):
    """AMUSE: Anytime Muon with Stable Gradient Evaluation.

    Integrates Muon's rapid bulk progress with Schedule-Free averaging.
    Uses a time-varying interpolation coefficient:
      - Early: evaluate gradients near fast Muon sequence (rapid adaptation)
      - Late: shift toward stable averaged sequence (suppress oscillations)

    This prevents the valley-wall oscillations that plague pure Muon.

    Args:
        params: model parameters
        lr: learning rate
        momentum: Muon momentum
        beta: averaging decay
        interpolation_schedule: 'sqrt' (default) or 'linear'
        ns_steps: Newton-Schulz iterations
    """

    def __init__(self, params: Iterable, lr=0.02, momentum=0.95,
                 beta=0.999, interpolation_schedule='sqrt', ns_steps=5):
        defaults = {'lr': lr, 'momentum': momentum, 'beta': beta,
                        'interpolation_schedule': interpolation_schedule,
                        'ns_steps': ns_steps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            beta = group['beta']
            sched = group['interpolation_schedule']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    state['X'] = p.clone()  # averaged (stable)
                    state['Z'] = p.clone()  # Muon fast sequence
                    state['step'] = 0

                state['step'] += 1
                t = state['step']
                mom = state['momentum']
                X, Z = state['X'], state['Z']

                # Muon update
                if grad.ndim == 2 and grad.shape[0] > 1 and grad.shape[1] > 1:
                    mom.mul_(momentum).add_(grad, alpha=1 - momentum)
                    update = _newton_schulz_5(mom, ns_steps)
                    update *= max(1, grad.shape[0] / grad.shape[1]) ** 0.5
                else:
                    mom.mul_(momentum).add_(grad, alpha=1 - momentum)
                    update = mom

                # Update Z (fast Muon sequence)
                Z.add_(update, alpha=-lr)

                # Time-varying interpolation coefficient
                # Early: c ≈ 1 (evaluate at Z, rapid adaptation)
                # Late: c ≈ 0 (evaluate at X, stable)
                if sched == 'sqrt':
                    c = 1.0 / math.sqrt(t)
                else:  # linear
                    c = max(0, 1.0 - t / 1000)

                # Evaluation point: interpolate between X (stable) and Z (fast)
                eval_point = X * (1 - c) + Z * c

                # Update X (averaging)
                X.mul_(beta).add_(Z, alpha=1 - beta)

                # Set parameter to evaluation point
                p.copy_(eval_point)

        return loss


class MONA(Optimizer):
    """MONA: Muon Optimizer with Nesterov Acceleration.

    Adds Nesterov acceleration to Muon:
      - Computes EMA of gradient differences (acceleration term)
      - Adds acceleration to Muon update
      - Escapes sharp minima while preserving spectral-norm regularization

    SOTA on MoE pretraining (1B-68B params, 1T tokens).

    Args:
        params: model parameters
        lr: learning rate
        momentum: Muon momentum
        accel_decay: EMA decay for acceleration term
        accel_weight: weight for acceleration term
        ns_steps: Newton-Schulz iterations
    """

    def __init__(self, params: Iterable, lr=0.02, momentum=0.95,
                 accel_decay=0.9, accel_weight=0.1, ns_steps=5):
        defaults = {'lr': lr, 'momentum': momentum, 'accel_decay': accel_decay,
                        'accel_weight': accel_weight, 'ns_steps': ns_steps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            accel_decay = group['accel_decay']
            accel_weight = group['accel_weight']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)
                    state['accel'] = torch.zeros_like(p)  # EMA of grad differences
                    state['step'] = 0

                state['step'] += 1
                mom = state['momentum']
                prev_grad = state['prev_grad']
                accel = state['accel']

                # Compute gradient difference (acceleration signal)
                grad_diff = grad - prev_grad

                # Update acceleration EMA
                accel.mul_(accel_decay).add_(grad_diff, alpha=1 - accel_decay)

                # Save current grad for next step
                prev_grad.copy_(grad)

                # Muon update + acceleration
                combined = grad + accel_weight * accel

                if combined.ndim == 2 and combined.shape[0] > 1 and combined.shape[1] > 1:
                    mom.mul_(momentum).add_(combined, alpha=1 - momentum)
                    update = _newton_schulz_5(mom, ns_steps)
                    update *= max(1, combined.shape[0] / combined.shape[1]) ** 0.5
                else:
                    mom.mul_(momentum).add_(combined, alpha=1 - momentum)
                    update = mom

                p.add_(update, alpha=-lr)

        return loss


class SAMuonLite(Optimizer):
    """SAMuon-lite: Spectral-Aware Muon (arXiv 2608.25990).

    Standard Muon step (momentum → Newton-Schulz orthogonalization → O),
    then a spectral split of the update:

      - (u1, v1): top singular pair of the momentum buffer, estimated with
        a few power-iteration steps (bf16, cheap — no full SVD).
      - head = c · u1 v1ᵀ with c = u1ᵀ O v1  (scalar overlap of O with the
        dominant momentum direction).
      - bulk = O − head  (all remaining spectral directions).
      - update = head + bulk_gain · bulk.

    Rationale: orthogonalization already fully scales the head direction,
    while the bulk directions carry most of the long-tail progress —
    amplifying them (bulk_gain ≈ 1.5) reaches the same loss in 13–24%
    fewer tokens. bulk_gain = 1.0 reduces to plain Muon up to bf16
    rounding.

    Args:
        params: model parameters
        lr: learning rate
        momentum: Muon momentum
        bulk_gain: gain on the non-head (bulk) part of the update
        power_iters: power-iteration steps for the top singular pair
        weight_decay: decoupled decay applied to the parameter
        ns_steps: Newton-Schulz iterations
    """

    def __init__(self, params: Iterable, lr=0.02, momentum=0.95,
                 bulk_gain=1.5, power_iters=3, weight_decay=0.0,
                 ns_steps=5):
        defaults = {'lr': lr, 'momentum': momentum, 'bulk_gain': bulk_gain,
                        'power_iters': power_iters, 'weight_decay': weight_decay,
                        'ns_steps': ns_steps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            bulk_gain = group['bulk_gain']
            power_iters = group['power_iters']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)

                mom = state['momentum']

                # SAMuon-lite update (for 2D matrix params)
                if grad.ndim == 2 and grad.shape[0] > 1 and grad.shape[1] > 1:
                    # Standard Muon: momentum → orthogonalize.
                    mom.mul_(momentum).add_(grad, alpha=1 - momentum)
                    O = _newton_schulz_5(mom, ns_steps)
                    # Spectral split on the momentum buffer's head.
                    u1, v1 = _top_singular_pair(mom, power_iters)
                    c = u1 @ O @ v1  # scalar: u1ᵀ O v1
                    head = c * torch.outer(u1, v1)
                    bulk = O - head
                    update = head + bulk_gain * bulk
                    update *= max(1, grad.shape[0] / grad.shape[1]) ** 0.5
                else:
                    # Momentum for non-matrix params (vectors, scalars).
                    mom.mul_(momentum).add_(grad, alpha=1 - momentum)
                    update = mom

                # Decoupled weight decay, then parameter update.
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)

        return loss


class SAMuonLiteWithAuxAdam:
    """SAMuon-lite for 2D hidden weights + AdamW for embeddings/scalars.

    Duck-typed composite mirroring MuonScheduleFree (muon_sf_blockwise.py):
    param groups carry a ``use_samuon`` flag — flagged groups go to an
    internal SAMuonLite, the rest to an internal AdamW. Only the SAMuon
    param groups are exposed via ``.param_groups`` (same convention as
    MuonScheduleFree, so the trainer's flat LR rewrite hits the Muon
    groups, not the AdamW auxiliaries).
    """

    def __init__(self, param_groups):
        samuon_groups, adam_groups = [], []
        for g in param_groups:
            g = dict(g)
            use_samuon = g.pop('use_samuon', False)
            if not g.get('params'):
                continue
            (samuon_groups if use_samuon else adam_groups).append(g)
        self._samuon = SAMuonLite(samuon_groups) if samuon_groups else None
        self._adam = (torch.optim.AdamW(adam_groups) if adam_groups else None)

    @property
    def param_groups(self):
        return self._samuon.param_groups if self._samuon is not None else []

    def step(self, closure=None):
        loss = self._samuon.step(closure) if self._samuon is not None else None
        if self._adam is not None:
            self._adam.step()
        return loss

    def zero_grad(self, set_to_none=True):
        if self._samuon is not None:
            self._samuon.zero_grad(set_to_none=set_to_none)
        if self._adam is not None:
            self._adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'samuon': (self._samuon.state_dict()
                       if self._samuon is not None else None),
            'adam': (self._adam.state_dict()
                     if self._adam is not None else None),
        }

    def load_state_dict(self, state_dict):
        if self._samuon is not None and state_dict.get('samuon') is not None:
            self._samuon.load_state_dict(state_dict['samuon'])
        if self._adam is not None and state_dict.get('adam') is not None:
            self._adam.load_state_dict(state_dict['adam'])


def build_samuon_lite(model, max_lr: float, weight_decay: float = 0.1,
                      bulk_gain: float = 1.5) -> SAMuonLiteWithAuxAdam:
    """Build a SAMuonLiteWithAuxAdam optimizer for a ForgeAI model.

    Same parameter split as build_muon_sf_plain (muon_sf_blockwise.py):
      - SAMuon-lite: 2D hidden matrices (not embeddings/head)
      - AdamW: embeddings, head, scalars (<2D)

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
            "use_samuon": False,
            "weight_decay": 0.0,
        },
        {
            "params": hidden_params,
            "lr": muon_lr,
            "momentum": 0.95,
            "use_samuon": True,
            "bulk_gain": bulk_gain,
            "weight_decay": weight_decay,
        },
        {
            "params": scalar_params,
            "lr": scalar_lr,
            "betas": (0.8, 0.95),
            "eps": 1e-10,
            "use_samuon": False,
            "weight_decay": 0.0,
        },
    ]
    print(
        f"Using SAMuon-lite (lr={muon_lr:.4f}, bulk_gain={bulk_gain}) for "
        f"{len(hidden_params)} hidden matrices + AdamW (embed={embed_lr:.4f}, "
        f"scalar={scalar_lr:.4f}) for {len(embed_params)} embed + "
        f"{len(scalar_params)} scalar params."
    )
    return SAMuonLiteWithAuxAdam(param_groups)
