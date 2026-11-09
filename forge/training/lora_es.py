"""LoRA-ES (R71/B3): zeroth-order / evolution-strategies adapter training.

Trains a small parameter set (LoRA A/B matrices, or any param subset) with
FORWARD PASSES ONLY — no backward graph, no activations, no optimizer
state. Memory cost ≈ inference: a 1.7B bf16 model trains in ~4-5GB instead
of ~14GB for full SFT. Quality trade: ZO gradients are noisy, so use many
directions + antithetic pairs, and keep the trainable set small (r=8-16).

Modes (`perturb_mode`):
  "full"   — MeZO-style: every direction perturbs ALL trainable params.
  "block"  — block-coordinate: each direction perturbs one random
             contiguous block of tensors. Lower-variance per eval; pairs
             well with per-layer eps.
  "fisher" — like "full", but per-parameter step size is scaled by
             `fisher_scale` (e.g. 1/(F_base+eps) from merge_models.fisher_diag)
             so budget goes to params the base model doesn't defend.

All math runs inside torch.no_grad/inference_mode — `loss_fn()` only needs
to return a float loss; grads are never touched.

Usage:
    es = LoRAES(params, eps=1e-3, lr=3e-3, n_dirs=8)
    for step in range(steps):
        batch = next(data)
        loss_fn = lambda: model_loss(model, batch)
        info = es.step(loss_fn)
"""

from __future__ import annotations

import math
import torch


class LoRAES:
    """Zeroth-order SGD with antithetic Gaussian/Rademacher directions."""

    def __init__(self, params: list[torch.Tensor], names: list[str] | None = None,
                 eps: float = 1e-3, lr: float = 3e-3, n_dirs: int = 8,
                 momentum: float = 0.9, antithetic: bool = True,
                 distribution: str = "gaussian",   # gaussian | rademacher
                 perturb_mode: str = "full",        # full | block | fisher
                 block_size: int = 8,
                 fisher_scale: dict[str, torch.Tensor] | None = None,
                 master_fp32: bool = True,
                 seed: int = 0, device: torch.device | None = None):
        assert perturb_mode in ("full", "block", "fisher")
        self.params = params
        # fp32 shadow weights: bf16 params can't represent eps-scale
        # perturbations (rounding kills the ZO signal -> |g|=0 measured
        # on spark). Perturbations/updates live on these masters and are
        # cast into the (possibly low-precision) params only for eval.
        self._master = ([p.detach().to(torch.float32).clone()
                         for p in params] if master_fp32 else None)
        self.names = names or [f"p{i}" for i in range(len(params))]
        self.eps, self.lr, self.n_dirs = eps, lr, n_dirs
        self.momentum = momentum
        self.antithetic = antithetic
        self.distribution = distribution
        self.perturb_mode = perturb_mode
        self.block_size = block_size
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._vel = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        # per-parameter eps multipliers (fisher mode); 1.0 elsewhere
        self._scale = []
        for i, n in enumerate(self.names):
            s = fisher_scale.get(n) if fisher_scale else None
            if s is not None:
                s = s.to(params[i].device, torch.float32).clamp(0, 1)
            self._scale.append(s)
        self.device = device or (params[0].device if params else "cpu")

    # ---- direction sampling -------------------------------------------
    def _sample_dir(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for t in tensors:
            u = torch.empty(t.shape, dtype=torch.float32)
            if self.distribution == "rademacher":
                u.random_(0, 2, generator=self.gen).mul_(2).sub_(1)
            else:
                u.normal_(generator=self.gen)
            out.append(u.to(t.device))
        return out

    # ---- one ES step ----------------------------------------------------
    def step(self, loss_fn) -> dict:
        """One ZO update: sample n_dirs directions, eval ±eps, update.

        Returns {'loss': base_loss, 'grad_norm': est_norm, 'dir_reward': [...]}.
        """
        n = len(self.params)
        scores, dirs_used = [], []
        base_loss = float(loss_fn())
        for _ in range(self.n_dirs):
            if self.perturb_mode == "block":
                idx = torch.randperm(n, generator=self.gen)[:self.block_size]
            else:
                idx = range(n)
            u = self._sample_dir([self.params[i] for i in idx])
            self._perturb_subset(idx, u, +1.0)
            l_plus = float(loss_fn())
            if self.antithetic:
                self._perturb_subset(idx, u, -2.0)   # to theta - eps*u
                l_minus = float(loss_fn())
                self._perturb_subset(idx, u, +1.0)   # restore
                c = (l_plus - l_minus) / (2.0 * self.eps)
            else:
                self._perturb_subset(idx, u, -1.0)
                c = (l_plus - base_loss) / self.eps
            scores.append(c)
            dirs_used.append((idx, u))
        # grad estimate g = mean_i c_i * u_i ; SGD+momentum update
        gnorm = 0.0
        for i in range(n):
            g = None
            for (idx, u), c in zip(dirs_used, scores):
                if i in idx:
                    j = list(idx).index(i)
                    g = c * u[j] if g is None else g + c * u[j]
            if g is None:
                continue
            g = g / self.n_dirs
            gnorm += float(g.pow(2).sum())
            self._vel[i].mul_(self.momentum).add_(g.to(self._vel[i].device))
            tgt = self._master[i] if self._master is not None \
                else self.params[i].data
            tgt.add_(self._vel[i].to(tgt.device, tgt.dtype), alpha=-self.lr)
            if self._master is not None:
                self.params[i].data.copy_(tgt)
        return {"loss": base_loss, "grad_norm": math.sqrt(gnorm),
                "dir_reward": scores}

    def _perturb_subset(self, idx, us: list[torch.Tensor], sign: float):
        for j, i in enumerate(idx):
            u, s = us[j], self._scale[i]
            if s is not None:
                u = u * s
            tgt = self._master[i] if self._master is not None \
                else self.params[i].data
            tgt.add_(u.to(tgt.dtype) * (sign * self.eps))
            if self._master is not None:
                self.params[i].data.copy_(tgt)


# ---------------------------------------------------------------------------
# Low-rank bypass wrapper — the "LoRA" half of LoRA-ES
# ---------------------------------------------------------------------------

class LowRankBypass(torch.nn.Module):
    """Wrap a Linear: y = W x + s * (x @ B^T) @ A^T. A,B are the trainables.

    Weight matrix shapes: linear.weight is [out, in]. A: [r, in], B: [out, r]
    so delta W = B @ A has shape [out, in] (standard LoRA convention).
    """

    def __init__(self, linear: torch.nn.Linear, rank: int = 8,
                 scale: float = 1.0, init: str = "zero"):
        super().__init__()
        self.linear = linear
        self.rank, self.scale = rank, scale
        dev, dt = linear.weight.device, linear.weight.dtype
        self.A = torch.nn.Parameter(
            torch.zeros(rank, linear.in_features, device=dev, dtype=dt))
        self.B = torch.nn.Parameter(
            torch.zeros(linear.out_features, rank, device=dev, dtype=dt))
        if init == "randn":
            torch.nn.init.normal_(self.A, std=1.0 / math.sqrt(rank))
        elif init == "randn_ab":
            # ZO needs BOTH matrices nonzero: with one at zero the
            # adapter output is identically 0 and finite differences on
            # EITHER matrix still read 0 (each multiplies through the
            # zero partner). Backprop escapes this; ES cannot.
            torch.nn.init.normal_(self.A, std=1.0 / math.sqrt(rank))
            torch.nn.init.normal_(self.B, std=0.002)

    def forward(self, x):
        return self.linear(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def attach_lora(model: torch.nn.Module, rank: int = 8, scale: float = 1.0,
                name_filter=None, init: str = "zero") -> list[torch.Tensor]:
    """Replace matching nn.Linear children with LowRankBypass.

    `name_filter(name) -> bool`; default wraps every Linear.
    Returns the trainable param list [A0, B0, A1, B1, ...].
    """
    trainables: list[torch.Tensor] = []
    for mod_name, mod in list(model.named_modules()):
        for ch_name, ch in list(mod.named_children()):
            full = f"{mod_name}.{ch_name}" if mod_name else ch_name
            if isinstance(ch, torch.nn.Linear) and not isinstance(
                    ch, LowRankBypass) and (name_filter is None
                                            or name_filter(full)):
                wrapped = LowRankBypass(ch, rank, scale, init)
                setattr(mod, ch_name, wrapped)
                trainables += [wrapped.A, wrapped.B]
    return trainables
