"""R38-3/4/5 + MiCA: PiSSA, AdaLoRA, rsLoRA, MiCA adapter variants.

LoRA initialization / scaling / allocation strategies that build on
the manual LoRA path in ``bitnet_lora.py`` (no PEFT dependency, works with
BitNetLinear / NF4Linear / nn.Linear).

  R38-3  PiSSAInitializer  — principal-singular-value LoRA init (faster
                             convergence than random A + zero B).
  R38-4  AdaLoRAClass      — adaptive rank budget allocation across layers
                             with prune / grow / reallocate on importance
                             sensitivity scores.
  R38-5  rsLoRALinear      — scale-free LoRA (1/sqrt(r) scaling) for stable
                             high-rank training without LR tuning.
  MiCA   MiCAInitializer   — minor-singular-component init: frozen B spans
          + MiCALinear       the LEAST important directions of W, trainable
                             zero-init A.  Inverse of PiSSA; ~5.9x better
                             knowledge acquisition per PEFT experiments —
                             use for CPT / knowledge-injection SFT, not
                             instruction tuning.

Refs:
  PiSSA   — Meng et al. arXiv 2404.02948
  AdaLoRA — Zhang et al. arXiv 2308.13160
  rsLoRA  — Kalajdzievski arXiv 2312.03732
  MiCA    — arXiv 2604.01694 (HF PEFT: init_lora_weights="mica")
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# ── R38-3: PiSSA (Principal Singular values LoRA Initialization) ─────────

class PiSSAInitializer:
    """Initialize LoRA from the principal singular components of W.

    Standard LoRA uses random A + zero B so the adapter starts as a no-op.
    PiSSA instead decomposes W = U S Vh and splits the *top-r* singular
    component into the adapter (A = U[:, :r] * S[:r], B = Vh[:r, :]). The
    residual (bottom singular values) stays in the frozen base weight, so
    the trainable adapter carries the most informative directions from the
    start — empirically converges faster and to a lower loss than random
    init.

    All methods are no-grad / pure-tensor: they produce (A, B) or merged
    weights and do not mutate nn.Modules in place. The caller is responsible
    for assigning the results to LoRAAdapter.lora_A / lora_B (or equivalent).
    """

    @staticmethod
    @torch.no_grad()
    def initialize(weight_matrix: torch.Tensor, rank: int,
                   clamp_rank: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """SVD init: returns (A, B) for a LoRA adapter.

        Args:
            weight_matrix: base weight W of shape (out_features, in_features).
            rank: target LoRA rank r. Clamped to min(rank, min(dims)) when
                ``clamp_rank`` is True (SVD cannot return more components than
                the smaller dimension).
            clamp_rank: if False, raise when rank exceeds the SVD limit.

        Returns:
            A: shape (rank, in_features)  — scaled left singular vectors.
            B: shape (out_features, rank) — right singular vectors.

        With these, ``A.T @ B.T`` (LoRAAdapter convention) reconstructs the
        top-r singular component of W, i.e. ``U[:, :r] @ diag(S[:r]) @ Vh[:r, :]``.
        """
        if weight_matrix.ndim != 2:
            raise ValueError(f"weight_matrix must be 2D, got {weight_matrix.ndim}D")
        W = weight_matrix.to(torch.float32)
        out_features, in_features = W.shape
        max_rank = min(out_features, in_features)
        if rank > max_rank:
            if not clamp_rank:
                raise ValueError(
                    f"rank {rank} exceeds SVD limit {max_rank} for {tuple(W.shape)}")
            rank = max_rank

        # full_matrices=False keeps U (out, k), S (k,), Vh (k, in) with k=min(dims)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        Sr = S[:rank]
        Ur = U[:, :rank]                # (out, r)
        Vhr = Vh[:rank, :]              # (r, in)

        # A: (rank, in_features) = scaled right singular vectors (Vh rows)
        # B: (out_features, rank) = left singular vectors
        # so that B @ A = Ur * Sr @ Vhr = top-r reconstruction of W.
        # LoRAAdapter.forward computes scale * (x @ A.T @ B.T) = scale * x @ (B@A).T
        # → with scale=1.0 this is x @ (top-r W).T = top-r component of base(x).
        A = (Vhr * Sr.unsqueeze(1)).contiguous()   # (r, in)
        B = Ur.contiguous()                         # (out, r)
        return A, B

    @staticmethod
    @torch.no_grad()
    def merge(A: torch.Tensor, B: torch.Tensor,
              base_weight: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Merge adapter back into the base weight: W + scale * (B @ A).

        Args:
            A: (rank, in_features).
            B: (out_features, rank).
            base_weight: (out_features, in_features).
            scale: LoRA scaling factor (default 1.0 — PiSSA uses scale=1).

        Returns:
            merged weight of shape (out_features, in_features).
        """
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be 2D")
        if A.shape[0] != B.shape[1]:
            raise ValueError(
                f"rank mismatch: A.shape[0]={A.shape[0]} != B.shape[1]={B.shape[1]}")
        delta = scale * (B.to(torch.float32) @ A.to(torch.float32))
        return base_weight.to(torch.float32) + delta


# ── MiCA (Minor-Component Adaptation) ────────────────────────────────────

class MiCAInitializer:
    """Initialize LoRA from the MINOR (bottom-r) singular subspace of W.

    Exact inverse of PiSSA: where PiSSA puts the top-r singular component
    into a trainable adapter, MiCA freezes ``B`` to the *least* important
    left singular vectors ``U[:, -r:]`` and trains only a zero-initialized
    ``A``.  The update is confined to the directions the pretrained weight
    uses least, which preserves pretrained capability while still moving
    knowledge — PEFT's ``init_lora_weights="mica"`` recipe (arXiv
    2604.01694): ~5.9x better knowledge acquisition than vanilla LoRA for
    knowledge-injection / CPT-style SFT, with only A's ``r * in_features``
    params trainable (B frozen).

    Not for instruction tuning — use vanilla/PiSSA there.  Recommended:
    r=16–32, LR 5e-4..2e-3 (minor directions are flat and tolerate a large
    step), target q_proj/v_proj per the paper.
    """

    @staticmethod
    @torch.no_grad()
    def initialize(weight_matrix: torch.Tensor, rank: int,
                   clamp_rank: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Minor-subspace init: returns (A, B) for a LoRA adapter.

        Args:
            weight_matrix: base weight W of shape (out_features, in_features).
            rank: target LoRA rank r. Clamped to min(rank, min(dims)) when
                ``clamp_rank`` is True.
            clamp_rank: if False, raise when rank exceeds the SVD limit.

        Returns:
            A: (rank, in_features) — ALL ZEROS, the trainable factor.
            B: (out_features, rank) — U[:, -r:], the minor left singular
                vectors of W.  ORTHONORMAL columns; meant to be FROZEN
                (``requires_grad_(False)`` on the parameter).

        The adapter starts as an exact no-op (A=0 → x @ A.T @ B.T = 0),
        unlike PiSSA which starts carrying W's top-r component.  The base
        weight is NOT modified — there is no residual to fold back.
        """
        if weight_matrix.ndim != 2:
            raise ValueError(f"weight_matrix must be 2D, got {weight_matrix.ndim}D")
        W = weight_matrix.to(torch.float32)
        out_features, in_features = W.shape
        max_rank = min(out_features, in_features)
        if rank > max_rank:
            if not clamp_rank:
                raise ValueError(
                    f"rank {rank} exceeds SVD limit {max_rank} for {tuple(W.shape)}")
            rank = max_rank

        U, _S, _Vh = torch.linalg.svd(W, full_matrices=False)
        # U columns are sorted by descending singular value — the LAST r
        # are the minor subspace.
        B = U[:, -rank:].contiguous()                    # (out, r)
        A = torch.zeros(rank, in_features, dtype=W.dtype)
        return A, B


class MiCALinear(nn.Module):
    """Linear layer with a MiCA adapter (frozen minor-subspace B, trained A).

    Forward is identical to vanilla LoRA —
    ``y = base(x) + (alpha / r) * (x @ A.T @ B.T)`` — but ``lora_B`` is a
    frozen buffer holding the bottom-r left singular vectors of the base
    weight and ``lora_A`` is the only trainable factor, zero-initialized
    so the adapter is an exact no-op at init.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float = 1.0, bias: bool = False,
                 base_weight: torch.Tensor | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        if base_weight is not None:
            assert base_weight.shape == (out_features, in_features)
            self.base = nn.Linear(in_features, out_features, bias=bias)
            with torch.no_grad():
                self.base.weight.copy_(base_weight)
        else:
            self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        A, B = MiCAInitializer.initialize(self.base.weight, rank)
        self.lora_A = nn.Parameter(A.to(self.base.weight.dtype))
        # Frozen direction matrix — buffer, not Parameter, so it can never
        # collect a gradient or appear in optimizer groups.
        self.register_buffer(
            "lora_B", B.to(self.base.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = base(x) + scale * (x @ A.T @ B.T)."""
        out = self.base(x)
        delta = self.scale * (x @ self.lora_A.T @ self.lora_B.T)
        return out + delta.to(out.dtype)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Merge adapter into base weight and return a plain nn.Linear."""
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.base.bias is not None)
        delta = self.scale * (self.lora_B.to(torch.float32)
                              @ self.lora_A.to(torch.float32))
        w = self.base.weight.to(torch.float32) + delta
        merged.weight.copy_(w.to(self.base.weight.dtype))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias.data)
        merged = merged.to(self.base.weight.device)
        return merged

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, rank={self.rank}, "
                f"alpha={self.alpha}, scale=1/r={self.scale:.4f}, "
                f"B=frozen(U[:, -r:])")


# ── R38-4: AdaLoRA (Adaptive Budget Allocation) ──────────────────────────

class AdaLoRAClass:
    """Adaptive rank-budget allocation across LoRA layers (R38-4).

    Manages a per-layer rank budget that is redistributed over training
    based on an importance / sensitivity score. Layers that matter more
    (high gradient * weight norm) get more singular values; unimportant
    layers are pruned. This avoids the fixed-rank compromise of vanilla
    LoRA: important layers grow, unimportant ones shrink, total budget
    stays constant.

    The adapter is represented as a triple (A, B, singular_values) where
    A and B are the LoRA factors and ``singular_values`` is a 1-D tensor
    of length rank carrying the per-direction "importance mask" (analogous
    to the diagonal of S in a factored adapter). Prune = zero out small
    SVs; grow = restore / increase SVs on important directions.

    Importance score (sensitivity) follows AdaLoRA §3.3:
        s_i = |grad_i| * |w_i|
    aggregated per layer as the mean over the adapter's parameters.
    """

    def __init__(self, total_budget: int, n_layers: int,
                 min_rank: int = 1, max_rank: int | None = None,
                 prune_threshold: float = 0.5,
                 grow_step: float = 0.1):
        """
        Args:
            total_budget: total rank budget shared across all layers.
            n_layers: number of LoRA-adapted layers.
            min_rank: floor — no layer drops below this rank.
            max_rank: optional ceiling per layer.
            prune_threshold: fractions of the max SV below which an SV is
                pruned (0.5 = prune anything < 50% of the largest SV).
            grow_step: relative increment applied when growing an SV.
        """
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        # If budget is too small for min_rank, clamp budget up to n_layers*min_rank
        # (min_rank is a floor, not a hard constraint on total_budget)
        self.total_budget = total_budget
        self.n_layers = n_layers
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.prune_threshold = prune_threshold
        self.grow_step = grow_step
        self.ranks = self.allocate_budget(total_budget, n_layers)

    def allocate_budget(self, total_budget: int,
                        n_layers: int) -> list[int]:
        """Distribute ``total_budget`` across ``n_layers`` as evenly as possible.

        Remainder (total_budget % n_layers) is distributed one extra rank to
        the first ``remainder`` layers. Each layer is clamped to
        [min_rank, max_rank]. The sum of the returned ranks is exactly
        ``total_budget`` when no clamping is needed.
        """
        base = total_budget // n_layers
        remainder = total_budget % n_layers
        ranks = [base + (1 if i < remainder else 0) for i in range(n_layers)]
        # clamp to floor / ceiling
        ranks = [max(self.min_rank, r) for r in ranks]
        if self.max_rank is not None:
            ranks = [min(self.max_rank, r) for r in ranks]
        return ranks

    @staticmethod
    @torch.no_grad()
    def _importance(adapter: dict) -> float:
        """Mean |grad| * |weight| sensitivity for an adapter dict.

        ``adapter`` must contain 'A', 'B' (parameters or tensors) and
        optionally 'singular_values'. We use the actual gradient when
        available, falling back to the weight magnitude (no-grad case).
        """
        score = 0.0
        n = 0
        for key in ("A", "B"):
            t = adapter.get(key)
            if t is None:
                continue
            w = t.detach().abs()
            g = t.grad.detach().abs() if getattr(t, "grad", None) is not None else w
            score += float((g * w).mean().item())
            n += 1
        sv = adapter.get("singular_values")
        if sv is not None and getattr(sv, "grad", None) is not None:
            w = sv.detach().abs()
            g = sv.grad.detach().abs()
            score += float((g * w).mean().item())
            n += 1
        return score / max(n, 1)

    @torch.no_grad()
    def prune_singular_values(self, adapter: dict,
                              importance_score: float | None = None
                              ) -> dict:
        """Zero out singular values below the prune threshold.

        An SV is pruned when its magnitude < ``prune_threshold`` * max(SV).
        Pruned SVs are masked (set to 0) rather than removed, so the rank
        shape is preserved and growth can revive them later. Returns the
        (mutated) adapter dict with an added 'mask' key.
        """
        sv = adapter.get("singular_values")
        if sv is None:
            return adapter
        sv = sv.detach().clone()
        if sv.numel() == 0:
            adapter["mask"] = torch.ones_like(sv)
            return adapter
        thresh = self.prune_threshold * float(sv.abs().max().item())
        mask = (sv.abs() >= thresh).to(sv.dtype)
        adapter["singular_values"] = sv * mask
        adapter["mask"] = mask
        return adapter

    @torch.no_grad()
    def grow_singular_values(self, adapter: dict,
                             importance_score: float | None = None
                             ) -> dict:
        """Grow currently-masked (pruned) SVs back by ``grow_step``.

        Directions that were pruned but whose layer now has high importance
        get a small non-zero SV so they re-enter the active budget. The
        growth magnitude is ``grow_step`` * current max SV.
        """
        sv = adapter.get("singular_values")
        mask = adapter.get("mask")
        if sv is None or mask is None:
            return adapter
        sv = sv.detach().clone()
        max_sv = float(sv.abs().max().item()) if sv.numel() else 0.0
        grow_amt = self.grow_step * max_sv if max_sv > 0 else self.grow_step
        # revive pruned (mask==0) directions
        revived = (mask == 0)
        sv = torch.where(revived, torch.full_like(sv, grow_amt), sv)
        new_mask = torch.where(revived, torch.ones_like(mask), mask)
        adapter["singular_values"] = sv
        adapter["mask"] = new_mask
        return adapter

    @torch.no_grad()
    def update_budget(self, grad_norms: list[float] | dict[int, float]
                      ) -> list[int]:
        """Reallocate the total rank budget across layers from gradient norms.

        ``grad_norms`` is either a list (indexed by layer) or a dict
        {layer_idx: grad_norm}. Layers with larger gradient norms receive
        more rank; the total is conserved (sum == total_budget) subject to
        the [min_rank, max_rank] clamps.

        Returns the new per-layer rank list (also stored in ``self.ranks``).
        """
        if isinstance(grad_norms, dict):
            norms = [float(grad_norms.get(i, 0.0)) for i in range(self.n_layers)]
        else:
            norms = [float(g) for g in grad_norms]
            if len(norms) < self.n_layers:
                norms += [0.0] * (self.n_layers - len(norms))
            norms = norms[:self.n_layers]
        total = sum(norms)
        if total <= 0:
            # no signal → keep even split
            self.ranks = self.allocate_budget(self.total_budget, self.n_layers)
            return self.ranks
        # proportional allocation
        raw = [self.total_budget * (g / total) for g in norms]
        ranks = [int(round(r)) for r in raw]
        # clamp
        ranks = [max(self.min_rank, r) for r in ranks]
        if self.max_rank is not None:
            ranks = [min(self.max_rank, r) for r in ranks]
        # fix rounding drift so sum == total_budget
        _redistribute_drift(ranks, self.total_budget,
                             self.min_rank, self.max_rank)
        self.ranks = ranks
        return ranks


def _redistribute_drift(ranks: list[int], target_sum: int,
                        min_rank: int, max_rank: int | None) -> None:
    """In-place adjust ``ranks`` so sum(ranks) == target_sum, respecting clamps."""
    while sum(ranks) != target_sum:
        diff = target_sum - sum(ranks)
        if diff > 0:
            # add to the largest ranks first (they're the "important" ones)
            order = sorted(range(len(ranks)), key=lambda i: -ranks[i])
            added = False
            for i in order:
                if max_rank is None or ranks[i] < max_rank:
                    ranks[i] += 1
                    added = True
                    break
            if not added:
                break
        else:
            # remove from the smallest ranks first
            order = sorted(range(len(ranks)), key=lambda i: ranks[i])
            removed = False
            for i in order:
                if ranks[i] > min_rank:
                    ranks[i] -= 1
                    removed = True
                    break
            if not removed:
                break


# ── R38-5: rsLoRA (Scale-Free LoRA) ──────────────────────────────────────

class rsLoRALinear(nn.Module):
    """Linear layer with rsLoRA (scale-free LoRA) adapter.

    Standard LoRA scales the low-rank update by ``alpha / r`` (≈ 1/r for
    fixed alpha). At high ranks this shrinks the update so much that the
    effective LR on the adapter becomes vanishingly small, forcing LR
    re-tuning per rank. rsLoRA instead scales by ``alpha / sqrt(r)`` which
    keeps the update magnitude roughly rank-invariant, enabling stable
    training across a wide rank range without per-rank LR tuning.

    Forward:  y = base(x) + (alpha / sqrt(r)) * (x @ A.T @ B.T)

    The base ``nn.Linear`` is frozen (requires_grad=False); only A and B
    are trainable. ``merge()`` folds the adapter back into the base weight
    and returns a plain ``nn.Linear``.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float = 1.0, bias: bool = False,
                 base_weight: torch.Tensor | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        # rsLoRA scaling: 1/sqrt(r) instead of 1/r
        self.scale = alpha / math.sqrt(rank)

        if base_weight is not None:
            assert base_weight.shape == (out_features, in_features)
            self.base = nn.Linear(in_features, out_features, bias=bias)
            with torch.no_grad():
                self.base.weight.copy_(base_weight)
        else:
            self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # A: (rank, in_features), B: (out_features, rank)
        # Standard LoRA init: kaiming A, zero B (starts as no-op).
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = base(x) + scale * (x @ A.T @ B.T)."""
        out = self.base(x)
        delta = self.scale * (x @ self.lora_A.T @ self.lora_B.T)
        return out + delta.to(out.dtype)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Merge adapter into base weight and return a plain nn.Linear.

        The returned layer has weight = base.weight + scale * (B @ A) and
        (if present) the original bias. The adapter factors are discarded.
        """
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.base.bias is not None)
        delta = self.scale * (self.lora_B.to(torch.float32)
                              @ self.lora_A.to(torch.float32))
        w = self.base.weight.to(torch.float32) + delta
        merged.weight.copy_(w.to(self.base.weight.dtype))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias.data)
        merged = merged.to(self.base.weight.device)
        return merged

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, rank={self.rank}, "
                f"alpha={self.alpha}, scale=1/sqrt(r)={self.scale:.4f}")
