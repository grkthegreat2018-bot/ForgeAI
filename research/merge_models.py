"""Model merging — SLERP / TIES / DARE / SVD / Task Arithmetic / Linear.

Operates directly on state dicts (safetensors or .pt) so it works with
ForgeAI's custom LFM2.5 architecture without needing HF-format models
(mergekit's YAML pipeline expects HF-format models, incompatible here).

Methods:
  linear       — weighted average of N models (model soup).
  task_arith   — add/subtract task vectors (theta_ft - theta_base).
  slerp        — spherical linear interpolation between 2 models.
  ties         — Trim, Elect Sign, Disjoint Merge of N task vectors.
  dare         — random drop + rescale of task vectors (Yu et al. 2023).
  svd          — SVD-based low-rank merge of task vectors (Ainsworth et al.).

All methods are pure PyTorch. `torch.kthvalue` is used instead of
`torch.quantile` for magnitude thresholds (quantile has a 2^24 element
limit on CUDA; embeddings exceed it).

Usage:
    python -m research.merge_models --method slerp \\
        --model-a research/checkpoints/a.safetensors \\
        --model-b research/checkpoints/b.safetensors \\
        --t 0.5 --out research/checkpoints/merged.safetensors

    python -m research.merge_models --method ties \\
        --base research/checkpoints/base.safetensors \\
        --models a.safetensors b.safetensors c.safetensors \\
        --density 0.5 --out research/checkpoints/merged.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch

from forge.checkpoint_io import load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return only the tensor entries."""
    state = load_checkpoint(path, map_location="cpu")
    return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}


def _task_vectors(ft: dict[str, torch.Tensor],
                  base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute per-parameter task vectors: ft - base."""
    out = {}
    for k, v in ft.items():
        if k in base and base[k].shape == v.shape:
            out[k] = (v.float() - base[k].float())
    return out


def _kth_threshold(values: torch.Tensor, density: float) -> torch.Tensor:
    """Per-row magnitude threshold keeping fraction `density` of elements.

    Uses torch.kthvalue (not torch.quantile) to avoid the 2^24 element
    CUDA limit on large tensors (e.g. embeddings).
    `values`: [N] magnitudes for one parameter flattened.
    Returns the scalar threshold below which elements are trimmed.
    """
    n = values.numel()
    n_keep = max(1, int(n * density))
    if n_keep >= n:
        return values.min()
    # kthvalue returns the k-th smallest; we want the n_keep-th largest,
    # i.e. the (n - n_keep + 1)-th smallest.
    k = n - n_keep + 1
    return values.kthvalue(k).values


# ---------------------------------------------------------------------------
# Merge methods
# ---------------------------------------------------------------------------

def merge_fisher(base: dict[str, torch.Tensor],
                 new: dict[str, torch.Tensor],
                 f_base: dict[str, torch.Tensor],
                 f_new: dict[str, torch.Tensor],
                 eps: float = 1e-20,
                 mode: str = "gate",
                 alpha_new: float = 1.0,
                 dtype=None) -> dict[str, torch.Tensor]:
    """FisherFold — diagonal-Fisher-weighted merge (R71/B4).

    mode="gate" (rescue/anti-forgetting semantics):
        w' = w_b + (w_n - w_b) * F_n / (F_b + F_n + eps)
        Params the new model actually *uses* on its task (high F_new relative
        to F_base) keep their delta; params that drifted without purpose
        (both Fishers ~0) or that are base-critical (F_b >> F_n) revert to
        base. This is the merge that extracts a regressed sftN's real
        skills without importing its damage.

    mode="interp" (symmetric skill union):
        w' = (F_b * w_b + F_n * w_n) / (F_b + F_n + eps * (w_b prior))
        Importance-weighted interpolation; eps floor leans to base.

    Fisher dicts are keyed like the state dicts; missing keys -> 0 Fisher.
    """
    out = {}
    for k, wb in base.items():
        wn = new.get(k)
        if wn is None or wn.shape != wb.shape:
            out[k] = wb.clone()
            continue
        wb32 = wb.float()
        wn32 = wn.float()
        fb = (f_base[k].float().to(wb32.device) if k in f_base
              else torch.zeros_like(wb32))
        fn = (f_new[k].float().to(wb32.device) if k in f_new
              else torch.zeros_like(wb32))
        if alpha_new != 1.0:
            fn = fn * alpha_new
        if mode == "interp":
            out[k] = ((fb * wb32 + fn * wn32 + eps * wb32)
                      / (fb + fn + eps)).to(wb.dtype)
        else:  # gate
            g = fn / (fb + fn + eps)
            out[k] = (wb32 + (wn32 - wb32) * g).to(wb.dtype)
    return out


def fisher_diag(model, loss_fn, data_iter, n_batches: int,
                fisher_dtype: torch.dtype = torch.float32,
                accumulate_device: str | None = None,
                n_groups: int = 1) -> dict[str, torch.Tensor]:
    """Empirical diagonal Fisher information: E_batch[(dL/dw_i)^2].

    `loss_fn(model, batch)` must return a scalar loss tensor (already
    backward-able). Runs exactly `n_batches` draws from `data_iter`
    (re-iterates from the start if the iterator is exhausted early).
    Returns {param_name: tensor} — same names as model.named_parameters().

    `accumulate_device`: where the running sum lives (default: param
    device). Use "cpu" when GPU headroom is tight (slower: D2H per step).

    `n_groups` > 1 splits the pass into that many sequential runs, each
    with requires_grad enabled on ONE contiguous group of params only —
    GPU grad memory drops to ~1/n_groups of the model (the VRAM-safe
    path for 12GB cards). Fisher is exact per-param; it just costs
    n_groups backward passes per batch.
    """
    named = [(n, p) for n, p in model.named_parameters()]
    if n_groups <= 1:
        groups = [named]
    else:
        gs = max(1, len(named) // n_groups)
        groups = [named[i:i + gs] for i in range(0, len(named), gs)]

    fisher: dict[str, torch.Tensor] = {}
    dev = accumulate_device
    prev_rg = {n: p.requires_grad for n, p in named}
    try:
        for gi, group in enumerate(groups):
            gset = {n for n, _ in group}
            for n, p in named:
                p.requires_grad_(n in gset and prev_rg[n])
            for n, p in group:
                if p.requires_grad:
                    fisher[n] = torch.zeros(
                        p.shape, dtype=fisher_dtype,
                        device=dev or p.device)
            model.zero_grad(set_to_none=True)
            it = iter(data_iter)
            count = 0
            while count < n_batches:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(data_iter)
                    batch = next(it)
                loss = loss_fn(model, batch)
                loss.backward()
                for n, p in group:
                    if p.grad is not None:
                        acc = fisher[n]
                        g = p.grad.detach().to(acc.device,
                                               dtype=acc.dtype)
                        acc += g * g
                model.zero_grad(set_to_none=True)
                count += 1
    finally:
        for n, p in named:
            p.requires_grad_(prev_rg[n])
    for n in fisher:
        fisher[n] /= max(1, n_batches)
    return fisher


def merge_linear(models: list[dict[str, torch.Tensor]],
                 weights: list[float] | None = None
                 ) -> dict[str, torch.Tensor]:
    """Weighted average of N models (model soup)."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    assert len(weights) == len(models), "weights/models length mismatch"
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]
    out = {}
    keys = models[0].keys()
    for k in keys:
        acc = torch.zeros_like(models[0][k], dtype=torch.float32)
        for m, w in zip(models, weights):
            acc = acc + w * m[k].float()
        out[k] = acc.to(models[0][k].dtype)
    return out


def merge_task_arith(base: dict[str, torch.Tensor],
                     task_vectors: list[dict[str, torch.Tensor]],
                     scales: list[float] | None = None
                     ) -> dict[str, torch.Tensor]:
    """Task arithmetic: base + sum(scale_i * task_vector_i)."""
    if scales is None:
        scales = [1.0] * len(task_vectors)
    out = {}
    for k, v in base.items():
        acc = v.float().clone()
        for tv, s in zip(task_vectors, scales):
            if k in tv:
                acc = acc + s * tv[k]
        out[k] = acc.to(v.dtype)
    return out


def merge_slerp(a: dict[str, torch.Tensor],
                b: dict[str, torch.Tensor],
                t: float = 0.5) -> dict[str, torch.Tensor]:
    """Spherical linear interpolation between two models.

    Falls back to linear interpolation when vectors are nearly collinear
    (dot > 0.9995), matching the standard SLERP convention.
    """
    out = {}
    for k in a:
        if k not in b or a[k].shape != b[k].shape:
            out[k] = a[k].clone()
            continue
        wa = a[k].flatten().float()
        wb = b[k].flatten().float()
        na = wa.norm() + 1e-8
        nb = wb.norm() + 1e-8
        ua = wa / na
        ub = wb / nb
        dot = (ua * ub).sum().clamp(-1.0, 1.0)
        if dot > 0.9995:
            result = (1 - t) * wa + t * wb
        else:
            theta = torch.arccos(dot)
            sin_theta = torch.sin(theta)
            result = (torch.sin((1 - t) * theta) / sin_theta) * wa + \
                     (torch.sin(t * theta) / sin_theta) * wb
        out[k] = result.reshape(a[k].shape).to(a[k].dtype)
    return out


def merge_ties(base: dict[str, torch.Tensor],
               task_vectors: list[dict[str, torch.Tensor]],
               density: float = 0.5) -> dict[str, torch.Tensor]:
    """TIES merging (Yadav et al. 2023).

    1. Trim: keep top-`density` fraction by magnitude per parameter per model.
    2. Elect Sign: per parameter, the consensus sign is the sign of the
       sum of trimmed vectors; only vectors matching the consensus sign
       contribute.
    3. Disjoint Merge: sum the sign-consistent vectors, divide by count.
    """
    out = {}
    keys = task_vectors[0].keys()
    for k in keys:
        if k not in base:
            continue
        # Stack trimmed task vectors: [N, *param_shape]
        stacked = []
        for tv in task_vectors:
            if k not in tv:
                continue
            v = tv[k]
            mag = v.abs().flatten()
            thresh = _kth_threshold(mag, density)
            mask = (v.abs() >= thresh).to(v.dtype)
            stacked.append(v * mask)
        if not stacked:
            out[k] = base[k].clone()
            continue
        stacked = torch.stack(stacked, dim=0)  # [N, *shape]
        # Elect sign: consensus = sign of sum across models.
        total = stacked.sum(dim=0)
        consensus = torch.sign(total)
        # Keep only vectors whose sign matches consensus; zero others.
        agree = (torch.sign(stacked) == consensus.unsqueeze(0)).to(stacked.dtype)
        pruned = stacked * agree
        # Disjoint merge: sum / count of contributors (avoid div-by-zero).
        count = agree.sum(dim=0).clamp(min=1.0)
        merged_delta = pruned.sum(dim=0) / count
        out[k] = (base[k].float() + merged_delta).to(base[k].dtype)
    return out


def merge_dare(base: dict[str, torch.Tensor],
               task_vectors: list[dict[str, torch.Tensor]],
               drop_rate: float = 0.1,
               seed: int = 0) -> dict[str, torch.Tensor]:
    """DARE merging (Yu et al. 2023).

    For each task vector, randomly drop `drop_rate` fraction of deltas and
    rescale survivors by 1/(1-drop_rate) to preserve expected magnitude,
    then average the rescaled deltas onto the base.
    """
    gen = torch.Generator().manual_seed(seed)
    scale = 1.0 / max(1.0 - drop_rate, 1e-6)
    out = {}
    keys = task_vectors[0].keys()
    for k in keys:
        if k not in base:
            continue
        acc = torch.zeros_like(base[k], dtype=torch.float32)
        n = 0
        for tv in task_vectors:
            if k not in tv:
                continue
            v = tv[k]
            mask = (torch.rand(v.shape, generator=gen) >= drop_rate).to(v.dtype)
            acc = acc + (v * mask * scale)
            n += 1
        if n == 0:
            out[k] = base[k].clone()
            continue
        out[k] = (base[k].float() + acc / n).to(base[k].dtype)
    return out


def merge_svd(base: dict[str, torch.Tensor],
              task_vectors: list[dict[str, torch.Tensor]],
              rank_ratio: float = 0.5,
              scales: list[float] | None = None) -> dict[str, torch.Tensor]:
    """SVD-based low-rank merge of task vectors.

    For each parameter, stack the N task vectors into a matrix [N, numel]
    (or [numel, N] for 1-D params), compute SVD, truncate to rank
    floor(rank_ratio * min(N, numel)), reconstruct, and apply weighted sum
    onto the base. This compresses the task-vector space to its principal
    components, reducing interference between conflicting updates
    (Ainsworth et al., "SVDETFix").
    """
    if scales is None:
        scales = [1.0] * len(task_vectors)
    out = {}
    keys = task_vectors[0].keys()
    for k in keys:
        if k not in base:
            continue
        # Collect available task vectors for this param.
        vecs = [tv[k].flatten().float() for tv in task_vectors if k in tv]
        if not vecs:
            out[k] = base[k].clone()
            continue
        M = torch.stack(vecs, dim=0)  # [N, numel]
        n_models, numel = M.shape
        # SVD of the task-vector matrix.
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        k_rank = max(1, int(rank_ratio * min(n_models, numel)))
        k_rank = min(k_rank, S.numel())
        U_k = U[:, :k_rank]
        S_k = S[:k_rank]
        Vh_k = Vh[:k_rank, :]
        # Weighted reconstruction: scale per-model contribution.
        w = torch.tensor(scales[:n_models], dtype=M.dtype, device=M.device)
        # Reconstruct a single consensus delta: weighted combination of
        # the top-k right singular directions, scaled by singular values.
        weighted = (U_k * w.unsqueeze(1)) @ torch.diag(S_k)  # [N, k]
        # Sum across models into the principal subspace, then project back.
        consensus = weighted.sum(dim=0) @ Vh_k  # [numel]
        # Normalize by number of models so magnitude stays comparable.
        consensus = consensus / n_models
        out[k] = (base[k].float() + consensus.reshape(base[k].shape)).to(base[k].dtype)
    return out


# ---------------------------------------------------------------------------
# Evolutionary operators — "sexual reproduction" of LLM weights
# ---------------------------------------------------------------------------
# Implements the GENOME framework (Zhang et al. 2026) operators:
#   crossover  — recombine weights of 2 parents → offspring
#   mutation   — small random perturbations for diversity
#   selection  — keep high-fitness individuals
#   succession — carry elite parents forward unchanged (elitism)
#
# Block-level crossover uses the ForgeAI state-dict key convention
# `blocks.{i}.<submodule>.<param>`. Non-block keys (embed/head/norm) are
# inherited from parent A by default (or averaged) — they are the "shared
# genome" that shouldn't be spliced mid-model.
#
# All operators run on CPU (state dicts loaded to CPU). This is the 12GB-
# VRAM-friendly path: evolution happens in system RAM, only the fitness
# evaluation needs the GPU (and only for the candidate being scored).
#
# Novel twist for ForgeAI: `mutate_quant_perturb` treats quantization
# scales (`.qscale` keys) as mutable alleles — evolutionary search over
# the quant-precision axis, which the source papers do not explore.

import re

_BLOCK_RE = re.compile(r'^blocks\.(\d+)\.')


def _block_index(key: str) -> int | None:
    """Return the transformer-block index for a state-dict key, or None."""
    m = _BLOCK_RE.match(key)
    return int(m.group(1)) if m else None


def _non_block_keys(state: dict[str, torch.Tensor]) -> list[str]:
    """Keys that are NOT inside any transformer block (embed/head/norm/etc)."""
    return [k for k in state if _block_index(k) is None]


def _n_blocks(state: dict[str, torch.Tensor]) -> int:
    """Number of transformer blocks in a state dict."""
    max_idx = -1
    for k in state:
        idx = _block_index(k)
        if idx is not None and idx > max_idx:
            max_idx = idx
    return max_idx + 1


def crossover_blockwise(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    split_block: int | None = None,
    non_block_source: str = "a",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Block-level crossover: blocks [0:split] from A, [split:] from B.

    This is the canonical "sexual reproduction" operator — each offspring
    gets a contiguous prefix of layers from one parent and the suffix from
    the other, mimicking chromosomal crossover.

    Args:
        a, b: parent state dicts (must share keys + shapes).
        split_block: block index at which to switch from A→B. If None,
            picks a random split point in [1, n_blocks-1].
        non_block_source: how to handle non-block keys (embed/head/norm).
            "a" = take from A, "b" = take from B, "avg" = average A+B.
        seed: RNG seed for random split.
    """
    n = _n_blocks(a)
    if split_block is None:
        gen = torch.Generator().manual_seed(seed)
        split_block = int(torch.randint(1, max(n, 2), (1,), generator=gen).item())
    split_block = max(1, min(split_block, max(n - 1, 1)))

    out = {}
    for k, v_a in a.items():
        if k not in b or b[k].shape != v_a.shape:
            out[k] = v_a.clone()
            continue
        idx = _block_index(k)
        if idx is None:
            # Non-block key
            if non_block_source == "b":
                out[k] = b[k].clone()
            elif non_block_source == "avg":
                out[k] = ((v_a.float() + b[k].float()) * 0.5).to(v_a.dtype)
            else:
                out[k] = v_a.clone()
        else:
            out[k] = v_a.clone() if idx < split_block else b[k].clone()
    return out


def crossover_block_random(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    p: float = 0.5,
    non_block_source: str = "a",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Per-block random crossover: each block inherited from A or B.

    Each transformer block is taken wholesale from parent A (prob 1-p) or
    parent B (prob p). This is uniform crossover at the block granularity.

    Args:
        p: probability a given block comes from B.
    """
    n = _n_blocks(a)
    gen = torch.Generator().manual_seed(seed)
    from_b = torch.rand(n, generator=gen) < p

    out = {}
    for k, v_a in a.items():
        if k not in b or b[k].shape != v_a.shape:
            out[k] = v_a.clone()
            continue
        idx = _block_index(k)
        if idx is None:
            if non_block_source == "b":
                out[k] = b[k].clone()
            elif non_block_source == "avg":
                out[k] = ((v_a.float() + b[k].float()) * 0.5).to(v_a.dtype)
            else:
                out[k] = v_a.clone()
        else:
            out[k] = b[k].clone() if bool(from_b[idx]) else v_a.clone()
    return out


def crossover_uniform(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    p: float = 0.5,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Per-tensor uniform crossover: each tensor from A or B independently.

    Finer-grained than block crossover — every parameter tensor is inherited
    as a whole from one parent. Element-wise mixing is NOT done here (that
    would be SLERP/linear); this preserves each parent's per-layer solutions.

    Args:
        p: probability a given tensor comes from B.
    """
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, v_a in a.items():
        if k not in b or b[k].shape != v_a.shape:
            out[k] = v_a.clone()
            continue
        use_b = bool(torch.rand(1, generator=gen).item() < p)
        out[k] = b[k].clone() if use_b else v_a.clone()
    return out


def mutate_gaussian(
    state: dict[str, torch.Tensor],
    sigma: float = 0.01,
    rate: float = 0.01,
    seed: int = 0,
    skip_keys: tuple[str, ...] = ("embed.weight", "head.weight"),
) -> dict[str, torch.Tensor]:
    """Gaussian mutation: perturb a fraction of elements by N(0, sigma).

    Only a `rate` fraction of elements in each tensor are perturbed (sparse
    mutation), keeping most of the parent's weights intact. Embedding/head
    keys are skipped by default to avoid vocabulary disruption.

    Args:
        sigma: std dev of Gaussian noise (relative to weight magnitude scale
            — noise is scaled by per-tensor absmean so small and large
            tensors get proportionally-sized perturbations).
        rate: fraction of elements to perturb (0.01 = 1%).
        skip_keys: state-dict keys to leave untouched.
    """
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in state.items():
        if k in skip_keys or v.dtype not in (torch.float32, torch.bfloat16,
                                             torch.float16, torch.float64):
            out[k] = v.clone()
            continue
        vf = v.float()
        mask = (torch.rand(v.shape, generator=gen) < rate).to(vf.dtype)
        absmean = vf.abs().mean().clamp(min=1e-8)
        noise = torch.randn(v.shape, generator=gen).to(vf.dtype) * (sigma * absmean)
        out[k] = (vf + noise * mask).to(v.dtype)
    return out


def mutate_quant_perturb(
    state: dict[str, torch.Tensor],
    sigma: float = 0.05,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Novel: perturb quantization scales (`.qscale` keys) as mutable alleles.

    ForgeAI stores per-tensor quantization scales under `<param>.qscale`.
    This operator perturbs only those scales, leaving the int8/ternary weights
    untouched — an evolutionary search over the quantization-precision axis
    that the source papers (GENOME, Darwin, Sakana) do not explore.

    Args:
        sigma: relative perturbation factor for scales (0.05 = ±5%).
    """
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in state.items():
        if k.endswith(".qscale") and v.is_floating_point():
            scale = 1.0 + torch.randn(v.shape, generator=gen).to(v.float().dtype) * sigma
            out[k] = (v.float() * scale).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def mutate_block_swap(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    n_swaps: int = 1,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Translocation: swap n whole blocks from B into A.

    A different mutation operator — instead of noise, graft entire blocks
    from a donor. This is the BES "translocation" operator adapted to weights.
    """
    n = _n_blocks(a)
    gen = torch.Generator().manual_seed(seed)
    n_swaps = min(n_swaps, n)
    swap_idxs = set(torch.randperm(n, generator=gen)[:n_swaps].tolist())

    out = {}
    for k, v_a in a.items():
        if k not in b or b[k].shape != v_a.shape:
            out[k] = v_a.clone()
            continue
        idx = _block_index(k)
        if idx is not None and idx in swap_idxs:
            out[k] = b[k].clone()
        else:
            out[k] = v_a.clone()
    return out


def select_tournament(
    fitnesses: list[float],
    k: int = 3,
    seed: int = 0,
) -> int:
    """Tournament selection: pick k random individuals, return the best index.

    Args:
        fitnesses: list of fitness scores (higher = better).
        k: tournament size.
    """
    gen = torch.Generator().manual_seed(seed)
    n = len(fitnesses)
    contenders = torch.randperm(n, generator=gen)[:k].tolist()
    best = contenders[0]
    for c in contenders[1:]:
        if fitnesses[c] > fitnesses[best]:
            best = c
    return best


def select_rank(
    fitnesses: list[float],
    selection_pressure: float = 1.5,
    seed: int = 0,
) -> int:
    """Rank-based selection (linear ranking, Baker 1985).

    Less sensitive to fitness magnitude than tournament or roulette —
    works well when one individual has a disproportionately high score
    (which would dominate roulette wheel). Assigns selection probabilities
    based on rank, not raw fitness.

    Args:
        fitnesses: list of fitness scores (higher = better).
        selection_pressure: SP in [1.0, 2.0]. 1.0 = uniform (no pressure),
            2.0 = max pressure (best individual is 2x as likely as median).
        seed: RNG seed.
    """
    n = len(fitnesses)
    if n <= 1:
        return 0
    sp = max(1.0, min(2.0, selection_pressure))
    # Rank: worst=0, best=n-1
    ranked = sorted(range(n), key=lambda i: fitnesses[i])
    rank_of = [0] * n
    for r, idx in enumerate(ranked):
        rank_of[idx] = r
    # Linear ranking probabilities: p_i = (2 - SP + 2*(SP-1)*rank_i/(n-1)) / n
    probs = []
    for i in range(n):
        p = (2.0 - sp + 2.0 * (sp - 1.0) * rank_of[i] / max(n - 1, 1)) / n
        probs.append(max(p, 1e-10))
    total = sum(probs)
    probs = [p / total for p in probs]
    gen = torch.Generator().manual_seed(seed)
    r = torch.rand(1, generator=gen).item()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return n - 1


def select_roulette(
    fitnesses: list[float],
    seed: int = 0,
) -> int:
    """Fitness-proportionate (roulette wheel) selection.

    Individuals are selected with probability proportional to their fitness.
    Requires non-negative fitness (auto-shifts if min < 0).

    Args:
        fitnesses: list of fitness scores (higher = better).
        seed: RNG seed.
    """
    n = len(fitnesses)
    if n <= 1:
        return 0
    # Shift to non-negative
    min_f = min(fitnesses)
    shifted = [f - min_f + 1e-8 for f in fitnesses]
    total = sum(shifted)
    if total <= 0:
        return seed % n
    probs = [f / total for f in shifted]
    gen = torch.Generator().manual_seed(seed)
    r = torch.rand(1, generator=gen).item()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return n - 1


def _state_distance(a: dict[str, torch.Tensor],
                    b: dict[str, torch.Tensor]) -> float:
    """L2 distance between two state dicts (mean of per-tensor L2 distances).

    Used for diversity computation — measures how different two individuals
    are in weight space. Cheaper than full flatten+norm for large models.
    """
    dists = []
    for k in a:
        if k in b and a[k].shape == b[k].shape and a[k].is_floating_point():
            d = (a[k].float() - b[k].float()).norm().item()
            dists.append(d / max(a[k].numel(), 1))
    return sum(dists) / max(len(dists), 1)


def _population_centroid(population: list[dict[str, torch.Tensor]]
                         ) -> dict[str, torch.Tensor]:
    """Compute the centroid (element-wise mean) of the population."""
    if not population:
        return {}
    keys = population[0].keys()
    centroid = {}
    for k in keys:
        acc = torch.zeros_like(population[0][k], dtype=torch.float32)
        for ind in population:
            if k in ind and ind[k].shape == population[0][k].shape:
                acc += ind[k].float()
        centroid[k] = acc / len(population)
    return centroid


def _population_diversity(population: list[dict[str, torch.Tensor]],
                          centroid: dict[str, torch.Tensor] | None = None
                          ) -> float:
    """Measure population diversity as mean distance from centroid.

    High diversity = spread out in weight space (good for exploration).
    Low diversity = converged (good for exploitation, bad if premature).
    """
    if len(population) <= 1:
        return 0.0
    if centroid is None:
        centroid = _population_centroid(population)
    dists = [_state_distance(ind, centroid) for ind in population]
    return sum(dists) / len(dists)


def select_diversity(
    fitnesses: list[float],
    population: list[dict[str, torch.Tensor]],
    diversity_weight: float = 0.3,
    seed: int = 0,
) -> int:
    """Diversity-aware selection (novelty-quality balance).

    Combines fitness with diversity: individuals that are both fit AND
    different from the population centroid get higher selection probability.
    This prevents premature convergence by maintaining behavioral diversity.

    Based on Lehman & Stanley novelty search + quality-diversity hybrid.

    Args:
        fitnesses: list of fitness scores (higher = better).
        population: list of state dicts (for diversity computation).
        diversity_weight: 0.0 = pure fitness, 1.0 = pure diversity.
            0.3 = 70% fitness, 30% diversity (good default).
        seed: RNG seed.
    """
    n = len(fitnesses)
    if n <= 1:
        return 0
    centroid = _population_centroid(population)
    diversities = [_state_distance(ind, centroid) for ind in population]
    # Normalize both to [0, 1]
    f_min, f_max = min(fitnesses), max(fitnesses)
    d_min, d_max = min(diversities), max(diversities)
    f_range = max(f_max - f_min, 1e-10)
    d_range = max(d_max - d_min, 1e-10)
    norm_f = [(f - f_min) / f_range for f in fitnesses]
    norm_d = [(d - d_min) / d_range for d in diversities]
    # Combined score: weighted sum of normalized fitness + diversity
    scores = [(1 - diversity_weight) * nf + diversity_weight * nd
              for nf, nd in zip(norm_f, norm_d)]
    # Roulette selection on combined scores
    total = sum(scores)
    if total <= 0:
        return seed % n
    probs = [s / total for s in scores]
    gen = torch.Generator().manual_seed(seed)
    r = torch.rand(1, generator=gen).item()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return n - 1


def scale_fitness_sigma(fitnesses: list[float],
                        c: float = 2.0) -> list[float]:
    """Linear sigma scaling (Forrest & Tanomura 1990).

    Scales raw fitness to prevent premature convergence when one individual
    dominates. scaled = max(f - (mean - c*std), 0).

    Args:
        fitnesses: raw fitness scores.
        c: scaling constant (2.0 = classic, higher = more pressure).
    """
    n = len(fitnesses)
    if n <= 1:
        return list(fitnesses)
    mean = sum(fitnesses) / n
    var = sum((f - mean) ** 2 for f in fitnesses) / n
    std = var ** 0.5
    if std < 1e-10:
        # All equal — return uniform
        return [1.0] * n
    scaled = [max(f - (mean - c * std), 0.0) for f in fitnesses]
    total = sum(scaled)
    if total <= 0:
        return [1.0] * n
    return scaled


def scale_fitness_rank(fitnesses: list[float],
                       selection_pressure: float = 1.5) -> list[float]:
    """Rank-based fitness scaling.

    Replaces raw fitness with rank-based values, making selection robust
    to outlier fitness values. Uses the same linear ranking as select_rank.
    """
    n = len(fitnesses)
    if n <= 1:
        return list(fitnesses)
    sp = max(1.0, min(2.0, selection_pressure))
    ranked = sorted(range(n), key=lambda i: fitnesses[i])
    rank_of = [0] * n
    for r, idx in enumerate(ranked):
        rank_of[idx] = r
    scaled = []
    for i in range(n):
        p = (2.0 - sp + 2.0 * (sp - 1.0) * rank_of[i] / max(n - 1, 1)) / n
        scaled.append(max(p, 1e-10))
    total = sum(scaled)
    return [s / total for s in scaled]


def evolve(
    population: list[dict[str, torch.Tensor]],
    fitness_fn,
    n_generations: int = 5,
    population_size: int | None = None,
    crossover: str = "blockwise",
    crossover_kwargs: dict | None = None,
    mutation: str = "gaussian",
    mutation_kwargs: dict | None = None,
    mutation_rate: float = 0.5,
    elitism: int = 1,
    seed: int = 0,
    save_fn=None,
    out_dir: str | None = None,
    verbose: bool = True,
    selection: str = "tournament",
    selection_kwargs: dict | None = None,
    fitness_scaling: str = "none",
    progress_bonus: float = 0.0,
    diversity_bonus: float = 0.0,
    adaptive_mutation: bool = False,
    hall_of_fame_size: int = 0,
    convergence_patience: int = 0,
) -> dict:
    """Run evolutionary model merging ("sexual reproduction" of LLM weights).

    GENOME-style loop with sophisticated candidate selection and score rewarding:
      1. Evaluate fitness of every individual (fitness_fn(state_dict) -> float)
      2. Apply score rewarding: fitness scaling + progress bonus + diversity bonus
      3. Select parents via the chosen selection strategy
      4. Crossover two parents → offspring
      5. Mutate offspring (with adaptive probability if enabled)
      6. Carry forward `elitism` best individuals unchanged (succession)
      7. Update hall of fame (best-ever individuals across all generations)
      8. Repeat for n_generations (early stop if convergence_patience hit)

    All state dicts live in CPU RAM. Only fitness_fn touches the GPU (and
    only for the candidate being scored) — this is the 12GB-VRAM-friendly
    operating point.

    Selection strategies (``selection`` parameter):
      - "tournament": k-way tournament (classic, robust, default)
      - "rank": linear ranking with selection pressure (outlier-resistant)
      - "roulette": fitness-proportionate (fast, needs non-negative fitness)
      - "diversity": novelty-quality hybrid (prevents premature convergence)

    Score rewarding (applied before selection):
      - ``fitness_scaling``: "none" (raw), "sigma" (linear sigma scaling),
        "rank" (rank-based scaling). Prevents single-individual dominance.
      - ``progress_bonus``: weight for rewarding offspring that beat their
        parents' fitness (encourages progressive improvement).
      - ``diversity_bonus``: weight for rewarding individuals far from the
        population centroid (maintains behavioral diversity).
      - ``adaptive_mutation``: dynamically adjusts mutation_rate based on
        population diversity — increases when diversity drops (stagnation),
        decreases when diversity is high (converging toward optimum).

    Args:
        population: initial population of state dicts (on CPU).
        fitness_fn: callable(state_dict) -> float. Higher is better.
        n_generations: number of generations to evolve.
        population_size: target population size per gen. If None,
            keeps the initial population size.
        crossover: "blockwise", "block_random", or "uniform".
        crossover_kwargs: extra kwargs for the crossover function.
        mutation: "gaussian", "quant_perturb", or "block_swap".
        mutation_kwargs: extra kwargs for the mutation function.
        mutation_rate: base probability each offspring gets mutated.
        elitism: number of top individuals carried forward unchanged.
        seed: base RNG seed (incremented per generation for variety).
        save_fn: optional callable(state_dict, path) to save checkpoints.
        out_dir: directory for saving per-generation best checkpoints.
        verbose: print per-generation progress.
        selection: parent selection strategy (see above).
        selection_kwargs: extra kwargs for the selection function
            (e.g. {"k": 5} for tournament, {"selection_pressure": 1.8}
            for rank, {"diversity_weight": 0.4} for diversity).
        fitness_scaling: "none", "sigma", or "rank".
        progress_bonus: weight (0.0=disabled) for rewarding improvement
            over parent fitness. Added to offspring fitness after eval.
        diversity_bonus: weight (0.0=disabled) for rewarding distance
            from population centroid.
        adaptive_mutation: if True, dynamically adjust mutation_rate
            based on population diversity.
        hall_of_fame_size: size of the hall of fame (0=disabled). The HoF
            stores the best-ever individuals and can participate in
            crossover as elite donors.
        convergence_patience: stop early if best fitness doesn't improve
            for N consecutive generations (0=disabled).

    Returns:
        dict with keys:
          - "best": best state dict
          - "best_fitness": float
          - "best_path": path to saved best (if save_fn + out_dir) else None
          - "history": list of per-gen {best_fitness, mean_fitness,
            min_fitness, diversity, mutation_rate}
          - "final_population": list of state dicts
          - "hall_of_fame": list of (state_dict, fitness) tuples
    """
    crossover_kwargs = crossover_kwargs or {}
    mutation_kwargs = mutation_kwargs or {}
    selection_kwargs = selection_kwargs or {}
    if population_size is None:
        population_size = len(population)

    crossover_fns = {
        "blockwise": crossover_blockwise,
        "block_random": crossover_block_random,
        "uniform": crossover_uniform,
    }
    mutation_fns = {
        "gaussian": mutate_gaussian,
        "quant_perturb": mutate_quant_perturb,
        "block_swap": mutate_block_swap,  # note: needs a donor (uses next pop member)
    }
    xfn = crossover_fns[crossover]
    mfn = mutation_fns[mutation]

    # Selection function dispatch
    def _select_parent(fits, pop, gen_seed):
        if selection == "rank":
            return select_rank(fits, seed=gen_seed,
                               **{k: v for k, v in selection_kwargs.items()
                                  if k != "diversity_weight"})
        elif selection == "roulette":
            return select_roulette(fits, seed=gen_seed)
        elif selection == "diversity":
            return select_diversity(fits, pop, seed=gen_seed,
                                    **{k: v for k, v in selection_kwargs.items()
                                       if k == "diversity_weight"})
        else:  # tournament (default)
            k = selection_kwargs.get("k", 3)
            return select_tournament(fits, k=k, seed=gen_seed)

    history = []
    best_state = None
    best_fitness = float("-inf")
    best_path = None

    # Hall of Fame: best-ever individuals across all generations
    hall_of_fame: list[tuple[dict, float]] = []

    # Track parent fitnesses for progress bonus
    prev_fitnesses: list[float] | None = None

    # Adaptive mutation state
    current_mutation_rate = mutation_rate
    initial_diversity: float | None = None

    # Convergence tracking
    gens_without_improvement = 0

    for gen in range(n_generations):
        # 1. Evaluate fitness
        raw_fitnesses = [fitness_fn(ind) for ind in population]
        fitnesses = list(raw_fitnesses)  # working copy for selection

        # 2. Score rewarding: apply fitness scaling
        if fitness_scaling == "sigma":
            fitnesses = scale_fitness_sigma(fitnesses)
        elif fitness_scaling == "rank":
            fitnesses = scale_fitness_rank(fitnesses)

        # 3. Score rewarding: diversity bonus
        gen_diversity = 0.0
        if diversity_bonus > 0 or adaptive_mutation or selection == "diversity":
            centroid = _population_centroid(population)
            gen_diversity = _population_diversity(population, centroid)
            if initial_diversity is None:
                initial_diversity = max(gen_diversity, 1e-10)
            if diversity_bonus > 0:
                diversities = [_state_distance(ind, centroid)
                               for ind in population]
                d_min, d_max = min(diversities), max(diversities)
                d_range = max(d_max - d_min, 1e-10)
                for i, d in enumerate(diversities):
                    norm_d = (d - d_min) / d_range
                    fitnesses[i] += diversity_bonus * norm_d

        # 4. Score rewarding: progress bonus (offspring vs parents)
        if progress_bonus > 0 and prev_fitnesses is not None:
            # Compare each individual's fitness to the best parent fitness
            # from the previous generation. If this individual beats it,
            # add a progress bonus.
            prev_best = max(prev_fitnesses)
            for i, rf in enumerate(raw_fitnesses):
                if rf > prev_best:
                    fitnesses[i] += progress_bonus * (rf - prev_best)

        # 5. Adaptive mutation rate
        if adaptive_mutation and initial_diversity is not None:
            # When diversity drops below 50% of initial, increase mutation
            # to escape stagnation. When diversity is high, decrease to
            # allow convergence.
            diversity_ratio = gen_diversity / initial_diversity
            if diversity_ratio < 0.5:
                # Stagnating — increase mutation
                current_mutation_rate = min(mutation_rate * 2.5, 0.95)
            elif diversity_ratio > 1.5:
                # High diversity — decrease mutation to converge
                current_mutation_rate = max(mutation_rate * 0.5, 0.05)
            else:
                current_mutation_rate = mutation_rate

        # 6. Track best (using raw fitness, not scaled)
        gen_best_idx = max(range(len(raw_fitnesses)),
                           key=lambda i: raw_fitnesses[i])
        gen_best_fit = raw_fitnesses[gen_best_idx]
        gen_mean = sum(raw_fitnesses) / len(raw_fitnesses)
        gen_min = min(raw_fitnesses)
        improved = gen_best_fit > best_fitness
        history.append({
            "generation": gen,
            "best_fitness": gen_best_fit,
            "mean_fitness": gen_mean,
            "min_fitness": gen_min,
            "best_index": gen_best_idx,
            "diversity": gen_diversity,
            "mutation_rate": current_mutation_rate,
        })
        if improved:
            best_fitness = gen_best_fit
            best_state = {k: v.clone() for k, v in population[gen_best_idx].items()}
            gens_without_improvement = 0
        else:
            gens_without_improvement += 1

        if verbose:
            div_str = f" div={gen_diversity:.6f}" if gen_diversity > 0 else ""
            mut_str = f" mut={current_mutation_rate:.2f}" if adaptive_mutation else ""
            print(f"  [evolve] gen {gen}: best={gen_best_fit:.4f} "
                  f"mean={gen_mean:.4f} min={gen_min:.4f}{div_str}{mut_str}")

        # Save per-generation best
        if save_fn is not None and out_dir is not None:
            from pathlib import Path
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            path = str(Path(out_dir) / f"gen_{gen}_best.safetensors")
            save_fn(population[gen_best_idx], path)
            if gen_best_fit >= best_fitness:
                best_path = path

        # 7. Update hall of fame
        if hall_of_fame_size > 0:
            for i, rf in enumerate(raw_fitnesses):
                hall_of_fame.append((
                    {k: v.clone() for k, v in population[i].items()},
                    rf,
                ))
            # Keep only top-N by fitness
            hall_of_fame.sort(key=lambda x: x[1], reverse=True)
            hall_of_fame = hall_of_fame[:hall_of_fame_size]

        # 8. Convergence check
        if convergence_patience > 0 and gens_without_improvement >= convergence_patience:
            if verbose:
                print(f"  [evolve] converged at gen {gen} "
                      f"(no improvement for {convergence_patience} gens)")
            break

        # 9. Build next generation
        ranked = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i],
                        reverse=True)
        new_pop = []

        # Elitism: carry forward top individuals (succession)
        for i in range(min(elitism, len(population))):
            new_pop.append({k: v.clone() for k, v in population[ranked[i]].items()})

        # Fill rest via crossover + mutation
        gen_seed = seed + gen
        # Build a donor pool that includes hall of fame members (if enabled)
        donor_pool = list(population)
        donor_fits = list(fitnesses)
        if hall_of_fame_size > 0 and hall_of_fame:
            for hof_state, hof_fit in hall_of_fame:
                donor_pool.append(hof_state)
                donor_fits.append(hof_fit)

        while len(new_pop) < population_size:
            parent_a_idx = _select_parent(fitnesses, population, gen_seed)
            gen_seed += 1
            parent_b_idx = _select_parent(fitnesses, population, gen_seed)
            gen_seed += 1
            if parent_a_idx == parent_b_idx:
                parent_b_idx = (parent_a_idx + 1) % len(population)

            offspring = xfn(population[parent_a_idx],
                            population[parent_b_idx],
                            seed=gen_seed, **crossover_kwargs)
            gen_seed += 1

            # Mutation (with adaptive rate)
            mut_thresh = current_mutation_rate if adaptive_mutation else mutation_rate
            if torch.rand(1, generator=torch.Generator().manual_seed(gen_seed)).item() < mut_thresh:
                gen_seed += 1
                if mutation == "block_swap":
                    # block_swap needs a donor — use a random donor (may be HoF)
                    donor_idx = torch.randperm(len(donor_pool),
                                               generator=torch.Generator().manual_seed(gen_seed))[0].item()
                    gen_seed += 1
                    offspring = mfn(offspring, donor_pool[donor_idx],
                                    seed=gen_seed, **mutation_kwargs)
                else:
                    offspring = mfn(offspring, seed=gen_seed, **mutation_kwargs)
                gen_seed += 1

            new_pop.append(offspring)

        population = new_pop
        prev_fitnesses = list(raw_fitnesses)

    # Final evaluation of last generation
    raw_fitnesses = [fitness_fn(ind) for ind in population]
    gen_best_idx = max(range(len(raw_fitnesses)), key=lambda i: raw_fitnesses[i])
    if raw_fitnesses[gen_best_idx] > best_fitness:
        best_fitness = raw_fitnesses[gen_best_idx]
        best_state = {k: v.clone() for k, v in population[gen_best_idx].items()}
    # Final diversity
    final_diversity = _population_diversity(population) if len(population) > 1 else 0.0
    history.append({
        "generation": len(history),
        "best_fitness": best_fitness,
        "mean_fitness": sum(raw_fitnesses) / len(raw_fitnesses),
        "min_fitness": min(raw_fitnesses),
        "best_index": gen_best_idx,
        "diversity": final_diversity,
        "mutation_rate": current_mutation_rate,
    })
    if verbose:
        print(f"  [evolve] final: best={best_fitness:.4f}")

    # Save final best
    if save_fn is not None and out_dir is not None and best_state is not None:
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        best_path = str(Path(out_dir) / "best.safetensors")
        save_fn(best_state, best_path)

    return {
        "best": best_state,
        "best_fitness": best_fitness,
        "best_path": best_path,
        "history": history,
        "final_population": population,
        "hall_of_fame": hall_of_fame,
    }


# ---------------------------------------------------------------------------
# Depth upscaling (R50-5) — mergekit "passthrough" layer duplication
# ---------------------------------------------------------------------------
# Grows a checkpoint's depth with zero training: each output block i
# copies the weights of source block ``layer_map[i]``. Duplicated layers
# are NOT lossless (a duplicated Mamba/attention block changes the
# function), so this is a warm start for continued training, not a
# finished model — the copied blocks act as strong priors that typically
# recover within a few hundred steps (SOLAR/mergekit depth-upscaling).
#
# ForgeAI caveat: blocks are TYPED (Mamba-2 vs GQA attention). The layer
# map therefore only makes sense when duplicated source blocks share the
# type expected at their destination slot; the caller supplies the new
# ``layer_types`` ordering via --layer-types when the target config
# needs it.

def parse_layer_map(spec: str) -> list[int]:
    """Parse a layer-map spec like ``"0-15,8,9,10,16-23"`` → list[int].

    Ranges are inclusive on both ends (``0-3`` = [0,1,2,3]). Commas
    separate individual indices and ranges; whitespace is ignored.
    """
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"descending range {part!r} in layer map")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("empty layer map")
    return out


def depth_upscale(state: dict[str, torch.Tensor],
                  layer_map: list[int],
                  layer_types: list[str] | None = None,
                  ) -> tuple[dict[str, torch.Tensor], dict]:
    """Duplicate/reorder blocks per ``layer_map`` — returns (state, meta).

    Args:
        state: source state dict with ``blocks.{i}.``-prefixed keys.
        layer_map: source block index for each output block. Length =
            new depth; every entry must be < source depth.
        layer_types: optional per-output-block type labels ("mamba",
            "attention", ...). When omitted, derived by mapping each
            output slot to its source block index — the caller should
            still verify type consistency against the target config.

    Returns:
        (new_state, meta) where meta = {"layer_map", "n_layers",
        "layer_types"} — write it as a sidecar so the loader can
        rebuild the per-layer type schedule.
    """
    n_src = _n_blocks(state)
    if n_src == 0:
        raise ValueError("state dict has no blocks.* keys")
    for src in layer_map:
        if not (0 <= src < n_src):
            raise ValueError(
                f"layer_map entry {src} out of range for {n_src} blocks")

    # Partition keys into block-indexed and passthrough (non-block) sets.
    block_keys: dict[int, list[str]] = {i: [] for i in range(n_src)}
    passthrough: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        idx = _block_index(k)
        if idx is None:
            passthrough[k] = v
        else:
            block_keys[idx].append(k)

    out = dict(passthrough)
    for new_idx, src in enumerate(layer_map):
        for k in block_keys[src]:
            suffix = k.split(".", 2)[2]  # strip "blocks.{i}."
            out[f"blocks.{new_idx}.{suffix}"] = state[k].clone()

    derived_types = (
        list(layer_types) if layer_types is not None else None)
    if derived_types is not None and len(derived_types) != len(layer_map):
        raise ValueError(
            f"layer_types length {len(derived_types)} != "
            f"new depth {len(layer_map)}")

    meta = {
        "layer_map": list(layer_map),
        "n_layers": len(layer_map),
        "layer_types": derived_types,
        "source_n_layers": n_src,
    }
    return out, meta


def depth_layer_types(state: dict[str, torch.Tensor],
                      layer_map: list[int],
                      source_types: list[str]) -> list[str]:
    """Derive new per-layer types from the source schedule + layer map."""
    if len(source_types) != _n_blocks(state):
        raise ValueError(
            f"source_types length {len(source_types)} != "
            f"source depth {_n_blocks(state)}")
    return [source_types[src] for src in layer_map]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Model merging for ForgeAI (LFM2.5)")
    p.add_argument("--method", required=True,
                   choices=["linear", "task_arith", "slerp", "ties", "dare", "svd",
                            "crossover_blockwise", "crossover_block_random",
                            "crossover_uniform", "mutate_gaussian",
                            "mutate_quant_perturb", "mutate_block_swap", "evolve",
                            "depth"])
    p.add_argument("--base", default=None,
                   help="Base model checkpoint (required for task_arith/ties/dare/svd)")
    p.add_argument("--model-a", default=None, help="First model (slerp/linear 2-model)")
    p.add_argument("--model-b", default=None, help="Second model (slerp/linear 2-model)")
    p.add_argument("--models", nargs="*", default=None,
                   help="List of model checkpoints (linear/task_arith/ties/dare/svd)")
    p.add_argument("--weights", nargs="*", type=float, default=None,
                   help="Per-model weights (linear/task_arith/svd)")
    p.add_argument("--t", type=float, default=0.5, help="SLERP interpolation factor")
    p.add_argument("--density", type=float, default=0.5,
                   help="TIES: fraction of magnitudes to keep (0.5 = keep top 50%)")
    p.add_argument("--drop-rate", type=float, default=0.1, help="DARE: drop fraction")
    p.add_argument("--rank-ratio", type=float, default=0.5,
                   help="SVD: rank truncation ratio (0.5 = keep top half of singular values)")
    p.add_argument("--seed", type=int, default=0, help="DARE/evolutionary: RNG seed")
    p.add_argument("--out", required=True, help="Output checkpoint path")
    # Evolutionary operator args
    p.add_argument("--split-block", type=int, default=None,
                   help="crossover_blockwise: block index to split at")
    p.add_argument("--p", type=float, default=0.5,
                   help="crossover_uniform/block_random: prob tensor/block from B")
    p.add_argument("--sigma", type=float, default=0.01,
                   help="mutate_gaussian/quant_perturb: noise std dev")
    p.add_argument("--rate", type=float, default=0.01,
                   help="mutate_gaussian: fraction of elements to perturb")
    p.add_argument("--n-swaps", type=int, default=1,
                   help="mutate_block_swap: number of blocks to swap")
    p.add_argument("--n-generations", type=int, default=5,
                   help="evolve: number of generations")
    p.add_argument("--population-size", type=int, default=None,
                   help="evolve: target population size (default = initial)")
    p.add_argument("--mutation-rate", type=float, default=0.5,
                   help="evolve: probability each offspring is mutated")
    p.add_argument("--elitism", type=int, default=1,
                   help="evolve: number of top individuals carried forward")
    p.add_argument("--fitness", default="neg_norm",
                   help="evolve: fitness function (neg_norm=random, benchmark=GPU)")
    # Sophisticated selection + score rewarding args
    p.add_argument("--selection", default="tournament",
                   choices=["tournament", "rank", "roulette", "diversity"],
                   help="evolve: parent selection strategy")
    p.add_argument("--fitness-scaling", default="none",
                   choices=["none", "sigma", "rank"],
                   help="evolve: fitness scaling method")
    p.add_argument("--progress-bonus", type=float, default=0.0,
                   help="evolve: weight for rewarding improvement over parents")
    p.add_argument("--diversity-bonus", type=float, default=0.0,
                   help="evolve: weight for rewarding distance from centroid")
    p.add_argument("--adaptive-mutation", action="store_true",
                   help="evolve: dynamically adjust mutation rate based on diversity")
    p.add_argument("--hall-of-fame", type=int, default=0,
                   help="evolve: hall of fame size (0=disabled)")
    p.add_argument("--convergence-patience", type=int, default=0,
                   help="evolve: stop if no improvement for N gens (0=disabled)")
    p.add_argument("--layers", default=None,
                   help='depth: layer map spec, e.g. "0-15,8-23" (ranges '
                        'inclusive). Duplicated blocks warm-start deeper '
                        'models; write a .depth.json sidecar with the map.')
    p.add_argument("--layer-types", default=None,
                   help="depth: comma-separated per-output-block types "
                        "(mamba,...,attention); optional — defaults to "
                        "the source type schedule remapped by --layers.")
    args = p.parse_args()

    out_path = args.out
    if not out_path.endswith(".safetensors") and not out_path.endswith(".pt"):
        out_path = out_path + ".safetensors"

    # --- 2-model methods ---
    if args.method == "slerp":
        if not args.model_a or not args.model_b:
            p.error("--model-a and --model-b required for slerp")
        a = _load_state_dict(args.model_a)
        b = _load_state_dict(args.model_b)
        print(f"SLERP(t={args.t}): {args.model_a} <-> {args.model_b}")
        merged = merge_slerp(a, b, t=args.t)

    elif args.method == "linear" and args.model_a and args.model_b and not args.models:
        a = _load_state_dict(args.model_a)
        b = _load_state_dict(args.model_b)
        w = args.weights or [0.5, 0.5]
        print(f"Linear soup: {args.model_a} ({w[0]}) + {args.model_b} ({w[1]})")
        merged = merge_linear([a, b], w)

    # --- Evolutionary single-shot operators (2 parents) ---
    elif args.method in ("crossover_blockwise", "crossover_block_random",
                         "crossover_uniform"):
        if not args.model_a or not args.model_b:
            p.error(f"--model-a and --model-b required for {args.method}")
        a = _load_state_dict(args.model_a)
        b = _load_state_dict(args.model_b)
        if args.method == "crossover_blockwise":
            print(f"Blockwise crossover: {args.model_a} x {args.model_b} "
                  f"(split={args.split_block})")
            merged = crossover_blockwise(a, b, split_block=args.split_block,
                                         seed=args.seed)
        elif args.method == "crossover_block_random":
            print(f"Block-random crossover: p={args.p}")
            merged = crossover_block_random(a, b, p=args.p, seed=args.seed)
        else:
            print(f"Uniform crossover: p={args.p}")
            merged = crossover_uniform(a, b, p=args.p, seed=args.seed)

    elif args.method == "mutate_gaussian":
        if not args.model_a:
            p.error("--model-a required for mutate_gaussian")
        a = _load_state_dict(args.model_a)
        print(f"Gaussian mutation: sigma={args.sigma} rate={args.rate}")
        merged = mutate_gaussian(a, sigma=args.sigma, rate=args.rate,
                                 seed=args.seed)

    elif args.method == "mutate_quant_perturb":
        if not args.model_a:
            p.error("--model-a required for mutate_quant_perturb")
        a = _load_state_dict(args.model_a)
        print(f"Quant-scale perturbation: sigma={args.sigma}")
        merged = mutate_quant_perturb(a, sigma=args.sigma, seed=args.seed)

    elif args.method == "depth":
        if not args.model_a:
            p.error("--model-a required for depth (the source checkpoint)")
        if not args.layers:
            p.error('--layers required for depth, e.g. "0-15,8-23"')
        a = _load_state_dict(args.model_a)
        layer_map = parse_layer_map(args.layers)
        layer_types = (
            [t.strip() for t in args.layer_types.split(",")]
            if args.layer_types else None)
        print(f"Depth upscale: {args.model_a} "
              f"{_n_blocks(a)} -> {len(layer_map)} blocks, map={args.layers}")
        merged, meta = depth_upscale(a, layer_map, layer_types=layer_types)
        # Sidecar with the layer map for loader/config reconstruction.
        import json as _json
        sidecar = str(Path(out_path).with_suffix(".depth.json"))
        with open(sidecar, "w") as f:
            _json.dump(meta, f, indent=2)
        print(f"  layer map sidecar: {sidecar}")

    elif args.method == "mutate_block_swap":
        if not args.model_a or not args.model_b:
            p.error("--model-a (recipient) and --model-b (donor) required")
        a = _load_state_dict(args.model_a)
        b = _load_state_dict(args.model_b)
        print(f"Block swap: {args.n_swaps} blocks from {args.model_b} -> {args.model_a}")
        merged = mutate_block_swap(a, b, n_swaps=args.n_swaps, seed=args.seed)

    elif args.method == "evolve":
        if not args.models or len(args.models) < 2:
            p.error("--models (>=2 checkpoints) required for evolve")
        pop = [_load_state_dict(m) for m in args.models]
        print(f"Evolutionary merge: {len(pop)} parents, "
              f"{args.n_generations} generations, pop_size={args.population_size}")

        # Default fitness: negative L2 norm of weights (cheap proxy —
        # smaller-norm models tend to generalize better). Use --fitness
        # benchmark to plug in a real GPU evaluator from ForgeEngine.
        if args.fitness == "neg_norm":
            def fitness_fn(state):
                return -sum(v.float().norm().item() for v in state.values()
                            if v.is_floating_point()) / 1e6
        else:
            p.error(f"unknown fitness: {args.fitness} (use 'neg_norm' or "
                    f"call ForgeEngine.evolve_merge for 'benchmark')")

        result = evolve(
            pop, fitness_fn,
            n_generations=args.n_generations,
            population_size=args.population_size,
            mutation_rate=args.mutation_rate,
            elitism=args.elitism,
            seed=args.seed,
            save_fn=save_checkpoint,
            out_dir=str(Path(out_path).parent),
            selection=args.selection,
            fitness_scaling=args.fitness_scaling,
            progress_bonus=args.progress_bonus,
            diversity_bonus=args.diversity_bonus,
            adaptive_mutation=args.adaptive_mutation,
            hall_of_fame_size=args.hall_of_fame,
            convergence_patience=args.convergence_patience,
        )
        merged = result["best"]
        print(f"Evolution complete: best_fitness={result['best_fitness']:.4f}")
        print(f"  history: {[h['best_fitness'] for h in result['history']]}")
        if result.get("hall_of_fame"):
            print(f"  hall of fame: {len(result['hall_of_fame'])} individuals")

    # --- N-model methods requiring base ---
    else:
        if not args.base:
            p.error(f"--base required for {args.method}")
        if not args.models:
            p.error(f"--models required for {args.method}")
        base = _load_state_dict(args.base)
        ft_models = [_load_state_dict(m) for m in args.models]
        task_vectors = [_task_vectors(m, base) for m in ft_models]
        print(f"{args.method}: base={args.base} + {len(ft_models)} task vectors")

        if args.method == "linear":
            merged = merge_linear([base, *ft_models], args.weights)
        elif args.method == "task_arith":
            merged = merge_task_arith(base, task_vectors, args.weights)
        elif args.method == "ties":
            merged = merge_ties(base, task_vectors, density=args.density)
        elif args.method == "dare":
            merged = merge_dare(base, task_vectors, drop_rate=args.drop_rate,
                                seed=args.seed)
        elif args.method == "svd":
            merged = merge_svd(base, task_vectors, rank_ratio=args.rank_ratio,
                               scales=args.weights)
        else:
            p.error(f"unknown method: {args.method}")

    save_checkpoint(merged, out_path)
    print(f"Saved merged model to {out_path} ({len(merged)} tensors)")


if __name__ == "__main__":
    main()
