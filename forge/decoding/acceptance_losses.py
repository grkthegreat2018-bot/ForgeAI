"""Acceptance-aware draft-head losses (arXiv 2609.24150).

Speculative-decoding draft heads are usually trained with plain
cross-entropy, which optimizes next-token likelihood — not the quantity
that actually gates speedup: the expected number of draft tokens the
target model accepts per verification window. These two losses optimize
acceptance directly, each matched to a verification mode.

eal_loss — Expected Accepted Length (greedy verification)
    For window positions t = 1..W let a_t = argmax_v p_target(v|t) be the
    target's greedy pick and

        p_t = softmax(draft_logits_t)[a_t]

    the probability mass the draft assigns to that pick. Under greedy
    verification a draft token is accepted iff it equals the target's
    argmax; p_t is the differentiable relaxation of that indicator. The
    expected accepted length is then

        EAL = Σ_{t=1..W} Π_{j<=t} p_j

    and the loss is -log(EAL / W) ∈ [0, ∞): 0 when the draft puts all its
    mass on the target's greedy picks at every position, increasing
    monotonically with mismatch. (The unnormalized variant -EAL is
    available via ``expected_accepted_length``.)

wtv_loss — Window Total-Variation (sampling verification)
    Under speculative sampling, the per-position acceptance probability
    is 1 - TV(q_t, p_t) with q_t = softmax(draft_t / T),
    p_t = softmax(target_t / T). Position t is only reached if all
    earlier positions were accepted, which weights each term by the
    sequential acceptance product:

        L = Σ_{t=1..W} w_t · TV_t,   w_t = Π_{j<t} (1 - TV_j)

    Minimizing this puts the most gradient on the earliest — most likely
    to be verified — positions, matching the acceptance profile rather
    than a uniform per-position average.

Both accept (B, W, V) or (W, V) logits and return a scalar tensor.
Intended call site: ``DSparkTrainer.compute_loss`` in dspark.py via
``loss_type="eal"|"wtv"``; usable standalone anywhere draft and target
logits over a draft window are available.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _ensure_batched(t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize (W, V) or (B, W, V) to (B, W, V). Returns (tensor, was_2d)."""
    if t.dim() == 2:
        return t.unsqueeze(0), True
    if t.dim() == 3:
        return t, False
    raise ValueError(f"expected (W, V) or (B, W, V) logits, got shape {tuple(t.shape)}")


def expected_accepted_length(draft_logits: torch.Tensor,
                             target_logits: torch.Tensor) -> torch.Tensor:
    """EAL = Σ_t Π_{j<=t} p_j for greedy verification. Returns (B,) tensor."""
    draft, _ = _ensure_batched(draft_logits)
    target, _ = _ensure_batched(target_logits)
    target_pick = target.argmax(dim=-1)  # (B, W)
    draft_probs = F.softmax(draft.float(), dim=-1)
    p_match = draft_probs.gather(
        -1, target_pick.unsqueeze(-1)).squeeze(-1)  # (B, W)
    return torch.cumprod(p_match.clamp(min=0.0), dim=-1).sum(dim=-1)  # (B,)


def eal_loss(draft_logits: torch.Tensor,
             target_logits: torch.Tensor) -> torch.Tensor:
    """Expected Accepted Length loss for GREEDY speculative verification.

    L = -log(EAL / W) where EAL = Σ_{t=1..W} Π_{j<=t} p_match_j and
    p_match_j is the draft's softmax mass on the target's argmax at
    position j. Zero when the draft is a one-hot match to the target's
    greedy picks at every window position; monotonically increasing with
    mismatch. Draft logits carry gradients; target logits are treated as
    fixed (argmax selection is non-differentiable by nature).

    Args:
        draft_logits: (W, V) or (B, W, V) draft-head logits over the window.
        target_logits: same shape — target model logits (no grad needed).

    Returns:
        scalar loss tensor.
    """
    draft, _ = _ensure_batched(draft_logits)
    target, _ = _ensure_batched(target_logits)
    W = draft.shape[1]
    eal = expected_accepted_length(draft, target)  # (B,)
    return -torch.log(eal.clamp(min=1e-8) / W).mean()


def wtv_loss(draft_logits: torch.Tensor, target_logits: torch.Tensor,
             temperature: float = 1.0) -> torch.Tensor:
    """Window Total-Variation loss for SAMPLING speculative verification.

    Per position: TV_t = 0.5 · ‖softmax(draft_t/T) − softmax(target_t/T)‖₁
    is the rejection probability under speculative sampling. The window
    objective weights each position by the probability it is reached:

        L = Σ_{t=1..W} w_t · TV_t,   w_t = Π_{j<t} (1 − TV_j),  w_1 = 1.

    Earlier positions (verified most often) get the largest weight, so
    gradient effort is allocated where acceptance is actually decided.

    Args:
        draft_logits: (W, V) or (B, W, V) draft-head logits over the window.
        target_logits: same shape — target model logits (no grad needed).
        temperature: softmax temperature matching the sampling temperature
            used at decode time (default 1.0).

    Returns:
        scalar loss tensor.
    """
    draft, _ = _ensure_batched(draft_logits)
    target, _ = _ensure_batched(target_logits)
    q = F.softmax(draft.float() / temperature, dim=-1)
    p = F.softmax(target.float().detach() / temperature, dim=-1)
    tv = 0.5 * (q - p).abs().sum(dim=-1)  # (B, W)
    accept = (1.0 - tv).clamp(min=0.0)
    # w_t = Π_{j<t} accept_j: cumulative product shifted right by one.
    w = torch.cumprod(accept, dim=-1)
    w = torch.cat([torch.ones_like(w[..., :1]), w[..., :-1]], dim=-1)
    return (w * tv).sum(dim=-1).mean()
