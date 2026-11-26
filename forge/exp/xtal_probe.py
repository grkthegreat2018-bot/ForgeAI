"""EXP-XTAL measurement machinery — per-tag loss decomposition, gradient
alignment, embedding geometry. Greenfield: no per-group gradient or
alignment utilities existed in the repo (closest idiom: sift.py's
cross-batch accumulation). All measurements run under no_grad or use
throwaway backward passes that never touch the optimizer.

Conventions:
- tokens/tags tensors are [B, T+1]; position i is scored as a prediction
  target with tag tags[:, i]. Input = tokens[:, :-1], target = tokens[:, 1:],
  so target position index j (0-based in the shifted frame) carries
  tag tags[:, j+1].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from forge.exp.xtal_grammar import TAG_A, TAG_AH, TAG_B, TAG_C, TAG_NAMES


def per_tag_losses(model, tokens, tags) -> dict:
    """Mean CE per tag over a batch. Returns {tag: (loss_sum, count)} so
    callers can pool across batches correctly."""
    with torch.no_grad():
        logits = model(tokens[:, :-1])[0]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        tokens[:, 1:].reshape(-1), reduction="none").view(tokens.size(0), -1)
    tgt_tags = tags[:, 1:]
    out = {}
    for t in torch.unique(tgt_tags).tolist():
        mask = tgt_tags == t
        out[int(t)] = (float(ce[mask].sum()), int(mask.sum()))
    return out


def eval_fixed_batches(model, eval_sets: dict) -> dict:
    """eval_sets: {name: (tokens, tags)}. Returns {name: mean_ce} pooled
    over the batch, plus per-tag breakdown as '{name}/tag_<n>'."""
    res = {}
    for name, (toks, tags) in eval_sets.items():
        parts = per_tag_losses(model, toks, tags)
        tot_sum = sum(s for s, _ in parts.values())
        tot_n = sum(n for _, n in parts.values())
        res[name] = tot_sum / max(tot_n, 1)
        for t, (s, n) in parts.items():
            res[f"{name}/{TAG_NAMES.get(t, t)}"] = s / max(n, 1)
    return res


def batch_loss(model, tokens, tags, tag_filter: int | None = None) -> torch.Tensor:
    """Differentiable mean CE over a batch, optionally restricted to one
    tag's positions. Used for both training steps (tag_filter=None) and
    segregated gradient probes."""
    logits = model(tokens[:, :-1])[0]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        tokens[:, 1:].reshape(-1), reduction="none").view(tokens.size(0), -1)
    if tag_filter is None:
        return ce.mean()
    mask = (tags[:, 1:] == tag_filter).float()
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def flat_grad(model, loss: torch.Tensor) -> torch.Tensor:
    """Full-parameter gradient vector of loss, without touching optimizer
    state. Zeroes grads, backward, collects, zeroes again."""
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().reshape(-1).clone())
    model.zero_grad(set_to_none=True)
    return torch.cat(grads)


def grad_alignment(model, probe_sets: dict) -> dict:
    """Cosine alignment between full-parameter gradient vectors of
    segregated per-rule batches. probe_sets: {name: (tokens, tags)}.
    Uses mean batch CE (all positions) as each rule's 'task gradient'.
    Returns {'align/X_Y': cos} for all pairs plus per-set grad norms."""
    vecs = {}
    norms = {}
    for name, (toks, tags) in probe_sets.items():
        loss = batch_loss(model, toks, tags)
        g = flat_grad(model, loss)
        vecs[name] = g
        norms[name] = float(g.norm())
    out = {}
    keys = sorted(vecs)
    for i, a in enumerate(keys):
        out[f"gnorm/{a}"] = norms[a]
        for b in keys[i + 1:]:
            denom = (vecs[a].norm() * vecs[b].norm()).clamp_min(1e-12)
            out[f"align/{a}_{b}"] = float((vecs[a] * vecs[b]).sum() / denom)
    return out


def tag_grad_alignment(model, eval_sets: dict, tags_of_interest=(TAG_A, TAG_C)) -> dict:
    """Finer-grained alignment: gradient of loss restricted to positions of
    ONE tag within each eval set — isolates rule-specific learning signal
    from the shared background (embedding/unembedding calibration)."""
    vecs = {}
    for t in tags_of_interest:
        kind = {TAG_A: "A", TAG_AH: "AH", TAG_B: "B", TAG_C: "C"}[t]
        toks, tags = eval_sets[kind]
        loss = batch_loss(model, toks, tags, tag_filter=t if kind != "AH" else TAG_AH)
        vecs[TAG_NAMES[t]] = flat_grad(model, loss)
    out = {}
    keys = sorted(vecs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            denom = (vecs[a].norm() * vecs[b].norm()).clamp_min(1e-12)
            out[f"tagalign/{a}_{b}"] = float((vecs[a] * vecs[b]).sum() / denom)
    return out


def embedding_geometry(model, n_classes: int = 8, members: int = 8,
                       item0: int = 16, n_holdout: int = 2) -> dict:
    """Within-class vs across-class cosine similarity of input embedding
    rows — direct observation of equivalence-class discovery. As rule A
    crystallizes, members of the same class should cluster (they are
    behaviorally identical as triggers).

    Split by train vs holdout members: holdout members never appear in
    trigger roles — their only class signal is successor-side statistics
    (the controlled leak). 'emb/holdout_own_cls' tracks whether their
    embeddings drift toward their TRUE class cluster (train-member
    centroid) vs other classes — direct observation of distributional
    class inference in weight space."""
    emb = model.embed.weight.detach().float()
    rows = emb[item0: item0 + n_classes * members]  # [n_items, d]
    rows = F.normalize(rows, dim=-1)
    sim = rows @ rows.T
    cls = torch.arange(n_classes * members, device=sim.device) // members
    same = cls[:, None] == cls[None, :]
    eye = torch.eye(len(cls), dtype=torch.bool, device=sim.device)
    mem_idx = torch.arange(len(cls), device=sim.device) % members
    holdout = mem_idx >= (members - n_holdout)
    train = ~holdout
    within = sim[same & ~eye & train[:, None] & train[None, :]].mean()
    across = sim[~same & train[:, None] & train[None, :]].mean()
    out = {"emb/within_class_cos": float(within),
           "emb/across_class_cos": float(across),
           "emb/class_gap": float(within - across)}
    # holdout members: mean cos to own-class train members minus mean cos
    # to other classes' train members — positive means joining the cluster
    ho_rows = rows[holdout]
    if len(ho_rows) > 0:
        ho_cls = cls[holdout]
        tr_rows = rows[train]
        tr_cls = cls[train]
        sims = ho_rows @ tr_rows.T  # [n_ho, n_tr]
        own = torch.zeros(len(ho_rows), device=sim.device)
        other = torch.zeros(len(ho_rows), device=sim.device)
        for i in range(len(ho_rows)):
            own_m = tr_cls == ho_cls[i]
            own[i] = sims[i][own_m].mean()
            other[i] = sims[i][~own_m].mean()
        out["emb/holdout_own_cls"] = float(own.mean())
        out["emb/holdout_other_cls"] = float(other.mean())
        out["emb/holdout_cluster_gap"] = float((own - other).mean())
    return out
