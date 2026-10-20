"""Reverse-consistency memory for FluxLM — backward bigram sidecar.

Motivation (EXP-1 prevtok finding): greedy-generated text is measurably
backward-INCONSISTENT — p(x_{t-1} | x_t) under a reverse model is worse
than on real text — and reranking candidates by forward+backward joint
score fixes it nearly for free.  On a transformer that needed a trained
head; on FluxLM the reverse model is just another count table, written
for free at ingest:

    write: rev[x_t][x_{t-1}] += 1            (O(1) per token)
    read:  rev_logp(c, p) = log p(x_{t-1}=p | x_t=c)
           smoothed by unigram backoff + uniform floor

The decode-side use is candidate rerank: take the mixture's top-K
candidates and score `fwd_logp(c) + alpha * rev_logp(c, last)` — the
self-correcting decode from the transformer experiment, now as pure
memory (no parameters, journaled like every other live surface).

Host-dict prototype in the SuffixMemory style; if this graduates it
belongs in _GPUCellTables next to the forward cells.
"""
from __future__ import annotations

import math


class ReverseMem:
    """rev[succ] -> {prev: count} + unigram stats.

    Smoothing mirrors NGramMemory: row probability blended with the
    unigram distribution (uni_mix) and a uniform floor so a known
    successor with a novel predecessor gives a finite lift, never -inf.
    """

    def __init__(self, vocab_size: int, alpha: float = 1.0,
                 uni_mix: float = 0.1, floor: float = 0.01):
        self.vocab = int(vocab_size)
        self.alpha = float(alpha)
        self.uni_mix = float(uni_mix)
        self.floor = float(floor)
        self.table: dict[int, dict[int, int]] = {}
        self.unigram = [0] * self.vocab
        self.total = 0
        self._last: int | None = None

    # ── write path (O(1) per token) ───────────────────────────────────

    def feed_one(self, tok: int) -> None:
        """Observe one committed token: rev[tok][prev] += 1."""
        tok = int(tok)
        if self._last is not None:
            row = self.table.get(tok)
            if row is None:
                row = self.table[tok] = {}
            row[self._last] = row.get(self._last, 0) + 1
        if 0 <= tok < self.vocab:
            self.unigram[tok] += 1
        self.total += 1
        self._last = tok

    def feed(self, ids) -> int:
        for t in ids:
            self.feed_one(t)
        return self.total

    # ── read path ─────────────────────────────────────────────────────

    def rev_logp(self, cand: int, prev: int) -> float:
        """log p(x_{t-1}=prev | x_t=cand), smoothed."""
        cand, prev = int(cand), int(prev)
        row = self.table.get(cand)
        tot = self.total or 1
        pu = (self.unigram[prev] / tot) if 0 <= prev < self.vocab else 0.0
        if not row:
            return math.log(self.uni_mix * pu + self.floor)
        c = row.get(prev, 0)
        rs = sum(row.values())
        p = (c + self.uni_mix * pu * rs + self.floor) \
            / (rs * (1.0 + self.uni_mix + self.floor))
        return math.log(max(p, 1e-12))

    def rev_probs(self, cand: int) -> dict[int, float] | None:
        """{prev: prob} for a candidate successor — None if unseen."""
        row = self.table.get(int(cand))
        if not row:
            return None
        rs = sum(row.values())
        return {p: c / rs for p, c in row.items()}

    # ── live-surface contract (journaled like the rest) ───────────────

    def snapshot_state(self) -> dict:
        return {"table": {c: dict(r) for c, r in self.table.items()},
                "unigram": list(self.unigram),
                "total": self.total, "last": self._last}

    def restore_state(self, snap: dict) -> None:
        self.table = {c: dict(r) for c, r in snap["table"].items()}
        self.unigram = list(snap["unigram"])
        self.total = snap["total"]
        self._last = snap["last"]

    def reset_state(self) -> None:
        self.table.clear()
        self.unigram = [0] * self.vocab
        self.total = 0
        self._last = None

    def info(self) -> dict:
        return {"type": "reverse_mem", "rows": len(self.table),
                "total": self.total, "alpha": self.alpha}


# ── decode-time rerank ────────────────────────────────────────────────

def generate_rerank(model, rev: ReverseMem, prompt_ids, max_new: int = 64,
                    k: int = 4, learn: bool = True, tag: str = "gen",
                    margin: float | None = None,
                    generator=None) -> dict:
    """FluxLM greedy decode with backward-consistency rerank.

    Each step: mixture logits -> top-K candidates -> rescore
    ``fwd_logp(c) + rev.alpha * rev_logp(c, last_token)`` -> argmax.
    No extra model calls (the rerank is host-side table lookups), so
    forwards/token stays 1.0 — the mechanism is free vs K model
    forwards on a transformer.

    ``margin``: if set, rerank only when the mixture is contested
    (top1-top2 logp gap < margin) — the consistency/accuracy tension
    says consistency pressure should not override a confident
    prediction, only break ties.

    Returns dict(ids=generated, corrections=n_swapped, steps=n).
    """
    import torch
    model.ingest([int(t) for t in prompt_ids], tag=tag, learn=learn)
    rev.feed(prompt_ids)
    out: list[int] = []
    corrections = contested = 0
    for _ in range(max_new):
        lg = torch.as_tensor(model._predict()).float()
        k_c = min(k, lg.numel())
        topv, topi = lg.topk(k_c)
        fwd = torch.log_softmax(topv, dim=-1)
        last = out[-1] if out else int(prompt_ids[-1])
        if margin is not None and k_c > 1 \
                and float(fwd[0] - fwd[1]) >= margin:
            best = 0                      # confident -> trust mixture
        else:
            contested += 1
            scores = [float(fwd[j])
                      + rev.alpha * rev.rev_logp(int(topi[j]), last)
                      for j in range(k_c)]
            best = int(scores.index(max(scores)))
        x = int(topi[best])
        if best != 0:
            corrections += 1
        out.append(x)
        if learn:
            model._learn_token(x)
        else:
            model._push(x)
        rev.feed_one(x)
    return {"ids": out, "corrections": corrections,
            "contested": contested, "steps": max_new}


def joint_prev_nll(model, rev: ReverseMem, ids) -> dict:
    """Backward consistency of a token stream under the reverse table:
    mean -rev_logp(x_t, x_{t-1}).  Lower = more self-consistent text."""
    losses = [-rev.rev_logp(ids[t + 1], ids[t])
              for t in range(len(ids) - 1)]
    return {"prev_nll": sum(losses) / max(len(losses), 1),
            "n_pairs": len(losses)}
