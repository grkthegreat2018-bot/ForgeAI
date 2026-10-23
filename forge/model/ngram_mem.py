"""NGramMemory — exact-suffix associative store for LiveT.

Counterfactual experiment: TitanMemory/GDN2 retrieve by *fuzzy* hidden-
state keys (context vectors blur across positions). This module stores
the same context->successor associations keyed by *exact* token-suffix
hashes — the infini-gram/FLUX semantics — as a sparse logit sidecar.

    write: counts[order][mix64(suffix)] [next_token] += 1   (journalable)
    read:  lift[t] = gate * w_o * log P(next | order-o suffix)
           added to logits — pure retrieval, no learned decode needed.

Not a replacement for the fuzzy channels: exact keys give zero-noise
verbatim recall but cannot generalize; hidden-state keys blur but
interpolate. The hybrid (exact + fuzzy) is what this probe measures.

Journaled like the rest of the live surface: snapshot_state()/
restore_state() deep-copy the count dicts; a revert pops pre-call
snapshots. Host-side dict tables for the research prototype — the
_GPUCellTables path exists in forge/model/flux if this graduates.

Also hosts ``SuffixMemory`` (R65): the production streaming variant —
the FluxLM-style per-order suffix→successor tables ported for engine
use on foreign checkpoints.  Exact verbatim recall over an unbounded
token stream at bounded host RAM: the context-independence piece for
KV-cache models whose sliding window would otherwise forget names,
facts and established beats.
"""
from __future__ import annotations

import numpy as np
import torch


def _mix64(x: int) -> int:
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 33
    return x


class NGramMemory:
    """Per-order suffix -> successor count tables + gated logit lifts.

    orders: tuple of context lengths (e.g. (4, 8)). counts[order] maps
    the rolling suffix hash to an int32 vocab-count vector. `backoff`
    shorter orders smooth the longest order's read (weighted by count).
    """

    def __init__(self, vocab_size: int, orders=(4, 8), gate: float = 0.0,
                 order_weight: float = 0.5):
        self.vocab = vocab_size
        self.orders = tuple(sorted(orders))
        self.gate = gate                    # 0 = lossless sidecar at init
        self.order_weight = order_weight    # per-order discount
        self.tables = {o: {} for o in self.orders}
        self.unigram = np.zeros(vocab_size, dtype=np.float64)
        self.total = 0

    # ── keying ──────────────────────────────────────────────────────────

    def _key(self, ids: torch.Tensor, end: int, order: int) -> int:
        h = 0x9E3779B97F4A7C15
        for tok in ids[max(0, end - order):end].tolist():
            h = _mix64(h ^ (tok + 1))
        return h or 1

    # ── live surface contract (matches LiveSession) ─────────────────────

    def snapshot_state(self) -> dict:
        return {
            "tables": {o: {k: v.copy() for k, v in t.items()}
                       for o, t in self.tables.items()},
            "unigram": self.unigram.copy(),
            "total": self.total,
        }

    def restore_state(self, snap: dict) -> None:
        self.tables = {o: dict(t) for o, t in snap["tables"].items()}
        self.unigram = snap["unigram"].copy()
        self.total = snap["total"]

    def reset_state(self) -> None:
        self.tables = {o: {} for o in self.tables}
        self.unigram[:] = 0
        self.total = 0

    def update(self, ids: torch.Tensor, **_kw) -> dict:
        """Write every position's suffix->successor count. ids: (B,T)."""
        ids1 = ids.reshape(-1)
        for o, tab in self.tables.items():
            for t in range(ids1.numel() - 1):
                key = self._key(ids1, t + 1, o)
                row = tab.get(key)
                if row is None:
                    row = np.zeros(self.vocab, dtype=np.int32)
                    tab[key] = row
                row[int(ids1[t + 1])] += 1
        for tok in ids1.tolist():
            self.unigram[int(tok)] += 1
            self.total += 1
        return {"rows": sum(len(t) for t in self.tables.values())}

    # ── read path ───────────────────────────────────────────────────────

    def lift(self, ids: torch.Tensor) -> torch.Tensor:
        """(B,T,V) logit lift = gate * Σ_o w_o * log P_o(next|suffix)."""
        B, T = ids.shape
        device = ids.device
        out = torch.zeros(B, T, self.vocab, device=device)
        flat = ids.reshape(-1)
        uni = self.unigram / max(self.total, 1)
        for o, tab in self.tables.items():
            w = self.order_weight ** (self.orders[-1] - o)
            for t in range(flat.numel()):
                row = tab.get(self._key(flat, t + 1, o))
                if row is None or row.sum() == 0:
                    continue
                p = (torch.from_numpy(row).float() +
                     0.1 * self.vocab * torch.from_numpy(uni).float()
                     + 0.01)
                p = p / p.sum()
                b, tt = divmod(t, T)
                out[b, tt] += (w * p.log()).to(device)
        return out * self.gate


class SuffixMemory:
    """Streaming exact-suffix sidecar for KV-cache models (R65).

    The FluxLM mechanism ported onto a transformer: per-order tables map
    the rolling hash of the last ``o`` context tokens to sparse successor
    counts ``{hash -> {tok: count}}``.  At decode time the last-position
    suffix keys are looked up and turned into an additive logit lift —
    exact verbatim recall of anything ever seen in the stream, no matter
    how far back the KV window evicted it.

    Unlike the LiveT ``NGramMemory`` prototype this variant is built for
    unbounded streams and big vocabs:

      - sparse ``{tok: count}`` rows (no int32[V] per suffix — a 128k-vocab
        stream would need ~512KB/row dense; sparse rows cost ~tens of bytes),
      - bounded: per-order ``max_cells`` cap; on overflow all counts halve
        (FluxLM-style decay — ratios survive, cold cells die),
      - incremental: ``feed``/``feed_one`` maintain an internal token tail
        and rolling writes in O(orders) per token,
      - doubles as a prompt-lookup drafter via ``draft(n)``.

    All state is host-side; ``lift_last`` materializes a dense (V,) lift
    (one np pass over the vocab — µs) which the caller adds to the
    last-position logits before sampling.
    """

    def __init__(self, vocab_size: int,
                 orders=(4, 8, 16, 32, 64),
                 gate: float = 1.0,
                 order_weight: float = 0.7,
                 max_cells_per_order: int = 200_000,
                 unigram_mix: float = 0.1,
                 uniform_floor: float = 0.01):
        self.vocab = int(vocab_size)
        self.orders = tuple(sorted(int(o) for o in orders))
        self.max_order = self.orders[-1]
        self.gate = float(gate)
        self.order_weight = float(order_weight)
        self.max_cells = int(max_cells_per_order)
        self.unigram_mix = float(unigram_mix)
        self.uniform_floor = float(uniform_floor)

        self.tables: dict[int, dict[int, dict[int, int]]] = {
            o: {} for o in self.orders}
        self.unigram = np.zeros(self.vocab, dtype=np.float64)
        self.total = 0
        self._decays = 0
        # Internal tail: the last ``max_order`` observed token ids.  Single
        # source of truth — callers never pass context explicitly.
        self._tail: list[int] = []

    # ── write path ──────────────────────────────────────────────────────

    def _suffix_key(self, toks: list[int], order: int) -> int:
        h = 0x9E3779B97F4A7C15
        for tok in toks[-order:]:
            h = _mix64(h ^ (tok + 1))
        return h or 1

    def _write_pair(self, succ_tok: int) -> None:
        """Record ``self._tail -> succ_tok`` at every order."""
        for o in self.orders:
            if len(self._tail) >= o:
                key = self._suffix_key(self._tail, o)
                tab = self.tables[o]
                row = tab.get(key)
                if row is None:
                    if len(tab) >= self.max_cells:
                        self._decay(o)
                        row = tab.get(key)
                        if row is None:
                            row = {}
                            tab[key] = row
                    else:
                        row = {}
                        tab[key] = row
                row[succ_tok] = row.get(succ_tok, 0) + 1

    def _decay(self, order: int) -> None:
        """Halve every count in ``tables[order]``; drop rows that die."""
        tab = self.tables[order]
        for key in list(tab.keys()):
            row = tab[key]
            for tok in list(row.keys()):
                row[tok] //= 2
                if row[tok] == 0:
                    del row[tok]
            if not row:
                del tab[key]
        self._decays += 1

    def feed(self, ids) -> int:
        """Bulk-ingest a token-id sequence (prompt or chunk): writes every
        position's suffix→successor pair and advances the tail."""
        for tok in ids:
            self.feed_one(int(tok))
        return len(self._tail)

    def feed_one(self, tok: int) -> None:
        """Observe one accepted/emitted token (O(orders))."""
        self._write_pair(tok)
        if 0 <= tok < self.vocab:
            self.unigram[tok] += 1
        self.total += 1
        self._tail.append(tok)
        if len(self._tail) > self.max_order:
            del self._tail[: len(self._tail) - self.max_order]

    # ── read path ───────────────────────────────────────────────────────

    def _row_probs(self, order: int) -> tuple[dict[int, float] | None, float]:
        """Successor distribution for the current tail at ``order``.

        Returns (probs_dict, backoff_weight) — probs over observed
        successors only (uniform/unigram floors added at combine time).
        """
        if len(self._tail) < order:
            return None, 0.0
        row = self.tables[order].get(self._suffix_key(self._tail, order))
        if not row:
            return None, 0.0
        total = sum(row.values())
        return {t: c / total for t, c in row.items()}, float(total)

    def lift_last(self, device=None, dtype=torch.float32,
                  gate: float | None = None) -> torch.Tensor:
        """(V,) additive logit lift for the next token, given the tail.

        Geometric mixture over orders: lift = gate * Σ_o w_o * log p_o,
        where p_o is the order-o successor distribution smoothed by the
        unigram table and a uniform floor (R64 fix: a known suffix with a
        novel successor must produce a finite lift, not -inf).

        ``gate`` overrides ``self.gate`` for this call only — per-request
        overrides must not write the shared sidecar's ``.gate``.
        """
        uni = self.unigram / max(self.total, 1)
        floor = np.full(self.vocab, np.log(self.uniform_floor),
                        dtype=np.float32)
        lift = np.zeros(self.vocab, dtype=np.float32)
        max_o = self.orders[-1]
        matched = 0
        for o in self.orders:
            probs, _cnt = self._row_probs(o)
            if probs is None:
                continue
            matched += 1
            w = self.order_weight ** (max_o - o)
            for tok, p in probs.items():
                # p smoothed: count-prop + unigram backoff + uniform floor
                p_s = p + self.unigram_mix * uni[tok] + self.uniform_floor
                lift[tok] += w * np.log(p_s)
        if matched == 0:
            return torch.zeros(self.vocab, dtype=dtype, device=device)
        # Blend: matched successors keep their mixture lift; unseen tokens
        # get a mild negative pull (scale the floor so a memory hit is
        # always distinguishable but never a hard veto).
        g = self.gate if gate is None else float(gate)
        out = g * (lift + 0.25 * floor)
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    def draft(self, n: int = 4) -> list[int]:
        """Greedy suffix continuation of the tail — the prompt-lookup
        draft a speculative verifier can check.  Follows the deepest
        matching order's argmax successor, rolling the virtual tail."""
        tail = list(self._tail)
        out: list[int] = []
        for _ in range(n):
            best = None
            for o in reversed(self.orders):  # deepest first
                if len(tail) < o:
                    continue
                row = self.tables[o].get(self._suffix_key(tail, o))
                if row:
                    best = max(row.items(), key=lambda kv: kv[1])[0]
                    break
            if best is None:
                break
            out.append(best)
            tail.append(best)
            if len(tail) > self.max_order:
                tail = tail[-self.max_order:]
        return out

    # ── sidecar contract (snapshot/restore like LiveT surfaces) ─────────

    def snapshot_state(self) -> dict:
        return {
            "tables": {o: {k: dict(r) for k, r in t.items()}
                       for o, t in self.tables.items()},
            "unigram": self.unigram.copy(),
            "total": self.total,
            "tail": list(self._tail),
            "decays": self._decays,
        }

    def restore_state(self, snap: dict) -> None:
        self.tables = {o: {k: dict(r) for k, r in t.items()}
                       for o, t in snap["tables"].items()}
        self.unigram = snap["unigram"].copy()
        self.total = snap["total"]
        self._tail = list(snap["tail"])
        self._decays = snap.get("decays", 0)

    def reset_state(self) -> None:
        self.tables = {o: {} for o in self.orders}
        self.unigram[:] = 0
        self.total = 0
        self._tail = []
        self._decays = 0

    def info(self) -> dict:
        return {
            "type": "suffix_memory",
            "orders": self.orders,
            "cells": {o: len(t) for o, t in self.tables.items()},
            "total_tokens": self.total,
            "gate": self.gate,
            "decays": self._decays,
            "tail_len": len(self._tail),
        }
