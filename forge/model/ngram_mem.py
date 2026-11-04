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
                 order_weight: float = 0.5, write_decay: float = 0.5,
                 row_mass: float = 8.0):
        self.vocab = vocab_size
        self.orders = tuple(sorted(orders))
        self.gate = gate                    # 0 = lossless sidecar at init
        self.order_weight = order_weight    # per-order discount
        self.write_decay = float(write_decay)  # EMA rate: 1 = last-write-wins
        self.row_mass = float(row_mass)     # effective count mass in lift()
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
                    row = np.zeros(self.vocab, dtype=np.float32)
                    tab[key] = row
                # EMA successor distribution (normalized rows): recency-
                # weighted so an overwritten fact's latest value dominates
                # exponentially fast, while lift mass stays constant —
                # raw count decay dilutes the smoothed lift uniformly
                a = self.write_decay
                row *= 1.0 - a
                row[int(ids1[t + 1])] += a
                row /= row.sum()
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
        for i, (o, tab) in enumerate(self.tables.items()):
            w = self.order_weight ** (len(self.orders) - 1 - i)
            for t in range(flat.numel()):
                row = tab.get(self._key(flat, t + 1, o))
                if row is None or row.sum() == 0:
                    continue
                p = (self.row_mass * torch.from_numpy(row).float() +
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
                 uniform_floor: float = 0.01,
                 write_decay: float = 0.5):
        self.vocab = int(vocab_size)
        self.orders = tuple(sorted(int(o) for o in orders))
        self.max_order = self.orders[-1]
        self.gate = float(gate)
        self.order_weight = float(order_weight)
        self.max_cells = int(max_cells_per_order)
        self.write_decay = float(write_decay)  # EMA rate: 1 = last-write-wins
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
                # EMA successor distribution: recency-weighted — an
                # overwritten fact's latest value dominates exponentially
                a = self.write_decay
                if a < 1.0:
                    for t in list(row.keys()):
                        row[t] *= 1.0 - a
                        if row[t] < 1e-3:
                            del row[t]
                    row[succ_tok] = row.get(succ_tok, 0.0) + a
                else:
                    row.clear()
                    row[succ_tok] = 1.0

    def _decay(self, order: int) -> None:
        """Halve every mass in ``tables[order]``; drop rows that die."""
        tab = self.tables[order]
        for key in list(tab.keys()):
            row = tab[key]
            for tok in list(row.keys()):
                row[tok] *= 0.5
                if row[tok] < 0.02:
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
        lift = np.zeros(self.vocab, dtype=np.float32)
        matched = 0
        for i, o in enumerate(self.orders):
            probs, _cnt = self._row_probs(o)
            if probs is None:
                continue
            matched += 1
            w = self.order_weight ** (len(self.orders) - 1 - i)
            for tok, p in probs.items():
                # log-likelihood-ratio vs the smoothing prior: memorized
                # successors get a POSITIVE boost proportional to how much
                # the row beats the unigram/uniform reference; tokens the
                # memory has no opinion on get exactly 0
                p_s = p + self.unigram_mix * uni[tok] + self.uniform_floor
                p_ref = self.unigram_mix * uni[tok] + self.uniform_floor
                lift[tok] += w * np.log(p_s / p_ref)
        if matched == 0:
            return torch.zeros(self.vocab, dtype=dtype, device=device)
        g = self.gate if gate is None else float(gate)
        return torch.from_numpy(g * lift).to(device=device, dtype=dtype)

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

    # ── disk persistence ────────────────────────────────────────────────
    # Pickle on purpose: a live-memory artifact is self-produced local
    # state, same trust class as checkpoint .pt bundles (see AGENTS.md
    # security notes — load only self-produced artifacts).

    _PERSIST_VERSION = 1

    def save(self, path) -> None:
        """Persist config + full memory state to ``path``."""
        import pickle
        payload = {
            "version": self._PERSIST_VERSION,
            "vocab": self.vocab,
            "orders": self.orders,
            "gate": self.gate,
            "order_weight": self.order_weight,
            "max_cells": self.max_cells,
            "write_decay": self.write_decay,
            "unigram_mix": self.unigram_mix,
            "uniform_floor": self.uniform_floor,
            "state": self.snapshot_state(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)

    @classmethod
    def load(cls, path) -> "SuffixMemory":
        """Rebuild a SuffixMemory saved via ``save`` (exact state parity)."""
        import pickle
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("version") != cls._PERSIST_VERSION:
            raise ValueError(
                f"unsupported SuffixMemory version {payload.get('version')}")
        mem = cls(payload["vocab"], orders=payload["orders"],
                  gate=payload["gate"], order_weight=payload["order_weight"],
                  max_cells_per_order=payload["max_cells"],
                  unigram_mix=payload["unigram_mix"],
                  uniform_floor=payload["uniform_floor"],
                  write_decay=payload["write_decay"])
        mem.restore_state(payload["state"])
        return mem


class SuffixDraftModel(torch.nn.Module):
    """Draft-model-shaped shim over ``SuffixMemory`` — a zero-VRAM
    prompt-lookup drafter for ``external_draft_speculative``.

    The spec-decode draft contract is model-shaped: ``draft(ids,
    past_key_values=None, use_cache=...) -> (logits, kv)``, one forward
    per proposed token. This shim answers each call from the memory's
    suffix tables instead of running weights:

      * ``mem._tail`` is the committed context — the decoding strategy
        feeds every committed token to the sidecar (``feed_one``), so
        the shim never tracks committed tokens itself. T>1 calls
        (prefill / post-verify resync) only end the in-round proposal
        chain.
      * T==1 calls roll the virtual tail: a token equal to the shim's
        own last argmax is a mid-round proposal (folded into the
        prediction context); anything else is an out-of-band committed
        token (the strategy's no-draft resync), which ends the chain —
        it already lives in ``mem._tail``.
      * ``past_key_values`` is ignored — the memory state IS the draft
        state; ``(logits, None)`` unpacks as stateless.
      * Returned logits carry the memory's real successor probability
        at the predicted token (uniform -20 elsewhere), so
        ``confidence_tau`` scheduling sees an honest survival prior
        instead of a fake 1.0 spike.
    """

    def __init__(self, mem: "SuffixMemory", device: str = "cpu"):
        super().__init__()
        self.mem = mem
        self.device = device
        self._proposal: list[int] = []      # in-round unverified drafts
        self._last_pred: int | None = None  # argmax of last emitted logits

    def _predict_logits(self, B: int) -> torch.Tensor:
        tail = list(self.mem._tail) + self._proposal
        logits = torch.full((B, 1, self.mem.vocab), -20.0,
                            device=self.device, dtype=torch.float32)
        self._last_pred = None
        for b in range(B):
            best_tok, best_p = None, 0.0
            for o in reversed(self.mem.orders):
                if len(tail) < o:
                    continue
                row = self.mem.tables[o].get(
                    self.mem._suffix_key(tail, o))
                if row:
                    tot = sum(row.values())
                    k, c = max(row.items(), key=lambda kv: kv[1])
                    best_tok, best_p = k, (c / tot) if tot else 0.0
                    break
            if best_tok is not None:
                p = min(max(best_p, 1e-3), 1.0)
                logits[b, 0, best_tok] = float(
                    torch.logit(torch.tensor(p)))
                if b == 0:
                    self._last_pred = best_tok
        return logits

    def forward(self, ids, past_key_values=None, use_cache: bool = False,
                **_kw):
        ids_t = (ids if torch.is_tensor(ids)
                 else torch.tensor(ids, dtype=torch.long))
        B = ids_t.shape[0] if ids_t.dim() else 1
        if ids_t.dim() != 2:
            return torch.zeros(B, 1, self.mem.vocab, device=self.device), None
        if ids_t.shape[1] > 1:
            # T>1: prompt prefill or post-verify resync — committed
            # context lives in mem._tail (the caller feeds it); only
            # the in-round proposal chain ends here.
            self._proposal = []
            self._last_pred = None
        else:
            tok = int(ids_t[0, -1].item())
            if self._last_pred is not None and tok == self._last_pred:
                # In-round proposal roll — virtual tail only.
                self._proposal.append(tok)
                self._proposal = self._proposal[-self.mem.max_order:]
            else:
                # Committed token the shim didn't propose (no-draft
                # resync) — already fed into mem._tail; end the chain.
                self._proposal = []
                self._last_pred = None
        return self._predict_logits(B), None
