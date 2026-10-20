"""StrataLM - the M1 model skeleton on the packed-cell substrate.

CPU-eager reference per docs/STRATA_M1_DESIGN.md: rolling suffix-hash
context keys (10 orders), a channel registry (order/uni/backoff/
recency/fatigue), a hedge-weighted geometric mixture, surprise-gated
writes, and teach/learn/observe.

Weight/param/knob separation:
  * stored info lives in the BlockStore (kinds "o<k>" per order) -
    journaled/revertable via per-block undo arenas;
  * channel mixer weights are PARAMS ("w_mix" -> .iq on snapshot);
  * behavioral scalars (smoothing, decays, gates) are KNOBS - ingest
    never writes them.

Journal: the substrate reverts cell bytes by tag/seq; the model-level
journal only needs non-cell state - token markers carrying the store
write-clock watermark, unigram bumps, recency writes, hedge snapshots,
episode boundaries.  revert_since(pos) is a point-in-time rewind: it
unwinds to a stream position AND drops every store write with seq >=
the earliest reverted token's watermark (teach/write_counts cells
interleaved inside the span go too - they happened after the mark).
revert_tag(tag) additionally removes the tag's position-free writes
that precede the mark (exact per-cell undo in the arenas).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from forge.model.flux.helpers import _s64

from .blocks import BlockStore
from .cells import foll_tok, foll_cntq, foll_valid, meta_wtag
from .config import StrataConfig
from .io import save_container, load_container
from .knobs import KnobSet, DEFAULT_KNOBS
from .params import ParamSet

_FNV = 1099511628211
_MOD64 = 1 << 64

# M1 conditioning surface: scalars the model reads at run time.  Kept in
# KnobSet (never mutated by learn/teach) - extending DEFAULT_KNOBS here
# rather than editing knobs.py keeps the substrate file untouched.
MODEL_KNOBS = {
    "alpha": 0.4,             # cell smoothing toward unigram
    "uni_alpha": 0.05,        # unigram add-alpha
    "backoff_min_tot": 2.0,   # min cell mass for a backoff vote
    "backoff_kappa": 8.0,     # confidence = tot / (tot + kappa)
    "recency_decay": 0.90,
    "fatigue_decay": 0.80,
    "fatigue_strength": 2.0,
    "hedge_eta_max": 4.0,     # AdaHedge cap (eta = ln(C)/mix_gap)
    "hedge_min": 1e-3,
    "hedge_max": 50.0,
    "evidence_lift": 1.0,     # 1 = w += tot/(tot+kappa) on evidence
}


class StrataLM(nn.Module):
    """Fixed-budget associative-memory LM over labeled cell blocks."""

    # Engine capability flags (same contract as FluxLM).
    unbounded_context = True      # O(1) state - no prompt truncation
    supports_kv_cache = False     # memory is the state

    def __init__(self, config: StrataConfig | None = None,
                 knobs: KnobSet | None = None,
                 params: ParamSet | None = None):
        super().__init__()
        self.config = config or StrataConfig()
        c = self.config
        self.vocab_size = c.vocab_size
        self.eos_token_id = c.eos_token_id
        self._dev = torch.device(
            c.device if (c.device == "cpu" or torch.cuda.is_available())
            else "cpu")
        self.store = BlockStore(
            slots=c.slots_per_block, M=c.followers,
            dev=c.device if c.device == "cpu" or
            torch.cuda.is_available() else "cpu",
            store_raw=c.store_raw, fill_cap=c.fill_cap,
            max_blocks=c.max_blocks_per_kind,
            cold_blocks=c.cold_blocks_per_kind, undo_cap=c.undo_cap,
            vocab=c.vocab_size)
        self.knobs = knobs or KnobSet({**DEFAULT_KNOBS, **MODEL_KNOBS})
        self.params = params or ParamSet()

        # channel registry: order channels + dense/fallback channels
        self.channels = ([f"o{k}" for k in c.orders]
                         + ["uni", "backoff", "recency", "fatigue"])
        self._chix = {ch: i for i, ch in enumerate(self.channels)}
        self._w = np.full(len(self.channels), 1.0 / len(self.channels),
                          dtype=np.float64)
        self._hedge_mix = 0.0        # AdaHedge cumulative mix gap

        V = c.vocab_size
        # dense aux slabs: unigram counts + maintained log numerator
        self._uni = np.zeros(V, dtype=np.float64)
        self._total = 0.0
        ua = float(self.knobs.get("uni_alpha", 0.05))
        self._log_uni_num = np.full(V, math.log(ua), dtype=np.float64)

        # token ring + rolling prefix-hash ring (suffix-hash substrate)
        self._ring = np.zeros(c.ring_capacity, dtype=np.int32)
        self._H = np.zeros(c.ring_capacity, dtype=np.uint64)
        self._count = 0
        self._ep_start = 0
        self._pows = np.empty(max(c.orders) + 1, dtype=np.uint64)
        self._pows[0] = 1
        for i in range(1, len(self._pows)):
            self._pows[i] = (int(self._pows[i - 1]) * _FNV) % _MOD64

        # recency/fatigue state: last position per token + window deque
        self._last_seen: dict[int, int] = {}
        self._recent: deque = deque(
            maxlen=max(c.recency_span, c.fatigue_span))

        # model-level journal for non-cell state (cells ride the block
        # undo arenas).  Entries:
        #   ("t", tok, tag, pos, store_seq_before)  stream marker
        #   ("u", tok)                            unigram +1
        #   ("r", tok, prev_pos_or_-1)            last_seen write
        #   ("h", w_fp64[C], mix_gap)             pre-update hedge state
        #   ("b", ep_start)                       episode boundary
        #   ("ws", tag)/("we", tag)               position-free write
        #                                         block markers
        self._journal: list[tuple] = []
        self.journal_dropped = 0

        self.learning = True
        self.tag = "boot"

    # ------------------------------------------------------------------
    # knobs / context keys
    # ------------------------------------------------------------------

    def _k(self, name: str) -> float:
        return float(self.knobs.get(name, MODEL_KNOBS.get(name, 0.0)))

    def _suffix_key(self, k: int, end: int) -> int | None:
        """Hash of the last ``k`` tokens ending at abs pos ``end``;
        None when the episode is shorter than ``k``."""
        if end - k + 1 < self._ep_start or end < k - 1:
            return None
        cap = self.config.ring_capacity
        return (int(self._H[end % cap])
                - int(self._H[(end - k) % cap]) * int(self._pows[k])) \
            % _MOD64

    def _keys_at(self, end: int) -> dict[int, int]:
        keys = {}
        for k in self.config.orders:
            key = self._suffix_key(k, end)
            if key is not None:
                keys[k] = key
        return keys

    def _raw_hash(self, k: int, end: int) -> int:
        """Suffix hash ignoring the episode boundary (write_counts)."""
        cap = self.config.ring_capacity
        return (int(self._H[end % cap])
                - int(self._H[(end - k) % cap]) * int(self._pows[k])) \
            % _MOD64

    @staticmethod
    def _prefix_hashes(ids) -> np.ndarray:
        h = np.empty(len(ids), dtype=np.uint64)
        prev = 0
        for i, t in enumerate(ids):
            prev = (prev * _FNV + int(t) + 1) % _MOD64
            h[i] = prev
        return h

    def _seq_key(self, H: np.ndarray, k: int, e: int) -> int | None:
        """Suffix key over a standalone token list (teach)."""
        if e - k + 1 < 0 or e < k - 1:
            return None
        return (int(H[e]) - int(H[e - k]) * int(self._pows[k])) % _MOD64

    # ------------------------------------------------------------------
    # cell reads
    # ------------------------------------------------------------------

    def _read_cell(self, kind: str, key: int):
        """-> (followers {tok: cnt}, tot, found).  Merged across blocks
        newest-first by the store."""
        kt = torch.tensor([_s64(key)], dtype=torch.int64)
        toks, cnts, tot, found = self.store.read(kind, kt)
        if not bool(found[0]):
            return {}, 0.0, False
        cell = {}
        for t, cv in zip(toks[0].tolist(), cnts[0].tolist()):
            if t >= 0 and cv > 0.0:
                cell[t] = cell.get(t, 0.0) + cv
        return cell, float(tot[0]), True

    # ------------------------------------------------------------------
    # channel probabilities (hedge rewards + surprise need p(x) per
    # channel; the emitted distribution is geometric, the reward is
    # arithmetic - same split as FluxLM)
    # ------------------------------------------------------------------

    def _uni_p(self, t: int, denom: float) -> float:
        return (self._uni[t] + self._k("uni_alpha")) / denom

    def _recent_lift_z(self, span: int, decay: float,
                       sign: float) -> float:
        """Σ p_uni(t)·(exp(sign·decay^age) − 1) over the live window -
        latest occurrence per token wins (flux _recent_lift_z)."""
        end = self._count - 1
        seen: dict[int, int] = {}
        for pos, t in reversed(self._recent):
            age = end - pos
            if age >= span:
                break
            if age >= 0 and t not in seen:
                seen[t] = age
        if not seen:
            return 0.0
        denom = self._total + self._k("uni_alpha") * self.vocab_size
        return float(sum(
            self._uni_p(t, denom)
            * (math.exp(sign * decay ** a) - 1.0)
            for t, a in seen.items()))

    def _backoff_pick(self, cells: dict[int, tuple]) -> int:
        """Longest order with enough evidence; -1 when thin."""
        best = -1
        bmin = self._k("backoff_min_tot")
        for k, (_cell, tot, found) in cells.items():
            if found and tot >= bmin and k > best:
                best = k
        return best

    def _channel_prob(self, ch: str, x: int, keys: dict[int, int],
                      cells: dict[int, tuple], denom: float
                      ) -> tuple[float, bool]:
        """(prob of x under channel, awake) - asleep channels take the
        unigram prior and abstain from the reward renormalization."""
        if ch == "uni":
            return self._uni_p(x, denom), True
        kappa = self._k("backoff_kappa")
        if ch == "backoff":
            bk = self._backoff_pick(cells)
            if bk < 0:
                return self._uni_p(x, denom), False
            cell, tot, _ = cells[bk]
            conf = tot / (tot + kappa)
            return (conf * (cell.get(x, 0.0)
                            + self._k("alpha") * self._uni_p(x, denom))
                    / (tot + self._k("alpha"))), True
        if ch == "recency":
            c = self.config
            age = self._count - 1 - self._last_seen.get(x, -10 ** 9)
            rec = self._k("recency_decay") ** age \
                if 0 <= age < c.recency_span else 0.0
            z = 1.0 + self._recent_lift_z(c.recency_span,
                                          self._k("recency_decay"), 1.0)
            return self._uni_p(x, denom) * math.exp(rec) / z, True
        if ch == "fatigue":
            c = self.config
            age = self._count - 1 - self._last_seen.get(x, -10 ** 9)
            fat = self._k("fatigue_decay") ** age \
                if 0 <= age < c.fatigue_span else 0.0
            z = 1.0 + self._recent_lift_z(
                c.fatigue_span, self._k("fatigue_decay"),
                -self._k("fatigue_strength"))
            return (self._uni_p(x, denom)
                    * math.exp(-self._k("fatigue_strength") * fat)
                    / z), True
        # order channels "o<k>"
        k = int(ch[1:])
        cell, tot, found = cells.get(k, ({}, 0.0, False))
        if not found or tot <= 0.0:
            return self._uni_p(x, denom), False
        return ((cell.get(x, 0.0)
                 + self._k("alpha") * self._uni_p(x, denom))
                / (tot + self._k("alpha"))), True

    # ------------------------------------------------------------------
    # predict - geometric mixture: log p_uni + Σ w_c·(log p_c − log p_uni)
    # over the sparse candidate set, plus recency/fatigue lifts, plus an
    # arithmetic unigram escape floor on the emitted distribution.
    # ------------------------------------------------------------------

    def _predict(self, cap: dict | None = None) -> torch.Tensor:
        """Next-token logits [V] (fp32). ``cap`` captures per-channel
        additive lifts for explain()."""
        c = self.config
        V = self.vocab_size
        end = self._count - 1
        keys = self._keys_at(end) if end >= self._ep_start else {}
        denom = self._total + self._k("uni_alpha") * V
        log_denom = math.log(denom)
        logits = torch.from_numpy(self._log_uni_num - log_denom).float()
        if cap is not None:
            cap["uni"] = logits.clone()
        w = {ch: self._w[i] for i, ch in enumerate(self.channels)}

        def log_uni(t: int) -> float:
            return math.log(self._uni[t] + self._k("uni_alpha")) \
                - log_denom

        lift: dict[int, float] = {}

        def acc(t: int, dv: float, ch: str) -> None:
            lift[t] = lift.get(t, 0.0) + dv
            if cap is not None:
                cd = cap.setdefault(ch, {})
                cd[t] = cd.get(t, 0.0) + dv

        alpha = self._k("alpha")
        kappa = self._k("backoff_kappa")
        el = self._k("evidence_lift")
        cells: dict[int, tuple] = {}
        for k, key in keys.items():
            cell, tot, found = self._read_cell(f"o{k}", key)
            cells[k] = (cell, tot, found)
            if not found or tot <= 0.0:
                continue
            wk = w[f"o{k}"] + (el * tot / (tot + kappa))
            for t, cnt in cell.items():
                p = (cnt + alpha * self._uni_p(t, denom)) / (tot + alpha)
                acc(t, wk * (math.log(max(p, 1e-12)) - log_uni(t)),
                    f"o{k}")

        # backoff: longest evidenced order, confidence-discounted
        bk = self._backoff_pick(cells)
        if bk >= 0:
            cell, tot, _ = cells[bk]
            conf = tot / (tot + kappa)
            wb = w["backoff"] + (el * conf)
            for t, cnt in cell.items():
                p = conf * (cnt + alpha * self._uni_p(t, denom)) \
                    / (tot + alpha)
                acc(t, wb * (math.log(max(p, 1e-12)) - log_uni(t)),
                    "backoff")

        # recency: additive boost inside the live window
        wr = w["recency"]
        if wr > 1e-4:
            rd = self._k("recency_decay")
            for pos, t in reversed(self._recent):
                age = end - pos
                if age >= c.recency_span:
                    break
                if age >= 0:
                    acc(t, wr * rd ** age, "recency")

        # fatigue: subtractive penalty for just-emitted tokens
        wf = w["fatigue"] * self._k("fatigue_strength")
        if wf > 1e-4:
            fd = self._k("fatigue_decay")
            for pos, t in reversed(self._recent):
                age = end - pos
                if age >= c.fatigue_span:
                    break
                if age >= 0:
                    acc(t, -wf * fd ** age, "fatigue")

        if lift:
            idx = torch.tensor(list(lift.keys()), dtype=torch.long)
            val = torch.tensor(list(lift.values()), dtype=torch.float32)
            logits.index_add_(0, idx, val)
        if cap is not None:
            cap["_pre_escape"] = logits.clone()
        eps = float(self.knobs.get("uni_escape", 0.12))
        if eps > 0:
            uni_pv = np.exp(self._log_uni_num - log_denom)
            logits = torch.log(
                (1.0 - eps) * torch.softmax(logits, -1)
                + eps * torch.from_numpy(uni_pv).float())
        return logits.to(self._dev)

    def predict(self) -> torch.Tensor:
        return self._predict()

    def explain(self, top_k: int = 10) -> dict:
        """Additive per-channel decomposition + cell provenance: every
        firing cell reports (kind, block, slot, key, tag, followers)."""
        cap: dict = {}
        logits = self._predict(cap=cap)
        end = self._count - 1
        keys = self._keys_at(end) if end >= self._ep_start else {}
        names = {v: n for n, v in self.store.tag_ids.items()}
        cells = []
        for k, key in keys.items():
            pk_key = torch.tensor([_s64(key)], dtype=torch.int64)
            for bi, b in enumerate(self.store.kinds.get(f"o{k}", [])):
                slot, found = b.probe(pk_key.to(b.dev))
                if not bool(found[0]):
                    continue
                s = int(slot[0])
                row = b.foll[s]
                valid = foll_valid(row)
                fl = sorted(
                    ((int(t), float(c))
                     for t, c in zip(foll_tok(row[valid]).tolist(),
                                     b.lut[foll_cntq(row[valid])
                                           .long()].tolist())),
                    key=lambda kv: -kv[1])[:8]
                cells.append({
                    "channel": f"o{k}", "block": b.id, "slot": s,
                    "key": int(key), "winner": bi == 0,
                    "tag": names.get(int(meta_wtag(b.metaw[s])), "?"),
                    "total": float(b._row_tot(
                        torch.tensor([s], device=b.dev))[0]),
                    "followers": fl})
        top = torch.topk(logits, min(top_k, logits.numel()))
        lifts = {}
        for ch, d in cap.items():
            if ch.startswith("_") or ch == "uni":
                continue
            lifts[ch] = {int(t): round(v, 6) for t, v in d.items()
                         if int(t) in set(top.indices.tolist())}
        return {"end": end, "logits_top": [
                    (int(t), float(v)) for t, v in
                    zip(top.indices.tolist(), top.values.tolist())],
                "weights": self.channel_weights(), "cells": cells,
                "lifts": lifts}

    def channel_weights(self) -> dict:
        return {ch: float(self._w[i])
                for i, ch in enumerate(self.channels)}

    # ------------------------------------------------------------------
    # stream state
    # ------------------------------------------------------------------

    def _push(self, tok: int) -> None:
        """Append to the stream: ring + prefix hash + counters only.
        No memory writes - this is the observe path."""
        c = self.config
        pos = self._count
        slot = pos % c.ring_capacity
        self._ring[slot] = tok
        prev = int(self._H[(pos - 1) % c.ring_capacity]) if pos > 0 else 0
        self._H[slot] = (prev * _FNV + tok + 1) % _MOD64
        self._count += 1
        if c.auto_episode and tok == c.eos_token_id:
            self.soft_reset()

    def soft_reset(self) -> None:
        """New episode: reset context/recency state, keep all memory."""
        self._journal.append(("b", self._ep_start))
        self._ep_start = self._count
        self._last_seen.clear()
        self._recent.clear()

    def observe(self, ids) -> int:
        """Advance context over ``ids`` with zero writes."""
        n = 0
        for x in ids:
            self._push(int(x))
            n += 1
        return n

    def ingest(self, ids, tag: str | None = None,
               learn: bool = True) -> int:
        """Stream-learn on token ids (instant, hot)."""
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        prev_tag, prev_learn = self.tag, self.learning
        if tag is not None:
            self.tag = tag
        self.learning = learn
        n = 0
        for x in ids:
            if self.learning:
                self._learn_token(int(x))
            else:
                self._push(int(x))
            n += 1
        self.tag, self.learning = prev_tag, prev_learn
        return n

    # ------------------------------------------------------------------
    # learn
    # ------------------------------------------------------------------

    def _learn_token(self, x: int) -> None:
        """All live updates for observing token ``x`` (journaled)."""
        c = self.config
        kn = self.knobs
        end = self._count - 1            # context ends just before x
        keys = self._keys_at(end) if end >= self._ep_start else {}
        # one probe per order feeds both the hedge reward and the write
        cells = {k: self._read_cell(f"o{k}", key)
                 for k, key in keys.items()}
        denom = self._total + self._k("uni_alpha") * self.vocab_size

        # channel rewards (arithmetic p_mix over awake channels -
        # sleeping experts renorm: absent channels take p_mix neutrally)
        probs = np.empty(len(self.channels))
        awake = np.ones(len(self.channels), dtype=bool)
        for i, ch in enumerate(self.channels):
            probs[i], awake[i] = self._channel_prob(
                ch, x, keys, cells, denom)
        wv = self._w
        wsum = float((wv * awake).sum())
        p_mix = float((wv * probs * awake).sum()) / max(wsum, 1e-12)
        rewards = np.where(awake, probs, p_mix)
        # AdaHedge: lr adapts to accumulated mixability gap
        hmx = self._hedge_mix + float(rewards.max() - p_mix)
        eta = min(self._k("hedge_eta_max"),
                  math.log(len(self.channels)) / max(hmx, 1e-9))
        factors = np.exp(np.clip(eta * (rewards - p_mix), -30.0, 30.0))
        self._j(("h", wv.copy(), self._hedge_mix))
        self._hedge_mix = hmx
        wv *= factors
        np.clip(wv, self._k("hedge_min"), self._k("hedge_max"), out=wv)
        wv /= wv.sum()

        # surprise: predictable tokens barely write
        s = (max(1.0 - p_mix, self._k("surprise_floor"))
             if kn.get("surprise_gate", 1.0) else 1.0)

        # unigram + total
        self._uni[x] += 1.0
        self._total += 1.0
        self._log_uni_num[x] = math.log(
            self._uni[x] + self._k("uni_alpha"))
        self._j(("u", x))

        # order cells: (suffix key -> x) at surprise magnitude, tagged.
        seq0 = self.store.seq
        kt = torch.tensor([x], dtype=torch.int64)
        dt = torch.tensor([s], dtype=torch.float32)
        for k, key in keys.items():
            self.store.write(
                f"o{k}", torch.tensor([_s64(key)], dtype=torch.int64),
                kt, dt, tag=self.tag)

        # recency bookkeeping
        prev = self._last_seen.get(x, -1)
        self._last_seen[x] = end
        self._recent.append((end, x))
        self._j(("r", x, prev))

        # token marker closes the write group (carries the store
        # watermark for point-in-time cell revert)
        self._j(("t", x, self.tag, self._count, seq0))
        self._push(x)

    def _j(self, e: tuple) -> None:
        self._journal.append(e)
        if len(self._journal) > self.config.journal_cap:
            del self._journal[: self.config.journal_cap // 10]
            self.journal_dropped += self.config.journal_cap // 10

    # ------------------------------------------------------------------
    # teach / manual writes (position-free; journaled under tag)
    # ------------------------------------------------------------------

    def teach(self, context_ids, target_ids, weight: float = 25.0,
              tag: str = "teach") -> dict:
        """Bind context -> target: ``weight`` follower mass under every
        valid order key of each target position's context.  Keys are
        derived from the token sequence itself - no stream presence
        needed, and the model's own context is untouched.  Revertible
        via ``revert_tag(tag)``."""
        ctx = [int(t) for t in context_ids]
        tgt = [int(t) for t in target_ids]
        rep = {"cells": 0}
        if not tgt:
            return rep
        seq = ctx + tgt
        H = self._prefix_hashes(seq)
        self._j(("ws", tag))
        for i, t in enumerate(tgt):
            e = len(ctx) + i - 1           # context end inside seq
            for k in self.config.orders:
                key = self._seq_key(H, k, e)
                if key is None:
                    continue
                self.store.write(
                    f"o{k}",
                    torch.tensor([_s64(key)], dtype=torch.int64),
                    torch.tensor([t], dtype=torch.int64),
                    torch.tensor([weight], dtype=torch.float32),
                    tag=tag)
                rep["cells"] += 1
        self._j(("we", tag))
        return rep

    def write_counts(self, ctx_end: int, deltas: dict[int, float],
                     tag: str = "distill") -> int:
        """Write fractional follower counts under the context keys valid
        at absolute position ``ctx_end`` (manual injection / distill).
        Uses RAW suffix hashes (not episode-gated) - the stored key
        still matches whenever the same tokens recur."""
        floor = max(0, self._count - self.config.ring_capacity)
        keys = {}
        for k in self.config.orders:
            if ctx_end - 1 - k + 1 >= floor:
                keys[k] = self._raw_hash(k, ctx_end - 1)
        toks = torch.tensor(list(deltas.keys()), dtype=torch.int64)
        ds = torch.tensor(list(deltas.values()), dtype=torch.float32)
        self._j(("ws", tag))
        n = 0
        for k, key in keys.items():
            self.store.write(
                f"o{k}", torch.tensor([_s64(key)], dtype=torch.int64),
                toks, ds, tag=tag)
            n += len(deltas)
        self._j(("we", tag))
        return n

    # ------------------------------------------------------------------
    # revert / journal
    # ------------------------------------------------------------------

    def _undo_entry(self, e: tuple) -> bool:
        op = e[0]
        if op == "t":
            # rewind stream head; ring bytes stay (harmless once head
            # moved).  Cell undo is batched by the caller via the seq
            # watermark.  "t" stores the token's own position.
            self._count = e[3]
            return True
        if op == "b":
            self._ep_start = e[1]
            return True
        if op == "u":
            if self._uni[e[1]] >= 1.0:
                self._uni[e[1]] -= 1.0
                self._total -= 1.0
                self._log_uni_num[e[1]] = math.log(
                    self._uni[e[1]] + self._k("uni_alpha"))
                return True
            return False
        if op == "r":
            if e[2] < 0:
                self._last_seen.pop(e[1], None)
            else:
                self._last_seen[e[1]] = e[2]
            return True
        if op == "h":
            self._w[:] = e[1]
            self._hedge_mix = e[2]
            return True
        return op in ("ws", "we")          # structural markers

    def revert_since(self, seq_pos: int | None = None,
                     n_entries: int | None = None) -> dict:
        """Undo journaled state back to a stream position (or N entries,
        rounded up to a token boundary).  Cell bytes are restored via
        the substrate's write-clock arenas: every store write with seq
        >= the earliest reverted token's watermark is undone - including
        position-free teach/write_counts cells inside the span."""
        j = self._journal
        reverted = inexact = popped = 0
        min_seq = None
        while j:
            at_boundary = j[-1][0] in ("t", "b", "ws", "we")
            if seq_pos is not None and self._count <= seq_pos \
                    and at_boundary:
                break
            if n_entries is not None and popped >= n_entries \
                    and at_boundary:
                break
            if seq_pos is None and n_entries is None:
                break
            e = j.pop()
            popped += 1
            if e[0] == "t":
                min_seq = e[4] if min_seq is None else min(min_seq,
                                                           e[4])
            if self._undo_entry(e):
                reverted += 1
            else:
                inexact += 1
        if min_seq is not None:
            cells = self.store.revert_since(min_seq - 1)
        else:
            cells = 0
        self._rebuild_recent()
        self._rebuild_seen()
        return {"reverted": reverted, "inexact": inexact,
                "cells_undone": cells, "stream_len": self._count}

    def revert_tag(self, tag: str) -> dict:
        """Un-learn everything journaled under ``tag``:
        suffix-rewind to the earliest tagged token (undoes later tokens
        too, causal consistency), then a tag-scoped cell sweep for the
        tag's position-free writes that precede the mark."""
        target = None
        for e in self._journal:
            if e[0] == "t" and e[2] == tag:
                target = e[3] if target is None else min(target, e[3])
        rep = {"reverted": 0, "inexact": 0, "cells_undone": 0}
        if target is not None:
            rep = self.revert_since(seq_pos=target)
            rep["found"] = True
        # residual tag writes (teach/write_counts cells anywhere still
        # in the arenas; wholly-owned sealed blocks drop, partial taint)
        srep = self.store.revert_tag(tag)
        rep["cells_undone"] = rep.get("cells_undone", 0) \
            + srep["undone"]
        rep["dropped_blocks"] = srep["dropped"]
        rep["tainted"] = srep["tainted"]
        if srep["undone"] or srep["dropped"] or srep["tainted"]:
            rep["found"] = True
        # drop the tag's write-block markers from the journal
        self._journal = [e for e in self._journal
                         if not (e[0] in ("ws", "we") and e[1] == tag)]
        rep.setdefault("found", False)
        rep["tag"] = tag
        rep["stream_len"] = self._count
        return rep

    def revert_writes(self, tag: str) -> dict:
        """Undo a tag's position-free writes only (teach/write_counts)
        - no stream rewind."""
        srep = self.store.revert_tag(tag)
        self._journal = [e for e in self._journal
                         if not (e[0] in ("ws", "we") and e[1] == tag)]
        return {"cells_undone": srep["undone"],
                "dropped_blocks": srep["dropped"],
                "tainted": srep["tainted"],
                "found": bool(srep["undone"] or srep["dropped"]
                              or srep["tainted"]),
                "tag": tag}

    def _rebuild_recent(self) -> None:
        """Rebuild the in-span recency tail from the ring - exact since
        ring bytes before the head are untouched by revert."""
        span = self.config.recency_span
        cap = self.config.ring_capacity
        self._recent = deque(
            ((p, int(self._ring[p % cap]))
             for p in range(max(0, self._count - span), self._count)),
            maxlen=self._recent.maxlen)

    def _rebuild_seen(self) -> None:
        """Rebuild last_seen inside the live window.  "r" undos already
        restored per-token state exactly; this additionally covers
        entries cleared by a reverted episode boundary (which journals
        no contents - the ring is the only surviving evidence) and drops
        stale entries past the head."""
        span = max(self.config.recency_span, self.config.fatigue_span)
        cap = self.config.ring_capacity
        for t in [t for t, p in self._last_seen.items() if p >=
                  self._count]:
            del self._last_seen[t]
        for p in range(max(self._ep_start, self._count - span),
                       self._count):
            self._last_seen[int(self._ring[p % cap])] = p

    def audit(self, n: int = 50, tokenizer=None) -> list[dict]:
        """Decode the last ``n`` learned tokens: tok, tag, position."""
        out = []
        tail: deque = deque(maxlen=8)
        for e in self._journal[-(n * 8):]:
            if e[0] == "t":
                rec = {"pos": e[3], "tok": e[1], "tag": e[2]}
                if tokenizer is not None:
                    rec["token"] = tokenizer.decode([e[1]])
                    rec["context"] = tokenizer.decode(list(tail))
                tail.append(e[1])
                out.append(rec)
        return out[-n:]

    def writes_since(self, n_entries: int) -> dict:
        from collections import Counter
        ops: Counter = Counter()
        tags: Counter = Counter()
        for e in self._journal[-n_entries:]:
            ops[e[0]] += 1
            if e[0] == "t":
                tags[e[2]] += 1
        return {"ops": dict(ops), "tags": dict(tags)}

    # ------------------------------------------------------------------
    # forward / generate (engine contract)
    # ------------------------------------------------------------------

    def forward(self, idx: torch.Tensor, targets=None,
                past_key_values=None, use_cache: bool = False,
                return_hidden: bool = False, **kw):
        """Returns (logits, loss_or_None, None).  Each input token is
        observed (learned when ``learning`` and ``learn_prefill``
        allow), then used as context for the next-position logits."""
        rows = idx.tolist() if isinstance(idx, torch.Tensor) \
            else [list(idx)]
        all_logits, losses = [], []
        tgt_rows = (targets.tolist()
                    if isinstance(targets, torch.Tensor) else None)
        for b, row in enumerate(rows):
            tgts = tgt_rows[b] if tgt_rows else None
            prev_learn = self.learning
            self.learning = bool(
                prev_learn and (self.config.learn_prefill
                                or len(row) <= 1))
            try:
                outs = []
                for i, x in enumerate(row):
                    x = int(x)
                    if self.learning:
                        self._learn_token(x)
                    else:
                        self._push(x)
                    outs.append(self._predict())
                    if tgts is not None and i + 1 < len(tgts):
                        lp = torch.log_softmax(outs[-1].float(), -1)
                        losses.append(-float(lp[int(tgts[i + 1])]))
                outs_t = (torch.stack(outs) if outs else
                          torch.zeros((0, self.vocab_size)))
            finally:
                self.learning = prev_learn
            all_logits.append(outs_t)
        logits = torch.stack(all_logits)
        if isinstance(idx, torch.Tensor) and idx.device != self._dev:
            logits = logits.to(idx.device)
        loss = (torch.tensor(float(np.mean(losses)))
                if losses else None)
        return logits, loss, None

    def generate(self, prompt_ids, max_new: int = 64,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 0, learn: bool = True, tag: str = "gen",
                 generator: torch.Generator | None = None) -> list[int]:
        """Autoregressive decode with support-restricted sampling.
        temp=0 is argmax; >0 samples after top-p/top-k and the
        ``support_thresh`` mask (a candidate must beat thresh * p_uni -
        some channel must have voted for it)."""
        self.ingest(list(prompt_ids), tag=tag, learn=learn)
        out = []
        thresh = float(self.knobs.get("support_thresh", 1.5))
        for _ in range(max_new):
            logits = self._predict().float()
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                p = torch.softmax(logits / temperature, -1)
                denom = (self._total
                         + self._k("uni_alpha") * self.vocab_size)
                uni_pv = torch.from_numpy(
                    (self._uni + self._k("uni_alpha"))
                    / max(denom, 1e-12)).float()
                p = torch.where(p >= thresh * uni_pv, p,
                                torch.zeros_like(p))
                if top_k > 0:
                    vk, vi = torch.topk(p, top_k)
                    m = torch.zeros_like(p)
                    m[vi] = vk
                    p = m
                if top_p < 1.0:
                    srt, si = torch.sort(p, descending=True)
                    cum = torch.cumsum(srt, 0)
                    keep = cum - srt < top_p
                    m = torch.zeros_like(p)
                    m[si[keep]] = srt[keep]
                    p = m
                tot = p.sum()
                if float(tot) <= 0:               # nothing on support
                    nxt = int(logits.argmax())    # -> argmax fallback
                else:
                    nxt = int(torch.multinomial(
                        p / tot, 1, generator=generator))
            out.append(nxt)
            if learn:
                self._learn_token(nxt)
            else:
                self._push(nxt)
        return out

    # ------------------------------------------------------------------
    # persistence + reporting
    # ------------------------------------------------------------------

    def snapshot(self, path) -> Path:
        """Write the three artifacts: ``path`` (.strata - cells + aux
        state), ``path`` with .iq suffix (params incl. mixer weights),
        and .cond (knobs).  Journal/undo arenas are session state and do
        not serialize - a loaded model reverts only post-load writes."""
        p = Path(path)
        self.params.set("w_mix",
                        torch.from_numpy(self._w.astype(np.float32)))
        tensors = dict(self.store.tensors())
        tensors["aux.uni"] = torch.from_numpy(self._uni.astype(np.int64))
        tensors["aux.ring"] = torch.from_numpy(self._ring)
        tensors["aux.H"] = torch.from_numpy(self._H.view(np.int64))
        st = sorted(self._last_seen.items())
        tensors["aux.seen_tok"] = torch.tensor(
            [t for t, _ in st], dtype=torch.int64)
        tensors["aux.seen_pos"] = torch.tensor(
            [p_ for _, p_ in st], dtype=torch.int64)
        meta = {"kind": "strata_model", "version": 1,
                "config": asdict(self.config),
                "store": self.store.manifest(),
                "model": {"count": self._count,
                          "ep_start": self._ep_start,
                          "total": self._total,
                          "hedge_mix": self._hedge_mix}}
        save_container(p, tensors, meta)
        self.params.save(p.with_suffix(".iq"))
        self.knobs.save(p.with_suffix(".cond"))
        return p

    @classmethod
    def load(cls, path, device: str | None = None) -> "StrataLM":
        p = Path(path)
        meta, tensors = load_container(p, copy=True)
        cfg = StrataConfig(**meta["config"])
        if device is not None:
            cfg.device = device
        knobs = (KnobSet.load(p.with_suffix(".cond"))
                 if p.with_suffix(".cond").exists() else None)
        params = (ParamSet.load(p.with_suffix(".iq"))
                  if p.with_suffix(".iq").exists() else None)
        m = cls(cfg, knobs=knobs, params=params)
        m.store = BlockStore.from_manifest(meta["store"], tensors,
                                           dev=cfg.device if
                                           cfg.device == "cpu" or
                                           torch.cuda.is_available()
                                           else "cpu")
        m._uni = tensors["aux.uni"].numpy().astype(np.float64)
        m._log_uni_num = np.log(m._uni + m._k("uni_alpha"))
        m._ring = tensors["aux.ring"].numpy().astype(np.int32)
        m._H = tensors["aux.H"].numpy().view(np.uint64)
        st = tensors["aux.seen_tok"].tolist()
        sp = tensors["aux.seen_pos"].tolist()
        m._last_seen = dict(zip(st, sp))
        md = meta["model"]
        m._count = md["count"]
        m._ep_start = md["ep_start"]
        m._total = md["total"]
        m._hedge_mix = md["hedge_mix"]
        m._rebuild_recent()
        if "w_mix" in m.params.names():
            w = m.params.get("w_mix").numpy().astype(np.float64)
            if w.shape == m._w.shape:
                m._w = w
        return m

    def memory_report(self) -> dict:
        aux = (self._uni.astype(np.int64).nbytes + self._ring.nbytes
               + self._H.nbytes
               + len(self._last_seen) * 24)
        cell = sum(b.S * (8 + 4 * b.M + (8 if b.raw is not None else 0))
                   for bl in self.store.kinds.values() for b in bl)
        return {"total_MB_est": (aux + cell) / 1e6,
                "aux_MB": aux / 1e6, "cell_MB": cell / 1e6,
                "journal_len": len(self._journal),
                "journal_dropped": self.journal_dropped,
                "blocks": self.store.stats()}
