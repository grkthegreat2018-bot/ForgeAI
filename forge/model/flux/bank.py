"""FluxBank - cold-tier weight bank for FluxLM order tables.

FluxLM "weights" are addressable memory cells.  The GPU/host tables are
the HOT tier; FluxBank is the COLD tier: a host-resident (optionally
file-persisted) store of whole order rows that fell off the hot end.
Reads merge cold evidence through the same order-channel prob formula
- the union of hot + cold counts is the true union of disjoint write
sets, so a context spilled then re-written needs no resurrection.
Spill dedups: re-spilling a banked context merges its row, never
duplicates.

Layout per bank (one per order k):
  keys   int64 [N]      - probed key (pk), unique
  raws   int64 [N]      - raw context key (audit/debug)
  toks   int32 [N, M]   - follower token columns (-1 = empty)
  cnt    f32   [N, M]   - follower counts
  tot    f32   [N]      - row total (sum of cnt)
  occ    f32   [N]      - utility score (tocc at spill time)
  pos    i64   [N]      - last block position that wrote the row

The index is a python dict {key: row} - O(1) probe/dedup.  Column
arrays live in numpy and grow geometrically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FluxBank:
    """Cold-tier row store for one order table."""

    def __init__(self, order: int, M: int = 16):
        self.order = order
        self.M = M
        self._ix: dict[int, int] = {}
        self._n = 0                     # live rows (arrays overcap)
        self.keys = np.zeros(0, dtype=np.int64)
        self.raws = np.zeros(0, dtype=np.int64)
        self.toks = np.zeros((0, M), dtype=np.int32)
        self.cnt = np.zeros((0, M), dtype=np.float32)
        self.tot = np.zeros(0, dtype=np.float32)
        self.occ = np.zeros(0, dtype=np.float32)
        self.pos = np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return self._n

    # -- probes --------------------------------------------------------

    def probe(self, key: int) -> int:
        """Row index for ``key`` or -1."""
        return self._ix.get(int(key), -1)

    def cell(self, key: int) -> tuple[dict, float] | None:
        """(followers {tok: cnt}, tot) for ``key`` or None - same
        contract as a hot ``_tables[k]`` cell + ``_totals[k]``."""
        i = self._ix.get(int(key), -1)
        if i < 0:
            return None
        t = self.toks[i]
        cn = self.cnt[i]
        cell = {int(t[j]): float(cn[j])
                for j in range(self.M) if t[j] >= 0}
        return cell, float(self.tot[i])

    def probe_keys(self, keys: np.ndarray) -> np.ndarray:
        """Vectorized probe: row index per key (-1 miss)."""
        out = np.full(keys.shape, -1, dtype=np.int64)
        for i, k in enumerate(keys):
            out[i] = self._ix.get(int(k), -1)
        return out

    # -- writes --------------------------------------------------------

    def _grow(self, extra: int) -> None:
        n = self._n + extra
        cap = self.keys.shape[0]
        if n <= cap:
            return
        cap = max(cap, 1 << 10)
        while cap < n:
            cap *= 2
        def _ext(a, fill=0):
            out = np.full((cap,) + a.shape[1:], fill, dtype=a.dtype)
            out[:self._n] = a[:self._n]
            return out
        self.keys = _ext(self.keys)
        self.raws = _ext(self.raws)
        self.toks = _ext(self.toks, -1)
        self.cnt = _ext(self.cnt)
        self.tot = _ext(self.tot)
        self.occ = _ext(self.occ)
        self.pos = _ext(self.pos)

    def add_rows(self, keys, raws, toks, cnt, tot, occ, pos) -> int:
        """Spill rows into the bank.  numpy array args, shapes
        [B], [B], [B,M], [B,M], [B], [B], [B].  Existing keys MERGE
        (follower counts sum, tot sums, occ takes the max) - a
        re-spilled context keeps one row.  Returns rows added."""
        keys = np.asarray(keys, dtype=np.int64).reshape(-1)
        B = keys.shape[0]
        if not B:
            return 0
        raws = np.asarray(raws, dtype=np.int64).reshape(-1)
        toks = np.asarray(toks, dtype=np.int32).reshape(B, self.M)
        cnt = np.asarray(cnt, dtype=np.float32).reshape(B, self.M)
        tot = np.asarray(tot, dtype=np.float32).reshape(-1)
        occ = np.asarray(occ, dtype=np.float32).reshape(-1)
        pos = np.asarray(pos, dtype=np.int64).reshape(-1)
        n0 = self._n
        self._grow(B)
        added = 0
        for b in range(B):
            k = int(keys[b])
            i = self._ix.get(k, -1)
            if i < 0:
                i = n0 + added
                self._ix[k] = i
                self.keys[i] = k
                self.raws[i] = raws[b]
                self.toks[i] = toks[b]
                self.cnt[i] = cnt[b]
                self.tot[i] = tot[b]
                self.occ[i] = occ[b]
                added += 1
            else:
                # merge follower columns by token id
                for j in range(self.M):
                    t = int(toks[b, j])
                    if t < 0:
                        continue
                    hit = np.nonzero(self.toks[i] == t)[0]
                    if hit.size:
                        self.cnt[i, hit[0]] += cnt[b, j]
                    else:
                        free = np.nonzero(self.toks[i] < 0)[0]
                        if free.size:
                            self.toks[i, free[0]] = t
                            self.cnt[i, free[0]] = cnt[b, j]
                        else:
                            # row full: weakest column loses the merge
                            w = int(np.argmin(self.cnt[i]))
                            if cnt[b, j] > self.cnt[i, w]:
                                self.toks[i, w] = t
                                self.cnt[i, w] = cnt[b, j]
                self.tot[i] += tot[b]
                self.occ[i] = max(self.occ[i], float(occ[b]))
            self.pos[i] = pos[b]
        self._n = n0 + added
        return added

    def drop_rows(self, idx: np.ndarray) -> None:
        """Physically remove rows (controller shrink / coldest purge)."""
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        keep = np.ones(self._n, dtype=bool)
        keep[idx] = False
        for k in self.keys[idx]:
            self._ix.pop(int(k), None)
        def _cut(a):
            out = a[:self._n][keep]
            return out
        self.keys = _cut(self.keys)
        self.raws = _cut(self.raws)
        self.toks = _cut(self.toks)
        self.cnt = _cut(self.cnt)
        self.tot = _cut(self.tot)
        self.occ = _cut(self.occ)
        self.pos = _cut(self.pos)
        self._n = int(keep.sum())
        self._ix = {int(k): i
                    for i, k in enumerate(self.keys[:self._n])}

    def coldest(self, n: int) -> np.ndarray:
        """Row indices of the n lowest-occ entries (shrink victims)."""
        if not self._n:
            return np.zeros(0, dtype=np.int64)
        n = min(n, self._n)
        return np.argsort(self.occ[:self._n], kind="stable")[:n]

    # -- persistence ----------------------------------------------------

    def save(self, path) -> None:
        """Write the bank to a ``.fluxbank`` npz file - a weight bank
        as a loadable artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:          # keep the .fluxbank ext
            np.savez_compressed(
                f, order=np.int64(self.order), M=np.int64(self.M),
                keys=self.keys[:self._n], raws=self.raws[:self._n],
                toks=self.toks[:self._n], cnt=self.cnt[:self._n],
                tot=self.tot[:self._n], occ=self.occ[:self._n],
                pos=self.pos[:self._n])
        meta = path.with_suffix(".meta.json")
        meta.write_text(json.dumps(
            {"order": self.order, "M": self.M, "rows": int(len(self))}))

    @classmethod
    def load(cls, path) -> "FluxBank":
        z = np.load(Path(path))
        b = cls(int(z["order"]), int(z["M"]))
        b.keys = z["keys"].astype(np.int64)
        b.raws = z["raws"].astype(np.int64)
        b.toks = z["toks"].astype(np.int32)
        b.cnt = z["cnt"].astype(np.float32)
        b.tot = z["tot"].astype(np.float32)
        b.occ = z["occ"].astype(np.float32)
        b.pos = z["pos"].astype(np.int64)
        b._n = int(b.keys.shape[0])
        b._ix = {int(k): i for i, k in enumerate(b.keys)}
        return b
