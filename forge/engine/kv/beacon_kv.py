"""BeaconKV: beacon-centroid KV cache eviction (R60).

Inspired by "BeaconKV: Beacon-guided KV Cache Compression" (arXiv
2609.04971).  The paper clusters *queries* to find a small set of
"beacon" representatives that predict which cached positions future
attention will need.  This implementation maintains an equivalent
beacon set over observed *keys* — a documented approximation that needs
no query-side hooks:

  - each layer keeps ``n_beacons`` centroids, learned online
    (k-means-lite): every observed key is assigned to its nearest
    beacon by cosine similarity and the beacon is nudged toward the
    assigned mean via EMA,
  - a cached position's importance = max cosine similarity to the
    beacon set + a linear recency bonus,
  - on overflow the cache keeps the first ``sink`` positions (attention
    sinks), the last ``window`` positions (recent context), then fills
    the rest of ``budget`` with the top-scoring middle positions.

Two live paths:

  1. classic strategy API (``append``/``get``) — the strategy zoo
     interface; beacons update on every append,
  2. ``compact_past_kv(past_kv)`` — the decode-time bridge hook.  The
     engine calls this on the model's per-layer ``presents`` list; only
     attention-layer ``(k, v)`` tuple entries are sliced (``None``,
     dicts and other conv/SSM state pass through untouched).

RoPE note: keys are stored post-RoPE, so evicting a middle position
does not corrupt the remaining keys (standard eviction semantics —
the effective RoPE offset of subsequent tokens is compressed along
with the cache).

Follow-up: a query-projection hook would close the gap to the paper's
query-clustering — the key-centroid scoring already approximates it
because keys and queries live in the same attention space.

Usage:
    from forge.engine.kv.beacon_kv import BeaconKVCache
    cache = BeaconKVCache(budget=512, window=128, n_beacons=16)
    cache.init(n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from forge.engine.kv_backend import KVCacheStrategy


def _is_kv_entry(entry) -> bool:
    """True for attention-layer ``(k, v)`` tuple entries (the only
    present-KV shape Beacon/bridge code is allowed to slice)."""
    return (isinstance(entry, tuple) and len(entry) == 2
            and torch.is_tensor(entry[0]) and torch.is_tensor(entry[1])
            and entry[0].dim() >= 3 and entry[0].shape == entry[1].shape)


class BeaconKVCache(KVCacheStrategy):
    """KV cache with beacon-centroid importance eviction.

    Importance of a cached position = ``max_c cos(k_t, beacon_c) +
    recency_weight * t/T``.  Beacons are maintained per layer as
    normalized centroids updated by EMA over nearest-centroid
    assignments of every observed key (online k-means, no stored
    assignments — O(T * n_beacons) per update).

    Retention contract (both the append path and ``compact_past_kv``):
    first ``sink`` + last ``window`` positions always kept, then the
    highest-scoring middle positions up to ``budget`` total.  Eviction
    triggers when the cache exceeds ``compact_over`` (= budget + slack),
    so compaction has hysteresis and does not run every token.
    """

    def __init__(self, budget: int = 1024, sink: int = 4, window: int = 128,
                 n_beacons: int = 16, ema: float = 0.9,
                 recency_weight: float = 0.5, slack: int | None = None):
        self.budget = budget
        self.sink = sink
        self.window = window
        self.n_beacons = n_beacons
        self.ema = ema
        self.recency_weight = recency_weight
        self.slack = slack  # hysteresis band above budget; default = window

    # ── KVCacheStrategy interface ────────────────────────────────────

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # Clamp geometry to the configured sequence length.
        self.budget = max(1, min(self.budget, max_seq_len))
        self.window = max(0, min(self.window, max_seq_len))
        # Resolve slack once so callers can introspect the real geometry.
        self.slack = self.window if self.slack is None else self.slack
        # Watermark used by the decode bridge: compact once a tuple
        # entry grows past this length; compaction brings it back to
        # ~budget, leaving `slack` tokens of headroom between runs.
        self.compact_over = self.budget + self.slack
        self.max_capacity = self.compact_over  # append-path trigger

        # Full-precision store for the append/get API path.
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0

        # Per-layer beacon centroids: layer_key -> (n_beacons, head_dim)
        # normalized float32 rows.  layer_key is the presents index for
        # the bridge path and 0 for the append path.
        self._beacons: dict[int, torch.Tensor] = {}

        # Stats for info()
        self.evictions = 0

    def append(self, k, v, position, attention_weights=None):
        """Append K/V ([B, n_kv, T, head_dim]); evict when over capacity."""
        B, _, T, _ = k.shape
        self._ensure_buffer(B, T, k.dtype)
        end = self.seq_len + T
        self.k_cache[:, :, self.seq_len:end].copy_(k)
        self.v_cache[:, :, self.seq_len:end].copy_(v)
        self.seq_len = end
        # Beacons learn from every observed key (single-layer key 0 on
        # the append path; the bridge keys beacons by presents index).
        self._beacons[0] = self._update_beacons(0, self._pooled(k))
        if self.seq_len > self.max_capacity:
            self._evict()

    def get(self, positions=None):
        """Return the retained K/V — [B, n_kv, seq_len, head_dim].
        ``positions`` is accepted for the strategy interface; the evicted
        cache always returns the full retained set (sibling-cache
        convention: SnapKVCache.get() takes no argument)."""
        if self.k_cache is None:
            return None, None
        return (self.k_cache[:, :, :self.seq_len],
                self.v_cache[:, :, :self.seq_len])

    def get_past_kv(self):
        if self.k_cache is None or self.seq_len == 0:
            return None
        return self.get(None)

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self._beacons = {}
        self.seq_len = 0
        self.evictions = 0

    def info(self):
        return {
            "type": "beacon",
            "seq_len": self.seq_len,
            "budget": self.budget,
            "sink": self.sink,
            "window": self.window,
            "n_beacons": self.n_beacons,
            "compact_over": self.compact_over,
            "beacons_tracked": len(self._beacons),
            "evictions": self.evictions,
            "compression": self.max_seq_len / max(1, self.seq_len),
        }

    # ── decode-time bridge ───────────────────────────────────────────

    def compact_past_kv(self, past_kv, **kw):
        """Compact the model's per-layer present-KV list in place
        semantically (rebuilds the list; tuple entries are sliced).

        Only ``(k, v)`` tensor tuples are touched — conv/SSM ``None``
        entries, dicts and any other objects pass through by identity.
        Beacons are refreshed from every observed attention layer even
        when no eviction is needed, so the centroid estimate tracks the
        key distribution across the whole sequence.
        """
        if not isinstance(past_kv, (list, tuple)):
            return past_kv
        out = []
        for layer_idx, entry in enumerate(past_kv):
            if not _is_kv_entry(entry):
                out.append(entry)
                continue
            k, v = entry
            pooled = self._pooled(k)
            beacons = self._update_beacons(layer_idx, pooled)
            self._beacons[layer_idx] = beacons
            T = k.shape[-2]
            if self.budget >= T:
                out.append(entry)
                continue
            keep = self._keep_mask(T, self._scores(pooled, beacons), k.device)
            idx = keep.nonzero(as_tuple=False).squeeze(-1)
            out.append((k.index_select(-2, idx), v.index_select(-2, idx)))
            self.evictions += 1
        return type(past_kv)(out) if isinstance(past_kv, tuple) else out

    # ── internals ────────────────────────────────────────────────────

    def _ensure_buffer(self, B, T, dtype):
        """Grow the fp store geometrically (no per-append torch.cat)."""
        needed = max(self.max_capacity + T, self.seq_len + T)
        if (self.k_cache is None or self.k_cache.shape[0] != B
                or self.k_cache.shape[2] < needed
                or self.k_cache.dtype != dtype):
            old_k, old_v, old_len = self.k_cache, self.v_cache, self.seq_len
            new_k = torch.zeros(B, self.n_kv, needed, self.head_dim,
                                device=self.device, dtype=dtype)
            new_v = torch.zeros_like(new_k)
            if old_k is not None and old_len > 0:
                copy_len = min(old_len, old_k.shape[2])
                new_k[:, :, :copy_len].copy_(old_k[:, :, :copy_len])
                new_v[:, :, :copy_len].copy_(old_v[:, :, :copy_len])
            self.k_cache, self.v_cache = new_k, new_v

    @staticmethod
    def _pooled(k):
        """Pool keys to one normalized vector per position: [T, hd].

        Averages over batch and KV heads — the shared position mask
        applies to every head anyway, so scoring works on the
        head-mean key.  Returns float32 for stable cosine math.
        """
        pooled = k.float().mean(dim=(0, 1))
        return F.normalize(pooled, dim=-1, eps=1e-8)

    def _update_beacons(self, layer_key, pooled):
        """Online k-means-lite: assign each pooled key to its nearest
        beacon (cosine), EMA-nudge the beacon toward the assigned mean,
        re-normalize.  Cold start seeds beacons with evenly spaced keys
        (deterministic — no RNG, reproducible eviction).

        Returns the layer's beacon tensor — a fresh tensor each call, so
        callers can use it immediately for scoring.
        """
        beacons = self._beacons.get(layer_key)
        if beacons is None:
            T = pooled.shape[0]
            if self.n_beacons <= T:
                idx = torch.linspace(0, T - 1, self.n_beacons,
                                     device=pooled.device).long()
            else:
                # Fewer keys than beacons: take them all, repeat the
                # last one to fill — EMA splits them apart as data
                # arrives.
                idx = torch.arange(T, device=pooled.device).repeat(
                    (self.n_beacons + T - 1) // max(T, 1))[:self.n_beacons]
            beacons = F.normalize(pooled[idx].clone(), dim=-1, eps=1e-8)
            return beacons
        sims = pooled @ beacons.T                     # [T, n_beacons]
        assign = sims.argmax(dim=-1)                  # [T]
        # Rebuild rather than mutate — safe under torch.inference_mode
        # (beacons may be inference tensors created during decode).
        rows = []
        for c in range(beacons.shape[0]):
            sel = assign == c
            if bool(sel.any()):
                mean_vec = F.normalize(
                    pooled[sel].float().mean(dim=0), dim=-1, eps=1e-8)
                rows.append(F.normalize(
                    self.ema * beacons[c] + (1.0 - self.ema) * mean_vec,
                    dim=-1, eps=1e-8))
            else:
                rows.append(beacons[c])
        return torch.stack(rows)

    def _scores(self, pooled, beacons=None):
        """Importance per position: max beacon cosine + recency ramp."""
        T = pooled.shape[0]
        if beacons is None or beacons.shape[0] == 0:
            sim = torch.zeros(T, device=pooled.device, dtype=pooled.dtype)
        else:
            sim = (pooled @ beacons.T).amax(dim=-1)
        ramp = (torch.arange(T, device=pooled.device, dtype=pooled.dtype)
                / max(T - 1, 1))
        return sim + self.recency_weight * ramp

    def _keep_mask(self, T, scores, device):
        """sink ∪ window ∪ top-(budget - protected) middle positions."""
        keep = torch.zeros(T, dtype=torch.bool, device=device)
        n_sink = min(self.sink, T)
        win_start = max(0, T - self.window)
        keep[:n_sink] = True
        keep[win_start:] = True
        n_slots = max(0, self.budget - int(keep.sum().item()))
        if n_slots > 0 and win_start > n_sink:
            cand = scores[n_sink:win_start]
            n_take = min(n_slots, cand.shape[0])
            top = cand.topk(n_take).indices + n_sink
            keep[top] = True
        return keep

    def _evict(self):
        """Append-path eviction: compact the fp store to ~budget."""
        total = self.seq_len
        scores = self._scores(self._pooled(
            self.k_cache[:, :, :total]), self._beacons.get(0))
        keep = self._keep_mask(total, scores, self.device)
        new_len = int(keep.sum().item())
        # Buffers may be larger than `total` (growth slack) — mask only
        # the first `total` slots or the boolean index will not
        # broadcast (R50-4 fix pattern).
        self.k_cache[:, :, :new_len] = \
            self.k_cache[:, :, :total][:, :, keep]
        self.v_cache[:, :, :new_len] = \
            self.v_cache[:, :, :total][:, :, keep]
        self.seq_len = new_len
        self.evictions += 1
