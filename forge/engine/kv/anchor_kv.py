"""AnchorKV: anchor-based KV cache eviction (R60).

Inspired by "AnchorKV: Anchor-Based KV Cache Compression" (arXiv
2608.02901).  Each layer picks ``n_anchors`` representative keys by
farthest-point sampling (FPS) over the pooled, cosine-normalized keys.
Every non-anchor position is then mapped to its nearest anchor with a
per-head scalar residual scale

    s_h = <k_t,h, k_a,h> / <k_a,h, k_a,h>

so the evicted position is approximated by ``s * anchor``.  Positions
whose reconstruction error is LOW are well covered by the anchor set —
they are evicted first.  Positions with high error (unique keys no
anchor explains) are the expensive ones to lose, so they survive
alongside the protected set:

  - first ``sink`` positions (attention sinks) — always kept,
  - all anchor positions — always kept,
  - last ``window`` positions (recent context) — always kept,
  - remaining budget filled with the HIGHEST-error middle positions.

Two live paths:

  1. classic strategy API (``append``/``get``).  ``get`` returns the
     retained full-fidelity K/V; ``get_full()`` additionally returns a
     reconstructed full-length K/V where evicted positions are rebuilt
     from their anchor + stored residual scale (the paper's read path).
  2. ``compact_past_kv(past_kv)`` — the decode-time bridge hook.  Only
     attention-layer ``(k, v)`` tuple entries are sliced; ``None``,
     dicts and other conv/SSM state pass through by identity.  On the
     bridge path the true tensors are kept for every retained position
     (no reconstruction needed — anchors simply dominate the keep set).

RoPE note: keys are stored post-RoPE, so evicting a middle position
does not corrupt the remaining keys (standard eviction semantics).

Usage:
    from forge.engine.kv.anchor_kv import AnchorKVCache
    cache = AnchorKVCache(budget=512, n_anchors=32)
    cache.init(n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from forge.engine.kv_backend import KVCacheStrategy


def _is_kv_entry(entry) -> bool:
    """True for attention-layer ``(k, v)`` tuple entries (the only
    present-KV shape the bridge is allowed to slice)."""
    return (isinstance(entry, tuple) and len(entry) == 2
            and torch.is_tensor(entry[0]) and torch.is_tensor(entry[1])
            and entry[0].dim() >= 3 and entry[0].shape == entry[1].shape)


class AnchorKVCache(KVCacheStrategy):
    """KV cache with anchor-coverage eviction.

    Retention contract (append path and ``compact_past_kv``): sinks ∪
    anchors ∪ recent window ∪ top-error middle positions, truncated to
    ~``budget`` total.  ``compact_over`` (= budget + slack) is the
    watermark the decode bridge uses so compaction runs with hysteresis
    rather than every token.
    """

    def __init__(self, budget: int = 1024, sink: int = 4, window: int = 128,
                 n_anchors: int = 32, slack: int | None = None):
        self.budget = budget
        self.sink = sink
        self.window = window
        self.n_anchors = n_anchors
        self.slack = slack

    # ── KVCacheStrategy interface ────────────────────────────────────

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        self.budget = max(1, min(self.budget, max_seq_len))
        self.window = max(0, min(self.window, max_seq_len))
        # Anchors live inside the budget alongside sinks + window.
        self.n_anchors = max(1, min(self.n_anchors,
                                  max(1, self.budget - self.sink - self.window)))
        slack = self.window if self.slack is None else self.slack
        self.compact_over = self.budget + slack
        self.max_capacity = self.compact_over  # append-path trigger

        # Full-precision store for the append/get API path.
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0

        # Reconstruction bookkeeping (append path only).  Original
        # position ids of the retained slots, plus per-evicted-position
        # anchor + scalar residual scales for ``get_full()``.
        self.orig_len = 0
        self._kept_orig = None        # LongTensor [seq_len]
        self._recon_pos = None        # LongTensor [E] original positions
        self._recon_anchor = None     # LongTensor [E] original positions
        self._recon_sk = None         # [B, n_kv, E, 1]
        self._recon_sv = None         # [B, n_kv, E, 1]

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
        # Original positions are a pure counter — seq_len is in
        # compressed slot space after an eviction, so end-of-buffer
        # arithmetic would under-count.
        self.orig_len += T
        if self.seq_len > self.max_capacity:
            self._evict()

    def get(self, positions=None):
        """Return the retained full-fidelity K/V. ``positions`` is
        accepted for the strategy interface; the retained set is always
        returned in full (SnapKVCache.get() convention)."""
        if self.k_cache is None:
            return None, None
        return (self.k_cache[:, :, :self.seq_len],
                self.v_cache[:, :, :self.seq_len])

    def get_past_kv(self):
        if self.k_cache is None or self.seq_len == 0:
            return None
        return self.get(None)

    def get_full(self):
        """Reconstructed full-length K/V — the paper's read path.

        Retained positions carry their true K/V; evicted positions are
        rebuilt as ``s * anchor`` using the stored per-head scalar
        residual scale recorded at eviction time.  Returns
        ``(k, v)`` of shape [B, n_kv, orig_len, head_dim].
        """
        k_ret, v_ret = self.get(None)
        if k_ret is None:
            return None, None
        B = k_ret.shape[0]
        k_full = torch.zeros(B, self.n_kv, self.orig_len, self.head_dim,
                             device=self.device, dtype=k_ret.dtype)
        v_full = torch.zeros_like(k_full)
        if self._kept_orig is not None and self._kept_orig.numel() > 0:
            k_full[:, :, self._kept_orig] = k_ret
            v_full[:, :, self._kept_orig] = v_ret
        else:
            # Nothing evicted yet — slots are in original order.
            k_full[:, :, :self.seq_len] = k_ret
            v_full[:, :, :self.seq_len] = v_ret
        if self._recon_pos is not None and self._recon_pos.numel() > 0:
            k_full[:, :, self._recon_pos] = \
                self._recon_sk * k_full[:, :, self._recon_anchor]
            v_full[:, :, self._recon_pos] = \
                self._recon_sv * v_full[:, :, self._recon_anchor]
        return k_full, v_full

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0
        self.orig_len = 0
        self._kept_orig = None
        self._recon_pos = None
        self._recon_anchor = None
        self._recon_sk = None
        self._recon_sv = None
        self.evictions = 0

    def info(self):
        n_recon = (0 if self._recon_pos is None
                   else int(self._recon_pos.numel()))
        return {
            "type": "anchor",
            "seq_len": self.seq_len,
            "orig_len": self.orig_len,
            "budget": self.budget,
            "sink": self.sink,
            "window": self.window,
            "n_anchors": self.n_anchors,
            "compact_over": self.compact_over,
            "evictions": self.evictions,
            "reconstructed": n_recon,
            "compression": self.max_seq_len / max(1, self.seq_len),
        }

    # ── decode-time bridge ───────────────────────────────────────────

    def compact_past_kv(self, past_kv, **kw):
        """Compact the model's per-layer present-KV list.

        For each attention-layer ``(k, v)`` tuple entry: pick anchors
        (FPS), score every position by anchor-reconstruction error, and
        keep sinks ∪ anchors ∪ window ∪ top-error positions to
        ~``budget``.  Non-tuple entries pass through by identity.  The
        bridge keeps true tensors for retained positions — no
        reconstruction is needed on this path.
        """
        if not isinstance(past_kv, (list, tuple)):
            return past_kv
        out = []
        for entry in past_kv:
            if not _is_kv_entry(entry):
                out.append(entry)
                continue
            k, v = entry
            T = k.shape[-2]
            if self.budget >= T:
                out.append(entry)
                continue
            keep, _, _, _, _ = self._plan(k, v)
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
            old_orig = self._kept_orig
            new_k = torch.zeros(B, self.n_kv, needed, self.head_dim,
                                device=self.device, dtype=dtype)
            new_v = torch.zeros_like(new_k)
            if old_k is not None and old_len > 0:
                copy_len = min(old_len, old_k.shape[2])
                new_k[:, :, :copy_len].copy_(old_k[:, :, :copy_len])
                new_v[:, :, :copy_len].copy_(old_v[:, :, :copy_len])
            self.k_cache, self.v_cache = new_k, new_v
            if old_orig is None and old_len > 0:
                # Before the first eviction slots are in original order.
                self._kept_orig = torch.arange(
                    old_len, device=self.device)
            elif old_orig is not None:
                self._kept_orig = old_orig

    def _fps(self, pooled, seed_norms=None):
        """Farthest-point sampling over normalized keys.

        Deterministic greedy FPS: seed with the largest-norm key
        (near-zero keys are poor anchors), then repeatedly take the
        point farthest (cosine distance) from the current anchor set.
        ``seed_norms`` are the pre-normalization norms — ``pooled``
        itself is unit-norm so its own norm carries no signal.
        Returns sorted original indices — deterministic, no RNG.
        """
        T = pooled.shape[0]
        n = min(self.n_anchors, T)
        if seed_norms is None:
            seed_norms = pooled.norm(dim=-1)
        anchors = [int(seed_norms.argmax().item())]
        # Distance to the nearest selected anchor (cosine distance).
        d = 1.0 - pooled @ pooled[anchors[0]]
        for _ in range(n - 1):
            i = int(d.argmax().item())
            anchors.append(i)
            d = torch.minimum(d, 1.0 - pooled @ pooled[i])
        return sorted(anchors)

    def _residual_errors(self, k, v, pooled, anchor_idx):
        """Per-position anchor-reconstruction error: [T].

        Each position is mapped to its nearest anchor (cosine on the
        pooled keys); the residual scale is the least-squares per-head
        scalar ``s = <x, x_a>/<x_a, x_a>`` and the error is the
        head-mean relative residual ``||x - s*x_a|| / ||x||`` summed
        over K and V.
        """
        anchor_vecs = pooled[anchor_idx]                # [A, hd]
        nearest = (pooled @ anchor_vecs.T).argmax(dim=-1)   # [T]
        a_idx = anchor_idx[nearest]                     # [T]

        k_a = k[:, :, a_idx]                            # [B, nkv, T, hd]
        v_a = v[:, :, a_idx]
        eps = 1e-8
        sk = ((k * k_a).sum(-1)
              / (k_a * k_a).sum(-1).clamp_min(eps))     # [B, nkv, T]
        sv = ((v * v_a).sum(-1)
              / (v_a * v_a).sum(-1).clamp_min(eps))
        err_k = ((k - sk.unsqueeze(-1) * k_a).norm(dim=-1)
                 / k.norm(dim=-1).clamp_min(eps)).mean(dim=(0, 1))
        err_v = ((v - sv.unsqueeze(-1) * v_a).norm(dim=-1)
                 / v.norm(dim=-1).clamp_min(eps)).mean(dim=(0, 1))
        return err_k + err_v, a_idx, sk, sv

    def _plan(self, k, v):
        """Eviction plan for one layer's K/V.

        Returns ``(keep_mask, anchor_idx, a_idx, sk, sv)`` — the mask
        plus the residual metadata the append path records for
        ``get_full()``: ``a_idx[t]`` is the slot index of position t's
        nearest anchor, ``sk``/``sv`` the per-head residual scales.
        """
        T = k.shape[-2]
        raw = k.float().mean(dim=(0, 1))              # [T, hd]
        pooled = F.normalize(raw, dim=-1, eps=1e-8)
        anchor_idx = torch.tensor(
            self._fps(pooled, raw.norm(dim=-1)),
            device=k.device, dtype=torch.long)
        errors, a_idx, sk, sv = self._residual_errors(k, v, pooled,
                                                    anchor_idx)

        keep = torch.zeros(T, dtype=torch.bool, device=k.device)
        n_sink = min(self.sink, T)
        win_start = max(0, T - self.window)
        keep[:n_sink] = True
        keep[win_start:] = True
        keep[anchor_idx] = True

        n_slots = max(0, self.budget - int(keep.sum().item()))
        if n_slots > 0:
            # Highest-error unprotected positions survive — they are
            # the ones anchors explain worst.  Lowest error = evicted
            # first (well covered by the anchor set).
            cand = (~keep).nonzero(as_tuple=False).squeeze(-1)
            n_take = min(n_slots, cand.shape[0])
            if n_take > 0:
                top = errors[cand].topk(n_take).indices
                keep[cand[top]] = True
        return keep, anchor_idx, a_idx, sk, sv

    def _evict(self):
        """Append-path eviction: compact fp store + record residuals."""
        total = self.seq_len
        k = self.k_cache[:, :, :total]
        v = self.v_cache[:, :, :total]
        keep, _anchor_idx, a_idx, sk, sv = self._plan(k, v)
        new_len = int(keep.sum().item())

        # Current slots' original positions (identity before the first
        # eviction).
        if self._kept_orig is None:
            cur_orig = torch.arange(total, device=self.device)
        else:
            cur_orig = self._kept_orig

        # Record residuals for the positions this eviction drops so
        # get_full() can rebuild them.  Scales are per-head scalars
        # against the anchor's ORIGINAL position.
        evict = ~keep
        if bool(evict.any()):
            ev_pos = cur_orig[evict]
            ev_anchor = cur_orig[a_idx[evict]]
            ev_sk = sk[:, :, evict].unsqueeze(-1)     # [B, nkv, E, 1]
            ev_sv = sv[:, :, evict].unsqueeze(-1)
            if self._recon_pos is None:
                self._recon_pos = ev_pos
                self._recon_anchor = ev_anchor
                self._recon_sk = ev_sk
                self._recon_sv = ev_sv
            else:
                self._recon_pos = torch.cat([self._recon_pos, ev_pos])
                self._recon_anchor = torch.cat(
                    [self._recon_anchor, ev_anchor])
                self._recon_sk = torch.cat([self._recon_sk, ev_sk], dim=2)
                self._recon_sv = torch.cat([self._recon_sv, ev_sv], dim=2)

        # Compact buffers (R50-4: mask only the first `total` slots —
        # buffers may carry growth slack past `total`).
        self.k_cache[:, :, :new_len] = k[:, :, keep]
        self.v_cache[:, :, :new_len] = v[:, :, keep]
        self._kept_orig = cur_orig[keep]
        self.seq_len = new_len
        self.evictions += 1
