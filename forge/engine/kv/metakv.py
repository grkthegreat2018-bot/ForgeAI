"""MetaKV: feature-driven dispatcher over KV cache strategies (R60).

Inspired by "MetaKV: Meta-Learning for KV Cache Strategy Selection"
(arXiv 2609.07966).  The paper trains a predictor that maps request
features (context length, memory pressure, task type) onto the
best-suited KV compression strategy.  This implementation ships a
small, documented heuristic table in place of the learned predictors:

  - long context (``max_seq_len`` >= ``LONG_CTX``) + memory pressure
    (free VRAM < ``LOW_MEM_BYTES``)  ->  ``anchor``
      (anchor reconstruction covers long contexts when VRAM is tight)
  - everything else                 ->  ``beacon``
      (beacon-centroid eviction is the balanced default)

``select(features)`` is the single seam a learned predictor plugs
into later — it takes the same feature dict ``collect_features``
builds and returns a strategy name accepted by ``build_kv_cache``.
``MetaKVCache(force_inner="anchor")`` pins the inner strategy for
ablation/tests.

The wrapper delegates the whole ``KVCacheStrategy`` interface —
including the R60 decode-time ``compact_past_kv`` bridge hook and the
``compact_over`` watermark — to the chosen inner strategy, so the
engine sees one uniform strategy object.

Usage:
    from forge.engine.kv.metakv import MetaKVCache
    cache = MetaKVCache()                     # heuristic dispatch
    cache = MetaKVCache(force_inner="anchor") # pinned inner strategy
    cache.init(n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype)
"""

from __future__ import annotations

import torch

from forge.engine.kv_backend import KVCacheStrategy


class MetaKVCache(KVCacheStrategy):
    """Dispatcher strategy: pick an inner KV cache from request features.

    Feature dict keys (all optional; ``collect_features`` fills what it
    can observe):

      - ``max_seq_len`` / ``seq_len`` — configured context length,
      - ``free_vram_bytes`` — bytes free on the active CUDA device
        (``None`` on CPU or when ``torch.cuda.mem_get_info`` fails),
      - ``device`` — device string.
    """

    # Heuristic table constants (documented stand-ins for the paper's
    # learned predictors — see module docstring).
    LONG_CTX = 4096
    LOW_MEM_BYTES = 6 * 1024**3  # 6 GiB free-VRAM watermark

    def __init__(self, force_inner: str | None = None,
                 inner_overrides: dict | None = None):
        self.force_inner = force_inner
        self.inner_overrides = dict(inner_overrides or {})
        self._inner = None
        self.selected = None
        self.features = None
        self.seq_len = 0
        self.compact_over = None

    # ── feature collection + selection ───────────────────────────────

    def collect_features(self, max_seq_len, device) -> dict:
        """Build the request-feature dict for ``select``.

        Memory pressure is probed via ``torch.cuda.mem_get_info`` when
        the target device is CUDA; on CPU (or if the probe fails) the
        feature is ``None`` and the low-memory branch never fires —
        heuristics degrade to the ``beacon`` default.
        """
        free = None
        if str(device).startswith("cuda") and torch.cuda.is_available():
            try:
                free, _ = torch.cuda.mem_get_info()
            except Exception:
                free = None
        return {
            "max_seq_len": max_seq_len,
            "seq_len": max_seq_len,
            "free_vram_bytes": free,
            "device": str(device),
        }

    def select(self, features: dict) -> str:
        """Map request features to an inner strategy name.

        Documented heuristic table (see module docstring).  A learned
        predictor replaces this method — same signature, same return
        contract (a name ``build_kv_cache`` accepts).
        """
        features = features or {}
        seq = features.get("seq_len") or features.get("max_seq_len") or 0
        free = features.get("free_vram_bytes")
        low_mem = free is not None and free < self.LOW_MEM_BYTES
        if seq >= self.LONG_CTX and low_mem:
            return "anchor"
        return "beacon"

    # ── KVCacheStrategy interface (delegated) ────────────────────────

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from forge.engine.kv_backend import build_kv_cache

        self.features = self.collect_features(max_seq_len, device)
        name = self.force_inner or self.select(self.features)
        self.selected = name
        inner = build_kv_cache(name)
        # Inner-strategy constructor overrides (e.g. a tighter budget)
        # are applied as attribute patches post-build where supported.
        for attr, val in self.inner_overrides.items():
            if hasattr(inner, attr):
                setattr(inner, attr, val)
        inner.init(n_heads, head_dim, n_kv_heads, max_seq_len, device,
                   dtype)
        self._inner = inner
        self.seq_len = 0
        # Bridge watermark passthrough so the decode loop triggers
        # compaction on the inner strategy's schedule.
        self.compact_over = getattr(inner, "compact_over", None)

    def append(self, k, v, position, attention_weights=None,
               token_ids=None):
        self._inner.append(k, v, position,
                           attention_weights=attention_weights)
        self.seq_len = getattr(self._inner, "seq_len", self.seq_len)

    def get(self, positions):
        return self._inner.get(positions)

    def get_past_kv(self):
        if hasattr(self._inner, "get_past_kv"):
            return self._inner.get_past_kv()
        return self._inner.get(None)

    def compact_past_kv(self, past_kv, **kw):
        """Decode-time bridge hook — delegated to the inner strategy."""
        return self._inner.compact_past_kv(past_kv, **kw)

    def clear(self):
        if self._inner is not None:
            self._inner.clear()
        self.seq_len = 0

    def info(self):
        d = dict(self._inner.info()) if self._inner is not None else {}
        # Keep the inner strategy's live stats (seq_len etc.) and nest
        # its type name — the outer "type" stays "metakv".
        d["inner_type"] = d.get("type")
        d.update({
            "type": "metakv",
            "inner": self.selected,
            "features": self.features,
        })
        return d
