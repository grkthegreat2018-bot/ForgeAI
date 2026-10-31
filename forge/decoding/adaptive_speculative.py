"""Adaptive n-gram + EAGLE-3 combo speculative decoding.

Based on findings from "Speculative Decoding: Performance or Illusion?"
(MLSys 2026): adaptively combining n-gram and EAGLE can push theoretical
speedup to 4.9× on code-editing workloads. EAGLE-3 is the best all-around
choice (2.89× average), but n-gram dominates on code-editing / high-overlap
tasks (where input and output share substantial token sequences).

Strategy:
  1. Maintain an n-gram cache of recently generated token sequences
  2. For each decode step, try n-gram lookup first (free, no model forward)
  3. If n-gram has a high-confidence match, use it as the draft
  4. Otherwise, fall back to EAGLE-3 (or MTP) for drafting
  5. The target model verifies both types of drafts identically

This is training-free for the n-gram path and composes with existing
EAGLE-3/MTP heads. The key insight: n-gram is optimal when the output
contains repeated phrases from the input (code editing, RAG, summarization),
while EAGLE-3 is better for novel generation (creative writing, math).

Adaptive selection:
  - Track per-request n-gram hit rate and EAGLE acceptance rate
  - If n-gram hit rate > 60%: prefer n-gram (code/RAG workload)
  - If EAGLE acceptance > 75%: prefer EAGLE (novel generation)
  - Otherwise: try both in parallel and pick the longer accepted sequence
"""
from __future__ import annotations

from collections import defaultdict, deque

import torch


class NGramCache:
    """N-gram lookup cache for speculative decoding.

    Maintains a cache of n-gram → continuation mappings from recently
    seen token sequences. When a matching n-gram is found in the current
    context, the cached continuation is used as a speculative draft.

    The cache is updated incrementally as new tokens are generated.
    """

    def __init__(self, n: int = 3, max_cache_size: int = 10000,
                 max_draft_len: int = 5):
        self.n = n
        self.max_cache_size = max_cache_size
        self.max_draft_len = max_draft_len
        self._cache: dict[tuple[int, ...], deque[list[int]]] = defaultdict(
            lambda: deque(maxlen=4))  # keep up to 4 continuations per n-gram
        self._recent_tokens: list[int] = []

    def update(self, tokens: list[int]):
        """Update the cache with new tokens."""
        self._recent_tokens.extend(tokens)
        # Only keep last 4096 tokens for memory
        if len(self._recent_tokens) > 4096:
            self._recent_tokens = self._recent_tokens[-4096:]

        # Extract n-grams and their continuations
        for i in range(len(tokens) - self.n):
            ngram = tuple(tokens[i:i + self.n])
            continuation = tokens[i + self.n:i + self.n + self.max_draft_len]
            if continuation:
                self._cache[ngram].append(continuation)

        # Evict old entries if cache is too large
        if len(self._cache) > self.max_cache_size:
            # Remove oldest entries (FIFO — dict preserves insertion order in Python 3.7+)
            keys = list(self._cache.keys())
            for k in keys[:len(keys) - self.max_cache_size]:
                del self._cache[k]

    def lookup(self, tokens: list[int]) -> list[int] | None:
        """Look up the best continuation for the current token sequence.

        Args:
            tokens: recent tokens (at least n tokens)

        Returns:
            continuation: list of draft tokens, or None if no match
        """
        if len(tokens) < self.n:
            return None

        ngram = tuple(tokens[-self.n:])
        continuations = self._cache.get(ngram)
        if not continuations:
            return None

        # Return the most recent continuation (highest recency priority)
        return list(continuations[-1])

    def hit_rate(self) -> float:
        """Estimate cache hit rate (fraction of lookups that found a match)."""
        if not self._recent_tokens or len(self._recent_tokens) < self.n:
            return 0.0
        hits = 0
        total = 0
        for i in range(self.n, len(self._recent_tokens)):
            ngram = tuple(self._recent_tokens[i - self.n:i])
            if ngram in self._cache:
                hits += 1
            total += 1
        return hits / max(total, 1)

    def clear(self):
        self._cache.clear()
        self._recent_tokens.clear()


class AdaptiveSpeculativeDecoder:
    """Adaptive n-gram + EAGLE-3 combo speculative decoder.

    Combines n-gram lookup (free, great for code/RAG) with EAGLE-3
    (model-based, great for novel generation). Adaptively selects the
    best drafter per request based on observed hit/acceptance rates.
    """

    def __init__(self, n_gram_size: int = 3, max_draft_len: int = 5,
                 ngram_threshold: float = 0.6,  # prefer n-gram if hit_rate > 60%
                 eagle_threshold: float = 0.75,  # prefer EAGLE if acceptance > 75%
                 eagle_head=None,  # optional EAGLE-3 head
                 mtp_module=None):  # optional MTP module
        self.ngram_cache = NGramCache(n=n_gram_size, max_draft_len=max_draft_len)
        self.ngram_threshold = ngram_threshold
        self.eagle_threshold = eagle_threshold
        self.eagle_head = eagle_head
        self.mtp_module = mtp_module

        # Per-request stats
        self._ngram_attempts = 0
        self._ngram_hits = 0
        self._eagle_attempts = 0
        self._eagle_accepts = 0

        # Current drafter preference: "ngram", "eagle", or "both"
        self._drafter = "both"

    def draft(self, tokens: list[int], hidden_state: torch.Tensor | None = None,
               ) -> tuple[list[int], str]:
        """Generate a speculative draft using the best drafter.

        Args:
            tokens: recently generated token IDs
            hidden_state: optional model hidden state (for EAGLE)

        Returns:
            (draft_tokens, drafter_name)
        """
        # Update drafter preference based on stats
        self._update_preference()

        ngram_draft = None
        eagle_draft = None

        # Try n-gram first (free)
        if self._drafter in ("ngram", "both"):
            ngram_draft = self.ngram_cache.lookup(tokens)
            if ngram_draft is not None:
                self._ngram_attempts += 1

        # Try EAGLE/MTP if available
        if self._drafter in ("eagle", "both") and self.eagle_head is not None:
            eagle_draft = self._eagle_draft(hidden_state)
            if eagle_draft is not None:
                self._eagle_attempts += 1
        elif self._drafter in ("eagle", "both") and self.mtp_module is not None:
            eagle_draft = self._mtp_draft(hidden_state)
            if eagle_draft is not None:
                self._eagle_attempts += 1

        # Pick the best draft
        if ngram_draft and eagle_draft:
            # Both available: pick the longer one (more accepted tokens = faster)
            if len(ngram_draft) >= len(eagle_draft):
                return ngram_draft, "ngram"
            return eagle_draft, "eagle"
        elif ngram_draft:
            return ngram_draft, "ngram"
        elif eagle_draft:
            return eagle_draft, "eagle"
        return [], "none"

    def _eagle_draft(self, hidden_state: torch.Tensor | None) -> list[int] | None:
        """Generate draft using EAGLE-3 head."""
        if hidden_state is None or self.eagle_head is None:
            return None
        try:
            with torch.inference_mode():
                draft = self.eagle_head.predict(hidden_state)
            return draft.tolist() if hasattr(draft, 'tolist') else list(draft)
        except Exception:
            return None

    def _mtp_draft(self, hidden_state: torch.Tensor | None) -> list[int] | None:
        """Generate draft using MTP module."""
        if hidden_state is None or self.mtp_module is None:
            return None
        try:
            with torch.inference_mode():
                draft = self.mtp_module.predict(hidden_state)
            return draft.tolist() if hasattr(draft, 'tolist') else list(draft)
        except Exception:
            return None

    def record_result(self, drafter: str, n_accepted: int, n_draft: int):
        """Record the result of a speculative decoding step."""
        if drafter == "ngram":
            if n_accepted > 0:
                self._ngram_hits += 1
        elif drafter == "eagle" and n_accepted > 0:
            self._eagle_accepts += 1

    def _update_preference(self):
        """Update drafter preference based on observed stats."""
        ngram_rate = self._ngram_hits / max(self._ngram_attempts, 1)
        eagle_rate = self._eagle_accepts / max(self._eagle_attempts, 1)

        if ngram_rate > self.ngram_threshold and ngram_rate > eagle_rate:
            self._drafter = "ngram"
        elif eagle_rate > self.eagle_threshold:
            self._drafter = "eagle"
        else:
            self._drafter = "both"

    def stats(self) -> dict:
        return {
            "drafter": self._drafter,
            "ngram_attempts": self._ngram_attempts,
            "ngram_hits": self._ngram_hits,
            "ngram_rate": self._ngram_hits / max(self._ngram_attempts, 1),
            "eagle_attempts": self._eagle_attempts,
            "eagle_accepts": self._eagle_accepts,
            "eagle_rate": self._eagle_accepts / max(self._eagle_attempts, 1),
        }
