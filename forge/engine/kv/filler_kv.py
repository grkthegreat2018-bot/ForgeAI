"""Filler-token KV cache eviction (R50-4).

Semantic token-type-aware eviction: function words, punctuation, and
other "filler" tokens carry far less retrieval signal than entities and
content words — yet they occupy the same KV budget. This policy tracks
which cached positions hold filler tokens and evicts them first when the
cache exceeds capacity, before falling back to attention-score eviction
(SnapKV-style) for the remaining non-filler positions.

The filler set is caller-supplied via ``filler_pred`` (a ``token_id ->
bool`` predicate) or ``filler_ids`` (an id set); ``filler_token_ids()``
builds a sensible default for English function words + punctuation from
the active tokenizer. Callers that never pass ``token_ids`` degrade
gracefully to SnapKV-style score eviction.

Safety rails (all configurable):
  - ``n_sink`` leading positions are always kept (attention sinks).
  - the trailing ``observation_window`` tokens are always kept — fillers
    in the recent window still shape local syntax.
  - ``filler_keep_ratio`` keeps the highest-attention share of flagged
    fillers (0.0 = strict drop-all; 0.1 is a safer default for models
    that lean on determiners/negation).
  - evictions happen only when ``seq_len > max_capacity``.

Usage:
    from forge.engine.kv.filler_kv import FillerKVCache, filler_token_ids
    ids = filler_token_ids(tokenizer)
    cache = FillerKVCache(budget=512, filler_ids=ids)
    cache.append(k, v, position, attention_weights=attn, token_ids=toks)
"""

import torch

# English function words, auxiliaries, conjunctions, prepositions,
# pronouns, and punctuation that typically carry low retrieval signal.
# Encoded per-tokenizer by ``filler_token_ids`` — multi-token encodings
# contribute all of their pieces.
DEFAULT_FILLER_STRINGS: tuple[str, ...] = (
    " the", " a", " an", " and", " or", " but", " if", " then", " else",
    " of", " at", " by", " for", " with", " about", " against", " between",
    " into", " through", " during", " before", " after", " above", " below",
    " to", " from", " up", " down", " in", " out", " on", " off", " over",
    " under", " again", " further", " once", " here", " there", " when",
    " where", " why", " how", " all", " any", " both", " each", " few",
    " more", " most", " other", " some", " such", " no", " nor", " not",
    " only", " own", " same", " so", " than", " too", " very", " just",
    " is", " are", " was", " were", " be", " been", " being", " have",
    " has", " had", " having", " do", " does", " did", " doing", " would",
    " could", " should", " shall", " will", " may", " might", " must",
    " can", " i", " you", " he", " she", " it", " we", " they", " me",
    " him", " her", " us", " them", " my", " your", " his", " its", " our",
    " their", " this", " that", " these", " those", " what", " which",
    " who", " whom", " as", " because", " while", " although", " though",
    " since", " until", " unless", ",", ".", ";", ":", "!", "?", "'",
    '"', "(", ")", "-", "\n", "\n\n",
)


def filler_token_ids(tokenizer, extra_strings=None) -> frozenset[int]:
    """Encode filler strings to a token-id set for ``filler_pred``.

    Every id produced by each filler string is included (subword pieces
    of multi-token fillers are filler-ish too). Strings that fail to
    encode are skipped. ``extra_strings`` appends caller-supplied
    fillers (e.g. chat-template markers).
    """
    ids: set[int] = set()
    for s in (*DEFAULT_FILLER_STRINGS, *(extra_strings or ())):
        try:
            encoded = tokenizer.encode(s, add_special_tokens=False)
        except Exception:
            continue
        ids.update(encoded)
    return frozenset(ids)


class FillerKVCache:
    """KV cache with filler-token-first eviction.

    Mirrors :class:`forge.engine.kv.snapkv.SnapKVCache` semantics —
    pre-allocated buffers, observation-window protection, accumulated
    attention scores — with an additional per-position ``is_filler``
    flag. On overflow, eviction runs in two passes:

      1. filler positions outside sink + observation window are dropped
         (highest-score ``filler_keep_ratio`` fraction may be retained),
      2. if still over capacity, lowest-score non-fillers are evicted
         exactly as SnapKV does.
    """

    def __init__(self, observation_window: int = 128, budget: int = 512,
                 n_kv_heads: int = 2, head_dim: int = 128,
                 n_sink: int = 4, filler_keep_ratio: float = 0.0,
                 filler_ids: frozenset[int] | set[int] | None = None,
                 filler_pred=None,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.obs_window = observation_window
        self.budget = budget
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.n_sink = max(0, n_sink)
        self.filler_keep_ratio = min(max(filler_keep_ratio, 0.0), 1.0)
        self.filler_ids = set(filler_ids) if filler_ids is not None else None
        self.filler_pred = filler_pred
        self.device = device
        self.dtype = dtype

        self.k_cache = None
        self.v_cache = None
        self.attention_scores = None  # [B, n_kv, capacity] accumulated
        self.is_filler = None         # [capacity] bool flags
        self.seq_len = 0
        self.max_capacity = budget + observation_window
        # Stats for info()/diagnostics
        self.filler_evicted = 0
        self.score_evicted = 0

    def _is_filler(self, token_id: int) -> bool:
        if self.filler_pred is not None:
            return bool(self.filler_pred(token_id))
        if self.filler_ids is not None:
            return token_id in self.filler_ids
        return False

    def _ensure_buffer(self, B: int, T: int, dtype: torch.dtype):
        needed = max(self.max_capacity + T, self.seq_len + T)
        if self.k_cache is None:
            self.k_cache = torch.zeros(B, self.n_kv, needed, self.head_dim,
                                       device=self.device, dtype=dtype)
            self.v_cache = torch.zeros_like(self.k_cache)
            self.attention_scores = torch.zeros(
                B, self.n_kv, needed, device=self.device,
                dtype=torch.bfloat16 if self.dtype == torch.bfloat16
                else self.dtype)
            self.is_filler = torch.zeros(needed, dtype=torch.bool,
                                         device=self.device)
        elif self.k_cache.shape[2] < needed:
            new_size = max(needed, self.k_cache.shape[2] * 2)
            new_k = torch.zeros(B, self.n_kv, new_size, self.head_dim,
                                device=self.device, dtype=self.k_cache.dtype)
            new_v = torch.zeros_like(new_k)
            new_scores = torch.zeros(
                B, self.n_kv, new_size, device=self.device,
                dtype=self.attention_scores.dtype)
            new_flags = torch.zeros(new_size, dtype=torch.bool,
                                    device=self.device)
            new_k[:, :, :self.seq_len] = self.k_cache[:, :, :self.seq_len]
            new_v[:, :, :self.seq_len] = self.v_cache[:, :, :self.seq_len]
            new_scores[:, :, :self.seq_len] = \
                self.attention_scores[:, :, :self.seq_len]
            new_flags[:self.seq_len] = self.is_filler[:self.seq_len]
            self.k_cache, self.v_cache = new_k, new_v
            self.attention_scores = new_scores
            self.is_filler = new_flags

    def append(self, k: torch.Tensor, v: torch.Tensor, position: int,
               attention_weights: torch.Tensor | None = None,
               token_ids: list[int] | torch.Tensor | None = None):
        """Append K/V; optionally flag filler tokens and score attention.

        Args:
            k, v: [B, n_kv, T, head_dim]
            position: logical position (unused; kept for interface parity)
            attention_weights: [B, n_kv, T, cache_size] — accumulated into
                per-position importance scores (SnapKV semantics).
            token_ids: length-T ids for the appended positions; flagged
                via ``filler_pred``/``filler_ids`` when provided.
        """
        B, _, T, _ = k.shape
        self._ensure_buffer(B, T, k.dtype)

        if attention_weights is not None and self.seq_len > 0:
            scores = attention_weights.sum(dim=2)  # [B, n_kv, cache_size]
            cs = scores.shape[-1]
            if cs <= self.seq_len:
                self.attention_scores[:, :, :cs] += scores
            else:
                self.attention_scores[:, :, :self.seq_len] += \
                    scores[:, :, :self.seq_len]

        end = self.seq_len + T
        self.k_cache[:, :, self.seq_len:end].copy_(k)
        self.v_cache[:, :, self.seq_len:end].copy_(v)
        self.attention_scores[:, :, self.seq_len:end].zero_()
        if token_ids is not None:
            flags = torch.tensor(
                [self._is_filler(int(t)) for t in token_ids],
                dtype=torch.bool, device=self.device)
            self.is_filler[self.seq_len:end] = flags[:end - self.seq_len]
        else:
            self.is_filler[self.seq_len:end] = False
        self.seq_len = end

        if self.seq_len > self.max_capacity:
            self._evict()

    def _evict(self):
        """Filler-first eviction, then score-based fallback."""
        total = self.seq_len
        n_to_evict = total - self.max_capacity
        obs_start = total - self.obs_window

        # Protected: sink prefix + observation window (never evicted).
        protected = torch.zeros(total, dtype=torch.bool, device=self.device)
        protected[:min(self.n_sink, total)] = True
        protected[obs_start:] = True

        scores = self.attention_scores[:, :, :total].mean(dim=1).mean(dim=0)

        evict = torch.zeros(total, dtype=torch.bool, device=self.device)

        # Pass 1: evict filler positions outside the protected region.
        filler_region = self.is_filler[:total].clone() & ~protected
        n_fillers = int(filler_region.sum().item())
        if n_fillers > 0:
            n_keep_fillers = int(n_fillers * self.filler_keep_ratio)
            if n_keep_fillers > 0:
                # Keep the highest-attention fillers (they may carry
                # signal — e.g. negation particles).
                keep_idx = scores.masked_fill(
                    ~filler_region, float("-inf")).topk(
                    n_keep_fillers).indices
                filler_region[keep_idx] = False
            evict |= filler_region
            self.filler_evicted += int(evict.sum().item())

        remaining = n_to_evict - int(evict.sum().item())

        # Pass 2: SnapKV-style lowest-score eviction for the rest.
        if remaining > 0:
            candidates = ~protected & ~evict
            n_cand = int(candidates.sum().item())
            if n_cand > 0:
                n_drop = min(remaining, n_cand)
                drop_idx = scores.masked_fill(
                    ~candidates, float("inf")).topk(
                    n_drop, largest=False).indices
                evict[drop_idx] = True
                self.score_evicted += int(n_drop)

        keep = ~evict
        new_seq_len = int(keep.sum().item())
        # Buffers may be larger than `total` (growth slack) — mask only the
        # first `total` slots or the boolean index will not broadcast.
        self.k_cache[:, :, :new_seq_len] = \
            self.k_cache[:, :, :total][:, :, keep]
        self.v_cache[:, :, :new_seq_len] = \
            self.v_cache[:, :, :total][:, :, keep]
        self.attention_scores[:, :, :new_seq_len] = \
            self.attention_scores[:, :, :total][:, :, keep]
        self.is_filler[:new_seq_len] = self.is_filler[:total][keep]
        self.seq_len = new_seq_len

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.k_cache[:, :, :self.seq_len],
                self.v_cache[:, :, :self.seq_len])

    def get_past_kv(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.k_cache is None or self.seq_len == 0:
            return None
        return (self.k_cache[:, :, :self.seq_len],
                self.v_cache[:, :, :self.seq_len])

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.attention_scores = None
        self.is_filler = None
        self.seq_len = 0
        self.filler_evicted = 0
        self.score_evicted = 0

    def info(self) -> dict:
        n_fillers = (
            int(self.is_filler[:self.seq_len].sum().item())
            if self.is_filler is not None else 0)
        return {
            "type": "filler",
            "observation_window": self.obs_window,
            "budget": self.budget,
            "max_capacity": self.max_capacity,
            "current_size": self.seq_len,
            "seq_len": self.seq_len,
            "n_sink": self.n_sink,
            "fillers_cached": n_fillers,
            "filler_evicted": self.filler_evicted,
            "score_evicted": self.score_evicted,
            "filler_keep_ratio": self.filler_keep_ratio,
            "compression": max(1.0, self.seq_len / max(1, self.seq_len)),
        }
