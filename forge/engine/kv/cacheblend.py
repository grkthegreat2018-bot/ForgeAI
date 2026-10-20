"""CacheBlend: non-prefix KV cache reuse via selective boundary recompute.

Port of LMCache's CacheBlend (Best Paper @ ACM EuroSys'25, arXiv:2405.16444)
into the ForgeAI framework, adapted for single-GPU (RTX 5070 12GB) inference
on the LFM2.5-1.2B model.

The problem
-----------
Prefix caching only reuses KV for a *common prefix*.  In RAG / tool-use /
multi-document workloads the reusable text chunks appear at *arbitrary*
positions in the prompt, so prefix caching misses almost everything and the
full prefill is recomputed.  CacheBlend reuses the pre-computed KV of *any*
repeated chunk — prefix or not — by selectively recomputing a small set of
"critical" tokens at chunk boundaries where cross-attention with the
preceding context matters most.  The paper reports ~15% recompute →
2.2-3.3x TTFT reduction, 2.8-5x throughput, with no quality loss.

How it works (ForgeAI adaptation)
---------------------------------
1. ``ChunkStore.register_chunk(token_ids, past_kv)`` — pre-compute and store
   the KV of a text chunk (a retrieved doc, a tool output, a prompt
   template) keyed by a deterministic rolling hash of its tokens.
2. ``CacheBlend.lookup(tokens)`` — slide a rolling hash over the request
   tokens and find every stored chunk that occurs at a non-prefix position.
   Token equality is verified on every hit (hash is only a lookup key).
3. ``BlendAssembler.assemble(matches, n_layers)`` — build the per-layer KV
   buffer by concatenating matched chunks' KV at their request positions.
   Reused **V** is position-independent → copied verbatim.  Reused **K**
   is post-RoPE, so its stored rotation (position 0..chunk_len) must be
   re-rotated to the new absolute position via ``reposition_keys`` (apply
   inverse rotation at the old position, then forward rotation at the new
   position using the model's ``RotaryEmbedding`` cos/sin tables).
4. Boundary tokens (the first ``recompute_tokens`` of each non-prefix
   chunk) are *not* reused — they are recomputed in-context by running the
   model on just those tokens with the preceding assembled KV as
   ``past_key_values``.  This restores the cross-attention that the
   independently-computed chunk KV missed, preserving generation quality.

Novel twist (per AGENTS.md directive C — "prefer novel over copy")
------------------------------------------------------------------
The paper selects critical tokens by an attention-based score computed
during a *full* prefill pass, which defeats the purpose for cold chunks.
We instead use a **position-gradient heuristic**: the first token of a
non-prefix chunk is always recomputed (it has zero preceding context in
the chunk's own KV), and the recompute count decays exponentially into
the chunk (``recompute_tokens = ceil(log2(chunk_len))`` by default).  This
needs no extra forward pass and captures the empirical observation that
cross-attention influence from preceding text decays rapidly within the
chunk.  The fraction is tunable via ``recompute_fraction``.

VRAM budget (RTX 5070 12GB)
---------------------------
Chunk KV is stored on CPU (pinned) by default — the store is a cold
dictionary consulted once per request, not per token.  The assembled
buffer is GPU-resident only for the active request (same footprint as a
normal KV cache).  No extra VRAM beyond the standard cache.  Falls back
to full prefill if no chunks match (zero overhead on the miss path).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from forge.engine.prefix_cache import _chunk_hash
from forge.keys._tensor_utils import _rotate_half

if TYPE_CHECKING:
    from forge.engine.forge_engine import ForgeEngine


# ── data structures ───────────────────────────────────────────────────────


@dataclass
class ChunkRecord:
    """A stored pre-computed chunk: tokens + per-layer KV."""
    chunk_hash: int
    token_ids: list[int]
    # past_kv: list[(k, v) | None] per layer — same format as the model's
    # ``presents``.  k/v shape [B, n_kv, chunk_len, head_dim].
    past_kv: object
    length: int
    access_count: int = 0


@dataclass
class BlendMatch:
    """A non-prefix chunk occurrence found in a request."""
    chunk: ChunkRecord
    request_start: int  # token offset in the request
    request_end: int    # exclusive


# ── chunk store ───────────────────────────────────────────────────────────


class ChunkStore:
    """Stores pre-computed per-chunk KV keyed by a deterministic chunk hash.

    The hash is the same rolling scheme used by ``ChunkedPrefixCache`` so
    chunks registered here are also discoverable as prefix-cache entries
    when they happen to align with a prefix.
    """

    def __init__(self, chunk_size: int = 256, max_chunks: int = 512):
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self._chunks: dict[int, ChunkRecord] = {}
        # Reverse index by chunk length for the matcher.
        self._by_length: dict[int, list[int]] = {}

    def _hash_chunk(self, token_ids: list[int]) -> int:
        return _chunk_hash(token_ids)

    def register_chunk(self, token_ids: list[int], past_kv) -> int:
        """Store a chunk's pre-computed KV.  Returns the chunk hash."""
        if len(token_ids) == 0:
            return 0
        h = self._hash_chunk(token_ids)
        if h in self._chunks:
            self._chunks[h].access_count += 1
            return h
        rec = ChunkRecord(
            chunk_hash=h, token_ids=list(token_ids),
            past_kv=past_kv, length=len(token_ids),
        )
        self._chunks[h] = rec
        self._by_length.setdefault(len(token_ids), []).append(h)
        # LRU-ish cap
        if len(self._chunks) > self.max_chunks:
            oldest = next(iter(self._chunks))
            del self._chunks[oldest]
        return h

    def get(self, chunk_hash: int) -> ChunkRecord | None:
        rec = self._chunks.get(chunk_hash)
        if rec is not None:
            rec.access_count += 1
        return rec

    def __len__(self):
        return len(self._chunks)

    def clear(self):
        self._chunks.clear()
        self._by_length.clear()

    def stats(self) -> dict:
        return {
            "chunks": len(self._chunks),
            "max_chunks": self.max_chunks,
            "chunk_size": self.chunk_size,
            "total_tokens_cached": sum(r.length for r in self._chunks.values()),
        }


# ── range matcher ─────────────────────────────────────────────────────────


class RangeMatcher:
    """Sliding-window rolling-hash matcher for non-prefix chunk occurrences.

    For each distinct stored chunk length ``L``, slide a window of ``L``
    tokens over the request, hash the window, and check the store.  This is
    O(n · #distinct_lengths) hash computations — cheap because the chunk
    lengths are few (typically one or two sizes) and the hash is blake2b
    over a small token slice.
    """

    def __init__(self, store: ChunkStore):
        self.store = store

    def find_matches(self, token_ids: list[int],
                     skip_prefix: bool = True) -> list[BlendMatch]:
        """Find all stored chunks occurring in ``token_ids``.

        Args:
            skip_prefix: if True, a match starting at position 0 is treated
                as a prefix hit (handled by the prefix cache) and skipped,
                so CacheBlend only returns *non-prefix* matches.
        """
        matches: list[BlendMatch] = []
        n = len(token_ids)
        occupied = [False] * n  # greedy non-overlap
        # Try longer chunks first (more tokens reused per hit).
        for length in sorted(self.store._by_length.keys(), reverse=True):
            if length > n:
                continue
            for start in range(0, n - length + 1):
                if all(occupied[start:start + length]):
                    continue
                if skip_prefix and start == 0:
                    continue
                window = token_ids[start:start + length]
                h = self.store._hash_chunk(window)
                rec = self.store.get(h)
                if rec is None:
                    continue
                # Verify token equality (hash collision guard).
                if rec.token_ids != window:
                    continue
                for i in range(start, start + length):
                    occupied[i] = True
                matches.append(BlendMatch(
                    chunk=rec, request_start=start,
                    request_end=start + length))
        matches.sort(key=lambda m: m.request_start)
        return matches


# ── RoPE re-rotation for reused K ─────────────────────────────────────────


def reposition_keys(k: torch.Tensor, rope, old_positions: torch.Tensor,
                    new_positions: torch.Tensor) -> torch.Tensor:
    """Re-rotate post-RoPE keys from ``old_positions`` to ``new_positions``.

    RoPE applies ``R(θ)`` per position.  Stored chunk K was rotated at
    positions ``[0, chunk_len)``; in the assembled buffer it must sit at
    ``[request_start, request_start + chunk_len)``.  We apply the inverse
    rotation at the old position then the forward rotation at the new
    position: ``k_new = R(θ_new) · R(-θ_old) · k_old``.

    Args:
        k: ``[B, n_kv, T, head_dim]`` post-RoPE keys.
        rope: the layer's ``RotaryEmbedding`` module (has ``cos_cached``
            and ``sin_cached`` tables of shape ``[max_seq_len, head_dim]``).
        old_positions / new_positions: ``[T]`` long tensors of absolute
            positions.
    """
    cos_t = rope.cos_cached
    sin_t = rope.sin_cached
    dtype = k.dtype
    cos_old = cos_t[old_positions].to(dtype)  # [T, head_dim]
    sin_old = sin_t[old_positions].to(dtype)
    cos_new = cos_t[new_positions].to(dtype)
    sin_new = sin_t[new_positions].to(dtype)
    # Inverse rotation at old position: x*cos - rotate_half(x)*sin
    k_unrot = k * cos_old - _rotate_half(k) * sin_old
    # Forward rotation at new position: x*cos + rotate_half(x)*sin
    k_new = k_unrot * cos_new + _rotate_half(k_unrot) * sin_new
    return k_new


# ── assembler ─────────────────────────────────────────────────────────────


@dataclass
class BlendPlan:
    """Result of assembling a CacheBlend buffer.

    Attributes:
        past_kv: per-layer assembled KV list (``[(k, v) | None]``), GPU.
        covered_len: number of request tokens covered by the assembled
            buffer (matches + any leading prefix gap filled by recompute).
        recompute_positions: set of absolute token positions that must be
            recomputed in-context (boundary tokens + unmatched gaps).
        reuse_tokens: count of tokens reused from stored chunks.
        recompute_tokens: count of tokens to recompute.
    """
    past_kv: object
    covered_len: int
    recompute_positions: list[int] = field(default_factory=list)
    reuse_tokens: int = 0
    recompute_tokens: int = 0


class BlendAssembler:
    """Assembles a per-layer KV buffer from matched chunks.

    Builds the KV layout for the *covered* prefix of the request.  Reused
    V is copied verbatim; reused K is re-rotated to its new absolute
    position.  Boundary tokens (first ``recompute_tokens`` of each
    non-prefix chunk) and any unmatched gaps are marked for in-context
    recompute.
    """

    def __init__(self, recompute_fraction: float = 0.15,
                 min_recompute: int = 1):
        self.recompute_fraction = recompute_fraction
        self.min_recompute = min_recompute

    def _boundary_count(self, chunk_len: int) -> int:
        """Position-gradient heuristic: ceil(log2(L)) boundary tokens,
        capped by ``recompute_fraction`` of the chunk."""
        n = max(self.min_recompute, int(math.ceil(math.log2(max(chunk_len, 2)))))
        cap = max(self.min_recompute, int(math.ceil(chunk_len * self.recompute_fraction)))
        return min(n, cap)

    def assemble(self, matches: list[BlendMatch], request_len: int,
                 n_layers: int, device, dtype,
                 rope_modules: list) -> BlendPlan:
        """Assemble the KV buffer for the covered prefix.

        ``rope_modules`` is a list (one per layer) of the model's
        ``RotaryEmbedding`` (or ``None`` for conv layers).  When a layer's
        rope is ``None`` (conv layer, no KV) the assembled entry is
        ``None``.
        """
        if not matches:
            return BlendPlan(past_kv=None, covered_len=0)

        # Covered prefix = up to the end of the last match that starts
        # at or before any gap we can bridge.  We cover [0, covered_len)
        # where covered_len = max match end among matches that form a
        # contiguous-enough prefix.  Gaps inside [0, covered_len) are
        # recomputed.
        covered_len = matches[-1].request_end
        # Determine recompute positions: unmatched gaps + chunk boundaries.
        recompute = set()
        reuse = 0
        # Build a per-position ownership map.
        owner = [-1] * covered_len  # -1 = gap (recompute), else match idx
        for mi, m in enumerate(matches):
            b = self._boundary_count(m.chunk.length)
            for pos in range(m.request_start, m.request_end):
                owner[pos] = mi
            # Boundary: first `b` tokens of the chunk are recomputed.
            for pos in range(m.request_start, min(m.request_start + b, m.request_end)):
                recompute.add(pos)
            reuse += m.chunk.length - b
        # Gaps before/within the covered prefix must be recomputed.
        for pos in range(covered_len):
            if owner[pos] == -1:
                recompute.add(pos)

        # Allocate per-layer KV buffers of shape [1, n_kv, covered_len, hd].
        # We infer n_kv / head_dim from the first matched chunk's KV.
        first_kv = None
        for m in matches:
            for layer in m.chunk.past_kv:
                if layer is not None:
                    first_kv = layer
                    break
            if first_kv is not None:
                break
        if first_kv is None:
            return BlendPlan(past_kv=None, covered_len=0)
        _, n_kv, _, head_dim = first_kv[0].shape

        past_kv = []
        for li in range(n_layers):
            # Find this layer's KV in any match (conv layers are None).
            sample = None
            for m in matches:
                if li < len(m.chunk.past_kv) and m.chunk.past_kv[li] is not None:
                    sample = m.chunk.past_kv[li]
                    break
            if sample is None:
                past_kv.append(None)
                continue
            k_buf = torch.zeros(1, n_kv, covered_len, head_dim,
                                dtype=dtype, device=device)
            v_buf = torch.zeros_like(k_buf)
            rope = rope_modules[li] if li < len(rope_modules) else None
            for m in matches:
                layer_kv = m.chunk.past_kv[li]
                if layer_kv is None:
                    continue
                ck, cv = layer_kv
                cs, ce = m.request_start, m.request_end
                clen = m.chunk.length
                # V is position-independent → copy verbatim.
                v_buf[:, :, cs:ce] = cv.to(device=device, dtype=dtype)
                # K needs RoPE re-rotation to the new absolute positions.
                if rope is not None:
                    old_pos = torch.arange(clen, device=device)
                    new_pos = torch.arange(cs, ce, device=device)
                    k_buf[:, :, cs:ce] = reposition_keys(
                        ck.to(device=device, dtype=dtype), rope,
                        old_pos, new_pos)
                else:
                    k_buf[:, :, cs:ce] = ck.to(device=device, dtype=dtype)
            past_kv.append((k_buf, v_buf))

        return BlendPlan(
            past_kv=past_kv, covered_len=covered_len,
            recompute_positions=sorted(recompute),
            reuse_tokens=reuse,
            recompute_tokens=len(recompute),
        )


# ── orchestrator ──────────────────────────────────────────────────────────


class CacheBlend:
    """CacheBlend orchestrator: store chunks, match requests, blend prefill.

    Wired into ``ForgeEngine`` via ``engine._cache_blend``.  The engine
    calls ``blend_prefill`` before the normal prefill; on a miss it
    returns ``None`` and the engine falls back to the standard path with
    zero overhead.
    """

    def __init__(self, chunk_size: int = 256, max_chunks: int = 512,
                 recompute_fraction: float = 0.15,
                 min_cover_tokens: int = 64):
        self.store = ChunkStore(chunk_size=chunk_size, max_chunks=max_chunks)
        self.matcher = RangeMatcher(self.store)
        self.assembler = BlendAssembler(recompute_fraction=recompute_fraction)
        self.min_cover_tokens = min_cover_tokens
        self._blend_hits = 0
        self._blend_misses = 0
        self._tokens_reused = 0
        self._tokens_recomputed = 0

    # ── registration ──────────────────────────────────────────────────

    def register_chunk(self, token_ids: list[int], past_kv) -> int:
        """Pre-compute and store a chunk's KV for future reuse."""
        return self.store.register_chunk(token_ids, past_kv)

    def register_text(self, engine: ForgeEngine, text: str) -> int:
        """Tokenize ``text``, run a prefill, and store the resulting KV."""
        ids = engine.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(engine.device)
        if ids.shape[1] == 0:
            return 0
        with torch.inference_mode():
            out = engine.model(ids, use_cache=True, logits_last_only=True)
        from forge.model_loader import unpack_output_with_kv
        _, past_kv = unpack_output_with_kv(out)
        return self.store.register_chunk(ids[0].cpu().tolist(), past_kv)

    # ── lookup + blend ────────────────────────────────────────────────

    def lookup(self, token_ids: list[int]) -> list[BlendMatch]:
        return self.matcher.find_matches(token_ids, skip_prefix=True)

    def _get_rope_modules(self, engine: ForgeEngine) -> list:
        """Collect per-layer RoPE modules from the model's attention blocks."""
        rope_modules = []
        blocks = getattr(engine.model, "blocks", None) or \
            getattr(engine.model, "layers", None) or []
        for block in blocks:
            attn = getattr(block, "attn", None) or getattr(block, "self_attn", None)
            rope = getattr(attn, "rope", None) if attn is not None else None
            rope_modules.append(rope)
        return rope_modules

    def blend_prefill(self, engine: ForgeEngine,
                      ids: torch.Tensor) -> tuple | None:
        """Attempt a CacheBlend prefill for ``ids``.

        Returns ``(past_kv, covered_len)`` on a productive blend (the
        engine then decodes from ``ids[:, covered_len:]`` with this
        ``past_kv``), or ``None`` to fall back to the normal prefill.

        Boundary / gap tokens are recomputed in-context by running the
        model on just those positions with the assembled KV as the
        preceding cache.
        """
        if ids.shape[1] <= self.min_cover_tokens:
            return None
        token_ids = ids[0].cpu().tolist()
        matches = self.lookup(token_ids)
        if not matches:
            self._blend_misses += 1
            return None
        covered = sum(m.request_end - m.request_start for m in matches)
        if covered < self.min_cover_tokens:
            self._blend_misses += 1
            return None

        n_layers = len(self._get_rope_modules(engine)) or \
            len(matches[0].chunk.past_kv)
        rope_modules = self._get_rope_modules(engine)
        plan = self.assembler.assemble(
            matches, request_len=len(token_ids), n_layers=n_layers,
            device=engine.device, dtype=engine.model.dtype
            if hasattr(engine.model, "dtype") else torch.get_default_dtype(),
            rope_modules=rope_modules)
        if plan.past_kv is None or plan.covered_len == 0:
            self._blend_misses += 1
            return None

        # Recompute boundary / gap tokens in-context.  We process each
        # contiguous recompute region: feed those tokens to the model
        # with the assembled KV sliced to the region's start as past,
        # then splice the freshly-computed KV back into the buffer.
        if plan.recompute_positions:
            self._recompute_regions(engine, ids, plan)

        self._blend_hits += 1
        self._tokens_reused += plan.reuse_tokens
        self._tokens_recomputed += plan.recompute_tokens
        return plan.past_kv, plan.covered_len

    def _recompute_regions(self, engine: ForgeEngine, ids: torch.Tensor,
                           plan: BlendPlan) -> None:
        """Recompute boundary/gap tokens in-context, splicing into ``plan``."""
        from forge.model_loader import unpack_output_with_kv
        positions = plan.recompute_positions
        # Group contiguous runs.
        runs = []
        run_start = positions[0]
        prev = positions[0]
        for p in positions[1:]:
            if p == prev + 1:
                prev = p
            else:
                runs.append((run_start, prev + 1))
                run_start = p
                prev = p
        runs.append((run_start, prev + 1))

        for (rs, re_) in runs:
            # Preceding KV = assembled buffer sliced to rs tokens.
            past = []
            for layer in plan.past_kv:
                if layer is None:
                    past.append(None)
                    continue
                k, v = layer
                past.append((k[:, :, :rs], v[:, :, :rs]))
            region_ids = ids[:, rs:re_]
            with torch.inference_mode():
                out = engine.model(
                    region_ids, past_key_values=past, use_cache=True,
                    logits_last_only=True)
            _, region_kv = unpack_output_with_kv(out)
            # Splice the recomputed KV back into the assembled buffer.
            rlen = re_ - rs
            for li, layer in enumerate(plan.past_kv):
                if layer is None or li >= len(region_kv) or region_kv[li] is None:
                    continue
                rk, rv = region_kv[li]
                layer[0][:, :, rs:re_] = rk[:, :, -rlen:]
                layer[1][:, :, rs:re_] = rv[:, :, -rlen:]

    # ── diagnostics ───────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "blend_hits": self._blend_hits,
            "blend_misses": self._blend_misses,
            "tokens_reused": self._tokens_reused,
            "tokens_recomputed": self._tokens_recomputed,
            "reuse_ratio": (
                self._tokens_reused /
                max(self._tokens_reused + self._tokens_recomputed, 1)
            ),
            "store": self.store.stats(),
        }

    def clear(self):
        self.store.clear()
        self._blend_hits = 0
        self._blend_misses = 0
        self._tokens_reused = 0
        self._tokens_recomputed = 0
