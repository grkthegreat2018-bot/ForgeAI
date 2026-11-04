"""Decoding strategy backends.

Pluggable decoding strategies selectable at runtime:
  - StandardDecoding: autoregressive token-by-token (baseline)
  - ExternalDraftSpeculativeDecoding: draft model + verify ("speculative" alias)
  - MedusaDecoding: parallel prediction heads (wraps medusa.py)
  - DSparkDecoding: semi-autoregressive + confidence scheduling (wraps dspark.py)
  - MTPSelfSpecDecoding: use MTP heads from checkpoint for self-speculative decoding
  - SelfSpeculativeSparse: same model as draft+target with sparse-attention draft (R39-4)
  - SpecBoundDecoding: early-exit draft + calibrated per-layer bound (ACL'26, R60)

All implement the DecodingStrategy interface:
  generate(model, input_ids, max_new_tokens, temperature, top_p) -> output_ids
"""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from forge.engine.engine_common import (
    _DEFAULT_EOS_TOKEN_IDS,
    _apply_dry,
    _dry_penalties,
    _hash_multinomial,
)
from forge.engine.engine_common import (
    _min_k_filter as _min_k_filter_logits,
)
from forge.model_loader import unpack_output_with_kv
from forge.evolution.auto_targets import self_improve


def _sample_from_logits(
    next_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    generated_ids: list[int],
    min_p: float,
    min_k: float,
    dry_penalties: dict[int, float] | None,
    top_p_fn,
    seed: int | None = None,
    sample_pos: int | None = None,
) -> torch.Tensor:
    """Shared sampling chain: rep-penalty → temp → DRY → min-p →
    min-k → top-k → top-p → multinomial (argmax at temperature==0).

    Repetition penalty precedes temperature (HF processor ordering) and
    also applies to greedy decode — penalized logits can change argmax.

    Bit-identical to the inline chain previously duplicated across
    StandardDecoding; ``top_p_fn`` is the strategy's nucleus filter
    (StandardDecoding._top_p → research.sampling_utils.top_p_filter_logits).
    """
    if repetition_penalty != 1.0 and generated_ids:
        # raw logits may be an inference tensor (model ran under
        # inference_mode) — clone so the in-place index_copy is legal
        # outside that context.
        next_logits = next_logits.clone()
        toks = torch.tensor(
            sorted(set(generated_ids[-64:])), device=next_logits.device,
            dtype=torch.long)
        next_logits.index_copy_(
            -1, toks,
            next_logits.index_select(-1, toks) / repetition_penalty)
    next_logits = next_logits / max(temperature, 1e-5)
    if temperature == 0:
        return next_logits.argmax(-1, keepdim=True)
    if dry_penalties:
        next_logits = _apply_dry(next_logits, dry_penalties)
    if min_p > 0.0:
        probs = F.softmax(next_logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = min_p * max_prob
        next_logits = torch.where(
            probs < threshold,
            torch.full_like(next_logits, float('-inf')),
            next_logits,
        )
    if min_k > 0.0:
        next_logits = _min_k_filter_logits(next_logits, min_k)
    if top_k > 0:
        indices_to_remove = next_logits < torch.topk(
            next_logits, top_k)[0][..., -1, None]
        next_logits.masked_fill_(indices_to_remove, float('-inf'))
    if top_p < 1.0:
        next_logits = top_p_fn(next_logits, top_p)
    probs = F.softmax(next_logits, dim=-1)
    if seed is not None:
        # DeepInfer seq_seed path: the draw is a SplitMix64 hash of
        # (seed, absolute position) inverted through the CDF — a pure
        # function of the filtered distribution, reproducible across
        # runs/chunkings without RNG-order bookkeeping.
        pos = torch.tensor(
            [0 if sample_pos is None else sample_pos],
            dtype=torch.int64, device=probs.device)
        return _hash_multinomial(probs, seed, pos)
    return torch.multinomial(probs, num_samples=1)


class DecodingStrategy(ABC):
    """Base interface for decoding strategies."""

    @abstractmethod
    def generate(self, model, input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 top_k: int = 80,
                 repetition_penalty: float = 1.05,
                 **kwargs) -> torch.Tensor:
        """Generate tokens. Returns full sequence [1, prompt_len + gen_len].

        top_k and repetition_penalty are LFM2.5-recommended defaults; they are
        only applied when temperature > 0 (ignored for greedy decoding).
        Subclasses may accept additional kwargs (min_p, min_k, etc.) — the base
        signature accepts **kwargs so the engine can uniformly pass all sampling
        params without TypeError on strategies that don't use them.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


def _evolve_inputs_clone_state_tree():
    """Representative cases for _clone_state_tree: nested past_kv trees
    mixing (k,v) tuples, SSM/conv dicts, Nones and scalars — the shapes a
    Jamba-style hybrid cache actually carries per decode step."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)

    def t(*shape):
        return torch.randn(*shape, generator=g).to(dev)

    cases = []
    for layers in (8, 32):
        tree = []
        for _ in range(layers):
            tree.append((t(1, 8, 64, 64), t(1, 8, 64, 64)))
            tree.append({"conv_state": t(1, 256, 3), "ssm": t(1, 64)})
        tree += [None, "meta", 42, [t(2, 4), {"k": t(8)}]]
        cases.append((tree,))
    return cases


@self_improve(gen_input=_evolve_inputs_clone_state_tree, evals=80,
              tags=("decoding", "spec-decode", "state", "hot-path"),
              prompt=("Faster _clone_state_tree: deep-clone tensor leaves "
                      "in a nested tuple/list/dict tree. Ideas: dispatch "
                      "on exact type, avoid per-node isinstance chains, "
                      "iterative traversal, torch._C fast paths. Output "
                      "must equal the seed: same structure, every tensor "
                      "an equal-valued clone."))
def _clone_state_tree(obj):
    """Deep-clone every tensor leaf of a nested cache structure.

    torch.compile(mode='reduce-overhead') runs under cudagraph trees:
    outputs of compiled run N live in a shared pool that run N+1
    overwrites. The decode loop feeds past_kv back as inputs to the next
    compiled call, which trips the 'overwritten by a subsequent run'
    guard. Cloning in the same step — before the next compiled call —
    produces stable tensors. KV is ~12-57KB/token here, negligible vs a
    forward pass."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone_state_tree(o) for o in obj)
    if isinstance(obj, list):
        return [_clone_state_tree(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _clone_state_tree(v) for k, v in obj.items()}
    return obj


def _model_is_compiled(model) -> bool:
    """True when the model is a dynamo-compiled callable whose outputs may
    live in cudagraph-managed memory."""
    if hasattr(model, '_orig_mod'):  # torch.compile(model) -> OptimizedModule
        return True
    fwd = getattr(model, 'forward', None)
    return fwd is not None and hasattr(fwd, '_torchdynamo_orig_callable')


def _evolve_inputs_ngram_lookup():
    """Representative cases for _ngram_lookup: token sequences with a
    planted earlier copy of the tail n-gram ~60% of the time — the
    lookup-hot distribution the spec-decode draft path sees."""
    import random
    rng = random.Random(0)
    cases = []
    for _ in range(240):
        n = rng.randint(8, 512)
        seq = [rng.randint(0, 5000) for _ in range(n)]
        if rng.random() < 0.6 and n > 6:
            k = rng.randint(2, 4)
            src = rng.randint(0, n - k - 1)
            seq[-k:] = seq[src:src + k]
        cases.append((seq, rng.choice((2, 3, 4, 5)),
                      rng.choice((4, 8, 16))))
    cases += [([1, 2], 3, 4), ([], 3, 4), ([7, 7, 7, 7, 7], 2, 4)]
    return cases


@self_improve(gen_input=_evolve_inputs_ngram_lookup, evals=80,
              tags=("decoding", "spec-decode", "ngram", "hot-path"),
              prompt=("Faster _ngram_lookup(seq, ngram_size, "
                      "draft_length): find the rightmost earlier copy of "
                      "the tail n-gram and return up to draft_length "
                      "tokens after it. Ideas: dict of ngram->last_pos, "
                      "rolling hash, bytes/array compare, early exit. "
                      "Same return value for every input."))
def _ngram_lookup(seq: list[int], ngram_size: int,
                  draft_length: int) -> list[int]:
    """Find draft tokens by n-gram matching against the sequence."""
    if len(seq) < ngram_size + 1:
        return []
    ngram = seq[-ngram_size:]
    # Search for this ngram earlier in the sequence
    for i in range(len(seq) - ngram_size - 1, ngram_size - 2, -1):
        if seq[i:i + ngram_size] == ngram:
            # Return up to draft_length tokens after the match
            start = i + ngram_size
            end = min(start + draft_length, len(seq))
            return seq[start:end]
    return []


def _evolve_inputs_jsd():
    """Representative cases for _jsd: softmax probability rows over
    vocab-scale widths (DoLa contrasts final vs early-layer rows)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    cases = []
    for v in (1024, 65536, 128256):
        p = torch.softmax(torch.randn(1, v, generator=g).to(dev) * 4, -1)
        q = torch.softmax(torch.randn(1, v, generator=g).to(dev) * 4, -1)
        cases.append((p, q))
        cases.append((p, p))                    # identical rows
    return cases


@self_improve(gen_input=_evolve_inputs_jsd, evals=80,
              tags=("decoding", "dola", "math", "hot-path"),
              prompt=("Faster _jsd(p, q): Jensen-Shannon divergence "
                      "between two probability rows -> float. Seed makes "
                      "~6 full-vocab temporaries. Ideas: fuse via "
                      "log-add operations, one pass, avoid intermediate "
                      "m tensor, half the clamps. Must return the same "
                      "float within 1e-5."))
def _jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence between two probability rows."""
    m = 0.5 * (p + q)
    log_m = m.clamp_min(1e-12).log()
    kl_pm = (p * (p.clamp_min(1e-12).log() - log_m)).sum(-1)
    kl_qm = (q * (q.clamp_min(1e-12).log() - log_m)).sum(-1)
    return float((0.5 * (kl_pm + kl_qm)).mean().item())


class StandardDecoding(DecodingStrategy):
    """Standard autoregressive decoding with KV cache."""

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 min_p: float = 0.0, min_k: float = 0.0,
                 dry_multiplier: float = 0.0, dry_base: float = 1.75,
                 dry_allowed_length: int = 2, dry_penalty_last_n: int = 512,
                 stop=None, tokenizer=None,
                 kv_compact=None, suffix_mem=None,
                 cancel_check=None, seed: int | None = None,
                 **_ignored):
        ids = input_ids.clone()
        # EOS detection: check model attr, config, then known defaults
        eos = getattr(model, "eos_token_id", None)
        if eos is None:
            cfg = getattr(model, "config", None)
            eos = getattr(cfg, "eos_token_id", None) if cfg else None
        # LFM2.5 <|im_end|>=7, Qwen2.5 151643/151645,
        # Jamba <|endoftext|>=2, <|im_end|>=519
        eos_set = set(_DEFAULT_EOS_TOKEN_IDS)
        if isinstance(eos, (list, tuple, set, frozenset)):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)
        # Track generated token ids for repetition penalty + degeneration.
        # DRY operates on prompt + generated context (llama.cpp semantics).
        plen = input_ids.shape[1]
        prompt_ids = input_ids[0].tolist()
        generated_ids: list[int] = []
        # Collect generated token tensors for a single final cat (O(n) vs O(n²))
        _generated_tokens: list[torch.Tensor] = []
        # Degeneration guard: stop if same token repeats too many times.
        MAX_REPEAT = 8  # same token 8x in a row = degenerate

        # R65: reset + ingest the prompt so the suffix sidecar's lift sees
        # the whole context stream (its tail is the single source of truth).
        if suffix_mem is not None:
            suffix_mem.reset_state()
            suffix_mem.feed(prompt_ids)

        # Incremental stop-string matching inside the decode loop —
        # the engine previously truncated the decoded text post-hoc, so a
        # stop string never actually STOPPED generation (wasted tokens,
        # and callers got full-length output with the tail cut off).
        from forge.engine.engine_common import _StopMatcher
        stop_matcher = (_StopMatcher(tokenizer, stop)
                        if stop and tokenizer is not None else None)

        # Prefill — model returns (logits, loss, presents) when use_cache=True.
        # logits_last_only: only the last position is ever read — materializing
        # [B,T,V] logits costs ~512MB transient at 4k tokens / V=128k.
        # Static-KV decode graph path: prefill eager into the persistent
        # preallocated cache, then replay the captured per-token graph.
        # Only preferred over compiled functional decode when the model is
        # NOT compiled — measured on RTX 5070 / proposer_mla_moe:
        #   eager-KV  ~39 tok/s | static-graph ~76 | compiled ~107.
        # A compiled decode-step capture would beat both, but the fused
        # mask+cumsum+SDPA kernel fails Triton codegen on this box.
        bcg = getattr(model, '_bcg', None)
        state = getattr(model, '_decode_state', None)
        if (bcg is not None and state is not None
                and (state.get('compiled_step')
                     or not _model_is_compiled(model))
                and any(k in bcg._decode_graphs
                        for k in (f"kv{b}" for b in state['buckets']))
                and input_ids.shape[0] == 1
                and input_ids.shape[1] + max_new_tokens
                <= state['cache'].max_seq_len):
            try:
                return self._generate_static(
                    model, input_ids, max_new_tokens, eos_set,
                    temperature, top_p, top_k, repetition_penalty,
                    min_p, min_k, dry_multiplier, dry_base,
                    dry_allowed_length, dry_penalty_last_n, prompt_ids,
                    suffix_mem, stop_matcher, cancel_check, seed=seed)
            except Exception:
                import logging
                logging.getLogger(__name__).warning(
                    "static-KV graph decode failed — falling back to "
                    "functional path", exc_info=True)
                ids = input_ids.clone()  # _generate_static may have mutated

        needs_kv_clone = _model_is_compiled(model)
        with torch.inference_mode():
            out = model(ids, use_cache=True, logits_last_only=True)
            logits, past_kv = unpack_output_with_kv(out)
            if needs_kv_clone:
                logits = logits.clone()
                past_kv = _clone_state_tree(past_kv)
            # R65: a prompt that already overflows the KV budget is
            # compacted before decode (the strategy no-ops when T fits).
            if kv_compact is not None:
                past_kv = kv_compact(past_kv, force=True)

        # Deferred readback (plan 4.6 + DeepInfer zero-overhead
        # scheduling): the next forward launches BEFORE the .item() sync,
        # so CPU bookkeeping overlaps GPU work — the DeepInfer pipeline
        # shape "launch_forward(N+1) → post-process(N)" on one request.
        # Hybrid models snapshot their mutable recurrent buffers first:
        # on a break the speculative step is discarded and the buffers
        # rolled back (clone cost ≪ the per-step sync it removes — for
        # LFM3's 13 conv layers ≈ 320KB vs ~0.5-2ms of CPU latency).
        # Checked AFTER prefill so lazily-created state attrs exist.
        # (_v0 capture fires only on the non-cached prefill; _conv_state_
        # reset/prefix are one-shot flags, not per-token buffers.)
        _defer_snap = any(
            callable(getattr(h, "snapshot_state", None))
            or any(getattr(h, a, None) is not None
                   for a in _MUTABLE_RECURRENT_ATTRS)
            for h in _spec_state_holders(model))

        # Prompt-boundary KV for the prefix cache — captured before decode
        # mutates the state (end-of-generation KV must not be stored under
        # a prompt-only key).
        model._forge_prefill_kv = past_kv

        for _ in range(max_new_tokens):
            # Cooperative abort (POST /v1/abort): keep the partial text.
            if cancel_check is not None and cancel_check():
                break
            # DRY penalty: suppress tokens that continue a repeated
            # n-gram suffix anywhere in prompt+generated context.
            dry_pen = (
                _dry_penalties(
                    prompt_ids + generated_ids,
                    dry_penalty_last_n,
                    dry_allowed_length,
                    dry_multiplier,
                    dry_base,
                )
                if dry_multiplier > 0.0 else None
            )
            # R65: exact-suffix lift — verbatim recall of anything the
            # stream has seen, including content the KV window evicted.
            last_logits = logits[:, -1, :]
            if suffix_mem is not None:
                last_logits = last_logits + suffix_mem.lift_last(
                    device=last_logits.device, dtype=last_logits.dtype)
            next_token = _sample_from_logits(
                last_logits,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
                min_p=min_p, min_k=min_k,
                dry_penalties=dry_pen,
                top_p_fn=self._top_p,
                seed=seed,
                sample_pos=plen + len(generated_ids))

            # Snapshot mutable recurrent state BEFORE the speculative
            # launch — any break below discards the step and restores.
            _snap = _spec_snapshot_recurrent(model) if _defer_snap else None
            with torch.inference_mode():
                out = model(next_token, past_key_values=past_kv,
                            use_cache=True)
                _nl, _nk = unpack_output_with_kv(out)
                if needs_kv_clone:
                    _nl = _nl.clone()
                    _nk = _clone_state_tree(_nk)
                if kv_compact is not None:
                    _nk = kv_compact(_nk)
                _next = (_nl, _nk)

            # Single GPU->CPU sync per token: .item() brings the id over,
            # then the EOS check runs on the CPU scalar (no second sync).
            # The D2H wait overlaps the speculative forward already in
            # flight — DeepInfer's post-process(N) under forward(N+1).
            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if suffix_mem is not None:
                suffix_mem.feed_one(tok_id)
            if tok_id in eos_set:
                _defer_restore_recurrent(_snap)
                break

            # Degeneration guard: detect repetitive garbage and stop early.
            # Critical for base models that haven't learned to emit EOS.
            _stop = (
                (len(generated_ids) >= MAX_REPEAT
                 and all(g == tok_id for g in generated_ids[-MAX_REPEAT:]))
                or (len(generated_ids) >= 20
                    and len(set(generated_ids[-20:])) / 20 < 0.4))
            if _stop:
                _defer_restore_recurrent(_snap)
                break

            # Collect tokens for final cat (avoids O(n²) torch.cat per step)
            _generated_tokens.append(next_token)

            # Stop-string match runs AFTER the append: the stop-completing
            # token must land in output_ids so the caller's post-hoc find()
            # can trim the full stop text (earlier chunks of the match were
            # already emitted in prior steps).
            if stop_matcher is not None and stop_matcher.push(tok_id):
                _defer_restore_recurrent(_snap)
                break
            logits, past_kv = _next

        # Expose final KV cache state + next-position logits for the
        # _finish_to_stop fast path (the last emitted token is already
        # in past_kv — its logits let us continue without re-feeding it).
        model._forge_last_kv = past_kv
        model._forge_last_logits = logits
        # Single cat at the end instead of per-step (O(n) vs O(n²))
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
        return ids

    def _generate_static(self, model, input_ids, max_new_tokens, eos_set,
                         temperature, top_p, top_k, repetition_penalty,
                         min_p, min_k, dry_multiplier, dry_base,
                         dry_allowed_length, dry_penalty_last_n, prompt_ids,
                         suffix_mem, stop_matcher, cancel_check=None,
                         seed: int | None = None):
        """Decode via a replayed CUDA graph over static KV.

        Prefill runs eagerly into the persistent PreAllocatedKVCache
        (pos_t tensor mode); each token then updates the three device
        buffers the graph reads — mask_buf (attention_mask → positions
        AND future-slot masking), pos_t (append index), input_ids — and
        replays. Sampling/EOS/degen logic stays eager per token.
        """
        bcg = model._bcg
        state = model._decode_state
        cache, mask_buf = state['cache'], state['mask']
        # Prefill on the UNWRAPPED model — the graph was captured on it, and
        # routing a new (mask+preallocated) signature through torch.compile
        # invites an inductor recompile (Triton cache race on this box).
        fwd = getattr(model, '_orig_mod', model)
        ids = input_ids.clone()
        plen = ids.shape[1]
        generated_ids: list[int] = []
        _generated_tokens: list[torch.Tensor] = []
        MAX_REPEAT = 8

        cache.reset()
        mask_buf.zero_()
        with torch.inference_mode():
            prompt_mask = torch.ones(1, plen, dtype=torch.long,
                                     device=ids.device)
            out = fwd(ids, attention_mask=prompt_mask,
                      preallocated_cache=cache, use_cache=True,
                      logits_last_only=True)
            logits, _ = unpack_output_with_kv(out)
        cache.position = plen
        mask_buf[:, :plen] = 1
        model._forge_prefill_kv = None

        # Per-KV-length-bucket graphs: pick the smallest captured bucket
        # covering pos+1 visible slots; each graph reads a fixed [:b]
        # view of the SAME cache storage, so switching is free.
        buckets = state.get('buckets') or [cache.max_seq_len]
        graphs = bcg._decode_graphs
        buffers = bcg._static_buffers

        def _step_bufs(need):
            for b in buckets:
                if b >= need:
                    tag = f"kv{b}"
                    if tag in graphs:
                        return graphs[tag], buffers[tag]
            return None

        # Deferred readback (plan 4.6 + DeepInfer zero-overhead
        # scheduling): launch this token's replay BEFORE .item() so the
        # D2H overlaps the replay.  A break rewinds position/mask — and
        # for hybrids (conv/SSM/Titan banks) also restores a per-step
        # recurrent snapshot taken before the replay.  Dict-carried
        # presents (mamba) are functional — the speculative replay's new
        # entries are simply discarded, no rollback needed.
        _defer_snap = any(
            callable(getattr(h, "snapshot_state", None))
            or any(getattr(h, a, None) is not None
                   for a in _MUTABLE_RECURRENT_ATTRS)
            for h in _spec_state_holders(model))

        for _ in range(max_new_tokens):
            if cancel_check is not None and cancel_check():
                break
            dry_pen = (
                _dry_penalties(
                    prompt_ids + generated_ids,
                    dry_penalty_last_n,
                    dry_allowed_length,
                    dry_multiplier,
                    dry_base,
                )
                if dry_multiplier > 0.0 else None
            )
            last_logits = logits[:, -1, :]
            if suffix_mem is not None:
                last_logits = last_logits + suffix_mem.lift_last(
                    device=last_logits.device, dtype=last_logits.dtype)
            next_token = _sample_from_logits(
                last_logits,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
                min_p=min_p, min_k=min_k,
                dry_penalties=dry_pen,
                top_p_fn=self._top_p,
                seed=seed,
                sample_pos=plen + len(generated_ids))

            _replayed = None
            _pos = cache.position
            _step = (_step_bufs(_pos + 1)
                     if _pos < cache.max_seq_len
                     else None)
            _snap = None
            if _step is not None:
                _snap = (_spec_snapshot_recurrent(model)
                         if _defer_snap else None)
                _prev_logits = logits.clone()  # graph overwrites the buf
                graph, bufs = _step
                mask_buf[:, _pos] = 1
                cache.pos_t.fill_(_pos)
                bufs['input_ids'].copy_(next_token)
                with torch.inference_mode():
                    graph.replay()
                    _new_logits, _ = unpack_output_with_kv(bufs['output'])
                cache.position += 1
                _replayed = (_new_logits, _pos, _prev_logits)

            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if suffix_mem is not None:
                suffix_mem.feed_one(tok_id)
            _stop = tok_id in eos_set
            if not _stop and len(generated_ids) >= MAX_REPEAT and all(g == tok_id for g in generated_ids[-MAX_REPEAT:]):
                _stop = True
            if not _stop and len(generated_ids) >= 20:
                recent = generated_ids[-20:]
                if len(set(recent)) / len(recent) < 0.4:
                    _stop = True
            if not _stop:
                _generated_tokens.append(next_token)
                if stop_matcher is not None and stop_matcher.push(tok_id):
                    _stop = True
            if _stop:
                if _replayed is not None:
                    _, _p, logits = _replayed
                    cache.position = _p
                    mask_buf[:, _p] = 0
                _defer_restore_recurrent(_snap)
                break
            if _replayed is not None:
                logits = _replayed[0]
                continue

            # Drive static state and replay the captured step.
            pos = cache.position
            if pos >= cache.max_seq_len:
                break
            step = _step_bufs(pos + 1)
            if step is None:
                break
            graph, bufs = step
            mask_buf[:, pos] = 1
            cache.pos_t.fill_(pos)
            bufs['input_ids'].copy_(next_token)
            with torch.inference_mode():
                graph.replay()
                logits, _ = unpack_output_with_kv(bufs['output'])
            cache.position += 1

        # _forge_last_logits is consumed by the _finish_to_stop fast path —
        # clone: the raw tensor is a graph buffer overwritten on next replay.
        model._forge_last_kv = None
        model._forge_last_logits = logits.clone()
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
        return ids

    def _top_p(self, logits, top_p):
        from research.sampling_utils import top_p_filter_logits
        return top_p_filter_logits(logits, top_p)


class NGramSpeculativeDecoding(DecodingStrategy):
    """N-gram speculative decoding (prompt-lookup).

    Uses n-gram matching against the prompt to generate draft tokens —
    no separate draft model needed. Works well for code completion and
    RAG where output often repeats prompt substrings.

    Based on vLLM's prompt_lookup_speculative_decoding.
    """

    def __init__(self, ngram_size: int = 3, draft_length: int = 4):
        self.ngram_size = ngram_size
        self.draft_length = draft_length

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        device = input_ids.device
        generated = 0
        eos = getattr(model, "eos_token_id", None)
        eos_set = set(_DEFAULT_EOS_TOKEN_IDS)
        if isinstance(eos, (list, tuple, set, frozenset)):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)

        with torch.inference_mode():
            # Prefill
            out = model(ids, use_cache=True, logits_last_only=True)
            logits, past_kv = unpack_output_with_kv(out)
        model._forge_prefill_kv = past_kv

        while generated < max_new_tokens:
            # Generate main token
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                main_token = next_logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                main_token = torch.multinomial(probs.view(-1), 1).unsqueeze(0)
            if int(main_token.item()) in eos_set:
                ids = torch.cat([ids, main_token], dim=-1)
                break

            # N-gram lookup: find draft tokens by matching last ngram_size tokens
            draft_tokens = self._ngram_lookup(ids, int(main_token.item()))

            if len(draft_tokens) > 0:
                # Verify draft tokens in one forward pass
                draft_tensor = torch.tensor(draft_tokens, device=device).unsqueeze(0)
                verify_ids = torch.cat([main_token, draft_tensor], dim=-1)
                # Snapshot before the T>1 verify advances live conv/SSM
                # state past the committed boundary.
                snap = _spec_snapshot_recurrent(model)
                past_len = _spec_past_len(past_kv)
                if verify_ids.shape[1] > 1:
                    _spec_install_conv_prefix(snap)
                with torch.inference_mode():
                    out = model(verify_ids, past_key_values=past_kv, use_cache=True)
                    verify_logits, new_past_kv = unpack_output_with_kv(out)

                # Accept tokens that match
                accepted = 0
                for i, draft_tok in enumerate(draft_tokens):
                    pred_tok = verify_logits[0, i, :].argmax()
                    if pred_tok.item() == draft_tok:
                        accepted += 1
                    else:
                        break

                # KV rollback: the verify pass appended positions for the
                # WHOLE draft suffix — slice back to the committed prefix
                # (main_token + accepted drafts).  The old code kept the
                # full suffix: attention saw rejected drafts as context.
                n_draft = len(draft_tokens)
                old_past_kv = past_kv
                past_kv = _spec_slice_kv(
                    new_past_kv, past_len + 1 + accepted)
                # Recurrent repair on rejection: restore + replay kept span.
                if accepted < n_draft and (
                        _spec_has_recurrent_state(snap)
                        or _spec_has_recurrent_presents(new_past_kv)):
                    rep_kv = _spec_repair_recurrent(
                        model, snap, old_past_kv, verify_ids,
                        1 + accepted)
                    if rep_kv is not None:
                        past_kv = rep_kv

                # Append accepted + 1 (the token predicted at the mismatch)
                new_tokens = [main_token.squeeze().item()] + draft_tokens[:accepted]
                if accepted < n_draft:
                    # Mismatch — resample the corrected token and FORWARD
                    # it: its position's KV was computed for the rejected
                    # draft, and the next round's logits must come from
                    # the true committed context (the old code reused the
                    # stale verify row conditioned on the rejected token).
                    resample = verify_logits[
                        :, accepted:accepted + 1, :].argmax(-1)
                    new_tokens.append(int(resample.item()))
                    with torch.inference_mode():
                        out = model(
                            resample.to(device).long(),
                            past_key_values=past_kv, use_cache=True)
                        logits, past_kv = unpack_output_with_kv(out)
                else:
                    # Full accept — keep_len already covers the whole
                    # verify window; prediction after the last draft.
                    logits = verify_logits[:, -1:, :]

                stop = False
                for tok in new_tokens:
                    ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=-1)
                    generated += 1
                    if tok in eos_set or generated >= max_new_tokens:
                        stop = True
                        break
                if stop:
                    break
            else:
                # No draft — standard step
                ids = torch.cat([ids, main_token], dim=-1)
                generated += 1
                with torch.inference_mode():
                    out = model(main_token, past_key_values=past_kv, use_cache=True)
                    logits, past_kv = unpack_output_with_kv(out)

        return ids

    def _ngram_lookup(self, ids: torch.Tensor, last_token) -> list[int]:
        """Find draft tokens by n-gram matching against the sequence."""
        # last_token may arrive as a 0-dim tensor — coerce to a real int
        # (a tensor in the list only compared equal by accident).
        seq = ids[0].tolist() + [int(last_token)]
        return _ngram_lookup(seq, self.ngram_size, self.draft_length)


class ExternalDraftSpeculativeDecoding(DecodingStrategy):
    """Speculative decoding with an external draft model.

    The draft model is a smaller/faster model that proposes tokens.
    The main model verifies them in a single forward pass.
    """

    def __init__(self, draft_model, draft_length: int = 4,
                 confidence_tau: float = 0.0):
        self.draft_model = draft_model
        self.draft_length = draft_length
        # Acceptance telemetry: (proposed, accepted) across all rounds —
        # the honest measure of whether a draft source pays for itself.
        self.spec_stats = {"proposed": 0, "accepted": 0, "rounds": 0}
        # DeepInfer confidence-scheduled verification, bs=1 form: the
        # draft's own softmax probability proxies the acceptance rate;
        # the cumprod prefix-survival falling below ``confidence_tau``
        # ends the confident draft span — positions past it are neither
        # produced nor verified (DeepSeek trims the verify subset at a
        # global budget; on a single request a threshold prefix is the
        # same cut and also saves the doomed draft forwards).  0 =
        # classic fixed-length verify.
        self.confidence_tau = confidence_tau

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 suffix_mem=None, **kwargs):
        ids = input_ids.clone()
        device = input_ids.device
        generated = 0

        # R65: when the draft IS the suffix sidecar (SuffixDraftModel),
        # the sidecar's tail must see the prompt before its first
        # proposal — its _tail is the committed-context source of truth.
        if suffix_mem is not None:
            suffix_mem.reset_state()
            suffix_mem.feed(ids[0].tolist())

        with torch.inference_mode():
            # Prefill both models
            out = model(ids, use_cache=True, logits_last_only=True)
            logits, past_kv = unpack_output_with_kv(out)
            draft_out = self.draft_model(ids, use_cache=True)
            draft_logits, draft_past_kv = unpack_output_with_kv(draft_out)
        model._forge_prefill_kv = past_kv

        while generated < max_new_tokens:
            # Draft model proposes k tokens.  Its KV absorbs all drafted
            # positions regardless of acceptance — resync it below once
            # the committed span is known.
            draft_past_len = _spec_past_len(draft_past_kv)
            draft_snap = _spec_snapshot_recurrent(self.draft_model)
            draft_tokens = []
            cur_logits = draft_logits[:, -1, :]
            survival = 1.0
            for _ in range(self.draft_length):
                if temperature == 0:
                    tok = cur_logits.argmax(-1, keepdim=True)
                    conf = F.softmax(cur_logits, dim=-1).gather(
                        -1, tok) if self.confidence_tau > 0 else None
                else:
                    probs = F.softmax(cur_logits / max(temperature, 1e-5), dim=-1)
                    tok = torch.multinomial(probs.view(-1), 1).unsqueeze(0)
                    conf = probs.view(-1).gather(
                        0, tok.view(-1)) if self.confidence_tau > 0 else None
                if conf is not None:
                    # One sync for (token, draft-prob): cumprod prefix
                    # survival ends the round early below the tau floor.
                    tok_p = torch.stack(
                        [tok.float().view(-1)[0], conf.view(-1)[0]]).tolist()
                    survival *= float(tok_p[1])
                    if survival < self.confidence_tau:
                        break
                    draft_tokens.append(int(tok_p[0]))
                else:
                    draft_tokens.append(tok.squeeze().item())
                with torch.inference_mode():
                    draft_out = self.draft_model(tok, past_key_values=draft_past_kv, use_cache=True)
                    draft_logits, draft_past_kv = unpack_output_with_kv(draft_out)
                    cur_logits = draft_logits[:, -1, :]

            # Main model generates next token
            main_token = (logits[:, -1, :] / max(temperature, 1e-5)).argmax(-1, keepdim=True)

            if draft_tokens:
                # Verify draft in one forward pass
                draft_tensor = torch.tensor(draft_tokens, device=device).unsqueeze(0)
                verify_ids = torch.cat([main_token, draft_tensor], dim=-1)
                snap = _spec_snapshot_recurrent(model)
                past_len = _spec_past_len(past_kv)
                if verify_ids.shape[1] > 1:
                    _spec_install_conv_prefix(snap)
                with torch.inference_mode():
                    out = model(verify_ids, past_key_values=past_kv, use_cache=True)
                    verify_logits, new_past_kv = unpack_output_with_kv(out)

                # Accept matching prefix
                accepted = 0
                new_tokens = [main_token.squeeze().item()]
                self.spec_stats["proposed"] += len(draft_tokens)
                self.spec_stats["rounds"] += 1
                for i, draft_tok in enumerate(draft_tokens):
                    pred_tok = verify_logits[0, i, :].argmax()
                    if pred_tok.item() == draft_tok:
                        accepted += 1
                        new_tokens.append(draft_tok)
                    else:
                        break
                self.spec_stats["accepted"] += accepted

                # Roll the target KV back to the committed prefix
                # (main + accepted drafts) — rejected draft positions
                # must not remain as attention context.
                n_draft = len(draft_tokens)
                old_past_kv = past_kv
                past_kv = _spec_slice_kv(
                    new_past_kv, past_len + 1 + accepted)
                if accepted < n_draft and (
                        _spec_has_recurrent_state(snap)
                        or _spec_has_recurrent_presents(new_past_kv)):
                    rep_kv = _spec_repair_recurrent(
                        model, snap, old_past_kv, verify_ids,
                        1 + accepted)
                    if rep_kv is not None:
                        past_kv = rep_kv

                if accepted < n_draft:
                    # Resample the corrected token at the first mismatch
                    # and forward it — that position's KV belongs to the
                    # rejected draft, and next-round logits conditioned on
                    # it (old verify_logits[:, -1:] reuse) are stale.
                    resample = verify_logits[
                        :, accepted:accepted + 1, :].argmax(-1)
                    new_tokens.append(int(resample.item()))
                    with torch.inference_mode():
                        out = model(resample.to(device).long(),
                                    past_key_values=past_kv,
                                    use_cache=True)
                        logits, past_kv = unpack_output_with_kv(out)
                else:
                    logits = verify_logits[:, -1:, :]

                for tok in new_tokens:
                    ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=-1)
                    generated += 1
                    if suffix_mem is not None:
                        # Committed tokens feed the sidecar BEFORE the
                        # draft resync below — SuffixDraftModel predicts
                        # next-round proposals from mem._tail.
                        suffix_mem.feed_one(tok)
                    if generated >= max_new_tokens:
                        break

                # Resync the draft model: slice its KV to the pre-draft
                # boundary and feed the committed tokens (incl. main +
                # resample it never drafted) so next-round proposals come
                # from the true committed context.  Recurrent state is
                # restored on EVERY round — even a full accept missed
                # main_token's contribution to conv/SSM buffers.
                draft_past_kv = _spec_slice_kv(draft_past_kv,
                                               draft_past_len)
                if _spec_has_recurrent_state(draft_snap) or \
                        _spec_has_recurrent_presents(draft_past_kv):
                    _spec_restore_recurrent(draft_snap)
                commit_t = torch.tensor(new_tokens, device=device,
                                        dtype=torch.long).unsqueeze(0)
                if commit_t.shape[1] > 1:
                    _spec_install_conv_prefix(draft_snap)
                with torch.inference_mode():
                    draft_out = self.draft_model(
                        commit_t, past_key_values=draft_past_kv,
                        use_cache=True)
                    draft_logits, draft_past_kv = unpack_output_with_kv(
                        draft_out)
            else:
                ids = torch.cat([ids, main_token], dim=-1)
                generated += 1
                if suffix_mem is not None:
                    suffix_mem.feed_one(int(main_token.item()))
                with torch.inference_mode():
                    out = model(main_token, past_key_values=past_kv, use_cache=True)
                    logits, past_kv = unpack_output_with_kv(out)
                with torch.inference_mode():
                    draft_out = self.draft_model(
                        main_token, past_key_values=draft_past_kv,
                        use_cache=True)
                    draft_logits, draft_past_kv = unpack_output_with_kv(
                        draft_out)

        return ids

    def spec_report(self) -> dict:
        s = self.spec_stats
        return {**s,
                "accept_rate": (s["accepted"] / s["proposed"]
                                if s["proposed"] else 0.0)}


class MedusaDecoding(DecodingStrategy):
    """Medusa parallel prediction heads."""

    def __init__(self, medusa_heads):
        self.medusa = medusa_heads

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        from forge.decoding.medusa import medusa_generate
        return medusa_generate(
            model, self.medusa, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, device=str(input_ids.device),
        )


class DSparkDecoding(DecodingStrategy):
    """DSpark semi-autoregressive speculative decoding."""

    def __init__(self, dspark_head=None):
        self.dspark = dspark_head

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        from forge.decoding.dspark import dspark_generate
        return dspark_generate(
            model, self.dspark, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, device=str(input_ids.device),
        )


class Eagle3Decoding(DecodingStrategy):
    """EAGLE-3 feature-level speculative decoding.

    Uses a lightweight draft head that operates on the target model's
    multi-layer hidden states (low/mid/high) to predict draft tokens.
    No separate draft model needed — the head is attached to the target.

    Args:
        eagle_head: trained Eagle3Head module (optional, can be loaded later)
        draft_length: number of tokens to draft per iteration (default 4)
    """

    def __init__(self, eagle_head=None, draft_length: int = 4):
        self.eagle_head = eagle_head
        self.draft_length = draft_length

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        if self.eagle_head is None:
            # Fallback to standard if no head loaded
            return StandardDecoding().generate(
                model, input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                top_k=top_k, repetition_penalty=repetition_penalty,
                **kwargs,
            )
        # eagle3_generate takes a prompt string; here we work with token ids
        # so we use the internal generation loop directly.
        return _eagle_generate_from_ids(
            model, self.eagle_head, input_ids,
            max_new_tokens=max_new_tokens,
            draft_length=self.draft_length,
            temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=str(input_ids.device),
        )


class MTPSelfSpecDecoding(DecodingStrategy):
    """Self-speculative decoding using MTP heads from KeyStack checkpoint.

    The XP model checkpoint already contains MTP heads (mtp_head.heads.0-3.weight)
    initialized from the LM head. These predict tokens at positions t+1, t+2,
    t+3, t+4 in parallel. We use them as a draft model for self-speculative
    decoding — no separate draft model needed.

    Flow:
    1. Main model generates token t (with full attention)
    2. MTP heads predict draft tokens t+1..t+k (parallel, no attention)
    3. Main model verifies all k+1 tokens in ONE forward pass (EAGLE-style
       tree verification — a linear chain's tree mask is the standard causal
       mask, so no custom mask is needed). Predictions at all k positions are
       compared against the drafts in parallel; the first mismatch is found
       and the accepted prefix is kept. KV cache is rolled back by slicing
       the returned presents tensors to the accepted length.
    4. Accept matching prefix, reject and resample at first mismatch

    This is "free" speculative decoding — the MTP heads are already in the
    checkpoint and add zero inference cost for the draft phase. The batch
    verification makes it actually faster than standard decoding (one forward
    pass verifies k drafts, vs k forward passes for one-at-a-time).

    Evolution-discovered defaults: 7 draft tokens (k=7), 0.95 acceptance
    threshold, 0.10 draft_model_ratio → 5.37x speedup over standard decoding.
    These values were found by evolution search on RTX 5070 + LFM2.5-1.2B
    and represent the sweet spot between draft accuracy and verification
    overhead.
    """

    def __init__(self, k=7, mtp_module=None, acceptance_threshold=0.95,
                 draft_model_ratio=0.10, replay_ssm_cache=None):
        self.k = k
        self.mtp = mtp_module  # Optional: pre-loaded MTP module
        self.acceptance_threshold = acceptance_threshold
        self.draft_model_ratio = draft_model_ratio
        self.replay_ssm = replay_ssm_cache  # ReplaySSMCache for SSM rollback

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        eos_raw = getattr(model, "eos_token_id", None)
        eos_set = set(_DEFAULT_EOS_TOKEN_IDS)
        if isinstance(eos_raw, (list, tuple, set, frozenset)):
            eos_set.update(eos_raw)
        elif eos_raw is not None:
            eos_set.add(eos_raw)

        # Get MTP module from model if not provided
        mtp = self.mtp
        if mtp is None and hasattr(model, "mtp_head"):
            mtp = model.mtp_head
        if mtp is None and getattr(model, "mtp_module", None) is not None:
            mtp = model.mtp_module  # integrated MTPModule (V14+)
        if mtp is None:
            # No MTP heads — fall back to standard
            return StandardDecoding().generate(
                model, input_ids, max_new_tokens, temperature, top_p,
                top_k=top_k, repetition_penalty=repetition_penalty,
                **kwargs)

        # Prefill — need both KV cache (presents) and hidden states for MTP
        with torch.inference_mode():
            out = model(ids, use_cache=True, return_hidden=True,
                        logits_last_only=True)
            # (logits, loss, presents, hidden) when use_cache +
            # return_hidden — unpack defensively (EC-169).
            logits, past_kv = unpack_output_with_kv(out)
            hidden = (out[3] if isinstance(out, tuple) and len(out) > 3
                      else getattr(out, "hidden_states", None))

        generated = 0
        while generated < max_new_tokens:
            # Step 1: Main model predicts token t (using KV cache, single token)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                main_token = next_logits.argmax(-1, keepdim=True)
            else:
                main_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            if int(main_token.item()) in eos_set:
                ids = torch.cat([ids, main_token], dim=-1)
                break

            # ReplaySSM: checkpoint before drafting so we can rollback
            # the SSM state efficiently if the draft is rejected.
            ssm_checkpoint = None
            if self.replay_ssm is not None:
                ssm_checkpoint = self.replay_ssm.checkpoint()

            # Step 2: MTP heads predict draft tokens (parallel, from last hidden)
            if hidden is not None:
                with torch.inference_mode():
                    mtp_out = mtp(hidden[:, -1:, :])  # Use last hidden state
                draft_tokens = [main_token]  # [1, 1]
                if isinstance(mtp_out, (list, tuple)):
                    for head_out in mtp_out[:self.k]:
                        draft_tok = head_out.argmax(-1)[:, -1:]  # [B, 1]
                        draft_tokens.append(draft_tok)
                else:
                    draft_tokens.append(mtp_out.argmax(-1)[:, -1:])
            else:
                draft_tokens = [main_token]

            # Step 3: Batch-verify ALL draft tokens in ONE forward pass
            # (EAGLE-style tree verification for a linear chain — the tree
            # mask for a chain is just the standard causal mask, so no custom
            # mask is needed). This replaces the previous one-at-a-time loop
            # which defeated the purpose of speculative decoding (each token
            # did a full forward pass, making it slower than standard decoding).
            #
            # verify_seq = [main_token, draft_1, ..., draft_k]  (length n_draft)
            # logits[:, i, :] predicts the token after position i.
            # We compare the model's prediction at position i against draft_{i+1}
            # for all i in parallel, find the first mismatch, accept the prefix,
            # and resample at the mismatch.
            n_draft = len(draft_tokens) - 1  # number of MTP-drafted tokens
            if n_draft == 0:
                # No drafts — just accept the main token.
                ids = torch.cat([ids, main_token], dim=-1)
                generated += 1
                # logits already set for the next iteration.
                continue

            verify_seq = torch.cat(draft_tokens, dim=-1)  # [B, n_draft+1]
            # First tuple entry may be None/dict on conv-first hybrid
            # stacks — scan for a seq-dim KV entry instead of [0][0].
            past_len = _spec_past_len(past_kv)
            # Snapshot live recurrent state before the T>1 verify advances
            # it past the committed boundary (rejected drafts must not
            # leak into conv/SSM state).
            snap = _spec_snapshot_recurrent(model)
            if verify_seq.shape[1] > 1:
                _spec_install_conv_prefix(snap)
            with torch.inference_mode():
                out = model(verify_seq, past_key_values=past_kv,
                            use_cache=True, return_hidden=True)
                verify_logits, new_past_kv = unpack_output_with_kv(out)
                verify_hidden = (
                    out[3] if isinstance(out, tuple) and len(out) > 3
                    else getattr(out, "hidden_states", None))

            # Predictions for positions 0..n_draft-1 (what should follow each
            # accepted token) vs the drafted tokens at 1..n_draft.
            preds = verify_logits[:, :n_draft, :].argmax(-1)        # [B, n_draft]
            draft_stack = torch.cat(draft_tokens[1:], dim=-1)       # [B, n_draft]
            matches = (preds == draft_stack)                        # [B, n_draft]

            # First mismatch index (per batch). For B=1 this is a scalar.
            # n_accepted = number of accepted drafts (main_token + first n_accepted drafts).
            # We take the per-batch min of the first-mismatch index so the
            # accepted prefix is valid for every sequence in the batch.
            # first_false[i] = index of first False in matches[i], or n_draft if all True.
            not_match = ~matches
            # argmax returns first True index; if row is all False, argmax returns 0.
            any_mismatch = not_match.any(dim=-1)                    # [B]
            first_mismatch = torch.where(
                any_mismatch,
                not_match.float().argmax(dim=-1),
                torch.full_like(any_mismatch, n_draft, dtype=torch.long),
            )                                                       # [B]
            n_accepted = int(first_mismatch.min().item())           # accept the common prefix

            # KV cache rollback: keep only past_len + n_accepted + 1 entries
            # (the accepted prefix). The new_past_kv tensors have length
            # past_len + n_draft + 1; slice off the rejected tail.  Dict/
            # None layer entries (conv/SSM) have no seq dim — pass through.
            # Keep the pre-verify cache for the recurrent repair replay —
            # the kept span is replayed against it (passing the extended
            # cache would show attention the committed span twice).
            old_past_kv = past_kv
            keep_len = past_len + n_accepted + 1
            past_kv = _spec_slice_kv(new_past_kv, keep_len)

            # ReplaySSM: if some drafts were rejected, rollback the SSM
            # input cache to the checkpoint so the next reconstruct_state
            # replays only the accepted inputs.  When all drafts are
            # accepted, no rollback is needed.
            if (self.replay_ssm is not None and ssm_checkpoint is not None
                    and n_accepted < n_draft):
                self.replay_ssm.rollback(ssm_checkpoint + n_accepted + 1)

            # Recurrent-state repair: the verify forward advanced live
            # conv/SSM buffers over the WHOLE draft suffix — on rejection
            # restore the snapshot and replay the kept span one token at a
            # time (bit-exact incremental path), chained presents included.
            if n_accepted < n_draft and (
                    _spec_has_recurrent_state(snap)
                    or _spec_has_recurrent_presents(new_past_kv)):
                rep_kv = _spec_repair_recurrent(
                    model, snap, old_past_kv, verify_seq, n_accepted + 1)
                if rep_kv is not None:
                    past_kv = rep_kv

            # Accepted tokens: main_token + drafts 1..n_accepted.
            accepted = verify_seq[:, :n_accepted + 1]
            ids = torch.cat([ids, accepted], dim=-1)
            generated += n_accepted + 1

            # Set up logits + hidden for the NEXT iteration:
            #  - If a mismatch occurred at index j=n_accepted, the model's
            #    prediction at position j (verify_logits[:, j, :]) is the
            #    corrected next token — use it as the next main_token's logits.
            #  - If all drafts accepted, use the prediction after the last
            #    draft (verify_logits[:, n_draft, :]).
            next_idx = min(n_accepted, n_draft)
            logits = verify_logits[:, next_idx:next_idx + 1, :]    # [B, 1, V]
            hidden = verify_hidden[:, next_idx:next_idx + 1, :]    # [B, 1, d]

            if any(t in eos_set for t in accepted.flatten().tolist()):
                break

        return ids


class SelfSpeculativeSparse(DecodingStrategy):
    """Self-speculative decoding with sparse-attention draft (R39-4).

    The **same model** serves as both draft and target — no separate
    draft model is needed, so there is **zero extra memory**.

    Flow per iteration:
      1. **Draft phase**: generate ``draft_len`` tokens autoregressively
         using *sparse attention* (top-``sparse_k`` KV positions).  The
         model forward is called with ``sparse_k=<int>`` so attention
         layers can limit the KV cache to the top-k most relevant
         positions.  This is faster than full attention because the
         attention computation is O(seq_len * sparse_k) instead of
         O(seq_len²).
      2. **Verify phase**: run the model with **full attention** on the
         entire draft sequence in a single forward pass.  The full-
         attention logits at each position tell us what the *correct*
         next token is.
      3. **Accept**: compare draft tokens against the full-attention
         predictions.  Accept the longest matching prefix.  Resample at
         the first mismatch using the full-attention logits.
      4. **KV selection feedback**: the verification attention scores
         from the accepted positions are reused to select which KV
         positions to keep for the next draft round's sparse attention.

    Because verification uses full attention, the output is **lossless**
    — every accepted token is exactly what standard full-attention
    decoding would produce.  The sparse draft only affects *speed*, not
    *quality*.

    Args:
        draft_len: number of tokens to draft per iteration (default 4).
        sparse_k: number of top KV positions to attend to during the
            draft phase (default 64).  Larger = more accurate draft but
            slower; smaller = faster but lower acceptance rate.
    """

    def __init__(self, draft_len: int = 4, sparse_k: int = 64,
                 replay_ssm_cache=None):
        self.draft_len = draft_len
        self.sparse_k = sparse_k
        self.replay_ssm = replay_ssm_cache  # ReplaySSMCache for SSM rollback
        # Acceptance statistics (updated during generate).
        self.acceptance_rate: float = 0.0
        self._total_drafts: int = 0
        self._total_accepted: int = 0
        # KV selection scores from the last verification pass — reused
        # for sparse attention position selection in the next draft round.
        self._kv_scores: torch.Tensor | None = None
        # Capability probes, resolved on first generate() call:
        #   _sparse_ok — model accepts sparse_k (production
        #     ConfigurableResearchLLM does NOT — it would TypeError).
        #   _exit_layers — layer_indices early-exit draft fallback.
        self._sparse_ok: bool | None = None
        self._exit_layers: list[int] | None = None

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        eos = getattr(model, "eos_token_id", None)
        eos_set = set(_DEFAULT_EOS_TOKEN_IDS)
        if isinstance(eos, (list, tuple, set, frozenset)):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)

        # Prefill with full attention.
        with torch.inference_mode():
            out = model(ids, use_cache=True, logits_last_only=True)
            logits, past_kv = unpack_output_with_kv(out)
        model._forge_prefill_kv = past_kv

        generated = 0
        _generated_tokens: list[torch.Tensor] = []

        while generated < max_new_tokens:
            # ── Step 1: Base token from full-attention logits ──────────
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                base_token = next_logits.argmax(-1, keepdim=True)
            else:
                base_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            # ReplaySSM: checkpoint before drafting so we can rollback
            # the SSM state efficiently if the draft is rejected.
            ssm_checkpoint = None
            if self.replay_ssm is not None:
                ssm_checkpoint = self.replay_ssm.checkpoint()

            # ── Step 2: Draft phase — sparse attention ────────────────
            # Generate draft_len tokens autoregressively with sparse
            # attention (top-k KV positions).  The model forward receives
            # sparse_k so attention layers can limit the KV window.
            #
            # The production ConfigurableResearchLLM.forward has no
            # sparse_k parameter — probe once and fall back to the
            # layer_indices early-exit draft (a real cheap-draft path on
            # this model) or a plain forward if neither is supported.
            if self._sparse_ok is None:
                import inspect
                try:
                    params = inspect.signature(model.forward).parameters
                    self._sparse_ok = "sparse_k" in params
                    if "layer_indices" in params:
                        n_layers = len(getattr(model, "blocks", []) or []) \
                            or getattr(getattr(model, "config", None),
                                       "n_layers", 0) or 0
                        depth = max(1, n_layers // 2)
                        self._exit_layers = list(range(depth)) \
                            if n_layers else None
                except (TypeError, ValueError):
                    self._sparse_ok = False
            draft_tokens = [base_token]
            # Snapshot live recurrent state — T==1 draft forwards mutate
            # conv/SSM buffers even under use_cache=False; restore before
            # the verify pass so drafts never leak into committed state.
            snap = _spec_snapshot_recurrent(model)
            try:
                draft_kv = past_kv
                for _ in range(self.draft_len):
                    with torch.inference_mode():
                        if self._sparse_ok:
                            draft_out = model(
                                draft_tokens[-1],
                                past_key_values=draft_kv,
                                use_cache=True,
                                sparse_k=self.sparse_k,
                            )
                        elif self._exit_layers is not None:
                            draft_out = model(
                                draft_tokens[-1],
                                past_key_values=draft_kv,
                                use_cache=True,
                                layer_indices=self._exit_layers,
                                logits_last_only=True)
                        else:
                            draft_out = model(
                                draft_tokens[-1],
                                past_key_values=draft_kv,
                                use_cache=True)
                        draft_logits_i, draft_kv = unpack_output_with_kv(
                            draft_out)
                    draft_tok = draft_logits_i[:, -1, :].argmax(
                        -1, keepdim=True)
                    draft_tokens.append(draft_tok)
            finally:
                _spec_restore_recurrent(snap)

            n_draft = len(draft_tokens) - 1  # number of drafted tokens

            # ── Step 3: Verify phase — full attention ─────────────────
            # Run the model with full attention on the entire draft
            # sequence in ONE forward pass.  The draft KV cache (sparse)
            # is discarded — we re-process from the original full KV.
            verify_seq = torch.cat(draft_tokens, dim=-1)  # [B, n_draft+1]
            past_len = _spec_past_len(past_kv)
            if verify_seq.shape[1] > 1:
                # Committed-boundary conv context for the T>1 segment —
                # zero left-padding would corrupt the first kernel_size-1
                # positions' logits.
                _spec_install_conv_prefix(snap)
            with torch.inference_mode():
                verify_out = model(
                    verify_seq,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                verify_logits, new_past_kv = unpack_output_with_kv(verify_out)

            # ── Step 4: Compare draft vs full-attention predictions ───
            # verify_logits[:, i, :] predicts the token after position i.
            # We compare against draft_tokens[i+1] for i in [0, n_draft).
            preds = verify_logits[:, :n_draft, :].argmax(-1)     # [B, n_draft]
            draft_stack = torch.cat(draft_tokens[1:], dim=-1)    # [B, n_draft]
            matches = (preds == draft_stack)                     # [B, n_draft]

            # First mismatch index (per batch).
            not_match = ~matches
            any_mismatch = not_match.any(dim=-1)
            first_mismatch = torch.where(
                any_mismatch,
                not_match.float().argmax(dim=-1),
                torch.full_like(any_mismatch, n_draft, dtype=torch.long),
            )
            n_accepted = int(first_mismatch.min().item())

            # ── Acceptance statistics ────────────────────────────────
            self._total_drafts += n_draft
            self._total_accepted += n_accepted
            self.acceptance_rate = (
                self._total_accepted / max(self._total_drafts, 1)
            )

            # ── Step 5: KV cache rollback to accepted prefix ─────────
            # Dict/None layer entries (conv/SSM) pass through — seq-dim
            # slicing only applies to (k, v) attention presents.
            old_past_kv = past_kv
            keep_len = past_len + n_accepted + 1
            past_kv = _spec_slice_kv(new_past_kv, keep_len)

            # ReplaySSM: if some drafts were rejected, rollback the SSM
            # input cache to the checkpoint + accepted prefix so the next
            # reconstruct_state replays only the accepted inputs.
            if (self.replay_ssm is not None and ssm_checkpoint is not None
                    and n_accepted < n_draft):
                self.replay_ssm.rollback(ssm_checkpoint + n_accepted + 1)

            # Recurrent repair: restore pre-draft state + replay the kept
            # span token-by-token so conv/SSM buffers land bit-exact.
            if n_accepted < n_draft and (
                    _spec_has_recurrent_state(snap)
                    or _spec_has_recurrent_presents(new_past_kv)):
                rep_kv = _spec_repair_recurrent(
                    model, snap, old_past_kv, verify_seq, n_accepted + 1)
                if rep_kv is not None:
                    past_kv = rep_kv

            # ── Step 6: Accept tokens ────────────────────────────────
            accepted = verify_seq[:, :n_accepted + 1]
            _generated_tokens.append(accepted)
            generated += n_accepted + 1

            # Check EOS in accepted tokens.
            if any(t.item() in eos_set for t in accepted.flatten()):
                break

            # ── Set up logits for next iteration ─────────────────────
            # If mismatch at index j=n_accepted, verify_logits[:, j, :]
            # is the corrected next token's logits.
            # If all accepted, use the prediction after the last draft.
            next_idx = min(n_accepted, n_draft)
            logits = verify_logits[:, next_idx:next_idx + 1, :]

        # Single cat at the end (O(n) vs O(n²)).
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
        # Truncate to max_new_tokens (speculative acceptance can overshoot).
        max_len = input_ids.shape[1] + max_new_tokens
        if ids.shape[1] > max_len:
            ids = ids[:, :max_len]
        return ids


class DoLaDecoding(DecodingStrategy):
    """DoLa self-contrastive decoding (Chuang et al., ICLR 2024).

    Decoding by Contrasting Layers: the next-token distribution is
    ``softmax(log_softmax(final) − log_softmax(early))`` restricted to the
    final distribution's top-``candidate_top_k`` tokens. Amplifies the
    factual signal concentrated in late layers against shallow-layer
    surface priors — no auxiliary model or training required.

    On the ForgeLM hybrid, contrasting a Mamba block's hidden state
    against the final logits also isolates the attention layers' recall
    contribution (``early_layer`` accepts any block index; the block
    outputs are uniform hidden states regardless of block type).

    Args:
        early_layer: fixed early-layer block index, or ``None`` for
            per-step auto-selection (max JSD vs final distribution — the
            paper's "dynamic premature layer" choice) over
            ``early_candidates``.
        early_candidates: layer indices for auto-selection; default
            ``{n/4, n/2, 3n/4}`` clamped to valid block indices.
        candidate_top_k: restrict the contrasted distribution to the
            top-k tokens by final-layer probability (paper's candidate
            set; 0 = score the whole vocab).
    """

    def __init__(self, early_layer: int | None = None,
                 early_candidates: list[int] | None = None,
                 candidate_top_k: int = 64):
        self.early_layer = early_layer
        self.early_candidates = early_candidates
        self.candidate_top_k = candidate_top_k

    @staticmethod
    def _early_logits(model, hidden: torch.Tensor) -> torch.Tensor:
        """Project a block-output hidden state through ln_f + lm head."""
        h = hidden[:, -1, :]
        ln_f = getattr(model, "ln_f", None)
        if ln_f is not None:
            h = ln_f(h)
        return model.head(h)

    @staticmethod
    def _jsd(p: torch.Tensor, q: torch.Tensor) -> float:
        """Jensen-Shannon divergence between two probability rows."""
        return _jsd(p, q)

    @torch.no_grad()
    def _contrast_logits(self, model, final_logits, hidden_list,
                         candidates) -> torch.Tensor:
        """DoLa contrasted logits for the last position (fp32).

        Runs under no_grad (NOT inference_mode): the hidden_list inputs
        are inference tensors produced during the cached forwards — model
        params (requires_grad=True) would save them for backward outside
        a no-grad scope. Outputs stay normal tensors so the caller's
        in-place sampling ops still work.
        """
        p_final = F.softmax(final_logits.float(), dim=-1)
        if len(candidates) > 1:
            # Dynamic premature-layer selection: argmax JSD vs final.
            early_logits, best_jsd = None, -1.0
            for layer_idx in candidates:
                e_logits = self._early_logits(model, hidden_list[layer_idx])
                jsd = self._jsd(p_final, F.softmax(e_logits.float(), dim=-1))
                if jsd > best_jsd:
                    best_jsd, early_logits = jsd, e_logits
        else:
            early_logits = self._early_logits(
                model, hidden_list[candidates[0]])

        contrasted = (F.log_softmax(final_logits.float(), dim=-1)
                      - F.log_softmax(early_logits.float(), dim=-1))
        if self.candidate_top_k > 0:
            k = min(self.candidate_top_k, contrasted.shape[-1])
            top_idx = p_final.topk(k, dim=-1).indices
            mask = torch.ones_like(contrasted, dtype=torch.bool)
            mask.scatter_(-1, top_idx, False)
            contrasted = contrasted.masked_fill(mask, float("-inf"))
        return contrasted

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0, top_k=80,
                 repetition_penalty=1.05,
                 min_p: float = 0.0, min_k: float = 0.0,
                 dry_multiplier: float = 0.0, dry_base: float = 1.75,
                 dry_allowed_length: int = 2, dry_penalty_last_n: int = 512,
                 **_ignored):
        ids = input_ids.clone()
        eos = getattr(model, "eos_token_id", None)
        if eos is None:
            cfg = getattr(model, "config", None)
            eos = getattr(cfg, "eos_token_id", None) if cfg else None
        eos_set = set(_DEFAULT_EOS_TOKEN_IDS)
        if isinstance(eos, (list, tuple, set, frozenset)):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)

        prompt_ids = input_ids[0].tolist()
        generated_ids: list[int] = []
        _generated_tokens: list[torch.Tensor] = []
        MAX_REPEAT = 8

        n_layers = len(getattr(model, "blocks", [])) or getattr(
            getattr(model, "config", None), "n_layers", 0)
        if self.early_layer is not None:
            candidates = [max(0, min(int(self.early_layer), n_layers - 1))]
        else:
            if self.early_candidates is not None:
                candidates = list(self.early_candidates)
            else:
                candidates = sorted({
                    max(0, min(int(f), n_layers - 1))
                    for f in (n_layers // 4, n_layers // 2,
                              3 * n_layers // 4)})
            candidates = [c for c in candidates if 0 <= c < n_layers] or [0]

        # Prefill — return_hidden_states yields (logits, loss, kv, hs_list).
        with torch.inference_mode():
            out = model(ids, use_cache=True, return_hidden_states=True,
                        logits_last_only=True)
            logits, past_kv = unpack_output_with_kv(out)
            hidden_list = (out[3] if isinstance(out, tuple)
                           and len(out) > 3
                           else getattr(out, "hidden_states", None))
        model._forge_prefill_kv = past_kv

        for _ in range(max_new_tokens):
            next_logits = self._contrast_logits(
                model, logits[:, -1, :], hidden_list, candidates)
            dry_pen = (
                _dry_penalties(
                    prompt_ids + generated_ids, dry_penalty_last_n,
                    dry_allowed_length, dry_multiplier, dry_base)
                if dry_multiplier > 0.0 else None
            )
            next_token = _sample_from_logits(
                next_logits,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
                min_p=min_p, min_k=min_k,
                dry_penalties=dry_pen,
                top_p_fn=self._top_p)

            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if tok_id in eos_set:
                break
            if len(generated_ids) >= MAX_REPEAT and all(g == tok_id for g in generated_ids[-MAX_REPEAT:]):
                break
            if len(generated_ids) >= 20:
                recent = generated_ids[-20:]
                if len(set(recent)) / len(recent) < 0.4:
                    break

            _generated_tokens.append(next_token)
            with torch.inference_mode():
                out = model(next_token, past_key_values=past_kv,
                            use_cache=True, return_hidden_states=True)
                logits, past_kv = unpack_output_with_kv(out)
                hidden_list = (out[3] if isinstance(out, tuple)
                               and len(out) > 3
                               else getattr(out, "hidden_states", None))

        model._forge_last_kv = past_kv
        model._forge_last_logits = logits
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
        return ids

    def _top_p(self, logits, top_p):
        from research.sampling_utils import top_p_filter_logits
        return top_p_filter_logits(logits, top_p)


# Recurrent-state attributes snapshotted around SpecBound draft phases.
# use_cache=False protects the KV cache, but conv/mamba incremental paths
# still mutate their live state buffers on T==1 forwards — these are the
# buffers that need save/restore for bit-exact verification semantics.
_RECURRENT_STATE_ATTRS = (
    "_conv_state", "_conv_state_reset", "_conv_state_prefix",
    "_ssm_state", "_ssm_state_real", "_ssm_state_imag", "_state_reset",
    "_v0", "_v0_capture",
)

# The subset that advances on EVERY cached decode step — presence of any
# of these means a speculatively-launched forward can't be discarded on a
# break (no cheap rollback), so deferred readback must stay off.
_MUTABLE_RECURRENT_ATTRS = (
    "_conv_state", "_ssm_state", "_ssm_state_real", "_ssm_state_imag",
    "_memory",
)


# ── shared speculative-decode state helpers ─────────────────────────
# The snapshot/restore + KV-rollback machinery every draft/verify
# strategy needs (previously trapped inside SpecBoundDecoding — MTP,
# n-gram and self-spec paths crashed on dict presents or leaked
# over-advanced conv/SSM state on rejection).

def _spec_state_holders(model):
    """Yield every object that may carry live recurrent state."""
    yield model
    for block in getattr(model, "blocks", []) or []:
        yield block
        for attr in ("attn", "_kda", "_gdn2", "_forge_hybrid_ssm"):
            sub = getattr(block, attr, None)
            if sub is not None:
                yield sub


def _spec_snapshot_recurrent(model) -> list:
    """Clone every live recurrent-state buffer on the model."""
    snap = []
    for obj in _spec_state_holders(model):
        ss = getattr(obj, "snapshot_state", None)
        if callable(ss):
            d = ss()
            if any(v is not None for v in d.values()):
                snap.append((obj, "__snap__", d))
        for attr in _RECURRENT_STATE_ATTRS:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                snap.append((obj, attr,
                             val.clone() if torch.is_tensor(val) else val))
    return snap


def _spec_restore_recurrent(snap) -> None:
    for obj, attr, val in snap:
        if attr == "__snap__":
            if torch.is_tensor(val.get("state")):
                obj._state = val["state"].clone()
            else:
                obj._state = val.get("state")
            for ck in ("conv_q", "conv_k", "conv_v"):
                cv = getattr(obj, ck, None)
                if cv is not None:
                    s = val.get(ck)
                    cv.state = s.clone() if torch.is_tensor(s) else s
        else:
            setattr(obj, attr,
                    val.clone() if torch.is_tensor(val) else val)


def _defer_restore_recurrent(snap) -> None:
    """Undo a speculatively-launched decode step on a hybrid model.

    Same restore surface as :func:`_spec_restore_recurrent` but writes
    back IN PLACE when the live attr is a same-shaped tensor —
    ``_conv_state`` aliases the persistent ``_conv_state_buf`` storage
    and ``_state`` banks can be captured inside CUDA graphs, so a plain
    ``setattr`` would orphan the recorded storage pointer and let the
    graph keep mutating a buffer the module no longer reads.
    ``None`` snaps are a no-op (pure-attention / dict-presents models).
    """
    if not snap:
        return
    for obj, attr, val in snap:
        if attr == "__snap__":
            cur = getattr(obj, "_state", None)
            s = val.get("state")
            if torch.is_tensor(cur) and torch.is_tensor(s) \
                    and cur.shape == s.shape:
                cur.copy_(s)
            else:
                obj._state = s.clone() if torch.is_tensor(s) else s
            for ck in ("conv_q", "conv_k", "conv_v"):
                cv = getattr(obj, ck, None)
                if cv is not None:
                    sv = val.get(ck)
                    cst = getattr(cv, "state", None)
                    if torch.is_tensor(cst) and torch.is_tensor(sv) \
                            and cst.shape == sv.shape:
                        cst.copy_(sv)
                    else:
                        cv.state = sv.clone() if torch.is_tensor(sv) else sv
        else:
            cur = getattr(obj, attr, None)
            if torch.is_tensor(cur) and torch.is_tensor(val) \
                    and cur.shape == val.shape:
                cur.copy_(val)
            else:
                setattr(obj, attr,
                        val.clone() if torch.is_tensor(val) else val)


def _spec_has_recurrent_state(snap) -> bool:
    return any(
        (a == "__snap__" and any(x is not None for x in v.values()))
        or (torch.is_tensor(v) and ("_conv_state" in a or "_ssm_state" in a))
        for _, a, v in snap)


def _spec_has_recurrent_presents(past_kv) -> bool:
    """True when presents carry O(1) recurrent state (mamba-style dict
    entries) that a seq-dim slice cannot roll back."""
    if not past_kv:
        return False
    return any(isinstance(e, dict) for e in past_kv)


def _spec_install_conv_prefix(snap) -> None:
    """Install the snapshot's conv state as ``_conv_state_prefix`` so the
    NEXT forward's T>1 causal conv uses the real committed-boundary
    context instead of zero left-padding."""
    for obj, attr, val in snap:
        if attr == "_conv_state" and torch.is_tensor(val):
            obj._conv_state_prefix = val


def _spec_past_len(past_kv) -> int:
    if not past_kv:
        return 0
    for e in past_kv:
        if e is None or isinstance(e, dict):
            continue  # conv/SSM entries carry no seq-dim KV
        return e[0].shape[-2]
    return 0


def _evolve_inputs_spec_slice_kv():
    """Representative cases for _spec_slice_kv: hybrid (k,v)-tuple +
    conv-dict + None per-layer presents at verify-time rollback lens."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    cases = []
    for n_layers, seq, keep in ((16, 256, 128), (32, 512, 200),
                                (8, 64, 0)):
        past = []
        for i in range(n_layers):
            if i % 3 == 0:
                past.append(None)
            elif i % 3 == 1:
                past.append({"conv_state":
                             torch.randn(1, 64, 4, generator=g).to(dev)})
            else:
                k = torch.randn(1, 8, seq, 64, generator=g).to(dev)
                v = torch.randn(1, 8, seq, 64, generator=g).to(dev)
                past.append((k, v))
        cases.append((past, keep))
    cases.append((None, 4))
    return cases


@self_improve(gen_input=_evolve_inputs_spec_slice_kv, evals=80,
              tags=("decoding", "spec-decode", "kv-cache", "hot-path"),
              prompt=("Faster _spec_slice_kv(past_kv, keep_len): new "
                      "list with each (k,v) tuple sliced [:, :, :keep, :]"
                      "; None/dict entries pass through unchanged, None "
                      "input -> None. Ideas: list-comp prealloc, single "
                      "type dispatch, skip re-slicing views."))
def _spec_slice_kv(past_kv, keep_len: int):
    """Rollback KV presents to ``keep_len`` (None-safe; dict presents
    pass through — recurrent state is repaired by the replay below)."""
    if past_kv is None:
        return None
    out = []
    for layer_kv in past_kv:
        if layer_kv is None or isinstance(layer_kv, dict):
            out.append(layer_kv)
        else:
            k, v = layer_kv
            out.append((k[:, :, :keep_len, :], v[:, :, :keep_len, :]))
    return out


def _spec_repair_recurrent(model, snap, old_past_kv, seq, n_kept: int):
    """Restore pre-draft recurrent state, then replay ``seq[:, :n_kept]``
    one token at a time against ``old_past_kv`` (the pre-verify cache),
    chaining presents.  Returns the canonical post-repair cache.  T=1
    calls take the incremental conv/SSM path — bit-exact vs sequential
    StandardDecoding."""
    _spec_restore_recurrent(snap)
    rep_kv = old_past_kv
    for i in range(n_kept):
        with torch.inference_mode():
            rep_out = model(seq[:, i:i + 1],
                            past_key_values=rep_kv, use_cache=True)
        _, rep_kv = unpack_output_with_kv(rep_out)
    return rep_kv


class SpecBoundDecoding(DecodingStrategy):
    """SpecBound: adaptive bounded self-speculation (ACL 2026 findings).

    Self-speculative decoding where the *same model* drafts via EARLY
    EXIT.  The model forward accepts ``layer_indices`` — skipped blocks
    are a KV passthrough — so a partial forward over layers
    ``0..exit_layer`` yields a cheap approximate next-token distribution
    that proposes up to ``draft_len`` lookahead tokens.  ONE full forward
    then verifies the drafted suffix with the standard speculative
    accept/resample rule (same as SelfSpeculativeSparse), which makes the
    output **bit-identical to StandardDecoding under greedy
    (temperature=0)**: every emitted token is either the verified argmax
    of the full model or its correction at the first mismatch.

    Bounded, calibrated speculation (the paper's contribution):

    * Per-candidate exit layer keeps an EMA of its acceptance fraction.
      Each round runs on the deepest candidate whose EMA >=
      ``conf_threshold`` (default 0.6); if none qualifies, the deepest
      candidate is used (deepest = closest to the full distribution =
      highest draft quality while the EMAs re-calibrate).
    * ``draft_len`` adapts within [``min_draft``, ``max_draft``]: +1 when
      a verify accepts every drafted token, -1 on any rejection.

    State-safety notes (why greedy equivalence holds on hybrid models):

    * Draft forwards run ``use_cache=False`` with
      ``past_key_values=past_kv`` — attention layers attend over the
      committed prefix read-only (``torch.cat`` allocates; the passed
      tensors are never mutated) and no new presents are returned.
      Skipped layers contribute nothing.
    * T>1 segment forwards (multi-token draft calls + the verify pass)
      install ``_conv_state_prefix`` from the committed-boundary conv
      state so the causal conv uses the real left context instead of
      zero-padding — the same mechanism ``apply_recurrent_state_prefix``
      uses for prefix-cache hits.  Without it the first
      ``kernel_size - 1`` positions of each segment would produce logits
      that diverge from token-by-token decoding.
    * ``use_cache=False`` does NOT protect *recurrent* block state at
      T==1: conv/mamba incremental paths update ``_conv_state`` /
      ``_ssm_state`` as a side effect.  We snapshot those buffers before
      the draft loop and restore them afterwards.
    * On verify rejection (or an early stop inside the accepted span)
      the kept tokens are replayed one at a time with chained presents —
      the model's incremental path — so live conv/SSM buffers and
      mamba-style dict presents all end up bit-exact vs
      StandardDecoding's sequential decode.
    * ``replay_ssm_cache`` (optional) is honoured like the other
      speculative strategies for engines that wire ReplaySSM rollback.

    Caveat: on blocks whose T>1 continuation lacks a prefix/state
    mechanism (mamba-family internal convs zero-pad the first
    ``d_conv - 1`` segment positions; KDA/GDN2 state paths behave
    similarly — the same limitation every speculative strategy in this
    file shares), verify logits inherit that approximation; exactness
    is guaranteed on pure-attention and ``DoubleGatedConvLayer``-style
    conv hybrids, which is also what the unit tests cover.

    On models whose forward lacks ``layer_indices`` (or rejects the
    kwarg), the draft falls back to a full forward — no early-exit
    speedup, but verification still guarantees identical output.

    Args:
        exit_layer: fixed early-exit block index, or None for
            confidence-calibrated selection over ``exit_candidates``.
        exit_candidates: candidate exit indices; default
            ``{n//4, n//2, 3n//4}`` clamped to valid block indices
            (same candidate set as DoLa).
        conf_threshold: minimum per-layer acceptance EMA (default 0.6).
        draft_len: initial speculation depth (default 4).
        min_draft / max_draft: adaptive bound clamps (default 2 / 8).
        ema_alpha: EMA rate for per-layer acceptance stats (default 0.3).
        replay_ssm_cache: optional ReplaySSMCache for SSM rollback.
    """

    def __init__(self, exit_layer: int | None = None,
                 exit_candidates: list[int] | None = None,
                 conf_threshold: float = 0.6,
                 draft_len: int = 4, min_draft: int = 2, max_draft: int = 8,
                 ema_alpha: float = 0.3, replay_ssm_cache=None):
        self.exit_layer = exit_layer
        self.exit_candidates = exit_candidates
        self.conf_threshold = conf_threshold
        self.min_draft = max(1, int(min_draft))
        self.max_draft = max(self.min_draft, int(max_draft))
        self._cur_draft = max(self.min_draft,
                              min(int(draft_len), self.max_draft))
        self.ema_alpha = float(ema_alpha)
        self.replay_ssm = replay_ssm_cache
        # Per-exit-layer acceptance EMA (untested layers read as the
        # threshold — eligible until evidence says otherwise).
        self._layer_ema: dict[int, float] = {}
        self._cur_exit: int | None = None
        # Acceptance statistics (same convention as SelfSpeculativeSparse).
        self.acceptance_rate: float = 0.0
        self._total_drafts: int = 0
        self._total_accepted: int = 0
        # Set during generate: False once a forward rejects layer_indices.
        self._partial_ok = True

    # ── recurrent-state snapshot / restore ─────────────────────────
    # Thin delegates to the module-level _spec_* helpers shared by all
    # draft/verify strategies.

    @staticmethod
    def _state_holders(model):
        return _spec_state_holders(model)

    @classmethod
    def _snapshot_recurrent(cls, model) -> list:
        return _spec_snapshot_recurrent(model)

    @staticmethod
    def _restore_recurrent(snap) -> None:
        _spec_restore_recurrent(snap)

    @staticmethod
    def _has_recurrent_state(snap) -> bool:
        return _spec_has_recurrent_state(snap)

    @staticmethod
    def _has_recurrent_presents(past_kv) -> bool:
        return _spec_has_recurrent_presents(past_kv)

    @staticmethod
    def _install_conv_prefix(snap) -> None:
        _spec_install_conv_prefix(snap)

    # ── draft helpers ───────────────────────────────────────────────

    def _select_exit_layer(self, candidates: list[int]) -> int:
        """Deepest candidate whose acceptance EMA meets the threshold;
        deepest overall when none qualifies."""
        if self.exit_layer is not None:
            return candidates[0] if len(candidates) == 1 else min(
                max(int(self.exit_layer), 0), candidates[-1])
        eligible = [c for c in candidates
                    if self._layer_ema.get(c, self.conf_threshold)
                    >= self.conf_threshold]
        return max(eligible) if eligible else max(candidates)

    def _draft_logits(self, model, suffix, past_kv, active) -> torch.Tensor:
        """Last-position logits of the (partial) draft distribution."""
        if self._partial_ok and active is not None:
            try:
                out = model(suffix, past_key_values=past_kv,
                            use_cache=False, layer_indices=active,
                            logits_last_only=True)
                logits, _ = unpack_output_with_kv(out)
                return logits[:, -1, :]
            except TypeError:
                self._partial_ok = False
        try:
            out = model(suffix, past_key_values=past_kv,
                        use_cache=False, logits_last_only=True)
        except TypeError:
            out = model(suffix, past_key_values=past_kv, use_cache=False)
        logits, _ = unpack_output_with_kv(out)
        return logits[:, -1, :]

    @staticmethod
    def _past_len(past_kv) -> int:
        return _spec_past_len(past_kv)

    @staticmethod
    def _slice_kv(past_kv, keep_len: int):
        return _spec_slice_kv(past_kv, keep_len)

    def _top_p(self, logits, top_p):
        from research.sampling_utils import top_p_filter_logits
        return top_p_filter_logits(logits, top_p)

    # ── main loop ───────────────────────────────────────────────────

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0, top_k=80,
                 repetition_penalty=1.05,
                 min_p: float = 0.0, min_k: float = 0.0,
                 dry_multiplier: float = 0.0, dry_base: float = 1.75,
                 dry_allowed_length: int = 2, dry_penalty_last_n: int = 512,
                 **_ignored):
        if input_ids.shape[0] > 1:
            # The per-token emit/guard path below is B=1-only (like the
            # engine's speculative decode paths); batch inputs get the
            # equivalent standard loop.
            return StandardDecoding().generate(
                model, input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                min_p=min_p, min_k=min_k, dry_multiplier=dry_multiplier,
                dry_base=dry_base, dry_allowed_length=dry_allowed_length,
                dry_penalty_last_n=dry_penalty_last_n)
        ids = input_ids.clone()
        device = input_ids.device
        # EOS detection identical to StandardDecoding.
        eos = getattr(model, "eos_token_id", None)
        if eos is None:
            cfg = getattr(model, "config", None)
            eos = getattr(cfg, "eos_token_id", None) if cfg else None
        eos_set = set(_DEFAULT_EOS_TOKEN_IDS)
        if isinstance(eos, (list, tuple, set, frozenset)):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)

        prompt_ids = input_ids[0].tolist()
        generated_ids: list[int] = []
        _generated_tokens: list[torch.Tensor] = []
        MAX_REPEAT = 8
        generated = 0

        # Exit-layer candidates: {n//4, n//2, 3n//4} clamped (DoLa set).
        n_layers = len(getattr(model, "blocks", []) or []) or getattr(
            getattr(model, "config", None), "n_layers", 0)
        if self.exit_layer is not None:
            candidates = [max(0, min(int(self.exit_layer),
                                     max(n_layers - 1, 0)))]
        else:
            if self.exit_candidates is not None:
                candidates = sorted(set(self.exit_candidates))
            else:
                candidates = sorted({
                    max(0, min(int(f), max(n_layers - 1, 0)))
                    for f in (n_layers // 4, n_layers // 2,
                              3 * n_layers // 4)})
            candidates = [c for c in candidates
                          if 0 <= c < n_layers] or [0]
        self._cur_exit = self._select_exit_layer(candidates)

        # Model capability probe for the partial-forward draft path.
        import inspect
        try:
            self._partial_ok = "layer_indices" in inspect.signature(
                model.forward).parameters
        except (TypeError, ValueError):
            self._partial_ok = False

        # Prefill — full model, KV cache.
        with torch.inference_mode():
            out = model(ids, use_cache=True, logits_last_only=True)
            logits, past_kv = unpack_output_with_kv(out)
        model._forge_prefill_kv = past_kv

        while generated < max_new_tokens:
            # ── Base token: identical sampling chain to standard ────
            dry_pen = (
                _dry_penalties(
                    prompt_ids + generated_ids, dry_penalty_last_n,
                    dry_allowed_length, dry_multiplier, dry_base)
                if dry_multiplier > 0.0 else None)
            base_token = _sample_from_logits(
                logits[:, -1, :],
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
                min_p=min_p, min_k=min_k,
                dry_penalties=dry_pen,
                top_p_fn=self._top_p)
            base_id = base_token.item()

            # EOS on the base token: stop before drafting.  Mirrors
            # StandardDecoding exactly — the id goes into generated_ids
            # but the token tensor is NOT collected (standard breaks
            # before _generated_tokens.append), so EOS stays out of the
            # emitted output.
            if base_id in eos_set:
                generated_ids.append(base_id)
                generated += 1
                break

            # ReplaySSM: checkpoint before drafting (engine-wired path).
            ssm_checkpoint = None
            if self.replay_ssm is not None:
                ssm_checkpoint = self.replay_ssm.checkpoint()

            # ── Draft phase: early-exit proposals ────────────────────
            # Snapshot live recurrent state — conv/mamba T==1 paths
            # mutate it even under use_cache=False.
            snap = self._snapshot_recurrent(model)
            draft_tokens = [base_token]
            draft_budget = min(self._cur_draft,
                               max_new_tokens - generated - 1)
            active = (list(range(self._cur_exit + 1))
                      if self._partial_ok else None)
            try:
                for _ in range(max(0, draft_budget)):
                    suffix = torch.cat(draft_tokens, dim=-1)
                    if suffix.shape[1] > 1:
                        # Give the T>1 causal-conv the committed-boundary
                        # left context (consumed by the forward each time).
                        self._install_conv_prefix(snap)
                    with torch.inference_mode():
                        d_logits = self._draft_logits(
                            model, suffix, past_kv, active)
                    if temperature == 0:
                        d_tok = d_logits.argmax(-1, keepdim=True)
                    else:
                        d_tok = torch.multinomial(
                            F.softmax(d_logits / max(temperature, 1e-5),
                                      dim=-1), num_samples=1)
                    draft_tokens.append(d_tok)
                    if int(d_tok.item()) in eos_set:
                        break
            finally:
                self._restore_recurrent(snap)

            n_draft = len(draft_tokens) - 1

            # ── Verify phase: ONE full forward on the draft suffix ──
            verify_seq = torch.cat(draft_tokens, dim=-1)
            past_len = self._past_len(past_kv)
            if verify_seq.shape[1] > 1:
                # Same committed-boundary conv context for the verify
                # pass — without it the first kernel_size-1 positions
                # would see zero-padded conv context and could emit
                # logits that diverge from token-by-token decoding.
                self._install_conv_prefix(snap)
            with torch.inference_mode():
                verify_out = model(verify_seq, past_key_values=past_kv,
                                   use_cache=True)
                verify_logits, new_past_kv = unpack_output_with_kv(verify_out)

            # ── Accept longest matching prefix (greedy semantics) ──
            if n_draft > 0:
                preds = verify_logits[:, :n_draft, :].argmax(-1)
                draft_stack = torch.cat(draft_tokens[1:], dim=-1)
                matches = (preds == draft_stack)
                not_match = ~matches
                any_mismatch = not_match.any(dim=-1)
                first_mismatch = torch.where(
                    any_mismatch,
                    not_match.float().argmax(dim=-1),
                    torch.full_like(any_mismatch, n_draft,
                                    dtype=torch.long))
                n_accepted = int(first_mismatch.min().item())
            else:
                n_accepted = 0

            # KV rollback to the accepted prefix.  Keep the pre-verify
            # cache too: the recurrent-state repair below replays the
            # kept span against it (passing the already-extended cache
            # would make attention see the committed span twice and
            # rebuild conv/SSM state over the wrong hidden stream).
            old_past_kv = past_kv
            keep_len = past_len + n_accepted + 1
            past_kv = self._slice_kv(new_past_kv, keep_len)

            # ReplaySSM rollback for rejected tails (engine-wired path).
            if (self.replay_ssm is not None and ssm_checkpoint is not None
                    and n_accepted < n_draft):
                self.replay_ssm.rollback(ssm_checkpoint + n_accepted + 1)

            if n_draft > 0:
                # ── Acceptance stats + EMA + adaptive bound ─────────
                frac = n_accepted / n_draft
                self._total_drafts += n_draft
                self._total_accepted += n_accepted
                self.acceptance_rate = (
                    self._total_accepted / max(self._total_drafts, 1))
                old_ema = self._layer_ema.get(
                    self._cur_exit, self.conf_threshold)
                self._layer_ema[self._cur_exit] = (
                    (1 - self.ema_alpha) * old_ema
                    + self.ema_alpha * frac)
                # Adaptive bound: grow when everything was accepted.
                if n_accepted == n_draft:
                    self._cur_draft = min(self._cur_draft + 1,
                                          self.max_draft)
                else:
                    self._cur_draft = max(self._cur_draft - 1,
                                          self.min_draft)
                self._cur_exit = self._select_exit_layer(candidates)

            # ── Emit accepted tokens (same per-token guards as std) ─
            # Ordering mirrors StandardDecoding exactly: the id lands in
            # generated_ids first, and an EOS/degeneration break fires
            # BEFORE the token tensor is collected — the break-triggering
            # token is therefore excluded from the emitted output.
            n_kept = 0
            stop = False
            for tok in verify_seq[0, :n_accepted + 1].tolist():
                generated_ids.append(tok)
                generated += 1
                if tok in eos_set:
                    stop = True
                    break
                if len(generated_ids) >= MAX_REPEAT and all(g == tok
                       for g in generated_ids[-MAX_REPEAT:]):
                    stop = True
                    break
                if len(generated_ids) >= 20:
                    recent = generated_ids[-20:]
                    if len(set(recent)) / len(recent) < 0.4:
                        stop = True
                        break
                _generated_tokens.append(
                    torch.tensor([[tok]], device=device))
                n_kept += 1

            # KV over-provisions whenever we kept fewer tokens than the
            # accepted span — trim to exactly the emitted prefix.
            if n_kept < n_accepted + 1:
                past_kv = self._slice_kv(past_kv, past_len + n_kept)

            # ── Recurrent-state repair ───────────────────────────────
            # The verify forward advanced live conv/SSM state (and any
            # dict-carried recurrent presents) over the WHOLE verify_seq.
            # Whenever that consumed span differs from what we kept — a
            # rejection, or an early stop inside the accepted span —
            # restore the pre-draft snapshot and replay the kept tokens
            # ONE AT A TIME, chaining the returned presents.  T=1 calls
            # take the model's incremental path (``_incremental_conv`` /
            # per-step SSM update), which is bit-exact vs what
            # StandardDecoding would have produced — for live-attr state
            # (conv buffers) AND presents-carried state (mamba dicts)
            # alike.  rep_kv then becomes the canonical cache: its
            # recurrent entries hold the correct kept-boundary state.
            needs_repair = (
                self._has_recurrent_state(snap)
                or self._has_recurrent_presents(old_past_kv))
            if needs_repair and n_kept != n_draft + 1:
                self._restore_recurrent(snap)
                rep_kv = old_past_kv
                for i in range(n_kept):
                    with torch.inference_mode():
                        rep_out = model(verify_seq[:, i:i + 1],
                                        past_key_values=rep_kv,
                                        use_cache=True)
                    _, rep_kv = unpack_output_with_kv(rep_out)
                if rep_kv is not None:
                    past_kv = rep_kv

            if stop:
                break

            # ── Logits for the next round ───────────────────────────
            # Rejection at n_accepted: verify_logits[:, n_accepted] is
            # the corrected next-token distribution.  Full accept: the
            # prediction after the last drafted position.
            next_idx = min(n_accepted, n_draft)
            logits = verify_logits[:, next_idx:next_idx + 1, :]

        model._forge_last_kv = past_kv
        model._forge_last_logits = logits
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
        # Truncate to max_new_tokens (acceptance can overshoot).
        max_len = input_ids.shape[1] + max_new_tokens
        if ids.shape[1] > max_len:
            ids = ids[:, :max_len]
        return ids


def build_decoding(strategy: str = "standard", **kwargs) -> DecodingStrategy:
    """Factory: build decoding strategy by name.

    Strict: unknown names raise ActivationError with the valid list —
    a typo must never silently become StandardDecoding (EC-019).
    """
    strategies = {
        "standard": StandardDecoding,
        "speculative": ExternalDraftSpeculativeDecoding,
        "ngram_speculative": NGramSpeculativeDecoding,
        "external_draft_speculative": ExternalDraftSpeculativeDecoding,
        "medusa": MedusaDecoding,
        "dspark": DSparkDecoding,
        "eagle3": Eagle3Decoding,
        "mtp_selfspec": MTPSelfSpecDecoding,
        "self_speculative_sparse": SelfSpeculativeSparse,
        "specbound": SpecBoundDecoding,
        "dola": DoLaDecoding,
        "batched": None,  # set below to avoid circular import
    }
    if strategy == "batched":
        from forge.engine.batched_decoding import BatchedDecoding
        return BatchedDecoding(**kwargs)
    if strategy == "uno":
        from forge.decoding.uno import NgramProposer, UnoDecoding
        kwargs.setdefault("proposer", NgramProposer())
        return UnoDecoding(**kwargs)
    cls = strategies.get(strategy)
    if cls is None:
        from forge.engine.errors import ActivationError
        valid = sorted(set(strategies) | {"uno"})
        raise ActivationError(
            f"Unknown decoding strategy {strategy!r}. "
            f"Valid: {', '.join(valid)}")
    return cls(**kwargs)


@torch.inference_mode()
def _eagle_generate_from_ids(
    model, head, input_ids, max_new_tokens=100, draft_length=4,
    temperature=0.0, top_k=0, repetition_penalty=1.0, device="cuda",
):
    """EAGLE-3 generation from token ids (used by Eagle3Decoding strategy)."""
    from forge.decoding.eagle import extract_hidden_states

    model.eval()
    head.eval()

    extract_layers = [head.low_layer, head.mid_layer, head.high_layer]
    eos_id = getattr(model, 'eos_token_id', None) or 7

    # Prefill
    hidden_list, final_hidden, presents = extract_hidden_states(
        model, input_ids, extract_layers, use_cache=True,
    )
    fused = head.fuse_hidden_states(hidden_list)
    target_logits = model.head(final_hidden)
    last_logits = target_logits[:, -1, :]

    if temperature <= 0:
        next_token = last_logits.argmax(dim=-1, keepdim=True)
    else:
        l = last_logits / temperature
        if top_k > 0:
            idx_rm = l < torch.topk(l, top_k)[0][..., -1, None]
            l.masked_fill_(idx_rm, float('-inf'))
        probs = F.softmax(l, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    generated = [next_token.item()]
    _eagle_tokens: list[torch.Tensor] = [next_token]  # collect for final cat

    while len(generated) < max_new_tokens:
        if generated[-1] == eos_id:
            break

        # Draft tokens autoregressively
        draft_tokens = []
        cur_fused = fused[:, -1:, :]
        cur_token = next_token
        for _ in range(draft_length):
            draft_tok, _ = head.predict_next(cur_fused, cur_token, temperature=temperature, top_k=top_k)
            draft_tokens.append(draft_tok.item())
            cur_token = draft_tok

        # Verify with target
        draft_tensor = torch.tensor([draft_tokens], device=device, dtype=input_ids.dtype)
        verify_hidden, verify_final, presents = extract_hidden_states(
            model, draft_tensor, extract_layers,
            past_key_values=presents, use_cache=True,
        )
        verify_logits = model.head(verify_final)

        # Accept longest prefix
        n_accepted = 0
        for i, dt in enumerate(draft_tokens):
            target_tok = verify_logits[:, i, :].argmax(dim=-1).item()
            if target_tok == dt:
                n_accepted += 1
            else:
                break

        for i in range(n_accepted):
            generated.append(draft_tokens[i])
            if len(generated) >= max_new_tokens:
                break

        if n_accepted < draft_length:
            next_token = verify_logits[:, n_accepted, :].argmax(dim=-1, keepdim=True)
        else:
            next_token = verify_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        fused = head.fuse_hidden_states(verify_hidden)
        if next_token.item() == eos_id:
            break
        generated.append(next_token.item())
        _eagle_tokens.append(next_token)

    # Single cat at the end instead of per-step (O(n) vs O(n²))
    ids = torch.cat([input_ids] + _eagle_tokens, dim=1)
    return ids[:, input_ids.shape[1]:]
