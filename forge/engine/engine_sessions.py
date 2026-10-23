"""Session/KV-cache management mixin for ForgeEngine.

Layer map: session_cache.py = per-session KV reuse implementation;
session_manager.py = server-side history + batching (different layer).
This mixin is the engine-facing surface (begin/continue/end_session)."""
from .engine_common import *  # noqa: F403
from forge.model_loader import unpack_output_with_kv  # noqa: F401
from .prefix_cache import (
    apply_recurrent_state_prefix as _apply_recurrent_state_prefix,
    capture_recurrent_state as _capture_recurrent_state,
)
from .engine_common import (  # noqa: F401
    _CKPT_CACHE_MAX,
    _DEFAULT_CPU_MEMORY_BYTES,
    _DEFAULT_EOS_TOKEN_IDS,
    _QWEN_TOKENIZER_PATH,
    _QWEN_VOCAB,
    _checkpoint_metadata_cache,
    _checkpoint_size_cache,
    _ckpt_cache_lock,
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    _truncate_at_stop,
    logger,
)


class _SessionMixin:
    # ── Session-aware generation ────────────────────────────────────────

    def begin_session(self, session_id: str, ttl: float | None = None) -> None:
        """Start a new generation session with persistent KV cache.

        Sessions maintain KV cache state across multiple ``continue_session``
        calls, enabling O(Δt) per-turn cost instead of O(n) re-prefill.

        Args:
            session_id: unique identifier for this conversation/session.
            ttl: time-to-live in seconds. If set, the session's KV cache
                is auto-evicted after this many seconds of inactivity.
                None = no TTL (persists until end_session or LRU eviction).
        """
        return self._session_cache.begin_session(session_id, ttl)

    def continue_session(self, session_id: str, prompt: str,
                         max_new_tokens: int = 100,
                         temperature: float = 0.0,
                         top_p: float = 1.0,
                         top_k: int = 80,
                         repetition_penalty: float = 1.05,
                         stop: list[str] | None = None,
                         skip_special_tokens: bool = True) -> str:
        """Continue a session with a new prompt — reuses cached KV.

        Only the delta (new tokens not in the session's cached prefix)
        needs prefilling. Previous turns' KV is reused as-is.

        Args:
            session_id: must match a session started with ``begin_session``.
            prompt: new user input (appended to session history).
            max_new_tokens, temperature, top_p, top_k, repetition_penalty:
                same as ``generate()``.

        Returns:
            Generated text string.
        """
        self._require_awake()
        params = SamplingParams(
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, stop=stop,
            skip_special_tokens=skip_special_tokens)
        self._validate_generation_params(prompt, params)

        # The cache manager mutates session.token_ids here — keep it
        # inside _gen_lock so end_session/evict can't race the prefill.
        with self._gen_lock:
            ids, past_kv, cached_len, recurrent = \
                self._session_cache.continue_session(session_id, prompt)
            return self._continue_session_impl(
                session_id, ids, past_kv, cached_len, recurrent, params)

    def _continue_session_impl(self, session_id, ids, past_kv, cached_len,
                               recurrent, params) -> str:
        # If we have cached KV, only prefill the delta — and restore the
        # conv/recurrent boundary context first so the delta doesn't
        # zero-pad its first kernel-1 positions.
        generated_ids: list[int] = []
        if cached_len > 0 and past_kv is not None:
            _apply_recurrent_state_prefix(self.model, recurrent)
            delta_ids = ids[:, cached_len:]
            if delta_ids.shape[1] > 0:
                with torch.inference_mode():
                    out = self.model(
                        delta_ids, past_key_values=past_kv, use_cache=True)
                    logits, past_kv = unpack_output_with_kv(out)
            else:
                # No new tokens (e.g. caller re-sent the last prompt) —
                # reuse the stored next-position logits; we cannot
                # re-forward the last token without duplicating it in KV.
                session = self._session_cache.get_session(session_id)
                logits = (session.last_logits if session is not None
                          else None)
                if logits is None:
                    past_kv = None  # can't seed the decode — full prefill
            if past_kv is not None:
                # Decode from current logits
                eos_set = self._eos_token_ids()
                for _ in self._decode_tokens(
                    logits, past_kv, params, eos_set, generated_ids,
                ):
                    pass
                result = _truncate_at_stop(
                    self._safe_decode_ids(
                        generated_ids,
                        skip_special_tokens=params.skip_special_tokens),
                    params.stop)
            else:
                result, generated_ids = self._generate_with_oom_recovery(
                    lambda: self._full_generate(ids, params))
        else:
            # No cache hit — full prefill + decode
            result, generated_ids = self._generate_with_oom_recovery(
                lambda: self._full_generate(ids, params))

        # Update session KV + recurrent state — generated ids are part of
        # the cached sequence now, so the next turn's delta is the new
        # prompt only.
        last_kv = getattr(self.model, '_forge_last_kv', None)
        if last_kv is not None:
            self._session_cache.update_session_kv(
                session_id, last_kv,
                recurrent_state=_capture_recurrent_state(self.model),
                generated_ids=generated_ids,
                last_logits=getattr(self.model, '_forge_last_logits', None))

        self._record_generation(len(generated_ids))
        return result

    def pin_session(self, session_id: str, ttl: float | None = None) -> None:
        """Pin a session's KV cache to prevent eviction during tool calls.

        Args:
            session_id: the session to pin.
            ttl: optional TTL for the pin. If set, the session auto-evicts
                after this many seconds (useful for tool call timeouts).
        """
        self._session_cache.pin_session(session_id, ttl)

    def unpin_session(self, session_id: str) -> None:
        """Unpin a session (allow eviction again)."""
        self._session_cache.unpin_session(session_id)

    def end_session(self, session_id: str) -> None:
        """End a session and release its KV cache."""
        self._session_cache.end_session(session_id)

    def session_stats(self) -> dict:
        """Get session cache statistics."""
        return self._session_cache.stats()

    def _full_generate(self, ids, params) -> tuple[str, list[int]]:
        """Full prefill + decode (no session cache).

        Returns (text, generated_ids) — the caller records the token
        count once and appends the ids to the session's cached sequence.
        """
        eos_set = self._eos_token_ids()
        generated_ids: list[int] = []
        logits, past_kv = self._prefill(ids)
        for _ in self._decode_tokens(
            logits, past_kv, params, eos_set, generated_ids,
        ):
            pass
        text = _truncate_at_stop(
            self._safe_decode_ids(
                generated_ids,
                skip_special_tokens=params.skip_special_tokens),
            params.stop)
        return text, generated_ids
