"""Chunked / hybrid prefill strategies for long prompts.

Merged from chunked_prefill.py + hybrid_prefill.py (both <7KB, same domain).

ChunkedPrefiller: splits long prompts into chunks for interleaved decode,
preventing long prompts from blocking the decode queue.

HybridChunkedPrefiller: adaptive chunking — chunks only when decode is
active, falls back to continuous prefill when no decode requests are running.
Based on vLLM PR #26625 (Hybrid Chunked Prefill).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# ── ChunkedPrefiller ───────────────────────────────────────────────────

class ChunkedPrefiller:
    """Chunked prefill for long prompts.

    Splits a long prompt into chunks of `chunk_size` tokens and processes
    them sequentially, accumulating the KV cache. This prevents long prompts
    from blocking the decode queue.

    Args:
        model: the LLM model
        chunk_size: tokens per prefill chunk (default 512)
        device: cuda or cpu
    """

    def __init__(self, model: nn.Module, chunk_size: int = 512,
                 device: str = "cuda"):
        self.model = model
        self.chunk_size = chunk_size
        self.device = torch.device(device)

    def prefill_chunked(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = True,
        on_chunk_done: callable = None,
    ) -> tuple[torch.Tensor, list]:
        """Process a long prompt in chunks, accumulating KV cache.

        Args:
            input_ids: (1, prompt_len) token IDs
            use_cache: whether to return KV cache
            on_chunk_done: optional callback(chunk_idx, n_chunks) for
                           yielding between chunks (e.g., to run decode steps)

        Returns:
            (logits, past_kv) — logits for the last token, past_kv for decode
        """
        B, T = input_ids.shape
        assert B == 1, "ChunkedPrefiller only supports batch=1"

        chunks = []
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunks.append((start, end))

        n_chunks = len(chunks)
        past_kv = None
        logits = None

        with torch.inference_mode():
            for i, (start, end) in enumerate(chunks):
                chunk_ids = input_ids[:, start:end]

                if past_kv is not None:
                    out = self.model(
                        chunk_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                        logits_last_only=True,
                    )
                else:
                    out = self.model(
                        chunk_ids,
                        use_cache=True,
                        logits_last_only=True,
                    )

                # Unpack output
                if isinstance(out, tuple):
                    if len(out) >= 3:
                        logits = out[0]
                        past_kv = out[2] if len(out) > 2 else out[1]
                    else:
                        logits = out[0]
                        past_kv = out[1] if len(out) > 1 else None
                else:
                    logits = out

                # Callback between chunks (yield to decode queue)
                if on_chunk_done is not None and i < n_chunks - 1:
                    on_chunk_done(i + 1, n_chunks)

        return logits, past_kv

    def prefill_interleaved(
        self,
        input_ids: torch.Tensor,
        decode_fn: callable,
        decode_state: dict,
    ) -> tuple[torch.Tensor, list]:
        """Interleave chunked prefill with ongoing decode steps.

        Between each prefill chunk, calls decode_fn(decode_state) to run
        one decode step for an ongoing generation. This keeps decode latency
        low while making progress on the prefill.

        Args:
            input_ids: the long prompt to prefill
            decode_fn: function(state) -> state, runs one decode step
            decode_state: mutable state for the ongoing decode

        Returns:
            (logits, past_kv) for the prefilled prompt
        """
        def on_chunk_done(chunk_idx, n_chunks):
            # Run one decode step between prefill chunks
            decode_state.update(decode_fn(decode_state))

        return self.prefill_chunked(
            input_ids, use_cache=True, on_chunk_done=on_chunk_done)


def should_chunk(prompt_len: int, chunk_size: int = 512,
                 threshold: int = 1024) -> bool:
    """Decide whether to use chunked prefill for a prompt.

    Chunking adds overhead for short prompts (extra forward calls).
    Only chunk if the prompt is longer than the threshold.
    """
    return prompt_len > threshold


# ── HybridChunkedPrefiller ─────────────────────────────────────────────

class HybridChunkedPrefiller(ChunkedPrefiller):
    """Adaptive chunked prefill: chunk only when decode is active.

    Extends ChunkedPrefiller with awareness of whether decode requests
    are currently running. When decode is active, uses chunked prefill
    to avoid blocking decode. When no decode is running, uses continuous
    (non-chunked) prefill for maximum throughput.
    """

    def __init__(self, model: nn.Module, chunk_size: int = 512,
                 device: str = "cuda"):
        super().__init__(model, chunk_size=chunk_size, device=device)
        self._decode_active = False

    def set_decode_active(self, active: bool):
        """Inform the prefiller whether decode requests are running."""
        self._decode_active = active

    def prefill_hybrid(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = True,
        on_chunk_done: callable = None,
    ) -> tuple[torch.Tensor, list]:
        """Prefill with adaptive chunking.

        If decode is active: use chunked prefill (interleave with decode).
        If decode is NOT active: use continuous prefill (single forward pass).
        """
        B, T = input_ids.shape

        # Decision: chunk only when decode is active AND prompt is long
        use_chunking = self._decode_active and should_chunk(T, self.chunk_size)

        if not use_chunking:
            # Continuous prefill: single forward pass (maximum throughput)
            with torch.inference_mode():
                if T > 0:
                    out = self.model(input_ids, use_cache=use_cache,
                                     logits_last_only=True)
                    if isinstance(out, tuple):
                        logits = out[0]
                        past_kv = out[1] if len(out) > 1 else None
                        if len(out) > 2:
                            past_kv = out[2]
                    else:
                        logits = out
                        past_kv = None
                    return logits, past_kv
                return None, None

        # Chunked prefill: interleave with decode
        return self.prefill_chunked(
            input_ids, use_cache=use_cache, on_chunk_done=on_chunk_done)


class HybridPrefillScheduler:
    """Scheduler that coordinates hybrid chunked prefill with decode.

    Tracks whether decode requests are active and informs the prefiller.
    Also manages the interleaving of prefill chunks and decode steps.
    """

    def __init__(self, model: nn.Module, chunk_size: int = 512,
                 device: str = "cuda"):
        self.prefiller = HybridChunkedPrefiller(model, chunk_size, device)
        self._decode_requests: list = []
        self._prefill_queue: list = []

    def add_decode_request(self, request):
        """Add an ongoing decode request."""
        self._decode_requests.append(request)
        self.prefiller.set_decode_active(True)

    def remove_decode_request(self, request):
        """Remove a completed decode request."""
        if request in self._decode_requests:
            self._decode_requests.remove(request)
        if not self._decode_requests:
            self.prefiller.set_decode_active(False)

    def add_prefill_request(self, input_ids: torch.Tensor):
        """Queue a prefill request."""
        self._prefill_queue.append(input_ids)

    def run_step(self) -> list:
        """Run one scheduling step.

        If decode is active and prefill queue is non-empty:
          → run one prefill chunk + one decode step (interleaved)

        If only prefill is queued:
          → run continuous prefill (no chunking)

        If only decode is active:
          → run decode step

        Returns:
            list of completed request results
        """
        results = []

        if self._prefill_queue and self._decode_requests:
            # Hybrid mode: chunked prefill + decode
            input_ids = self._prefill_queue.pop(0)

            def on_chunk_done(chunk_idx, n_chunks):
                # Run one decode step between prefill chunks
                for req in self._decode_requests[:]:
                    result = req.step()
                    if req.done:
                        results.append(result)
                        self.remove_decode_request(req)

            logits, past_kv = self.prefiller.prefill_hybrid(
                input_ids, use_cache=True, on_chunk_done=on_chunk_done)
            results.append(("prefill", logits, past_kv))

        elif self._prefill_queue:
            # Prefill-only mode: continuous prefill (no chunking)
            input_ids = self._prefill_queue.pop(0)
            logits, past_kv = self.prefiller.prefill_hybrid(input_ids)
            results.append(("prefill", logits, past_kv))

        elif self._decode_requests:
            # Decode-only mode
            for req in self._decode_requests[:]:
                result = req.step()
                if req.done:
                    results.append(result)
                    self.remove_decode_request(req)

        return results

    @property
    def decode_active(self) -> bool:
        return len(self._decode_requests) > 0

    @property
    def prefill_pending(self) -> int:
        return len(self._prefill_queue)

    def stats(self) -> dict:
        return {
            "decode_requests": len(self._decode_requests),
            "prefill_pending": self.prefill_pending,
            "decode_active": self.decode_active,
            "mode": "hybrid" if (self.decode_active and self.prefill_pending) else
                    "prefill_only" if self.prefill_pending else
                    "decode_only" if self.decode_active else "idle",
        }
