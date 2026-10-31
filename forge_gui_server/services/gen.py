"""Generation service — async pumps over the resident engine.

Port of forge_gui/api/generation.py: the same four engine entry points
(generate_stream / generate_adaptive / generate_batch / generate_raw),
but tokens flow through an asyncio.Queue to SSE/WebSocket consumers
instead of Qt signals.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

# Generation calls are blocking; they run here and fight over the
# EngineService lease lock — same serialization as the Qt version.
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gen")


class StreamCancel:
    """Thread-safe cancel token shared with the worker thread."""

    def __init__(self) -> None:
        self._flag = threading.Event()

    def cancel(self) -> None:
        self._flag.set()

    @property
    def cancelled(self) -> bool:
        return self._flag.is_set()


async def stream_tokens(runtime, prompt: str, max_new_tokens: int = 128,
                        temperature: float = 0.7, top_k: int = 50,
                        top_p: float = 0.95,
                        cancel: StreamCancel | None = None,
                        ) -> AsyncIterator[dict]:
    """Yield {"type": "token"|"done"|"error", ...} events."""
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancel = cancel or StreamCancel()

    def emit(item):
        loop.call_soon_threadsafe(q.put_nowait, item)

    def worker():
        t0 = time.perf_counter()
        try:
            if not runtime.is_ready():
                emit(("error", "No model resident — load one on the "
                       "Models page first (one shared engine serves "
                       "every page)."))
                return
            chunks: list[str] = []
            with runtime.acquire(timeout_s=60.0) as engine:
                for tok in engine.generate_stream(
                        prompt, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_k=top_k, top_p=top_p):
                    if cancel.cancelled:
                        break
                    chunks.append(tok)
                    emit(("token", tok))
            dt = max(1e-6, time.perf_counter() - t0)
            full = "".join(chunks)
            emit(("done", {"text": full,
                           "tok_s": round(len(full.split()) / dt, 2)}))
        except Exception as e:
            logger.warning("generation failed: %s", e, exc_info=True)
            emit(("error", f"{type(e).__name__}: {e}"))
        finally:
            emit(("end", None))

    fut = loop.run_in_executor(_pool, worker)
    try:
        while True:
            kind, data = await q.get()
            if kind == "end":
                break
            yield {"type": kind, "data": data}
    finally:
        # Consumer gone (disconnect/GeneratorExit) — flag the worker so a
        # running generate_stream breaks instead of holding the lease.
        cancel.cancel()
        fut.cancel()


async def run_adaptive(runtime, prompt: str, think_max_tokens: int = 512,
                       no_think_max_tokens: int = 256,
                       temperature: float = 0.0, top_p: float = 1.0,
                       top_k: int = 80) -> dict[str, Any]:
    loop = asyncio.get_running_loop()

    def worker():
        t0 = time.perf_counter()
        with runtime.acquire(timeout_s=120.0) as engine:
            text, did_think = engine.generate_adaptive(
                prompt, think_max_tokens=think_max_tokens,
                no_think_max_tokens=no_think_max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k)
        dt = max(1e-6, time.perf_counter() - t0)
        return {"text": text, "did_think": did_think,
                "tok_s": round(len(text.split()) / dt, 2)}

    if not runtime.is_ready():
        return {"error": "No model resident — load a model first."}
    try:
        return await loop.run_in_executor(_pool, worker)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def run_batch(runtime, prompts: list[str], max_new_tokens: int = 256,
                    temperature: float = 0.0, top_p: float = 1.0,
                    top_k: int = 80) -> dict[str, Any]:
    loop = asyncio.get_running_loop()

    def worker():
        t0 = time.perf_counter()
        with runtime.acquire(timeout_s=300.0) as engine:
            results = engine.generate_batch(
                prompts, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k)
        dt = max(1e-6, time.perf_counter() - t0)
        words = sum(len(r.split()) for r in results)
        return {"results": results, "tok_s": round(words / dt, 2)}

    if not runtime.is_ready():
        return {"error": "No model resident — load a model first."}
    try:
        return await loop.run_in_executor(_pool, worker)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


async def run_raw(runtime, prompt: str, max_new_tokens: int = 256,
                  temperature: float = 0.2, top_p: float = 1.0,
                  top_k: int = 80, repetition_penalty: float = 1.05,
                  min_p: float = 0.0, min_k: float = 0.0,
                  skip_special_tokens: bool = False) -> dict[str, Any]:
    loop = asyncio.get_running_loop()

    def worker():
        t0 = time.perf_counter()
        with runtime.acquire(timeout_s=120.0) as engine:
            text = engine.generate_raw(
                prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                min_p=min_p, min_k=min_k,
                skip_special_tokens=skip_special_tokens)
        dt = max(1e-6, time.perf_counter() - t0)
        return {"text": text, "tok_s": round(len(text.split()) / dt, 2)}

    if not runtime.is_ready():
        return {"error": "No model resident — load a model first."}
    try:
        return await loop.run_in_executor(_pool, worker)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
