"""EngineService — async port of forge_gui.api.engine_runtime.EngineRuntime.

One resident ForgeEngine shared by chat / agent / tools. The engine is
heavy (torch + checkpoint in VRAM) so exactly one instance lives here and
callers borrow it through ``acquire()`` — a threading.Lock lease identical
in semantics to the Qt version (engine calls run in executor threads, so
a threading lock is correct and preserves blocking-acquire behavior).

Load/reactivate run in a dedicated single-worker ThreadPoolExecutor;
progress is published to the event hub instead of Qt signals.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from forge_gui.api.status_reader import project_root
import contextlib

logger = logging.getLogger(__name__)


def _set_no_compile(use_compile: bool | None) -> None:
    if use_compile is True:
        os.environ.pop("FORGE_NO_COMPILE", None)
    elif use_compile is False:
        os.environ["FORGE_NO_COMPILE"] = "1"


def _iter_learn_texts(path: str):
    """Yield text docs from a learn corpus file — .txt/.md whole,
    .jsonl/.ndjson per-row (text/content/output or messages)."""
    import json
    if path.lower().endswith((".jsonl", ".ndjson")):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k in ("text", "content", "output", "response"):
                    v = row.get(k) if isinstance(row, dict) else None
                    if isinstance(v, str) and v:
                        yield v
                        break
    elif path.lower().endswith(".json"):
        try:
            rows = json.load(open(path, encoding="utf-8",
                                  errors="replace"))
        except Exception:
            return
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, str):
                yield row
            elif isinstance(row, dict):
                v = row.get("text") or row.get("content")
                if v:
                    yield str(v)
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield f.read()


class EngineService:
    """Owns the resident engine; serializes generation across tasks."""

    def __init__(self, hub) -> None:
        self._hub = hub
        self._engine = None
        self._state = "idle"           # idle | loading | ready | error
        self._info: dict[str, Any] = {}
        self._error = ""
        self._lock = threading.Lock()  # generation lease
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engine")
        self._load_task = None
        self._reactivating = False
        self._loop: asyncio.AbstractEventLoop | None = None
        # Load coordination: a generation counter invalidates in-flight
        # loads superseded by unload()/a newer load(), _loading_req records
        # the in-flight request, and _queued_load holds a different request
        # that arrived mid-load (runs when the current one settles).
        self._load_seq = 0
        self._loading_req: tuple | None = None
        self._queued_load: tuple | None = None
        self._learn_task = None
        self._learn_cancel = threading.Event()
        self._reactivate_task = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ── state ─────────────────────────────────────────────────────────
    @property
    def state(self) -> str:
        return self._state

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    @property
    def error(self) -> str:
        return self._error

    def is_ready(self) -> bool:
        return self._state == "ready" and self._engine is not None

    def is_busy(self) -> bool:
        return self._state == "loading" or self._reactivating

    def snapshot(self) -> dict[str, Any]:
        return {"state": self._state, "info": dict(self._info),
                "error": self._error, "busy": self.is_busy()}

    def _set_state(self, s: str) -> None:
        self._state = s
        self._hub.publish("engine", self.snapshot())

    def _progress(self, msg: str) -> None:
        self._hub.publish("engine_progress", {"message": msg})

    @staticmethod
    def _resolve_ckpt(checkpoint: str) -> str:
        """Resolve a load request to the absolute checkpoint path —
        mirrors the normalization in _load_blocking."""
        root = project_root()
        ckpt = checkpoint or str(root / "research" / "checkpoints" /
                               "ForgeLM_V2.safetensors")
        if not os.path.isabs(ckpt):
            ckpt = str(root / ckpt)
        return ckpt

    # ── load / unload ─────────────────────────────────────────────────
    def load(self, checkpoint: str, config_name: str,
             use_compile: bool | None = None,
             activation: dict | None = None,
             config_overrides: dict | None = None) -> None:
        """Kick off an async load. Progress/result arrives via hub events."""
        if str(checkpoint).lower().endswith(".flux"):
            config_name = "flux"   # .flux routes to from_flux; normalize
                                   # so dedupe/reload checks stay stable
        req = (self._resolve_ckpt(checkpoint), config_name)
        if self._state == "loading":
            if self._loading_req == req:
                return  # identical load already in flight
            # Different request mid-load (e.g. user picked another model
            # during startup preload) — queue it instead of dropping it.
            self._queued_load = (checkpoint, config_name, use_compile,
                                 activation, config_overrides)
            self._progress("load queued behind in-flight load")
            return
        old_engine = None
        if self.is_ready():
            # A bare request for the resident model is a no-op — avoids a
            # pointless unload+reload cycle (e.g. re-clicking Load on the
            # checkpoint the startup preload just brought up).
            if (self._info.get("checkpoint"),
                    self._info.get("config_name")) == req \
                    and activation is None and use_compile is None \
                    and not config_overrides:
                return
            # Detach cheaply HERE; the blocking sleep(2)+empty_cache runs
            # on the pool thread inside the load, not on the event loop.
            old_engine = self._detach()
        self._error = ""
        self._load_seq += 1
        seq = self._load_seq
        self._loading_req = req
        self._set_state("loading")
        loop = self._loop or asyncio.get_event_loop()
        coro = self._load_async(
            seq, loop, checkpoint, config_name, use_compile, activation,
            config_overrides, old_engine)
        if loop.is_running():
            self._load_task = asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            self._load_task = asyncio.ensure_future(coro)

    async def _load_async(self, seq, loop, checkpoint, config_name,
                          use_compile, activation, config_overrides,
                          old_engine=None) -> None:
        try:
            engine, info = await loop.run_in_executor(
                self._pool, self._load_blocking, checkpoint, config_name,
                use_compile, activation, config_overrides, old_engine)
        except Exception as e:
            if seq == self._load_seq:
                self._loading_req = None
                self._error = str(e)
                self._engine = None
                self._set_state("error")
                self._drain_queue()
            return
        if seq != self._load_seq:
            # Superseded by a newer load/unload while in flight — free the
            # VRAM this engine holds instead of installing it.
            self._discard_engine(engine)
            return
        self._loading_req = None
        self._engine = engine
        self._info = info
        self._set_state("ready")
        self._drain_queue()

    def _drain_queue(self) -> None:
        q, self._queued_load = self._queued_load, None
        if q is not None:
            self.load(*q)

    @staticmethod
    def _discard_engine(engine) -> None:
        """Free a fully-loaded engine whose result was superseded."""
        try:
            engine.sleep(level=2)
        except Exception:
            logger.debug("superseded engine cleanup failed", exc_info=True)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_blocking(self, checkpoint, config_name, use_compile,
                       activation, config_overrides, old_engine=None):
        """Blocking load — runs in the engine executor thread."""
        t0 = time.perf_counter()
        if old_engine is not None:
            # Previous resident model — teardown here (pool thread), not
            # on the event loop where sleep(2)+empty_cache used to stall
            # every endpoint + /ws.
            self._progress("unloading previous model…")
            self._discard_engine(old_engine)
        if activation is not None:
            _set_no_compile(bool(activation.get("use_compile", False)))
        else:
            _set_no_compile(use_compile if use_compile is not None else False)
        self._progress("importing torch / ForgeEngine…")
        from forge.engine.forge_engine import ForgeEngine  # type: ignore
        ckpt = self._resolve_ckpt(checkpoint)
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"checkpoint not found: {ckpt}")

        import torch
        if ckpt.lower().endswith(".flux"):
            return self._load_flux(ckpt, t0)
        from forge.engine.engine_common import _fast_load_vram_required
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            need = _fast_load_vram_required(os.path.getsize(ckpt))
            if free < need:
                raise RuntimeError(
                    f"not enough free VRAM: {free / 1e9:.1f} GB free, "
                    f"loading this checkpoint needs ~{need / 1e9:.1f} GB. "
                    f"Close other GPU apps / GUI instances (or unload the "
                    f"current model) and retry.")

        self._progress(f"loading {ckpt} ({config_name})…")
        if activation is not None:
            engine = ForgeEngine.from_checkpoint(
                checkpoint=ckpt, config_name=config_name,
                auto_activate=False, config_overrides=config_overrides)
            self._progress("activating features (manual preset)…")
            engine.activate(**activation)
        else:
            engine = ForgeEngine.from_checkpoint(
                checkpoint=ckpt, config_name=config_name,
                config_overrides=config_overrides)
            self._progress("activating features (optimal preset)…")

        # ForgeGate probe bundle (~44KB): chat uses the route head to skip
        # the think pass on easy turns. Missing/stale bundle is non-fatal.
        probes = project_root() / "research" / "checkpoints" / "gate_probes.pt"
        if probes.is_file():
            try:
                engine.load_gate_probes(str(probes))
            except Exception:
                logger.warning("gate probes failed to load", exc_info=True)

        active: dict = {}
        try:
            ac = getattr(engine, "active_config", None)
            if ac is not None:
                active = ac.to_dict()
        except Exception:
            logger.debug("Failed to get active config dict", exc_info=True)
        info = {
            "checkpoint": ckpt,
            "config_name": config_name,
            "load_s": round(time.perf_counter() - t0, 2),
            "device": str(getattr(getattr(engine, "device", None), "type", "?")),
            "dtype": str(getattr(engine, "dtype", "?")),
            "use_compile": bool(active.get("use_compile",
                                         use_compile or False)),
            "activation": active,
        }
        return engine, info

    def _load_flux(self, ckpt: str, t0: float):
        """FluxLM memory snapshot — bypasses checkpoint detection, VRAM
        sizing, activation presets and gate probes (no dense weights,
        no hidden states).  GPU-resident tables when CUDA is up."""
        self._progress(f"loading FLUX memory {ckpt}…")
        import torch

        from forge.engine.forge_engine import ForgeEngine  # type: ignore
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        engine = ForgeEngine.from_flux(
            checkpoint=ckpt, device=dev, cuda_primary=(dev == "cuda"))
        info = {
            "checkpoint": ckpt,
            "config_name": "flux",
            "load_s": round(time.perf_counter() - t0, 2),
            "device": dev,
            "dtype": "memory",
            "use_compile": False,
            "activation": {},
        }
        return engine, info

    def _detach(self):
        """Cheap state teardown — invalidate in-flight loads and drop the
        resident-engine reference. Returns the engine (or None) so the
        caller can dispose of it on a worker thread."""
        self._load_seq += 1  # invalidate any in-flight load
        self._queued_load = None
        self._loading_req = None
        self._load_task = None
        self._reactivating = False
        engine, self._engine = self._engine, None
        self._info = {}
        self._set_state("idle")
        return engine

    def unload(self) -> None:
        """Detach + blocking teardown. Callers on the event loop should
        run this via run_in_executor."""
        engine = self._detach()
        if engine is not None:
            self._discard_engine(engine)

    # ── live re-activation ────────────────────────────────────────────
    def reactivate(self, activation: dict) -> dict:
        """Schedule a re-activation. Returns {"ok": bool, ...} — refusals
        come back as errors instead of the old always-ok route reply."""
        if not self.is_ready():
            self._hub.publish("engine_error",
                              {"error": "no resident engine to re-activate"})
            return {"ok": False,
                    "error": "no resident engine to re-activate"}
        if self.is_busy():
            self._hub.publish("engine_error",
                              {"error": "engine is busy"})
            return {"ok": False, "error": "engine is busy"}
        unknown: list[str] = []
        try:
            from dataclasses import fields as _dc_fields

            from forge.engine.activation import ActivationConfig
            valid = {f.name for f in _dc_fields(ActivationConfig)}
            unknown = sorted(k for k in activation if k not in valid)
        except Exception:
            pass
        self._reactivating = True
        self._set_state(self._state)  # republish busy flag
        loop = self._loop or asyncio.get_event_loop()
        coro = self._reactivate_async(loop, activation)
        if loop.is_running():
            self._reactivate_task = asyncio.run_coroutine_threadsafe(
                coro, loop)
        else:
            self._reactivate_task = asyncio.ensure_future(coro)
        out: dict = {"ok": True}
        if unknown:
            # ActivationConfig.from_kwargs drops unknown keys silently —
            # surface them so a typo doesn't masquerade as applied.
            out["warnings"] = [
                "unknown activation keys ignored: " + ", ".join(unknown)]
        return out

    async def _reactivate_async(self, loop, activation) -> None:
        t0 = time.perf_counter()
        try:
            active = await loop.run_in_executor(
                self._pool, self._reactivate_blocking, activation)
        except Exception as e:
            self._reactivating = False
            self._hub.publish("engine_error", {"error": str(e)})
            self._set_state(self._state)
            return
        self._progress(f"features applied in {time.perf_counter() - t0:.1f}s")
        self._info = dict(self._info, activation=active,
                          use_compile=bool(active.get("use_compile", False)))
        self._reactivating = False
        self._set_state(self._state)

    def _reactivate_blocking(self, activation):
        _set_no_compile(bool(activation.get("use_compile", False)))
        with self.acquire(timeout_s=120.0) as engine:
            self._progress("applying features to resident engine…")
            engine.activate(**activation)
            try:
                ac = getattr(engine, "active_config", None)
                return ac.to_dict() if ac is not None else {}
            except Exception:
                return {}

    # ── live-learn (FluxLM) ───────────────────────────────────────────
    LEARN_SLICE = 1 << 13          # ~8k tokens per lease hold

    def learn(self, path: str, tag: str | None = None,
              max_tokens: int = 0) -> None:
        """Async live-learn: ingest a corpus file/dir into the resident
        FluxLM.  Progress arrives via engine_progress / learn_done hub
        events; cancel via learn_cancel()."""
        if self._learn_task is not None and not self._learn_task.done():
            self._hub.publish("engine_error",
                              {"error": "a learn run is already active"})
            return
        self._learn_cancel.clear()
        loop = self._loop or asyncio.get_event_loop()
        coro = self._learn_async(loop, path, tag, max_tokens)
        if loop.is_running():
            self._learn_task = asyncio.run_coroutine_threadsafe(
                coro, loop)
        else:
            self._learn_task = asyncio.ensure_future(coro)

    def learn_cancel(self) -> None:
        self._learn_cancel.set()

    async def _learn_async(self, loop, path, tag, max_tokens) -> None:
        try:
            rep = await loop.run_in_executor(
                self._pool, self._learn_blocking, path, tag, max_tokens)
        except Exception as e:
            self._hub.publish("engine_error", {"error": str(e)})
            return
        if rep is not None:
            self._hub.publish("learn_done", rep)
            self._info = dict(self._info, learn=rep)
            self._set_state(self._state)

    def _learn_blocking(self, path, tag, max_tokens):
        """Slice-wise ingest under the generation lease — each hold is
        ~8k tokens (~4s GPU) so chat can interleave between slices."""
        if not self.is_ready():
            raise RuntimeError("no resident engine — load a model first")
        eng = self._engine
        if eng.model.__class__.__name__ != "FluxLM":
            raise RuntimeError(
                "learn requires a resident FluxLM engine "
                "(load a .flux checkpoint first)")
        root = project_root()
        p = path if os.path.isabs(path) else str(root / path)
        pp = os.path.abspath(p)
        if os.path.isdir(pp):
            files = sorted(
                os.path.join(pp, f) for f in os.listdir(pp)
                if f.lower().endswith((".txt", ".md", ".jsonl",
                                       ".ndjson", ".json")))
        elif os.path.isfile(pp):
            files = [pp]
        else:
            raise RuntimeError(f"learn path not found: {pp}")
        tok = eng.tokenizer
        total, docs, t0 = 0, 0, time.perf_counter()
        last_log = t0
        self._progress(f"learning {len(files)} file(s) into memory…")
        for fp in files:
            ftag = tag or os.path.splitext(os.path.basename(fp))[0]
            for text in _iter_learn_texts(fp):
                ids = tok.encode(text, add_special_tokens=False) \
                    if hasattr(tok, "encode") else tok(text).input_ids
                for off in range(0, len(ids), self.LEARN_SLICE):
                    sl = ids[off:off + self.LEARN_SLICE]
                    with self.acquire(timeout_s=300.0) as e:
                        total += e.model.ingest(sl, tag=ftag)
                    now = time.perf_counter()
                    if now - last_log > 2.0:
                        last_log = now
                        rate = total / max(now - t0, 1e-9)
                        self._progress(
                            f"learn {ftag}: {total:,} tok "
                            f"({rate:,.0f} tok/s)")
                    if self._learn_cancel.is_set() or (
                            max_tokens and total >= max_tokens):
                        return {"tokens": total, "docs": docs,
                                "tag": ftag, "cancelled": True,
                                "tok_s": round(
                                    total / max(now - t0, 1e-9), 1)}
                with self.acquire(timeout_s=120.0) as e:
                    e.model.soft_reset()      # docs are independent
                docs += 1
                if self._learn_cancel.is_set() or (
                        max_tokens and total >= max_tokens):
                    break
            if self._learn_cancel.is_set() or (
                    max_tokens and total >= max_tokens):
                break
        el = time.perf_counter() - t0
        rep = {"tokens": total, "docs": docs,
               "tok_s": round(total / max(el, 1e-9), 1),
               "elapsed_s": round(el, 1), "cancelled":
                   bool(self._learn_cancel.is_set())}
        return rep

    # ── borrowing the engine ──────────────────────────────────────────
    def acquire(self, timeout_s: float = 600.0):
        """Sync context manager — call from executor threads (blocking)."""
        return _EngineLease(self, timeout_s)

    def try_engine(self):
        return self._engine if self.is_ready() else None

    @staticmethod
    def _as_awaitable(fut):
        """concurrent.futures.Future → awaitable; asyncio futures pass
        through. asyncio.wait_for() rejects raw concurrent futures
        (TypeError) — that was the old shutdown bug."""
        if isinstance(fut, asyncio.Future):
            return fut
        return asyncio.wrap_future(fut)

    async def shutdown(self) -> None:
        # Stop accepting new work + cancel the learn loop first.
        self._learn_cancel.set()
        lt = self._learn_task
        if lt is not None and not lt.done():
            with contextlib.suppress(Exception):
                lt.cancel()
        for fut, timeout in ((self._load_task, 60), (self._learn_task, 30)):
            if fut is None:
                continue
            try:
                if fut.done():
                    continue
                await asyncio.wait_for(self._as_awaitable(fut),
                                       timeout=timeout)
            except (Exception, asyncio.CancelledError):
                pass
        # Blocking engine teardown off the loop, on the engine pool.
        with contextlib.suppress(Exception):
            await asyncio.get_running_loop().run_in_executor(
                self._pool, self.unload)
        self._pool.shutdown(wait=True, cancel_futures=True)


class _EngineLease:
    def __init__(self, rt: EngineService, timeout_s: float) -> None:
        self._rt = rt
        self._timeout = timeout_s

    def __enter__(self):
        if not self._rt._lock.acquire(timeout=self._timeout):
            raise RuntimeError("engine busy — another generation is running")
        if not self._rt.is_ready():
            self._rt._lock.release()
            raise RuntimeError("engine not loaded — load a model first")
        return self._rt._engine

    def __exit__(self, *exc) -> None:
        self._rt._lock.release()
        return None


# singleton — created in app.py lifespan
engine_service: EngineService | None = None
