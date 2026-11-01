"""Integration test: cross-thread ForgeEngine hand-off (critique F25).

PROBLEM (F25): ForgeEngine is constructed in a QThread worker
(``_LoadWorker`` in ``forge_gui/api/engine_runtime.py``) and then passed
to the main thread for generation via ``EngineRuntime._on_loaded``.
This is a potential thread-safety issue with PyTorch CUDA tensors.

THREAD-SAFETY CONTRACT
======================

Current GUI pattern (``forge_gui/api/engine_runtime.py``):

    1. ``_LoadWorker`` (a QThread) runs ``ForgeEngine.from_checkpoint()``
       — this constructs the model, moves tensors to the device, and
       activates strategies, all on the **worker thread**.
    2. The finished engine object is emitted via ``finished_ok.emit(engine,
       info)`` and stored in ``EngineRuntime._engine`` on the **main
       thread** (Qt signal/slot delivery).
    3. Generation happens on yet another thread (``_EngineChatWorker``)
       which acquires the engine through ``EngineRuntime.acquire()`` and
       calls ``engine.generate_stream()`` / ``engine.generate()``.

Why this works on CPU
---------------------
PyTorch **CPU** tensors are thread-safe for read-only inference operations.
Once the model is constructed and ``model.eval()`` is called, the
parameters are frozen (no autograd, no in-place mutation). Multiple threads
can read the same parameter tensors concurrently without data races.
The only shared mutable state is the KV cache, which is per-request (each
``generate()`` call builds its own cache) — so cross-thread generation on
CPU is safe as long as only one thread generates at a time (the
``EngineRuntime._lock`` serializes this).

The risk with CUDA tensors
--------------------------
CUDA operations are **not** automatically thread-safe across threads that
use different CUDA streams. If the engine is constructed on a worker
thread (which creates a default stream on that thread's CUDA context) and
then generation runs on a different thread, the operations may land on
different streams without explicit synchronization. Symptoms include:

  * ``RuntimeError: CUDA error: invalid device context``
  * Silent data corruption (wrong outputs) from unsynchronised stream
    operations.
  * Deadlocks if one thread holds the GIL while waiting for a CUDA event
    on another stream.

Recommended pattern for CUDA
---------------------------
  * **Option A (simplest):** construct AND generate on the **same
    dedicated inference thread**. The GUI already does this for chat
    (``_EngineChatWorker`` runs on its own QThread), but the *load* worker
    and the *generation* worker are different threads — so the engine
    crosses a thread boundary.
  * **Option B:** construct on the worker thread, then call
    ``torch.cuda.synchronize()`` before handing off, and ensure the
    receiving thread uses the same device / stream. This is fragile.
  * **Option C (best):** keep a single long-lived inference thread. The
    load worker sends a "load" command to it; the inference thread
    constructs the engine and then serves generation requests from the
    same thread. No tensor ever crosses a thread boundary.

This test validates the **CPU** path (Option A-equivalent: construct on
worker, generate on main) to document the contract and catch regressions.
The CUDA path is left as a documented risk — a full CUDA cross-thread test
would need a GPU and careful stream management.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from forge.config import get_config
from forge.engine.forge_engine import ForgeEngine
from forge.model_loader import ConfigurableResearchLLM


def _build_engine_cpu(preset="forgelm_tiny", vocab=65536):
    """Build a ForgeEngine on CPU (no checkpoint, config-only tiny model).

    Mirrors the ``_build_engine`` helper in test_end_to_end_generate.py
    but forces CPU device to avoid CUDA thread-safety complications.
    Uses vocab=65536 to match the lfm25 tokenizer.
    """
    cfg = get_config(preset)
    cfg.vocab_size = vocab
    cfg.dtype = "float32"  # CPU-safe dtype (bf16 ops are limited on CPU)
    cfg.device = "cpu"

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("cpu"):
            model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    model.eval()

    from research.tokenizer_cache import get_tokenizer

    tok = get_tokenizer("research/checkpoints/forgelm_v2_tokenizer")
    engine = ForgeEngine(model, tok, device="cpu")
    return engine, model


@pytest.mark.integration
@pytest.mark.usefixtures("v2_assets")
class TestCrossThreadEngineHandoff:
    """Validate that a ForgeEngine constructed on a worker thread can be
    used for generation on the main thread (the GUI's actual pattern).

    These tests run on CPU to isolate the thread-hand-off logic from
    CUDA stream-safety issues. See the module docstring for the CUDA
    risk analysis and recommended patterns.
    """

    def test_construct_on_worker_generate_on_main(self):
        """Engine built in a worker thread, generate() called on main.

        This is the exact pattern used by the GUI:
          1. _LoadWorker (QThread) constructs ForgeEngine
          2. Engine is stored on the main thread
          3. generate() is called from the main thread (or another worker)

        On CPU this is safe because parameter tensors are read-only after
        model.eval() and the KV cache is per-request.
        """
        engine_holder = {}
        error_holder = {}

        def worker_construct():
            """Simulate _LoadWorker.run() — construct engine off-thread."""
            try:
                eng, mdl = _build_engine_cpu()
                engine_holder["engine"] = eng
                engine_holder["model"] = mdl
            except Exception as e:
                error_holder["error"] = e

        # Step 1: construct on a worker thread (simulates QThread)
        t = threading.Thread(target=worker_construct, name="LoadWorker")
        t.start()
        t.join(timeout=120)  # generous for CPU model init

        assert not t.is_alive(), "Worker thread did not finish within 120s"
        assert "error" not in error_holder, f"Engine construction failed on worker thread: {error_holder.get('error')}"
        assert "engine" in engine_holder, "Engine was not stored by worker"

        engine = engine_holder["engine"]

        # Step 2: generate on the MAIN thread (simulates GUI main-thread
        # usage after the engine is handed off via Qt signal).
        try:
            result = engine.generate(
                "Hello",
                max_new_tokens=10,
                finish_sentence=False,
                temperature=0.0,
            )
            assert isinstance(result, str), f"Expected str from generate(), got {type(result)}"
            assert len(result) > 0, "generate() returned empty string"
        finally:
            del engine_holder["engine"]
            del engine_holder["model"]
            del engine

    def test_multiple_generate_calls_after_cross_thread_handoff(self):
        """After hand-off, multiple generate() calls on the main thread
        should all succeed without state corruption.

        This catches the case where the first call works (because the KV
        cache is freshly allocated) but subsequent calls fail because the
        engine's internal state was set up on a different thread.
        """
        engine_holder = {}
        error_holder = {}

        def worker_construct():
            try:
                eng, mdl = _build_engine_cpu()
                engine_holder["engine"] = eng
                engine_holder["model"] = mdl
            except Exception as e:
                error_holder["error"] = e

        t = threading.Thread(target=worker_construct, name="LoadWorker")
        t.start()
        t.join(timeout=120)

        assert not t.is_alive(), "Worker thread did not finish"
        assert "error" not in error_holder, f"Construction failed: {error_holder.get('error')}"

        engine = engine_holder["engine"]
        try:
            results = []
            for i in range(3):
                r = engine.generate(
                    f"Test prompt number {i}",
                    max_new_tokens=5,
                    finish_sentence=False,
                    temperature=0.0,
                )
                assert isinstance(r, str), f"Call {i}: expected str, got {type(r)}"
                results.append(r)

            # All calls should produce non-empty output
            assert all(len(r) > 0 for r in results), f"Some generate() calls returned empty: {results!r}"
        finally:
            del engine_holder["engine"]
            del engine_holder["model"]
            del engine

    def test_streaming_after_cross_thread_handoff(self):
        """generate_stream() should also work after cross-thread hand-off.

        The GUI's _EngineChatWorker uses generate_stream(), not generate(),
        so this is the actual production code path.
        """
        engine_holder = {}
        error_holder = {}

        def worker_construct():
            try:
                eng, mdl = _build_engine_cpu()
                engine_holder["engine"] = eng
                engine_holder["model"] = mdl
            except Exception as e:
                error_holder["error"] = e

        t = threading.Thread(target=worker_construct, name="LoadWorker")
        t.start()
        t.join(timeout=120)

        assert not t.is_alive(), "Worker thread did not finish"
        assert "error" not in error_holder, f"Construction failed: {error_holder.get('error')}"

        engine = engine_holder["engine"]
        try:
            chunks = list(
                engine.generate_stream(
                    "Hello world",
                    max_new_tokens=10,
                    temperature=0.0,
                )
            )
            assert len(chunks) > 0, "Stream produced no chunks"
            assert all(isinstance(c, str) for c in chunks), f"Non-string chunk: {[type(c) for c in chunks]}"
        finally:
            del engine_holder["engine"]
            del engine_holder["model"]
            del engine

    def test_engine_device_is_cpu(self):
        """Sanity: the engine constructed on the worker thread must report
        CPU as its device (confirms no accidental CUDA tensor creation)."""
        engine_holder = {}
        error_holder = {}

        def worker_construct():
            try:
                eng, mdl = _build_engine_cpu()
                engine_holder["engine"] = eng
                engine_holder["model"] = mdl
            except Exception as e:
                error_holder["error"] = e

        t = threading.Thread(target=worker_construct, name="LoadWorker")
        t.start()
        t.join(timeout=120)

        assert not t.is_alive(), "Worker thread did not finish"
        assert "error" not in error_holder

        engine = engine_holder["engine"]
        try:
            assert engine.device.type == "cpu", f"Expected CPU device, got {engine.device}"
            # All model parameters should be on CPU
            for name, param in engine.model.named_parameters():
                assert param.device.type == "cpu", f"Parameter {name} on {param.device}, expected CPU"
        finally:
            del engine_holder["engine"]
            del engine_holder["model"]
            del engine
