"""Boot & Test — async port of forge_gui.api.model_boot.ModelBootWorker.

Loads a checkpoint on the shared runtime if needed (waiting for the load
to finish), then runs a small test generation to verify the model works.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="boot")


class BootService:
    def __init__(self, hub, runtime) -> None:
        self._hub = hub
        self._runtime = runtime
        self._cancel = threading.Event()
        self.running = False

    def cancel(self) -> None:
        self._cancel.set()

    async def boot_and_test(self, checkpoint: str, config_name: str,
                            prompt: str = "def fibonacci(n):",
                            max_tokens: int = 64,
                            timeout_s: float = 300.0) -> dict:
        if self.running:
            return {"ok": False, "error": "boot test already running"}
        self.running = True
        self._cancel.clear()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                _pool, self._run, checkpoint, config_name, prompt,
                max_tokens, timeout_s)
        finally:
            self.running = False

    def _status(self, msg: str) -> None:
        self._hub.publish("boot", {"kind": "status", "message": msg})

    def _run(self, checkpoint, config_name, prompt, max_tokens,
             timeout_s) -> dict:
        try:
            # info.checkpoint is the RESOLVED absolute path; compare it to
            # the resolved request — the old raw-string compare reloaded
            # the engine even when the same model was already resident.
            resolved_req = self._runtime._resolve_ckpt(checkpoint)
            needs_reload = (
                not self._runtime.is_ready()
                or self._runtime.info.get("checkpoint") != resolved_req
                or self._runtime.info.get("config_name") != config_name)
            if needs_reload:
                self._status(f"loading {config_name}…")
                # runtime.load() is async-driven; invoke + poll state
                self._runtime.load(checkpoint, config_name)
                t0 = time.time()
                while time.time() - t0 < timeout_s:
                    if self._cancel.is_set():
                        return {"ok": False, "error": "cancelled"}
                    if self._runtime.is_ready():
                        break
                    if self._runtime.state == "error":
                        return {"ok": False,
                                "error": self._runtime.error}
                    time.sleep(0.4)
                if not self._runtime.is_ready():
                    return {"ok": False,
                            "error": "load timed out or cancelled"}

            self._status("running test generation…")
            t0 = time.perf_counter()
            with self._runtime.acquire(timeout_s=120.0) as engine:
                output = engine.generate(
                    prompt, max_new_tokens=max_tokens,
                    temperature=0.7, top_k=50, top_p=0.95)
            dt = max(1e-6, time.perf_counter() - t0)
            tokens = len(output.split())
            result = {"ok": True, "output": output, "tokens": tokens,
                      "time_s": round(dt, 2),
                      "tps": round(tokens / dt, 2)}
            self._hub.publish("boot", {"kind": "result", "data": result})
            return result
        except Exception as e:
            logger.warning("boot test failed: %s", e, exc_info=True)
            self._hub.publish("boot", {"kind": "error",
                                       "error": f"{type(e).__name__}: {e}"})
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


boot_service: BootService | None = None
