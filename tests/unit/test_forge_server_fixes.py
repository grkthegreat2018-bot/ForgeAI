"""Tests for forge_server / model_registry critique fixes.

Covers: EC-057 (worker-thread SSE streaming), EC-058 (FORGE_API_KEY
middleware), EC-061 (registry token delta), EC-062 (kv-cache-type via
hotswap), EC-065/086 (X-Request-ID, /metrics, deep /health), EC-073
(stream consumer must not hang on producer error), EC-152 subset
(/version, /v1/tokenize, /v1/detokenize), EC-156 (seeded fork_rng),
EC-161 (is_awake sync), EC-198 (per-task serialization), EC-199
(in-flight tracking + busy engines not evicted), EC-218 (batch cap,
executor offload, real token counts).

All CPU-only with a fake engine — no checkpoint loads.
"""
import asyncio
import threading
import time
from concurrent.futures import Future

import pytest
import torch
from fastapi.testclient import TestClient

from forge.engine.forge_server import ForgeServer, _MAX_BATCH_PROMPTS
from forge.engine.model_registry import (
    EngineEntry,
    ModelRegistry,
    VRAMBudgetExceeded,
)

MODEL = "forgelm-v2-jamba"


# ── Fakes ───────────────────────────────────────────────────────────────────

class _FakeTokenizer:
    eos_token_id = 7

    def encode(self, text):
        return [ord(c) % 128 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor(
            [[ord(c) % 128 for c in text]])}


class _FakeHotswap:
    """Records update/apply calls instead of re-activating an engine."""

    def __init__(self):
        self._pending = {}
        self.apply_calls = 0
        self.applied = []

    def update_from_dict(self, d):
        self._pending.update(d)

    def apply_pending(self):
        self.apply_calls += 1
        self.applied.append(dict(self._pending))
        self._pending = {}
        return True

    def get_pending_changes(self):
        return dict(self._pending)

    def has_pending(self):
        return bool(self._pending)

    def get_settings(self):
        return {}


class _FakeEngine:
    """Minimal engine: enough surface for registry + server tests."""

    def __init__(self, chunks=("a", "b", "c")):
        self.tokenizer = _FakeTokenizer()
        self.hotswap = _FakeHotswap()
        self.device = torch.device("cpu")
        self._gen_lock = threading.RLock()
        self._awake = True
        self.total_tokens_generated = 0
        self.generation_count = 0
        self.activate_calls = []
        self.sleep_calls = []
        self.rng_values = []
        self._chunks = list(chunks)
        self._gen_gate = None       # Event to stall generate()
        self._stream_gate = None    # Event to stall generate_stream()
        self.stream_thread = None

    def generate(self, prompt, max_new_tokens=100, **kw):
        if self._gen_gate is not None:
            self._gen_gate.wait(10)
        # Record RNG draw so tests can verify seeding actually applied.
        self.rng_values.append(torch.rand(1).item())
        self.total_tokens_generated += 3
        self.generation_count += 1
        return "out"

    def generate_stream(self, prompt, **kw):
        with self._gen_lock:
            self.stream_thread = threading.get_ident()
            yield self._chunks[0]
            if self._stream_gate is not None:
                self._stream_gate.wait(10)
            for c in self._chunks[1:]:
                yield c
            self.total_tokens_generated += len(self._chunks)

    def generate_batch(self, prompts, **kw):
        return [f"r{i}" for i in range(len(prompts))]

    def activate(self, **kw):
        self.activate_calls.append(kw)

    def sleep(self, level=1):
        self.sleep_calls.append(level)
        self._awake = False

    def wake(self):
        self._awake = True

    # BatchQueue boot-config check reads these
    quantize = None
    acceleration = None


def _registry_with(engine, model_id=MODEL):
    reg = ModelRegistry(device="cpu")
    reg._entries[model_id] = EngineEntry(
        model_id=model_id, engine=engine,
        vram_budget_bytes=1 << 20, checkpoint_path="",
        config_name="forgelm_v2", tokenizer_path="")
    return reg


def _server(engine=None):
    engine = engine or _FakeEngine()
    return ForgeServer(registry=_registry_with(engine)), engine


# ── Registry fixes ──────────────────────────────────────────────────────────

class TestRegistryFixes:
    def test_token_delta_not_cumulative(self):
        """EC-061: entry.total_tokens grows by the per-request delta."""
        reg = _registry_with(_FakeEngine())
        reg.generate(MODEL, "hi")
        reg.generate(MODEL, "hi")
        entry = reg._entries[MODEL]
        assert entry.engine.total_tokens_generated == 6
        assert entry.total_tokens == 6  # not 3 + 6 = 9

    def test_seed_scoped_and_restored(self):
        """EC-156: per-request seed applies, then global RNG is restored."""
        eng = _FakeEngine()
        reg = _registry_with(eng)
        torch.manual_seed(123)
        snap = torch.random.get_rng_state()
        reg.generate(MODEL, "hi", seed=5)
        assert torch.equal(torch.random.get_rng_state(), snap)
        first = eng.rng_values[-1]
        reg.generate(MODEL, "hi", seed=5)
        assert eng.rng_values[-1] == first  # same seed → same draw

    def test_in_flight_and_busy_engine_not_evicted(self):
        """EC-199: in-flight engines are skipped as eviction victims."""
        eng = _FakeEngine()
        eng._gen_gate = threading.Event()
        reg = _registry_with(eng)
        # Second idle model that could be evicted.
        idle = _FakeEngine()
        reg._entries["idle-model"] = EngineEntry(
            model_id="idle-model", engine=idle,
            vram_budget_bytes=1 << 20, checkpoint_path="",
            config_name="forgelm_v2", tokenizer_path="")

        done = []
        t = threading.Thread(
            target=lambda: done.append(reg.generate(MODEL, "x")),
            daemon=True)
        t.start()
        for _ in range(200):
            if reg._entries[MODEL].in_flight:
                break
            time.sleep(0.01)
        assert reg._entries[MODEL].in_flight == 1

        with reg._lock:
            with pytest.raises(VRAMBudgetExceeded):
                # More than the idle model's budget — would have to evict
                # the busy model too, which is refused.
                reg._evict_lru(10 ** 9, exclude=None)
        assert not eng.sleep_calls           # busy engine untouched
        assert idle.sleep_calls == [1]       # idle one was slept
        eng._gen_gate.set()
        t.join(timeout=5)
        assert reg._entries[MODEL].in_flight == 0

    def test_is_awake_reflects_external_sleep(self):
        """EC-161: engine.sleep() outside the registry shows in listing."""
        eng = _FakeEngine()
        reg = _registry_with(eng)
        eng.sleep(level=1)  # external sleep — registry never called
        models = reg.list_models()
        assert models[0]["awake"] is False

    def test_sleep_does_not_hold_registry_lock(self):
        """EC-199: a sleep blocked on _gen_lock doesn't freeze the registry."""
        eng = _FakeEngine()
        eng._gen_lock.acquire()          # simulate in-flight decode
        reg = _registry_with(eng)
        t = threading.Thread(
            target=lambda: reg.sleep(MODEL), daemon=True)
        t.start()
        # Registry must stay responsive while sleep waits on _gen_lock.
        assert reg.list_models()  # would deadlock pre-fix
        eng._gen_lock.release()
        t.join(timeout=5)
        assert eng.sleep_calls == [1]


# ── Server observability / auth / utility endpoints ─────────────────────────

class TestServerSurface:
    def test_request_id_assigned_and_echoed(self, monkeypatch):
        """EC-086/065: X-Request-ID on every response, inbound honored."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()
        c = TestClient(server.app)
        r = c.get("/health")
        assert r.status_code == 200
        rid = r.headers.get("x-request-id")
        assert rid
        r2 = c.get("/health", headers={"x-request-id": "req-abc"})
        assert r2.headers["x-request-id"] == "req-abc"

    def test_health_deep(self, monkeypatch):
        """EC-065/072: per-model state + public queue counter."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()
        c = TestClient(server.app)
        body = c.get("/health").json()
        assert body["status"] == "ok"
        assert body["batch_queue_pending"] == 0
        assert body["models"][0]["id"] == MODEL
        assert "uptime_s" in body

    def test_metrics(self, monkeypatch):
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()
        c = TestClient(server.app)
        c.get("/health")
        body = c.get("/metrics").json()
        assert body["requests_total"] >= 2
        assert MODEL in body["models"]
        assert body["batch_queue_pending"] == 0

    def test_version(self, monkeypatch):
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()
        c = TestClient(server.app)
        body = c.get("/version").json()
        assert body["server"] == "forgeai"

    def test_api_key_gate(self, monkeypatch):
        """EC-058: FORGE_API_KEY gates mutating /v1 methods only."""
        monkeypatch.setenv("FORGE_API_KEY", "secret")
        server, _ = _server()
        c = TestClient(server.app)
        payload = {"model": MODEL, "text": "hi"}
        assert c.post("/v1/tokenize", json=payload).status_code == 401
        assert c.get("/health").status_code == 200        # GET free
        r = c.post("/v1/tokenize", json=payload,
                   headers={"authorization": "Bearer secret"})
        assert r.status_code == 200
        r = c.post("/v1/tokenize", json=payload,
                   headers={"authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_tokenize_detokenize(self, monkeypatch):
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()
        c = TestClient(server.app)
        r = c.post("/v1/tokenize", json={"model": MODEL, "text": "ab"})
        assert r.status_code == 200
        assert r.json()["count"] == 2
        r = c.post("/v1/detokenize",
                   json={"model": MODEL, "tokens": [97, 98]})
        assert r.status_code == 200
        assert r.json()["text"] == "ab"

    def test_kv_cache_type_via_hotswap(self, monkeypatch):
        """EC-062: route through hotswap, never bare activate()."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, eng = _server()
        c = TestClient(server.app)
        r = c.post("/v1/engine/kv-cache-type", json={
            "model": MODEL, "cache_type_k": "q8_0",
            "cache_type_v": "q8_0"})
        assert r.status_code == 200
        assert eng.hotswap.apply_calls == 1
        assert eng.hotswap.applied[-1] == {"kv_bits": 8}
        assert eng.activate_calls == []  # flag-wiping activate avoided

    def test_kv_cache_type_mismatched_v_rejected(self, monkeypatch):
        """EC-062: cache_type_v != cache_type_k is an honest 400."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, eng = _server()
        c = TestClient(server.app)
        r = c.post("/v1/engine/kv-cache-type", json={
            "model": MODEL, "cache_type_k": "q8_0",
            "cache_type_v": "f16"})
        assert r.status_code == 400
        assert eng.hotswap.apply_calls == 0

    def test_kv_cache_type_unknown_type_rejected(self, monkeypatch):
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()
        c = TestClient(server.app)
        r = c.post("/v1/engine/kv-cache-type", json={
            "model": MODEL, "cache_type_k": "turbo9",
            "cache_type_v": "turbo9"})
        assert r.status_code == 400

    def test_batch_prompt_cap(self, monkeypatch):
        """EC-218: oversized batches are rejected."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()
        c = TestClient(server.app)
        r = c.post("/v1/batch/completions", json={
            "model": MODEL,
            "prompts": ["p"] * (_MAX_BATCH_PROMPTS + 1)})
        assert r.status_code == 400

    def test_batch_completions_ok(self, monkeypatch):
        """EC-218: results returned with real (non-word) token counts."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()
        c = TestClient(server.app)
        r = c.post("/v1/batch/completions", json={
            "model": MODEL, "prompts": ["p1", "p2"]})
        assert r.status_code == 200
        body = r.json()
        assert body["results"] == ["r0", "r1"]
        assert body["total_tokens"] == 4  # 2 chars each via fake tokenizer


# ── Streaming internals ─────────────────────────────────────────────────────

class TestStreaming:
    def test_aiter_stream_runs_off_loop(self, monkeypatch):
        """EC-057: the blocking generator iterates on a worker thread."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, eng = _server()
        caller = threading.get_ident()

        async def run():
            return [c async for c in server._aiter_stream(
                lambda: eng.generate_stream("x"))]

        assert asyncio.run(run()) == ["a", "b", "c"]
        assert eng.stream_thread is not None
        assert eng.stream_thread != caller

    def test_aiter_stream_propagates_errors(self, monkeypatch):
        """EC-057: engine errors reach the SSE consumer, never hang."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()

        def bad_gen():
            yield "partial"
            raise RuntimeError("decode boom")

        async def run():
            out = []
            with pytest.raises(RuntimeError, match="decode boom"):
                async for c in server._aiter_stream(bad_gen):
                    out.append(c)
            return out

        assert asyncio.run(run()) == ["partial"]

    def test_aiter_stream_break_frees_gen_lock(self, monkeypatch):
        """EC-066: abandoning the stream closes the generator → _gen_lock
        is released at the next decode boundary."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        eng = _FakeEngine()
        eng._stream_gate = threading.Event()  # never set: stall mid-stream
        server, _ = _server(engine=eng)

        async def run():
            async for _c in server._aiter_stream(
                    lambda: eng.generate_stream("x")):
                break  # client goes away after the first chunk

        asyncio.run(run())
        # The worker is parked in the gate-wait inside the generator;
        # releasing the gate lets it observe cancel → gen.close() →
        # _gen_lock freed.
        eng._stream_gate.set()
        for _ in range(200):
            if eng._gen_lock.acquire(blocking=False):
                eng._gen_lock.release()
                return
            time.sleep(0.01)
        pytest.fail("engine _gen_lock still held after stream break")

    def test_stream_task_surfaces_producer_error(self, monkeypatch):
        """EC-073: a failed producer future can't hang the SSE consumer."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()
        q = asyncio.Queue()
        fut = Future()
        fut.set_exception(RuntimeError("dispatch boom"))

        async def run():
            return [c async for c in server._stream_task("t1", q, fut)]

        out = "".join(asyncio.run(run()))
        assert "dispatch boom" in out
        assert "[DONE]" in out

    def test_stream_task_delivers_chunks(self, monkeypatch):
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, _ = _server()

        async def run():
            q = asyncio.Queue()
            fut = Future()

            async def feed():
                await q.put("hello")
                await q.put(None)          # sentinel
                fut.set_result("hello")

            feeder = asyncio.ensure_future(feed())
            out = [c async for c in server._stream_task("t2", q, fut)]
            await feeder
            return out

        text = "".join(asyncio.run(run()))
        assert "hello" in text
        assert "[DONE]" in text


# ── Per-task serialization ──────────────────────────────────────────────────

class TestTaskSerialization:
    def test_task_lock_identity_and_bound(self):
        """EC-198: same task → same lock; map stays bounded."""
        server, _ = _server()
        assert server._task_lock("t") is server._task_lock("t")
        server._max_task_locks = 4
        for i in range(10):
            server._task_lock(f"t{i}")
        assert len(server._task_locks) <= server._max_task_locks

    def test_concurrent_messages_ordered(self, monkeypatch):
        """EC-198: two concurrent POSTs produce user/assistant pairs in
        order — not interleaved user,user,assistant,assistant."""
        monkeypatch.delenv("FORGE_API_KEY", raising=False)
        server, eng = _server()
        server.batch_queue.start()
        try:
            c = TestClient(server.app)
            tid = c.post("/v1/tasks", json={"model": MODEL}
                         ).json()["task_id"]

            errors = []

            def post(i):
                try:
                    r = c.post(f"/v1/tasks/{tid}/messages",
                               json={"content": f"m{i}"})
                    assert r.status_code == 200, r.text
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            threads = [threading.Thread(target=post, args=(i,))
                       for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
            assert not errors

            session = server.session_manager.get_task(tid)
            roles = [m["role"] for m in session.history]
            assert roles == ["user", "assistant", "user", "assistant"]
            # Each user message was rendered AFTER the prior pair landed.
            contents = [m["content"] for m in session.history]
            assert sorted(contents[0::2]) == ["m0", "m1"]
        finally:
            server.batch_queue.stop()

