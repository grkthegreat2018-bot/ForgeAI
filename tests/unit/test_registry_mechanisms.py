"""Phase 6b hosting-parity mechanism tests.

Covers: HF-repo JIT pull (register_hf_model + `model`-field auto-register +
offline guard + revision pinning), keep_alive TTL eviction
(evict_expired), cooperative abort (request_cancel / _gen_cancel),
BatchQueue priority ordering + queue timeout + _MAX_PENDING
backpressure + queue abort, and logits_processor forwarding on the
non-stream generate path.

All CPU-only with fake engines — no checkpoint loads, no network.
"""

import socket
import threading
import time
import urllib.request
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import forge.engine.forge_loader as forge_loader
import forge.engine.model_registry as registry_mod
from forge.engine.engine_common import SamplingParams
from forge.engine.engine_generation import _GenerationMixin
from forge.engine.model_registry import EngineEntry, ModelRegistry
from forge.engine.session_manager import (
    _MAX_PENDING,
    BatchQueue,
    PendingRequest,
    SessionManager,
)

MODEL = "forgelm-v2-jamba"


# ── Fakes ───────────────────────────────────────────────────────────────────


class _FakeTokenizer:
    eos_token_id = 7

    def encode(self, text):
        return [ord(c) % 128 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, return_tensors=None, **_kw):
        return {"input_ids": torch.tensor([[ord(c) % 128 for c in text]])}


class _FakeEngine:
    """Minimal engine surface for registry tests (mirrors
    test_forge_server_fixes stubs)."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.device = torch.device("cpu")
        self._gen_lock = threading.RLock()
        self._awake = True
        self.total_tokens_generated = 0
        self.generation_count = 0
        self.sleep_calls = []
        self.generate_kwargs = {}
        self._gen_cancel = False
        self.cancel_calls = 0

    # Phase 6b abort surface — the decode loops check _gen_cancel and
    # break with partial output; flag cleared at call start.
    def request_cancel(self):
        self._gen_cancel = True
        self.cancel_calls += 1

    def cancel_generation(self):
        self.request_cancel()

    def generate(self, prompt, max_new_tokens=100, **kw):
        self.generate_kwargs = kw
        self._gen_cancel = False  # clear at call start
        toks = []
        for _ in range(3):
            if getattr(self, "_gen_cancel", False):
                break  # cooperative abort → partial text
            toks.append("x")
        self.total_tokens_generated += len(toks)
        self.generation_count += 1
        return "".join(toks) or "out"

    def generate_stream(self, prompt, **kw):
        yield "a"
        yield "b"

    def generate_batch(self, prompts, **kw):
        return [f"r{i}" for i in range(len(prompts))]

    def sleep(self, level=1):
        self.sleep_calls.append(level)
        self._awake = False

    def wake(self):
        self._awake = True

    quantize = None
    acceleration = None


def _registry_with(engine, model_id=MODEL, **entry_kw):
    reg = ModelRegistry(device="cpu")
    reg._entries[model_id] = EngineEntry(
        model_id=model_id,
        engine=engine,
        vram_budget_bytes=1 << 20,
        checkpoint_path="",
        config_name="forgelm_v2",
        tokenizer_path="",
        **entry_kw,
    )
    return reg


# ── HF JIT load ─────────────────────────────────────────────────────────────


class TestHFJitLoad:
    def test_register_hf_model_offline_raises(self, monkeypatch):
        """HF_HUB_OFFLINE=1 → ValueError before any network access."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        reg = ModelRegistry(device="cpu")
        with pytest.raises(ValueError, match="offline"):
            reg.register_hf_model("m", "user/repo")

    def test_download_hf_checkpoint_offline_raises(self, monkeypatch):
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        with pytest.raises(ValueError, match="offline"):
            forge_loader.ForgeLoader().download_hf_checkpoint("user/repo")

    def test_download_hf_file_offline_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with pytest.raises(ValueError, match="offline"):
            forge_loader.ForgeLoader._download_hf_file(
                "user/repo", tmp_path / "f.gguf")

    def test_download_hf_file_uses_revision(self, monkeypatch, tmp_path):
        """Per-call revision pins resolve/<rev>/; FORGE_HF_REVISION is the
        global default and loses to an explicit arg."""
        urls = []
        monkeypatch.setattr(
            urllib.request, "urlretrieve",
            lambda url, dest: (urls.append(url), open(dest, "wb").close()))
        # SSRF check does a real getaddrinfo — stub it to a public IP so
        # the test never touches DNS/network.
        monkeypatch.setattr(
            socket, "getaddrinfo",
            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])

        monkeypatch.setenv("FORGE_HF_REVISION", "envrev")
        forge_loader.ForgeLoader._download_hf_file(
            "user/repo", tmp_path / "w.safetensors",
            filename="model.safetensors")
        assert urls[-1].endswith(
            "/user/repo/resolve/envrev/model.safetensors")

        forge_loader.ForgeLoader._download_hf_file(
            "user/repo", tmp_path / "w2.safetensors",
            filename="model.safetensors", revision="pinned")
        assert urls[-1].endswith(
            "/user/repo/resolve/pinned/model.safetensors")

        monkeypatch.delenv("FORGE_HF_REVISION")
        forge_loader.ForgeLoader._download_hf_file(
            "user/repo", tmp_path / "w3.safetensors",
            filename="model.safetensors")
        assert urls[-1].endswith(
            "/user/repo/resolve/main/model.safetensors")

    def test_hf_select_files_prefers_safetensors(self):
        files = ["config.json", "model.safetensors",
                 "big-Q4_K_M.gguf", "tokenizer.json", "sub/dir/file.json"]
        keep = forge_loader._hf_select_files("u/r", files)
        assert "model.safetensors" in keep
        assert "config.json" in keep
        assert not any(f.endswith(".gguf") for f in keep)
        assert "sub/dir/file.json" not in keep  # top-level only

    def test_hf_select_files_picks_single_gguf(self):
        files = ["m-Q4_K_M.gguf", "m-F16.gguf", "README.md"]
        keep = forge_loader._hf_select_files("u/r", files)
        assert keep == ["m-Q4_K_M.gguf"]

    def test_register_hf_model_routes_to_register(
            self, monkeypatch, tmp_path):
        """Downloads land in a local path, then register() takes over —
        with keep_alive_s + an explicit dir-size budget."""
        ckpt = tmp_path / "weights.safetensors"
        ckpt.write_bytes(b"x" * 2048)
        seen = {}

        def fake_download(self, repo_id, revision=None, dest_dir=None):
            seen["repo"] = repo_id
            seen["revision"] = revision
            return ckpt

        monkeypatch.setattr(
            forge_loader.ForgeLoader, "download_hf_checkpoint",
            fake_download)

        reg = ModelRegistry(device="cpu")
        calls = []

        def fake_register(model_id, **kw):
            calls.append(kw)
            reg._entries[model_id] = EngineEntry(
                model_id=model_id, engine=_FakeEngine(),
                vram_budget_bytes=1 << 20, checkpoint_path=str(ckpt),
                config_name="forgelm_v2", tokenizer_path="",
                keep_alive_s=kw.get("keep_alive_s"))
            return reg._entries[model_id].engine

        monkeypatch.setattr(reg, "register", fake_register)
        out = reg.register_hf_model(
            "my-model", "user/repo", revision="r1", keep_alive_s=30.0)
        assert out == "my-model"
        assert seen == {"repo": "user/repo", "revision": "r1"}
        assert calls[0]["checkpoint"] == str(ckpt)
        assert calls[0]["keep_alive_s"] == 30.0
        assert calls[0]["vram_budget_gb"] > 0
        assert reg._entries["my-model"].keep_alive_s == 30.0

    def test_generate_auto_registers_hf_repo_id(
            self, monkeypatch, tmp_path):
        """`model`-field JIT: an unknown owner/name id is pulled and then
        served — no explicit register call needed."""
        ckpt = tmp_path / "w.safetensors"
        ckpt.write_bytes(b"x" * 1024)
        monkeypatch.setattr(
            forge_loader.ForgeLoader, "download_hf_checkpoint",
            lambda self, repo, revision=None, dest_dir=None: ckpt)
        monkeypatch.setattr(
            registry_mod.ForgeEngine, "from_checkpoint",
            classmethod(lambda cls, **kw: _FakeEngine()))

        reg = ModelRegistry(device="cpu")
        out = reg.generate("user/repo", "hi")
        assert out == "xxx"
        assert "user/repo" in reg._entries

    def test_generate_unknown_nonrepo_id_still_keyerror(self):
        reg = ModelRegistry(device="cpu")
        with pytest.raises(KeyError, match="not-a-model"):
            reg.generate("not-a-model", "hi")
        with pytest.raises(KeyError):
            list(reg.generate_stream("also/many/slashes", "hi"))

    def test_load_huggingface_offline_guard(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with pytest.raises(ValueError, match="offline"):
            forge_loader.ForgeLoader()._load_huggingface(
                "user/repo", "forgelm_v2")


# ── keep_alive TTL ──────────────────────────────────────────────────────────


class TestKeepAliveTTL:
    def test_register_stores_keep_alive(self, monkeypatch, tmp_path):
        ckpt = tmp_path / "w.safetensors"
        ckpt.write_bytes(b"x" * 16)
        monkeypatch.setattr(
            registry_mod.ForgeEngine, "from_checkpoint",
            classmethod(lambda cls, **kw: _FakeEngine()))
        reg = ModelRegistry(device="cpu")
        reg.register("m", checkpoint=str(ckpt), config_name="c",
                     keep_alive_s=42.0)
        assert reg._entries["m"].keep_alive_s == 42.0

    def test_evict_expired_sleeps_stale_model(self):
        eng = _FakeEngine()
        reg = _registry_with(eng, keep_alive_s=10.0)
        entry = reg._entries[MODEL]
        entry.last_used = time.time() - 100
        slept = reg.evict_expired()
        assert slept == [MODEL]
        assert eng.sleep_calls == [1]
        assert entry.is_awake is False

    def test_evict_expired_skips_fresh_and_unpinned(self):
        fresh = _FakeEngine()
        reg = _registry_with(fresh, keep_alive_s=1000.0)
        reg._entries[MODEL].last_used = time.time()
        reg._entries["no-ttl"] = EngineEntry(
            model_id="no-ttl", engine=_FakeEngine(),
            vram_budget_bytes=1 << 20, checkpoint_path="",
            config_name="c", tokenizer_path="",
            keep_alive_s=None)
        reg._entries["no-ttl"].last_used = time.time() - 10**9
        assert reg.evict_expired() == []
        assert fresh.sleep_calls == []

    def test_evict_expired_skips_in_flight(self):
        eng = _FakeEngine()
        reg = _registry_with(eng, keep_alive_s=0.5)
        entry = reg._entries[MODEL]
        entry.last_used = time.time() - 100
        entry.in_flight = 1
        assert reg.evict_expired() == []
        assert eng.sleep_calls == []

    def test_evict_expired_now_arg(self):
        eng = _FakeEngine()
        reg = _registry_with(eng, keep_alive_s=10.0)
        entry = reg._entries[MODEL]
        entry.last_used = 500.0
        assert reg.evict_expired(now=505.0) == []
        assert reg.evict_expired(now=600.0) == [MODEL]


# ── Cooperative abort ───────────────────────────────────────────────────────


class _TinyModel:
    """Forward returning (logits, loss, presents) for _decode_tokens."""

    def __init__(self, vocab=16):
        self._forge_last_kv = None
        self._forge_last_logits = None
        self.vocab = vocab
        self.calls = 0

    def __call__(self, ids, past_key_values=None, use_cache=False, **_kw):
        self.calls += 1
        logits = torch.zeros(1, 1, self.vocab)
        logits[0, 0, (self.calls % self.vocab)] = 10.0
        return logits, None, []


class _TinyEngine(_GenerationMixin):
    """Real _decode_tokens loop over a stub model — CPU-only."""

    def __init__(self):
        self._gen_cancel = False
        self._suffix_mem = None
        self.kv_cache = None
        self.tokenizer = _FakeTokenizer()
        self.model = _TinyModel()
        self._recovery = SimpleNamespace(enabled=False)
        self.device = torch.device("cpu")

    def _suffix_mem_for(self, params):
        return None


class TestAbort:
    def test_abort_signals_in_flight_and_counts(self):
        eng = _FakeEngine()
        reg = _registry_with(eng)
        reg._entries[MODEL].in_flight = 2
        n = reg.abort()
        assert n == 2
        assert eng._gen_cancel is True
        assert eng.cancel_calls == 1

    def test_abort_model_scoped_and_unknown_id(self):
        a, b = _FakeEngine(), _FakeEngine()
        reg = _registry_with(a, model_id="m1")
        reg._entries["m2"] = EngineEntry(
            model_id="m2", engine=b, vram_budget_bytes=1 << 20,
            checkpoint_path="", config_name="c", tokenizer_path="")
        reg._entries["m1"].in_flight = 1
        reg._entries["m2"].in_flight = 1
        assert reg.abort("m1") == 1
        assert a._gen_cancel is True
        assert b._gen_cancel is False
        with pytest.raises(KeyError):
            reg.abort("ghost")

    def test_decode_tokens_loop_honors_cancel(self):
        """The real _decode_tokens loop breaks early on _gen_cancel,
        keeping the partial sequence."""
        eng = _TinyEngine()
        logits = torch.zeros(1, 1, 16)
        logits[0, 0, 3] = 10.0
        params = SamplingParams(max_new_tokens=50)
        gen_ids: list[int] = []
        it = eng._decode_tokens(logits, [], params, set(), gen_ids)
        next(it)  # token 1
        next(it)  # token 2
        eng.request_cancel()
        with pytest.raises(StopIteration):
            next(it)
        assert len(gen_ids) == 2

    def test_fake_engine_partial_text_on_cancel(self):
        """Contract-level: a generate loop honoring _gen_cancel returns
        the partial text (cancel lands after 2 tokens)."""
        eng = _FakeEngine()
        # Same per-token check the real _decode_tokens loop performs.
        def gen(prompt, max_new_tokens=100, **kw):
            toks = []
            for i in range(5):
                if getattr(self, "_gen_cancel", False):
                    break
                toks.append("x")
                if i == 1:
                    self._gen_cancel = True  # abort lands mid-decode
            return "".join(toks)
        eng.generate = gen
        assert eng.generate("x") == "xx"

    def test_flag_set_before_decode_yields_empty(self):
        eng = _TinyEngine()
        eng.request_cancel()
        logits = torch.zeros(1, 1, 16)
        params = SamplingParams(max_new_tokens=10)
        ids: list[int] = []
        for _ in eng._decode_tokens(logits, [], params, set(), ids):
            pass
        assert ids == []


# ── Priority / timeout / backpressure ───────────────────────────────────────


class TestQueueMechanisms:
    def test_pending_request_fields(self):
        req = PendingRequest(
            task_id=None, model_id="m", prompt="p", max_tokens=4,
            temperature=0.0, top_p=1.0, top_k=80,
            repetition_penalty=1.0, seed=None, stop=None, stream=False,
            future=Future(), priority=3, timeout_s=1.5)
        assert req.priority == 3
        assert req.timeout_s == 1.5
        assert req.enqueued_at <= time.time()
        assert req.cancelled is False

    def test_submit_priority_ordering_lower_first(self):
        """LOWER priority number pops first; ties keep FIFO."""
        bq = BatchQueue(registry=MagicMock(), session_manager=SessionManager())
        bq.submit(None, "m", "a", priority=5)
        bq.submit(None, "m", "b", priority=0)
        bq.submit(None, "m", "c", priority=5)
        bq.submit(None, "m", "d", priority=0)
        assert [r.prompt for r in bq._pending] == ["b", "d", "a", "c"]

    def test_submit_max_pending_rejects(self):
        """Queue-full → RuntimeError backpressure (server maps to 503)."""
        bq = BatchQueue(registry=MagicMock(), session_manager=SessionManager())
        bq._max_pending = 2
        bq.submit(None, "m", "a")
        bq.submit(None, "m", "b")
        with pytest.raises(RuntimeError, match="full"):
            bq.submit(None, "m", "c")
        assert _MAX_PENDING == 256  # documented default

    def test_queue_timeout_expires_queued_request(self):
        """A request still queued past timeout_s dies with TimeoutError."""
        bq = BatchQueue(registry=MagicMock(), session_manager=SessionManager(),
                        batch_window_ms=10)
        bq.start()
        try:
            fut = bq.submit(None, "m", "a", timeout_s=0.0)
            with pytest.raises(TimeoutError, match="timed out"):
                fut.result(timeout=5)
            assert bq._pending == []
        finally:
            bq.stop()

    def test_sweep_expired_removes_timed_out(self):
        bq = BatchQueue(registry=MagicMock(), session_manager=SessionManager())
        stale = PendingRequest(
            task_id=None, model_id="m", prompt="old", max_tokens=4,
            temperature=0.0, top_p=1.0, top_k=80,
            repetition_penalty=1.0, seed=None, stop=None, stream=False,
            future=Future(), timeout_s=1.0,
            enqueued_at=time.time() - 100)
        fresh = PendingRequest(
            task_id=None, model_id="m", prompt="new", max_tokens=4,
            temperature=0.0, top_p=1.0, top_k=80,
            repetition_penalty=1.0, seed=None, stop=None, stream=False,
            future=Future(), timeout_s=1000.0)
        bq._pending.extend([stale, fresh])
        with bq._cv:
            bq._sweep_expired(time.time())
        assert bq._pending == [fresh]
        with pytest.raises(TimeoutError):
            stale.future.result(timeout=1)
        assert not fresh.future.done()

    def test_pending_cancel_flag(self):
        req = PendingRequest(
            task_id=None, model_id="m", prompt="p", max_tokens=4,
            temperature=0.0, top_p=1.0, top_k=80,
            repetition_penalty=1.0, seed=None, stop=None, stream=False,
            future=Future())
        assert req.cancel() is True
        assert req.cancelled is True
        assert req.future.cancelled()

    def test_batch_queue_abort_sweeps_pending(self):
        bq = BatchQueue(registry=MagicMock(), session_manager=SessionManager())
        f1 = bq.submit(None, "m1", "a")
        f2 = bq.submit(None, "m2", "b")
        f3 = bq.submit(None, "m1", "c")
        assert bq.abort("m1") == 2
        assert f1.cancelled() and f3.cancelled()
        assert not f2.done()
        assert [r.prompt for r in bq._pending] == ["b"]
        assert bq.abort() == 1
        assert f2.cancelled()
        assert bq._pending == []


# ── logits_processor on non-stream generate ─────────────────────────────────


class TestLogitsProcessor:
    def test_registry_forwards_logits_processor(self):
        eng = _FakeEngine()
        reg = _registry_with(eng)
        def proc(lg, ids):
            return lg
        reg.generate(MODEL, "hi", logits_processor=proc)
        assert eng.generate_kwargs.get("logits_processor") is proc

    def test_decode_tokens_applies_processor(self):
        """_decode_tokens invokes the processor before sampling each step."""
        eng = _TinyEngine()
        calls = []

        def proc(lg, ids):
            calls.append(len(ids))
            lg = lg.clone()
            lg[..., :] = -100.0
            lg[..., 5] = 100.0  # force token 5
            return lg

        logits = torch.zeros(1, 1, 16)
        params = SamplingParams(max_new_tokens=3)
        ids: list[int] = []
        for _ in eng._decode_tokens(logits, [], params, set(), ids,
                                    logits_processor=proc):
            pass
        assert ids == [5, 5, 5]
        assert calls == [0, 1, 2]
