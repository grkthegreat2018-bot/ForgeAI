"""Regression tests for the engine correctness/concurrency fix round.

Covers:
1.  Reentrant engine generation lock (_gen_lock) exists and is reentrant.
2.  BatchedDecoding passes a growing attention_mask through decode steps
    (pad-KV pollution fix) and applies top_p.
3.  cache_prompt_prefix reuses captured KV — no second full prefill.
4.  SemanticKVAnchors.find_reuse_point verifies text prefix equality.
5.  SessionCacheManager.continue_session falls back on KV-length mismatch.
6.  _ToolCallStreamFilter: incremental marker state machine (split markers,
    tail after end marker preserved).
7.  Conv-state boundary: prefix-cache hit must reproduce full-prefill
    logits (hybrid conv+attention model, CPU).
8.  CPU flash_attention fallback handles T<S delta prefill (causal mask
    anchoring fix).
9.  BatchQueue: boot-config grouping, per-request loop capture, scoped
    seeding (no global RNG leak).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import threading
from unittest.mock import MagicMock

import pytest
import torch

# ── Shared tiny-model fixture ─────────────────────────────────────────────


@pytest.fixture
def tiny_model():
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM

    cfg = get_config("forgelm_tiny")
    cfg.vocab_size = 256
    cfg.device = "cpu"
    cfg.dtype = "float32"
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old)
    model.eval()
    return model


# ── 1. Engine generation lock ─────────────────────────────────────────────


def _make_engine(tiny_model):
    from forge.engine.forge_engine import ForgeEngine

    tok = MagicMock()
    tok.eos_token_id = 0
    tok.return_value.input_ids = torch.tensor([[1, 2, 3, 4]])
    tok.decode = MagicMock(return_value="x")
    return ForgeEngine(tiny_model, tok, device="cpu")


class TestGenerationLock:
    def test_lock_is_reentrant(self, tiny_model):
        """_gen_lock must be an RLock — generate_with_tools() -> generate()
        nests, so a plain Lock would self-deadlock."""
        eng = _make_engine(tiny_model)
        assert isinstance(eng._gen_lock, type(threading.RLock()))
        eng._gen_lock.acquire()
        eng._gen_lock.acquire()  # must not block — reentrant
        eng._gen_lock.release()
        eng._gen_lock.release()

    def test_stream_generator_holds_lock(self, tiny_model):
        """generate_stream must hold _gen_lock for the whole iteration —
        checked from a second thread (RLock re-acquires in the owner)."""
        eng = _make_engine(tiny_model)
        acquired = []

        def try_acquire():
            acquired.append(eng._gen_lock.acquire(blocking=False))
            if acquired[-1]:
                eng._gen_lock.release()

        gen = eng.generate_stream("x", max_new_tokens=2)
        next(gen)  # lock now held by this thread for the iteration
        t = threading.Thread(target=try_acquire)
        t.start()
        t.join()
        assert acquired == [False], "lock was free during iteration"
        gen.close()
        t2 = threading.Thread(target=try_acquire)
        t2.start()
        t2.join()
        assert acquired == [False, True], "lock held after close()"


# ── 2. BatchedDecoding mask + top_p ──────────────────────────────────────


class _RecordingModel(torch.nn.Module):
    """Wrap a real model, recording every forward's kwargs."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = []

    def forward(self, idx, **kwargs):
        self.calls.append({"T": idx.shape[1], **kwargs})
        return self.inner(idx, **kwargs)


class TestBatchedDecoding:
    def test_attention_mask_extended_through_decode(self, tiny_model):
        """Decode steps must pass a mask covering cached+new positions —
        otherwise pad-token KV pollutes attention for unequal prompts."""
        from forge.engine.batched_decoding import BatchedDecoding

        rec = _RecordingModel(tiny_model)
        dec = BatchedDecoding(eos_token_id=255)
        p1 = torch.tensor([[5, 6, 7, 8]])  # len 4
        p2 = torch.tensor([[9, 10]])  # len 2 (padded)
        dec.generate_batch(rec, [p1, p2], [3, 3], [0.0, 0.0], [1.0, 1.0])
        decode_calls = [c for c in rec.calls if c["T"] == 1]
        assert decode_calls, "no decode steps ran"
        for c in decode_calls:
            mask = c.get("attention_mask")
            assert mask is not None, "decode step missing attention_mask"
            kv_len = c["past_key_values"][2][0].shape[2]
            # mask covers cached KV + the new token column
            assert mask.shape[1] == kv_len + 1
            # pad positions of the short prompt stay masked forever
            assert not bool(mask[1, 0])
            assert not bool(mask[1, 1])

    def test_top_p_applied(self, tiny_model):
        """top_p < 1.0 must actually filter — previously plumbed but ignored."""
        from forge.engine.batched_decoding import BatchedDecoding

        class Fixed(torch.nn.Module):
            """Always emits the same logit profile: token 10 dominant, a
            long tail of near-equal tokens summing past any top_p < ~0.9."""

            def forward(self, idx, **kw):
                B, T = idx.shape
                logits = torch.zeros(B, T, 256)
                logits[..., 10] = 10.0
                logits[..., 20:200] = 1.0  # uniform tail
                past = kw.get("past_key_values")
                if past is None:
                    past = [None] * 4
                return logits, None, past

        dec = BatchedDecoding(eos_token_id=255)
        out = dec.generate_batch(
            Fixed(),
            [torch.tensor([[1, 2]])],
            [5],
            [1.0],  # temperature > 0 → sampling
            [0.05],  # top_p keeps only token 10
            seed_list=[1234],
        )
        gen = out[0][0]
        # gen[0:2] is the prompt; every generated token must be id 10.
        assert (gen[2:] == 10).all(), f"top_p ignored: {gen}"


# ── 3. cache_prompt_prefix reuses captured KV ────────────────────────────


class TestCachePromptPrefix:
    def test_no_second_prefill(self, tiny_model):
        """cache_prompt_prefix must store the captured KV, not re-run the
        model (the old path paid a second full prefill per prompt)."""
        from forge.engine.prefix_cache import ChunkedPrefixCache, cache_prompt_prefix

        engine = MagicMock()
        engine.model = tiny_model
        engine._prefix_cache = ChunkedPrefixCache(max_entries=8)

        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]])

        with torch.inference_mode():
            _, past_kv = __import__("forge.model_loader", fromlist=["unpack_output_with_kv"]).unpack_output_with_kv(
                tiny_model(ids, use_cache=True)
            )
        # Model snapshot is produced automatically at prefill end.
        assert getattr(tiny_model, "_last_prefill_recurrent", None)

        calls = []
        orig = tiny_model.forward

        def counting(*a, **kw):
            calls.append(1)
            return orig(*a, **kw)

        tiny_model.forward = counting
        try:
            cache_prompt_prefix(engine, ids, past_kv)
        finally:
            tiny_model.forward = orig
        assert not calls, "cache_prompt_prefix re-ran the model"

        # Entry stored and reusable
        hit = engine._prefix_cache.lookup_longest_prefix(ids)
        assert hit is not None
        assert hit[0] == ids.shape[1]
        assert hit[2] is not None  # conv snapshot stored

    def test_skips_when_no_kv(self, tiny_model):
        """past_kv=None (streaming strategies) → skip, never re-prefill."""
        from forge.engine.prefix_cache import ChunkedPrefixCache, cache_prompt_prefix

        engine = MagicMock()
        engine.model = tiny_model
        engine._prefix_cache = ChunkedPrefixCache()
        calls = []
        orig = tiny_model.forward
        tiny_model.forward = lambda *a, **kw: (calls.append(1), orig(*a, **kw))[1]
        try:
            cache_prompt_prefix(engine, torch.tensor([[1] * 20]), None)
        finally:
            tiny_model.forward = orig
        assert not calls


# ── 4. Semantic anchor text verification ─────────────────────────────────


class TestSemanticAnchors:
    def test_reuse_requires_matching_text(self):
        from forge.engine.prefix_cache import SemanticKVAnchors

        anchors = SemanticKVAnchors()
        src = "Hello, how are you? Tell me about X"
        anchors.save_anchor(token_pos=6, text_pos=17, anchor_type="paragraph", kv_state=MagicMock(), source_text=src)
        # Same prefix → hit
        hit = anchors.find_reuse_point("Hello, how are you? And then Y")
        assert hit is not None
        # Different text at same position → miss (previously position-only)
        miss = anchors.find_reuse_point("Hello, completely different text")
        assert miss is None

    def test_legacy_anchor_without_hash_skipped(self):
        from forge.engine.prefix_cache import SemanticKVAnchors

        anchors = SemanticKVAnchors()
        anchors.save_anchor(token_pos=6, text_pos=5, anchor_type="x", kv_state=MagicMock())
        assert anchors.find_reuse_point("12345_rest") is None


# ── 5. Session KV-length verification ────────────────────────────────────


class TestSessionKVVerification:
    def _engine(self):
        eng = MagicMock()
        eng.device = torch.device("cpu")
        eng.tokenizer = MagicMock(return_value=MagicMock(input_ids=torch.tensor([[9, 9, 9]])))
        eng._log = MagicMock()
        return eng

    def test_kv_mismatch_falls_back(self):
        from forge.engine.session_cache import SessionCacheManager

        eng = self._engine()
        scm = SessionCacheManager(eng)
        scm.begin_session("s1")
        sess = scm._sessions["s1"]
        sess.token_ids = [1, 2, 3, 4, 5]
        # KV only covers 2 tokens (e.g. eviction shrank it) — must not be
        # trusted as covering all 5 cached tokens.
        k = torch.zeros(1, 8, 2, 16)
        sess.past_kv = [(k, k.clone()), None]
        _ids, past_kv, cached_len, _recurrent = scm.continue_session("s1", "hi")
        assert cached_len == 0
        assert past_kv is None
        eng._log.assert_called()  # warn logged

    def test_kv_match_prefills_delta(self):
        from forge.engine.session_cache import SessionCacheManager

        eng = self._engine()
        scm = SessionCacheManager(eng)
        scm.begin_session("s1")
        sess = scm._sessions["s1"]
        sess.token_ids = [1, 2, 3]
        k = torch.zeros(1, 8, 3, 16)
        sess.past_kv = [(k, k.clone()), None]
        _ids, past_kv, cached_len, _recurrent = scm.continue_session("s1", "hi")
        assert cached_len == 3
        assert past_kv is not None


# ── 6. Streaming marker filter ────────────────────────────────────────────


class TestToolCallStreamFilter:
    def _filter(self):
        from forge.engine.forge_server import _ToolCallStreamFilter

        return _ToolCallStreamFilter()

    def test_plain_text_passthrough(self):
        f = self._filter()
        assert f.feed("hello ") == "hello "
        assert f.feed("world") == "world"
        assert f.flush() == ""

    def test_tool_call_hidden(self):
        f = self._filter()
        out = f.feed('pre <tool_call>{"a":1}</tool_call> post')
        assert out == "pre  post"
        assert f.tool_calls_seen == 1

    def test_split_markers_across_chunks(self):
        f = self._filter()
        chunks = ["hel", "lo <to", "ol_ca", 'll>{"x":', "1}</tool", "_call>", "tail"]
        visible = "".join(f.feed(c) for c in chunks) + f.flush()
        assert visible == "hello tail"
        assert f.tool_calls_seen == 1

    def test_tail_after_end_marker_not_dropped(self):
        """The chunk containing </tool_call> previously lost trailing text."""
        f = self._filter()
        out = f.feed("<tool_call>{}</tool_call>AND MORE")
        assert out == "AND MORE"

    def test_marker_lookalike_text(self):
        f = self._filter()
        out = f.feed("a <tool_cally> b")
        assert out == "a <tool_cally> b"

    def test_unterminated_tool_call_hides_tail(self):
        f = self._filter()
        out = f.feed('ok <tool_call>{"x":1}')
        out += f.flush()
        assert out == "ok "


# ── 7. Conv-state boundary equivalence ───────────────────────────────────


class TestConvStateBoundary:
    def test_prefix_hit_matches_full_prefill(self, tiny_model):
        """Cached-prefix continuation must produce identical logits to a
        fresh full prefill — conv boundary state restored via snapshot."""
        from forge.engine.prefix_cache import apply_recurrent_state_prefix
        from forge.model_loader import unpack_output_with_kv

        torch.manual_seed(0)
        full = torch.randint(0, 256, (1, 12))
        prompt, suffix = full[:, :8], full[:, 8:]

        with torch.inference_mode():
            logits_full, _ = unpack_output_with_kv(tiny_model(full, use_cache=True))

        with torch.inference_mode():
            _, past = unpack_output_with_kv(tiny_model(prompt, use_cache=True))
        snap = tiny_model._last_prefill_recurrent
        assert snap, "prefill did not record recurrent snapshot"

        # Simulate an intervening sequence clobbering live conv state.
        with torch.inference_mode():
            tiny_model(torch.randint(0, 256, (1, 5)), use_cache=True)

        apply_recurrent_state_prefix(tiny_model, snap)
        with torch.inference_mode():
            logits_cached, _ = unpack_output_with_kv(tiny_model(suffix, past_key_values=past, use_cache=True))

        torch.testing.assert_close(logits_cached, logits_full[:, 8:], atol=1e-5, rtol=1e-4)

    def test_prefix_hit_without_snapshot_is_corrupt(self, tiny_model):
        """Control: WITHOUT the snapshot restore, cached-prefix logits must
        differ (zero-pad conv boundary) — proves the test above actually
        exercises the fix."""
        from forge.model_loader import unpack_output_with_kv

        torch.manual_seed(0)
        full = torch.randint(0, 256, (1, 12))
        prompt, suffix = full[:, :8], full[:, 8:]
        with torch.inference_mode():
            logits_full, _ = unpack_output_with_kv(tiny_model(full, use_cache=True))
            _, past = unpack_output_with_kv(tiny_model(prompt, use_cache=True))
            # clobber conv state with an unrelated sequence
            tiny_model(torch.randint(0, 256, (1, 5)), use_cache=True)
            logits_stale, _ = unpack_output_with_kv(tiny_model(suffix, past_key_values=past, use_cache=True))
        assert not torch.allclose(logits_stale, logits_full[:, 8:], atol=1e-5, rtol=1e-4)


# ── 8. CPU flash_attention delta-prefill mask ────────────────────────────


class TestFlashAttentionFallback:
    def test_causal_mask_with_past(self):
        """T<S (query shorter than keys) must anchor the causal mask at the
        bottom-right so the delta sees all of the past."""
        from forge.model_loader import flash_attention

        torch.manual_seed(0)
        q = torch.randn(1, 2, 3, 8)  # 3 new queries
        k = torch.randn(1, 2, 7, 8)  # 7 keys (4 past + 3 new)
        v = torch.randn(1, 2, 7, 8)
        out = flash_attention(q, k, v, is_causal=True)
        assert out.shape == (1, 2, 3, 8)
        # First query row attends to keys 0..4 (past 4 + itself) — verify by
        # brute-force reference.
        scores = torch.matmul(q, k.transpose(-2, -1)) / (8**0.5)
        q_pos = torch.arange(7 - 3, 7)
        mask = torch.arange(7).unsqueeze(0) <= q_pos.unsqueeze(1)
        ref = torch.softmax(scores.masked_fill(~mask, float("-inf")), -1) @ v
        torch.testing.assert_close(out, ref)


# ── 9. BatchQueue fixes ──────────────────────────────────────────────────


class TestBatchQueue:
    def test_boot_key_groups_configs(self):
        """Requests with different boot configs must not share a batch."""
        from forge.engine.session_manager import BatchQueue, SessionManager, TaskBootConfig

        sm = SessionManager()
        bq = BatchQueue(registry=MagicMock(), session_manager=sm)
        t1 = sm.create_task("m", boot_config=TaskBootConfig(kv_cache="paged"))
        t2 = sm.create_task("m", boot_config=TaskBootConfig(kv_cache="snapkv"))
        r1 = MagicMock(task_id=t1)
        r2 = MagicMock(task_id=t2)
        r3 = MagicMock(task_id=None)
        assert bq._boot_key(r1) != bq._boot_key(r2)
        assert bq._boot_key(r3) == ()

    def test_push_stream_without_loop_falls_back(self):
        from concurrent.futures import Future

        from forge.engine.session_manager import BatchQueue, PendingRequest, SessionManager

        bq = BatchQueue(registry=MagicMock(), session_manager=SessionManager())
        req = PendingRequest(
            task_id=None,
            model_id="m",
            prompt="p",
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            top_k=80,
            repetition_penalty=1.0,
            seed=None,
            stop=None,
            stream=True,
            future=Future(),
            stream_queue=MagicMock(),
            loop=None,
        )
        # No captured loop → returns False instead of crashing in thread
        assert bq._push_stream(req, "x") is False

    def test_submit_captures_running_loop(self):
        import asyncio

        from forge.engine.session_manager import BatchQueue, SessionManager

        bq = BatchQueue(registry=MagicMock(), session_manager=SessionManager())

        async def _submit():
            q = asyncio.Queue()
            return bq.submit(None, "m", "p", stream=True, stream_queue=q)

        asyncio.run(_submit())
        # The pending request captured the caller's event loop — the
        # dispatcher thread can now run_coroutine_threadsafe against it
        # instead of crashing on asyncio.get_event_loop().
        req = bq._pending[0]
        assert req.loop is not None
