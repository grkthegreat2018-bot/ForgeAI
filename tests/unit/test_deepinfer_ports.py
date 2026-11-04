"""DeepInfer-port unit tests: deterministic seq_seed sampling, deferred
readback recurrent rollback, confidence-scheduled draft verification, and
bounded replay for partial prefix-cache hits.

All tests run on CPU with stub models — no checkpoint required.
"""
from types import SimpleNamespace

import torch

from forge.engine.engine_common import (
    SamplingParams,
    _hash_multinomial,
    _hash_sample_positions,
)
from forge.engine.decoding import (
    _defer_restore_recurrent,
    _sample_from_logits,
    _spec_snapshot_recurrent,
    build_decoding,
)
from forge.engine.prefix_cache import (
    ChunkedPrefixCache,
    _bounded_replay_recurrent,
    _reset_live_recurrent,
)


# ── deterministic sampling (DeepInfer seq_seed port) ─────────────────────


def test_hash_uniforms_in_unit_range_and_deterministic():
    pos = torch.arange(128, dtype=torch.int64)
    u1 = _hash_sample_positions(pos, seed=42)
    u2 = _hash_sample_positions(pos, seed=42)
    assert torch.equal(u1, u2)                      # reproducible
    assert ((u1 >= 0.0) & (u1 < 1.0)).all()         # in [0,1)
    assert u1.unique().numel() > 64                  # not degenerate


def test_hash_uniforms_vary_with_seed_and_position():
    pos = torch.arange(64, dtype=torch.int64)
    a = _hash_sample_positions(pos, seed=1)
    b = _hash_sample_positions(pos, seed=2)
    c = _hash_sample_positions(pos + 1000, seed=1)
    assert not torch.equal(a, b)                    # different seed
    assert not torch.equal(a, c)                    # different position


def test_hash_multinomial_reproducible_and_cdf_correct():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(1, 512), dim=-1)
    pos = torch.tensor([17])
    picks = {_hash_multinomial(probs, 7, pos).item() for _ in range(8)}
    assert len(picks) == 1                          # same input -> same pick
    # One-hot distribution: CDF inversion must always return the argmax.
    onehot = torch.zeros(1, 16)
    onehot[0, 5] = 1.0
    for p in range(8):
        assert _hash_multinomial(
            onehot, 99, torch.tensor([p])).item() == 5


def test_hash_multinomial_explores_across_positions():
    # Uniform-ish probs over 16 tokens: over many positions a seed should
    # pick several distinct tokens, and two seeds should diverge somewhere.
    probs = torch.ones(1, 16) / 16.0
    pos = torch.arange(256, dtype=torch.int64)
    a = _hash_multinomial(probs.expand(1, -1), 1, pos[:1]).item()
    seen_a = {_hash_multinomial(probs, 1, torch.tensor([int(p)])).item()
              for p in pos}
    seen_b = {_hash_multinomial(probs, 2, torch.tensor([int(p)])).item()
              for p in pos}
    assert len(seen_a) > 4                          # explores the support
    assert len(seen_b) > 4
    assert isinstance(a, int)


def test_sample_from_logits_seed_deterministic():
    logits = torch.randn(1, 256)
    kw = dict(temperature=0.8, top_p=0.95, top_k=50,
              repetition_penalty=1.0, generated_ids=[],
              min_p=0.0, min_k=0.0, dry_penalties=None,
              top_p_fn=lambda l, p: l)
    t1 = _sample_from_logits(logits.clone(), seed=1234, sample_pos=9, **kw)
    t2 = _sample_from_logits(logits.clone(), seed=1234, sample_pos=9, **kw)
    assert torch.equal(t1, t2)
    # No seed -> stochastic path still works and returns (B,1).
    t3 = _sample_from_logits(logits.clone(), **kw)
    assert t3.shape == (1, 1)
    # Greedy ignores the seed entirely.
    g = _sample_from_logits(logits.clone(), temperature=0.0, top_p=1.0,
                            top_k=0, repetition_penalty=1.0,
                            generated_ids=[], min_p=0.0, min_k=0.0,
                            dry_penalties=None, top_p_fn=lambda l, p: l,
                            seed=1234, sample_pos=9)
    assert g.item() == logits.argmax(-1).item()


def test_sampling_params_seed_field():
    assert SamplingParams().seed is None
    assert SamplingParams(seed=7).seed == 7


# ── deferred readback recurrent rollback (DeepInfer ZOS, hybrid-safe) ────


class _ConvHolder(SimpleNamespace):
    pass


def _hybrid_stub():
    holder = _ConvHolder(
        _conv_state=torch.randn(2, 8, 3),
        _conv_state_reset=False,
        _conv_state_prefix=None,
        _ssm_state=None,
        _ssm_state_real=None,
        _ssm_state_imag=None,
        _state_reset=False,
    )
    block = SimpleNamespace(attn=holder, _kda=None, _gdn2=None,
                            _forge_hybrid_ssm=None)
    return SimpleNamespace(blocks=[block]), holder


def test_defer_restore_recurrent_inplace_preserves_alias():
    model, holder = _hybrid_stub()
    buf = holder._conv_state
    snap = _spec_snapshot_recurrent(model)
    # Speculative step mutates the live buffer in place.
    holder._conv_state.copy_(torch.randn_like(holder._conv_state))
    _defer_restore_recurrent(snap)
    # Values restored AND the aliased buffer object is the same storage —
    # captured graphs / _conv_state_buf keep pointing at it.
    assert holder._conv_state is buf
    saved = [v for _, a, v in snap if a == "_conv_state"][0]
    assert torch.equal(holder._conv_state, saved)


def test_defer_restore_recurrent_none_noop():
    _defer_restore_recurrent(None)   # must not raise
    _defer_restore_recurrent([])


# ── bounded replay on partial prefix hits (DeepInfer SWA replay) ─────────


class _ReplayStubModel:
    """Minimal callable model: records the replayed span, returns
    (logits, kv) tuple for unpack_output_with_kv, carries a conv holder
    so _reset_live_recurrent / capture_recurrent_state have surfaces."""

    def __init__(self, vocab=64):
        self.vocab = vocab
        self.calls = []
        holder = _ConvHolder(
            _conv_state=torch.randn(2, 8, 3),
            _conv_state_reset=False,
            _conv_state_prefix=None,
            _ssm_state=None,
            _ssm_state_real=None,
            _ssm_state_imag=None,
            _state_reset=False,
        )
        self.blocks = [SimpleNamespace(attn=holder, _kda=None, _gdn2=None,
                                       _forge_hybrid_ssm=None)]

    def __call__(self, ids, past_key_values=None, use_cache=False,
                 logits_last_only=False, **kw):
        self.calls.append(ids.shape[1])
        kv = list(past_key_values) if past_key_values else []
        # Append replayed span to tuple entries, rebuild dict presents.
        out = []
        for e in kv:
            if isinstance(e, tuple):
                k, v = e
                pad = ids.shape[1]
                k = torch.cat([k, torch.zeros(1, k.shape[1], pad,
                                              k.shape[3])], dim=2)
                v = torch.cat([v, torch.zeros(1, v.shape[1], pad,
                                              v.shape[3])], dim=2)
                out.append((k, v))
            elif e is None:
                out.append({"conv_state": torch.randn(1, 8)})
            else:
                out.append(dict(e))
        if not out:
            out = [{"conv_state": torch.randn(1, 8)}]
        return torch.zeros(1, ids.shape[1], self.vocab), out


def test_reset_live_recurrent_zeroes_and_flags():
    model, holder = _hybrid_stub()
    holder._ssm_state = torch.randn(4, 4)
    holder._conv_state_prefix = torch.randn(2, 8, 3)
    _reset_live_recurrent(model)
    assert holder._conv_state_reset is True
    assert holder._ssm_state is None
    assert holder._conv_state_prefix is None


def test_bounded_replay_rebuilds_boundary_state():
    model = _ReplayStubModel()
    n = 512
    ids = torch.randint(0, 64, (1, n))
    g = torch.Generator().manual_seed(0)
    cached_kv = [
        (torch.randn(1, 4, n, 8, generator=g),
         torch.randn(1, 4, n, 8, generator=g)),
        {"conv_state": torch.randn(1, 8)},   # entry-end position — wrong
        None,
    ]
    matched_len, replay_len = 256, 128
    repaired_kv, snap = _bounded_replay_recurrent(
        model, ids, matched_len, cached_kv, replay_len)

    # Replayed exactly the last `replay_len` tokens before the boundary.
    assert model.calls == [replay_len]
    # Returned KV covers the full matched prefix (cached slice + replay).
    k, _ = repaired_kv[0]
    assert k.shape[2] == matched_len
    # Cached attention prefix is preserved bit-for-bit.
    assert torch.equal(k[:, :, :matched_len - replay_len],
                       cached_kv[0][0][:, :, :matched_len - replay_len])
    # Boundary snapshot captured for the suffix forward.
    assert 0 in snap and "conv" in snap[0]


def test_legacy_cache_hit_yields_none_snap_for_replay_path():
    # The real stale-state path: legacy dict / LRUPrefixCache entries
    # carry NO recurrent snapshot — get_cached_prefix returns
    # conv_snap=None, which routes generate_from_prefix_cache through
    # bounded replay instead of trusting stale live conv state.
    from forge.engine.prefix_cache import cache_key, get_cached_prefix

    ids = torch.randint(0, 100, (1, 64))
    kv = [(torch.randn(1, 2, 32, 4), torch.randn(1, 2, 32, 4))]
    cache = {cache_key(ids): (32, kv)}
    cached = get_cached_prefix(cache, ids)
    assert cached is not None
    assert cached[2] is None    # -> bounded replay in the caller


def test_chunked_boundary_hit_returns_position_correct_snap():
    cache = ChunkedPrefixCache(chunk_size=64)
    ids = torch.randint(0, 100, (1, 64))
    kv = [(torch.randn(1, 2, 64, 4), torch.randn(1, 2, 64, 4))]
    snap = {0: {"conv": torch.randn(2, 4, 3)}}
    cache.put((cache.prefix_hashes(ids[0].tolist())[-1],
               ids[0].tolist()), kv, 64, conv_snap=snap)
    hit = cache.lookup_longest_prefix(ids)
    assert hit is not None
    matched_len, _, conv_snap = hit
    assert matched_len == 64
    assert conv_snap is snap    # position-correct at its own boundary


# ── confidence-scheduled verification (DeepInfer varlen verify, bs=1) ────


class _ScriptedDraft:
    """Draft model proposing a fixed token stream with scripted
    per-step softmax confidence."""

    def __init__(self, tokens, probs, vocab=256):
        self.tokens = tokens
        self.probs = probs
        self.vocab = vocab
        self.calls = 0

    def _peaked(self, T, tok, p):
        logits = torch.full((1, T, self.vocab), -20.0)
        rest = (1.0 - p) / (self.vocab - 1)
        logits[0, -1] = torch.log(torch.tensor(rest))
        logits[0, -1, tok] = torch.log(torch.tensor(p))
        return logits

    def __call__(self, ids, past_key_values=None, use_cache=False, **kw):
        i = self.calls % len(self.tokens)
        if ids.shape[1] == 1:
            # autoregressive draft step after prefill
            logits = self._peaked(1, self.tokens[i], self.probs[i])
            self.calls += 1
        else:
            # prefill: last-position logits produce the FIRST draft token
            logits = self._peaked(ids.shape[1], self.tokens[0],
                                  self.probs[0])
        return logits, past_key_values


class _AcceptAllTarget:
    """Target model that accepts every draft token (greedy preds match
    the draft proposals) — isolates the confidence-trim effect."""

    def __init__(self, draft_tokens, vocab=256):
        self.draft_tokens = draft_tokens
        self.vocab = vocab
        self.blocks = []

    def __call__(self, ids, past_key_values=None, use_cache=False, **kw):
        T = ids.shape[1]
        logits = torch.full((1, T, self.vocab), -20.0)
        for j in range(T):
            if j < len(self.draft_tokens):
                logits[0, j, self.draft_tokens[j]] = 20.0
        return logits, (past_key_values or [])


def test_confidence_tau_trims_draft_length():
    from forge.engine.decoding import ExternalDraftSpeculativeDecoding

    def run(tau):
        draft = _ScriptedDraft([10, 11, 12, 13], [0.9, 0.9, 0.01, 0.9])
        target = _AcceptAllTarget([10, 11, 12, 13])
        strat = ExternalDraftSpeculativeDecoding(
            draft_model=draft, draft_length=4, confidence_tau=tau)
        strat.generate(target, torch.tensor([[1, 2, 3]]),
                       max_new_tokens=6, eos_token_ids={999})
        return draft.calls

    flat = run(0.0)
    trimmed = run(0.5)
    # cumprod: 0.9 -> 0.81 -> 0.0081 < 0.5 — the tau path ends each round
    # at the confident prefix; flat always drafts the full length.
    assert trimmed < flat


def test_build_decoding_passes_confidence_tau():
    strat = build_decoding("speculative", draft_model=object(),
                           draft_length=4, confidence_tau=0.25)
    assert strat.confidence_tau == 0.25


def test_spec_stats_acceptance_telemetry():
    """spec_stats must count proposed/accepted/rounds so accept_rate
    honestly reports whether a draft source pays for itself."""
    from forge.engine.decoding import ExternalDraftSpeculativeDecoding

    # Accept-all target: every proposed token verifies.
    draft = _ScriptedDraft([10, 11, 12, 13], [0.9] * 4)
    target = _AcceptAllTarget([10, 11, 12, 13])
    strat = ExternalDraftSpeculativeDecoding(
        draft_model=draft, draft_length=4)
    strat.generate(target, torch.tensor([[1, 2, 3]]),
                   max_new_tokens=8, eos_token_ids={999})
    rep = strat.spec_report()
    assert rep["rounds"] >= 1
    assert rep["proposed"] > 0
    assert rep["accepted"] > 0
    assert 0.0 < rep["accept_rate"] <= 1.0

    # Reject-everything target: propose but never accept.
    class _RejectAll(_AcceptAllTarget):
        def __call__(self, ids, past_key_values=None, use_cache=False,
                     **kw):
            T = ids.shape[1]
            logits = torch.full((1, T, self.vocab), -20.0)
            logits[0, :, 7] = 20.0  # always predicts 7, never a draft
            return logits, (past_key_values or [])

    strat2 = ExternalDraftSpeculativeDecoding(
        draft_model=_ScriptedDraft([10, 11, 12, 13], [0.9] * 4),
        draft_length=4)
    strat2.generate(_RejectAll([10, 11, 12, 13]),
                    torch.tensor([[1, 2, 3]]), max_new_tokens=8,
                    eos_token_ids={999})
    rep2 = strat2.spec_report()
    assert rep2["proposed"] > 0
    assert rep2["accepted"] == 0
    assert rep2["accept_rate"] == 0.0
