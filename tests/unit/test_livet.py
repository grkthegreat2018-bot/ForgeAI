"""LiveT unit tests — journaled live training over ConfigurableResearchLLM.

Covers the LiveSession contract: frozen substrate stays bit-identical,
live surfaces (GDN2 states, TitanMemory fast weights) update on ingest,
revert is exact, persistent memory survives stream resets, and eval
never mutates live state. CPU-only, tiny config.
"""

import torch

from forge.model.live import LiveConfig, LiveSession, livet_config
from forge.model_loader import ConfigurableResearchLLM


def _model(seed: int = 0, **kw):
    torch.manual_seed(seed)
    cfg = livet_config(vocab_size=256, d_model=64, n_layers=2, n_heads=2, **kw)
    return ConfigurableResearchLLM(cfg).eval()


def _seq(n: int = 64, lo: int = 0, hi: int = 255) -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    return torch.randint(lo, hi, (n,), generator=g)


def _param_fingerprint(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_live_surfaces_discovered():
    m = _model()
    s = LiveSession(m)
    assert len(s.mem_layers) == 2 and len(s.gdn2_layers) == 2


def test_lossless_init():
    """gate=0 memory + gdn2 branches contribute nothing at init."""
    torch.manual_seed(0)
    plain = ConfigurableResearchLLM(
        livet_config(vocab_size=256, d_model=64, n_layers=2, n_heads=2, use_gdn2=False, use_titan_memory=False)
    ).eval()
    live = _model()
    live.load_state_dict(plain.state_dict(), strict=False)
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        p1, _ = plain(ids)
        p2, _ = live(ids)
    assert torch.allclose(p1, p2, atol=1e-6)


def test_ingest_writes_memory_not_params():
    m = _model()
    s = LiveSession(m, LiveConfig(chunk=16, open_gates=True))
    pre = _param_fingerprint(m)
    stats = s.ingest(_seq(64))
    assert stats["writes"] > 0
    for mem in s.mem_layers.values():
        assert mem.memory.abs().sum() > 0
    # frozen substrate: no parameter moved
    for n, p in m.named_parameters():
        assert torch.equal(p.detach(), pre[n]), n
    # GDN2 states populated
    for g in s.gdn2_layers.values():
        assert g._state is not None


def test_revert_exact():
    m = _model()
    s = LiveSession(m, LiveConfig(chunk=16, open_gates=True))
    s.ingest(_seq(32))
    mark = s.mark()
    probe = _seq(16).unsqueeze(0)
    logits_pre = m(probe)[0].clone()
    s.ingest(_seq(64, lo=10, hi=200))
    s.ingest(_seq(64))
    assert s.revert(mark) == 2
    logits_post = m(probe)[0]
    assert torch.equal(logits_pre, logits_post)
    assert s.stream_pos == 32


def test_memory_persists_across_reset():
    m = _model()
    s = LiveSession(m, LiveConfig(chunk=16, open_gates=True))
    s.ingest(_seq(48))
    mem = s.mem_layers[0]
    w_before = mem.memory.clone()
    s.reset()  # new stream: working mem cleared
    assert m.blocks[0]._gdn2._state is None
    assert torch.equal(mem.memory, w_before)  # long-term keeps
    s.reset(memory_too=True)  # full wipe only when asked
    assert mem.memory.abs().sum() == 0


def test_eval_nll_does_not_mutate():
    m = _model()
    s = LiveSession(m, LiveConfig(chunk=16, open_gates=True))
    s.ingest(_seq(48))
    snap = s._snapshot_live()
    s.eval_nll(_seq(64))
    s.eval_nll(_seq(64), use_live_state=False)
    cur = s._snapshot_live()
    for i in snap["mem"]:
        for k in snap["mem"][i]:
            assert torch.equal(cur["mem"][i][k], snap["mem"][i][k])
    for i in snap["gdn2"]:
        a, b = snap["gdn2"][i]["state"], cur["gdn2"][i]["state"]
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b)


def test_snapshot_roundtrip():
    m = _model()
    s = LiveSession(m, LiveConfig(chunk=16, open_gates=True))
    s.ingest(_seq(48))
    snap = s.snapshot()
    s.ingest(_seq(48))
    s.load(snap)
    cur = s._snapshot_live()
    for i in snap["live"]["mem"]:
        for k in snap["live"]["mem"][i]:
            assert torch.equal(cur["mem"][i][k], snap["live"]["mem"][i][k])


def test_deterministic_ingest():
    ids = _seq(48)
    outs = []
    for _ in range(2):
        m = _model()
        s = LiveSession(m, LiveConfig(chunk=16, open_gates=True))
        s.ingest(ids)
        outs.append(s.mem_layers[0].memory.clone())
    assert torch.equal(outs[0], outs[1])


def test_bootstrap_then_live_recall():
    """The Titans claim: after the outer loop shapes k/v/q/o + head
    alignment, *novel* key→value pairs written at test time are recalled.
    Bootstrap on the pair grammar (different pairings), freeze, ingest
    novel pairs, verify loss on value positions drops."""
    torch.manual_seed(0)
    n_pairs, val_base = 8, 100
    cfg = livet_config(vocab_size=256, d_model=64, n_layers=2, n_heads=2)
    model = ConfigurableResearchLLM(cfg)
    s = LiveSession(model, LiveConfig(chunk=32, mem_lr=1.0))
    s.open_gates()

    def pairs_stream(pair_map, reps=40, sep=200):
        seq = []
        for _r in range(reps):
            for k, v in pair_map.items():
                seq += [sep, k, v]
        return torch.tensor(seq)

    # Bootstrap: pair format, DIFFERENT pairings than the live facts.
    boot_map = {10 + i: 50 + i for i in range(n_pairs)}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    boot = pairs_stream(boot_map, reps=10)
    for _ in range(8):
        _logits, loss = model(boot[:-1].unsqueeze(0), targets=boot[1:].unsqueeze(0))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    s.freeze_substrate()

    # Live facts: novel pairings (same keys, new values).
    live_map = {10 + i: val_base + i for i in range(n_pairs)}
    probe = pairs_stream(live_map, reps=1)
    # value positions = every 3rd offset (sep, k, v) → idx % 3 == 2
    val_idx = torch.arange(len(probe)) % 3 == 2

    def fact_loss():
        with torch.no_grad():
            logits = model(probe[:-1].unsqueeze(0))[0]
            ce = torch.nn.functional.cross_entropy(logits[0].float(), probe[1:], reduction="none")
        return float(ce[val_idx[1:]].mean())

    s.reset()
    loss_before = fact_loss()
    s.ingest(pairs_stream(live_map, reps=6))
    s.reset()  # memory persists; working state clears
    loss_after = fact_loss()
    assert loss_after < loss_before


def test_sparse_exact_sidecar():
    """NGramMemory: exact-suffix store journals + reverts with the
    session, lifts verbatim recall far beyond fuzzy fast weights."""
    from forge.model.ngram_mem import NGramMemory

    m = _model()
    s = LiveSession(m, LiveConfig(chunk=16, open_gates=True, mem_lr=0.0))  # fuzzy writes off
    s.attach_sparse(NGramMemory(256, orders=(4,), gate=1.0))
    ids = _seq(64)
    nll_before = s.eval_nll(ids)
    mark = s.mark()
    s.ingest(ids)
    nll_after = s.eval_nll(ids)
    assert nll_after < nll_before - 0.5  # exact recall, not marginal
    assert s.revert(mark) == 1
    nll_reverted = s.eval_nll(ids)
    assert abs(nll_reverted - nll_before) < 1e-6


def test_eval_nll_warm_parity():
    """warm=True eval preserves deterministic numbers (stream state is
    snapshotted/restored around the eval either way)."""
    m = _model()
    s = LiveSession(m, LiveConfig(chunk=16, open_gates=True))
    ids = _seq(64)
    s.ingest(ids)
    a = s.eval_nll(ids)
    b = s.eval_nll(ids, warm=True)
    assert abs(a - b) < 1.0  # same ballpark; state fully restored after


def test_ngram_lift_novel_successor_finite():
    """R64 fix: a target token with unigram=0 inside a KNOWN suffix used
    to get p=0 -> lift=-inf -> CE=inf -> surprise=inf -> NaN writes.
    The uniform floor must keep every lifted logit finite."""
    import math

    from forge.model.ngram_mem import NGramMemory

    mem = NGramMemory(256, orders=(4,), gate=1.0)
    # Teach only tokens < 64 so token 200 keeps unigram=0.
    mem.update(torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]]))
    # Query ends in a known suffix (1,2,3,4) but predicts novel token 200.
    ids = torch.tensor([[1, 2, 3, 4, 200]])
    lift = mem.lift(ids)
    assert torch.isfinite(lift).all()
    assert not math.isinf(float(lift[0, -1, 200]))
    m = _model()
    s = LiveSession(m, LiveConfig(chunk=16, open_gates=True, mem_lr=0.0))
    s.attach_sparse(mem)
    stats = s.ingest(torch.tensor([[1, 2, 3, 4, 200]]))
    assert math.isfinite(stats["mean_loss"])


def test_ngram_overwrite_recency():
    """decay-on-rewrite: retraining a suffix with a new successor makes
    the latest value dominate the lift (was: permanent 50/50 stale blend
    - measured ow=2.35/stale=2.36 in exp_live_v2)."""
    from forge.model.ngram_mem import NGramMemory

    mem = NGramMemory(64, orders=(2,), gate=1.0, write_decay=0.5)
    ctx = torch.tensor([[9, 10]])
    mem.update(torch.tensor([[9, 10, 30]] * 6))
    assert float(mem.lift(ctx)[0, -1, 30]) > float(mem.lift(ctx)[0, -1, 31])
    mem.update(torch.tensor([[9, 10, 31]] * 3))
    lift = mem.lift(ctx)
    assert float(lift[0, -1, 31]) > float(lift[0, -1, 30])


def test_ngram_overwrite_decay_one_is_last_write_wins():
    """write_decay=1.0 -> EMA rate 1 = row collapses to the latest successor."""
    from forge.model.ngram_mem import NGramMemory

    mem = NGramMemory(64, orders=(2,), gate=1.0, write_decay=1.0)
    mem.update(torch.tensor([[9, 10, 30]] * 6))
    mem.update(torch.tensor([[9, 10, 31]]))
    lift = mem.lift(torch.tensor([[9, 10]]))
    assert float(lift[0, -1, 31]) > float(lift[0, -1, 30]) + 1.0


def test_suffixmem_overwrite_recency():
    """Same decay-on-rewrite contract on the production streaming variant."""
    from forge.model.ngram_mem import SuffixMemory

    sm = SuffixMemory(64, orders=(2,), gate=1.0, write_decay=0.5)
    for _ in range(6):
        sm.feed([9, 10, 30])
    sm.feed([9, 10])
    assert float(sm.lift_last()[30]) > float(sm.lift_last()[31])
    for _ in range(3):
        sm.feed([9, 10, 31])
    sm.feed([9, 10])
    lift = sm.lift_last()
    assert float(lift[31]) > float(lift[30])
    assert sm.draft(1) == [31]

def test_suffixmem_save_load_parity(tmp_path):
    import numpy as np

    from forge.model.ngram_mem import SuffixMemory
    sm = SuffixMemory(64, orders=(4, 8), gate=1.0, write_decay=0.5)
    sm.feed(list(range(1, 20)))
    sm.feed([5, 6, 7, 8, 42])
    sm.feed([5, 6, 7, 8])
    lift_before = sm.lift_last().numpy().copy()
    p = tmp_path / "mem.pkl"
    sm.save(p)
    sm2 = SuffixMemory.load(p)
    lift_after = sm2.lift_last().numpy()
    assert np.array_equal(lift_before, lift_after)
    assert sm2.info()["cells"] == sm.info()["cells"]
    # continued ingest works identically on both instances
    sm.feed_one(9)
    sm2.feed_one(9)
    assert np.array_equal(sm.lift_last().numpy(), sm2.lift_last().numpy())

def test_suffix_draft_model_round_semantics():
    # SuffixDraftModel follows the external_draft_speculative contract:
    # committed tokens live in mem._tail (the strategy feeds them);
    # T==1 calls roll only the virtual proposal chain.
    import torch

    from forge.model.ngram_mem import SuffixDraftModel, SuffixMemory
    mem = SuffixMemory(64, orders=(4,), gate=1.0, write_decay=0.5)
    seq = [10, 11, 12, 13, 42, 43, 10, 11, 12, 13, 42, 43]
    d = SuffixDraftModel(mem)
    # Strategy feeds the prompt BEFORE the T>1 prefill call.
    mem.feed(seq)
    logits, kv = d(torch.tensor([seq]), use_cache=True)
    assert kv is None
    # tail ends ..42,43 -> order-4 key (12,13,42,43) -> successor 10
    t1 = int(logits[0, -1].argmax())
    assert t1 == 10
    # roll proposal 10 -> next prediction 11
    logits, _ = d(torch.tensor([[t1]]), use_cache=True)
    assert int(logits[0, -1].argmax()) == 11
    # roll 11 -> 12
    logits, _ = d(torch.tensor([[11]]), use_cache=True)
    assert int(logits[0, -1].argmax()) == 12
    # resync: verifier commits [10, 11] + corrected token 99 — the
    # strategy feeds them, THEN the T>1 call ends the proposal chain.
    for t in (10, 11, 99):
        mem.feed_one(t)
    logits, _ = d(torch.tensor([[10, 11, 99]]), use_cache=True)
    # tail ends ..11,99 -> no learned successor -> uniform-ish logits
    assert float(logits.max()) <= -19.0
    # feed committed tokens completing a known suffix -> predict 10
    for t in (12, 13, 42, 43):
        mem.feed_one(t)
    logits, _ = d(torch.tensor([[12, 13, 42, 43]]), use_cache=True)
    assert int(logits[0, -1].argmax()) == 10
    # a T==1 token that is NOT the last prediction is treated as an
    # out-of-band commit (strategy's no-draft resync), not a proposal.
    logits, _ = d(torch.tensor([[7]]), use_cache=True)
    assert d._proposal == []
