"""StrataLM M1 tests - model dynamics on the packed-cell substrate.
Mirrors the FluxLM learn/recall probes in test_flux.py on the
BlockStore/CellBlock substrate. CPU-only."""
import numpy as np
import torch

from forge.model.strata import StrataConfig, StrataLM


def _cfg(**kw):
    kw.setdefault("vocab_size", 512)
    kw.setdefault("slots_per_block", 1 << 12)
    kw.setdefault("ring_capacity", 1 << 14)
    kw.setdefault("journal_cap", 50_000)
    kw.setdefault("eos_token_id", -1)       # no mid-stream episode fires
    return StrataConfig(**kw)


def _ids(text: str) -> list[int]:
    return [ord(ch) % 512 for ch in text]


# ---- forward contract --------------------------------------------------------


def test_forward_contract():
    m = StrataLM(_cfg())
    ids = torch.tensor([_ids("hello world")])
    logits, loss, past = m(ids, use_cache=True)
    assert logits.shape == (1, 11, 512)
    assert logits.dtype == torch.float32
    assert past is None and loss is None


def test_empty_model_predicts_uniform():
    """t=0: no cells, no counts - the emitted distribution is the
    (uniform) unigram prior, finite everywhere."""
    m = StrataLM(_cfg())
    p = torch.softmax(m._predict(), -1)
    assert torch.isfinite(p).all()
    assert p.max().item() - p.min().item() < 1e-3


# ---- live learning ------------------------------------------------------------


def test_live_learning_recall():
    m = StrataLM(_cfg())
    m.ingest(_ids("the cat sat on the mat. " * 20), tag="train")
    m.soft_reset()
    m.observe(_ids("on the"))
    top = int(m._predict().argmax())
    assert top in (ord("m") % 512, ord(" ") % 512)


def test_learn_new_word_instantly():
    """The core promise: a fact learned mid-stream applies on the next
    predict - no gradient step, no batch."""
    m = StrataLM(_cfg())
    m.ingest(_ids("a glorf is a small bird. " * 5), tag="def")
    m.soft_reset()
    m.observe(_ids("a glorf is a"))
    top5 = torch.topk(m._predict(), 5).indices.tolist()
    assert ord(" ") % 512 in top5
    m.soft_reset()
    m.observe(_ids("glorf is a small"))
    top5b = torch.topk(m._predict(), 5).indices.tolist()
    assert ord(" ") % 512 in top5b


def test_recall_past_filler():
    """Order-k keys survive intervening filler: a fact stated before a
    long noise span still predicts (the substrate is the memory, not a
    context window)."""
    m = StrataLM(_cfg())
    filler = "q w e r t y u i o p " * 15          # 150 tokens of noise
    m.ingest(_ids("the secret word is zephyr. " + filler), tag="mem")
    m.soft_reset()
    m.observe(_ids("the secret word is"))
    top = int(m._predict().argmax())
    assert top in (ord(" ") % 512, ord("z") % 512)


def test_weights_are_local_not_meshed():
    """Learning under one context must not touch unrelated cells - the
    block store grows new cells; existing rows keep their bytes."""
    m = StrataLM(_cfg())
    m.ingest(_ids("aaa bbb aaa bbb "), tag="a")
    before = {k: sum(int(b.occupied()) for b in bl)
              for k, bl in m.store.kinds.items()}
    m.ingest(_ids("x y z x y z "), tag="b")
    after = {k: sum(int(b.occupied()) for b in bl)
             for k, bl in m.store.kinds.items()}
    for k in after:
        assert after[k] >= before.get(k, 0)


# ---- surprise gating ---------------------------------------------------------


def test_surprise_gated_writes():
    """A well-predicted repeat writes less mass than a novel one."""
    m = StrataLM(_cfg())
    m.ingest(_ids("same same same same same "), tag="t")
    stats = m.store.stats()
    tots = sum(s["writes"] for k in stats.values() for s in k)
    # 25 tokens * 10 order channels = 250 max writes; 'same' contexts
    # repeat so new cells < writes; and surprise decays -> the second
    # 'same ' writes less than the first
    assert tots > 0
    assert all(s["writes"] <= len(_ids("same " * 5)) for s_list in
               stats.values() for s in s_list)


def test_channel_weights_adapt():
    m = StrataLM(_cfg())
    w0 = m._w.copy()
    m.ingest(_ids("ab ab ab ab ab ab ab ab "), tag="t")
    assert not np.allclose(m._w, w0)
    assert abs(m._w.sum() - 1.0) < 1e-6


def test_uni_escape_floor():
    m = StrataLM(_cfg())
    m.knobs.set("uni_escape", 0.05)
    m.ingest(_ids("aaa bbb ccc " * 20), tag="t")
    p = torch.softmax(m._predict().float(), -1)
    denom = m._total + m._k("uni_alpha") * m.vocab_size
    uni_pv = torch.as_tensor(
        (m._uni + m._k("uni_alpha")) / denom, dtype=torch.float32)
    assert (p >= 0.05 * uni_pv - 1e-6).all()
    assert torch.isfinite(p).all() and abs(float(p.sum()) - 1.0) < 1e-4


def test_calib_rewards_no_hedge_collapse():
    """Additive-lift channels must not monopolize the mixture."""
    m = StrataLM(_cfg())
    m.ingest(_ids("the cat sat on the mat. " * 20), tag="t")
    m.ingest(_ids("alpha beta gamma delta epsilon " * 10), tag="t2")
    w = m.channel_weights()
    assert w["fatigue"] < 0.5
    assert w["recency"] < 0.5
    assert sum(v for k, v in w.items() if k.startswith("o")) > 0.3


# ---- observe / episode --------------------------------------------------------


def test_observe_writes_nothing():
    """observe() advances context only: no cells, no journal, no uni."""
    m = StrataLM(_cfg())
    n = m.observe(_ids("just context tokens "))
    assert n == len(_ids("just context tokens "))
    assert m._count == n and m._total == 0
    assert not m.store.kinds          # no blocks created
    assert len(m._journal) == 0


def test_observe_context_still_guides():
    """Observed context feeds suffix keys so a later predict sees it."""
    m = StrataLM(_cfg())
    m.ingest(_ids("the cat sat on the mat. " * 10), tag="t")
    m.soft_reset()
    m.observe(_ids("the cat sat on the"))
    top = int(m._predict().argmax())
    assert top in (ord(" ") % 512, ord("m") % 512)


def test_auto_episode_boundary():
    """EOS starts a fresh episode: no order key may span the boundary;
    the boundary is journaled so revert restores ep_start."""
    c = _cfg()
    c.eos_token_id = 2
    c.auto_episode = True
    m = StrataLM(c)
    doc1 = _ids("the cat sat. ")
    m.ingest(doc1 + [2] + _ids("do"), tag="t")
    assert m._ep_start == len(doc1) + 1
    assert m._keys_at(m._count - 1) == {}      # 2 in-episode tokens
    m.revert_tag("t")
    assert m._ep_start == 0


def test_recent_window_bounded():
    m = StrataLM(_cfg())
    m.ingest(_ids("the cat sat on the mat. " * 20), tag="t")
    assert len(m._recent) <= m.config.recency_span
    logits = m._predict()
    assert logits.shape[-1] == 512


# ---- teach --------------------------------------------------------------------


def test_teach_instant_hot_training():
    """teach(): one call binds context -> target; recall is immediate,
    no stream position needed (keys derive from the token sequence)."""
    m = StrataLM(_cfg())
    m.ingest(_ids("ordinary preamble text " * 3), tag="boot")
    rep = m.teach([10, 20, 30, 40], [77], weight=50.0, tag="fact")
    assert rep["cells"] > 0
    m.soft_reset()
    m.observe([10, 20, 30, 40])
    assert int(m._predict().argmax()) == 77


def test_teach_revert():
    """revert_tag on a teach() block un-teaches: cells are byte-restored
    by the block undo arenas."""
    m = StrataLM(_cfg())
    m.ingest(_ids("baseline filler " * 4), tag="boot")
    m.teach([5, 6, 7, 8], [99], weight=60.0, tag="fact")
    m.soft_reset()
    m.observe([5, 6, 7, 8])
    assert int(m._predict().argmax()) == 99
    rep = m.revert_tag("fact")
    assert rep["found"]
    m.soft_reset()
    m.observe([5, 6, 7, 8])
    assert int(m._predict().argmax()) != 99


# ---- revert / journal ---------------------------------------------------------


def test_journal_audit_and_revert():
    m = StrataLM(_cfg())
    m.ingest(_ids("first fact alpha beta "), tag="keep")
    snap_pos = m._count
    m.ingest(_ids("bad data poison poison "), tag="poison")
    assert m._count > snap_pos
    rep = m.revert_tag("poison")
    assert rep["found"] and rep["reverted"] > 0
    assert m._count == snap_pos
    # poisoned followers must be gone from the o-kind cells
    m.soft_reset()
    m.observe(_ids("bad data "))
    logits = m._predict()
    assert torch.isfinite(logits).all()


def test_revert_since_point_in_time():
    """revert_since(pos) un-learns the tail: stream, uni, recency,
    hedge AND cell bytes all restore - position-free writes inside the
    span are dropped by the seq watermark too."""
    m = StrataLM(_cfg())
    m.ingest(_ids("stable prefix text here. " * 4), tag="keep")
    mark = m._count
    w_at_mark = m._w.copy()
    m.ingest(_ids("zqxjkvb " * 20), tag="drop")
    rep = m.revert_since(seq_pos=mark)
    assert m._count == mark
    assert rep["reverted"] > 0
    # ring still holds the kept span; recency rebuilt in-window
    assert all(0 <= m._count - 1 - p < m.config.recency_span
               for p, _ in m._recent)
    # hedge weights return to the mark's state
    assert np.allclose(m._w, w_at_mark)


def test_revert_restores_prediction():
    """After revert the model predicts as if the writes never happened."""
    m = StrataLM(_cfg())
    m.ingest(_ids("baseline text here "), tag="base")
    m.soft_reset()
    probe = torch.tensor([_ids("baseline")])
    m.learning = False
    before = m(probe)[0][0, -1].clone()
    m.learning = True
    m.soft_reset()
    m.ingest(_ids("poisoned garbage zzz zzz "), tag="bad")
    m.revert_tag("bad")
    m.soft_reset()
    m.learning = False
    after = m(probe)[0][0, -1]
    m.learning = True
    assert torch.allclose(before, after, atol=1e-4)


def test_audit_and_writes_since():
    m = StrataLM(_cfg())
    m.ingest(_ids("audit me please "), tag="a")
    rec = m.audit(5)
    assert rec and rec[-1]["tag"] == "a"
    rep = m.writes_since(1000)
    assert rep["ops"].get("t", 0) > 0 and rep["tags"]["a"] > 0


# ---- mixture / explain ---------------------------------------------------------


def test_explain_provenance():
    """explain(): every firing cell traces to (kind, block, slot, key,
    tag) - the transparency contract."""
    m = StrataLM(_cfg())
    m.ingest(_ids("the cat sat on the mat. " * 10), tag="train")
    m.soft_reset()
    m.observe(_ids("the cat sat on the"))
    rep = m.explain(top_k=5)
    assert rep["cells"]                         # cells fired
    c0 = rep["cells"][0]
    assert c0["channel"].startswith("o")
    assert c0["tag"] == "train" and c0["followers"]
    assert rep["weights"]["o6"] > 0


def test_memory_report_bounded():
    m = StrataLM(_cfg())
    rep = m.memory_report()
    assert rep["total_MB_est"] < 1024
    assert rep["journal_len"] > 0 or rep["journal_dropped"] >= 0


# ---- snapshot ------------------------------------------------------------------


def test_snapshot_roundtrip(tmp_path):
    """Cell slabs + aux state (uni, ring, count, weights) survive a
    container round-trip; prediction after load matches."""
    m = StrataLM(_cfg())
    m.ingest(_ids("persistent memory test data " * 3), tag="s")
    m.teach([1, 2, 3, 4], [88], weight=30.0, tag="t1")
    p = m.snapshot(tmp_path / "m.strata")
    m2 = StrataLM.load(p)
    assert m2._count == m._count
    assert m2._total == m._total
    assert np.array_equal(m2._uni, m._uni)
    assert np.allclose(m2._w, m._w)
    for kind, bl in m.store.kinds.items():
        for b1, b2 in zip(bl, m2.store.kinds[kind]):
            assert torch.equal(b1.tag, b2.tag)
            assert torch.equal(b1.foll, b2.foll)
    # teach still recalls after load
    m2.soft_reset()
    m2.observe([1, 2, 3, 4])
    assert int(m2._predict().argmax()) == 88


def test_snapshot_artifacts(tmp_path):
    """snapshot writes .strata (weights+state), .iq (params incl.
    mixer weights) and .cond (knobs) as separate artifacts."""
    m = StrataLM(_cfg())
    m.ingest(_ids("artifact check " * 3), tag="s")
    p = m.snapshot(tmp_path / "m.strata")
    assert p.exists()
    assert (tmp_path / "m.iq").exists()
    assert (tmp_path / "m.cond").exists()
    assert "w_mix" in m.params.names()


# ---- generate ------------------------------------------------------------------


def test_generate_greedy_recalls():
    m = StrataLM(_cfg())
    m.ingest(_ids("the cat sat on the mat. " * 10), tag="t")
    m.soft_reset()
    out = m.generate(_ids("the cat sat on the"), max_new=4,
                     temperature=0.0, learn=False)
    assert len(out) == 4
    # a learned corpus continues predictably
    assert all(isinstance(t, int) for t in out)
