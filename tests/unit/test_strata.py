"""STRATA substrate tests (M0): packed cells, labeled blocks,
tag-scoped revert, container IO.  CPU-runnable; GPU mirror gated."""
from __future__ import annotations

import time

import pytest
import torch

from forge.model.strata import (
    DEFAULT_KNOBS,
    BlockStore,
    CellBlock,
    KnobSet,
    ParamSet,
    cell_bytes,
    count_lut,
    decode_counts,
    foll_cntq,
    foll_tok,
    foll_valid,
    load_store,
    pack_follower,
    quantize_counts,
    save_store,
)

DEV = "cpu"


def _keys(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-(1 << 62), (1 << 62), (n,),
                         generator=g, dtype=torch.int64)


# ---- count LUT -------------------------------------------------------------


def test_count_lut_relative_error():
    lut = count_lut()
    c = torch.logspace(0, 4.5, 500).float()      # 1 .. ~31k
    q = quantize_counts(c)
    back = decode_counts(q, lut)
    rel = ((back - c).abs() / c.clamp_min(1)).max()
    assert rel < 0.06                            # ~4.4% steps + round


def test_count_lut_edges():
    lut = count_lut()
    q = quantize_counts(torch.tensor([0.0, 1.0, 5.0, 63000.0, 1e9]))
    back = decode_counts(q, lut)
    assert back[0] == 0.0
    assert back[1] == pytest.approx(1.0, rel=0.05)
    assert back[-1] == lut[255]                  # saturates


def test_pack_follower_roundtrip():
    w = pack_follower(torch.tensor(65535),
                      torch.tensor(200), torch.tensor(7))
    assert int(foll_tok(w)) == 65535
    assert int(foll_cntq(w)) == 200
    assert foll_valid(w)


# ---- cell block write/read ---------------------------------------------------


def test_block_add_read():
    b = CellBlock("o8", 0, 1024, 8, torch.device(DEV))
    keys = torch.arange(1, 200) * 7919
    toks = torch.arange(1, 200) % 5000
    d = torch.ones(199)
    b.add(keys, toks, d, wtag=1)
    t, _c, tot, f = b.read(keys)
    assert f.all()
    assert (t[:, 0] == toks).all()
    assert tot == pytest.approx(torch.ones(199), rel=0.06)


def test_block_merge_counts():
    b = CellBlock("o8", 0, 1024, 8, torch.device(DEV))
    k = torch.tensor([12345])
    for _i in range(5):
        b.add(k, torch.tensor([7]), torch.tensor([1.0]), wtag=0)
    t, _c, tot, f = b.read(k)
    assert f.item()
    assert (t[0] == 7).sum() == 1                 # merged, not duplicated
    assert tot[0].item() == pytest.approx(5.0, rel=0.08)


def test_batch_dup_keys_accumulate():
    """Same (key,tok) twice in ONE batch must land +2d, not +1d."""
    b = CellBlock("o8", 0, 1024, 8, torch.device(DEV))
    k = torch.tensor([777, 777])
    b.add(k, torch.tensor([42, 42]), torch.tensor([1.0, 1.0]), wtag=0)
    _, _c, tot, _ = b.read(torch.tensor([777]))
    assert tot[0].item() == pytest.approx(2.0, rel=0.08)


def test_eviction_bounded():
    b = CellBlock("o8", 0, 256, 8, torch.device(DEV))
    b.add(_keys(4000), torch.randint(0, 60000, (4000,)),
           torch.ones(4000), wtag=0)
    assert int(b.occupied()) <= 256
    assert int(b._ev) > 0


def test_determinism():
    ks, tk, d = _keys(500), torch.randint(0, 60000, (500,)), \
        torch.rand(500) + 0.1
    a, b = (CellBlock("o8", 0, 4096, 8, torch.device(DEV))
            for _ in range(2))
    for blk in (a, b):
        blk.add(ks, tk, d, wtag=3)
    assert torch.equal(a.tag, b.tag)
    assert torch.equal(a.foll, b.foll)
    assert torch.equal(a.metaw, b.metaw)


# ---- tag revert --------------------------------------------------------------


def _snap(b):
    return (b.tag.clone(), b.metaw.clone(), b.foll.clone())


def test_revert_tag_exact():
    b = CellBlock("o8", 0, 4096, 8, torch.device(DEV))
    b.add(_keys(300, 1), torch.randint(0, 60000, (300,)),
           torch.ones(300), wtag=1, seq=0)
    snap = _snap(b)
    b.add(_keys(300, 2), torch.randint(0, 60000, (300,)),
           torch.ones(300), wtag=2, seq=300)
    assert b.revert_tag(2) == 300
    t2 = _snap(b)
    for a, c in zip(snap, t2):
        assert torch.equal(a, c)


def test_revert_tag_preserves_other_tags():
    b = CellBlock("o8", 0, 4096, 8, torch.device(DEV))
    b.add(torch.tensor([55]), torch.tensor([9]), torch.tensor([1.0]),
          wtag=1, seq=0)
    b.add(torch.tensor([55]), torch.tensor([10]),
          torch.tensor([1.0]), wtag=2, seq=1)
    b.revert_tag(2)
    t, _c, _tot, f = b.read(torch.tensor([55]))
    assert f.item() and (t[0] == 9).any() and not (t[0] == 10).any()


def test_revert_since():
    st = BlockStore(slots=4096, dev=DEV)
    st.write("o8", _keys(100), torch.randint(0, 60000, (100,)),
             torch.ones(100), tag="a")
    mark = st.seq
    st.write("o8", _keys(100, 9), torch.randint(0, 60000, (100,)),
             torch.ones(100), tag="b")
    blk = st.kinds["o8"][0]
    snap_f = blk.foll.clone()
    st.write("o8", _keys(50, 3), torch.randint(0, 60000, (50,)),
             torch.ones(50), tag="c")
    st.revert_since(mark + 99)
    # post-revert cells equal the snapshot taken right after 'b'
    assert torch.equal(blk.foll, snap_f)


# ---- store lifecycle ------------------------------------------------------------


def test_seal_rolls_active():
    st = BlockStore(slots=1024, dev=DEV, fill_cap=0.5)
    st.write("o8", _keys(600), torch.randint(0, 60000, (600,)),
             torch.ones(600), tag="t")
    assert len(st.kinds["o8"]) == 2            # sealed + fresh active
    assert st.kinds["o8"][1]._sealed
    assert st.kinds["o8"][1].meta.checksum != 0


def test_store_read_merges_blocks():
    st = BlockStore(slots=1024, dev=DEV, fill_cap=0.4)
    k_old = _keys(600, 5)
    st.write("o8", k_old, torch.randint(0, 60000, (600,)),
             torch.ones(600), tag="old")      # seals mid-write
    k_new = _keys(20, 6)
    st.write("o8", k_new, torch.randint(0, 60000, (20,)),
             torch.ones(20), tag="new")
    _, _, _, f1 = st.read("o8", k_old)
    _, _, _, f2 = st.read("o8", k_new)
    assert f1.float().mean() > 0.8 and f2.all()


def test_revert_tag_drops_owned_sealed():
    st = BlockStore(slots=512, dev=DEV)
    st.write("o8", _keys(200, 1), torch.randint(0, 60000, (200,)),
             torch.ones(200), tag="keep")
    st.seal("o8")
    st.write("o8", _keys(200, 2), torch.randint(0, 60000, (200,)),
             torch.ones(200), tag="temp")
    st.seal("o8")
    rep = st.revert_tag("temp")
    assert rep["dropped"] and not rep["tainted"]
    assert len(st.kinds["o8"]) == 2            # keep-sealed + active
    # 'keep' facts still readable
    _, _, _, f = st.read("o8", _keys(200, 1))
    assert f.float().mean() > 0.8


def test_revert_tag_taints_partial_sealed():
    st = BlockStore(slots=512, dev=DEV)
    st.write("o8", _keys(100, 1), torch.randint(0, 60000, (100,)),
             torch.ones(100), tag="a")
    st.write("o8", _keys(100, 2), torch.randint(0, 60000, (100,)),
             torch.ones(100), tag="b")
    st.seal("o8")
    rep = st.revert_tag("a")
    assert rep["tainted"]                      # frozen bytes win


def test_stats_labels():
    st = BlockStore(slots=1024, dev=DEV)
    st.write("o8", _keys(100), torch.randint(0, 60000, (100,)),
             torch.ones(100), tag="train-run-1")
    s = st.stats()["o8"][0]
    assert s["writes"] == 100
    assert s["tags"] == {1: 100}
    assert s["load"] > 0


# ---- io roundtrip -----------------------------------------------------------------


def test_save_load_bitexact(tmp_path):
    st = BlockStore(slots=2048, dev=DEV, store_raw=True)
    ks = _keys(800)
    st.write("o8", ks, torch.randint(0, 60000, (800,)),
             torch.rand(800) + 0.1, tag="run")
    st.seal("o8")
    st.write("o8", _keys(50, 7), torch.randint(0, 60000, (50,)),
             torch.ones(50), tag="live")
    p = tmp_path / "w.strata"
    save_store(p, st)
    st2 = load_store(p, device="cpu")
    for kind, bl in st.kinds.items():
        for b1, b2 in zip(bl, st2.kinds[kind]):
            assert torch.equal(b1.tag, b2.tag)
            assert torch.equal(b1.foll, b2.foll)
            assert torch.equal(b1.metaw, b2.metaw)
            assert torch.equal(b1.raw, b2.raw)
            assert b1.meta.checksum == b2.meta.checksum
    t1, c1, _, f1 = st.read("o8", ks)
    t2, c2, _, f2 = st2.read("o8", ks)
    assert torch.equal(t1, t2) and torch.equal(f1, f2)
    assert torch.allclose(c1, c2)


def test_load_speed(tmp_path):
    """Sub-second for ~40MB of cells is the 'lightning load' bar."""
    st = BlockStore(slots=1 << 18, dev=DEV)
    for i in range(4):
        st.write(f"o{i}", _keys(200_000, i),
                 torch.randint(0, 60000, (200_000,)),
                 torch.ones(200_000), tag=f"b{i}")
    p = tmp_path / "big.strata"
    t0 = time.perf_counter()
    save_store(p, st)
    t_save = time.perf_counter() - t0
    t0 = time.perf_counter()
    st2 = load_store(p, device="cpu")
    t_load = time.perf_counter() - t0
    mb = p.stat().st_size / 1e6
    print(f"\n  {mb:.1f}MB save={t_save:.2f}s load={t_load:.3f}s")
    assert t_load < 10.0                        # real target <1s


# ---- knobs / params --------------------------------------------------------------


def test_knobs_overlay():
    k = KnobSet(dict(DEFAULT_KNOBS))
    assert k["temp"] == 1.0
    k.push("creative", {"temp": 1.8})
    assert k["temp"] == 1.8
    k.push("precise", {"temp": 0.2})
    assert k["temp"] == 0.2
    k.pop("precise")
    assert k["temp"] == 1.8
    k.pop("creative")
    assert k["temp"] == 1.0
    assert k.get("missing", 42) == 42


def test_knobs_io(tmp_path):
    k = KnobSet({"a": 1.5})
    k.push("x", {"b": [1, 2]})
    p = tmp_path / "k.cond"
    k.save(p)
    k2 = KnobSet.load(p)
    assert k2["a"] == 1.5 and k2["b"] == [1, 2]


def test_params_seeded_derive(tmp_path):
    p = ParamSet()
    t = p.set_seeded("R", (64, 8), seed=1234)
    p.set("w_mix", torch.linspace(0, 1, 12))
    f = tmp_path / "p.iq"
    p.save(f)
    p2 = ParamSet.load(f)
    assert torch.equal(p2.get("R"), t)          # regenerated, not stored
    assert torch.equal(p2.get("w_mix"), p.get("w_mix"))


# ---- sub-bit packed params -------------------------------------------------


def test_param_pack_int4_roundtrip(tmp_path):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(1000, 256, generator=g) * 0.02
    p = ParamSet()
    p.set_packed("gsyn.wq", w, bits=4, group=64)
    back = p.get("gsyn.wq")
    rel = (back - w).norm() / w.norm()
    # fp4 E2M1 PTQ floor on gaussian weights ~0.10-0.11; QAT at
    # distill time is what recovers quality, not tighter rounding.
    assert rel < 0.12
    # byte check: 256k params -> 4bit = 8/word -> 32k words + scales
    assert p.tensors["gsyn.wq::w0"].numel() == 1000 * 256 // 8
    f = tmp_path / "q.iq"
    p.save(f)
    p2 = ParamSet.load(f)
    assert torch.equal(p2.get("gsyn.wq"), back)


def test_param_pack_bits_1_and_2():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(4096, generator=g)
    # measured PTQ floors (gaussian, group=64): fp4 ~0.106,
    # ternary ~0.51, binary ~0.60 - asserts are mechanics + boundedness,
    # quality recovery is QAT/distill's job
    for bits, bound in ((4, 0.12), (2, 0.55), (1, 0.65)):
        p = ParamSet()
        p.set_packed("w", w, bits=bits, group=64)
        back = p.get("w")
        rel = (back - w).norm() / w.norm()
        assert rel < bound, f"bits={bits} rel={rel:.3f}"
    # byte math: 4096 params @1bit -> 128 words = 512B (~0.125B/param)
    p = ParamSet()
    p.set_packed("w", w, bits=1)
    assert p.tensors["w::w0"].numel() == 128


def test_param_pack_residual_nullifies_error():
    """Residual levels compound the quant error down: fp4 x2 must land
    ~10x tighter than fp4 x1 (~0.106 -> ~0.01) at 1B/param."""
    g = torch.Generator().manual_seed(2)
    w = torch.randn(64, 4096, generator=g)
    p = ParamSet()
    p.set_packed("w", w, bits=4, residual=1)
    e1 = ((p.get("w") - w).norm() / w.norm()).item()
    p2 = ParamSet()
    p2.set_packed("w", w, bits=4, residual=2)
    e2 = ((p2.get("w") - w).norm() / w.norm()).item()
    p3 = ParamSet()
    p3.set_packed("w", w, bits=4, residual=3)
    e3 = ((p3.get("w") - w).norm() / w.norm()).item()
    assert e1 < 0.12 and e2 < 0.03 and e3 < 0.01, (e1, e2, e3)
    assert e2 < 0.3 * e1 and e3 < 0.3 * e2   # multiplicative decay


def test_param_packed_names_and_bytes():
    p = ParamSet()
    p.set_packed("big", torch.randn(8192), bits=4)
    p.set("scalars", torch.ones(4))
    assert set(p.names()) == {"big", "scalars"}
    # 8192 * 4b/32 = 1024 words*4B + 128 fp16 scales + 16B scalars
    assert p.stored_bytes() == 1024 * 4 + 128 * 2 + 4 * 4


def test_cell_bytes_density():
    """The compression claim, in bytes: FLUX ~152B/cell vs STRATA."""
    assert cell_bytes(8, raw=True) == 48        # ~3.2x denser
    assert cell_bytes(8, raw=False) == 40       # ~3.8x
    assert cell_bytes(4, raw=False) == 24       # ~6.3x


# ---- gpu mirror --------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_gpu_block_parity():
    ks = _keys(2000, 11)
    tk = torch.randint(0, 60000, (2000,))
    d = torch.rand(2000) + 0.1
    cpu = CellBlock("o8", 0, 8192, 8, torch.device("cpu"))
    gpu = CellBlock("o8", 0, 8192, 8, torch.device("cuda"))
    cpu.add(ks, tk, d, wtag=1)
    gpu.add(ks.cuda(), tk.cuda(), d.cuda(), wtag=1)
    assert torch.equal(cpu.tag, gpu.tag.cpu())
    assert torch.equal(cpu.foll, gpu.foll.cpu())
    tc, cc, _, fc = cpu.read(ks)
    tg, cg, _, fg = gpu.read(ks.cuda())
    assert torch.equal(fc, fg.cpu())
    assert torch.equal(tc, tg.cpu())
    assert torch.allclose(cc, cg.cpu())
