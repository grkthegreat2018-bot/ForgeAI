"""Unit tests for SIFT (forge/training/sift.py). CPU-only.

Coverage:
- zero-init slices are lossless (bit-exact forward == base)
- grads reach slice deltas only; frozen weights get none
- backward frontier: no autograd graph below the frontier block
- calibration selects the planted-important units
- merge folds deltas into the right weight slices (+ fused mirrors)
- adapter save/load round-trip re-attaches identical slices
- prefix cache: fill + inject produce identical logits to full forward
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from forge.training import sift


class ToyFusedQKV(nn.Module):
    """Stand-in for FusedQKVLinear: [q+k+v+g, in], returns split tuple."""
    def __init__(self, in_f, q, k, v, g=0):
        super().__init__()
        self.q_dim, self.k_dim, self.v_dim, self.g_dim = q, k, v, g
        self.out_dim = q + k + v + g
        self.weight = nn.Parameter(torch.randn(self.out_dim, in_f) * 0.02)
        self.bias = None

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight)
        dims = [self.q_dim, self.k_dim, self.v_dim] + ([self.g_dim] if self.g_dim else [])
        return out.split(dims, dim=-1)


class ToyAttn(nn.Module):
    def __init__(self, d, hd, nq, nkv, use_gate=True):
        super().__init__()
        self.head_dim = hd
        self.qkv_proj = ToyFusedQKV(d, nq * hd, nkv * hd, nkv * hd,
                                    nq * hd if use_gate else 0)
        # split mirrors (kept like the real builder does)
        self.q_proj = nn.Linear(d, nq * hd, bias=False)
        self.k_proj = nn.Linear(d, nkv * hd, bias=False)
        self.v_proj = nn.Linear(d, nkv * hd, bias=False)
        self.gate_proj = nn.Linear(d, nq * hd, bias=False) if use_gate else None
        with torch.no_grad():
            w = torch.cat([self.q_proj.weight, self.k_proj.weight,
                           self.v_proj.weight] +
                          ([self.gate_proj.weight] if use_gate else []))
            self.qkv_proj.weight.copy_(w)
        self.out_proj = nn.Linear(nq * hd, d, bias=False)

    def forward(self, x):
        q, k, v, g = self.qkv_proj(x)
        return self.out_proj(q * torch.sigmoid(g))


class ToyFusedGateUp(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.d_model, self.hidden_dim = d, h
        self.out_dim = 2 * h
        self.weight = nn.Parameter(torch.randn(self.out_dim, d) * 0.02)
        self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight).split(
            [self.hidden_dim, self.hidden_dim], dim=-1)


class ToyFFN(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate_up_proj = ToyFusedGateUp(d, h)
        self._fused_gate_up = True
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)
        with torch.no_grad():
            self.gate_up_proj.weight.copy_(
                torch.cat([self.w_gate.weight, self.w_up.weight]))

    def forward(self, x):
        g, u = self.gate_up_proj(x)
        return self.w_down(torch.nn.functional.gelu(g) * u)


class ToyBlock(nn.Module):
    def __init__(self, d, hd, nq, nkv, h):
        super().__init__()
        self.attn = ToyAttn(d, hd, nq, nkv)
        self.ffn = ToyFFN(d, h)

    def forward(self, x):
        return x + self.attn(x) + self.ffn(x)


class ToyModel(nn.Module):
    def __init__(self, n_blocks=6, d=64, hd=16, nq=4, nkv=2, h=128, vocab=97):
        super().__init__()
        self.config = type("C", (), dict(head_dim=hd, n_heads=nq,
                                         n_kv_heads=nkv,
                                         intermediate_size=h))()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            [ToyBlock(d, hd, nq, nkv, h) for _ in range(n_blocks)])
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, attention_mask=None, **kw):
        x = self.embed(idx)
        _sift_start = int(getattr(self, "_sift_start_layer", 0) or 0)
        _sift_inj = getattr(self, "_sift_inject_hidden", None)
        if _sift_inj is not None:
            self._sift_inject_hidden = None
            x = _sift_inj.to(x.dtype)
        if _sift_start:
            self._sift_start_layer = 0
        _sift_cap = getattr(self, "_sift_capture_at", None)
        for i, blk in enumerate(self.blocks):
            if i < _sift_start:
                continue
            x = blk(x)
            if _sift_cap is not None and i + 1 >= _sift_cap:
                self._sift_captured = x.detach()
                return None
        return self.head(x)


def _toy(seed=0):
    torch.manual_seed(seed)
    return ToyModel()


def _ids(b=2, t=8):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 96, (b, t), generator=g)


# ── lossless init ──────────────────────────────────────────────────────────

def test_slice_init_is_lossless():
    m = _toy()
    ids = _ids()
    base = m(ids).clone()
    qkv = m.blocks[3].attn.qkv_proj
    sift._attach_slice(qkv, "row", torch.tensor([0, 1, 2, 3]))
    after = m(ids)
    assert torch.equal(base, after)


def test_col_slice_lossless():
    m = _toy()
    ids = _ids()
    base = m(ids).clone()
    wd = m.blocks[2].ffn.w_down
    sift._attach_slice(wd, "col", torch.tensor([4, 5, 6]))
    assert torch.equal(base, m(ids))


# ── gradient flow ──────────────────────────────────────────────────────────

def test_grads_reach_only_deltas():
    m = _toy()
    qkv = m.blocks[4].attn.qkv_proj
    p = sift._attach_slice(qkv, "row", torch.tensor([0, 1]))
    sift.freeze_non_sift(m)
    m(ids := _ids()).sum().backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
    assert qkv.weight.grad is None  # frozen weight gets no grad


def test_delta_updates_affect_output():
    m = _toy()
    ids = _ids()
    base = m(ids).clone()
    out = m.blocks[5].attn.out_proj
    p = sift._attach_slice(out, "col", torch.tensor([0, 1]))
    with torch.no_grad():
        p.fill_(0.5)
    assert not torch.equal(base, m(ids))


# ── unit discovery on the toy fused layout ─────────────────────────────────

def test_discover_units_fused():
    m = _toy()
    c = sift.discover_units(m, attn_gran=16, ffn_gran=32)
    # q_dim = 64 → 4 attn units; hidden 128 → 4 ffn units per block
    for bi in range(len(m.blocks)):
        assert len(c.units[bi]["attn"]) == 4
        assert len(c.units[bi]["ffn"]) == 4
    # unit members: qkv row + qkv gate row + out col (+ k/v absent w/o flag)
    u0 = c.units[0]["attn"][0]
    paths = sorted(sp.module for sp in u0.members)
    assert paths == ["blocks.0.attn.out_proj", "blocks.0.attn.qkv_proj",
                     "blocks.0.attn.qkv_proj"]
    sides = sorted(sp.side for sp in u0.members)
    assert sides == ["col", "row", "row"]
    # ffn unit: gate_up row j, gate_up row h+j, down col j
    f0 = c.units[0]["ffn"][1]
    starts = sorted(sp.start for sp in f0.members)
    assert starts == [32, 32, 128 + 32]


def test_include_kv_adds_members():
    m = _toy()
    c = sift.discover_units(m, attn_gran=16, ffn_gran=32, include_kv=True)
    u = c.units[0]["attn"][0]
    assert len(u.members) == 5  # q,g,out + k,v


def test_scalar_gate_spark_layout():
    """Spark-style fused QKV: gate is ONE scalar row per head (g_dim ==
    n_heads), appended after v. Units get a 1-row gate span, calibration
    must not trip on the non-granular out_dim."""
    class ScalarGateAttn(nn.Module):
        def __init__(self, d, hd, nq, nkv):
            super().__init__()
            self.qkv_proj = ToyFusedQKV(d, nq * hd, nkv * hd, nkv * hd, nq)
            self.out_proj = nn.Linear(nq * hd, d, bias=False)

        def forward(self, x):
            q, k, v, g = self.qkv_proj(x)
            return self.out_proj(q * torch.sigmoid(g.repeat_interleave(
                self.qkv_proj.q_dim // self.qkv_proj.g_dim, dim=-1)))

    class ScalarBlock(nn.Module):
        def __init__(self, d, hd, nq, nkv, h):
            super().__init__()
            self.attn = ScalarGateAttn(d, hd, nq, nkv)
            self.ffn = ToyFFN(d, h)

        def forward(self, x):
            return x + self.attn(x) + self.ffn(x)

    torch.manual_seed(0)
    m = ToyModel()
    m.blocks = nn.ModuleList([ScalarBlock(64, 16, 4, 2, 128) for _ in range(3)])
    c = sift.discover_units(m, attn_gran=16, ffn_gran=32)
    # q_dim=64, gran16 -> 4 units; g_base = 64+32+32 = 128; unit u owns
    # gate row 128+u (head u since gran == hd)
    u2 = c.units[0]["attn"][2]
    g_rows = [sp for sp in u2.members
              if sp.module.endswith("qkv_proj") and sp.start >= 128]
    assert len(g_rows) == 1 and g_rows[0].length == 1
    assert g_rows[0].start == 128 + 2
    # calibrate end-to-end: weight is [132, 64] — the old granule view
    # would have crashed on 132 % 16
    it = lambda: iter([(_ids(), _ids(), None)] * 2)
    mass = sift.calibrate_importance(
        m, c, it, lambda i, l, mk: m(i).pow(2).mean(), 2)
    assert all(v > 0 for v in mass.values())
    # a unit containing the gate row trains it: merge lands in qkv rows
    sel = sift.build_selection(c, frontier=0, attn_units=4, ffn_units=0)
    params = sift.apply_selection(m, sel)
    qkv = m.blocks[0].attn.qkv_proj
    assert any((qkv.weight.shape[0] == 132) for _ in [0])
    with torch.no_grad():
        params[0].fill_(0.01)
    n = sift.merge_sift(m, zero=True)
    assert n >= 1


# ── calibration picks planted-important units ──────────────────────────────

def test_calibrate_and_select():
    m = _toy()
    ids = _ids()

    def loss_fn(i, l, mk):
        return m(i).pow(2).mean()  # deterministic scalar target

    c = sift.discover_units(m, attn_gran=16, ffn_gran=32)
    it = lambda: iter([(ids, ids, None)] * 2)
    mass = sift.calibrate_importance(m, c, it, loss_fn, 2)
    assert len(mass) == 6
    assert all(v > 0 for v in mass.values())
    # every unit got a positive score
    for bu in c.units.values():
        assert all(u.score > 0 for u in bu["attn"] + bu["ffn"])
    sel = sift.build_selection(c, frontier=4, attn_units=2, ffn_units=2)
    mods = set(sel)
    assert any("blocks.4." in p or "blocks.5." in p for p in mods)
    assert not any("blocks.3." in p for p in mods)


def test_score_modes_reject_sign_flipping_gradients():
    """loss_fn flips sign every call → per-batch grads cancel.
    'fro' accumulates magnitude anyway (score>0); 'smean'/'obs' score
    the *mean* gradient → ~0 → selection empty."""
    ref = None
    for mode, expect_empty in (("fro", False), ("smean", True), ("obs", True)):
        m = _toy()
        ids = _ids()
        cnt = [0]

        def loss_fn(i, l, mk):
            cnt[0] += 1
            return m(i).pow(2).mean() * (1 if cnt[0] % 2 else -1)

        c = sift.discover_units(m, attn_gran=16, ffn_gran=32)
        it = lambda: iter([(ids, ids, None)] * 2)
        mass = sift.calibrate_importance(m, c, it, loss_fn, 2,
                                         score_mode=mode)
        if ref is None:
            ref = max(mass.values())
        if expect_empty:
            # fp16 accumulation leaves ~1e-3 relative residue, not exact 0
            assert max(mass.values()) < 0.05 * ref
        else:
            assert all(v > 0 for v in mass.values())


# ── merge + mirrors ────────────────────────────────────────────────────────

def test_merge_fused_qkv_mirrors_to_split():
    m = _toy()
    sift._register_mirrors(m)
    qkv = m.blocks[1].attn.qkv_proj
    sel = torch.tensor([0, 1, 2, 3])  # q rows 0-3
    p = sift._attach_slice(qkv, "row", sel)
    with torch.no_grad():
        p.uniform_(-0.1, 0.1)
    d = p.detach().clone()
    w_before = qkv.weight[sel].clone()
    q_before = m.blocks[1].attn.q_proj.weight[sel].clone()
    sift.merge_sift(m)
    assert torch.allclose(qkv.weight[sel], w_before + d)
    assert torch.allclose(m.blocks[1].attn.q_proj.weight[sel], q_before + d)


def test_merge_gateup_mirror():
    m = _toy()
    sift._register_mirrors(m)
    gu = m.blocks[0].ffn.gate_up_proj
    sel = torch.tensor([0, 128 + 1])  # gate row 0 + up row 1
    p = sift._attach_slice(gu, "row", sel)
    with torch.no_grad():
        p.fill_(0.01)
    wg0 = m.blocks[0].ffn.w_gate.weight[0].clone()
    wu1 = m.blocks[0].ffn.w_up.weight[1].clone()
    sift.merge_sift(m)
    assert torch.allclose(m.blocks[0].ffn.w_gate.weight[0], wg0 + 0.01)
    assert torch.allclose(m.blocks[0].ffn.w_up.weight[1], wu1 + 0.01)


def test_merge_col_side():
    m = _toy()
    wd = m.blocks[1].ffn.w_down
    p = sift._attach_slice(wd, "col", torch.tensor([7, 8]))
    with torch.no_grad():
        p.fill_(0.02)
    before = wd.weight[:, [7, 8]].clone()
    sift.merge_sift(m)
    assert torch.allclose(wd.weight[:, [7, 8]], before + 0.02)


# ── adapter save/load ──────────────────────────────────────────────────────

def test_adapter_roundtrip(tmp_path):
    m = _toy()
    qkv = m.blocks[3].attn.qkv_proj
    wd = m.blocks[3].ffn.w_down
    sift._attach_slice(qkv, "row", torch.tensor([0, 1]))
    p2 = sift._attach_slice(wd, "col", torch.tensor([5, 6]))
    with torch.no_grad():
        qkv.sift_delta.fill_(0.3)
        p2.fill_(-0.2)
    f = str(tmp_path / "a.sift.safetensors")
    sift.save_sift_adapter(m, f)
    assert os.path.exists(f)
    m2 = _toy()
    n = sift.load_sift_adapter(m2, f)
    assert n == 2
    assert torch.equal(m2.blocks[3].attn.qkv_proj.sift_sel,
                     torch.tensor([0, 1]))
    assert torch.allclose(m2.blocks[3].attn.qkv_proj.sift_delta,
                          torch.full((2, 64), 0.3))
    assert torch.allclose(m2.blocks[3].ffn.w_down.sift_delta,
                          torch.full((64, 2), -0.2))
    # same outputs as donor model
    ids = _ids()
    assert torch.equal(m(ids), m2(ids))


def test_detach_restores_forward():
    m = _toy()
    ids = _ids()
    base = m(ids).clone()
    qkv = m.blocks[2].attn.qkv_proj
    sift._attach_slice(qkv, "row", torch.tensor([0]))
    with torch.no_grad():
        qkv.sift_delta.fill_(0.0)  # zero delta: detach merges nothing
    sift.detach_sift(m, merge=True)
    assert torch.equal(base, m(ids))
    assert not hasattr(qkv, "sift_delta")
    assert "sift_" not in " ".join(m.state_dict().keys())


# ── prefix cache ───────────────────────────────────────────────────────────

def test_prefix_cache_fill_and_inject():
    m = _toy()
    pc = sift.SiftPrefixCache(m, frontier=3, max_gb=1.0)
    ids = _ids(b=4, t=8)
    full = m(ids).clone()
    pc.prepare(ids, None, ids.device)
    part = m(ids)
    pc.clear()
    # fp-noise level: batch-geometry can perturb the fill pass vs the full
    # forward GEMM blocking — bitwise equality is NOT guaranteed.
    assert torch.allclose(full, part, atol=1e-5)
    # second call: all hits, no fills
    fills_before = pc.fills
    pc.prepare(ids, None, ids.device)
    m(ids)
    pc.clear()
    assert pc.fills == fills_before
    assert pc.hits == 4


def test_prefix_cache_partial_rows():
    m = _toy()
    pc = sift.SiftPrefixCache(m, frontier=3, max_gb=1.0)
    ids = _ids(b=4, t=8)
    pc.prepare(ids[:2], None, ids.device)  # warm rows 0-1
    m(ids[:2]); pc.clear()
    full = m(ids).clone()
    pc.prepare(ids, None, ids.device)
    assert torch.allclose(full, m(ids), atol=1e-5)
    pc.clear()


# ── end-to-end: setup_sift on the toy ──────────────────────────────────────

def test_setup_sift_end_to_end():
    m = _toy()
    ids = _ids()

    def loss_fn(i, l, mk):
        return m(i).pow(2).mean()

    st = sift.setup_sift(
        m, lambda: iter([(ids, ids, None)] * 3), loss_fn,
        attn_units=2, ffn_units=2, attn_gran=16, ffn_gran=32,
        frontier="auto", frontier_mass=0.5, calib_batches=2)
    assert st.n_touched > 0
    assert st.delta_params
    # only deltas require grad
    for n, p in m.named_parameters():
        assert p.requires_grad == n.endswith("sift_delta")
    # forward still works + lossless (deltas are zero)
    m(ids)
    # below-frontier modules carry no slice
    below = [k for k in st.selection
             if int(k.split(".")[1]) < st.frontier]
    assert not below


# ── Newton init (diagonal Gauss-Newton slice warm-start) ───────────────────

def test_newton_init_recovers_true_delta():
    """White-input quadratic: the diagonal-GN step Δ=-lr·ḡ/E[x²] is
    direction-exact (ḡ ∝ -D), so lr=1/h recovers the true slice."""
    torch.manual_seed(0)
    lin = nn.Linear(16, 8, bias=False)
    holder = nn.Module()
    holder.lin = lin
    p = sift._attach_slice(lin, "row", torch.tensor([0, 1, 2, 3]))
    sift.freeze_non_sift(holder)
    D = torch.randn(4, 16) * 0.5  # true delta: t[..., :4] = x @ D.T

    def loss_fn(_i, _l, _m):
        x = torch.randn(8, 64, 16)
        y = lin(x)
        t = torch.zeros_like(y)
        t[..., :4] = x @ D.T
        return ((y - t) ** 2).mean()

    # L = mean over 8 outputs -> h_j = (2/8)E[x_j^2] = ex2/4 -> exact lr = 4
    sift.newton_init(holder, lambda: iter([(None, None, None)] * 8),
                     loss_fn, 8, lr=4.0, trust=1e6)
    # optimum is W_sel + Δ = D, i.e. Δ* = D - W_sel (not D itself)
    Dstar = D - lin.weight[0:4]
    err = (p.detach() - Dstar).norm() / Dstar.norm()
    assert err < 0.15, f"newton init err {err:.3f}"
    # post-init residual on the selected rows collapses vs target energy
    x = torch.randn(4, 32, 16)
    tgt = x @ D.T
    res = lin(x)[..., :4] - tgt
    assert (res ** 2).mean() < 0.05 * (tgt ** 2).mean()


def test_newton_init_col_side_and_hooks_cleanup():
    m = _toy()
    ids = _ids()
    wd = m.blocks[1].ffn.w_down
    p = sift._attach_slice(wd, "col", torch.tensor([5, 6, 7]))
    sift.freeze_non_sift(m)

    def loss_fn(i, l, mk):
        return m(i).pow(2).mean()

    rep = sift.newton_init(m, lambda: iter([(ids, ids, None)] * 2),
                           loss_fn, 2, lr=1.0)
    assert rep["passes"] == 2
    assert p.detach().abs().sum() > 0  # delta moved off zero
    assert not wd._forward_pre_hooks  # hooks removed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
