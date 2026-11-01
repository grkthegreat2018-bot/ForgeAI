"""E-dynscale unit tests — HeadGate saliency gating + EarlyExit decode.

CPU-only: tiny all-attention ConfigurableResearchLLM built from the
forgelm_tiny preset with layer_types=None (no downloads, no checkpoints).

Covers:
  * HeadGate attach/detach, lossless-at-tau=0, gate math correctness,
    all tau_mode semantics, and EMA-driven gate recovery.
  * EarlyExitController manual decode loop parity vs the stock model()
    decode path, per-layer KV-cache skip bookkeeping, and
    teacher-forced first-exit NLL wiring (tau=inf == full-depth NLL,
    tau=0 → min_depth everywhere).
  * DiagTempScaler attach/fold equivalence.
"""

import torch
import torch.nn as nn

from forge.config import get_config
from forge.exp.dynscale import (
    DiagTempScaler,
    EarlyExitController,
    HeadGate,
    HeadGateController,
    OutGain,
    confidence,
    find_linear,
)
from forge.model_loader import ConfigurableResearchLLM, unpack_output_with_kv

LAYERS = 3
HEADS = 4
D_MODEL = 64
VOCAB = 97


def _model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = get_config(
        "forgelm_tiny",
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_layers=LAYERS,
        n_heads=HEADS,
        n_kv_heads=2,
        intermediate_size=2 * D_MODEL,
        layer_types=None,
        max_seq_len=64,
    )
    return ConfigurableResearchLLM(cfg).eval()


def _ids(n: int = 16, hi: int = VOCAB, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, hi, (1, n), generator=g)


def _prefill(model, ids):
    out = model(ids, use_cache=True, logits_last_only=True)
    return unpack_output_with_kv(out)


# ── HeadGate unit math ────────────────────────────────────────────────


def test_gate_list_modes():
    lin = nn.Linear(8, 8, bias=False)
    g = HeadGate(lin, n_heads=2, head_dim=4, tau=0.5, tau_mode="rel_mean")
    g.ema = [10.0, 1.0]
    assert g.gate_list() == [1.0, 0.0]  # 1.0 < 0.5 * 5.5
    g.tau_mode = "rel_max"
    assert g.gate_list() == [1.0, 0.0]  # 1.0 < 0.5 * 10
    g.tau_mode = "abs"
    g.tau = 5.0
    assert g.gate_list() == [1.0, 0.0]  # 1.0 < 5
    g.tau_mode = "frac"
    g.tau = 0.5
    assert g.gate_list() == [1.0, 0.0]  # lowest 50% = head 1
    g.tau = 0.0
    assert g.gate_list() is None  # tau=0 → no gating
    g.tau_mode = "off"
    g.tau = 0.9
    assert g.gate_list() is None


def test_gate_never_kills_all_heads():
    lin = nn.Linear(8, 8, bias=False)
    g = HeadGate(lin, n_heads=2, head_dim=4, tau=2.0, tau_mode="rel_max")
    g.ema = [5.0, 1.0]
    assert g.gate_list() == [1.0, 0.0]  # best head survives


def test_forward_applies_gate_to_out_proj_input():
    lin = nn.Linear(8, 8, bias=False)
    g = HeadGate(lin, n_heads=2, head_dim=4, tau=0.9, tau_mode="rel_max")
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = g(x)
    gate = torch.tensor(g.last_gate).view(1, 1, 2, 1)
    expect = lin((x.view(1, 3, 2, 4) * gate).reshape(1, 3, 8))
    assert torch.equal(y, expect)


def test_ema_measures_raw_input_so_gates_can_recover():
    """A gated head that starts producing large outputs must recover —
    saliency is read pre-gate, not from the zeroed channel."""
    lin = nn.Linear(8, 8, bias=False)
    g = HeadGate(lin, n_heads=2, head_dim=4, ema_alpha=0.9, tau=0.9, tau_mode="rel_max")
    g.ema = [10.0, 0.01]
    x = torch.zeros(1, 1, 8)
    x.view(1, 1, 2, 4)[0, 0, 1] = 1000.0  # head 1 now dominates
    with torch.no_grad():
        g(x)
    # sal[1] = ||[1000]*4|| = 2000 → ema ≈ [0.9*10, 0.9*.01 + .1*2000]
    # = [9.0, 200.009] → thr = 0.9*200: head 0 falls below, head 1 recovers.
    assert g.last_gate == [0.0, 1.0]
    assert abs(g.ema[1] - 200.009) < 1e-2


def test_update_every_skips_ema_refresh():
    lin = nn.Linear(8, 8, bias=False)
    g = HeadGate(lin, n_heads=2, head_dim=4, update_every=4)
    xs = [torch.randn(1, 1, 8) for _ in range(4)]
    with torch.no_grad():
        g(xs[0])  # call 1: EMA init
        e0 = list(g.ema)
        g(xs[1])  # call 2: skipped
        g(xs[2])  # call 3: skipped
        assert g.ema == e0
        g(xs[3])  # call 4: updates
        assert g.ema != e0


# ── HeadGateController on the tiny model ──────────────────────────────


def test_controller_attach_detach_lossless():
    m = _model()
    ids = _ids(12)
    with torch.no_grad():
        ref = m(ids)[0]
    ctl = HeadGateController(m, tau=0.0)
    n = ctl.attach()
    assert n == LAYERS  # all attention blocks
    with torch.no_grad():
        out = m(ids)[0]
    assert torch.equal(out, ref)  # tau=0 → bit-exact
    ctl.detach()
    assert all(isinstance(b.attn.out_proj, nn.Linear) for b in m.blocks)
    with torch.no_grad():
        out2 = m(ids)[0]
    assert torch.equal(out2, ref)


def test_controller_gating_changes_output_and_stats():
    m = _model()
    ids = _ids(12)
    with torch.no_grad():
        ref = m(ids)[0]
    ctl = HeadGateController(m, tau=0.5, tau_mode="frac")
    ctl.attach()
    with torch.no_grad():
        m(ids)  # populates EMA + gates
        out = m(ids)[0]
    st = ctl.stats()
    assert st["heads_total"] == LAYERS * HEADS
    # frac=0.5 → exactly 2 of 4 heads gated per layer (post-EMA)
    assert st["frac_gated"] == 0.5
    assert not torch.equal(out, ref)  # gating changed output
    ctl.detach()


def test_controller_set_tau_and_disable():
    m = _model()
    ids = _ids(12)
    with torch.no_grad():
        ref = m(ids)[0]
    ctl = HeadGateController(m, tau=0.5, tau_mode="frac")
    ctl.attach()
    with torch.no_grad():
        m(ids)
    ctl.set_tau(0.0)
    with torch.no_grad():
        out = m(ids)[0]
    assert torch.equal(out, ref)  # tau=0 restores exactly
    ctl.set_tau(0.75)
    ctl.set_enabled(False)
    with torch.no_grad():
        out2 = m(ids)[0]
    assert torch.equal(out2, ref)  # disabled = passthrough
    ctl.detach()


# ── EarlyExitController ───────────────────────────────────────────────


def test_ee_decode_parity_with_model_loop():
    """Manual block loop (tau=None) must match stock model() greedy
    decode token-for-token — validates position_ids + past plumbing."""
    m = _model()
    ids = _ids(8)
    ee = EarlyExitController(model=m)
    res = ee.decode(*_prefill(m, ids), ids.shape[1], 6, tau=None)
    logits, past = _prefill(m, ids)
    ref = []
    with torch.no_grad():
        for _ in range(6):
            tok = logits[:, -1].argmax(-1, keepdim=True)
            ref.append(int(tok[0, 0]))
            logits, past = unpack_output_with_kv(m(tok, past_key_values=past, use_cache=True))
    assert res["tokens"] == ref
    assert res["mean_depth"] == LAYERS


def test_ee_early_exit_skips_blocks_and_kv():
    m = _model()
    ids = _ids(8)
    ee = EarlyExitController(model=m)
    # margin >= 0 is always true → every token exits at min_depth=2
    res = ee.decode(*_prefill(m, ids), ids.shape[1], 4, tau=0.0, metric="margin", min_depth=2)
    assert res["mean_depth"] == 2
    P = ids.shape[1]
    for i, kv in enumerate(res["past"]):
        L = kv[0].shape[-2] if kv is not None else 0
        # layers 0,1 ran 4 decode steps; layers ≥2 were skipped → P only
        assert (P + 4 if i < 2 else P) == L


def test_exit_nll_tau_inf_parity():
    """tau=inf → no position ever exits → identical to stock CE loss."""
    m = _model()
    ids = _ids(14)
    ee = EarlyExitController(model=m)
    nll, dep = ee.exit_nll(ids, tau=float("inf"), metric="prob")
    out = m(ids[:, :-1], targets=ids[:, 1:])
    assert abs(nll - float(out[1])) < 1e-5
    assert dep == LAYERS


def test_exit_nll_tau_none_parity_and_tau0_depth():
    m = _model()
    ids = _ids(14)
    ee = EarlyExitController(model=m)
    nll_none, _ = ee.exit_nll(ids, tau=None)
    out = m(ids[:, :-1], targets=ids[:, 1:])
    assert abs(nll_none - float(out[1])) < 1e-9
    # tau=0 prob: every position clears at the first check (min_depth=2)
    nll0, dep0 = ee.exit_nll(ids, tau=0.0, metric="prob", min_depth=2)
    assert dep0 == 2.0
    assert nll0 != nll_none  # early logits differ


def test_confidence_metrics():
    z = torch.tensor([[[0.0, 10.0, -1.0, -2.0]]])
    p = confidence(z, "prob")
    mg = confidence(z, "margin")
    lm = confidence(z, "logit_margin")
    assert p.shape == (1, 1)
    assert p.item() > 0.99
    assert 0.0 < mg.item() < 1.0
    assert lm.item() == 10.0  # top1=10.0, top2=0.0


# ── DiagTempScaler ────────────────────────────────────────────────────


def test_diagtemp_attach_and_fold_equivalence():
    m = _model()
    ids = _ids(10)
    sc = DiagTempScaler(m)
    n = sc.attach()
    assert n == 2 * LAYERS  # out_proj + w_down
    # non-unit gains: fold must preserve the scaled function exactly
    for i, w in enumerate(sc.wraps):
        w["gain_mod"].gain.data.fill_(1.0 + 0.05 * (i + 1))
    with torch.no_grad():
        ref = m(ids)[0]
    sc.fold()
    assert all(isinstance(b.attn.out_proj, nn.Linear) or not isinstance(b.attn.out_proj, OutGain) for b in m.blocks)
    with torch.no_grad():
        out = m(ids)[0]
    assert torch.allclose(out, ref, atol=1e-5)


def test_diagtemp_detach_restores():
    m = _model()
    ids = _ids(10)
    with torch.no_grad():
        ref = m(ids)[0]
    sc = DiagTempScaler(m)
    sc.attach()
    for w in sc.wraps:
        w["gain_mod"].gain.data.fill_(2.0)
    sc.detach()  # discard, don't fold
    with torch.no_grad():
        out = m(ids)[0]
    assert torch.equal(out, ref)


def test_find_linear_through_wrappers():
    lin = nn.Linear(8, 8, bias=False)
    g = HeadGate(lin, n_heads=2, head_dim=4)
    og = OutGain(g, 8)
    assert find_linear(og) is lin
    assert find_linear(g) is lin
    assert find_linear(lin) is lin
    assert find_linear(nn.ReLU()) is None
