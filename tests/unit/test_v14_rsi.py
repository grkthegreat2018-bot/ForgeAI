"""V14-RSI tests: preset lineage, budget, subset flags, new heads, mamba3 fix."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forge.config import get_config, validate_preset_lineage


def _tiny_v14(**overrides):
    """Tiny V14-shaped config for CPU tests."""
    return get_config(
        "forgelm_v14_rsi",
        vocab_size=1024,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=1,
        head_dim=32,
        layer_types=["conv", "mamba", "conv", "attention"],
        moe_n_experts=4,
        moe_top_k=2,
        moe_d_ff=64,
        gdn2_n_heads=2,
        gdn2_head_dim=32,
        titan_layers=[1],
        titan_memory_rank=16,
        max_seq_len=512,
        **overrides,
    )


def test_preset_lineage_valid():
    assert validate_preset_lineage() == []


def test_preset_param_budget():
    """V14 must land ~2.5B total (int4 <= ~1.35GB)."""
    with torch.device("meta"):
        from forge.model.llm import ConfigurableResearchLLM

        m = ConfigurableResearchLLM(get_config("forgelm_v14_rsi"))
    total = sum(p.numel() for p in m.parameters())
    assert 2.2e9 <= total <= 2.8e9, f"total {total / 1e9:.3f}B outside 2.2-2.8B"


def test_layer_mix_and_subsets():
    cfg = _tiny_v14()
    with torch.device("meta"):
        from forge.model.llm import ConfigurableResearchLLM

        m = ConfigurableResearchLLM(cfg)
    types = [b.layer_type for b in m.blocks]
    assert types == ["conv", "mamba", "conv", "attention"]
    # titan subset [1] -> only block 1 has _memory
    have_mem = [i for i, b in enumerate(m.blocks) if b._memory is not None]
    assert have_mem == [1]
    # gdn2 applies to all blocks (gdn2_layers=None)
    assert all(b._gdn2 is not None for b in m.blocks)


def test_gdn2_layer_subset():
    cfg = _tiny_v14(gdn2_layers=[0, 3])
    with torch.device("meta"):
        from forge.model.llm import ConfigurableResearchLLM

        m = ConfigurableResearchLLM(cfg)
    have = [i for i, b in enumerate(m.blocks) if b._gdn2 is not None]
    assert have == [0, 3]


def test_mtp_adapter_heads_param_reduction():
    """Adapter mode: n*d^2 params vs n*d*V; requires tied head."""
    from forge.decoding.mtp import MTPModule

    full = MTPModule(d_model=128, vocab_size=1024, n_heads=4)
    adapt = MTPModule(d_model=128, vocab_size=1024, n_heads=4, adapter_heads=True)
    # Compare head-list params only (shared trunk is constant).
    n_full = sum(p.numel() for h in full.heads if h is not None for p in h.parameters())
    n_adapt = sum(p.numel() for h in adapt.heads if h is not None for p in h.parameters())
    assert n_adapt == 4 * 128 * 128 and n_full == 4 * 128 * 1024
    assert n_adapt < n_full * 0.15  # d/V scaling: ~8x at toy dims, ~26x real


def test_mtp_adapter_draft_mode():
    """Draft call mtp(hidden) returns per-head (B,1,V) logits."""
    from forge.decoding.mtp import MTPModule

    mtp = MTPModule(d_model=64, vocab_size=256, n_heads=3, adapter_heads=True)
    head_w = torch.nn.Linear(64, 256, bias=False)
    mtp.tie_head_to_model(head_w.weight)
    h = torch.randn(1, 1, 64)
    out = mtp(h)  # draft mode: single arg
    assert isinstance(out, list) and len(out) == 3
    assert out[0].shape == (1, 1, 256)


def test_edit_head_pointer_shapes():
    from forge.model.edit_head import CopySpanPointerHead, copy_span_loss

    head = CopySpanPointerHead(d_model=64, ptr_dim=32)
    src_h = torch.randn(10, 64)
    src_k = head.key_cache(src_h)
    h_t = torch.randn(1, 1, 64)
    s, e = head(h_t, src_k)
    assert s.shape == (1, 10) and e.shape == (1, 10)
    loss = copy_span_loss(s, e, torch.tensor([2]), torch.tensor([7]))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_verifier_head_and_bt_loss():
    from forge.model.edit_head import VerifierHead, verifier_bt_loss

    vh = VerifierHead(d_model=64, hidden=32)
    h = torch.randn(4, 64)
    s = vh(h)
    assert s.shape == (4,)
    loss = verifier_bt_loss(s[:2], s[2:])
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_model_builds_heads():
    cfg = _tiny_v14()
    with torch.device("meta"):
        from forge.model.llm import ConfigurableResearchLLM

        m = ConfigurableResearchLLM(cfg)
    assert m.edit_head is not None and m.verifier_head is not None
    assert m.mtp_module is not None and m.mtp_module.adapter_heads


def test_mamba3_bf16_out_proj_dtype():
    """R60 regression: complex scan output must cast back to weight dtype."""
    from forge.engine.mamba3 import Mamba3Block

    blk = Mamba3Block(d_model=64, d_state=8, d_conv=4, expand=2, dt_rank=8, layer_idx=0)
    blk = blk.to(torch.bfloat16)
    x = torch.randn(1, 8, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        out, _ = blk(x)
    assert out.dtype == torch.bfloat16
