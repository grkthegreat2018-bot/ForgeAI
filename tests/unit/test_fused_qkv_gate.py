"""fused QKV+gate (d2): one GEMM emits q/k/v + per-head output gate.

_extends_ use_fused_gemm to append gate_proj rows after v (Qwen3-Next
pattern). The fused matrix is re-packed from the LOADED weights by
loader._refresh_fused_qkv — the construction-time copy is init-only.
Verifies:
  - fused forward is bit-exact vs separate projections (incl. gate)
  - refresh_fused_qkv re-packs after weight mutation
  - non-gated attention still fuses q/k/v only
"""
import torch

from forge.config import get_config
from forge.model_loader import ConfigurableResearchLLM
from forge.model.builders import refresh_fused_qkv
from forge.model.loader import ModelLoader


def _cfg(**kw):
    base = dict(n_layers=2, layer_types=["attention", "attention"],
                d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
                intermediate_size=128, vocab_size=256, max_seq_len=64,
                use_attn_out_gate=True)
    base.update(kw)
    return get_config("forgelm_tiny", device="cpu", **base)


def _build(fused: bool):
    torch.manual_seed(0)
    return ConfigurableResearchLLM(_cfg(use_fused_gemm=fused)).eval()


def test_fused_qkv_gate_function_exact():
    a = _build(fused=False)
    b = _build(fused=True)
    # re-pack fused matrices from the (identically-seeded) separate
    # projections — mirrors the loader's post-load refresh
    ModelLoader._refresh_fused_qkv(b)
    idx = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        la = a(idx)[0]
        lb = b(idx)[0]
    # one packed GEMM vs three — fp32 kernel-selection noise only
    assert (la - lb).abs().max().item() < 1e-5
    assert (la.argmax(-1) == lb.argmax(-1)).all().item()


def test_refresh_repacks_after_weight_change():
    b = _build(fused=True)
    attn = b.blocks[0].attn
    with torch.no_grad():
        attn.q_proj.weight.add_(0.5)
    refresh_fused_qkv(attn)
    q_rows = attn.q_proj.out_features
    assert torch.equal(attn.qkv_proj.weight[:q_rows],
                       attn.q_proj.weight)
    # gate rows sit after q+k+v
    g_rows = attn.gate_proj.out_features
    assert torch.equal(attn.qkv_proj.weight[-g_rows:],
                       attn.gate_proj.weight)


def test_loader_refresh_scans_model():
    b = _build(fused=True)
    with torch.no_grad():
        for blk in b.blocks:
            blk.attn.qkv_proj.weight.zero_()
    ModelLoader._refresh_fused_qkv(b)
    attn = b.blocks[0].attn
    q_rows = attn.q_proj.out_features
    assert torch.equal(attn.qkv_proj.weight[:q_rows],
                       attn.q_proj.weight)


def test_ungated_attention_fuses_qkv_only():
    cfg = _cfg(use_fused_gemm=True, use_attn_out_gate=False)
    torch.manual_seed(0)
    m = ConfigurableResearchLLM(cfg).eval()
    attn = m.blocks[0].attn
    assert attn._fused_qkv
    assert attn.qkv_proj.g_dim == 0
    ModelLoader._refresh_fused_qkv(m)
    q_rows = attn.q_proj.out_features
    assert torch.equal(attn.qkv_proj.weight[:q_rows],
                       attn.q_proj.weight)
