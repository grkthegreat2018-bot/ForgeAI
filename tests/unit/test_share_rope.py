"""share_rope (d2): identical RotaryEmbedding tables fold to one instance.

Layers with the same (dim, theta, scaling, partial) build byte-identical
cos/sin buffers, so rebinding duplicates is function-exact. Verifies:
  - same-param layers share ONE RotaryEmbedding instance
  - distinct-param layers (e.g. SWA theta) keep their own
  - logits are bit-exact vs the unshared model
"""
import torch

from forge.config import get_config
from forge.model_loader import ConfigurableResearchLLM


def _cfg(**kw):
    base = {
        "n_layers": 4,
        "layer_types": ["attention", "attention", "attention_swa",
                     "attention"],
        "sliding_window": 16,
        "max_seq_len": 64,
        "rope_layer_params": {
            "attention": {"rope_theta": 1_000_000.0,
                          "partial_rotary_factor": 0.5},
            "attention_swa": {"rope_theta": 10_000.0,
                              "partial_rotary_factor": 1.0},
        },
        "d_model": 64, "n_heads": 4, "n_kv_heads": 2, "head_dim": 16,
        "intermediate_size": 128, "vocab_size": 256,
    }
    base.update(kw)
    return get_config("forgelm_tiny", device="cpu", **base)


def _build(share: bool):
    torch.manual_seed(0)
    return ConfigurableResearchLLM(_cfg(share_rope=share)).eval()


def _attn_ropes(model):
    out = []
    for b in model.blocks:
        rope = getattr(getattr(b, 'attn', None), 'rope', None)
        if rope is not None:
            out.append(rope)
    return out


def test_identical_rope_layers_share_one_instance():
    model = _build(share=True)
    ropes = _attn_ropes(model)
    assert len(ropes) == 4
    # three full-attention layers (identical params) collapse to one table
    full = [ropes[0], ropes[1], ropes[3]]
    assert full[0] is full[1] is full[2]
    # the SWA layer has different theta/partial -> its own table
    assert ropes[2] is not full[0]
    assert ropes[2].base != full[0].base


def test_shared_rope_bit_exact():
    torch.manual_seed(0)
    a = _build(share=False)
    torch.manual_seed(0)
    b = _build(share=True)
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        la = a(idx)[0]
        lb = b(idx)[0]
    assert (la - lb).abs().max().item() == 0.0


def test_shared_rope_reduces_table_memory():
    model = _build(share=True)
    seen = {}
    for b in model.blocks:
        rope = getattr(getattr(b, 'attn', None), 'rope', None)
        if rope is not None:
            seen[id(rope)] = rope
    # 4 attention layers -> 2 distinct table sets (full + swa)
    assert len(seen) == 2
