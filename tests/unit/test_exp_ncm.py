"""CPU tests for forge/exp/ncm.py."""

import torch

from forge.exp.ncm import NCMLayer, attach_ncm


def _tiny_attn_model():
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM

    cfg = get_config("forgelm_tiny", layer_types=["attention"] * 4)
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


def test_ncm_shapes_and_gate_zero():
    ncm = NCMLayer(32)
    x = torch.randn(2, 8, 32)
    y = ncm(x)
    assert y.shape == x.shape
    assert torch.all(y == 0)  # gate=0


def test_attach_is_lossless():
    model = _tiny_attn_model()
    ids = torch.randint(0, 256, (1, 24))
    ref = model(ids)[0]
    attach_ncm(model, [1, 2])
    out = model(ids)[0]
    assert torch.allclose(ref, out, atol=1e-5)


def test_open_gate_changes_output():
    model = _tiny_attn_model()
    ids = torch.randint(0, 256, (1, 24))
    ref = model(ids)[0]
    attach_ncm(model, [1])
    model.blocks[1]._ncm.gate.data.fill_(1.0)
    model.blocks[1]._ncm_gate_zero = None
    out = model(ids)[0]
    assert not torch.allclose(ref, out, atol=1e-5)


def test_recurrence_is_causal():
    ncm = NCMLayer(16)
    ncm.gate.data.fill_(1.0)
    x = torch.randn(1, 10, 16)
    x2 = x.clone()
    x2[0, 7:] = 99.0  # perturb the tail — early outputs must not move
    y1, y2 = ncm(x), ncm(x2)
    assert torch.allclose(y1[0, :6], y2[0, :6], atol=1e-5)
