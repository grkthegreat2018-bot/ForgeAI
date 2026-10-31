"""CPU tests for forge/exp/diffdraft.py."""
import torch

from forge.exp.diffdraft import DiffusionDraftHead, _truncate_pasts, spec_verify


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


def test_denoise_shapes():
    head = DiffusionDraftHead(32, k=4)
    h = torch.randn(2, 32)
    z = torch.randn(2, 4, 32)
    assert head.denoise(h, z, 0.5).shape == z.shape
    assert head.denoise(h, z, torch.rand(2)).shape == z.shape


def test_sample_returns_ids():
    head = DiffusionDraftHead(32, k=4, steps=2)
    emb = torch.randn(64, 32)
    ids = head.sample(torch.randn(1, 32), emb)
    assert ids.shape == (1, 4) and ids.max() < 64


def test_spec_verify_commits_corrected_and_truncates():
    model = _tiny_attn_model()
    ids = torch.randint(0, 256, (1, 16))
    out = model(ids, use_cache=True)
    pasts, last = out[2], out[0][:, -1:]
    draft = torch.randint(0, 256, (1, 4))
    res = spec_verify(model, pasts, last, draft)
    assert 1 <= len(res["committed"]) <= 4
    # KV length must equal context + accepted draft tokens
    ctx = 16 + res["n_accepted"]
    assert res["kv"][0][0].shape[-2] == ctx


def test_truncate_pasts_tuples():
    k = torch.randn(1, 2, 10, 4)
    v = torch.randn(1, 2, 10, 4)
    pasts = _truncate_pasts([(k, v), None], 6)
    assert pasts[0][0].shape[-2] == 6 and pasts[1] is None
