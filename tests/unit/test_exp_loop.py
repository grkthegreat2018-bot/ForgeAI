"""CPU tests for forge/exp/loopdepth.py (E-loop recurrent depth)."""
import torch

from forge.exp.loopdepth import LoopGates, looped_decode_step, looped_prefill


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


def test_looped_prefill_one_iter_matches_baseline():
    model = _tiny_attn_model()
    ids = torch.randint(0, 256, (1, 32))
    logits_ref = model(ids)[0]
    logits, pasts, st = looped_prefill(model, ids, 1, 3, max_iters=1)
    assert st["iters_run"] == 1
    assert torch.allclose(logits, logits_ref, atol=1e-5)
    assert pasts is not None and len(pasts) == 4


def test_gates_zero_make_loop_lossless():
    model = _tiny_attn_model()
    ids = torch.randint(0, 256, (1, 32))
    logits0, _, _ = looped_prefill(model, ids, 1, 3, max_iters=0)
    g = torch.zeros(2)
    logits_g, _, st = looped_prefill(model, ids, 1, 3, max_iters=3,
                                   gates=g)
    assert st["iters_run"] == 3
    assert torch.allclose(logits_g, logits0, atol=1e-5)


def test_decode_step_runs_and_updates_cache():
    model = _tiny_attn_model()
    ids = torch.randint(0, 256, (1, 16))
    _, pasts, _ = looped_prefill(model, ids, 1, 3, max_iters=1)
    tok = torch.randint(0, 256, (1, 1))
    logits, pasts = looped_decode_step(model, tok, pasts, 1, 3,
                                       max_iters=2)
    assert logits.shape[-1] == 256
    # looped layers must have committed KV for the new token
    assert pasts[1] is not None and pasts[1][0].shape[-2] == 17


def test_loopgates_init_zero():
    g = LoopGates(3, device="cpu")
    assert torch.all(g() == 0)
