"""CPU tests for forge/exp/abshead.py."""
import torch

from forge.exp.abshead import LatentThoughtHead, collect_pairs, future_pairs, hint_logits, hint_nll, train_latent


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


def test_head_shapes_and_zero_init():
    head = LatentThoughtHead(32, hidden=16, k=4)
    h = torch.randn(2, 8, 32)
    pred = head(h)
    assert pred.shape == h.shape
    assert torch.all(pred == 0)          # zero-init output layer
    assert torch.all(head.hint(h) == 0)  # gate=0


def test_forward_deterministic():
    head = LatentThoughtHead(32, hidden=16, k=4).eval()
    with torch.no_grad():
        head.out.weight.normal_(0, 0.1)   # nonzero output
        h = torch.randn(2, 8, 32)
        assert torch.equal(head(h), head(h))


def test_hint_is_future_delta():
    head = LatentThoughtHead(16, hidden=8, k=2)
    with torch.no_grad():
        head.out.weight.normal_(0, 0.1)
        head.gate.fill_(0.5)
        h = torch.randn(1, 4, 16)
        hint = head.hint(h)
        assert torch.allclose(hint, 0.5 * (head(h) - h))


def test_future_pairs_hidden_slices():
    ids = torch.arange(10).unsqueeze(0)
    hidden = torch.randn(1, 10, 8)
    src, tgt = future_pairs(ids, hidden, None, k=3, mode="hidden")
    assert src.shape == tgt.shape == (1, 7, 8)
    assert torch.equal(src, hidden[:, :7])
    assert torch.equal(tgt, hidden[:, 3:])  # h_{t+3}


def test_future_pairs_embed_mean():
    ids = torch.arange(10).unsqueeze(0)
    emb = torch.randn(64, 8)
    hidden = torch.randn(1, 10, 8)
    _src, tgt = future_pairs(ids, hidden, emb, k=2, mode="embed")
    # position t=0 target = mean(embed(tok1), embed(tok2))
    assert torch.allclose(tgt[0, 0], (emb[1] + emb[2]) / 2)
    # position t=7 target = mean(embed(tok8), embed(tok9))
    assert torch.allclose(tgt[0, 7], (emb[8] + emb[9]) / 2)


def test_hint_readout_lossless_at_gate_zero():
    model = _tiny_attn_model()
    head = LatentThoughtHead(model.config.d_model, hidden=32, k=2)
    with torch.no_grad():
        head.out.weight.normal_(0, 0.1)   # nonzero pred, gate still 0
    ids = torch.randint(0, 256, (1, 24))
    ref = model(ids)[0]
    out = model(ids, return_hidden=True)
    hinted = hint_logits(model, head, out[-1])
    assert torch.allclose(ref, hinted, atol=1e-5)


def test_open_gate_changes_logits():
    model = _tiny_attn_model()
    head = LatentThoughtHead(model.config.d_model, hidden=32, k=2)
    with torch.no_grad():
        head.out.weight.normal_(0, 0.1)
        head.gate.fill_(0.5)
    ids = torch.randint(0, 256, (1, 24))
    ref = model(ids)[0]
    out = model(ids, return_hidden=True)
    hinted = hint_logits(model, head, out[-1])
    assert not torch.allclose(ref, hinted, atol=1e-5)


def test_hint_nll_matches_model_loss_at_gate_zero():
    model = _tiny_attn_model()
    head = LatentThoughtHead(model.config.d_model, hidden=32, k=2)
    ids = torch.randint(0, 256, (1, 24))
    ref = float(model(ids[:, :-1], targets=ids[:, 1:])[1])
    assert abs(hint_nll(model, head, [ids]) - ref) < 1e-5


def test_collect_pairs_and_train_latent_smoke():
    torch.manual_seed(0)
    model = _tiny_attn_model()
    head = LatentThoughtHead(model.config.d_model, hidden=64, k=2)
    ids_list = [torch.randint(0, 256, (1, 32)) for _ in range(2)]
    src, tgt = collect_pairs(model, ids_list[0], k=2)
    assert src.shape == tgt.shape == (1, 30, model.config.d_model)
    hist = train_latent(model, head, ids_list, steps=12, lr=3e-3)
    assert len(hist["losses"]) == 12
    assert hist["losses"][-1] < hist["losses"][0]
    # gate must stay 0 through latent training (lossless contract)
    assert float(head.gate) == 0.0
    # model weights were not trained
    assert all(p.requires_grad for p in model.parameters())
