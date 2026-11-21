"""Q101: chunked policy-term helpers vs full-materialization reference.

Covers ChunkedGatherLogProbs (value + grads, incl. logit_scale) and
chunked_token_entropy, plus the model._skip_head escape hatch.
"""
import torch
import torch.nn.functional as F

from forge.training.losses.chunked_ce import (
    chunked_gather_logprobs, chunked_token_entropy)


def _rand(n=37, h=64, v=911):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, h, generator=g, requires_grad=True)
    w = torch.randn(v, h, generator=g, requires_grad=True)
    t = torch.randint(0, v, (n,), generator=g)
    return x, w, t


def test_gather_logprobs_matches_full_path():
    x, w, t = _rand()
    ref = F.log_softmax((x @ w.t()), dim=-1).gather(
        1, t.unsqueeze(1)).squeeze(1)
    got = chunked_gather_logprobs(x, w, t, chunk_size=8)
    assert torch.allclose(got, ref, atol=1e-5)


def test_gather_logprobs_logit_scale():
    x, w, t = _rand()
    ref = F.log_softmax((x @ w.t()) / 10.0, dim=-1).gather(
        1, t.unsqueeze(1)).squeeze(1)
    got = chunked_gather_logprobs(x, w, t, chunk_size=8, logit_scale=10.0)
    assert torch.allclose(got, ref, atol=1e-5)


def test_gather_logprobs_grad_matches():
    x, w, t = _rand()
    x2, w2 = x.detach().clone().requires_grad_(True), \
        w.detach().clone().requires_grad_(True)
    ref = F.log_softmax((x @ w.t()), dim=-1).gather(
        1, t.unsqueeze(1)).squeeze(1)
    g = torch.randn_like(ref)
    ref.backward(g)
    got = chunked_gather_logprobs(x2, w2, t, chunk_size=8)
    got.backward(g)
    assert torch.allclose(x2.grad, x.grad, atol=1e-4)
    assert torch.allclose(w2.grad, w.grad, atol=1e-4)


def test_kl_divergence_matches_full_path_and_grads():
    from forge.training.losses.chunked_ce import chunked_kl_divergence
    x, w, _ = _rand()
    xr = torch.randn_like(x)  # detached reference
    lp_c = F.log_softmax(x @ w.t(), dim=-1)
    lp_r = F.log_softmax(xr @ w.t(), dim=-1)
    ref = (lp_c.exp() * (lp_c - lp_r)).sum(-1)
    got = chunked_kl_divergence(x, xr, w, chunk_size=8)
    assert torch.allclose(got, ref, atol=1e-5)
    # grads: ref path via autograd vs chunked custom backward
    x2, xr2 = x.detach().clone().requires_grad_(True), \
        xr.detach().clone().requires_grad_(True)
    kl = chunked_kl_divergence(x2, xr2, w, chunk_size=8)
    kl.sum().backward()
    ref.sum().backward()
    assert torch.allclose(x2.grad, x.grad, atol=1e-4)
    assert xr2.grad is None  # reference carries no gradient


def test_token_entropy_matches_full_path():
    x, w, _ = _rand()
    lp = F.log_softmax((x @ w.t()), dim=-1)
    ref = -(lp.exp() * lp).sum(dim=-1)
    got = chunked_token_entropy(x.detach(), w, chunk_size=8)
    assert torch.allclose(got, ref, atol=1e-5)


def test_skip_head_returns_none_logits():
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM
    cfg = get_config("forgelm_tiny", layer_types=["attention"] * 2)
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
    ids = torch.randint(0, 256, (1, 16))
    model._skip_head = True
    logits, loss, hidden = model(ids, return_hidden=True)
    assert logits is None and loss is None
    assert hidden.shape[:2] == ids.shape
    assert model._skip_head is False  # flag consumed
    # hidden feeds the same numbers the head would have produced
    ref_logits = model.head(hidden)
    logits2, _, _ = model(ids, return_hidden=True)
    assert torch.allclose(ref_logits, logits2, atol=1e-5)


def _real_tiny():
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM
    cfg = get_config("forgelm_tiny", layer_types=["attention"] * 2)
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


class _WordTok:
    pad_token_id = 0

    def __call__(self, text, **kw):
        from types import SimpleNamespace
        ids = [(ord(w[0]) % 14) + 1 for w in str(text).split()]
        e = SimpleNamespace()
        e.input_ids = torch.tensor([ids], dtype=torch.long)
        return e


def test_grpo_trainer_hidden_path_matches_logits():
    """End-to-end on a real model: chunked gather over _skip_head hidden
    must equal the old logits-path gather (Q101 closed loop)."""
    from forge.self_play.grpo_trainer import GRPOConfig, GRPOTrainer
    model = _real_tiny()
    tr = GRPOTrainer(model, _WordTok(), None, device="cpu",
                     config=GRPOConfig())
    ids = torch.randint(0, 14, (2, 24)) + 1
    mask = torch.ones_like(ids)

    m, tensor, is_h = tr._hidden_forward(ids, mask)
    assert is_h and tensor.shape[-1] == 64 or tensor.shape[-1] < 257
    tgt = torch.randint(0, 14, (tensor.shape[1],)) + 1

    chunked = tr._gather_sol_logp(tensor[0], m, tgt)
    ref = F.log_softmax(model.head(tensor[0]).float(), dim=-1).gather(
        1, tgt.unsqueeze(1)).squeeze(-1)
    assert torch.allclose(chunked, ref, atol=1e-4)

    ent = tr._sol_entropy(tensor[0], m)
    lp_full = F.log_softmax(model.head(tensor[0]).float(), dim=-1)
    assert torch.allclose(ent, -(lp_full.exp() * lp_full).sum(-1),
                          atol=1e-4)
