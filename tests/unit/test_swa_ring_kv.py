"""swa_ring_kv (d2 strict-lossless): sliding-window layers store KV in a
ring buffer of capacity = window (slot = position % W).

Attention is permutation-invariant over keys and the SWA mask never
admits keys older than W, so wrapped storage is output-identical to the
masked linear buffer — while decode length becomes unbounded on those
layers. Verifies:
  - ring writes wrap (append + append_t/pos_t)
  - get_layer returns the last-C keys in position order
  - end-to-end decode is function-identical vs a linear-buffer model
    decoding past 2x the window (same argmax trajectory; logits agree
    within fp32 kernel-tiling noise — SDPA sees different lengths so
    bitwise equality is not guaranteed)
"""
import torch

from forge.config import get_config
from forge.model.kv_cache import create_kv_cache
from forge.model_loader import ConfigurableResearchLLM

W = 8


def _cfg(**kw):
    base = {
        "n_layers": 3,
        "layer_types": ["attention", "attention_swa", "attention"],
        "sliding_window": W,
        "max_seq_len": 64,
        "d_model": 64, "n_heads": 4, "n_kv_heads": 2, "head_dim": 16,
        "intermediate_size": 128, "vocab_size": 256,
    }
    base.update(kw)
    return get_config("forgelm_tiny", device="cpu", **base)


def _build(ring: bool):
    torch.manual_seed(0)
    return ConfigurableResearchLLM(_cfg(swa_ring_kv=ring)).eval()


def _cache(model, max_total=64):
    return create_kv_cache(model, max_total=max_total, batch=1,
                           device=torch.device("cpu"))


def test_ring_buffer_allocated_at_window():
    model = _build(ring=True)
    cache = _cache(model)
    # layer 1 is the SWA layer -> cap W; the others stay max_seq_len
    assert cache.ring_cap(1) == W
    assert cache.ring_cap(0) is None
    assert cache.k_caches[1].shape[2] == W
    assert cache.k_caches[0].shape[2] == 64


def test_ring_append_wraps_and_get_layer_orders():
    torch.manual_seed(0)
    model = _build(ring=True)
    cache = _cache(model)
    ks = torch.randn(1, 2, 2 * W + 3, 16)
    vs = torch.randn(1, 2, 2 * W + 3, 16)
    # prefill past the window in one append
    cache.append(1, ks, vs)
    cache.advance(ks.shape[2])
    P = ks.shape[2]
    k, v = cache.get_layer(1)
    # last W positions, in position order
    assert torch.equal(k, ks[:, :, P - W:])
    assert torch.equal(v, vs[:, :, P - W:])
    # single-token appends wrap too
    k2 = torch.randn(1, 2, 1, 16)
    v2 = torch.randn(1, 2, 1, 16)
    cache.append(1, k2, v2)
    cache.advance(1)
    k, v = cache.get_layer(1)
    want_k = torch.cat([ks[:, :, P - W + 1:], k2], dim=2)
    assert torch.equal(k, want_k)


def test_ring_append_t_pos_t():
    torch.manual_seed(0)
    model = _build(ring=True)
    cache = _cache(model)
    cache.enable_tensor_pos()
    for p in range(W + 3):
        k = torch.full((1, 2, 1, 16), float(p))
        v = torch.full((1, 2, 1, 16), float(p))
        cache.append_t(1, k, v)
        cache.advance(1)
        cache.sync_tensor_pos()
    kk, _vv = cache.kv_view(1)
    # slots hold positions W+3-W .. W+2 in wrapped slot order
    assert kk.shape[2] == W
    vals = sorted(kk[0, 0, :, 0].tolist())
    assert vals == [float(p) for p in range(3, W + 3)]


def test_ring_decode_equivalent_vs_linear():
    """Same model, linear vs ring SWA buffer: identical logits when
    decoding well past the window (prefill + steps)."""
    torch.manual_seed(0)
    m_lin = _build(ring=False)
    torch.manual_seed(0)
    m_ring = _build(ring=True)
    for p, q in zip(m_lin.state_dict().values(),
                    m_ring.state_dict().values(), strict=False):
        assert torch.equal(p, q)  # same weights, buffers only differ

    prompt = torch.randint(0, 256, (1, 2 * W + 2))
    n_steps = W + 4

    def run(model, ring):
        cache = _cache(model)
        with torch.no_grad():
            out = model(prompt, preallocated_cache=cache, use_cache=True)
            logits = [out[0][:, -1:]]
            tok = out[0][:, -1:].argmax(-1)
            for _ in range(n_steps):
                out = model(tok, preallocated_cache=cache, use_cache=True)
                logits.append(out[0][:, -1:])
                tok = out[0][:, -1:].argmax(-1)
        return torch.cat(logits, 1)

    a = run(m_lin, False)
    b = run(m_ring, True)
    # SDPA reduces over different lengths (19 vs 9 keys) — the masked -inf
    # terms contribute exact zeros but kernel tiling differs, so bitwise
    # equality is not guaranteed. The real gates: identical argmax
    # trajectory and fp32-kernel-noise-level logit agreement.
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    assert (a - b).abs().max().item() < 2e-5


def test_ring_prefill_chunked_equivalent():
    """T>1 continuation forward over a wrapped ring == linear cache."""
    torch.manual_seed(0)
    m_lin = _build(ring=False)
    torch.manual_seed(0)
    m_ring = _build(ring=True)
    prompt = torch.randint(0, 256, (1, W + 2))
    cont = torch.randint(0, 256, (1, 5))
    out = {}
    for name, model in (("lin", m_lin), ("ring", m_ring)):
        cache = _cache(model)
        with torch.no_grad():
            model(prompt, preallocated_cache=cache, use_cache=True)
            out[name] = model(cont, preallocated_cache=cache,
                              use_cache=True)[0]
    assert torch.equal(out["lin"].argmax(-1), out["ring"].argmax(-1))
    assert (out["lin"] - out["ring"]).abs().max().item() < 2e-5
