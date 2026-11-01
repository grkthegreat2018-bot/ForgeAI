"""conv K widen (d2): zero-pad depthwise conv kernels to a larger K.

The causal conv left-pads the input by K-1; widening keeps the original
taps on the LAST weight positions and zeroes the new leading ones, so
the extended lookback contributes nothing — function-exact. Verifies:
  - _warm_start_conv pads only attn.conv.weight 3-D keys
  - prefill (T>1) output is bit-exact K7 vs padded-K11
  - incremental decode (T=1, conv_state) is bit-exact
"""
import torch
import torch.nn as nn
from types import SimpleNamespace

from forge.model.layers import DoubleGatedConvLayer
from forge.model.loader import ModelLoader


def _pair(d=32, k=7, k_new=11):
    torch.manual_seed(0)
    l7 = DoubleGatedConvLayer(d, kernel_size=k).eval()
    # state dict keys as the loader sees them (layer lives under .attn)
    state = {f"attn.{k}": v.clone() for k, v in l7.state_dict().items()}
    cfg = SimpleNamespace(conv_kernel_size=k_new)
    padded = ModelLoader._warm_start_conv(dict(state), cfg)
    l11 = DoubleGatedConvLayer(d, kernel_size=k_new).eval()
    l11.load_state_dict({k[len("attn."):]: v
                         for k, v in padded.items()})
    return l7, l11, padded


def test_warm_start_pads_only_conv_weight():
    l7, l11, padded = _pair()
    w = padded["attn.conv.weight"]
    assert w.shape == (32, 1, 11)
    # original taps sit on the LAST 7 positions; leading 4 are zero
    assert w[..., :4].abs().max().item() == 0.0
    assert torch.equal(w[..., 4:], l7.conv.weight)
    # non-conv keys pass through untouched
    assert torch.equal(padded["attn.in_proj.weight"],
                       l7.in_proj.weight)


def test_warm_start_idempotent_and_noop():
    l7, _, padded = _pair()
    cfg = SimpleNamespace(conv_kernel_size=11)
    again = ModelLoader._warm_start_conv(dict(padded), cfg)
    assert torch.equal(again["attn.conv.weight"],
                       padded["attn.conv.weight"])
    # smaller/equal target is a no-op
    cfg7 = SimpleNamespace(conv_kernel_size=7)
    same = ModelLoader._warm_start_conv(
        {"attn.conv.weight": torch.zeros(8, 1, 7)}, cfg7)
    assert same["attn.conv.weight"].shape[-1] == 7


def test_prefill_bit_exact():
    l7, l11, _ = _pair()
    torch.manual_seed(1)
    x = torch.randn(2, 24, 32)
    with torch.no_grad():
        y7 = l7(x)[0]
        y11 = l11(x)[0]
    assert (y7 - y11).abs().max().item() == 0.0


def test_incremental_decode_bit_exact():
    l7, l11, _ = _pair()
    torch.manual_seed(2)
    xs = torch.randn(1, 16, 32)
    outs7, outs11 = [], []
    with torch.no_grad():
        for t in range(xs.shape[1]):
            outs7.append(l7(xs[:, t:t + 1])[0])
            outs11.append(l11(xs[:, t:t + 1])[0])
    d = (torch.cat(outs7, 1) - torch.cat(outs11, 1)).abs().max().item()
    assert d == 0.0
