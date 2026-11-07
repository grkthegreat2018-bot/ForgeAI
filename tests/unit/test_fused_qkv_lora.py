"""Fused-QKV LoRA: attach, forward delta, merge consistency, dead-path guard."""
import torch
import torch.nn as nn

from forge.keys.quantization.fused_gemm_key import FusedQKVLinear
from forge.training.bitnet_lora import add_lora_adapters, convert_to_bitnet_everywhere, merge_lora_adapters


class _Attn(nn.Module):
    def __init__(self, d=64, q=64, k=16, v=16, g=8):
        super().__init__()
        self.q_proj = nn.Linear(d, q, bias=False)
        self.k_proj = nn.Linear(d, k, bias=False)
        self.v_proj = nn.Linear(d, v, bias=False)
        self.gate_proj = nn.Linear(d, g, bias=False)
        self.out_proj = nn.Linear(q, d, bias=False)
        w = torch.cat([self.q_proj.weight, self.k_proj.weight,
                       self.v_proj.weight, self.gate_proj.weight], dim=0)
        self.qkv_proj = FusedQKVLinear(d, q, k, v, bias=False, g_dim=g)
        with torch.no_grad():
            self.qkv_proj.weight.copy_(w)
        self._fused_qkv = True


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()


def test_fused_qkv_adapter_attaches_and_applies():
    net = _Net()
    n, _params = add_lora_adapters(net, rank=4, alpha=8,
                                target_modules=["qkv_proj", "out_proj"])
    assert n == 2
    lora = net.attn.qkv_proj.lora_adapter
    with torch.no_grad():
        lora.lora_B.fill_(0.1)
    x = torch.randn(2, 5, 64)
    import torch.nn.functional as F
    full = F.linear(x, net.attn.qkv_proj.weight) + lora(
        x.to(lora.lora_A.dtype)).float()
    exp = full.split([64, 16, 16, 8], dim=-1)
    got = net.attn.qkv_proj(x)
    for e, g in zip(exp, got, strict=False):
        assert torch.allclose(e, g, atol=1e-5)


def test_fused_qkv_merge_mirrors_into_separate():
    net = _Net()
    add_lora_adapters(net, rank=4, alpha=8, target_modules=["qkv_proj"])
    lora = net.attn.qkv_proj.lora_adapter
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.02)
    delta = (lora.scale * (lora.lora_B @ lora.lora_A)).detach()
    q_before = net.attn.q_proj.weight.clone()
    n = merge_lora_adapters(net)
    assert n == 1
    assert torch.allclose(net.attn.qkv_proj.weight[:64],
                          net.attn.q_proj.weight, atol=1e-6)
    assert (net.attn.q_proj.weight - q_before
            - delta[:64].to(q_before.dtype)).abs().max() < 1e-5


def test_bitnet_conversion_disables_fused_path():
    net = _Net()
    n_conv, _ = convert_to_bitnet_everywhere(net)
    assert n_conv > 0
    assert net.attn._fused_qkv is False


if __name__ == "__main__":
    test_fused_qkv_adapter_attaches_and_applies()
    test_fused_qkv_merge_mirrors_into_separate()
    test_bitnet_conversion_disables_fused_path()
    print("all pass")
