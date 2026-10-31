"""Tests for the QuantizedLinearMixin protocol (critique F14).

Verifies that:
1. All known quantized linear classes have the mixin / ``is_quantized_linear``.
2. ``add_lora_adapters`` detects new (previously unknown) quantized linears
   automatically via the protocol — no string matching needed.
3. ``merge_lora_adapters`` dispatches to ``merge_lora()`` on quantized linears.
"""
import torch
import torch.nn as nn

from forge.quant.protocol import QuantizedLinearMixin, is_quantized_linear

# ── 1. All known quantized linear classes have the protocol ────────────────

def _make_linear(in_f=64, out_f=64, bias=True):
    return nn.Linear(in_f, out_f, bias=bias)


def test_quantized_linear_has_mixin():
    """QuantizedLinear (inference_quant) has the mixin."""
    from forge.quant.inference_quant import QuantizedLinear
    lin = QuantizedLinear(_make_linear(), bits=8)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)
    assert lin.is_quantized_linear is True


def test_fast_int8_linear_has_mixin():
    """FastINT8Linear (inference_quant) has the mixin."""
    from forge.quant.inference_quant import FastINT8Linear
    lin = FastINT8Linear(_make_linear())
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_nf4_linear_has_mixin():
    """NF4Linear (bitnet_lora) has the mixin."""
    from forge.training.bitnet_lora import NF4Linear
    lin = NF4Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_iri_fp4_linear_key_has_mixin():
    """IRIFP4Linear (iri_fp4_key) has the mixin."""
    from forge.keys.quantization.iri_fp4_key import IRIFP4Linear
    lin = IRIFP4Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_iri_fp4_linear_novel_quant_has_mixin():
    """IRIFP4Linear (novel_quant) has the mixin."""
    from forge.engine.quant.novel_quant import IRIFP4Linear
    lin = IRIFP4Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_forge_quant_linear_has_mixin():
    """ForgeQuantLinear (forge_quant) has the mixin."""
    from forge.engine.quant.forge_quant import ForgeQuantLinear
    lin = ForgeQuantLinear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_grinqh_linear_has_mixin():
    """GRINQHLinear (grinqh) has the mixin."""
    from forge.engine.quant.grinqh import GRINQHLinear
    lin = GRINQHLinear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_nvfp4_linear_has_mixin():
    """NVFP4Linear (nvfp4_quant) has the mixin."""
    from forge.engine.quant.nvfp4_quant import NVFP4Linear
    lin = NVFP4Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_w8a8_linear_has_mixin():
    """W8A8Linear (w8a8_quant) has the mixin."""
    from forge.engine.quant.w8a8_quant import W8A8Linear
    lin = W8A8Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_fp8_linear_w8a8_has_mixin():
    """FP8Linear (w8a8_quant) has the mixin."""
    from forge.engine.quant.w8a8_quant import FP8Linear
    lin = FP8Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_fp8_linear_fp8_infer_has_mixin():
    """FP8Linear (fp8_infer) has the mixin."""
    from forge.quant.fp8_infer import FP8Linear
    lin = FP8Linear(_make_linear())
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_asfp4_linear_has_mixin():
    """ASFP4Linear (novel_quant) has the mixin."""
    from forge.engine.quant.novel_quant import ASFP4Linear
    lin = ASFP4Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_residual_fp4_linear_has_mixin():
    """ResidualFP4Linear (novel_quant) has the mixin."""
    from forge.engine.quant.novel_quant import ResidualFP4Linear
    lin = ResidualFP4Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_sr_fp4_linear_has_mixin():
    """SRFP4Linear (novel_quant) has the mixin."""
    from forge.engine.quant.novel_quant import SRFP4Linear
    lin = SRFP4Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_acbq_linear_has_mixin():
    """ACBQLinear (acbq) has the mixin."""
    from forge.engine.quant.acbq import ACBQLinear
    lin = ACBQLinear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_mixllm_linear_has_mixin():
    """MixLLMLinear (mixllm) has the mixin."""
    from forge.engine.quant.mixllm import MixLLMLinear
    lin = MixLLMLinear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


def test_quamba2_linear_has_mixin():
    """Quamba2Linear (quamba2) has the mixin."""
    from forge.quant.quamba2 import Quamba2Linear
    lin = Quamba2Linear(64, 64, bias=False)
    assert isinstance(lin, QuantizedLinearMixin)
    assert is_quantized_linear(lin)


# ── 2. add_lora_adapters detects new quantized linears automatically ──────

class _DummyQuantLinear(QuantizedLinearMixin):
    """A dummy quantized linear that the bitnet_lora code has never seen.

    If detection relies on string matching this class name, it will NOT be
    detected.  With the protocol-based check, it IS detected automatically.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_adapter = None
        # Store a dummy quantized weight buffer so forward works.
        self.register_buffer(
            "weight_packed",
            torch.zeros(out_features, (in_features + 1) // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scales",
            torch.ones(out_features, max(1, in_features // 64), dtype=torch.float16),
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.group_size = 64

    def _dequantize_weight(self, dtype=torch.bfloat16, cache=False):
        # Return a zero weight — we only care about detection, not values.
        return torch.zeros(self.out_features, self.in_features, dtype=dtype)

    def forward(self, x):
        w = self._dequantize_weight(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        out = torch.nn.functional.linear(x, w, bias)
        if self.lora_adapter is not None:
            out = out + self.lora_adapter(x)
        return out

    @torch.no_grad()
    def merge_lora(self) -> bool:
        """Merge LoRA by zeroing the adapter (dummy implementation)."""
        if self.lora_adapter is None:
            return False
        self.lora_adapter = None
        return True


def test_is_quantized_linear_detects_dummy():
    """is_quantized_linear detects the dummy class via the mixin."""
    lin = _DummyQuantLinear(64, 64)
    assert is_quantized_linear(lin)
    assert isinstance(lin, QuantizedLinearMixin)


def test_add_lora_adapters_detects_new_quant_linear():
    """add_lora_adapters detects a previously-unknown quantized linear.

    This is the core regression test for critique F14: a new quantized
    linear class that was NOT in the old string-matching list should be
    automatically detected via the protocol.
    """
    from forge.training.bitnet_lora import add_lora_adapters

    model = nn.Sequential(_DummyQuantLinear(64, 64))
    n_adapters, lora_params = add_lora_adapters(model, rank=8, alpha=16)
    assert n_adapters == 1, "add_lora_adapters should detect the dummy quant linear"
    assert len(lora_params) == 2  # lora_A + lora_B
    # The adapter should be attached.
    dummy = model[0]
    assert hasattr(dummy, "lora_adapter") and dummy.lora_adapter is not None


def test_add_lora_adapters_skips_plain_linear_below_min_size():
    """add_lora_adapters respects min_size for quantized linears too."""
    from forge.training.bitnet_lora import add_lora_adapters

    model = nn.Sequential(_DummyQuantLinear(32, 32))
    n_adapters, _ = add_lora_adapters(model, rank=8, alpha=16, min_size=64)
    assert n_adapters == 0, "Layers below min_size should be skipped"


# ── 3. merge_lora works on quantized linears ─────────────────────────────

def test_merge_lora_adapters_dispatches_to_quant_linear():
    """merge_lora_adapters calls merge_lora() on quantized linears."""
    from forge.training.bitnet_lora import LoRAAdapter, merge_lora_adapters

    dummy = _DummyQuantLinear(64, 64)
    dummy.lora_adapter = LoRAAdapter(64, 64, rank=8, alpha=16)
    model = nn.Sequential(dummy)

    n_merged = merge_lora_adapters(model)
    assert n_merged == 1
    # merge_lora should have removed the adapter.
    assert dummy.lora_adapter is None


def test_merge_lora_adapters_on_nf4_linear():
    """merge_lora_adapters works on a real NF4Linear."""
    from forge.training.bitnet_lora import LoRAAdapter, NF4Linear, merge_lora_adapters

    nf4 = NF4Linear(64, 64, bias=False, group_size=64)
    # Load some weights so merge has something to work with.
    w = torch.randn(64, 64) * 0.1
    nf4.load_from_weight(w)
    nf4.lora_adapter = LoRAAdapter(64, 64, rank=8, alpha=16)
    model = nn.Sequential(nf4)

    n_merged = merge_lora_adapters(model)
    assert n_merged == 1
    assert nf4.lora_adapter is None


def test_merge_lora_default_returns_false():
    """The default merge_lora() in the mixin returns False (no-op)."""
    from forge.quant.inference_quant import QuantizedLinear

    lin = QuantizedLinear(_make_linear(), bits=8)
    # QuantizedLinear doesn't override merge_lora, so the default no-op runs.
    assert lin.merge_lora() is False


# ── 4. Backward compatibility: getattr fallback ────────────────────────────

def test_is_quantized_linear_attr_fallback():
    """is_quantized_linear returns True for objects with is_quantized_linear=True
    even if they don't inherit from the mixin (backward compat)."""

    class ExternalQuantLinear(nn.Module):
        is_quantized_linear = True
        in_features = 64
        out_features = 64

    ext = ExternalQuantLinear()
    assert is_quantized_linear(ext)
    assert not isinstance(ext, QuantizedLinearMixin)  # not a mixin subclass


def test_is_quantized_linear_false_for_plain_linear():
    """is_quantized_linear returns False for a plain nn.Linear."""
    lin = nn.Linear(64, 64)
    assert not is_quantized_linear(lin)


def test_is_quantized_linear_false_for_random_module():
    """is_quantized_linear returns False for a non-linear module."""
    mod = nn.ReLU()
    assert not is_quantized_linear(mod)
