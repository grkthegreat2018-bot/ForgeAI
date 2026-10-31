"""Regression test: fused attention wrappers must apply the LFM3
per-head sigmoid output gate (``_finish_attn`` / ``gate_proj``).

Bug (2026-10-30): FusedQKNormRopeCacheWrapper.fused_forward and
SequenceAwareSplitWrapper.split_forward (plus atflash/compact/cosa/
flex/mosa/triroute wrappers) ended their fused path with a bare
``self.out_proj(out)``, skipping ``_finish_attn``. LFM3 uses
``use_attn_out_gate`` (out = out_proj(attn * sigmoid(gate_proj(x)))),
so the *functional* decode path silently produced semantically
different output than the unfused reference — and was wrongly blamed
on the static CUDA-graph path it was compared against.

Both wrappers run on CPU (fused_qk_norm_rope falls back to
_py_qk_norm_rope; split_merge_attention is pure torch), so the
fused-vs-original parity assertion is a real semantic check —
pre-fix it fails by the gate factor.
"""
import torch

from forge.model.layers import GroupedQueryAttention
from forge.model.kv_cache import PreAllocatedKVCache
from forge.engine.attention.fused_qk_norm_rope_cache import (
    FusedQKNormRopeCacheWrapper,
)
from forge.engine.attention.seq_aware_split import (
    SequenceAwareSplitWrapper,
)


def _gated_gqa(d_model=64, n_heads=4, n_kv_heads=2):
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        max_seq_len=128, use_out_gate=True, use_rope=True,
        use_qk_norm=False)
    attn.eval()
    # Make the gate decisive: random init gives sigmoid(gate(x)) ~0.5
    # but a strong input makes pre/post-gate outputs unmistakably differ.
    with torch.no_grad():
        attn.gate_proj.weight.mul_(3.0)
    return attn


class TestFusedWrapperOutGate:
    def test_fused_qk_norm_rope_applies_gate(self):
        """fused_forward output must equal the unfused forward (which
        applies _finish_attn) — not the bare out_proj result."""
        attn = _gated_gqa()
        x = torch.randn(1, 5, 64)
        with torch.no_grad():
            ref, _ = attn(x, use_cache=False)

        wrapper = FusedQKNormRopeCacheWrapper(kv_quant_bits=None)
        wrapper._patch(attn, "blocks.0.attn")
        with torch.no_grad():
            fused, _ = attn(x, use_cache=False)

        assert torch.allclose(ref, fused, atol=1e-4), (
            "fused_forward diverged from the reference forward — the "
            "per-head output gate is being skipped (pre-fix behavior)")
        # Positive control: gate must be live — bypassing it must differ.
        with torch.no_grad():
            raw = attn.out_proj(
                torch.randn(1, 5, attn.n_heads * attn.head_dim))
        assert not torch.allclose(ref, raw, atol=1e-4)

    def test_seq_aware_split_applies_gate(self):
        """split_forward fused branch must route through _finish_attn."""
        attn = _gated_gqa()
        x1 = torch.randn(1, 1, 64)

        # Reference: T=1 decode over a prefilled preallocated cache,
        # unfused original forward.
        cache_a = PreAllocatedKVCache(
            n_layers=1, batch=1, n_kv_heads=2, max_seq_len=64,
            head_dim=16, dtype=torch.float32, device=torch.device("cpu"))
        with torch.no_grad():
            pre = torch.randn(1, 6, 64)
            attn(pre, preallocated_cache=cache_a, layer_idx=0)
            ref, _ = attn(x1, preallocated_cache=cache_a, layer_idx=0)

        # Fused: same cache state, patched forward takes the split path.
        attn2 = _gated_gqa()
        attn2.load_state_dict(attn.state_dict())
        cache_b = PreAllocatedKVCache(
            n_layers=1, batch=1, n_kv_heads=2, max_seq_len=64,
            head_dim=16, dtype=torch.float32, device=torch.device("cpu"))
        wrapper = SequenceAwareSplitWrapper(n_sms=4, min_seq_len=4)
        wrapper._patch(attn2, "blocks.0.attn")
        with torch.no_grad():
            attn2(pre, preallocated_cache=cache_b, layer_idx=0)
            fused, _ = attn2(x1, preallocated_cache=cache_b, layer_idx=0)

        assert cache_b.position == cache_a.position
        assert torch.allclose(ref, fused, atol=1e-4), (
            "split_forward diverged from the reference forward — the "
            "per-head output gate is being skipped (pre-fix behavior)")

    def test_no_gate_modules_still_project(self):
        """Modules without a gate (gate_proj=None) keep the bare
        out_proj path — the _finish_attn fallback must not break them."""
        torch.manual_seed(0)
        attn = GroupedQueryAttention(
            d_model=64, n_heads=4, n_kv_heads=2, max_seq_len=128,
            use_out_gate=False, use_rope=True, use_qk_norm=False)
        attn.eval()
        assert attn.gate_proj is None
        x = torch.randn(1, 5, 64)
        with torch.no_grad():
            ref, _ = attn(x, use_cache=False)
        wrapper = FusedQKNormRopeCacheWrapper(kv_quant_bits=None)
        wrapper._patch(attn, "blocks.0.attn")
        with torch.no_grad():
            fused, _ = attn(x, use_cache=False)
        assert torch.allclose(ref, fused, atol=1e-4)
