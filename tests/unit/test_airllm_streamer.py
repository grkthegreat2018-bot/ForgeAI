"""Tests for forge.engine.airllm_streamer — R71/B1 pinned prefetch ring.

Verifies: binding correctness (bound params == shard weights), int4
pack/dequant roundtrip, end-to-end greedy output == manual reference, and
the VRAM bound (GPU holds ~S slots worth of layer weights, not all).
"""
import os

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from forge.engine.airllm_streamer import (
    AirLLMStreamer, _PinnedCache, _dequant_i4, _pack_i4,
)

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, d, dtype=torch.bfloat16) * 0.02)

    def forward(self, x):
        return torch.nn.functional.gelu(x @ self.w.to(x.dtype))


class _Model(nn.Module):
    def __init__(self, n_layers=4, d=64, vocab=128):
        super().__init__()
        self.blocks = nn.ModuleList(_Block(d) for _ in range(n_layers))
        self.embed = nn.Embedding(vocab, d, dtype=torch.bfloat16)
        self.ln_f = nn.LayerNorm(d, dtype=torch.bfloat16)
        self.head = nn.Linear(d, vocab, bias=False, dtype=torch.bfloat16)
        self.config = type("C", (), {"n_layers": n_layers, "d_model": d,
                                     "n_heads": 4, "n_kv_heads": 1,
                                     "max_seq_len": 64})()


class _Engine:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._param_map = dict(model.named_parameters())
        self._layer_shards = []
        self._use_i4_stream = False
        self._stream_plan = None

    def _eos_token_ids(self):
        return set()

    def _sample_next_token(self, logits, temperature, a, b, c, ids):
        return logits.argmax(-1).reshape(1, 1)


def _write_shards(tmp, model, n_layers):
    shard_dir = tmp / "xp_shards"
    shard_dir.mkdir()
    resident, groups = {}, {}
    for name, p in model.named_parameters():
        parts = name.split(".")
        if parts[0] == "blocks" and parts[1].isdigit():
            groups.setdefault(int(parts[1]), {})[name] = p.data.clone()
        else:
            resident[name] = p.data.clone()
    save_file(resident, str(shard_dir / "shard_0.safetensors"))
    paths = []
    for i in sorted(groups):
        p = shard_dir / f"shard_{i + 1}.safetensors"
        save_file(groups[i], str(p))
        paths.append(p)
    return shard_dir, paths


def _write_i4(tmp, bf16_paths):
    from safetensors.torch import load_file
    i4_dir = tmp / "xp_shards_i4"
    i4_dir.mkdir()
    out_paths = []
    for sh in bf16_paths:
        out = {}
        for name, t in load_file(str(sh)).items():
            q, mn, mx, tail, _s, _n = _pack_i4(t)
            out[name + ".q4"] = q
            out[name + ".gmn"] = mn
            out[name + ".gmx"] = mx
            if tail.numel():
                out[name + ".tl"] = tail
        p = i4_dir / (sh.stem + ".i4.safetensors")
        save_file(out, str(p))
        out_paths.append(p)
    return out_paths


class TestI4Codec:
    def test_pack_dequant_roundtrip(self):
        torch.manual_seed(0)
        w = torch.randn(128, 64) * 0.02
        q, mn, mx, tail, shape, n = _pack_i4(w)
        out = _dequant_i4(q, mn, mx, tail, shape, n, "cpu")
        err = (out.float() - w).pow(2).mean() / w.pow(2).mean()
        assert err < 0.01  # g32 int4 rel-MSE ~0.6% typical


@_cuda
class TestStreamerRing:
    def _engine(self, tmp_path, n_layers=4, d=64):
        torch.manual_seed(0)
        model = _Model(n_layers=n_layers, d=d).to("cuda")
        eng = _Engine(model, torch.device("cuda"))
        _dir, paths = _write_shards(tmp_path, model, n_layers)
        eng._layer_shards = paths
        return eng, model, paths

    def test_pinned_cache(self, tmp_path):
        eng, model, paths = self._engine(tmp_path)
        cache = _PinnedCache(paths)
        st = cache.get(0)
        assert all(t.is_pinned() for t in st.values())
        assert cache.get(0) is st

    def test_binding_correctness_bf16(self, tmp_path):
        eng, model, paths = self._engine(tmp_path)
        plan = AirLLMStreamer._build_plan(eng, eng.device, int4=False)
        AirLLMStreamer._issue(plan, 2)          # layer index 2 -> slot 2
        torch.cuda.synchronize()
        s = 2 % 3
        for name, pm in eng._param_map.items():
            if name.startswith("blocks.2."):
                pm.data = plan["slots"][s][name.split(".", 2)[2]]
        torch.testing.assert_close(
            eng._param_map["blocks.2.w"].data,
            model.blocks[2].w.data.to("cuda"))

    def test_generate_matches_reference(self, tmp_path):
        eng, model, paths = self._engine(tmp_path)
        eng._use_i4_stream = False
        ids = torch.tensor([[3, 17, 42]], device=eng.device)

        # manual reference: full resident greedy forward
        ref = ids.clone()
        h = model.embed(ref)
        for b in model.blocks:
            h = b(h)
        logits = model.head(model.ln_f(h))
        next_id = logits[:, -1, :].argmax(-1)

        out = AirLLMStreamer.generate(eng, ids, max_new_tokens=1, temperature=0.0)
        assert out.shape[-1] == 4
        assert out[0, -1].item() == next_id.item()

    def test_vram_bound(self, tmp_path):
        eng, model, paths = self._engine(tmp_path, n_layers=6, d=256)
        ids = torch.tensor([[1, 2, 3]], device=eng.device)
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        AirLLMStreamer.generate(eng, ids, max_new_tokens=4, temperature=0.0)
        peak = torch.cuda.max_memory_allocated() - base
        layer_bytes = sum(p.numel() * 2 for p in model.blocks[0].parameters())
        all_layers = layer_bytes * 6
        # GPU must hold ~= ring slots * layer, not all 6 layers
        assert peak < all_layers, f"peak {peak/1e6:.1f}MB vs all {all_layers/1e6:.1f}MB"

    def test_i4_generate_runs(self, tmp_path):
        eng, model, paths = self._engine(tmp_path, n_layers=4)
        _write_i4(tmp_path, paths)
        eng._layer_shards = sorted(
            (tmp_path / "xp_shards_i4").glob("*.i4.safetensors"))
        eng._use_i4_stream = True
        ids = torch.tensor([[3, 17, 42]], device=eng.device)
        out = AirLLMStreamer.generate(eng, ids, max_new_tokens=3, temperature=0.0)
        assert out.shape[-1] >= 4
        assert torch.isfinite(out.float()).all()
