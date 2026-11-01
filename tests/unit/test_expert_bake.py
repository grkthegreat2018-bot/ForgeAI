"""Tests for research.training_free.expert_bake — AirMoE expert consolidation."""

import torch

from forge.checkpoint_io import load_checkpoint, save_checkpoint
from research.training_free.expert_bake import (
    _layer_from_filename,
    bake_expert,
    decompress_expert,
)


def _save(path, tensors):
    save_checkpoint(dict(tensors), str(path))


def _make_target(tmp_path):
    t = {
        "blocks.0.ffn.w_gate.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "blocks.0.ffn.w_up.weight": torch.arange(6, 12, dtype=torch.float32).reshape(3, 2),
        "blocks.0.ffn.w_down.weight": torch.arange(12, 18, dtype=torch.float32).reshape(3, 2),
        "blocks.1.ffn.w_gate.weight": torch.ones(3, 2, dtype=torch.float32),
    }
    path = tmp_path / "target.safetensors"
    _save(path, t)
    return path, t


class TestDecompress:
    def test_raw_format(self):
        state = {"w1.weight": torch.randn(3, 2), "w2.weight": torch.randn(3, 2)}
        expert = decompress_expert(state)
        assert set(expert.keys()) == {"w1", "w2"}
        assert torch.equal(expert["w1"], state["w1.weight"])

    def test_svd_format(self):
        W = torch.randn(4, 3)
        U, S, Vh = torch.linalg.svd(W)
        state = {"w1_U": U, "w1_S": S, "w1_Vh": Vh}
        expert = decompress_expert(state)
        assert torch.allclose(expert["w1"], W, atol=1e-4)

    def test_int4_group_format(self):
        W = torch.randn(8, 4)
        U, S, Vh = torch.linalg.svd(W)
        gs = 4
        uf = U.flatten()
        n_groups = (uf.numel() + gs - 1) // gs
        pad = n_groups * gs - uf.numel()
        uf_pad = torch.cat([uf, torch.zeros(pad)])
        ug = uf_pad.reshape(n_groups, gs)
        u_scale = ug.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        u_q = torch.clamp(torch.round(ug / u_scale), -127, 127).to(torch.int8)
        state = {
            "w1_U_q": u_q,
            "w1_U_scale": u_scale.squeeze(-1),
            "w1_S": S,
            "w1_Vh_q": Vh,
            "w1_Vh_scale": torch.ones(Vh.shape[0]),
            "w1_U_shape": torch.tensor([U.shape[0], U.shape[1]]),
            "w1_Vh_shape": torch.tensor([Vh.shape[0], Vh.shape[1]]),
        }
        expert = decompress_expert(state)
        assert torch.allclose(expert["w1"], W, atol=1e-2)


class TestBakeExpert:
    def test_folds_delta_into_dense_ffn(self, tmp_path):
        target_path, t = _make_target(tmp_path)
        # Expert for layer 0 = base FFN + known delta.
        gate = t["blocks.0.ffn.w_gate.weight"] + torch.full((3, 2), 2.0)
        up = t["blocks.0.ffn.w_up.weight"] + torch.full((3, 2), 1.0)
        down = t["blocks.0.ffn.w_down.weight"] + torch.full((3, 2), 0.5)
        expert_path = tmp_path / "expert_l0_math.safetensors"
        _save(expert_path, {"w1.weight": gate, "w2.weight": up, "w3.weight": down})

        out = str(tmp_path / "baked.safetensors")
        bake_expert(str(target_path), [str(expert_path)], alpha=0.5, out_path=out)

        merged = load_checkpoint(out, map_location="cpu")
        # target + 0.5 * delta
        assert torch.allclose(
            merged["blocks.0.ffn.w_gate.weight"].float(), t["blocks.0.ffn.w_gate.weight"] + 1.0, atol=1e-5
        )
        assert torch.allclose(
            merged["blocks.0.ffn.w_up.weight"].float(), t["blocks.0.ffn.w_up.weight"] + 0.5, atol=1e-5
        )
        # Unrelated layer untouched.
        assert torch.equal(merged["blocks.1.ffn.w_gate.weight"], t["blocks.1.ffn.w_gate.weight"])

    def test_multi_expert_averages(self, tmp_path):
        target_path, t = _make_target(tmp_path)
        e1 = tmp_path / "expert_l0_a.safetensors"
        e2 = tmp_path / "expert_l0_b.safetensors"
        base = t["blocks.0.ffn.w_gate.weight"]
        _save(e1, {"w1.weight": base + 4.0})
        _save(e2, {"w1.weight": base + 8.0})
        out = str(tmp_path / "baked.safetensors")
        bake_expert(str(target_path), [str(e1), str(e2)], alpha=1.0, out_path=out)
        merged = load_checkpoint(out, map_location="cpu")
        # mean delta = 6.0
        assert torch.allclose(merged["blocks.0.ffn.w_gate.weight"].float(), base + 6.0, atol=1e-5)

    def test_svd_expert_folds(self, tmp_path):
        target_path, t = _make_target(tmp_path)
        gate = t["blocks.0.ffn.w_gate.weight"] + torch.full((3, 2), 3.0)
        U, S, Vh = torch.linalg.svd(gate)
        expert_path = tmp_path / "expert_l0_math.safetensors"
        _save(expert_path, {"w1_U": U, "w1_S": S, "w1_Vh": Vh})

        out = str(tmp_path / "baked.safetensors")
        bake_expert(str(target_path), [str(expert_path)], alpha=1.0, out_path=out)
        merged = load_checkpoint(out, map_location="cpu")
        expected = t["blocks.0.ffn.w_gate.weight"] + 3.0
        assert torch.allclose(merged["blocks.0.ffn.w_gate.weight"].float(), expected, atol=1e-3)

    def test_layer_parsing(self):
        assert _layer_from_filename("expert_l7_math.safetensors") == 7
        assert _layer_from_filename("experts/expert_l12_code.safetensors") == 12
        assert _layer_from_filename("nope.safetensors") is None
