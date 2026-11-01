"""Tests for research.training_free.bake — offline weight baking.

All tensor-level / CPU — no model or GPU required.
"""

import json
import os

import torch

from research.training_free.bake import (
    bake_task_vector,
    extract_distill_dataset,
    fuse_lora,
)


def _save_ckpt(path, tensors: dict):
    from forge.checkpoint_io import save_checkpoint

    save_checkpoint(dict(tensors), path)


class TestBakeTaskVector:
    def test_adds_delta_to_target(self, tmp_path):
        base = {"w": torch.tensor([1.0, 2.0, 3.0])}
        ft = {"w": torch.tensor([3.0, 4.0, 5.0])}
        target = {"w": torch.tensor([10.0, 20.0, 30.0])}

        bp, fp, tp = (tmp_path / "b.safetensors", tmp_path / "f.safetensors", tmp_path / "t.safetensors")
        out = str(tmp_path / "o.safetensors")
        _save_ckpt(str(bp), base)
        _save_ckpt(str(fp), ft)
        _save_ckpt(str(tp), target)

        bake_task_vector(str(tp), str(fp), str(bp), alpha=0.5, out_path=out)

        from forge.checkpoint_io import load_checkpoint

        merged = load_checkpoint(out, map_location="cpu")
        # target + 0.5 * (ft - base) = [10,20,30] + 0.5*[2,2,2]
        expected = torch.tensor([11.0, 21.0, 31.0])
        assert torch.allclose(merged["w"].float(), expected, atol=1e-5)

    def test_alpha_zero_is_identity(self, tmp_path):
        base = {"w": torch.tensor([1.0])}
        ft = {"w": torch.tensor([9.0])}
        target = {"w": torch.tensor([5.0])}
        paths = [str(tmp_path / f"{n}.safetensors") for n in "bft"]
        _save_ckpt(paths[0], base)
        _save_ckpt(paths[1], ft)
        _save_ckpt(paths[2], target)

        out = str(tmp_path / "o.safetensors")
        bake_task_vector(paths[2], paths[1], paths[0], alpha=0.0, out_path=out)

        from forge.checkpoint_io import load_checkpoint

        merged = load_checkpoint(out, map_location="cpu")
        assert torch.allclose(merged["w"].float(), target["w"].float())


class TestExtractDistill:
    def _packet(self, task, code, correct=True, score=0.9):
        return {
            "task": task,
            "prompt": task,
            "generated_code": code,
            "execution": {
                "returncode": 0 if correct else 1,
                "output_matches_expected": correct,
                "stdout": "ok" if correct else "",
            },
            "quality_score": score,
        }

    def test_filters_and_formats(self, tmp_path):
        pk = tmp_path / "packets.jsonl"
        with open(pk, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._packet("t1", "def a(): pass")) + "\n")
            f.write(json.dumps(self._packet("t2", "def b(): pass", correct=False, score=0.2)) + "\n")
            f.write(json.dumps(self._packet("t3", "def c(): pass", correct=True, score=0.5)) + "\n")

        out = tmp_path / "distill.jsonl"
        n = extract_distill_dataset([str(pk)], str(out), min_score=0.6, require_correct=True)
        assert n == 1
        with open(out, encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert row == {"prompt": "t1", "response": "def a(): pass"}

    def test_require_correct_false(self, tmp_path):
        pk = tmp_path / "p.jsonl"
        with open(pk, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._packet("t1", "code", correct=True, score=0.3)) + "\n")
        out = tmp_path / "d.jsonl"
        n = extract_distill_dataset([str(pk)], str(out), min_score=0.0, require_correct=True)
        assert n == 1  # correct=True wins even with low score


class TestFuseLora:
    def _make_adapter(self, adapter_dir, r=2, alpha=4):
        os.makedirs(adapter_dir, exist_ok=True)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump({"r": r, "lora_alpha": alpha}, f)

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (r, in=2)
        b = torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])  # (out=3, r)
        from safetensors.torch import save_file

        save_file(
            {
                "base_model.model.lin.lora_A.weight": a,
                "base_model.model.lin.lora_B.weight": b,
            },
            os.path.join(adapter_dir, "adapter_model.safetensors"),
        )
        return a, b

    def test_fuses_into_base(self, tmp_path):
        w = torch.ones(3, 2)
        base_path = str(tmp_path / "base.safetensors")
        _save_ckpt(base_path, {"lin.weight": w.clone()})

        adapter_dir = str(tmp_path / "lora")
        a, b = self._make_adapter(adapter_dir, r=2, alpha=4)  # scale = 2

        out = str(tmp_path / "fused.safetensors")
        fuse_lora(base_path, adapter_dir, out_path=out)

        from forge.checkpoint_io import load_checkpoint

        fused = load_checkpoint(out, map_location="cpu")
        expected = w + 2.0 * (b @ a)
        assert torch.allclose(fused["lin.weight"].float(), expected, atol=1e-5)

    def test_alpha_override(self, tmp_path):
        w = torch.zeros(3, 2)
        base_path = str(tmp_path / "base.safetensors")
        _save_ckpt(base_path, {"lin.weight": w.clone()})

        adapter_dir = str(tmp_path / "lora")
        a, b = self._make_adapter(adapter_dir, r=2, alpha=4)
        # Override alpha to 2 -> scale = 1
        out = str(tmp_path / "fused.safetensors")
        fuse_lora(base_path, adapter_dir, out_path=out, alpha_override=2.0)

        from forge.checkpoint_io import load_checkpoint

        fused = load_checkpoint(out, map_location="cpu")
        assert torch.allclose(fused["lin.weight"].float(), b @ a, atol=1e-5)
