"""Tests for forge.checkpoint_io — save/load with safetensors and .pt formats."""

import json
import os
import struct

import pytest
import torch

from forge.checkpoint_io import (
    _is_safetensors_path,
    _jsonable,
    _read_safetensors_entries,
    load_checkpoint,
    load_safetensors_pipelined,
    save_checkpoint,
)


class TestIsSafetensorsPath:
    def test_safetensors_extension(self):
        assert _is_safetensors_path("model.safetensors") is True

    def test_pt_extension(self):
        assert _is_safetensors_path("model.pt") is False

    def test_full_path(self):
        assert _is_safetensors_path("/tmp/checkpoints/model.safetensors") is True

    def test_no_extension(self):
        assert _is_safetensors_path("model") is False


class TestJsonable:
    def test_int(self):
        assert _jsonable(42) == 42

    def test_float(self):
        assert _jsonable(3.14) == 3.14

    def test_string(self):
        assert _jsonable("hello") == "hello"

    def test_list(self):
        assert _jsonable([1, 2, 3]) == [1, 2, 3]

    def test_nested_dict(self):
        result = _jsonable({"a": 1, "b": {"c": 2}})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_tensor_falls_back_to_repr(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _jsonable(t)
        assert isinstance(result, str)  # repr fallback


class TestSaveLoadSafetensors:
    """Round-trip save/load with .safetensors format."""

    def test_save_creates_safetensors_file(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        result = save_checkpoint(small_state_dict, path)
        assert result == path
        assert os.path.exists(path)

    def test_save_creates_meta_json_sidecar(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        meta_path = path + ".meta.json"
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["step"] == 100
        assert meta["config"]["lr"] == 1e-4

    def test_load_returns_tensors(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert torch.equal(loaded["weight_a"], small_state_dict["weight_a"])
        assert torch.equal(loaded["weight_b"], small_state_dict["weight_b"])

    def test_load_returns_metadata(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert loaded["step"] == 100
        assert loaded["config"]["lr"] == 1e-4

    def test_roundtrip_preserves_shapes(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert loaded["weight_a"].shape == (4, 8)
        assert loaded["weight_b"].shape == (16,)

    def test_roundtrip_bf16(self, tmp_checkpoint_dir, bf16_state_dict):
        path = str(tmp_checkpoint_dir / "bf16.safetensors")
        save_checkpoint(bf16_state_dict, path)
        loaded = load_checkpoint(path)
        assert loaded["weight_a"].dtype == torch.bfloat16
        assert torch.equal(loaded["weight_a"], bf16_state_dict["weight_a"])

    def test_roundtrip_fp32(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "fp32.safetensors")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert loaded["weight_a"].dtype == torch.float32

    def test_save_creates_parent_dir(self, tmp_path, small_state_dict):
        path = str(tmp_path / "nested" / "deep" / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        assert os.path.exists(path)

    def test_no_tmp_file_left_after_save(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "clean.safetensors")
        save_checkpoint(small_state_dict, path)
        assert not os.path.exists(path + ".tmp")
        assert not os.path.exists(path + ".meta.json.tmp")


class TestSaveLoadPt:
    """Round-trip save/load with legacy .pt format."""

    def test_save_pt_file(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.pt")
        result = save_checkpoint(small_state_dict, path)
        assert result == path
        assert os.path.exists(path)

    def test_load_pt_file(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.pt")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert torch.equal(loaded["weight_a"], small_state_dict["weight_a"])
        assert loaded["step"] == 100

    def test_pt_no_meta_sidecar(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.pt")
        save_checkpoint(small_state_dict, path)
        assert not os.path.exists(path + ".meta.json")


class TestSaveLoadEdgeCases:
    """Edge cases and error conditions."""

    def test_empty_state_dict(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "empty.safetensors")
        save_checkpoint({}, path)
        loaded = load_checkpoint(path)
        assert len(loaded) == 0

    def test_metadata_only(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "meta_only.safetensors")
        state = {"step": 10, "name": "test"}
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert loaded["step"] == 10
        assert loaded["name"] == "test"

    def test_single_tensor(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "single.safetensors")
        state = {"weight": torch.randn(3, 3)}
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert torch.equal(loaded["weight"], state["weight"])

    def test_overwrite_existing(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "overwrite.safetensors")
        save_checkpoint(small_state_dict, path)
        new_state = {"weight_a": torch.randn(4, 8), "step": 200}
        save_checkpoint(new_state, path)
        loaded = load_checkpoint(path)
        assert loaded["step"] == 200
        assert torch.equal(loaded["weight_a"], new_state["weight_a"])


class TestReadSafetensorsEntries:
    """Header parsing for the pipelined loader (CPU-safe)."""

    def test_entries_parsed_sorted(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "hdr.safetensors")
        state = {
            "b": torch.randn(4, 4, dtype=torch.bfloat16),
            "a": torch.randn(8, dtype=torch.float32),
            "i": torch.arange(5, dtype=torch.int64),
        }
        save_checkpoint(state, path)
        entries, data_start = _read_safetensors_entries(path)
        names = [e[0] for e in entries]
        assert set(names) == {"a", "b", "i"}
        # sorted by data offset
        offs = [e[3] for e in entries]
        assert offs == sorted(offs)
        assert data_start > 8
        # file size = data_start + last end
        assert os.path.getsize(path) == data_start + entries[-1][4]

    def test_dtypes_mapped(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "dt.safetensors")
        save_checkpoint({"w": torch.randn(2, 2, dtype=torch.bfloat16)}, path)
        entries, _ = _read_safetensors_entries(path)
        assert entries[0][1] == torch.bfloat16

    def test_unknown_dtype_raises(self, tmp_path):
        # Hand-craft a safetensors file with an unsupported dtype name.
        path = str(tmp_path / "bad.safetensors")
        hdr = json.dumps({"x": {"dtype": "NOPE", "shape": [1], "data_offsets": [0, 4]}})
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hdr)))
            f.write(hdr.encode())
            f.write(b"\x00" * 4)
        with pytest.raises(ValueError, match="unsupported"):
            _read_safetensors_entries(path)


class TestPipelinedLoad:
    """Pipelined safetensors -> CUDA loader."""

    def test_cpu_device_rejected(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "cpu.safetensors")
        save_checkpoint({"w": torch.randn(2, 2)}, path)
        with pytest.raises(ValueError, match="CUDA"):
            load_safetensors_pipelined(path, "cpu")

    def test_meta_device_rejected(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "meta.safetensors")
        save_checkpoint({"w": torch.randn(2, 2)}, path)
        with pytest.raises(ValueError, match="CUDA"):
            load_safetensors_pipelined(path, "meta")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pipelined loader requires CUDA")
    def test_roundtrip_bit_exact(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "rt.safetensors")
        state = {
            "w_bf16": torch.randn(64, 64, dtype=torch.bfloat16),
            "w_f32": torch.randn(33, 17, dtype=torch.float32),
            "w_i64": torch.arange(1000, dtype=torch.int64).view(10, -1),
            "w_i8": torch.randint(-128, 127, (256,), dtype=torch.int8),
            "w_u8": torch.randint(0, 255, (17,), dtype=torch.uint8),
        }
        save_checkpoint(state, path)
        loaded = load_safetensors_pipelined(path, "cuda")
        for k, ref in state.items():
            assert loaded[k].dtype == ref.dtype
            assert loaded[k].shape == ref.shape
            assert loaded[k].device.type == "cuda"
            assert torch.equal(loaded[k].cpu(), ref), k

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pipelined loader requires CUDA")
    def test_multi_chunk_spanning(self, tmp_checkpoint_dir):
        """A tensor larger than chunk_mb is filled across chunk boundaries."""
        path = str(tmp_checkpoint_dir / "big.safetensors")
        big = torch.randn(1024, 1024, dtype=torch.bfloat16)  # 2 MiB
        save_checkpoint({"big": big, "tail": torch.randn(7)}, path)
        # 1 MiB chunks -> big spans 2 chunks, tail follows in the same range
        loaded = load_safetensors_pipelined(path, "cuda", num_threads=1, chunk_mb=1)
        assert torch.equal(loaded["big"].cpu(), big)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pipelined loader requires CUDA")
    def test_multi_thread_ranges(self, tmp_checkpoint_dir):
        """Many tensors -> split across threads -> all bytes land."""
        path = str(tmp_checkpoint_dir / "mt.safetensors")
        state = {f"w{i:03d}": torch.randn(256, 256, dtype=torch.bfloat16) for i in range(16)}
        save_checkpoint(state, path)
        loaded = load_safetensors_pipelined(path, "cuda", num_threads=4, chunk_mb=1)
        for k, ref in state.items():
            assert torch.equal(loaded[k].cpu(), ref), k

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="pipelined loader requires CUDA")
    def test_truncated_file_raises(self, tmp_checkpoint_dir, tmp_path):
        src = str(tmp_checkpoint_dir / "full.safetensors")
        save_checkpoint({"w": torch.randn(64, 64)}, src)
        dst = str(tmp_path / "trunc.safetensors")
        with open(src, "rb") as f:
            data = f.read()
        with open(dst, "wb") as f:
            f.write(data[:-32])  # cut the tail of the data blob
        # Either the metadata validator rejects the lying data_offsets
        # (ValueError) or the reader hits the short blob (OSError).
        with pytest.raises((OSError, ValueError)):
            load_safetensors_pipelined(dst, "cuda")


class TestLoraOnlyTrainingCheckpoint:
    """Periodic LoRA-only snapshots: no frozen base weights, self-describing
    sidecar for adapter re-injection on resume."""

    def _model(self):
        import torch.nn as nn
        model = nn.Linear(8, 8)
        model.lora_adapter = nn.Module()
        model.lora_adapter.lora_A = nn.Parameter(torch.randn(2, 8))
        model.lora_adapter.lora_B = nn.Parameter(torch.zeros(8, 2))
        return model

    def test_lora_only_snapshot(self, tmp_path):
        from safetensors import safe_open
        from forge.checkpoint_io import (
            load_training_state, save_training_checkpoint)

        path = str(tmp_path / "step50.safetensors")
        save_training_checkpoint(self._model(), path, step=50, lora_only=True,
                                 meta={"base_checkpoint": "base/dir"})
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
        assert keys, "no adapter tensors written"
        assert all("lora_" in k or "qscale" in k for k in keys)

        meta = json.loads((tmp_path / "step50.safetensors.meta.json")
                          .read_text(encoding="utf-8"))
        assert meta["lora_only"] is True
        assert meta["step"] == 50
        assert meta["base_checkpoint"] == "base/dir"
        assert load_training_state(path)["step"] == 50

    def test_default_snapshot_keeps_base_weights(self, tmp_path):
        from safetensors import safe_open
        from forge.checkpoint_io import save_training_checkpoint

        path = str(tmp_path / "full.safetensors")
        save_training_checkpoint(self._model(), path, step=1)
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
        assert "weight" in keys and "bias" in keys
        meta = json.loads((tmp_path / "full.safetensors.meta.json")
                          .read_text(encoding="utf-8"))
        assert "lora_only" not in meta
