"""Unit tests for forge_gui.api.lora_store (adapter scan, header parsing,
base-hint guessing, _human_bytes).

Uses temp directories + fake safetensors headers — no real adapters or
torch needed.
"""

import json
import struct
import time

from forge_gui.api.lora_store import (
    _base_hint,
    _human_bytes,
    read_adapter_header,
    scan_lora_adapters,
)

# ── _human_bytes ───────────────────────────────────────────────────────


def test_human_bytes_units():
    assert _human_bytes(0) == "0.0 B"
    assert _human_bytes(512) == "512.0 B"
    assert _human_bytes(1024) == "1.0 KB"
    assert _human_bytes(1024 * 1024) == "1.0 MB"
    assert _human_bytes(1024**3) == "1.0 GB"


def test_human_bytes_small():
    assert _human_bytes(100) == "100.0 B"


# ── _base_hint ─────────────────────────────────────────────────────────


def test_base_hint_underscore_lora():
    assert _base_hint("forgelm_v2_tooluse.lora.safetensors") == "forgelm_v2_tooluse"


def test_base_hint_dot_lora():
    assert _base_hint("epoch3.lora.safetensors") == "epoch3"


def test_base_hint_no_marker():
    assert _base_hint("adapter.safetensors") == "adapter.safetensors"


# ── read_adapter_header ────────────────────────────────────────────────


def _write_fake_safetensors(path, header_dict: dict) -> None:
    """Write a minimal safetensors file (header + zero data)."""
    header_bytes = json.dumps(header_dict).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)


def test_read_header_valid_lora(tmp_path):
    p = tmp_path / "test_lora.safetensors"
    header = {
        "model.layers.0.attn.lora_A.weight": {"shape": [32, 1024], "dtype": "BF16"},
        "model.layers.0.attn.lora_B.weight": {"shape": [1024, 32], "dtype": "BF16"},
        "__metadata__": {"format": "pt"},
    }
    _write_fake_safetensors(p, header)
    info = read_adapter_header(p)
    assert "error" not in info
    assert info["rank"] == 32
    assert info["n_tensors"] == 2
    assert info["n_params"] == 32 * 1024 * 2
    assert "BF16" in info["dtype"]


def test_read_header_no_lora_tensors(tmp_path):
    p = tmp_path / "base.safetensors"
    header = {
        "model.layers.0.attn.weight": {"shape": [1024, 1024], "dtype": "BF16"},
    }
    _write_fake_safetensors(p, header)
    info = read_adapter_header(p)
    assert info["rank"] is None
    assert info["n_tensors"] == 1


def test_read_header_bad_file(tmp_path):
    p = tmp_path / "corrupt.safetensors"
    p.write_bytes(b"\x00\x01\x02\x03")
    info = read_adapter_header(p)
    assert "error" in info


def test_read_header_missing_file(tmp_path):
    info = read_adapter_header(tmp_path / "nonexistent.safetensors")
    assert "error" in info


# ── scan_lora_adapters ─────────────────────────────────────────────────


def test_scan_finds_lora_files(tmp_path, monkeypatch):
    """scan_lora_adapters uses project_root()/research/checkpoints —
    monkeypatch project_root to our tmp dir."""
    ckpt_dir = tmp_path / "research" / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    # write a lora adapter
    lora_file = ckpt_dir / "ForgeLM_V10_lora.safetensors"
    header = {
        "lora_A.weight": {"shape": [16, 512], "dtype": "BF16"},
        "lora_B.weight": {"shape": [512, 16], "dtype": "BF16"},
    }
    _write_fake_safetensors(lora_file, header)

    # write a non-lora file (should be skipped)
    base_file = ckpt_dir / "ForgeLM_V10_base.safetensors"
    _write_fake_safetensors(base_file, {"weight": {"shape": [10], "dtype": "F32"}})

    import forge_gui.api.lora_store as ls

    monkeypatch.setattr(ls, "project_root", lambda: tmp_path)

    entries = scan_lora_adapters()
    assert len(entries) == 1
    e = entries[0]
    assert e.name == "ForgeLM_V10_lora.safetensors"
    assert e.rank == 16
    assert e.n_tensors == 2
    assert e.base_hint == "ForgeLM_V10"
    assert e.size_bytes > 0
    assert "error" not in e.header_error


def test_scan_empty_when_no_checkpoints(tmp_path, monkeypatch):
    import forge_gui.api.lora_store as ls

    monkeypatch.setattr(ls, "project_root", lambda: tmp_path)
    entries = scan_lora_adapters()
    assert entries == []


def test_scan_sorted_by_modified_desc(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "research" / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    header = {"lora_A": {"shape": [8, 64], "dtype": "BF16"}}
    old = ckpt_dir / "old_lora.safetensors"
    new = ckpt_dir / "new_lora.safetensors"
    _write_fake_safetensors(old, header)
    _write_fake_safetensors(new, header)

    # make 'new' newer
    import os

    os.utime(new, (time.time() + 100, time.time() + 100))

    import forge_gui.api.lora_store as ls

    monkeypatch.setattr(ls, "project_root", lambda: tmp_path)

    entries = scan_lora_adapters()
    assert len(entries) == 2
    assert entries[0].name == "new_lora.safetensors"
    assert entries[1].name == "old_lora.safetensors"
