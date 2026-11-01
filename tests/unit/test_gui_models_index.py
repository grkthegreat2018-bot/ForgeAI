"""Regression tests for ModelsIndex meta parsing.

Bug (2026-09-07): SFT checkpoints store ``"config"`` in their .meta.json as a
full config *dict* (not a registry name string). ``ModelsIndex.models()`` fed
that dict into ``dict.get()`` → ``TypeError: unhashable type: 'dict'``, which
broke the Dashboard refresh and the Engine page checkpoint list on every tick.
"""

from __future__ import annotations

import json

import pytest

from forge_gui.api.models_index import ModelsIndex


@pytest.fixture()
def ckpt_root(tmp_path, monkeypatch):
    """Temp project root with research/checkpoints/ + monkeypatched root."""
    ckpt_dir = tmp_path / "research" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    import forge_gui.api.models_index as mi_mod

    monkeypatch.setattr(mi_mod, "project_root", lambda: tmp_path)
    return ckpt_dir


def _write_ckpt(ckpt_dir, name, meta):
    (ckpt_dir / name).write_bytes(b"\x00" * 16)
    if meta is not None:
        (ckpt_dir / (name + ".meta.json")).write_text(json.dumps(meta), encoding="utf-8")


class TestMetaConfigParsing:
    def test_dict_config_does_not_crash(self, ckpt_root):
        """meta['config'] as a dict must not raise unhashable-dict TypeError."""
        _write_ckpt(ckpt_root, "m.safetensors", {"step": 40, "config": {"d_model": 2048, "n_layers": 16}})
        idx = ModelsIndex()
        models = idx.models()  # used to raise TypeError
        assert len(models) == 1
        entry = models[0]
        assert entry.config == {"d_model": 2048, "n_layers": 16}
        assert entry.config_name is None  # not a registry name

    def test_string_config_resolves_registry(self, ckpt_root):
        _write_ckpt(ckpt_root, "m.safetensors", {"config": "forgelm_v2"})
        idx = ModelsIndex()
        idx._configs = {"forgelm_v2": {"d_model": 2048}}
        models = idx.models()
        assert len(models) == 1
        assert models[0].config_name == "forgelm_v2"
        assert models[0].config == {"d_model": 2048}

    def test_unknown_string_config_yields_empty_cfg(self, ckpt_root):
        _write_ckpt(ckpt_root, "m.safetensors", {"config": "no_such_preset"})
        idx = ModelsIndex()
        idx._configs = {}
        models = idx.models()
        assert len(models) == 1
        assert models[0].config_name == "no_such_preset"
        assert models[0].config == {}

    def test_no_meta_yields_empty_cfg(self, ckpt_root):
        _write_ckpt(ckpt_root, "m.safetensors", None)
        idx = ModelsIndex()
        models = idx.models()
        assert len(models) == 1
        assert models[0].config_name is None
        assert models[0].config == {}
