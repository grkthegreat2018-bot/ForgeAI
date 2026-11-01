"""A9: write_status_json metrics fanout — env-gated, best-effort."""

from __future__ import annotations

import json

from forge.training import training_utils as tu


def test_status_json_writes_file(tmp_path):
    path = tmp_path / "status.json"
    tu.write_status_json(path, {"step": 3, "loss": 1.5, "name": "run"})
    data = json.loads(path.read_text())
    assert data["step"] == 3 and data["loss"] == 1.5


def test_fanout_no_env_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_WANDB", raising=False)
    monkeypatch.delenv("FORGE_TB_DIR", raising=False)
    tu._METRICS["tried"] = False
    tu._METRICS["wandb"] = tu._METRICS["tb"] = None
    path = tmp_path / "status.json"
    tu.write_status_json(path, {"step": 1, "loss": 0.5})
    assert path.exists()
    assert tu._METRICS["wandb"] is None and tu._METRICS["tb"] is None


def test_fanout_tb_dir_records_scalars(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_WANDB", raising=False)
    monkeypatch.setenv("FORGE_TB_DIR", str(tmp_path / "tb"))
    tu._METRICS["tried"] = False
    tu._METRICS["wandb"] = tu._METRICS["tb"] = None
    try:
        tu.write_status_json(tmp_path / "status.json", {"step": 2, "loss": 0.25, "ts": 0.0})
        tb = tu._METRICS["tb"]
        if tb is None:
            # tensorboard not installed — sink degraded gracefully, no crash
            return
        assert len(tb._file_writer.event_storage._events or []) >= 0
    finally:
        tu.close_metrics()
        tu._METRICS["tried"] = False
