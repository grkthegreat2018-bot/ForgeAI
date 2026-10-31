"""Unit tests for forge_gui.api.lora_training_trigger.

Tests tool definitions and execution with mocked ProcessManager.
Pure stdlib — no Qt/torch.
"""
from unittest.mock import MagicMock

import pytest

from forge_gui.api.lora_training_trigger import (
    LoraTrainingTrigger,
    lora_training_tool_defs,
)


class TestLoraTrainingToolDefs:
    def test_defs_present(self):
        defs = lora_training_tool_defs()
        names = [d["function"]["name"] for d in defs]
        assert "train_lora" in names
        assert "check_training" in names

    def test_train_lora_has_category_param(self):
        defs = lora_training_tool_defs()
        train_def = next(d for d in defs if d["function"]["name"] == "train_lora")
        props = train_def["function"]["parameters"]["properties"]
        assert "category" in props
        assert "rank" in props
        assert "data_source" in props


class TestLoraTrainingTrigger:
    @pytest.fixture()
    def trigger(self):
        proc_mgr = MagicMock()
        proc_mgr.launch.return_value = "task-123"
        return LoraTrainingTrigger(proc_mgr=proc_mgr,
                                   chat_store=None,
                                   checkpoint="fake.ckpt")

    def test_unknown_tool(self, trigger):
        result = trigger.execute("nonexistent", {})
        assert result["ok"] is False

    def test_train_lora_no_category(self, trigger):
        result = trigger.execute("train_lora", {})
        assert result["ok"] is False
        assert "category" in result["result"]["error"]

    def test_train_lora_no_proc_mgr(self, tmp_path):
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"messages": []}', encoding="utf-8")
        trigger = LoraTrainingTrigger(proc_mgr=None,
                                      chat_store=None,
                                      checkpoint="fake.ckpt")
        result = trigger.execute("train_lora",
                                 {"category": "coding",
                                  "data_source": "file",
                                  "data_file": str(data_file)})
        assert result["ok"] is False
        assert "ProcessManager" in result["result"]["error"]

    def test_train_lora_file_not_found(self, trigger):
        result = trigger.execute("train_lora",
                                 {"category": "coding",
                                  "data_source": "file",
                                  "data_file": "nonexistent.jsonl"})
        assert result["ok"] is False
        assert "not found" in result["result"]["error"]

    def test_train_lora_no_chat_store(self):
        trigger = LoraTrainingTrigger(proc_mgr=MagicMock(),
                                      chat_store=None,
                                      checkpoint="fake.ckpt")
        result = trigger.execute("train_lora",
                                 {"category": "coding",
                                  "data_source": "chat_history"})
        assert result["ok"] is False
        assert "chat_store" in result["result"]["error"]

    def test_check_training_unknown_task(self, trigger):
        result = trigger.execute("check_training", {"task_id": "unknown"})
        assert result["ok"] is False

    def test_training_tasks_property(self, trigger):
        assert trigger.training_tasks == {}
