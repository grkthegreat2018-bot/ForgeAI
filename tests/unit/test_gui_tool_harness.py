"""Unit tests for forge_gui.api.tool_harness.

Tests the unified harness: tool dispatch, safety integration, read-only
mode, memory tools, and LoRA tools (mocked). Pure stdlib — no Qt/torch.
"""

import pytest

from forge_gui.api.lorebook import Lorebook
from forge_gui.api.tool_harness import ToolHarness


@pytest.fixture()
def harness(tmp_path):
    lore = Lorebook(root=tmp_path)
    return ToolHarness(workspace=str(tmp_path), lorebook=lore, lora_harness=None, read_only=False, enable_safety=True)


@pytest.fixture()
def read_only_harness(tmp_path):
    lore = Lorebook(root=tmp_path)
    return ToolHarness(workspace=str(tmp_path), lorebook=lore, lora_harness=None, read_only=True, enable_safety=True)


# ── Tool definitions ───────────────────────────────────────────────────


class TestToolDefs:
    def test_coding_tools_present(self, harness):
        names = [d["function"]["name"] for d in harness.tool_defs()]
        assert "list_dir" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "run_python" in names

    def test_memory_tools_present(self, harness):
        names = [d["function"]["name"] for d in harness.tool_defs()]
        assert "remember" in names
        assert "recall_memory" in names
        assert "forget" in names

    def test_lora_tools_absent_without_harness(self, harness):
        names = [d["function"]["name"] for d in harness.tool_defs()]
        assert "load_lora" not in names

    def test_read_only_filters_side_effects(self, read_only_harness):
        names = [d["function"]["name"] for d in read_only_harness.tool_defs()]
        assert "write_file" not in names
        assert "delete_file" not in names
        assert "run_python" not in names
        assert "run_cmd" not in names
        # read tools still present
        assert "list_dir" in names
        assert "read_file" in names
        # memory tools still present
        assert "remember" in names


# ── Coding tools ───────────────────────────────────────────────────────


class TestCodingTools:
    def test_list_dir(self, harness):
        rec = harness.execute("list_dir", {"path": "."})
        assert rec["ok"]
        assert "entries" in rec["result"]

    def test_write_and_read(self, harness):
        harness.execute("write_file", {"path": "test.txt", "content": "hello"})
        rec = harness.execute("read_file", {"path": "test.txt"})
        assert rec["ok"]
        assert rec["result"]["content"] == "hello"

    def test_read_only_blocks_write(self, read_only_harness):
        rec = read_only_harness.execute("write_file", {"path": "test.txt", "content": "x"})
        assert not rec["ok"]
        assert "read-only" in rec["result"]["error"]


# ── Safety integration ─────────────────────────────────────────────────


class TestSafetyIntegration:
    def test_safe_write_passes(self, harness):
        rec = harness.execute("write_file", {"path": "safe.py", "content": "x = 1"})
        assert rec["ok"]

    def test_dangerous_write_blocked(self, harness):
        rec = harness.execute("write_file", {"path": "evil.py", "content": "import os\nos.system('rm -rf /')"})
        assert not rec["ok"]
        assert "SAFETY" in rec["result"]["error"]
        assert rec["result"]["strikes"] == 1

    def test_dangerous_command_blocked(self, harness):
        rec = harness.execute("run_cmd", {"command": "rm -rf /"})
        assert not rec["ok"]
        assert "SAFETY" in rec["result"]["error"]

    def test_three_strikes_terminate(self, harness):
        for i in range(3):
            harness.execute("write_file", {"path": f"evil{i}.py", "content": "import os\nos.system('rm -rf /')"})
        assert harness.strikes.terminated

    def test_safe_python_passes(self, harness):
        rec = harness.execute("run_python", {"code": "print(1+1)"})
        assert rec["ok"]

    def test_dangerous_python_blocked(self, harness):
        rec = harness.execute("run_python", {"code": "import os\nos.system('ls')"})
        assert not rec["ok"]
        assert "SAFETY" in rec["result"]["error"]

    def test_reset_strikes(self, harness):
        harness.execute("write_file", {"path": "evil.py", "content": "import os\nos.system('rm -rf /')"})
        assert harness.strikes.count == 1
        harness.reset_strikes()
        assert harness.strikes.count == 0


# ── Memory tools ───────────────────────────────────────────────────────


class TestMemoryToolsInHarness:
    def test_remember(self, harness):
        rec = harness.execute("remember", {"keys": ["python"], "content": "use snake_case"})
        assert rec["ok"]
        assert "id" in rec["result"]

    def test_recall_memory(self, harness):
        harness.execute("remember", {"keys": ["python"], "content": "use snake_case"})
        rec = harness.execute("recall_memory", {"query": "python"})
        assert rec["ok"]
        assert rec["result"]["count"] == 1

    def test_forget(self, harness):
        harness.execute("remember", {"keys": ["python"], "content": "use snake_case"})
        rec = harness.execute("forget", {"keys": ["python"]})
        assert rec["ok"]
        assert rec["result"]["deleted"] == 1


# ── Summary ────────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_includes_strikes(self, harness):
        s = harness.summary()
        assert "strikes" in s
        assert "no safety violations" in s["strikes"]

    def test_summary_includes_memory(self, harness):
        s = harness.summary()
        assert "memory" in s
        assert s["memory"]["total"] == 0

    def test_summary_read_only(self, read_only_harness):
        s = read_only_harness.summary()
        assert s["read_only"] is True
