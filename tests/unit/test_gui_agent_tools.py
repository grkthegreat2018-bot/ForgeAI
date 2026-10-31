"""Unit tests for forge_gui.api.agent_tools (workspace jail + tool behavior)."""
import pytest

from forge_gui.api.agent_tools import ToolSandbox, tool_defs, tool_results_to_text


@pytest.fixture()
def sandbox(tmp_path):
    return ToolSandbox(workspace=tmp_path, default_timeout_s=10.0)


def test_tool_defs_shape():
    defs = tool_defs()
    names = {d["function"]["name"] for d in defs}
    assert {"list_dir", "read_file", "write_file", "run_python",
            "run_cmd", "grep_project"} <= names
    for d in defs:
        assert d["type"] == "function"
        assert "parameters" in d["function"]


def test_path_jail_blocks_absolute(sandbox):
    rec = sandbox.execute("read_file", {"path": "C:/Windows/win.ini"})
    assert not rec["ok"]
    assert "absolute" in rec["result"]["error"]


def test_path_jail_blocks_escape(sandbox):
    rec = sandbox.execute("read_file", {"path": "../../../etc/passwd"})
    assert not rec["ok"]
    assert "escapes workspace" in rec["result"]["error"]


def test_write_read_roundtrip(sandbox):
    sandbox.execute("write_file", {"path": "src/main.py", "content": "print(1)\n"})
    rec = sandbox.execute("read_file", {"path": "src/main.py"})
    assert rec["ok"]
    assert rec["result"]["content"] == "print(1)\n"
    assert not rec["result"]["truncated"]


def test_append_and_delete(sandbox):
    sandbox.execute("write_file", {"path": "log.txt", "content": "one\n"})
    sandbox.execute("append_file", {"path": "log.txt", "content": "two\n"})
    rec = sandbox.execute("read_file", {"path": "log.txt"})
    assert rec["result"]["content"] == "one\ntwo\n"
    rec = sandbox.execute("delete_file", {"path": "log.txt"})
    assert rec["ok"]
    rec = sandbox.execute("read_file", {"path": "log.txt"})
    assert not rec["ok"]


def test_delete_disabled(tmp_path):
    sb = ToolSandbox(workspace=tmp_path, allow_delete=False)
    sb.execute("write_file", {"path": "x.txt", "content": "x"})
    rec = sb.execute("delete_file", {"path": "x.txt"})
    assert not rec["ok"]
    assert "disabled" in rec["result"]["error"]


def test_list_dir(sandbox):
    sandbox.execute("write_file", {"path": "a.py", "content": "x"})
    rec = sandbox.execute("list_dir", {"path": "."})
    names = [e["name"] for e in rec["result"]["entries"]]
    assert "a.py" in names


def test_run_python_success(sandbox):
    rec = sandbox.execute("run_python", {"code": "print('hello')"})["result"]
    assert rec["exit_code"] == 0
    assert "hello" in rec["stdout"]


def test_run_python_failure(sandbox):
    rec = sandbox.execute("run_python", {"code": "raise ValueError('boom')"})["result"]
    assert rec["exit_code"] != 0
    assert "boom" in rec["stderr"]


def test_run_cmd_allowlist(sandbox):
    rec = sandbox.execute("run_cmd", {"command": "del /q important.txt"})["result"]
    assert rec.get("exit_code") != 0 or "error" in rec
    rec = sandbox.execute("run_cmd", {"command": "format C:"})["result"]
    assert "not in allowlist" in rec["error"]


def test_grep_project(sandbox):
    sandbox.execute("write_file", {"path": "mod.py",
                                   "content": "def target_fn():\n    pass\n"})
    rec = sandbox.execute("grep_project", {"pattern": "target_fn", "glob": "*.py"})
    matches = rec["result"]["matches"]
    assert len(matches) == 1
    assert matches[0]["path"] == "mod.py"
    assert matches[0]["line"] == 1


def test_unknown_tool(sandbox):
    rec = sandbox.execute("rm_rf_everything", {})
    assert not rec["ok"]
    assert "unknown tool" in rec["result"]["error"]


def test_calls_journal(sandbox):
    sandbox.execute("write_file", {"path": "a.txt", "content": "x"})
    assert len(sandbox.calls) == 1
    assert sandbox.summary()["n_calls"] == 1
    assert "write_file" in sandbox.summary()["tools_used"]


def test_results_to_text_serializable(sandbox):
    rec = sandbox.execute("write_file", {"path": "a.txt", "content": "x"})
    text = tool_results_to_text(rec)
    assert "written" in text
