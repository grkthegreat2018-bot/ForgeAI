"""Tests for safety guardrails, backup manager, sub-agents, and coding tools."""
import tempfile
import zipfile
from pathlib import Path

import pytest

from forge_gui.api.safety_checker import check_command, check_patterns

# ── Safety: email/forum/web-send blocking ──────────────────────────────

class TestSafetyGuardrails:
    def test_block_smtplib(self):
        v = check_patterns("import smtplib; smtplib.SMTP('smtp.gmail.com')", "test.py")
        assert not v.safe
        assert "smtplib" in v.reason.lower() or "smtp" in v.reason.lower()

    def test_block_requests_post(self):
        v = check_patterns("requests.post('https://api.github.com/...')", "test.py")
        assert not v.safe
        assert "http" in v.reason.lower() or "post" in v.reason.lower()

    def test_block_curl_post(self):
        v = check_command("curl -X POST https://example.com/api -d 'data'")
        assert not v.safe

    def test_block_curl_data(self):
        v = check_command("curl -d 'secret' https://example.com")
        assert not v.safe

    def test_block_wget_post(self):
        v = check_command("wget --post-data='data' https://example.com")
        assert not v.safe

    def test_block_scp(self):
        v = check_command("scp file.txt user@host:/tmp/")
        assert not v.safe

    def test_block_ssh(self):
        v = check_command("ssh user@host")
        assert not v.safe

    def test_block_sendmail_cmd(self):
        v = check_command("sendmail recipient@example.com")
        assert not v.safe

    def test_block_webhook(self):
        v = check_patterns("response = requests.post(webhook_url)", "test.py")
        assert not v.safe

    def test_block_twilio(self):
        v = check_patterns("from twilio.rest import Client", "test.py")
        assert not v.safe

    def test_block_discord_bot(self):
        v = check_patterns("import discord; bot = discord.Client()", "test.py")
        assert not v.safe

    def test_allow_curl_get(self):
        # plain curl GET (no -X POST, no -d) should pass pattern check
        # (though curl itself may not be in the allowlist for run_cmd)
        v = check_patterns("curl https://example.com/api/data", "test.sh")
        assert v.safe

    def test_allow_requests_get(self):
        # requests.get is fine (read-only)
        v = check_patterns("response = requests.get('https://example.com')", "test.py")
        assert v.safe

    def test_block_social_media_api(self):
        v = check_patterns("requests.post('https://reddit.com/api/submit')", "test.py")
        assert not v.safe


# ── Coding tools: git, search_replace, create_file ─────────────────────

class TestCodingTools:
    @pytest.fixture()
    def workspace(self, tmp_path):
        (tmp_path / "test.py").write_text("print('hello')\n", encoding="utf-8")
        return tmp_path

    def test_search_replace_unique(self, workspace):
        from forge_gui.api.agent_tools import ToolSandbox
        sb = ToolSandbox(workspace)
        rec = sb.execute("search_replace", {
            "path": "test.py",
            "old_text": "print('hello')",
            "new_text": "print('world')"})
        assert rec["ok"]
        assert "replacements" in rec["result"]
        assert workspace.joinpath("test.py").read_text() == "print('world')\n"

    def test_search_replace_not_found(self, workspace):
        from forge_gui.api.agent_tools import ToolSandbox
        sb = ToolSandbox(workspace)
        rec = sb.execute("search_replace", {
            "path": "test.py",
            "old_text": "nonexistent",
            "new_text": "whatever"})
        assert not rec["ok"]
        assert "not found" in rec["result"]["error"]

    def test_search_replace_not_unique(self, workspace):
        from forge_gui.api.agent_tools import ToolSandbox
        (workspace / "dup.py").write_text("foo\nfoo\n", encoding="utf-8")
        sb = ToolSandbox(workspace)
        rec = sb.execute("search_replace", {
            "path": "dup.py",
            "old_text": "foo",
            "new_text": "bar"})
        assert not rec["ok"]
        assert "not unique" in rec["result"]["error"]

    def test_search_replace_all(self, workspace):
        from forge_gui.api.agent_tools import ToolSandbox
        (workspace / "dup.py").write_text("foo\nfoo\n", encoding="utf-8")
        sb = ToolSandbox(workspace)
        rec = sb.execute("search_replace", {
            "path": "dup.py",
            "old_text": "foo",
            "new_text": "bar",
            "replace_all": True})
        assert rec["ok"]
        assert workspace.joinpath("dup.py").read_text() == "bar\nbar\n"

    def test_create_file_new(self, workspace):
        from forge_gui.api.agent_tools import ToolSandbox
        sb = ToolSandbox(workspace)
        rec = sb.execute("create_file", {
            "path": "new_file.py",
            "content": "# new file\n"})
        assert rec["ok"]
        assert workspace.joinpath("new_file.py").exists()

    def test_create_file_exists(self, workspace):
        from forge_gui.api.agent_tools import ToolSandbox
        sb = ToolSandbox(workspace)
        rec = sb.execute("create_file", {
            "path": "test.py",
            "content": "# overwrite"})
        assert not rec["ok"]
        assert "already exists" in rec["result"]["error"]

    def test_git_status(self, workspace):
        from forge_gui.api.agent_tools import ToolSandbox
        sb = ToolSandbox(workspace)
        rec = sb.execute("git_status", {})
        # git may not be initialized, but should not crash
        assert "exit_code" in rec["result"] or "error" in rec["result"]

    def test_git_log(self, workspace):
        from forge_gui.api.agent_tools import ToolSandbox
        sb = ToolSandbox(workspace)
        rec = sb.execute("git_log", {"n": 5})
        assert "exit_code" in rec["result"] or "error" in rec["result"]


# ── Backup manager ──────────────────────────────────────────────────────

class TestBackupManager:
    @pytest.fixture()
    def project(self, tmp_path):
        (tmp_path / "main.py").write_text("print('v1')\n", encoding="utf-8")
        (tmp_path / "utils.py").write_text("def foo(): pass\n", encoding="utf-8")
        (tmp_path / "data").mkdir()
        return tmp_path

    def test_create_backup(self, project):
        from forge_gui.api.backup_manager import BackupManager
        bm = BackupManager(project)
        path = bm.create_backup()
        assert path is not None
        assert Path(path).exists()
        assert path.endswith(".zip")
        assert project.name in path

    def test_backup_naming(self, project):
        from forge_gui.api.backup_manager import BackupManager
        bm = BackupManager(project)
        path = bm.create_backup()
        name = Path(path).name
        # should be projectname_date_time.zip
        assert name.startswith(project.name + "_")
        assert name.endswith(".zip")

    def test_backup_zip_content(self, project):
        from forge_gui.api.backup_manager import BackupManager
        bm = BackupManager(project)
        path = bm.create_backup()
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "main.py" in names
            assert "utils.py" in names

    def test_list_backups(self, project):
        from forge_gui.api.backup_manager import BackupManager
        bm = BackupManager(project)
        bm.create_backup()
        backups = bm.list_backups()
        assert len(backups) >= 1
        assert "name" in backups[0]
        assert "size_mb" in backups[0]
        assert "date" in backups[0]

    def test_count_changes(self, project):
        from forge_gui.api.backup_manager import BackupManager
        bm = BackupManager(project)
        bm._snapshot_hashes()
        # modify files
        (project / "main.py").write_text("print('v2')\n", encoding="utf-8")
        (project / "utils.py").write_text("def bar(): pass\n", encoding="utf-8")
        (project / "new.py").write_text("# new\n", encoding="utf-8")
        changes = bm._count_changes()
        assert changes >= 3

    def test_prune_old_backups(self, project):
        from forge_gui.api.backup_manager import BackupManager
        bm = BackupManager(project)
        bm.MAX_BACKUPS = 2
        for _ in range(5):
            bm.create_backup()
        backups = bm.list_backups()
        assert len(backups) <= 2

    def test_backup_skip_dirs(self, project):
        from forge_gui.api.backup_manager import BackupManager
        bm = BackupManager(project)
        path = bm.create_backup()
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            # data/ dir should be skipped
            assert not any(n.startswith("data/") for n in names)


# ── Sub-agent manager (no engine needed for basic tests) ───────────────

class TestSubAgentManager:
    def test_tool_defs(self):
        from forge_gui.api.sub_agent import sub_agent_tool_defs
        defs = sub_agent_tool_defs()
        names = {d["function"]["name"] for d in defs}
        assert "spawn_sub_agent" in names
        assert "spawn_sub_agents" in names
        assert "check_sub_agent" in names
        assert "wait_sub_agents" in names
        assert "list_sub_agents" in names

    def test_spawn_returns_task_id(self):
        from forge_gui.api.sub_agent import SubAgentManager
        # use a mock engine runtime that raises immediately
        class MockRuntime:
            def acquire(self, timeout_s=30):
                raise RuntimeError("no engine")
        mgr = SubAgentManager(MockRuntime())
        tid = mgr.spawn("test prompt")
        assert tid.startswith("sub_")
        # wait for the future to complete (avoids race with fixed sleep)
        future = mgr._futures[tid]
        future.result(timeout=10.0)
        task = mgr.get_result(tid)
        assert task.status == "error"
        mgr.shutdown()

    def test_spawn_batch(self):
        from forge_gui.api.sub_agent import SubAgentManager
        class MockRuntime:
            def acquire(self, timeout_s=30):
                raise RuntimeError("no engine")
        mgr = SubAgentManager(MockRuntime())
        ids = mgr.spawn_batch([
            {"prompt": "task 1"},
            {"prompt": "task 2"},
            {"prompt": "task 3"},
        ])
        assert len(ids) == 3
        import time as _t
        _t.sleep(0.5)
        tasks = mgr.list_tasks()
        assert len(tasks) == 3
        mgr.shutdown()

    def test_list_tasks(self):
        from forge_gui.api.sub_agent import SubAgentManager
        class MockRuntime:
            def acquire(self, timeout_s=30):
                raise RuntimeError("no engine")
        mgr = SubAgentManager(MockRuntime())
        mgr.spawn("test")
        import time as _t
        _t.sleep(0.5)
        tasks = mgr.list_tasks()
        assert len(tasks) == 1
        assert "task_id" in tasks[0]
        assert "status" in tasks[0]
        mgr.shutdown()


# ── Tool harness integration ────────────────────────────────────────────

class TestToolHarnessIntegration:
    def test_backup_tools_in_defs(self):
        from forge_gui.api.backup_manager import BackupManager
        from forge_gui.api.tool_harness import ToolHarness
        with tempfile.TemporaryDirectory() as td:
            bm = BackupManager(Path(td))
            harness = ToolHarness(td, backup_manager=bm)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            assert "list_backups" in names
            assert "create_backup" in names
            assert "load_backup" in names

    def test_sub_agent_tools_in_defs(self):
        from forge_gui.api.sub_agent import SubAgentManager
        from forge_gui.api.tool_harness import ToolHarness
        class MockRuntime:
            def acquire(self, timeout_s=30):
                raise RuntimeError("no engine")
        mgr = SubAgentManager(MockRuntime())
        with tempfile.TemporaryDirectory() as td:
            harness = ToolHarness(td, sub_agent_manager=mgr)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            assert "spawn_sub_agent" in names
            assert "list_sub_agents" in names

    def test_new_coding_tools_in_defs(self):
        from forge_gui.api.tool_harness import ToolHarness
        with tempfile.TemporaryDirectory() as td:
            harness = ToolHarness(td)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            assert "search_replace" in names
            assert "create_file" in names
            assert "git_status" in names
            assert "git_diff" in names
            assert "git_log" in names
            assert "git_revert" in names

    def test_read_only_hides_backup_tools(self):
        from forge_gui.api.backup_manager import BackupManager
        from forge_gui.api.tool_harness import ToolHarness
        with tempfile.TemporaryDirectory() as td:
            bm = BackupManager(Path(td))
            harness = ToolHarness(td, backup_manager=bm, read_only=True)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            assert "list_backups" not in names
            assert "create_backup" not in names
