"""Tests for time manager, library install, and process protection."""
import json
import time

from forge_gui.api.safety_checker import check_command, check_process_kill

# ── Time Manager ────────────────────────────────────────────────────────

class TestTimeManager:
    def test_get_time(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        info = tm.get_time()
        assert "time" in info
        assert "date" in info
        assert "datetime" in info
        assert "weekday" in info
        assert "unix_timestamp" in info
        assert "uptime_seconds" in info
        tm.shutdown()

    def test_set_timer(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        tid = tm.set_timer(60, label="test timer")
        assert tid.startswith("timer_")
        status = tm.check_timer(tid)
        assert status["status"] == "active"
        assert status["remaining_seconds"] > 55
        assert status["label"] == "test timer"
        tm.shutdown()

    def test_cancel_timer(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        tid = tm.set_timer(60, label="cancel me")
        ok = tm.cancel_timer(tid)
        assert ok
        status = tm.check_timer(tid)
        assert status["status"] == "cancelled"
        tm.shutdown()

    def test_list_timers(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        tm.set_timer(60, label="t1")
        tm.set_timer(120, label="t2")
        timers = tm.list_timers()
        assert len(timers) == 2
        labels = [t["label"] for t in timers]
        assert "t1" in labels
        assert "t2" in labels
        tm.shutdown()

    def test_set_alarm_valid(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        # set alarm for 23 hours from now (should be tomorrow)
        tid = tm.set_alarm("23:59", label="late alarm")
        assert tid.startswith("timer_")
        status = tm.check_timer(tid)
        assert status["kind"] == "alarm"
        assert status["status"] == "active"
        tm.shutdown()

    def test_set_alarm_12h_format(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        tid = tm.set_alarm("11:30 PM", label="night alarm")
        assert tid != ""
        status = tm.check_timer(tid)
        assert status["kind"] == "alarm"
        tm.shutdown()

    def test_set_alarm_invalid(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        tid = tm.set_alarm("not a time", label="bad")
        assert tid == ""
        tm.shutdown()

    def test_timer_conditions(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        tid = tm.set_timer(60, on_process_exit="my_training_job")
        # simulate process exit
        cancelled = tm.check_conditions(active_processes=[], user_prompted=False)
        assert tid in cancelled
        status = tm.check_timer(tid)
        assert status["status"] == "cancelled"
        tm.shutdown()

    def test_timer_on_user_prompt(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        tid = tm.set_timer(60, on_user_prompt=True)
        cancelled = tm.check_conditions(active_processes=None, user_prompted=True)
        assert tid in cancelled
        tm.shutdown()

    def test_timer_not_cancelled_when_process_running(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        tid = tm.set_timer(60, on_process_exit="my_job")
        cancelled = tm.check_conditions(active_processes=["my_job", "other"])
        assert tid not in cancelled
        tm.shutdown()

    def test_timer_fires(self):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        tm = TimeManager()
        fired_signals = []
        tm.timer_fired.connect(lambda tid, label, msg: fired_signals.append((tid, label, msg)))
        # 0.1 second timer
        tid = tm.set_timer(0.1, label="quick", message="fired!")
        # process events to let the QTimer fire
        app.processEvents()
        time.sleep(0.3)
        app.processEvents()
        assert len(fired_signals) >= 1
        tm.shutdown()

    def test_parse_time_24h(self):
        import datetime

        from forge_gui.api.time_manager import _parse_time
        t = _parse_time("17:30")
        assert t == datetime.time(17, 30)

    def test_parse_time_12h_pm(self):
        import datetime

        from forge_gui.api.time_manager import _parse_time
        t = _parse_time("5:00 PM")
        assert t == datetime.time(17, 0)

    def test_parse_time_12h_am(self):
        import datetime

        from forge_gui.api.time_manager import _parse_time
        t = _parse_time("5:00 AM")
        assert t == datetime.time(5, 0)

    def test_parse_time_midnight(self):
        import datetime

        from forge_gui.api.time_manager import _parse_time
        t = _parse_time("12:00 AM")
        assert t == datetime.time(0, 0)

    def test_parse_time_noon(self):
        import datetime

        from forge_gui.api.time_manager import _parse_time
        t = _parse_time("12:00 PM")
        assert t == datetime.time(12, 0)

    def test_parse_time_invalid(self):
        from forge_gui.api.time_manager import _parse_time
        assert _parse_time("25:00") is None
        assert _parse_time("abc") is None


# ── Library Install Manager ─────────────────────────────────────────────

class TestLibraryInstallManager:
    def test_default_allowlist(self, tmp_path, monkeypatch):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        # patch project_root to use tmp_path
        import forge_gui.api.library_install as li_mod
        from forge_gui.api.library_install import LibraryInstallManager
        monkeypatch.setattr(li_mod, "project_root", lambda: tmp_path)
        mgr = LibraryInstallManager()
        allowed = mgr.get_allowlist()
        assert "numpy" in allowed
        assert "requests" in allowed
        assert "pytest" in allowed

    def test_is_allowed(self, tmp_path, monkeypatch):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        import forge_gui.api.library_install as li_mod
        monkeypatch.setattr(li_mod, "project_root", lambda: tmp_path)
        from forge_gui.api.library_install import LibraryInstallManager
        mgr = LibraryInstallManager()
        assert mgr.is_allowed("numpy")
        assert mgr.is_allowed("numpy>=1.20")
        assert not mgr.is_allowed("some_malicious_pkg")

    def test_add_to_allowlist(self, tmp_path, monkeypatch):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        import forge_gui.api.library_install as li_mod
        monkeypatch.setattr(li_mod, "project_root", lambda: tmp_path)
        from forge_gui.api.library_install import LibraryInstallManager
        mgr = LibraryInstallManager()
        mgr.add_to_allowlist("my_custom_lib")
        assert mgr.is_allowed("my_custom_lib")
        # verify persisted
        allowlist_path = tmp_path / "data" / "library_allowlist.json"
        assert allowlist_path.is_file()
        data = json.loads(allowlist_path.read_text())
        assert "my_custom_lib" in data["allowed"]

    def test_allowlist_persistence(self, tmp_path, monkeypatch):
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        import forge_gui.api.library_install as li_mod
        monkeypatch.setattr(li_mod, "project_root", lambda: tmp_path)
        from forge_gui.api.library_install import LibraryInstallManager
        mgr1 = LibraryInstallManager()
        mgr1.add_to_allowlist("persisted_lib")
        # create new manager — should load from disk
        mgr2 = LibraryInstallManager()
        assert mgr2.is_allowed("persisted_lib")

    def test_tool_defs(self):
        from forge_gui.api.library_install import library_tool_defs
        defs = library_tool_defs()
        names = {d["function"]["name"] for d in defs}
        assert "install_library" in names
        assert "check_library" in names
        assert "list_allowed_libraries" in names


# ── Process Protection ──────────────────────────────────────────────────

class TestProcessProtection:
    def test_block_kill_python(self):
        v = check_command("kill -9 python")
        assert not v.safe
        assert "protected" in v.reason.lower()

    def test_block_kill_forge_gui(self):
        v = check_command("taskkill /F /IM forge_gui.exe")
        assert not v.safe
        assert "protected" in v.reason.lower()

    def test_block_pkill_python(self):
        v = check_command("pkill -9 python")
        assert not v.safe

    def test_block_killall_python(self):
        v = check_command("killall python")
        assert not v.safe

    def test_block_kill_forge_server(self):
        v = check_command("kill forge_server")
        assert not v.safe

    def test_block_kill_sft_train(self):
        v = check_command("taskkill /F /IM sft_train.exe")
        assert not v.safe

    def test_allow_kill_non_protected(self):
        # killing a zombie process that's not in the protected list
        v = check_command("kill -9 12345")
        assert v.safe

    def test_allow_non_kill_command(self):
        v = check_command("git status")
        assert v.safe

    def test_check_process_kill_directly(self):
        v = check_process_kill("kill python")
        assert not v.safe

    def test_check_process_kill_safe(self):
        v = check_process_kill("git status")
        assert v.safe


# ── Tool Harness Integration ────────────────────────────────────────────

class TestToolHarnessTimeLibrary:
    def test_time_tools_in_defs(self):
        import sys
        import tempfile

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        from forge_gui.api.tool_harness import ToolHarness
        tm = TimeManager()
        with tempfile.TemporaryDirectory() as td:
            harness = ToolHarness(td, time_manager=tm)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            assert "get_time" in names
            assert "set_timer" in names
            assert "set_alarm" in names
            assert "check_timer" in names
            assert "cancel_timer" in names
            assert "list_timers" in names
        tm.shutdown()

    def test_library_tools_in_defs(self):
        import sys
        import tempfile

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.library_install import LibraryInstallManager
        from forge_gui.api.tool_harness import ToolHarness
        with tempfile.TemporaryDirectory() as td:
            mgr = LibraryInstallManager()
            harness = ToolHarness(td, library_manager=mgr)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            assert "install_library" in names
            assert "check_library" in names
            assert "list_allowed_libraries" in names

    def test_time_tools_available_in_read_only(self):
        """Time tools should be available even in read-only mode (no side effects)."""
        import sys
        import tempfile

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.time_manager import TimeManager
        from forge_gui.api.tool_harness import ToolHarness
        tm = TimeManager()
        with tempfile.TemporaryDirectory() as td:
            harness = ToolHarness(td, time_manager=tm, read_only=True)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            assert "get_time" in names
            assert "list_timers" in names
        tm.shutdown()

    def test_library_tools_hidden_in_read_only(self):
        """Library install should be hidden in read-only mode."""
        import sys
        import tempfile

        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from forge_gui.api.library_install import LibraryInstallManager
        from forge_gui.api.tool_harness import ToolHarness
        with tempfile.TemporaryDirectory() as td:
            mgr = LibraryInstallManager()
            harness = ToolHarness(td, library_manager=mgr, read_only=True)
            defs = harness.tool_defs()
            names = {d["function"]["name"] for d in defs}
            assert "install_library" not in names
            assert "list_allowed_libraries" not in names
