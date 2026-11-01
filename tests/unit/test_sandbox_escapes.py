"""Regression tests for the hardened sandbox workers.

Covers research/persistent_executor.py (PEP 578 audit-hook jail) and the
self_play SandboxExecutor fallback path — the vectors that used to bypass
the old import-blocklist: direct module import, importlib, preloaded
sys.modules, builtins mutation persistence, raw open(), env leakage,
process/network spawn, and timeout/respawn behavior.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.persistent_executor import PersistentCodeExecutor


@pytest.fixture()
def executor(tmp_path):
    ex = PersistentCodeExecutor(timeout_s=10, memory_limit_mb=512, fs_root=str(tmp_path))
    try:
        yield ex
    finally:
        ex.close()


def _run(ex, code):
    r = ex.execute(code)
    assert r["returncode"] == 0, f"worker error: {r}"
    return r


def _fail(ex, code):
    r = ex.execute(code)
    assert r["returncode"] != 0, f"escape succeeded: stdout={r.get('stdout')!r}"
    return r


class TestBasics:
    def test_arithmetic(self, executor):
        r = _run(executor, "print(6 * 7)")
        assert r["stdout"].strip() == "42"

    def test_stdlib_math(self, executor):
        r = _run(executor, "import math, json\nprint(json.dumps({'r': round(math.sqrt(2), 3)}))")
        assert '"r": 1.414' in r["stdout"]

    def test_scratch_write_allowed(self, executor, tmp_path):
        _run(executor, "f = open('ok.txt', 'w'); f.write('hi'); f.close()")
        assert (tmp_path / "ok.txt").read_text() == "hi"


class TestProcessNetworkDenied:
    @pytest.mark.parametrize(
        "code",
        [
            "import os; os.system('echo pwned')",
            "import os; os.popen('echo hi')",
            "import subprocess; subprocess.run(['echo','x'])",
            "import subprocess",
            "import multiprocessing",
            "import socket; socket.socket()",
            "import socket",
            "import urllib.request",
            "import ctypes",
            "import ctypes.windll",
            "import signal",
            "import marshal",
            "import pickle",
        ],
    )
    def test_denied(self, executor, code):
        _fail(executor, code)

    def test_builtins_exec_removed(self, executor):
        r = _run(
            executor,
            "print('exec' in dir(__builtins__) if not isinstance(__builtins__, dict) else 'exec' in __builtins__)",
        )
        assert r["stdout"].strip() == "False"

    def test_dynamic_import_blocked(self, executor):
        _fail(executor, "import importlib; importlib.import_module('os')")

    def test_os_system_via_preloaded(self, executor):
        # os is importable (needed for legit stdlib) but every dangerous
        # call is audit-denied regardless of how it was reached.
        r = _fail(executor, "import os; os.system('echo should_not_run')")
        assert "pwn" not in r.get("stdout", "")


class TestFilesystemJail:
    def test_open_outside_denied(self, executor):
        _fail(executor, "open(r'C:\\Windows\\System32\\drivers\\etc\\hosts')")

    def test_open_outside_write_denied(self, executor):
        _fail(executor, "open(r'..\\escape.txt', 'w').write('x')")

    def test_listdir_outside_denied(self, executor):
        _fail(executor, "import os; os.listdir('C:\\\\')")

    def test_env_scrubbed(self, executor):
        r = _run(executor, "import os; print(sorted(os.environ.keys()))")
        keys = r["stdout"].lower()
        for secret in ("api_key", "token", "secret", "hf_"):
            assert secret not in keys


class TestWorkerResilience:
    def test_builtins_patch_does_not_persist(self, executor):
        _run(executor, "__builtins__['pwned'] = 1; print('patched')")
        r = _run(executor, "print('pwned' in __builtins__)")
        assert r["stdout"].strip() == "False"

    def test_timeout_kills_and_respawns(self, tmp_path):
        ex = PersistentCodeExecutor(timeout_s=2, memory_limit_mb=512, fs_root=str(tmp_path / "ex2"))
        try:
            t0 = time.time()
            r = ex.execute("while True: pass")
            assert time.time() - t0 < 15
            assert r["returncode"] != 0 or r.get("timed_out")
            # worker respawned: ordinary code still executes
            r2 = ex.execute("print(1 + 1)")
            assert r2["stdout"].strip() == "2"
        finally:
            ex.close()


class TestOnceMode:
    def test_once_script_runs(self, tmp_path):
        import json
        import subprocess

        from research.persistent_executor import build_once_script, scrub_environ

        once = tmp_path / "once.py"
        once.write_text(build_once_script(), encoding="utf-8")
        code = tmp_path / "code.py"
        code.write_text("print('once_ok')", encoding="utf-8")
        res = tmp_path / "res.json"
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        subprocess.run(
            [sys.executable, str(once), str(scratch), str(tmp_path / "proto"), "--once", str(code), str(res)],
            env=scrub_environ(),
            timeout=60,
            check=False,
        )
        data = json.loads(res.read_text(encoding="utf-8"))
        assert data["returncode"] == 0
        assert "once_ok" in data["stdout"]
