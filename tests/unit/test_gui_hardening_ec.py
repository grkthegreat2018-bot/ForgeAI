"""Focused tests for the ENGINE_CRITIQUE_DB GUI-server hardening fixes.

Self-contained, CPU-only: no engine, no torch, no Qt event loop. Each
test names the EC item it guards so a regression maps straight back to
the critique entry.
"""

from __future__ import annotations

import json
import threading
import time
import zipfile
from pathlib import Path

import pytest

from forge_gui.api.agent_tools import CMD_ALLOWLIST, ToolSandbox
from forge_gui.api.chat_store import ChatStore
from forge_gui.api.events_reader import EventsReader
from forge_gui.api.lorebook import Lorebook
from forge_gui.api.process_manager import ProcessPreset, build_command, preset_script_exists
from forge_gui.api.status_reader import StatusReader
from forge_gui_server.hub import EventHub
from forge_gui_server.services.procs import ProcessService, TaskInfo
from forge_gui_server.services.support import ApprovalChannel, BackupService


class _FakeHub:
    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def publish(self, kind, payload=None):
        evt = {"kind": kind, "payload": payload}
        self.events.append((kind, payload))
        return evt


# ── EC-184: agent command allowlist + argv[0] confinement ────────────


def test_cmd_allowlist_excludes_interpreters_and_vcs():
    for banned in ("python", "python3", "pip", "pip3", "git"):
        assert banned not in CMD_ALLOWLIST


def test_run_cmd_rejects_path_argv0(tmp_path):
    sb = ToolSandbox(tmp_path)
    for evil in (
        r"..\tools\pytest.exe --version",
        r"..\\evil\\pytest.exe",
        "C:/Windows/System32/cmd.exe /c dir",
        "./pytest --version",
    ):
        res = sb.tool_run_cmd(evil)
        assert "error" in res, evil
        assert "bare name" in res["error"]


def test_run_cmd_rejects_non_allowlisted_names(tmp_path):
    sb = ToolSandbox(tmp_path)
    res = sb.tool_run_cmd('python -c "print(1)"')
    assert "not in allowlist" in res["error"]
    res = sb.tool_run_cmd("git status")
    assert "not in allowlist" in res["error"]


# ── EC-185 / EC-077: launch presets ───────────────────────────────────


def test_preset_overrides_fill_only_declared_args():
    preset = ProcessPreset(
        name="t", script="scripts/x.py", description="", arg_defaults={"--data": "", "--epochs": "3"}
    )
    cmd = build_command(preset, {"--data": "d.jsonl", "--evil-flag": "1", "argv0": "calc.exe"})
    assert cmd[1].endswith("scripts\\x.py") or cmd[1].endswith("scripts/x.py")
    joined = " ".join(cmd)
    assert "--evil-flag" not in joined
    assert "calc.exe" not in joined
    i = cmd.index("--data")
    assert cmd[i + 1] == "d.jsonl"


def test_all_presets_have_existing_scripts():
    from forge_gui.api.process_manager import get_presets

    presets = get_presets()
    assert presets, "expected at least one launch preset"
    for p in presets:
        assert preset_script_exists(p), f"preset script missing: {p.script}"


# ── EC-186: backup restore zip validation + staging ──────────────────


def test_zip_member_validation():
    v = BackupService._validate_zip_member
    v("data/chats/conversations.json")  # clean member OK
    v("nested/dir/file.txt")
    for bad in (
        "../escape.txt",
        "a/../../x",
        "..\\win\\x",
        "C:/abs/x",
        "C:\\abs\\x",
        "/root/x",
        "\\unc\\x",
        "x/../../y",
    ):
        with pytest.raises(ValueError):
            v(bad)


def _make_service(tmp_path):
    return BackupService(_FakeHub(), None, root=tmp_path)


def test_restore_rejects_outside_backup_dir(tmp_path):
    svc = _make_service(tmp_path)
    evil = tmp_path / "outside.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("data/x.txt", "x")
    with pytest.raises(ValueError):
        svc._restore_backup(str(evil))


def test_restore_rejects_traversal_members_without_wiping(tmp_path):
    svc = _make_service(tmp_path)
    keep = tmp_path / "keep.txt"
    keep.write_text("precious", encoding="utf-8")
    evil = svc.backup_dir / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../evil.txt", "pwned")
        zf.writestr("fine.txt", "ok")
    with pytest.raises(ValueError):
        svc._restore_backup(str(evil))
    # Validation happens BEFORE the wipe — the project is untouched.
    assert keep.read_text() == "precious"
    assert not (tmp_path / "evil.txt").exists()


def test_restore_stages_then_swaps(tmp_path):
    svc = _make_service(tmp_path)
    (tmp_path / "old.txt").write_text("old", encoding="utf-8")
    good = svc.backup_dir / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("data/new.txt", "new-content")
    svc._restore_backup(str(good))
    assert (tmp_path / "data" / "new.txt").read_text() == "new-content"
    assert not (tmp_path / "old.txt").exists()  # wiped then swapped
    assert not list(svc.backup_dir.glob(".restore_staging_*"))


# ── EC-203: file change hashing ───────────────────────────────────────


def test_hash_file_catches_late_edits(tmp_path):
    f = tmp_path / "big.txt"
    f.write_bytes(b"A" * 65536)
    h1 = BackupService._hash_file(f)
    time.sleep(0.01)
    # Edit only the TAIL — the old size+first-4KB hash missed this.
    data = bytearray(f.read_bytes())
    data[-10:] = b"ZZZZZZZZZZ"
    f.write_bytes(bytes(data))
    assert BackupService._hash_file(f) != h1


# ── EC-211 / EC-212: store locking + immutable fields + counter ──────


def test_lorebook_update_cannot_rewrite_identity(tmp_path):
    lb = Lorebook(root=tmp_path)
    e = lb.add(keys=["python"], content="original")
    old_created = e.created
    lb.update(e.id, id="hijacked", keys=["other"], created="x", trigger_count=99, content="updated", constant=True)
    e2 = lb.get(e.id)
    assert e2.id == e.id
    assert e2.keys == ["python"]
    assert e2.created == old_created
    assert e2.trigger_count == 0
    assert e2.content == "updated" and e2.constant is True


def test_approval_ids_unique_under_threads():
    ch = ApprovalChannel(_FakeHub())
    ids = []
    threading.Lock()

    def req():
        # timeout 0 → returns immediately; id still consumed
        ch.request("k", {}, timeout_s=0.05)

    ts = [threading.Thread(target=req) for _ in range(25)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for _k, payload in ch._hub.events:
        if payload and "id" in payload:
            ids.append(payload["id"])
    assert len(ids) == len(set(ids)) == 25


# ── EC-194: chat-store batch persistence ──────────────────────────────


def test_append_messages_single_save(tmp_path):
    store = ChatStore(root=tmp_path)
    conv = store.create()
    calls = []
    orig = store.save
    store.save = lambda: (calls.append(1), orig())
    msgs = [
        {"role": "assistant", "content": "a"},
        {"role": "tool", "name": "t", "content": "{}"},
        {"role": "assistant", "content": "b"},
    ]
    store.append_messages(conv["id"], msgs)
    assert calls == [1]
    got = store.get(conv["id"])
    assert [m["content"] for m in got["messages"]] == ["a", "{}", "b"]


# ── EC-207: log tail + status_file confinement ───────────────────────


def test_log_tail_seeks_and_confines(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "task_1"
    run_dir.mkdir(parents=True)
    status = run_dir / "status.json"
    status.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    log = run_dir / "log.txt"
    lines = [f"line-{i}" for i in range(3000)]
    log.write_text("\n".join(lines), encoding="utf-8")
    sr = StatusReader(search_dirs=[runs])
    tail = sr.read_log_tail(str(status), lines=10)
    assert tail == lines[-10:]
    # out-of-bounds status file → refused
    outside = tmp_path / "elsewhere" / "status.json"
    outside.parent.mkdir()
    outside.write_text("{}")
    assert sr.read_log_tail(str(outside), lines=5) == []


# ── EC-208: cheap task summaries + kill guard ─────────────────────────


def test_to_dict_no_lines_unless_tailed():
    t = TaskInfo(id="t1", name="n", command=["x"], started_at=time.time(), status="done", ended_at=time.time())
    t.lines = [f"l{i}" for i in range(5000)]
    d = t.to_dict()
    assert "lines" not in d and d["n_lines"] == 5000
    d = t.to_dict(tail=3)
    assert d["lines"] == ["l4997", "l4998", "l4999"]


def test_kill_does_not_clobber_finished_task(tmp_path):
    svc = ProcessService(_FakeHub())
    t = TaskInfo(id="t1", name="n", command=["x"], started_at=time.time() - 5, status="done", ended_at=time.time() - 1)
    svc.tasks["t1"] = t
    svc.kill("t1")
    assert svc.tasks["t1"].status == "done"


# ── EC-215: task id collision after restart ───────────────────────────


def test_task_ids_skip_stale_dirs(tmp_path, monkeypatch):
    import forge_gui_server.services.procs as procs_mod

    monkeypatch.setattr(procs_mod, "project_root", lambda: tmp_path)
    stale = tmp_path / "research" / "tasks" / "task_0001"
    stale.mkdir(parents=True)
    svc = ProcessService(_FakeHub())
    tid = svc.launch("t", ["definitely-not-a-real-exec-xyz"])
    assert tid != "task_0001"  # didn't reuse the stale dir
    assert (tmp_path / "research" / "tasks" / tid).is_dir()


# ── EC-216: hub cross-thread safety ──────────────────────────────────


def test_hub_publish_without_loop_keeps_recent():
    hub = EventHub()
    q = hub.subscribe()
    evt = hub.publish("engine", {"state": "ready"})
    assert evt["seq"] == 1
    # no running loop → not fanned out to a foreign-thread queue
    assert q.empty()
    q2 = hub.subscribe()  # replay sees it
    assert not q2.empty()


# ── EC-220: self-play status path caching ────────────────────────────


def test_status_path_cached_between_polls(tmp_path):
    ckpts = tmp_path / "checkpoints"
    sp = ckpts / "self_play"
    sp.mkdir(parents=True)
    (sp / "status.json").write_text(json.dumps({"status": "running", "step": 5}), encoding="utf-8")
    er = EventsReader(search_dirs=[ckpts])
    s1 = er.latest_status()
    assert s1 and s1["step"] == 5
    # Second poll must not re-discover (rglob) — poison the finder.
    er._find_status_file = lambda: pytest.fail("re-discovered")
    s2 = er.latest_status()
    assert s2["step"] == 5


# ── EC-190 / EC-206: cancel token + lora confine ─────────────────────


def test_cancel_tokens():
    from forge_gui_server.services.chat_loop import ChatCancel
    from forge_gui_server.services.gen import StreamCancel

    for tok in (ChatCancel(), StreamCancel()):
        assert not tok.cancelled
        tok.cancel()
        assert tok.cancelled


def test_lora_path_confinement():
    pytest.importorskip("forge_gui.api.lora_store")
    from forge_gui_server.services.lora import LoraService

    ck = Path(__file__).resolve().parents[2] / "research" / "checkpoints"
    with pytest.raises(ValueError):
        LoraService._confine("../../etc/passwd", [ck])
    with pytest.raises(ValueError):
        LoraService._confine("C:/Windows/x.safetensors", [ck])
