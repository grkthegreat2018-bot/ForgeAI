"""System routes: tasks, launch presets, logs, compute, lorebook, MCP,
approvals, timers, backups, master prompt."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from ..deps import services

logger = logging.getLogger(__name__)
router = APIRouter()


# ── tasks (process manager) ───────────────────────────────────────────

@router.get("/tasks")
async def tasks():
    return {"tasks": services.procs.all_tasks()}


@router.get("/tasks/{task_id}")
async def task_detail(task_id: str, tail: int = 500):
    info = services.procs.tasks.get(task_id)
    if info is None:
        return {"error": "not found"}
    return info.to_dict(tail=tail)


class LaunchRequest(BaseModel):
    # Arbitrary caller-supplied commands are disabled — launches must go
    # through a blessed preset (same as /launch/preset).
    preset_name: str = ""
    name: str = ""
    overrides: dict[str, str] = {}


def _preset_command(preset_name: str, overrides: dict[str, str]):
    """Resolve a preset name to an argv list, or an error string."""
    from forge_gui.api.process_manager import build_command, get_presets, preset_script_exists
    preset = next((p for p in get_presets()
                   if p.name == preset_name), None)
    if preset is None:
        return None, f"unknown preset: {preset_name}"
    if not preset_script_exists(preset):
        return None, f"preset script missing: {preset.script}"
    return build_command(preset, overrides), None


@router.post("/tasks/launch")
async def task_launch(body: LaunchRequest):
    cmd, err = _preset_command(body.preset_name, body.overrides)
    if err:
        return {"ok": False, "error": err}
    task_id = services.procs.launch(body.name or body.preset_name, cmd)
    return {"ok": True, "task_id": task_id}


@router.post("/tasks/{task_id}/kill")
async def task_kill(task_id: str):
    return {"ok": services.procs.kill(task_id)}


@router.delete("/tasks/{task_id}")
async def task_remove(task_id: str):
    return {"ok": services.procs.remove(task_id)}


@router.post("/tasks/clear_finished")
async def tasks_clear():
    removed = [tid for tid in list(services.procs.tasks)
               if services.procs.remove(tid)]
    return {"removed": len(removed)}


# ── launch presets ────────────────────────────────────────────────────

@router.get("/launch/presets")
async def launch_presets():
    from forge_gui.api.process_manager import get_presets
    return {"presets": [
        {"name": p.name, "script": p.script, "description": p.description,
         "arg_defaults": p.arg_defaults}
        for p in get_presets()]}


class PresetLaunchRequest(BaseModel):
    preset_name: str = ""
    overrides: dict[str, str] = {}


@router.post("/launch/preset")
async def launch_preset(body: PresetLaunchRequest):
    cmd, err = _preset_command(body.preset_name, body.overrides)
    if err:
        return {"ok": False, "error": err}
    task_id = services.procs.launch(body.preset_name, cmd)
    return {"ok": True, "task_id": task_id, "cmd": cmd}


# ── logs ──────────────────────────────────────────────────────────────

@router.get("/logs/sources")
async def log_sources():
    loop = asyncio.get_running_loop()
    sources = await loop.run_in_executor(
        None, services.log_tailer.discover)
    return {"sources": sources}


@router.get("/logs")
async def logs(query: str = "", levels: str = "",
               sources: str = "", limit: int = 3000):
    loop = asyncio.get_running_loop()

    def fetch():
        tailer = services.log_tailer
        tailer.poll()
        lv = {l.strip() for l in levels.split(",") if l.strip()} or None
        src = {s.strip() for s in sources.split(",") if s.strip()} or None
        return tailer.filtered(query or None, lv, src, limit)

    lines = await loop.run_in_executor(None, fetch)
    buf = services.log_tailer.buffer
    n_err = sum(1 for l in buf if getattr(l, "level", "") in
                ("ERROR", "FATAL", "CRITICAL"))
    n_warn = sum(1 for l in buf if getattr(l, "level", "") == "WARN")
    return {"lines": [l.fmt() if hasattr(l, "fmt") else str(l)
                      for l in lines],
            "total": len(buf), "errors": n_err, "warnings": n_warn,
            "sources": list(getattr(services.log_tailer, "sources", []))}


@router.post("/logs/clear")
async def logs_clear():
    services.log_tailer.clear()
    return {"ok": True}


# ── compute ───────────────────────────────────────────────────────────

@router.get("/compute")
async def compute():
    snap = services.gpu.snapshot()
    out = dict(snap)
    try:
        loop = asyncio.get_running_loop()

        def torch_info():
            import torch
            info = {"torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available()}
            if torch.cuda.is_available():
                info["device"] = torch.cuda.get_device_name(0)
                info["capability"] = ".".join(
                    str(x) for x in torch.cuda.get_device_capability(0))
                info["allocated_gb"] = round(
                    torch.cuda.memory_allocated() / 1e9, 3)
                info["reserved_gb"] = round(
                    torch.cuda.memory_reserved() / 1e9, 3)
                info["max_allocated_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1e9, 3)
            return info
        out["torch"] = await loop.run_in_executor(None, torch_info)
    except Exception as e:
        out["torch"] = {"error": str(e)}
    return out


# ── lorebook (memory) ─────────────────────────────────────────────────

@router.get("/lorebook")
async def lorebook_entries():
    lb = services.lorebook
    entries = lb.entries() if callable(lb.entries) else lb.entries
    return {"entries": [e.to_dict() for e in entries],
            "stats": lb.stats()}


@router.post("/lorebook")
async def lorebook_add(body: dict):
    e = services.lorebook.add(
        keys=body.get("keys", []), content=body.get("content", ""),
        category=body.get("category", "note"),
        priority=int(body.get("priority", 100)),
        constant=bool(body.get("constant", False)))
    return e.to_dict() if e else {"error": "add failed"}


@router.post("/lorebook/{entry_id}")
async def lorebook_update(entry_id: str, body: dict):
    e = services.lorebook.update(entry_id, **body)
    return e.to_dict() if e else {"error": "not found"}


@router.delete("/lorebook/{entry_id}")
async def lorebook_delete(entry_id: str):
    return {"ok": services.lorebook.delete(entry_id)}


# ── MCP servers ───────────────────────────────────────────────────────

@router.get("/mcp")
async def mcp_status():
    return {"servers": services.mcp.status()}


@router.post("/mcp/connect/{name}")
async def mcp_connect(name: str):
    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, services.mcp.connect, name)
    return {"ok": ok}


@router.post("/mcp/disconnect/{name}")
async def mcp_disconnect(name: str):
    loop = asyncio.get_running_loop()
    # client.close() blocks on proc.wait(timeout=5) — keep it off the loop
    await loop.run_in_executor(None, services.mcp.disconnect, name)
    return {"ok": True}


# ── approvals ─────────────────────────────────────────────────────────

@router.get("/approvals")
async def approvals_pending():
    return {"pending": services.approvals.pending()}


@router.post("/approvals/{req_id}")
async def approvals_respond(req_id: str, body: dict):
    return {"ok": services.approvals.respond(
        req_id, bool(body.get("granted")))}


# ── timers ────────────────────────────────────────────────────────────

@router.get("/timers")
async def timers():
    return {"timers": services.timers.list_timers(),
            "fired": services.timers.get_fired_timers()}


@router.post("/timers/{timer_id}/cancel")
async def timer_cancel(timer_id: str):
    return {"ok": services.timers.cancel_timer(timer_id)}


# ── backups ───────────────────────────────────────────────────────────

@router.get("/backups")
async def backups():
    return {"backups": services.backups.list_backups(),
            "active": services.backups.is_active}


@router.post("/backups")
async def backup_create():
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(
        None, services.backups.create_backup)
    return {"path": path}


@router.post("/backups/monitor")
async def backup_monitor(body: dict):
    if body.get("active"):
        services.backups.start()
    else:
        services.backups.stop()
    return {"active": services.backups.is_active}


# ── master prompt ─────────────────────────────────────────────────────

@router.get("/master_prompt")
async def master_prompt(config_name: str = ""):
    from forge_gui.api.master_prompt import get_default_prompt_for_config
    cfg = config_name or services.engine.info.get(
        "config_name", "forgelm_v2")
    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None, get_default_prompt_for_config, cfg)
        return {"prompt": text, "config": cfg}
    except Exception as e:
        return {"prompt": "", "error": str(e)}


# ── sub-agents ────────────────────────────────────────────────────────

@router.get("/sub_agents")
async def sub_agents():
    return {"tasks": services.sub_agents.list_tasks()}


@router.post("/sub_agents/clear")
async def sub_agents_clear():
    services.sub_agents.clear()
    return {"ok": True}
