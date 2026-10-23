"""Chat, agent, and generation routes — CRUD + SSE streaming."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..deps import services
from ..services import chat_loop, gen

logger = logging.getLogger(__name__)
router = APIRouter()

# Approval modes a remote caller may request for /agent/runs. "none" runs
# every side-effecting tool without asking — gated behind an explicit
# opt-in env flag so a casual API caller can't disable the gate.
_APPROVAL_MODES = {"destructive", "all"}
_NO_APPROVAL_ENV = "FORGE_AGENT_ALLOW_NO_APPROVAL"


def _allowed_approval_modes() -> set[str]:
    modes = set(_APPROVAL_MODES)
    if os.environ.get(_NO_APPROVAL_ENV, "").strip().lower() in (
            "1", "true", "yes"):
        modes.add("none")
    return modes


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


# ── conversations CRUD ────────────────────────────────────────────────

@router.get("/chats")
async def list_chats():
    store = services.chat_store
    out = []
    for c in store.conversations:
        good, bad = store.count_ratings(c)
        out.append({"id": c["id"], "title": c["title"],
                    "model": c.get("model", ""),
                    "created_at": c.get("created_at"),
                    "updated_at": c.get("updated_at"),
                    "n_messages": len(c.get("messages", [])),
                    "good": good, "bad": bad})
    return {"conversations": out}


@router.post("/chats")
async def create_chat(body: dict):
    conv = services.chat_store.create(
        title=body.get("title", "New chat"),
        model=body.get("model", ""))
    return conv


@router.get("/chats/{conv_id}")
async def get_chat(conv_id: str):
    conv = services.chat_store.get(conv_id)
    if conv is None:
        return {"error": "not found"}
    return conv


@router.delete("/chats/{conv_id}")
async def delete_chat(conv_id: str):
    return {"ok": services.chat_store.delete(conv_id)}


@router.post("/chats/{conv_id}/rename")
async def rename_chat(conv_id: str, body: dict):
    services.chat_store.rename(conv_id, body.get("title", ""))
    return {"ok": True}


@router.post("/chats/{conv_id}/truncate")
async def truncate_chat(conv_id: str, body: dict):
    """Drop messages[index:] — the UI uses this for regenerate/edit."""
    conv = services.chat_store.truncate_messages(
        conv_id, int(body.get("index", 0)))
    if conv is None:
        return {"error": "not found"}
    return conv


@router.post("/chats/{conv_id}/rate")
async def rate_message(conv_id: str, body: dict):
    new = services.chat_store.rate_message(
        conv_id, int(body.get("msg_idx", -1)), body.get("rating"))
    return {"rating": new}


@router.post("/chats/export")
async def export_rated(body: dict | None = None):
    loop = asyncio.get_running_loop()
    path, n = await loop.run_in_executor(
        None, services.chat_store.export_training_data)
    return {"path": str(path), "examples": n}


@router.get("/chats/exports")
async def list_exports():
    return {"exports": services.chat_store.list_exports()}


class ChatImportRequest(BaseModel):
    text: str = ""
    title: str = ""


@router.post("/chats/import")
async def import_chat(body: ChatImportRequest):
    """Import a pasted transcript (JSON / ChatML / role-marked / plain
    alternating paragraphs) as a new conversation."""
    from forge_gui.api.chat_store import parse_transcript
    msgs = parse_transcript(body.text)
    if not msgs:
        return {"error": "no chat turns detected — paste ChatML, a JSON "
                "message list, or 'User:'/'Assistant:'-marked text"}
    conv = services.chat_store.create()
    for m in msgs:
        services.chat_store.append_message(
            conv["id"], m["role"], m["content"])
    title = body.title.strip() or next(
        (m["content"].strip().splitlines()[0][:60]
         for m in msgs if m["role"] == "user" and m["content"].strip()),
        "Imported chat")
    services.chat_store.rename(conv["id"], title)
    return services.chat_store.get(conv["id"])


# ── chat send (SSE) ───────────────────────────────────────────────────

class ChatSendRequest(BaseModel):
    conv_id: str = ""
    message: str = ""
    system_prompt: str = ""
    use_master: bool = True
    model: str = ""
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 80
    repetition_penalty: float = 1.05
    tools_enabled: bool = True
    thinking: bool = True
    think_budget: int | None = None
    min_p: float = 0.0
    dry_multiplier: float = 0.0
    dry_base: float = 1.75


@router.post("/chat/send")
async def chat_send(body: ChatSendRequest):
    store = services.chat_store
    conv_id = body.conv_id
    if conv_id:
        conv = store.get(conv_id)
    else:
        conv = None
    if conv is None:
        conv = store.create(model=body.model)
        conv_id = conv["id"]
    store.append_message(conv_id, "user", body.message)
    store.touch(conv_id, body.model)

    # history = system + lorebook-injected messages (mirrors Qt chat page)
    history: list[dict] = []
    sys_prompt = body.system_prompt
    if body.use_master and not sys_prompt:
        from forge_gui.api.master_prompt import (
            get_default_prompt_for_config)
        cfg = services.engine.info.get("config_name", "forgelm_v2")
        try:
            sys_prompt = get_default_prompt_for_config(
                cfg, tools_enabled=body.tools_enabled,
                thinking_enabled=body.thinking)
        except Exception:
            sys_prompt = ""
    prior = [{"role": m["role"], "content": m.get("content", ""),
              "tool_calls": m.get("tool_calls"), "name": m.get("name", ""),
              "reasoning_content": m.get("reasoning_content")}
             for m in conv["messages"]]
    try:
        # inject() returns the memory TEXT (str) — fold it into the system
        # prompt; the old code discarded it by extending a str as messages.
        injected = services.lorebook.inject(prior)
        if isinstance(injected, str) and injected:
            sys_prompt = (f"{sys_prompt}\n{injected}"
                          if sys_prompt else injected)
    except Exception:
        pass
    if sys_prompt:
        history.append({"role": "system", "content": sys_prompt})
    history.extend(prior)

    harness = services.make_harness(read_only=True) \
        if body.tools_enabled else None
    cfg_name = services.engine.info.get("config_name", "")
    cancel = chat_loop.ChatCancel()

    async def event_stream():
        yield _sse({"type": "conv", "data": {"id": conv_id}})
        collected: list[dict] = []
        try:
            async for evt in chat_loop.run_chat(
                    services.engine, harness, history,
                    config_name=cfg_name,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature, top_p=body.top_p,
                    top_k=body.top_k,
                    repetition_penalty=body.repetition_penalty,
                    tools_enabled=body.tools_enabled,
                    thinking=body.thinking,
                    think_budget=body.think_budget,
                    min_p=body.min_p,
                    dry_multiplier=body.dry_multiplier,
                    dry_base=body.dry_base,
                    cancel=cancel):
                if evt["type"] == "done":
                    collected = evt["data"].get("messages", [])
                yield _sse(evt)
        finally:
            # Client disconnected (or stream ended) — stop generation and
            # any in-flight tool execution instead of orphaning the worker.
            cancel.cancel()
        # persist produced messages (one save, not N+1 rewrites)
        if collected:
            try:
                store.append_messages(conv_id, collected)
            except Exception:
                logger.debug("persist messages failed", exc_info=True)
        yield _sse({"type": "saved", "data": {"id": conv_id}})

    return StreamingResponse(event_stream(),
                             media_type="text/event-stream")


# ── agent runs ────────────────────────────────────────────────────────

def _project_slug(text: str, max_len: int = 40) -> str:
    import re as _re
    slug = _re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:max_len].strip("-") or "project"


def _agent_workspace(body: "AgentStartRequest") -> tuple[str, str]:
    """Every agent project lives in its own subfolder of ForgeAI_Projects.

    Returns (workspace_path, project_name). An explicit workspace path
    still wins (escape hatch); otherwise the project name — or a slug of
    the task — becomes the subfolder, created if missing.
    """
    if body.workspace:
        # An explicit workspace must still live under the project root —
        # an arbitrary path would jail the agent's file/exec tools to
        # whatever directory the remote caller picked.
        from forge_gui.api.status_reader import project_root
        root = Path(project_root()).resolve()
        ws = Path(body.workspace).resolve()
        if not (ws == root or root in ws.parents):
            raise ValueError(
                f"workspace outside project root: {body.workspace}")
        ws.mkdir(parents=True, exist_ok=True)
        return str(ws), body.project or ws.name
    from forge_gui.api.status_reader import project_root
    project = _project_slug(body.project or body.task)
    root = Path(project_root()) / "ForgeAI_Projects"
    ws = root / project
    ws.mkdir(parents=True, exist_ok=True)
    return str(ws), project


class AgentStartRequest(BaseModel):
    task: str = ""
    workspace: str = ""
    project: str = ""
    system_prompt: str = ""
    max_rounds: int | None = None  # None → model decides when done
    max_new_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 80
    repetition_penalty: float = 1.05
    enabled_tools: list[str] | None = None
    approval_mode: str = "destructive"
    history: list[dict] | None = None


@router.post("/agent/runs")
async def agent_start(body: AgentStartRequest):
    if not services.engine.is_ready():
        return {"ok": False, "error": "engine not loaded — load a model first"}
    mode = (body.approval_mode or "destructive").lower()
    if mode not in _allowed_approval_modes():
        return {"ok": False, "error":
                f"approval_mode '{body.approval_mode}' not allowed "
                f"(allowed: {sorted(_allowed_approval_modes())})"}
    try:
        ws, project = _agent_workspace(body)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    kw = {}
    if body.system_prompt:
        kw["system_prompt"] = body.system_prompt
    run = services.agent.start(
        task=body.task, workspace=ws, project=project,
        max_rounds=body.max_rounds, max_new_tokens=body.max_new_tokens,
        temperature=body.temperature, top_p=body.top_p, top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
        enabled_tools=body.enabled_tools,
        approval_mode=mode,
        history=body.history, **kw)
    return {"ok": True, "run_id": run.run_id}


@router.get("/agent/runs")
async def agent_list():
    return {"runs": services.agent.list_runs()}


@router.post("/agent/runs/{run_id}/cancel")
async def agent_cancel(run_id: str):
    return {"ok": services.agent.cancel(run_id)}


@router.delete("/agent/runs/{run_id}")
async def agent_delete(run_id: str):
    return {"ok": services.agent.delete(run_id)}


@router.post("/agent/runs/{run_id}/respond")
async def agent_respond(run_id: str, body: dict):
    return {"ok": services.agent.respond(
        run_id, bool(body.get("granted")))}


@router.get("/agent/runs/{run_id}")
async def agent_detail(run_id: str):
    """Full run state + persisted event history — lets the UI rehydrate
    a run's chat view after navigating away or reconnecting."""
    d = services.agent.detail(run_id)
    if d is None:
        return {"ok": False, "error": "unknown run"}
    return {"ok": True, **d}


@router.post("/agent/runs/{run_id}/message")
async def agent_message(run_id: str, body: dict):
    """Steer a running agent — the message is injected as a user turn
    before the next round renders."""
    msg = str(body.get("message") or "").strip()
    if not msg:
        return {"ok": False, "error": "empty message"}
    return {"ok": services.agent.steer(run_id, msg)}


@router.get("/agent/tools")
async def agent_tools():
    """Tool defs + category metadata for the agent config panel."""
    from forge_gui.api.tool_harness import ToolHarness
    harness = services.make_harness()
    defs = harness.tool_defs()
    return {"tools": [{"name": d["function"]["name"],
                       "description": d["function"].get("description", "")}
                      for d in defs]}


# ── generations ───────────────────────────────────────────────────────

class GenStreamRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95


@router.post("/gen/stream")
async def gen_stream(body: GenStreamRequest):
    cancel = gen.StreamCancel()

    async def event_stream():
        try:
            async for evt in gen.stream_tokens(
                    services.engine, body.prompt,
                    max_new_tokens=body.max_new_tokens,
                    temperature=body.temperature, top_k=body.top_k,
                    top_p=body.top_p, cancel=cancel):
                yield _sse(evt)
        finally:
            cancel.cancel()
    return StreamingResponse(event_stream(),
                             media_type="text/event-stream")


@router.post("/gen/adaptive")
async def gen_adaptive(body: dict):
    return await gen.run_adaptive(
        services.engine, body.get("prompt", ""),
        think_max_tokens=body.get("think_max_tokens", 512),
        no_think_max_tokens=body.get("no_think_max_tokens", 256),
        temperature=body.get("temperature", 0.0),
        top_p=body.get("top_p", 1.0), top_k=body.get("top_k", 80))


@router.post("/gen/batch")
async def gen_batch(body: dict):
    return await gen.run_batch(
        services.engine, body.get("prompts", []),
        max_new_tokens=body.get("max_new_tokens", 256),
        temperature=body.get("temperature", 0.0),
        top_p=body.get("top_p", 1.0), top_k=body.get("top_k", 80))


@router.post("/gen/raw")
async def gen_raw(body: dict):
    return await gen.run_raw(
        services.engine, body.get("prompt", ""),
        max_new_tokens=body.get("max_new_tokens", 256),
        temperature=body.get("temperature", 0.2),
        top_p=body.get("top_p", 1.0), top_k=body.get("top_k", 80),
        repetition_penalty=body.get("repetition_penalty", 1.05),
        min_p=body.get("min_p", 0.0), min_k=body.get("min_k", 0.0),
        skip_special_tokens=body.get("skip_special_tokens", False))
