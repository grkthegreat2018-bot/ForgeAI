"""Sub-agent runner — concurrent generation via ForgeEngine multi-gen.

ForgeEngine supports multiple concurrent generation calls with full
hotswappable model settings. This module provides a SubAgentManager
that can spawn parallel sub-agents for tasks like:
- Parallel code review (multiple files at once)
- Multi-perspective analysis (security, performance, style)
- Parallel test generation + documentation
- Research + implementation in parallel

Each sub-agent gets its own generation call with independent settings
(temperature, max_tokens, etc.) but shares the same resident engine.
The engine's concurrent gen control handles serialization.

Architecture:
- SubAgentManager: manages a pool of sub-agent generation requests
- spawn_sub_agent: tool that the main agent calls to delegate work
- Results are collected and returned to the main agent

The sub-agent system uses the ToolHarness for safety — all sub-agent
tool calls go through the same safety checker as the main agent.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from ._signal import SimpleSignal

logger = logging.getLogger(__name__)


@dataclass
class SubAgentTask:
    """A single sub-agent task."""
    task_id: str
    prompt: str
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    top_k: int = 80
    status: str = "pending"  # pending, running, done, error
    result: str = ""
    error: str = ""
    elapsed_s: float = 0.0
    started_at: float = 0.0


class SubAgentManager:
    """Manages concurrent sub-agent generation calls.

    Uses ForgeEngine's concurrent generation support. Each sub-agent
    gets an independent generation call with hotswappable settings.

    Signals (SimpleSignal):
        sub_agent_started(task_id): a sub-agent started generating
        sub_agent_done(task_id, result): a sub-agent finished
        sub_agent_error(task_id, error): a sub-agent failed
        all_done(): all pending sub-agents completed
    """

    def __init__(self, engine_runtime, max_concurrent: int = 3) -> None:
        self.sub_agent_started = SimpleSignal()
        self.sub_agent_done = SimpleSignal()
        self.sub_agent_error = SimpleSignal()
        self.all_done = SimpleSignal()
        self.engine_runtime = engine_runtime
        self.max_concurrent = max_concurrent
        self._tasks: dict[str, SubAgentTask] = {}
        self._futures: dict[str, Future] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"sub_{self._counter:03d}"

    def spawn(self, prompt: str, system_prompt: str = "",
              temperature: float = 0.7, max_tokens: int = 512,
              top_p: float = 0.95, top_k: int = 80) -> str:
        """Spawn a sub-agent task. Returns the task_id.

        The sub-agent runs concurrently with other sub-agents and the
        main agent. Results are collected via get_result() or the
        sub_agent_done signal.
        """
        task_id = self._next_id()
        task = SubAgentTask(
            task_id=task_id, prompt=prompt, system_prompt=system_prompt,
            temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, top_k=top_k, started_at=time.time())
        self._tasks[task_id] = task

        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent,
                thread_name_prefix="subagent")

        future = self._executor.submit(self._run_sub_agent, task)
        self._futures[task_id] = future
        self.sub_agent_started.emit(task_id)
        return task_id

    def spawn_batch(self, tasks: list[dict]) -> list[str]:
        """Spawn multiple sub-agents at once. Returns list of task_ids.

        Each task dict should have: prompt, system_prompt (optional),
        temperature (optional), max_tokens (optional), etc.
        """
        ids = []
        for t in tasks:
            tid = self.spawn(
                prompt=t.get("prompt", ""),
                system_prompt=t.get("system_prompt", ""),
                temperature=t.get("temperature", 0.7),
                max_tokens=t.get("max_tokens", 512),
                top_p=t.get("top_p", 0.95),
                top_k=t.get("top_k", 80))
            ids.append(tid)
        return ids

    def _run_sub_agent(self, task: SubAgentTask) -> None:
        """Run a single sub-agent generation (in a worker thread)."""
        task.status = "running"
        try:
            # build the message list
            messages = []
            if task.system_prompt:
                messages.append({"role": "system", "content": task.system_prompt})
            messages.append({"role": "user", "content": task.prompt})

            # render via qwen adapter (dispatch on config name)
            from forge.self_play.discovery.qwen_adapter import (
                render_messages_for_config,
            )
            cfg_name = self.engine_runtime.info.get("config_name", "")
            rendered = render_messages_for_config(
                messages, config_name=cfg_name, add_generation_prompt=True)

            # generate via the engine runtime
            parts: list[str] = []
            with self.engine_runtime.acquire(timeout_s=60.0) as engine:
                for tok in engine.generate_stream(
                        rendered, max_new_tokens=task.max_tokens,
                        temperature=task.temperature, top_p=task.top_p,
                        top_k=task.top_k):
                    parts.append(tok)

            task.result = "".join(parts)
            task.status = "done"
            task.elapsed_s = time.time() - task.started_at
            self.sub_agent_done.emit(task.task_id, task.result)
        except Exception as e:
            task.error = f"{type(e).__name__}: {e}"
            task.status = "error"
            task.elapsed_s = time.time() - task.started_at
            logger.warning("sub-agent %s failed: %s", task.task_id, task.error)
            self.sub_agent_error.emit(task.task_id, task.error)

    def get_result(self, task_id: str) -> SubAgentTask | None:
        """Get the status/result of a sub-agent task."""
        return self._tasks.get(task_id)

    def wait_all(self, timeout_s: float = 120) -> dict[str, SubAgentTask]:
        """Wait for all pending sub-agents to complete. Returns all tasks."""
        for _tid, fut in list(self._futures.items()):
            try:
                fut.result(timeout=timeout_s)
            except Exception:
                logger.debug("Error waiting for sub-agent future (already recorded in task)", exc_info=True)  # error already recorded in task
        self._futures.clear()
        self.all_done.emit()
        return dict(self._tasks)

    def list_tasks(self) -> list[dict]:
        """List all sub-agent tasks with their status."""
        return [
            {"task_id": t.task_id, "status": t.status,
             "elapsed_s": round(t.elapsed_s, 2),
             "result_preview": t.result[:200] if t.result else "",
             "error": t.error}
            for t in self._tasks.values()
        ]

    def clear(self) -> None:
        """Clear all completed tasks."""
        self._tasks = {k: v for k, v in self._tasks.items()
                       if v.status in ("pending", "running")}

    def shutdown(self) -> None:
        """Shutdown the executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


# ── sub-agent tool definitions ──────────────────────────────────────────

def sub_agent_tool_defs() -> list[dict]:
    """Tool definitions for sub-agent operations."""
    return [
        {
            "type": "function",
            "function": {
                "name": "spawn_sub_agent",
                "description": (
                    "Spawn a concurrent sub-agent to work on a task in "
                    "parallel. The sub-agent gets its own generation call "
                    "with independent settings. Use for parallel code "
                    "review, multi-perspective analysis, or delegating "
                    "subtasks. Returns a task_id."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The task prompt for the sub-agent.",
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "Optional system prompt for the sub-agent.",
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (default 0.7).",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Max tokens to generate (default 512).",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spawn_sub_agents",
                "description": (
                    "Spawn multiple sub-agents at once for parallel work. "
                    "Each task runs concurrently. Returns list of task_ids."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "prompt": {"type": "string"},
                                    "system_prompt": {"type": "string"},
                                    "temperature": {"type": "number"},
                                    "max_tokens": {"type": "integer"},
                                },
                            },
                            "description": "List of sub-agent tasks to spawn.",
                        },
                    },
                    "required": ["tasks"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_sub_agent",
                "description": (
                    "Check the status and result of a sub-agent task. "
                    "Returns the task status (pending/running/done/error) "
                    "and result if complete."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task_id from spawn_sub_agent.",
                        },
                    },
                    "required": ["task_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wait_sub_agents",
                "description": (
                    "Wait for all pending sub-agents to complete and "
                    "return all results. Blocks until all are done."),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_sub_agents",
                "description": (
                    "List all sub-agent tasks with their current status "
                    "and result previews."),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
