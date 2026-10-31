"""AgentService — asyncio port of forge_gui.api.agent_runner.AgentRunner.

The agent loop itself stays a blocking function executed in a worker
thread (engine calls are synchronous); Qt signals become hub events and
the approval threading.Event is resolved by a REST call. Logic is
line-faithful to the Qt version.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from forge_gui.api.agent_tools import tool_results_to_text

logger = logging.getLogger(__name__)

SIDE_EFFECT_TOOLS = frozenset({
    "write_file", "append_file", "delete_file",
    "run_python", "run_cmd",
    "search_replace", "create_file", "git_revert",
    "rename_file", "project_search_replace", "undo_edit",
    "git_branch", "git_stash", "run_tests",
})
DESTRUCTIVE_TOOLS = frozenset({
    "delete_file", "git_revert", "project_search_replace",
    "git_branch", "git_stash",
})
APPROVAL_NONE = "none"
APPROVAL_DESTRUCTIVE = "destructive"
APPROVAL_ALL = "all"

DEFAULT_SYSTEM = (
    "You are Forge Agent, an expert coding and research agent. You MUST use "
    "tools to complete the task — never just write explanations. Call a tool "
    "every turn until done, then give a short summary.\n"
    "For research tasks: start with web_search to find information online, "
    "then web_fetch to read full pages. Use news_search for current "
    "headlines, wikipedia_search for factual background and arxiv_search "
    "for academic papers.\n"
    "For coding tasks: start with list_dir and read_file to explore the "
    "workspace, then use write_file or search_replace to make changes.\n"
    "Format: <tool_call>\n{\"name\": \"tool_name\", \"arguments\": "
    "{\"arg\": \"value\"}}\n</tool_call>"
)

NO_TOOL_RETRIES = 2

# Safety ceiling for model-decided (auto) mode — the run ends when the
# model stops calling tools; this only guards against a true runaway.
AUTO_MAX_ROUNDS = 256

# Jamba token ids — generation must stop at </tool_call> or the start of
# a <tool_response> block; otherwise the model continues past its call and
# hallucinates the tool's output instead of waiting for the real result.
_TOOL_STOP_IDS = [2, 519, 532, 539]
_TOOL_CALL_START = "<tool_call>"
_RE_NAME_HINT = re.compile(r'\{"name"\s*:\s*"')

# Agent think budget — the render opens a <think> block every round, but
# an agent's job is action: past this many think tokens the loop injects
# a bare </think> so the model proceeds to the call/answer. Generous vs
# chat's 160 (agent steps legitimately reason more), still bounded.
_AGENT_THINK_BUDGET = 384
_AGENT_THINK_CLOSE = "\n</think>\n"


def _musing_before_first_call(raw: str) -> str:
    """Text the model produced before its first tool call.

    Anything after the call is a hallucinated continuation (a simulated
    tool response) — never persist or display it.
    """
    cut = raw.find(_TOOL_CALL_START)
    if cut < 0:
        m = _RE_NAME_HINT.search(raw)
        cut = m.start() if m else len(raw)
    return raw[:cut]


_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent")


class AgentRun:
    def __init__(self, run_id: str, runtime, task: str, workspace: str,
                 system_prompt: str = DEFAULT_SYSTEM,
                 max_rounds: int | None = None, max_new_tokens: int = 2048,
                 temperature: float = 0.2, top_p: float = 0.95,
                 top_k: int = 80, repetition_penalty: float = 1.05,
                 enabled_tools: list[str] | None = None,
                 approval_mode: str = APPROVAL_DESTRUCTIVE,
                 history: list[dict] | None = None,
                 project: str = "",
                 tool_harness=None) -> None:
        self.run_id = run_id
        self._runtime = runtime
        self.task = task
        self.workspace = workspace
        self.project = project
        self.system_prompt = system_prompt
        # None or <=0 → auto: the model decides when the task is done
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.enabled_tools = enabled_tools
        self.approval_mode = approval_mode
        self.history = list(history or [])
        self.tool_harness = tool_harness
        self._cancel = False
        self._approval = threading.Event()
        self._approval_granted = False
        self._steer: queue.Queue[str] = queue.Queue()
        self.events: list[dict] = []
        self._evt_seq = 0
        self.rounds_done = 0
        self.status = "running"
        self.result: dict | None = None
        self.error = ""
        self.started_at = time.time()

    def cancel(self) -> None:
        self._cancel = True
        self._approval_granted = False
        self._approval.set()

    def respond_approval(self, granted: bool) -> None:
        self._approval_granted = granted
        self._approval.set()

    def steer(self, message: str) -> None:
        """Queue a user message injected between rounds mid-run."""
        self._steer.put(message)

    def info(self) -> dict:
        return {"run_id": self.run_id, "task": self.task,
                "status": self.status, "started_at": self.started_at,
                "error": self.error, "project": self.project,
                "workspace": self.workspace,
                "rounds": self.rounds_done,
                "n_events": self._evt_seq}


class AgentService:
    """Manages agent runs; emits every trace event on the hub."""

    def __init__(self, hub, runtime) -> None:
        self._hub = hub
        self._runtime = runtime
        self._ids = itertools.count(1)
        self.runs: dict[str, AgentRun] = {}
        self._harness_factory = None  # callable() -> ToolHarness
        self._aio_loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._aio_loop = loop

    def set_harness_factory(self, factory) -> None:
        self._harness_factory = factory

    def start(self, task: str, workspace: str, **kw) -> AgentRun:
        run_id = f"agent_{next(self._ids):04d}"
        if self._harness_factory and "tool_harness" not in kw:
            try:
                kw["tool_harness"] = self._harness_factory(workspace)
            except Exception as e:
                logger.warning("harness factory failed, using default: %s", e)
        run = AgentRun(run_id, self._runtime, task, workspace, **kw)
        self.runs[run_id] = run
        run.events.append({"run_id": run_id, "kind": "created",
                           "round": None, "seq": 0, "data": None})
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._aio_loop
        if loop is not None:
            loop.run_in_executor(_pool, self._drive, run)
        else:
            _pool.submit(self._drive, run)
        self._emit(run, "started", {"task": task, "workspace": workspace,
                                    "project": run.project})
        return run

    def _emit(self, run: AgentRun, kind: str, data: Any = None,
              round_idx: int | None = None) -> None:
        run._evt_seq += 1
        payload = {"run_id": run.run_id, "kind": kind, "seq": run._evt_seq,
                   "round": round_idx, "data": data}
        run.events.append(payload)
        self._hub.publish("agent_event", payload)

    def _drive(self, run: AgentRun) -> None:
        t0 = time.perf_counter()
        try:
            result = self._loop(run)
            result["elapsed_s"] = round(time.perf_counter() - t0, 2)
            run.result = result
            run.status = "cancelled" if run._cancel else "done"
            self._emit(run, "finished", result)
        except Exception as e:
            logger.warning("agent loop failed: %s", e, exc_info=True)
            run.error = f"{type(e).__name__}: {e}"
            run.status = "failed"
            self._emit(run, "failed", run.error)

    def _loop(self, run: AgentRun) -> dict:
        from forge.self_play.discovery.qwen_adapter import (  # type: ignore
            TOOL_CALL_END,
            TOOL_CALL_START,
            qwen_parse_tool_calls,
            render_messages_for_config,
        )
        from forge_gui.api.tool_harness import ToolHarness
        from forge_gui.api.web_tools import WebTools
        harness = run.tool_harness or ToolHarness(
            run.workspace, enable_safety=True,
            web_tools=WebTools(enabled=True))
        defs = harness.tool_defs()
        if run.enabled_tools is not None:
            allow = set(run.enabled_tools)
            defs = [d for d in defs if d["function"]["name"] in allow]

        messages: list[dict] = list(run.history)
        messages.append({"role": "system", "content": run.system_prompt})
        messages.append({"role": "user", "content": run.task})

        all_calls: list[dict] = []
        content = ""
        rounds = 0
        # max_rounds None/<=0 → the model decides when it is done (a round
        # with no tool calls ends the run); a high ceiling only guards
        # against a true runaway. Fixed mode keeps the no-tool retry.
        auto = run.max_rounds is None or run.max_rounds <= 0
        cap = AUTO_MAX_ROUNDS if auto else run.max_rounds

        round_idx = 0
        while round_idx < cap:
            if run._cancel:
                break
            rounds = round_idx + 1
            run.rounds_done = rounds
            # drain user steering messages queued mid-run before the
            # next prompt is rendered so the model sees them immediately
            while True:
                try:
                    steer_msg = run._steer.get_nowait()
                except queue.Empty:
                    break
                messages.append({"role": "user", "content": steer_msg})
            self._emit(run, "round_started", None, round_idx)

            cfg_name = self._runtime.info.get("config_name", "")
            rendered = render_messages_for_config(
                messages, config_name=cfg_name,
                tools=defs, add_generation_prompt=True)
            self._emit(run, "prompt_rendered", rendered, round_idx)

            raw = self._generate(run, rendered)
            if run._cancel:
                break

            tool_calls, content = qwen_parse_tool_calls(raw)
            if tool_calls:
                content = _musing_before_first_call(raw)
            # The prompt opens <think> every round — split reasoning out
            # of the visible text and keep it on the message so the
            # template hands it back next round (interleaved thinking).
            # Fallback: an EOS inside the block leaves body empty — for
            # a final (no-call) round the reasoning IS the deliverable,
            # surface it rather than returning "".
            from .chat_loop import _split_reasoning
            reasoning, content = _split_reasoning(content or "")
            if not content and reasoning and not tool_calls:
                content, reasoning = reasoning, ""
            self._emit(run, "raw_output", raw, round_idx)
            self._emit(run, "text", content or "", round_idx)
            messages.append({"role": "assistant", "content": content or "",
                             "reasoning_content": reasoning or None,
                             "tool_calls": tool_calls or None})

            if not tool_calls:
                if auto:
                    break  # model stopped calling tools — task is done
                retried = False
                for _retry in range(NO_TOOL_RETRIES):
                    if run._cancel:
                        break
                    err_msg = (
                        f"No tool call found in your response. You MUST use "
                        f"tools to complete the task. To call a tool, output: "
                        f"{TOOL_CALL_START}\n{{\"name\": \"tool_name\", "
                        f"\"arguments\": {{...}}}}\n{TOOL_CALL_END}\n"
                        f"Available tools: {', '.join(d['function']['name'] for d in defs)}. "
                        f"Try again — start by calling list_dir or read_file "
                        f"to explore the workspace."
                    )
                    messages.append({"role": "tool", "name": "system",
                                     "content": err_msg})
                    rendered = render_messages_for_config(
                        messages, config_name=cfg_name,
                        tools=defs, add_generation_prompt=True)
                    self._emit(run, "prompt_rendered", rendered, round_idx)
                    raw = self._generate(run, rendered)
                    if run._cancel:
                        break
                    tool_calls, content = qwen_parse_tool_calls(raw)
                    if tool_calls:
                        content = _musing_before_first_call(raw)
                    reasoning, content = _split_reasoning(content or "")
                    if not content and reasoning and not tool_calls:
                        content, reasoning = reasoning, ""
                    self._emit(run, "raw_output", raw, round_idx)
                    self._emit(run, "text", content or "", round_idx)
                    messages.append({"role": "assistant",
                                     "content": content or "",
                                     "reasoning_content": reasoning or None,
                                     "tool_calls": tool_calls or None})
                    if tool_calls:
                        retried = True
                        break
                if not retried:
                    break

            for tc in tool_calls:
                if run._cancel:
                    break
                call = tc if isinstance(tc, dict) else {"name": str(tc)}
                self._emit(run, "tool_call", call, round_idx)
                if not self._may_execute(run, call):
                    results = [{"ok": False,
                                "result": {"error": "blocked by policy"}}]
                else:
                    if self._needs_approval(run, call):
                        run._approval.clear()
                        self._emit(run, "approval_requested", call, round_idx)
                        run._approval.wait()
                        if run._cancel or not run._approval_granted:
                            results = [{"ok": False,
                                        "result": {"error": "denied by user"}}]
                            rec = results[0]
                            self._emit(run, "tool_result", rec, round_idx)
                            messages.append({
                                "role": "tool",
                                "name": call.get("name", "tool"),
                                "content": tool_results_to_text(rec),
                            })
                            continue
                    results = harness.execute_calls([call])
                rec = results[0]
                all_calls.append(call)
                self._emit(run, "tool_result", rec, round_idx)
                messages.append({
                    "role": "tool",
                    "name": call.get("name", "tool"),
                    "content": tool_results_to_text(rec),
                })
                if rec.get("result", {}).get("terminated"):
                    run._cancel = True
                    break
            round_idx += 1

        return {
            "content": content or "",
            "rounds": rounds,
            "tool_calls": all_calls,
            "sandbox": harness.summary(),
            "cancelled": run._cancel,
            "safety": harness.strikes.summary(),
        }

    def _generate(self, run: AgentRun, rendered: str) -> str:
        from .chat_loop import _think_cap_processor
        with self._runtime.acquire() as engine:
            gen_raw = getattr(engine, "generate_raw", None)
            if gen_raw is not None:
                # The render opens <think> every round: mask <tool_call>
                # while it's open (trained order is think → </think> →
                # text → call; a mid-think call leaves reasoning leaking
                # into the visible text) and cap the think with a bare
                # </think> injection — not the chat "Answer:" suffix, so
                # a capped think can still proceed to a tool call.
                cap = _think_cap_processor(
                    _AGENT_THINK_BUDGET,
                    engine.tokenizer.encode(
                        _AGENT_THINK_CLOSE, add_special_tokens=False),
                    tool_calls_allowed=True)
                return gen_raw(
                    rendered, max_new_tokens=run.max_new_tokens,
                    temperature=run.temperature, top_p=run.top_p,
                    top_k=run.top_k,
                    repetition_penalty=run.repetition_penalty,
                    skip_special_tokens=False,
                    logits_processor=cap,
                    eos_token_ids=_TOOL_STOP_IDS)
            return engine.generate(
                rendered, max_new_tokens=run.max_new_tokens,
                temperature=run.temperature, top_p=run.top_p,
                top_k=run.top_k,
                repetition_penalty=run.repetition_penalty,
                skip_special_tokens=False,
                stop=["</tool_call>", "<tool_response>"])

    def _may_execute(self, run: AgentRun, call: dict) -> bool:
        if run.enabled_tools is None:
            return True
        return call.get("name") in set(run.enabled_tools)

    def _needs_approval(self, run: AgentRun, call: dict) -> bool:
        name = call.get("name", "")
        if run.approval_mode == APPROVAL_NONE:
            return False
        if run.approval_mode == APPROVAL_ALL:
            return name in SIDE_EFFECT_TOOLS
        return name in DESTRUCTIVE_TOOLS

    # ── management ────────────────────────────────────────────────────
    def get(self, run_id: str) -> AgentRun | None:
        return self.runs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if run:
            run.cancel()
            return True
        return False

    def delete(self, run_id: str) -> bool:
        """Remove a run — cancels it first if still running."""
        run = self.runs.pop(run_id, None)
        if run is None:
            return False
        if run.status == "running":
            run.cancel()
            run.status = "cancelled"
        return True

    def respond(self, run_id: str, granted: bool) -> bool:
        run = self.runs.get(run_id)
        if run:
            run.respond_approval(granted)
            return True
        return False

    def steer(self, run_id: str, message: str) -> bool:
        run = self.runs.get(run_id)
        if run and run.status == "running":
            run.steer(message)
            self._emit(run, "user_message", message)
            return True
        return False

    def detail(self, run_id: str) -> dict | None:
        run = self.runs.get(run_id)
        if run is None:
            return None
        return {"run": run.info(), "events": list(run.events)}

    def list_runs(self) -> list[dict]:
        return [r.info() for r in sorted(
            self.runs.values(), key=lambda r: -r.started_at)]


agent_service: AgentService | None = None
