"""Agentic distillation client — teacher models call tools to generate rich training data.

⚠️  TRAINING DATA GENERATION ONLY — uses discovery_tools.py for the self-play
    tool set. Not used during inference. For inference tool registry, see
    forge/engine/engine_tools.py.

Takes the self-play loop process from `tool_use_loop.py` and applies it to the
distillation model router. Teacher API models (gpt-oss, DeepSeek, Qwen, GLM, etc.)
act as the "engine" in an agentic tool-use loop:

  1. Teacher receives a task + tool definitions (OpenAI function-calling format)
  2. Teacher emits tool calls → we execute them (run_script, web_search, think, etc.)
  3. Results injected back into conversation → teacher continues
  4. Loop until teacher gives final answer or max_turns reached
  5. Full trajectory collected as SFT training data (messages + tool_calls + reward)

Fine-tuning is DISABLED — this is pure data collection. The collected trajectories
can later be used for SFT or GRPO training of the local ForgeLM model.

## Task Generation

Teachers can also generate their own tasks (like GoalGenerator in infinite_tool_loop):
  - Teacher model proposes coding tasks with test cases
  - Tasks are filtered for quality (non-filler, requires tool use, has verifiable output)
  - Filtered tasks are added to the task pool for agentic execution

## Usage

    from forge.distillation.agentic_distill import AgenticDistillClient

    client = AgenticDistillClient()
    # Run agentic tasks (teachers call tools)
    trajectories = client.run_agentic_batch(tasks, n_samples_per_task=3)
    # Let teachers generate their own tasks
    new_tasks = client.generate_tasks(n_tasks=20)
    # Save trajectories as SFT data
    client.save_trajectories(trajectories, "agentic_distill_data.jsonl")
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.distillation.distill_client import (
    DistillationClient,
    DistillModel,
)
from forge.self_play.discovery.discovery_db import DiscoveryDB
from forge.self_play.discovery.discovery_tools import ToolRegistry

# compute_reward was in tool_use_loop.py, which was removed in the AZR loop
# merge. The reward-scoring path is legacy — keep the module importable (other
# code imports helpers without scoring), but fail clearly at the call site.
try:
    from forge.self_play.discovery.tool_use_loop import compute_reward
except ImportError:
    compute_reward = None  # type: ignore


class _TrajectoryReward:
    """Minimal stand-in for the removed tool_use_loop.compute_reward."""
    __slots__ = ("total", "parts")

    def __init__(self, total: float, parts: dict):
        self.total = total
        self.parts = parts

    def to_dict(self) -> dict:
        return dict(self.parts)


def _local_compute_reward(task: str, tool_calls: list, final_answer,
                          stopped_after_tools: bool,
                          stopped_after_answer: bool,
                          seen_outputs: set) -> _TrajectoryReward:
    """Heuristic trajectory reward (0..1): answered (0.30), productive
    tool use (0.25), clean stop (0.20), answer substance (0.15), novel
    vs seen outputs (0.10). The verified/oracle gate still applies
    downstream at export time."""
    parts = {}
    parts["answered"] = 0.30 if final_answer and str(final_answer).strip() \
        else 0.0
    n_tools = len(tool_calls or [])
    parts["tool_use"] = 0.25 if n_tools >= 1 else 0.0
    if stopped_after_tools and not stopped_after_answer:
        parts["tool_use"] = 0.15
    parts["clean_stop"] = 0.20 if (stopped_after_answer or
                                   (final_answer and not stopped_after_tools)) \
        else 0.0
    alen = len(str(final_answer or ""))
    parts["answer_substance"] = min(0.15, alen / 2000.0) if alen >= 40 else 0.0
    novel = final_answer and final_answer[:200] not in (seen_outputs or set())
    parts["novelty"] = 0.10 if novel else 0.0
    return _TrajectoryReward(min(1.0, sum(parts.values())), parts)


if compute_reward is None:
    compute_reward = _local_compute_reward  # type: ignore


# ── Tool schema conversion ───────────────────────────────────────────────

def _schemas_to_openai_tools(schemas: list[dict]) -> list[dict]:
    """Convert ToolRegistry schemas to OpenAI function-calling format.

    ToolRegistry schema: {"name": ..., "description": ..., "parameters": {...}}
    OpenAI format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
    """
    tools = []
    for s in schemas:
        params = s.get("parameters", {})
        # Ensure parameters is a valid JSON schema dict
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        if "type" not in params:
            params["type"] = "object"
        if "properties" not in params:
            params["properties"] = {}
        tools.append({
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": params,
            },
        })
    return tools


# ── Trajectory dataclass ─────────────────────────────────────────────────

@dataclass
class AgenticTrajectory:
    """A single agentic tool-use trajectory from a teacher model."""
    task: str
    teacher_model: str
    messages: list[dict]           # full conversation including tool calls/results
    tool_calls: list[dict]         # [{name, args, result, success}, ...]
    final_answer: str | None
    reward: float                  # computed reward (0..1)
    reward_breakdown: dict         # detailed reward components
    n_turns: int
    stopped_after_tools: bool
    stopped_after_answer: bool
    latency_ms: float
    tokens_in: int
    tokens_out: int
    error: str = ""

    def to_sft_dict(self) -> dict:
        """Convert to SFT training format."""
        return {
            "task": self.task,
            "teacher_model": self.teacher_model,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "final_answer": self.final_answer,
            "reward": self.reward,
            "reward_breakdown": self.reward_breakdown,
            "n_turns": self.n_turns,
            "tokens_out": self.tokens_out,
        }


# ── Task generation ──────────────────────────────────────────────────────

# Filler patterns for task filtering (same as infinite_tool_loop)
_FILLER_PATTERNS = [
    "hello", "test", "hi ", "what is your name", "say ",
    "tell me a joke", "what can you do", "introduce yourself",
    "write hello world", "print hello", "count to ten",
    "what is 1+1", "what is 2+2", "simple", "basic test",
    "just ", "anything", "something", "whatever",
]

_TOOL_KEYWORDS = [
    "search", "research", "script", "code", "build", "implement",
    "investigate", "analyze", "compare", "benchmark", "design",
    "experiment", "debug", "create", "explore", "study",
    "write a", "run a", "test a", "find", "calculate",
    "summarize", "propose", "record", "think about",
    "hypothesis", "theory", "discovery", "function", "algorithm",
    "parse", "sort", "reverse", "check", "convert", "generate",
    "implement", "solve", "compute", "encrypt", "decode",
]


def _is_filler_task(task: str) -> bool:
    """Return True if task is too easy/generic/filler."""
    t = task.lower().strip()
    if len(t) < 20:
        return True
    if t.startswith("{") or t.startswith("["):
        return True
    for pat in _FILLER_PATTERNS:
        if pat in t:
            return True
    has_keyword = any(kw in t for kw in _TOOL_KEYWORDS)
    return bool(not has_keyword)


# System prompt for task generation
_TASK_GEN_SYSTEM = """\
You are a task generator for an AI training pipeline. Generate coding tasks that:
1. Require writing a Python function or script
2. Have verifiable output (can be tested with specific inputs/outputs)
3. Range from easy to hard difficulty
4. Are specific and well-defined (not vague)

Output format — one task per line, JSON object:
{"task": "Write a Python function ...", "test_cases": [{"input": "...", "output": "..."}]}

Generate diverse tasks: string manipulation, math, data structures, algorithms, \
file I/O, error handling, recursion, dynamic programming, etc."""


# System prompt for agentic tool use
_AGENTIC_SYSTEM = """\
You are an expert Python programmer with access to tools. Use tools proactively to:
- Verify your solutions (run_script to test code)
- Research unknowns (web_search, wikipedia_search)
- Record your reasoning (think)
- Calculate values (calculate)

When given a coding task:
1. Think about the approach (use the think tool)
2. Write and test the solution (use run_script)
3. Verify it passes all test cases
4. Provide the final solution

Be thorough — always test your code before giving the final answer."""


# ── Agentic distillation client ──────────────────────────────────────────

class AgenticDistillClient(DistillationClient):
    """Distillation client with agentic tool-use capabilities.

    Extends DistillationClient with:
    - run_agentic_task(): teacher models call tools in an agentic loop
    - generate_tasks(): teachers generate their own coding tasks
    - save_trajectories(): save tool-use trajectories as SFT data

    Fine-tuning is DISABLED — this is pure data collection.

    Note: Not all models support OpenAI function calling. Models that don't
    support tool use will return errors and be auto-cooled down. The client
    filters to known tool-capable providers on init.
    """

    # Providers/models known to support OpenAI function calling
    _TOOL_CAPABLE_PROVIDERS = {"groq", "deepseek", "nvidia", "mistral",
                               "sambanova", "cerebras", "huggingface"}
    # OpenRouter: only some models support tools (gpt-oss does, GLM doesn't)
    _OPENROUTER_TOOL_MODELS = {"gpt-oss", "deepseek"}
    # Cloudflare: doesn't support function calling yet
    # Z AI: doesn't support function calling yet

    def __init__(self, *args, max_turns: int = 8,
                 max_gen_tokens: int = 2048,
                 agentic_temperature: float = 0.4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns
        self.max_gen_tokens = max_gen_tokens
        self.agentic_temperature = agentic_temperature
        # Shared DB for tool registry
        self._db: DiscoveryDB | None = None
        self._seen_outputs: set[str] = set()

        # Filter to tool-capable models only
        self.models = [m for m in self.models if self._supports_tools(m)]
        # Rebuild canonical groups with filtered models
        self._canonical_groups = {}
        for m in self.models:
            canon = m.canonical or m.model_id
            self._canonical_groups.setdefault(canon, []).append(m)

    @staticmethod
    def _supports_tools(m: DistillModel) -> bool:
        """Check if a model/provider supports OpenAI function calling."""
        if m.provider in AgenticDistillClient._TOOL_CAPABLE_PROVIDERS:
            return True
        if m.provider == "openrouter":
            model_lower = m.model_id.lower()
            return any(kw in model_lower
                       for kw in AgenticDistillClient._OPENROUTER_TOOL_MODELS)
        # cloudflare, zai, siliconflow: no function calling support yet
        return False

    def _get_db(self) -> DiscoveryDB:
        if self._db is None:
            import tempfile
            tmp = Path(tempfile.gettempdir()) / "forge_agentic_distill.sqlite3"
            self._db = DiscoveryDB(str(tmp))
        return self._db

    def _build_tools(self, session_id: str) -> tuple[ToolRegistry, list[dict]]:
        """Build tool registry and OpenAI-format tool definitions."""
        registry = ToolRegistry(self._get_db(), session_id)
        tools = _schemas_to_openai_tools(registry.schemas)
        return registry, tools

    def run_agentic_task(self, task: str, model: DistillModel | None = None,
                         test_cases: list[dict] | None = None) -> AgenticTrajectory:
        """Run a single task through the agentic tool-use loop with a teacher model.

        The teacher model receives the task + tool definitions, emits tool calls,
        we execute them and feed results back, until the teacher gives a final
        answer or max_turns is reached.

        Args:
            task: the task description
            model: specific teacher model to use (None = auto-pick via rotation)
            test_cases: optional test cases to include in the prompt

        Returns:
            AgenticTrajectory with full conversation, tool calls, and reward
        """
        # Pick a model if not specified
        if model is None:
            picked = self._pick_model_with_rotation(task, 1)
            if not picked:
                return AgenticTrajectory(
                    task=task, teacher_model="none", messages=[],
                    tool_calls=[], final_answer=None, reward=0.0,
                    reward_breakdown={}, n_turns=0,
                    stopped_after_tools=False, stopped_after_answer=False,
                    latency_ms=0, tokens_in=0, tokens_out=0,
                    error="No models available",
                )
            model = picked[0]

        client = self._get_client(model)
        if client is None:
            return AgenticTrajectory(
                task=task, teacher_model=f"{model.provider}/{model.model_id}",
                messages=[], tool_calls=[], final_answer=None, reward=0.0,
                reward_breakdown={}, n_turns=0,
                stopped_after_tools=False, stopped_after_answer=False,
                latency_ms=0, tokens_in=0, tokens_out=0,
                error="No API client available",
            )

        # Build tools
        session_id = str(uuid.uuid4())[:8]
        registry, tools = self._build_tools(session_id)

        # Build initial messages
        user_content = task
        if test_cases:
            cases_str = "\n".join(
                f"  Input: {tc.get('input', '')} → Output: {tc.get('output', '')}"
                for tc in test_cases[:10]
            )
            user_content += f"\n\nTest cases:\n{cases_str}"

        messages = [
            {"role": "system", "content": _AGENTIC_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        tool_call_records: list[dict] = []
        final_answer: str | None = None
        stopped_after_tools = False
        stopped_after_answer = False
        total_tokens_in = 0
        total_tokens_out = 0
        t0 = time.time()
        turn = 0

        # Put provider on cooldown if we get repeated errors
        try:
            for turn in range(self.max_turns):
                # Call the teacher model with tool definitions
                kwargs: dict[str, Any] = {
                    "model": model.model_id,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": self.agentic_temperature,
                    "max_completion_tokens": self.max_gen_tokens,
                }
                # Reasoning models
                if model.reasoning and "gpt-oss" in model.model_id.lower():
                    kwargs["reasoning_effort"] = "medium"
                if model.reasoning and "qwen" in model.model_id.lower() and "3.6" not in model.model_id:
                    kwargs["reasoning_effort"] = "medium"

                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                msg = choice.message

                # Track tokens
                if response.usage:
                    total_tokens_in += response.usage.prompt_tokens or 0
                    total_tokens_out += response.usage.completion_tokens or 0

                # Check for tool calls
                tool_calls = msg.tool_calls if hasattr(msg, "tool_calls") else None

                if not tool_calls:
                    # No tool calls — this is the final answer
                    final_answer = (msg.content or "").strip()
                    messages.append({"role": "assistant", "content": final_answer})
                    stopped_after_answer = True
                    break

                # Execute tool calls
                stopped_after_tools = True
                # Add assistant message with tool calls
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
                messages.append(assistant_msg)

                had_error = False
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}

                    result = registry.call(name, args)
                    success = not (isinstance(result, dict) and "error" in result)
                    tool_call_records.append({
                        "name": name,
                        "args": args,
                        "result": result,
                        "success": success,
                    })

                    # Add tool result to conversation
                    content = json.dumps(result, ensure_ascii=False)[:4000]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": content,
                    })
                    if not success:
                        had_error = True

                # Nudge after errors
                if had_error and turn < self.max_turns - 1:
                    messages.append({
                        "role": "system",
                        "content": "A tool returned an error. Analyze what went "
                                   "wrong and try a different approach.",
                    })

        except Exception as e:
            latency = (time.time() - t0) * 1000
            err_str = str(e)
            # Blacklist provider permanently for this process on:
            # - 402 Payment required (no credits)
            # - 429 with daily/monthly quota exhausted
            # - 404 Model not found
            should_blacklist = False
            if "402" in err_str or "Payment required" in err_str or ("429" in err_str and any(kw in err_str.lower() for kw in
                    ("per-day", "per_day", "daily", "per-month", "per_month",
                     "free-models-per-day", "rate limit exceeded"))) or ("404" in err_str and "model" in err_str.lower()):
                should_blacklist = True

            if should_blacklist:
                if model.provider not in self._blacklisted_providers:
                    self._blacklisted_providers.add(model.provider)
                    print(f"  [BLACKLIST] {model.provider} exhausted for this "
                          f"process — {err_str[:80]}")
            elif "429" in err_str:
                self._provider_cooldowns[model.provider] = time.time() + 60
            elif "402" in err_str:
                self._provider_cooldowns[model.provider] = time.time() + self._cooldown_seconds
            return AgenticTrajectory(
                task=task, teacher_model=f"{model.provider}/{model.model_id}",
                messages=messages, tool_calls=tool_call_records,
                final_answer=final_answer, reward=0.0, reward_breakdown={},
                n_turns=turn, stopped_after_tools=stopped_after_tools,
                stopped_after_answer=stopped_after_answer,
                latency_ms=latency, tokens_in=total_tokens_in,
                tokens_out=total_tokens_out, error=err_str,
            )

        # Track request counts
        self._request_counts[model.model_id] = \
            self._request_counts.get(model.model_id, 0) + 1
        self._provider_request_counts[model.provider] = \
            self._provider_request_counts.get(model.provider, 0) + 1
        self._daily_counts[model.model_id] = \
            self._daily_counts.get(model.model_id, 0) + 1

        latency = (time.time() - t0) * 1000

        # Compute reward using the same reward function from tool_use_loop
        if compute_reward is None:
            raise RuntimeError(
                "compute_reward is not available: research.self_play.discovery."
                "tool_use_loop could not be imported (it was removed in the AZR "
                "loop merge). agentic_distill teacher-model tool-use distillation "
                "requires it; restore tool_use_loop.py or disable this path."
            )
        reward = compute_reward(
            task=task,
            tool_calls=tool_call_records,
            final_answer=final_answer,
            stopped_after_tools=stopped_after_tools,
            stopped_after_answer=stopped_after_answer,
            seen_outputs=self._seen_outputs,
        )

        # Track novel outputs
        if final_answer:
            self._seen_outputs.add(final_answer[:200])

        return AgenticTrajectory(
            task=task,
            teacher_model=f"{model.provider}/{model.model_id}",
            messages=messages,
            tool_calls=tool_call_records,
            final_answer=final_answer,
            reward=reward.total,
            reward_breakdown=reward.to_dict(),
            n_turns=turn + 1,
            stopped_after_tools=stopped_after_tools,
            stopped_after_answer=stopped_after_answer,
            latency_ms=latency,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
        )

    def run_agentic_batch(self, tasks: list[str],
                          test_cases_per_task: list[list[dict]] | None = None,
                          n_samples_per_task: int = 2,
                          delay_between: float = 0.5,
                          min_reward: float = 0.3,
                          ) -> list[AgenticTrajectory]:
        """Run agentic tool-use loop on multiple tasks.

        Args:
            tasks: list of task descriptions
            test_cases_per_task: parallel list of test case lists
            n_samples_per_task: number of teacher models to try per task
            delay_between: delay between API calls (rate limiting)
            min_reward: only keep trajectories with reward >= this

        Returns:
            list of AgenticTrajectory objects (filtered by min_reward)
        """
        if test_cases_per_task is None:
            test_cases_per_task = [None] * len(tasks)

        all_trajectories: list[AgenticTrajectory] = []

        for i, task in enumerate(tasks):
            # Early exit: all providers exhausted
            if self.all_providers_exhausted():
                print(f"\n  ⚠ ALL PROVIDERS EXHAUSTED — ending early at "
                      f"task {i}/{len(tasks)}")
                print(f"    Blacklisted: {sorted(self._blacklisted_providers)}")
                break

            tc = test_cases_per_task[i] if i < len(test_cases_per_task) else None
            print(f"\n[{i+1}/{len(tasks)}] Task: {task[:70]}...")

            active = self.active_providers()
            if len(active) <= 2:
                print(f"  ⚠ Only {len(active)} active providers: {active}")

            # Pick n different teacher models for this task
            models = self._pick_model_with_rotation(task, n_samples_per_task)

            for j, model in enumerate(models):
                # Skip if provider got blacklisted mid-task
                if model.provider in self._blacklisted_providers:
                    print(f"  → Teacher {j+1}/{n_samples_per_task}: "
                          f"{model.provider}/{model.model_id} [BLACKLISTED, skip]")
                    continue
                print(f"  → Teacher {j+1}/{n_samples_per_task}: "
                      f"{model.provider}/{model.model_id}")
                traj = self.run_agentic_task(task, model=model, test_cases=tc)

                if traj.error:
                    print(f"    ✗ Error: {traj.error[:100]}")
                else:
                    print(f"    {'✓' if traj.reward >= min_reward else '○'} "
                          f"reward={traj.reward:.2f} turns={traj.n_turns} "
                          f"tools={len(traj.tool_calls)} "
                          f"tokens={traj.tokens_out}")

                if not traj.error and traj.reward >= min_reward:
                    all_trajectories.append(traj)

                if delay_between > 0:
                    time.sleep(delay_between)

        return all_trajectories

    def generate_tasks(self, n_tasks: int = 20,
                       model: DistillModel | None = None,
                       ) -> list[dict]:
        """Have a teacher model generate coding tasks with test cases.

        Args:
            n_tasks: number of tasks to generate
            model: specific teacher model (None = auto-pick)

        Returns:
            list of {"task": str, "test_cases": list[dict]} dicts
        """
        if model is None:
            picked = self._pick_model_with_rotation("task_generation", 3)
            if not picked:
                return []
            # Try up to 3 providers for resilience
            models_to_try = picked
        else:
            models_to_try = [model]

        # Try each provider until one succeeds
        for model in models_to_try:
            client = self._get_client(model)
            if client is None:
                continue

            prompt = (
                f"Generate {n_tasks} diverse Python coding tasks. Each task should:\n"
                "1. Require writing a Python function\n"
                "2. Have 2-5 test cases with specific inputs and expected outputs\n"
                "3. Range from easy to hard\n"
                "4. Cover diverse topics: strings, math, data structures, algorithms, "
                "recursion, file I/O, error handling, parsing, etc.\n\n"
                "Output one task per line as JSON:\n"
                '{"task": "...", "test_cases": [{"input": "...", "output": "..."}]}\n\n'
                "Example:\n"
                '{"task": "Write a Python function is_palindrome(s) that returns True if s is a palindrome", '
                '"test_cases": [{"input": "racecar", "output": "True"}, {"input": "hello", "output": "False"}]}'
            )

            messages = [
                {"role": "system", "content": _TASK_GEN_SYSTEM},
                {"role": "user", "content": prompt},
            ]

            try:
                kwargs: dict[str, Any] = {
                    "model": model.model_id,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_completion_tokens": 4096,
                    "timeout": 60,  # 60s per-call timeout
                }
                if model.reasoning and "gpt-oss" in model.model_id.lower():
                    kwargs["reasoning_effort"] = "medium"

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""

                # Track request
                self._request_counts[model.model_id] = \
                    self._request_counts.get(model.model_id, 0) + 1
                self._provider_request_counts[model.provider] = \
                    self._provider_request_counts.get(model.provider, 0) + 1
            except Exception as e:
                err_str = str(e)
                print(f"Task generation error ({model.provider}): {err_str[:100]}")
                # Blacklist if quota exhausted
                if any(kw in err_str for kw in ("402", "Payment required")) or ("429" in err_str and any(kw in err_str.lower() for kw in
                        ("per-day", "daily", "per-month", "rate limit exceeded"))):
                    self._blacklisted_providers.add(model.provider)
                continue  # try next provider

            # Parse tasks from response
            tasks = []
            for line in content.split("\n"):
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    task = json.loads(line)
                    if "task" in task and isinstance(task["task"], str):
                        if not _is_filler_task(task["task"]):
                            task.setdefault("test_cases", [])
                            tasks.append(task)
                except json.JSONDecodeError:
                    continue

            if tasks:
                return tasks  # success — stop trying providers

        return []  # all providers failed

    def save_trajectories(self, trajectories: list[AgenticTrajectory],
                          path: str | Path) -> int:
        """Save trajectories as JSONL for SFT training.

        Args:
            trajectories: list of AgenticTrajectory objects
            path: output file path

        Returns:
            number of trajectories saved
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj.to_sft_dict(), ensure_ascii=False) + "\n")

        return len(trajectories)

    def agentic_stats(self) -> dict:
        """Return statistics including agentic-specific metrics."""
        base = self.stats()
        base.update({
            "seen_outputs": len(self._seen_outputs),
            "max_turns": self.max_turns,
            "max_gen_tokens": self.max_gen_tokens,
        })
        return base
