"""Infinite Curriculum Engine — AZR-style self-play task generation.

Implements the Absolute Zero Reasoner (AZR) paradigm:
  - The model PROPOSES coding tasks with unit tests
  - A Python executor VERIFIES (ground truth, not model judgment)
  - Goldilocks difficulty: proposer rewarded when solver succeeds 40-60%
  - Three reasoning modes: deduction, abduction, induction
  - Infinite, adaptive curriculum — never runs out of questions

Architecture:
  1. Proposer: model generates (task_description, function_signature, test_cases)
  2. Validator: Python executor checks that test_cases are self-consistent
     (reference solution passes all tests — filters garbage proposals)
  3. Solver: model solves the task (existing self-play infrastructure)
  4. Difficulty tracker: per-domain rolling success rate
     - >60% success → proposer must generate harder tasks
     - <40% success → proposer must generate easier tasks
  5. Task queue: validated tasks stored for replay and curriculum ordering

Research basis:
  - AZR (NeurIPS 2025): zero-data SOTA via propose-solve-verify loop
  - SQLM (2025): asymmetric self-play, Goldilocks reward
  - SPICE (Meta FAIR, 2025): corpus grounding prevents collapse
  - SOAR (2026): reward = measured improvement, not intrinsic proxy

Usage:
    from forge.self_play.infinite_curriculum import InfiniteCurriculum
    curriculum = InfiniteCurriculum(model, tokenizer, device="cuda")
    tasks = curriculum.propose_tasks(domain="algorithms", n=10)
    for task in tasks:
        result = curriculum.solve_task(task)
        curriculum.record_result(task, result["success"])
"""
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from forge.engine.scheduler.async_d2h import AsyncTokenReader
from research.json_compat import loads

# Pre-compiled regex patterns (avoids 450+ recompilations per session).
_RE_PYTHON_BLOCK = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
_RE_TASK_DESC = re.compile(r'#\s*(?:TASK|Task):\s*(.+?)(?:\n|$)')
_RE_QUOTED = re.compile(r'["\'](.+?)["\']')
_RE_SOLVE_SIG = re.compile(r'def\s+solve\s*(\(.*?\))')
_RE_SOLVE_TEST = re.compile(r'#\s*solve\s*\((.+?)\)\s*==\s*(.+?)(?:\n|$)')
_RE_ASSERT_TEST = re.compile(r'#?\s*assert\s+solve\s*\((.+?)\)\s*==\s*(.+?)(?:,\s*["\']|\n|$)')
_RE_PRINT_TEST = re.compile(
    r'print\(\s*solve\s*\((.+?)\)\s*\)\s*#\s*'
    r'(?:[Ee]xpected\s*[:=]?\s*|==\s*|->\s*)?(.+?)(?:\n|$)')
# Bare input lines: "# solve(<args>)" with no declared expected — AZR-style
# proposals where the executor derives o = solve(args) as ground truth.
_RE_SOLVE_INPUT = re.compile(
    r'#\s*solve\s*\((.*?)\)\s*(?:->[^\n]*)?(?:\n|$)')
# Bare test calls at line start: "solve(<args>)  # comment" — the model
# drifts to this form inside the code block (observed live). Disjoint from
# _RE_SOLVE_INPUT, which requires a leading '#'.
_RE_SOLVE_BARE = re.compile(
    r'^\s*solve\s*\((.*?)\)\s*(?:#.*)?$', re.MULTILINE)
_RE_SOLVE_IMPL = re.compile(r'(def\s+solve\s*\([^)]*\)(?:\s*->\s*[^\n:]+)?:.*?)(?:\nassert|\n#|\n\ndef|\Z)', re.DOTALL)
_RE_RETURN = re.compile(r'return\s+(.+)')
_RE_CODEBLOCK_OPEN = re.compile(r'```python\s*\n?')
_RE_CODEBLOCK_CLOSE = re.compile(r'```\s*$', re.MULTILINE)
# Control-looking vocab tokens (<|im_start|>, <|reserved_NNN|>, <tool_call>,
# <think>, …). Banned wholesale in self-play generated text — observed live:
# the 3B model samples <|reserved_567|>-class garbage mid-answer, and the
# GUI's six-id structural set doesn't cover the reserved range. Byte-fallback
# tokens (<0x0A> = '\n', etc.) are EXCLUDED — they're how the tokenizer
# emits newlines/unicode; banning them starves the sampler (observed:
# force-injected suffix began with a banned <0x0A> → all -inf logits →
# multinomial device-side assert).
_RE_CONTROL_TOKEN = re.compile(r'^<(?!0x[0-9A-Fa-f]{2}>)[^>\n]+>$')
# Ids generation legitimately needs: <|endoftext|> EOS, <|im_end|> turn end,
# </think> (the model must be able to close its own think block).
_CONTROL_TOKEN_KEEP = frozenset({2, 519, 542})

from forge.evaluation.goal_tasks import GoalTask
import contextlib

# ─── Persistent sandbox worker pool ──────────────────────────────────
# Module-level ThreadPool reused across sandbox executions to avoid the
# overhead of creating/destroying threads for every validation run.
_SANDBOX_POOL = None


def _get_sandbox_pool():
    """Return (or lazily create) the persistent sandbox ThreadPool."""
    global _SANDBOX_POOL
    if _SANDBOX_POOL is None:
        from multiprocessing.pool import ThreadPool
        _SANDBOX_POOL = ThreadPool(processes=4)
    return _SANDBOX_POOL


# ─── Data structures ──────────────────────────────────────────────────

@dataclass
class ProposedTask:
    """A task proposed by the model, with self-generated test cases."""
    id: str
    domain: str                          # algorithms, math, strings, logic
    difficulty: str                      # easy, medium, hard (model-declared)
    description: str                     # what the function should do
    signature: str                       # e.g. "(n: int) -> int"
    solve_name: str = "solve"
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    # each: {"args": (...,), "expected": ...}
    # Bare inputs the model proposed without declaring outputs — expected
    # values are derived by executing solve(*args) in validation (AZR-style:
    # the executor, not the model, is the source of ground truth).
    pending_inputs: list[tuple] = field(default_factory=list)
    stress_index: int | None = None
    proposer_confidence: float = 0.0     # model's own confidence
    validated: bool = False              # passed self-consistency check
    archetype: str = "model_proposed"    # marks this as model-generated
    raw_output: str = ""                 # full model output for debugging
    generation_mode: str = "induction"   # induction/abduction/deduction


@dataclass
class CurriculumStats:
    """Tracks curriculum health metrics."""
    total_proposed: int = 0
    total_validated: int = 0       # passed self-consistency
    total_solved: int = 0          # solver succeeded
    total_failed: int = 0          # solver failed
    # Per-domain rolling success rate (for Goldilocks difficulty)
    domain_results: dict[str, list[bool]] = field(default_factory=dict)
    # AZR proposer reward tracking (r_propose = 1 - |success_rate - 0.5|)
    proposer_rewards: list[float] = field(default_factory=list)
    # Task diversity tracking (semantic distance between proposed tasks)
    task_embeddings: list[list[float]] = field(default_factory=list)

    @property
    def validation_rate(self) -> float:
        return self.total_validated / max(self.total_proposed, 1)

    @property
    def solve_rate(self) -> float:
        return self.total_solved / max(self.total_validated, 1)

    def domain_success_rate(self, domain: str) -> float:
        results = self.domain_results.get(domain, [])
        if not results:
            return 0.5  # unknown → assume medium
        return sum(results) / len(results)

    @property
    def mean_proposer_reward(self) -> float:
        """AZR learnability reward: peaks at 50% solver success."""
        if not self.proposer_rewards:
            return 0.0
        return sum(self.proposer_rewards) / len(self.proposer_rewards)

    @property
    def diversity_score(self) -> float:
        """Mean pairwise cosine distance between recent task embeddings.
        Target >0.6 (alert if <0.6 = mode collapse)."""
        if len(self.task_embeddings) < 2:
            return 1.0
        import numpy as np
        embs = np.array(self.task_embeddings[-100:])  # last 100
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        normed = embs / (norms + 1e-8)
        sim_matrix = normed @ normed.T
        n = len(embs)
        # mean of off-diagonal cosine similarities
        total_sim = (sim_matrix.sum() - n) / (n * (n - 1))
        return float(1.0 - total_sim)


# ─── Task proposal prompts ────────────────────────────────────────────

# Mode 1: Induction — model writes a function + test cases (code completion style)
# Push-limits: the prompt asks for NON-TRIVIAL tasks with branching, edge cases.
# Few-shot example pins the exact format the parser expects:
#   `def solve(...):` body + `# solve(args) == expected` comment test cases.
# IMPORTANT: the example uses a DIFFERENT signature than the target so the
# model can't just copy it verbatim. The desc_hint is placed FIRST and
# emphasized so it drives the task, not the example.
_INDUCTION_EXAMPLE = '''# FORMAT EXAMPLE (different task — do NOT copy):
# def solve(nums: list) -> int:
#     total = 0
#     for n in nums:
#         if n > 0:
#             total += n
#     return total
# # solve([1, -2, 3]) == 4
# # solve([]) == 0
# # solve([-1, -2]) == 0
# # solve([5]) == 5
'''

INDUCTION_PROMPT = '''# TASK: {desc_hint}
# Write solve{signature} that solves the ABOVE problem using CONDITIONAL LOGIC or LOOPS.
# Do NOT just call a builtin (sum, max, len, sorted, etc.) — implement the logic yourself.
# After the function, write 4+ test cases as comments in EXACTLY this format:
#   # solve(<args>) == <expected>
# Include edge cases (empty, negative, single element, large).

''' + _INDUCTION_EXAMPLE + '''
# Now implement the TASK at the top. Start with `def solve{signature}:`

def solve{signature}:
'''

# Mode 2: Abduction — given output, write code that produces it
ABLATION_PROMPT = """# You are a task designer. Create a Python programming challenge.
# Mode: ABDUCTION — given a target output, the solver must write code that produces it.
#
# Rules:
# 1. The task must be solvable with a single function called `solve`.
# 2. Provide the function signature and 3-5 (input, output) pairs.
# 3. The function should transform the input to produce the output.
# 4. Focus on: {domain}
#
# Generate a {difficulty} {domain} task in the same format:"""

# Mode 3: Deduction — given code, predict output (simpler, for warmup)
DEDUCTION_PROMPT = """# You are a task designer. Create a Python trace challenge.
# Mode: DEDUCTION — given code and input, the solver must predict the output.
#
# Rules:
# 1. Provide a short Python function (5-15 lines).
# 2. Provide 3-5 (input, expected_output) pairs.
# 3. The function should be deterministic (no randomness, no I/O).
# 4. Focus on: {domain}
#
# Generate a {difficulty} {domain} task:"""

# Solver prompt — shown to the model that must solve the task
SOLVER_PROMPT = """# Goal: {description}
# Define a function `solve{signature}`.
# It must produce the correct output for ALL of these test inputs:
{test_lines}
# Implement solve any way you choose."""

# ─── Chat-mode prompts (engine path) ─────────────────────────────────
# When a ForgeEngine is attached, prompts render as ChatML user turns and
# the model replies with an instruction-following answer — NOT a raw code
# continuation. The content requirements are identical to the completion
# prompts above; only the surface format differs (single ```python block).

CHAT_PROPOSE_PROMPT = """Here is an EXAMPLE of a Python programming task — study the format, do NOT copy it:

```python
# Task: Return the sum of the digits of n.
def solve(n: int) -> int:
    total = 0
    n = abs(n)
    while n > 0:
        total += n % 10
        n //= 10
    return total

# solve(0)
# solve(123)
# solve(-45)
# solve(9999)
```

Now create ONE DIFFERENT {difficulty} task in the {domain} domain (idea: "{desc_hint}", or invent your own). Same format exactly: a `# Task:` line, a `def solve{signature}:` implementation using loops/conditionals (not just a builtin call), then 4+ `# solve(input)` test-INPUT lines covering edge cases. Expected outputs are computed automatically — do NOT write `== <expected>`. Reply with ONLY the code block."""

CHAT_SOLVE_PROMPT = """{description}

Write a complete Python function `def {solve_name}{sig_params}:` that returns (not prints) the correct output for ALL of these cases:

{test_lines}

Reply with ONLY a single ```python code block containing the full function definition."""

# GUI _split_reasoning parity: the think-mode render opens <think> in the
# assistant turn; everything before the last </think> is reasoning_content
# (dropped), everything after is the reply. A leading Answer:/Final Answer:
# anchor is stripped the same way the GUI does (_strip_answer_anchor).
_RE_ANSWER_ANCHOR = re.compile(r"^\s*(?:Final\s+)?Answer\s*:\s*\n?")

# Chat-mode replies include the model's <think> reasoning, so they need a
# larger token floor than raw completion continuations, and the think block
# itself gets a hard cap (GUI _think_cap_processor — after `budget` think
# tokens without a </think>, force-inject the think-close + code-fence
# suffix, dropping the model straight into code-completion mode; a bare
# </think> or "Answer:" anchor left it musing — observed live). Self-play
# bans tool markers outright (no tools exist).
_CHAT_MIN_SOLVE_TOKENS = 384
_CHAT_MIN_PROPOSE_TOKENS = 640
_CHAT_SOLVE_THINK_BUDGET = 192
_CHAT_PROPOSE_THINK_BUDGET = 320
# Force-close suffixes for the think-cap: instead of the chat "Answer:"
# anchor (the model keeps musing on meta-prompts — observed live), inject
# the think-close PLUS the code-fence opener so the model lands directly
# in code-completion mode on the required format.
_CHAT_SOLVE_CLOSE = "\n</think>\n```python\n"
_CHAT_PROPOSE_CLOSE = "\n</think>\n```python\n# Task: "
# Propose prefill: instead of a think block (the 3B self-closes it early
# then muses about the meta-instructions — observed live on every batch),
# render the direct turn and prefill the code-fence opener so the model
# continues as code. Task proposal is format imitation, not reasoning.
_CHAT_PROPOSE_PREFILL = "```python\n"
# Solve prefill: same mechanism — the solver's think block mused in
# answer voice or truncated mid-function (observed: 0/4 solved, all
# SyntaxError). Prefilled code keeps the whole budget for the function.
_CHAT_SOLVE_PREFILL = "```python\n"
# Post-think fence deadline: if the model closes </think> itself but the
# "answer" rambles this many tokens without opening a code fence, inject
# the fence part of the close suffix (observed live: 3B self-closes think
# then muses in answer voice for the whole budget — think-cap can't fire
# because think_open is False once 542 lands).
_POST_THINK_FENCE_DEADLINE = 48


# Structural markers that may survive decode when skip_special_tokens=False.
# </think> is needed for the reasoning split, so decode must preserve it;
# trailing turn markers (<|im_end|>, <|endoftext|>) get dropped here.
_RE_TURN_MARKERS = re.compile(
    r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>"
    r"|<｜start▁of▁sentence｜>|<｜end▁of▁sentence｜>"
    r"|<\|User\|>|<\|Bot\|>|<\|System\|>|<\|Tool\|>")


def _strip_reply(text: str) -> str:
    """Split reasoning from a chat reply (GUI _split_reasoning parity).

    Drops everything through the last ``</think>`` (that text is the
    model's reasoning_content, not the reply), removes a leading
    ``Answer:``/``Final Answer:`` anchor if present, and strips any
    surviving ChatML turn markers.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = _RE_ANSWER_ANCHOR.sub("", text, count=1)
    return _RE_TURN_MARKERS.sub("", text).strip("\n")


class _DocstringStripper(ast.NodeTransformer):
    """Remove docstring Expr nodes (module/function/class level)."""

    def _strip(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


def _clean_code(code: str) -> str:
    """Strip all comments + docstrings; canonicalize via AST round-trip.

    The solver dumps reasoning into ``#`` comments and docstrings (observed
    live: 40+ lines of musing inside the exported solution). Neither affects
    execution, so dropping them yields terse, correct training code.
    Returns "" when the code doesn't parse — the caller keeps the raw text
    so verification can fail it naturally.
    """
    try:
        tree = ast.parse(code)
        cleaned = ast.unparse(_DocstringStripper().visit(tree))
        return cleaned
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return ""


def _json_default(o):
    """json default= handler for task persistence: coerce values the model
    can smuggle into test_cases via eval'd literals or sandbox outputs
    (sets, bytes, arbitrary objects) into serializable forms."""
    if isinstance(o, (set, frozenset)):
        try:
            return sorted(o)
        except TypeError:
            return list(o)
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", "replace")
    return repr(o)


# ─── Generation tracer (debug rounds) ────────────────────────────────

class GenTracer:
    """Records every generation call so a debug round reveals exactly
    what the model emitted.

    Attached to ``InfiniteCurriculum.gen_trace`` by the loop when
    ``LoopConfig.debug_round`` is set. Each ``_generate``/``_generate_batch``
    call records the rendered prompt fed to the model, the RAW reply
    (pre-strip — ``<think>``/``<|im_end|>``/fence markers all visible),
    the post-strip reply the pipeline actually consumes, and per-token
    ids + decoded pieces. Events print live and persist for the
    ``debug_round_epN`` transcript the loop dumps after Phase 1.

    Engine paths return text only, so token ids are re-encoded from the
    raw output (``ids_source="reencoded"`` — special markers are added
    tokens and re-encode faithfully; boundary merges may differ by ±1).
    The legacy path supplies the true sampled ids (``"generated"``).
    """

    def __init__(self, tokenizer, live: bool = True):
        self.tokenizer = tokenizer
        self.live = live
        self.events: list[dict] = []

    def record(self, *, phase: str, prompt: str, rendered: str,
               raw: str, reply: str,
               gen_ids: list[int] | None = None, **meta) -> dict:
        ids_source = "generated"
        ids = [int(t) for t in gen_ids] if gen_ids is not None else None
        if ids is None:
            ids_source = "reencoded"
            ids = [int(t) for t in self.tokenizer(
                raw, add_special_tokens=False).input_ids]
        ev = {
            "i": len(self.events), "phase": phase, "prompt": prompt,
            "rendered": rendered, "raw": raw, "reply": reply,
            "n_tokens": len(ids), "ids_source": ids_source,
            "tokens": [{"id": t, "text": self._piece(t)} for t in ids],
            **meta,
        }
        self.events.append(ev)
        if self.live:
            self._print_live(ev)
        return ev

    def _piece(self, tid: int) -> str:
        try:
            return self.tokenizer.decode([tid])
        except Exception:
            return "<?>"

    def _print_live(self, ev: dict) -> None:
        # ASCII separators only — box-drawing chars mojibake on the
        # Windows console's non-UTF-8 codepage.
        bar = "-" * 66
        print(f"\n{bar}\n  GEN #{ev['i']} [{ev['phase']}] "
              f"{ev['n_tokens']} tokens ({ev['ids_source']})")
        try:
            print(ev["raw"] if ev["raw"].strip() else "  <empty output>")
        except UnicodeEncodeError:
            print(ev["raw"].encode("ascii", "replace").decode("ascii"))
        print(bar)


# ─── Infinite Curriculum Engine ───────────────────────────────────────

class InfiniteCurriculum:
    """AZR-style infinite curriculum: propose → validate → solve → score.

    The model generates tasks with test cases. A Python executor validates
    that the proposer's own reference solution passes the tests (self-consistency).
    Validated tasks are then given to the solver. Success rate feeds back into
    difficulty adjustment (Goldilocks zone: 40-60% solver success).
    """

    # Domains and their difficulty progression
    DOMAINS = ["algorithms", "math", "strings", "logic", "data_structures"]
    DIFFICULTIES = ["easy", "medium", "hard"]

    # AZR-optimal Goldilocks zone (NeurIPS 2025 Spotlight + "Survive or Collapse"
    # arxiv 2605.22217): proposer reward peaks at 50% solver success rate.
    # Tasks too easy (>80%) or too hard (<20%) get zero/negative proposer reward.
    # Previous setting (target=0.2) was too hard — model failed 80% and learned
    # almost nothing. 50% is the proven optimal learnability point.
    GOLDILOCKS_LOW = 0.2
    GOLDILOCKS_HIGH = 0.8
    TARGET_SUCCESS = 0.5  # AZR optimal: model solves ~50% (max learnability)
    MIN_DIFFICULTY = "easy"  # allow easy tasks early in curriculum
    DEFAULT_DIFFICULTY = "medium"  # start at medium, adapt based on success rate

    # Minimum complexity requirements — reject trivial tasks
    MIN_TEST_CASES = 4          # need at least 4 test cases
    MIN_SIGNATURE_ARGS = 1      # at least 1 argument
    BANNED_PATTERNS = {         # reject tasks whose solution is just a builtin
        "sum", "max", "min", "len", "sorted", "abs", "round",
        "int", "str", "list", "set", "dict", "tuple", "bool",
        "print", "type", "hash", "id", "ord", "chr",
    }

    # Rolling window for per-domain success tracking
    WINDOW_SIZE = 15

    def __init__(self, model, tokenizer, device: str = "cuda",
                 max_gen_tokens: int = 200,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 task_queue_dir: str | None = None,
                 live_status=None,
                 engine=None,
                 config_name: str = "forgelm_v2"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.live = live_status
        # When set, all generation routes through the ForgeEngine (activated
        # features: prefix cache, spec decode, DRY, proper EOS) and prompts
        # are rendered in the GUI's canonical Jamba ChatML — the model is
        # chat-tuned, so raw completion prompts are off-distribution.
        self.engine = engine
        # Chat-format preset (e.g. "spark_x26_1p7b") — selects the turn
        # renderer in _render_chat so proposals use the model's trained
        # chat dialect instead of always Jamba ChatML.
        self.config_name = config_name
        # Lazily-encoded force-close suffix ids for the think-cap
        # processor, keyed by suffix text (needs the tokenizer).
        self._think_suffix_ids = {}
        # Lazily-built list of control-looking vocab ids banned in
        # generated text (see _banned_special_ids).
        self._banned_ids = None
        # Debug-round generation tracer (LoopConfig.debug_round) — None in
        # normal runs; _generate/_generate_batch record every call when set.
        self.gen_trace: GenTracer | None = None
        self._trace_phase = "gen"
        if task_queue_dir is None:
            from research.paths import CURRICULUM_DIR
            task_queue_dir = CURRICULUM_DIR
        self.task_queue_dir = Path(task_queue_dir)
        self.task_queue_dir.mkdir(parents=True, exist_ok=True)

        self.stats = CurriculumStats()
        self._task_counter = 0
        self.rng = random.Random(42)

        # Task queue: validated tasks ready for training
        self._task_queue: list[ProposedTask] = []
        # Failed/rejected tasks (for diversity — don't re-propose similar)
        self._rejected_signatures: set = set()
        # All submitted task descriptions (valid + rejected) to prevent clones.
        # Stored normalized (lowercase, stripped, collapsed whitespace).
        self._seen_descriptions: set[str] = set()
        self._seen_desc_tokens: list[set[str]] = []
        self._seen_desc_path = self.task_queue_dir / "seen_descriptions.json"
        # Functional fingerprints: probe-input output signatures of
        # validated reference impls — catches same-function clones whose
        # descriptions differ ("count even digits" vs "number of even
        # digits" are the same function → same fingerprint → reject).
        self._seen_fingerprints: set[str] = set()
        self._seen_fp_path = self.task_queue_dir / "seen_fingerprints.json"

        # Forward cache for repeated prompt prefixes (C2 — 20-40% fewer forward passes)
        from forge.runtime.forward_cache import ForwardCache
        self.fwd_cache = ForwardCache(max_entries=200, device=device)

        # Self-modeling: confidence-based retry policy (D5)
        from forge.runtime.self_model import ConfidenceScorer, RetryPolicy
        self.confidence_scorer = ConfidenceScorer()
        self.retry_policy = RetryPolicy(
            min_confidence=0.4,  # low bar — we're pushing limits, low confidence is normal
            max_retries=1,       # one retry on very low confidence
            temp_increase=0.2,
            base_temp=temperature,
        )

        # ELO-driven curriculum matchmaking: rates both model and individual
        # prompts, targets ~50% expected success (Goldilocks zone). Augments
        # the per-domain rolling success rate with per-prompt difficulty tracking.
        from forge.self_play.elo_tracker import EloTracker
        self.elo = EloTracker(initial_rating=1200.0, seed=42)

    # ─── Task Proposal ─────────────────────────────────────────────

    def propose_tasks(self, domain: str = "algorithms",
                      n: int = 5,
                      difficulty: str | None = None,
                      mode: str = "induction",
                      batch_size: int = 8) -> list[ProposedTask]:
        """Generate n tasks via the model, validate them, return valid ones.

        Uses batched generation: multiple prompts are generated in parallel
        in a single forward pass (batch_size at a time). This gives
        batch_size× speedup on GPU since batch=1 underutilizes compute.

        Args:
            domain: target domain (algorithms, math, strings, etc.)
            n: number of tasks to attempt (may return fewer if validation fails)
            difficulty: override adaptive difficulty (None = auto)
            mode: induction / abduction / deduction
            batch_size: number of prompts to generate in parallel (8 = good default)
        """
        if difficulty is None:
            difficulty = self._adaptive_difficulty(domain)

        self._trace_phase = "propose"
        valid_tasks = []
        proposed = 0
        parse_failed = 0
        validated_count = 0

        # Degeneration guard: if the model emits >MAX_CONSECUTIVE_FAILS parse
        # failures in a row, stop early (was looping 100 times producing
        # garbage). The few-shot prompt should fix most failures, but this
        # prevents pathological stalls.
        MAX_CONSECUTIVE_FAILS = 12
        consecutive_fails = 0

        while proposed < n:
            # Generate a batch of prompts
            current_batch = min(batch_size, n - proposed)
            prompts = []
            self._batch_used_hints = set()  # reset per-batch hint dedup
            for _ in range(current_batch):
                sig_hint = self._random_signature(domain)
                _t_args, _t_expected = self._example_test_cases(domain, sig_hint)
                desc_hints = {
                    "algorithms": [
                        "Find the k-th smallest element using partitioning (not sorted())",
                        "Detect if a list contains a cycle using Floyd's algorithm",
                        "Merge two sorted lists into one sorted list (not using sorted())",
                        "Find the longest increasing subsequence length",
                        "Compute the running median of a stream",
                        "Rotate a matrix 90 degrees in-place",
                        "Find all pairs that sum to a target (no nested loops if possible)",
                        "Implement binary search on a rotated sorted array",
                        "Count inversions in a list using merge sort",
                        "Find the majority element (appears > n/2 times) in O(n) space",
                    ],
                    "math": [
                        "Compute the n-th prime number using the Sieve of Eratosthenes",
                        "Find all prime factors of n (not just check if prime)",
                        "Compute the digital root of n (recursive sum until single digit)",
                        "Check if n is a perfect number (sum of proper divisors equals n)",
                        "Compute the n-th Catalan number",
                        "Find the GCD of a list of numbers (not just two)",
                        "Compute the Euler totient function phi(n)",
                        "Check if n is a power of 2 using bit manipulation",
                        "Compute the continued fraction representation of sqrt(n)",
                        "Find the modular inverse of a mod m using extended Euclidean",
                    ],
                    "strings": [
                        "Find the longest palindromic substring (not just check palindrome)",
                        "Compress a string using run-length encoding",
                        "Check if two strings are anagrams without sorting",
                        "Find the first non-repeating character in a string",
                        "Implement string matching using KMP algorithm",
                        "Convert a string to title case handling edge cases",
                        "Find the minimum window in s that contains all chars of t",
                        "Validate if a string is a valid IPv4 address",
                        "Count distinct palindromic substrings",
                    ],
                    "logic": [
                        "Evaluate a boolean expression string with AND/OR/NOT",
                        "Check if a number is a valid Luhn checksum",
                        "Determine if a Sudoku row/column is valid",
                        "Check if parentheses/brackets are balanced",
                        "Evaluate a simple arithmetic expression string (no eval)",
                    ],
                    "data_structures": [
                        "Implement a min-stack (push/pop/min in O(1))",
                        "Evaluate reverse polish notation",
                        "Implement a circular queue with wraparound",
                        "Flatten a nested list (arbitrary depth) iteratively",
                        "Detect if a binary tree is a valid BST",
                    ],
                }
                # Sample desc_hint without replacement within a batch to
                # encourage diverse prompts (prevents identical prompts → clone tasks)
                available_hints = desc_hints.get(domain, desc_hints["algorithms"])
                unused = [h for h in available_hints
                          if h not in self._batch_used_hints]
                if unused:
                    desc_hint = self.rng.choice(unused)
                else:
                    # All used — reset and pick fresh
                    self._batch_used_hints = set()
                    desc_hint = self.rng.choice(available_hints)
                self._batch_used_hints.add(desc_hint)
                prompts.append(self._propose_prompt(
                    domain, difficulty, sig_hint, desc_hint))

            # Batched generation — all prompts in one forward pass sequence.
            # Chat replies carry the <think> block — propose gets a higher
            # floor and a longer think budget.
            completions = self._generate_batch(
                prompts, min_tokens=_CHAT_MIN_PROPOSE_TOKENS,
                think_budget=_CHAT_PROPOSE_THINK_BUDGET,
                close_suffix=_CHAT_PROPOSE_CLOSE,
                prefill=_CHAT_PROPOSE_PREFILL)

            # Parse and validate each completion. In engine/chat mode the
            # reply IS the task spec; in raw mode the completion continues
            # the prompt's open `def solve` stub (prompt + completion).
            for j, completion in enumerate(completions):
                proposed += 1
                full_code = (completion if self.engine is not None
                             else prompts[j] + completion)
                task = self._parse_proposal(full_code, domain, difficulty, mode)
                if task is not None:
                    task.raw_output = full_code

                if task is None:
                    parse_failed += 1
                    consecutive_fails += 1
                    with contextlib.suppress(OSError):
                        print(f"  [Curriculum] Task {proposed}/{n}: PARSE FAILED")
                    if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                        print(f"  [Curriculum] {consecutive_fails} consecutive parse failures — "
                              f"stopping early (got {len(valid_tasks)} valid)")
                        if self.live is not None:
                            self.live.curriculum_progress(
                                proposed, validated_count, parse_failed)
                        return valid_tasks
                    continue

                consecutive_fails = 0  # reset on a successful parse
                self.stats.total_proposed += 1
                safe_desc = task.description[:40].encode('ascii', 'replace').decode('ascii')
                with contextlib.suppress(OSError):
                    print(f"  [Curriculum] Task {proposed}/{n}: sig={task.signature}, "
                          f"tests={len(task.test_cases)}, desc={safe_desc}")

                # Clone guard: reject tasks with descriptions we've already seen
                if self._is_seen_description(task.description):
                    print("    -> REJECTED (duplicate description — clone)")
                    continue

                # Complexity filter: reject trivial tasks (push-limits mode)
                if not self._is_complex_enough(task):
                    print("    -> REJECTED (too simple)")
                    self._mark_description_seen(task.description)
                    continue

                # Validate: run proposer's reference solution against test cases
                if self._validate_task(task):
                    # Functional clone guard: run the ref impl on fixed
                    # probes — identical output signature = same function
                    # regardless of description wording.
                    fp = self._functional_fingerprint(
                        self._extract_reference_code(task.raw_output), task)
                    if fp is not None and fp in self._seen_fingerprints:
                        self._mark_description_seen(task.description)
                        print("    -> REJECTED (functional clone — same "
                              "outputs on probe inputs)")
                        continue
                    if fp is not None:
                        self._seen_fingerprints.add(fp)
                    task.validated = True
                    self.stats.total_validated += 1
                    validated_count += 1
                    valid_tasks.append(task)
                    self._task_queue.append(task)
                    self._save_task(task)
                    self._mark_description_seen(task.description)
                    print(f"    -> VALIDATED (conf={task.proposer_confidence})")
                else:
                    self._rejected_signatures.add(task.signature)
                    self._mark_description_seen(task.description)
                    print("    -> REJECTED (validation failed)")

            # Live progress per batch
            if self.live is not None:
                self.live.curriculum_progress(proposed, validated_count, parse_failed)

        # Final flush of seen descriptions to disk
        self._save_seen_descriptions()
        return valid_tasks

    def _render_chat(self, instruction: str,
                     prefill: str | None = None) -> str:
        """Render an instruction as the GUI does on a think turn:
        canonical Jamba ChatML user turn, generation prompt opening the
        assistant ``<think>`` block (thinking=True — the trained format;
        direct-mode rendering left the model musing in think-voice and
        starved the reply of tokens).

        With ``prefill``, renders the direct (thinking=False) turn and
        appends the prefill text — the model continues it as code, which
        sidesteps the 3B's meta-musing failure on task-creation prompts.
        """
        from forge.self_play.discovery.qwen_adapter import render_messages_for_config
        rendered = render_messages_for_config(
            [{"role": "user", "content": instruction}],
            config_name=self.config_name,
            thinking=(prefill is None))
        return rendered + prefill if prefill else rendered

    def _generate_batch(self, prompts: list[str],
                        min_tokens: int = _CHAT_MIN_SOLVE_TOKENS,
                        think_budget: int = _CHAT_SOLVE_THINK_BUDGET,
                        close_suffix: str = _CHAT_SOLVE_CLOSE,
                        prefill: str | None = None,
                        seeds: list[int | None] | None = None
                        ) -> list[str]:
        """Generate completions for multiple prompts in a single batched pass.

        Engine path: ForgeEngine.generate_batch (BatchedDecoding — all
        prompts in one forward sequence, proper EOS set, gen-lock, OOM
        fallback). Legacy path: left-padding with proper attention_mask and
        position_ids so pad tokens don't corrupt generation.

        Args:
            prompts: list of prompt strings (1-16 typically)

        Returns:
            list of completion strings (same length as prompts)
        """
        if self.engine is not None:
            rendered = [self._render_chat(p, prefill=prefill)
                        for p in prompts]
            # One think-cap processor per sequence — each holds its own
            # force-injection queue, so instances can't be shared.
            # Prefill mode: budget=None — no think block to cap, the
            # processor only applies the structural bans.
            caps = [self._think_cap(
                None if prefill else think_budget, close_suffix)
                for _ in rendered]
            replies = self.engine.generate_batch(
                rendered,
                max_new_tokens=max(self.max_gen_tokens, min_tokens),
                temperature=self.temperature, top_p=self.top_p,
                top_k=self.top_k, seeds=seeds,
                logits_processors=caps,
                # Preserve </think> for the reasoning split.
                skip_special_tokens=False)
            stripped = [_strip_reply(r) for r in replies]
            if prefill:
                # Model sometimes echoes the fence opener — strip it so
                # the prepended prefill doesn't create an empty block.
                stripped = [re.sub(r'^\s*```\w*\s*\n?', '', r)
                            for r in stripped]
                stripped = [prefill + r for r in stripped]
            if self.gen_trace is not None:
                for p, rend, raw, rep in zip(
                        prompts, rendered, replies, stripped):
                    self.gen_trace.record(
                        phase=self._trace_phase, prompt=p, rendered=rend,
                        raw=raw, reply=rep, temperature=self.temperature,
                        max_new_tokens=max(self.max_gen_tokens, min_tokens))
            return stripped
        if len(prompts) == 1:
            return [self._generate(prompts[0])]

        B = len(prompts)
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id or 0

        # Tokenize all prompts — batch mode (single call vs N per-prompt calls)
        # Use return_tensors=None since prompts have variable lengths; we
        # left-pad manually below.
        enc = self.tokenizer(prompts, truncation=True, max_length=512)
        all_ids = [torch.tensor(ids, dtype=torch.long) for ids in enc.input_ids]

        # Left-pad to max prompt length (left pad so generation positions align)
        max_prompt_len = max(ids.shape[0] for ids in all_ids)
        padded = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(B, max_prompt_len, dtype=torch.long, device=self.device)
        for i, ids in enumerate(all_ids):
            t = ids.shape[0]
            padded[i, max_prompt_len - t:] = ids.to(self.device)
            attention_mask[i, max_prompt_len - t:] = 1

        # Generate
        with torch.no_grad():
            # Prefill: process all prompts at once with attention_mask
            logits, _, past_kvs = self.model(
                padded, past_key_values=None, use_cache=True, attention_mask=attention_mask)
            next_logits = logits[:, -1]  # (B, vocab)
            next_tokens = self._sample_batch(next_logits)  # (B,)

            # Async D2H: issue non-blocking copy for all B tokens.
            reader = AsyncTokenReader(B, next_logits.shape[-1], self.device)
            reader.issue(next_tokens, next_logits)

            gen_ids = [[] for _ in range(B)]
            # Sync for first token.
            token_list = reader.get_tokens()
            finished = [False] * B
            for b in range(B):
                gen_ids[b].append(token_list[b])
                if token_list[b] == eos_id:
                    finished[b] = True

            # Decode loop: all sequences continue in parallel
            # Pre-allocate mask buffer to avoid torch.cat per step (CPU spike fix).
            max_total = max_prompt_len + self.max_gen_tokens
            mask_buf = torch.zeros(B, max_total, dtype=torch.long, device=self.device)
            mask_buf[:, :max_prompt_len] = attention_mask
            step = 0
            while step < self.max_gen_tokens and not all(finished):
                cur = next_tokens.unsqueeze(-1)  # (B, 1)
                mask_buf[:, max_prompt_len + step] = 1
                # Launch forward (GPU starts working).
                logits, _, past_kvs = self.model(
                    cur, past_key_values=past_kvs, use_cache=True,
                    attention_mask=mask_buf[:, :max_prompt_len + step + 1])
                next_logits = logits[:, -1]  # (B, vocab)
                next_tokens = self._sample_batch(next_logits)  # (B,)

                # Async D2H: non-blocking copy of tokens.
                reader.issue(next_tokens, next_logits)

                # Sync to get tokens (GPU finished copy by now — near-free).
                token_list = reader.get_tokens()
                for b in range(B):
                    if not finished[b]:
                        gen_ids[b].append(token_list[b])
                        if token_list[b] == eos_id:
                            finished[b] = True

                step += 1

            del past_kvs

        # Decode each sequence
        completions = []
        for b in range(B):
            text = self.tokenizer.decode(
                torch.tensor(gen_ids[b]), skip_special_tokens=True)
            completions.append(text)
            if self.gen_trace is not None:
                raw = self.tokenizer.decode(
                    torch.tensor(gen_ids[b]), skip_special_tokens=False)
                self.gen_trace.record(
                    phase=self._trace_phase, prompt=prompts[b],
                    rendered=prompts[b], raw=raw, reply=text,
                    gen_ids=gen_ids[b], temperature=self.temperature,
                    max_new_tokens=self.max_gen_tokens)
        return completions

    def _sample_batch(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next tokens for a batch. logits: (B, vocab) -> (B,)."""
        if self.temperature <= 0.0:
            return logits.argmax(dim=-1)

        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < probs.shape[-1]:
            topk_vals, _ = torch.topk(probs, self.top_k, dim=-1)
            mask = probs < topk_vals[:, -1:]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if 0.0 < self.top_p < 1.0:
            # Per-row top-p filtering
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            keep = cumsum <= self.top_p
            keep[..., 0] = True  # always keep at least one
            probs = torch.zeros_like(probs, dtype=probs.dtype)
            probs.scatter_(-1, sorted_idx, (sorted_probs * keep.to(probs.dtype)))
            # Safety: avoid division by zero
            row_sums = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sums > 0,
                                probs / row_sums.clamp(min=1e-8),
                                torch.softmax(scaled, dim=-1))

        # Per-sequence sampling with different seeds to prevent cloning.
        B = probs.shape[0]
        out = torch.empty(B, dtype=torch.long, device=probs.device)
        for b in range(B):
            seed = int(torch.randint(0, 2**31 - 1, (1,), device=probs.device).item())
            gen = torch.Generator(device=probs.device)
            gen.manual_seed(seed)
            out[b] = torch.multinomial(probs[b], num_samples=1, generator=gen).squeeze()
        return out

    def _propose_prompt(self, domain: str, difficulty: str,
                        sig_hint: str, desc_hint: str) -> str:
        """Build the proposal prompt for the active generation mode.

        Engine mode: instruction-style (CHAT_PROPOSE_PROMPT) — the model
        replies with a complete ```python task spec. Legacy mode: raw
        completion prompt ending in an open `def solve` stub.
        """
        if self.engine is not None:
            return CHAT_PROPOSE_PROMPT.format(
                domain=domain, difficulty=difficulty,
                signature=sig_hint, desc_hint=desc_hint)
        return INDUCTION_PROMPT.format(
            domain=domain, difficulty=difficulty,
            signature=sig_hint, desc_hint=desc_hint)

    def _propose_one(self, domain: str, difficulty: str,
                     mode: str = "induction") -> ProposedTask | None:
        """Generate one task via the model and parse it."""
        self._trace_phase = "propose"
        # Pick a random signature pattern to guide the model
        sig_hint = self._random_signature(domain)

        # Generate concrete test case examples for the prompt template
        # These are EXAMPLES — the model will replace them with its own
        _t_args, _t_expected = self._example_test_cases(domain, sig_hint)

        # Description hint — challenging, non-trivial tasks that push the model
        desc_hints = {
            "algorithms": [
                "Find the k-th smallest element using partitioning (not sorted())",
                "Detect if a list contains a cycle using Floyd's algorithm",
                "Merge two sorted lists into one sorted list (not using sorted())",
                "Find the longest increasing subsequence length",
                "Compute the running median of a stream",
                "Rotate a matrix 90 degrees in-place",
                "Find all pairs that sum to a target (no nested loops if possible)",
                "Implement binary search on a rotated sorted array",
                "Count inversions in a list using merge sort",
                "Find the majority element (appears > n/2 times) in O(n) space",
            ],
            "math": [
                "Compute the n-th prime number using the Sieve of Eratosthenes",
                "Find all prime factors of n (not just check if prime)",
                "Compute the digital root of n (recursive sum until single digit)",
                "Check if n is a perfect number (sum of proper divisors equals n)",
                "Compute the n-th Catalan number",
                "Find the GCD of a list of numbers (not just two)",
                "Compute the Euler totient function phi(n)",
                "Check if n is a power of 2 using bit manipulation",
                "Compute the continued fraction representation of sqrt(n)",
                "Find the modular inverse of a mod m using extended Euclidean",
            ],
            "strings": [
                "Find the longest palindromic substring (not just check palindrome)",
                "Compress a string using run-length encoding",
                "Check if two strings are anagrams without sorting",
                "Find the first non-repeating character in a string",
                "Implement string matching using KMP algorithm",
                "Convert a string to title case handling edge cases",
                "Find the minimum window in s that contains all chars of t",
                "Validate if a string is a valid IPv4 address",
                "Count distinct palindromic substrings",
                "Implement regex for basic pattern matching (., *, +)",
            ],
            "logic": [
                "Evaluate a boolean expression string with AND/OR/NOT",
                "Check if a number is a valid Luhn checksum",
                "Determine if a Sudoku row/column is valid",
                "Check if parentheses/brackets are balanced",
                "Evaluate a simple arithmetic expression string (no eval)",
            ],
            "data_structures": [
                "Implement a min-stack (push/pop/min in O(1))",
                "Evaluate reverse polish notation",
                "Implement a circular queue with wraparound",
                "Flatten a nested list (arbitrary depth) iteratively",
                "Detect if a binary tree is a valid BST",
            ],
        }
        desc_hint = self.rng.choice(desc_hints.get(domain, desc_hints["algorithms"]))

        prompt = self._propose_prompt(domain, difficulty, sig_hint, desc_hint)

        # Generate — chat mode replies with the full spec (thinks first,
        # so raise the token floor); raw mode completes the open
        # `def solve` stub in the prompt.
        completion = self._generate(
            prompt, min_tokens=_CHAT_MIN_PROPOSE_TOKENS,
            think_budget=_CHAT_PROPOSE_THINK_BUDGET,
            close_suffix=_CHAT_PROPOSE_CLOSE,
            prefill=_CHAT_PROPOSE_PREFILL)

        # Full code = prompt + completion (the model sees the stub and
        # completes it) in raw mode; just the reply in chat mode.
        full_code = completion if self.engine is not None else prompt + completion

        # Parse the full code into a ProposedTask
        task = self._parse_proposal(full_code, domain, difficulty, mode)
        if task:
            task.raw_output = full_code  # store full code for validation
        return task

    # ─── API-driven task proposal ────────────────────────────────────

    def api_propose_tasks(self, n: int = 20,
                          domain: str = "algorithms",
                          difficulty: str = "medium",
                          ) -> list[ProposedTask]:
        """Generate tasks via distillation API teachers (Groq/DeepSeek/etc.).

        Uses AgenticDistillClient.generate_tasks() to get diverse coding tasks
        with test cases from strong external models, then converts them to
        ProposedTask format for the standard solve/verify pipeline.

        This bypasses the model's limited task vocabulary — external APIs
        provide unlimited diversity. The local model still does all solving.

        Args:
            n: number of tasks to request from the API
            domain: target domain (used for tagging, not sent to API)
            difficulty: difficulty hint (used for tagging)

        Returns:
            list of validated ProposedTask objects
        """
        from forge.distillation.agentic_distill import AgenticDistillClient

        client = AgenticDistillClient()
        raw_tasks = client.generate_tasks(n_tasks=n)
        if not raw_tasks:
            print("  [API-Propose] No tasks generated by API teachers")
            return []

        valid_tasks: list[ProposedTask] = []
        for raw in raw_tasks:
            task_desc = raw.get("task", "")
            raw_tests = raw.get("test_cases", [])

            # Convert API test_cases format to ProposedTask format.
            # API: {"input": "racecar", "output": "True"}
            #   or already-parsed: {"input": 42, "output": [2, 3, 7]}
            # Need: {"args": ("racecar",), "expected": True}
            test_cases = []
            for tc in raw_tests:
                # Handle various test_case formats from different API models:
                # - dict: {"input": ..., "output": ...}
                # - list/tuple: [input, output]
                # - str: "input -> output" or just "input"
                if isinstance(tc, dict):
                    inp = tc.get("input", "")
                    out = tc.get("output", "")
                elif isinstance(tc, (list, tuple)) and len(tc) >= 2:
                    inp, out = tc[0], tc[1]
                elif isinstance(tc, str):
                    # Try "input -> output" or "input, output" format
                    if "->" in tc:
                        parts = tc.split("->", 1)
                        inp, out = parts[0].strip(), parts[1].strip()
                    elif "," in tc:
                        parts = tc.split(",", 1)
                        inp, out = parts[0].strip(), parts[1].strip()
                    else:
                        continue  # can't parse
                else:
                    continue
                try:
                    # Parse input — handle both string and already-parsed types.
                    if isinstance(inp, str):
                        inp_stripped = inp.strip()
                        if not inp_stripped:
                            args: tuple = ()
                        elif inp_stripped.startswith("(") and inp_stripped.endswith(")"):
                            # Tuple literal — parse as multiple args
                            parsed = ast.literal_eval(inp_stripped)
                            args = parsed if isinstance(parsed, tuple) else (parsed,)
                        else:
                            # Try parsing as a single literal
                            try:
                                parsed = ast.literal_eval(inp_stripped)
                                args = (parsed,)
                            except (ValueError, SyntaxError):
                                args = (inp_stripped,)  # treat as raw string
                    else:
                        # Already a Python object (int, list, dict, etc.)
                        args = (inp,) if not isinstance(inp, tuple) else inp

                    # Parse output — handle both string and already-parsed types.
                    if isinstance(out, str):
                        out_stripped = out.strip()
                        try:
                            expected = ast.literal_eval(out_stripped)
                        except (ValueError, SyntaxError):
                            expected = out_stripped  # treat as string
                    else:
                        # Already a Python object
                        expected = out
                except Exception:
                    continue

                test_cases.append({"args": args, "expected": expected})

            if len(test_cases) < 2:
                continue  # need at least 2 test cases

            # Infer signature from test case args
            first_args = test_cases[0]["args"]
            type_hints = []
            for a in first_args:
                if isinstance(a, int):
                    type_hints.append("int")
                elif isinstance(a, str):
                    type_hints.append("str")
                elif isinstance(a, list):
                    type_hints.append("list")
                elif isinstance(a, float):
                    type_hints.append("float")
                elif isinstance(a, bool):
                    type_hints.append("bool")
                else:
                    type_hints.append("any")
            params = ", ".join(
                f"arg{i}: {t}" for i, t in enumerate(type_hints)
            ) if type_hints else ""
            ret = test_cases[0]["expected"]
            if isinstance(ret, int):
                ret_type = "int"
            elif isinstance(ret, str):
                ret_type = "str"
            elif isinstance(ret, list):
                ret_type = "list"
            elif isinstance(ret, bool):
                ret_type = "bool"
            else:
                ret_type = "any"
            signature = f"({params}) -> {ret_type}" if params else "() -> any"

            self._task_counter += 1
            task_id = f"api_{domain}_{difficulty}_{self._task_counter}"

            task = ProposedTask(
                id=task_id,
                domain=domain,
                difficulty=difficulty,
                description=task_desc,
                signature=signature,
                solve_name="solve",
                test_cases=test_cases,
                stress_index=len(test_cases) - 1 if len(test_cases) > 2 else None,
                archetype="api_generated",
                raw_output="",  # no reference impl from API
                generation_mode="api",
            )
            task.proposer_confidence = 0.5  # API tasks are pre-validated by teacher
            task.validated = True  # skip self-consistency (no ref code to check)

            # Dedup by description
            desc_key = task_desc.lower().strip()[:80]
            if desc_key in self._seen_descriptions:
                continue
            self._seen_descriptions.add(desc_key)

            valid_tasks.append(task)

        print(f"  [API-Propose] {len(valid_tasks)}/{len(raw_tasks)} tasks "
              f"validated from API teachers")
        return valid_tasks

    def _random_signature(self, domain: str) -> str:
        """Return a random function signature hint for the domain."""
        sigs = {
            "algorithms": ["(n: int) -> int", "(lst: list) -> int",
                          "(lst: list) -> list", "(n: int) -> list",
                          "(a: int, b: int) -> int"],
            "math": ["(n: int) -> int", "(x: int, y: int) -> int",
                    "(n: int) -> bool", "(n: int) -> float"],
            "strings": ["(s: str) -> str", "(s: str) -> int",
                       "(s: str, t: str) -> str", "(s: str) -> bool"],
            "logic": ["(n: int) -> bool", "(a: int, b: int) -> bool",
                     "(lst: list) -> bool"],
            "data_structures": ["(lst: list) -> list", "(lst: list, n: int) -> list",
                               "(lst: list) -> int", "(matrix: list) -> list"],
        }
        return self.rng.choice(sigs.get(domain, sigs["algorithms"]))

    def _example_test_cases(self, domain: str, signature: str) -> tuple[list[str], list[str]]:
        """Generate example test case strings for the prompt template.

        These are PLACEHOLDER examples that show the format — the model
        is expected to replace them with its own test cases.
        """
        # Parse the signature to know how many args
        # Simple heuristic based on domain
        if "lst" in signature:
            args = ["[1, 2, 3]", "[5]", "[]", "[3, 1, 4, 1, 5]"]
            expected = ["6", "5", "0", "14"]
        elif "s" in signature and "str" in signature:
            args = ["'hello'", "''", "'test'", "'abcabc'"]
            expected = ["5", "0", "4", "6"]
        elif "a" in signature and "b" in signature:
            args = ["3, 5", "0, 0", "10, 20", "7, 3"]
            expected = ["8", "0", "30", "10"]
        elif "n" in signature and "bool" in signature:
            args = ["7", "1", "10", "13"]
            expected = ["True", "False", "False", "True"]
        else:
            args = ["5", "0", "10", "20"]
            expected = ["25", "0", "100", "400"]
        return args, expected

    def _think_cap(self, budget: int, close_suffix: str = _CHAT_SOLVE_CLOSE):
        """Per-sequence think-cap + structural-ban logits processor.

        Reuses the GUI's own ``chat_loop._think_cap_processor``: while the
        think block is open it bans <tool_call>; after ``budget`` think
        tokens without a ``</think>`` it force-injects ``close_suffix``
        (one token per step — "\n</think>\n```python\n…" drops the model
        straight into code-completion mode); and it always bans
        im_start/tool markers/<think> re-open. With
        ``tool_calls_allowed=False`` the ban set is exactly the self-play
        structural set {518, 531, 532, 539, 540, 541}.
        """
        from forge_gui_server.services.chat_loop import _THINK_END_ID, _think_cap_processor
        ids = self._think_suffix_ids.get(close_suffix)
        if ids is None:
            ids = self.tokenizer.encode(
                close_suffix, add_special_tokens=False)
            self._think_suffix_ids[close_suffix] = ids
        cap = _think_cap_processor(
            budget, ids, tool_calls_allowed=False)
        banned = self._banned_special_ids()

        # Post-think fence deadline — when the model self-closes </think>
        # then muses instead of emitting a code block, force the fence part
        # of the close suffix after _POST_THINK_FENCE_DEADLINE fenceless
        # answer tokens.
        fence_suffix = close_suffix.split("</think>", 1)[-1]
        fence_ids = self._think_suffix_ids.get(fence_suffix)
        if fence_ids is None:
            fence_ids = self.tokenizer.encode(
                fence_suffix, add_special_tokens=False)
            self._think_suffix_ids[fence_suffix] = fence_ids
        try:
            fence_start = self.tokenizer.encode(
                "```", add_special_tokens=False)[0]
        except Exception:
            fence_start = None
        extra_queue: list[int] = []

        def proc(logits, generated_ids):
            masked = cap(logits, generated_ids)
            if banned:
                masked[..., banned] = float("-inf")
            if extra_queue:
                tid = extra_queue.pop(0)
                forced = torch.full_like(masked, float("-inf"))
                forced[..., tid] = logits[..., tid]
                return forced
            if fence_start is not None and _THINK_END_ID in generated_ids:
                last = (len(generated_ids) - 1
                        - generated_ids[::-1].index(_THINK_END_ID))
                post = generated_ids[last + 1:]
                if (len(post) >= _POST_THINK_FENCE_DEADLINE
                        and fence_start not in post):
                    extra_queue.extend(fence_ids)
            return masked
        return proc

    def _banned_special_ids(self) -> list[int]:
        """Every control-looking vocab id minus the legitimate terminators
        in _CONTROL_TOKEN_KEEP. One-time scan — the reserved-token class is
        too large to enumerate in the GUI's six-id structural set."""
        if self._banned_ids is None:
            try:
                vocab = self.tokenizer.get_vocab()
            except Exception:
                vocab = {}
            self._banned_ids = [
                tid for tok, tid in vocab.items()
                if tid not in _CONTROL_TOKEN_KEEP
                and _RE_CONTROL_TOKEN.match(tok)]
        return self._banned_ids

    def _generate(self, prompt: str,
                  min_tokens: int = _CHAT_MIN_SOLVE_TOKENS,
                  think_budget: int = _CHAT_SOLVE_THINK_BUDGET,
                  close_suffix: str = _CHAT_SOLVE_CLOSE,
                  prefill: str | None = None) -> str:
        """Generate text — through the ForgeEngine when attached.

        Engine path: ChatML think-mode render via generate_raw (activated
        features: prefix cache, mtp_selfspec, spectral KV, DRY, think-cap +
        structural token bans, EOS on <|im_end|>=519/<|endoftext|>=2); the
        reply is reasoning-stripped via _strip_reply. ``min_tokens`` raises
        the gen cap because chat replies carry the <think> block. Legacy
        path: raw model() with KV cache + async D2H.
        """
        if self.engine is not None:
            rendered = self._render_chat(prompt, prefill=prefill)
            # RoPE tables are built for 16384 — a runaway task prompt
            # (observed: 78k tokens) crashes _apply_rope. Cap at 12000:
            # keep the system/header head + task tail, drop the middle.
            ids = self.engine.tokenizer.encode(
                rendered, add_special_tokens=False)
            if len(ids) > 12000:
                n_orig = len(ids)
                ids = ids[:4000] + ids[-8000:]
                rendered = self.engine.tokenizer.decode(
                    ids, skip_special_tokens=False)
                print(f"  [Curriculum] prompt {n_orig} toks > 12000 — "
                      f"truncated (head+tail kept)")
            raw = self.engine.generate_raw(
                rendered,
                max_new_tokens=max(self.max_gen_tokens, min_tokens),
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=1.05,
                logits_processor=self._think_cap(
                    None if prefill else think_budget, close_suffix),
                eos_token_ids=[2, 519],
                # Keep </think> visible — _strip_reply needs it for the
                # reasoning/content split (GUI _split_reasoning parity).
                skip_special_tokens=False,
            )
            reply = _strip_reply(raw)
            if prefill:
                reply = prefill + re.sub(r'^\s*```\w*\s*\n?', '', reply)
            if self.gen_trace is not None:
                self.gen_trace.record(
                    phase=self._trace_phase, prompt=prompt,
                    rendered=rendered, raw=raw, reply=reply,
                    temperature=self.temperature,
                    max_new_tokens=max(self.max_gen_tokens, min_tokens))
            return reply

        enc = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=512)
        input_ids = enc.input_ids.to(self.device)

        with torch.no_grad():
            past_kvs = None
            gen_ids = []
            eos_id = self.tokenizer.eos_token_id

            # Initial pass on full prompt
            logits, _, past_kvs = self.model(
                input_ids, past_key_values=past_kvs, use_cache=True)
            next_logits = logits[0, -1]
            next_token_gpu = self._sample_gpu(next_logits)

            # Async D2H: issue non-blocking copy instead of blocking .item().
            reader = AsyncTokenReader(1, next_logits.shape[-1], self.device)
            # _sample_gpu returns (1,) for single-token; issue expects (B,)
            token_1d = next_token_gpu.squeeze() if next_token_gpu.dim() > 1 else next_token_gpu
            reader.issue(token_1d, next_logits.unsqueeze(0))

            # Pre-allocate single-token buffer.
            cur_token = torch.zeros(1, 1, dtype=input_ids.dtype, device=self.device)

            # Sync for first token.
            token_list = reader.get_tokens()
            next_token = token_list[0]
            gen_ids.append(next_token)

            # Subsequent tokens: feed only new token, reuse KV cache.
            # Async D2H pattern: issue copy, launch forward, sync for previous result.
            while len(gen_ids) < self.max_gen_tokens:
                if next_token == eos_id:
                    break
                cur_token[0, 0] = next_token_gpu
                # Launch forward (GPU starts working).
                logits, _, past_kvs = self.model(
                    cur_token, past_key_values=past_kvs, use_cache=True)
                next_logits = logits[0, -1]
                next_token_gpu = self._sample_gpu(next_logits)

                # Issue async D2H (non-blocking).
                token_1d = next_token_gpu.squeeze() if next_token_gpu.dim() > 1 else next_token_gpu
                reader.issue(token_1d, next_logits.unsqueeze(0))

                # Sync to get token (GPU finished copy by now — near-free).
                token_list = reader.get_tokens()
                next_token = token_list[0]
                gen_ids.append(next_token)

            del past_kvs

        text = self.tokenizer.decode(
            torch.tensor(gen_ids), skip_special_tokens=True)
        if self.gen_trace is not None:
            raw = self.tokenizer.decode(
                torch.tensor(gen_ids), skip_special_tokens=False)
            self.gen_trace.record(
                phase=self._trace_phase, prompt=prompt, rendered=prompt,
                raw=raw, reply=text, gen_ids=gen_ids,
                temperature=self.temperature,
                max_new_tokens=self.max_gen_tokens)
        return text

    def _sample_gpu(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token, keeping result on GPU (no .item() sync).

        Unlike _sample() which returns an int (forcing D2H sync), this returns
        a GPU tensor for use with async D2H transfer.
        """
        if self.temperature <= 0.0:
            return logits.argmax()

        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < probs.shape[0]:
            topk_vals, _ = torch.topk(probs, self.top_k)
            mask = probs < topk_vals[-1]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum()

        if 0.0 < self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = sorted_probs.cumsum(0)
            cutoff = (cumsum > self.top_p).to(sorted_probs.dtype).cumsum(0)
            sorted_probs = sorted_probs * (cutoff == 0).to(sorted_probs.dtype)
            # Safety: if all probs got filtered, fall back to top-1
            total = sorted_probs.sum()
            if total > 0:
                sorted_probs = sorted_probs / total
                probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
            # else: keep top-k filtered probs (already normalized)

        # Safety: ensure probs sum > 0 and no NaNs before multinomial
        if probs.sum() <= 0 or torch.isnan(probs).any():
            probs = torch.softmax(scaled, dim=-1)  # fall back to raw softmax

        return torch.multinomial(probs, 1)

    def _sample(self, logits: torch.Tensor) -> int:
        """Sample next token with temperature/top-k/top-p."""
        if self.temperature <= 0.0:
            return logits.argmax().item()

        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < probs.shape[0]:
            topk_vals, _ = torch.topk(probs, self.top_k)
            mask = probs < topk_vals[-1]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum()

        if 0.0 < self.top_p < 1.0:
            from research.sampling_utils import top_p_sample_probs
            probs = top_p_sample_probs(probs, self.top_p)

        return torch.multinomial(probs, num_samples=1).squeeze().item()

    def _parse_proposal(self, raw: str, domain: str, difficulty: str,
                        mode: str) -> ProposedTask | None:
        """Parse model output into a ProposedTask.

        Expected format:
          # Task: <description>
          def solve(...):
              ...
          # solve(args1) == expected1
          # solve(args2) == expected2
        """
        # Extract code block if wrapped in ```python ... ```
        code_match = _RE_PYTHON_BLOCK.search(raw)
        code = code_match.group(1) if code_match else raw

        # Extract task description — try "# Task:" or quoted description
        desc_match = _RE_TASK_DESC.search(code)
        if not desc_match:
            # Try quoted: "Given a list..." or 'Given a list...'
            desc_match = _RE_QUOTED.search(code)
        description = desc_match.group(1).strip() if desc_match else "Model-proposed task"
        # A code-looking or empty description (model skipped the '# Task:'
        # line) collides in the clone guard — make it unique.
        if not description or re.match(r'^(def|#|```)', description):
            description = (f"{domain} {difficulty} task "
                           f"#{self._task_counter + 1}")

        # Extract function signature — if no def solve found, infer from test cases
        sig_match = _RE_SOLVE_SIG.search(code)
        if sig_match:
            signature = sig_match.group(1)
        else:
            # Infer signature from first test case args
            # Will be set after test cases are parsed
            signature = None

        # Extract test cases — handle multiple formats:
        #   Format 1: # solve(args) == expected
        #   Format 2: assert solve(args) == expected
        #   Format 3: # assert solve(args) == expected  (commented assert)
        test_cases = []
        # Pattern 1: comment-style # solve(args) == expected
        for m in _RE_SOLVE_TEST.finditer(code):
            args_str = m.group(1).strip()
            expected_str = m.group(2).strip().rstrip(',')
            try:
                args = ast.literal_eval(f"({args_str},)")
                expected = ast.literal_eval(expected_str)
                test_cases.append({"args": args, "expected": expected})
            except Exception:
                continue
        # Pattern 2: assert-style (active or commented)
        if not test_cases:
            for m in _RE_ASSERT_TEST.finditer(code):
                args_str = m.group(1).strip()
                expected_str = m.group(2).strip().rstrip(',')
                try:
                    args = ast.literal_eval(f"({args_str},)")
                    expected = ast.literal_eval(expected_str)
                    test_cases.append({"args": args, "expected": expected})
                except Exception:
                    continue
        # Pattern 3: print-style — print(solve(args))  # expected
        # (chat models emit this often despite the requested format)
        if not test_cases:
            for m in _RE_PRINT_TEST.finditer(code):
                args_str = m.group(1).strip()
                expected_str = m.group(2).strip().rstrip(',')
                try:
                    args = ast.literal_eval(f"({args_str},)")
                    expected = ast.literal_eval(expected_str)
                    test_cases.append({"args": args, "expected": expected})
                except Exception:
                    continue

        # Bare inputs: "# solve(<args>)" lines with no declared expected.
        # AZR-style — the executor derives outputs during validation, so a
        # proposal only needs runnable code + inputs to be usable.
        pending_inputs = []
        for m in _RE_SOLVE_INPUT.finditer(code):
            args_str = m.group(1).strip()
            try:
                pending_inputs.append(ast.literal_eval(f"({args_str},)"))
            except Exception:
                continue
        for m in _RE_SOLVE_BARE.finditer(code):
            args_str = m.group(1).strip()
            try:
                pending_inputs.append(ast.literal_eval(f"({args_str},)"))
            except Exception:
                continue

        if len(test_cases) + len(pending_inputs) < 2:
            return None  # need at least 2 test cases/inputs

        # Infer signature from test case/input args if not found above
        if signature is None:
            first_args = (test_cases[0]["args"] if test_cases
                          else pending_inputs[0])
            type_hints = []
            for a in first_args:
                if isinstance(a, int):
                    type_hints.append("int")
                elif isinstance(a, str):
                    type_hints.append("str")
                elif isinstance(a, list):
                    type_hints.append("list")
                elif isinstance(a, float):
                    type_hints.append("float")
                elif isinstance(a, bool):
                    type_hints.append("bool")
                else:
                    type_hints.append("any")
            params = ", ".join(f"arg{i}: {t}" for i, t in enumerate(type_hints))
            # Infer return type from first expected (may be unknown when
            # all inputs are pending — executor derives them later)
            ret = test_cases[0]["expected"] if test_cases else None
            if isinstance(ret, int):
                ret_type = "int"
            elif isinstance(ret, str):
                ret_type = "str"
            elif isinstance(ret, list):
                ret_type = "list"
            elif isinstance(ret, bool):
                ret_type = "bool"
            else:
                ret_type = "any"
            signature = f"({params}) -> {ret_type}"

        # Extract the reference implementation (for validation)
        # Match from `def solve(...)` to the next assert/#/def or end
        impl_match = _RE_SOLVE_IMPL.search(code)
        impl_match.group(1) if impl_match else ""

        self._task_counter += 1
        task_id = f"azr_{domain}_{difficulty}_{self._task_counter}"

        return ProposedTask(
            id=task_id,
            domain=domain,
            difficulty=difficulty,
            description=description,
            signature=signature,
            solve_name="solve",
            test_cases=test_cases,
            pending_inputs=pending_inputs,
            stress_index=(len(test_cases) + len(pending_inputs) - 1
                          if len(test_cases) + len(pending_inputs) > 2
                          else None),
            archetype="model_proposed",
            raw_output=raw,
            generation_mode=mode,
        )

    # ─── Validation ────────────────────────────────────────────────

    def _validate_task(self, task: ProposedTask) -> bool:
        """Validate that the task is well-formed.

        In push-limits mode, we accept tasks even when the proposer's solution
        has bugs — the solver will determine if the task is solvable.
        We only reject tasks with:
        - No test cases or inputs
        - Syntax errors in the reference code (can't even parse)
        - Reference code that crashes on every proposed input

        AZR-style: when the reference implementation runs, the EXECUTOR is
        the source of truth — expected outputs are derived by running
        ``solve(*args)`` on the proposed inputs, not trusted from the
        model's own declarations.
        """
        if not task.test_cases and not task.pending_inputs:
            return False

        # Extract reference implementation from raw output
        ref_code = self._extract_reference_code(task.raw_output)
        if ref_code:
            derived = self._derive_outputs(ref_code, task)
            if derived:
                task.test_cases = derived
                task.pending_inputs = []
                task.proposer_confidence = 1.0
                return True
        if not task.test_cases:
            return False
        if not ref_code:
            # No reference code — accept with lower confidence
            task.proposer_confidence = 0.3
            return True

        # Try to validate: run ref code + check test cases
        test_lines = []
        for i, tc in enumerate(task.test_cases):
            args_str = ", ".join(repr(a) for a in tc["args"])
            expected_repr = repr(tc["expected"])
            test_lines.append(
                f"    result_{i} = solve({args_str})\n"
                f"    assert result_{i} == {expected_repr}, "
                f"Test {i}: solve({args_str}) = {{result_{i}!r}}, expected {expected_repr}"
            )

        validation_code = ref_code + "\n" + "\n".join(test_lines) + "\nprint('VALID')"

        # Execute in sandbox
        result = self._execute_sandbox(validation_code, timeout_s=3.0)
        if result.get("returncode") == 0 and "VALID" in result.get("stdout", ""):
            task.proposer_confidence = 1.0  # validated — high confidence
            return True

        # Proposer's solution failed validation — but the test cases might still be valid.
        # Check if the reference code at least parses (no syntax errors).
        try:
            compile(ref_code, "<proposer>", "exec")
            # Code parses but fails tests — accept with lower confidence.
            # The solver will try to solve it; if it can, we train on it.
            task.proposer_confidence = 0.4
            return True
        except SyntaxError:
            # Syntax error — the task is malformed, reject
            return False

    def _derive_outputs(self, ref_code: str,
                        task: ProposedTask) -> list[dict] | None:
        """Derive expected outputs by executing the reference impl.

        AZR semantics: the model proposes ``(program, inputs)``; the
        executor computes ``o = solve(*args)`` and THAT is the ground
        truth — the model never has to correctly predict its own outputs.
        Runs each input in a separate sandbox (one bad input can't kill
        the task). Returns ≥2 verified test cases or None.
        """
        all_inputs = ([tc["args"] for tc in task.test_cases]
                      + list(task.pending_inputs))
        if len(all_inputs) < 2:
            return None
        codes = []
        for inp in all_inputs:
            args_str = ", ".join(repr(a) for a in inp)
            codes.append(ref_code + f"\nprint(repr(solve({args_str})))")
        results = self._execute_sandbox_batch(codes, timeout_s=3.0)
        out = []
        for inp, res in zip(all_inputs, results):
            if res.get("returncode") != 0:
                continue
            lines = res.get("stdout", "").strip().splitlines()
            if not lines:
                continue
            try:
                expected = ast.literal_eval(lines[-1].strip())
            except Exception:
                continue  # non-literal output — can't verify downstream
            out.append({"args": inp, "expected": expected})
        return out if len(out) >= 2 else None

    def _is_complex_enough(self, task: ProposedTask) -> bool:
        """Check if a task is complex enough to be worth training on (push-limits).

        Rejects:
        - Too few test cases (< MIN_TEST_CASES)
        - Solutions that are just a single builtin call (sum, max, len, etc.)
        - Solutions with no branching (no if/for/while)
        - Tasks with trivial signatures (no arguments)
        - Identity mappings (all test cases: solve(x) == x) — proposer gaming
        - Constant mappings (all test cases: solve(x) == c) — proposer gaming
        """
        # Minimum test cases — count pending inputs too: input-only
        # proposals carry their cases in pending_inputs until
        # _validate_task derives outputs (this check runs BEFORE
        # validation, so tests=0 pending tasks were wrongly rejected).
        if (len(task.test_cases) + len(task.pending_inputs)
                < self.MIN_TEST_CASES):
            return False

        # Check signature has at least 1 argument
        sig_clean = task.signature.split("->")[0].strip().strip("()")
        if not sig_clean or sig_clean == "":
            return False

        # Reject identity mappings: all test cases where args[0] == expected
        # This catches the proposer gaming with `return n` tasks
        if len(task.test_cases) >= 3:
            identity_count = sum(
                1 for tc in task.test_cases
                if tc.get("args") and tc["args"][0] == tc.get("expected")
            )
            if identity_count >= len(task.test_cases) * 0.8:
                return False  # identity mapping — trivial, no learning value

        # Reject constant mappings: all test cases return the same constant
        if len(task.test_cases) >= 3:
            expected_vals = [tc.get("expected") for tc in task.test_cases]
            if len(set(repr(v) for v in expected_vals)) == 1:
                return False  # constant mapping — trivial

        # Check the reference implementation for complexity
        ref_code = self._extract_reference_code(task.raw_output)
        if ref_code:
            # Reject if the solution body is just a single builtin call
            # e.g., "return sum(lst)" or "return max(lst)"
            body = ref_code.split(":", 1)[-1].strip() if ":" in ref_code else ref_code
            # Remove common whitespace
            body_lines = [l.strip() for l in body.split("\n") if l.strip()]
            if body_lines:
                # Check if it's a single-line return of a builtin
                single_line = " ".join(body_lines)
                # Extract what's being returned
                ret_match = _RE_RETURN.search(single_line)
                if ret_match:
                    ret_expr = ret_match.group(1).strip()
                    # Check if it's just a builtin call: builtin(args)
                    builtin_match = re.match(
                        r'^(' + '|'.join(self.BANNED_PATTERNS) + r')\s*\(.+\)$',
                        ret_expr)
                    if builtin_match:
                        return False  # trivial: just a builtin wrapper
                    # Reject bare `return n` or `return x` (identity return)
                    if re.match(r'^return\s+[a-z_]\w*$', ret_expr) and len(body_lines) <= 2:
                        return False  # trivial: just returning the argument

                # Check for branching (if/for/while) — encourage but don't require
                # (the model often generates test cases without reference code)
                has_branch = any(
                    kw in ref_code
                    for kw in ["if ", "for ", "while ", "elif ", "and ", "or "]
                )
                # Only reject if NO branching AND the solution is a single line
                if not has_branch and len(body_lines) <= 2:
                    return False  # trivial: single-line no-branch solution

        return True

    def _extract_reference_code(self, raw: str) -> str:
        """Extract the def solve(...) implementation from raw model output."""
        # Remove code block markers
        code = _RE_CODEBLOCK_OPEN.sub('', raw)
        code = _RE_CODEBLOCK_CLOSE.sub('', code)

        # Find the function definition — stop at assert/#/def or end
        match = _RE_SOLVE_IMPL.search(code)
        if match:
            return match.group(1).strip()
        return ""

    def _execute_sandbox(self, code: str, timeout_s: float = 3.0) -> dict:
        """Execute Python code in a sandboxed one-shot worker (audit hook +
        jail dir + env scrub — model-generated code never runs unrestricted).
        """
        temp_path = None
        once_path = None
        result_path = None
        try:
            from research.persistent_executor import (
                assign_to_job, build_once_script, close_job, scrub_environ)
            scratch = tempfile.gettempdir()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                              delete=False, encoding='utf-8',
                                              dir=scratch) as f:
                f.write(code)
                temp_path = f.name
            once_path = temp_path + '_once.py'
            result_path = temp_path + '_result.json'
            with open(once_path, 'w', encoding='utf-8') as f:
                f.write(build_once_script())

            proc = subprocess.Popen(
                [sys.executable, once_path, scratch, scratch,
                 '--once', temp_path, result_path],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=scratch, env=scrub_environ())
            job = None
            handle = getattr(proc, '_handle', None)
            if handle is not None:
                job = assign_to_job(handle, allow_children=True)
            try:
                rc = proc.wait(timeout=timeout_s)
                if os.path.exists(result_path):
                    import json as _json
                    with open(result_path, encoding='utf-8') as f:
                        return _json.load(f)
                return {"stdout": "", "stderr": "sandbox produced no result",
                        "returncode": rc if rc is not None else -1}
            except subprocess.TimeoutExpired:
                proc.kill()
                return {"stdout": "", "stderr": "TIMEOUT", "returncode": -1}
            finally:
                close_job(job)
        except Exception as e:
            return {"stdout": "", "stderr": f"sandbox error: {e}",
                    "returncode": -1}
        finally:
            for p in (temp_path, once_path, result_path):
                if p is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(p)

    def _execute_sandbox_batch(self, codes: list[str],
                               timeout_s: float = 3.0) -> list[dict]:
        """Execute multiple Python code snippets in parallel via the persistent
        sandbox worker pool.

        Returns a list of result dicts in the same order as *codes*.
        """
        pool = _get_sandbox_pool()
        results = pool.map(
            lambda c: self._execute_sandbox(c, timeout_s=timeout_s), codes)
        return list(results)

    # ─── Solving ───────────────────────────────────────────────────

    def solve_task(self, task: ProposedTask,
                   engine: Any | None = None) -> dict:
        """Solve a proposed task using the model.

        Can use the existing RecursiveSelfPlay engine, or generate directly.
        Returns success/failure + solution code.
        """
        goal_task = self._to_goal_task(task)

        if engine is not None:
            # Use existing recursive self-play engine
            result = engine.run_goal_task(goal_task, k_samples=1, use_reasoning=True)
            return result
        else:
            # Direct generation (simpler, no retry loop)
            return self._solve_direct(goal_task)

    @staticmethod
    def _to_goal_task(task: ProposedTask) -> "GoalTask":
        """Convert ProposedTask to GoalTask for solver compatibility."""
        return GoalTask(
            id=task.id,
            domain=task.domain,
            difficulty=task.difficulty,
            description=task.description,
            input_signature=task.signature,
            solve_name=task.solve_name,
            test_cases=task.test_cases,
            stress_index=task.stress_index,
            archetype=task.archetype,
        )

    def _solve_prompt_text(self, task: "GoalTask") -> str:
        """Solve prompt for the active generation mode.

        Engine mode: instruction asking for the complete function in a
        ```python block (rendered as ChatML by _generate). Legacy mode:
        raw completion prompt ending in an open `def solve` stub.
        """
        from forge.evaluation.goal_tasks import build_goal_prompt
        if self.engine is None:
            return build_goal_prompt(task)
        test_lines = "\n".join(
            f"#   {task.solve_name}({', '.join(repr(a) for a in tc['args'])})"
            f" == {tc['expected']!r}"
            for tc in task.test_cases)
        return CHAT_SOLVE_PROMPT.format(
            description=task.description,
            solve_name=task.solve_name,
            sig_params=task.input_signature.split(" -> ")[0],
            test_lines=test_lines)

    def _assemble_solution(self, task: "GoalTask", prompt: str,
                           completion: str) -> str:
        """Turn a raw completion into an executable solution string."""
        code_match = _RE_PYTHON_BLOCK.search(completion)
        body = code_match.group(1) if code_match else completion
        if code_match is None:
            # Prefilled/unfenced reply — drop a dangling fence opener line
            # so it can't SyntaxError the assembled solution.
            body = re.sub(r'^\s*```\w*\s*\n', '', body)
        # Drop musing comments/docstrings — reasoning leaked into code
        # comments bloats the SFT/GRPO data (the "overthinking" leak).
        cleaned = _clean_code(body)
        if cleaned:
            body = cleaned
        if self.engine is not None:
            # Chat reply should carry the whole function; if the model
            # returned a bare body, wrap it under the requested signature.
            if re.search(r"def\s+" + re.escape(task.solve_name) + r"\s*\(",
                         body):
                return body
            sig = task.input_signature.split(" -> ")[0]
            indented = "\n".join(
                "    " + ln if ln.strip() else ln
                for ln in body.strip("\n").split("\n"))
            return f"def {task.solve_name}{sig}:\n{indented}"
        # Raw mode: prompt ends mid-function; completion is the body.
        return prompt + body

    def _verify_solution(self, task: "GoalTask",
                         full_solution: str) -> tuple[bool, dict]:
        """Run all test cases against a solution; (success, first_result)."""
        test_codes = []
        for tc in task.test_cases:
            args_str = ", ".join(repr(a) for a in tc["args"])
            expected_repr = repr(tc["expected"])
            tc_code = (
                full_solution + "\n"
                f"assert {task.solve_name}({args_str}) == {expected_repr}, "
                f"'{task.solve_name}({args_str}) failed'\n"
                "print('PASS')"
            )
            test_codes.append(tc_code)

        if not test_codes:
            return False, {"stdout": "", "stderr": "NO_TESTS",
                           "returncode": -1}
        if len(test_codes) == 1:
            result = self._execute_sandbox(test_codes[0], timeout_s=5.0)
            success = (result.get("returncode") == 0
                       and "PASS" in result.get("stdout", ""))
            return success, result
        tc_results = self._execute_sandbox_batch(test_codes, timeout_s=5.0)
        success = all(
            r.get("returncode") == 0 and "PASS" in r.get("stdout", "")
            for r in tc_results
        )
        # Aggregate error from first failing test case
        result = next(
            (r for r in tc_results if not (
                r.get("returncode") == 0 and "PASS" in r.get("stdout", ""))),
            tc_results[0],
        )
        return success, result

    def _solve_direct(self, task: GoalTask) -> dict:
        """Solve a task with self-modeling retry (D5).

        On low confidence, retry with higher temperature.
        Accept the best attempt (highest confidence or first that passes tests).
        """
        prompt = self._solve_prompt_text(task)
        self._trace_phase = "solve"

        attempts = []
        best_solution = None
        best_success = False

        for attempt in range(self.retry_policy.max_retries + 1):
            # Adjust temperature for retry (more exploration)
            old_temp = self.temperature
            self.temperature = self.retry_policy.temperature_for_attempt(attempt)

            completion = self._generate(prompt, prefill=_CHAT_SOLVE_PREFILL)
            self.temperature = old_temp  # restore

            full_solution = self._assemble_solution(task, prompt, completion)
            success, result = self._verify_solution(task, full_solution)

            attempts.append({
                "code": full_solution, "success": success,
                "error": result.get("stderr", ""),
                "round": 0, "sample": attempt,
                "temperature": self.retry_policy.temperature_for_attempt(attempt),
            })

            if success:
                best_solution = full_solution
                best_success = True
                break  # passed — no need to retry

            if best_solution is None:
                best_solution = full_solution  # keep first attempt as fallback

            # Self-modeling: check confidence to decide retry
            # (simplified — we use test failure as the error signal)
            # If this was the last attempt, stop
            if attempt >= self.retry_policy.max_retries:
                break

        return {
            "final_success": best_success,
            "attempts": attempts,
            "rounds_used": len(attempts),
            "best_quality": 1.0 if best_success else 0.0,
        }

    def solve_tasks_batch(self, tasks: list[ProposedTask],
                          group_size: int = 1) -> list[tuple]:
        """Solve all tasks with one concurrent batched generation pass.

        First attempts for every task go through a single
        ``engine.generate_batch`` call (all prompts decode in parallel —
        replaces the serial per-task generate loop). Failed tasks then get
        their temperature-bumped retries serially. Works without an engine
        too (falls back to the built-in raw batcher).

        ``group_size`` > 1 (engine path only) generates k independent
        rollouts per task prompt in the same batch — GRPO needs
        within-group reward variance, and pairing completions of
        DIFFERENT tasks collapses to zero advantage (observed: rewards
        [1,1] → loss 0 → no learning).

        Returns a list of ``(task, result_dict, elapsed_ms)`` in the same
        order as *tasks*.
        """
        if not tasks:
            return []
        self._trace_phase = "solve"
        goal_tasks = [self._to_goal_task(t) for t in tasks]
        prompts = [self._solve_prompt_text(t) for t in goal_tasks]

        t_batch = time.time()
        if group_size > 1 and self.engine is not None:
            ex_prompts = [p for p in prompts for _ in range(group_size)]
            seeds = [7919 * i + 13 for i in range(len(ex_prompts))]
            flat = self._generate_batch(
                ex_prompts, prefill=_CHAT_SOLVE_PREFILL, seeds=seeds)
            completions = [flat[i * group_size:(i + 1) * group_size]
                           for i in range(len(tasks))]
        else:
            flat = self._generate_batch(prompts, prefill=_CHAT_SOLVE_PREFILL)
            completions = [[c] for c in flat]
        batch_ms = (time.time() - t_batch) * 1000.0

        results = []
        for i, (task, goal_task, task_comps) in enumerate(
                zip(tasks, goal_tasks, completions)):
            t0 = time.time()
            attempts = []
            best_success = False
            for sample_i, completion in enumerate(task_comps):
                full_solution = self._assemble_solution(
                    goal_task, prompts[i], completion)
                success, result = self._verify_solution(
                    goal_task, full_solution)
                attempts.append({
                    "code": full_solution, "success": success,
                    "error": result.get("stderr", ""),
                    "round": 0, "sample": sample_i,
                    "temperature": self.retry_policy.temperature_for_attempt(0),
                })
                if success and not best_success:
                    best_success = True

            # Retries stay serial — temperature bump per the retry policy.
            for attempt in range(1, self.retry_policy.max_retries + 1):
                if best_success:
                    break
                old_temp = self.temperature
                self.temperature = self.retry_policy.temperature_for_attempt(
                    attempt)
                retry_completion = self._generate(
                    prompts[i], prefill=_CHAT_SOLVE_PREFILL)
                self.temperature = old_temp

                full_solution = self._assemble_solution(
                    goal_task, prompts[i], retry_completion)
                success, result = self._verify_solution(
                    goal_task, full_solution)
                attempts.append({
                    "code": full_solution, "success": success,
                    "error": result.get("stderr", ""),
                    "round": 0, "sample": len(attempts),
                    "temperature": self.retry_policy.temperature_for_attempt(
                        attempt),
                })
                if success:
                    best_success = True

            elapsed_ms = batch_ms / len(tasks) + (time.time() - t0) * 1000.0
            results.append((task, {
                "final_success": best_success,
                "attempts": attempts,
                "rounds_used": len(attempts),
                "best_quality": 1.0 if best_success else 0.0,
            }, elapsed_ms))
        return results

    # ─── LADDER: Recursive Problem Decomposition ──────────────────

    LADDER_MAX_DEPTH = 3  # max decomposition depth (LADDER paper uses 3-5)
    LADDER_N_VARIANTS = 3  # easier variants per decomposition step

    def solve_with_ladder(self, task: ProposedTask,
                          engine: Any | None = None,
                          depth: int = 0) -> dict:
        """Solve a task using LADDER recursive decomposition.

        LADDER (arxiv 2503.00735): when the solver fails a hard task, generate
        easier variants, solve those first, then retry the original.
        Llama 3.2 3B: 1% → 82% on MIT Integration Bee with this approach.

        Args:
            task: the proposed task to solve
            engine: optional RecursiveSelfPlay engine
            depth: current decomposition depth (0 = original task)

        Returns:
            result dict with final_success, attempts, decomposition_chain
        """
        # First, try solving the task directly
        result = self.solve_task(task, engine=engine)

        if result.get("final_success", False):
            result["decomposition_chain"] = []
            return result

        # Failed — if at max depth, return failure
        if depth >= self.LADDER_MAX_DEPTH:
            result["decomposition_chain"] = []
            result["ladder_exhausted"] = True
            return result

        # Generate easier variants of the task
        variants = self._generate_easier_variants(task, n=self.LADDER_N_VARIANTS)

        if not variants:
            result["decomposition_chain"] = []
            return result

        # Solve each variant recursively (stepping stones) — in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        variant_solutions = []
        with ThreadPoolExecutor(max_workers=min(4, len(variants))) as executor:
            futures = {
                executor.submit(self.solve_with_ladder, v, engine=engine, depth=depth + 1): v
                for v in variants
            }
            variant_results = {}
            for future in as_completed(futures):
                v = futures[future]
                variant_results[id(v)] = future.result()

            # Reassemble in original variant order
            for variant in variants:
                v_result = variant_results[id(variant)]
                variant_solutions.append({
                    "variant": variant,
                    "result": v_result,
                    "depth": depth + 1,
                })

        # Check if any variant succeeded — use as context for retrying original
        successful_variants = [v for v in variant_solutions if v["result"].get("final_success")]
        if not successful_variants:
            result["decomposition_chain"] = variant_solutions
            return result

        # Retry original task with variant solutions as context (stepping stones)
        # Build a hint prompt that includes the simpler solution pattern
        hint_context = self._build_ladder_hint(task, successful_variants)
        result = self._solve_with_hint(task, hint_context, engine=engine)

        result["decomposition_chain"] = variant_solutions
        result["ladder_depth"] = depth
        result["n_variants_solved"] = len(successful_variants)
        return result

    def _generate_easier_variants(self, task: ProposedTask,
                                  n: int = 3) -> list[ProposedTask]:
        """Generate n easier variants of a task for LADDER decomposition.

        Strategies (in order of preference):
        1. Fewer test cases (reduce input complexity)
        2. Simpler inputs (smaller numbers, shorter strings)
        3. Relaxed constraints (allow simpler algorithms)
        """
        variants = []

        # Strategy 1: Reduce test cases to simplest ones
        if len(task.test_cases) > 2:
            simple_tests = task.test_cases[:2]  # keep 2 simplest
            variants.append(ProposedTask(
                id=f"{task.id}_simple_tests",
                domain=task.domain,
                difficulty="easy",
                description=task.description + " (simplified: fewer tests)",
                signature=task.signature,
                solve_name=task.solve_name,
                test_cases=simple_tests,
                stress_index=None,
                archetype=task.archetype,
            ))

        # Strategy 2: Simplify test case inputs
        simplified_tests = []
        for tc in task.test_cases[:4]:  # take up to 4
            simple_args = self._simplify_args(tc["args"])
            simplified_tests.append({"args": simple_args, "expected": tc["expected"]})

        if simplified_tests and len(simplified_tests) < len(task.test_cases):
            variants.append(ProposedTask(
                id=f"{task.id}_simple_inputs",
                domain=task.domain,
                difficulty="easy",
                description=task.description + " (simplified: smaller inputs)",
                signature=task.signature,
                solve_name=task.solve_name,
                test_cases=simplified_tests,
                stress_index=None,
                archetype=task.archetype,
            ))

        # Strategy 3: Model-generated easier variant
        if len(variants) < n:
            variant_desc = self._generate_easier_description(task)
            if variant_desc:
                variants.append(ProposedTask(
                    id=f"{task.id}_model_variant",
                    domain=task.domain,
                    difficulty="easy",
                    description=variant_desc,
                    signature=task.signature,
                    solve_name=task.solve_name,
                    test_cases=task.test_cases[:3],
                    stress_index=None,
                    archetype=task.archetype,
                ))

        return variants[:n]

    def _simplify_args(self, args: list) -> list:
        """Simplify function arguments to smaller values."""
        simplified = []
        for arg in args:
            if isinstance(arg, int):
                simplified.append(abs(arg) % 10 if abs(arg) > 10 else arg)
            elif isinstance(arg, list):
                simplified.append(arg[:3] if len(arg) > 3 else arg)
            elif isinstance(arg, str):
                simplified.append(arg[:10] if len(arg) > 10 else arg)
            else:
                simplified.append(arg)
        return simplified

    def _generate_easier_description(self, task: ProposedTask) -> str | None:
        """Ask the model to generate an easier variant of the task description."""
        self._trace_phase = "ladder_variant"
        prompt = f"""Simplify this programming task to make it easier:

Original: {task.description}

Provide a simpler version that handles a subset of the problem. Just the description, no code:
"""
        try:
            completion = self._generate(prompt)
            # Take first line as simplified description
            desc = completion.strip().split("\n")[0][:200]
            return desc if desc else None
        except Exception:
            return None

    def _build_ladder_hint(self, task: ProposedTask,
                           successful_variants: list[dict]) -> str:
        """Build a hint prompt from successful variant solutions."""
        hints = []
        for sv in successful_variants[:2]:  # use at most 2 hints
            v_result = sv["result"]
            if v_result.get("attempts"):
                best_attempt = v_result["attempts"][0]
                hints.append(f"Similar simpler problem solution:\n{best_attempt['code'][:300]}")

        hint_text = "\n\n".join(hints) if hints else ""
        return f"Here are solutions to simpler versions of this problem:\n{hint_text}\n\nNow solve the original:"

    def _solve_with_hint(self, task: ProposedTask, hint: str,
                         engine: Any | None = None) -> dict:
        """Solve a task with an additional hint context (from LADDER variants)."""
        self._trace_phase = "solve_hint"
        from forge.evaluation.goal_tasks import build_goal_prompt
        goal_task = GoalTask(
            id=task.id, domain=task.domain, difficulty=task.difficulty,
            description=task.description, input_signature=task.signature,
            solve_name=task.solve_name, test_cases=task.test_cases,
            stress_index=task.stress_index, archetype=task.archetype,
        )
        # Prepend hint to the goal prompt — chat mode uses the instruction
        # prompt (rendered as ChatML by _generate); raw mode uses the
        # completion-style goal prompt.
        if self.engine is not None:
            hinted_prompt = hint + "\n\n" + self._solve_prompt_text(goal_task)
        else:
            hinted_prompt = hint + "\n\n" + build_goal_prompt(goal_task)

        if engine is not None:
            # Patch the prompt in the engine call
            result = engine.run_goal_task(goal_task, k_samples=1, use_reasoning=True)
            return result
        else:
            # Direct solve with hint — chat mode assembles the reply;
            # raw mode concatenates prompt + body.
            completion = self._generate(
                hinted_prompt, prefill=_CHAT_SOLVE_PREFILL)
            full_solution = self._assemble_solution(
                goal_task, hinted_prompt, completion)

            success, result = self._verify_solution(goal_task, full_solution)

            return {
                "final_success": success,
                "attempts": [{"code": full_solution, "success": success,
                              "error": result.get("stderr", ""), "round": 0}],
                "rounds_used": 1,
                "best_quality": 1.0 if success else 0.0,
                "used_ladder_hint": True,
            }

    # ─── Difficulty Adaptation ─────────────────────────────────────

    def record_result(self, task: ProposedTask, success: bool):
        """Record solver result for adaptive difficulty and proposer reward.

        Also updates ELO ratings for both the model and the individual prompt,
        enabling per-prompt difficulty tracking (vs the coarser per-domain
        rolling success rate).
        """
        self.stats.total_solved += success
        self.stats.total_failed += (not success)

        results = self.stats.domain_results.setdefault(task.domain, [])
        results.append(success)
        if len(results) > self.WINDOW_SIZE:
            results = results[-self.WINDOW_SIZE:]

        # ELO update: adjust model and prompt ratings based on solve outcome.
        # This gives per-prompt difficulty tracking — prompts the model fails
        # get higher ratings (harder), prompts it solves get lower (easier).
        self.elo.register_prompt(task.id)
        self.elo.update_model_rating(task.id, success=success)

        # AZR proposer reward: r_propose = 1 - |domain_success_rate - 0.5|
        # Peaks at 50% success (max learnability). Zero outside [0.0, 1.0].
        domain_rate = self.stats.domain_success_rate(task.domain)
        proposer_reward = 1.0 - abs(domain_rate - 0.5)
        self.stats.proposer_rewards.append(proposer_reward)
        if len(self.stats.proposer_rewards) > 1000:
            self.stats.proposer_rewards = self.stats.proposer_rewards[-1000:]

    def record_task_embedding(self, task: ProposedTask, embedding: list[float]):
        """Record a task embedding for diversity tracking.
        Call this after a task is validated. Embedding can be a simple
        bag-of-words hash or a model-generated embedding."""
        self.stats.task_embeddings.append(embedding)
        if len(self.stats.task_embeddings) > 500:
            self.stats.task_embeddings = self.stats.task_embeddings[-500:]

    def _adaptive_difficulty(self, domain: str) -> str:
        """Pick difficulty based on recent solver success rate (AZR Goldilocks).

        AZR-optimal: target 50% success rate. Adapt difficulty to keep the model
        in the [0.2, 0.8] learnability zone. Easy tasks are allowed early in the
        curriculum (when success rate is very low) to bootstrap.
        """
        rate = self.stats.domain_success_rate(domain)

        if rate > self.GOLDILOCKS_HIGH:
            # Model is solving >80% → escalate to hard
            return "hard"
        elif rate < self.GOLDILOCKS_LOW:
            # Model is failing >80% → ease to easy (bootstrap)
            return "easy"
        elif rate < 0.35:
            # Below 35% — mostly easy/medium
            return self.rng.choice(["easy", "medium", "medium"])
        elif rate > 0.65:
            # Above 65% — mostly hard
            return self.rng.choice(["medium", "hard", "hard"])
        else:
            # In the Goldilocks zone (35-65%) — balanced mix
            return self.rng.choice(["easy", "medium", "medium", "hard"])

    # ─── Task Queue Management ─────────────────────────────────────

    def get_training_batch(self, n: int,
                           domain: str | None = None,
                           difficulty: str | None = None) -> list[ProposedTask]:
        """Get n validated tasks from the queue for training.

        Returns tasks sorted by difficulty (easy first = curriculum order).
        """
        # Filter queue
        candidates = [t for t in self._task_queue
                      if (domain is None or t.domain == domain)
                      and (difficulty is None or t.difficulty == difficulty)]

        # Sort by difficulty (curriculum: easy → medium → hard)
        diff_order = {"easy": 0, "medium": 1, "hard": 2}
        candidates.sort(key=lambda t: diff_order.get(t.difficulty, 1))

        return candidates[:n]

    def get_training_batch_elo(self, n: int,
                               domain: str | None = None,
                               exploration_ratio: float = 0.2
                               ) -> list[ProposedTask]:
        """Get n validated tasks using ELO matchmaking.

        Selects prompts where the model's expected win probability is closest
        to 50% (the Goldilocks zone for maximum learning signal). A fraction
        of prompts are randomly selected for exploration (discovering new
        difficulty levels).

        This augments the difficulty-sorted `get_training_batch` with
        per-prompt ELO ratings — instead of just "easy first", it picks
        prompts at the model's current skill boundary.

        Args:
            n: number of tasks to select.
            domain: filter by domain (None = all).
            exploration_ratio: fraction of prompts for random exploration.

        Returns:
            list of ProposedTask, ELO-matched to the model's skill level.
        """
        candidates = [t for t in self._task_queue
                      if domain is None or t.domain == domain]
        if not candidates:
            return []

        # Use ELO to select prompt IDs that match the model's skill
        candidate_ids = [t.id for t in candidates]
        selected_ids = self.elo.select_mixed_prompts(
            candidate_ids, n=n, exploration_ratio=exploration_ratio)

        # Map back to ProposedTask objects
        id_to_task = {t.id: t for t in candidates}
        return [id_to_task[pid] for pid in selected_ids if pid in id_to_task]

    def queue_size(self, domain: str | None = None) -> int:
        """Number of validated tasks in the queue."""
        if domain is None:
            return len(self._task_queue)
        return sum(1 for t in self._task_queue if t.domain == domain)

    def _save_task(self, task: ProposedTask):
        """Save validated task to disk for replay."""
        path = self.task_queue_dir / f"{task.id}.json"
        data = {
            "id": task.id,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "description": task.description,
            "signature": task.signature,
            "solve_name": task.solve_name,
            "test_cases": task.test_cases,
            "stress_index": task.stress_index,
            "archetype": task.archetype,
            "generation_mode": task.generation_mode,
            "proposer_confidence": task.proposer_confidence,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=_json_default)
        except (OSError, TypeError, ValueError) as e:
            # Task persistence is auxiliary — a save failure must NOT kill
            # the epoch (observed: a set-valued test arg aborted an entire
            # self-play epoch mid-propose via TypeError here).
            print(f"  [Curriculum] _save_task failed for {task.id}: {e}")

    def load_queue(self, max_tasks: int = 1000):
        """Load previously saved tasks from disk into the queue."""
        files = sorted(self.task_queue_dir.glob("*.json"))[:max_tasks]
        for f in files:
            try:
                data = loads(f.read_text())
                task = ProposedTask(
                    id=data["id"],
                    domain=data["domain"],
                    difficulty=data["difficulty"],
                    description=data["description"],
                    signature=data["signature"],
                    solve_name=data.get("solve_name", "solve"),
                    test_cases=data["test_cases"],
                    stress_index=data.get("stress_index"),
                    archetype=data.get("archetype", "model_proposed"),
                    generation_mode=data.get("generation_mode", "induction"),
                    proposer_confidence=data.get("proposer_confidence", 0),
                    validated=True,
                )
                self._task_queue.append(task)
            except (json.JSONDecodeError, KeyError):
                continue

        print(f"  [Curriculum] Loaded {len(self._task_queue)} tasks from disk")
        self._load_seen_descriptions()

    def _load_seen_descriptions(self):
        """Load seen task descriptions + functional fingerprints from disk
        to prevent clone tasks."""
        if self._seen_desc_path.exists():
            try:
                self._seen_descriptions = set(loads(self._seen_desc_path.read_text()))
                print(f"  [Curriculum] Loaded {len(self._seen_descriptions)} seen descriptions")
            except (json.JSONDecodeError, OSError):
                self._seen_descriptions = set()
        self._seen_desc_tokens = [
            self._desc_tokens(d) for d in self._seen_descriptions]
        if self._seen_fp_path.exists():
            try:
                self._seen_fingerprints = set(
                    loads(self._seen_fp_path.read_text()))
            except (json.JSONDecodeError, OSError):
                self._seen_fingerprints = set()

    def _save_seen_descriptions(self):
        """Persist seen descriptions + fingerprints to disk."""
        try:
            with open(self._seen_desc_path, "w") as f:
                json.dump(sorted(self._seen_descriptions), f)
        except OSError:
            pass
        try:
            with open(self._seen_fp_path, "w") as f:
                json.dump(sorted(self._seen_fingerprints), f)
        except OSError:
            pass

    @staticmethod
    def _normalize_desc(desc: str) -> str:
        """Normalize a task description for dedup comparison."""
        import re
        # Lowercase, collapse whitespace, strip trailing punctuation
        d = re.sub(r'\s+', ' ', desc.lower().strip())
        d = d.rstrip('.;:')
        return d

    _DESC_STOPWORDS = frozenset(
        ["a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "is", "are", "it", "its", "with", "without", "by", "from", "return", "returns", "returned", "given", "write", "create", "one", "all", "any", "each"])

    @classmethod
    def _desc_tokens(cls, desc: str) -> set[str]:
        """Content-word set of a description (stopwords removed, naive
        plural stem) for similarity dedup."""
        toks = re.findall(r"[a-z]+", desc.lower())
        return {(t[:-1] if len(t) > 3 and t.endswith("s") else t)
                for t in toks if t not in cls._DESC_STOPWORDS}

    def _is_seen_description(self, desc: str) -> bool:
        """Check if a task description has been seen — exact normalized
        match OR high content-word overlap (semantic clone)."""
        norm = self._normalize_desc(desc)
        if norm in self._seen_descriptions:
            return True
        toks = self._desc_tokens(desc)
        if toks:
            for seen in self._seen_desc_tokens:
                union = toks | seen
                if union and len(toks & seen) / len(union) >= 0.5:
                    return True
        return False

    def _mark_description_seen(self, desc: str):
        """Record a task description as seen (valid or rejected)."""
        self._seen_descriptions.add(self._normalize_desc(desc))
        self._seen_desc_tokens.append(self._desc_tokens(desc))
        # Periodically save (every 10 new descriptions)
        if len(self._seen_descriptions) % 10 == 0:
            self._save_seen_descriptions()

    _FP_PROBES = {
        int: [0, 1, 7, 42, -3],
        str: ["", "a", "xyzzy", "Hello World", "aa bb"],
        list: [[], [0], [2, 4, 6], [-1, 9], [5] * 5],
        float: [0.0, 1.5, -2.25, 100.0, 3.75],
        bool: [True, False, True, False, True],
    }

    def _functional_fingerprint(self, ref_code: str,
                                task: ProposedTask) -> str | None:
        """Execute the reference impl on fixed probe inputs; return the
        output signature. Two tasks whose reference impls produce identical
        outputs on the same probes are the same function — a clone,
        whatever the description says. Returns None when probes can't be
        built for the signature (no dedup applied then).
        """
        if not ref_code or not task.test_cases:
            return None
        arg_types = [type(a) for a in task.test_cases[0]["args"]]
        if not arg_types or any(t not in self._FP_PROBES for t in arg_types):
            return None
        probe_vals = [self._FP_PROBES[t] for t in arg_types]
        n_p = min(len(p) for p in probe_vals)
        lines = [ref_code]
        for i in range(n_p):
            args = tuple(pv[i] for pv in probe_vals)
            lines.append(
                "try:\n"
                f"    print(repr(solve(*{args!r})))\n"
                "except Exception as _e:\n"
                "    print('ERR:' + type(_e).__name__)")
        res = self._execute_sandbox("\n".join(lines), timeout_s=3.0)
        if res.get("returncode") != 0:
            return None
        return (f"{len(arg_types)}:{','.join(t.__name__ for t in arg_types)}:"
                + res["stdout"].strip())

    # ─── Convert to GoalTask (for existing training pipeline) ──────

    def to_goal_tasks(self, tasks: list[ProposedTask]) -> list[GoalTask]:
        """Convert ProposedTasks to GoalTasks for the existing training pipeline."""
        return [GoalTask(
            id=t.id,
            domain=t.domain,
            difficulty=t.difficulty,
            description=t.description,
            input_signature=t.signature,
            solve_name=t.solve_name,
            test_cases=t.test_cases,
            stress_index=t.stress_index,
            archetype=t.archetype,
        ) for t in tasks]

    # ─── Stats ─────────────────────────────────────────────────────

    def print_stats(self):
        """Print curriculum health metrics."""
        try:
            print("\n  [Curriculum] Stats:")
            print(f"    Proposed: {self.stats.total_proposed}")
            print(f"    Validated: {self.stats.total_validated} "
                  f"({self.stats.validation_rate:.1%})")
            print(f"    Solved: {self.stats.total_solved} "
                  f"({self.stats.solve_rate:.1%})")
            print(f"    Queue: {len(self._task_queue)} tasks")
            print(f"    Proposer reward: {self.stats.mean_proposer_reward:.3f} "
                  f"(AZR optimal=0.5 at 50% success)")
            div = self.stats.diversity_score
            div_alert = " [WARNING: low diversity]" if div < 0.6 else ""
            print(f"    Diversity score: {div:.3f}{div_alert}")
            for domain in self.DOMAINS:
                rate = self.stats.domain_success_rate(domain)
                n = len(self.stats.domain_results.get(domain, []))
                if n > 0:
                    print(f"    {domain}: {rate:.1%} success ({n} samples)")
        except OSError:
            pass  # Windows console encoding issue — skip stats print
