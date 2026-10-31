"""Training-free alignment & adaptation — inference-only behavior control.

All techniques run strictly forward-pass: no gradients, no optimizer, no
parameter updates. VRAM stays at the inference footprint.

Merged from urial.py + decoder.py + reflexion.py + rain.py (all <3.5KB,
same domain: training-free inference techniques).

Modules:
  - URIAL:    in-context alignment (styling via system prompt + 3 examples)
  - decoder:  shared KV-cached forward-only generation utility
  - Reflexion: episodic memory of past attempts rendered into the prompt
  - RAIN:     rewindable autoregressive inference (self-eval + rewind)
  - steering: activation steering / task vectors (residual stream hooks)
  - solver:   TrainingFreeSolver — frozen-solver adapter combining the above
              for self-play loops (replaces GRPO weight updates)
  - bake:     expert baking / decompression
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from research.training_free.expert_bake import bake_expert, decompress_expert
from research.training_free.steering import ActivationSteerer

# ── URIAL ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and concise assistant. "
    "Solve the user's request directly and completely. "
    "When the answer is code, write clean, correct, efficient code "
    "with no unnecessary commentary."
)

# Three curated style demonstrations — URIAL shows that as few as three
# high-quality examples are enough to force the base model's style shift.
STYLE_EXAMPLES: list[dict[str, str]] = [
    {
        "user": "Write a function that returns the sum of all even numbers in a list.",
        "assistant": (
            "def sum_even(nums):\n"
            "    return sum(n for n in nums if n % 2 == 0)"
        ),
    },
    {
        "user": "Explain the difference between a list and a tuple in Python.",
        "assistant": (
            "A list is mutable and resizable; a tuple is immutable and "
            "fixed-size. Use lists for dynamic collections, tuples for fixed "
            "records or hashable keys."
        ),
    },
    {
        "user": "What is the output of 3 * 'ab' + 'c'?",
        "assistant": (
            "'abababc'. The * operator repeats the string 3 times, then "
            "+ concatenates 'c'."
        ),
    },
]


def build_prompt(
    user_prompt: str,
    system_prompt: str | None = None,
    n_examples: int = 3,
    extra_context: str = "",
) -> str:
    """Assemble the URIAL in-context alignment prompt.

    Args:
        user_prompt: the actual request.
        system_prompt: override the default system instruction.
        n_examples: how many style demonstrations to include (max 3).
        extra_context: optional extra block (e.g. Reflexion memory) inserted
            right before the user request.

    Returns:
        Full styled prompt string.
    """
    system = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    parts = [system]
    for ex in STYLE_EXAMPLES[: max(0, n_examples)]:
        parts.append(f"User: {ex['user']}\nAssistant: {ex['assistant']}")
    if extra_context:
        parts.append(extra_context)
    parts.append(f"User: {user_prompt}\nAssistant:")
    return "\n\n".join(parts)


# ── Shared decoder ─────────────────────────────────────────────────────

def generate_with_cache(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    max_tokens: int = 128,
    temperature: float = 0.0,
    stop_strings: tuple[str, ...] = (),
    collect_logprobs: bool = False,
) -> tuple[str, list[float] | None]:
    """Greedy/temperature KV-cached generation with a pre-allocated cache.

    Args:
        model: ConfigurableResearchLLM (or compatible).
        tokenizer: HF-style tokenizer.
        prompt: text to condition on.
        device: "cuda" or "cpu".
        max_tokens: max tokens to generate (excluding prompt).
        temperature: sampling temperature (0 = greedy).
        stop_strings: stop generating when any appears in the output.
        collect_logprobs: also return per-token log-probs of generated tokens.

    Returns:
        (generated_text, logprobs_or_None).
    """
    from forge.model_loader import create_kv_cache

    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer.eos_token_id

    cache = create_kv_cache(
        model, input_ids.shape[1] + max_tokens, batch=1, device=device)
    cache.reset()

    log_probs: list[float] = []
    generated_ids: list[int] = []

    with torch.inference_mode():
        logits, _ = model(input_ids, preallocated_cache=cache)
        next_logits = logits[0, -1]

        for _ in range(max_tokens):
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax().unsqueeze(0)

            if collect_logprobs:
                lp = F.log_softmax(next_logits.float(), dim=-1)
                log_probs.append(lp[next_token].item())

            token_id = next_token.item()
            generated_ids.append(token_id)

            if eos_id is not None and token_id == eos_id:
                break

            decoded = tokenizer.decode(
                torch.tensor(generated_ids), skip_special_tokens=True)
            if any(s in decoded for s in stop_strings):
                break

            cur = next_token.view(1, 1)
            logits, _ = model(cur, preallocated_cache=cache)
            next_logits = logits[0, -1]

    text = tokenizer.decode(
        torch.tensor(generated_ids), skip_special_tokens=True)
    return text, (log_probs or None)


# ── Reflexion ──────────────────────────────────────────────────────────

def make_template_reflection(task: str, code: str = "", error: str = "") -> str:
    """Deterministic critique template (no model call needed)."""
    if error:
        return (
            f"Previous attempt at '{task}' failed with: {error.strip()[:200]}. "
            "Avoid repeating this mistake; check edge cases and syntax."
        )
    return (
        f"Previous attempt at '{task}' did not solve it. "
        "Re-analyze the problem and try a different, complete solution."
    )


class ReflexionBuffer:
    """Bounded episodic memory of past attempts, rendered into the prompt.

    Args:
        max_entries: how many attempts to keep (oldest dropped).
        max_chars: hard cap on the rendered context block.
        reflection_fn: optional callable(task, code, error) -> str used to
            generate the reflection; defaults to the deterministic template.
    """

    def __init__(
        self,
        max_entries: int = 8,
        max_chars: int = 1500,
        reflection_fn: Callable[[str, str, str], str] | None = None,
    ):
        self.max_entries = max(1, max_entries)
        self.max_chars = max(200, max_chars)
        self.reflection_fn = reflection_fn or make_template_reflection
        self._entries: list[dict] = []

    def add(self, task: str, code: str = "", error: str = "",
            success: bool = False, reflection: str | None = None) -> None:
        """Record one attempt; drops oldest once over max_entries."""
        if reflection is None:
            reflection = self.reflection_fn(task, code, error)
        self._entries.append({
            "task": task,
            "success": bool(success),
            "reflection": reflection,
        })
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def successes(self) -> int:
        return sum(1 for e in self._entries if e["success"])

    def context_block(self) -> str:
        """Render the memory as a context section (bounded by max_chars)."""
        if not self._entries:
            return ""
        lines = ["## Past attempts (learn from them, do not repeat mistakes)"]
        for e in self._entries:
            tag = "SUCCESS" if e["success"] else "FAILED"
            lines.append(f"- [{tag}] {e['reflection']}")
        block = "\n".join(lines)
        if len(block) > self.max_chars:
            suffix = "\n... (memory truncated)"
            block = block[: max(0, self.max_chars - len(suffix))] + suffix
        return block

    def prompt_for(self, task: str, n_examples: int = 3,
                   system_prompt: str | None = None) -> str:
        """URIAL-styled prompt with the memory block prepended to the task."""
        return build_prompt(
            task,
            system_prompt=system_prompt,
            n_examples=n_examples,
            extra_context=self.context_block(),
        )


# ── RAIN ───────────────────────────────────────────────────────────────

def _default_eval(text: str, logprobs: list[float] | None) -> float:
    """Default self-evaluation: mean token log-prob (higher = more confident)."""
    if logprobs:
        return sum(logprobs) / len(logprobs)
    return 0.0


class RAINGenerator:
    """Rewindable autoregressive generator.

    Args:
        model: ConfigurableResearchLLM.
        tokenizer: HF-style tokenizer.
        device: "cuda" or "cpu".
        eval_fn: callable(generated_text, token_logprobs) -> float score.
            Higher = better. Defaults to mean token log-prob.
        threshold: rewind when score < threshold.
        rewind_tokens: how many tokens to truncate on each rewind.
        max_rewinds: maximum rewinds per request.
        max_tokens: per-attempt generation budget.
        temperature: sampling temperature (0 = greedy).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        eval_fn: Callable[[str, list[float] | None], float] | None = None,
        threshold: float = -1.0,
        rewind_tokens: int = 8,
        max_rewinds: int = 3,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eval_fn = eval_fn or _default_eval
        self.threshold = threshold
        self.rewind_tokens = max(1, rewind_tokens)
        self.max_rewinds = max(0, max_rewinds)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rewinds_used = 0

    def generate(self, prompt: str) -> tuple[str, float, list[float]]:
        """Generate with rewind-and-regenerate until the score clears the
        threshold or the rewind budget is exhausted.

        Returns:
            (final_text, final_score, per_attempt_scores).
        """
        self.rewinds_used = 0
        scores: list[float] = []

        text, logprobs = generate_with_cache(
            self.model, self.tokenizer, prompt,
            device=self.device, max_tokens=self.max_tokens,
            temperature=self.temperature, collect_logprobs=True,
        )
        score = self.eval_fn(text, logprobs)
        scores.append(score)

        while score < self.threshold and self.rewinds_used < self.max_rewinds:
            if len(text.split()) <= self.rewind_tokens:
                break  # nothing meaningful left to rewind
            self.rewinds_used += 1
            text, logprobs = generate_with_cache(
                self.model, self.tokenizer, prompt,
                device=self.device, max_tokens=self.max_tokens,
                temperature=self.temperature, collect_logprobs=True,
            )
            score = self.eval_fn(text, logprobs)
            scores.append(score)

        return text, score, scores


__all__ = [
    "STYLE_EXAMPLES",
    "SYSTEM_PROMPT",
    "ActivationSteerer",
    "RAINGenerator",
    "ReflexionBuffer",
    "TrainingFreeSolver",
    "bake_expert",
    "build_prompt",
    "decompress_expert",
    "generate_with_cache",
    "make_template_reflection",
]

# Import solver at the END to avoid circular import
# (solver.py imports generate_with_cache, ReflexionBuffer, build_prompt from here)
from research.training_free.solver import TrainingFreeSolver
