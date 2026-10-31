"""TrainingFreeSolver — frozen-solver adapter for self-play.

Replaces GRPO weight updates with three forward-only mechanisms:
  1. URIAL in-context alignment (styling without SFT/RLHF).
  2. Reflexion episodic memory (success/failure pushed into the prompt).
  3. Task-vector steering extracted from successful runs (activation
     steering, injected into the residual stream).

VRAM stays at inference footprint: no optimizer states, no backward passes.
"""
from __future__ import annotations

import torch

from research.training_free import ReflexionBuffer, build_prompt, generate_with_cache
from research.training_free.steering import ActivationSteerer


class TrainingFreeSolver:
    """Inference-only solver that adapts via context and steering.

    Args:
        model: ConfigurableResearchLLM (kept frozen).
        tokenizer: HF-style tokenizer.
        device: "cuda" or "cpu".
        max_tokens: per-request generation budget.
        temperature: sampling temperature (0 = greedy).
        capture_activations: collect residual activations on each recorded
            run so build_task_vector() has data (needs a bit of VRAM/CPU).
        memory_kwargs: passed to ReflexionBuffer (max_entries, max_chars...).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_tokens: int = 128,
        temperature: float = 0.0,
        capture_activations: bool = True,
        memory_kwargs: dict | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.capture_activations = capture_activations

        self.memory = ReflexionBuffer(**(memory_kwargs or {}))
        self.steerer = ActivationSteerer(model)
        self._pos_acts: list[dict[int, torch.Tensor]] = []
        self._neg_acts: list[dict[int, torch.Tensor]] = []
        self._vector: dict[int, torch.Tensor] | None = None

    # ── generation ────────────────────────────────────────────────

    def style(self, task: str, n_examples: int = 3,
              system_prompt: str | None = None) -> str:
        """Assemble the in-context alignment prompt (URIAL + Reflexion memory).

        Does not generate — lets callers (e.g. SelfPlaySandbox) run their own
        KV-cached/conformal generation from the styled prompt.
        """
        return build_prompt(
            task,
            system_prompt=system_prompt,
            n_examples=n_examples,
            extra_context=self.memory.context_block(),
        )

    def generate(self, task: str, n_examples: int = 3,
                 system_prompt: str | None = None) -> str:
        """Generate a solution for *task* with in-context alignment applied.

        The prompt includes: URIAL style demonstrations + Reflexion memory
        of past attempts + the current task. If a task vector is active it
        steers the forward pass; the model itself is never modified.
        """
        prompt = self.style(task, n_examples=n_examples,
                            system_prompt=system_prompt)
        text, _ = generate_with_cache(
            self.model, self.tokenizer, prompt,
            device=self.device, max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return text

    # ── memory ────────────────────────────────────────────────────

    def record(self, task: str, output: str = "", error: str = "",
               success: bool = False) -> None:
        """Record an attempt outcome into Reflexion memory.

        If capture_activations is enabled, the (task + output) residual
        activations are also collected — successes feed the positive set,
        failures the negative set, for later task-vector extraction.
        """
        self.memory.add(task, code=output, error=error, success=success)

        if not self.capture_activations or not output:
            return
        probe = f"{task}\n{output}"
        try:
            acts = self.steerer.collect_activations(
                self.tokenizer, [probe], device=self.device)
            if acts:
                (self._pos_acts if success else self._neg_acts).append(acts)
        except Exception:
            pass  # activation capture is best-effort

    # ── task vector steering ──────────────────────────────────────

    def build_task_vector(self, normalize: bool = True) -> dict[int, torch.Tensor]:
        """Extract the success-minus-failure direction from collected runs.

        Requires at least one positive and one negative activation set.
        Returns {layer: (D,)} and stores it for apply_steering().
        """
        if not self._pos_acts or not self._neg_acts:
            return {}
        pos_mean = _mean_acts(self._pos_acts)
        neg_mean = _mean_acts(self._neg_acts)
        self._vector = ActivationSteerer.task_vectors(
            pos_mean, neg_mean, normalize=normalize)
        return self._vector

    def apply_steering(self, alpha: float = 1.0) -> bool:
        """Apply the stored task vector at strength alpha. Returns False if
        no vector is available yet."""
        if not self._vector:
            return False
        self.steerer.apply(self._vector, alpha=alpha)
        return True

    def clear_steering(self) -> None:
        """Remove steering hooks (baseline behavior)."""
        self.steerer.remove()

    @property
    def steering_active(self) -> bool:
        return self.steerer.active

    def stats(self) -> dict:
        return {
            "memory_entries": len(self.memory),
            "successes": self.memory.successes,
            "pos_activation_sets": len(self._pos_acts),
            "neg_activation_sets": len(self._neg_acts),
            "task_vector_ready": self._vector is not None,
            "steering_active": self.steering_active,
        }


def _mean_acts(sets: list[dict[int, torch.Tensor]]) -> dict[int, torch.Tensor]:
    """Average activations across collected runs, per layer."""
    if not sets:
        return {}
    layers = sets[0].keys()
    out = {}
    for i in layers:
        stacked = torch.stack([s[i] for s in sets if i in s]).float()
        out[i] = stacked.mean(dim=0)
    return out
