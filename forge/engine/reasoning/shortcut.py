"""Shortcut Decoding + SyncThink — training-free CoT termination.

Two complementary training-free methods for efficient chain-of-thought:

1. **ShortcutDecoder** (ACL 2026, long.1330 "Think Faster Than Words"):
   Dual-signal adaptive controller that detects reasoning convergence via:
   - Internal confidence score from lightweight MLP probe on hidden states
   - Step-level entropy from the logit distribution
   When convergence is detected, switches directly to final answer generation,
   skipping remaining reasoning steps. 35% token reduction, maintains accuracy.

2. **SyncThinkTerminator** (OpenReview Hc9jAnIB3f):
   Monitors the model's reasoning-transition signal: answer tokens attend
   weakly to early reasoning and focus on boundary tokens (e.g. <|im_start|>).
   When attention mass shifts to boundary tokens, reasoning has saturated.
   62% accuracy with 656 tokens vs 61.22% with 2141 tokens (+8.1 on GPQA
   by preventing over-thinking).

Both are training-free, plug-and-play, and require no base model updates.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class ShortcutConfig:
    """Configuration for ShortcutDecoder."""
    # Entropy threshold below which reasoning is considered converged.
    # Lower = more conservative (longer CoT), higher = more aggressive.
    entropy_threshold: float = 0.5
    # Minimum number of reasoning steps before convergence can be triggered.
    min_steps: int = 5
    # Window size for rolling entropy average (smoother signal).
    entropy_window: int = 3
    # Probe confidence threshold (if probe is used).
    probe_threshold: float = 0.8
    # Whether to use the hidden-state probe (requires calibration).
    use_probe: bool = False
    # Token IDs that indicate transition to final answer (e.g. "The answer is").
    answer_trigger_tokens: set[int] = field(default_factory=set)


@dataclass
class SyncThinkConfig:
    """Configuration for SyncThinkTerminator."""
    # Attention mass threshold on boundary tokens to trigger termination.
    # If attention to boundary tokens exceeds this fraction, reasoning is
    # considered saturated.
    boundary_attention_threshold: float = 0.3
    # Minimum reasoning steps before saturation can be triggered.
    min_steps: int = 5
    # Window for rolling attention average.
    attention_window: int = 3
    # Whether to check for answer-token pattern (tokens that look like
    # final answers attend to boundary, not early reasoning).
    check_answer_pattern: bool = True


class ShortcutDecoder:
    """Shortcut Decoding — dual-signal convergence detection for CoT.

    Detects when the model has internally converged to the correct answer
    before completing the full CoT text, then switches to final answer
    generation. Training-free (uses entropy + optional hidden-state probe).

    Usage:
        shortcut = ShortcutDecoder(config)
        for step, (logits, hidden_states) in enumerate(reasoning_loop):
            if shortcut.should_terminate(step, logits, hidden_states):
                break  # switch to answer generation
    """

    def __init__(self, config: ShortcutConfig | None = None):
        self.config = config or ShortcutConfig()
        self._entropy_history: list[float] = []
        self._probe_history: list[float] = []

    def should_terminate(
        self,
        step: int,
        logits: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        probe_fn: Callable | None = None,
    ) -> bool:
        """Check if reasoning has converged and should terminate.

        Args:
            step: current reasoning step index (0-based).
            logits: (batch, vocab) logits for the current token.
            hidden_states: (batch, d_model) hidden states for probe (optional).
            probe_fn: callable(hidden_states) -> confidence score [0, 1].

        Returns:
            True if reasoning should terminate (convergence detected).
        """
        if step < self.config.min_steps:
            # Still record entropy for stats, but don't trigger termination.
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
            self._entropy_history.append(entropy)
            return False

        # Signal 1: Step-level entropy (low entropy = high confidence = converged).
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
        self._entropy_history.append(entropy)

        # Use rolling average for smoother signal.
        window = self._entropy_history[-self.config.entropy_window:]
        avg_entropy = sum(window) / len(window)

        entropy_converged = avg_entropy < self.config.entropy_threshold

        # Signal 2: Hidden-state probe (if available).
        probe_converged = True  # default: don't block on probe
        if self.config.use_probe and probe_fn is not None and hidden_states is not None:
            try:
                confidence = probe_fn(hidden_states).mean().item()
                self._probe_history.append(confidence)
                probe_converged = confidence > self.config.probe_threshold
            except Exception:
                probe_converged = True  # ignore probe errors

        # Signal 3: Answer trigger tokens (if configured).
        trigger_detected = False
        if self.config.answer_trigger_tokens:
            top_token = logits.argmax(dim=-1).item()
            trigger_detected = top_token in self.config.answer_trigger_tokens

        # Terminate if entropy is low AND (probe agrees OR probe disabled).
        # Also terminate if answer trigger token is detected.
        if entropy_converged and probe_converged:
            return True
        return bool(trigger_detected)

    def reset(self):
        """Reset internal state for a new reasoning session."""
        self._entropy_history.clear()
        self._probe_history.clear()

    @property
    def stats(self) -> dict:
        """Return statistics about the last reasoning session."""
        return {
            "n_steps": len(self._entropy_history),
            "final_entropy": self._entropy_history[-1] if self._entropy_history else None,
            "avg_entropy": (sum(self._entropy_history) / len(self._entropy_history)
                            if self._entropy_history else None),
            "min_entropy": min(self._entropy_history) if self._entropy_history else None,
        }


class SyncThinkTerminator:
    """SyncThink — training-free reasoning saturation detection.

    Monitors attention patterns to detect reasoning saturation: when answer
    tokens start attending to boundary tokens (e.g. <|im_start|>) instead of
    early reasoning tokens, the reasoning has saturated and further tokens
    are redundant.

    Usage:
        terminator = SyncThinkTerminator(config)
        for step, (logits, attention_weights) in enumerate(reasoning_loop):
            if terminator.should_terminate(step, attention_weights):
                break
    """

    def __init__(self, config: SyncThinkConfig | None = None):
        self.config = config or SyncThinkConfig()
        self._boundary_attention_history: list[float] = []

    def should_terminate(
        self,
        step: int,
        attention_weights: torch.Tensor,
        boundary_positions: set[int] | None = None,
    ) -> bool:
        """Check if reasoning has saturated and should terminate.

        Args:
            step: current reasoning step index (0-based).
            attention_weights: (n_heads, seq_len) or (seq_len,) attention
                weights from the last decoding step. The last row corresponds
                to the current token's attention over all positions.
            boundary_positions: positions of boundary tokens (e.g. <|im_start|>,
                system prompt end). If None, uses first and last positions.

        Returns:
            True if reasoning should terminate (saturation detected).
        """
        if step < self.config.min_steps:
            return False

        # Handle different attention shapes.
        if attention_weights.dim() > 2:
            # (n_heads, seq_len, seq_len) -> average over heads, take last row.
            attn = attention_weights.mean(dim=0)[-1]  # (seq_len,)
        elif attention_weights.dim() == 2:
            # (n_heads, seq_len) -> average over heads.
            attn = attention_weights.mean(dim=0)  # (seq_len,)
        else:
            attn = attention_weights  # (seq_len,)

        seq_len = attn.shape[-1]

        # Default boundary positions: first token (BOS/sink) and last few.
        if boundary_positions is None:
            boundary_positions = {0, seq_len - 1}

        # Compute attention mass on boundary positions.
        boundary_mass = sum(attn[i].item() for i in boundary_positions
                           if i < seq_len)
        self._boundary_attention_history.append(boundary_mass)

        # Rolling average for smoother signal.
        window = self._boundary_attention_history[-self.config.attention_window:]
        avg_boundary = sum(window) / len(window)

        return avg_boundary > self.config.boundary_attention_threshold

    def reset(self):
        """Reset internal state for a new reasoning session."""
        self._boundary_attention_history.clear()

    @property
    def stats(self) -> dict:
        """Return statistics about the last reasoning session."""
        return {
            "n_steps": len(self._boundary_attention_history),
            "final_boundary_attention": (
                self._boundary_attention_history[-1]
                if self._boundary_attention_history else None),
            "avg_boundary_attention": (
                sum(self._boundary_attention_history) /
                len(self._boundary_attention_history)
                if self._boundary_attention_history else None),
        }


def compute_step_entropy(logits: torch.Tensor) -> float:
    """Compute the entropy of a logit distribution for a single step.

    Useful as a standalone utility for custom reasoning termination logic.

    Args:
        logits: (batch, vocab) or (vocab,) logit tensor.

    Returns:
        Entropy value (float). Lower = more confident = more converged.
    """
    if logits.dim() > 1:
        logits = logits[-1]  # take last batch element
    probs = F.softmax(logits, dim=-1)
    return -(probs * (probs + 1e-10).log()).sum().item()


def detect_reasoning_saturation(
    entropy_history: list[float],
    patience: int = 3,
    improvement_threshold: float = 0.01,
) -> bool:
    """Detect if reasoning has saturated based on entropy history.

    Reasoning is saturated if entropy has not improved (decreased) by more
    than improvement_threshold over the last `patience` steps.

    Args:
        entropy_history: list of per-step entropy values.
        patience: number of steps to look back.
        improvement_threshold: minimum entropy improvement to continue.

    Returns:
        True if reasoning has saturated (should terminate).
    """
    if len(entropy_history) < patience + 1:
        return False
    recent = entropy_history[-patience:]
    baseline = entropy_history[-(patience + 1)]
    return all(e > baseline - improvement_threshold for e in recent)
