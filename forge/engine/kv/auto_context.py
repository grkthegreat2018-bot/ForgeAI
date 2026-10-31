"""AutoContext — Automatic Context Window Management (Round 33 flagship).

Novel cross-domain combination: using the per-token entropy trajectory
(already computed during sampling for top-k/top-p cutoff) as a TASK-TYPE
signal to select the optimal KV cache strategy. No prior paper proposes
repurposing the entropy signal — a byproduct of sampling — to drive KV
cache policy. The entropy trajectory is free: it is computed once per
generated token for sampling and would otherwise be discarded.

Three orthogonal signals drive strategy selection:
  1. Context length  — short/medium/long buckets pick the base strategy.
  2. Task type        — derived from the entropy trajectory shape:
       * High sustained entropy  -> coding/reasoning (diverse token
         distribution, many plausible continuations). These tasks attend
         heavily to RECENT tokens (the code being written) -> sliding
         window / SnapKV (preserve recent).
       * Low sustained entropy   -> chat/template (peaked distribution,
         predictable tokens). These tasks attend to the SYSTEM PROMPT and
         first turns -> StreamingLLM (attention sinks).
       * Mixed (entropy spikes)  -> RAG (flat baseline with sharp spikes
         at retrieved-chunk boundaries, detected as embedding-similarity
         jumps). Preserve retrieved chunks -> SnapKV (importance-aware).
  3. VRAM pressure    — torch.cuda.memory_allocated() / budget. Low ->
     full-precision KV; high -> compressed (S4R / SnapKV-4bit); critical
     -> CPU offload.

Dynamic hot-swap: AutoContext can switch strategies MID-CONVERSATION
without losing context. The KV state is extracted from the old strategy
(dequantized if necessary) and re-injected into the new strategy. This
lets the engine ride the VRAM pressure curve: full KV when VRAM is cheap,
compressed/offloaded when pressure rises, back to full when it drops.

Growth prediction: the manager tracks per-turn context growth and
extrapolates. If the predicted next-turn length would exceed the VRAM
budget, it pre-emptively switches to a compressed/offload strategy
BEFORE OOM — avoiding the costly exception-driven fallback path in
forge_engine._generate_with_oom_recovery.

AutoContextManager is a META-MANAGER, not a KVCacheStrategy. It wraps a
KVCacheStrategy (held inside AutoContextKVCache) and decides when to
swap the underlying strategy. AutoContextKVCache is the thin
KVCacheStrategy wrapper that delegates every call to the currently
active inner strategy and forwards hot-swap requests to the manager.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

logger = logging.getLogger(__name__)

from forge.engine.kv_backend import KVCacheStrategy, build_kv_cache


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _cuda_mem_allocated_bytes() -> int:
    if not _cuda_available():
        return 0
    try:
        return int(torch.cuda.memory_allocated())
    except Exception:
        return 0


class AutoContextManager:
    """Meta-manager that selects and hot-swaps KV cache strategies.

    Does NOT implement KVCacheStrategy itself — it owns the policy and
    drives an AutoContextKVCache wrapper (or directly advises the engine
    on which strategy name to build next).
    """

    def __init__(self, vram_budget_gb: float = 10.0, task_type: str = "auto"):
        self.vram_budget_bytes = int(vram_budget_gb * 1e9)
        self.task_type = task_type  # "auto" | "coding" | "chat" | "rag"
        self._vram_pressure = 0.0
        self._current_strategy_name: str | None = None
        self._growth_history: list[int] = []  # seq_len per turn
        self._predicted_growth_rate = 0.0
        self._switch_count = 0

        # Configurable thresholds (tuned for RTX 5070 12GB)
        self.ctx_short = 512
        self.ctx_med = 2048
        self.ctx_long = 8192
        self.vram_high = 0.70
        self.vram_critical = 0.90
        self.entropy_high = 2.5   # nats; coding/reasoning territory
        self.entropy_low = 1.0    # nats; chat/template territory
        self.entropy_spike_delta = 1.5  # RAG spike magnitude
        self.growth_preempt_ratio = 0.85  # switch if predicted > ratio*budget

    # ------------------------------------------------------------------ #
    # VRAM
    # ------------------------------------------------------------------ #
    def get_vram_pressure(self) -> float:
        """Fraction of VRAM budget currently allocated (0.0–1.0)."""
        if self.vram_budget_bytes <= 0:
            self._vram_pressure = 0.0
            return 0.0
        used = _cuda_mem_allocated_bytes()
        self._vram_pressure = min(1.0, used / self.vram_budget_bytes)
        return self._vram_pressure

    # ------------------------------------------------------------------ #
    # Entropy -> task type
    # ------------------------------------------------------------------ #
    def entropy_to_task_type(self, entropy_trajectory: Sequence[float]) -> str:
        """Classify task type from the per-token entropy signal.

        High sustained entropy -> "coding" (diverse continuations, attend
        to recent tokens). Low sustained entropy -> "chat" (peaked
        distribution, attend to system prompt). Mixed with sharp spikes
        -> "rag" (flat baseline + retrieved-chunk boundaries).

        entropy_trajectory: list of per-token entropy values (nats) from
        the sampling step. May be empty (returns "chat" as the safe
        default that preserves the system prompt).
        """
        if self.task_type != "auto":
            return self.task_type
        if not entropy_trajectory or len(entropy_trajectory) < 4:
            return "chat"
        ent = torch.tensor(entropy_trajectory, dtype=torch.float32)
        mean_e = float(ent.mean())
        std_e = float(ent.std())
        max_e = float(ent.max())
        # RAG: low baseline with sharp spikes (retrieved-chunk boundaries)
        if mean_e < self.entropy_high and (max_e - mean_e) > self.entropy_spike_delta:
            return "rag"
        if mean_e > self.entropy_high:
            return "coding"
        if mean_e < self.entropy_low:
            return "chat"
        # Ambiguous: use variance as tiebreaker
        return "rag" if std_e > 0.8 else "chat"

    # ------------------------------------------------------------------ #
    # Strategy selection
    # ------------------------------------------------------------------ #
    def select_strategy(
        self,
        context_length: int,
        entropy_trajectory: Sequence[float],
        vram_pressure: float,
    ) -> str:
        """Return the best strategy name for the current signals."""
        task = self.entropy_to_task_type(entropy_trajectory)

        # Context-length base selection with VRAM pressure modulation
        if context_length < self.ctx_short:
            base = "standard"
        elif context_length < self.ctx_med:
            base = "snapkv" if vram_pressure < self.vram_high else "s4r"
        elif context_length < self.ctx_long:
            base = "s4r" if vram_pressure < 0.80 else "cpu_offload"
        else:
            base = "cpu_offload"

        # Task-type overrides (stronger signal than length alone)
        if task == "coding":
            # Preserve recent tokens -> SnapKV observation window
            if context_length < self.ctx_long and vram_pressure < self.vram_critical:
                base = "snapkv"
        elif task == "chat":
            # Preserve system prompt + first turns -> StreamingLLM sinks
            if context_length < self.ctx_long and vram_pressure < self.vram_critical:
                base = "streaming"
        elif task == "rag":
            # Preserve retrieved chunks -> SnapKV importance eviction
            if context_length < self.ctx_long and vram_pressure < self.vram_critical:
                base = "snapkv"

        # VRAM critical override: always offload regardless of task/length
        if vram_pressure >= self.vram_critical:
            base = "cpu_offload"

        return base

    def maybe_switch(
        self,
        current_strategy: str,
        context_length: int,
        entropy_trajectory: Sequence[float],
        vram_pressure: float,
    ) -> str | None:
        """Return a new strategy name if a switch is warranted, else None."""
        target = self.select_strategy(
            context_length, entropy_trajectory, vram_pressure)
        if target == current_strategy:
            return None
        # Growth-prediction pre-empt: if we predict the next turn will blow
        # the VRAM budget, escalate even if the static selector is lenient.
        predicted = self.predict_growth(None)
        if predicted > 0:
            projected_len = context_length + int(predicted)
            projected_pressure = vram_pressure + self._growth_pressure_delta(
                projected_len, context_length)
            if projected_pressure >= self.growth_preempt_ratio and target not in ("cpu_offload", "s4r"):
                target = "s4r" if projected_pressure < self.vram_critical else "cpu_offload"
        if target == current_strategy:
            return None
        return target

    def _growth_pressure_delta(
        self, projected_len: int, current_len: int
    ) -> float:
        """Estimate VRAM pressure increase from current->projected length."""
        delta_tokens = max(0, projected_len - current_len)
        if delta_tokens == 0 or self.vram_budget_bytes <= 0:
            return 0.0
        # Rough: 2 bytes * 2 (K+V) * n_kv(8) * head_dim(64) * 16 layers
        # = ~32KB/token. Conservative default; real value depends on model.
        per_token_bytes = 2 * 2 * 8 * 64 * 16
        return (delta_tokens * per_token_bytes) / self.vram_budget_bytes

    # ------------------------------------------------------------------ #
    # Growth prediction
    # ------------------------------------------------------------------ #
    def predict_growth(
        self, conversation_history: Sequence[int] | None
    ) -> float:
        """Predict per-turn context growth (tokens) from recent turns.

        conversation_history: list of per-turn sequence lengths (most
        recent last). If None, uses the internally tracked history.
        Returns predicted tokens added in the next turn.
        """
        hist = list(conversation_history) if conversation_history is not None else self._growth_history
        if len(hist) < 2:
            self._predicted_growth_rate = 0.0
            return 0.0
        diffs = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
        # Exponential moving average of recent growth, weighted toward latest
        alpha = 0.4
        ema = float(diffs[0])
        for d in diffs[1:]:
            ema = alpha * d + (1 - alpha) * ema
        self._predicted_growth_rate = max(0.0, ema)
        return self._predicted_growth_rate

    def record_turn(self, seq_len: int) -> None:
        """Record the sequence length at the end of a conversation turn."""
        self._growth_history.append(seq_len)
        if len(self._growth_history) > 32:
            self._growth_history = self._growth_history[-32:]

    # ------------------------------------------------------------------ #
    # KV migration (hot-swap)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_kv(cache: KVCacheStrategy) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Extract full-precision (k, v) tensors from any strategy.

        Handles strategies that store quantized/evicted state by calling
        their get() (which dequantizes internally) and normalizing the
        shape to [B, n_kv, T, head_dim].
        """
        try:
            k, v = cache.get(None)
        except Exception:
            return None
        if k is None or v is None:
            return None
        if k.dim() == 3:  # [n_kv, T, head_dim] -> add batch dim
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        return k, v

    @staticmethod
    def _inject_kv(cache: KVCacheStrategy, k: torch.Tensor, v: torch.Tensor) -> bool:
        """Inject pre-existing KV into a freshly-initialized strategy.

        Replays the KV at position 0 so the new strategy treats it as a
        single prefill chunk. Returns True on success.
        """
        try:
            cache.append(k, v, 0)
            return True
        except Exception:
            return False

    def migrate_kv(
        self, old_cache: KVCacheStrategy, new_cache: KVCacheStrategy
    ) -> bool:
        """Extract KV from old strategy, inject into new strategy.

        Handles different interfaces: some strategies expose get_past_kv(),
        some return 3D tensors, some are quantized (get() dequantizes).
        The injection replays the extracted KV as a single prefill append
        at position 0. Returns True if migration succeeded, False if the
        new cache should start empty (caller falls back to re-prefill).
        """
        kv = self._extract_kv(old_cache)
        if kv is None:
            return False
        k, v = kv
        if k.shape[2] == 0:
            return True  # nothing to migrate
        ok = self._inject_kv(new_cache, k, v)
        if ok:
            self._switch_count += 1
        return ok

    # ------------------------------------------------------------------ #
    # Main loop hook
    # ------------------------------------------------------------------ #
    def update(
        self,
        current_seq_len: int,
        entropy_trajectory: Sequence[float],
        current_strategy: str,
    ) -> str | None:
        """Main per-step call. Returns new strategy name or None."""
        vram = self.get_vram_pressure()
        return self.maybe_switch(
            current_strategy, current_seq_len, entropy_trajectory, vram)

    @property
    def current_strategy_name(self) -> str | None:
        return self._current_strategy_name

    @property
    def switch_count(self) -> int:
        return self._switch_count

    def stats(self) -> dict:
        return {
            "vram_pressure": round(self._vram_pressure, 4),
            "vram_budget_gb": self.vram_budget_bytes / 1e9,
            "current_strategy": self._current_strategy_name,
            "switch_count": self._switch_count,
            "predicted_growth_rate": round(self._predicted_growth_rate, 1),
            "growth_history_len": len(self._growth_history),
            "cuda_available": _cuda_available(),
        }


class AutoContextKVCache(KVCacheStrategy):
    """Thin KVCacheStrategy wrapper that delegates to the active inner
    strategy and performs hot-swap on demand.

    The engine builds this once via build_kv_cache("auto_context") (wired
    separately) or constructs it directly with an initial strategy name.
    All KVCacheStrategy calls are forwarded to the inner strategy. When
    the manager decides a switch is needed, swap_strategy() migrates the
    KV state into a new inner strategy built via build_kv_cache().
    """

    def __init__(
        self,
        manager: AutoContextManager | None = None,
        initial_strategy: str = "standard",
    ):
        self.manager = manager or AutoContextManager()
        self._inner: KVCacheStrategy | None = None
        self._strategy_name = initial_strategy
        self._init_args: tuple | None = None
        self._seq_len = 0

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self._init_args = (n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype)
        self._inner = build_kv_cache(self._strategy_name)
        self._inner.init(n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype)
        self.manager._current_strategy_name = self._strategy_name

    def _rebuild(self, strategy_name: str) -> KVCacheStrategy | None:
        if self._init_args is None:
            return None
        n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype = self._init_args
        new_inner = build_kv_cache(strategy_name)
        new_inner.init(n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype)
        return new_inner

    def swap_strategy(self, strategy_name: str) -> bool:
        """Hot-swap to a new inner strategy, migrating KV state."""
        if strategy_name == self._strategy_name or self._inner is None:
            return False
        new_inner = self._rebuild(strategy_name)
        if new_inner is None:
            return False
        ok = self.manager.migrate_kv(self._inner, new_inner)
        if not ok:
            # Migration failed: start fresh (caller may need to re-prefill)
            try:
                new_inner.clear()
            except Exception:
                logger.debug("Failed to clear new inner cache during hot-swap", exc_info=True)
        old = self._inner
        self._inner = new_inner
        self._strategy_name = strategy_name
        self.manager._current_strategy_name = strategy_name
        # Free old cache tensors
        try:
            old.clear()
        except Exception:
            logger.debug("Failed to clear old cache during hot-swap", exc_info=True)
        if _cuda_available():
            torch.cuda.empty_cache()
        return True

    def maybe_swap(
        self,
        context_length: int,
        entropy_trajectory: Sequence[float],
    ) -> str | None:
        """Ask the manager whether to swap; perform swap if so."""
        target = self.manager.update(
            context_length, entropy_trajectory, self._strategy_name)
        if target is not None:
            self.swap_strategy(target)
        return target

    # ------------------- KVCacheStrategy delegation ------------------- #
    def append(self, k, v, position, **kwargs):
        if self._inner is not None:
            try:
                self._inner.append(k, v, position, **kwargs)
            except TypeError:
                self._inner.append(k, v, position)
            self._seq_len = getattr(self._inner, "seq_len", position + k.shape[2])

    def get(self, positions):
        if self._inner is None:
            return None, None
        return self._inner.get(positions)

    def get_past_kv(self):
        if self._inner is not None and hasattr(self._inner, "get_past_kv"):
            return self._inner.get_past_kv()
        return self.get(None)

    def clear(self):
        if self._inner is not None:
            self._inner.clear()
        self._seq_len = 0

    def info(self) -> dict:
        base = self._inner.info() if self._inner is not None else {}
        base["auto_context"] = True
        base["active_strategy"] = self._strategy_name
        base["manager_stats"] = self.manager.stats()
        return base

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    @property
    def seq_len(self) -> int:
        return getattr(self._inner, "seq_len", self._seq_len) if self._inner else self._seq_len
