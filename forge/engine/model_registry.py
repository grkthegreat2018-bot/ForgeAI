"""Model Registry — multi-engine VRAM-budgeted manager.

Manages multiple ForgeEngine instances on a single GPU with explicit
byte-level VRAM budgeting. Supports sleep/wake hot-swap, concurrent
multi-model loading, and automatic VRAM defragmentation.

Key differentiator vs vLLM/LM Studio:
  - Byte budgets, not utilization fractions (avoids vLLM V1 bug)
  - Concurrent multi-model when VRAM allows (LM Studio does one at a time)
  - Sleep/wake with sub-3s round-trip target
  - Automatic eviction of idle models under VRAM pressure

Usage:
    from forge.engine.model_registry import ModelRegistry

    registry = ModelRegistry()
    registry.register("forgelm-v2-jamba", checkpoint="...", config="forgelm_v2", vram_budget_gb=8.0)
    registry.register("qwen2.5", checkpoint="...", config="qwen25_05b", vram_budget_gb=3.5)

    # Generate with either model — registry handles wake/sleep automatically
    out = registry.generate("forgelm-v2-jamba", "def fibonacci(n):")
    out = registry.generate("qwen2.5", "Explain quantum computing")
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from forge.engine.engine_common import _DEFAULT_TOKENIZER_PATH
from forge.engine.forge_engine import ForgeEngine, _checkpoint_size_cache
import contextlib

logger = logging.getLogger(__name__)


@dataclass
class EngineEntry:
    """Metadata for a registered engine."""
    model_id: str
    engine: ForgeEngine
    vram_budget_bytes: int
    checkpoint_path: str
    config_name: str
    tokenizer_path: str
    is_awake: bool = True
    last_used: float = field(default_factory=time.time)
    generation_count: int = 0
    total_tokens: int = 0
    # Registry-driven generations in progress (EC-199): busy engines are
    # never eviction candidates.
    in_flight: int = 0
    # Actual VRAM footprint measured via mem_get_info delta at load
    # (EC-161); 0 when unmeasured (CPU device or pre-measure loads).
    measured_vram_bytes: int = 0


class VRAMBudgetExceeded(Exception):
    """Not enough VRAM to load the model even after evicting idle engines."""
    pass


class ModelRegistry:
    """Multi-engine manager with explicit VRAM budgeting.

    Key properties:
      - Byte-level budgets: each engine reserves an absolute byte budget.
        This avoids the vLLM V1 bug where `gpu_memory_utilization` fractions
        don't account for already-loaded models.
      - Sleep/wake: idle engines are put to sleep (Level 1: CPU offload)
        when VRAM is needed for a new model.
      - Concurrent loading: if VRAM budget allows, multiple models stay
        awake simultaneously.
      - Thread-safe: all public methods use a reentrant lock.
    """

    def __init__(self, device: str = "cuda", safety_margin: float = 0.10):
        self.device = torch.device(device)
        self.safety_margin = safety_margin  # 10% VRAM headroom
        self._entries: dict[str, EngineEntry] = {}
        self._lock = threading.RLock()
        # Bytes reserved by models currently loading off-lock (EC-199
        # moved from_checkpoint out of the lock) — counted as unavailable
        # so two concurrent register() calls can't both pass _make_room.
        self._inflight_loads: dict[str, int] = {}

        # VRAM capacity
        if self.device.type == "cuda":
            self._total_vram = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self._total_vram = 32 * 1024**3  # Assume 32GB system RAM

    # ── Public API ────────────────────────────────────────────────────────

    def register(self, model_id: str, checkpoint: str, config_name: str,
                 tokenizer_path: str | None = None,
                 vram_budget_gb: float = 0,
                 **engine_kwargs) -> ForgeEngine:
        """Register and load a model, managing VRAM automatically.

        If VRAM is insufficient, idle engines are put to sleep to free space.
        If vram_budget_gb is 0, auto-calculates from checkpoint file size.

        Returns the loaded ForgeEngine (or existing if already registered).
        """
        auto_budget = vram_budget_gb <= 0
        with self._lock:
            # Return existing if already loaded
            if model_id in self._entries:
                entry = self._entries[model_id]
                self._ensure_awake(model_id)
                entry.last_used = time.time()
                return entry.engine

            # Calculate VRAM budget
            if auto_budget:
                ckpt_size = _checkpoint_size_cache.get(checkpoint)
                if ckpt_size is None:
                    ckpt_size = Path(checkpoint).stat().st_size
                    _checkpoint_size_cache[checkpoint] = ckpt_size
                vram_budget_gb = (ckpt_size * 1.5) / 1e9  # 50% overhead for KV + activations
                logger.info("Auto budget for %s: %.2f GB", model_id, vram_budget_gb)
            budget_bytes = int(vram_budget_gb * 1e9)

            # Ensure VRAM is available
            self._make_room(budget_bytes, exclude=model_id)
            tok_path = tokenizer_path or _DEFAULT_TOKENIZER_PATH
            self._inflight_loads[model_id] = budget_bytes

        # Load the model OUTSIDE the registry lock: checkpoint I/O + warmup
        # take seconds and must not freeze generate()/stats()/list_models()
        # for every other model (EC-199).
        free_before = (self._free_vram()
                       if self.device.type == "cuda" else None)
        try:
            engine = ForgeEngine.from_checkpoint(
                checkpoint=checkpoint,
                config_name=config_name,
                tokenizer_path=tok_path,
                device=str(self.device),
                **engine_kwargs,
            )
        finally:
            with self._lock:
                self._inflight_loads.pop(model_id, None)
        # EC-161: measure the real footprint instead of trusting the
        # declared 1.5x-checkpoint estimate.
        measured = (max(0, free_before - self._free_vram())
                    if free_before is not None else 0)

        with self._lock:
            existing = self._entries.get(model_id)
            if existing is not None:
                # Lost a concurrent register() race — discard the duplicate
                # load and hand back the winner.
                with contextlib.suppress(Exception):
                    engine.sleep(level=2)
                return existing.engine
            entry = EngineEntry(
                model_id=model_id,
                engine=engine,
                vram_budget_bytes=(
                    measured if (auto_budget and measured > 0)
                    else budget_bytes),
                checkpoint_path=checkpoint,
                config_name=config_name,
                tokenizer_path=tok_path,
                measured_vram_bytes=measured,
            )
            self._entries[model_id] = entry

            logger.info("Registered: %s (budget=%.2fGB%s, free=%.2fGB)",
                        model_id, entry.vram_budget_bytes / 1e9,
                        " measured" if (auto_budget and measured > 0)
                        else " declared",
                        self._free_vram() / 1e9)
            return engine

    def generate(self, model_id: str, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.0, top_p: float = 1.0,
                 finish_sentence: bool = True,
                 top_k: int = 80, repetition_penalty: float = 1.05,
                 stop: list[str] | None = None, seed: int | None = None,
                 **kwargs) -> str:
        """Generate text from a registered model. Auto-wakes if asleep."""
        with self._lock:
            if model_id not in self._entries:
                raise KeyError(f"Model '{model_id}' not registered. "
                               f"Available: {list(self._entries.keys())}")

            self._ensure_awake(model_id)
            entry = self._entries[model_id]
            entry.last_used = time.time()
            entry.in_flight += 1

        def _call():
            return entry.engine.generate(
                prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                finish_sentence=finish_sentence,
                top_k=top_k, repetition_penalty=repetition_penalty,
                stop=stop,
                min_p=kwargs.get('min_p', 0.0),
                min_k=kwargs.get('min_k', 0.0),
                dry_multiplier=kwargs.get('dry_multiplier', 0.0),
                dry_base=kwargs.get('dry_base', 1.75),
                dry_allowed_length=kwargs.get('dry_allowed_length', 2),
                dry_penalty_last_n=kwargs.get('dry_penalty_last_n', 512),
                logprobs=kwargs.get('logprobs'),
                prompt_logprobs=kwargs.get('prompt_logprobs'),
                return_logprobs=kwargs.get('return_logprobs', False),
            )

        try:
            # Hold the engine gen lock across the whole call: the token
            # delta below is then exact (EC-061), and a per-request seed
            # can be applied inside a forked RNG scope without another
            # request's seed interleaving (EC-156).  _gen_lock is an
            # RLock, so engine.generate() re-enters harmlessly.
            devices = ([entry.engine.device]
                       if getattr(entry.engine.device, "type", "")
                       == "cuda" else [])
            with entry.engine._gen_lock:
                tokens_before = getattr(
                    entry.engine, "total_tokens_generated", 0)
                if seed is not None:
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(seed)
                        output = _call()
                else:
                    output = _call()
        finally:
            with self._lock:
                entry.in_flight -= 1

        with self._lock:
            entry.generation_count += 1
            # EC-061: credit the request DELTA, not the engine's lifetime
            # cumulative counter (which double-counted quadratically).
            entry.total_tokens += max(
                0, getattr(entry.engine, "total_tokens_generated", 0)
                - tokens_before)
        return output

    def generate_stream(self, model_id: str, prompt: str, max_new_tokens: int = 256,
                        temperature: float = 0.0, top_p: float = 1.0,
                        top_k: int = 80, repetition_penalty: float = 1.05,
                        min_p: float = 0.0, min_k: float = 0.0,
                        dry_multiplier: float = 0.0, dry_base: float = 1.75,
                        dry_allowed_length: int = 2,
                        dry_penalty_last_n: int = 512,
                        eos_token_ids: list[int] | None = None,
                        skip_special_tokens: bool = True,
                        logits_processor=None,
                        seed: int | None = None,
                        stop: list[str] | None = None):
        """Token-by-token streaming generator. Auto-wakes if asleep.

        Yields decoded text chunks (one per token) as they are generated.
        Sampling args mirror ForgeEngine.generate_stream.
        """
        with self._lock:
            if model_id not in self._entries:
                raise KeyError(f"Model '{model_id}' not registered. "
                               f"Available: {list(self._entries.keys())}")
            self._ensure_awake(model_id)
            entry = self._entries[model_id]
            entry.last_used = time.time()
            entry.in_flight += 1
            engine = entry.engine

        def _stream():
            return engine.generate_stream(
                prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty, min_p=min_p,
                min_k=min_k, eos_token_ids=eos_token_ids,
                dry_multiplier=dry_multiplier, dry_base=dry_base,
                dry_allowed_length=dry_allowed_length,
                dry_penalty_last_n=dry_penalty_last_n,
                skip_special_tokens=skip_special_tokens,
                logits_processor=logits_processor,
                stop=stop,
            )

        tokens_before = 0
        try:
            # Same discipline as generate(): engine._gen_lock (RLock)
            # around the iteration so the token delta is exact (EC-061)
            # and a forked RNG scope keeps a per-request seed from
            # clobbering concurrent sampling / leaking globally (EC-156).
            devices = ([engine.device]
                       if getattr(engine.device, "type", "") == "cuda"
                       else [])
            with engine._gen_lock:
                tokens_before = getattr(
                    engine, "total_tokens_generated", 0)
                if seed is not None:
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(seed)
                        for chunk in _stream():
                            yield chunk
                else:
                    for chunk in _stream():
                        yield chunk
        finally:
            with self._lock:
                entry.in_flight -= 1
                entry.generation_count += 1
                entry.total_tokens += max(
                    0, getattr(engine, "total_tokens_generated", 0)
                    - tokens_before)

    def sleep(self, model_id: str, level: int = 1):
        """Put a model to sleep to free VRAM."""
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None or not self._sync_awake(entry):
                return
            # engine.sleep() blocks on the engine _gen_lock for the whole
            # in-flight decode — run it off the registry lock (EC-199).
            self._sleep_unlocked(entry, level)
            if self._entries.get(model_id) is entry:
                entry.is_awake = False
            logger.info("%s → sleep level %s (free=%.2fGB)",
                        model_id, level, self._free_vram() / 1e9)

    def wake(self, model_id: str):
        """Wake a sleeping model."""
        with self._lock:
            self._ensure_awake(model_id)

    def switch(self, from_model: str, to_model: str):
        """Sleep one model, wake another. Target: <3s round-trip."""
        with self._lock:
            entry = self._entries.get(from_model)
            if entry is not None and self._sync_awake(entry):
                self._sleep_unlocked(entry, 1)
                if self._entries.get(from_model) is entry:
                    entry.is_awake = False
            self._ensure_awake(to_model)

    def unregister(self, model_id: str):
        """Remove a model from the registry and free all its resources."""
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                return
            if self._sync_awake(entry):
                self._sleep_unlocked(entry, 2)  # Discard weights
                entry.is_awake = False
            del entry.engine
            del self._entries[model_id]
            torch.cuda.empty_cache()
            logger.info("Unregistered: %s", model_id)

    def list_models(self) -> list[dict]:
        """List all registered models with status."""
        with self._lock:
            return [
                {
                    "id": e.model_id,
                    # Reflect sleeps triggered outside the registry
                    # (direct engine.sleep()) — EC-161.
                    "awake": self._sync_awake(e),
                    "vram_budget_gb": e.vram_budget_bytes / 1e9,
                    "config": e.config_name,
                    "last_used_ago_s": time.time() - e.last_used,
                    "generations": e.generation_count,
                    "total_tokens": e.total_tokens,
                    "in_flight": e.in_flight,
                }
                for e in self._entries.values()
            ]

    def get_engine(self, model_id: str):
        """Get the ForgeEngine for a registered model (auto-wakes if asleep).

        Returns None if model_id is not registered.
        """
        with self._lock:
            if model_id not in self._entries:
                return None
            self._ensure_awake(model_id)
            entry = self._entries[model_id]
            entry.last_used = time.time()
            return entry.engine

    def config_name_for(self, model_id: str) -> str | None:
        """Return the registered config preset name for a model_id."""
        with self._lock:
            entry = self._entries.get(model_id)
            return entry.config_name if entry is not None else None

    def stats(self) -> dict:
        """Aggregate registry statistics."""
        with self._lock:
            awake_count = sum(
                1 for e in self._entries.values() if self._sync_awake(e))
            asleep_count = len(self._entries) - awake_count
            awake_vram = sum(
                e.vram_budget_bytes for e in self._entries.values() if e.is_awake)
            return {
                "total_models": len(self._entries),
                "awake": awake_count,
                "asleep": asleep_count,
                "awake_vram_gb": awake_vram / 1e9,
                "total_vram_gb": self._total_vram / 1e9,
                "free_vram_gb": self._free_vram() / 1e9,
                "models": self.list_models(),
            }

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _sync_awake(entry: EngineEntry) -> bool:
        """Sync ``entry.is_awake`` with the engine's live sleep state.

        Sleeps triggered outside the registry (a direct ``engine.sleep()``
        via ``get_engine``) used to leave the registry claiming the model
        was awake (EC-161).  Returns the synced state.
        """
        awake = getattr(entry.engine, "_awake", None)
        if awake is not None:
            entry.is_awake = bool(awake)
        return entry.is_awake

    def _release_lock(self):
        """Fully release this thread's hold on ``self._lock``.

        Returns an opaque token for :meth:`_reacquire_lock`.  Works at any
        recursion depth (RLock ``_release_save``/``_acquire_restore`` —
        the same protocol ``threading.Condition`` uses internally).
        Blocking engine calls (``sleep``/``wake``/load) wait on
        ``engine._gen_lock``, which an in-flight decode can hold for
        seconds; running them while holding the registry lock freezes
        every other registry op (EC-199).
        """
        release_save = getattr(self._lock, "_release_save", None)
        if release_save is not None:
            return ("saved", release_save())
        self._lock.release()  # fallback: single-level release
        return ("single", None)

    def _reacquire_lock(self, token):
        """Undo :meth:`_release_lock`, restoring the prior hold depth."""
        kind, state = token
        if kind == "saved":
            self._lock._acquire_restore(state)
        else:
            self._lock.acquire()

    def _sleep_unlocked(self, entry: EngineEntry, level: int):
        """``entry.engine.sleep()`` without holding the registry lock."""
        token = self._release_lock()
        try:
            entry.engine.sleep(level=level)
        finally:
            self._reacquire_lock(token)

    def _free_vram(self) -> int:
        """Get currently free VRAM in bytes."""
        if self.device.type == "cuda":
            return torch.cuda.mem_get_info(self.device)[0]
        return self._total_vram  # CPU: assume all free

    def _usable_vram(self) -> int:
        """VRAM available for new models (after safety margin)."""
        return int(self._total_vram * (1.0 - self.safety_margin))

    def _reserved_vram(self) -> int:
        """Total VRAM reserved by awake engines."""
        return sum(e.vram_budget_bytes for e in self._entries.values() if e.is_awake)

    def _ensure_awake(self, model_id: str):
        """Wake a model, sleeping others if VRAM is tight.

        Caller must hold ``self._lock``; the blocking ``engine.wake()``
        is internally run with the lock released (EC-199).
        """
        entry = self._entries[model_id]
        if self._sync_awake(entry):
            return

        # Check if we need to free VRAM.  Bytes reserved by in-flight
        # register() loads aren't reclaimable — subtract them.
        needed = entry.vram_budget_bytes
        free = max(0, self._free_vram() - sum(self._inflight_loads.values()))
        reserved = self._reserved_vram()

        if free + reserved < needed:
            # Not enough total VRAM even if we sleep everything
            raise VRAMBudgetExceeded(
                f"Model '{model_id}' needs {needed/1e9:.2f}GB but only "
                f"{(free + reserved)/1e9:.2f}GB is available on the device"
            )

        if free < needed:
            # Sleep idle models (LRU order) until we have enough
            self._evict_lru(needed - free, exclude=model_id)

        # engine.wake() blocks on _gen_lock — drop the registry lock so
        # in-flight decodes can't freeze every other registry op.
        token = self._release_lock()
        try:
            entry.engine.wake()
        finally:
            self._reacquire_lock(token)
        if self._entries.get(model_id) is not entry:
            raise KeyError(
                f"Model '{model_id}' was unregistered during wake")
        entry.is_awake = True

    def _make_room(self, needed_bytes: int, exclude: str | None = None):
        """Ensure `needed_bytes` of free VRAM by sleeping idle engines."""
        free = max(0, self._free_vram() - sum(self._inflight_loads.values()))
        reserved = self._reserved_vram()

        if free + reserved < needed_bytes:
            raise VRAMBudgetExceeded(
                f"Need {needed_bytes/1e9:.2f}GB but device only has "
                f"{(free + reserved)/1e9:.2f}GB total (free={free/1e9:.2f}GB)"
            )

        if free < needed_bytes:
            self._evict_lru(needed_bytes - free, exclude=exclude)

    def _evict_lru(self, needed_bytes: int, exclude: str | None = None):
        """Sleep idle engines in LRU order until `needed_bytes` is freed.

        Caller must hold ``self._lock``.  Victims are selected under the
        lock, but ``engine.sleep()`` runs with the lock released — sleep
        waits on the engine ``_gen_lock``, so holding the registry lock
        across it froze all registry ops for the duration of an in-flight
        decode, and a busy engine merely *blocked* the eviction loop
        (EC-199).  Engines with in-flight registry generations are never
        eviction candidates.
        """
        # Sort awake, idle engines by last_used (oldest first), exclude
        # the target and any engine with an in-flight generation.
        candidates = sorted(
            (e for e in self._entries.values()
             if self._sync_awake(e) and e.model_id != exclude
             and e.in_flight == 0),
            key=lambda e: e.last_used,
        )

        freed = 0
        slept: list[EngineEntry] = []
        token = self._release_lock()
        try:
            for entry in candidates:
                try:
                    entry.engine.sleep(level=1)
                except Exception as e:
                    # A failed sleep must not wedge the whole eviction —
                    # skip it and keep trying other victims.
                    logger.warning("Eviction sleep failed for %s: %s",
                                   entry.model_id, e)
                    continue
                slept.append(entry)
                freed += entry.vram_budget_bytes
                if freed >= needed_bytes:
                    break
        finally:
            self._reacquire_lock(token)

        for entry in slept:
            if self._entries.get(entry.model_id) is entry:
                entry.is_awake = False
            logger.info("Evicted %s (idle %.0fs, freed %.2fGB)",
                        entry.model_id, time.time() - entry.last_used,
                        entry.vram_budget_bytes / 1e9)
        if freed < needed_bytes:
            raise VRAMBudgetExceeded(
                f"Could not free {needed_bytes/1e9:.2f}GB after evicting "
                f"{len(slept)} idle engines "
                f"({len(candidates) - len(slept)} skipped/failed)"
            )


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    registry = ModelRegistry()
    print(f"Device: {registry.device}")
    print(f"Total VRAM: {registry._total_vram / 1e9:.2f} GB")
    print(f"Free VRAM: {registry._free_vram() / 1e9:.2f} GB")
    print(f"Usable VRAM: {registry._usable_vram() / 1e9:.2f} GB")
