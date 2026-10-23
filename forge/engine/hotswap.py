"""Hot-swap configuration manager for ForgeEngine.

Allows runtime changes to engine settings without restarting or reloading
the model. Supports:
  - KV cache strategy switching (standard → s4r → kvzip → etc.)
  - Decoding strategy switching (standard → eagle3 → medusa → etc.)
  - Generation parameter overrides (temperature, top_k, max_tokens, etc.)
  - Context limit adjustment (infinite context with adjustable ceiling)
  - Feature flag toggling (prefix cache, chunked prefill, etc.)
  - VRAM budget adjustment
  - Per-request overrides

All changes take effect on the NEXT generation request — in-flight
generations are not interrupted.

Usage:
    from forge.engine.hotswap import HotSwapManager
    manager = HotSwapManager(engine)
    manager.set_kv_cache("kvzip")
    manager.set_decoding("eagle3")
    manager.set_context_limit(65536)
    manager.set_generation_defaults(temperature=0.7, max_tokens=512)
    manager.apply_pending()  # called automatically by generate()
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any

from forge.engine.activation import ActivationConfig


@dataclass
class EngineSettings:
    """Live engine settings — all hot-swappable.

    Changes to any field take effect on the next generate() call
    (via HotSwapManager.apply_pending()).
    """
    # ── Core strategies (require re-activation) ──
    kv_cache: str = "paged"
    decoding: str = "standard"
    quantize: str | None = None
    acceleration: str | None = None

    # ── KV cache parameters ──
    kv_cache_tokens: int | None = None  # None = unlimited (VRAM-bounded)
    kv_bits: int = 4

    # ── Context limits ──
    max_context_tokens: int = 32768  # adjustable context window ceiling
    infinite_context: bool = False  # if True, use eviction for unbounded context

    # ── Generation defaults (overridable per-request) ──
    default_temperature: float = 0.0
    default_max_tokens: int = 256
    default_top_p: float = 1.0
    default_top_k: int = 80
    default_repetition_penalty: float = 1.05

    # ── Feature flags ──
    use_compile: bool = False
    use_triton_conv: bool = False
    use_prefix_cache: bool = True
    use_chunked_prefill: bool = False
    use_fused_qk_norm_rope_cache: bool = False
    use_seq_split: bool = False
    warmup: bool = True

    # ── VRAM management ──
    vram_safety_margin_gb: float = 0.5
    auto_offload: bool = True  # auto-offload to CPU when VRAM low

    # ── Parallel generation ──
    max_batch_size: int = 8
    batch_timeout_ms: int = 50  # request batching window

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineSettings:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def diff(self, other: EngineSettings) -> dict[str, Any]:
        """Return fields that differ from `other`."""
        changes = {}
        for f in self.__dataclass_fields__:
            if getattr(self, f) != getattr(other, f):
                changes[f] = getattr(self, f)
        return changes


class HotSwapManager:
    """Manages hot-swappable engine configuration.

    Thread-safe: settings can be changed from any thread (e.g. API handler)
    while the engine is generating. Changes are applied atomically between
    generation requests.
    """

    def __init__(self, engine):
        self.engine = engine
        self._lock = threading.RLock()
        self._current = EngineSettings()
        self._pending: EngineSettings | None = None
        self._overrides: dict[str, dict] = {}  # per-request overrides by request_id
        # Set when a reactivation attempt raised — surfaced via
        # get_settings() so the failure isn't invisible (EC-100).
        self._last_reactivation_error: str | None = None

    @property
    def current(self) -> EngineSettings:
        """Current active settings (read-only)."""
        return self._current

    @property
    def pending(self) -> EngineSettings | None:
        """Pending settings that will be applied on next generate()."""
        return self._pending

    def has_pending(self) -> bool:
        """Check if there are pending changes to apply."""
        return self._pending is not None

    # ── Strategy hot-swap ──

    def set_kv_cache(self, strategy: str):
        """Switch KV cache strategy (e.g. 'standard' → 'kvzip' → 's4r').

        Takes effect on the next generate() call — current KV cache is
        cleared and rebuilt with the new strategy.
        """
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            self._pending.kv_cache = strategy

    def set_decoding(self, strategy: str):
        """Switch decoding strategy (e.g. 'standard' → 'eagle3' → 'medusa')."""
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            self._pending.decoding = strategy

    def set_quantize(self, quant: str | None):
        """Switch quantization (None, 'int8', 'int4', 'fp8', 'w8a8').

        Note: quantization changes require re-loading weights and may
        take several seconds. The engine will be briefly unavailable.
        """
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            self._pending.quantize = quant

    def set_acceleration(self, accel: str | None):
        """Switch acceleration mode (None, 'cuda_graph', 'megakernel')."""
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            self._pending.acceleration = accel

    # ── Context limit hot-swap ──

    def set_context_limit(self, max_tokens: int):
        """Set the maximum context window (in tokens).

        Set to a very large number (e.g. 1_000_000) for effectively infinite
        context. The KV cache eviction strategy will keep only the most
        important tokens within the VRAM budget.
        """
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            self._pending.max_context_tokens = max_tokens
            self._pending.kv_cache_tokens = max_tokens

    def set_infinite_context(self, enabled: bool = True, budget: int = 100_000):
        """Enable/disable infinite context mode.

        When enabled, uses KV cache eviction (streaming/snapkv/kvzip) to
        maintain unbounded context within the VRAM budget. The `budget`
        controls how many tokens are kept in VRAM before eviction kicks in.
        """
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            self._pending.infinite_context = enabled
            if enabled:
                self._pending.max_context_tokens = 1_000_000
                self._pending.kv_cache_tokens = budget
                # Auto-select an eviction strategy if currently using standard
                if self._pending.kv_cache in ("standard", "paged"):
                    self._pending.kv_cache = "streaming"
            else:
                self._pending.max_context_tokens = 32768
                self._pending.kv_cache_tokens = None

    # ── Generation defaults ──

    def set_generation_defaults(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ):
        """Set default generation parameters. Overridable per-request."""
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            if temperature is not None:
                self._pending.default_temperature = temperature
            if max_tokens is not None:
                self._pending.default_max_tokens = max_tokens
            if top_p is not None:
                self._pending.default_top_p = top_p
            if top_k is not None:
                self._pending.default_top_k = top_k
            if repetition_penalty is not None:
                self._pending.default_repetition_penalty = repetition_penalty

    # ── Feature flags ──

    def set_feature(self, flag: str, enabled: bool):
        """Toggle a feature flag (e.g. 'use_prefix_cache', 'use_compile')."""
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            if hasattr(self._pending, flag):
                setattr(self._pending, flag, enabled)
            else:
                self.engine._log(
                    f"hotswap.set_feature: unknown flag '{flag}' ignored",
                    level="warn", source="hotswap")

    def set_features(self, **flags):
        """Toggle multiple feature flags at once."""
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            for flag, enabled in flags.items():
                if hasattr(self._pending, flag):
                    setattr(self._pending, flag, enabled)
                else:
                    self.engine._log(
                        f"hotswap.set_features: unknown flag '{flag}' "
                        "ignored", level="warn", source="hotswap")

    # ── VRAM management ──

    def set_vram_margin(self, margin_gb: float):
        """Set VRAM safety margin (controls when auto-offload triggers)."""
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            self._pending.vram_safety_margin_gb = margin_gb

    def set_auto_offload(self, enabled: bool):
        """Enable/disable automatic CPU offload when VRAM is low."""
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            self._pending.auto_offload = enabled

    # ── Parallel generation ──

    def set_batch_config(self, max_batch: int | None = None,
                         timeout_ms: int | None = None):
        """Configure request batching for parallel generation."""
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            if max_batch is not None:
                self._pending.max_batch_size = max_batch
            if timeout_ms is not None:
                self._pending.batch_timeout_ms = timeout_ms

    # ── Bulk update ──

    def update(self, **changes):
        """Update multiple settings at once.

        Example:
            manager.update(kv_cache="kvzip", decoding="eagle3",
                          temperature=0.7, max_context_tokens=65536)
        """
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            for key, value in changes.items():
                if hasattr(self._pending, key):
                    setattr(self._pending, key, value)
                else:
                    self.engine._log(
                        f"hotswap.update: unknown setting '{key}' ignored",
                        level="warn", source="hotswap")

    def update_from_dict(self, d: dict[str, Any]):
        """Update from a dictionary (e.g. JSON API request body)."""
        with self._lock:
            if self._pending is None:
                self._pending = EngineSettings(**asdict(self._current))
            for key, value in d.items():
                if value is None:
                    continue
                if hasattr(self._pending, key):
                    setattr(self._pending, key, value)
                else:
                    self.engine._log(
                        f"hotswap.update_from_dict: unknown setting "
                        f"'{key}' ignored", level="warn", source="hotswap")

    # ── Per-request overrides ──

    def set_request_override(self, request_id: str, **overrides):
        """Set per-request generation overrides.

        These override the global defaults for a single request.
        Cleared after the request completes.
        """
        with self._lock:
            self._overrides[request_id] = overrides

    def get_request_overrides(self, request_id: str) -> dict:
        """Get per-request overrides and clean up."""
        with self._lock:
            return self._overrides.pop(request_id, {})

    # ── Apply pending changes ──

    def apply_pending(self) -> bool:
        """Apply pending settings to the engine.

        Called automatically by generate() before each generation.
        Returns True if changes were applied.

        Strategy changes (KV cache, decoding) require re-activation.
        Generation defaults are applied instantly.
        """
        with self._lock:
            if self._pending is None:
                return False

            pending = self._pending
            changes = self._current.diff(pending)
            if not changes:
                self._pending = None
                return False

            # Determine what needs re-activation vs instant update
            reactivation_fields = {
                "kv_cache", "decoding", "quantize", "acceleration",
                "kv_cache_tokens", "kv_bits", "max_context_tokens",
                "use_compile", "use_triton_conv", "use_prefix_cache",
                "use_chunked_prefill", "use_fused_qk_norm_rope_cache",
                "use_seq_split", "warmup", "infinite_context",
            }
            needs_reactivation = bool(changes.keys() & reactivation_fields)

            if needs_reactivation:
                try:
                    self._apply_reactivation(pending)
                except Exception as e:
                    # A failed reactivation must not leave _pending set —
                    # apply_pending() runs before EVERY generate(), so the
                    # known-failing config would retry forever.  Clear it,
                    # keep _current, record the failure for get_settings().
                    self._pending = None
                    self._last_reactivation_error = str(e)
                    raise
                self._last_reactivation_error = None

            self._current = EngineSettings(**asdict(pending))
            self._pending = None
            return True

    def _apply_reactivation(self, settings: EngineSettings):
        """Re-activate engine strategies with new settings."""
        engine = self.engine
        old_kv = engine.kv_cache

        # Seed from the engine's ACTIVE config so previously-enabled
        # feature flags (avmp, learned_prefix_cache, ...) survive the
        # swap — a bare ActivationConfig would silently drop them while
        # leaving the stale instances half-active (EC-115).
        base = engine.active_config
        if base is None:
            base = ActivationConfig()
        config = ActivationConfig(**base.to_dict())
        config.kv_cache = settings.kv_cache
        config.decoding = settings.decoding
        config.quantize = settings.quantize
        config.acceleration = settings.acceleration
        config.kv_bits = settings.kv_bits
        config.kv_cache_tokens = settings.kv_cache_tokens
        config.use_compile = settings.use_compile
        config.use_triton_conv = settings.use_triton_conv
        config.use_prefix_cache = settings.use_prefix_cache
        config.use_chunked_prefill = settings.use_chunked_prefill
        config.use_fused_qk_norm_rope_cache = (
            settings.use_fused_qk_norm_rope_cache)
        config.use_seq_split = settings.use_seq_split
        config.warmup = False  # don't re-warmup on hot-swap

        # Clear old KV cache if strategy changed
        kv_changed = settings.kv_cache != self._current.kv_cache
        if old_kv is not None and kv_changed:
            if hasattr(old_kv, 'clear'):
                old_kv.clear()

        # Re-activate
        try:
            engine.activate_config(config)
            engine._log(
                f"Hot-swap applied: {settings.diff(self._current)}",
                source="hotswap")
        except Exception as e:
            engine._log(f"Hot-swap failed: {e}", level="error",
                        source="hotswap")
            # Rollback: restore old settings
            raise

    # ── Query ──

    def get_settings(self) -> dict[str, Any]:
        """Get current settings as a dict (for API responses)."""
        with self._lock:
            result = self._current.to_dict()
            if self._pending:
                result["_pending"] = self._pending.diff(self._current)
            if self._last_reactivation_error:
                result["_reactivation_error"] = self._last_reactivation_error
            return result

    def get_pending_changes(self) -> dict[str, Any]:
        """Get pending changes (not yet applied)."""
        with self._lock:
            if self._pending is None:
                return {}
            return self._pending.diff(self._current)
