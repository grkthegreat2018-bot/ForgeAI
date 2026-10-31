"""Sleep/wake/VRAM lifecycle and crash-recovery mixin for ForgeEngine."""
import os

import torch

from .activation import ActivationConfig
from .engine_common import (  # noqa: F401
    _CKPT_CACHE_MAX,
    _DEFAULT_CPU_MEMORY_BYTES,
    _DEFAULT_EOS_TOKEN_IDS,
    _QWEN_TOKENIZER_PATH,
    _QWEN_VOCAB,
    _checkpoint_metadata_cache,
    _checkpoint_size_cache,
    _ckpt_cache_lock,
    _is_memory_model,
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    logger,
)
from .errors import CheckpointError, ConfigurationError


class _LifecycleMixin:
    # ── Sleep / wake ──────────────────────────────────────────────────────

    def sleep(self, level: int = 1) -> None:
        """Release GPU memory by offloading model weights.

        Level 1 (default): Move weights to CPU RAM. Fast wake (~2-3s).
            Preserves tokenizer, config, KV cache strategies, and CUDA context.
        Level 2: Discard weights entirely. Slower wake (reload from disk).
            Use for model switching when Level 1 CPU RAM is insufficient.

        After sleep, generation will fail until wake() is called.
        """
        # Serialise against in-flight generation: swapping weights off the
        # device mid-forward corrupts the active decode state.
        with self._gen_lock:
            self._sleep_impl(level)

    def _sleep_impl(self, level: int) -> None:
        if level not in (1, 2):
            raise ConfigurationError(
                f"sleep level must be 1 or 2, got {level}")
        if not self._awake and level <= self._sleep_level:
            return  # Already asleep
        is_flux = _is_memory_model(self.model)
        if level == 2 and not self.checkpoint_path and not is_flux:
            raise RuntimeError("Sleep level 2 requires a checkpoint path")

        if level == 1:
            if is_flux:
                # FluxLM tensors are plain attrs — nn.Module.to() is a
                # no-op on them.  Dense readout offloads via to_device();
                # cuda_primary GPU tables can't be torn down piecemeal,
                # so escalate to the snapshot path: VRAM is actually freed
                # and live memory survives the sleep (EC-120).
                if getattr(self.model, "_gpu", False):
                    self._log("Sleep level 1 on cuda_primary FluxLM: GPU "
                              "tables can't offload piecemeal — "
                              "escalating to snapshot sleep",
                              level="warn")
                    level = 2
                else:
                    self.model.to_device("cpu")
                    self._clear_cuda_cache()
                    self._awake = False
                    self._sleep_level = 1
                    self._log("Sleep level 1: weights offloaded to CPU")
                    return
            else:
                self.model.to("cpu", non_blocking=True)
                self._clear_cuda_cache()
                self._awake = False
                self._sleep_level = 1
                self._log("Sleep level 1: weights offloaded to CPU")
                return

        # Store minimal state, discard model
        self._stored_config = getattr(self.model, "config", None)
        self._stored_dtype = self.dtype
        self._stored_is_flux = is_flux
        if is_flux:
            # FluxLM's weights ARE its live memory — snapshot to a temp
            # file before discarding so session learning survives the
            # sleep; wake() reloads from it (not the last manual save)
            import tempfile
            fd, snap = tempfile.mkstemp(suffix=".flux",
                                        prefix="flux_sleep_")
            os.close(fd)
            self.model.snapshot(snap)
            self._stored_checkpoint = snap
        else:
            self._stored_checkpoint = self.checkpoint_path
        self._profiler.model = None
        # Release acceleration resources that hold CUDA memory/graphs.
        # Without this, sleep(level=2) leaks CUDA graph + megakernel memory.
        self._release_acceleration_resources()
        self.model = None
        self._clear_cuda_cache()
        self._awake = False
        self._sleep_level = 2
        if self.device.type == "cuda":
            free, _ = self._memory_info(self.device)
            self._log(f"Sleep level 2: weights discarded, "
                      f"{free / 1e9:.1f}GB free")
        else:
            self._log("Sleep level 2: weights discarded")

    def wake(self) -> None:
        """Restore model to GPU and resume inference.

        Level 1 wake: CPU→GPU copy (~2-3s). Preserves all strategies.
        Level 2 wake: Reload from checkpoint (~5-10s). Strategies must be re-activated.
        """
        with self._gen_lock:
            self._wake_impl()

    def _wake_impl(self) -> None:
        if self._awake:
            return  # Already awake

        if self._sleep_level == 1:
            if _is_memory_model(self.model):
                # cuda_primary tables never left; only the dense readout
                # may need moving back.
                if not getattr(self.model, "_gpu", False):
                    self.model.to_device(str(self.device))
            else:
                self.model.to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._awake = True
            self._sleep_level = 0
            self._log("Woke from level 1 sleep")
            return

        if not getattr(self, "_stored_checkpoint", None):
            raise CheckpointError(
                "Level 2 wake requires stored checkpoint path",
                suggestion="Load from checkpoint again with from_checkpoint().")
        if getattr(self, "_stored_is_flux", False):
            from forge.model.flux import FluxLM
            try:
                self.model = FluxLM.load(self._stored_checkpoint,
                                         device=str(self.device))
            except Exception as e:
                # Keep _stored_checkpoint so a retry can still succeed;
                # _awake stays False and generate() refuses via
                # _require_awake — surface the reason instead of
                # wedging silently (EC-120).
                self.model = None
                raise CheckpointError(
                    f"Level 2 wake failed — FluxLM snapshot load: {e}",
                    suggestion="Retry wake(); if the temp snapshot is "
                               "corrupt, reload with from_flux().") from e
            try:
                os.unlink(self._stored_checkpoint)   # temp sleep snapshot
            except OSError as e:
                self._log(f"Could not remove temp sleep snapshot "
                          f"{self._stored_checkpoint}: {e}", level="warn")
        else:
            from forge.model_loader import ModelLoader
            self.model = ModelLoader.build_model_fast(
                self._stored_config, checkpoint_path=self._stored_checkpoint,
                dtype=self._stored_dtype)
            self.model.to(self.device)
        self.model.eval()
        self._profiler.model = self.model
        del self._stored_config
        del self._stored_dtype
        del self._stored_checkpoint
        if hasattr(self, "_stored_is_flux"):
            del self._stored_is_flux
        self._awake = True
        self._sleep_level = 0
        self._log("Woke from level 2 sleep (reloaded from checkpoint)")

        # Re-activate strategies that were lost when the model was discarded.
        params = getattr(self, "_last_activation_params", None)
        if params:
            self._awake = True
            config = ActivationConfig.from_kwargs(**params)
            self.activate_config(config)
            self._log("Re-activated strategies after level 2 wake")

    @property
    def is_awake(self) -> bool:
        return self._awake

    def vram_usage(self) -> dict:
        """Report current VRAM usage for this engine."""
        if self.device.type != "cuda":
            return {
                "total_gb": 0,
                "free_gb": 0,
                "used_gb": 0,
                "model_weights_gb": 0,
                "percent": 0,
            }
        free, total = self._memory_info(self.device)
        used = total - free
        model_bytes = 0
        if self.is_awake and self.model is not None:
            # Memory models (FluxLM) store weights in plain tensor
            # attrs/GPU tables — parameters() reports ~0 while GBs are
            # resident. Prefer the model's own accounting hook (EC-128).
            report = getattr(self.model, "memory_report", None)
            if callable(report):
                try:
                    model_bytes = int(report().get("vram_MB_est", 0) * 1e6)
                except Exception:
                    model_bytes = 0
            if not model_bytes:
                model_bytes = sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in self.model.parameters()
                    if parameter.device.type == "cuda"
                )
        return {
            "total_gb": total / 1e9,
            "free_gb": free / 1e9,
            "used_gb": used / 1e9,
            "model_weights_gb": model_bytes / 1e9,
            "percent": used / total * 100,
        }

    def recover(self) -> dict | None:
        """Recover state from disk after a crash or restart.

        Returns a dict with recovered data:
          - ``generation``: last partial generation (prompt + output + token count)
          - ``event_log``: event log history
          - ``output_history``: past generation outputs
          - ``kv_snapshot``: KV cache state (if available)

        Returns None if no recovery data found.
        """
        return self._recovery.recover()

    def clear_recovery(self) -> None:
        """Remove all crash recovery files from disk."""
        self._recovery.clear()

