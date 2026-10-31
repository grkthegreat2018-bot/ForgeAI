"""Benchmarking, stats, and diagnostics mixin for ForgeEngine."""
import time

import torch

from .diagnostics import build_health_report
from .engine_common import (  # noqa: F401
    _CKPT_CACHE_MAX,
    _DEFAULT_CPU_MEMORY_BYTES,
    _DEFAULT_EOS_TOKEN_IDS,
    _QWEN_TOKENIZER_PATH,
    _QWEN_VOCAB,
    _checkpoint_metadata_cache,
    _checkpoint_size_cache,
    _ckpt_cache_lock,
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    logger,
)
from .errors import ConfigurationError


class _DiagnosticsMixin:
    # ── Benchmarking & stats ──────────────────────────────────────────────

    def benchmark(self, prompt: str, max_new_tokens: int = 50,
                  n_runs: int = 3) -> dict:
        """Benchmark generation speed."""
        self._require_awake()
        if n_runs < 1:
            raise ConfigurationError("n_runs must be positive")
        # Hold _gen_lock for the whole measurement — without it another
        # request can interleave between generate() calls and skew the
        # timings (EC-130). RLock: the inner generate() re-acquires fine.
        with self._gen_lock:
            self.generate(prompt, max_new_tokens=10,
                          finish_sentence=False)  # warmup
            times = []
            token_counts = []
            for _ in range(n_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                tokens_before = self.total_tokens_generated
                start = time.perf_counter()
                self.generate(
                    prompt, max_new_tokens=max_new_tokens,
                    finish_sentence=False)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                times.append(time.perf_counter() - start)
                token_counts.append(
                    self.total_tokens_generated - tokens_before)
        total_time = sum(times)
        total_tokens = sum(token_counts)
        average = total_time / n_runs
        tokens_per_second = total_tokens / total_time if total_time else 0
        self._log(f"{tokens_per_second:.0f} tok/s | "
                  f"{average * 1000:.0f}ms average", level="profile")
        return {
            "tokens_per_sec": tokens_per_second,
            "latency_ms": average * 1000,
            "tokens": max_new_tokens,
            "generated_tokens": total_tokens,
            "runs": n_runs,
        }

    def stats(self) -> dict:
        """Get engine statistics."""
        vram_info = self.vram_usage() if self.device.type == "cuda" else {}
        return {
            "generation_count": self.generation_count,
            "total_tokens_generated": self.total_tokens_generated,
            "keystack_features": self.keystack_features,
            "kv_cache": self.kv_cache.info() if self.kv_cache else None,
            "decoding": self.decoding.name,
            "quantization": self.quantize,
            "acceleration": self.acceleration,
            "mrl_adapter": self.mrl_adapter.info() if self.mrl_adapter else None,
            "quarot_kv": self.quarot_kv.info() if self.quarot_kv else None,
            "v0_warm": self.v0_warm.info() if self.v0_warm else None,
            "progressive_kv": self.progressive_kv.info() if self.progressive_kv else None,
            "vram": vram_info,
            "active_config": (self.active_config.to_dict()
                              if self.active_config else None),
            # Subsystems previously invisible to stats()/diagnose()
            # (EC-133): suffix memory, hotswap settings + last
            # reactivation error, session cache, tool registry,
            # OOM-recovery KV downgrade (EC-114).
            "suffix_mem": (self._suffix_mem.info()
                           if getattr(self, "_suffix_mem", None) else None),
            "hotswap": (self.hotswap.get_settings()
                        if getattr(self, "hotswap", None) else None),
            "sessions": (self.session_stats()
                         if hasattr(self, "session_stats") else None),
            "tools": (len(self.tools._handlers)
                      if getattr(self, "tools", None) is not None else 0),
            "kv_downgrade": getattr(self, "_kv_downgrade", None),
        }

    # ── Built-in diagnostics ────────────────────────────────────────────
    # These methods eliminate the need for one-off profiling/log-reading scripts.

    def bottleneck(self, prompt: str = "The quick brown fox",
                   max_new_tokens: int = 16) -> dict:
        """Profile a generation pass and identify the slowest transformer layers.

        Runs a short generation with per-layer forward hooks to measure
        wall-clock time per block. Returns a dict with per-layer timings,
        top-5 bottlenecks, and overall throughput.

        No external profiling script needed — call this directly:
            report = engine.bottleneck()
            print(report["bottlenecks"])
        """
        self._require_awake()
        self.events.log("Starting bottleneck profiling", source="profile",
                        max_new_tokens=max_new_tokens)
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        result = self._profiler.profile_generate(ids, max_new_tokens=max_new_tokens)
        bottlenecks = result.get("bottlenecks", [])
        if not bottlenecks:
            self.events.warn(
                result.get("error", "No bottlenecks found"), source="profile")
            return result
        slowest = bottlenecks[0]
        self.events.log(
            f"Bottleneck: {result.get('tok_s', 0)} tok/s, "
            f"slowest={slowest['type']}#{slowest['index']} "
            f"({slowest['time_ms']}ms)",
            source="profile", level="profile", bottlenecks=bottlenecks,
        )
        return result

    def read_log(self, n: int = 50, level: str | None = None,
                 source: str | None = None) -> list[dict]:
        """Read recent engine events as structured dicts.

        Replaces log-tailing scripts. Optional filters by level/source:
            engine.read_log(n=20, level="error")  # recent errors
            engine.read_log(n=10, source="profile")  # recent timings
        """
        return self.events.read_log(n=n, level=level, source=source)

    def read_output(self, n: int = 10) -> list[dict]:
        """Read recent generation outputs with metadata.

        Replaces output-capture scripts. Returns last n generations:
            engine.read_output(n=5)  # last 5 generations with tok/s, timing
        """
        return self.outputs.read_output(n=n)

    def diagnose(self) -> dict:
        """Full health report: stats + VRAM + warnings + recent errors.

        Non-invasive (does not run generation). Combines everything a
        debugging script would check into one call:
            report = engine.diagnose()
            if report["status"] != "healthy":
                print(report["warnings"])
        """
        report = build_health_report(self)
        self.events.log(f"Diagnose: {report['status']}",
                        source="engine",
                        warnings=len(report.get("warnings", [])))
        return report

    def profile_bandwidth(self) -> dict:
        """Profile CPU-GPU bandwidth and compute optimal split."""
        return self.bandwidth_profiler.profile_all()

    # ── Evolutionary model merging ("sexual reproduction") ────────────────
    # Implements the GENOME framework (Zhang et al. 2026): crossover,
    # mutation, selection, succession over a population of model checkpoints.
    # All weight recombination runs in CPU RAM; the GPU is only used for
    # fitness evaluation (one candidate at a time → fits 12GB VRAM).
    #
    # See research/merge_models.py for the operator implementations.

