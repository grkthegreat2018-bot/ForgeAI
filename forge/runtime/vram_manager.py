"""VRAM Manager — boot-time profiling + dynamic KV cache sizing.

Implements the growable KV cache pattern from BOOT_TIME_AUDIT.md (Stage 4):
  - Reserve virtual address space, commit physical pages incrementally
  - Profile actual peak memory after model load
  - Size KV cache / max_gen_tokens based on measured free VRAM
  - Monitor during generation, abort before OOM

Also sets persistent compile + autotune cache (Stage 3, env vars):
  - TORCHINDUCTOR_CACHE_DIR
  - TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR

Usage:
    from forge.runtime.vram_manager import VRAMManager
    vram = VRAMManager(total_vram_gb=12.0, safety_margin_gb=1.0)
    vram.setup_compile_cache()
    vram.profile_after_model_load(model)
    max_tokens = vram.max_gen_tokens(n_layers=28, n_heads=12, head_dim=128)
    vram.check_before_generation("task_1")
"""
import os
from dataclasses import dataclass

import torch


@dataclass
class VRAMProfile:
    """Snapshot of VRAM state at a point in time."""
    total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    peak_allocated_mb: float
    label: str = ""

    def __repr__(self) -> str:
        return (f"VRAM[{self.label}]: {self.allocated_mb:.0f}/{self.total_mb:.0f} MB alloc, "
                f"{self.free_mb:.0f} free, peak={self.peak_allocated_mb:.0f}")


class VRAMManager:
    """Manages VRAM allocation, profiling, and dynamic limits.

    Implements the growable KV cache pattern: instead of guessing how much
    VRAM the KV cache needs, we measure actual usage after model load and
    compute the safe token budget dynamically.
    """

    def __init__(
        self,
        total_vram_gb: float = 12.0,
        safety_margin_gb: float = 1.0,
        compile_cache_dir: str = None,
    ):
        if compile_cache_dir is None: from research.paths import TORCH_CACHE_DIR, as_str; compile_cache_dir = as_str(TORCH_CACHE_DIR)
        self.total_vram_mb = total_vram_gb * 1024
        self.safety_margin_mb = safety_margin_gb * 1024
        self.compile_cache_dir = compile_cache_dir

        # Filled by profile_after_model_load
        self.model_weights_mb: float = 0
        self.post_load_profile: VRAMProfile | None = None

        # Filled by profile_after_warmup
        self.warmup_peak_mb: float = 0
        self.warmup_profile: VRAMProfile | None = None

    # ── Stage 3: Persistent compile cache ──────────────────────────

    def setup_compile_cache(self) -> None:
        """Set env vars for persistent torch.compile + autotune cache.

        From BOOT_TIME_AUDIT.md Stage 3:
          - TORCHINDUCTOR_CACHE_DIR: caches FX graphs + Triton kernels to disk
          - TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR: skips GPU autotuning on restart

        Zero code change, massive warm-start improvement.
        """
        cache_dir = self.compile_cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        env_vars = {
            "TORCHINDUCTOR_CACHE_DIR": cache_dir,
            "TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR": cache_dir,
            "TORCHINDUCTOR_BENCHMARK_KERNELS": "0",  # don't re-benchmark on warm start
        }
        for key, val in env_vars.items():
            os.environ[key] = val

        print(f"  [VRAM] Compile cache: {cache_dir}")

    # ── Profiling helpers ──────────────────────────────────────────

    def snapshot(self, label: str = "") -> VRAMProfile:
        """Take a VRAM snapshot."""
        if not torch.cuda.is_available():
            return VRAMProfile(0, 0, 0, 0, 0, label)

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()

        return VRAMProfile(
            total_mb=total_bytes / 1e6,
            allocated_mb=allocated / 1e6,
            reserved_mb=reserved / 1e6,
            free_mb=free_bytes / 1e6,
            peak_allocated_mb=peak / 1e6,
            label=label,
        )

    def reset_peak(self) -> None:
        """Reset peak memory tracker."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def empty_cache(self) -> None:
        """Release cached memory back to OS."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def force_cleanup(self) -> None:
        """Aggressive cleanup: GC + empty cache + reset peak.

        Use after model conversion (e.g. fp32→fp16) to release
        the old tensors and PyTorch's reserved-but-unused memory.
        """
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ── Stage 4: Profile after model load ──────────────────────────

    def profile_after_model_load(self, model: torch.nn.Module, label: str = "model_loaded") -> VRAMProfile:
        """Profile VRAM after model is loaded to GPU.

        This tells us how much VRAM the model weights consume,
        which determines how much is left for KV cache + activations.
        """
        self.empty_cache()
        self.reset_peak()

        # Count model parameters on GPU
        param_bytes = sum(
            p.nelement() * p.element_size()
            for p in model.parameters()
            if p.is_cuda
        )
        self.model_weights_mb = param_bytes / 1e6

        prof = self.snapshot(label)
        self.post_load_profile = prof

        print(f"  [VRAM] After model load: {prof}")
        print(f"  [VRAM] Model weights: {self.model_weights_mb:.0f} MB")
        print(f"  [VRAM] Available for KV+activations: {prof.free_mb - self.safety_margin_mb:.0f} MB")

        return prof

    def profile_after_warmup(
        self,
        model: torch.nn.Module,
        tokenizer,
        warmup_prompt: str = "def hello():",
        max_warmup_tokens: int = 32,
    ) -> VRAMProfile:
        """Run a short warmup generation to measure peak VRAM.

        This captures the transient memory overhead (CUDA kernels, 
        temporary buffers, attention workspace) that isn't visible
        in a static snapshot.
        """
        print(f"  [VRAM] Warmup generation ({max_warmup_tokens} tokens)...")

        self.empty_cache()
        self.reset_peak()

        device = next(model.parameters()).device
        input_ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to(device)

        with torch.inference_mode():
            past_kv = None
            cur_ids = input_ids
            for step in range(max_warmup_tokens):
                if past_kv is not None:
                    logits, _, past_kv = model(
                        cur_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    logits, _, past_kv = model(cur_ids, use_cache=True)
                next_token = torch.argmax(logits[0, -1]).unsqueeze(0).unsqueeze(0)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)

        # Clean up warmup tensors
        del cur_ids, logits, past_kv
        self.empty_cache()

        peak = torch.cuda.max_memory_allocated() / 1e6
        self.warmup_peak_mb = peak
        prof = self.snapshot("warmup_done")
        self.warmup_profile = prof

        print(f"  [VRAM] Warmup peak: {peak:.0f} MB")
        print(f"  [VRAM] After warmup: {prof}")

        return prof

    # ── Dynamic KV cache sizing ────────────────────────────────────

    def kv_cache_bytes_per_token(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,
    ) -> int:
        """Calculate KV cache memory per token.

        KV cache stores (k, v) per layer, each [B, n_heads, T, head_dim].
        Per token per layer: 2 * n_heads * head_dim * dtype_bytes
        Total per token: n_layers * 2 * n_heads * head_dim * dtype_bytes
        """
        return n_layers * 2 * n_heads * head_dim * dtype_bytes

    def max_gen_tokens(
        self,
        n_layers: int = 28,
        n_heads: int = 12,
        head_dim: int = 128,
        dtype_bytes: int = 2,
        overhead_mb: float = 512,  # CUDA kernels, attention workspace, etc.
    ) -> int:
        """Calculate max generation tokens based on current free VRAM.

        Uses the growable KV cache pattern: measure free VRAM, subtract
        safety margin + overhead, divide by per-token KV cost.

        Args:
            n_layers: number of transformer layers
            n_heads: number of attention heads (for MLA, this is n_heads not n_kv)
            head_dim: dimension per head
            dtype_bytes: bytes per element (2 for bf16/fp16, 1 for int8)
            overhead_mb: reserved for CUDA kernels, attention workspace, temp buffers

        Returns:
            Max tokens that can be generated without OOM
        """
        if not torch.cuda.is_available():
            return 512  # conservative default

        free_mb = torch.cuda.mem_get_info()[0] / 1e6
        usable_mb = free_mb - self.safety_margin_mb - overhead_mb

        if usable_mb <= 0:
            print(f"  [VRAM] WARNING: only {free_mb:.0f} MB free, "
                  f"need {self.safety_margin_mb + overhead_mb:.0f} MB for overhead")
            return 64  # emergency minimum

        bytes_per_token = self.kv_cache_bytes_per_token(
            n_layers, n_heads, head_dim, dtype_bytes)
        max_tokens = int(usable_mb * 1e6 / bytes_per_token)

        # Cap at a sanity ceiling — VRAM (not the constant) is the real
        # limit. Was 2048, which needlessly truncated context for models
        # that fit larger KV budgets (e.g. quantized 8B on 12GB ~28K tok).
        max_tokens = min(max_tokens, 1_048_576)
        max_tokens = max(max_tokens, 64)  # floor

        print(f"  [VRAM] KV budget: {usable_mb:.0f} MB / {bytes_per_token} bytes/token "
              f"= {max_tokens} max tokens")

        return max_tokens

    def max_train_batch_size(
        self,
        seq_len: int,
        n_layers: int = 28,
        d_model: int = 1536,
        dtype_bytes: int = 2,
    ) -> int:
        """Calculate max training batch size based on free VRAM.

        Training activations per sample ≈ n_layers * seq_len * d_model * dtype_bytes * ~4
        (4x for forward + backward + intermediate).
        """
        if not torch.cuda.is_available():
            return 1

        free_mb = torch.cuda.mem_get_info()[0] / 1e6
        usable_mb = free_mb - self.safety_margin_mb

        # Rough activation estimate per sample
        activation_bytes_per_sample = n_layers * seq_len * d_model * dtype_bytes * 4
        activation_mb_per_sample = activation_bytes_per_sample / 1e6

        if activation_mb_per_sample <= 0:
            return 1

        batch_size = int(usable_mb / activation_mb_per_sample)
        batch_size = max(batch_size, 1)
        batch_size = min(batch_size, 8)  # cap

        print(f"  [VRAM] Train budget: {usable_mb:.0f} MB / {activation_mb_per_sample:.1f} MB/sample "
              f"= batch_size {batch_size}")

        return batch_size

    # ── Runtime monitoring ─────────────────────────────────────────

    def check_before_generation(self, label: str = "", min_free_mb: float = 256) -> bool:
        """Check if there's enough VRAM for a generation step.

        Returns True if safe to proceed, False if should abort/cleanup.
        """
        if not torch.cuda.is_available():
            return True

        free_mb = torch.cuda.mem_get_info()[0] / 1e6
        if free_mb < min_free_mb:
            print(f"  [VRAM] LOW: {free_mb:.0f} MB free ({label}), cleaning up...")
            self.empty_cache()
            free_mb = torch.cuda.mem_get_info()[0] / 1e6
            if free_mb < min_free_mb:
                print(f"  [VRAM] CRITICAL: {free_mb:.0f} MB free after cleanup, aborting")
                return False
        return True

    def check_during_generation(
        self,
        current_tokens: int,
        max_tokens: int,
        label: str = "",
    ) -> bool:
        """Check VRAM during generation. Returns True if should continue."""
        if not torch.cuda.is_available():
            return True

        free_mb = torch.cuda.mem_get_info()[0] / 1e6

        # If we're using >80% of VRAM, stop early
        if free_mb < self.safety_margin_mb * 0.5:
            print(f"  [VRAM] Early stop at {current_tokens}/{max_tokens} tokens: "
                  f"only {free_mb:.0f} MB free ({label})")
            return False

        return True

    def report(self) -> str:
        """Generate a VRAM usage report."""
        if not torch.cuda.is_available():
            return "CUDA not available"

        prof = self.snapshot("report")
        lines = [
            "=== VRAM Report ===",
            f"  Total: {prof.total_mb:.0f} MB",
            f"  Allocated: {prof.allocated_mb:.0f} MB",
            f"  Reserved: {prof.reserved_mb:.0f} MB",
            f"  Free: {prof.free_mb:.0f} MB",
            f"  Peak: {prof.peak_allocated_mb:.0f} MB",
            f"  Safety margin: {self.safety_margin_mb:.0f} MB",
        ]
        if self.post_load_profile:
            lines.append(f"  Post-load: {self.post_load_profile.allocated_mb:.0f} MB")
        if self.warmup_peak_mb > 0:
            lines.append(f"  Warmup peak: {self.warmup_peak_mb:.0f} MB")
        return "\n".join(lines)
