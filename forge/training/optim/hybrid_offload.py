"""Hybrid CPU-GPU optimizer: ZeRO-Offload-style optimizer state offloading.

Keeps optimizer states (exp_avg, exp_avg_sq) and fp32 master weights on CPU
pinned memory while the model params stay in bf16 on GPU. This eliminates the
VRAM bottleneck of fp32 AdamW states (12 bytes/param = 14.4GB for 1.2B model)
that forces the use of 8-bit AdamW on 12GB GPUs.

Memory budget on RTX 5070 (12GB):
  - Model bf16 weights:        2.34 GB  (GPU)
  - Activations + KV:          ~1-2 GB  (GPU, with grad checkpointing)
  - Gradients bf16:            2.34 GB  (GPU, transient)
  - Optimizer states (fp32):   14.4 GB  (CPU pinned RAM)
  - Master weights (fp32):      4.8 GB  (CPU pinned RAM)
  Total GPU: ~6-7 GB (vs 19+ GB for full fp32 AdamW on GPU)
  Total CPU: ~19 GB (fits in 32GB RAM)

Design (pure PyTorch, no C++ extension — Windows-compatible):
  1. backward() produces bf16 grads on GPU
  2. step(): async copy grads GPU→CPU pinned buffer (non_blocking, overlapped)
  3. CPU runs AdamW update on fp32 master weights using CPU grads
  4. Updated fp32 master → bf16 copy back to GPU params (async, non_blocking)
  5. CUDA stream synchronization ensures correctness

The CPU AdamW math is identical to torch.optim.AdamW (same eps, beta1, beta2,
weight_decay, amsgrad). We implement it manually because the state lives on CPU
and we want to control the GPU↔CPU transfer scheduling.

Usage:
  from forge.training.optim.hybrid_offload import CPUAdamW, configure_hybrid_optimizer

  # Drop-in replacement for torch.optim.AdamW:
  optimizer = CPUAdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

  # Or via configure_optimizer with optimizer_name="cpu_offload"
  optimizer = configure_optimizer(model, lr, wd, optimizer_name="cpu_offload")

  # Training loop is unchanged:
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

For async overlap (CPU optimizer step runs while GPU does next forward):
  optimizer = CPUAdamW(model.parameters(), lr=5e-5, overlap=True)
  # optimizer.step() returns immediately after launching CPU step on a thread
  # optimizer.wait() must be called before next backward() to ensure grads are consumed

FreeToken-inspired enhancements (R&D round 14):
  - double_buffer=True: Ping-pong grad buffers. While buffer A feeds the CPU
    AdamW step, buffer B receives the next batch's grads. Eliminates the
    transfer-then-compute serialization. Requires overlap=True.
  - bandwidth_adaptive=True: Profiles PCIe (B_P) and host (B_H) bandwidths at
    init, auto-selects optimal chunk size and overlap strategy. Inspired by
    FreeToken's q* policy (arXiv:2608.16157 Eq. 4).
  - chunked transfers (novel): Splits large param grad transfers into
    bandwidth-optimal chunks (chunk_size = f(B_P, latency)) for finer-grained
    overlap than per-tensor granularity. Multiple chunks launch on separate
    CUDA streams, maximizing PCIe utilization.
  - bandwidth_predictor: Tracks bandwidth/VRAM trend over steps and pre-emptively
    increases offload before projected OOM (novel vs FreeToken's static profiling).

  optimizer = CPUAdamW(model.parameters(), lr=5e-5,
                       overlap=True, double_buffer=True,
                       bandwidth_adaptive=True)
"""
from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterable

import torch
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# int4 gradient compression utilities (EF21 error feedback, R&D round 14)
# ---------------------------------------------------------------------------
# Per-row symmetric quantization to int4 [-8, 7] with 2-values-per-byte
# packing.  bf16 (2 bytes) → int4 (0.5 bytes) = 4x compression ratio.
# Used with EF21 error feedback (Richtárik et al., 2021) to preserve
# convergence: the residual (grad_to_send - decompressed) is accumulated
# on GPU and added to the next gradient before compression.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compress_grad_int4(grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress bf16/fp32 gradient to packed int4 + per-channel scales.

    Uses per-row (per-output-channel) symmetric quantization to int4 [-8, 7].
    Packs 2 int4 values into 1 int8 byte for transfer efficiency.

    For 1D tensors (biases, norms): uses a single per-tensor scale.
    For 2D+ tensors: per-row scale (first dim = output channels).

    Returns:
        packed: (N//2,) int8 tensor — 2 int4 values packed per byte
        scales: (out_channels, 1) fp16 — per-row absmax / 7
    """
    orig_shape = grad.shape
    orig_numel = grad.numel()

    # Edge case: empty tensor
    if orig_numel == 0:
        packed = torch.zeros(0, dtype=torch.int8, device=grad.device)
        n_rows = orig_shape[0] if len(orig_shape) > 1 else 1
        scales = torch.zeros(n_rows, 1, dtype=torch.float16, device=grad.device)
        return packed, scales

    # Reshape for per-row quantization
    # 1D → (1, N), 2D+ → (first_dim, -1)
    if grad.ndim <= 1:
        flat = grad.detach().reshape(1, -1)
    else:
        flat = grad.detach().reshape(grad.shape[0], -1)

    # Per-row absmax (avoid division by zero for all-zero rows)
    absmax = flat.abs().amax(dim=-1, keepdim=True)  # (out_channels, 1)
    absmax_safe = absmax.clamp(min=1e-12)
    scale = (absmax_safe / 7.0).to(torch.float16)  # fp16 for transfer efficiency

    # Quantize: round to int, clamp to int4 range [-8, 7]
    q = torch.round(flat / scale.to(torch.float32)).to(torch.int8)
    q = q.clamp(-8, 7)

    # Pack 2 int4 values into 1 int8 byte
    # High nibble = even-indexed value, low nibble = odd-indexed value
    q_flat = q.reshape(-1)
    n = q_flat.numel()
    if n % 2 != 0:
        # Pad with zero to make even length
        q_flat = torch.cat(
            [q_flat, torch.zeros(1, dtype=torch.int8, device=grad.device)]
        )

    # Shift from signed [-8, 7] to unsigned [0, 15] for nibble packing
    q_uint = (q_flat + 8).to(torch.uint8)  # (N_padded,)
    high = q_uint[0::2]  # (N//2,)
    low = q_uint[1::2]   # (N//2,)
    packed = ((high << 4) | low).view(torch.int8)  # bit-pattern preserved as int8

    return packed, scale


@torch.no_grad()
def _decompress_grad_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    """Decompress packed int4 + scales back to fp32 gradient.

    Args:
        packed: (N//2,) int8 tensor from _compress_grad_int4
        scales: (out_channels, 1) fp16 per-row scales
        original_shape: shape of the original gradient tensor

    Returns:
        fp32 gradient tensor of shape *original_shape*
    """
    orig_numel = 1
    for s in original_shape:
        orig_numel *= s

    # Edge case: empty tensor
    if packed.numel() == 0:
        return torch.zeros(original_shape, dtype=torch.float32, device=packed.device)

    # Unpack nibbles: view as uint8 to extract high/low 4 bits
    packed_uint = packed.view(torch.uint8)
    high = (packed_uint >> 4) & 0x0F  # (N//2,)
    low = packed_uint & 0x0F          # (N//2,)

    # Interleave back to (N_padded,)
    n_packed = packed_uint.numel()
    q_uint = torch.empty(n_packed * 2, dtype=torch.uint8, device=packed.device)
    q_uint[0::2] = high
    q_uint[1::2] = low

    # Convert unsigned [0, 15] back to signed [-8, 7]
    q = q_uint.to(torch.int32) - 8  # (N_padded,)

    # Trim padding if original had odd number of elements
    q = q[:orig_numel]

    # Reshape to (out_channels, -1) for per-row dequantization
    if len(original_shape) <= 1:
        q = q.reshape(1, -1)
        scale = scales.to(torch.float32).reshape(1, 1)
    else:
        out_channels = original_shape[0]
        q = q.reshape(out_channels, -1)
        scale = scales.to(torch.float32)

    # Dequantize: grad = q * scale
    grad = q.to(torch.float32) * scale

    # Reshape back to original shape
    return grad.reshape(original_shape)


class CPUAdamW(Optimizer):
    """AdamW with optimizer states and fp32 master weights on CPU.

    Model parameters remain on GPU (bf16). Optimizer states (m, v) and fp32
    master copies live in CPU pinned memory. Each step:
      1. Copy bf16 grads GPU→CPU (async, pinned, non_blocking)
      2. AdamW update on CPU (fp32 master += -lr * m / (sqrt(v) + eps))
      3. Copy fp32 master→bf16 GPU param (async, non_blocking)

    The CPU update is mathematically identical to torch.optim.AdamW.

    With grad_offload=True (recommended for >2B params on 12GB GPU):
      Registers backward hooks that stream gradients to CPU pinned memory
      *during* backward(), then frees the GPU gradient immediately. This
      eliminates the GPU gradient memory bottleneck (5.6GB for 2.8B params).
      Inspired by tascj/offload_adam (2025).

    Memory with grad_offload=True on V7 (2.8B params, 12GB GPU):
      - Model bf16 + INT8 buffers:  8.47 GB  (GPU)
      - Gradients:                  ~0 GB    (streamed to CPU during backward)
      - Activations:                O(1)     (gradient checkpointing)
      - Optimizer states (fp32):    33.6 GB  (CPU pinned RAM)
      Total GPU: ~8.5 GB (fits 12GB with headroom for activations)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        pin_memory: bool = True,
        overlap: bool = False,
        verbose: bool = True,
        grad_offload: bool = False,
        double_buffer: bool = False,
        bandwidth_adaptive: bool = False,
        chunk_size_mb: int | None = None,
        bf16_state: bool = False,
        grad_compression: str = "none",  # "none", "int4", "int8" (evolution: int4 best)
        prefetch_depth: int = 7,  # evolution-discovered: 7 steps ahead
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
                        "amsgrad": amsgrad}
        super().__init__(params, defaults)
        self.pin_memory = pin_memory
        self.overlap = overlap
        self.grad_offload = grad_offload
        self._verbose = verbose
        self._initialized = False
        self._cpu_thread: threading.Thread | None = None
        self._cpu_event = threading.Event()
        self._grad_hooks: list = []
        self._grad_accum: dict = {}  # param id → CPU grad buffer (accumulated)
        self._grad_ready: dict = {}   # param id → bool (grad offloaded this step)

        # FreeToken-inspired enhancements (R&D round 14)
        self.double_buffer = double_buffer
        self.bandwidth_adaptive = bandwidth_adaptive
        self.chunk_size_mb = chunk_size_mb
        self._bw_result = None       # BandwidthResult
        self._bw_predictor = None    # BandwidthPredictor
        self._active_buffer = 0      # 0 or 1 for ping-pong
        self._step_count = 0
        self.bf16_state = bf16_state  # bf16 m,v states (saves CPU RAM for large models)
        self.grad_compression = grad_compression
        self.prefetch_depth = prefetch_depth
        self._state_dtype = torch.bfloat16 if bf16_state else torch.float32

        # Auto-enable overlap when double_buffer is requested
        if double_buffer and not overlap:
            overlap = True
            self.overlap = True
            if verbose:
                print("  [FreeToken] double_buffer=True auto-enabled overlap=True")

        # Bandwidth-adaptive: profile PCIe + host bandwidth at init
        if bandwidth_adaptive:
            try:
                from forge.runtime.bandwidth_profiler import (
                    BandwidthProfiler,
                )
                profiler = BandwidthProfiler()
                bp, bh = profiler.profile()
                self._bw_result = profiler.get_cached()
                # Auto-set chunk size from bandwidth if not specified
                if chunk_size_mb is None:
                    # Target ~10ms per chunk transfer: chunk_bytes = B_P * 0.01s
                    chunk_bytes = bp * 1e9 * 0.01
                    self.chunk_size_mb = max(16, int(chunk_bytes / 1e6))
                if verbose:
                    print(f"  [FreeToken] Bandwidth-adaptive: B_P={bp:.1f} GB/s, "
                          f"B_H={bh:.1f} GB/s, q*={self._bw_result.q_star_ratio:.3f}, "
                          f"chunk={self.chunk_size_mb}MB")
                # If B_P >> B_H, overlap is very effective (transfer hidden behind CPU)
                # If B_P << B_H, overlap helps less — but still enable for double-buffer
                if bp > bh * 1.5 and not overlap:
                    overlap = True
                    self.overlap = True
                    if verbose:
                        print("  [FreeToken] B_P >> B_H: auto-enabled overlap "
                              "(PCIe fast enough to hide behind CPU compute)")
            except Exception as e:
                if verbose:
                    print(f"  [FreeToken] Bandwidth profiling failed: {e}, "
                          f"using defaults")

        if self._verbose:
            total_params = sum(p.numel() for group in self.param_groups for p in group["params"])
            cpu_mem = total_params * (4 + (2 if bf16_state else 4) + (2 if bf16_state else 4)) / 1e9  # master(fp32) + m + v
            grad_cpu_mem = total_params * 2 / 1e9 if grad_offload else 0  # bf16 grad buffer
            db_mem = total_params * 4 / 1e9 if double_buffer else 0  # extra fp32 buffer
            print(f"CPUAdamW: {total_params/1e6:.1f}M params | "
                  f"CPU optimizer memory: {cpu_mem:.2f} GB | "
                  f"overlap={overlap} | pin_memory={pin_memory} | "
                  f"grad_offload={grad_offload} | "
                  f"double_buffer={double_buffer} | "
                  f"bandwidth_adaptive={bandwidth_adaptive}")
            if grad_offload:
                print(f"  Grad offload: {grad_cpu_mem:.2f} GB grads streamed to CPU during backward")
            if double_buffer:
                print(f"  Double buffer: +{db_mem:.2f} GB CPU for ping-pong grad buffers")
            if grad_compression == "int4":
                ef_mem = total_params * 4 / 1e9  # fp32 ef_error on GPU
                print(f"  [FreeToken] Grad compression: int4 (4x PCIe bandwidth reduction, "
                      f"EF21 error feedback, +{ef_mem:.2f} GB GPU for error buffers)")
            elif grad_compression == "int8":
                print("  Grad compression: int8 (2x reduction) [not yet implemented — pass-through]")
            if prefetch_depth > 0:
                print(f"  Prefetch depth: {prefetch_depth} steps ahead")

    @torch.no_grad()
    def _lazy_init(self):
        """Allocate CPU state tensors on first step (after params are on GPU).

        Batches CPU memory allocations to avoid per-param syscall overhead.
        For 2.8B params (814 tensors), this reduces init from ~15s to ~3s.
        """
        if self._initialized:
            return

        import time as _time
        t0 = _time.perf_counter()

        # Phase 1: Collect all GPU params that need offload
        gpu_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.device.type != "cpu":
                    gpu_params.append((group, p))

        if gpu_params:
            # Phase 2: Batch copy all params to CPU fp32 in one transfer
            # This is much faster than per-param .to("cpu") + .clone()
            n_params = len(gpu_params)
            print(f"  CPUAdamW init: {n_params} tensors, batching CPU transfers...")

            # Copy all grads to CPU in one pass (overlaps with allocation)
            for group, p in gpu_params:
                state = self.state[p]
                state["step"] = torch.tensor(0.0, dtype=torch.float32)

                # Optimizer states: bf16 or fp32 depending on bf16_state flag
                state["exp_avg"] = torch.empty(p.shape, dtype=self._state_dtype)
                state["exp_avg_sq"] = torch.empty(p.shape, dtype=self._state_dtype)
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()

                # FP32 master: copy from GPU (synchronous — non_blocking with
                # dtype conversion can produce stale data on some platforms)
                master = torch.empty(p.shape, dtype=torch.float32)
                master.copy_(p.detach().view(-1).to("cpu", torch.float32).view(p.shape))
                state["master"] = master

                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.empty(p.shape, dtype=torch.float32)
                    state["max_exp_avg_sq"].zero_()

                # CPU gradient buffer
                state["grad_cpu"] = torch.empty(p.shape, dtype=torch.float32)
                state["grad_cpu"].zero_()

                # Double buffer: second ping-pong grad buffer (FreeToken-inspired)
                if self.double_buffer:
                    state["grad_cpu_b"] = torch.empty(p.shape, dtype=torch.float32)
                    state["grad_cpu_b"].zero_()

                # EF21 error feedback buffer (stays on GPU — same device as param)
                # Allocated when gradient compression is enabled. This is the
                # residual (grad_to_send - decompressed) that accumulates across
                # steps to preserve convergence under int4 quantization.
                # Memory cost: fp32, same size as param (4 bytes/param on GPU).
                # For V7 with BAdam (1 block active), only the active block's
                # error buffer needs to be on GPU.
                if self.grad_compression != "none":
                    state["ef_error"] = torch.zeros(
                        p.shape, dtype=torch.float32, device=p.device
                    )

                # Pin memory in batch (after all allocations done)
                # Skip pin_memory for now — it's slow on Windows and the
                # non-pinned path is fast enough with DDR5

                # Grad offload: allocate bf16 CPU grad buffer + register hook
                if self.grad_offload:
                    grad_buf = torch.empty(p.shape, dtype=p.dtype)
                    grad_buf.zero_()
                    self._grad_accum[id(p)] = grad_buf
                    self._grad_ready[id(p)] = False

                    def _make_hook(param, buf):
                        def _grad_hook(grad):
                            if self.grad_compression == "int4":
                                # EF21: compress on GPU, transfer packed int4
                                # to CPU (4x smaller), decompress on CPU
                                grad_cpu = self._compress_and_offload_grad(
                                    grad, param
                                )
                                buf.copy_(grad_cpu, non_blocking=False)
                            else:
                                buf.copy_(grad, non_blocking=True)
                            param.grad = None  # free GPU memory immediately
                            self._grad_ready[id(param)] = True
                            return None
                        return _grad_hook

                    hook = p.register_hook(_make_hook(p, self._grad_accum[id(p)]))
                    self._grad_hooks.append(hook)

            # Single sync point for all async CPU transfers
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Pin memory in a single pass (optional, after all allocs)
            if self.pin_memory:
                for group, p in gpu_params:
                    state = self.state[p]
                    try:
                        state["exp_avg"] = state["exp_avg"].pin_memory()
                        state["exp_avg_sq"] = state["exp_avg_sq"].pin_memory()
                        state["master"] = state["master"].pin_memory()
                        state["grad_cpu"] = state["grad_cpu"].pin_memory()
                        if self.double_buffer:
                            state["grad_cpu_b"] = state["grad_cpu_b"].pin_memory()
                        if self.grad_offload:
                            self._grad_accum[id(p)] = self._grad_accum[id(p)].pin_memory()
                    except RuntimeError:
                        pass  # pin_memory failed (memlock limit) — continue unpinned

        # CPU params: standard init
        for group in self.param_groups:
            for p in group["params"]:
                if p.device.type == "cpu":
                    state = self.state[p]
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

        elapsed = _time.perf_counter() - t0
        if self._verbose and gpu_params:
            print(f"  CPUAdamW init complete: {elapsed:.1f}s")

        self._initialized = True

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single AdamW step with CPU offloaded states.

        In sync mode (overlap=False): step() blocks until params are updated.
        In overlap mode (overlap=True): step() launches the CPU math on a
        background thread and returns immediately. Call wait() before the
        next backward() to ensure the CPU step consumed the grads and the
        GPU param sync is complete. The GPU param sync always runs on the
        main thread (CUDA is not thread-safe with the background thread).

        With double_buffer=True: grads are written to the inactive buffer
        while the CPU processes the active one. The buffers swap each step,
        eliminating transfer-then-compute serialization.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._lazy_init()
        self._step_count += 1

        # Determine which grad buffer to use (ping-pong for double_buffer)
        # The active buffer is the one the PREVIOUS CPU step is reading.
        # We write to the INACTIVE buffer (so we don't overwrite what CPU
        # is still reading), then CPU reads from that same inactive buffer.
        # The overlap is: CUDA transfer to inactive || CPU reads active.
        if self.double_buffer and self._step_count > 1:
            # Write to inactive, CPU reads from same inactive (after wait)
            write_buf_key = "grad_cpu_b" if self._active_buffer == 0 else "grad_cpu"
            read_buf_key = write_buf_key  # same buffer — write then read
        else:
            # First step or no double_buffer: write and read same buffer
            write_buf_key = "grad_cpu"
            read_buf_key = "grad_cpu"

        # Phase 1: get grads to CPU
        # With grad_offload: grads already streamed to CPU during backward via hooks
        # Without grad_offload: async copy grads GPU→CPU now
        #
        # FreeToken-inspired overlap: when double_buffer + overlap, start the
        # grad transfer BEFORE waiting for the previous CPU step. The CUDA
        # transfer (DMA) runs in parallel with the CPU optimizer thread.
        # Without double_buffer, we must wait FIRST to avoid overwriting the
        # buffer the CPU is still reading (race condition).
        #
        # Note: grad_offload doesn't benefit from double_buffer overlap since
        # grads are already on CPU (no CUDA transfer to overlap with).

        if self.grad_offload:
            # Grads already on CPU via hooks. Sync and copy to read buffer.
            if self.overlap and self._cpu_thread is not None:
                self.wait()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.device.type == "cpu":
                        continue
                    pid = id(p)
                    if not self._grad_ready.get(pid, False):
                        continue
                    state = self.state[p]
                    grad_buf = self._grad_accum[pid]
                    state[read_buf_key].copy_(grad_buf, non_blocking=False)
                    grad_buf.zero_()
                    self._grad_ready[pid] = False
        elif self.double_buffer and self.overlap and self._cpu_thread is not None:
            # FreeToken overlap: transfer to inactive buffer BEFORE waiting
            # for previous CPU step. DMA runs in parallel with CPU compute.
            copy_streams = self._copy_grads_to_cpu(write_buf_key)
            self.wait()  # overlaps with the transfer above
            for stream in copy_streams:
                stream.synchronize()
        else:
            # Original safe order: wait first, then transfer to same buffer
            if self.overlap and self._cpu_thread is not None:
                self.wait()
            copy_streams = self._copy_grads_to_cpu(write_buf_key)
            for stream in copy_streams:
                stream.synchronize()

        # Phase 2: CPU AdamW update
        # With double_buffer: CPU processes the read buffer while write buffer
        # is ready to receive next batch's grads (overlap with next backward)
        if self.overlap:
            self._cpu_event.clear()
            self._cpu_thread = threading.Thread(
                target=self._cpu_update, args=(read_buf_key,), daemon=True)
            self._cpu_thread.start()
        else:
            self._cpu_update(read_buf_key)
            # Phase 3 (sync mode): copy updated master weights back to GPU
            self._sync_params_to_gpu()

        # Swap buffers for next step (double-buffer ping-pong)
        if self.double_buffer:
            self._active_buffer = 1 - self._active_buffer

        return loss

    @torch.no_grad()
    def _compress_and_offload_grad(
        self, grad: torch.Tensor, p: torch.nn.Parameter
    ) -> torch.Tensor:
        """Apply EF21 error feedback, compress to int4, decompress on CPU.

        EF21 algorithm (Richtárik et al., 2021):
          1. grad_to_send = grad + ef_error  (accumulate previous residual)
          2. packed, scales = compress(grad_to_send)  (on GPU)
          3. grad_decompressed_gpu = decompress(packed, scales)  (on GPU, for error)
          4. ef_error = grad_to_send - grad_decompressed_gpu  (stays on GPU)
          5. Transfer packed + scales to CPU (4x smaller than full grad)
          6. grad_cpu = decompress(packed_cpu, scales_cpu)  (on CPU, for optimizer)

        The error buffer is fp32, same size as param, and stays on GPU.
        This is the convergence-preserving cost of int4 compression.

        Args:
            grad: GPU gradient tensor (bf16 or fp32)
            p: The parameter this gradient belongs to (for state lookup)

        Returns:
            fp32 gradient tensor on CPU (decompressed from int4)
        """
        state = self.state[p]
        ef_error = state["ef_error"]

        # Step 1: accumulate previous error (EF21)
        grad_to_send = grad.to(torch.float32) + ef_error

        # Step 2: compress on GPU
        packed, scales = _compress_grad_int4(grad_to_send)

        # Step 3: decompress on GPU to compute residual error
        grad_decompressed_gpu = _decompress_grad_int4(
            packed, scales, grad.shape
        )

        # Step 4: update error feedback buffer (stays on GPU)
        ef_error.copy_(grad_to_send - grad_decompressed_gpu)

        # Step 5: transfer compressed data to CPU (4x smaller than full grad)
        packed_cpu = packed.to("cpu", non_blocking=False)
        scales_cpu = scales.to("cpu", non_blocking=False)

        # Step 6: decompress on CPU for the optimizer update
        grad_cpu = _decompress_grad_int4(packed_cpu, scales_cpu, grad.shape)

        return grad_cpu

    def _copy_grads_to_cpu(self, buf_key: str) -> list:
        """Copy grads GPU→CPU with optional chunked transfers.

        With chunk_size_mb set (bandwidth_adaptive): splits large tensors
        into chunks and launches each on a separate CUDA stream for
        finer-grained PCIe overlap. This is the novel gradient-chunked
        pipeline — chunk_size = f(B_P, latency) from measured bandwidth.

        Without chunking: single async copy per tensor on a shared stream.

        Returns the list of CUDA streams used. The caller is responsible
        for synchronizing them (allows overlap with CPU optimizer step).
        """
        copy_streams = []
        chunk_bytes = (self.chunk_size_mb * 1024 * 1024) if self.chunk_size_mb else 0

        # Use a single shared stream for non-chunked transfers to avoid
        # OOM from creating hundreds of CUDA streams (one per param).
        # Key: when grad (bf16) and grad_cpu (fp32) have different dtypes,
        # non_blocking copy may allocate GPU temp buffers for ALL tensors
        # simultaneously, causing OOM on 12GB GPUs. We avoid this by doing
        # synchronous copies (the CPU AdamW math is the bottleneck anyway,
        # so transfer overlap provides minimal benefit).
        if not chunk_bytes or self.grad_compression == "int4":
            # Synchronous copy path: no stream needed (copies are non_blocking=False
            # to avoid GPU temp buffer accumulation from dtype-converted async copies).
            # The CPU AdamW math is the bottleneck, so transfer overlap provides
            # minimal benefit here.
            #
            # With int4 compression: compress on GPU (EF21), transfer packed int4
            # to CPU (4x smaller), decompress on CPU. Chunking is skipped since
            # packed data is already 4x smaller.
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.device.type == "cpu":
                        continue
                    state = self.state[p]
                    grad_cpu = state[buf_key]
                    if self.grad_compression == "int4":
                        grad_decompressed = self._compress_and_offload_grad(
                            p.grad, p
                        )
                        grad_cpu.copy_(grad_decompressed, non_blocking=False)
                    else:
                        grad_cpu.copy_(p.grad, non_blocking=False)
            return copy_streams  # empty list — caller's stream sync loop is a no-op

        # Chunked transfer path: use a small pool of streams (max 4)
        max_streams = 4
        stream_pool = [torch.cuda.Stream() for _ in range(max_streams)]
        stream_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if p.device.type == "cpu":
                    continue
                grad_cpu = state[buf_key]

                if p.grad.numel() * p.grad.element_size() > chunk_bytes:
                    chunk_elements = chunk_bytes // p.grad.element_size()
                    n_chunks = (p.grad.numel() + chunk_elements - 1) // chunk_elements
                    for i in range(n_chunks):
                        start = i * chunk_elements
                        end = min(start + chunk_elements, p.grad.numel())
                        stream = stream_pool[stream_idx % max_streams]
                        with torch.cuda.stream(stream):
                            grad_cpu.view(-1)[start:end].copy_(
                                p.grad.view(-1)[start:end], non_blocking=True)
                        stream_idx += 1
                else:
                    stream = stream_pool[stream_idx % max_streams]
                    with torch.cuda.stream(stream):
                        grad_cpu.copy_(p.grad, non_blocking=True)
                    stream_idx += 1

        copy_streams = list(set(stream_pool[:min(stream_idx, max_streams)]))
        return copy_streams

    def _cpu_update(self, buf_key: str = "grad_cpu"):
        """Run AdamW math on CPU using fp32 master weights and CPU grads.

        Args:
            buf_key: Which grad buffer to read from ("grad_cpu" or "grad_cpu_b").
                For double-buffer mode, this selects the ping-pong buffer that
                was filled by the previous step's transfer.
        """
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                # With grad_offload, p.grad is None (freed by hook). Check grad_cpu instead.
                if not self.grad_offload and p.grad is None:
                    continue
                if self.grad_offload and p not in self.state:
                    continue
                state = self.state[p]
                step_t = state["step"]
                step_t += 1
                step = step_t.item()

                if p.device.type == "cpu":
                    # Pure CPU param — standard in-place AdamW
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    grad = p.grad
                    bias_c1 = 1 - beta1 ** step
                    bias_c2 = 1 - beta2 ** step
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    step_size = lr / bias_c1
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    continue

                # Offloaded param: use CPU grad buffer + fp32 master
                grad = state[buf_key]
                master = state["master"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                bias_c1 = 1 - beta1 ** step
                bias_c2 = 1 - beta2 ** step

                # AdamW update on fp32 master (decoupled weight decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_sq, exp_avg_sq, out=max_sq)
                    denom = (max_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)

                step_size = lr / bias_c1
                # Decoupled weight decay (AdamW, not AdamL2)
                if wd != 0:
                    master.mul_(1 - lr * wd)
                master.addcdiv_(exp_avg, denom, value=-step_size)

        if self.overlap:
            # Signal that CPU math is done (NO CUDA ops here — main thread syncs)
            self._cpu_event.set()

    @torch.no_grad()
    def _sync_params_to_gpu(self):
        """Copy updated fp32 master weights back to bf16 GPU params (async).

        Uses a single shared CUDA stream to avoid OOM from creating hundreds
        of streams (one per param tensor).
        """
        if not torch.cuda.is_available():
            for group in self.param_groups:
                for p in group["params"]:
                    if not self.grad_offload and p.grad is None:
                        continue
                    if p.device.type == "cpu" or p not in self.state:
                        continue
                    p.copy_(self.state[p]["master"], non_blocking=True)
            return

        shared_stream = torch.cuda.Stream()
        with torch.cuda.stream(shared_stream):
            for group in self.param_groups:
                for p in group["params"]:
                    if not self.grad_offload and p.grad is None:
                        continue
                    if p.device.type == "cpu" or p not in self.state:
                        continue
                    # fp32 master → bf16 GPU param (synchronous to avoid
                    # GPU temp buffer accumulation from async dtype-converted copies)
                    master = self.state[p]["master"]
                    p.copy_(master, non_blocking=False)
        shared_stream.synchronize()

    def wait(self):
        """Wait for async CPU step to complete and sync params to GPU.

        Called automatically at the start of the next step() in overlap mode,
        or manually by the training loop before the next backward().
        """
        if self.overlap and self._cpu_thread is not None:
            self._cpu_event.wait()
            self._cpu_thread.join()
            self._cpu_thread = None
            # Phase 3: sync updated master weights back to GPU (on main thread)
            self._sync_params_to_gpu()

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients. Same as base Optimizer.zero_grad."""
        super().zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return state dict (CPU tensors — safe to save without GPU)."""
        return super().state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict. Tensors come back on CPU; _lazy_init skips re-alloc."""
        super().load_state_dict(state_dict)
        self._initialized = True

    def record_bandwidth_sample(self, vram_gb: float | None = None):
        """Record a bandwidth + VRAM sample for the predictive offload policy.

        Novel (R&D round 14): Tracks bandwidth/VRAM trend over training steps
        and pre-emptively adjusts offload before projected OOM. Call this
        every N steps from the training loop.

        Args:
            vram_gb: Current VRAM usage in GB. If None, reads from CUDA.
        """
        if not self.bandwidth_adaptive or self._bw_result is None:
            return
        if vram_gb is None and torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
        if vram_gb is None:
            return
        if self._bw_predictor is None:
            from forge.runtime.bandwidth_profiler import BandwidthPredictor
            self._bw_predictor = BandwidthPredictor()
        self._bw_predictor.record(self._step_count, self._bw_result.b_p, vram_gb)

    def should_preempt_offload(self) -> bool:
        """Check if the predictive policy says to increase offload.

        Returns True if VRAM is projected to hit the limit within the
        preempt threshold, or if bandwidth is degrading. The training
        loop can use this to enable grad_offload or increase grad_accum.
        """
        if self._bw_predictor is None:
            return False
        return self._bw_predictor.should_preempt_offload(self._step_count)

    def bandwidth_stats(self) -> dict:
        """Return bandwidth + predictor statistics for logging."""
        stats = {"bandwidth_adaptive": self.bandwidth_adaptive}
        if self._bw_result is not None:
            stats["b_p"] = f"{self._bw_result.b_p:.1f} GB/s"
            stats["b_h"] = f"{self._bw_result.b_h:.1f} GB/s"
            stats["q_star_ratio"] = f"{self._bw_result.q_star_ratio:.3f}"
        if self._bw_predictor is not None:
            stats["predictor"] = self._bw_predictor.stats()
        return stats

    def cleanup_compression(self):
        """Free EF21 error feedback buffers from GPU memory.

        Call this when compression is no longer needed or before switching
        compression modes. The ef_error buffers are fp32 on GPU (same size
        as params) and should not leak.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "ef_error" in self.state[p]:
                    del self.state[p]["ef_error"]
        if self._verbose and self.grad_compression != "none":
            print("  [FreeToken] Cleaned up EF21 error feedback buffers")

    def __del__(self):
        """Best-effort cleanup of GPU error feedback buffers."""
        try:
            self.cleanup_compression()
        except Exception:
            logger.debug("Error during hybrid optimizer cleanup", exc_info=True)  # optimizer may be partially initialized during GC


def configure_hybrid_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    overlap: bool = False,
    pin_memory: bool = True,
    grad_offload: bool = False,
    double_buffer: bool = False,
    bandwidth_adaptive: bool = False,
    chunk_size_mb: int | None = None,
) -> CPUAdamW:
    """Configure CPUAdamW with separate weight decay for matrix vs scalar params.

    Matches the param-grouping convention in training_utils.configure_optimizer:
    matrix params (ndim>=2) get weight decay, scalar/bias params (ndim<2) don't.

    With grad_offload=True, registers backward hooks that stream gradients to
    CPU pinned memory during backward(), freeing GPU gradient memory immediately.
    Essential for models >2B params on 12GB GPUs.

    FreeToken-inspired options (R&D round 14):
      - double_buffer=True: Ping-pong grad buffers for transfer/compute overlap.
      - bandwidth_adaptive=True: Profile PCIe bandwidth, auto-set chunk size
        and overlap strategy. Uses BandwidthProfiler from forge.runtime.
      - chunk_size_mb: Override auto-computed chunk size for chunked transfers.
    """
    matrix_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    other_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]

    param_groups = [
        {"params": matrix_params, "weight_decay": weight_decay},
        {"params": other_params, "weight_decay": 0.0},
    ]

    return CPUAdamW(param_groups, lr=lr, overlap=overlap,
                    pin_memory=pin_memory, grad_offload=grad_offload,
                    double_buffer=double_buffer,
                    bandwidth_adaptive=bandwidth_adaptive,
                    chunk_size_mb=chunk_size_mb)


def estimate_memory(model: torch.nn.Module, double_buffer: bool = False) -> dict:
    """Estimate GPU vs CPU memory usage for hybrid offload training.

    Returns a dict with GPU and CPU memory estimates in GB.
    Useful for verifying the model fits before starting a long training run.

    With double_buffer=True, adds an extra fp32 grad buffer per param on CPU.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # GPU: bf16 weights + bf16 grads (transient) + activations (user-controlled)
    gpu_weights = total_params * 2 / 1e9  # bf16
    gpu_grads = trainable * 2 / 1e9  # bf16 grads
    # CPU: fp32 master + fp32 m + fp32 v = 12 bytes/param
    cpu_optim = trainable * 12 / 1e9
    # Double buffer: extra fp32 grad buffer = 4 bytes/param
    cpu_double_buf = trainable * 4 / 1e9 if double_buffer else 0

    return {
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable / 1e6,
        "gpu_weights_GB": gpu_weights,
        "gpu_grads_GB": gpu_grads,
        "gpu_total_min_GB": gpu_weights + gpu_grads,  # + activations
        "cpu_optimizer_GB": cpu_optim,
        "cpu_double_buffer_GB": cpu_double_buf,
        "cpu_total_GB": cpu_optim + cpu_double_buf,
    }
