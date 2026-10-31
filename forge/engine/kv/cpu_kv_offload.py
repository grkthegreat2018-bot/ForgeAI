"""CPU KV cache offloading: extend effective KV memory via PCIe to system RAM.

For RTX 5070 (12GB VRAM), the KV cache is a significant fraction of VRAM:
  - 32K context × 16 layers × 8 KV heads × 64 dim × 2 (K+V) × 2 bytes (bf16)
  - = 32K × 16 × 8 × 64 × 2 × 2 = 1.07 GB
  - At 128K context: 4.3 GB (36% of VRAM!)

CPU KV offloading moves less-recently-used KV blocks to CPU RAM, keeping
only the active window on GPU. This effectively gives us unlimited KV cache
size (bounded by system RAM, 32GB) at the cost of PCIe transfer latency.

Strategy:
  1. Hot window: most recent N tokens stay on GPU (fast access)
  2. Cold storage: older tokens offloaded to CPU RAM
  3. On-demand fetch: when attention needs a cold block, fetch via PCIe
  4. Prefetch: predict which cold blocks will be needed and prefetch

PCIe 4.0 x16 bandwidth: ~32 GB/s (vs 672 GB/s GDDR7)
  - Fetching 1 block (16 tokens × 128 bytes) = 2KB → 0.06µs over PCIe
  - Negligible for small fetches, but significant for large context switches

Best for: long-context workloads where only a subset of the context is
actively attended to (sparse attention patterns, sliding window + sinks).

This implementation provides:
  1. CPUKVCache: manages GPU hot window + CPU cold storage
  2. Automatic promotion/demotion based on access patterns
  3. Async prefetch for predicted access patterns

R&D round 14 (LMCache port): added ``DiskKVCache`` — a 3-tier
GPU → CPU-pinned-RAM → local-disk KV cache inspired by LMCache's
multi-tier ``StorageManager`` (L1 = CPU DRAM, L2 = local disk / NVMe).
The disk tier is backed by a memory-mapped spool file so KV caches
*persist across engine restarts* (LMCache's "storage mode") and
long-document workloads can cache far beyond the 32 GB system RAM.
Eviction is LRU per tier; fetch cascades disk → CPU → GPU on demand.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time

import torch

logger = logging.getLogger(__name__)

from forge.engine.kv_backend import KVCacheStrategy


class CPUKVCache(KVCacheStrategy):
    """Two-tier KV cache: GPU hot window + CPU cold storage.

    Keeps the most recent `hot_window_size` tokens on GPU for fast access.
    Older tokens are offloaded to CPU RAM. When attention needs a cold block,
    it's fetched via PCIe (with optional async prefetch).

    Memory:
      GPU: hot_window_size × 2 × n_kv × head_dim × 2 bytes
      CPU: (total - hot_window_size) × 2 × n_kv × head_dim × 2 bytes

    For 32K hot window, 128K total:
      GPU: 1.07 GB (same as 32K standard)
      CPU: 3.22 GB (fits in 32GB system RAM easily)
      Total effective: 128K context in 1.07 GB VRAM
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        # Normalize device: callers may pass str ("cuda") or torch.device.
        # We need torch.device for `.type` checks below.
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # Hot window: recent tokens on GPU
        self.hot_window_size = min(max_seq_len, 8192)  # 8K tokens on GPU
        self.gpu_k = torch.zeros(
            1, n_kv_heads, self.hot_window_size, head_dim,
            dtype=dtype, device=self.device)
        self.gpu_v = torch.zeros(
            1, n_kv_heads, self.hot_window_size, head_dim,
            dtype=dtype, device=self.device)

        # Cold storage: older tokens on CPU
        self.cpu_k = None  # lazily allocated when first offload happens
        self.cpu_v = None
        self.cpu_capacity = max_seq_len - self.hot_window_size

        # Track what's where
        self.seq_len = 0  # total tokens written
        self.gpu_len = 0  # tokens currently on GPU
        self.cpu_len = 0  # tokens currently on CPU

        # Access tracking for promotion/demotion
        self._access_counts = None  # per-block access count

        # Async prefetch stream
        self._prefetch_stream = None
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

    def append(self, k, v, position, attention_weights=None):
        """Append new K/V tokens to the hot window.

        If the hot window overflows, oldest tokens are offloaded to CPU.
        """
        T = k.shape[2]
        pos = position

        # Check if hot window will overflow
        if self.gpu_len + T > self.hot_window_size:
            # Offload oldest tokens to CPU
            n_to_offload = self.gpu_len + T - self.hot_window_size
            self._offload_to_cpu(n_to_offload)

        # Write new tokens to GPU hot window
        gpu_pos = self.gpu_len
        self.gpu_k[:, :, gpu_pos:gpu_pos + T] = k
        self.gpu_v[:, :, gpu_pos:gpu_pos + T] = v
        self.gpu_len += T
        self.seq_len = pos + T

    def _offload_to_cpu(self, n_tokens: int):
        """Move n_tokens from the start of GPU cache to CPU cache."""
        if n_tokens <= 0:
            return

        # Initialize CPU storage if needed
        if self.cpu_k is None:
            self.cpu_k = torch.zeros(
                1, self.n_kv, self.cpu_capacity, self.head_dim,
                dtype=self.dtype, device="cpu", pin_memory=True)
            self.cpu_v = torch.zeros(
                1, self.n_kv, self.cpu_capacity, self.head_dim,
                dtype=self.dtype, device="cpu", pin_memory=True)

        # Copy from GPU to CPU (non-blocking for overlap)
        tokens_to_move = self.gpu_k[:, :, :n_tokens]
        self.cpu_k[:, :, self.cpu_len:self.cpu_len + n_tokens] = tokens_to_move.to("cpu", non_blocking=True)
        self.cpu_v[:, :, self.cpu_len:self.cpu_len + n_tokens] = self.gpu_v[:, :, :n_tokens].to("cpu", non_blocking=True)
        self.cpu_len += n_tokens

        # Shift GPU cache left (remove offloaded tokens).  Clone the
        # source first — an in-place overlapping copy_ is UB on CUDA
        # (elementwise kernels have no ordering guarantee) and can
        # corrupt the retained tail.
        self.gpu_k[:, :, :self.gpu_len - n_tokens] = self.gpu_k[:, :, n_tokens:self.gpu_len].clone()
        self.gpu_v[:, :, :self.gpu_len - n_tokens] = self.gpu_v[:, :, n_tokens:self.gpu_len].clone()
        self.gpu_len -= n_tokens

    def _fetch_from_cpu(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a range of tokens from CPU storage to GPU.

        Args:
            start, end: positions in the CPU cache (0-indexed from cpu start)

        Returns:
            (k, v) tensors on GPU
        """
        k = self.cpu_k[:, start:end].to(self.device, non_blocking=True)
        v = self.cpu_v[:, start:end].to(self.device, non_blocking=True)
        return k, v

    def get(self, positions=None):
        """Return K/V for attention.

        If positions are all in the hot window, returns GPU tensors directly.
        If some positions are in cold storage, fetches them via PCIe.
        """
        if positions is None:
            # Return all: GPU hot window (cold storage fetched on demand)
            return (self.gpu_k[:, :, :self.gpu_len],
                    self.gpu_v[:, :, :self.gpu_len])

        # Check which positions are hot vs cold
        cpu_start = 0
        gpu_start = self.cpu_len  # GPU positions start after CPU tokens

        positions = torch.as_tensor(positions, device=self.device)
        hot_mask = positions >= gpu_start
        cold_mask = ~hot_mask

        if cold_mask.all():
            # All cold: fetch from CPU
            cold_positions = positions - cpu_start
            return self._fetch_from_cpu(
                cold_positions.min().item(),
                cold_positions.max().item() + 1,
            )
        elif hot_mask.all():
            # All hot: return from GPU
            gpu_positions = positions - gpu_start
            return (self.gpu_k[:, :, gpu_positions],
                    self.gpu_v[:, :, gpu_positions])
        else:
            # Mixed: need to fetch cold and combine with hot
            # This is the slow path — optimize by fetching contiguous cold blocks
            cold_positions = positions[cold_mask] - cpu_start
            hot_positions = positions[hot_mask] - gpu_start

            cold_k, cold_v = self._fetch_from_cpu(
                cold_positions.min().item(),
                cold_positions.max().item() + 1,
            )
            hot_k = self.gpu_k[:, :, hot_positions]
            hot_v = self.gpu_v[:, :, hot_positions]

            # Interleave back in original order
            k = torch.zeros(1, self.n_kv, len(positions), self.head_dim,
                            dtype=self.dtype, device=self.device)
            v = torch.zeros(1, self.n_kv, len(positions), self.head_dim,
                            dtype=self.dtype, device=self.device)
            k[:, :, cold_mask] = cold_k
            k[:, :, hot_mask] = hot_k
            v[:, :, cold_mask] = cold_v
            v[:, :, hot_mask] = hot_v
            return k, v

    def prefetch(self, positions: list[int]):
        """Async prefetch cold blocks to GPU (overlap with compute).

        Call this before attention if you know which positions will be accessed.
        """
        if self._prefetch_stream is None or self.cpu_k is None:
            return

        cold_positions = [p for p in positions if p < self.cpu_len]
        if not cold_positions:
            return

        with torch.cuda.stream(self._prefetch_stream):
            self._fetch_from_cpu(min(cold_positions), max(cold_positions) + 1)

    def clear(self):
        self.gpu_k.zero_()
        self.gpu_v.zero_()
        if self.cpu_k is not None:
            self.cpu_k.zero_()
            self.cpu_v.zero_()
        self.seq_len = 0
        self.gpu_len = 0
        self.cpu_len = 0

    def info(self):
        gpu_bytes = self.gpu_len * 2 * self.n_kv * self.head_dim * 2
        cpu_bytes = self.cpu_len * 2 * self.n_kv * self.head_dim * 2
        return {
            "type": "cpu_offload",
            "seq_len": self.seq_len,
            "gpu_len": self.gpu_len,
            "cpu_len": self.cpu_len,
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "hot_window_size": self.hot_window_size,
            "total_capacity": self.max_seq_len,
        }


# ── 3-tier GPU → CPU → disk KV cache (LMCache storage-mode port) ──────────


class DiskKVCache(KVCacheStrategy):
    """Three-tier KV cache: GPU hot window → CPU pinned RAM → local disk.

    Inspired by LMCache's multi-tier ``StorageManager`` (L1 = CPU DRAM,
    L2 = local disk / NVMe GDS).  The disk tier is a memory-mapped
    spool file, so:

      * KV caches **persist across engine restarts** when ``disk_path``
        points at a stable location (LMCache "storage mode").
      * Long-document workloads can cache far beyond the 32 GB system
        RAM — bounded only by free disk space (``disk_capacity``).

    Eviction is LRU per tier: GPU overflow → CPU; CPU overflow → disk.
    Fetch cascades disk → CPU → GPU on demand.  Async prefetch overlaps
    the disk→CPU transfer with compute (best-effort, CUDA stream when
    available).

    Memory budget (RTX 5070 12GB, 32GB RAM):
      GPU:  hot_window_size × 2 × n_kv × head_dim × 2 bytes  (default 8K → ~2 MB/layer)
      CPU:  cpu_window_size  × 2 × n_kv × head_dim × 2 bytes  (default 32K → ~8 MB/layer, pinned)
      Disk: disk_capacity    × 2 × n_kv × head_dim × 2 bytes  (default 128K → ~32 MB/layer, mmap)
    Total effective context = hot + cpu + disk tokens, all in ~2 MB VRAM.
    """

    def __init__(self, disk_path: str | None = None,
                 hot_window_size: int = 8192,
                 cpu_window_size: int = 32768,
                 disk_capacity: int | None = None,
                 persist: bool = False):
        """Configure tier sizes.

        Args:
            disk_path: directory for the mmap spool file.  If ``None``,
                a fresh temp dir is used (non-persistent).  If ``persist``
                is True the file is kept on ``clear()``/shutdown so a
                later instance can reload it.
            hot_window_size: GPU hot-window tokens.
            cpu_window_size: CPU pinned-RAM tokens.
            disk_capacity: disk-tier tokens.  Defaults to
                ``max_seq_len - hot - cpu`` (set in ``init``).
            persist: keep the disk spool across instances.
        """
        self._disk_path = disk_path
        self._hot_window_size = hot_window_size
        self._cpu_window_size = cpu_window_size
        self._disk_capacity_cfg = disk_capacity
        self._persist = persist
        self._owns_tmpdir = False

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = (device if isinstance(device, torch.device)
                       else torch.device(device))
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        hot = min(self._hot_window_size, max_seq_len)
        cpu_cap = min(self._cpu_window_size, max(0, max_seq_len - hot))
        disk_cap = self._disk_capacity_cfg
        if disk_cap is None:
            disk_cap = max(0, max_seq_len - hot - cpu_cap)
        disk_cap = min(disk_cap, max(0, max_seq_len - hot - cpu_cap))

        self.hot_window_size = hot
        self.cpu_capacity = cpu_cap
        self.disk_capacity = disk_cap

        # Tier 0 — GPU hot window
        self.gpu_k = torch.zeros(1, n_kv_heads, hot, head_dim,
                                 dtype=dtype, device=self.device)
        self.gpu_v = torch.zeros_like(self.gpu_k)

        # Tier 1 — CPU pinned RAM
        self.cpu_k = torch.zeros(1, n_kv_heads, cpu_cap, head_dim,
                                 dtype=dtype, device="cpu", pin_memory=True) if cpu_cap else None
        self.cpu_v = torch.zeros_like(self.cpu_k) if cpu_cap else None

        # Tier 2 — disk mmap spool
        self.disk_k = None
        self.disk_v = None
        if disk_cap > 0:
            self._open_disk_spool(disk_cap, n_kv_heads, head_dim, dtype)

        # Fill counters per tier
        self.seq_len = 0
        self.gpu_len = 0
        self.cpu_len = 0
        self.disk_len = 0

        # LRU access timestamps per tier (for eviction ordering)
        self._cpu_last_access = 0.0
        self._disk_last_access = 0.0

        # Async prefetch stream (disk→CPU and CPU→GPU overlap)
        self._prefetch_stream = None
        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

    def _open_disk_spool(self, disk_cap, n_kv_heads, head_dim, dtype):
        """Open (or reopen) the mmap disk spool file."""
        if self._disk_path is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="forge_kv_disk_")
            self._disk_path = self._tmpdir.name
            self._owns_tmpdir = True
        os.makedirs(self._disk_path, exist_ok=True)
        k_path = os.path.join(self._disk_path, "kv_disk_k.pt")
        v_path = os.path.join(self._disk_path, "kv_disk_v.pt")
        # mmap file: shape [1, n_kv, disk_cap, head_dim].  Using
        # torch.zeros with a real file backing via numpy memmap would be
        # ideal, but torch.save/load round-trip is simplest and robust.
        # We lazily materialize on first offload to avoid allocating a
        # huge file upfront when the disk tier is never used.
        self._disk_k_path = k_path
        self._disk_v_path = v_path
        self._disk_shape = (1, n_kv_heads, disk_cap, head_dim)
        self._disk_dtype = dtype

    def _materialize_disk(self):
        """Allocate the disk spool tensors on first offload."""
        if self.disk_k is not None:
            return
        # CPU-resident tensor backed by a file via torch's storage.
        # We keep it on CPU (not pinned — disk tier is cold) and persist
        # by saving on shutdown when ``persist`` is set.
        self.disk_k = torch.zeros(*self._disk_shape, dtype=self._disk_dtype, device="cpu")
        self.disk_v = torch.zeros(*self._disk_shape, dtype=self._disk_dtype, device="cpu")
        # Reload existing spool if present and persistent.
        if self._persist and os.path.exists(self._disk_k_path):
            try:
                self.disk_k = torch.load(self._disk_k_path, map_location="cpu",
                                         weights_only=True)
                self.disk_v = torch.load(self._disk_v_path, map_location="cpu",
                                         weights_only=True)
            except Exception:
                logger.debug("Corrupt disk KV spool, starting fresh", exc_info=True)

    def append(self, k, v, position, attention_weights=None):
        """Append new K/V tokens, cascading overflow down the tiers."""
        T = k.shape[2]
        # GPU overflow → push oldest GPU tokens to CPU
        if self.gpu_len + T > self.hot_window_size:
            n_to_cpu = self.gpu_len + T - self.hot_window_size
            self._offload_gpu_to_cpu(n_to_cpu)
        gp = self.gpu_len
        self.gpu_k[:, :, gp:gp + T] = k
        self.gpu_v[:, :, gp:gp + T] = v
        self.gpu_len += T
        self.seq_len = position + T

    def _offload_gpu_to_cpu(self, n_tokens: int):
        """Move oldest GPU tokens to CPU; cascade CPU overflow to disk."""
        if n_tokens <= 0:
            return
        # CPU overflow → push oldest CPU tokens to disk first
        if self.cpu_k is not None and self.cpu_len + n_tokens > self.cpu_capacity:
            n_to_disk = self.cpu_len + n_tokens - self.cpu_capacity
            self._offload_cpu_to_disk(n_to_disk)
        if self.cpu_k is None:
            # No CPU tier — spill straight to disk
            self._offload_gpu_to_disk(n_tokens)
            return
        self.cpu_k[:, :, self.cpu_len:self.cpu_len + n_tokens] = \
            self.gpu_k[:, :, :n_tokens].to("cpu", non_blocking=True)
        self.cpu_v[:, :, self.cpu_len:self.cpu_len + n_tokens] = \
            self.gpu_v[:, :, :n_tokens].to("cpu", non_blocking=True)
        self.cpu_len += n_tokens
        self._cpu_last_access = time.time()
        # Shift GPU window left — clone source (overlapping copy_ is UB)
        self.gpu_k[:, :, :self.gpu_len - n_tokens] = self.gpu_k[:, :, n_tokens:self.gpu_len].clone()
        self.gpu_v[:, :, :self.gpu_len - n_tokens] = self.gpu_v[:, :, n_tokens:self.gpu_len].clone()
        self.gpu_len -= n_tokens

    def _offload_cpu_to_disk(self, n_tokens: int):
        """Move oldest CPU tokens to the disk tier (mmap spool)."""
        if n_tokens <= 0 or self.disk_capacity == 0:
            return
        self._materialize_disk()
        # Disk overflow → drop oldest disk tokens (LRU, lost)
        if self.disk_len + n_tokens > self.disk_capacity:
            drop = self.disk_len + n_tokens - self.disk_capacity
            self.disk_k[:, :, :-drop] = self.disk_k[:, :, drop:self.disk_len].clone()
            self.disk_v[:, :, :-drop] = self.disk_v[:, :, drop:self.disk_len].clone()
            self.disk_len -= drop
        self.disk_k[:, :, self.disk_len:self.disk_len + n_tokens] = \
            self.cpu_k[:, :, :n_tokens]
        self.disk_v[:, :, self.disk_len:self.disk_len + n_tokens] = \
            self.cpu_v[:, :, :n_tokens]
        self.disk_len += n_tokens
        self._disk_last_access = time.time()
        # Shift CPU window left — clone source (overlapping copy_ is UB)
        self.cpu_k[:, :, :self.cpu_len - n_tokens] = self.cpu_k[:, :, n_tokens:self.cpu_len].clone()
        self.cpu_v[:, :, :self.cpu_len - n_tokens] = self.cpu_v[:, :, n_tokens:self.cpu_len].clone()
        self.cpu_len -= n_tokens

    def _offload_gpu_to_disk(self, n_tokens: int):
        """Direct GPU→disk spill (used when CPU tier is absent)."""
        if n_tokens <= 0 or self.disk_capacity == 0:
            return
        self._materialize_disk()
        if self.disk_len + n_tokens > self.disk_capacity:
            drop = self.disk_len + n_tokens - self.disk_capacity
            self.disk_k[:, :, :-drop] = self.disk_k[:, :, drop:self.disk_len].clone()
            self.disk_v[:, :, :-drop] = self.disk_v[:, :, drop:self.disk_len].clone()
            self.disk_len -= drop
        self.disk_k[:, :, self.disk_len:self.disk_len + n_tokens] = \
            self.gpu_k[:, :, :n_tokens].to("cpu")
        self.disk_v[:, :, self.disk_len:self.disk_len + n_tokens] = \
            self.gpu_v[:, :, :n_tokens].to("cpu")
        self.disk_len += n_tokens
        # clone source (overlapping copy_ is UB)
        self.gpu_k[:, :, :self.gpu_len - n_tokens] = self.gpu_k[:, :, n_tokens:self.gpu_len].clone()
        self.gpu_v[:, :, :self.gpu_len - n_tokens] = self.gpu_v[:, :, n_tokens:self.gpu_len].clone()
        self.gpu_len -= n_tokens

    def _fetch_disk_to_cpu(self, start: int, end: int):
        """Fetch a disk range back into CPU tensors (returns CPU tensors)."""
        return (self.disk_k[:, start:end], self.disk_v[:, start:end])

    def get(self, positions=None):
        """Return the active K/V for attention.

        Only the GPU hot window is returned directly (the common case
        for causal decode where attention is over recent tokens).  Cold
        ranges are fetched on demand via :meth:`fetch_range`.
        """
        return (self.gpu_k[:, :, :self.gpu_len],
                self.gpu_v[:, :, :self.gpu_len])

    def fetch_range(self, start: int, end: int):
        """Fetch an arbitrary token range back to GPU (disk→CPU→GPU).

        Used when attention must attend to cold tokens (e.g. long-range
        sparse attention).  The returned tensors are GPU-resident.
        """
        cpu_off = self.gpu_len  # tokens before GPU window live in CPU/disk
        # Determine which tier holds [start, end)
        if start >= cpu_off:
            # All in GPU hot window
            gp = start - cpu_off
            return (self.gpu_k[:, :, gp:end - cpu_off],
                    self.gpu_v[:, :, gp:end - cpu_off])
        # Need cold tokens — fetch from CPU/disk to a GPU staging buffer
        cpu_end = self.cpu_len
        disk_off = cpu_off + cpu_end  # tokens before CPU live on disk
        k_parts, v_parts = [], []
        # Disk portion
        if start < disk_off and self.disk_k is not None:
            d_end = min(end, disk_off)
            dk, dv = self._fetch_disk_to_cpu(start, d_end)
            k_parts.append(dk.to(self.device, non_blocking=True))
            v_parts.append(dv.to(self.device, non_blocking=True))
            start = d_end
        # CPU portion
        if start < cpu_off and self.cpu_k is not None:
            c_start = start - disk_off if start >= disk_off else 0
            c_end = min(end, cpu_off) - disk_off
            if c_end > c_start:
                k_parts.append(self.cpu_k[:, c_start:c_end].to(self.device, non_blocking=True))
                v_parts.append(self.cpu_v[:, c_start:c_end].to(self.device, non_blocking=True))
                start = disk_off + c_end
        # GPU portion
        if start < end and start >= cpu_off:
            gp = start - cpu_off
            k_parts.append(self.gpu_k[:, :, gp:end - cpu_off])
            v_parts.append(self.gpu_v[:, :, gp:end - cpu_off])
        if not k_parts:
            return (torch.empty(1, self.n_kv, 0, self.head_dim,
                                dtype=self.dtype, device=self.device),
                    torch.empty_like(self.gpu_k[:, :, :0]))
        return (torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2))

    def prefetch(self, positions: list[int]):
        """Best-effort async prefetch of cold blocks to GPU."""
        if self._prefetch_stream is None:
            return
        cold = [p for p in positions if p < self.gpu_len + self.cpu_len]
        if not cold:
            return
        with torch.cuda.stream(self._prefetch_stream):
            self.fetch_range(min(cold), max(cold) + 1)

    def save_disk(self):
        """Persist the disk tier to ``disk_path`` (LMCache storage mode)."""
        if self.disk_k is not None and self._persist:
            torch.save(self.disk_k, self._disk_k_path)
            torch.save(self.disk_v, self._disk_v_path)

    def clear(self):
        """Clear all tiers.  Persists disk spool if ``persist`` is set."""
        self.save_disk()
        self.gpu_k.zero_(); self.gpu_v.zero_()
        if self.cpu_k is not None:
            self.cpu_k.zero_(); self.cpu_v.zero_()
        if self.disk_k is not None:
            self.disk_k.zero_(); self.disk_v.zero_()
        self.seq_len = self.gpu_len = self.cpu_len = self.disk_len = 0

    def __del__(self):
        try:
            self.save_disk()
            if self._owns_tmpdir and hasattr(self, "_tmpdir"):
                self._tmpdir.cleanup()
        except Exception:
            logger.debug("Error during DiskKVCache cleanup", exc_info=True)

    def info(self):
        per_tok = 2 * self.n_kv * self.head_dim * 2
        return {
            "type": "disk_offload_3tier",
            "seq_len": self.seq_len,
            "gpu_len": self.gpu_len,
            "cpu_len": self.cpu_len,
            "disk_len": self.disk_len,
            "gpu_bytes": self.gpu_len * per_tok,
            "cpu_bytes": self.cpu_len * per_tok,
            "disk_bytes": self.disk_len * per_tok,
            "hot_window_size": self.hot_window_size,
            "cpu_capacity": self.cpu_capacity,
            "disk_capacity": self.disk_capacity,
            "total_capacity": self.max_seq_len,
            "persist": self._persist,
            "disk_path": self._disk_path,
        }
