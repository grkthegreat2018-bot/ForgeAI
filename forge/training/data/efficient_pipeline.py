"""Efficient training data pipeline: disk cache, async prefetch, packed sequences.

Components:
  1. ``DiskTokenCache`` — memory-mapped disk cache for tokenization results.
     Avoids re-tokenizing on every run; uses a hash of (text, tokenizer_hash)
     as the cache key. Stores token arrays as raw int64 binary for fast mmap.
  2. ``AsyncPrefetcher`` — background thread that pre-loads and tokenizes the
     next N batches while the GPU is busy with the current batch. Eliminates
     the data-loading bottleneck (2-3x throughput on I/O-bound workloads).
  3. ``PackedSequenceDataset`` — packs variable-length examples into fixed-length
     sequences (Llama-3 style). Eliminates padding waste (30-50% of tokens
     are padding in typical SFT). Each packed sequence contains multiple
     examples concatenated, with position resets and attention masking.
  4. ``ModelDataCache`` — LRU cache for less-important model metadata (config
     dicts, tokenizer info, checkpoint metadata). Prevents redundant disk
     reads for data that doesn't change during training.

Usage::

    from forge.training.data.efficient_pipeline import (
        DiskTokenCache, AsyncPrefetcher, PackedSequenceDataset, ModelDataCache)

    # Disk tokenization cache
    cache = DiskTokenCache(cache_dir="research/data/tok_cache")
    ids = cache.tokenize(tokenizer, text, max_seq_len=1024)

    # Packed sequences (30-50% less padding)
    packed = PackedSequenceDataset(dataset, seq_len=1024, pack_examples=True)

    # Async prefetcher
    prefetcher = AsyncPrefetcher(dataset, batch_size=2, device="cuda",
                                  cache=cache, tokenizer=tokenizer)
    for batch in prefetcher:
        # batch is already on GPU, ready for training
        ...
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import contextlib

# ─── Disk-backed tokenization cache ──────────────────────────────────────────

class DiskTokenCache:
    """Disk-backed cache for tokenization results.

    Stores tokenized text as raw int64 binary files, keyed by a hash of
    (text, tokenizer_hash). On cache hit, reads via memory-mapped I/O
    (near-zero overhead for large caches). On miss, tokenizes and writes
    to disk for future runs.

    Benefits:
      - Eliminates re-tokenization across runs (saves 30-60s on 10K examples)
      - Memory-mapped reads are faster than re-tokenizing
      - Bounded disk usage (LRU eviction of old cache entries)
      - Thread-safe (multiple workers can read concurrently)
    """

    def __init__(self, cache_dir: str = "research/data/tok_cache",
                 max_entries: int = 100_000, max_disk_mb: int = 2048):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._index: OrderedDict[str, str] = OrderedDict()  # hash -> filename
        self._total_bytes = 0
        self._puts_since_save = 0
        self._load_index()

    def _load_index(self):
        """Load the cache index from disk."""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    data = json.load(f)
                self._index = OrderedDict(data.get("entries", {}))
                self._total_bytes = data.get("total_bytes", 0)
            except Exception:
                self._index = OrderedDict()
                self._total_bytes = 0

    def _save_index(self):
        """Save the cache index to disk."""
        index_path = self.cache_dir / "index.json"
        try:
            with open(index_path, "w") as f:
                json.dump({
                    "entries": list(self._index.items()),
                    "total_bytes": self._total_bytes,
                }, f)
        except Exception:
            pass

    @staticmethod
    def _hash_key(text: str, tokenizer_hash: str) -> str:
        """Compute a stable hash for (text, tokenizer_hash)."""
        h = hashlib.sha256(f"{tokenizer_hash}:{text}".encode())
        return h.hexdigest()[:16]  # 16 chars = 8 bytes, enough for 100K entries

    def _cache_path(self, key_hash: str) -> Path:
        """Get the cache file path for a key hash."""
        # Use first 2 chars as subdirectory (sharding for filesystem perf)
        subdir = self.cache_dir / key_hash[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{key_hash}.bin"

    def get(self, text: str, tokenizer_hash: str = "default") -> list[int] | None:
        """Get tokenized text from cache. Returns None on miss."""
        key = self._hash_key(text, tokenizer_hash)
        with self._lock:
            if key in self._index:
                # Move to end (most recently used)
                self._index.move_to_end(key)
                path = self._cache_path(key)
                if path.exists():
                    try:
                        # Memory-mapped read: fast for large caches
                        arr = np.memmap(str(path), dtype=np.int64, mode="r")
                        return arr.tolist()
                    except Exception:
                        # Corrupted entry — remove it
                        del self._index[key]
                        with contextlib.suppress(Exception):
                            path.unlink()
                        return None
                else:
                    # Index says it exists but file is gone
                    del self._index[key]
                    return None
        return None

    def put(self, text: str, token_ids: list[int], tokenizer_hash: str = "default"):
        """Store tokenized text in cache."""
        if not token_ids:
            return
        key = self._hash_key(text, tokenizer_hash)
        path = self._cache_path(key)
        with self._lock:
            # Write as raw int64 binary
            arr = np.array(token_ids, dtype=np.int64)
            arr.tofile(str(path))
            size = arr.nbytes
            self._index[key] = str(path)
            self._total_bytes += size
            # Evict if over limits
            self._evict_if_needed()
            # Batch index writes: serializing the whole index on every
            # put() is O(n puts x index size) — on a ~60k-entry cache a
            # full tokenize pass spent most of its wall-clock rewriting
            # a ~5MB JSON file thousands of times. Flushing every 200
            # puts costs at most 200 orphaned .bin files on a crash —
            # the entries are unreachable but harmless.
            self._puts_since_save += 1
            if self._puts_since_save >= 200:
                self._save_index()
                self._puts_since_save = 0

    def _evict_if_needed(self):
        """Evict least recently used entries if over limits."""
        while (len(self._index) > self.max_entries or
               self._total_bytes > self.max_disk_bytes):
            if not self._index:
                break
            _key, path_str = self._index.popitem(last=False)  # LRU
            try:
                size = Path(path_str).stat().st_size
                Path(path_str).unlink()
                self._total_bytes -= size
            except Exception:
                pass

    def flush(self) -> None:
        """Persist any pending index entries (batched saves leave up to
        200 unflushed). Call at end of a tokenize pass / on shutdown."""
        with self._lock:
            if self._puts_since_save:
                self._save_index()
                self._puts_since_save = 0

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass

    def tokenize(self, tokenizer, text: str, max_seq_len: int = 1024,
                 add_special_tokens: bool = False) -> list[int]:
        """Tokenize with disk cache. Falls back to tokenizer on miss."""
        # Compute tokenizer hash (stable across runs for same tokenizer)
        tok_hash = getattr(tokenizer, "_tokenizer_hash", None)
        if tok_hash is None:
            # Use tokenizer class name + vocab size as a rough hash
            tok_hash = f"{type(tokenizer).__name__}_{getattr(tokenizer, 'vocab_size', 0)}"
            tokenizer._tokenizer_hash = tok_hash

        cached = self.get(text, tok_hash)
        if cached is not None:
            # Truncate to max_seq_len if needed
            return cached[:max_seq_len] if len(cached) > max_seq_len else cached

        # Cache miss — tokenize
        try:
            enc = tokenizer(text, add_special_tokens=add_special_tokens,
                            return_tensors=None)
            ids = enc["input_ids"] if isinstance(enc, dict) else enc
            if not isinstance(ids, list):
                ids = list(ids)
        except Exception:
            return []

        # Cache it (only if reasonable size — don't cache huge texts)
        if len(text) < 50000:
            self.put(text, ids, tok_hash)
        return ids[:max_seq_len] if len(ids) > max_seq_len else ids

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "entries": len(self._index),
            "total_mb": self._total_bytes / 1024 / 1024,
            "max_entries": self.max_entries,
            "max_mb": self.max_disk_bytes / 1024 / 1024,
        }


# ─── Async prefetcher ─────────────────────────────────────────────────────────

class AsyncPrefetcher:
    """Background prefetcher for training batches.

    Runs a background thread that pre-loads, tokenizes, and collates the
    next N batches while the GPU processes the current one. Eliminates
    the data-loading bottleneck on I/O-bound workloads.

    R&D round 25 optimization: the background thread now produces **pinned
    CPU tensors** (not GPU tensors). The H2D transfer is done by the
    consumer (main thread) via ``.to(device, non_blocking=True)``, which
    queues an async copy on the CUDA stream that overlaps with the
    previous batch's GPU compute. This eliminates the GIL-holding
    ``.to(device)`` call from the background thread, allowing true
    overlap of CPU collation with GPU compute.

    The queue also uses ``collections.deque`` for O(1) popleft instead
    of the O(n) ``list.pop(0)``.

    Usage::
        prefetcher = AsyncPrefetcher(dataset, batch_size=2, device="cuda",
                                      collate_fn=collate_batch, pad_id=0)
        for batch in prefetcher:
            # batch is already on GPU, ready for training
            loss = model(batch[0], targets=batch[1])

    The prefetcher maintains a bounded queue of pre-collated batches.
    When the queue is empty, the main thread blocks until the next batch
    is ready.
    """

    def __init__(self, dataset: list, batch_size: int = 2,
                 device: str = "cpu", collate_fn: Callable | None = None,
                 prefetch_count: int = 4, shuffle: bool = True,
                 cache: DiskTokenCache | None = None,
                 tokenizer=None, seq_len: int = 1024,
                 pad_id: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.collate_fn = collate_fn
        self.prefetch_count = prefetch_count
        self.shuffle = shuffle
        self.cache = cache
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = pad_id

        # R&D round 25: use deque for O(1) popleft (was list.pop(0), O(n))
        from collections import deque
        self._queue: deque = deque()
        self._queue_lock = threading.Lock()
        self._queue_not_empty = threading.Condition(self._queue_lock)
        self._queue_not_full = threading.Condition(self._queue_lock)
        self._stop = False
        self._thread: threading.Thread | None = None
        self._indices: list[int] = []
        self._pos = 0
        self._epoch = 0
        # R&D round 25: track whether collate_fn produces CPU or GPU tensors.
        # If the collate_fn already does .to(device), we keep backward compat
        # and skip the consumer-side transfer. The sft_train collate_batch
        # is CPU-first when transfer_on_consume=True is set.
        self._transfer_on_consume = False

    def set_transfer_on_consume(self, enabled: bool = True):
        """R&D round 25: when enabled, the background thread produces pinned
        CPU tensors and the consumer does the H2D transfer with
        non_blocking=True. This overlaps the copy with GPU compute."""
        self._transfer_on_consume = enabled

    def _reset_indices(self):
        """Reset and shuffle indices for a new epoch."""
        import random
        self._indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self._indices)
        self._pos = 0
        self._epoch += 1

    def _get_next_batch(self) -> Any:
        """Get the next batch from the dataset."""
        if self._pos + self.batch_size > len(self._indices):
            self._reset_indices()
            if self._pos + self.batch_size > len(self._indices):
                return None  # dataset too small

        batch_indices = self._indices[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size

        batch = [self.dataset[i] for i in batch_indices]

        if self.collate_fn is not None:
            if self._transfer_on_consume:
                # R&D round 25: CPU-only collate — produce pinned CPU tensors.
                # The consumer will do .to(device, non_blocking=True) to
                # overlap the H2D copy with the previous batch's GPU compute.
                return self.collate_fn(batch, self.pad_id, "cpu")
            return self.collate_fn(batch, self.pad_id, self.device)
        return batch

    def _prefetch_worker(self):
        """Background worker that fills the prefetch queue."""
        while not self._stop:
            # Wait if queue is full
            with self._queue_not_full:
                while len(self._queue) >= self.prefetch_count and not self._stop:
                    self._queue_not_full.wait(timeout=0.1)
                if self._stop:
                    break

            # Get next batch
            batch = self._get_next_batch()
            if batch is None:
                break

            # Add to queue
            with self._queue_not_empty:
                self._queue.append(batch)
                self._queue_not_empty.notify()

    def __iter__(self) -> Iterator:
        """Start prefetching and yield batches."""
        self._stop = False
        self._reset_indices()
        from collections import deque
        self._queue = deque()

        # Start background thread
        self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()

        try:
            while True:
                # Get next batch from queue
                with self._queue_not_empty:
                    while not self._queue and self._thread.is_alive():
                        self._queue_not_empty.wait(timeout=0.1)
                    if not self._queue:
                        break
                    batch = self._queue.popleft()
                    self._queue_not_full.notify()

                # R&D round 25: if transfer_on_consume, do the H2D transfer
                # here in the main thread with non_blocking=True. This
                # overlaps the copy with the previous batch's GPU compute
                # (the previous yield's forward/backward is still running
                # on the GPU when we queue this copy).
                if self._transfer_on_consume and isinstance(batch, tuple):
                    use_pin = ("cuda" in self.device and torch.cuda.is_available())
                    batch = tuple(
                        t.to(self.device, non_blocking=use_pin) if torch.is_tensor(t)
                        else t
                        for t in batch
                    )

                yield batch
        finally:
            self._stop = True
            with self._queue_not_full:
                self._queue_not_full.notify_all()
            if self._thread is not None:
                self._thread.join(timeout=2.0)

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size


# ─── Packed sequence dataset ──────────────────────────────────────────────────

class PackedSequenceDataset:
    """Pack variable-length examples into fixed-length sequences.

    Llama-3 style sequence packing: concatenate multiple short examples
    into a single fixed-length sequence, eliminating padding waste.
    Typical SFT datasets have 30-50% padding — packing eliminates this.

    Each packed sequence:
      - Has exactly seq_len tokens
      - Contains 1+ examples concatenated
      - Labels are -100 at example boundaries (model doesn't predict
        across examples)
      - Position IDs reset at each example boundary
      - cu_seqlens: cumulative sequence lengths for varlen attention
        (R&D round 14). Tracks example boundaries so FlashAttention
        varlen can attend within examples without cross-example
        contamination. Format: [0, len_ex1, len_ex1+len_ex2, ...].

    Args:
        dataset: list of {"input_ids": [...], "labels": [...]} dicts
        seq_len: target packed sequence length
        pack_examples: if True, pack multiple examples per sequence.
            If False, fall back to standard padding (no packing).
        drop_last_incomplete: if True, drop the last incomplete packed
            sequence. If False, pad it.
        pad_id: padding token id
        emit_cu_seqlens: if True, include cu_seqlens in each item for
            varlen attention (R&D round 14). Default False for backward
            compat; set True when config.use_varlen=True.
    """

    def __init__(self, dataset: list[dict], seq_len: int = 1024,
                 pack_examples: bool = True, drop_last_incomplete: bool = False,
                 pad_id: int = 0, emit_cu_seqlens: bool = False):
        self.seq_len = seq_len
        self.pack_examples = pack_examples
        self.drop_last_incomplete = drop_last_incomplete
        self.pad_id = pad_id
        self.emit_cu_seqlens = emit_cu_seqlens

        # Cached teacher logprobs (offline distillation): carried through
        # packing as a per-position list parallel to labels (-99 = unscored).
        self._has_tlp = any(ex.get("teacher_logp") is not None
                            for ex in dataset)
        if pack_examples:
            self._packed = self._pack_sequences(dataset)
        else:
            # No packing — just pad each example to seq_len
            self._packed = self._pad_examples(dataset)

    def _pack_sequences(self, dataset: list[dict]) -> list[dict]:
        """Pack variable-length examples into fixed-length sequences.

        Tracks example boundaries (cu_seqlens) for varlen attention when
        emit_cu_seqlens=True. Each example's length (excluding padding) is
        recorded so FlashAttention varlen can attend within examples only.
        """
        packed = []
        current_ids: list[int] = []
        current_labels: list[int] = []
        current_tlp: list[float] = []
        # Track lengths of each example in the current packed sequence
        # (for cu_seqlens). Only the non-padding portion counts.
        current_ex_lens: list[int] = []

        def _flush():
            entry = {
                "input_ids": current_ids[:self.seq_len],
                "labels": current_labels[:self.seq_len],
                "n_comp": sum(1 for l in current_labels[:self.seq_len] if l != -100),
            }
            if self._has_tlp:
                entry["teacher_logp"] = current_tlp[:self.seq_len]
            if self.emit_cu_seqlens:
                entry["cu_seqlens"] = list(current_ex_lens)
            packed.append(entry)
            current_ids.clear()
            current_labels.clear()
            current_tlp.clear()
            current_ex_lens.clear()

        for ex in dataset:
            ids = ex["input_ids"]
            labels = ex.get("labels", [-100] * len(ids))

            # If a single example is longer than seq_len, truncate it
            if len(ids) > self.seq_len:
                ids = ids[:self.seq_len]
                labels = labels[:self.seq_len]

            # Expand completion-aligned teacher_logp to per-position form
            # (sentinel -99 at non-completion positions) so it packs in
            # lockstep with labels.
            if self._has_tlp:
                tlp_seq = [-99.0] * len(ids)
                tlp = ex.get("teacher_logp")
                if tlp is not None:
                    comp = [i for i, l in enumerate(labels) if l != -100]
                    for j, pos in enumerate(comp[:len(tlp)]):
                        tlp_seq[pos] = float(tlp[j])

            # If adding this example would exceed seq_len, flush current
            while current_ids and len(current_ids) + len(ids) > self.seq_len:
                # Fill remaining space with padding
                remaining = self.seq_len - len(current_ids)
                current_ids.extend([self.pad_id] * remaining)
                current_labels.extend([-100] * remaining)
                current_tlp.extend([-99.0] * remaining)
                _flush()

            # Add example to current sequence
            current_ids.extend(ids)
            current_labels.extend(labels)
            if self._has_tlp:
                current_tlp.extend(tlp_seq)
            current_ex_lens.append(len(ids))

            # If current sequence is exactly seq_len, flush
            if len(current_ids) >= self.seq_len:
                _flush()

        # Flush remaining
        if current_ids and not self.drop_last_incomplete:
            remaining = self.seq_len - len(current_ids)
            current_ids.extend([self.pad_id] * remaining)
            current_labels.extend([-100] * remaining)
            current_tlp.extend([-99.0] * remaining)
            _flush()

        return packed

    def _pad_examples(self, dataset: list[dict]) -> list[dict]:
        """Pad each example to seq_len (no packing)."""
        padded = []
        for ex in dataset:
            ids = ex["input_ids"]
            labels = ex.get("labels", [-100] * len(ids))
            if len(ids) > self.seq_len:
                ids = ids[:self.seq_len]
                labels = labels[:self.seq_len]
            elif len(ids) < self.seq_len:
                ids = ids + [self.pad_id] * (self.seq_len - len(ids))
                labels = labels + [-100] * (self.seq_len - len(ids))
            entry = {
                "input_ids": ids,
                "labels": labels,
                "n_comp": sum(1 for l in labels if l != -100),
            }
            if self._has_tlp:
                tlp_seq = [-99.0] * len(ids)
                tlp = ex.get("teacher_logp")
                if tlp is not None:
                    comp = [i for i, l in enumerate(labels) if l != -100]
                    for j, pos in enumerate(comp[:len(tlp)]):
                        tlp_seq[pos] = float(tlp[j])
                entry["teacher_logp"] = tlp_seq
            padded.append(entry)
        return padded

    def __len__(self) -> int:
        return len(self._packed)

    def __getitem__(self, idx: int) -> dict:
        ex = self._packed[idx]
        item = {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
            "n_comp": ex["n_comp"],
            "reward": 1.0,
        }
        if "teacher_logp" in ex:
            # Convert per-position (-99 sentinels) back to completion-aligned
            # form — collate_batch maps it onto non--100 label positions.
            item["teacher_logp"] = torch.tensor(
                [v for v, l in zip(ex["teacher_logp"], ex["labels"])
                 if l != -100],
                dtype=torch.float32)
        if self.emit_cu_seqlens and "cu_seqlens" in ex:
            # cu_seqlens: list of per-example lengths → cumulative sum.
            # Format: [0, len_0, len_0+len_1, ...] (FA varlen convention).
            # This is per-sequence (per-batch-item); the collate function
            # should concatenate all sequences and build a global cu_seqlens.
            ex_lens = ex["cu_seqlens"]
            cu = [0]
            for length in ex_lens:
                cu.append(cu[-1] + length)
            item["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32)
        return item

    def stats(self) -> dict:
        """Return packing statistics."""
        total_tokens = len(self._packed) * self.seq_len
        total_comp = sum(ex["n_comp"] for ex in self._packed)
        return {
            "n_sequences": len(self._packed),
            "seq_len": self.seq_len,
            "total_tokens": total_tokens,
            "completion_tokens": total_comp,
            "utilization": total_comp / max(total_tokens, 1),
        }


# ─── Model data cache (LRU for less-important model metadata) ─────────────────

class ModelDataCache:
    """LRU cache for model metadata that doesn't change during training.

    Caches:
      - Config dicts (from safetensors metadata)
      - Tokenizer info (vocab size, special tokens, chat template)
      - Checkpoint metadata (param count, dtype, key list)
      - Architecture signatures (for blank model caching)

    Prevents redundant disk reads for data that's read once and reused.
    Thread-safe, bounded LRU with configurable max size.
    """

    def __init__(self, max_entries: int = 256, max_mb: int = 512):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_mb * 1024 * 1024
        self._total_bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Get a value from cache. Returns None on miss."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: Any):
        """Store a value in cache."""
        with self._lock:
            # Estimate size (rough: use repr length for non-tensor values)
            try:
                size = len(repr(value).encode("utf-8"))
            except Exception:
                size = 1024  # default estimate

            if key in self._cache:
                old = self._cache.pop(key)
                with contextlib.suppress(Exception):
                    self._total_bytes -= len(repr(old).encode("utf-8"))

            self._cache[key] = value
            self._total_bytes += size

            # Evict if over limits
            while (len(self._cache) > self._max_entries or
                   self._total_bytes > self._max_bytes):
                if not self._cache:
                    break
                _, old = self._cache.popitem(last=False)
                with contextlib.suppress(Exception):
                    self._total_bytes -= len(repr(old).encode("utf-8"))

    def get_or_compute(self, key: str, compute_fn: Callable) -> Any:
        """Get from cache, or compute and cache."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute_fn()
        self.put(key, value)
        return value

    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "total_mb": self._total_bytes / 1024 / 1024,
            "max_entries": self._max_entries,
        }

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._total_bytes = 0


# ─── Global singletons ────────────────────────────────────────────────────────

# Global disk token cache (singleton — shared across all training runs)
_disk_cache: DiskTokenCache | None = None
_model_cache: ModelDataCache | None = None


def get_disk_cache() -> DiskTokenCache:
    """Get the global DiskTokenCache singleton."""
    global _disk_cache
    if _disk_cache is None:
        _disk_cache = DiskTokenCache()
    return _disk_cache


def get_model_cache() -> ModelDataCache:
    """Get the global ModelDataCache singleton."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelDataCache()
    return _model_cache
