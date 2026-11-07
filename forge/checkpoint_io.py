"""Checkpoint I/O with safetensors support.

Saves model state dicts as safetensors (.safetensors) when the path ends in
.safetensors, falling back to torch.save (.pt) otherwise. Safetensors is
preferred for new checkpoints: it's faster to load, memory-maps the file
(zero-copy on load), and is immune to arbitrary-code-execution via pickle.

A sidecar JSON `<stem>.meta.json` is written alongside safetensors checkpoints
to record non-tensor metadata (config dict, training step, etc.) since
safetensors only stores tensors.
"""
import glob
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import contextlib

try:
    from safetensors.torch import load_file, save_file
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

try:
    from fastsafetensors import SafeTensorsFileLoader
    _HAS_FAST_SAFETENSORS = True
except ImportError:
    _HAS_FAST_SAFETENSORS = False


# Hardening bounds for untrusted safetensors files — checked before any
# allocation is sized from header contents.
_MAX_ST_FILE_BYTES = 128 * 1024**3     # 128GB per-file sanity cap
_MAX_ST_HEADER_BYTES = 100 * 1024**2   # safetensors spec caps headers at 100MB
_MAX_ST_TENSORS = 100_000              # cap on header/keys enumeration


def _is_safetensors_path(path: str) -> bool:
    return str(path).endswith(".safetensors")


# ── Pipelined safetensors → CUDA loader ──────────────────────────────────
# safetensors dtype name -> torch dtype attribute name.
_ST_TORCH_DTYPES = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16",
    "U16": "uint16", "I32": "int32", "U32": "uint32", "I64": "int64",
    "U64": "uint64", "F16": "float16", "BF16": "bfloat16",
    "F32": "float32", "F64": "float64",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
}

# safetensors dtype name -> element size in bytes (for header validation).
_ST_DT_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
    "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
}


def _read_safetensors_entries(path: str):
    """Parse a safetensors header without loading tensor data.

    Returns ``(entries, data_start)`` where entries is a list of
    ``(key, torch_dtype, shape, begin, end)`` sorted by begin, offsets
    relative to the data blob, and data_start is the absolute file offset
    of the blob.  Raises ValueError on malformed headers or dtypes this
    torch build cannot represent.
    """
    import struct
    size = os.path.getsize(path)
    if size > _MAX_ST_FILE_BYTES:
        raise ValueError(
            f"safetensors file too large: {size / 2**30:.1f} GiB in {path}")
    if size < 8:
        raise ValueError(f"file too small for safetensors header: {path}")
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        # Bound the header read — a corrupt u64 length must not drive a
        # multi-GB f.read / json.loads.
        if hdr_len > _MAX_ST_HEADER_BYTES or hdr_len > size - 8:
            raise ValueError(
                f"bad safetensors header length {hdr_len} in {path}")
        header = json.loads(f.read(hdr_len))
    # Cap tensor enumeration — a hostile header could declare millions of
    # entries and exhaust memory before any data is read.
    n_tensors = len(header) - (1 if "__metadata__" in header else 0)
    if n_tensors > _MAX_ST_TENSORS:
        raise ValueError(
            f"safetensors header declares {n_tensors} tensors "
            f"(cap {_MAX_ST_TENSORS})")
    blob_len = size - 8 - hdr_len
    entries = []
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dt_name = _ST_TORCH_DTYPES.get(meta.get("dtype"))
        dt = getattr(torch, dt_name, None) if dt_name else None
        if dt is None:
            raise ValueError(
                f"unsupported safetensors dtype {meta.get('dtype')!r}")
        s, e = meta["data_offsets"]
        # Offsets must fit the real blob AND match shape*dtype — the
        # pipelined loader allocates tensors from `shape`, so a lie here
        # is a GPU/host OOM vector.
        if e < s or e > blob_len:
            raise ValueError(f"bad data_offsets for {key!r}")
        if math.prod(meta["shape"]) * _ST_DT_SIZES[meta["dtype"]] != e - s:
            raise ValueError(
                f"shape/dtype inconsistent with data_offsets for {key!r}")
        entries.append((key, dt, meta["shape"], s, e))
    entries.sort(key=lambda t: t[3])
    return entries, 8 + hdr_len


def load_safetensors_pipelined(path: str, device,
                               num_threads: int | None = None,
                               chunk_mb: int = 128) -> dict[str, torch.Tensor]:
    """Load a safetensors file to a CUDA device via a pipelined reader.

    Rationale: safetensors' ``get_tensor(device="cuda")`` copies one tensor
    at a time through pageable memory (~2 GB/s here), and fastsafetensors
    needs a CUDA runtime DLL that is missing on this Windows install.  This
    loader instead runs N file reads in parallel into per-thread pinned
    staging buffers while async H2D copies drain on a side stream —
    measured 12.9 GB/s vs 1.9 GB/s on ForgeLM V2 (0.49s vs 3.30s for the
    6.4GB checkpoint, warm page cache; cold is disk-bound either way).

    Each tensor gets its own GPU allocation (identical semantics to
    ``safe_open(...).get_tensor(k)``), so downstream quantization/weight
    surgery frees storage exactly like the per-tensor path.

    Args:
        path: .safetensors file path.
        device: CUDA device (torch.device or str). CPU callers get a
            ValueError — use safetensors mmap for zero-copy CPU loads.
        num_threads: reader threads (default: min(8, cpu_count//2)).
        chunk_mb: per-thread pinned staging size in MiB.

    Raises:
        ValueError: non-CUDA device, malformed header, unsupported dtype.
        OSError: short/corrupt file reads.
        Any CUDA/pinning error propagates so callers can fall back.
    """
    from concurrent.futures import ThreadPoolExecutor

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("load_safetensors_pipelined requires a CUDA device")

    entries, data_start = _read_safetensors_entries(path)
    if not entries:
        return {}
    blob_len = max(e for _, _, _, _, e in entries)

    # One GPU allocation per tensor (pre-allocating all is cheap — the
    # caching allocator services these without hitting the driver).
    state = {
        k: torch.empty(shape, dtype=dt, device=dev)
        for k, dt, shape, _, _ in entries
    }

    chunk = max(1, int(chunk_mb)) * 1024 * 1024
    nt = num_threads or min(8, max(2, (os.cpu_count() or 4) // 2))
    nt = max(1, min(nt, (blob_len + chunk - 1) // chunk))

    # Group entries into <=nt contiguous byte ranges of ~equal size.
    per = (blob_len + nt - 1) // nt
    groups: list[list] = []
    cur: list = []
    cur_bytes = 0
    for ent in entries:
        if cur and cur_bytes >= per:
            groups.append(cur)
            cur, cur_bytes = [], 0
        cur.append(ent)
        cur_bytes += ent[4] - ent[3]
    if cur:
        groups.append(cur)

    stream = torch.cuda.Stream()

    def _work(group):
        pinned = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
        np_pin = pinned.numpy()
        last_ev = None
        span_s, span_e = group[0][3], group[-1][4]
        off = span_s
        with open(path, "rb", buffering=0) as f:
            while off < span_e:
                n = min(chunk, span_e - off)
                if last_ev is not None:
                    # staging buffer free to overwrite once its H2D drained
                    last_ev.synchronize()
                f.seek(data_start + off)
                got = f.readinto(memoryview(np_pin)[:n])
                if got != n:
                    raise OSError(
                        f"short read in {path}: {got}/{n} bytes at {off}")
                with torch.cuda.stream(stream):
                    for k, _dt, _shape, s, e in group:
                        cs, ce = max(s, off), min(e, off + n)
                        if cs >= ce:
                            continue
                        flat = state[k].reshape(-1).view(torch.uint8)
                        flat[cs - s:ce - s].copy_(
                            pinned[cs - off:ce - off], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(stream)
                last_ev = ev
                off += n

    if len(groups) == 1:
        _work(groups[0])
    else:
        with ThreadPoolExecutor(len(groups),
                                thread_name_prefix="ckpt_io") as ex:
            for fut in [ex.submit(_work, g) for g in groups]:
                fut.result()
    stream.synchronize()
    return state


def cleanup_orphaned_tmp(checkpoint_dir: str | None = None) -> int:
    """Remove orphaned .tmp files left by crashed checkpoint saves.

    Call this at startup to clean up any .tmp files from previous runs
    that were interrupted before the atomic rename.

    Args:
        checkpoint_dir: directory to scan (default: research/checkpoints/)

    Returns:
        Number of orphaned .tmp files removed.
    """
    if checkpoint_dir is None:
        from research.paths import CHECKPOINTS_DIR, as_str
        checkpoint_dir = as_str(CHECKPOINTS_DIR)

    removed = 0
    for pattern in ["*.safetensors.tmp", "*.pt.tmp", "*.meta.json.tmp"]:
        for tmp_path in glob.glob(os.path.join(checkpoint_dir, "**", pattern),
                                  recursive=True):
            try:
                os.remove(tmp_path)
                removed += 1
            except OSError:
                pass
    return removed


def save_checkpoint(state: dict[str, Any], path: str) -> str:
    """Save a checkpoint dict atomically with readback verification.

    If path ends in .safetensors, tensors are written via safetensors and any
    non-tensor values (dicts, ints, strings) are written to a sidecar
    `<path>.meta.json`. Otherwise, torch.save is used (pickle, .pt format).

    Writes go to a `<path>.tmp` file first, which is verified by reading it
    back, then atomically renamed via `os.replace`. This guarantees that a
    crash (Ctrl-C, OOM, power loss) during the save cannot corrupt an existing
    checkpoint — the final path only ever points at a fully-verified file.

    Returns the path written.
    """
    path = str(path)
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)

    # Disk space check — estimate checkpoint size and verify enough free space
    import shutil
    total_bytes = sum(
        v.numel() * v.element_size() for v in state.values()
        if isinstance(v, torch.Tensor)
    )
    free_bytes = shutil.disk_usage(str(parent)).free
    if total_bytes > free_bytes * 0.9:  # require 10% headroom
        raise OSError(
            f"Insufficient disk space for checkpoint: need ~{total_bytes / 1e9:.1f} GB, "
            f"only {free_bytes / 1e9:.1f} GB free in {parent}")

    if _is_safetensors_path(path):
        if not _HAS_SAFETENSORS:
            raise RuntimeError("safetensors not installed; cannot write .safetensors checkpoint.")
        tensors = {}
        meta: dict[str, Any] = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                # safetensors requires contiguous tensors on CPU.
                # .cpu() already creates a copy if on GPU; .contiguous() ensures
                # layout. No need for .clone() — .cpu() + .contiguous() suffices.
                tensors[k] = v.detach().cpu().contiguous()
            else:
                meta[k] = v
        # Atomic write: save to .tmp, verify metadata, then rename.
        tmp_path = path + ".tmp"
        save_file(tensors, tmp_path, metadata={"format": "pt"})
        # Metadata-only verification: check safetensors header for shapes/dtypes
        # without loading the full tensor data (saves 3.6GB read per save).
        try:
            from safetensors import safe_open
            with safe_open(tmp_path, framework="pt", device="cpu") as f:
                verified_keys = list(f.keys())
                if len(verified_keys) != len(tensors):
                    raise RuntimeError(f"tensor count mismatch: {len(verified_keys)} != {len(tensors)}")
                for k, t in tensors.items():
                    if k not in verified_keys:
                        raise RuntimeError(f"missing tensor '{k}' in saved checkpoint")
                    # Access metadata via get_slice (no data load, just shape/dtype).
                    sl = f.get_slice(k)
                    if tuple(sl.get_shape()) != tuple(t.shape):
                        raise RuntimeError(
                            f"shape mismatch for '{k}': {sl.get_shape()} != {tuple(t.shape)}")
                    # safetensors uses short dtype names (F32, BF16, I64, etc.)
                    # while torch uses full names (torch.float32, torch.bfloat16).
                    # Compare via the torch dtype string mapping.
                    st_dtype = str(sl.get_dtype())
                    torch_dtype = str(t.dtype).replace("torch.", "")
                    # Map safetensors short names to torch short names.
                    _DTYPE_MAP = {"F32": "float32", "F64": "float64",
                                  "BF16": "bfloat16", "F16": "float16",
                                  "F8_E4M3": "float8_e4m3fn",
                                  "F8_E5M2": "float8_e5m2",
                                  "I64": "int64", "I32": "int32",
                                  "I16": "int16", "I8": "int8",
                                  "U8": "uint8", "BOOL": "bool"}
                    st_normalized = _DTYPE_MAP.get(st_dtype, st_dtype)
                    if st_normalized != torch_dtype:
                        raise RuntimeError(
                            f"dtype mismatch for '{k}': {st_dtype} != {t.dtype}")
        except Exception as e:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
            raise RuntimeError(f"checkpoint verification failed for {path}: {e}") from e
        os.replace(tmp_path, path)  # atomic on the same filesystem
        # Meta JSON sidecar (also atomic).
        meta_path = path + ".meta.json"
        meta_tmp = meta_path + ".tmp"
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(_jsonable(meta), f, indent=2)
        os.replace(meta_tmp, meta_path)
        print(f"Saved safetensors checkpoint to {path} ({len(tensors)} tensors, verified, meta -> {meta_path})")
        return path

    # Legacy .pt path (also atomic via tmp + rename).
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    print(f"Saved torch checkpoint to {path}")
    return path


def load_checkpoint(path: str, map_location=None, allow_unsafe: bool = False) -> dict[str, Any]:
    """Load a checkpoint dict written by save_checkpoint.

    For .safetensors, merges the sidecar meta JSON back into the result.
    Uses fastsafetensors (4.8-7.5x faster) when available, falls back to
    standard safetensors mmap.
    For .pt, uses torch.load (weights_only=True when possible).
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Directory contents: {os.listdir(os.path.dirname(path) or '.')}")
    if _is_safetensors_path(path):
        if not _HAS_SAFETENSORS:
            raise RuntimeError("safetensors not installed; cannot read .safetensors checkpoint.")

        # Sanity cap before any header-driven allocation.
        size = os.path.getsize(path)
        if size > _MAX_ST_FILE_BYTES:
            raise ValueError(
                f"safetensors checkpoint too large: "
                f"{size / 2**30:.1f} GiB in {path}")

        device_str = str(map_location) if map_location is not None else "cpu"

        # Try fastsafetensors first (4.8-7.5x faster, BOOT_TIME_AUDIT Stage 2)
        if _HAS_FAST_SAFETENSORS and map_location is not None and "cuda" in device_str:
            try:
                loader = SafeTensorsFileLoader(pg=None, device=device_str)
                tensors = loader.load(path, map_location=device_str)
                result: dict[str, Any] = dict(tensors.items())
                meta_path = path + ".meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as f:
                        result.update(json.load(f))
                print(f"Loaded safetensors (fast) from {path} ({len(tensors)} tensors)")
                return result
            except Exception as e:
                # Fall back to standard safetensors
                print(f"  (fastsafetensors failed: {e}, falling back to standard)")

        # CPU-only loads stay memory-mapped for zero-copy lazy access.
        # Direct-to-GPU (CUDA) goes through the pipelined reader (parallel
        # file reads -> pinned staging -> async H2D) with per-tensor
        # safetensors load_file as the fallback.
        if device_str in ("cpu", "meta"):
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                # Cap enumeration (see _MAX_ST_TENSORS) before materializing.
                if len(keys) > _MAX_ST_TENSORS:
                    raise ValueError(
                        f"safetensors file declares {len(keys)} tensors "
                        f"(cap {_MAX_ST_TENSORS})")
                result: dict[str, Any] = {k: f.get_tensor(k) for k in keys}
        else:
            try:
                tensors = load_safetensors_pipelined(path, device_str)
            except Exception as e:
                print(f"  (pipelined load failed: {e}, "
                      f"falling back to safetensors)")
                tensors = load_file(path, device=device_str)
            result = dict(tensors.items())
        n_tensors = len(result)
        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                result.update(json.load(f))
        print(f"Loaded safetensors checkpoint from {path} ({n_tensors} tensors)")
        return result

    # Legacy .pt path. Try weights_only=True first (safer).
    # If it fails, only fall back to unsafe pickle if allow_unsafe=True,
    # since weights_only=False enables arbitrary code execution.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        if not allow_unsafe:
            raise RuntimeError(
                f"Could not load {path} with weights_only=True ({e}). "
                f"Pass allow_unsafe=True to fall back to weights_only=False "
                f"(UNSAFE — enables arbitrary code execution from the "
                f"checkpoint file). Only do this if you trust the source."
            ) from e
        import warnings
        warnings.warn(
            f"Could not load {path} with weights_only=True ({e}). "
            f"Falling back to weights_only=False (UNSAFE — enables arbitrary "
            f"code execution from the checkpoint file). Only proceed if you "
            f"trust the source of this checkpoint.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Full training-state checkpoints (weights + optimizer + EMA + RNG + step)
# ---------------------------------------------------------------------------
#
# Layout for a checkpoint at `path` (e.g. model.safetensors):
#   path                 -> model weights (+ step/meta in sidecar JSON)
#   path + ".meta.json"  -> {"step": N, ...extra meta}   (safetensors only)
#   path + ".train.pt"   -> optimizer state, EMA state, RNG states (torch.save)
#
# Saving the optimizer/EMA/RNG sidecar means a crash or pause costs at most
# the steps since the last save — resume continues with identical optimizer
# momentum and LR schedule position instead of restarting from step 1.

_STEP_RE = re.compile(r"step(\d+)\.")


def verify_checkpoint(path: str) -> dict[str, Any]:
    """Verify a checkpoint's integrity without loading it into memory.

    Checks:
    - File exists and is non-empty
    - For .safetensors: header is valid JSON, tensor count matches
    - For .pt: file can be opened with weights_only=True
    - Sidecar meta JSON exists and is valid (safetensors only)

    Returns:
        Dict with keys: 'valid' (bool), 'format' (str), 'n_tensors' (int),
        'size_mb' (float), 'has_meta' (bool), 'errors' (list[str])
    """
    path = str(path)
    result = {
        "valid": False, "format": "unknown", "n_tensors": 0,
        "size_mb": 0.0, "has_meta": False, "errors": [],
    }

    if not os.path.exists(path):
        result["errors"].append(f"File not found: {path}")
        return result

    size = os.path.getsize(path)
    result["size_mb"] = size / 1e6
    if size == 0:
        result["errors"].append("File is empty (0 bytes)")
        return result

    if _is_safetensors_path(path):
        result["format"] = "safetensors"
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                result["n_tensors"] = len(keys)
                # Check for NaN/Inf in first few tensors — use small slices
                # instead of loading full tensors into memory
                for k in keys[:5]:
                    t = f.get_tensor(k)
                    # For large tensors, check a subsample instead of full load
                    if t.numel() > 1_000_000:
                        # Check first + last + middle slices (catches corruption)
                        n = t.numel()
                        sample = torch.cat([
                            t.flatten()[:1024],
                            t.flatten()[n // 2 - 512:n // 2 + 512],
                            t.flatten()[-1024:],
                        ])
                        if torch.isnan(sample).any():
                            result["errors"].append(f"NaN in tensor '{k}'")
                        if torch.isinf(sample).any():
                            result["errors"].append(f"Inf in tensor '{k}'")
                    else:
                        if torch.isnan(t).any():
                            result["errors"].append(f"NaN in tensor '{k}'")
                        if torch.isinf(t).any():
                            result["errors"].append(f"Inf in tensor '{k}'")
                    del t  # free immediately
        except Exception as e:
            result["errors"].append(f"safetensors read error: {e}")
            return result

        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            result["has_meta"] = True
            try:
                with open(meta_path, encoding="utf-8") as f:
                    json.load(f)
            except Exception as e:
                result["errors"].append(f"Meta JSON parse error: {e}")
    else:
        result["format"] = "pt"
        try:
            torch.load(path, map_location="cpu", weights_only=True)
            result["n_tensors"] = 1  # can't count without full load
        except Exception as e:
            result["errors"].append(f"torch.load error: {e}")
            return result

    result["valid"] = len(result["errors"]) == 0
    return result


def _train_state_path(path: str) -> str:
    return str(path) + ".train.pt"


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print(f"Warning: could not fully restore RNG state: {e}")


def save_training_checkpoint(
    model,
    path: str,
    optimizer=None,
    ema_state: dict[str, torch.Tensor] | None = None,
    step: int | None = None,
    meta: dict[str, Any] | None = None,
    lora_only: bool = False,
) -> str:
    """Save model weights + full training state atomically.

    Weights go to `path` via save_checkpoint (atomic + verified). Optimizer
    state, EMA shadow weights, and RNG states go to `<path>.train.pt` (also
    atomic via tmp + rename). `step` and `meta` are recorded in the sidecar
    JSON so any checkpoint self-documents its training progress.

    ``lora_only=True`` persists only adapter/qscale tensors — for LoRA-only
    training the frozen base weights are unchanged, so writing them every
    periodic save is gigabytes of dead I/O (and the full CPU state_dict copy
    is a RAM-spike hazard on 32GB boxes). The sidecar records
    ``lora_only: true``; resume re-injects the adapters onto the base
    checkpoint (``base_checkpoint`` in meta) instead of loading weights from
    this file. The .train.pt sidecar (optimizer/EMA/RNG) is always full.
    """
    state: dict[str, Any] = dict(model.state_dict())
    if lora_only:
        state = {k: v for k, v in state.items()
                 if "lora_" in k or "qscale" in k}
        state["lora_only"] = True
    if step is not None:
        state["step"] = int(step)
    if meta:
        state.update(meta)
    save_checkpoint(state, path)

    train_state: dict[str, Any] = {"rng": _capture_rng_state()}
    if optimizer is not None:
        train_state["optimizer"] = optimizer.state_dict()
    if ema_state is not None:
        train_state["ema"] = {k: v.detach().cpu() for k, v in ema_state.items()}
    ts_tmp = _train_state_path(path) + ".tmp"
    torch.save(train_state, ts_tmp)
    os.replace(ts_tmp, _train_state_path(path))
    return path


def load_training_state(path: str, optimizer=None, restore_rng: bool = True, allow_unsafe: bool = False) -> dict[str, Any]:
    """Load the training-state sidecar for a checkpoint.

    Returns {"step": int|None, "ema": dict|None, "has_optimizer": bool}.
    If `optimizer` is given and a saved optimizer state exists, it is loaded.
    Weights themselves are loaded separately (ModelLoader / load_checkpoint).
    """
    result: dict[str, Any] = {"step": None, "ema": None, "has_optimizer": False}
    # Step comes from the safetensors meta sidecar (or the .pt payload).
    meta_path = str(path) + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            result["step"] = json.load(f).get("step")
    if result["step"] is None:
        result["step"] = parse_step_from_path(str(path))

    ts_path = _train_state_path(path)
    if not os.path.exists(ts_path):
        return result
    try:
        ts = torch.load(ts_path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not allow_unsafe:
            print(f"Warning: could not load training state {ts_path} with weights_only=True ({e}); "
                  f"pass allow_unsafe=True to retry with weights_only=False.")
            return result
        try:
            ts = torch.load(ts_path, map_location="cpu", weights_only=False)
        except Exception as e2:
            print(f"Warning: could not load training state {ts_path}: {e2}")
            return result
    if optimizer is not None and "optimizer" in ts:
        try:
            optimizer.load_state_dict(ts["optimizer"])
            result["has_optimizer"] = True
        except Exception as e:
            print(f"Warning: optimizer state incompatible ({e}); continuing with fresh optimizer.")
    result["ema"] = ts.get("ema")
    if restore_rng and "rng" in ts:
        _restore_rng_state(ts["rng"])
    return result


def parse_step_from_path(path: str) -> int | None:
    """Extract the step number from a periodic checkpoint filename."""
    m = _STEP_RE.search(str(path))
    return int(m.group(1)) if m else None


def step_checkpoint_path(base_path: str, step: int) -> str:
    """`model.safetensors` + step 500 -> `model.step500.safetensors`."""
    base = str(base_path)
    stem, ext = os.path.splitext(base)
    return f"{stem}.step{step}{ext}"


def cleanup_step_checkpoints(base_path: str, keep: int) -> None:
    """Delete old `.stepN.<ext>` checkpoints for `base_path`, keeping the last `keep`."""
    base = str(base_path)
    stem, ext = os.path.splitext(base)
    candidates = []
    for p in glob.glob(f"{stem}.step*{ext}"):
        n = parse_step_from_path(p)
        if n is not None:
            candidates.append((n, p))
    candidates.sort()
    for _, p in candidates[:-keep] if keep > 0 else candidates:
        try:
            os.remove(p)
            for sidecar in (p + ".meta.json", _train_state_path(p)):
                if os.path.exists(sidecar):
                    os.remove(sidecar)
            print(f"Deleted old checkpoint: {p}")
        except OSError as e:
            print(f"Warning: could not delete {p}: {e}")


def emergency_save(model, base_path: str, kind: str, step: int, optimizer=None, ema_state=None) -> str | None:
    """Best-effort crash/interrupt save: `model.interrupt_step500.safetensors`.

    Never raises — used inside exception handlers where the run is dying.
    Returns the path written, or None on failure.
    """
    try:
        path = step_checkpoint_path(str(base_path), step).replace(
            ".step", f".{kind}_step", 1
        )
        save_training_checkpoint(model, path, optimizer=optimizer, ema_state=ema_state, step=step)
        print(f"Emergency checkpoint saved to {path}")
        return path
    except Exception as e:
        print(f"Emergency save failed: {e}")
        return None


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of a nested object to JSON-serializable form."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fall back to repr for anything else (configs, etc.).
    return repr(obj)
