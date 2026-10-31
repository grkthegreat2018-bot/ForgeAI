"""Vast.ai cloud backend for compute-heavy ForgeAI training.

Rents on-demand GPU instances on Vast.ai, syncs the ForgeAI repo +
checkpoints + training data, launches an ``sft_train.py`` run on the
remote box, streams the logs back to the local terminal, downloads the
resulting checkpoint, and tears down the instance.

## Architecture (v2 — SDK + persistent lifecycle)

Three transport layers, kept cleanly separated:

1. **Vast.ai control plane** — rent / list / stop / start / destroy
   instances, search offers, manage volumes, fetch SSH connection
   details. Driven by the official ``vastai`` Python SDK
   (``from vastai import VastAI``). The SDK provides typed returns,
   handles auth, and is the maintained interface.

2. **Remote execution plane** — SSH command exec, SFTP file sync, and
   log streaming. Driven by ``paramiko`` (Python-native, cross-platform,
   no dependency on Windows OpenSSH / rsync). The SDK's ``execute()``
   polls for only ~9s (too short for provisioning or training), so
   paramiko is used for long-running commands. The SDK's ``logs()``
   is used for historical log retrieval.

3. **Incremental sync layer** — manifest-based file sync. Computes a
   local file manifest (path → size + mtime + sha256), compares with
   the remote manifest, and uploads only changed files. This eliminates
   the "re-upload everything every run" problem.

## Instance lifecycle (persistent reuse)

Instead of destroy-after-run, the connector supports:

- **Reuse by label**: instances are tagged ``forgeai-<config>`` and
  found on subsequent runs via ``show_instances``.
- **Stop/start**: after training, the instance is **stopped** (not
  destroyed). Container disk persists (venv, repo, checkpoints all
  survive). GPU charges stop; only cheap disk charges continue.
- **Next run**: find stopped instance by label → start → sync only
  changed files → skip provisioning if requirements hash unchanged →
  run training → stop again.
- **Volume persistence** (optional): a Vast.ai volume can be attached
  at creation time to survive instance destruction (for GPU-type
  switches on the same machine).
- **Full destroy**: only when the user explicitly requests it, or when
  the host machine goes away (poll trap: ``exited``/``unknown``/
  ``offline``).

## Hardware context (RTX 5070 12GB local)

Full fine-grain training of the 1.2B model (and especially the V10 preset) does not fit comfortably on 12GB VRAM for high-throughput
runs. Vast.ai lets you rent an H100 / A100 / RTX 4090 by the hour for the
heavy lift, then pull the checkpoint back to the local box for inference /
quantization. The connector is the bridge.

## Setup (one-time)

1. Create a Vast.ai account at https://console.vast.ai and add billing.
2. Grab your API key from Account → API Key.
3. Export it:  ``set VAST_API_KEY=<key>``  (or put in ``.env``).
4. ``pip install vastai paramiko``
5. ``vastai set api-key <key>``  (so the SDK can find it)
6. Generate an SSH keypair if you don't have one::

       ssh-keygen -t ed25519 -f %USERPROFILE%\\.ssh\\id_ed25519

   then register the public key with Vast::

       vastai create ssh-key %USERPROFILE%\\.ssh\\id_ed25519.pub

   Every instance you rent after this will authorise that key.

## Usage (standalone)

    python -m research.cloud.vast_connector run \\
        --data research/data/finetune/tool_use_fc_70.jsonl \\
        --config forgelm_v2 \\
        --checkpoint research/checkpoints/ForgeLM_V2.safetensors \\
        --save ForgeLM_V10.sft.safetensors \\
        --max-steps 500 --gpu-filter "gpu_name=RTX_4090" --max-price 0.5

## Usage (via sft_train.py)

    python -m forge.training.runners.sft_train \\
        --data research/data/finetune/tool_use_fc_70.jsonl \\
        --remote-vast --gpu-filter "gpu_name=RTX_4090" --max-price 0.5 \\
        --max-steps 500 --no-lora --no-bitnet-everywhere

The ``--remote-vast`` flag short-circuits local training: all other args are
forwarded to the remote ``sft_train.py`` invocation (with ``--remote-vast``
stripped and local paths remapped to the remote workspace).

## New CLI subcommands

    python -m research.cloud.vast_connector reuse      # Reuse stopped instance
    python -m research.cloud.vast_connector sync       # Sync only changed files
    python -m research.cloud.vast_connector logs <id>  # Fetch historical logs
    python -m research.cloud.vast_connector wipe <id>  # Wipe remote data
    python -m research.cloud.vast_connector stop <id>  # Stop (preserve disk)
    python -m research.cloud.vast_connector start <id> # Start stopped instance
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from loguru import logger
except Exception:  # pragma: no cover - loguru is a project dep but stay safe
    import logging
    logger = logging.getLogger("vast_connector")

import contextlib

from research.paths import PROJECT_ROOT

# ─── Remote workspace layout on the rented instance ────────────────────────
REMOTE_ROOT = "/workspace/forgeai"
REMOTE_VENV = "/workspace/venv"
REMOTE_REPO = f"{REMOTE_ROOT}/repo"
REMOTE_CKPT_DIR = f"{REMOTE_ROOT}/checkpoints"
REMOTE_DATA_DIR = f"{REMOTE_ROOT}/data"
REMOTE_OUT_DIR = f"{REMOTE_ROOT}/output"
REMOTE_MANIFEST = f"{REMOTE_ROOT}/.file_manifest.json"
REMOTE_PROVISION_HASH = f"{REMOTE_ROOT}/.provision_hash"
REMOTE_EXIT_CODE = f"{REMOTE_ROOT}/exit_code"

# ─── Critical files for training (minimal upload set) ──────────────────────
# Only these files are synced to the remote — NOT the entire forge/ tree.
# Derived by tracing sft_train.py's import tree for the forgelm_v2 config.
# Adding a new training dependency? Add its path here.
CRITICAL_SOURCE_FILES: list[str] = [
    # Core
    "research/paths.py",
    "forge/config.py",
    "forge/checkpoint_io.py",
    "forge/model_loader.py",
    "research/tokenizer_cache.py",
    # ForgeEngine + inference
    "forge/engine/forge_engine.py",
    "forge/engine/activation.py",
    "forge/engine/airllm_streamer.py",
    "forge/engine/crash_recovery.py",
    "forge/engine/decoding.py",
    "forge/engine/diagnostics.py",
    "forge/engine/engine_tools.py",
    "forge/engine/errors.py",
    "forge/engine/hotswap.py",
    "forge/engine/innovations.py",
    "forge/engine/kv_backend.py",
    "forge/engine/library.py",
    "forge/engine/prefix_cache.py",
    "forge/engine/session_cache.py",
    "forge/engine/tool_security.py",
    # Architecture keys
    "forge/keys/misc/base.py",
    "forge/keys/misc/pit_key.py",
    "forge/keys/misc/keystack.py",
    "forge/keys/architecture/attn_residual_key.py",
    "forge/keys/architecture/factorized_embed_key.py",
    "forge/keys/architecture/hyperloop_key.py",
    "forge/keys/architecture/mhc_key.py",
    "forge/keys/architecture/mod_router_key.py",
    "forge/keys/architecture/titan_memory_key.py",
    "forge/keys/attention/csa_key.py",
    "forge/keys/attention/differential_attn_key.py",
    "forge/keys/attention/gla_key.py",
    "forge/keys/attention/gta_key.py",
    "forge/keys/attention/lisa_key.py",
    "forge/keys/compression/kron_ffn_key.py",
    "forge/keys/compression/monarch_ffn_key.py",
    "forge/keys/compression/nlrq_ffn_key.py",
    "forge/keys/compression/tt_ffn_key.py",
    "forge/keys/position/lerope_key.py",
    "forge/keys/quantization/bitnet_b158_key.py",
    "forge/keys/quantization/fused_gemm_key.py",
    "forge/keys/moe/expert_tying_key.py",
    "forge/keys/safety.py",
    # MoE
    "forge/moe/moe.py",
    # Runtime
    "forge/runtime/task_logger.py",
    # Training
    "forge/training/bitnet_lora.py",
    "forge/training/training_utils.py",
    "forge/training/data/curriculum_augment.py",
    "forge/training/data/efficient_pipeline.py",
    "forge/training/data/parquet_dataset.py",
    "forge/training/optim/advanced_norm.py",
    "forge/training/optim/badam.py",
    "forge/training/runners/sft_train.py",
    "forge/training/runners/lazy_train.py",
    "forge/training/runners/oomb_trainer.py",
    "forge/training/runners/optimal_checkpoint.py",
    # From-scratch 8B init functions
    "forge/training/runners/train_8b_all.py",
    # Compression key (NLRQ FFN, needed by from-scratch init)
    "forge/keys/compression/nlrq_ffn_key.py",
]

# __init__.py files needed for each package in the tree
CRITICAL_INIT_FILES: list[str] = [
    "research/__init__.py",
    "forge/engine/__init__.py",
    "forge/keys/__init__.py",
    "forge/keys/architecture/__init__.py",
    "forge/keys/attention/__init__.py",
    "forge/keys/misc/__init__.py",
    "forge/keys/quantization/__init__.py",
    "forge/keys/compression/__init__.py",
    "forge/keys/position/__init__.py",
    "forge/keys/moe/__init__.py",
    "forge/moe/__init__.py",
    "forge/runtime/__init__.py",
    "forge/training/__init__.py",
    "forge/training/data/__init__.py",
    "forge/training/optim/__init__.py",
    "forge/training/runners/__init__.py",
]

# Default CUDA image. PyTorch 2.4 + CUDA 12.1 + cuDNN 9.
# This is a large image (~5GB) but has everything we need pre-installed.
# Boot takes 2-5 min on first pull, ~30s if cached on the host.
# For newer GPUs (H100, RTX 5090) with CUDA 12.4+, override --image.
DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"

# Instance label prefix for reuse discovery.
INSTANCE_LABEL_PREFIX = "forgeai"

# Poll-trap states: if actual_status becomes one of these, the instance will
# never reach "running" — must destroy and retry with a different offer.
POLL_TRAP_STATES = {"exited", "unknown", "offline"}

# Terminal colors for structured log output (disabled on non-TTY).
class _Color:
    RESET = "\033[0m" if sys.stderr.isatty() else ""
    RED = "\033[31m" if sys.stderr.isatty() else ""
    YELLOW = "\033[33m" if sys.stderr.isatty() else ""
    GREEN = "\033[32m" if sys.stderr.isatty() else ""
    CYAN = "\033[36m" if sys.stderr.isatty() else ""
    DIM = "\033[2m" if sys.stderr.isatty() else ""
    BOLD = "\033[1m" if sys.stderr.isatty() else ""


# ─── Dataclasses ───────────────────────────────────────────────────────────
@dataclass
class VastOffer:
    """A single Vast.ai GPU offer returned by ``vast.search_offers()``."""
    id: int
    gpu_name: str
    num_gpus: int
    vram_gb: float
    dph_total: float          # $/hour total (instance + GPU)
    reliability: float
    inet_down: float          # Mbps
    inet_up: float
    dlperf: float = 0.0       # deep-learning performance score (Vast.ai)
    disk_gb: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def perf_per_dollar(self) -> float:
        """Performance per dollar — higher is better. Falls back to
        num_gpus*vram if dlperf is missing (older API versions)."""
        perf = self.dlperf if self.dlperf > 0 else self.num_gpus * self.vram_gb
        return perf / max(self.dph_total, 1e-6)

    @property
    def label(self) -> str:
        ppd = self.perf_per_dollar
        return (f"#{self.id} {self.gpu_name} x{self.num_gpus} "
                f"{self.vram_gb:.0f}GB ${self.dph_total:.3f}/h "
                f"perf/$={ppd:.1f} rel={self.reliability:.2f}")


@dataclass
class RemoteTrainingSpec:
    """Everything needed to launch an sft_train run on a remote Vast box."""
    # Training args forwarded to remote sft_train.py (dict of arg→value).
    # Values may be str / int / float / bool / list[str]. Bool True => flag,
    # False => omitted. None => omitted.
    train_args: dict = field(default_factory=dict)
    # Local files/dirs to upload. Checkpoints + data are resolved to absolute
    # paths here; on the remote they land in REMOTE_CKPT_DIR / REMOTE_DATA_DIR.
    checkpoints: list[str] = field(default_factory=list)
    data_files: list[str] = field(default_factory=list)
    # Vast selection
    gpu_filter: str = ""            # empty = auto-select any GPU by perf/$
    max_price: float = 0.0          # 0 = no per-hour cap (use budget instead)
    min_vram_gb: float = 24.0
    min_reliability: float = 0.9
    disk_gb: int = 100
    image: str = DEFAULT_IMAGE
    on_demand: bool = True          # interruptible=False (more stable)
    # Budget: max total dollars to spend on the run. The connector estimates
    # training hours from max_steps + est_sec_per_step, computes the max
    # affordable $/hr, and filters offers accordingly. 0 = no budget cap.
    budget: float = 10.0
    est_sec_per_step: float = 5.0   # rough: V10 model on an A100. Override for 1.2B (~1.5s).
    # SSH
    ssh_key: str = ""               # private key path; default ~/.ssh/id_ed25519
    # Lifecycle (v2: persistent reuse)
    # auto_destroy=False (default): stop instance after training, preserve disk.
    # auto_destroy=True: destroy instance after training (irreversible).
    auto_destroy: bool = False
    # reuse_instance: if True, look for a stopped/running instance with our
    # label before creating a new one. Default True.
    reuse_instance: bool = True
    # use_volume: if True, create + attach a Vast.ai volume at /workspace
    # for cross-destroy persistence. Default False (container disk is enough
    # for stop/start reuse).
    use_volume: bool = False
    volume_size_gb: int = 200
    stream_logs: bool = True
    download_output: bool = True
    poll_interval: float = 10.0
    startup_timeout: float = 600.0  # seconds to wait for SSH readiness
    # Throughput: auto-tune training args for the rented GPU's VRAM.
    maximize_throughput: bool = True
    # Train from scratch (random init, no checkpoint).
    from_scratch: bool = False
    # Log filtering: grep pattern for live log stream (None = all lines).
    log_filter: str | None = None


# ─── File manifest for incremental sync ────────────────────────────────────
@dataclass
class FileEntry:
    """A single file in the sync manifest."""
    size: int
    mtime: float
    sha256: str  # first 16 hex chars of sha256, for quick comparison

    def to_dict(self) -> dict:
        return {"size": self.size, "mtime": self.mtime, "sha256": self.sha256}

    @staticmethod
    def from_dict(d: dict) -> FileEntry:
        return FileEntry(
            size=int(d.get("size", 0)),
            mtime=float(d.get("mtime", 0)),
            sha256=str(d.get("sha256", "")),
        )


def _compute_file_hash(path: Path) -> str:
    """Compute sha256 of file content, return first 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _compute_local_manifest(local_dir: str,
                             exclude: list[str] | None = None) -> dict[str, dict]:
    """Walk a local directory and return {relative_path: FileEntry.to_dict()}."""
    exclude = exclude or []
    manifest: dict[str, dict] = {}
    local_dir = str(Path(local_dir))
    for root, dirs, files in os.walk(local_dir):
        rel = Path(root).relative_to(local_dir).as_posix()
        dirs[:] = [d for d in dirs
                   if not any(x in d or x in f"{rel}/{d}" for x in exclude)]
        for f in files:
            if any(x in f or x in f"{rel}/{f}" for x in exclude):
                continue
            fp = Path(root) / f
            rel_path = f if rel == "." else f"{rel}/{f}"
            try:
                st = fp.stat()
                manifest[rel_path] = FileEntry(
                    size=st.st_size, mtime=st.st_mtime,
                    sha256=_compute_file_hash(fp),
                ).to_dict()
            except OSError:
                continue
    return manifest


# ─── Connector ─────────────────────────────────────────────────────────────
class VastConnector:
    """Vast.ai control-plane (SDK) + remote execution-plane (paramiko) wrapper.

    All Vast.ai API calls go through the ``vastai.VastAI`` SDK instance.
    All SSH/SFTP goes through a lazily-created :class:`paramiko.SSHClient`
    with keepalive + reconnect-on-drop.
    """

    def __init__(self, api_key: str | None = None, ssh_key: str = ""):
        self.api_key = api_key or os.environ.get("VAST_API_KEY", "")
        self.ssh_key = ssh_key or os.environ.get(
            "VAST_SSH_KEY",
            str(Path.home() / ".ssh" / "id_ed25519"),
        )
        self._ssh = None  # lazy paramiko client
        self._ssh_host: str | None = None
        self._ssh_port: int | None = None
        # Lazily-created SDK client.
        self._vast: Any = None

    @property
    def vast(self) -> Any:
        """Lazily create the vastai.VastAI SDK client."""
        if self._vast is None:
            try:
                from vastai import VastAI
            except ImportError:
                raise RuntimeError(
                    "vastai SDK not found. Install with: pip install vastai"
                )
            kwargs: dict = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._vast = VastAI(**kwargs)
        return self._vast

    # ── Control plane (SDK) ───────────────────────────────────────────────
    def search_offers(self, gpu_filter: str = "",
                      max_price: float = 0.0, min_vram_gb: float = 24.0,
                      min_reliability: float = 0.9, limit: int = 50,
                      sort_by_perf: bool = True) -> list[VastOffer]:
        """Search Vast.ai offers matching the criteria.

        Uses the official ``vast.search_offers()`` SDK method.
        The default query includes ``external=false rentable=true verified=true``,
        so we only add the user's gpu_filter + ``direct_port_count>=1``.

        Args:
            gpu_filter: Vast query string (e.g. "gpu_name=RTX_4090"). Empty
                string = no GPU filter (search all GPUs, auto-select best).
            max_price: Max $/hour. 0 = no per-hour cap.
            min_vram_gb: Minimum GPU VRAM in GiB (checked post-query).
            min_reliability: Minimum host reliability (0..1).
            sort_by_perf: If True, sort by performance-per-dollar (best first).

        Returns filtered + sorted list of VastOffer.
        """
        query_parts = ["direct_port_count>=1", "rentable=true",
                       "cuda_max_good>=12.1", "verified=true"]
        if gpu_filter:
            query_parts.insert(0, gpu_filter)
        query = " ".join(query_parts)
        order = "dlperf_usd-" if sort_by_perf else "dph_total"
        rows = self.vast.search_offers(
            query=query, order=order, limit=limit, no_default=False,
        )
        offers: list[VastOffer] = []
        for r in rows:
            try:
                dph = float(r.get("dph_total", r.get("dph", 0)) or 0)
                if dph <= 0:
                    continue
                if max_price > 0 and dph > max_price:
                    continue
                # VRAM: API returns MB. 24GB GPU = 24564 MB.
                # Use /1000 (decimal GB, matching Vast.ai's convention) not /1024.
                # A 24GB 4090 reports gpu_total_ram=24564 → 24.56 GB (decimal),
                # but /1024 gives 23.99 GiB which wrongly fails a 24GB filter.
                vram_mb = float(r.get("gpu_total_ram",
                                       float(r.get("gpu_ram", 0) or 0) *
                                       int(r.get("num_gpus", 1))))
                vram = vram_mb / 1000.0  # MB → GB (decimal, matches Vast.ai)
                if vram < min_vram_gb:
                    continue
                rel = float(r.get("reliability", 0) or 0)
                if rel < min_reliability:
                    continue
                dlperf = float(r.get("dlperf", 0) or 0)
                offers.append(VastOffer(
                    id=int(r["id"]),
                    gpu_name=str(r.get("gpu_name", "?")),
                    num_gpus=int(r.get("num_gpus", 1)),
                    vram_gb=vram,
                    dph_total=dph,
                    reliability=rel,
                    inet_down=float(r.get("inet_down", 0) or 0),
                    inet_up=float(r.get("inet_up", 0) or 0),
                    dlperf=dlperf,
                    disk_gb=float(r.get("disk_space", 0) or 0),
                    raw=r,
                ))
            except (KeyError, ValueError, TypeError):
                continue
        if sort_by_perf:
            offers.sort(key=lambda o: o.perf_per_dollar, reverse=True)
        else:
            offers.sort(key=lambda o: o.dph_total)
        return offers

    def estimate_training_hours(self, spec: RemoteTrainingSpec) -> float:
        """Rough estimate of training wall-clock hours from max_steps."""
        max_steps = spec.train_args.get("--max-steps", 500)
        try:
            max_steps = int(max_steps)
        except (ValueError, TypeError):
            max_steps = 500
        return max_steps * spec.est_sec_per_step / 3600.0

    def select_best_offer(self, spec: RemoteTrainingSpec) -> VastOffer | None:
        """Auto-select the best GPU by performance-per-dollar within budget."""
        est_hours = self.estimate_training_hours(spec)
        max_dph = spec.budget / max(est_hours, 0.01) if spec.budget > 0 else 0.0
        logger.info(
            f"Budget: ${spec.budget:.2f} | est {est_hours:.2f}h "
            f"({spec.train_args.get('--max-steps', '?')} steps × "
            f"{spec.est_sec_per_step}s) | max ${max_dph:.3f}/h"
        )
        offers = self.search_offers(
            gpu_filter=spec.gpu_filter,
            max_price=max_dph if max_dph > 0 else spec.max_price,
            min_vram_gb=spec.min_vram_gb,
            min_reliability=spec.min_reliability,
            sort_by_perf=True,
        )
        if offers:
            logger.info(f"Found {len(offers)} offers within budget. "
                        f"Best: {offers[0].label}")
            return offers[0]
        logger.warning(f"No offers within ${max_dph:.3f}/h budget. "
                       f"Relaxing price cap...")
        offers = self.search_offers(
            gpu_filter=spec.gpu_filter, max_price=0.0,
            min_vram_gb=spec.min_vram_gb,
            min_reliability=spec.min_reliability, sort_by_perf=False,
        )
        if offers:
            est_cost = offers[0].dph_total * est_hours
            logger.warning(f"Cheapest viable: {offers[0].label} | "
                           f"est cost ${est_cost:.2f} (budget ${spec.budget:.2f})")
            if est_cost > spec.budget:
                logger.warning(f"WARNING: est cost ${est_cost:.2f} exceeds "
                               f"budget ${spec.budget:.2f}.")
            return offers[0]
        return None

    def create_instance(self, offer_id: int, image: str = DEFAULT_IMAGE,
                        disk_gb: int = 100, on_demand: bool = True,
                        ssh_pubkey: str = "", label: str = "",
                        volume_id: int | None = None) -> int:
        """Rent an offer. Returns the new instance (contract) id.

        Uses the SDK's ``create_instance()`` with ``runtype="ssh_direct"``
        for SSH + direct connections. If ``ssh_pubkey`` is provided, injects
        it via ``onstart_cmd`` so the key is authorized without needing
        ``vastai create ssh-key`` (which requires 2FA on some accounts).
        """
        onstart_parts: list[str] = []
        if ssh_pubkey:
            safe_key = ssh_pubkey.replace("'", "'\\''")
            onstart_parts.append(
                f"mkdir -p ~/.ssh && echo '{safe_key}' >> ~/.ssh/authorized_keys "
                f"&& chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
            )
        kwargs: dict = {
            "image": image,
            "disk": disk_gb,
            "runtype": "ssh_direct",
        }
        if label:
            kwargs["label"] = label
        if onstart_parts:
            kwargs["onstart_cmd"] = " && ".join(onstart_parts)
        if not on_demand:
            kwargs["price"] = 0.0  # bid_price for interruptible
        if volume_id:
            kwargs["volume_info"] = {
                "volume_id": volume_id,
                "mount_path": "/workspace",
            }
        result = self.vast.create_instance(id=offer_id, **kwargs)
        new_id = result.get("new_contract") or result.get("id")
        if new_id is None:
            raise RuntimeError(f"Could not parse instance id from: {result}")
        return int(new_id)

    def list_instances(self) -> list[dict]:
        """List current Vast.ai instances (all states)."""
        rows = self.vast.show_instances()
        return rows if isinstance(rows, list) else []

    def get_instance(self, instance_id: int) -> dict | None:
        """Get a single instance by ID via SDK."""
        try:
            inst = self.vast.show_instance(id=instance_id)
            if isinstance(inst, list):
                return inst[0] if inst else None
            return inst
        except Exception:
            return None

    def destroy_instance(self, instance_id: int) -> None:
        """Destroy (permanently delete) an instance and free billing."""
        self.vast.destroy_instance(id=instance_id)
        logger.info(f"Destroyed Vast instance {instance_id}")

    def stop_instance(self, instance_id: int) -> None:
        """Stop an instance (preserves disk, stops GPU charges)."""
        self.vast.stop_instance(id=instance_id)
        logger.info(f"Stopped Vast instance {instance_id} (disk preserved)")

    def start_instance(self, instance_id: int) -> None:
        """Start a stopped instance."""
        self.vast.start_instance(id=instance_id)
        logger.info(f"Started Vast instance {instance_id}")

    def label_instance(self, instance_id: int, label: str) -> None:
        """Tag an instance with a label for reuse discovery."""
        self.vast.label_instance(id=instance_id, label=label)

    def find_instance_by_label(self, label: str) -> dict | None:
        """Find an instance (any state) by its label.

        Returns the instance dict if found, None otherwise.
        """
        for inst in self.list_instances():
            if inst.get("label") == label:
                return inst
        return None

    def get_ssh_info(self, instance_id: int) -> tuple[str, int]:
        """Return (host, port) for SSH to the instance.

        Prefers the **direct port mapping** (container port 22 → host port)
        from the ``ports`` field, connecting to the machine's public IP
        directly. Falls back to the Vast SSH proxy (``ssh_url()``) if the
        direct mapping is not available.

        The direct port is more reliable — the Vast proxy can have
        timeouts and connection refused issues, especially on fresh
        instances where the proxy hasn't fully registered yet.
        """
        inst = self.get_instance(instance_id)
        if inst:
            # ── Preferred: direct port mapping ──
            # ports field format: {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "17573"}]}
            ports = inst.get("ports") or {}
            ip = inst.get("public_ipaddr")
            for cport, mappings in ports.items():
                if "22" in cport and mappings:
                    host_port = mappings[0].get("HostPort")
                    if ip and host_port:
                        return str(ip), int(host_port)
            # ── Fallback: Vast SSH proxy ──
            try:
                url = self.vast.ssh_url(id=instance_id)
                if url:
                    m = re.search(r"@([\w.-]+):(\d+)", url)
                    if m:
                        return m.group(1), int(m.group(2))
            except Exception:
                pass
            # ── Last resort: ssh_host + ssh_port from instance ──
            host = inst.get("ssh_host") or inst.get("public_ipaddr")
            port = inst.get("ssh_port")
            if host and port:
                return str(host), int(port)
        raise RuntimeError(
            f"Could not determine SSH info for instance {instance_id}."
        )

    def get_logs(self, instance_id: int, tail: int = 1000,
                 filter_str: str | None = None) -> str:
        """Retrieve historical container logs via SDK.

        Uses ``vast.logs()`` which requests logs from S3 and polls until
        ready. Useful for debugging after a disconnect, or reviewing past
        training output without re-connecting via SSH.
        """
        return self.vast.logs(
            instance_id=instance_id,
            tail=str(tail) if tail else None,
            filter=filter_str,
        )

    # ── Volume management (SDK) ───────────────────────────────────────────
    def show_volumes(self) -> list[dict]:
        """List all owned volumes."""
        return self.vast.show_volumes()

    def create_volume(self, offer_id: int, size_gb: int = 200,
                      name: str = "forgeai-persistent") -> int:
        """Create a persistent volume. Returns the volume ID.

        The volume must be created from a volume offer on the same machine
        as the target instance. Use ``search volumes`` to find offers.
        """
        result = self.vast.create_volume(id=offer_id, size=size_gb, name=name)
        vol_id = result.get("id") or result.get("volume_id")
        if vol_id is None:
            raise RuntimeError(f"Could not parse volume id from: {result}")
        return int(vol_id)

    # ── Instance lifecycle (persistent reuse) ─────────────────────────────
    def _instance_label(self, spec: RemoteTrainingSpec) -> str:
        """Generate a deterministic label for instance reuse."""
        config = spec.train_args.get("--config", "default")
        gpu = spec.gpu_filter or "auto"
        # Sanitize for label use (max 1024 chars, no special chars needed).
        return f"{INSTANCE_LABEL_PREFIX}-{config}-{gpu}"

    def ensure_instance(self, spec: RemoteTrainingSpec) -> tuple[int, bool]:
        """Find or create an instance for this training spec.

        **Enforces a hard single-instance cap**: at most ONE instance may
        exist at any time. If multiple instances are found, extras are
        destroyed and the first viable one is reused.

        Returns (instance_id, is_new). If ``reuse_instance`` is True,
        looks for any existing instance (preferably with our label, but
        will reuse any live instance to avoid spawning a second).

        - Found + running: reuse as-is.
        - Found + stopped: start it, wait for running.
        - Found + in poll-trap state: destroy + create new.
        - Not found: create new with label.
        """
        # ── Single-instance enforcement: list all, destroy extras ──
        all_insts = self.list_instances()
        if len(all_insts) > 1:
            logger.warning(
                f"Found {len(all_insts)} instances — enforcing single-instance "
                f"cap. Destroying extras, keeping the first viable one."
            )
            # Keep the first non-trapped instance; destroy the rest.
            keep_id: int | None = None
            for inst in all_insts:
                status = str(inst.get("actual_status", "")).lower()
                if status not in POLL_TRAP_STATES:
                    keep_id = int(inst.get("id", -1))
                    break
            for inst in all_insts:
                iid = int(inst.get("id", -1))
                if iid != keep_id:
                    try:
                        self.destroy_instance(iid)
                        logger.info(f"  Destroyed extra instance #{iid}")
                    except Exception as e:
                        logger.warning(f"  Failed to destroy #{iid}: {e}")
            all_insts = [i for i in all_insts if int(i.get("id", -1)) == keep_id]

        if not spec.reuse_instance:
            # Even with reuse=False, destroy any existing instance first
            # to maintain the single-instance cap.
            for inst in all_insts:
                iid = int(inst.get("id", -1))
                try:
                    self.destroy_instance(iid)
                    logger.info(f"Destroyed existing #{iid} (reuse=False)")
                except Exception as e:
                    logger.warning(f"Failed to destroy #{iid}: {e}")
            return self._create_new_instance(spec), True

        # ── Reuse logic: prefer label match, fall back to any instance ──
        label = self._instance_label(spec)
        existing = None
        if all_insts:
            # Prefer instance with our label, else take the first.
            for inst in all_insts:
                if inst.get("label") == label:
                    existing = inst
                    break
            if existing is None:
                existing = all_insts[0]
                logger.info(
                    f"Reusing instance #{existing.get('id')} "
                    f"(label mismatch: have '{existing.get('label','')}', "
                    f"want '{label}') — single-instance cap."
                )

        if existing:
            inst_id = int(existing.get("id", -1))
            status = str(existing.get("actual_status", "")).lower()
            cur = str(existing.get("cur_state", "")).lower()
            if status in POLL_TRAP_STATES or cur in POLL_TRAP_STATES:
                logger.warning(
                    f"Existing instance {inst_id} in poll-trap state "
                    f"'{status}' — destroying and creating new."
                )
                try:
                    self.destroy_instance(inst_id)
                except Exception as e:
                    logger.warning(f"Failed to destroy trapped instance: {e}")
                return self._create_new_instance(spec, label=label), True
            if status == "running" or cur == "running":
                logger.info(f"Reusing running instance {inst_id} (label={label})")
                return inst_id, False
            if status == "stopped" or cur == "stopped":
                logger.info(f"Starting stopped instance {inst_id} (label={label})")
                self.start_instance(inst_id)
                return inst_id, False
            logger.info(f"Reusing instance {inst_id} in state '{status}'")
            return inst_id, False
        # No existing instance — create new.
        return self._create_new_instance(spec, label=label), True

    def _create_new_instance(self, spec: RemoteTrainingSpec,
                             label: str = "") -> int:
        """Select best offer, create instance, label it."""
        if not label:
            label = self._instance_label(spec)
        chosen = self.select_best_offer(spec)
        if chosen is None:
            raise RuntimeError(
                f"No Vast offers match: filter={spec.gpu_filter or '(any)'} "
                f"min_vram={spec.min_vram_gb}GB "
                f"min_reliability={spec.min_reliability}"
            )
        logger.info(f"Renting best perf/$ offer: {chosen.label}")
        self._auto_tune_for_gpu(spec, chosen.vram_gb, chosen.num_gpus)
        ssh_pubkey = self._read_ssh_pubkey()
        inst_id = self.create_instance(
            chosen.id, image=spec.image, disk_gb=spec.disk_gb,
            on_demand=spec.on_demand, ssh_pubkey=ssh_pubkey,
            label=label,
        )
        logger.info(f"Instance created: {inst_id} (label={label})")
        return inst_id

    def _read_ssh_pubkey(self) -> str:
        """Read the SSH public key for injection at instance creation."""
        pub_path = self.ssh_key + ".pub"
        if not Path(pub_path).exists():
            pub_path = str(Path(self.ssh_key).with_suffix(".pub"))
        if Path(pub_path).exists():
            try:
                return Path(pub_path).read_text().strip()
            except Exception:
                pass
        return ""

    def wait_for_running(self, instance_id: int, timeout: float = 600.0,
                         poll: float = 10.0) -> dict:
        """Poll until the instance is ``running`` and SSH-reachable.

        Handles the poll-trap: if ``actual_status`` becomes ``exited``,
        ``unknown``, or ``offline``, raises RuntimeError immediately
        (these states never recover — per Vast.ai docs).

        Vast.ai status fields:
        - ``actual_status``: container-level (none, loading, running, exited)
        - ``cur_state``: instance-level (running, stopped, error)
        During image pull, ``actual_status='none'`` or ``'loading'`` while
        ``cur_state`` may already be ``'running'``. We only attempt SSH
        when ``actual_status`` is ``'running'`` (container fully started).
        """
        deadline = time.time() + timeout
        last = None
        ssh_attempts = 0
        while time.time() < deadline:
            inst = self.get_instance(instance_id)
            if inst is None:
                raise RuntimeError(f"Instance {instance_id} disappeared")
            last = inst
            actual = str(inst.get("actual_status", "")).lower()
            cur = str(inst.get("cur_state", "")).lower()
            logger.info(f"Instance {instance_id}: actual={actual!r} cur={cur!r}")
            # Poll-trap: these states never recover.
            if actual in POLL_TRAP_STATES or cur in POLL_TRAP_STATES:
                raise RuntimeError(
                    f"Instance {instance_id} entered poll-trap state "
                    f"'{actual or cur}'. Destroy and retry with a different offer."
                )
            # Only attempt SSH when the container is actually running
            # (not just 'loading' or 'none' even if cur_state='running').
            if actual in ("running", "executing"):
                try:
                    host, port = self.get_ssh_info(instance_id)
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(5.0)
                    s.connect((host, port))
                    s.close()
                    logger.info(f"SSH port {host}:{port} is open")
                    return inst
                except Exception as e:
                    ssh_attempts += 1
                    logger.info(f"SSH not ready yet (attempt {ssh_attempts}): {e}")
            time.sleep(poll)
        raise TimeoutError(
            f"Instance {instance_id} not running after {timeout}s. "
            f"Last: actual={last.get('actual_status')} "
            f"cur={last.get('cur_state') if last else '?'}"
        )

    # ── SSH / SFTP plane (paramiko) ───────────────────────────────────────
    def _ssh_connect(self, host: str, port: int, retries: int = 6,
                     retry_delay: float = 10.0) -> Any:
        """Connect via SSH with retries + keepalive.

        If an existing connection to the same host:port is alive, reuses it.
        Otherwise creates a new connection with keepalive enabled.
        """
        if self._ssh is not None and self._ssh_host == host \
                and self._ssh_port == port:
            transport = self._ssh.get_transport()
            if transport and transport.is_active():
                return self._ssh
            # Connection died — close and reconnect.
            self.ssh_disconnect()
        import paramiko  # lazy import
        key_path = self.ssh_key
        if not Path(key_path).exists():
            raise FileNotFoundError(
                f"SSH private key not found: {key_path}. "
                f"Generate one with ssh-keygen and register its .pub with "
                f"`vastai create ssh-key <pub>`."
            )
        last_err = None
        for attempt in range(retries):
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(host, port=port, username="root",
                               key_filename=key_path, timeout=30.0,
                               banner_timeout=30.0)
                # Enable keepalive: send a packet every 30s to detect drops.
                transport = client.get_transport()
                if transport:
                    transport.set_keepalive(30)
                self._ssh = client
                self._ssh_host = host
                self._ssh_port = port
                return client
            except Exception as e:
                last_err = e
                logger.debug(f"SSH attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
        raise RuntimeError(f"SSH connection to {host}:{port} failed after "
                           f"{retries} attempts: {last_err}")

    def ssh_disconnect(self) -> None:
        if self._ssh is not None:
            self._ssh.close()
            self._ssh = None
            self._ssh_host = None
            self._ssh_port = None

    def exec_remote(self, host: str, port: int, cmd: str,
                    timeout: float | None = None) -> tuple[int, str, str]:
        """Run a command on the remote box, return (rc, stdout, stderr).

        Reconnects automatically if the SSH connection has dropped.
        """
        client = self._ssh_connect(host, port)
        _stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        rc = stdout.channel.recv_exit_status()
        return rc, stdout.read().decode("utf-8", "replace"), \
            stderr.read().decode("utf-8", "replace")

    def stream_remote(self, host: str, port: int, cmd: str,
                      reconnect: bool = True) -> Iterator[str]:
        """Run a long command and yield stdout lines as they arrive.

        Used for streaming training logs back to the local terminal.
        If ``reconnect`` is True and the connection drops mid-stream,
        attempts to reconnect and resume (the command is re-run; output
        may have gaps but the stream continues).
        """
        while True:
            try:
                client = self._ssh_connect(host, port)
                transport = client.get_transport()
                if not transport:
                    raise RuntimeError("No SSH transport")
                chan = transport.open_session()
                chan.get_pty()
                chan.exec_command(cmd)
                buf = ""
                while not chan.exit_status_ready():
                    while chan.recv_ready():
                        buf += chan.recv(4096).decode("utf-8", "replace")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            yield line
                    while chan.recv_stderr_ready():
                        yield chan.recv_stderr(4096).decode("utf-8", "replace").rstrip("\n")
                    time.sleep(0.2)
                # flush trailing
                buf += chan.recv(65536).decode("utf-8", "replace")
                for line in buf.split("\n"):
                    if line:
                        yield line
                return  # normal completion
            except Exception as e:
                if not reconnect:
                    raise
                logger.warning(f"SSH stream dropped: {e}. Reconnecting in 5s...")
                self.ssh_disconnect()
                time.sleep(5.0)
                # Re-run the command. The remote process may still be running
                # (we'll get its continued output) or may have finished (we'll
                # get the exit marker immediately).
                logger.info("Resuming log stream after reconnect...")

    def upload_file(self, host: str, port: int, local: str, remote: str,
                    show_progress: bool = False) -> None:
        """Upload a single file over SFTP. Creates remote parent dirs."""
        client = self._ssh_connect(host, port)
        sftp = client.open_sftp()
        try:
            self._ensure_remote_dir(sftp, str(PurePosixPath(remote).parent))
            sftp.put(local, remote,
                     callback=self._progress_cb if show_progress else None)
        finally:
            sftp.close()

    def upload_rsync(self, host: str, port: int,
                     local_path: str, remote_path: str,
                     ssh_key: str | None = None) -> int:
        """Upload a file via rsync or scp over SSH (much faster than SFTP).

        rsync uses a streaming protocol that saturates the network link,
        while SFTP has high per-packet overhead. For a 95 MB zip, rsync
        is typically 10-50x faster than paramiko SFTP.

        Tries rsync first (best), then scp (good, available on Windows
        via OpenSSH), then falls back to SFTP (slow).

        Returns: bytes transferred, or -1 if fell back to SFTP.
        """
        import shutil
        import subprocess
        key = ssh_key or self.ssh_key
        key_arg = f"-i {key}"
        ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

        # ── Try rsync first (fastest) ──
        if shutil.which("rsync"):
            rsync_cmd = [
                "rsync", "-az", "--progress",
                "-e", f"ssh {key_arg} -p {port} {ssh_opts}",
                local_path,
                f"root@{host}:{remote_path}",
            ]
            try:
                result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    size = Path(local_path).stat().st_size
                    logger.info(f"rsync complete: {size / 1e6:.1f} MB uploaded")
                    return size
                logger.warning(f"rsync rc={result.returncode}: {result.stderr[:200]}")
            except (subprocess.TimeoutExpired, Exception) as e:
                logger.warning(f"rsync failed: {e}")

        # ── Try scp (available on Windows via OpenSSH) ──
        if shutil.which("scp"):
            scp_cmd = [
                "scp", "-i", key, "-P", str(port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                local_path,
                f"root@{host}:{remote_path}",
            ]
            try:
                result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    size = Path(local_path).stat().st_size
                    logger.info(f"scp complete: {size / 1e6:.1f} MB uploaded")
                    return size
                logger.warning(f"scp rc={result.returncode}: {result.stderr[:200]}")
            except (subprocess.TimeoutExpired, Exception) as e:
                logger.warning(f"scp failed: {e}")

        # ── Fallback: SFTP (slow but always works) ──
        logger.info("Falling back to SFTP...")
        self.upload_file(host, port, local_path, remote_path, show_progress=True)
        return -1

    def upload_and_unzip(self, host: str, port: int,
                         local_paths: list[str],
                         remote_dest: str,
                         zip_name: str = "upload.zip",
                         compression: int = 6) -> int:
        """Zip local paths, upload the zip, unzip on remote.

        Much faster than uploading many small files individually:
        - Single SFTP transfer (one TCP stream, no per-file overhead)
        - Compressed (JSONL text compresses ~5-10x)
        - Remote unzip is near-instant (CPU, not network-bound)

        Args:
            local_paths: List of local files/dirs to include in the zip.
            remote_dest: Remote directory to unzip into (created if missing).
            zip_name: Name of the zip file on remote (under remote_dest).
            compression: zipfile compression level (0=store, 6=default, 9=max).

        Returns:
            Size of the uploaded zip in bytes.
        """
        import tempfile
        import zipfile

        # 1. Create zip locally
        tmp_zip = Path(tempfile.gettempdir()) / zip_name
        logger.info(f"Zipping {len(local_paths)} paths -> {tmp_zip.name}...")
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=compression) as zf:
            for local_path in local_paths:
                p = Path(local_path)
                if p.is_file():
                    # Use just the filename as arcname (flat, no dirs).
                    # The remote_dest is the target directory.
                    zf.write(p, p.name)
                    logger.debug(f"  + {p.name} ({p.stat().st_size} bytes)")
                elif p.is_dir():
                    for root, dirs, files in os.walk(p):
                        rel = Path(root).relative_to(p).as_posix()
                        dirs[:] = [d for d in dirs
                                   if d not in ("__pycache__", ".git")]
                        for f in files:
                            fp = Path(root) / f
                            arcname = (f"{rel}/{f}" if rel != "."
                                       else f)
                            zf.write(fp, arcname)
        zip_size = tmp_zip.stat().st_size
        total_uncompressed = sum(
            Path(lp).stat().st_size if Path(lp).is_file()
            else sum(f.stat().st_size for _, _, fs in os.walk(lp) for f in
                     [Path(_) / fn for fn in fs])
            for lp in local_paths
        )
        ratio = zip_size / max(total_uncompressed, 1) * 100
        logger.info(f"Zip: {zip_size / 1e6:.1f} MB "
                    f"(from {total_uncompressed / 1e6:.1f} MB, {ratio:.0f}%)")

        # 2. Upload zip via rsync (fast) or SFTP (fallback)
        remote_zip = f"{remote_dest}/{zip_name}"
        logger.info(f"Uploading {zip_name} ({zip_size / 1e6:.1f} MB)...")
        self.upload_rsync(host, port, str(tmp_zip), remote_zip)

        # 3. Unzip on remote
        logger.info(f"Unzipping on remote → {remote_dest}...")
        rc, out, err = self.exec_remote(
            host, port,
            f"cd {remote_dest} && unzip -o {zip_name} && rm {zip_name}",
            timeout=120.0,
        )
        if rc != 0:
            # Fall back to python unzip if `unzip` command not available
            logger.warning(f"unzip command failed (rc={rc}), trying python...")
            rc, out, err = self.exec_remote(
                host, port,
                f"python3 -c \"import zipfile; zipfile.ZipFile('{remote_zip}').extractall('{remote_dest}')\" && rm {remote_zip}",
                timeout=120.0,
            )
            if rc != 0:
                raise RuntimeError(
                    f"Remote unzip failed (rc={rc}): {err or out}"
                )
        logger.info("Unzip complete.")
        # Cleanup local temp zip
        with contextlib.suppress(OSError):
            tmp_zip.unlink()
        return zip_size

    def upload_dir(self, host: str, port: int, local_dir: str,
                   remote_dir: str, exclude: list[str] | None = None) -> None:
        """Recursively upload a directory over SFTP (full, non-incremental).

        For incremental sync (only changed files), use :meth:`sync_dir`.
        """
        exclude = exclude or ["__pycache__", ".git", ".pytest_cache",
                              "node_modules", ".venv", "venv"]
        client = self._ssh_connect(host, port)
        sftp = client.open_sftp()
        local_dir = str(Path(local_dir))
        try:
            self._ensure_remote_dir(sftp, remote_dir)
            for root, dirs, files in os.walk(local_dir):
                rel = Path(root).relative_to(local_dir).as_posix()
                dirs[:] = [d for d in dirs
                           if not any(x in d or x in f"{rel}/{d}" for x in exclude)]
                rdir = remote_dir if rel == "." else f"{remote_dir}/{rel}"
                self._ensure_remote_dir(sftp, rdir)
                for f in files:
                    if any(x in f or x in f"{rel}/{f}" for x in exclude):
                        continue
                    sftp.put(str(Path(root) / f), f"{rdir}/{f}")
        finally:
            sftp.close()

    def sync_critical_files(self, host: str, port: int) -> tuple[int, int]:
        """Sync only the critical source files needed for training.

        Instead of uploading the entire ``research/`` tree (427+ files),
        uploads only the ~53 files in :data:`CRITICAL_SOURCE_FILES` +
        :data:`CRITICAL_INIT_FILES` — the minimal set traced from
        sft_train.py's import tree for the forgelm_v2 config.

        Uses manifest-based incremental sync: only uploads files that
        changed since the last sync (sha256 + mtime comparison).

        Returns (num_uploaded, num_skipped).
        """
        all_files = CRITICAL_SOURCE_FILES + CRITICAL_INIT_FILES
        # Build local manifest for just these files.
        local_manifest: dict[str, dict] = {}
        for rel_path in all_files:
            local_path = PROJECT_ROOT / rel_path
            if not local_path.exists():
                logger.debug(f"  Skip (not found locally): {rel_path}")
                continue
            try:
                st = local_path.stat()
                local_manifest[rel_path] = FileEntry(
                    size=st.st_size, mtime=st.st_mtime,
                    sha256=_compute_file_hash(local_path),
                ).to_dict()
            except OSError:
                continue
        # Fetch remote manifest.
        remote_manifest = self._get_remote_manifest(host, port, REMOTE_REPO)
        # Determine changed files.
        to_upload: list[tuple[str, str, str]] = []  # (rel, local, remote)
        skipped = 0
        for rel_path, local_entry in local_manifest.items():
            remote_entry = remote_manifest.get(rel_path)
            if remote_entry and remote_entry == local_entry:
                skipped += 1
                continue
            local_path = str(PROJECT_ROOT / rel_path)
            remote_path = f"{REMOTE_REPO}/{rel_path}"
            to_upload.append((rel_path, local_path, remote_path))
        if not to_upload:
            logger.info(f"Sync: all {len(local_manifest)} critical files "
                        f"up-to-date (0 uploaded, {skipped} skipped)")
            return 0, skipped
        logger.info(f"Sync: {len(to_upload)} changed, {skipped} unchanged "
                    f"of {len(local_manifest)} critical files")
        # Upload changed files.
        # Use shell `mkdir -p` for all needed dirs upfront (more reliable
        # than SFTP mkdir which can fail silently on some servers).
        remote_dirs = set()
        for rel_path, _, remote_path in to_upload:
            remote_dirs.add(str(PurePosixPath(remote_path).parent))
        mkdir_cmd = "mkdir -p " + " ".join(
            f'"{d}"' for d in sorted(remote_dirs)
        )
        rc, _out, err = self.exec_remote(host, port, mkdir_cmd, timeout=30.0)
        logger.info(f"mkdir -p {len(remote_dirs)} dirs: rc={rc}")
        if rc != 0:
            logger.warning(f"mkdir -p failed (rc={rc}): {err}")
        else:
            # Verify dirs exist
            rc2, out2, _ = self.exec_remote(
                host, port,
                f"ls -d {REMOTE_REPO}/research/ 2>&1",
                timeout=10.0,
            )
            logger.info(f"Verify {REMOTE_REPO}/research/: rc={rc2} out={out2.strip()}")
        # Now upload files via SFTP.
        client = self._ssh_connect(host, port)
        sftp = client.open_sftp()
        try:
            for rel_path, local_path, remote_path in to_upload:
                logger.info(f"  Uploading {rel_path}")
                try:
                    sftp.put(local_path, remote_path)
                except OSError as e:
                    logger.error(f"  Failed: {rel_path}: {e}")
                    raise
        finally:
            sftp.close()
        # Update remote manifest.
        self._put_remote_manifest(host, port, REMOTE_REPO, local_manifest)
        logger.info(f"Sync complete: {len(to_upload)} uploaded, {skipped} skipped")
        return len(to_upload), skipped

    def sync_dir(self, host: str, port: int, local_dir: str,
                 remote_dir: str,
                 exclude: list[str] | None = None) -> tuple[int, int]:
        """Incrementally sync a directory: only upload changed files.

        Computes a local manifest (path → size + mtime + sha256), compares
        with the remote manifest (cached on the instance), and uploads only
        files that differ. After upload, updates the remote manifest.

        Returns (num_uploaded, num_skipped).
        """
        exclude = exclude or ["__pycache__", ".git", ".pytest_cache",
                              "node_modules", ".venv", "venv", ".tmp"]
        local_manifest = _compute_local_manifest(local_dir, exclude)
        # Fetch remote manifest from the instance.
        remote_manifest = self._get_remote_manifest(host, port, remote_dir)
        # Determine which files need uploading.
        to_upload: list[tuple[str, str]] = []  # (local_path, remote_path)
        skipped = 0
        for rel_path, local_entry in local_manifest.items():
            remote_entry = remote_manifest.get(rel_path)
            if remote_entry and remote_entry == local_entry:
                skipped += 1
                continue
            local_path = str(Path(local_dir) / rel_path)
            remote_path = f"{remote_dir}/{rel_path}"
            to_upload.append((local_path, remote_path))
        # Also detect deleted files (on remote but not local).
        deleted = [rp for rp in remote_manifest if rp not in local_manifest]
        if not to_upload and not deleted:
            logger.info(f"Sync: all {len(local_manifest)} files up-to-date "
                        f"(0 uploaded, {skipped} skipped)")
            return 0, skipped
        logger.info(f"Sync: {len(to_upload)} changed, {skipped} unchanged, "
                    f"{len(deleted)} deleted")
        # Upload changed files.
        client = self._ssh_connect(host, port)
        sftp = client.open_sftp()
        try:
            # Ensure the top-level remote_dir exists first (fresh instance).
            self._ensure_remote_dir(sftp, remote_dir)
            for local_path, remote_path in to_upload:
                rel = Path(remote_path).relative_to(remote_dir).as_posix() \
                    if remote_path.startswith(remote_dir) else Path(local_path).name
                logger.info(f"  Uploading {rel}")
                self._ensure_remote_dir(sftp, str(PurePosixPath(remote_path).parent))
                try:
                    sftp.put(local_path, remote_path)
                except OSError as e:
                    logger.error(f"  Failed to upload {rel}: {e} "
                                 f"(local={local_path} remote={remote_path})")
                    raise
            # Delete removed files.
            for rel_path in deleted:
                remote_path = f"{remote_dir}/{rel_path}"
                try:
                    sftp.remove(remote_path)
                    logger.debug(f"  Deleted {rel_path}")
                except OSError:
                    pass  # already gone
        finally:
            sftp.close()
        # Update remote manifest.
        self._put_remote_manifest(host, port, remote_dir, local_manifest)
        logger.info(f"Sync complete: {len(to_upload)} uploaded, {skipped} skipped, "
                    f"{len(deleted)} deleted")
        return len(to_upload), skipped

    def _get_remote_manifest(self, host: str, port: int,
                             remote_dir: str) -> dict:
        """Fetch the remote file manifest (cached JSON on the instance)."""
        manifest_path = f"{remote_dir}/.file_manifest.json"
        try:
            client = self._ssh_connect(host, port)
            sftp = client.open_sftp()
            try:
                with sftp.open(manifest_path, "r") as f:
                    data = f.read()
                    if isinstance(data, bytes):
                        data = data.decode("utf-8", "replace")
                    elif not isinstance(data, str):
                        # Mock or unexpected type — treat as empty.
                        return {}
                return json.loads(data) if data else {}
            finally:
                sftp.close()
        except (OSError, json.JSONDecodeError):
            return {}  # no manifest yet — first sync

    def _put_remote_manifest(self, host: str, port: int,
                             remote_dir: str, manifest: dict) -> None:
        """Write the file manifest to the instance for next sync."""
        manifest_path = f"{remote_dir}/.file_manifest.json"
        client = self._ssh_connect(host, port)
        sftp = client.open_sftp()
        try:
            self._ensure_remote_dir(sftp, remote_dir)
            with sftp.open(manifest_path, "w") as f:
                f.write(json.dumps(manifest))
        finally:
            sftp.close()

    def download_file(self, host: str, port: int, remote: str, local: str,
                      show_progress: bool = True) -> None:
        """Download a single file (e.g. the trained checkpoint)."""
        client = self._ssh_connect(host, port)
        sftp = client.open_sftp()
        try:
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, local,
                     callback=self._progress_cb if show_progress else None)
        finally:
            sftp.close()

    def download_dir(self, host: str, port: int, remote_dir: str,
                     local_dir: str) -> None:
        """Recursively download a directory (e.g. LoRA adapter dir)."""
        client = self._ssh_connect(host, port)
        sftp = client.open_sftp()
        try:
            self._download_dir_recursive(sftp, remote_dir, local_dir)
        finally:
            sftp.close()

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _progress_cb(transferred: int, total: int) -> None:
        pct = 100.0 * transferred / max(total, 1)
        sys.stdout.write(f"\r  SFTP {transferred}/{total} bytes ({pct:.1f}%)")
        sys.stdout.flush()
        if transferred >= total:
            sys.stdout.write("\n")

    @staticmethod
    def _ensure_remote_dir(sftp, remote_path: str) -> None:
        """mkdir -p over SFTP (paramiko has no recursive mkdir).

        Walks each path component, stat-checks it, and mkdir if missing.
        Uses a recursive approach: if mkdir fails, ensure parent first
        (handles the case where intermediate dirs don't exist yet).
        """
        parts = [p for p in remote_path.split("/") if p]
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}"
            try:
                sftp.stat(cur)
            except OSError:
                # Dir doesn't exist — try to create it.
                # If mkdir fails (parent missing), recurse to create parent
                # first, then retry.
                try:
                    sftp.mkdir(cur)
                except OSError:
                    parent = "/".join(cur.split("/")[:-1])
                    if parent:
                        VastConnector._ensure_remote_dir(sftp, parent)
                        sftp.mkdir(cur)

    def _download_dir_recursive(self, sftp, remote_dir: str, local_dir: str) -> None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dir):
            rpath = f"{remote_dir}/{entry.filename}"
            lpath = str(Path(local_dir) / entry.filename)
            try:
                sftp.listdir(rpath)
                self._download_dir_recursive(sftp, rpath, lpath)
            except OSError:
                sftp.get(rpath, lpath)

    # ── Log parsing + formatting ──────────────────────────────────────────
    @staticmethod
    def _parse_log_line(line: str) -> dict:
        """Parse a training log line into structured components.

        Detects:
        - Step/loss/metrics: "step 42 loss 3.14 lr 1e-4"
        - Log levels: ERROR/WARNING/INFO/DEBUG
        - Progress bars: lines with % or it/s
        - Exit markers: __FORGE_EXIT__:N
        """
        result: dict = {"raw": line, "level": "INFO", "type": "log"}
        # Exit marker
        m = re.match(r"__FORGE_EXIT__:(-?\d+)", line)
        if m:
            result["type"] = "exit"
            result["exit_code"] = int(m.group(1))
            return result
        # Log level detection
        upper = line.upper()
        if "ERROR" in upper or "TRACEBACK" in upper or "EXCEPTION" in upper:
            result["level"] = "ERROR"
        elif "WARN" in upper:
            result["level"] = "WARNING"
        elif "DEBUG" in upper:
            result["level"] = "DEBUG"
        # Step/loss/metrics
        m = re.search(r"step\s+(\d+)\s+loss\s+([\d.]+)", line, re.IGNORECASE)
        if m:
            result["type"] = "metric"
            result["step"] = int(m.group(1))
            result["loss"] = float(m.group(2))
            m2 = re.search(r"lr\s+([\d.e-]+)", line, re.IGNORECASE)
            if m2:
                result["lr"] = float(m2.group(1))
        return result

    @staticmethod
    def _format_log_line(parsed: dict) -> str:
        """Color-code a parsed log line for terminal output."""
        level = parsed.get("level", "INFO")
        line = parsed["raw"]
        if parsed.get("type") == "exit":
            code = parsed.get("exit_code", -1)
            color = _Color.GREEN if code == 0 else _Color.RED
            return f"{color}{line}{_Color.RESET}"
        if parsed.get("type") == "metric":
            return f"{_Color.CYAN}{line}{_Color.RESET}"
        colors = {
            "ERROR": _Color.RED,
            "WARNING": _Color.YELLOW,
            "INFO": _Color.GREEN,
            "DEBUG": _Color.DIM,
        }
        color = colors.get(level, "")
        if color:
            return f"{color}{line}{_Color.RESET}"
        return line

    @staticmethod
    def _should_show_line(parsed: dict,
                          filter_str: str | None) -> bool:
        """Apply log filter (grep-like)."""
        if not filter_str:
            return True
        return filter_str.lower() in parsed["raw"].lower()

    # ── Provisioning ──────────────────────────────────────────────────────
    def _build_provision_cmd(self) -> str:
        """Shell command to set up the remote venv + install ForgeAI deps."""
        return (
            f"set -e; "
            f"python -m venv {REMOTE_VENV} && "
            f". {REMOTE_VENV}/bin/activate && "
            f"pip install --upgrade pip wheel && "
            f"pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && "
            f"pip install datasets==4.5.0 transformers==4.51.3 "
            f"gigatoken==0.9.0 bitsandbytes==0.49.2 "
            f"python-dotenv==1.0.1 safetensors>=0.4.0 msgspec>=0.21 "
            f"rich>=14.0 loguru>=0.7.3 torchao>=0.17 "
            f"vastai>=1.4.0 paramiko>=3.4 2>&1 | tail -10"
        )

    def _provision_hash(self) -> str:
        """Compute a hash of the provisioning command + requirements file.

        If this hash matches the remote cached hash, provisioning can be
        skipped entirely (deps already installed from a prior run).
        """
        h = hashlib.sha256()
        h.update(self._build_provision_cmd().encode())
        req_path = PROJECT_ROOT / "requirements.txt"
        if req_path.exists():
            h.update(req_path.read_bytes())
        return h.hexdigest()[:16]

    def _check_provision_hash(self, host: str, port: int) -> bool:
        """Check if the remote provision hash matches (skip provisioning).

        Returns True if provisioning can be skipped (hash matches).
        """
        expected = self._provision_hash()
        try:
            client = self._ssh_connect(host, port)
            sftp = client.open_sftp()
            try:
                with sftp.open(REMOTE_PROVISION_HASH, "r") as f:
                    remote_hash = f.read().decode("utf-8", "replace").strip()
                return remote_hash == expected
            finally:
                sftp.close()
        except OSError:
            return False  # no hash file — need to provision

    def _write_provision_hash(self, host: str, port: int) -> None:
        """Write the provision hash to the instance for next run."""
        try:
            client = self._ssh_connect(host, port)
            sftp = client.open_sftp()
            try:
                self._ensure_remote_dir(sftp, REMOTE_ROOT)
                with sftp.open(REMOTE_PROVISION_HASH, "w") as f:
                    f.write(self._provision_hash())
            finally:
                sftp.close()
        except Exception as e:
            logger.debug(f"Could not write provision hash: {e}")

    # ── High-level: remote training lifecycle ─────────────────────────────
    def _auto_tune_for_gpu(self, spec: RemoteTrainingSpec,
                           gpu_vram_gb: float, num_gpus: int) -> None:
        """Override VRAM-saving defaults to maximize throughput on a big GPU.

        The local RTX 5070 has 12GB VRAM, so sft_train defaults to aggressive
        VRAM saving: grad checkpointing, BitNet int8, batch_size=1, LoRA. On
        a rented 80GB A100/H100, these are wasteful — they trade compute for
        VRAM we don't need to save. This method reconfigures for max speed.
        """
        if not spec.maximize_throughput:
            return
        ta = spec.train_args
        if gpu_vram_gb >= 40:
            ta["--grad-checkpoint"] = False
            logger.info(f"  Auto: disabled grad checkpointing (+30% throughput on {gpu_vram_gb:.0f}GB)")
        if gpu_vram_gb >= 60:
            ta["--bitnet-everywhere"] = False
            logger.info(f"  Auto: disabled BitNet int8 (full bf16 on {gpu_vram_gb:.0f}GB)")
            ta["--lora"] = False
            logger.info(f"  Auto: disabled LoRA (full fine-tune on {gpu_vram_gb:.0f}GB)")
            ta["--grad-compression"] = "none"
            logger.info(f"  Auto: disabled grad compression (no CPU offload on {gpu_vram_gb:.0f}GB)")
        current_bs = ta.get("--batch-size", 1)
        if current_bs == 1:
            if gpu_vram_gb >= 80:
                ta["--batch-size"] = 4
            elif gpu_vram_gb >= 60:
                ta["--batch-size"] = 2
            else:
                ta["--batch-size"] = 1
            logger.info(f"  Auto: batch_size={ta['--batch-size']} for {gpu_vram_gb:.0f}GB")
        if gpu_vram_gb >= 40:
            ta["--compile"] = True
            logger.info("  Auto: enabled torch.compile (1.3-2x kernel fusion)")
        if spec.from_scratch:
            ta["--checkpoint"] = "scratch"
            ta["--use-forge-engine"] = False
            logger.info("  Auto: set --checkpoint scratch, disabled ForgeEngine")
            current_opt = ta.get("--optimizer", "muon_sf")
            bitnet_active = ta.get("--bitnet-everywhere", True) is not False
            if not bitnet_active and gpu_vram_gb >= 60:
                if current_opt not in ("flash_adamw", "fused", "fira_nlrq"):
                    ta["--optimizer"] = "flash_adamw"
                    logger.info("  Auto: optimizer=flash_adamw")
            else:
                if current_opt not in ("badam", "fira_nlrq"):
                    ta["--optimizer"] = "badam"
                    logger.info("  Auto: optimizer=badam (BitNet int8)")
        if num_gpus > 1:
            logger.info(f"  Note: {num_gpus} GPUs available, using GPU 0 "
                        f"(multi-GPU DDP not yet supported)")

    def run_remote_training(self, spec: RemoteTrainingSpec) -> int:
        """Full lifecycle: ensure instance → sync → train → download → stop/destroy.

        v2 lifecycle:
        1. Find or create instance (by label, with stop/start reuse).
        2. Wait for running + SSH.
        3. Incrementally sync only changed source files.
        4. Skip provisioning if requirements hash unchanged.
        5. Upload checkpoints + data (incremental via manifest).
        6. Launch training, stream structured logs.
        7. Download output checkpoint.
        8. Stop instance (preserve disk) or destroy (if auto_destroy).
        """
        # 1. Find or create instance.
        inst_id, is_new = self.ensure_instance(spec)
        try:
            return self._drive_instance(inst_id, spec, is_new)
        finally:
            if spec.auto_destroy:
                try:
                    self.destroy_instance(inst_id)
                except Exception as e:
                    logger.warning(f"Failed to auto-destroy {inst_id}: {e}")
            else:
                try:
                    self.stop_instance(inst_id)
                except Exception as e:
                    logger.warning(f"Failed to auto-stop {inst_id}: {e}")
            self.ssh_disconnect()

    def _drive_instance(self, inst_id: int, spec: RemoteTrainingSpec,
                        is_new: bool) -> int:
        """Provision + run training on an already-rented/reused instance."""
        # 2. Wait for running + SSH.
        self.wait_for_running(inst_id, timeout=spec.startup_timeout,
                              poll=spec.poll_interval)
        host, port = self.get_ssh_info(inst_id)
        logger.info(f"SSH ready: root@{host}:{port}")

        # 3. Sync only critical source files (not the entire research/ tree).
        logger.info("Syncing critical source files (incremental)...")
        self.sync_critical_files(host, port)
        for fname in ["requirements.txt", "pyproject.toml"]:
            src = str(PROJECT_ROOT / fname)
            if Path(src).exists():
                self.upload_file(host, port, src, f"{REMOTE_REPO}/{fname}")
        # Ensure remote dirs exist.
        client = self._ssh_connect(host, port)
        sftp = client.open_sftp()
        try:
            for d in [REMOTE_CKPT_DIR, REMOTE_DATA_DIR, REMOTE_OUT_DIR]:
                self._ensure_remote_dir(sftp, d)
        finally:
            sftp.close()

        # 4. Provision (skip if hash matches and not a new instance).
        if not is_new and self._check_provision_hash(host, port):
            logger.info("Provisioning skipped (requirements unchanged).")
        else:
            logger.info("Provisioning remote environment (pip install)...")
            provision_cmd = self._build_provision_cmd()
            rc, out, err = self.exec_remote(host, port, provision_cmd, timeout=900.0)
            if rc != 0:
                logger.error(f"Provisioning failed:\n{out}\n{err}")
                return rc
            logger.info("Provisioning complete.")
            self._write_provision_hash(host, port)

        # 5. Upload checkpoints + data (check if already on remote).
        for ckpt in spec.checkpoints:
            rpath = f"{REMOTE_CKPT_DIR}/{Path(ckpt).name}"
            if self._remote_file_exists(host, port, rpath, Path(ckpt).stat().st_size):
                logger.info(f"Checkpoint {Path(ckpt).name} already on remote, skipping.")
            else:
                logger.info(f"Uploading checkpoint {ckpt} -> {rpath}")
                self.upload_file(host, port, ckpt, rpath, show_progress=True)
        for data in spec.data_files:
            rpath = f"{REMOTE_DATA_DIR}/{Path(data).name}"
            if self._remote_file_exists(host, port, rpath, Path(data).stat().st_size):
                logger.info(f"Data {Path(data).name} already on remote, skipping.")
            else:
                logger.info(f"Uploading data {data} -> {rpath}")
                self.upload_file(host, port, data, rpath, show_progress=True)

        # 6. Build + launch the remote sft_train command.
        remote_cmd = self._build_remote_train_cmd(spec)
        logger.info(f"Launching remote training:\n  {remote_cmd}")
        exit_code = -1
        if spec.stream_logs:
            for line in self.stream_remote(host, port, remote_cmd):
                parsed = self._parse_log_line(line)
                if not self._should_show_line(parsed, spec.log_filter):
                    continue
                if parsed.get("type") == "exit":
                    exit_code = parsed.get("exit_code", -1)
                    print(self._format_log_line(parsed), flush=True)
                else:
                    print(self._format_log_line(parsed), flush=True)
            # Fallback: probe exit code file if no marker was seen.
            if exit_code < 0:
                try:
                    _rc2, out2, _ = self.exec_remote(
                        host, port, f"cat {REMOTE_EXIT_CODE} 2>/dev/null || echo -1")
                    exit_code = int(out2.strip())
                except ValueError:
                    exit_code = -1
        else:
            rc, out, err = self.exec_remote(host, port, remote_cmd, timeout=None)
            exit_code = rc
            print(out)
            if err:
                print(err, file=sys.stderr)

        # 7. Download the output checkpoint.
        if spec.download_output and exit_code == 0:
            save_arg = spec.train_args.get("--save")
            if save_arg:
                local_save = str(Path(save_arg))
                if not os.path.isabs(local_save):
                    local_save = str(PROJECT_ROOT / local_save)
                remote_save = self._remap_path(local_save, spec)
                logger.info(f"Downloading trained checkpoint {remote_save} -> {local_save}")
                try:
                    self.download_file(host, port, remote_save, local_save)
                except Exception as e:
                    logger.warning(f"Download failed: {e}")
                    try:
                        self.download_dir(host, port, remote_save, local_save)
                    except Exception as e2:
                        logger.error(f"Dir download also failed: {e2}")
        return exit_code

    def _remote_file_exists(self, host: str, port: int, remote_path: str,
                            expected_size: int) -> bool:
        """Check if a file exists on the remote with the expected size."""
        try:
            client = self._ssh_connect(host, port)
            sftp = client.open_sftp()
            try:
                stat = sftp.stat(remote_path)
                return stat.st_size == expected_size
            finally:
                sftp.close()
        except OSError:
            return False

    def wipe_remote_data(self, instance_id: int) -> None:
        """Wipe uploaded data + checkpoints on an instance without recreating it.

        Useful when you want to re-upload fresh data but keep the provisioned
        venv + repo. Clears REMOTE_DATA_DIR, REMOTE_CKPT_DIR, REMOTE_OUT_DIR,
        and the file manifest. Does NOT touch the venv or repo source.
        """
        self.wait_for_running(instance_id, timeout=300.0)
        host, port = self.get_ssh_info(instance_id)
        logger.info(f"Wiping remote data on instance {instance_id}...")
        wipe_cmd = (
            f"rm -rf {REMOTE_DATA_DIR}/* {REMOTE_CKPT_DIR}/* "
            f"{REMOTE_OUT_DIR}/* {REMOTE_MANIFEST} 2>/dev/null; "
            f"mkdir -p {REMOTE_DATA_DIR} {REMOTE_CKPT_DIR} {REMOTE_OUT_DIR}; "
            f"echo 'wipe complete'"
        )
        rc, out, err = self.exec_remote(host, port, wipe_cmd, timeout=60.0)
        if rc != 0:
            logger.error(f"Wipe failed:\n{out}\n{err}")
        else:
            logger.info("Remote data wiped (venv + repo preserved).")

    def _build_remote_train_cmd(self, spec: RemoteTrainingSpec) -> str:
        """Construct the remote ``python -m forge.training.runners.sft_train ...`` line."""
        setup = [
            f". {REMOTE_VENV}/bin/activate",
            f"cd {REMOTE_REPO}",
        ]
        train_parts = ["python -m forge.training.runners.sft_train"]
        for key, val in spec.train_args.items():
            if val is None or val is False:
                continue
            if val is True:
                train_parts.append(key)
            elif isinstance(val, list):
                train_parts.append(key)
                train_parts.extend(
                    self._remap_path(str(v), spec) if self._is_path_arg(key, v) else str(v)
                    for v in val
                )
            else:
                train_parts.append(key)
                train_parts.append(
                    self._remap_path(str(val), spec) if self._is_path_arg(key, val) else str(val))
        train_cmd = " ".join(train_parts)
        setup_cmd = " && ".join(setup)
        full = f"{setup_cmd} && PYTHONPATH={REMOTE_REPO} {train_cmd}"
        return (f'bash -lc \'set -e; {full}; '
                f'echo $? > {REMOTE_EXIT_CODE}; '
                f'code=$(cat {REMOTE_EXIT_CODE}); '
                f'echo "__FORGE_EXIT__:$code"\'')

    @staticmethod
    def _is_path_arg(key: str, val=None) -> bool:
        if key == "--checkpoint" and val in ("scratch", "none", "", None):
            return False
        return key in ("--data", "--checkpoint", "--save", "--anchor",
                       "--teacher-checkpoint")

    def _remap_path(self, local_path: str, spec: RemoteTrainingSpec) -> str:
        """Map a local path to its remote location after upload."""
        p = Path(local_path)
        name = p.name
        for ckpt in spec.checkpoints:
            if Path(ckpt).resolve() == p.resolve() or Path(ckpt).name == name:
                return f"{REMOTE_CKPT_DIR}/{name}"
        for data in spec.data_files:
            if Path(data).resolve() == p.resolve() or Path(data).name == name:
                return f"{REMOTE_DATA_DIR}/{name}"
        if key_save := spec.train_args.get("--save"):
            if local_path == str(key_save) or Path(local_path).name == Path(str(key_save)).name:
                return f"{REMOTE_OUT_DIR}/{name}"
        return f"{REMOTE_ROOT}/{name}"


# ─── Spec builder: convert sft_train argparse Namespace → RemoteTrainingSpec ─
def build_spec_from_args(args, train_arg_names: set[str]) -> RemoteTrainingSpec:
    """Convert a parsed sft_train argparse Namespace into a RemoteTrainingSpec."""
    spec = RemoteTrainingSpec(
        gpu_filter=getattr(args, "gpu_filter", ""),
        max_price=getattr(args, "max_price", 0.0),
        min_vram_gb=getattr(args, "min_vram_gb", 24.0),
        min_reliability=getattr(args, "min_reliability", 0.9),
        disk_gb=getattr(args, "vast_disk_gb", 100),
        image=getattr(args, "vast_image", None) or DEFAULT_IMAGE,
        on_demand=getattr(args, "vast_on_demand", True),
        budget=getattr(args, "vast_budget", 10.0),
        est_sec_per_step=getattr(args, "vast_est_sec_per_step", 1.5),
        ssh_key=getattr(args, "vast_ssh_key", ""),
        auto_destroy=getattr(args, "vast_auto_destroy", False),
        reuse_instance=getattr(args, "vast_reuse_instance", True),
        use_volume=getattr(args, "vast_use_volume", False),
        volume_size_gb=getattr(args, "vast_volume_size_gb", 200),
        stream_logs=getattr(args, "vast_stream_logs", True),
        download_output=getattr(args, "vast_download_output", True),
        poll_interval=getattr(args, "vast_poll_interval", 10.0),
        startup_timeout=getattr(args, "vast_startup_timeout", 600.0),
        maximize_throughput=getattr(args, "vast_maximize_throughput", True),
        from_scratch=getattr(args, "from_scratch", False),
        log_filter=getattr(args, "vast_log_filter", None),
    )
    train_args: dict[str, Any] = {}
    for dest in train_arg_names:
        if not hasattr(args, dest):
            continue
        val = getattr(args, dest)
        if val is None:
            continue
        flag = "--" + dest.replace("_", "-")
        train_args[flag] = val
    spec.train_args = train_args

    ckpts: list[str] = []
    if (cp := train_args.get("--checkpoint")) and isinstance(cp, str):
        ckpts.append(str(PROJECT_ROOT / cp) if not os.path.isabs(cp) else cp)
    if (tp := train_args.get("--teacher-checkpoint")) and isinstance(tp, str):
        ckpts.append(str(PROJECT_ROOT / tp) if not os.path.isabs(tp) else tp)
    if (ap := train_args.get("--anchor")) and isinstance(ap, str):
        ckpts.append(str(PROJECT_ROOT / ap) if not os.path.isabs(ap) else ap)
    spec.checkpoints = ckpts

    data_files: list[str] = []
    if (d := train_args.get("--data")) and isinstance(d, list):
        for df in d:
            data_files.append(str(PROJECT_ROOT / df) if not os.path.isabs(df) else df)
    spec.data_files = data_files
    return spec


# ─── CLI entrypoint (standalone use) ───────────────────────────────────────
def main() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Vast.ai cloud backend for ForgeAI training (v2: SDK + persistent).")
    sub = p.add_subparsers(dest="command", required=True)

    p_offers = sub.add_parser("offers", help="List matching GPU offers.")
    p_offers.add_argument("--gpu-filter", default="")
    p_offers.add_argument("--max-price", type=float, default=0.0)
    p_offers.add_argument("--min-vram-gb", type=float, default=24.0)
    p_offers.add_argument("--min-reliability", type=float, default=0.9)

    p_instances = sub.add_parser("instances", help="List current instances.")

    p_destroy = sub.add_parser("destroy", help="Destroy an instance.")
    p_destroy.add_argument("instance_id", type=int)

    p_stop = sub.add_parser("stop", help="Stop instance (preserve disk).")
    p_stop.add_argument("instance_id", type=int)

    p_start = sub.add_parser("start", help="Start stopped instance.")
    p_start.add_argument("instance_id", type=int)

    p_logs = sub.add_parser("logs", help="Fetch historical logs.")
    p_logs.add_argument("instance_id", type=int)
    p_logs.add_argument("--tail", type=int, default=1000)
    p_logs.add_argument("--filter", dest="filter_str", default=None)

    p_wipe = sub.add_parser("wipe", help="Wipe remote data (keep venv+repo).")
    p_wipe.add_argument("instance_id", type=int)

    p_run = sub.add_parser("run", help="Ensure instance, sync, train, download, stop.")
    p_run.add_argument("--data", nargs="+", required=True)
    p_run.add_argument("--config", default="forgelm_v2")
    p_run.add_argument("--checkpoint",
                       default="research/checkpoints/ForgeLM_V2.safetensors")
    p_run.add_argument("--save", default="ForgeLM_V10.sft.safetensors")
    p_run.add_argument("--max-steps", type=int, default=500)
    p_run.add_argument("--lr", type=float, default=2e-4)
    p_run.add_argument("--batch-size", type=int, default=1)
    p_run.add_argument("--seq-len", type=int, default=1024)
    p_run.add_argument("--no-lora", dest="lora", action="store_false", default=True)
    p_run.add_argument("--no-bitnet-everywhere", dest="bitnet_everywhere",
                       action="store_false", default=True)
    p_run.add_argument("--from-scratch", action="store_true", default=False)
    p_run.add_argument("--vast-maximize-throughput",
                       action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--gpu-filter", default="")
    p_run.add_argument("--max-price", type=float, default=0.0)
    p_run.add_argument("--min-vram-gb", type=float, default=24.0)
    p_run.add_argument("--min-reliability", type=float, default=0.9)
    p_run.add_argument("--vast-budget", type=float, default=10.0)
    p_run.add_argument("--vast-est-sec-per-step", type=float, default=5.0)
    p_run.add_argument("--vast-disk-gb", type=int, default=100)
    p_run.add_argument("--vast-image", default=DEFAULT_IMAGE)
    p_run.add_argument("--vast-on-demand",
                       action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--vast-ssh-key", default="")
    p_run.add_argument("--vast-auto-destroy",
                       action=argparse.BooleanOptionalAction, default=False,
                       help="Destroy after run (default False = stop+preserve).")
    p_run.add_argument("--vast-reuse-instance",
                       action=argparse.BooleanOptionalAction, default=True,
                       help="Find+reuse stopped instance by label (default True).")
    p_run.add_argument("--vast-use-volume",
                       action=argparse.BooleanOptionalAction, default=False,
                       help="Attach a persistent volume (default False).")
    p_run.add_argument("--vast-volume-size-gb", type=int, default=200)
    p_run.add_argument("--vast-stream-logs",
                       action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--vast-download-output",
                       action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--vast-poll-interval", type=float, default=10.0)
    p_run.add_argument("--vast-startup-timeout", type=float, default=600.0)
    p_run.add_argument("--vast-log-filter", default=None,
                       help="Grep filter for live log stream.")

    args = p.parse_args()
    conn = VastConnector()

    if args.command == "offers":
        for o in conn.search_offers(args.gpu_filter, args.max_price,
                                    args.min_vram_gb, args.min_reliability,
                                    sort_by_perf=True):
            print(o.label)
        return 0
    if args.command == "instances":
        for inst in conn.list_instances():
            print(f"#{inst.get('id')} {inst.get('actual_status', inst.get('cur_state'))} "
                  f"{inst.get('gpu_name', '?')} ${inst.get('dph_total', '?')}/h "
                  f"label={inst.get('label', '')}")
        return 0
    if args.command == "destroy":
        conn.destroy_instance(args.instance_id)
        return 0
    if args.command == "stop":
        conn.stop_instance(args.instance_id)
        return 0
    if args.command == "start":
        conn.start_instance(args.instance_id)
        return 0
    if args.command == "logs":
        logs = conn.get_logs(args.instance_id, tail=args.tail,
                             filter_str=args.filter_str)
        print(logs)
        return 0
    if args.command == "wipe":
        conn.wipe_remote_data(args.instance_id)
        return 0
    if args.command == "run":
        train_args = {
            "--data": args.data,
            "--config": args.config,
            "--checkpoint": args.checkpoint,
            "--save": args.save,
            "--max-steps": args.max_steps,
            "--lr": args.lr,
            "--batch-size": args.batch_size,
            "--seq-len": args.seq_len,
            "--lora": args.lora,
            "--bitnet-everywhere": args.bitnet_everywhere,
        }
        spec = RemoteTrainingSpec(
            train_args=train_args,
            gpu_filter=args.gpu_filter,
            max_price=args.max_price,
            min_vram_gb=args.min_vram_gb,
            min_reliability=args.min_reliability,
            budget=args.vast_budget,
            est_sec_per_step=args.vast_est_sec_per_step,
            disk_gb=args.vast_disk_gb,
            image=args.vast_image,
            on_demand=args.vast_on_demand,
            ssh_key=args.vast_ssh_key,
            auto_destroy=args.vast_auto_destroy,
            reuse_instance=args.vast_reuse_instance,
            use_volume=args.vast_use_volume,
            volume_size_gb=args.vast_volume_size_gb,
            stream_logs=args.vast_stream_logs,
            download_output=args.vast_download_output,
            poll_interval=args.vast_poll_interval,
            startup_timeout=args.vast_startup_timeout,
            maximize_throughput=args.vast_maximize_throughput,
            from_scratch=args.from_scratch,
            log_filter=args.vast_log_filter,
        )
        if not args.from_scratch:
            spec.checkpoints = [
                str(PROJECT_ROOT / args.checkpoint)
                if not os.path.isabs(args.checkpoint) else args.checkpoint
            ]
        spec.data_files = [
            str(PROJECT_ROOT / d) if not os.path.isabs(d) else d
            for d in args.data
        ]
        return conn.run_remote_training(spec)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
