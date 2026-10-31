"""Training routes: fine-tune datasets + launcher, self-play control,
training-run telemetry."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from ..deps import services

logger = logging.getLogger(__name__)
router = APIRouter()

from forge_gui.api.status_reader import project_root
import contextlib

OPTIMIZERS = ["muon_sf", "muon_sf_plain", "muon", "fused", "lion",
              "flash_adamw", "flash_lion", "sf_normuon", "amuse", "mona",
              "samuon_lite",
              "bnb", "forge", "cpu_offload", "badam", "fira_nlrq"]
LOSSES = ["ce", "focal", "label_smoothing", "lovasz", "dynamic_focal",
          "mixture"]
CURRICULA = ["none", "vanilla", "pacing", "interleaved", "warmup"]
SELFPLAY_TOPICS = [
    "python_algorithms", "python_math", "python_strings",
    "python_general", "python_oop", "python_file_io",
    "math_arithmetic", "all_topics",
]


# ── datasets ──────────────────────────────────────────────────────────

def _scan_datasets() -> list[dict]:
    root = project_root()
    data_dir = root / "data"
    out = []
    if not data_dir.is_dir():
        return out
    for p in sorted(data_dir.rglob("*.jsonl")):
        try:
            n = 0
            with open(p, encoding="utf-8", errors="ignore") as f:
                for _ in f:
                    n += 1
            rel = str(p.relative_to(root)).replace("\\", "/")
            out.append({"path": rel, "name": p.name,
                        "examples": n,
                        "size_bytes": p.stat().st_size,
                        "modified": p.stat().st_mtime})
        except OSError:
            continue
    out.sort(key=lambda d: d["modified"], reverse=True)
    return out


@router.get("/finetune/datasets")
async def datasets():
    loop = asyncio.get_running_loop()
    ds = await loop.run_in_executor(None, _scan_datasets)
    return {"datasets": ds,
            "exports": services.chat_store.list_exports(),
            "choices": {"optimizers": OPTIMIZERS, "losses": LOSSES,
                        "curricula": CURRICULA}}


@router.post("/finetune/export")
async def export_rated():
    loop = asyncio.get_running_loop()
    path, n = await loop.run_in_executor(
        None, services.chat_store.export_training_data)
    return {"path": str(path), "examples": n}


class FinetuneRequest(BaseModel):
    datasets: list[str] = []
    config: str = "forgelm_v2"
    checkpoint: str = "research/checkpoints/ForgeLM_V2.safetensors"
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    save_adapter: bool = True
    bitnet: bool = True
    max_steps: int = 500
    lr: float = 5e-5
    min_lr: float = 5e-6
    warmup: int = 20
    batch_size: int = 1
    grad_accum: int = 5
    seq_len: int = 1024
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    qk_clip_tau: float = 0.0
    optimizer: str = "muon_sf"
    loss_function: str = "ce"
    entropy_alpha: float = 0.5
    curriculum: str = "none"
    ema: bool = False
    augment: bool = False
    synpro: bool = False
    val_every: int = 0
    save: str = "research/checkpoints/forge_finetuned.safetensors"


def _build_finetune_cmd(body: FinetuneRequest) -> list[str] | None:
    if not body.datasets:
        return None
    root = project_root()
    # Canonical trainer — research/ migrated to forge/ (the old path
    # launched a nonexistent script and the task died instantly).
    runner = root / "forge" / "training" / "runners" / "sft_train.py"
    if not runner.is_file():
        return None
    venv_python = root / "venv" / "Scripts" / "python.exe"
    cmd = [str(venv_python if venv_python.is_file() else "python"),
           str(runner),
           "--data", *body.datasets,
           "--config", body.config or "forgelm_v2",
           "--checkpoint", body.checkpoint,
           "--max-steps", str(body.max_steps),
           "--lr", f"{body.lr:.7g}",
           "--min-lr", f"{body.min_lr:.7g}",
           "--batch-size", str(body.batch_size),
           "--grad-accum", str(body.grad_accum),
           "--seq-len", str(body.seq_len),
           "--warmup-steps", str(body.warmup),
           "--weight-decay", f"{body.weight_decay:.3g}",
           "--grad-clip", f"{body.grad_clip:.3g}",
           "--optimizer", body.optimizer,
           "--loss-function", body.loss_function,
           "--entropy-alpha", f"{body.entropy_alpha:.2g}",
           "--curriculum", body.curriculum,
           "--save", body.save]
    if body.qk_clip_tau > 0:
        cmd += ["--qk-clip-tau", f"{body.qk_clip_tau:.3g}"]
    if body.use_lora:
        cmd += ["--lora", "--lora-r", str(body.lora_r),
                "--lora-alpha", str(body.lora_alpha)]
        if body.save_adapter:
            cmd.append("--save-lora-adapter")
    else:
        cmd.append("--no-lora")
    cmd.append("--bitnet-everywhere" if body.bitnet
               else "--no-bitnet-everywhere")
    if body.ema:
        cmd.append("--ema")
    if body.augment:
        cmd.append("--augment")
    if body.synpro:
        cmd.append("--synpro")
    if body.val_every > 0:
        cmd += ["--val-every", str(body.val_every)]
    return cmd


@router.post("/finetune/preview")
async def finetune_preview(body: FinetuneRequest):
    cmd = _build_finetune_cmd(body)
    return {"cmd": cmd, "preview": " ".join(cmd) if cmd else ""}


@router.post("/finetune/launch")
async def finetune_launch(body: FinetuneRequest):
    runner = project_root() / "forge" / "training" / "runners" / "sft_train.py"
    if not runner.is_file():
        return {"ok": False, "error":
                f"trainer not found: {runner.relative_to(project_root())}"}
    cmd = _build_finetune_cmd(body)
    if cmd is None:
        return {"ok": False, "error": "select at least one dataset"}
    name = ("LoRA fine-tune" if body.use_lora else "Full fine-tune")
    task_id = services.procs.launch(name, cmd)
    return {"ok": True, "task_id": task_id}


# ── self-play ─────────────────────────────────────────────────────────

def _selfplay_dir() -> Path:
    return project_root() / "research" / "checkpoints" / "self_play"


def _flux_checkpoints() -> list[str]:
    """.flux snapshots under research/checkpoints/flux/ — selectable as
    the starting model for a flux-mode self-play run (or fresh)."""
    d = project_root() / "research" / "checkpoints" / "flux"
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.flux"))


@router.get("/selfplay/status")
async def selfplay_status():
    loop = asyncio.get_running_loop()

    def fetch():
        ev = services.events
        return (ev.latest_status(), ev.all_events()[-200:],
                ev.heartbeat_age(), ev.heartbeat_stalled())

    # JSON parse + file stats stay off the loop (EC: self-play status offload)
    latest, events, hb_age, hb_stalled = await loop.run_in_executor(
        None, fetch)
    return {"status": latest or {}, "events": events,
            "heartbeat_age_s": hb_age,
            "heartbeat_stalled": hb_stalled,
            "topics": SELFPLAY_TOPICS,
            "flux_models": _flux_checkpoints()}


class SelfPlayStartRequest(BaseModel):
    topic: str = "python_algorithms"
    epochs: int = 3
    tasks_per_epoch: int = 50
    mode: str = "sft"               # 'sft' | 'grpo' | 'flux'
    flux_checkpoint: str = ""       # .flux path, or "" for a fresh model
    flux_group_size: int = 4        # sampled attempts per task
    flux_workers: int = 4           # cloned-memory gen workers


@router.post("/selfplay/start")
async def selfplay_start(body: SelfPlayStartRequest):
    root = project_root()
    sp_dir = _selfplay_dir()
    sp_dir.mkdir(parents=True, exist_ok=True)
    # kill stray self-play tasks, wipe stale telemetry (matches Qt page)
    for t in services.procs.tasks.values():
        if t.is_live and "self" in t.name.lower() and "play" in t.name.lower():
            services.procs.kill(t.id)
    for f in ("events.jsonl", "status.json", "heartbeat.json",
              "STOP_REQUESTED"):
        with contextlib.suppress(OSError):
            (sp_dir / f).unlink(missing_ok=True)
    venv_py = root / "venv" / "Scripts" / "python.exe"
    cmd = [str(venv_py if venv_py.is_file() else "python"), "-u",
           "-m", "forge.self_play.infinite_loop",
           "--epochs", str(body.epochs),
           "--tasks-per-epoch", str(body.tasks_per_epoch)]
    if body.mode == "flux":
        ckpt = body.flux_checkpoint
        if ckpt and not ckpt.startswith(("research/", "research\\")):
            # name from the flux picker → checkpoint dir path
            ckpt = str(root / "research" / "checkpoints" / "flux" / ckpt)
        cmd += ["--training-mode", "flux",
                "--checkpoint", ckpt or "none",
                "--flux-group-size", str(max(1, body.flux_group_size)),
                "--flux-workers", str(max(1, body.flux_workers))]
    else:
        cmd += ["--checkpoint",
                str(root / "research" / "checkpoints"
                    / "ForgeLM_V2.safetensors"),
                "--config", "forgelm_v2",
                "--training-mode", body.mode,
                "--ft-batch-size", "8"]
    task_id = services.procs.launch("Self-Play Training", cmd)
    return {"ok": True, "task_id": task_id}


@router.post("/selfplay/stop")
async def selfplay_stop():
    stopped = False
    for t in services.procs.tasks.values():
        if t.is_live and "self" in t.name.lower() and "play" in t.name.lower():
            services.procs.kill(t.id)
            stopped = True
    # also drop the STOP_REQUESTED sentinel for cooperative shutdown
    try:
        (_selfplay_dir() / "STOP_REQUESTED").write_text("stop")
        stopped = True
    except OSError:
        pass
    return {"ok": stopped}
