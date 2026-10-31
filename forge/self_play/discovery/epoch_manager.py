"""Epoch manager — compare, keep best, archive loser, trigger distill.

After each fine-tune (or distill) run, the new candidate epoch is evaluated
alongside the current best on quality + skill + compute. The winner becomes
(or stays) the 'best' epoch; the loser is moved to an archive directory so
the user can inspect it but it won't be used as the parent for the next
fine-tune. This implements the user's "best stays, loser is archived" rule.

Every DISTILL_EVERY epochs, a distill run is forced (see distill.py) to
filter bloat and outdated knowledge — the model is rebuilt from DB-curated
content only.

This module is the single entry point the discovery loop calls:
    EpochManager.maybe_advance(db, device) -> dict | None
It decides whether to fine-tune, distill, or do nothing, based on DB size +
quality triggers and the epoch count.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from forge.self_play.discovery.discovery_db import DiscoveryDB
from research.paths import DATA_DIR

_EPOCHS_DIR = DATA_DIR / "discovery" / "epochs"
_ARCHIVE_DIR = _EPOCHS_DIR / "archive"

# Trigger thresholds for "DB is high quality and large enough".
_MIN_THOUGHTS = 50
_MIN_SCRIPTS_OK = 20
_MIN_THEORIES = 10
_MIN_DISCOVERIES = 5
_MIN_RESEARCH = 10
_MIN_TRAJECTORIES = 15  # tool-use trajectories for SFT
# Distill cadence.
DISTILL_EVERY = 12


@dataclass
class EpochComparison:
    best_composite: float
    candidate_composite: float
    winner: str  # 'best' | 'candidate' | 'tie'
    archived_path: str | None
    detail: dict


def _db_quality_score(db: DiscoveryDB) -> float:
    """Heuristic 0..1 score for DB readiness. Used as the fine-tune trigger.

    Now includes tool-use trajectory quality as a component.
    """
    counts = db.table_counts()
    if counts["thoughts"] < _MIN_THOUGHTS:
        return 0.0
    theories = db.query("SELECT status, COUNT(*) AS n FROM theories GROUP BY status")
    status = {r["status"]: r["n"] for r in theories}
    resolved = status.get("supported", 0) + status.get("refuted", 0)
    total_th = max(sum(status.values()), 1)
    resolution_rate = resolved / total_th
    ok_scripts = db.query(
        "SELECT COUNT(*) AS n FROM scripts WHERE returncode=0 AND length(stdout) > 0")[0]["n"]
    script_quality = min(ok_scripts / _MIN_SCRIPTS_OK, 1.0) if _MIN_SCRIPTS_OK else 1.0

    # Tool-use trajectory quality
    traj_stats = db.trajectory_stats()
    n_traj = traj_stats.get("n", 0) or 0
    avg_traj_reward = traj_stats.get("avg_reward", 0.0) or 0.0
    traj_quality = min(n_traj / _MIN_TRAJECTORIES, 1.0) * avg_traj_reward

    coverage = min(1.0, (
        min(counts["thoughts"] / _MIN_THOUGHTS, 1.0) * 0.25 +
        min(counts["discoveries"] / _MIN_DISCOVERIES, 1.0) * 0.25 +
        min(counts["research"] / _MIN_RESEARCH, 1.0) * 0.15 +
        min(counts["theories"] / _MIN_THEORIES, 1.0) * 0.15 +
        min(n_traj / _MIN_TRAJECTORIES, 1.0) * 0.20))
    return coverage * 0.4 + resolution_rate * 0.2 + script_quality * 0.2 + traj_quality * 0.2


def db_is_ready(db: DiscoveryDB, min_score: float = 0.6) -> bool:
    """True if the DB is high-quality and large enough to fine-tune from."""
    return _db_quality_score(db) >= min_score


def _load_model_at(checkpoint_path: str | None, device: str):
    """Load the ForgeLM V10 model, optionally with LoRA adapter or full checkpoint.

    Handles two checkpoint types:
      - LoRA adapter (*_lora.safetensors): Load R30 base + attach LoRA via
        engine.load_lora() (hot-loading, no merge needed).
      - Full checkpoint (*.safetensors): Load directly into model weights.
    """
    from forge.checkpoint_io import load_checkpoint
    from forge.model_loader import load_default_model

    model, tok = load_default_model("forgelm_v2")

    if checkpoint_path:
        # LoRA adapter checkpoint: load R30 base + hot-load LoRA
        if "_lora" in Path(checkpoint_path).name:
            from forge.engine.forge_engine import ForgeEngine
            from research.paths import V2_CHECKPOINT
            r30_path = str(V2_CHECKPOINT.parent / "ForgeLM_V2.safetensors")
            # Reload from R30 base (LoRA was trained on R30)
            engine = ForgeEngine.from_checkpoint(
                r30_path, config_name="forgelm_v2",
                device=device, auto_activate=False)
            engine.load_lora(checkpoint_path, rank=32, alpha=64)
            model = engine.model
            tok = engine.tokenizer
        else:
            # Full checkpoint: load directly
            sd = load_checkpoint(checkpoint_path)
            model.load_state_dict(
                {k: v.to(model.device) for k, v in sd.items()}, strict=False)

    model.to(device).eval()
    return model, tok


def _archive(checkpoint_path: str) -> str:
    """Move a loser checkpoint into the archive dir. Returns archive path."""
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(checkpoint_path)
    dst = _ARCHIVE_DIR / src.name
    if src.exists():
        shutil.move(str(src), str(dst))
        # Move sidecar meta too if present.
        meta = src.with_name(src.name + ".meta.json")
        if meta.exists():
            shutil.move(str(meta), str(_ARCHIVE_DIR / meta.name))
    return str(dst)


class EpochManager:
    """Orchestrates fine-tune, distill, and best-vs-loser comparison."""

    def __init__(self, db: DiscoveryDB, device: str = "cuda"):
        self.db = db
        self.device = device

    def maybe_advance(self) -> dict | None:
        """Decide + execute the next epoch action. Returns a summary or None.

        Logic:
          1. If epoch count is a multiple of DISTILL_EVERY and DB is ready ->
             run a distill pass.
          2. Else if DB is ready -> run a fine-tune pass.
          3. Always: compare candidate vs current best; keep winner, archive
             loser.
        """
        if not db_is_ready(self.db):
            return None
        epoch_num = self.db.last_epoch_num()
        do_distill = (epoch_num > 0 and epoch_num % DISTILL_EVERY == 0)
        from forge.self_play.discovery import distill as distill_mod
        from forge.self_play.discovery import finetune as ft_mod

        if do_distill:
            teacher = (self.db.best_epoch() or {}).get("checkpoint_path")
            candidate_path = distill_mod.distill_run(
                self.db, teacher_checkpoint=teacher, device=self.device)
            kind = "distill"
        else:
            parent = (self.db.best_epoch() or {}).get("checkpoint_path")
            candidate_path = ft_mod.finetune_from_db(
                self.db, base_checkpoint=parent, device=self.device)
            kind = "finetune"

        comp = self._compare_and_promote(candidate_path, kind)
        self.db.emit("epoch_advanced", {"kind": kind, "comparison": comp.__dict__})
        return {"kind": kind, "candidate": candidate_path,
                "winner": comp.winner, "archived": comp.archived_path,
                "best_composite": comp.best_composite,
                "candidate_composite": comp.candidate_composite}

    def _compare_and_promote(self, candidate_path: str, kind: str) -> EpochComparison:
        """Evaluate candidate vs current best; keep winner, archive loser."""
        from forge.self_play.discovery.quality_eval import evaluate

        best = self.db.best_epoch()
        # Score candidate.
        cand_model, tok = _load_model_at(candidate_path, self.device)
        cand_score = evaluate(cand_model, tok, self.db, self.device)
        del cand_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if best is None:
            # First epoch — promote by default.
            self.db.add_epoch(self.db.last_epoch_num(), candidate_path,
                              kind=kind, **cand_score.as_db_tuple(), status="best")
            return EpochComparison(best_composite=0.0,
                                   candidate_composite=cand_score.composite,
                                   winner="candidate", archived_path=None,
                                   detail={"first": True})

        # Score current best.
        best_model, _ = _load_model_at(best["checkpoint_path"], self.device)
        best_score = evaluate(best_model, tok, self.db, self.device)
        del best_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Update the candidate epoch row with scores.
        self.db.query("SELECT 1")  # noop to ensure connection works
        # Record candidate scores (the add_epoch in finetune/distill already
        # inserted a candidate row; update it).
        epoch_num = self.db.last_epoch_num()
        with self.db._lock, self.db._conn() as c:
            c.execute(
                "UPDATE epochs SET quality=?, skill=?, compute=?, composite=? "
                "WHERE epoch_num=? AND status='candidate'",
                (cand_score.quality, cand_score.skill, cand_score.compute,
                 cand_score.composite, epoch_num))

        if cand_score.composite > best_score.composite + 1e-4:
            # Candidate wins: demote best -> archived, promote candidate.
            self.db.set_epoch_status(best["id"], "archived")
            archived = _archive(best["checkpoint_path"])
            with self.db._lock, self.db._conn() as c:
                c.execute("UPDATE epochs SET status='best' WHERE epoch_num=?",
                          (epoch_num,))
            return EpochComparison(best_composite=best_score.composite,
                                   candidate_composite=cand_score.composite,
                                   winner="candidate", archived_path=archived,
                                   detail={"best_score": best_score.__dict__,
                                           "cand_score": cand_score.__dict__})
        else:
            # Best stays; archive the candidate.
            with self.db._lock, self.db._conn() as c:
                c.execute("UPDATE epochs SET status='archived' WHERE epoch_num=?",
                          (epoch_num,))
            archived = _archive(candidate_path)
            return EpochComparison(best_composite=best_score.composite,
                                   candidate_composite=cand_score.composite,
                                   winner="best", archived_path=archived,
                                   detail={"best_score": best_score.__dict__,
                                           "cand_score": cand_score.__dict__})
