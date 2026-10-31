"""Tagged LoRA registry — the management layer for hot-swap adapters.

A registry is a directory of ``*.lora.safetensors`` adapters plus one
``manifest.json`` mapping a *tag* to its adapter + load params:

    {
      "tags": {
        "min_k_filter": {
          "file": "min_k_filter.lora.safetensors",
          "rank": 32, "alpha": 64,
          "target_modules": null,
          "base": "Qwen/Qwen3-4B-Instruct-2507",
          "trained_rows": 41, "created": "2026-02-23T22:00:00"
        },
        "default": {...}
      }
    }

Resolution is a fallback chain: exact tag → each domain tag → ``"default"``.
Base-model weights are never touched by adapters — ``ForgeEngine.load_lora``
patches forwards only — so a shared engine can hot-swap skills per request.

Used by ``LLMProposer`` (per-target adapter select) and
``scripts/finetune_proposer.py --mode lora`` (writes new entries).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LoRAEntry:
    tag: str
    file: str
    rank: int = 32
    alpha: int = 64
    target_modules: list[str] | None = None
    base: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def path(self) -> str:
        return self.file  # resolved relative to registry root by callers

    def to_dict(self) -> dict:
        return {
            "file": self.file, "rank": self.rank, "alpha": self.alpha,
            "target_modules": self.target_modules, "base": self.base,
            **self.meta,
        }


class LoRARegistry:
    """Manifest-backed tagged adapter store. Read-safe while writers run."""

    MANIFEST = "manifest.json"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _manifest_path(self) -> Path:
        return self.root / self.MANIFEST

    def _load(self) -> dict:
        p = self._manifest_path()
        if not p.exists():
            return {"tags": {}}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"tags": {}}

    def _save(self, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / (self.MANIFEST + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._manifest_path())   # atomic on same volume

    def tags(self) -> list[str]:
        return sorted(self._load().get("tags", {}).keys())

    def get(self, tag: str) -> LoRAEntry | None:
        d = self._load().get("tags", {}).get(tag)
        if d is None:
            return None
        return LoRAEntry(tag=tag, **{k: v for k, v in d.items()
                                     if k in LoRAEntry.__dataclass_fields__
                                     and k != "tag"},
                         meta={k: v for k, v in d.items()
                               if k not in LoRAEntry.__dataclass_fields__})

    def resolve(self, *candidate_tags: str) -> LoRAEntry | None:
        """First hit in the fallback chain; ``default`` is the final stop."""
        data = self._load().get("tags", {})
        for tag in [*candidate_tags, "default"]:
            if tag and tag in data:
                return self.get(tag)
        return None

    def resolve_path(self, entry: LoRAEntry) -> Path:
        p = Path(entry.file)
        return p if p.is_absolute() else self.root / p

    def register(self, tag: str, file: str | Path, rank: int = 32,
                 alpha: int = 64, target_modules: list[str] | None = None,
                 base: str = "", **meta) -> LoRAEntry:
        file = str(file)
        if not Path(file).is_absolute() and not (self.root / file).exists():
            raise FileNotFoundError(
                f"adapter file not in registry root: {self.root / file}")
        entry = LoRAEntry(tag=tag,
                          file=Path(file).name if Path(file).is_absolute()
                          else file,
                          rank=rank, alpha=alpha,
                          target_modules=target_modules, base=base,
                          meta={**meta,
                                "created": datetime.now(timezone.utc)
                                .isoformat(timespec="seconds")})
        data = self._load()
        data.setdefault("tags", {})[tag] = entry.to_dict()
        self._save(data)
        return entry

    def mark(self, tag: str, **meta) -> None:
        """Merge health stats into an entry (e.g. post-campaign acceptance
        rates the manager reads on the next plan)."""
        data = self._load()
        if tag in data.get("tags", {}):
            data["tags"][tag].update(meta)
            self._save(data)


@dataclass
class LoRAPlan:
    """Manager decision for one training group."""
    action: str                 # "new" | "update" | "retrain"
    tag: str
    rows: list = field(default_factory=list)
    init_from: str | None = None  # adapter to warm-start (update only)


class LoRAManager:
    """Decides *which* adapter a batch of new rows should train — or
    whether a new tag is justified at all.

    Policy (per row-group keyed on ``target``):
      - tag exists + healthy          -> ``update`` (warm-start adapter)
      - tag exists + ``stale`` flag    -> ``retrain`` (fresh weights, same tag)
      - tag missing + rows >= min_new -> ``new`` adapter for that target
      - tag missing + too few rows    -> fold into ``default`` (anti-spam:
        niche topics ride the shared adapter until they earn their own)

    Staleness is signaled via ``reg.mark(tag, stale=True)`` — e.g. when
    post-train acceptance regresses — or forced per-tag with
    ``plan(..., fresh={tag})``.
    """

    def __init__(self, root: str | Path):
        self.reg = LoRARegistry(root)

    def plan(self, rows: list[dict], min_new: int = 10,
             fresh: set[str] | None = None) -> list[LoRAPlan]:
        fresh = fresh or set()
        groups: dict[str, list] = {}
        for r in rows:
            groups.setdefault(r.get("target") or "misc", []).append(r)
        plans: list[LoRAPlan] = []
        for tag, rs in sorted(groups.items()):
            ent = self.reg.get(tag)
            if ent is not None:
                stale = ent.meta.get("stale") or tag in fresh
                plans.append(LoRAPlan(
                    "retrain" if stale else "update", tag, rs,
                    None if stale else str(self.reg.resolve_path(ent))))
            elif len(rs) >= min_new:
                plans.append(LoRAPlan("new", tag, rs))
            else:
                dft = self.reg.get("default")
                plans.append(LoRAPlan(
                    "update" if dft else "new", "default", rs,
                    str(self.reg.resolve_path(dft)) if dft else None))
        return plans
