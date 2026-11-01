"""Best-config export/import — the live link between ForgeEvolve and
production code (critique F21/NC8).

``python -m forge.evolution --apply-best`` reads ``forge_evolve.db`` and
writes the highest-scoring config per domain to
``forge/evolution/best_configs.json``.  Production entrypoints
(``sft_train.py``) call :func:`apply_to_namespace` to override argparse
defaults for ``# evolution-discovered`` parameters — the frozen comments
become a live file instead.

Safety: only params listed in :data:`SFT_PARAM_MAP` are applied, and only
when the user did not pass the flag explicitly.  Values matching
documented *reverted* promotions (see AGENTS.md "Reverted promotions")
are refused so a stale DB cannot silently re-promote a known-bad value.
A23 score floor: export only emits configs whose discovery recorded a
``metadata.verified.delta > 0`` win — rows without a verified promotion
never reach production defaults.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical output — versioned inside the package so it travels with the code.
BEST_CONFIGS_PATH = Path(__file__).resolve().parent / "best_configs.json"

# Allowlist mapping argparse dest → (evolution domain, config key, type).
# Only params on this list are ever overridden — everything else in the DB
# stays advisory.  Keep this in sync with the ``# evolution-discovered``
# comments in sft_train.py.
SFT_PARAM_MAP: dict[str, tuple[str, str, type]] = {
    "grad_accum": ("grad_accum_config", "accum_steps", int),
    "sync_freq": ("grad_accum_config", "sync_freq", int),
    "focal_gamma": ("loss_config", "focal_gamma", float),
}

# Reverted-promotion guards (AGENTS.md "Reverted promotions"): conditions
# under which an evolved value must NOT be applied.  Each entry is
# (argparse dest, predicate) — if predicate(value) is True the value is
# refused with a warning.
REVERTED_GUARDS: dict[str, object] = {
    # label_smoothing > 0.2 was a scoring artifact — reverted to 0.1.
    "label_smoothing_eps": lambda v: v > 0.2,
    # warmup_steps == 0 broke training stability — reverted to 500/20.
    "warmup_steps": lambda v: v == 0,
}


# Rows scanned per domain when looking for an exportable config. The
# score floor (below) rejects the top row far more often than not, so a
# single-row peek is not enough.
_EXPORT_SCAN_LIMIT = 200


def _verified_delta(row: dict) -> float | None:
    """Return the verified-win delta recorded on a discovery, else None.

    Scout/CodeEvolve write ``metadata.verified = {delta, margin, ...}``
    when a challenger beats the incumbent past a noise-aware margin.
    Rows without that block (plain archive saves, failed verifies, the
    synthetic meta-inner rows) have no promotion claim — unexportable.
    """
    meta = row.get("metadata") or {}
    ev = meta.get("verified")
    if not ev:
        return None
    try:
        delta = float(ev.get("delta"))
    except (TypeError, ValueError, AttributeError):
        return None
    return delta


def export_best_configs(
    db_path: str = "forge_evolve.db",
    out_path: str | Path | None = None,
    domains: list[str] | None = None,
) -> dict[str, dict]:
    """Read the findings DB and write the best config per domain to JSON.

    Score floor (A23): a config is only exportable when its discovery
    carries ``metadata.verified.delta > 0`` — a challenger that provably
    beat its incumbent. Unconditional top-score export would happily
    promote negative-scored rows (the main DB's discoveries are ALL
    negative) and poison production defaults. With no qualifying rows
    the file is still written honestly as ``{"configs": {}, "skipped": N}``
    rather than left stale.

    Returns the exported mapping ``{domain: {"config": ..., "score": ...}}``.
    """
    from .database import FindingsDB
    from .domain_spec import list_specs
    from .domains import DOMAINS

    out = Path(out_path) if out_path else BEST_CONFIGS_PATH
    db = FindingsDB(db_path)
    try:
        if domains is None:
            # Domains actually present in the DB ∪ registered specs —
            # spec-only iteration silently misses rows whose domain was
            # renamed/removed (e.g. the main DB's 'quant' rows) and would
            # under-report `skipped`.
            db_domains = {r[0] for r in db.conn.execute(
                "SELECT DISTINCT domain FROM discoveries")}
            domain_names = sorted(
                db_domains | set(list_specs()) | set(DOMAINS.keys()))
        else:
            domain_names = domains
        exported: dict[str, dict] = {}
        skipped = 0  # discovery rows rejected by the verified-delta floor
        for name in domain_names:
            # Rows arrive score-DESC, so the first qualifying row is the
            # best *verified* config — not merely the best-scoring one.
            chosen, delta = None, None
            for row in db.query_best_configs(name, limit=_EXPORT_SCAN_LIMIT):
                if row.get("config") is None:
                    continue
                d = _verified_delta(row)
                if d is not None and d > 0:
                    chosen, delta = row, d
                    break
                skipped += 1
            if chosen is None:
                continue
            exported[name] = {
                "config": chosen["config"],
                "score": chosen.get("score"),
                "delta": delta,
            }
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "configs": exported,
        "skipped": skipped,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return exported


def load_best_configs(path: str | Path | None = None) -> dict[str, dict]:
    """Load the exported best-configs file. Returns {} if absent/corrupt.

    Reads the current ``"configs"`` key; the legacy ``"domains"`` key is
    still accepted so older exports/tests keep working.
    """
    p = Path(path) if path else BEST_CONFIGS_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data.get("configs") or data.get("domains") or {}


def apply_to_namespace(args, defaults: dict, param_map: dict | None = None) -> list[str]:
    """Override ``args`` attributes with evolution-best values.

    Only params in ``param_map`` (default :data:`SFT_PARAM_MAP`) are
    considered, only when the current value still equals the argparse
    default (explicit user flags always win), and only when the value
    passes the :data:`REVERTED_GUARDS` checks.

    Returns a list of human-readable strings describing each override
    applied (for logging).
    """
    param_map = param_map or SFT_PARAM_MAP
    domains = load_best_configs()
    if not domains:
        return []
    applied: list[str] = []
    for dest, (domain, key, typ) in param_map.items():
        entry = domains.get(domain)
        if not entry or key not in (entry.get("config") or {}):
            continue
        current = getattr(args, dest, None)
        if current is None or current != defaults.get(dest):
            continue  # user passed the flag explicitly — never override
        try:
            value = typ(entry["config"][key])
        except (TypeError, ValueError):
            continue
        guard = REVERTED_GUARDS.get(dest)
        if guard is not None and guard(value):
            logger.warning(
                "Evolution value for --%s (%r) matches a reverted promotion; "
                "keeping default %r", dest.replace("_", "-"), value, current)
            continue
        setattr(args, dest, value)
        applied.append(
            f"--{dest.replace('_', '-')}={value} "
            f"(domain={domain}, score={entry.get('score')})")
    return applied
