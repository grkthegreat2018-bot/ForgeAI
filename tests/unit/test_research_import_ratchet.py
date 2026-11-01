"""A3 ratchet: `research.*` imports inside `forge/` may only shrink.

`research/` is a legacy package that still holds live modules
(paths.py, tokenizer_cache.py, merge_models.py, persistent_executor.py,
training_free/, vision/, ...). Migrating all ~44 import sites in one pass
is unsafe, so the boundary rule is:

  * NEW code in ``forge/`` must not add ``from research.`` / ``import
    research.`` imports — new shared helpers belong under ``forge/``.
  * Existing sites shrink opportunistically as modules get ported.

This test pins the current count and fails if it grows.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FORGE = REPO / "forge"

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+research\b", re.M)

#: Baseline measured 2026-10-31 (competitive audit A3). Lower is progress;
#: bump the pin DOWN when sites are migrated — never up.
RESEARCH_IMPORT_BASELINE = 61


def _count_research_imports() -> int:
    total = 0
    for path in FORGE.rglob("*.py"):
        total += len(_IMPORT_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
    return total


def test_research_imports_do_not_grow():
    n = _count_research_imports()
    assert n <= RESEARCH_IMPORT_BASELINE, (
        f"forge/ now has {n} `research.*` import sites "
        f"(baseline {RESEARCH_IMPORT_BASELINE}). New forge/ code must "
        "import shared helpers from forge/, not research/ — see "
        "test_research_import_ratchet.py and AGENTS.md canonical layout."
    )
