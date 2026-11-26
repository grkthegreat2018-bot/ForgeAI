# Flux/EXP Rules — research vehicles, not production

## Mental model
- **FluxLM** (`forge/model/flux/`) = research vehicle (associative
  memory); **STRATA** (`forge/model/strata/`) = deployment container
  format. Division of labor: `docs/design/STRATA_VS_FLUXLM.md` (A12).
- **`forge/exp/` + `scripts/exp_*.py`** = the EXP experiment surfaces
  (EXP-1..8, documented in AGENTS.md sections ~1743+). Baseline model
  for EXP-1 is `.devin/qwen25-05b/` (kept deliberately — don't delete).
- FluxLM is semi-dormant vs the SPARK self-play mainline — check
  `docs/design/FLUXLM_V2_DESIGN.md` + `docs/FLUX_IMPROVEMENT_ROADMAP.md`
  for the live queue before assuming a lane is open.

## Rules
- Results log to `.devin/exp_results.jsonl` via `log_result`
  (`forge/exp/harness.py`) — ad-hoc prints don't count.
- `scripts/exp_flux*.py` read `docs/research/*.md` +
  `docs/archive/CODEBASE_CRITIQUE_2026-09-16.md` as a BYTE CORPUS —
  never move the archive file into `docs/research/` (glob double-counts)
  or reorder it (breaks train/test slice comparability).
- `flux_domains.py` `_CORPUS_PATH` → `.devin/test_corpus.txt` (kept).
- Params are never written by ingest — only journaled buffers move
  (LiveT contract; `mark()/revert()/snapshot()/load()` are exact).
