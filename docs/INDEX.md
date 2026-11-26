# ForgeAI Documentation Index

**Start here:** `docs/CODEBASE_MAP.md` (task→file routing) ·
`scripts/README.md` (script index) · `.devin/rules/` (scoped edit
rules) · `.devin/config.json` (machine-readable env facts).

## Active Documents
- **AGENTS.md** — Agent operating directives (A-H), canonical layout, build/test commands
- **docs/CODEBASE_MAP.md** — Quick-find routing: task→file table, package map, data roots, freshness rules
- **scripts/README.md** — Categorized index of all scripts (regenerate: `scripts/build_script_index.py`)
- **docs/CHANGELOG.md** — Historical R&D round notes, refactor retrospectives (extracted from AGENTS.md)
- **docs/project_overview.md** — High-level project overview
- **docs/forge_engine_guide.md** — ForgeEngine inference engine guide
- **docs/ADAPTER_AUTHORING.md** — Recipe for adding a new HF model (preset + key map + forge_convert.json + bit-exact port test)
- **docs/RD_SUNSET_POLICY.md** — use_* flag sunset policy (beat the incumbent within 3 R&D rounds or the flag is deleted)
- **docs/EVOLVE_ROADMAP.md** — Evolve-loop roadmap: shipped/queued work + theory
- **docs/ENGINE_IMPROVEMENT_PLAN.md** — ForgeEngine improvement plan (research-complete, awaiting execution)
- **docs/ENGINE_CRITIQUE_DB.md** — Living catalogue of ForgeEngine weaknesses, dead code, and improvement opportunities
- **docs/FAILURE_CATALOG.md** — Hardware/environment failure log (SMR drive, zombie procs, VRAM traps)
- **docs/COMPETITIVE_AUDIT_2026-10-31.md** — External-model competitive audit driving directives + eval anchors
- **docs/evolve_research.md** — Evolve-loop web-research digest (referenced by `forge/evolution/research.py`)
- **docs/FLUX_IMPROVEMENT_ROADMAP.md** — FluxLM RSI-loop improvement roadmap (R54+)

## Design Documents (`docs/design/`)
- **V14_RSI_DESIGN.md** — ForgeLM V14-RSI ~2.5B RSI-loop coder design (preset `forgelm_v14_rsi`)
- **R71_STREAMED_INTELLIGENCE_DESIGN.md** — R71: StreamHead streaming, ECQ codec, FisherFold, LoRA-ES, LoopThink
- **LIVET_DESIGN.md** — LiveT live-training design (journaled GDN2/Titan/NGram live surfaces)
- **STRATA_M1_DESIGN.md** — STRATA M1 model-dynamics design (FluxLM successor)
- **STRATA_VS_FLUXLM.md** — STRATA vs FluxLM division-of-labor decision (audit A12) + consolidation trigger
- **FLUXLM_V2_DESIGN.md** — FluxLM V2 design notes

## Research Notes (`docs/research/`)
Long-form research notes (component atlas, technique survey, key-mapping/novelty audits, ideation docs). These double as the byte-corpus for `scripts/exp_flux*.py` experiments.
- **RSI_LOOP_BASE_LORA_RESEARCH.md** — Base-model decision (LFM3 vs Spark/Granite) + LoRA-family methods for the non-SFT RSI update path (2026-11-23)
- **RSI_LOOP_TRAINING_OBJECTIVES.md** — Training-objective layer: GRPO-variant gaps, preference objectives (DPO/KTO/oracle-weighted pairs), iterative-round failure modes (2026-11-23)
- **RSI_LOOP_MEMORY_TOPOLOGY.md** — Train/infer memory decoupling on 12GB: measured budgets, `_fq`/CUDA-graph caveats, merge-cycle canonical fix (2026-11-23)
- **RSI_LOOP_CAPACITY_EXPANSION.md** — Beating the 1.7B param ceiling: depth-graft surgery, post-hoc looping (UpLoop/Dory), MoE upcycle, param-free capability multipliers (2026-11-23)
- **RSI_LOOP_WHITESPACE_RECON.md** — Reconciled two external "white space" digests vs the codebase: what's already built, what confirms design, genuine gaps (harness-evolve genome, BES proposer, degeneracy guard) (2026-11-23)

## Archive (`docs/archive/`)
- **CODEBASE_CRITIQUE_2026-09-16.md** — Retired critique retained as corpus data for `scripts/exp_flux*.py` (read as raw bytes; do not move into `docs/research/` or the glob double-counts it)

## Bug Tracking
- **BUG_LOG.md** — Resolved bug log (see `log-bug` in `~/.codeium/windsurf/skills/` for the workflow)

## Notes
- R&D round numbering has gaps: CHANGELOG.md only keeps recent rounds
  (R60+); rounds 1-48 were rotated out in commit `0ada4ee` — see
  `git show 0ada4ee^:docs/CHANGELOG.md` and the wiki's `tried/` pages
  (Rounds-32-48, Rounds-49-70). Per-file rd_round_* notes were pruned in
  the 2026-11-20 cleanup — recover from git history or the wiki.
- Retired audits (CODEBASE_CRITIQUE_2026-09-09/-10, rd_quant_* lists,
  FFN/DIFFUSIONBLOCKS/GROQ research, RND_PLAN_V8, WebScrappers, Gemini
  notes) were removed in the same cleanup — `git log` for recovery.
