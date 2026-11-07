# ForgeAI Documentation Index

## Active Documents
- **AGENTS.md** — Agent operating directives (A-H), canonical layout, build/test commands
- **docs/CHANGELOG.md** — Historical R&D round notes, refactor retrospectives (extracted from AGENTS.md)
- **docs/CODEBASE_CRITIQUE_2026-09-09.md** — 25-issue audit driving the 2026-09-09 refactor
- **docs/CODEBASE_CRITIQUE_2026-09-10_followup.md** — follow-up critique pass
- **docs/CODEBASE_CRITIQUE_2026-09-16.md** — second critique pass
- **docs/project_overview.md** — High-level project overview
- **docs/forge_engine_guide.md** — ForgeEngine inference engine guide
- **docs/ADAPTER_AUTHORING.md** — Recipe for adding a new HF model (preset + key map + forge_convert.json + bit-exact port test)
- **docs/RD_SUNSET_POLICY.md** — use_* flag sunset policy (beat the incumbent within 3 R&D rounds or the flag is deleted)
- **docs/EVOLVE_ROADMAP.md** — Evolve-loop roadmap: shipped/queued work + theory
- **docs/ENGINE_IMPROVEMENT_PLAN.md** — ForgeEngine improvement plan (research-complete, awaiting execution)
- **docs/ENGINE_CRITIQUE_DB.md** — Living catalogue of ForgeEngine weaknesses, dead code, and improvement opportunities

## Design Documents
- **docs/V14_RSI_DESIGN.md** — ForgeLM V14-RSI ~2.5B RSI-loop coder design (preset `forgelm_v14_rsi`)
- **docs/LIVET_DESIGN.md** — LiveT live-training design (journaled GDN2/Titan/NGram live surfaces)
- **docs/STRATA_M1_DESIGN.md** — STRATA M1 model-dynamics design (FluxLM successor)
- **docs/STRATA_VS_FLUXLM.md** — STRATA vs FluxLM division-of-labor decision (audit A12) + consolidation trigger

## R&D Round Notes (chronological)
- `rd_round_32.md` through `rd_round_49_plan.md` — Per-round research notes
- `rd_quant_master_list.md` — Master list of quantization algorithms
- `rd_quant_plug_and_play.md` — Plug-and-play quantization reference
- `rd_round_44_quant.md` / `rd_round_45_research.md` / `rd_round_45_results.md` /
  `rd_round_46_results.md` / `rd_round_47_results.md` / `rd_round_48_results.md` —
  Quantization R&D rounds 44-48

## Research Documents
- `FFN_RESEARCH.md` — FFN architecture research
- `DIFFUSIONBLOCKS_RESEARCH.md` — DiffusionBlocks research
- `GROQ_DISTILLATION_RESEARCH.md` — Groq distillation research
- `Gemini deep research.md` — Gemini deep research notes
- `RND_PLAN_V8_AND_NOVEL_ARCH.md` — V8 and novel architecture plan
- `WebScrappers.md` — Web scraping tools reference
- `evolve_research.md` — Evolve-loop web-research digest
- `FLUX_IMPROVEMENT_ROADMAP.md` — FluxLM RSI-loop improvement roadmap (R54+)
- `docs/research/` — Long-form research notes (component atlas, technique survey, key-mapping/novelty audits, ideation docs)

## Bug Tracking
- `BUG_LOG.md` — Resolved bug log (see `log-bug` in `~/.codeium/windsurf/skills/` for the workflow)

## Notes
- R&D round numbering has gaps: CHANGELOG.md only keeps recent rounds
  (R60+); rounds 1-48 were rotated out in commit `0ada4ee` — see
  `git show 0ada4ee^:docs/CHANGELOG.md` and the wiki's `tried/` pages
  (Rounds-32-48, Rounds-49-70). Rounds 32+ also have individual files here.
- For the full critique driving the 2026-09-09 refactor, see
  `CODEBASE_CRITIQUE_2026-09-09.md`
