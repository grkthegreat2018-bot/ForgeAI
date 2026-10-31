# Evolve-Loop Roadmap — ideas & plans

Tracking what shipped, what's queued, and the theory behind each.
Ground truth lives here; AGENTS.md carries the stable summaries.

## Shipped (verified)

- **LFM3-1.7B proposer** — `lfm3_1p7b_cc` ForgeAI port (fp32 logit parity
  vs HF = 3.6e-5), ForgeQuant `_fq` packed dir (1.43GB), LoRA hone
  attaches adapters (rank-derivation reads the file, not the manifest).
- **Auto-targets** — `auto_targets.py`: `@self_improve` harvester →
  `targets/auto_*.py` specs from live production source; 12 functions
  annotated, merged into `rsi_evolve` by default.
- **Web-research tools** — `SEARCH: <query>` intercept in the proposer
  (budget+cache capped), `research.py` free keyless chain
  (ddgs → arXiv + Semantic Scholar → paid tavily/exa fallback),
  `docs/evolve_research.md` digest injected into prompts.
- **Lifecycle** — `LLMProposer.release_engines()` frees proposer VRAM
  between evolve and hone phases; `safety_margin_gb=3` keeps the
  desktop compositor alive.
- **Correctness guards** — `_lora_base_ok` (wrong-base adapters skip
  instead of crash; `_fq` ≡ base dir for packed weights), checkpoint-
  recompute `_pad_mask` fix, torchao int4 contiguity patch.
- **Steal-first round 1** (2026-10): FunSearch k=2 parents (`_parents`
  → `shots` = parent + next-best sorted v0/v1, "write v2"), STaR repair
  regen (error detail → one re-gated retry, `repair_regen`/`repair_
  per_batch=4`, works in basin/serial/epool), engine_tools web
  fallback (`free_search`→Exa search chain; `jina_fetch`→`trafilatura`
  scrape chain — dead paid keys no longer mean dead tools).
- **Research digests** — `.devin/scratchpad_research_{webapis,quant,rsi}.md`:
  search/fetch backend rankings, quant survey (GemLite+HQQ = the batched-
  decode winner, Marlin needs a build), proposer model survey (keep LFM3,
  A/B vs Qwen2.5-Coder-1.5B), ≤2h-training verdict (124M yes, >200M only
  via warm-start+distill — `sft_train --distill` is parsed but STUBBED),
  and the steal-first list below.
- **ForgeQuant upgrades** (2026-10): fused batched Triton GEMM
  (`_fq_int4_gemm_kernel`, M=2-32 dispatch — measured 3.96×@M=8,
  2.86×@M=32, 1.70×@M=1 on RTX 5070 vs dequant+cuBLAS) + opt-in
  Hadamard rotation (`use_rotation=True`/`--rotate`, 2.3× lower
  forward error on outlier data, flagged via `fq_rotation` in
  forge_convert.json — old packs stay compatible).
- **Proposer tools v2**: `FETCH: <url>` (jina→trafilatura, ~1.5KB
  injected) + `INSIGHT: <topic>` (code_insights DB query, local —
  no web budget) alongside `SEARCH:`, shared caps/cache.
- **Non-code domains** — `forge/evolution/domains/`: `cev_math_solver`
  (accuracy bench) + `cev_rle_codec` (compression-ratio bench),
  `Target.resolve` checks targets/ then domains/, both in
  DEFAULT_TARGETS — the loop isn't code-only anymore.
- **Audit fixes**: B1/B2/B3 LoRA/quant bugs (unload del→None, stale
  tag marker clear, block_recon guard) + `scripts/show_winners.py`.
- **Steal-first round 2** (2026-10): hint bandit (`_HINTS` Beta arms,
  credit persisted to `mutator_op_stats` — meta-prompt evolution),
  eval-cascade cheap gate (`cheap_bench`/`cheap_reps`/`cheap_margin`
  — reduced-rep bench before full), FunSearch island cull+clone
  (worst-half discard + best clone, `island_cull`), MAP-Elites bins
  (code-length quartiles per island, `map_elites`), `sft_train
  --resume`+SIGINT emergency-save + atomic lora/export writes.
- **Docs refresh** (2026-10): `forge_engine_guide.md`/`project_overview.md`
  un-staled (forge/ paths, V2→V14 chain), INDEX.md filled, durability
  contract + canonical-loop docstring added.
- **Prompt-contract round** (2026-10): hardened `_extract_code`
  (think-strip, leaked ChatML markers, fence-tag variants, required-name
  preference, prose/indent salvage), primed prompts ChatML-wrapped with
  the ```` ```python ```` prime inside the assistant turn, conditional
  output contract (primed = continue the body, unprimed = one fenced
  block), gate-side repair regen for syntax/missing/incomplete/
  undefined/dup rejects, same-batch dedup (`dup_batch`, `gen_dup`),
  `_tried_gists` anti-dup prompt line, proposal telemetry
  (raws/gated/valid_pct/stages), per-prompt prime preserved in basin
  batches, `export_validity_pairs` → `<sft>.validity.jsonl`
  (chosen=gated sibling, rejected=own malformed output — identical
  chosen/rejected pairs are rerouted to a different sibling or skipped),
  and `llm_domain_gen` hardened to the same wrap+prime+extract path.

## Queued / in-flight

### Steal-first round 2 remainder
- [ ] Randomized bench inputs per eval — reward-hack armor (Sakana lesson)
- [ ] API-teacher ensemble proposals via `forge/distillation` free-tier
     clients (Groq/NIM/OpenRouter)
- [ ] Code-fragment archive (winning snippets by tag) vs insight text

### Quant / proposer path
- [ ] GemLite+HQQ proposer path (research's winner for B=8-32 decode):
     `pip install gemlite hqq`, 20-line bench B∈{1,8,16,32} vs
     cached-dequant bf16, then `ForgeGemLiteLinear` w/ LoRA hook if it wins
- [ ] Hadamard-rotation ForgeQuant variant (steal from QuaRot —
     spreads outliers, could delete the sparse path entirely)
- [ ] Model A/B harness: ~50 real proposer prompts, LFM3 vs
     Qwen2.5-Coder-1.5B (near-free port — hit `forge_convert.json` preset)
- [ ] Fresh Tavily free key (1k/mo) + optional Jina key (500 RPM) in `.env`

### Training speed / distill
- [ ] Implement `sft_train --distill` (top-50 chunked-KL, spec'd in help
     text — unlocks >200M warm-start+KD training in ≤2h window)
- [ ] 2h sprint experiment if wanted: modded-stack ~124M (muon_sf +
     chunked CE + grad ckpt, seq-curriculum) — GPT-2-small fluency, NOT
     a coder; the real unlock is Inheritune-style warm-start

### Audit findings (2026-10, `.devin/scratchpad_audit.md` content in session)
- [x] FIXED B1: `unload_lora` `del lora_adapter` → `= None` (crash on
     ForgeQuant/NF4 forwards after standalone unload)
- [x] FIXED B2: stale `_evolve_lora_tag` — unload now clears the evolve
     markers so next swap re-attaches
- [x] FIXED B3: `block_recon._apply_optimized_params` —
     `_invalidate_cache` hasattr-guarded, falls back to `_cached_weight=None`
- [x] DONE: `scripts/show_winners.py` — human review of promoted code
- [x] DONE: `sft_train --resume` + SIGINT emergency-save (`--resume`
     consumes the `.train.pt` sidecar via `load_training_state`;
     SIGINT saves `interrupt_step<N>`)
- [x] DONE: atomic writes — `.lora.safetensors` save + `export_sft`/
     `export_pairs`/`export_validity_pairs` output (tmp+`os.replace`)
- [x] DONE: docs cold-onboarding — `forge_engine_guide.md`/
     `project_overview.md` un-staled, INDEX.md filled, R66 static-KV +
     LoRA lifecycle documented in AGENTS.md
- [ ] Hygiene top-10: two quant packages vs one protocol, two decoding
     roots, novel_quant_r44-48 versioned siblings, code_evolve 2.7k
     lines, vendored ckpt dirs inside `forge/evolution/`
- [ ] Orphan-hazard: eval workers survive `taskkill /F` (daemon procs
     clean on exceptions only) — job-object or parent-death watcher
- [ ] LLM-judge scoring toggle (same-model separate-context critique as
     a second score channel — needs an idle-context plan)

### Previously queued
- [ ] Bench: `_fq` cached-dequant bf16 vs torchao int4 vs bf16 on
     B=32 batch decode — pick the fastest proposer path. Needs a
     quiet GPU window.
- [ ] SEARCH adoption watch — if LFM3 still never emits `SEARCH:`,
     escalate to a forced few-shot example inside `_with_tools`
     (not just a contract line + hint).
- [ ] Remaining ~10 LoRA plans from the first hone (killed after
     verification) — rerun `finetune_proposer --mode lora` to
     complete per-target adapters + router.
- [ ] Expand auto-target coverage — annotate the next batch of
     hot-path functions (sampling chain, KV ops, tokenizer paths).
     Each annotation = more win surface for `--min-new` gating.
- [ ] `prequant_forge_quant.py` quality upgrade — AWQ/GPTQ-style
     activation-aware rounding (currently absmax RTN). The dense
     path is plain round-to-nearest; minimizing *output* error is
     the free accuracy.
- [ ] LoftQ/QA-LoRA init — initialize adapters to absorb quant error
     so int4+LoRA ≈ fp16+LoRA (the "no-sacrifice" tier). ~50 lines
     inside `_lora_mode`/finetune_proposer.

## Theory — where the gains compound

### Proposer skill (the flywheel)
1. LoRA-SFT on verified wins (live, ~32-row gate)
2. Per-target tagged adapters + router (once several land, the
   proposer specializes — the real RSI payoff)
3. DPO on `code_programs` chosen/rejected pairs when SFT plateaus
   (`dpo_align.py` exists, data accumulates already) — both
   `export_pairs` (child>parent quality) and `export_validity_pairs`
   (format compliance) feed it
4. GRPO next tier (`rlvr_train.py`) — reward = verified promotion

### Throughput (more candidates → more wins)
- `_cache_w` fixed the unfused dequant path; watch `[Batch]` ms.
- If decode still lags: `basin_batch 16` for lower latency, or keep
  32 for raw throughput.
- Chunked prefix cache shares ~90% of basin prompt prefix — the
  ~10x prefill amortization is already on.

### Win quality (promos that mean something)
- `ms_per_call` under GPU contention is noisy — if promotions look
  spurious, serialize CUDA-target evals into LLM-idle windows.
- Weight target budgets by production call frequency
  (sample_pipeline = every token >> ece = rare) via TAGS.
- Oracle hardening: double-gen inputs defeat naive memoization;
  add timing-floor sanity (a win >10x seed is suspicious) if gaming
  appears.

### Web research (the model pulls external technique)
- Keys: tavily 432 / exa 401 dead — free chain carries it; refresh
  keys when convenient (`.env` TAVILY_API_KEY / EXA_API_KEY).
- Optional: FIRECRAWL_API_KEY exists in `.env` — page-fetch backend
  for `research.py` (deeper dives than snippets; e.g., pull a
  technique's full impl notes into the proposer context).
- GitHub code search (unauth, 10/min) as a 4th backend — real
  implementations beat snippets for kernel work.

### Tokenizer note (GigaToken ask)
- Can't swap on a trained ckpt — tokenizer ↔ embeddings bound.
  LFM3 keeps its 128K ChatML vocab.
- GigaToken-style large-vocab is a *pretrain* decision — a real
  direction for the ForgeLM v15+ lineage (decoupled lightweight
  output head so ~1.4M vocab doesn't explode params).
- Nearest applicable: tokenizer-throughput auto-target for
  sft_train's tokenize path.

## Runbook

```powershell
# campaign (current shape)
.\venv\Scripts\python.exe -u scripts\rsi_evolve.py `
  --model research\checkpoints\lfm3_1p7b_cc_fq `
  --lora-root forge\evolution\loras_lfm3 `
  --proposer basin --llm-every 1 --basin-batch 32 `
  --min-new 32 --finetune-every 1 --research `
  --db forge_evolve.db --status results/rsi_lfm3 --minutes 60

# regenerate auto-target specs
.\venv\Scripts\python.exe -m forge.evolution.auto_targets

# refresh research digest
.\venv\Scripts\python.exe -m forge.evolution.research

# finish remaining LoRA plans (same as the verified run)
.\venv\Scripts\python.exe scripts\finetune_proposer.py `
  --db forge_evolve.db --data forge_evolve.rsi_sft.jsonl `
  --force-export --mode lora `
  --lora-root forge\evolution\loras_lfm3 `
  --base research\checkpoints\lfm3_1p7b_cc `
  --lora-steps 40 --lora-r 16 --lora-alpha 32
```

## Durability & resume

What survives a kill (SIGKILL, power loss, crash mid-campaign):

- `code_programs` rows — every candidate is INSERTed per eval
  (`_log_program`: code, metrics, score, stage, prompt, parent_hash),
  so syntax rejects, dups, and finished evals are all on disk.
- `code_insights` rows — verified wins land immediately; the
  cross-domain insight retrieval for the next run is intact.
- `mutator_op_stats` — the mutator bandit (alpha/beta per op+tag_sig)
  is persisted, so op selection warm-starts across runs.
- Incumbent resume — `cfg.resume=True` (default) re-gates the all-time
  best for the target through the CURRENT test, re-measures it, and
  repopulates island elites + the dedup set from `code_programs`. A
  killed run picks up where the DB left off, not from the seed.
- WAL durability — the DB runs `PRAGMA journal_mode=WAL`
  (`forge/evolution/database.py`): committed rows survive the process,
  and status/ro readers don't block the writer.

What is lost:

- In-flight evals — subprocess/eval-pool candidates die with the run;
  nothing is logged for them (no partial `code_programs` row).
- Per-island bandit arms — `self._arms` (Beta weights for island
  selection) is in-memory only; it resets to uniform on restart even
  though the elite pools themselves are repopulated.
- The `runs` table row — `db.save_run()` only fires at the end of
  `run()`, so a crash orphans the `code_programs` rows under their
  `run_id` (they still feed resume/dedup; there's just no run summary).

`sft_train` resume — SHIPPED:

`forge/training/runners/sft_train.py --resume <ckpt>` consumes the
`.train.pt` sidecar via `load_training_state` (weights + optimizer +
RNG + EMA + step, `start_step=step+1`, overrides `--checkpoint`), and a
SIGINT handler emergency-saves `interrupt_step<N>` (dpo_align pattern).
A killed proposer-SFT job now resumes from the last saved step.
