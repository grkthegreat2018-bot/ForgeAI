# ForgeAI Competitive Audit — Fix Handoff

**Date:** 2026-10-31 **Status:** research-complete, findings for fix agent
**Sources:** 3-agent codebase inventory + live competitor research
(vLLM, SGLang, llama.cpp, TRT-LLM, Ollama/LM Studio, exllamav2, Unsloth,
Axolotl, LLaMA-Factory, TRL, DeepSpeed/Megatron, fla, Titans/HOPE,
MemGPT/mem0, AlphaEvolve/OpenEvolve, Optuna).
**Scope note:** `docs/ENGINE_IMPROVEMENT_PLAN.md` (2026-09-25) already covers
the decode-path/serving internals. Items that plan covers are marked
**[PLANNED]** and only referenced — do not duplicate the analysis; execute it.
Everything marked **[NEW]** is not in any existing plan/critique doc.

---

## 0. TL;DR

ForgeAI is feature-dense single-user research infrastructure. The gaps that
matter most are not missing features but **credibility, wiring honesty, and
ecosystem interop**: no external benchmark numbers exist for any claim, ~half
the advertised capability zoo is dormant wiring, and repo/CI hygiene would
fail any outside contributor. The engine's own invariants ("report effective
state, not requested state"; "no half-dead flags") are violated by the
current state.

---

## A. New findings (not in existing plans)

### A1 [NEW] No external evaluation — the credibility gap. **P0**

- `forge/evaluation/` has 3 files: a hardcoded 50-question probe
  (`ckpt_tester.py`), GOSP goal tasks, goal scorer. Zero lm-eval-harness /
  MMLU / HumanEval / MT-Bench interop. MMLU appears once, in a docstring.
- Every performance claim in AGENTS.md/CHANGELOG is measured on internal
  corpora and internal baselines. Precedent in-repo for why this is
  dangerous: the reverted evolution promotions (synthetic metrics rewarded
  trivial solutions — documented in AGENTS.md "Reverted promotions").
- **Fix:** wire `lm-eval` (EleutherAI harness) or a vendored task subset
  against `ForgeEngine` behind a small adapter; produce a baseline report
  for ForgeLM V2 (MMLU/ARC/HellaSwag/GSM8K) and store it as the regression
  gate for preset/key changes. For FluxLM: build the held-out-QA
  head-to-head vs a dense retriever + reranker baseline before any more
  channel R&D — `held_qa=0.0` at 30K docs is already marked
  "retrieval/extraction-bound" in AGENTS.md; a vanilla RAG stack very
  likely beats it on that axis, and the comparison is the point.
- **Effort:** M. **Owner of risk:** all R&D claims are unverifiable until done.

### A2 [NEW] Dormant strategy zoo — quantify + cull or wire. **P1**

Confirmed by code inspection (not just plan-suspicion):

- `KVCacheStrategy` interface (`kv_backend.py`): ~30 registered strategies.
  Their `append`/`get` are **never invoked by the decode loop** — only
  `compact_past_kv` is consulted (`engine_generation.py:~1014`,
  beacon/anchor/metakv/flux family). Actual decode KV is per-layer `(k,v)`
  tuples in `past_kv`. Docstrings still advertise rotorquant as "the
  default" — misleading about effective decode precision.
- `CudaGraphRunner` (`runtime/cuda_graph.py`): captured decode call passes
  **no `past_kv`** — graph decode path is non-functional;
  `acceleration="cuda_graph"` sets a runner generation never calls.
- `use_spec_attn` appears in TaskBootConfig/hotswap with no
  feature-registry handler — dead flag.
- Plan 1.1 already proposes making strategies "real or deletable" via
  `PreAllocatedKVCache` wiring — treat this as the *enforcement* step:
  after that lands, delete every strategy whose code path still isn't
  invoked, or gate it behind an explicit `experimental=True` flag that
  `diagnostics()` reports honestly.
- **Effort:** M (mostly deletion + test updates). Aligns with Engine
  Invariant "Forgetting is expensive" — currently violated at scale.

### A3 [NEW] Repo hygiene — externally embarrassing. **P1**

Verified against `git ls-files` / `git status` (2026-10-31):

- **`tools/llama/` tracked binaries**: `cublas64_13.dll` (55 MB),
  `cudart.zip` (423 MB), `cublasLt64_13.dll` (493 MB), ~15 `ggml-*.dll`
  — **~1 GB of CUDA/ggml binaries committed to git**. `git rm -r --cached
  tools/llama` + gitignore; keep the binaries locally for parity testing.
- **No LICENSE file** despite `license = {text = "MIT"}` in pyproject.
- **CI is near-decorative** (`.github/workflows/ci.yml`): ruff checks only
  `tests/` (all ~181K lines of `forge/` unlinted); `test-gpu` is
  `if: false`; `test-cpu` almost certainly fails — 8+ test files hardcode
  `D:\windsurf\ForgeAI` paths (`test_code_evolve.py` subprocess cwd,
  `tests/evolution/test_*.py` `sys.path.insert`) and tests need the
  gitignored `research/checkpoints/forgelm_v2_tokenizer` asset. Fix: lint
  `forge/` (ratchet: add per-directory), replace hardcoded paths with
  repo-relative fixtures, commit a tiny test tokenizer or skip-gate on
  asset presence, delete or implement the GPU job.
- **Vendored model assets inside the package**:
  `forge/evolution/proposer_ckpt/`, `forge/evolution/proposer_mla_moe/`
  (vocab.json/merges.txt/tokenizer.json ~3-6 MB) — move under
  `checkpoints/` or a `forge/evolution/assets/` download-on-demand path.
- **Three venvs** (`venv`, `venv_ov`, `venv_train`) — only venv is
  documented; no `requirements_ov.txt`/`requirements_train.txt`; aux
  venvs unreproducible. Write both files + 5 lines in README.
- **`research/`→`forge/` migration incomplete**: ~33 production `forge/`
  modules import `research.*` (tokenizer_cache, paths, sampling_utils,
  merge_models, persistent_executor, vast_connector, training_free).
  Either finish the move or delete the "legacy/experimental" framing —
  production depends on it.
- Root scratch files (`parity_*.txt`, `keycheck.txt`, `tok_test.txt`,
  `results/`, `scratch/`) unignored; `forge_evolve*.db` correctly ignored
  (the 40 GB `forge_evolve_meta_inner.db` is untracked — fine, keep it
  out of any backup/archive path).
- **Effort:** S-M total. Highest-visibility cheap fixes in this doc.

### A4 [NEW] Adopt `fla` (flash-linear-attention) kernels for GDN2/KDA. **P1**

- ForgeAI implements GDN2/KDA side-paths in PyTorch; `fla-org` ships fused
  chunked Triton kernels (varlen, fused kkt+solve, autotuned) and is the
  implementation Qwen3-Next (GDN) and Kimi Linear (KDA) actually ship.
- Plan 1.9 already vendors vLLM's Mamba Triton kernels — extend the same
  "vendor don't reinvent" policy to the delta-rule family. Keeps the keys/
  lossless-port contract while removing the kernel-maintenance tax.
- **Effort:** M. Watch `triton-windows` compatibility (SM120) — fla is
  pure Triton so it should port; spot-check first.

### A5 [NEW] Proposer model is the weakest link in CodeEvolve. **P1**

- `LLMProposer` default = Qwen2.5-Coder-0.5B; measured 17/21 verbatim
  copies (AGENTS.md). AlphaEvolve's wins come substantially from
  Gemini Flash+Pro proposer quality.
- Cheapest multiplier on the whole evolution/RSI stack: a 3-4B coder
  (Qwen3-4B preset already exists + `proposer_quantize` int4 path) or the
  distill-client teacher path as proposer-of-record for basin leaps.
- **Effort:** S (config + smoke). Do before the next RSI campaign.

### A6 [NEW] FluxLM positioning vs RAG/memory baselines. **P1**

- FluxLM's unique property is journaled bit-exact revert — nobody ships
  that (MemGPT/mem0/Letta don't do exact undo). But as an LM it is far
  below a trained transformer (EXP-4: NLL ~6.7 vs Qwen-0.5B 4.3 on the
  same token stream) and held-out QA collapses at corpus scale.
- Honest repositioning: **a revertible memory layer, not an LM** — which
  is exactly what mem0-style products are. The competitive set changes
  from "LMs" to "agent memory", where bit-exact undo + explain() cell
  provenance are differentiators no one has.
- **Fix direction:** build the RAG head-to-head (A1) + expose the
  journal/revert/provenance as the API surface (explain(),
  inspect_cell, revert_tag already exist — they're the product).
- **Effort:** eval M; repositioning is docs-only.

### A7 [NEW] No user on-ramp. **P2**

- README = 32 lines, no install/quickstart/usage example; no PyPI; no
  release artifacts; docs/ is excellent *internal* notes but a new user
  cannot run the project. If ForgeAI stays private, downgrade to P3 — but
  then stop writing for an audience (README/docs imply one).
- **Effort:** S (quickstart) → M (pip package).

### A8 [NEW] Multi-LoRA serving. **P2**

- Engine hot-swaps one adapter at a time (`engine_lora.py` unloads the
  previous). vLLM/SGLang serve hundreds of adapters on one base. The
  tagged-LoRA registry + router already exists for the proposer — the
  serving piece (per-request `adapter_id` field, multi-adapter residency
  with per-request bind) is the missing 20%.
- **Effort:** M. Plan 6b already lists per-request selection as "verify".

### A9 [NEW] Standard observability for training. **P2**

- No wandb/tensorboard/MLflow — status.json + GUI events only. Every
  training competitor (Axolotl/TRT/torchtune) emits standard metrics a
  user can graph. Optional `wandb`/`tensorboard` writer hooks in
  `training_utils.py` heartbeat are cheap and non-invasive.
- **Effort:** S.

### A10 [NEW] Model coverage strategy. **P2**

- ~5 presets + arch adapters (Qwen3/Gemma3/Llama4/LFM3) vs thousands of
  models on HF for vLLM/llama.cpp. Reasonable for single-user; the gap is
  process, not count: no documented "how to add a preset/adapter" recipe,
  so each new model is bespoke. Write the adapter-authoring guide +
  template; pick by user demand, not breadth.
- **Effort:** S (docs) + per-model M.

---

## A18 [NEW] Sandbox/ACL threat-model notes. **P2**

- `sandbox.json` grants **`forge/evolution` read_write** to model-facing
  tools — the directory containing `code_evolve.py`, `eval_pool.py`,
  `mutator.py`, and `targets/`. Model-driven tools (`/v1/tools/*`,
  agent loop) can therefore rewrite the eval machinery and promotion
  gate while `forge/engine` is read-only. `PROTECTED_ENGINE_FILES`
  protects serving files but none of the self-improvement surface —
  the part an RSI loop is most motivated to modify. Combined with A13
  (candidates can write the same files from inside evals), the
  evolution stack is unguarded on both sides.
- Regex code-scanning (`RISKY_COMMAND_PATTERNS`) is obfuscation-brittle
  (`getattr(os,'system')`, string-concat eval) — fine as a tripwire,
  not as a boundary; the AST/audit-hook layer in
  `persistent_executor.py` is the real one, and it doesn't cover the
  tool path.
- `website_whitelist: []` = all domains allowed. Scraped
  wiki/arxiv/web content is model input — indirect prompt injection
  is the realistic attack shape here (fetched text driving `file`/tool
  calls). `.env`/secrets reads are denied, good; but `research/` +
  `forge/` reads allow source+checkpoint exfil in principle.
- **Fix:** move `forge/evolution` to read_only (or carve out
  `targets/`+`eval_pool`+`code_evolve` as protected); add the evolution
  stack to PROTECTED_ENGINE_FILES; document the `file`-tool
  read-own-source choice as deliberate.
- **Effort:** S.

## B. Already-covered items — verify & execute existing plan

These are documented in `docs/ENGINE_IMPROVEMENT_PLAN.md` (status:
"awaiting execution approval" as of 2026-09-25). The audit independently
re-derived them; execute the plan rather than re-planning:

| Item | Plan ref | Audit note |
|---|---|---|
| Iteration-level continuous batching | Phase 5 | Confirmed: `_gen_lock` serializes all generation; BatchQueue is window batching; batched streaming emits one whole-text chunk. README claims "continuous batching" — it isn't. Either ship Phase 5 or fix the claim. |
| Paged/state-pool KV + CUDA-graph decode | Phases 1, 3, 4 | Confirmed dead: `CudaGraphRunner` capture passes no `past_kv`. vLLM vendored-kernel path (1.9) is the right call. |
| `response_format`/xgrammar + usage/`n>1`/`logit_bias`/penalties | 2.3, 6.4, 6b | Engine has `json_schema` param + custom FSM; `ChatCompletionRequest` still doesn't expose `response_format` (verified in code). Plan's real-xgrammar-lib recommendation is correct — the hand FSM at 65k vocab is a per-token tax. |
| `/metrics` Prometheus, version/info, health split, abort, graceful drain | 6b | Partially done since plan: `/v1/tokenize`, `/v1/detokenize`, `/version`, `/metrics` (JSON, not Prometheus text) now exist. Re-audit the 6b table against current server before executing. |
| Security (auth/CORS/sandbox) | 6.1-6.7 | `FORGE_API_KEY` gates mutating routes; server binds 0.0.0.0 — plan's auth/CORS items still open. |
| Prefix-cache/SSM snapshot bugs, batched EOS/pad bugs, lifecycle races | Phase 0 | Several marked DONE; open remainder (0.1, 0.3, 0.5, 0.8) are still the top correctness risks. |
| Launch-bound decode (~800 kernels/tok) | Phase 4 | Confirmed design; Phase 4 is the single biggest decode multiplier. |

**Process flag:** the plan is 5 weeks old and marked "awaiting execution
approval". If it is still unapproved, that decision is the blocker on all
serving-tier competitiveness — approve or descope it explicitly.

---

## C. Where ForgeAI leads — do not regress these

- **Journaled bit-exact revert** of live memory (FluxJournal, LiveT
  mark/revert/snapshot). Unique in the field; it's the moat for A6.
- **Noise-aware promotion** in evolution (replicated verify reps,
  SEM/z-score gates, monotonic incumbent bars). OpenEvolve doesn't do
  this; AlphaEvolve doesn't publish it.
- **Proposer trains on its own verified wins** (CodeEvolve → export_sft →
  LoRA hot-swap). AlphaEvolve doesn't fine-tune its proposer.
- **Native Windows** — vLLM/SGLang are Linux-only. A real niche.
- TypeSafe `/v1/systemone` compat, `/v1/engine/*` live hotswap,
  `/v1/library`, sandboxed `/v1/tools/*` + approval flow, GGUF-native
  loading, model registry w/ byte-budgeted VRAM + sleep/wake LRU.
- Honest negative-result documentation (EXP-2/3/4 verdicts) — better
  science hygiene than most published work. Keep it.

## D. Competitor cheat-sheet (who to steal what from)

| Area | Best-in-class | Steal |
|---|---|---|
| Serving throughput | vLLM | Paged/state-pool KV, iteration-level batching, full-step CUDA graphs, PDL, async readback (all already in plan) |
| Hybrid-SSM serving | vLLM mamba path | `MambaManager` state slots + `state_indices_tensor`; chunk-boundary SSM snapshots for APC |
| Single-file local UX | Ollama / LM Studio | Model pull/registry UX, `keep_alive` TTL unload |
| Quant/kernels | llama.cpp, TRT-LLM | Hand-tuned low-bit kernels; accept that unoptimized quant loses to bf16 at ≤0.5B (own EXP-1 finding) |
| Structured output | xgrammar lib (MLC) | Plan 2.3 — bitmask + rollback, replaces hand FSM |
| Fine-tuning | Unsloth | Triton-rewritten fused CE/LoRA kernels = 2-5× on exactly this GPU class |
| Config-driven training | Axolotl/LLaMA-Factory | YAML recipes, dataset registry |
| RL stack | TRL | Learned reward model, full DPO/PPO/GRPO/GKD suite |
| Distributed | DeepSpeed/Megatron | Out of scope (1 GPU) — Vast.ai connector is the right answer |
| Linear-attn kernels | fla-org | GDN/KDA fused Triton kernels (A4) |
| Test-time memory | Titans/HOPE (Google) | Baseline-compared evals; two-timescale memory (already in gsyn `_Ss`) |
| Agent memory | mem0/MemGPT/Letta | A6 repositioning targets |
| Program evolution | OpenEvolve/AlphaEvolve | Proposer quality (A5), public-problem demonstrations |
| HPO | Optuna | Multivariate TPE, ASHA pruners for Scout |

## A11 [NEW] Evolution-program economics — measured, and ugly. **P1**

Pulled live from `forge_evolve.db` + `results/rsi_lfm3*.jsonl` (2026-10-31):

- **Config-space ForgeEvolve**: 500 runs / ~9,838 evals / 120 discovery
  events. The canonical `discoveries` table holds 43 rows — **all
  quant-domain, all negative scores, `applied=0` for every row.** The
  discovery ledger shows zero applied outcomes.
- **CodeEvolve**: 54,814 candidate programs → 41,310 (75%) rejected as
  duplicates → 4,238 (7.7%) test-passing → promotions tracked per-run.
  Current LFM3 round: **0 promotions across ~10 targets** (stagnation
  stop on every one); "improvement_pct" figures (85-95%) are vs the
  ORIGINAL seed and reflect the resumed incumbent's prior gains, not new
  wins this round.
- **Wins are micro-kernels** (`top_p_filter`, `quantize_4bit`,
  `suffix_mem`, `min_k_filter`) scored per-call in µs/ms — while the
  project's own diagnosis (ENGINE_IMPROVEMENT_PLAN §0) is launch-bound
  decode at ~800 kernels/token. A 94% faster top-p filter moves system
  tok/s by ~nothing when sampling is <1% of step time. **No end-to-end
  before/after benchmark proves the evolution loop's cumulative system
  benefit.** Dead tables in the DB (`evaluations`, `gen_models`,
  `gen_model_performance` all 0 rows) corroborate partial wiring.
- **Fix:** (a) add a system-level meta-target to CodeEvolve — decode
  step e2e tok/s on a fixed bench — so evolution optimizes the metric
  that matters; (b) run a quarterly promotion-ledger audit: % of
  discoveries ever wired into the default path; (c) dedup upstream —
  75% of pipeline compute spent generating already-seen programs is a
  proposer-quality problem (see A5), not a dedup problem.
- **Effort:** S-M.

## A12 [NEW] STRATA vs FluxLM — two live non-transformer families. **P2**

- `forge/model/strata/` (STRATA, "FluxLM's successor" per AGENTS.md) and
  `forge/model/flux/` are both actively developed (FluxLM R68-70 this
  month; STRATA M1 shipped ~86 tok/s CPU-eager learn). Own rule E:
  "if two files end up doing the same thing, merge them." The split
  needs an explicit division-of-labor decision — e.g. FluxLM = research
  vehicle, STRATA = deployment artifact format — or consolidation.
  Today they duplicate: suffix-hash order cells, hedge mixture,
  teach/ingest, journal/revert, snapshot.
- **Effort:** docs decision (S) or consolidation (L).

## A13 [NEW] Eval-pool integrity — untrusted code runs unsandboxed. **P1**

- `eval_pool.py:_eval_code` (lines 76-139) runs `exec(code, ns)`
  **in-process** in persistent workers; `_worker_main` performs zero
  post-eval checks. A candidate executes with full torch/subprocess/
  filesystem/network and shares the worker's module state across
  `max_jobs` evals. Consequences, all reachable today:
  - `torch.allclose`/`time.perf_counter`/`math.*` monkeypatching
    **persists into subsequent evals in the same worker** — the test
    gate and bench of later candidates (and the incumbent's verify
    re-measure in `eval_many`) can be poisoned by an earlier candidate.
    This directly undermines the promotion-integrity invariant added
    after the ratchet incident.
  - Candidates can write `forge/evolution/targets/*.py` on disk —
    redefining their own correctness gate, persisting across workers
    AND runs.
  - `forge_evolve.db` sits at `cwd=ROOT` — writable by candidate code;
    the promotion ledger itself is corruptible.
  - `.env` file readable (env-var scrubbing doesn't cover files);
    network unrestricted.
- **Asymmetry**: self-play model-generated code DOES route through
  `research/persistent_executor.py` (sys.audit jail + fs jail + env
  scrub) — generated code is treated as hostile in one pipeline and
  trusted in the other. Full audit-jail can't apply to evals (the jail
  denies torch/subprocess which benches need) — so the fix is OS-level
  + canaries:
  1. Worker init stores refs (`torch.allclose`, `time.perf_counter`,
     `math.sqrt`, `np.dot`…); post-eval `is`-check; on drift → discard
     result, respawn, record `stage="poison"` in DB.
  2. Parent-process sha256 of the target file pre/post eval; on change
     → restore + flag.
  3. Worker `cwd` → per-worker tempdir (PYTHONPATH still resolves
     `forge.*`; nothing needs repo-root cwd).
  4. Move the findings DB outside the eval cwd (or parent-side
     hash-check).
- **Effort:** S-M. Security of the optimization loop itself.

## A14 [NEW] KV dormancy quantified — 3/30 strategies are live. **P1**

- Of 30 registered names in `build_kv_cache` (`kv_backend.py:387-493`),
  only `beacon`, `anchor`, `metakv` implement `compact_past_kv` — the
  sole hook the decode loop calls (`engine_generation.py:1021-1283`).
  The other 27 strategy objects are constructed and reported in
  `info()`/diagnostics but their `append`/`get` are never invoked —
  decode KV stays per-layer `(k,v)` tuples. The engine knows: live logs
  show `kv_cache 'paged' has no live compaction hook, so an in-flight
  long generation can still OOM` during the RSI campaign — the runtime
  itself surfaces the dormancy as a VRAM warning.
- Sharpens A2's cull: either implement `compact_past_kv` (the real
  bridge) for survivors or delete. Also means `kv_bits`/diagnostics
  report configured-not-effective state — violates the top engine
  invariant verbatim.
- **Effort:** M.

## A15 [NEW] Decode profile verified — ~800 launches + 1 hard sync/token. **P1**

- Independently re-derived (matches ENGINE_IMPROVEMENT_PLAN §0):
  `mamba_probe._selective_scan_ref` (lines 135-189) is a pure-Python
  T-loop — ~15 kernels for the scan alone at T=1, ~30/layer incl.
  projections/conv/Jamba norms × 26 layers ≈ 780, + 2 attention layers
  with `torch.cat` KV growth (layers.py:320-321, O(T²) copy), + head
  GEMM over 65536 vocab + sampling chain.
- Hard syncs: `next_token.item()` every token
  (`decoding.py:381,505`); the ngram spec path is worse —
  `.item()` at :592, :597, :617, :641, :650 per step.
- Conclusion: the plan's Phase 4 (full-step CUDA graph) + 4.6 (async
  readback) are correctly aimed and remain the highest-value serving
  work — 800 launches → ~1 graph replay is the difference between
  research-engine and production-engine tok/s on this exact hardware.
- **Effort:** as planned (L).

## A16 [NEW] R&D sunset policy — the flag accretion problem. **P2**

- `ActivationConfig` has ~55 `use_*` flags (all default False);
  `_auto_activate_optimal` promotes ~8-12 for packed models → ~15-20%
  ship rate. FluxConfig similar (gcell/gsyn/gsyn_deep/dnet/rev/gv all
  default-off). Defensible hit rate for research, but each flag costs
  wiring surface, tests, and the dormant-zoo problem (A2). The codebase
  has the measurement infra to enforce a policy that doesn't exist:
  **a flag that hasn't beaten the current default on a fixed bench
  within N rounds gets removed, not just left off.** Combine with A2's
  cull.
- **Effort:** policy is free; enforcement rides A2.

## A17 [NEW] Self-play capability telemetry gap. **P2**

- The RSI loop reports `tasks_done`/`curriculum_validated` (live: ~37%
  proposal validation rate) but no capability-trajectory metric — is
  the model better at epoch N than epoch 0? No fixed held-out probe is
  re-run per epoch and logged. Same shape as A1: self-improvement is
  asserted by activity, not measured by outcome. Fix: a frozen eval set
  scored at every epoch boundary, appended to status.json — cheap,
  closes the loop's own feedback.
- **Effort:** S.

## F. Appendix — FluxLM-vs-RAG eval protocol (for A6)

Design for the head-to-head that settles FluxLM's positioning:

**Baseline arm (the "prof" competitor for the memory layer):**
- Embedder: `bge-small-en-v1.5` or `e5-small` (~33M params, CPU-ok).
- Store: FAISS flat / Chroma, 256-512-token chunks, top-k=8.
- Reader: the SAME generator ForgeAI serves (ForgeLM V2 or
  Qwen2.5-0.5B) so the comparison isolates the memory mechanism, not
  the model. Optional strong arm: + `bge-reranker-base`.
- FluxLM side: existing served corpus + `teacher_hard_arb` decode.

**Corpus + questions:**
1. The existing 30K-doc pool from R70 — reuse its 80-question held set
   (63/80 answers in-corpus; the "retrieval-bound" result is the thing
   to beat).
2. A public set (NQ-open subset or the scrape_feed wiki stream) so the
   eval isn't corpus-circular.
3. **Paraphrase set** — generalization (FluxLM's weak axis per R69:
   54-80% paraphrase generalization).
4. **Freshness set** — facts ingested after snapshot (live-write
   advantage both sides nominally share; RAG re-index is cheap).
5. **Correction/undo set** — ingest fact v1, then supersede with v2.
   RAG has no clean undo (delete+rebuild); FluxLM has `revert_tag`.
   This is the differentiator that justifies the architecture —
   measure it, don't assert it.

**Metrics:** EM/F1 per arm; retrieval recall@8 (RAG) vs fired-cell
provenance hit-rate (FluxLM `explain()`); ms/query; GB resident
memory; **revert-latency** (time to retract a poisoned doc); ingest
tok/s (write cost — FluxLM ~4.5-7.5K tok/s bulk vs embed+index).

**Arms:** (a) FluxLM alone, (b) RAG alone, (c) hybrid — FluxLM as
journalled provenance/undo layer + RAG for coverage. Prediction to
falsify: (b) wins held_qa/paraphrase, (a) wins verbatim recall + undo
latency, (c) is the honest product shape.

## E. Ranked execution order

1. **A1** external evals (unblocks credibility for everything else)
2. **Approve/descope the ENGINE_IMPROVEMENT_PLAN** (5-week-old decision
   blocking all serving items); re-audit 6b vs current server
3. **A3** repo hygiene (cheap, high-visibility)
4. **A5** proposer upgrade (S effort, multiplies RSI/evolution)
5. **A11** evolution system-level meta-target + promotion-ledger audit
   (cheap; redirects the whole RSI program at metrics that matter)
6. **A13** eval-pool integrity canaries (protects the promotion ledger
   that A5/A11 depend on)
7. **A2+A14** dormant-zoo cull (enforce the invariants already written;
   KV zoo is 3/30 live)
8. **A4** fla kernel adoption (with plan 1.9 as the pattern)
9. **A6** FluxLM repositioning + RAG head-to-head (+A12 consolidation
   decision at the same time)
10. Phase 4/5 decode core + continuous batching (A15 confirms the
    diagnosis — the real throughput win)
11. A18 sandbox ACL tightening (S; protects the same ledger as A13)
12. A8-A10, A16-A17 as capacity allows
