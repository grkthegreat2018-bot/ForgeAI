# ForgeAI — Agent Notes

## Agent Operating Directives (READ FIRST)

These directives govern how work is done in ForgeAI. They are non-negotiable
unless the user explicitly overrides them for a specific task.

### A. Model Versioning — Build On The Prior, Never Beside It
- **Every new custom model version MUST be derived from the immediately
  preceding version**, carrying forward all prior keys/architecture as the
  baseline, then adding or replacing only what's new. Current chain:
  `forgelm_v2` (Jamba base) → `forgelm_v12_jamba` (R37 keys) →
  `forgelm_v13_gdn2` (R60: Gated DeltaNet-2 side-paths, KDA-superset) →
  `forgelm_v14_rsi` (V14: ~2.5B RSI-loop coder, conv/mamba3/GLA + MoE-FFN +
  CopySpan/Verifier heads — non-lossless new topology, all divergences
  documented in dropped_keys; see docs/V14_RSI_DESIGN.md).
- **Port-first, train-second**: when introducing a new architecture key or
  attention variant, write the lossless checkpoint-conversion path
  (`XxxKey` class, identity/zero-init warm start, bit-exact load test)
  **before** any training experiment. This eliminates the "train first, port
  later" tax that has bitten past rounds.
- **No silent regressions**: a new preset that drops a prior key must
  document WHY in the preset line and in the R&D round notes. Dropping keys
  silently is a bug.
- **Preset lineage check**: before merging a new preset, run a bit-exact
  forward-pass comparison against the prior preset's checkpoint on the BSP
  base. Max logit diff must be 0.0 (lossless) unless the preset is
  intentionally non-lossless (document the delta).

### B. Confirm-Then-Fix — Never Leave A Known Bug Sitting
- **When you find an issue, confirm it** (reproduce with a minimal script or
  test) **then fix it in the same session**. Do not log it and move on.
- If a fix would be large/risky, scope a minimal failing test first, then
  implement the smallest correct fix. Prefer targeted edits over rewrites.
- **Always add or update a test** for the fix so it cannot silently return.
  Tests live in `tests/unit/` and run on CPU where possible.
- If a fix is genuinely blocked (needs user input, env change, or a
  destructive op), say so explicitly and create a `todo` — do not pretend
  it's done.

### C. R&D Is The Default Mode — Push For Novel Improvements
- **No area is "solved"**. Every existing technique (attention, KV cache,
  quantization, decoding, optimizer, loss, scheduler) is fair game for a
  novel variation. The codebase already has 13 R&D rounds; round 14+ is
  expected, not exceptional.
- **Prefer novel over copy**: when implementing a known technique, always
  ask "what's the novel twist that could beat the paper's number on our
  specific hardware (RTX 5070, 12GB, SM120, Blackwell)?" Implement the
  baseline AND at least one novel variation in the same round.
- **Cross-domain combinations are the highest-value R&D** — see the
  expanded Novel Discovery Protocol in `docs/CHANGELOG.md`.
- **When stuck on a hard optimization, pivot don't quit**: if 2 iterations
  fail to beat the known best, shelf it in `.devin/scratchpad.md` with the
  failed approaches documented, then touch up a *different* area. Fresh
  context often surfaces the missing idea. Return to the hard problem later.
- **Record failures as carefully as successes** — a documented dead end
  saves the next session hours. Use `.devin/scratchpad.md`.

### D. GPU/VRAM — Mixed Approaches Are Mandatory To Consider
- **Never propose a GPU-only or CPU-only solution when a mixed approach is
  viable.** The hardware is RTX 5070 12GB VRAM + 32GB system RAM + pinned
  CPU offload (`hybrid_offload.py`, `cpu_kv_offload.py`). The optimal
  operating point is almost always a split.
- **Always state the VRAM budget** for any new inference or training
  feature. If a feature pushes past 12GB on the 1.2B model, it MUST offer
  a mixed CPU/GPU fallback path (e.g. CPUAdamW for training, CPU KV
  offload for inference, BitNet int8 for weights).
- **Quantization is a first-class citizen**, not a fallback. BitNet b1.58,
  W8A8, NVFP4, OffQ, AAAC, SharQ, MosaicQuant are all production paths on
  this hardware — prefer them over "just use a smaller model".
- **Profile before assuming**: use `torch.cuda.memory_allocated()` /
  `torch.cuda.max_memory_allocated()` in test scripts. Guessing VRAM is
  how we get OOM at 3am.
- **Storage constraint — D: is SMR + USB-BOT**: the 4TB external
  (`WD Elements 2620`) is a device-managed SMR HDD on `USBSTOR`
  (no UASP, no command queue). Sequential ≈230 MB/s, but small-file
  reads ≈38 files/s and random writes collapse <10 MB/s once the
  drive's CMR cache fills. Keep hot random I/O (many-small-file
  caches) on C: NVMe; D: is for sequential data (checkpoints,
  datasets, packed single-file caches like `DiskTokenCache`).
  See `docs/FAILURE_CATALOG.md` 2026-11-05 entry.

### E. No Redundant Files — Search Before You Create
- **Before creating ANY new script or module, grep the codebase for an
  existing one that does the same thing.** The codebase has ~1,000 .py files
  after the aggressive refactor (was 908); the odds are high that a related
  implementation exists.
- **Prefer upgrading an existing file over spawning a new one.** If
  `forge/inference/kv/snapkv.py` exists and you want "smarter SnapKV",
  edit that file — do not create `snapkv_v2.py` or `smart_snapkv.py`.
  Versioned filenames fragment the codebase and hide the canonical path.
- **If two files end up doing the same thing, merge them** and delete the
  inferior one. Document the merge in the "Removed (consolidation)" section.
- **The canonical path wins**: when in doubt, the file already wired into
  `forge_engine.py` / `forge_server.py` / `sft_train.py` is canonical. New
  code hooks into those, not around them.

### F. Math Thinking + Script Testing — Find The True Optimum
- **Every optimization claim must be backed by a number from a script**,
  not a paper citation. Papers report numbers on different hardware/models;
  our numbers come from RTX 5070 + ForgeLM V2 (Jamba-3B).
- **Write the smallest possible test script first** (see Novel Discovery
  Protocol step 1 in `docs/CHANGELOG.md`). A 20-line script that runs in 5 seconds > a 200-line
  design doc.
- **Do the math by hand for small cases** before trusting a benchmark.
  If a KV compression scheme claims "15×", verify: `bytes_full /
  bytes_compressed` with the actual dims (d_model=2048, head_dim=64,
  8 KV heads, 16 layers). Catches off-by-one and per-layer-vs-total
  confusion.
- **Sweep parameters, don't guess them**: when a technique has
  hyperparameters (rank, block size, threshold, alpha), write a sweep
  script that tries 5-10 values and plots/logs the result. The optimum is
  rarely the paper's default.
- **Compare against the RIGHT baseline**: a new KV cache must beat the
  current production cache (`s4r` for V4), not a strawman. A new optimizer
  must beat `muon_sf_plain` (V3) or `cpu_offload` (V4), not raw AdamW.
- **No area is truly solved**: even "obviously optimal" choices (RoPE
  theta, head_dim, SwiGLU intermediate ratio) deserve a periodic
  re-challenge. If the improvement is hard to find, shelf it and touch up
  another area — fresh ideas surface indirectly.

### G. General Best Practices For This Codebase
- **Run the test suite before declaring done**: `pytest tests/ --tb=short -q`
  with the venv python and `PYTHONPATH=D:\windsurf\ForgeAI`.
- **Keep AGENTS.md current**: when you add a new file, feature, or R&D
  round, update the relevant section here in the SAME session. Stale
  AGENTS.md causes the next agent to duplicate your work.
- **Keep the GitHub wiki current**: the user-facing wiki at
  https://github.com/grkthegreat2018-bot/ForgeAI/wiki is a separate git
  repo (`ForgeAI.wiki.git`, cloned at `scratch/wiki_repo/`). When you
  change something the wiki documents — new preset, engine flag,
  decoding/KV/quant strategy, endpoint, script, security posture,
  build/test command — update the matching page in `scratch/wiki_repo/`
  and `git push` to `master` in the SAME session. If `scratch/wiki_repo/`
  is gone, re-clone `https://github.com/grkthegreat2018-bot/ForgeAI.wiki.git`.
  Only changes pushed to `master` go live.
- **Use `.devin/scratchpad.md` for long working notes**, not the main
  conversation. Keeps context clean and persists across sessions.
- **Debate the user's premise when warranted** — if a requested approach
  has a known-better alternative on this hardware, say so and propose the
  alternative. Don't silently implement a worse path.
- **Subagent delegation**: use `subagent_explore` for read-only research
  (file indexing, paper lookups) and `subagent_general` for parallel
  implementation tasks. Spawn 2-3 in parallel for independent work.
- **Skills**: invoke user-level skills (`log-bug`, `sync-memory`,
  `systemspecs`, `glm-supercharge`) — they live in
  `~/.codeium/windsurf/skills/`, not `.devin/skills/` (which does not
  exist). They encode project-specific workflows.
- **IDE crash trigger — batch QA/training-data edits**: The IDE crashes when
  a single edit contains 25+ lines matching Question/Answer or similar
  training-data-like patterns. **Code edits do NOT need batching** — only
  QA pairs, fact lists, training samples, and other data-like content.
  When writing scripts with embedded fact sets or training data, split the
  file write into multiple smaller edits (≤20 QA lines per edit) or write
  the data to a separate `.json`/`.jsonl` file and load it at runtime.

### H. Commit Hygiene — One Logical Change Per Commit
- **One logical change per commit.** If the commit message needs "etc." or a "+"
  to list multiple features, split the commit.
- **Stop amending pushed commits.** Rewriting history breaks bisect and
  collaboration.
- **Commit message format**: imperative mood, focused on WHY not WHAT.
  Example: "Fix int8 dtype mismatch in FastINT8Linear dequantize path"
  NOT: "Add V12 architecture + ForgeQuant + tool-use SFT + GUI fixes"
- **Bisectability**: every commit must build and pass tests. If a commit
  breaks tests, fix it or revert before pushing.

## Security Posture (updated 2026-02 audit+hardening)

Hard guarantees now enforced:
- `research/persistent_executor.py` is THE sandbox: PEP 578 audit hook
  (denies process/network/ctypes/frame/code-obj events, confines open() +
  all fs ops to a per-worker jail), restricted builtins, env scrub
  (secrets never reach workers), per-exec builtins/sys.modules restore,
  Windows Job Object (tree-kill + memory cap). `build_once_script()`
  provides the same containment for one-shot executions.
- All model-generated code paths route through it:
  `forge/self_play/self_play_sandbox.py` (persistent + once fallback —
  the old two-stage wrapper's import hook ran in the WRONG process and
  was inert; the in-process `exec(all_code)` self_critique is gone),
  `forge/self_play/infinite_curriculum.py` (_execute_sandbox).
- `forge/evolution/eval_pool.py` + `_candidate_eval.py` + `code_evolve.py`:
  secret-env scrub before candidate exec (deny-wins — secret-pattern
  matches survive only via exact-match `TOKENIZERS_PARALLELISM`; the old
  prefix allowlist leaked FORGE_API_KEY/NVIDIA_API_KEY/NODE_AUTH_TOKEN/
  CARGO_REGISTRY_TOKEN/OS_PASSWORD), Job Object on pool workers,
  `__RESULT__` lines carry a per-eval `_nonce` (env-passed, popped before
  candidate exec) and the evaluator exits via `os._exit` so candidate
  atexit forgeries never run; un-nonced markers fail closed.
- `forge_gui/api/agent_tools.py`: run_python env-scrubbed + timeout
  capped; run_tests args filtered to a safe pytest flag allowlist.
- `forge_gui_server/app.py`: loopback-only CORS + WebSocket Origin check
  (was wildcard CORS + no WS origin check = browser drive-by RCE on
  localhost).
- `forge/engine/forge_server.py`: default bind 127.0.0.1 (was 0.0.0.0),
  FORGE_API_KEY now gates ALL /v1 methods (was mutating-only — GETs
  leaked task history/security config), loopback-only CORS (was
  wildcard+credentials), lora_adapter/adapter_path confined to repo +
  FORGE_LORA_ROOTS.
- `forge/engine/tool_security.py`: default blacklist (.env/.ssh/.git/
  sandbox.json) is a NON-REMOVABLE floor — /v1/security/config PATCH can
  only add patterns.
- `forge/engine/library.py`: entry_id validated [A-Za-z0-9_-]{1,128} —
  was HTTP-controlled path traversal in entry files.
- `forge/checkpoint_io.py` + `forge/model/loader.py` +
  `forge/engine/engine_checkpoints.py`: safetensors header/tensor-count/
  size bounds (100MB header, 100k tensors, 128GB file, shape×dtype ==
  blob range) — corrupt safetensors can't OOM/OOB the loaders.
- `forge/aux_models.py`: manifest `kind=="module"` imports confined to
  project roots + FORGE_AUX_MODULE_ROOTS (was arbitrary class
  instantiation = RCE via remote manifest).
- `forge/evolution/llm_domain_gen.py`: AST gate before in-process exec.
- `forge/training/crash_recovery.py`: RLock (signal-handler deadlock),
  weights_only fallbacks now warn loudly everywhere.

Honest residuals — NOT guaranteed:
- "100% memory safe" is unattainable for a Python/CUDA/Triton codebase:
  audit hooks and Job Objects contain Python-level escapes, not
  interpreter bugs, native-extension memory errors, or GPU kernel OOB.
- Audit-hook jail assumes no buggy native code in the worker; a
  memory-corruption primitive in the sandboxed process could in
  principle defeat the hook (that's why Job tree-kill + mem cap exist).
- eval_pool/candidate workers intentionally keep full capabilities
  (torch.compile subprocess, CUDA) — only env secrets are stripped;
  treat evolved candidate code as hostile-but-jailed, not harmless.
- The `_nonce` marker token is env-delivered and popped before candidate
  exec — a candidate that reads its own PEB env-block snapshot via
  ctypes/native (eval workers have full capabilities) could recover it.
  eval_pool results travel via result_q + canaries (drift detection,
  not a trust boundary). Promotion-grade integrity still needs
  audit-hooked workers; today the nonce defeats print/atexit-level
  forgery, not a determined native-code attacker.
- Bench inputs in most evolve targets remain fixed-seed (only
  prefix_hash randomizes per-eval) — input-specialization resistance
  is partial pending the randomized-inputs roadmap item.
- `weights_only=False` torch.load sites remain for genuinely
  pickle-required formats (flux snapshots, optimizer/RNG resume
  bundles): load only self-produced artifacts; all now warn.
- forge_gui_server binds 127.0.0.1 and gates /api + /ws on a per-launch
  token (FORGE_GUI_TOKEN env or launcher-minted, carried as X-Forge-Token
  header / ?token=). It is still a local-trust boundary — the token
  travels in the launch URL; do not port-forward it or serve it on LAN.
- `tempfile`/os.stat inside the jail are fine (confined); os.stat on
  outside paths is a file-existence oracle only.


## Current Canonical Layout

Production code: `forge/` (was `research/`, migrated commit f1a7542)
- `forge/config.py` — ModelConfig dataclass + presets
- `forge/model_loader.py` — facade re-exporting `forge/model/` (kv_cache,
  attention_ops, layers, builders, llm/ConfigurableResearchLLM, loader/ModelLoader)
- `forge/checkpoint_io.py` — save/load_checkpoint (safetensors + .pt) and
  `load_safetensors_pipelined()` — the fast CUDA weight loader: N parallel
  file reads into per-thread pinned buffers + per-tensor async H2D on a
  side stream (0.53s vs 3.3s per-tensor safetensors on the 6.4GB V2 ckpt;
  fastsafetensors is broken on this Windows box — missing cudart DLL — so
  pipelined is the de-facto fast path; both have safetensors fallback).
  Wired into `ModelLoader._load_safetensors_mmap/_load_sharded_safetensors`
  and `load_checkpoint`. `from_checkpoint` also overlaps the tokenizer
  load on a daemon thread. Bench: `scripts/bench_weight_io.py` (I/O only),
  `scripts/bench_load_time.py` (full engine load phases — V2: ~2.4s total).
- `forge/engine/forge_engine.py` — ForgeEngine (inference engine; core init +
  8 mixins: engine_checkpoints, engine_activation, engine_generation,
  engine_diagnostics, engine_merging, engine_lora, engine_lifecycle,
  engine_sessions; shared helpers in engine_common.py).
  R60-4: `engine_generation._maybe_compact_past_kv` is the live bridge
  between the registered `kv_cache` strategy (beacon/anchor/metakv) and
  the per-request `past_kv` — it compacts only `(k, v)` tuple entries
  (dict SSM/conv state passes through), throttled to `compact_over`
  watermark crossings + every 32 steps + once forced after prefill.
  IMPORTANT invariant: any T>1 forward mid-sequence (chunked prefill,
  spec verify) must see the committed conv/SSM left context —
  `DoubleGatedConvLayer`/`MambaLayer`/`Mamba3Block` now read live
  state/`past["conv_state"]` as the continuation prefix; do NOT add
  T>1 decode-time paths that bypass this.
  R66 static-KV decode graphs: `PreAllocatedKVCache` gained `pos_t`
  tensor-position mode (`enable_tensor_pos`/`sync_tensor_pos`/
  `append_t` — `index_copy_` device-indexed writes, no python-int
  `[:pos]` slicing); GLA + GQA attention read the FULL buffer under
  `pos_t` with future slots masked by `attention_bias` (mask_buf is
  both the position_ids source via cumsum and the future-slot mask).
  `BreakableCudaGraph.capture_decode(decode_step=)` requires a
  static-KV step callable (refuses context-free captures — they were
  silently wrong) and validates warmup OUTSIDE `torch.cuda.graph`
  (exceptions inside capture poison the stream).
  `feature_registry._h_breakable_cuda_graph` builds the shared
  cache+mask at activation and captures one graph PER KV-LENGTH
  BUCKET (512..8192): each capture sets `cache.view_len=b` and
  passes a `mask_buf[:, :b]` view, so the recorded attention reads
  only [:b] of the shared storage — a bucket switch costs zero data
  movement and bounds wasted attention to ~4x (vs ~30x full-buffer).
  Compiled-model step tried ×2 per bucket, eager fallback per bucket;
  `_decode_state['compiled_step']` = all-buckets-fused.
  `StandardDecoding._generate_static` replays per token
  (mask_buf[pos]=1 → pos_t=pos → input copy → pick smallest bucket ≥
  pos+1 → replay).  Measured proposer_mla_moe RTX 5070
  (int4+rotorquant, 256-token decode, BIT-IDENTICAL output):
  eager-KV ~39, compiled-functional 598, **bucketed static-graph
  1258 tok/s (2.1x)**.  The "Triton cache race" was actually Windows
  MAX_PATH: ~200-char fused-kernel names + long cache base + ~85
  chars hash/tmp > 260 → deterministic tmp-dir FileNotFoundError.
  `configure.py` pins `TRITON_CACHE_DIR` to a ≤24-char drive-root
  dir (same rule as flux/model.py).
  `_forward_impl` skips conv/SSM state reset when
  `preallocated_cache.position > 0` so captured steps don't bake
  resets. Equivalence contract test:
  `TestKVCacheEquivalence.test_gqa_tensor_pos_equivalence`.
  R66-batched: `model._ensure_static_batch(B, plen, need_len)` lazily
  allocates a batch-B cache (own `pos_t`/mask) and captures only the
  buckets covering (plen..need_len] — tags `b{B}_kv{b}` share the
  `bcg._decode_graphs/_static_buffers` dicts with B=1's `kv{b}`.
  The batch cache is sized to the smallest covering bucket and any
  resize evicts the stale `b{B}_*` graphs (they hold pointers into
  the old storage).  `BatchedDecoding.generate_batch` drives it
  through a `step_fn` seam (left-pad → per-seq positions emerge from
  per-row mask cumsum; finished rows keep replaying, outputs masked
  by `active`).  B≤32 supported; need_len>max bucket → functional
  fallback.
  LFM3 (lfm3_1p7b_cc_fq) verified: conv-state is a single owned
  bank buffer + per-batch views (position-0 reset writes in place —
  captured graphs never lose storage), conv pad-mask reads the
  current position via `attention_mask.index_select(1, pos_t)` (the
  bucket-view `[:, -1]` column is always 0 → was zeroing all 13
  conv layers per replay).  All fused-attention wrappers route
  through `_finish_attn` so the per-head sigmoid output gate is
  applied — the functional path had been silently skipping it.
  Measured greedy equality static-vs-functional-batched with kernel
  parity (`--no-wrappers --no-compile`): B=8 8/8, B=16 16/16,
  B=32 32/32 exact rows; `compile=True` flips ~6% of rows at B=32
  on argmax near-ties (inductor numerics — coherent output, not
  corruption; irrelevant under temperature sampling).  Throughput
  (RTX 5070, cached-bf16 `_cache_w=True`): B=16 ~1.05k tok/s,
  **B=32 1271 tok/s aggregate** (peak 5.8GB).  int4-fused Triton
  GEMM at M=16 measured 3.3x SLOWER than dequant-cached cuBLAS —
  `_fused_gemv=False` is correct at batch decode.  Harness:
  `scripts/bench_lfm3_evolve.py` (equality + tok/s, releases the
  functional engine before loading static to avoid 12GB contention).
  R66-MoE probes (overclock_probe E5): proposer_mla_moe experts are
  DIVERGENT in all 24 layers (port-side specialization, not a load
  bug); the `_experts_identical` dense_bypass fast path is correct
  armor for fresh ports but never fires here.  Router is untrained
  (gating entropy 0.000): routed-k4 NLL +2.8, k2 +10.1, k1 +12.0 vs
  mean-of-4 baseline 3.41 — dense_bypass IS the semantics; cutting
  the 4x expert compute needs distillation, not a flag.
- `forge/model/llm.py` — runtime loop dial (Dory-style latent
  recurrence): `model._loop_iters/_loop_begin/_loop_end/_loop_reinject`
  re-run `blocks[begin:end)` K times per forward with last-loop-KV +
  conv/ssm state rollback on non-final passes (functional KV path only —
  raises on preallocated/static caches). Default 1 = plain schedule,
  zero overhead; train-time backprop flows through all passes (loop-FT
  ready). Probe `scripts/probe_lfm3_loops.py` on lfm3_1p7b_cc: zero-shot
  loops HURT — PPL 5.21→8.46@K2→45.7@K4 (reinject=1: 7.81/20.6), task
  checks 6/6→2/6@K2→1/6@K4 (reinject=1: 5/6@K2), decode cost 1.8x@K2 /
  3.1x@K4 per token. Verdict: needs Dory-style loop-sampled FT to pay
  off; reinjection halves the damage and is the right starting shape.
- `forge/aux_models.py` — generic non-LLM inference runner
  (`load_aux_model(manifest, ckpt)`): safetensors + manifest-declared
  `mlp`/`module` archs, VRAM-aware CUDA/CPU pick, `layer{i}` key remap.
- `forge/aux_train.py` — train-side sibling: `train_mlp(X, Y, ...)` on
  numpy arrays (AdamW + cosine, input-norm folded into layer0 so exported
  nets take raw features), `export_aux_checkpoint` writes `layer{i}.*`
  safetensors via `save_checkpoint`. Used by Oposition's `neural/train.py`.
- `forge/model/live.py` — LiveT live-training session wrapper
  (`LiveConfig`, `LiveSession`, `livet_config`): journaled live learning
  over ConfigurableResearchLLM. Live surfaces = GDN2 recurrent states
  (per-sequence working memory, auto-update under use_cache) +
  TitanMemory fast weights (persistent cross-sequence store, inner-loop
  surprise-gated updates). Substrate (embed/attn/FFN/head/projections)
  stays frozen — params are never written by ingest; only journaled
  buffers move. `mark()`/`revert()`/`snapshot()`/`load()` give exact
  undo; `eval_nll(use_live_state=)` measures without mutating.
  Design doc: `docs/LIVET_DESIGN.md`; tests `tests/unit/test_livet.py`.
  TitanMemory v2 (titan_memory_key.py): learned k/v/q/o projections +
  buffer fast weights + momentum/forget inner loop; `update()` target
  defaults to successor-token embedding (context→next-token store);
  `LiveConfig.solver` selects gd/cg/solve (ridge-regularized dense
  solve) inner loop. `session.attach_sparse(NGramMemory)` adds an
  exact-suffix sidecar (forge/model/ngram_mem.py): per-order suffix-hash
  successor counts -> gated logit lifts, journaled/reverted with the
  live surface — measured +2 NLL verbatim recall over fuzzy keys alone;
  hybrid (solve+ngram, all blocks) is the winning configuration.
  `forge/model/ngram_mem.py` — NGramMemory exact-suffix sidecar
  (the live-surface contract: update/lift/snapshot/restore/reset/gate);
  R64 fixes: `.to(device)` on lift logits (CUDA inputs), +0.01 uniform
  floor (novel successor in known suffix used to give -inf lift → inf
  surprise → NaN fast weights; real-text-only bug).
- `forge/model/flux/` — FluxLM (R52): sparse online associative-memory
  LM, NOT a transformer.  **Positioning (competitive audit A6): a
  revertible memory layer, not an LM — the moat is journaled bit-exact
  undo + `explain()` cell provenance; the competitive set is agent
  memory (mem0/MemGPT/Letta), not transformers.**  Held-out doc-grounded
  QA head-to-head vs a TF-IDF RAG baseline:
  `scripts/bench_flux_vs_rag.py`; STRATA division of labor:
  `docs/STRATA_VS_FLUXLM.md`.  Package layout: `config.py` (FluxConfig),
  `helpers.py` (`_mix64`/`_s64`/`_scatter_sum`/teacher loader),
  `journal.py` (FluxJournal), `tables.py` (_GPUCellTables),
  `kernels.py` (triton, HAS_TRITON-gated), `model.py` (FluxLM =
  nn.Module + mixins), `statemixin.py` (ring/epi/knn/csem helpers),
  `predictmixin.py` (all predict paths + forward), `learnmixin.py`
  (ingest/teach/fused steps), `revertmixin.py` (undo/audit),
  `iomixin.py` (snapshot/load/report).  `forge/model/flux.py` was
  deleted — `from forge.model.flux import FluxLM` resolves to the
  package facade.  Weights = addressable memory cells (per-order
  n-gram tables keyed by rolling suffix hashes, Hebbian topic matrix,
  episodic index) written live per token; context = order-agnostic
  sketch → unbounded, no positional training.  Prediction = Hedge-weighted
  geometric mixture over channels: `deep` = archived episodic retrieval —
  entries are `(ctx_end, succ, cert_hashes)` tuples deferred-inserted so
  positions past the ring horizon still vote via 24/96-gram suffix-hash
  certificates (R52-2); `knn` = approximate-match bank of normalized
  context fingerprints → successor votes via top-k cosine (R52-1,
  generalizes where `deep` needs verbatim repeats); `sem` = expected-
  next-token fingerprint cosine spread (generativity).  `teach(ctx,
  target)` = instant hot-training — key-addressed multi-order writes +
  csem + knn row, journaled under a tag (R52-3).  Every write is
  journaled (`FluxJournal`) for audit (`audit()`, `writes_since()`) and
  revert (`revert_since`, `revert_tag`, `snapshot`/`load`).  Engine path:
  `ForgeEngine.from_flux(config=FluxConfig(...), checkpoint=..., device=)`
  — bypasses checkpoint detect/quant/KV; `device="cuda"` runs fused
  CUDA-graph steps with GPU-resident tables (`cuda_primary`), knn bank,
  and host-side journal blocks (R52-4 — bulk ingest journal doesn't
  grow VRAM); `unbounded_context=True` skips tokenizer truncation;
  session learning tagged `"live"`.  Train: `scripts/train_flux.py`
  (ingest = training, instant); bench: `scripts/bench_flux.py`.
  R52-5: Triton kernels (deep-votes, logit-tail; `use_triton` flag + HAS_TRITON eager fallback), knn scan sliced to bucketed live prefix (graph recapture on bucket cross), ~692 tok/s live CUDA @V=4096.  R52-6: `graft_teacher` (V2 embed.weight → de-meaned randproj `_R`), `reinforce(tag,gain)` outcome replay, `consolidate()` episodic→ngram self-distillation, utility eviction via `_epi_hits`; distill_flux loads teacher via ForgeEngine.  R52-7: `.flux` snapshots load via ForgeEngine/GUI (models_index + engine_rt → `from_flux`); sleep/wake preserve live memory via temp snapshot; CPU→GPU sentinel-cell OOB fixed.  R52-9: `--training-mode flux` RSI in InfiniteSelfPlayLoop —
  `FluxLM.clone()` (BytesIO round-trip, zero shared state), flux-aware
  `generate_batch` (cloned-worker threads, per-prompt temp/seed, workers
  run `_use_cuda_graph=False` — concurrent captures invalidate);
  cross-epoch prompt dedup + eval-set exclusion; GUI launchable via
  SelfPlay page (mode=flux + .flux checkpoint picker), live events in
  status.json/events.jsonl.  R52-8: `train_flux.py --topic ai,code|science|custom-kw` high-precision
  domain filter (title accept / intro ≥2 distinct hits) for --hf streams,
  jsonl records, and --dict-flatten kaikki entries; corpus-scale flags
  --hf/--wiki-filter/--max-tokens/--ckpt-every; GUI live-learn via
  EngineService.learn + POST /engine/learn (sliced ingest under the
  generation lease).  Chat/QA corpora ingest as NATURAL LANGUAGE
  (c5fcd23): `_render_chat` maps roles to Question:/Answer:/Context:
  blocks — the model echoes memorized surface form, so ChatML markup
  would be reproduced at decode; `--chat-style natural|chatml|plain`,
  HF streams auto-detect messages/conversations + dolly/alpaca/squad
  schemas via `_auto_qa` (--hf-template still wins), `--domains`/
  `--domain` ingest under a named lane.  R52-9: `--training-mode flux` in
  `forge/self_play/infinite_loop.py` — RSI on _concise_qa_pairs:
  tagged attempts, immediate revert on fail, reinforce on success,
  teach_text(Q:/A:) on fail, consolidate, fixed-set eval gate
  (flux_eval_seed), demote = revert_since(epoch_mark).  Tests:
  `tests/unit/test_flux.py` (30), `tests/unit/test_train_flux.py` (12),
  `tests/unit/test_grpo_rsi_mode.py` TestFluxModeBranch.  R52-10:
  monolith split into `forge/model/flux/` package (mixin layout mirrors
  the engine); `forward()` GPU path now batches via `_fused_seq` — T
  device steps + ONE `_bout`/logits D2H readback, deferred host
  journal/mirror replay (epi eviction takes effect at row granularity);
  `learn_prefill` config flag (False = bulk forwards observe-only,
  decode still learns); `_scatter_sum` deterministic vote accumulation
  (knn/deep/order channels + reward path + triton kernel) —
  `index_add_`/triton `atomic_add` float ordering made logits
  timing-dependent; NOTE `_logits_dev` order-channel contrib also
  needed it (tokens repeat across order cells); epi deque maxlen drops
  now journaled ("eE") in the fused path; `_NullJournal.entries`
  instance attr; `load()` → `_restore_state`/`_rebuild_gpu_epi`;
  `memory_report` unified; `.flux` load documents its pickle trust
  boundary (dict-of-dict cells can't go through safetensors).  Engine
  side: `SamplingParams` collapses ~18 duplicated generation args;
  `generate_stream(stop=)` with cross-chunk hold-back (recovery
  checkpointing decodes generated_ids per step — mock decoders in
  tests must disable `eng._recovery`); logprobs captured during decode
  (no re-run — FLUX no longer double-ingests); capability checks
  (`supports_kv_cache`, `unbounded_context`) replace FluxLM
  name-sniffing; `PYTORCH_CUDA_ALLOC_CONF` moved to
  `forge.runtime.configure` and gated off Windows (expandable_segments
  unsupported → per-alloc warnings); CrashRecoveryManager weakref'd
  (atexit/signal roots pinned engine); flux `sleep(1)` uses
  `to_device` (nn.Module.to() was a no-op).  New regression tests:
  `_scatter_sum` order-invariance, `_fused_seq` vs `_fused_step`
  bit-equality, two-model CUDA forward determinism, learn_prefill
  observe-only, _NullJournal isolation, crash-recovery weakref +
  close(), flux sleep to_device, stream stop across chunks, logprobs
  no-double-ingest.  R52-11: finished the atomic-order sweep — every
  remaining float `index_add_` with duplicate-capable keys on the
  write/read paths now routes through `_group_add`/`_group_totals`
  (`helpers.py` — sort keys + cumsum + boundary diff, then ONE
  index_put_ per unique key; O(k log k), no O(k²) matmul, used for
  8k-token bulk chunks).  Converted: bulk `_A`/`_Ac` deltas, `_unig`
  (reuses the host np.unique counts), csem `asum`/`ssum`/`tocc`,
  backoff-channel logits (`_scatter_sum` like the other channels),
  `tables.add`/`add_multi`/`subtract` float writes, and the revert
  mirrors.  int64 counts (`_epi_cnt`, `bincount`) and [O]-unique/
  single-index adds were already order-immune.  NOTE `_group_add` has
  data-dependent shapes — eager paths only, never inside the captured
  `_fused_step` graph.  New test: `test_group_add_deterministic`
  (allclose vs index_add_, bitwise repeatability, exact negation
  round-trip).  `test_cuda_forward_deterministic` was flaky in-suite
  before this (12-cell `_Ac` drift ~3.7e-9); 3 clean runs after.
  R52-12: `write_gate` (thin-context `tot8<2|s>=0.1` epi + `s>=0.3`
  knn write suppression; CPU −80% knn rows, GPU −10%; teach/consolidate
  bypass; GPU gated knn writes steer to a `pos=-1` discard row so the
  captured graph stays branchless; CPU `_csem_cpu` capped at
  `csem_slots` with journaled `s`-op evictions; `adaptive_read`
  two-graph fused steps — `(lite,knn_bucket)` graph keys, the flag
  computed from the phase-2 order gather and riding `_bout`;
  `_finish_to_stop` uses `model._predict()` (past_kv=None re-prefill
  double-ingestion fixed); bounded `_recent` deque + revert rebuild.
  R52-13: `compile_dev` — `torch.compile` of `_logits_dev`/`_probes_at`
  INSIDE CUDA-graph captures (predict 630→150us, fused step
  1038→~800us, bench predict 2.5K→11.4K tok/s; logits 9.5e-07-exact).
  Dynamo-disabled islands keep bit-parity: `_topk_dev`/`_knn_votes_dev`
  (inductor topk picks different tie members → 4.2 logit diff on
  identical fingerprints), `_deep_votes_dev`/`_logit_tail_dev` (custom
  triton).  knn votes moved out of `_logits_dev` into `kv`/`kvt`
  tensor args (python-int args recompile during capture →
  cudaErrorStreamCaptureInvalidated); `kv`/`kvt` always tensors;
  `_warm` is per-(lite,bucket) for the same reason;
  `assert_indirect_indexing=False`; TORCHINDUCTOR_/TRITON_CACHE_DIR
  short paths (Windows MAX_PATH); flux sources ASCII-only (cp1252
  inductor codecache bug); `_scatter_sum` uses scatter_ stores.
  torch.compile(_step_dev) whole crashes triton PassManager — dead
  end; live steps stay ~1ms launch-bound (CPU still faster at V=512;
  GPU wins bulk ingest ~8x + predict ~40x).  Tests:
  `test_cuda_compiled_logits_parity`,
  `test_cuda_to_device_drops_graph_and_compile_state`.
  R52-14: whole-step `torch.compile(_step_dev)` now WORKS — the
  PassManager crash was two separate bugs: (a) the hedge block's fp64
  persistent-reduction kernel (kstar argmax + pmix sum + exp/clamp)
  dies in TritonGPURemoveLayoutConversions on sm_120 → extracted to
  `@torch._dynamo.disable _hedge_dev`; (b) inductor lowers
  `index_copy_`/`index_add_` on big flat views to whole-tensor
  read-modify-write (27.9µs compiled vs ~2µs scatter on a 2M int64
  tensor → ~1.1ms monster kernels on the 16M-cell order tables) → all
  big-tensor writes moved to disabled helpers `_w_order_wipe_dev`/
  `_w_order_cells_dev`/`_w_epi_dev`/`_w_knn_dev`/`_w_topic_dev`/
  `_w_csem_dev` (verbatim blocks, real scatters inside).  Inline knn
  `topk` → `_knn_reward_dev` (same tie hazard).  Fused step
  1044→492µs (~354 kernels vs 625; lite graph 398µs, full 580µs);
  CPU still ~1.8× faster on live steps (~285µs learn+predict) — the
  residual is ~300×~1µs launches; next lever = hand triton write-pack
  kernels or thin-helper producer relocation.  Compiled path is
  deterministic (twins bit-identical) but NOT eager-bitwise: fp16
  knn fingerprints can flip topk ties (≤3.2 logit swings, equal-
  quality alternatives); bout int fields exact vs eager.  Tests:
  `test_cuda_compiled_step_journal_parity`,
  `test_cuda_compiled_step_deterministic`.
  R52-15: head-format tuning shipped new defaults —
  `orders=(3,4,6,8,12,16,24,32,48,64)`, `knn_k=8`, `deep_order=4`,
  `epi_row=32` (sweep `.tmp/flux_head_sweep.py`, 2 corpus families +
  GPU).  Findings: order-1/2 are noise channels (drop alone: NLL
  0.465→0.272, div-eval better too); orders 48/64 add real in-domain
  reach; k=8 captures all of knn's contribution; deep_order=4 seeds
  shorter matches (-7% div); epi_row=32 unstarves GPU deep coverage
  (-19% GPU NLL for +0.23GB); `gpu_row_width` stays 16 (32 = no gain);
  cert_orders flat below ring horizon (untested at scale).  Net vs old
  defaults: -67..-69% CPU NLL, -9% diverse-eval, -30% GPU NLL, -22%
  GPU div, recall intact.  MAX_PATH bug fixed in `FluxLM.__init__`:
  cache-dir env values >24 chars (e.g. `.devin/torch_cache/triton` set
  by `forge/runtime/configure.py`) are overridden to drive-root
  `D:\ti`/`D:\tt` — a ~148-char fused kernel name only fits ≤~125-char
  bases... inverse: base≤24 fits ~170-char names.  Pinned by
  `test_tuned_head_format_defaults`; two tests re-pointed off
  hardcoded order-1/2 semantics.  R54 (measured-experiment round,
  harness `.tmp/flux_harness.py`, log `.devin/scratchpad.md`):
  `calib_rewards=True` — additive-lift channels (recency/fatigue/sem/
  csem) now score hedge rewards as their implied distribution
  `p_uni·exp(f)/Z` instead of raw feature values; fixes the measured
  hedge collapse (`fatigue≈0.98`) — weights spread over order channels.
  `uni_escape=0.05` — arithmetic ε·unigram mix on emitted logits
  bounds worst-case NLL (geometric mixtures let a confident-wrong
  channel veto the true token).  `sleeping_experts` (0/1/2 abstention
  modes) and engine `generate(learn=False)` predict-only gate +
  `generate_batch(prompt_rewards=)` passthrough added but NOT enabled
  (E1/E2 measured no-win; flags are diagnostics).  Numbers: GPU fact
  NLL 0.206→0.111, bound 1.36→0.98, all 4.64→3.68, ingest
  2265→9461 tok/s; CPU fact 0.77→0.25.  Regression tests:
  `test_calib_rewards_no_hedge_collapse`, `test_uni_escape_floor`;
  `test_reinforce_and_revert` teach weight recalibrated 0.4→0.05.
  R54 cont. — `sparse_taps` PROMOTED (E7): PAQ8-style gapped-order
  channels `p<j>` (stride-2/3 6-tap set default) — hash chains over
  (t, t-2, t-4…) survive single-token corruption that breaks every
  contiguous key; `sparse_thin_only` gates them to fallback experts
  (ungated dilutes warm contexts — fact .24→.35; gated keeps the win).
  Host-resident `_ptabs`/`_ptots` on BOTH backends: GPU bookkeep
  mirrors surprise-scaled writes ("C" journal, exact revert), bulk
  does vectorized ring-gather writes, predict thin flag rides bout
  (`_boff_thin`) and the host scatter-adds the lift into `_blogits`/
  `blog_seq`/`_plogits` post-step.  Sparse arms abstain from device
  hedge (neutral pmix via `_pmask` closed form — uni fallback would
  decay them to hedge_min and kill the lift).  Measured: div
  5.26→4.21, all 3.67→2.99 (CPU) / 3.66→2.99 (GPU, ~exact parity,
  zero VRAM).  cond_hedge (E6) and knn_utility_evict (E5) measured,
  NOT promoted — flags stay off (see scratchpad).  R54 cont. — E8:
  `sparse_taps` extended to 8 taps (stride-4 (0,4),(1,5)) → all 2.92,
  and PROMOTED `class_orders=(3,)`: g<k> channels = contiguous suffix
  keys over byte CLASSES (`_byte_class` LUT, PAQ word-model analog) —
  coarse keys generalize on novel text.  Same host-resident split
  (`_gtabs`/`_gtots`, "K" journal entries, thin-gate, abstention via
  `_pmask`/`_pidx`).  g3 alone is optimal (k≥4 keys are overconfident-
  wrong on real prose; class_min_tot floor no-op).  all 2.92→2.76,
  div→3.88, nat→3.40, GPU parity 2.764 @ 89.3MB.  R54 cont. — E9:
  `thin_tot=2.0` (order cells below 2 count as thin → fallback experts
  fire; all −1%), `uni_escape` retuned 0.05→0.12 on the new mixture
  (all −2%), `hybrid_taps=(((3,0),(2,0),(1,0),(0,1)),)` PROMOTED
  (h<j> = (offset,is_raw) class+raw patterns; all 2.676→2.662; extra
  correlated arms dilute).  `word_orders` (w<j> ring-scan word-hash
  keys) measured NOT promoted — plumbing kept opt-in.  Benchmark
  all: 3.67→2.66 cumulative.  R54 cont. — E10: `thin_boost=0.2`
  PROMOTED — fixed +w for deep/topic/sem/csem/knn in regime-0 (thin)
  contexts, bypassing the hedge floor where they are the only signal
  (all 2.662→2.647 CPU / 2.663→2.659 GPU; swept .1–.5).  GPU carries
  it via `wrow_b = wrow + tb*_tb_mask` into `_logit_tail_dev` + eager
  sites.  Teacher graft measured at token level (V=65536 harness,
  ForgeLM_V2 embed→randproj `_R`): graft alone 3.445→3.435, with boost
  3.471 — NOT promoted; semantic votes are too noisy at 65k vocab for
  a fixed boost, and channels hold ~0 hedge weight regardless.  E11:
  `FluxLM.generate()` + `support_thresh=1.5` — support-restricted
  sampling (token must beat thresh·p_uni) keeps temperature decoding
  off the unigram floor; greedy+masked sampling produce fluent NOVEL
  orderings on diverse corpora (the model composes at boundaries -
  earlier "retrieval only" was a fixed-order-corpus artifact).  Also
  fixed `_ingest_bulk` crash on short sequences.  PPMI self-`_R` graft
  measured, no gain — cosine-readout is the bottleneck, not features.
  E12: `distill_head` — parametric teacher channel: linear `_Wh
  [V,topic_dim]` trained by AdamW on (normalized context fingerprint
  -> teacher top-k) pairs, read as fixed-`head_w` additive logit lift
  in `_predict`/`_logits_dev`.  scope='all'+ids= trains at every
  corpus position (ring horizon would cap at ~3.3k).  Journaled 'WH'
  blob (fp16) + snapshot + to_device; CUDA graphs invalidated on
  write/revert (staleness bug found+fixed).  Token-level V=65536 vs
  ForgeLM_V2: graft+head(scope=all,ep4,w2.0) all 3.4046 vs 3.445
  baseline, recall 4/8, rep/fact preserved — the ONLY distill path
  that improves all+recall together (cells: rep 6.59 destroyed;
  hebbian: NLL worse).  Head alone on random _R memorizes (loss 9.84)
  vs grafted _R generalizes (8.79) — they compose.  CEILING: does not
  move greedy generation — topic-EMA fingerprint is bag-of-features,
  no word order; chat/agentic transfer needs an order-sensitive head
  input (bigram-bound sketch beside _c).  Opt-in: head_w=0 default.
  E13: order-sensitive head input landed — `concat(ĉ, R[t-1..t-8]/√d)`
  read live from `_ring`/`_ringg` (`head_orders=8`, graph-safe, zero
  new live state) + `head_thin=True` gate (regime-0 only — ungated
  head polluted rep/fact on off-domain data).  Harness: all
  3.445→3.2403, recall 8/8, head loss 8.79→7.26.  `head_hidden>0` =
  trained MLP channel (`Wh@relu(Wh1@x)`, Wh1 journaled in the 'WH'
  3-tuple + 'Wh1' snapshot key) — measured NOT better (head top-6 is
  marginals on every prompt; capacity is not the bottleneck, the
  fixed featurization is).  Retrieval is the generation mechanism:
  docs-60k ingested → coherent topical continuations; grafted _R
  gives semantic hits via cosine channels ('capital of France is'
  →'▁Berlin'); nat 10.02→9.25 from ingest alone.  Recipe: graft +
  distill_head(scope=all, ids=<served corpus>, ep4, lr3e-3), w≈2,
  thin-gated.  R54 cont. — readability + order-sketch: `explain(top_k)`
  returns per-channel logit attribution per top token + fired-cell
  provenance (raw keys, totals, followers, knn neighbours) — the
  mixture is additive so the decomposition is exact; `inspect_cell(
  'o8'|..'h0', key)` dumps one cell; `format_explain` renders.  CPU
  `_predict(cap=)` and device `_logits_dev(cap=)` record each
  channel's delta (cap forces the eager tail; triton tail skipped);
  host-table lift refactored to `_sparse_lift_channels_dev` (per-
  channel dicts; `_apply_sparse_lift_dev` merges — hot path numerics
  unchanged).  `bigram_sketch=True` maintains `_c2` — EMA over bound
  pairs `R[prev][bp1]*R[cur][bp2]` (seeded perms `_bp1/_bp2`), the
  order-sensitive head input E13 flagged: head in_dim += topic_dim,
  `_push`/`_step_dev`/bulk `_chunk_c2adv`/`_bind_pair`/`_bigram_vector`
  + `_rebuild_c2` (pre-sketch snapshots), 'b' journal entries now
  4-tuples (fp16 sketch), 't'/'bc' undo exact, 'c2' snapshot key.
  R55 — `gcell` channel (opt-in, default OFF): learned-key cell
  family — same dict-of-dict substrate as p/g/w/h host tables but
  the key is a SimHash code of the order-sensitive context
  fingerprint (`_gcell_feats` = [ĉ, ĉ2, R tail rows] like the head
  input; `x*512` int-quantize @ int8 `_Wk` projection, fp64
  accumulate = exact integer math → bit-identical CPU/GPU codes).
  Successor scores by delta-rule `s += lr·s_tok·(1-s)` scaled by the
  token's surprise; fixed-weight `gcell_w` additive lift, thin-gated
  (`gcell_thin`, regime-0 only) so recall owns warm contexts.
  Host-resident both backends (zero VRAM), journaled 'G'
  (pre-write values — bit-exact undo), 'gE' evictions,
  `gcell_max_cells` fifo cap, snapshot key 'gc', `inspect_cell(
  'gcell', code)` + explain()/memory_report wired.  Write sites:
  `_learn_token`, `_bookkeep_fused` (gc_code param — caller scores
  the PRE-x context since device _c is post-step), `_ingest_bulk`
  (`_gcell_codes_bulk` vectorized per-position codes from
  `_chunk_cprev`/`_chunk_c2prev`), `teach()` (`_gcell_code_ctx`,
  weight→repeated delta steps), `_fused_seq` (`_cprev_seq`/
  `_c2prev_seq` pure closed-form context rows — read code for
  emitted logits uses post-token row i+1, write code uses pre-token
  row i).  `_predict_gpu` lift gate now includes `config.gcell`.
  KNOWN (pre-existing, now documented in _fused_seq): `_next_lite`
  is host-side → seq mode can't flip lite/full graphs mid-row;
  surprise-scaled writes can diverge from _fused_step when
  adaptive_read toggles (pass adaptive_read=False for bit-equality).
  ALSO FIXED: `revert_since(seq_pos=)` over-ran one bulk block —
  after a 'bc' restore lands count<=floor the loop kept popping the
  PREVIOUS block's non-boundary writes (epi/knn/gcell); now breaks
  on the 'bc' boundary landing.
  R56 — `gsyn` channel (opt-in, default OFF): dense delta-rule
  generator, the parametric complement to gcell's sparse cells.
  `_S [topic_dim, gsyn_dim]` updated `S ← γS + β·s·(v_err ⊗ k)` per
  learned token where k = `_Wq`-projected `_gcell_feats`, v_err =
  `_R[x] − S@k` (delta rule = per-token SGD — the channel keeps
  self-modifying at inference like `_A`).  Two-timescale mirror
  `_Ss` (config `gsyn_slow_decay`>0, `gsyn_slow_beta`) = HOPE-style
  continuum memory pair: fast adapts to the passage, slow
  accumulates long-range regularities; read sums both.  Read =
  `w·(log p_softmax(R@(S@q)/τ) − log p_uni)` additive lift,
  thin-gated (`gsyn_thin`, regime-0).  Write sites mirror gcell:
  `_learn_token`, `_bookkeep_fused`/`_fused_seq`, `_ingest_bulk`
  (`_gcell_feats_bulk` shares one feature pass with the codes),
  `teach()` (`_gcell_feats_ctx`).  Journal 'GS' rides the fp16
  `_gsyn_ring` [coeff_f|coeff_s|v_err|k] — undo is the exact
  inverse `S_pre=(S−c·v_err⊗k)/γ` (same precision class as 'a').
  Lift is post-graph in `_apply_sparse_lift_dev` (host-computed
  per step like gcell) so CUDA-graph capture/replay is untouched.
  Snapshot 'gs' = (S, Ss) tuple — bare tensor = pre-two-timescale
  load path.  `memory_report()['gsyn_MB']` counts S+Ss+ring.
  `mem_bias` (SamplingParams → `generate(mem_bias=)` →
  `model._gen_bias`) is the recall↔gen runtime dial: scales both
  gcell and gsyn lift weights for the call, restored after; 0 =
  recall-only, >1 = lean generative.
  R60b — `r_live` (default ON, `r_eta` knob): the no-SFT substitute
  for `graft_teacher`/`distill_gen`.  Additive drift table
  `_Rl [V,topic_dim]` — each learned token's row EMAs toward the
  normalized PRE-x context sketch at rate `r_eta*s` (surprise-scaled
  co-occurrence embedding).  `_R` stays the frozen seeded bipolar
  matrix → `_c`/`_c2`/knn context-side EMAs keep bit-exact inverse
  undo; vocab-facing reads use materialized `_Reff = _R + _Rl` (gsyn
  logits + delta-rule targets; `teach` binds target rows too).
  Undo: 'rL'/'rB'/'bc'["rl"] first-touch restores — 'rB'/'rL' are
  revert boundaries (session-scoped, 'bc' class).  Snapshot key
  'Rl'; `_Reff` re-derives on load.  Write sites: `_learn_token`,
  `_bookkeep_fused` (inverts topic EMA for pre-x `_c`),
  `_ingest_bulk` (`_rl_bulk`, deterministic `_group_totals`), teach.
  `_R` dtype stays int8 — `_Rl` is the only live float table.
  R60c — recall→gen (`gen_recall` knob + `consolidate_gen()`): the
  recall side feeds the gen side.  Live: warm contexts (pre-write
  order-cell tot > thin_tot) write gcell/gsyn/_Rl at
  `max(surprise, gen_recall)` — recall-proven successors transfer
  even when unsurprising; wired in `_learn_token` (probe hoisted
  pre-write), `_bookkeep_fused` (`boff_thin` flag), `_ingest_bulk`
  (tots_f/found_f).  Batch: `consolidate_gen(mass, top_m, min_tot,
  tag)` replays on-ring positions (closed-form `_cprev_seq` seeded at
  zero inside the episode), gathers each context's cross-order
  successor distribution, writes top-m share-weighted followers into
  gcell + gsyn (explicit `v=` targets in `_gsyn_chunk`) + `_Rl`.
  One ws/we block; gsyn "gs" snapshot + "rB" session + per-cell "G"
  — revert_tag restores bit-exactly.  `_ringg` host fallback added
  for CPU `_gcell_feats_bulk`.  ~20s to basic topic competence
  measured (21.7K-token corpus, CPU).
  R60d — `gsyn_deep` (default OFF, 9 knobs): the nonlinear memory
  channel — a 2-layer MLP f_theta(k)→expected _Reff row trained
  ONLINE by surprise-scaled gradient steps (TTT-MLP/Titans: the
  memory is a learned function, not a table).  ~115K params +
  momentum buffers, <0.5MB; explicit no-autograd gradients, global
  clip (`gsyn_deep_clip`), heavy-ball (`gsyn_deep_momentum`),
  decay forget-gate (`gsyn_deep_decay`).  SEPARATE from linear
  gsyn (own lift w/thin/temp/journal/explain/report/probe) so
  ablations isolate it; `_gsd_chunk` = LaCT-style batched grad
  steps in bulk/consolidate.  Undo = whole-theta snapshots
  ("gd" entries + blk["gsd"] in "bc" blocks; `_gsd_session` flush
  per ingest span).  NOTE: "gd" had to be added to revert_since's
  boundary set — without it a rewind overshot into the previous
  span's session snapshot (latent bug class for any future
  span-scoped op).  `probe_gsyn_deep()` exposes k/h/o stats +
  param norms + theta drift.  Proof: XOR-corner task — antipodal
  keys → same target — linear gsyn 2/4 (chance, f(-u)=-f(u) is
  structural) vs gsyn_deep 4/4.
  R60e — `dnet` (default OFF, 8 knobs): the iterated-memory channel —
  a stack of gated delta-rule layers, i.e. a linear-attention
  "transformer" that learns online with no training loop, no KV
  cache, no optimizer state (~0.34MB total for 2 layers).  Each
  layer's S_l is the same delta-rule memory as gsyn; the STACK
  supplies what single-step channels structurally lack: iterated
  content routing (layer-l input = relu(o_{l-1}) — one layer's
  readout addresses the next layer's memory; relu breaks the
  odd-symmetry linear ceiling).  Content-gated writes
  `sigmoid(w_b·u)·beta·s` (GDN-style), scalar decay.  Same journal/
  revert/explain/probe/report parity as gsyn_deep ("dn" op +
  blk["dnet"], in the boundary set).  Depth proof: antipodal task
  1-layer=2/4 vs 2-layer=3/4.  Bulk writes are exact-sequential
  (layer l needs l-1's CURRENT state — the WY chunk shortcut
  doesn't lift; optimization path noted).
  R60f — **fuzzy channels are now hedge members + candidate-
  restricted**: `gcell`/`gsyn`/`gsd`/`dnet` sit at the tail of
  `self.channels` (flag-gated).  Real probs in `_channel_prob`
  (eager; `_pfq` memo per position) and `real_r` under
  `bulk_channel_rewards` (bulk, post-block-state approximation);
  abstain-neutral on the fused device step (`_fidx` filled with
  uni_x).  All four lifts multiply by w_rel = C·w (relative to
  uniform share — sinks when the channel loses; replaces the
  fixed-weight bypass that made them unpunishable).  Knobs
  `lift_topk=64`/`lift_topm=8` (0=off): fuzzy lifts vote only on
  top-k of the pre-lift row + own top-m proposals —
  `_sparse_lift_vec_dev` now returns (lift, flift); `_apply_sparse_
  lift_dev` computes the support mask once per predict.  Load:
  `wR` width mismatch replicates the global row.  Measured: grid
  `@2K docs seen_qa` 0.28 → 0.84 (hedge alone carried it;
  restriction = structural guarantee), `backoff_order_sharp=1`
  → 0.98.  Eval harness: `.devin/eval_flux_quality.py` +
  `eval_flux_grid.py`.
  R57 — gcell fuzzy read (SDM semantics, `gcell_read_radius`, default
  1): predict-time Hamming-ball probe over learned codes — similar
  contexts pool evidence, so generalization is a *read* of existing
  cells, not extra stored state.  `_gc_buckets` coarse index
  (`code >> gcell_bucket_shift`, default 6) prunes probes — codes
  within dist r differ in ≤r high-part bits, so enumerating
  high-parts within r covers all candidates; members distance-
  checked exactly (`(cc^q).bit_count()`).  Contributions decay by
  `gcell_neighbor_decay**dist` (default 0.5).  Bucket ops ride the
  existing 'G'/'gE' journal (index mirrors cell lifetime — undo
  maintains it inline); index is derived state, rebuilt on load.
  explain() cells now carry a `dist` field per fired gcell code.
  R58 — fine-grained credit + verifiers + exploration in flux
  self-play: `reinforce(tag, gain, span=(lo,hi))` replays only
  deltas whose token position falls in the span (positions are
  reconstructed from 't'/'bc' markers — 't' CLOSES its write group;
  absolute anchor = _count − journaled tokens, 'bc' blocks carry
  blk["count"]); `gain` may be a callable `gain(pos)` for shaped
  credit.  `flux_verifier` (callable(response, pair) → [0,1])
  replaces the binary grader: `attempt["reward"]` scales
  `reinforce` gain (`flux_gain * reward`); below-threshold-but-
  nonzero attempts (and parsed tool calls under wrong answers)
  apply their packets at `flux_gain * max(reward,
  flux_partial_gain)` before teach-fail.  `flux_explore_eps` /
  `flux_explore_temp` give ε-greedy temperature exploration in
  _temps.
  R59 — counterfactual credit via logged propensities (logged-bandit
  / CRM): `generate_batch(cf_topk=k)` makes FluxStreamPool record
  each generated token's top-k (id, prob) alternatives into
  `FluxPacket.alts` (post-temp/mask sampling distribution — the true
  propensities).  `FluxLM.reinforce_alts(tag, packet, gain, scale,
  block_idx)` walks the applied ws/we block for per-position order-
  cell keys ("c"/"t" groups host-side, "bc" writes_all on GPU;
  block_idx selects among same-tag blocks since apply order =
  journal order) and writes `gain*scale*pi/(1-pi_sampled)` to each
  untried alternative via `_write_counts_keys`, journaled under
  `cf:<tag>` (independently revertible).  Self-play wiring: success
  gives alts a share of the win; partial-credit failures boost alts
  at the same scaled gain; zero-reward failures (flux_cf_fail)
  apply the packet, boost the alternatives, then revert_writes(tag)
  — the untried actions keep credit for what the tried one lost.
  Greedy/degenerate positions (pi_s>=0.999) are skipped.
  R60 — bulk-ingest speedup round (~850-1900 → ~4.5-7.5K tok/s on
  GSM8K text, RTX 5070): (a) `_gsyn_chunk` replaced per-token
  `_gsyn_write` in the bulk path — WY/triangular-solve delta-rule
  blocks (fp64, exact vs sequential, `blk["gsyn"]` S/Ss snapshot for
  undo); block size default 256 with adaptive cap `bs <= 1 +
  -6ln10/ln(gam)` keeping `g^-(bs-1)` fp64-safe. (b) `add_multi`
  collision-repair loop batched: one `_slots2`/probe + grouped undo +
  batched follower-column pick for the whole stuck set; per-element
  sequential writes remain only for ordering, with live re-probe +
  host row-read restricted to elements whose target flat was
  conflicted or written by an earlier repair (`written` set). (c)
  journal flood fix — bulk per-token C/K/W/H/G/GS/eE entries ride a
  per-block sub-journal (`blk["sub"]`, replayed in `_undo_block`), so
  a 40K-token blob emits ONE journal entry instead of ~187K and `bc`
  headers can't be evicted mid-block (previously reverts silently
  no-oped). (d) `_A` stability: grouped topic deltas are damped by
  the geometric-saturation factor `phi` (batch-mode delta rule with
  m duplicate tokens overshoots past m·lr·s>2 → inf→NaN chain);
  proto EMA rewritten as bounded decayed-sum matvec (the cumsum/
  `pd^-j` construction overflowed at chunk ~8K); non-finite surprise
  treated as maximally surprising.  Adaptive knn read-skip already
  shipped (`adaptive_read`).  Deferred: FAISS binary index for
  gcell fuzzy reads (new dep), suffix-array recall core
  (architectural).  Bench: `.devin/bench_corpus.py`,
  `.devin/bench_stuck.py`, parity `.devin/gsyn_parity.py`.
  R61 — packed v2 `.flux` snapshot format: v1 pickled dict-of-dict
  cells verbatim, so a 5.9M-token corpus model serialized ~16GB
  (journal `bc` blocks ~13GB of tensors + 2.9GB `data.pkl` of Python
  objects) and took ~6min/~16GB RAM to load.  `snapshot(path,
  keep_journal=False)` now writes `fmt=2`: every `{key -> {tok: cnt}}`
  table becomes CSR columns (keys_u64/rowptr/toks_i32/cnts_f64 —
  f64 preserves exact Python floats), `_epi` becomes
  keys/rowptr/pos/succ/clen/cert columns, `last_seen`/`_epi_fifo`/
  `_gc_fifo` become column pairs, and journal entries are dropped
  unless `keep_journal=True` (deployment default off; in-session
  journal on the live model is unaffected — post-load reverts only
  reach pre-snapshot writes when the journal is kept).
  `_GPUCellTables.triplets()` vectorized (masked tensor dumps →
  np.repeat/reshape, was a ~100M-step Python loop) and
  `load_triplets` chunks `add` calls (2M/slice) so big restores
  don't spike VRAM.  knn bank slices to the live prefix
  (`_knn_live`) instead of serializing full capacity.  Load path is
  backward compatible: `_restore_state` detects `fmt==2` and
  `_v2_to_v1` unpacks columns back to the v1 structures before the
  shared restore code — v1 files still load.  `clone()` stays on
  the v1 in-memory path (keeps journal).  Migration:
  `.devin/migrate_v2.py`; round-trip test `.devin/v2_roundtrip.py`
  (predict/A/S/epi/gc bit-parity cuda+cpu).
  R68 — live generalizer "gen head" (`gen_live`, default OFF;
  `gen_live_topm`, `gen_live_min_tot`): consolidate_gen made
  incremental.  Every learned position whose order cells hold
  evidence (post-write tot > min_tot) transfers the cells' recalled
  successor DISTRIBUTION — top-M followers, share-weighted — into
  the gen channels at strength `gen_live*conf` with
  `conf = max_k tot/(tot+backoff_kappa)`: gcell per-follower cell
  writes, gsyn/gsd/dnet blended soft targets
  `sum(share*_Reff[t])`, and `_Rl` per-follower row EMAs.  The head
  owns NO weights — it moves mass between existing stores
  (`_genlive_write` in statemixin.py).  Unlike `gen_recall` (floors
  the realized token's own write), it covers followers that were
  not observed at the position, and unlike `gen_recall` it keeps
  writing as cells converge (realized writes stall at s→0, conf
  stays ~1).  All three learn paths read POST-write cells: eager
  `_learn_token` reads `_tables`/`_totals` after the order write;
  `_bookkeep_fused` re-probes post-step (`_fused_seq` batches one
  post-row probe over all positions — consolidate_gen-class
  approximation); `_ingest_bulk` reuses `slot2`/`found2` (the
  reward fix-up's post-add_multi re-probe — pre-write probes made
  first-pass bulk ingests transfer nothing; that was the GPU test
  bug).  `_predict_only` skips it; thin contexts self-gate.  All
  writes ride the existing journal ops — `revert_since` on stream
  ingests restores `_gc`/`_gct`/`_Rl` bit-exact and `_S` to
  fp16-factor precision.  Self-play: `--flux-gen-live` (implies
  `--flux-gen-channels`), `--flux-gen-consolidate` batch complement.
  Probe `.tmp/exp_gen_live.py`: mass +14-38% in gen stores, NLL
  neutral on paraphrase frames (transfers stay distribution-
  consistent — benefit is continuous consolidation, not noisy
  writes).  Tests `test_gen_live_*` (6).
  R69 — extractive knn recall lift for held-out QA (all knobs
  default OFF in `config.py`): `_knn_votes_dev` is now a pipeline —
  broad cosine pool (`knn_pool`) -> high-idf coverage rerank
  (`knn_cover`/`knn_cover_w`) -> votes (successor; `knn_span` span
  voting w/ `knn_dedup`+`knn_span_idf`; `knn_twin` answer-adoption
  from rows inside trained "Q:..A:" lines; `knn_spread*` corpus-wide
  content field, `knn_spread_doc` = flat doc-coverage maxpool x
  own-idf).  Fingerprints: `knn_window` last-W bag + `knn_idf` +
  `knn_residual` (prose-mean removal — fixes the cos~0.99 collapse)
  + `knn_episode` prompt-span queries (`_prompt_start`/`_prompt_g`
  set in `ingest()`).  Merged decode in `generate()`: `lg += knn_lift
  * log p_votes` for `knn_lift_steps` — recall merges into
  generation rather than replacing it, gated by deep-order cell
  existence (`knn_lift_deep`, seen keep verbatim recall),
  `knn_lift_thin`, `knn_lift_conf`, and vote concentration
  (`knn_lift_topconf`).  Answer masks: `attach_vocab(itos)` +
  `knn_atype` (word-initial alpha) + `knn_tokmin`/`knn_tokmax`.
  BUG FIX: the spread field used to include the prompt span in the
  live mask — the question self-matched its own rare tokens; the
  query span is now excluded from field source and read positions.
  Measured (4K trivia+evidence): seen_qa 0.86-0.96; paraphrase
  generalization 54-80%; held_qa ~5% — corpus-knowledge bound
  (answers absent from the docs), not a mechanism bound.
  R70 storage/scale pass (30K-doc corpus): fingerprints NEED the
  write-side set (`knn_window=16`+`knn_idf`+`knn_residual` +
  `--doc-eos` episode boundaries) or the bank collapses to a flat
  ~0.72 cos space (v10 seen_qa 0.15 -> v11 0.80, v12 +`knn_idf`
  numeric-bool parse fix = 0.825).  `_ingest_bulk` EOS split needed a
  reentrancy flag (terminal marker recursed).  Elastic storage:
  `auto_grow` grows order tables (lossless dedup-claim migration in
  `_GPUCellTables.grow`, anchored-force placement kills eviction
  ping-pong), knn bank, and ring at 85% pressure/wrap - journaled +
  snapshot-compatible, `grow_max_slots` bounds it.  `_epi_pos`/
  `_knn_pos` int32 (-272MB on the v12 layout; `index_put_`/`cat`/
  `index_copy_` sites need explicit `.long()`/`.to(int32)`).
  `knn_dedup_sim`/`knn_dedup_win` merge near-verbatim bank writes vs
  the recency window (redundant when `write_gate` is on - it already
  suppresses repeats).  `gpu_row_width=8` is quality-free at 30K
  (halves ttoks/tcnt: -1.28GB; M=12 slightly *helps*: 0.85).
  held_qa=0.0 at 30K is retrieval/extraction-bound, not coverage
  (63/80 held answers exist in corpus); `knn_atype`/`knn_span`/
  `knn_subq` + self-play are the open lever.  Head-to-head confirms it
  at small scale already (`scripts/bench_flux_vs_rag.py`, 500-2000
  synthetic docs, R69 knobs + Q/A-format scaffold): flux=0.0 vs TF-IDF
  rag=1.0 — knn retrieves fact-type-correct but entity-confused rows
  and the hedge mixture collapses to one channel.  LFM2's 64400 tokenizer
  is NOT flux's 65536 vocab - only ForgeLM V2 is a same-vocab
  teacher, and w8a8/int8 quant hits a CUBLAS scaled_mm limit on this
  box (fp16 works).  Tests: `test_grow_*` (3), `test_knn_dedup_*`,
  `test_teach_writes_all_gen_stores`, `test_teach_generalize_flag_*`,
  `test_gen_head_holds_no_params`.
- `forge/model/strata/` — STRATA (R60): FluxLM's successor as a
  three-artifact format — WEIGHTS `.strata` (labeled CellBlocks of
  packed 48B cells: tag32+meta+8×u32 followers+raw64, log-quantized
  counts, 2-choice cuckoo hash, active/sealed/cold lifecycle,
  per-block undo arenas, tag-scoped revert, memmap container IO),
  PARAMS `.iq` (ParamSet — mixer/dense weights; seeded tensors stored
  as {seed,shape} and regenerated), KNOBS `.cond` (KnobSet — scalar
  gates/lifts/sampling with named overlays).  M0 = substrate;
  **M1 `StrataLM` (model.py) DONE**: suffix-hash order channels
  o3..o64 + uni/backoff/recency/fatigue, hedge geometric mixture,
  surprise-gated writes, ingest/teach/observe/learn, forward→
  (logits,None,None), generate+knobs, explain/audit/revert/snapshot —
  CPU-eager reference, ~86 tok/s learn.  Design =
  `docs/STRATA_M1_DESIGN.md`; further milestones (dense gsyn channel
  should reuse FLUX's r_live live-feature pattern — params+knobs, no
  distill).  Bench: `.tmp/bench_strata_step.py` — eager is
  launch-bound dead (~28 tok/s); graph replay 585µs for 10
  probes+dense; `pk_of` per-call salt alloc breaks capture; add() not
  graphable → two-tier thin-write + deferred repair.  Tests:
  `tests/unit/test_strata.py` (27) + `test_strata_m1.py` (25).
  Division of labor (audit A12, `docs/STRATA_VS_FLUXLM.md`): FluxLM =
  research vehicle, STRATA = deployment artifact format — new
  capabilities land in FluxLM first; STRATA ports only stabilized
  pieces.  Consolidation trigger: fold into flux/ if the container
  ships no unique value by next round.
- `forge/scrape_feed.py` + `scripts/scrape_feed.py` — R61: continuous
  online scraper -> normalized doc feed.  Sources (`wiki-random` /
  `wiki-all` resumable allpages cursor / `wiki-recent` / `wiki-search` /
  `arxiv` / `wiktionary` / `urls` / `ddg`) -> Processor (persistent
  sha1 dedupe `seen.txt`, min/max chars, injected filter_fn — CLI wires
  `download_wikipedia.should_skip` + train_flux `--topic` scorer; wiki
  keeps `raw` wikitext for the filter since should_skip keys on
  {{disambig}} markup) -> ShardWriter (`feed_NNNNNN.jsonl.wip` ->
  `.ready` rotation; the filesystem queue is the bus) -> sinks:
  `--to shards` (corpus only), `--to api` (serialized POST
  /engine/learn per shard, polls /engine/stats `info.learn` for
  completion — the route is one-at-a-time), `--to model` (in-process
  FluxLM ingest + snapshot, tag "<tag>:<source>" for revert).  Gotcha:
  prop=extracts caps whole-article text at 1 page/request
  (exlimit>1 needs exintro) — live full-text uses prop=revisions
  wikitext (~20 pages/req) + local `wiki_to_text` cleaner.  All fetches
  injectable; tests `tests/unit/test_scrape_feed.py` (16, offline).
- `forge/self_play/flux_goals.py` — R53: source-grounded goal layer for
  flux-mode self-play.  `FluxTools` = budgeted cached executor for the
  model-facing tools (`wiki`/`dict`/`paper`/`web`, `parse_tool_call`
  accepts `wiki X`, `dict: x`, `[paper] x` + aliases); doc fetchers
  return rich payloads for goal building while `call()` returns
  one-line `obs_chars`-clipped observations.  Sources: Wikipedia REST
  summaries (search→summary or random), Wiktionary REST definitions
  (primary; dictionaryapi.dev fallback; local kaikki jsonl via
  `--flux-dict-file`), arXiv Atom (year/first-author/id goals).  Goal
  builders are EXTRACTIVE — every `response` is a verbatim span/field
  of the doc so `_flux_check` stays exact and teach-on-fail writes real
  knowledge.  `FluxKB` = append-only `flux_kb.jsonl` under status_dir
  (deduped (source,key); replayed for new question forms when the fetch
  budget is spent or offline).  `FluxGoalSampler.sample_web` =
  round-robin fresh fetches + KB replay, deduped vs cross-epoch seen
  AND the eval sets; `build_eval` persists a frozen
  `flux_eval_web.json`.  Loop wiring (`infinite_loop.py` R53):
  `flux_web`/`flux_goal_sources`/`flux_web_frac`/`flux_tool_frac`/
  `flux_rag_frac`/`flux_web_fetch_budget`/`flux_eval_web_n`/
  `flux_dict_file`/`flux_obs_chars` config+CLI; web goals get kind=
  "qa"|"rag"|"tool" — tool mode is two-phase (`T:` line → loop executes
  → `O:` injected → `A:` answer), failures teach_text the full
  trajectory so FLUX learns the tool-call binding verbatim; eval =
  procedural + frozen web sets combined.  Offline → `FluxTools.online()`
  probe fails once and everything degrades to the procedural stream.
  GOTCHA: Wikimedia hosts 403 browser-spoofed UAs (robot policy) —
  `web_primitives._ua_for` sends a descriptive BOT_UA to
  *.wikipedia.org/wiktionary.org/etc; also MediaWiki REST paths need
  `_` for spaces (`+` 404s).  Tests: `tests/unit/test_flux_goals.py`.
  Sandbox surface (afcc357/60955d3): `file <path>` = READ-ONLY local
  file tool — the loop may read anything incl. its own source
  (exact-component denylist for secrets/envs, 256KB cap, works
  offline; `flux_file_root` config).  `note <text>` appends to one
  fixed journal path (`flux_note_file`, default
  `status_dir/flux_notes.md`); `write <relpath>::<text>` appends a
  line under `flux_sandbox_dir` (default `status_dir/flux_sandbox`)
  — resolved path must stay inside the jail (no absolute paths, no
  `..`, deny components blocked), so model text can never redirect a
  write outside the sandbox.  Self-posed goals (2275df2): `flux_self_frac`
  of each epoch's tasks are MODEL-authored questions run through the
  tool path, scored by extractive grounding in the real observation
  (`_flux_score_self` — no canonical key); teach-fail internalizes the
  obs as the answer.  `flux_teacher_channel`/`flux_teacher_decode`
  wire the EXP-8 static Qwen Hedge channel into fresh loop models.
  Still missing vs full spec: write-to-sandbox file editing, stop-on-
  proven-advancement, execution sandbox for model code.
- `forge/engine/decoding.py` — Decoding strategies (standard, speculative
  family, `dola` self-contrastive R50-2, `uno`, `specbound` adaptive
  early-exit self-speculation R60-5 — greedy bit-identical to standard;
  DRY n-gram penalty R50-1 in the shared `_sample_from_logits` chain +
  all `engine_generation` paths, `dry_multiplier`/`dry_base`/
  `dry_allowed_length`/`dry_penalty_last_n` params, 0=off).
  DeepInfer port (deepseek-ai/DeepInfer, R&D round 15): (1) `seed=`
  on `SamplingParams`/`generate`/`generate_raw`/`generate_stream`/
  `decoding.generate` — seq_seed deterministic sampling: SplitMix64
  hash of (seed, absolute position) inverted through the filtered CDF
  (`engine_common._hash_multinomial`), batch/chunk-invariant at bs=1,
  opt-in only; (2) deferred readback now covers HYBRID models too —
  the next forward launches before `.item()` and a per-step
  `_spec_snapshot_recurrent` + `_defer_restore_recurrent` (in-place
  copy_ so `_conv_state_buf`/graph-captured banks keep their storage)
  rolls back on EOS/degen/stop breaks — DeepInfer ZOS
  "launch(N+1) → post-process(N)" on one request, applies to both
  `StandardDecoding.generate`/`_generate_static` and
  `engine_generation._decode_tokens`; (3) `ExternalDraftSpeculative`
  gained `confidence_tau` (build via `build_decoding("speculative",
  confidence_tau=t)`) — draft-prob cumprod prefix-survival ends each
  round at the confident prefix (DeepInfer varlen verify, bs=1 form;
  also fixed a latent bug where `cur_logits` was never updated inside
  the draft loop, so the draft re-proposed the first token every
  iteration); (4) `prefix_cache._bounded_replay_recurrent` —
  DeepInfer SWA bounded replay on prefix hits lacking a position-
  correct conv_snap (legacy dict/LRU entries — the stale cross-request
  `_conv_state` fallback): zeroes live recurrent state, re-forwards
  only the last max(128, sliding_window) prefix tokens, returns a
  full-length repaired KV (bit-exact attention, rebuilt conv/SSM
  boundary). Tests: `tests/unit/test_deepinfer_ports.py`.
- `forge/engine/gated.py` — ForgeGate three-probe gated generation
  (route/doom/convergence); `ForgeEngine.load_gate_probes()` +
  `generate_gated()`; probe bundle `research/checkpoints/gate_probes.safetensors`
  (~44KB, version-guarded). VRAM overhead ~0 (logistic heads on hidden
  states the model already computes). Trained on the gate_r 1450-label
  self-verified corpus; retrain pipeline documented in
  `.devin/scratchpad.md` (R&D: ForgeGate probes). Held-out eval on
  ForgeLM V2: +16.4pts acc at −34% tokens vs always-think; numbers are
  corpus-dependent — recalibrate probes on new domains/checkpoints
  before trusting thresholds. GUI wiring:
  `EngineService._load_blocking` auto-loads the bundle when present;
  `chat_loop._route_p_easy` scores the route head once per turn on a
  TOOLS-FREE render of the conversation (the ~17-schema `<tools>` block
  collapses h_mean and drags p_easy ~0.25 below threshold — "Hello":
  0.355 tools-free vs 0.098 with tools; probe was trained tools-free)
  and switches to a closed-think render (`<think></think>` + "Answer:"
  anchor — bare </think> leaves the model musing) when p_easy ≥ 0.2.
  Bare greetings/acks bypass the probe entirely (`_is_trivial_turn`,
  gate_r has no chit-chat class). Emits a `gate` SSE event the Chat page
  shows as a `gate → mode (p=…)` chip. Routing + conv exit run ONLY on
  fresh user turns (last conv msg is `user`) — tool-continuation rounds
  go straight to think+budget: both probes are OOD on synthesis turns
  (observed: conv-exit forced `Answer:` → model restated its plan and
  re-called the same tool 5×). Think turns get TWO exits: the
  conv probe early-exit (`_conv_exit_observer` scores per-step hidden
  states via `generate_stream(hidden_observer=)`, K=2 consecutive
  >0.75 past 32 tokens → inject force-answer; emits a `conv-exit` gate
  event) AND the hard cap backstop (`_think_cap_processor` injects the
  gated.py force-answer suffix after `think_budget` think tokens,
  default 160, per-request via `ChatSendRequest.think_budget` /
  Chat settings "Think budget", 0=no forced exit) — plus a `<think>`
  (id 541) ban in generated text in every mode (a generated <think> can
  only re-open a reasoning pass — observed: model emitted `Answer:` then
  re-opened <think> and re-looped). Direct path passes `budget=None`
  (ban only). Structural bans also apply every mode: `<|im_start|>` 518,
  `<tool_response>`/`</tool_response>` 539/540 (fake tool results), and
  `<tool_call>` 531 is masked while the think block is open — trained
  order is think → `</think>` → text → call; a mid-think call used to
  leave the block unclosed so reasoning leaked into the persisted reply
  (the "thinking escapes the tool call" bug). `tool_calls_allowed=False`
  (tools off, or defs dropped by the repeat guard) bans 531+532 outright.
  Text-level belt: a rolling 24-char tail stops the stream on literal
  `</tool_call>`/`<tool_response>` emitted as ordinary tokens (id-EOS
  can't see them). Post-parse: `_split_reasoning` puts pre-`</think>`
  text in `msg["reasoning_content"]` (re-rendered into the next tool
  round by the template, rendered as ThinkCard in the UI, dropped when
  a new user turn starts) — unclosed output is all-reasoning/empty-body,
  never a visible reply; `_strip_answer_anchor` drops an echoed
  `Answer:`/`Final Answer:` lead. Fed-back tool results are truncated at
  4000 chars. Two output guards: repeat-call detection (`_call_sig` —
  an identical name+args call drops tool defs so the next round must
  synthesize) and `_strip_direct_musing` (direct-path text before a
  stray `</think>` is reasoning voice — dropped; pre-call musing on
  direct tool turns is dropped entirely). Threshold calibrated via
  `scripts/bench_gate_route.py` — rerun it when the probes are
  retrained or the chat template changes. The agent loop
  (`agent_loop.py`) reuses the same processor with a 384-token think
  budget and a bare `\n</think>\n` close (not the "Answer:" suffix) so
  a capped think can still proceed to a tool call.
- `forge/engine/decide.py` — SystemOneEvaluator: TypeSafe AI "System One"-
  compatible typed decisions (noul/choice/score questions → TypeSafe-style
  probability answers) via candidate-continuation scoring — single forward
  passes, no generation. Reasoning-model aware: primes
  `assistant\n<think>\n</think>\nAnswer:` so answer tokens land at the
  read position. `ForgeEngine.decide(state, questions)` wraps it.
  Server exposes `POST /v1/systemone` (typesafe_sdk drop-in: point
  `base_url` at Forge; `/v1/models` sniffs `X-TypeSafe-SDK` header and
  returns the TypeSafe `{"models": [...]}` shape). Without a scorer,
  probabilities are raw LM softmax — consistent within a question but
  NOT calibrated.
- `forge/engine/decision_head.py` — Tier-1 `DecisionScorer`: linear
  verifier head on last-token hidden states, `s(q,c) = w·h(prompt+" "+cand)`
  softmaxed per question; covers noul/choice/score with one head. Also
  `ProcessRewardHead` (R50-3): sigmoid step verifier on step-end hidden
  states — `fit_prm(model, tok, device, dataset)` trains on per-step or
  outcome-broadcast labels (BCE, frozen base), `score_steps` returns
  P(step correct); ~10KB weights, feeds GRPO advantage shaping.
  Train via `fit_decision_scorer(model, tok, device, dataset)` —
  dataset rows `(state, question_spec, correct_candidate_idx)`, group
  softmax CE (proper scoring rule) + LBFGS temperature scaling; ~10 KB
  weights, `scorer.save()`/`DecisionScorer.load()`. Attach with
  `engine.load_decision_scorer(path)` or `decide(..., scorer=...)`.
  Real-model check (340 mixed templated examples): ECE 0.067 vs 0.141
  raw at equal accuracy — calibrated-style, not production-calibrated;
  validate on non-templated data before confidence gating.
- `forge/keys/` — KeyStack architecture keys (25 canonical + `kda` R49-2
  side-path: `use_kda` config flag, gate=0 bit-exact, KDAKey BI port;
  + `gdn2` R60-1: `use_gdn2` flag, Gated DeltaNet-2 erase/write gates,
  collapses to KDA at warm start — mutually exclusive with `use_kda`).
  `forge/keys/safety.py` is the production `safe_apply`/rollback harness
  (used by pit/lerope/attn_residual/mhc keys `safe=True` paths).
- `forge/quant/` — Quantization implementations. `inference_quant.py`
  `quantize_model_w4()` packs nn.Linear weights into torchao
  `Int4TilePackedTo4dTensor` (W4A16, group=128, tinygemm kernel —
  `aten::_convert_weight_to_int4pack` is CUDA-only, so each linear is
  moved to GPU, quantized from bf16, then released; peak VRAM = packed
  total + one layer). Wired into `from_checkpoint` via
  `load_quantize="int4"` (engine_checkpoints `_load_quantized`): loads
  bf16→meta→assign on CPU, per-module CUDA quantize, skips the
  post-load auto-quant pass AND compile/graph features
  (`_auto_activate_optimal` hard guards — torchao tensor subclasses
  hang inductor, capture is fragile; guards also override caller flags).
  Compatible features auto-enabled for packed pure-attention models:
  learned prefix cache, adaptive+suffix spec decode, FASER, lookahead
  gate, POD attention, hybrid prefill, hot prefix. Excluded: lossy
  KV/attention pruners, RoPE replacers, P-EAGLE (needs trained draft),
  arch-specific (quamba2/replay_ssm/MoE/triton_conv/fa4), extra weight
  quant. `from_checkpoint` splits ActivationConfig-named kwargs into
  `_user_activation_overrides` — forge_server flags:
  `--load-quantize int4`, `--kv-tokens N` (caps KV prealloc).
  Validated on Hermes-3-8B: ~11.5s load, ~7.6GB VRAM incl. KV,
  ~34 tok/s, 28k-token KV budget. Tests: `tests/unit/test_llama31_w4.py`.
- `forge/decoding/` — Decoding implementations
- `forge/training/` — Training runners + optimizers; `training/data/`
  holds the dataset pipeline modules (`efficient_pipeline`,
  `parquet_dataset`, `curriculum_augment`, data-prep one-shots) — it is
  tracked source, exempted from the `data/` gitignore rule.
  `runners/sft_train.py` renders all examples via the CANONICAL
  `chat_template.apply_chat_template` (`<|im_start|>`/`<|im_end|>`,
  `<tool_call>`/`<tool_response>`, thinking=False direct answers). The
  legacy `<|startofsegment|>`/`<|tool_call_start|>` markers are gone —
  they don't exist in the V2 tokenizer, so SFT was training a format
  inference never sees (fixed 2026-09-21; see BUG_LOG).
  Offline on-policy distillation (R&D 2026-11): `scripts/score_teacher_
  logprobs.py` scores each row's completion with an HF or Forge-native
  teacher and writes a `teacher_logp` list (one float per completion
  token, char-offset aligned back onto student tokens — tokenizer-
  agnostic, works LFM3↔Qwen). `sft_train --distill` then trains those
  rows with advantage-weighted CE: `w = floor + (1-floor)·clip(α·(tlp−slp))`
  via `chunked_distill_ce` (`forge/training/losses/chunked_ce.py`,
  never materializes [N,V]; sentinels `TEACHER_LOGP_MISSING=-99` →
  neutral CE; rides `model.config.teacher_logp` per-batch like
  `entropy_alpha`). Knobs: `--distill-alpha/clip/floor`; unscored rows
  train as plain CE; `--teacher-checkpoint`/`--distill-topk` remain
  reserved stubs. CodeEvolve `export_sft` now writes `reward` =
  1−exp(−Δ) from the verified score delta → per-example
  `reward_weights` in collate.
  2026-11-07 hardening (spark sft3 postmortems): tokenizer resolution is
  `_resolve_tokenizer_source` — `--tokenizer` → ckpt dir → bare-file
  `.meta.json` `tokenizer_path` → canonical default (a bare .safetensors
  ckpt previously fell through to the Qwen-vocab default and mis-tokenized
  every row: train loss ~10 vs true CE 0.64); vocab guard = hard error
  when tokenizer vocab > model vocab, loud warning otherwise. Loader
  dedup keys on prompt+response (RFT k-variants survive; prompt-only
  dedup had dropped 259→50 proposer rows). Periodic step saves under
  manual LoRA are adapter/qscale-only by default (`--full-step-saves`
  opts out) — ~7× less SMR I/O, no full-weight CPU copy (the RAM-spike
  kill moment); resume detects `lora_only` in the sidecar and re-injects
  adapters onto `base_checkpoint` instead of loading weights from the
  snapshot. `torch.compile` call fixed (no `cache_dir` kwarg on torch
  2.11 → `TORCHINDUCTOR_CACHE_DIR`; default mode, not reduce-overhead).
- `forge/evolution/` — Evolutionary optimizer (ForgeEvolve); CLI: `python -m forge.evolution --domain <name> --steps <N>` (use `--list-domains` to see all domains). Domain specs + run/focus profiles live in `forge/evolution/configs/` (canonical; the legacy `tests/evolution/configs/` copy was removed — the tests/evolution harness scripts now point at the canonical dir). `--fixate` pins a run to one domain: no cross-domain transfer, no refinement spawning, plateaus re-diversify in place (`_fixate_rediversify`); `run_evolve.py --fixate` additionally kills auto-discover/revisit/parallel waves. `forge/evolution/domains/livet_domains.py` — `MicroLiveLM` (R64): the micro live-format search domain; REAL evaluator (builds ConfigurableResearchLLM+LiveSession on CUDA, measures fact-recall live_gain, bs=1 decode tps, ingest tps, params, revert parity — not synthetic). Structural grid via `discrete_choices` (240 cells) + 9 continuous dims. Focus topic: `--focus micro_format`. Results: conv+titan+ngram substrate dominates the micro frontier (d64-class ~0.2-0.5M params, gain ~5, ~250tps); see scratchpad R64. `forge/evolution/domains/flux_domains.py` — `FluxTune` (R65): FluxLM self-tuning domain; builds real FluxLM (cuda_primary, graphs+compile_dev off — concurrent captures/poisoned RNG state crash threaded evals) and measures held-out NLL, 1-hop + 2-hop chain recall, pred/ingest/gen tps, memory_report+VRAM, argmax diversity; score = weighted log2 ratios vs production defaults (baseline anchors at 25). `forge/evolution/domains/meta_domains.py` — `EvolverTune` (R65): meta-evolution — candidates are ForgeEvolveConfig knobs scored by real inner ForgeEvolve runs on SyntheticDomain (progress vs random-floor + eval efficiency); fixated+warm_start=False inner runs isolate proposal quality. Both set `eval_thread_safe=False` (engine opt-out of `_threaded_eval` side-stream pools — added in engine.py `_gpu_batch_eval`). Categories/focus topics: `fluxlm`, `evolver`. DBs: `forge_evolve_flux.db`, `forge_evolve_meta.db`, `forge_evolve_meta_inner.db` (scratch). R66 proposal frontier (engine.py): proposals are submitted to a pending frontier every gen and popped for eval later — submission is decoupled from eval throughput (`_submitted` counter in logs; measured ~2.1K unique proposals/s on synthetic). `ForgeEvolveConfig.propose_mult` (proposals/gen = n_generators×propose_mult, multi-sample grouped-einsum forward in `BatchedGenerator`), `frontier_cap` (evict lowest surrogate-scored pending), `quick_ratio` (multi-fidelity: pop quick_ratio×, domain `quick_eval` ranks, top fraction promotes to full eval). Domains may set `cpu_eval_ok=False` to opt out of the `parallel_eval` CPU pool (FluxTune/EvolverTune do — GPU-bound/nested-CUDA). Dedup is one D2H transfer + vectorized round for all candidates (was per-candidate `.cpu()`, ~100x). FluxTune quick_eval: ~1/3 corpus, 200 held positions, no decode probe, quality+mem terms only (tps is wall-clock noise at that fidelity), ~3x cheaper, deterministic prod=25.0 anchor. FluxTune hop2 probe is real chain recall (`#V>` -> V_next, the hop-1 value becomes the hop-2 key — measures association chaining, baseline ~2.35 nats). R66b literature fixes (AlphaEvolve/FunSearch/pyribs/MEMES parity): dedup keys on the DECODED config (discrete-bin domains were re-evaluating identical configs under different param vectors); per-generator archive-conditioned contexts (60% UCB elite / 15% random cell / 15% two-elite blend / 10% best — was one shared elite for all generators); rank-shaped scores for trainer+evolve (OpenAI-ES centered-rank fitness shaping — -1e9 failure spikes no longer poison fitness_ema); explore emitters (first 20% of generators credited on rank-shaped novelty, MEMES-style); crossover emitter (`crossover_frac`, blends two elite encodings + jitter as direct proposals, generator_idx=-2); islands (`n_islands` deme archives — generator g feeds island g%K, weakest half import global elites every `island_migrate_gens`).
- `forge/evolution/scout.py` — Scout (OpenEvolve-lite): lightweight goal-driven discovery engine layered on the same domain specs + simulators + FindingsDB, replacing the 500-generator MLP population / gen-model manager / weight-BLOB storage with three small ML components: (1) numpy-only TPE proposer (Parzen l(x)/g(x) density-ratio over top-γ history + elite archive parents; per-dim bandwidths shrink where |corr(param,score)| is high — learned knob importance; `--mode random` = baseline), (2) Thompson-sampling Beta bandit over domains so cycles flow to whichever domain keeps paying off, (3) cross-domain param priors — `scout_param_stats` holds global per-(param-name, bin) Welford stats of within-domain z-scored evals, so a fresh domain inherits good regions for params shared with other domains (`transferable=false` specs neither leak nor receive). Promotion gate: candidate must beat the incumbent's re-measured mean by max(min_delta, rel_delta·|inc|, k·pooled SEM) over `--verify-reps` replicated evals of BOTH configs (fresh seeds; z-score recorded in metadata) — only then is it written to `discoveries`. Stagnation: domain retires after `--patience-dom` (40) dry cycles (basin hop at `--hop-after` 12 first), run stops after `--patience` (30) globally dry cycles or when all domains retire. Goal routing: `--focus memory|speed|quality|training` maps to spec categories (mirrors focus_profiles.json); `--require "metric>=v"` adds an end-result gate on promotions. CLI: `scripts/scout_loop.py` (e.g. `--focus memory --cycles 300`, `--report` for discoveries + cross-domain insights). ~20 evals/s on CPU, zero VRAM. Tests: `tests/unit/test_scout.py` (10). Measured smoke: 37 verified discoveries / 640 evals / 32s on `--focus memory`.
- `forge/evolution/code_evolve.py` — CodeEvolve (OpenEvolve-style *program* evolution, the code-level sibling of Scout): evolves actual implementations, not just parameter values. A `Target` file declares `SEED_CODE` + `REQUIRED` names + `test(ns)` correctness gate + `bench(ns)` returning real metrics + `METRIC`/`GOAL`/`BUDGETS`/`TAGS`. Loop: proposer -> free syntax/required-name/dup gates -> `_candidate_eval.py` subprocess runs test+bench in isolation (invoked BY FILE PATH, pure-stdlib, never imports `forge.evolution.__init__` = ~0.2s not ~10s/eval; preamble injects `np`/`torch` only if referenced) -> score = sign*metric vs incumbent -> replicated verification re-benches challenger AND incumbent `reps` times (mean/SEM/z + min_delta + rel margin) -> promote only on verified win. Compares vs original seed, last best, AND the round's own candidates (per-island elite archives feed parent selection). Islands + Thompson arms + basin hop at `hop_after`; stagnation check sits at TOP of loop so every cheap-reject `continue` still counts — run stops after `patience` dry cycles. Proposers (`--proposer`): `mut` = `forge/evolution/mutator.py` AST mutation engine (default; no model, ~0ms/candidate — sort-reuse, `_ctor` ctor hoisting, diff/masked_fill/funcify swaps, literal perturb, elite crossover) with a **Thompson bandit per op persisted in `mutator_op_stats`** — every eval feeds credit back via `Mutator.note()`, so the policy literally resumes across runs and shares by target tag_sig; `mix` = mutations + LLM leap every `--llm-every`; `llm` = coder model via forge/exp harness (Qwen2.5-Coder-0.5B default — measured weak: 17/21 verbatim copies; any other `--model` routes through transformers AutoModel + chat template, e.g. cached Instruct variants); `none` = seed-eval only. **Resume** (default on, `--fresh` to disable): prior best program becomes incumbent (re-measured), last-2000 code hashes pre-seed dedup, top-24 programs repopulate island elites. `code_programs`/`code_insights`/`mutator_op_stats` all live in the same FindingsDB; `--export-pairs out.jsonl` writes DPO pairs (chosen=improved child, rejected=parent) for the future proposer fine-tune. `--emit-target pkg.mod:fn` scaffolds a target file from any codebase function. Targets: `forge/evolution/targets/` — ForgeEngine hot paths: `sample_pipeline` (`_sample_next_token`'s rep→temp→DRY→min_p→min_k→top_k→top_p→vocab-order chain, every decoded token), `suffix_mem` (SuffixMemory feed/lift/draft per-token sidecar), `radix_prefix` (RadixPrefixCache match/insert/evict, list-API so the surface is the tree logic), `stop_matcher` (`_StopMatcher` incremental stop-string matching), `prefix_hash` (blake2b rolling chain), `canon_key` (mutator's own canonicalizer — self-target), plus `min_k_filter` (mutator found +33% verified, ~+50% cumulative — the production win is wired into `engine_common`), `top_p_filter`, `top_p_probs`, `quantize_4bit`, `dequantize_4bit`, `dry_penalties`, `fast_softmax`. Tests: `tests/unit/test_code_evolve.py` (26). Fixed bugs: `continue`-paths skipped bottom-of-loop stagnation check (infinite reject loop); `_load_insights` SELECT missed the `target` col → resume-time KeyError; module-level `sqlite3` import missing (ALTER-race handler would NameError); `_promos` never incremented (status/watch reported 0); `_detect_cuda_target` body-only AST scan missed module-level `.to("cuda")` (VRAM-capped autoscaler); `LLMProposer._engine` waiters crashed unpacking on a failed builder's entry (now retries on a fresh entry so the real error surfaces); `_seed_signature` couldn't prime `class` targets (SuffixMem/RadixPrefixCache got `def SuffixMem(` primes → function proposals, guaranteed test-fail).
  - **Throughput pipeline** (post-audit rebuild; measured 0.92→~23 evals/min on top_p_filter, dup rejects 91.6%→~6%): `forge/evolution/__init__.py` is LAZY (PEP 562 `__getattr__` exports) — spawn workers used to import the whole legacy chain (torch+engine+domains) just to resolve `_worker_main`, ~335MB RSS + seconds per worker; `domains/__init__.py` `DOMAINS` is a deferred-merge registry (`_DomainRegistry`) since domain_spec↔domains is a real import cycle once the package init stopped loading domains first. `forge/evolution/eval_pool.py` — `EvalPool`: N persistent `spawn` workers, target preloaded once, candidates over per-worker `mp.Queue`s, hung/dead workers killed+respawned (`max_jobs` self-recycle bounds RSS); `eval_many()` = parallel verify reps; `eval_sync()` for seed/resume. Default ON via `eval_pool=True` whenever `workers != 1` (`--no-eval-pool` / workers=1 = old subprocess path); pool size `min(max_workers, 4|16)` by cuda-target scan. Dead-job sweep distinguishes clean exit (`exitcode==0` = recycle raced our submit → job never started → REQUEUED onto the respawned worker) from crash (nonzero → fail); boot-failure circuit breaker: `_boot_trip` consecutive zero-completion worker deaths (target import crash, spawn env failure) opens the circuit → `submit()` returns staged `stage="spawn"` results instead of respawn-burning a process per proposal forever. `FindingsDB.release()` transiently drops the WAL conn between campaign rounds (lazy-reopen on next access — open conns pinned frames across sequential `evolve_all` targets); `close()` stays terminal. `_verified` caches the incumbent's rep scores for `inc_stale` (48) evals instead of re-benching every attempt. `Mutator.mutants_for(src, insp)` — deterministic op×site enumeration (per-site literal/funcify/masked_where/locals_bind/crossover + singles + two-op compositions, bandit-ordered) yields `(op_sig, code)` — uniqueness by construction keyed `(parent_h, op_sig)`; `proposer=basin`: background thread emits batched LLM parents (`LLMProposer.propose_batch`, `basin_batch`=8/pass) into `_basin_q`; families drain via `_propose_basin` — blocks up to 2.5s for a basin when the enumeration runs dry, `_dry` only when the emitter is dead/absent. Measured: +23.2% verified promotion on top_p_filter at eval 20 of a 44-eval run. **Promotion-integrity invariants** (regression-ratchet fix): `_handle_result` requires the verified `challenger_mean` to beat the *recorded* `incumbent_score`, not just the same-regime re-measure — the stored bar is monotonic within a run (kills the inflate-incumbent-remeasure-under-contention ratchet: measured 0.294->0.345->0.703 ms fake promotions on quantize_4bit). `_resume` floors the incumbent score at the resumed program's own DB-recorded score (`max(fresh, recorded)`) so a boot-contention-inflated re-measure can't lower the bar across runs either. Worker RSS watchdog: `worker_mem_mb` (default 2048) polls busy workers ~2Hz via psutil, kills+respawns on breach, result tagged `stage="mem"`; spawn baseline ~335MB (torch via forge chain) so caps must exceed it. `rsi_evolve.py` honing checks `manifest.json` (LoRARegistry.MANIFEST), not registry.json.
  - **Prior-art ports (R&D round: FunSearch/OpenEvolve mechanisms)**: `island_cull=True` basin hop now discards the worst ~half of islands and clones the best island's elite set + bandit arm into the least-bad culled slot (emptied slots reseed via `_pick_island`'s fresh-island preference; `island_cull=False` = legacy top-1 reset). `map_elites=True` adds a second elite feature dim — `_push_elite` keeps per-(island, code-length-bin) cells (bin = code_len vs island median; each bin gets a guaranteed floor of ⌊n_elites/4⌋ elites then remaining slots fill by score) so short and long program families both survive without shrinking a single-bin pool. `cheap_bench=True` eval cascade: first-pass candidate evals carry a `cheap={reps,metric,incumbent,goal,margin}` spec into `_candidate_eval.py`/`EvalPool` workers — when the spec defines `BENCH_REPS > cheap_reps` a reduced-rep bench runs first and candidates landing >`cheap_margin` (default 1.0 = ~2x) worse than the incumbent report `stage="cheap_gate"` and skip the full bench (verify/seed/resume evals always run full reps). `hint_bandit=True`: `LLMProposer._HINTS` strategy lines are now a Thompson Beta bandit persisted in `mutator_op_stats` as `op='hint:<idx>'` keyed by tag_sig — attribution flows prompt→hint via `_hint_by_prompt`, credit via `LLMProposer.note(code, credit, delta, prompt)` on the same scale as `Mutator.note` (which now takes the optional `prompt` kwarg; `_feedback` notifies both `self._proposer` and `self._basin_llm`, old 3-arg note() still works). `export_pairs`/`export_sft` write atomically (tmp+`os.replace`), as does sft_train's `--save-lora-adapter` snapshot. `sft_train --resume <ckpt>` consumes the `.train.pt` sidecar via `load_training_state` (optimizer+RNG+EMA+step, `start_step=step+1`, overrides `--checkpoint`) and a SIGINT handler emergency-saves `interrupt_step<N>` (dpo_align pattern, `current["step"]` tracked per step). Tests: `test_hint_bandit_attribution`, `test_island_cull_clone`, `test_map_elites_len_bins`, `test_cheap_gate_cascade`.
  - `scripts/rsi_evolve.py` — RSI campaign driver: per round runs every engine-target (basin proposer default, workers=0 auto — the 15-entry `DEFAULT_TARGETS`: 13 code targets incl. the engine decode/cache hot paths + `cev_math_solver`/`cev_rle_codec`, plus auto-harvested `auto_*` targets merged via `collect_targets()` — `--no-auto-targets` skips), then `export_sft` → `finetune_proposer --mode lora` → tagged adapters + router → next round's `lora_root` hot-swaps them — the proposer literally trains on its own verified wins (`--no-train` to evolve only, `--min-new` gates honing, `--quiet-rounds`/`--minutes` stop conditions, `--status` per-target + `.events.jsonl`). Each round also runs `export_validity_pairs` → `<sft>.validity.jsonl` (validity-DPO: chosen=gated program, rejected=the model's own malformed/dup emission — the format-compliance signal; a `dup` row whose stored code IS the gated program gets a DIFFERENT sibling as chosen, or is skipped when none differs). `scripts/evolve_all.py` — the evolve-only campaign sibling (13 code targets + `cev_math_solver`/`cev_rle_codec`; `--auto-targets` merges `collect_targets()` output as in rsi_evolve; `--rounds 0` = until quiet, `--status <stem>` per-target JSON for evolve_watch, basin/fanout knobs, `--workers 0` auto pool default, `--export-pairs`/`--export-validity` at campaign end); both drivers derive per-target seeds via `zlib.crc32(name)` (deterministic across processes — `hash()` salted by PYTHONHASHSEED made the same campaign config unreproducible). Prompt side: `_parents` also returns FunSearch best-shot **k=2 shots** — when an island holds ≥2 programs the prompt shows the pair sorted worst→best as `v0 (score X)`/`v1 (score Y)` and asks for `v2` (single parent = old "Current best" framing). Artifact feedback (`cfg.repair_regen`, `repair_per_batch`=4/batch): a REAL error result (exec/test/bench `detail` — tracebacks, not bare "failed" score-misses) queues ONE repair regen — original prompt + `# Previous attempt failed with:` + error truncated ~400 chars — drained by `_drain_repairs` as a single batched regen (same pattern as `_research_followup`), re-gated + re-submitted like a fresh candidate; the marker in the prompt prevents chained re-repairs, basin mode repairs through `_basin_llm`.
  - `scripts/fn_evolve.py` — the single-target CodeEvolve CLI (the campaign drivers wrap it): `--target <name>` (or `--emit-target MOD:FN` to scaffold a target spec from a live function), `--proposer mut|mix|llm|basin|none`, `--basin-batch`/`--fanout`, `--workers 0` = AUTO ramp (gated by CPU/RAM and free VRAM for CUDA-bench targets; 1 = serial subprocess debug path), `--evals`/`--patience`/`--hop-after`/`--reps`, `--report` (best-program table from the db, no run), `--export-pairs` (DPO child>parent) and `--export-validity` (the validity-DPO export above), `--lora-root`/`--lora-tag` adapter hot-swap, `--fresh` to ignore target history.
  - **Web-research tools for the proposer** (`forge/evolution/research.py` + `LLMProposer` SEARCH loop): `research.py` is a urllib-only tavily/exa client (`tavily_search`/`exa_search`, keys from env or repo `.env` — TAVILY_API_KEY / EXA_API_KEY; tavily→exa fallback when tavily yields only errors). `research()` queries the ~12-topic `TOPICS` table (real forge/ functions → search queries) and writes `docs/evolve_research.md`. With `--research` on `rsi_evolve.py` (→ `CodeEvolveConfig.research` + `research_digest` + `research_budget`): the digest refreshes at round start when missing/stale >20h, a ~1.6KB trimmed copy is injected into every proposal prompt, and the proposer may emit `SEARCH: <query>` lines instead of code — `_research_followup` intercepts them BEFORE `_extract_code`, runs the cached/budgeted search, injects a comment block of hits (title | url + 1-line snippet, `research_hits`=3 per query), and regenerates ONCE (one batched extra round-trip for `propose_batch`, serial for `__call__`). Caps: `research_max_queries`=2 queries/proposal, 1 regen/proposal, `research_budget`=32 uncached API calls/process (class-level `_SEARCH_SPENT`, shared across proposer instances — ~daily quota). Query cache: in-process `_SEARCH_CACHE` dict + append-only jsonl at `cfg.research_cache` (default `<lora_root>/research_cache.jsonl`) — repeat queries across runs cost nothing; keys are never printed or persisted. Proposal-side ONLY — eval workers never see this path. Two more tool lines share the same caps (`research_max_queries` total per proposal, ONE regen): `FETCH: <url>` pulls page text via `research.jina_fetch` → `trafilatura_fetch` (~1.5KB injected; consumes the web budget + rides the same query cache as `fetch:<url>` keys) and `INSIGHT: <topic>` retrieves `code_insights` rows by tag/topic overlap (local `mode=ro` sqlite read — free, no budget spent). Tests: `tests/unit/test_code_evolve.py` (`test_search_*`, `test_fetch_and_insight_tools`, stub `generate_batch` engine — no model load).
  - `forge/evolution/targets/canon_key.py` — self-referential target: `canon_code(src)` = the mutator's dedup/no-op canonicalizer (`ast.unparse(ast.parse(src))`). Test enforces identical equivalence classes vs the reference on a 30-pair corpus; bench uses unique program strings so memoization can't game it. First self-run: **+22.6% verified** (locals_bind attr→local binds; wired back into mutator as `_parse`/`_unparse` module-level binds). Pool safety: `mem_cap_mb`/`--worker-mem-mb` RSS watchdog (kill+respawn, stage="mem"; cap must exceed the ~335MB torch-boot baseline). `basin_starve_s` (240s) covers the basin-wait design: a live emitter that hasn't produced a productive basin isn't stagnation (model load + ~10-72s batches are latency). `_LocalsBind` binds attrs used ≥1× (was ≥3×) — thin single-return seeds get enumerable surface.
  - `forge/evolution/auto_targets.py` — auto-harvested evolution targets: production functions opt in via `@self_improve(gen_input=..., deps=..., globals=..., imports=..., evals=...)` (registration only, zero import cost); `collect_targets()` pulls the LIVE function source (seed can never drift like hand-copied seeds), wraps it in the standard spec contract (generic `_same` oracle: finite-mask+allclose tensors incl. complex, recursive list/tuple/dict, isclose numbers; `_gen_cases` called twice so candidate vs ref see fresh-identical inputs), writes `forge/evolution/targets/auto_<name>.py` (prefix reserved; stale auto_* pruned) and returns `{name: evals}`. `rsi_evolve.py` merges them into its budget map by default (`--no-auto-targets` to skip); `--targets` subsets accept auto_* names. First batch (12, `AUTO_MODULES`): `engine_common._truncate_at_stop`, `decoding._clone_state_tree`/`_spec_slice_kv`/`_ngram_lookup`/`_jsd` (latter two extracted module-level — methods delegate), `prefix_cache._slice_past_kv`, `mamba3.complex_rmsnorm`/`exp_trapezoidal_discretize`, `w8a8_quant.w8a8_quantize_activation` (extracted; method delegates), `decision_head.expected_calibration_error`, `attention_ops._build_block_diag_causal_mask`/`_causal_mask`. Contract: gen_input must be a zero-arg, self-contained, deterministic module-level `def` returning representative cases — fake inputs = fake wins. Regen: `python -m forge.evolution.auto_targets`. Tests `tests/unit/test_auto_targets.py`.
  - **Non-code domains** (`forge/evolution/domains/cev_*.py` — spec-contract targets, NOT BaseDomain): `cev_math_solver.py` (`solve_problem` over generated arith/linear/gcd/modpow cases, `accuracy` max, seed covers only the easy slice) and `cev_rle_codec.py` (`compress`/`decompress` byte codec, `ratio` min on a runs+patterns+noise corpus) — proof the loop isn't codegen-only. `Target.resolve` checks `targets/` then `domains/`; both are in `rsi_evolve.py` `DEFAULT_TARGETS` (60 evals) and runnable via `--targets cev_math_solver,cev_rle_codec` or standalone `python forge/evolution/domains/cev_*.py`. Test: `test_noncode_domain_specs`.
  - `forge/evolution/research.py` — web-research helper for the loop: `TOPICS` maps forge/ functions to search queries, writes `docs/evolve_research.md` {technique, url, applicability}. Backends: `free_search` — KEYLESS chain tried FIRST (ddgs multi-engine → arXiv Atom + Semantic Scholar for paper-ish queries); the SEARCH tool works with zero keys. Paid fallback: `tavily_search` → `exa_search` (keys from `.env`) only when the free chain yields no hits. Tavily 432 / Exa 401 as of this round — free chain carries the load. Scraping: `jina_fetch(url)` (keyless `r.jina.ai/<url>` reader, ~20 RPM) + `trafilatura_fetch(url)` (lazy optional dep) — shared by `engine_tools._web_scrape`, which now chains Firecrawl → jina → trafilatura; `_web_search` chains Tavily → `free_search` → Exa (paid first, keyless keeps the tools alive on missing/dead keys).
  - `_lora_base_ok(ent, engine_ckpt)` (code_evolve.py): adapter/base guard in `LLMProposer._ensure_lora` — registry `resolve()` falls back to `default` without checking provenance, and a wrong-base adapter crashes the forward on name-matched wrong-shape modules (observed: qwen3b_hone2_fq-trained `default` on the mla_moe proposer, every basin proposal died on 896-vs-2048). Dir-like bases compare resolved paths + dir-name tail; HF ids compare name tails; empty base keeps old behavior.
  - **LFM3 proposer pipeline pitfalls (verified 2026-10-30)**: (a) *Adapter poison*: thin-data scale-2.0 LoRA adapters look loadable yet drive the proposer to 0% parseable output (token soup) — observed `default` (17 rows) AND `min_k_filter`. `_ensure_lora` now runs a **canary** post-attach (two greedy probes must compile else unload + tag renamed `quarantined_*` in manifest); `finetune_proposer` defaults `--lora-alpha 32` (=rank, scale 1.0) and skips registering adapters with <`--min-rows` (32) trained rows. (b) *Primed prompts need ChatML*: `primed` prompts previously skipped the template entirely — chat-tuned ckpts babble on raw completions. `_wrap` now puts the ```` ```python`` `+signature prime inside the *assistant* turn (template + priming both intact; verified: `return [x ** 2 for x in ss]` vs raw-prime `a(a(b(`). (c) `<think>` budget: LFM3-CC is a thinking ckpt — unprimed prompts burn ~150 tok reasoning before code; `max_gen_tokens` raised 448→640 and the prompt says "skip think for small changes". (d) `gen_tokens`/`had_think` now logged per gen-stage row; `[cevolve] reasoning:` prints think-vs-direct ok-rate per target stop. (e) *Same-model judge* (`cfg.judge`, default on): promotions get a critique (winner vs beaten incumbent, `self._prev_incumbent`) and plausible-but-wrong rejects (stage in test/exec/bench — NOT 'gen' soup) get a why-failed critique; both land in `code_insights` kinds `judge`/`judge_fail` → retrievable by INSIGHT: and `_relevant_insights`. Judge calls are advisory try/except — never block eval; tests that need deterministic proposer queues set `cfg.judge=False`. (f) *Batched decode*: `batched_decoding.generate_batch` vectorized — uniform top_k/top_p run one batched op (mixed requests keep per-row), one `(B,V)` `multinomial` per step via `batch_gen` (seeded deterministically from the row-seed tuple — basin emitter seeds EVERY row, which had forced the per-row path), `.tolist()` stop-check (was B×`.item()` syncs/step), and `generated` preallocated to `prompt+max_tokens` with `gen_lens` tracking (was 32 `torch.cat` reallocs/step — allocator churn at ~12GB VRAM). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` is UNSUPPORTED on this Windows torch build (warns, no-op) — don't rely on it. (g) `--max-workers 16` default overcommits RAM (each spawn worker ~2GB torch RSS → >32GB); run campaigns with `--workers 4 --max-workers 4`. Worker native crashes (`python311.dll` 0xc0000005) observed at 16-worker auto-ramp = paging-commit exhaustion, not a Python error. `_cache_w` on packed linears stays False for the proposer — decode M≤32 uses the fused Triton GEMM on packed weights; caching ~142 bf16 copies costs ~4GB VRAM that eval workers need (prefill M>32 pays one dequant/batch instead).
- `scripts/finetune_proposer.py` — proposer fine-tune driver on the canonical trainer: exports verified parent->improved-child pairs from `code_programs` via `export_sft` (`{prompt, response}` — sft_train's contract; real logged prompts where present, template-reconstructed otherwise, `# Candidate: <parent_hash8>` line keeps rows unique under sft_train's prompt[:200] dedup), then invokes `forge.training.runners.sft_train --hf-model <HF id> --no-bitnet-everywhere` (bitnet/lora default ON — bitnet-everywhere keys would break plain HF reload; PEFT LoRA merges at save), and assembles the saved safetensors + base tokenizer/config into a loadable HF dir (`forge/evolution/proposer_ckpt`) that `fn_evolve --model <dir>` loads via `load_hf`. Acceptance on 33 rows/60 steps: ckpt loads, ChatML path consistent (`primed=False`), 3/3 batched candidates compile. `LLMProposer.primed` auto-detects: qwen25_05b ForgeEngine path = completion-primed; dirs with a chat_template / HF ids = chat path. Proposer engine is ALWAYS ForgeEngine (transformers only on arch mismatch): `use_chunked_prefix_cache=True` on every path — basin batches share ~90% prompt prefix, measured ~10x on warm shared-prefix batches (0.8s vs 8.2s). `__init__` spawns a `proposer_prewarm` daemon thread so engine build (~5s weights+warmup) overlaps pool boot — model ready before first basin. `rlvr_train.py` (GRPO, task-type code) and `dpo_align.py` remain the next tiers if SFT saturates.
- **Ported proposer (default since this round)**: `forge/evolution/proposer_mla_moe/` = `proposer_ckpt` (SFT'd Qwen2.5-Coder-0.5B) ported to ForgeAI MLA+MoE by `scripts/port_qwen_mla_moe.py` (GQA→latent-KV SVD d_c=256 full-rank, dense SwiGLU→4 experts dense_bypass; preset `qwen25_05b_mla_moe`; logit parity vs source cos≈0.99999). `LLMProposer._model_id` defaults to this dir when `forge_convert.json` exists — the evolve loop now dogfoods the engine's GLA/MoE path. `forge_convert.json` may carry `"primed": bool` — explicit override checked BEFORE the chat_template heuristic; proposer_mla_moe sets `primed:true` because its tokenizer ships a chat_template but the weights only produce code on primed completion prompts (ChatML wrap → degenerate output, verified). HF fallback after a ForgeEngine load failure is skipped for forge_convert dirs (renamed keys can never load in transformers — surfaces the real engine error instead). Basin emitter builds prompts with the LLM's own `primed` mode and logs EVERY raw completion as `stage='gen'` rows (~9x more captured SFT/DPO pairs; previously ~90% of model output was dropped silently). `FindingsDB._connect` sets `busy_timeout` BEFORE `journal_mode=WAL` + retries — parallel targets on a fresh db no longer crash on the WAL race.
- **Proposer default (audit A5, 2026-10-31)**: `LLMProposer._model_id` now prefers `Qwen/Qwen2.5-Coder-3B-Instruct` when already in the HF cache (`snap_dir` local check — no implicit download), then `research/checkpoints/lfm3_1p7b_cc_fq`/`lfm3_1p7b_cc` as the offline fallback. Gate-rate head-to-head (`scripts/bench_proposer.py`, 12-prompt grid → `results/proposer_bench*.json`): Qwen-3B **100% valid / 100% syntax** vs lfm3_1p7b_cc_fq 33%/50% and unquantized lfm3_1p7b_cc 50%/75% — int4 costs ~17pts, capability gap is the rest.
- **Tagged-LoRA subsystem** (per-topic hot-swap adapters; base weights never touched):
  - `forge/evolution/lora_registry.py` — `LoRARegistry` (manifest.json: tag -> {file, rank, alpha, target_modules, base, meta}) + `LoRAManager.plan(rows)`: decides per training group — existing tag → `update` (warm-start via sft_train `--lora-init`), ≥`min_new` rows unknown tag → `new`, small groups → fold into `default` (anti-spam), `stale` meta/`--lora-fresh` → `retrain` same tag fresh weights. `reg.mark(tag, **meta)` writes post-campaign health stats the next plan reads.
  - `forge/evolution/lora_router.py` — `LoRARouter`: decision-model routing. Features = `ForgeEngine.embed()` (model's OWN mean-pooled last hidden state), head = aux MLP (`train_mlp`/`load_aux_model`), stored as `router.safetensors`+`router.json` in the registry. Uncertain/untrained → registry fallback chain. Trained at end of `finetune_proposer --mode lora` on each tag's prompt corpus (disable: `--no-train-router`).
  - Proposer hot-swap: `LLMProposer._ensure_lora` swaps adapters on the shared engine inside the gen lock (idempotent via `eng._evolve_lora_tag`); per-prompt routing for serial gen, first-prompt routing per batch. Config: `lora_root`, `lora_tag` (override), `engine_config`, `proposer_quantize` (engine `load_quantize` int4/fp8/int8 for big bases like Qwen3-4B).
  - Provenance: `code_programs.adapter` column records which adapter produced each candidate (mut slots log NULL) — feeds router retraining + manager health stats.
  - Train: `sft_train --manual-lora --save-lora-adapter --lora-init <prev>` → `<save>.lora.safetensors`; `finetune_proposer.py --mode lora` runs the full plan→train→register→router pipeline. **The hone trains ON the deployed checkpoint** — `finetune_proposer._resolve_forge_base` resolves base→(ckpt_dir, forge_config, `--lora-targets`) and invokes sft_train via `--checkpoint/--config/--tokenizer` (never `--hf-model`), because adapters carry model-native names (`blocks.N.attn.*.lora_adapter.*`) and `_load_lora_impl` copies by exact match — HF-trained keys (`model.layers.*`) loaded 0 tensors on the Forge port and were a silent placebo. Per-arch targets: `qwen25_05b_mla_moe` → `q_proj,out_proj,kv_down_proj,k_up_proj,v_up_proj,k_rope_proj,w1,w2,w3` (router untouched, out<64). Overfit brakes wired in: `--lora-lr 1e-4`, `--lora-val-every 15`, `--lora-val-size 0.15`, `--lora-early-stop 2`, plus `_stale_adapter_tags` — a tag with ≥24 adapter-attributed children and <5% child-beats-parent win rate gets `reg.mark(stale=True)` → manager emits `retrain` (fresh init, not warm-start). `_ensure_lora` prints a loud warning when an adapter loads 0 tensors — a placebo hone can no longer pass silently. sft_train also gained `--tokenizer` (the port's Qwen tokenizer — canonical LFM2.5 would corrupt the 151936-vocab rows) and `--lora-targets` args.
- `forge/evolution/domains/mech_domains.py` — `MechSearch` (R67): mechanism discovery for the FluxLM/micro-format dev plan. 22-dim candidate space spanning the *mechanism surface*: attn_type (gqa/gta/gla), delta path (none/kda/gdn2), titan fast-weight rank compression (0/d2/d4), solver, ngram sidecar+orders, live_projections (bounded substrate drift), write_threshold (surprise admission gating), forget/surprise_cap/chunk — real eval builds ConfigurableResearchLLM+LiveSession on CUDA (live_gain, decode/ingest tps, params, revert parity; same scoreboard as MicroLiveLM). `quick_eval` tier + `gpu_pool_ok` (spawn workers; small eager models). Smoke: gla+live_proj+gate 49.36 ≥ R64 winner 49.33. Category `mech`.
- `forge/evolution/engine.py` GPU eval pool (R66c): `gpu_eval_workers>1` spawns N `spawn`-context worker processes (`_gpu_worker_init`/`_gpu_eval`), each owning its own CUDA context + domain instance — evals share the device but never a context/graph/RNG. Opt-in ONLY for domains declaring `gpu_pool_ok=True` + `gpu_worker_kwargs()` (currently just `FluxTune`; quick tier routes through `quick_eval` inside workers). Worker exceptions → -1e9 fail-score; pool failure → serial `_gpu_batch_eval` fallback. VRAM rule of thumb on the 12GB 5070: workers x worst-case candidate VRAM (prod mem_preset ≈ 4.2GB) must fit — `gpu_eval_workers=2` measured ~1.8x eval throughput; 3 is risky. Cleaned up in `run()` finally alongside `cpu_pool`.
- `forge/evolution/program_vm.py` + `domains/math_prog_domain.py` — `math_prog` / `MathProgDomain`: batched program-synthesis domain. Candidates ARE algorithms — fixed-length register-machine programs over a closed opcode table (15 total ops, protected div/mod/pow, no eval/IO/loops → nothing to scan; ISA is the sandbox). Decoded from generator `[0,1]^45` vectors (12 steps × op/a-addr/b-addr + 8 consts ±64 + out-sel). `run_programs` executes [B,T,R] register buffers fully vectorized on CUDA (~2700 programs/s; bench: `scripts/bench_math_prog.py`). Tasks randomized per eval: FUNC (match hidden f, held-out graded), GOAL ("A must equal T" — compose bounded consts; pinned via `goal_target`), GOAL_IO (constraint pairs incl. pinned `goal_inputs`+`goal_target`, oracle shifts `h(u)+(T-h(u0))` so held-out still forces real math), FIT (kind 3: approximate a pinned `target_fn` — e.g. tanh/sigmoid/log2/sin/rsqrt — graded by `fit_score`=-log10(max_err) + continuous proximity kernel; op whitelist via `ProgramSpec.op_ids`/`--ops cheap|cheapdiv` restricts the ISA to deployable ops). Anti-gaming: `heldout_exact`, `uses_inputs` (dependency-cone only — dead-end reads don't count), `const_cheat`, `degenerate` metrics → RewardGuard in `configs/domains/math_prog.json`. Batched path: `MathProgDomain.batch_evaluate` is auto-delegated by `BatchedEvaluator` (any domain defining `batch_evaluate` wins). Driver: `scripts/evolve_math_prog.py --target 794 [--inputs 12,7] | --fit tanh --ops cheapdiv`. Tests: `tests/unit/test_math_prog.py` (27).
- `forge/self_play/` — Self-play + discovery (`infinite_loop.py` is the RSI
  loop; `live_status.py` writes live telemetry — status.json, heartbeat.json
  with progress-coupled stall detection, events.jsonl — to
  `research/checkpoints/self_play/` for the GUI Self-Play page + CLI polling;
  `--status-dir`/`--no-live-status` flags on the loop). Live spectation:
  `Services._selfplay_watch` (deps.py) tails `EventsReader` every ~400ms and
  republishes onto the hub — `selfplay_event` / `selfplay_status` ws events,
  while `/api/selfplay/status` stays for hydration; `EventsReader.poll`
  survives shrink/ctime-rotation/torn O_TRUNC rewrites (rewinds mid-line).
  `task_done` carries bounded `preview` (model output ≤200 chars) +
  `expected` (≤80); flux mode pushes `flux_cells/journal/knn/epi/mb` +
  `flux_channels` hedge weights into status.json for the FLUX memory card. RSI data path
  (anti-overthinking): `_assemble_solution` AST-cleans every reply
  (comments + docstrings stripped — model musing can't leak into SFT/GRPO
  data); export picks the SHORTEST passing attempt and mixes
  `concise_qa_per_epoch` verified short-answer pairs (default 12, computed/
  curated answers — never model-generated) so each epoch trains direct
  minimal answers; GRPO groups get `grpo_concise_bonus` (default 0.1)
  length-efficiency shaping — correctness dominates, and all-pass groups
  regain nonzero advantage. CLI: `--concise-qa-per-epoch`,
  `--grpo-concise-bonus`. Debug: `--debug-round` runs ONE self-play round
  (propose→solve) with NO training — a `GenTracer` on the curriculum
  records every generation (rendered prompt, raw pre-strip output,
  per-token ids), prints it live, and dumps
  `status_dir/debug_round_ep*.txt|.jsonl`. Use it to check model behavior
  before a real RSI run.
- `forge_gui/` — Qt-free domain layer (`api/` only; PySide6 shell removed).
  Web tools live in `forge_gui/api/web_tools.py` (defs + dispatch) backed by
  stdlib-only `forge/web_primitives.py`: `web_search` (multi-engine chain:
  DDG HTML → DDG Lite → Bing → Google News RSS — first parseable result
  wins), `news_search` (Google News RSS — use for "the news"/current
  events; empty query → top headlines), `web_fetch` (ANY http(s) URL or
  bare domain — auto-https, follows redirects, reports `final_url`; HTML
  → markdown-ish text with inline `[label](url)` links + a deduped
  `links` list of every anchor so the model can follow link names;
  `offset`/`next_offset` page long docs; non-HTML bodies pass through raw
  with `content_type`), `wikipedia_search`, `arxiv_search`. All keyless
  GETs; never gated by safety checks, agent approval modes, or read-only
  chat mode. `parse_ddg_html` drops DDG ad/tracker links (y.js
  redirectors carry u3= not uddg= and can't be unwrapped). `web_fetch`
  clamp is 200–16000 chars via `max_chars`. The
  agent-loop fallback harness also wires `WebTools` so web tools exist even
  if the deps harness factory fails.
- `forge_gui_server/` — FastAPI GUI backend (REST `/api` + WebSocket `/ws`)
  serving the React UI; ports `forge_gui/api/` to plain async services.
  Launch: `python -m forge_gui_server` (desktop window via pywebview),
  `--browser`, `--no-window`, or `--dev` (API only, Vite dev on :5173).
  EngineService preloads the resident model in the background at startup
  (`FORGE_GUI_NO_PRELOAD=1` to disable, `FORGE_GUI_PRELOAD` /
  `FORGE_GUI_PRELOAD_CONFIG` to pick a different checkpoint/config);
  a different-model request mid-load is queued (not dropped), a bare
  re-request for the resident model is a no-op, and unload() uses
  sleep(level=2) — no wasted GPU→CPU copy on discard.
- `forge_ui/` — React 19 + TS + Vite + Tailwind v4 frontend
  (`npm run dev` / `npm run build`; design tokens in `src/index.css`).
  Shared chat/agent primitives live in `src/components/chat/` (Markdown,
  blocks=ThinkCard/ToolActivityCard/SegmentedBody/UserBubble/CopyBtn,
  Composer, ScrollFeed) — reuse them instead of duplicating message cards.
  Backend endpoints the pages rely on: `POST /api/chats/{id}/truncate`
  (regenerate/edit-and-resend), `POST /api/chats/import` (paste a
  transcript → new conversation; `parse_transcript` in
  `forge_gui/api/chat_store.py` handles JSON/ChatML/role-marked/plain
  alternating-paragraph formats), `DELETE /api/agent/runs/{id}`,
  `GET /api/agent/tools` (real harness tool defs for the Agent picker —
  do not hardcode tool lists in the UI).
- `tests/unit/` — Unit tests (CPU-runnable where possible)
- `tests/integration/` — Integration tests (GPU required)
- `docs/` — Documentation + R&D round notes; `docs/FLUX_IMPROVEMENT_ROADMAP.md`
  = FluxLM phased improvement plan + technique audit (R54+ planning);
  `docs/ENGINE_IMPROVEMENT_PLAN.md` = ForgeEngine phased plan (correctness
  bugs → wiring → SSM snapshots → graph decode → continuous batching; the
  "dead-wired" R&D-zoo diagnosis + external precedents)
- `scripts/` — Standalone scripts
- `research/` — legacy root still holding LIVE modules: `paths.py`
  (canonical path constants — `V2_CHECKPOINT`, `FORGE_TOKENIZER_DIR`),
  `tokenizer_cache.py`, `merge_models.py`, `sampling_utils.py`,
  `json_compat.py`, `persistent_executor.py`, `architecture/` (checkpoint
  port scripts), `training_free/`, `cloud/`, `vision/`, and `checkpoints/`
  (weights — ForgeLM_V2.safetensors etc.). New code goes in `forge/`;
  these are stable leftovers, not dead code.

## Current Architecture: ForgeLM V2 Jamba-3B (SOLE BASE)

**Base model**: ForgeLM V2 — lossless port of AI21 Jamba-Reasoning-3B.
- 28-layer hybrid: 26 Mamba-2 SSM + 2 GQA attention layers
- d_model=2560, 20 heads, 1 KV head (MQA), vocab=65536, max_seq_len=262144
- No RoPE (Mamba handles position), untied embeddings, ~3.2B params
- bf16 checkpoint ~6.1 GB; server VRAM budget 8.0 GB

**Base checkpoint**: `research/checkpoints/ForgeLM_V2.safetensors`
**Tokenizer**: `research/checkpoints/forgelm_v2_tokenizer/`
**Default config**: `forgelm_v2` (load_default_model() defaults to this)
**Path constant**: `research.paths.V2_CHECKPOINT`, `research.paths.FORGE_TOKENIZER_DIR`
**Server model ID**: `forgelm-v2-jamba` (DEFAULT_MODELS in forge_server.py)

All prior checkpoints (Jamba_Reasoning_3B source port, ForgeLM_V2_Light*,
evolution R30/R31 artifacts, LFM2.5 GGUFs) were deleted 2026-09 — ForgeLM V2
Jamba is the sole base model.

## Config Presets

Config presets in `forge/config.py`:
- `forgelm_v2` — **ForgeLM V2 Jamba-3B: THE DEFAULT AND SOLE BASE.** 28 layers
  (26 Mamba + 2 GQA), d_model=2560, ~3.2B params.
- `forgelm_v12_jamba` — R37 research preset derived from `forgelm_v2`
  (parent declared; Mamba-3, Kronecker embed, PIT, OutRo, ForgeHybrid — all
  zero/identity-init for lossless warm start).
- `forgelm_v13_gdn2` — R60 preset derived from `forgelm_v12_jamba`;
  adds `use_gdn2` (Gated DeltaNet-2 side-paths alongside each block,
  gate=0 → bit-exact warm start; supersedes KDA — mutually exclusive).
- `forgelm_v14_rsi` — V14 preset derived from `forgelm_v13_gdn2`
  (intentionally non-lossless new topology; docs/V14_RSI_DESIGN.md).
  ~2.465B params (int4 ~1.23GB), ~1.26B active/step: 24 layers =
  16 DoubleGatedConv + 4 Mamba3 + 4 GLA (d_c=256); MoE FFN (8 experts
  top-2 + shared, d_ff=768); GDN2 at n_heads=4/head_dim=128 on all blocks;
  TitanMemory rank=128 on stride-6 subset [5,11,17,23]; MTP-4
  adapter_heads (shared tied vocab head); CopySpan edit head +
  Verifier head (new `forge/model/edit_head.py`, driver
  `forge/decoding/edit_decode.py`). New subset fields `gdn2_layers` /
  `titan_layers` (None=all). Bench: `scripts/bench_v14.py`;
  tests `tests/unit/test_v14_rsi.py`.
- `qwen25_coder_3b_mla_moe` — ported Qwen2.5-Coder-3B→GLA(d_c=512)+MoE8x1376
  dense_bypass (36L d2048, vocab 151936, tied). Port verified argmax-100%.
  Checkpoints: `research/checkpoints/qwen25_coder_3b_mla_moe` (bf16 port),
  `..._mla_moe_fq` (ForgeQuant prequant — loader swaps ForgeQuantLinear
  shells on `*.dense_packed`, `_forgequant_prequant` marker keeps BCG on),
  `..._mla_moe_honed` (QLoRA-NF4 evolve-format hone, 300 steps; teaches
  primed→fenced-block emission — "missing" stage ~95%→~5%; built by
  `scripts/assemble_evolve_ckpt.py` — copies tokenizer + `forge_convert.json`
  from the source port dir onto the trained safetensors). Mem-safe
  paths: `load_quantize='int4'` (59.7 tok/s, no BCG — torchao packed
  tensors can't capture) or `_fq` dir (BCG captures; ~41 tok/s with the
  fused int4 GEMV, ~25 unfused — still behind torchao on speed but
  graph-capturable). `scripts/prequant_forge_quant.py` bakes the _fq dir
  CPU-only; `--rotate` packs the dense basis Hadamard-rotated
  (`use_rotation=True` — QuaRot-style per-group outlier spreading, ~2.3x
  lower fwd error on outlier-heavy weights, neutral on Gaussian;
  `forge_convert.json {"fq_rotation": true}` makes the loader
  materialize rotated shells — INT8 sparse rows stay original-basis so
  `in_f` recovery is unchanged). ForgeQuantLinear also has a fused
  batched Triton path for M<=32 (`_fq_int4_gemm_kernel`, tl.dot ieee,
  BLOCK_M 16/32 — BLOCK_N halved to 64 at BLOCK_M=32 for smem):
  measured 2048² gs=128 — M=8 3.96x, M=32 2.86x, M=1 GEMV 1.70x vs
  dequant+cuBLAS. NOTE sft_train gotchas: `--no-bitnet-everywhere` required
  for ported models (ternary default is lossy); `--tokenize-workers 0`
  on Windows; NF4 merged saves need offline `weight_packed`→`.weight`
  dequant (one-liner in scratchpad 2026-10-27).
  ForgeQ-R residual patch (R&D 2026-11): `prequant_forge_quant.py
  --residual-rank N` fits an fp16 low-rank correction `res_a[O,r] @
  res_b[r,I]` to each packed matrix's quant error (LoftQ-style
  alternation, `--residual-iters`, skipped above
  `--residual-max-elems`) — W ≈ deq(Q) + A@B, applied AFTER the
  quantized output + LoRA in `ForgeQuantLinear.forward` (all branches,
  original basis even under `--rotate`); `merge_lora()` refits it.
  Self-describing: loader reads `res_a.shape[1]`, `forge_convert.json`
  `fq_residual_rank` is informational. It is loss-REDUCED, NOT
  mathematically lossless — r=16 ≈ 4-6% size overhead. Tests:
  `tests/unit/test_distill_quant_patch.py`.
- `forgelm_tiny` — 4-layer tiny conv+attention model for fast tests
  (no checkpoint, config-only).
- `gen_model_tiny` — tiny generator config for evolution experiments.
- `qwen25_05b` — HF-compat preset for Qwen2.5-0.5B checkpoint detection.
- `qwen3_4b` — HF-compat preset for the Qwen3-4B-Instruct-2507 family
  (e.g. `0xA50C1A1/Qwen3-4B-Nymphaea-RP`). First preset to use the
  explicit `ModelConfig.head_dim` field (128; `d_model/n_heads`=80 would
  be wrong). GQA 32Q/8KV, QK-norm, no QKV bias, RoPE 5M, 256K ctx, tied
  embed. Auto-detected by `_detect_config_from_header` on
  (vocab=151936, d_model=2560, n_layers=36).
- `llama31_8b` — HF-compat preset for the Llama-3.1-8B family (e.g.
  Hermes-3-Llama-3.1-8B lorablated/abliterated tunes; ChatML format).
  GQA 32Q/8KV, no attn/MLP bias, SwiGLU 14336, RMSNorm eps 1e-5, RoPE
  base 500K + llama3 frequency scaling (factor 8 over orig 8K ctx,
  implemented in `forge/model/layers.py` + mirrored in
  `loader._reset_non_persistent_buffers`), untied embed/head, vocab
  128256, max_seq_len capped at 32768 for the 12GB card (native 131072).
  EOS set includes 128000/128001/128040 (im_end). Auto-detected by
  `_detect_config_from_header` (embed 128256x4096, q 4096x4096, kv
  1024x4096). The server renders plain ChatML for `llama*` configs via
  `render_messages_for_config` (no Jamba thinking prefix/`<think>`
  block — see `qwen_adapter.py`).
- `llama32_3b` — HF-compat preset for the Llama-3.2-3B family (e.g.
  NousResearch/Hermes-3-Llama-3.2-3B; same ChatML + tool-call format as
  the 8B). GQA 24Q/8KV head_dim=128, SwiGLU 8192, RoPE 500K + llama3
  scaling factor 32 (3.2 scales 32x over the 8K base, vs 3.1's 8x),
  TIED embed/lm_head (unlike 8B/70B), vocab 128256, ctx capped 32768.
  ~3x faster decode than the 8B; first consumer is ForgeChat's
  hermes3_3b catalog entry as the fast card-conversion/chat model.
- `lfm3_1p7b` — ported LFM3-1.7B tri-hybrid (24L: 13 gated short-conv
  K=7 + 5 SWA W=256 θ=10K + 6 full GQA θ=1M partial-RoPE-0.5-zero-tail;
  16Q/4KV hd128, per-head sigmoid out-gate `gate_proj`, QK-norm,
  pre-residual sandwich norms `sandwich_norm_mode="pre_residual"`,
  tied embed, vocab 128K, NO final-norm confusion: HF `embedding_norm`
  IS the post-layer norm → maps to `ln_f`). Checkpoint:
  `research/checkpoints/lfm3_1p7b_cc` (mondk Claude-Code ft — stock
  arch + tool/bench/self-RL post-training on LiquidAI/LFM3-1.7B;
  forge_convert.json names the preset; tokenizer.json in-dir, eos
  `<|im_end|>`=124900). `max_seq_len` capped 16384 (RoPE tables are
  per-layer × ~500MB at 262K → OOM on the 12GB card; evolve loop needs
  ≤~4K). Parity vs bundled HF ref (venv_train, TF5.20): fp32
  max|dlogit|=3.6e-5, fp16 top-k identical — `scripts/test_lfm3_port.py`.
  New config fields: `use_attn_out_gate`, `sliding_window`,
  `rope_layer_params` (per-layer-type θ/partial), `sandwich_norm_mode`.
  New GQA args: `sliding_window`, `use_out_gate`, `rope_partial_factor`.
  LoRA: separate registry root `forge/evolution/loras_lfm3/` (the
  qwen3b `loras/default` adapter would shape-crash on this base);
  targets `q_proj,k_proj,v_proj,out_proj,in_proj,w_gate,w_up,w_down` —
  `out_proj` covers GQA+conv (same path), `gate_proj` skipped by
  min_size. Conv `_pad_mask` reproduces HF apply_mask_to_padding_states
  for left-padded batch gen. Run: `rsi_evolve.py --model
  research/checkpoints/lfm3_1p7b_cc --lora-root forge/evolution/loras_lfm3`.
  (A20: the multi-GB extensionless *merged-model* saves sft_train used
  to drop next to the adapters — `loras*/<tag>` torch.save blobs — now
  live under `research/checkpoints/lora_merged/<registry>/`; the
  registry dirs keep only `*.lora.safetensors` + manifest. The stale
  `proposer_ckpt.safetensors` dup of `proposer_ckpt/model.safetensors`
  was deleted after byte-identical sha256; its `.meta.json`/`.train.pt`
  sidecars moved to `research/checkpoints/`.)
- **LFM3 d-series** (derived from `lfm3_1p7b`, same `lfm3_1p7b_cc` ckpt):
  `lfm3_1p7b_d1` — zero-init grafts, all dormant at init: GDN2 side-paths
  on conv layers 1,5,9,13,17,21 (immediately before each full-attn),
  ValueResidual, LearnedSink, AttnRes, CopySpan/edit head, Verifier head.
  Lineage gate `scripts/test_lfm3_d1_parity.py`: max|dlogit|=0.0 on the
  real ckpt (CPU fp32), 1.7288B→1.8029B (+73.9M dormant).
  d2 strict-lossless restructures are config flags, not a preset yet:
  `share_rope` (dedup identical RoPE tables; SWA vs full-attn tables stay
  distinct), `conv_kernel_size=11` + `_warm_start_conv` zero-pad port
  (K=7→11, prefill+decode bit-exact), `zero_centered_norm` (w→w−1 reparam,
  `_warm_start_zero_centered_norm` port; qk-norm identity fast-path
  re-derived post-load), `use_fused_gemm` now WIRED (was dead code — fused
  QKV+gate / gate-up GEMMs consumed in forward; ctor RNG forked so init
  streams match unfused builds; loader `_refresh_fused_gemm` repacks
  checkpoint rows post-load), `swa_ring_kv` (SWA-layer KV buffer capped at
  the window, slot=pos%W — decode unbounded on those layers; pos_t path
  stays graph-capturable via pos_t%W; eager reads re-materialize the held
  window in position order; int8-cache path excluded). NGramMemory +
  `use_suffix_spec` already existed (gate=0 inert, `.draft()` ready).
  Function-identical modulo SDPA kernel-tiling (~1e-7), verified by
  argmax-trajectory tests (`tests/unit/test_swa_ring_kv.py`,
  `test_share_rope.py`, `test_fused_qkv_gate.py`, `test_conv_widen`,
  `test_zero_centered_norm`).

- `spark_x26_1p7b` — ported Spark-X2.6-1.7B (XHToken spark2_5, pure-
  attention hybrid): 28L (SWA512×3 + full, ×7), GQA 8Q/2KV head_dim 256,
  fused HF q_k_v_proj split on remap, per-head sigmoid out-gate, GeGLU,
  slice-mode partial RoPE 0.25/θ5M on full layers (SWA θ10K full).
  Vocab 131072, bos=0 eos=1 pad=2, tied embeds.
  `research/checkpoints/spark_x26_1p7b` (HF dir + tokenizer/chat template)
  and `spark_x26_1p7b_fq` (1.42GB packed — loads 1.87GB w/ swa_ring_kv+fused GEMM+share_rope defaults, 62-76 tok/s w/ BCG).
  Parity `scripts/test_spark_port.py`: max|dlogit|=2.1e-5. Port required
  THREE new shared-code paths: slice-mode partial RoPE
  (`partial_rotary_mode`), GeGLU (`ffn_type="geglu"` → SwiGLUFFN
  act_mode), and the q_k_v_proj/g_proj/embedding keymap in
  `_remap_hf_keys`. NOTE: Spark tokenizer needs the plain-HF fallback
  (gigatoken can't parse its CJK pre-tokenizer).

All legacy presets (`forgelm_v2_light`, `forgelm_v2_pro`, `forgelm_v12`,
`forgelm_v10_1.2b`, `forgelm_v11_3b_vl`, `lfm25_tiny`, `lfm25_1.2b`, V3–V9)
have been DELETED.

### Evolution-Discovered Promotions (2026-08-24, from forge_evolve.db)
Promoted after validation against evolution data (39,631 discoveries scanned):
- **MTP**: n_heads=4, loss_weight=0.495 (was 2/0.3, score 27.69) ✓ validated
- **SpecDecode (PEAGLE)**: n_draft=7 (was 4, score 57.05) ✓ validated
- **BatchQueue**: batch_window=52ms, max_batch=15 (was 50ms/8, score 44.25) ✓ validated
- **SFT training**: grad_accum=5, grad_compression=int4 (score 30.00/11.10) ✓ validated
- **AirMoEKey**: cache_strategy=lfu, disk_cache_size=4096 (score 8.83, 0% miss rate) ✓ validated
- **ForgeEngine**: CREATIVE_SAMPLING preset (temp=1.98, top_p=0.989, top_k=69, score 10.45) ✓ validated
- Already applied prior: W8A8 fp8+alpha=0.999, PagedEvictKV page=64/LRU, ModConfig aux_loss=1e-8, CheckpointRecompute selective/block=512, FocalLoss gamma=4.93

### Reverted promotions (scoring artifacts — synthetic metrics didn't match real behavior)
- **MoE routing** (4/3/switch/6e-5): REVERTED to 8/2/aux_free/0.01. Evolution found top_k=1 scored higher (24.95 vs 24.87) but top_k=1 is a trivial solution (no ensemble). Scoring fixed: diversity penalty for top_k=1.
- **RoPE theta=10M**: REVERTED to 1M. Synthetic metric rewarded angle diversity, not attention quality. Scoring fixed: checkpoint compat penalty + frozen-dimension detection at long range.
- **Scheduler warmup=0**: REVERTED to warmup=500/20. Synthetic AUC rewarded no warmup, ignored training stability. Scoring fixed: stability penalty for zero warmup.
- **Label smoothing 0.29**: REVERTED to 0.1. Synthetic metric rewarded high smoothing for grad magnitude. Scoring fixed: smoothing penalty above 0.2 + focus ratio metric.

> Detailed evolution domain scoring fixes, DB rescores, focus profiles, and R&D round
> documentation have been moved to `docs/CHANGELOG.md`.

## EXP-1 Experiment Surface (broad LLM arch experiments on Qwen2.5-0.5B)

Reusable experiment package + drivers — `forge/exp/` (experiment-only,
NOT model lineage): `harness.py` (load_engine, wikitext_docs, log_result,
bench_decode — raw fixed-step decode bench that bypasses early-EOS
undercounting), `ncm.py` (CfC neural-circuit side-path), `loopdepth.py`
(recurrent-depth looping, zero-gated extra iterations), `diffdraft.py`
(parallel draft heads: DDIM denoiser + MedusaDraftHead, greedy
spec_verify with KV truncate+correction fixup), `dynscale.py`
(HeadGate/EarlyExitController/DiagTempScaler), `abshead.py`
(LatentThoughtHead h_t→h_{t+4}), `ndp.py` (exact online-softmax
attention split across host/GPU KV tiers), `prevtok.py`.
Drivers: `scripts/exp_{baseline_qwen,matrix,loop,ncm,trainless,
diffdraft,dynscale,abshead,prevtok,ndp,cxl}.py`. Results:
`.devin/exp_results.jsonl` (append-only log via `log_result`),
notes in `.devin/scratchpad.md` "EXP-1" section.

Load pattern (all drivers): `ForgeEngine.from_checkpoint(<HF snapshot
dir>, config_name="qwen25_05b", device="cuda", quantize=None)`.
Qwen2.5-0.5B corrected baseline: Forge NLL 2.996 (HF ref 3.042),
~70-81 tok/s raw decode bf16.

Fixed bugs (keep): `qwen25_05b` now `tie_word_embeddings=True` (checkpoint
has no lm_head — untied gave NLL=11.94); `research/tokenizer_cache.
_resolve_path` no longer redirects HF repo-ids (`org/name`) to the 65k
ForgeLM tokenizer (was NLL=9.99).

Conventions discovered for exp modules:
- Side-paths: zero-init gate + `blk._X_gate_zero` cached check —
  lossless attach verified delta=0.0 on every module.
- bf16 model + fp32 aux modules: cast at the boundary (fp32 aux internals,
  or `.to(p.dtype)`); `F.mse_loss`/`Linear` reject mixed dtypes in fwd
  AND backward. `CastTitan` shim in exp_trainless.py for fp32-only
  TitanMemory.
- Tokenizer emits int32 — CE targets need `.long()`.
- Eval forwards don't mutate TitanMemory/NGramMemory state → skip
  `_snapshot_live` deep-copies (OOM at 151936 vocab: 594KB dense rows
  per suffix; see `eval_live` in exp_trainless.py).

Headline findings: bf16+standard KV is the fastest config at 0.5B on
RTX 5070 (all quant modes slower — kernel-bound); train-less
generalization real but marginal held-out (+0.02 NLL) and strong
verbatim (+0.4-0.5); ~15% attention heads removable at small NLL cost
but Python-side gating costs more than it saves; self-speculative
drafting needs a much better drafter than 300 steps on 6 docs
(medusa 15-28% accept > diffusion 3-8%, both <1 tok/fwd); prevtok
self-correction rerank gives joint dNLL -3.26 at ~1.0 forwards/tok;
CXL tiered KV keeps GPU footprint flat (12.6MB at any ctx, bit-exact)
at -52% throughput; NDP partials beat page migration only at moderate
far-context sizes (CPU attention < PCIe transfer below ~8K pages).

## EXP-2 FluxLM "rev" channel (bidirectional memory)

Native reverse-consistency channel — the EXP-1 prevtok self-correction
finding as pure memory, no parameters:
- `FluxConfig.rev_channel` (default False, lossless off): adds "rev" to
  `self.channels`; `_rev1[last] -> {succ: surprise-scaled cnt}` written
  free at `_learn_token` (one dict update, journaled "R", snapshot key
  `rev1`, `_pmask` device abstain).
- Channel semantics mirror order channels: vote `v_t = uni_t·p(last|t)`
  (count-scale; `p(last|t)` = smoothed pair count / successor frequency
  — the Bayes flip of the forward bigram), `p_ch = (v_x + a·p_uni) /
  (vt + a)`.  First formulation `uni_p·(1-Z+p)` was capped at
  ~unigram-scale rewards → Hedge floored it at w=0.001; the
  count-scale vote lets it earn real weight.
- `forge/model/flux/revmem.py`: standalone sidecar (ReverseMem +
  `generate_rerank`) for model-agnostic use / decode-only rerank.
- Driver `scripts/exp_fluxrev.py` (Markov-chain + docs/research bytes
  corpora): tests `tests/unit/test_flux_rev.py`.

Measured (two corpora, `scripts/exp_fluxrev.py`):
- Markov V=2048: rev alone earns w=0.971, NLL 3.10→2.70 — but plain
  `o1` does better (2.50): the generative direction is strictly more
  predictive; rev+o1 → Hedge de-weights rev to 0.022 (redundant).  Gen
  prev_nll: rev 1.072 < o1/base 1.117 — rev's one edge.
- Text bytes V=256 (docs/research ~450KB): base 2.492 NLL; rev 2.490
  (mild +, weight floored); o1 2.562 (hurts — bigram noise vs deep
  channels); gen prev_nll: o1 1.335 vs base/rev 3.049 — the FORWARD
  bigram, not the reverse, is what makes generated text
  backward-consistent on real prose.
- Verdict (honest, negative-leaning): as implemented, rev is a weaker
  o1 — the Bayes flip loses predictive information.  Sidecar rerank
  also hurts on text (top1 0.456→0.429 @a=2).  Kept default-off +
  tested as the backward-channel substrate; a useful variant likely
  needs longer predecessor context (rev over prev-k-grams) or
  pair-keyed bidirectional cells, not a bigger weight.
- Corollary worth noting: consistency pressure trades against
  predictive NLL — o1 maximizes self-consistency while minimizing
  held-out accuracy.  Consistency is a generation regularizer, not a
  free accuracy win.
- Margin-gated rerank probe (.tmp/rev_margin_probe.py): unconditional
  rev-rerank at a=1 reproduces o1's consistency win EXACTLY (gen
  prev_nll 3.049→1.335, 42/64 corrections) — but margin gating kills
  it (2.89+): FluxLM's degeneracy lives at CONFIDENT steps (83%
  uncontested), not where the mixture is uncertain.  Deeper
  implication: Hedge weights optimized for next-token rewards are not
  optimal generation weights — a channel mediocre at prediction can be
  exactly what decode needs.
- `generate(w_boost={ch: w})` (predictmixin): decode-scoped channel-
  weight override, restored in finally — the two-regime mechanism as
  API.  Verified: rev boost 0.5 on docs bytes cuts gen prev_nll
  3.05→2.68 (native, saturates w≥0.5); sidecar rerank α=1 reaches
  1.335 (pure rescore, no mixture dilution).
- o1-boost probe (.tmp/o1_boost_probe.py): an o1 model generates
  1.335-consistent text even with o1 floored at w=0.001 — no boost
  needed.  Consistency pressure needs almost no weight: it works by
  nudging near-tie steps, not out-voting the mixture.
- Precision check (.tmp/gen_inspect.py): the "inconsistent" base
  generation is actually FLUENT markdown prose with zero novel
  byte-pairs — prev_nll 3.049 doesn't measure degeneration, it
  measures the asymmetry of language: p(prev|succ) is systematically
  low for common successors (every predecessor is atypical when the
  successor has thousands).  The "consistency win" of rev/o1-rerank
  biases decode toward cliché transitions (strongly-associated
  pairs), not toward objectively better text.  FINAL verdict: the
  reverse channel fights the generative direction of language — the
  trained prevtok head on the transformer beat it because learned
  predecessor entropy, not pair existence, is the right consistency
  signal.  rev stays as substrate; not a recommended default.

## EXP-3 FluxLM "gv" channel (stateless live generalizer)

Read-time generalization over the LIVE order tables — generalization
as a query transformation, not a stored augmentation.  Motivation:
class tables (`g<k>`) keep independent copies of order evidence, so
deleted/decayed order cells persist through them; gv owns NO state,
so removed weight information can never persist through it.

- `FluxConfig.gen_variants` (default False, lossless off): adds "gv".
  `gen_fix_tail` rightmost suffix positions stay verbatim (0 = all
  positions vary), `gen_min_tot` exact-cell mass that silences gv
  (thin-gate), `gen_var_w` per-count vote weight, `gen_max_probes`
  host budget, `gen_orders` restricts the order set, `gen_cls_cap`
  bounds device members/position (first-N by token id).
- Mechanism (`_gen_votes`, predictmixin): for each order k whose
  exact cell is missing or thin (< gen_min_tot), replace one suffix
  token x_p with each same-class v and probe
  `key + (v-x_p)*P^(end-p)` — rolling-hash linearity makes each
  variant key O(1), no re-fold.  Class membership `_cls_members` is
  static vocab metadata, not learned state.  Votes merge
  `gen_var_w*cnt` per hit cell; channel prob = count-scale
  `(v_x + a·p_uni)/(vt + a)` (same semantics as order channels, so
  Hedge can weigh generalization against verbatim memory).
- `_pmask` abstains gv for device rewards (neutral reward, weight-
  preserving).  cuda_primary prediction path: `_gv_lift_dev` runs
  post-graph inside `_apply_sparse_lift_dev` — probes `_gt` with one
  batched `probe_fixed` + one cell gather + 2 syncs per predict.
- Contract tests `tests/unit/test_flux_gv.py` (10): variant firing,
  thin-gate sleep, **delete → votes vanish**, journaled revert →
  votes shrink by exactly the reverted evidence, clone determinism,
  off==lossless, CUDA lift + GPU-row clear → lift dies.
- Driver `scripts/exp_fluxgv.py` (synth class-structured + docs
  bytes, cuda_primary): synth NLL 1.5777→1.5762, top1 +0.0003 —
  small positive where true class structure exists.  Text bytes:
  NLL 2.5422→2.5575 (+0.015, small NEGATIVE) — byte classes are too
  coarse for natural text (one-edit same-class variants mostly fire
  noise cells); eval ~20x slower (thin contexts common → probes
  fire often, 2 syncs/predict).
- Verdict (honest, negative-leaning like rev): the CONTRACT works
  (zero-state, deletion-propagating generalization — proven by
  tests) but byte-class 1-edits are the wrong variant space for
  natural text.  Two fixes would change the verdict: (a) semantic
  variant classes (embedding neighborhoods — but that reintroduces
  stored state), or (b) keep gv CPU-side where Hedge floors its
  weight on evidence it can't use — on cuda_primary gv abstains
  from rewards so its weight stays at init and can't self-limit.
  Kept default-off + tested as the generalization substrate; the
  invariant to preserve in any successor: generalization reads live
  weights, never caches them.

## EXP-4 FluxLM on Qwen-BPE tokens (same-unit LLM comparison)

Byte-level NLL can't be compared to a BPE LM.  Driver
`scripts/exp_fluxtok.py` runs FluxLM (V=151936, cuda_primary) on
Qwen2.5's OWN token stream of the docs corpus (190K train / 2.3K
held tokens), so NLL/token is directly comparable.  No gradient
training anywhere — ingest is the whole "training".

Measured (held-out docs text, Qwen2.5-0.5B NLL = **4.298**):

| variant | top1 | NLL |
|---|---|---|
| Qwen2.5-0.5B (pretrained) | ~0.45 | **4.298** |
| bpe orders(1..32) only | 0.176 | **6.692** |
| +knn | 0.1725 | 6.815 |
| deep orders(3..64)+knn | 0.1205 | 7.616 |
| +sparse taps | 0.176 | 6.811 |
| +BPE surface classes g2-4 | 0.1699 | 7.055 |
| +surface-cls gv | 0.1707 | 7.056 |
| +emb-cluster classes g2-3 | 0.1773 | 6.830 |
| +emb-cluster gv | 0.1742 | 6.840 |

- New infra: `FluxConfig.cls_table`/`cls_members` overrides +
  `helpers.bpe_class_table` (surface classes) and
  `helpers.emb_class_table` (k-means partition of a pretrained
  embedding table -> semantic classes; members centroid-sorted so
  `gen_cls_cap` probes prototypical variants).  Both are static
  metadata — no fitting to the corpus.
- Honest verdict: at 190K train tokens EVERY auxiliary channel
  dilutes the mixture — plain order mixture wins.  The binding
  constraint vs Qwen is *coverage*, not channels: an associative
  memory generalizes only where it has cells.  Next lever = corpus
  scale (`scripts/exp_fluxscale.py`) — 12MB local text pool
  (~3.5M BPE tokens) vs docs-only.

## EXP-5 FluxLM memory rework + cold-tier weight banks

Bulk ingest OOMed past ~1M BPE tokens.  Root cause was transient
geometry, not resident state: the probe/reward/write pass ran
unchunked over all n — `[O,n]` and `[O,n,M]` tensors (~20GB at
n=3.5M, O=12, M=16) plus a 5.4GB `free_cols` argsort in
`_GPUCellTables.add_multi`.  Fixes in `learnmixin._ingest_bulk`:

- probe/reward/write pass now slices by `gpu_bulk_chunk` (8192);
  only light `[O,n]` buffers needed by downstream reward/hedge/
  regime logic stay whole-block.
- journal-bound matrices (`kk_all`, `write_o`, `surprise`,
  eviction/col snapshots) are pinned HOST buffers filled by async
  D2H copies; one stream sync before host reads.
- csem projection `_A[ids] @ _Pcsem` chunked — `[n,192]` transient
  (3.6GB at n=3.5M) dies; only `[n,64]` result persists.
- `blk["deltas"]` stays fp32 host (exact-undo requirement).
- Smoke: 60K-token cuda_primary ingest peak 0.62GB.

`grow()` no longer hard-raises on structural overflow (>2 keys per
slot pair): the weakest contestant retargets to its alternate slot,
drops only when both anchors are taken, and reports `dropped` —
auto-grow mid-ingest can't crash the model.

FluxBank (`forge/model/flux/bank.py`, `FluxConfig.bank_*`) — cold-tier
weight bank, the "MoE for weight data" tier: spilled order rows
(toks/cnt/tot/occ) live host-side, still vote through the SAME
order-channel prob (`_cold_cell` + `_union_cell` merges hot+cold
counts — disjoint write sets sum exactly).  Re-spill dedups/merges
by key.  `_spill_order` fires when a table crosses
`bank_spill_hi`·slots instead of decaying; `regulate()` runs at
ingest seal (`bank_regulate`) — hot tables shed cold rows, banks
purge coldest past `bank_rows_cap`.  Device path: `_bank_lift_dev`
probes banks for hot-missed contexts post-graph.  Persisted in
`.flux` snapshots and standalone `.fluxbank` npz files.  Undo is
approximate-but-safe: reverts probe hot keys, spilled rows skip
cleanly.  `len(bank)` = live rows (`_n`), not capacity.

Tests: `tests/unit/test_flux_bank.py` (7) — spill keeps votes,
hot/cold union merge, regulate, snapshot round-trip, live-length.
Suite: 195 flux tests green.

## EXP-6 FluxLM domain lanes — mixed-corpus isolation

Adding a 3.1M-token code-heavy pool to a prose-eval model made it
WORSE (nll 7.13 vs docs-only 6.82).  Attribution (evaluating the
pool model under docs-adapted/uniform `_w`): hedge weights are NOT
the cause — learned ≈ uniform ≈ 7.13, docs-w = 7.38.  The damage
is in shared low-order cells ("the"/"ing" collect code+prose
followers) and global unigram dilution.

Domain-mixed keys (`FluxConfig.domains`, `dom_order_max≤2`,
`dommix_w`): `ingest(domain="code")` XORs `dom_id` into suffix keys
for low orders only — code-"the" and prose-"the" are distinct
cells, zero extra memory.  All reads route through `_keys_dev` /
`_keys_at` (single insertion points → fused graph, probes, bank
spill/merge see identical masked keys).  Router: AdaHedge over
lanes refreshed every `dom_route_every` positions from reward
(p(actual) under each lane's cell view), not coverage — v1 scored
raw hits and the 16×-bigger code lane won everything.
Lane'd priors: `_uniL[d]`/`_totalL[d]` per-lane unigram mass;
`_uni_p` and device `_unig` mirror swap to the active lane on
`_set_domain` (graph-safe input-tensor pattern, same as `_dommix_g`).
`blk["dom"]` journals the lane; revert decrements lane totals.

Measured (same held-out prose slice): pool_dom **nll 6.7878,
top1 0.148, 7.16GB VRAM** — now BEATS docs-only 6.82; the 16×
mixed corpus costs zero prose regression.  Progression:
undifferentiated 7.13 → coverage-routed 6.99 → reward-routed +
lane priors 6.79.  CUDA smoke: lanes isolated on GPU, lane-1 cells
masked, lane-specific follower wins the logit.  Tests:
`test_flux_domains.py` (7) — masked-key reads, lane isolation,
spill/snapshot round-trip, lane'd prior, router reward.

## EXP-7 FluxLM decode speed — the wall was Python, not GPU

Profiled the hot decode loop (V=151936, cuda_primary, 100K tokens):
graph replay costs 0.05ms; everything else was host overhead.

- `generate(learn=True)` ran `_predict` + `_learn_token` per token —
  two device passes while `_fused_step` already returns post-learn
  next-position logits.  Now pipelined: each emitted token is learned
  by the NEXT iteration's fused call (one pass/token, trailing learn
  flushed, `_dom_tick` preserved for the deferred learn).
- `_sparse_lift_vec_dev` built two `[V]` fp64 arrays + np.nonzero
  scans per predict (~1ms) while real support is ~hundreds of
  followers — now (idx, contrib) pair lists summed by the device
  `index_add_`; flv clip moved device-side post-accumulate.
- `_knn_write` spent ~5 scalar D2H reads/call (dedup check + journal
  prev-state) — batched into two stacked `.tolist()` transfers.

Measured (warm, 300-token greedy): observe `predict+push`
198→**990 tok/s**; `generate(learn=False)` ~200/s→~330/s;
`generate(learn=True)` 29.5→~70-240/s (pipelined, key-warmed).
Cold-start caveat: first ~6 steps per (lite, bucket) graph key run
eager warmups + capture — measure decode warm only.
Next lever (design in scratchpad): self-speculative batched decode —
`_fused_seq` returns [T,V] logits for a drafted row, verify-and-
revert; needs a write-free `_observe_seq` variant for the
learn=False path to reach >1000/s.

## EXP-8 FluxLM "tch" — static Qwen teacher as the generalization floor

The hybrid the spec asked for: Qwen2.5-0.5B fp16 as a **static,
read-only Hedge channel** — no training, nothing journaled (its
state derives from the token stream, which IS journaled).  Memory
channels supply recall boosts above the teacher's dense floor.
`forge/model/flux/teacher.py` `TeacherChannel`; config `teacher_*`.
Lazy (`_tch()` on first query; excluded from snapshots by
construction — snapshot is a hand-built dict).

- Predict: `_tch_lift` blends the post-lift row toward the teacher:
  `row <- (1-w_t)*row + w_t*log p_tch` (convex, self-bounding — at
  w_t→1 the mixture IS the teacher).  Device path only.
- Rewards: `bulk_logprobs` teacher-forces each ingest block →
  `real_r`; `tch` stays in `_pmask` for the fused step (no HF
  forward in-graph) but channels with a `real_r` entry are excluded
  from the closed-form abstention fill (`pm_fill`).
- Stream tracking: incremental KV via `sync_to`; `DynamicCache`
  (newer transformers requires it — a raw tuple dies).  Truncation
  is a HAND tail-slice — `DynamicCache.crop()` keeps the FIRST
  tokens, not the tail (~3.4 nats lost to that alone), and the
  cache must keep the first `teacher_sink=8` positions (attention
  sinks) plus pass ABSOLUTE `position_ids` since retained keys hold
  the RoPE rotations they were written at (StreamingLLM semantics).

Measured (docs190k, Qwen-BPE held slice):
`flux-only 6.69 | teacher-only ~4.33 | HYBRID 4.5543 NLL / 0.2852
top1 | Qwen-0.5B reference 4.298` — the floor is real; recall rides
on top via confident-memory arbitration.  Blend form:
`row += w_t*(1-gate)*(lp - row)`; gate = `_dconf_g` (any order cell
clearing the adaptive-read evidence bar = verbatim context → the
teacher yields; gated positions score NLL 2.10 vs 5.70 ungated).
Verbatim generation recalls trained facts (" 4242 and never 777")
that teacher-only and memory-only both miss.  Bug chain
to that result (each measured): log-vs-prob rewards, `_pmask`
abstention overwriting real_r, prefix-vs-tail crop, RoPE position
corruption past crop, missing sinks, len*w clip flattening teacher
ordering (NLL 62.9), lp-uni reference leaking mass (mixture 6.47),
free additive lifts on ungated text net-negative (+1.3).
`teacher_decode=False` skips the HF forward per generated token
(~20 tok/s cap) — memory-only decode stays ~200-990.
Tests: `test_flux_teacher.py` (8: lift math, abstention, CPU abstain,
GPU bulk rewards, tail-crop + sink pinning, snapshot exclusion,
persistent floor).
Peak VRAM at 190K tokens: 12.4GB — teacher ~1.4GB incl. 4K KV;
memory work continues (the 2-6GB gold stat still open).

15M-token collapse + floor (commit a12be79): at 13.77M tokens
AdaHedge drove `w_tch→0.001` / `o2→0.981` — trained-stream rewards
always favour memorized recall, so an unfloored teacher starves and
the mixture degenerates to ~bigram (wt 7.42 / doc 8.53 ≈ flux-only
7.23/8.37).  `teacher_w0` is now a **persistent share floor** —
`_teacher_floor_` after every hedge update (device/host/bulk +
cond_hedge rows) sets tch=w0 and rescales the rest to 1−w0; above
floor untouched.  Reweighting the collapsed snapshot to floor 0.6
(cells identical) recovers **wt 4.9277 / doc 5.7063, top1
0.2453/0.2301** — wt-held approaches the Qwen 4.30 floor.  NOTE:
rescaling must hit `_wg`/`_wgR` (device), not just the `_w` host
mirror, or evals silently read collapsed weights.  NOTE with
`cond_hedge` the lift reads `_wR[warm]` — rescale ALL rows.

Decode (commit 47a70e1): three eval-invisible bugs fixed — pipeline
now applies post-graph lifts to `_fused_step` rows; `teacher_hard_arb`
makes ungated positions adopt `lp` outright (trained cells lift
+20..40 nats — no blend weight <0.95 reaches teacher argmax) while
`_dconf_g` still hands verbatim contexts to memory; `generate()`
rebuilds the teacher KV as prompt-only (the window otherwise served
the ingest tail's register).  Measured on wt15m.flux: `Paris, France`
/ `capital of France is Paris` / `12-5+3 = 10 apples.` / real
fibonacci structure — base-Qwen-0.5B serving quality.  Serve with
`learn=False` (learn=True self-poisons: emitted tokens write cells,
the gate then fires on the model's own output).

## Build & Test Commands

```powershell
# Run tests
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -m pytest tests/ --tb=short -q

# Verify model loads
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -c "from forge.model_loader import ConfigurableResearchLLM; print('OK')"

# Engine load-phase benchmark (weight I/O + activation)
D:\windsurf\ForgeAI\venv\Scripts\python.exe D:\windsurf\ForgeAI\scripts\bench_load_time.py
```

## Engine Invariants (from the ENGINE_CRITIQUE_DB fix pass)

- **Report effective state, not requested state.** After any
  fallback/degradation (quantization, KV cache, decoding strategy,
  low-VRAM mode), `self.quantize`/`self.kv_bits`/diagnostics MUST reflect
  what is actually applied. Never report a mode that is not in effect.
- **No asyncio primitives shared across event loops.** `asyncio.Lock`,
  `asyncio.Event`, `asyncio.Queue` etc. bind waiters to the loop that
  created them — `TestClient`, multi-portal hosts, and any
  `run_coroutine_threadsafe` producer can strand waiters forever
  (`call_soon` does not wake a foreign loop). Cross-thread/cross-loop
  serialization in `forge_server.py` uses `threading.Lock` + the
  `_task_lock_pool` executor + `_task_lock_waiters` accounting. Do not
  reintroduce `asyncio.Lock` for per-task locking.
- **Unknown strategy names must fail loudly.** `build_kv_cache` /
  `build_decoding` raise on unknown names (EC-019) — never silently
  substitute `standard`.
- **Speculative KV rollback must snapshot + repair, not just truncate.**
  All spec strategies roll back via the shared `_spec_snapshot` /
  `_spec_truncate` / `_spec_repair` helpers in `decoding.py`; verify
  failures replay the committed recurrent context against the
  pre-verify cache. New spec strategies must reuse those helpers.
- **Forgetting is expensive.** If you strip a `use_*` flag or an
  innovation slot from `ActivationConfig`/feature_registry, the engine
  must either wire it to a real path or refuse it — half-dead flags are
  how the critique got to 220 entries.

## Competitive Audit 2026-10-31 — Implementation Status

`docs/COMPETITIVE_AUDIT_2026-10-31.md` ranked items, landed this round:

- **A1 external evals** (`forge/evaluation/external_eval.py`,
  `scripts/download_eval_data.py`, `scripts/run_external_eval.py`):
  vendored loglikelihood-MC + gsm8k evaluator over local JSONL
  (`data/eval/`), no lm_eval dep. Baseline report →
  `results/external_eval_v2.json`; `--compare` gates regressions.
- **A2 dormant-zoo honesty** (`forge/engine/kv_backend.py`): only
  beacon/anchor/metakv/flux are decode-live (`compact_past_kv` seam);
  every other KV strategy is a bookkeeping object whose `info()`
  reports `decode_live: False`. `CudaGraphRunner` dead path removed;
  `use_spec_attn` flag deleted from all six plumbing sites.
- **A4 fla kernels** (`forge/keys/attention/fla_delta.py`): KDA +
  GDN2 use fla 0.4.2 chunked delta kernels on CUDA with eager
  fallback (pin fla==0.4.2 — 0.6.0 needs triton `gluon`, absent in
  triton-windows 3.4).
- **A9 observability** (`forge/training/training_utils.py`):
  `write_status_json` fans scalars out to wandb (`FORGE_WANDB=1`) and
  tensorboard (`FORGE_TB_DIR=`) — env-gated, best-effort, covers every
  runner with zero call-site changes.
- **A14 capability trajectory** (`forge/self_play/infinite_loop.py`):
  frozen 8-item `_CAPABILITY_PROBE` scored on each epoch's candidate;
  `capability_score` per-epoch + `capability_trajectory` last-25 land in
  status.json (`--no-capability-probe` to disable).
- **A3 hygiene**: `scripts/check_assets.py` (tokenizer/checkpoint
  gate), `tests/unit/test_research_import_ratchet.py` (pins forge/
  `research.*` imports at 61 — shrink-only), CI ruff ratchet
  (tests/ full check; forge/+scripts/ F821,F823).
- **A18 evolution ACL**: `sandbox.json` `forge/evolution` → read_only
  for model-facing tools (self-improvement stack was the one writable
  forge dir); `PROTECTED_ENGINE_FILES` += code_evolve/eval_pool/
  mutator/database/best_configs. Verified: tool writes to
  `forge/evolution/*` denied, `forge/results` still writable.
- **A19 honest auto-activate**: `kv_cache="rotorquant"` is now
  assigned ONLY when `quarot` keystack exists (it wires QuaRot-KV);
  everything else gets `"standard"` — diagnostics no longer report a
  4-bit KV cache that does not exist. mamba_hybrid profile labels
  `kv_cache="standard"` too.
- **6b serving gaps (plan approved)**: `response_format`→xgrammar
  `json_schema` on chat completions (non-stream), `/info` capabilities,
  `/health/ready` 503-while-loading, Prometheus text `/metrics`
  (`?format=prometheus` or Accept: text/plain; JSON stays default).

## Live-Learning Stack Status (2026-11 — L1/L2 validated on real backbone)

Measured on `research/checkpoints/lfm3_1p7b_cc_fq` (int4, ~1.4B):

- **L1 = exact suffix memory only.** `SuffixMemory`/`NGramMemory` rows
  are EMA successor distributions (`row <- (1-a)*row + a*onehot`,
  `row_mass` fixed so lift strength is write-count-independent);
  `write_decay=1.0` = last-write-wins. Lift is a log-likelihood ratio
  `w*log(p_mem/p_ref)` vs the unigram+uniform prior — unseen suffixes
  give exactly 0. Order weights discount by RANK among configured
  orders, not order value (the old `w**(max_order-o)` made order-4
  ~5e-5 under orders (4,8,16,32)). Ingest rule: feed at sentence/doc
  granularity — BPE suffix keys only match fresh-tokenized queries
  when boundary tokens were ingested with matching left context.
  Probe `scripts/exp_l1_real.py`: recall CE 5.10->2.45, hit-rate
  0.25->0.50, immune to 3.2k distractor tokens, lift ~0.12ms (~0.3%
  decode), +0 VRAM. Fuzzy Titan/GDN2 memory failed the fact-recall
  gate (gain ~2.0n at 5/50/200 reps vs ngram ~4.3; GDN2-only 0.00) —
  kept off the fact path.
- **L2 = sleep-cycle LoRA consolidation.** `scripts/exp_l2_consolidate.py`:
  doc ingest -> SuffixMemory traces -> reconstructed rows ->
  LoRA(r8) fit on int4 backbone -> retention eval. Operating point:
  lr 3e-3 forgets catastrophically (+3.86 ctrl drift); **lr 5e-4 +
  replay buffer -> fact CE 7.22->5.42, drift +0.31**. Rollback path
  required by design (Sleeping-LLM failure mode).
- **Never-loops**: `SamplingParams.loop_detect` (default on) — verbatim
  cycle detector in `_decode_tokens`, checks every 16 gen tokens past
  64; stops when tail is >=3 copies covering >=48 tokens (period<=128,
  copies = max(3, ceil(48/p))). Detection-as-stop is the only mechanism
  that ENDS an attractor loop; DRY/penalties only deform it.
- **Teacher-logprob distill data path validated**:
  `scripts/score_teacher_logprobs.py` (char-offset teacher->student
  token map) scored 150 rows, 0 unmapped, via Qwen2.5-Coder-3B int4
  teacher -> `data/sft/evolve_sft_v3.scored.jsonl`; `sft_train.py
  --distill*` consumes `teacher_logp`.
- **ForgeQ-R residual verdict**: r16 patch on groupwise-int4 error cut
  rel Frobenius 0.1164->0.1149 (~1%) — int4 error is near-white, skip
  bigger ranks; FLINT-style self-distill recovery is the repair path.
- `forge/distillation/agentic_distill.py` reward path restored
  (`_local_compute_reward`; `compute_reward` was deleted in AZR merge).
- **sft_train correctness fixes (distill CUDA assert, resolved)**:
  `_tokenize_cached` now namespaces the disk/LRU token cache by
  tokenizer identity (was shared `"default"` — cross-vocab poisoning
  caused a device-side embed gather assert); tokenizer selection is
  checkpoint-aware when `--tokenizer` absent (default canonical is
  65536-vocab — wrong for foreign checkpoints); `use_chunked_ce`/
  `ce_chunk_size`/`use_mtp` are synced onto `model.config` post-build
  (ForgeEngine builds its own config — trainer-cfg mutations were lost);
  uniform-1.0 `sample_weights` drop to None so the chunked-CE /
  `teacher_logp` path actually engages (was dead code → `--distill`
  silently ran plain weighted CE + full [B,T,V] logits). Training-gated
  embed bounds check in `llm.py` turns this failure class into a
  readable error. Regression tests:
  `tests/unit/test_sft_tokenizer_namespace.py`.
- **Checkpoint tokenizer self-description**: `save_training_checkpoint`
  meta now records `"tokenizer_path"` (the tokenizer source sft_train
  resolved); `from_checkpoint` reads it via `_tokenizer_from_meta` for
  file checkpoints (bare `.safetensors` has no sibling tokenizer.json —
  it silently got the canonical 65536-vocab tokenizer and generated
  whitespace on the 128000-vocab distill60 save until fixed; weights
  were always fine — discriminate weights-vs-wrapper with a raw-logits
  probe). `_tokenizer_for_vocab` still covers no-meta saves.
- **Pipelined loader 0-dim fix**: scalar tensors (BitNet `qscale`)
  crashed `load_safetensors_pipelined` (`view(uint8)` needs dim>0);
  `reshape(-1)` first — load fell back 4.9s → now 0.8s.
- **`decoding="suffix_draft"`**: zero-VRAM speculative decoding — the
  SuffixMemory sidecar IS the draft model (`SuffixDraftModel` shim,
  `forge/model/ngram_mem.py`). Contract: `enable_suffix_memory()` FIRST,
  then `activate(decoding="suffix_draft")` → builds
  ExternalDraftSpeculativeDecoding on the shim. The spec loop feeds
  `suffix_mem` (prompt at prefill, `feed_one` per committed token) so
  `mem._tail` is the committed-context source of truth; the shim rolls
  only a virtual proposal chain (T==1 calls matching its last argmax;
  T>1 calls end the round). `confidence_tau=0.3` is REQUIRED at
  activation — without it, no-opinion rounds emit uniform logits →
  argmax token-0 spam burns verify forwards (measured: 96 proposals,
  0 accepts → 36/9 with tau). Telemetry: `spec_stats`/`spec_report()`
  on the strategy. E2E: bit-identical greedy output, 1.86x on
  repetitive code — note the speedup at accept_rate≈0 comes from
  amortized per-token Python overhead (2 commits/round), not
  acceptance; suffix drafts pay off when token-level recurrence is
  high, so check `spec_report()['accept_rate']` before crediting it.
- **`engine.consolidate_pairs()`** (engine_lora.py) — the L2 loop as an
  engine API: L1-gated (mem_filter drops pairs the sidecar can't
  replay) → in-engine LoRA fit (frozen base, AdamW) → honest metrics
  incl. held-out `control` CE drift (replay rows are TRAINED — they
  can never serve as the forgetting control). Options: `merge` folds
  into base weights; `save_adapter` writes a load_lora-compatible
  safetensors. Operating point on lfm3 1.7B: lr2e-4×20 steps → facts
  CE 4.35→0.0001, ctrl drift +0.80 (lr5e-4×60 → +2.17, too hot);
  replay SIZE bounds drift more than lr.
- **distill60 run record**: `research/checkpoints/lfm3_1p7b_cc.distill60.safetensors`
  — 60 steps, 29 packed seqs (37 usable conv of 39 / 150 scored rows),
  seq 2048, batch4×accum4, BitNet-everywhere + manual-LoRA(143 adapters,
  ~36M params, merged), chunked distill CE w/ teacher_logp advantage
  weighting. Loss oscillated 35.5–38.75 (advantage weights ≈3.8× CE at
  clip — not comparable to plain CE). VRAM steady 7.78GB. Checkpoint
  verified end-to-end: logits ≈ base delta-small, generation clean.

## Spark selfgen staged plan + external-DB fallback (2026-11-07)

`scripts/gen_sft3_data.py --mode selfgen` (codebase-grounded RFT: AST-harvest
`forge/`/`scripts/`/`research/` functions → fuzz inputs → sandbox ground
truth → execution-verified solver rollouts → Goldilocks keep). Proposal-
authored tasks are a documented DEAD END at 1.7B (0/10 + 0/4 smokes).
Staged plan: (1) codebase-grounded variety batch → (2) retry model-authored
prompts → (3) ONLINE DBs if variety still fails. Full annotated catalog +
licenses: `.devin/scratchpad.md` "online-DB fallback catalog" (2026-11-07).

Implementation status (2026-11-07): the mode is LIVE with three task
supplies — harvested codebase functions (fuzz + model-input rescue) and
`--external <task.json>` (reference must pass its own literal asserts in
the sandbox, then the shared solver/Goldilocks/novelty pipeline) plus a
describe-first pass for undocumented functions. `scripts/fetch_external_
tasks.py` wrote 500 opc-sft2 tasks from 506 rows (99% yield). Sandbox
worker now pre-imports pure-stdlib utility modules (93b568f). Tests:
`tests/unit/test_selfgen_rft.py` (12, incl. real sandbox execution).
Anti-overfit training recipe: `--val-every 25 --early-stop-patience 3`
+ fresh selfgen data each round + lora-only periodic saves. Val-loss
improvements now write `<save>.bestval.safetensors` adapter snapshots
(~76MB) so the best — not last — step is recoverable.

**sft4 lineage (2026-11-07)**: `spark_sft_v3.jsonl` (646 rows: 86
exec-verified opc-sft2 selfgen rows [48 goldilocks+3 easy of 392 tasks],
300 ReflectEvo reflect/repair pairs, 258 v2 replay, 2 codebase) →
sft3→sft4 LoRA-r32 150 steps, val 0.694→0.649 monotone (no early-stop
needed) → `spark_x26_1p7b.sft4.safetensors` + `.lora` + `.sft4_fq` 1.59GB.
Unseen eval 16/16 incl. 6/6 repair-prompt items. `--keep-easy-prob`
(default 0) keeps 1 verified row from always-solved tasks — all-k-pass
rows are the strongest-verified and add task variety (SFT learns from
every correct row; the 0<p<1 Goldilocks band is an RL-centric bias).
Held-out solver eval: `.devin/_solver_eval.py` + `data/external/
opc_sft2_heldout.json` (fetch `--skip 500`) measures real pass@k.

**sft4 vs sft3 verdict (2026-11-08)**: statistically indistinguishable.
Held-out pass@4 temp0.8 on 117 GT-verified tasks: sft3 1/117, sft4
0/117 — the unfiltered held-out slice sits above the 1.7B capability
ceiling, so pass@k there can't discriminate. On-manifold eval
(`.devin/_onmanifold_eval.py`: 60 unique known-solvable tasks joined
back from kept-row gen_keys, k=4): sft3 3/60, sft4 3/60 — dead heat;
only ~40% of rollouts even yield extractable+named code at temp .8.
Re-running the 86 stored corpus solutions through `run_tests` passes
86/86 — the verification loop is consistent; observed kept-task solve
rates are tail-lucky single draws, so re-solve at k=4 is expectedly
low (batch-composition A/B: no measurable layout effect; per-seed
diag: top-rate tasks re-solve 4-8/8, low-rate ~0-1/4). The eval
script now logs per-attempt passes + per-assert accuracy (dense
metric) to `results/solver_heldout_<label>.json`.
Stage-2 CONFIRMED DEAD on sft4: model-authored task proposals =
0/32 valid (abstract few-shot) + 0/24 (corpus-native few-shot —
outputs collapse to empty code fences, ~16 tokens each). External
verified DBs remain the task supply; `opc-sft2 educational_instruct`
has ~118k rows total (~660 consumed) — `opc_sft2_b.json` staged
(next 500). sft5 selfgen uses sft4_fq as solver at k=4/temp.8/
keep-easy .3.

**sft5 lineage + extraction-decline finding (2026-11-08)**:
`spark_sft_v4.jsonl` (1169 rows: 43 sft4-solved opc_sft2_b rows
[36 gold+1 easy of 489], 600 fresh ReflectEvo slice-B, 526 v3 replay)
→ sft4→sft5 LoRA-r32 275 steps, val .5629→.5297, bestval adapter.
Held-out: 0/117 pass@4 both (sft4 147 attempts acc .006; sft5 108
.010). On-manifold 60 known-solvable: sft3 3/60, sft4 3/60, sft5
1/60, sft4+bestval 3/60 — all noise-equal on solves. THE REAL
SIGNAL: runnable-extraction rate declines per round — attempts
102 (sft3) → 83 (sft4) → 53 (sft5, bestval arm identical). Each
SFT round makes the model less code-forward; the reflection-heavy
corpus is the prime suspect. Roadmap for sft6: code-heavier mix
(replay-frac ↑, reflection ↓ to ~sft4's absolute 300, add KodCode/
OpenCodeInstruct code supply — bigger verified pool), and watch
attempt-rate as the leading metric since binary pass@k is ceiling-
bound at this scale.

**sft6 lineage + abstention root cause (2026-11-08)**: `spark_sft_v5.jsonl`
(520 rows: 112 sft3-solved opc_sft2_c rows [72 gold+3 easy of 589 — best
yield yet], 200 ReflectEvo slice-C, 208 v2 replay) → sft5→sft6 LoRA-r32,
early-stop @101/175; bestval step26 val .4941, final .5042 →
`sft6.safetensors`+`.lora`+`.sft6_fq`; bestval adapter merged via
`.devin/merge_bestval.py` → `sft6.bestval_merged_fq` (224/224 adapters,
all 448 tensors attached). On-manifold: sft6_fq **5/60 pass@4 but only
25 runnable outputs** (24% precision); bestval arm 1/60, 45 runnable —
bestval is worse (more code-forward, much less precise). Extraction
decline continues: 102→83→53→**25**, yet solve precision rose ~4×.
ROOT CAUSE pinned by raw-output peek: **empty-fence abstention** — the
model emits ` ```python ` then EOS/empty bodies on hard prompts, a
learned give-up pattern, NOT prose drift. Corpus culprit: v5 fenced-code
bodies median 228ch vs v2's 770ch; 32% are <60ch stubs (v2: 17%).
Short repair rows + short external solutions taught stub/abstain
responses. sft7 direction: enforce a complete-function floor on corpus
rows (min body length, required names, no trivial returns), cut or
reformat reflection rows so they never end in stub code, replay
v2-style long solutions; lineage continues from sft6_fq (final — best
solve precision), not bestval. `opc_sft2` ~118k rows deep, ~1260
consumed; next fetch `--skip 1660`.

Stage-3 shortlist (verified on HF hub):
- **Code tasks w/ refs+tests** (drop into the exec-verified loop):
  `OpenCoder-LLM/opc-sft-stage2` `educational_instruct` (compiler-validated
  triples), `KodCode/KodCode-V1` (447k + tests; CC-BY-NC), `nvidia/
  OpenCodeInstruct` (5M, unit-test pass arrays; CC-BY-4.0), `bigcode/
  self-oss-instruct-sc2-exec-filter-50k` (execution-filtered self-instruct;
  ODC-BY), `open-r1/codeforces-cots`.
- **Reasoning**: `open-thoughts/OpenThoughts3-1.2M` (apache-2.0),
  `nvidia/OpenCodeReasoning` (735k; CC-BY-4.0), `open-r1/OpenR1-Math-220k`,
  `PrimeIntellect/SYNTHETIC-1` (2M verifier-annotated; apache-2.0),
  `allenai/big-reasoning-traces`, `nvidia/OpenMathReasoning`.
- **Token efficiency**: `horseee/MixChain-Z-*` CoT-Valve sets (same Q,
  multiple CoT lengths), `distilled-ai/condensed-thinking-sft-dataset`,
  `yrdu/S3-CoT-Self-Sampled-Data`, `SmallDoge/TokenSkip-10K`,
  `simplescaling/s1K-1.1`, `GAIR/LIMO`.
- **Behavior / self-improvement mindset**: `bigai-nlco/ReflectionEvo`
  (ReflectEvo-460k failure→locate→correct reflections; SFT+DPO),
  `nvidia/Nemotron-SFT-SWE-v3` (238k agentic SWE trajectories),
  `SWE-bench/SWE-smith-trajectories` (~102k), `lambda/hermes-agent-
  reasoning-traces`. Gap: no public DB teaches self-goal-proposal —
  AZR path remains the unique lever there.
- Pull order if triggered: opc-sft-stage2 + KodCode tests →
  OpenCodeReasoning slice → ReflectionEvo → MixChain/TokenSkip.
  License notes: KodCode non-commercial; Magicoder/OpenAI-output terms;
  OpenCodeInstruct/OpenThoughts3 are the clean-license picks.

## Environment

- OS: Windows 11, GPU: RTX 5070 12GB
- Python venv: `D:\windsurf\ForgeAI\venv\`
- Key packages: torch, transformers, safetensors, bitsandbytes, pytest

## Memory/Information-Floor Probes (2026-11-03)

Measured floors for identical-output memory reduction — see
`.devin/scratchpad.md` (2026-11-03 entry) for full numbers:
- `scripts/probe_llm_memory_math.py` — memory decomposition + Chinchilla
  coupling (halving loss-excess = x7.68 params) + softmax rank inflation
  (rank 64 -> 512: attention matrix is NOT exactly low-rank).
- `scripts/probe_exact_decode_tradeoff.py` — fractional-KV + recompute
  decode: bit-exact in fp64, argmax-identical ~98-99% fp16 on random init;
  KV memory -> 0 at ~2.3x decode cost (T=2048 mini model).
- `scripts/probe_weight_entropy.py` — trained weights are ~incompressible:
  byte-entropy 6.29/8 (floor 1.27x), mantissa 6.97/7 bits, delta entropy
  8.0/8, zeros 0.2%.
- `scripts/probe_kv_manifold.py` — real KV on proposer_mla_moe: entropy
  ~6.3/8, rank99(K)~64-90/128, int8 past-KV argmax-identical E2E.
- Dormant exact-memory paths (not wired to decode): ResidualStreamCache,
  CaptureKVCache, XQuantKVCache (rematerialize), HiSparseKVCache — all
  registered in kv_backend but decode never calls them.

## Historical Notes
R&D round documentation, refactor retrospectives, and historical notes
have been moved to `docs/CHANGELOG.md` (extracted 2026-09-09, critique F1).
