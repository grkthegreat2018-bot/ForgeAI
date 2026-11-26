# RSI-Loop Research: External "White Space" Digests — Reconciliation

Compiled 2026-11-23. Fifth in series. Two external landscape digests proposed
~25 additions. Reconciled against the actual codebase: most are already
built, several confirm existing design choices, and a handful are genuine
gaps. Sorted accordingly.

---

## A. Already built (claimed white space, actually present)

| Suggestion | Status in ForgeAI |
|---|---|
| KV eviction strategies | **20+ strategy registry** (`kv_backend.py`): paged_eviction, snapkv, hisparse, evo_sparse, **hqe_kv** (HqeKV literally implemented), kvzip, spectral_kv, beacon_kv, anchor_kv, moment_kv, metakv, kvpop, streaming_llm, filler_kv, conf_kv, vtoken_kv, vegas_kv, learned_prefix_cache, flux_kv, auto_context + cpu/disk offload. Caveat (honesty note kv_backend:1-10): most are bookkeeping-only; live seam is `compact_past_kv`. |
| RadixAttention-style prefix reuse | `scheduler/unified_radix.py` + ChunkedPrefixCache + session KV |
| `--n-cpu-moe` offloading | `airllm_streamer.py`/`airllm_ring.py` — pinned shard + 3-slot GPU ring + side-stream prefetch (**measured 3.3→30-36 tok/s**); hybrid conv-CPU/attn-GPU fallback in loader; `cpu_kv_offload.py`, DiskKVCache 3-tier |
| CUDA-graph decode priority | BCG decode measured **5.7× on V14 (108 tok/s bf16)**, 918 agg tok/s spark B=32, 1271 tok/s LFM3-_fq B=32. The "Crossover Point" paper (RTX 3060: graphs 1.85× > any speculation gain; fixed γ beats adaptive controller) **confirms existing choices** — keep BCG first, prefer fixed per-domain γ over `adaptive_spec` complexity |
| Speculative decoding | Full registry: mtp_selfspec, medusa, eagle3, dspark, ngram/suffix, specbound, self_speculative_sparse, external_draft |
| DAPO trainer | CISPO asymmetric clip (0.2/0.28 = DAPO clip-higher equivalent) + zero-variance pre-filter already in GRPO; remaining gap = **dynamic-sampling backfill** (flagged in TRAINING_OBJECTIVES doc §1a) |
| RLVR | `rlvr_train.py` + unit-test verified rewards — the loop's core |
| Model merging | `research/merge_models.py` (extensive); Zenodo negative result (naive SLERP fails, **TIES works @density 0.55**) confirms the TIES recommendation already in doc 2 |
| Structured agent memory | LiveT (GDN2 states + TitanMemory + NGramMemory sidecar, journaled) — in-model alternative to external DB memory (Hindsight/Severian class) |
| Mamba/SSM beyond-transformers | mamba3 key + GDN2 + conv operators in-tree; ahead of the suggestion |
| TTC as capability lever | Covered in CAPACITY doc: verifier head + BoN + gated loops + per-task TTT |
| Self-correction via feedback | `persistent_executor` sandbox + verifier = T1's tool-integrated verification already; reflexion-style self-judging documented as failing at ≤3B |

## B. Confirms existing decisions (no action — good signal)

- **"Fewer Bits, More Tries"** (arXiv 2610.15372): capacity-for-retries
  trade = exactly the `_fq` weights + verifier-head + batched-BoN design.
- **Self-rewriting weights** (27B rewrites own weights via reasoning,
  +6.6pts): paradigm-adjacent to fn_evolve's code-mutation targets —
  the loop already rewrites *its own* weights via LoRA hone; the
  gradient-free variant is a spike candidate, not a gap.
- **Crossover paper**: CUDA graphs > speculation on consumer GPU, and
  adaptive speculation-width controllers don't beat a well-chosen fixed γ
  (headroom only 5.8%). Validates: BCG-first strategy, `adaptive_spec`
  complexity probably not worth it — align with sunset policy.
- **Qwen3.5-9B 97%-empty-on-8GB caution**: per-model/per-GPU validation
  mandatory — our bench_proposer gate already enforces this.
- **SFT→GRPO→DPO ordering, sign-advantage, LoRA-RL rank ceiling** —
  all already covered in TRAINING_OBJECTIVES doc.

## C. Genuine white spaces (ranked by leverage × cost)

| Rank | Gap | What it is | Fit |
|---|---|---|---|
| **1** | **Harness-level evolution** (DarwinX 2608.07545, Mem2Evolve ACL'26) | The loop evolves *weights* (LoRA hone) and *code solutions* — but not the **harness around the frozen model**: proposer prompts, system prompts, tool configs, sampling params, gen budgets as an evolvable genome with preserve-and-extend admission. DarwinX: ~+17pts avg on frozen model. | Real architectural gap. `code_evolve`/`infinite_loop` already have: per-task stats, config objects, archive DB (`forge_evolve.db`) — the genome is one dataclass + archive-table away. Cheap, novel for this codebase. |
| **2** | **BES bidirectional search** | Forward trajectory recombination + **backward goal decomposition** (decompose the target into sub-goals, search backwards) — escapes rollout entropy shell; beats BoN/MCTS on competitive programming | New proposer family for `fn_evolve.py`/`code_evolve`: current proposers (basin/mut/llm) are all forward-only. Cheap: a backward-decomposition prompt pass + sub-goal queue. |
| **3** | **SGS conjecturer-degeneracy guard** (Self-Guided Self-Play) | Curriculum proposer can collapse to trivial/unsolvable tasks; model-guided away-from-degenerate steering | Real loop risk for `infinite_curriculum`; complements the AZR learnability reward (target ~50% pass) already recommended in doc 2. Cheap: add degeneracy detector + steer penalty to task gen. |
| **4** | **MoE-Lego bitplane / CommitMoE predictive router** | Dynamic-precision expert offload; predictive expert prefetch | Only relevant when `forgelm_v14_rsi` MoE + offload lands. Note for V14-P1, not now. |
| **5** | **PolyKV shared multi-agent KV pool** | Write-once shared KV across agents on same context | The loop is single-proposer + batched — no multi-agent same-context path today. Park until a multi-agent feature exists. |
| **6** | **Learned eviction (KVP RL-per-head / RoleKV)** | RL/learned eviction policies vs heuristic | KV is ~50-200MB at loop contexts — not the binding constraint (doc 3). Revisit only for long-context serving features. |

## D. Evaluated and deferred

- **Diffusion LLM backend (DiffusionGemma/BlockServe)**: new serving
  paradigm (parallel denoise), 2000+ tok/s claims on big quantized MoE —
  but a different execution model than the whole engine; earlier
  DIFFUSIONBLOCKS research was archived. Exploratory spike at most,
  after the LFM3 capacity path settles.
- **ik_llama.cpp IQ quants**: GGUF-ecosystem reference only; ForgeQuant
  is the in-engine path. Useful solely as a quant-quality benchmark
  baseline if `_fq` dNLL ever gets questioned.
- **Strata 125B@12GB / canifinetune / Mighty Mouse / InventiveHQ
  benchmarks**: reference implementations and measurement harnesses —
  mine for methodology, don't integrate.
- **Hindsight/Severian/Neo4j memory DBs**: external agent memory —
  LiveT covers the in-model axis; external memory only if a GUI/agent
  product need emerges.
- **Mamba-3/RWKV presets**: mamba3 already a key in-tree; RWKV
  not worth a port slot vs LFM3 work.

## E. Action notes

1. **Harness genome spike** (highest leverage): define `HarnessGenome`
   (proposer prompt template, sampling params, tool flags, gen budgets,
   mutation weights) + archive lineage in `forge_evolve.db` + preserve-
   and-extend admission gate. Frozen-model capability axis orthogonal
   to every weight-side lever in docs 1-4.
2. **BES proposer**: add `--proposer bes` — backward goal-decomposition
   pass emitting sub-goal queue + forward solve per sub-goal; eval vs
   basin/mut proposers on the standard evolve suite.
   **SHIPPED 2026-11-24** — `forge/evolution/bes_proposer.py`:
   `BESProposer` wraps `LLMProposer`; pass 1 strips the reply contract
   and asks for a bounded numbered sub-goal list (`cfg.bes_subgoals`,
   default 8), pass 2 injects them before the reply contract and emits
   one fenced program (joint solve). `propose_batch` = one decompose
   batch + one solve batch; non-proposal prompts (judge/critique) pass
   through single-pass; decompose failure = plain `llm` fallback.
   `note()` maps credit to the solve prompt for hint-bandit attribution.
   Remaining: measured eval vs basin/mut on the standard suite.
3. **Degeneracy guard**: infinite_curriculum — flag tasks where
   proposer success-rate drifts to extremes; steer by rephrasing/
   difficulty scaling (uses existing task-stats machinery).
4. Keep `adaptive_spec` OFF default per Crossover finding; fixed γ per
   domain (code vs prose) if γ is tuned at all.
5. Add VestigeKV-style **arch-aware eviction signal** to the KV backlog
   only if Granite-h (NoPE layers) ever ports — the idea generalizes:
   repurpose per-layer salience channels for query-free eviction.
