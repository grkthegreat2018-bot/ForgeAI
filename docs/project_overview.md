# ForgeAI Project Overview

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                    ForgeEngine (Inference)                    │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ KV Cache │  │ Decoding  │  │ Quantize │  │ Innovations│  │
│  │ (7 strats)│  │ (5 strats)│  │ (INT4/8/ │  │ (MRL/V0/  │  │
│  │           │  │           │  │  FP8)    │  │  QuaRot)  │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ConfigurableResearchLLM (ForgeLM V2, Jamba-3B)    │   │
│  │  26 Mamba + 2 attn | ~3B params | ~6GB bf16       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         ↕                    ↕                    ↕
┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐
│  Self-Play Loop │  │  SFT Training│  │   Evaluation      │
│                 │  │              │  │                   │
│ Task Curriculum │  │ JSONL data   │  │ Benchmark compare │
│ → Tool calls    │  │ → Render     │  │ (quality/code/    │
│ → Execute tools │  │ → Tokenize   │  │  reasoning/tool)  │
│ → Reward        │  │ → CE loss    │  │                   │
│ → Save to DB    │  │ → Save .sft  │  │ Promote/Demote    │
│ → Export traj   │  │              │  │                   │
│ → Finetune/GRPO │  │              │  │                   │
└─────────────────┘  └──────────────┘  └───────────────────┘
```

---

## 1. ForgeLM Model

**File**: `forge/model_loader.py`, `forge/config.py`

| Property | Value |
|----------|-------|
| Base | AI21 Jamba-Reasoning-3B (100% lossless port) |
| Params | ~3B |
| Size | ~6 GB (bf16) |
| Layers | 28 (26 Mamba + 2 GQA attention) |
| d_model | 2560 |
| Heads | 20 query, 1 KV |
| head_dim | 128 |
| FFN | SwiGLU (intermediate=8192) |
| Norm | RMSNorm |
| Position | No RoPE (Jamba attention is position-agnostic), 256K context |
| Vocab | 65536, untied embeddings |
| Configs | `forgelm_v2` → `forgelm_v12_jamba` → `forgelm_v13_gdn2` → `forgelm_v14_rsi` (preset lineage, AGENTS.md §A); `forgelm_tiny` (4-layer test) |

**Key classes**:
- `ConfigurableResearchLLM` — full model with chunked CE, MTP, MoE aux loss
- `PreAllocatedKVCache` — O(1) append, INT8 quant + tensor-position (`pos_t`) static-graph mode
- `ModelLoader` — architecture caching, memory-mapped loading, HF key remapping, hybrid offload

**Checkpoint**: `research/checkpoints/ForgeLM_V2.safetensors` (current production V2)
**Tokenizer**: `research/checkpoints/forgelm_v2_tokenizer/`

**Non-transformer memory families** (`forge/model/flux/`, `forge/model/strata/`):
FluxLM is a sparse online associative-memory store repositioned (competitive
audit A6) as *a revertible memory layer, not an LM* — journaled bit-exact
undo + `explain()` cell provenance; the competitive set is agent memory
(mem0/MemGPT/Letta), not transformers. Head-to-head vs a TF-IDF RAG
baseline on held-out doc-grounded QA: `scripts/bench_flux_vs_rag.py`.
STRATA is the deployment container format — division of labor in
`docs/STRATA_VS_FLUXLM.md`.

---

## 2. ForgeEngine

**File**: `forge/engine/forge_engine.py`

**Purpose**: Unified inference engine — orchestrates all runtime strategies.

**API**:
```python
engine = ForgeEngine.from_checkpoint("checkpoint.safetensors")
engine.activate(kv_cache="paged", decoding="standard", quantize="int4", warmup=True)
output = engine.generate("Hello", max_new_tokens=100, temperature=0.7)
```

**Features**:
- Auto-detects VRAM capacity → enables AirLLM streaming if model doesn't fit
- Auto-detects KeyStack features from checkpoint metadata
- CUDA graph capture for decode acceleration
- Warmup pass to initialize CUDA kernels
- Prefix caching for repeated prompts
- Sleep/wake for model switching (Level 1: CPU offload, Level 2: discard weights)

**Activate options**: kv_cache, decoding, quantize, acceleration, mrl_keep_ratio, kv_bits, use_v0_warm, use_progressive_kv, use_compile, use_prefix_cache, use_spec_attn, kv_cache_tokens, warmup

---

## 3. KV Cache Backends

**File**: `forge/engine/kv_backend.py` — audited 2026-10-31 (A2).

Decode runs on per-layer `past_kv` tensors directly; the ~30 registered
`KVCacheStrategy` classes are bookkeeping objects — only the
`compact_past_kv` seam is live. `info()` reports `decode_live` per
strategy; only these compact the real cache:

| Strategy | Description | Status |
|----------|-------------|--------|
| beacon | Importance-scored prefix eviction | ✅ decode-live |
| anchor | Anchor-token retention + eviction | ✅ decode-live |
| metakv | Learned/meta eviction policy | ✅ decode-live |
| flux | Flux-scored compaction | ✅ decode-live |
| standard / paged / rotorquant / hadamard_int4 / compressed / streaming / snapkv / ~20 more | Bookkeeping strategies (`append`/`get` never called by decode) | ⚠️ dormant (`decode_live: False`) |

---

## 4. Decoding Strategies

**File**: `forge/engine/decoding.py`, `forge/decoding/`

| Strategy | Description | Speedup | Status |
|----------|-------------|---------|--------|
| standard | Autoregressive with KV cache | 1x | ✅ Working |
| speculative | Draft model + verify | 2-3x | ✅ Working (needs draft model) |
| medusa | Parallel prediction heads | 2.2-3.6x | ✅ Working |
| dspark | Semi-AR + confidence scheduling | ~3x | ✅ Working (experimental) |
| mtp_selfspec | MTP heads self-speculative | 2-4x | ✅ Working |
| eagle3 | Feature extrapolation (EAGLE-3) | up to ~5x | ✅ Working (`forge/decoding/eagle.py`) |

**Key files**:
- `forge/decoding/medusa.py` — MedusaHeads, MedusaTrainer, medusa_generate
- `forge/decoding/dspark.py` — DSparkHead, DSparkTrainer
- `forge/decoding/mtp.py` — MTPHead, MTPTrainer

---

## 5. Quantization

**Files**: `forge/engine/quant/`, `forge/quant/`

| Method | Type | Compression | Status |
|--------|------|-------------|--------|
| INT4 weight-only | Per-group symmetric | ~3.8x | ✅ Working |
| INT8 weight-only | Per-channel | ~2x | ✅ Working |
| FP8 (Blackwell) | torch._scaled_mm native | ~2x | ✅ Working |
| RotorQuant KV | Givens + Lloyd-Max | 3-4 bit | ✅ Working |
| Paged KV | vLLM-style | N/A | ✅ Working |
| KV Compress | H2O + KVQuant | Variable | ✅ Working |
| BitNet b1.58 | 1.58-bit | ~8x | ✅ Working (`forge/keys/quantization/bitnet_b158_key.py`) |
| SpinQuant | Hadamard rotation | N/A | ⚠️ Key only |
| WANDA | Pruning | Variable | ❌ Not implemented |

---

## 6. Innovations (Runtime)

**File**: `forge/engine/innovations.py`

| Innovation | Description | Status |
|------------|-------------|--------|
| MRL Adaptive Context | Truncate to d_keep dimensions, reduce compute O(d²)→O(d'²) | ✅ Working |
| QuaRot KV | Hadamard rotation for better KV quantization | ✅ Working |
| V0 Warm Start | ValueResidual V_0 pre-populates KV cache | ✅ Working |
| Progressive KV | Anchor (MSB) + residual (LSB) split streams | ✅ Working |

---

## 7. Self-Play System

### 7a. Infinite Tool Loop

**File**: `forge/self_play/infinite_loop.py` (unified loop; merged tool-use + AZR curriculum)

**Flow**: Self-Play → Export → Finetune/GRPO → Evaluate → Promote/Demote → Repeat

**Config** (`LoopConfig`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| tasks_per_epoch | 50 | Tasks per self-play session |
| max_turns | 5 | Max tool-call turns per task |
| max_gen_tokens | 256 | Max tokens per generation |
| temperature | 0.7 | Sampling temperature |
| min_reward | 0.4 | Min reward to save trajectory |
| ft_max_steps | 200 | Finetune steps |
| ft_lr | 1e-5 | Finetune learning rate |
| ft_batch_size | 2 | Finetune batch size |
| ft_seq_len | 1536 | Finetune sequence length |
| use_grpo | False | Use GRPO instead of SFT |
| grpo_group_size | 4 | GRPO group size |
| max_epochs | 10 | Max loop iterations |
| eval_threshold | 0.7 | Min quality to promote |

### 7b. Tool Registry

**File**: `forge/self_play/discovery/discovery_tools.py`

| Tool | Purpose |
|------|---------|
| think | Record train-of-thought |
| sudo_think | Meta-reason about process |
| run_script | Sandboxed Python execution (8s timeout, no network) |
| web_search | DuckDuckGo HTML search (no API key) |
| save_research | Persist web research finding |
| propose_theory | Log hypothesis |
| update_theory | Update theory status/evidence |
| record_discovery | Record confirmed finding |
| query_db | Read-only SQL against memory DB |
| migrate_schema | Audited DDL (CREATE/ALTER/INDEX only) |
| summarize_context | Model summarizes its own context |
| finish_session | End discovery session |

### 7c. Qwen Adapter (Tool Call Format)

**File**: `forge/self_play/discovery/qwen_adapter.py`

- Renders tool calls with literal text markers (built from hex to avoid IDE issues)
- Parses with 3-phase fallback: marker-wrapped JSON → bare JSON → LFM2.5 Pythonic format
- Uses `json_repair` for malformed JSON
- **xGrammar integration**: Constrained decoding — free text until `{` emitted, then grammar enforces valid JSON with tool name enum
- Loads fresh HF tokenizer for xgrammar (gigatoken wrapper not supported)

### 7d. Context Manager

**File**: `forge/self_play/discovery/discovery_loop.py` (rolling transcript window — standalone `context_manager.py` was removed)

- Token budget: 32K max, 4K reserved for generation
- Compresses at 75% capacity
- Keeps system + tool defs + last 6 turns intact
- Never splits tool_call + tool_result pairs
- Heuristic or model-generated summaries

### 7e. Discovery DB

**File**: `forge/self_play/discovery/discovery_db.py`

SQLite database with 11 tables: sessions, thoughts, scripts, research, theories, discoveries, events, schema_migrations, epochs, distill_runs, tool_trajectories.

Key: `tool_trajectories` stores full multi-turn conversations with reward scores for SFT/RL training.

### 7f. Epoch Manager

**File**: `forge/self_play/discovery/epoch_manager.py`

- DB quality score: coverage (40%) + resolution (20%) + script quality (20%) + trajectory quality (20%)
- Distill every 12 epochs, finetune otherwise
- Compare candidate vs best on quality/skill/compute → promote or archive

### 7g. ForgeEvolve / CodeEvolve (program evolution)

**Files**: `forge/evolution/` (engine: `engine.py`, domains: `domains/`, DB: `database.py`), `forge/evolution/code_evolve.py`, `forge/evolution/targets/`, `forge/evolution/eval_pool.py`, `forge/evolution/mutator.py`

- **ForgeEvolve** — MAP-Elites/bandit evolutionary optimizer over `BaseDomain` specs; CLI `python -m forge.evolution --domain <name>`. `scout.py` is the OpenEvolve-lite discovery engine (TPE proposer + domain bandit + cross-domain param priors).
- **CodeEvolve** — OpenEvolve-style *program* evolution: `Target` files declare `SEED_CODE`/`REQUIRED`/`test(ns)`/`bench(ns)`/`METRIC`/`GOAL`; proposers (`mut`/`mix`/`llm`/`basin`/`none`) emit candidates → free gates → isolated eval (`_candidate_eval.py` subprocess or persistent `EvalPool` workers) → replicated verify → promote on verified win only.
- **RSI loop** — `scripts/rsi_evolve.py` runs the target set per round, exports `export_sft` (verified child>parent pairs) + `export_validity_pairs` (validity-DPO: malformed/dup emissions as rejected), then `finetune_proposer --mode lora` hones tagged adapters that hot-swap next round — the proposer trains on its own verified wins.
- **CLIs** — `scripts/fn_evolve.py` (single target), `scripts/evolve_all.py` (evolve-only campaign), `scripts/rsi_evolve.py` (evolve+hone campaign).
- See `docs/EVOLVE_ROADMAP.md` for the shipped/queued list and AGENTS.md for the full parameter reference.

---

## 8. Training / Finetune

### 8a. SFT Training

**File**: `forge/training/runners/sft_train.py`

**Data formats**:
- `tool_use_fc`: JSONL `{"messages": [...]}` — multi-turn function calling
- `short_cot/code/tool_use`: JSONL `{"prompt": ..., "response": ...}` — single-turn

**Training features**:
- Completion-only loss (prompt masked with -100)
- Per-turn splitting for multi-turn (each assistant turn = separate example)
- Chunked CE (memory-efficient, avoids materializing full logits)
- LoRA support
- Gradient checkpointing
- EMA (exponential moving average)
- torch.compile support
- Cosine LR schedule with warmup
- Gradient clipping (1.0)

**CLI**: `python -m forge.training.runners.sft_train --data ... --checkpoint ... --save ... --max-steps 30 --lr 1e-5`

### 8b. Chunked CE

**File**: `forge/training/losses/chunked_ce.py`

Pure-PyTorch fused linear+CE — avoids materializing full [B*T, V] logits. 2-3x slower than Liger Triton kernel but works on all GPUs.

### 8c. GRPO Trainer

**File**: `forge/self_play/grpo_trainer.py`

**Fully implemented** with advanced features:
- MC-GRPO: Median baseline for small groups
- Turn-level advantages for multi-turn tool use
- KL penalty: KL(π_current || π_ref)
- PPO-style clipping
- ACR (Advantage Collapse Rate) monitoring
- Extensions: SC-GRPO, OM-GRPO, GVPO
- Tool-use rewards: continuous 0..1

### 8d. Training Data

**Directory**: `research/data/finetune/`

| Dataset | Content |
|---------|---------|
| azr_epoch1.jsonl | AZR self-play curriculum output |
| r30_training.jsonl / r31_training.jsonl / r31_v2_training.jsonl | R30/R31 mixed SFT rounds |
| r30_bbh, r30_gsm8k, r30_metamath, r30_orca_math | R30 reasoning/math subsets |
| r30_code_alpaca, r30_magicoder_oss, r31_code | Code instruction data |
| r30_no_robots, r30_openhermes | General instruction data |
| r31_single_tool, r31_multi_turn | Tool-use / multi-turn data |
| tasks/ | Task-definition data |

**Distillation data**: `data/hf_datasets/` (moved out of the `forge/` package in A20) — bbh, bigbench, codeforces, code_alpaca, dolphin_r1, folio, gsm8k, hermes_fc, magicoder, math_hard, metamath, no_robots, openhermes, …

---

## 9. Evaluation

**Files**: `forge/evaluation/`

| Benchmark | Description | Status |
|-----------|-------------|--------|
| external_eval.py | ARC-E/C, HellaSwag, MMLU (loglikelihood MC) + GSM8K (greedy) over `data/eval/*.jsonl`; `scripts/run_external_eval.py --compare` regression gate | ✅ Working (A1) |
| goal_scorer.py | Multi-dimensional: minimalism, efficiency, diversity, consistency, confidence | ✅ Working |
| ckpt_tester.py / goal_tasks.py | Checkpoint eval harness + task set | ✅ Working |
| test_model_compare.py | Quality/code/reasoning/tool-use comparison (25 tests) | ❌ Removed |
| reasoning_benchmarks.py | ARC-AGI-2, NeoCoder, FineReason, ThinkBench | ❌ Removed |
| livecodebench_eval.py | LiveCodeBench (ICLR 2025, contamination-free) | ❌ Removed |
| prompt_tests*.py | Prompt-based evaluation | ❌ Removed |

**Benchmark categories** (formerly test_model_compare.py, now removed — coverage moved to `forge/evaluation/` goal tasks):
- Tool use (3 tests): novel tool, parallel tools, nested args
- Reasoning (5 tests): capitals, math, Fibonacci, syllogism, primes
- Code (5 tests): BST, LRU cache, binary search, etc.
- Conciseness (5 tests): word limits, lowercase constraints
- Instruction following (5 tests): format, style constraints
- Self-correction (2 tests): false premise, sycophancy

---

## 10. Model Merging & Injection

**Files**: `research/merge_models.py`

**Merge methods**: SLERP, TIES, DARE, SVD, Task Arithmetic, Linear (model soup)

**Injection types**: removed (`inject_and_merge.py` no longer exists)

**CLI**:
```bash
python -m research.merge_models --method slerp --model-a a.safetensors --model-b b.safetensors --out merged.safetensors
```

---

## 11. Runtime

**Files**: `forge/runtime/`

| Component | Description | Status |
|-----------|-------------|--------|
| cuda_graph.py | CudaGraphRunner + CudaGraphGenerator (capture/replay) | ✅ Working |
| vram_manager.py | Boot profiling, dynamic KV sizing, compile cache | ✅ Working |
| forward_cache.py | LRU cache for repeated forward passes (20-40% savings) | ✅ Working |
| self_model.py | Self-model implementation | ✅ Working |
| task_logger.py | Task logging | ✅ Working |

---

## 12. Current Production State

| Component | Checkpoint | Notes |
|-----------|------------|-------|
| Best model | ForgeLM_V2.safetensors (`forgelm_v2` preset) | Jamba-3B port; V12→V14 preset chain extends it |
| Self-play | Working end-to-end | xgrammar constrained decoding, 7/8 tool use |
| Tool format | Literal text markers | Model outputs messy but parser is lenient |
| Training | SFT with completion-only loss | No anti-regression yet, no token reweighting |
| Evaluation | `forge/evaluation/` goal scorer + ckpt tester | Multi-dimensional goal tasks (25-test compare suite removed) |

**Known issues**:
- Model outputs `<tool_call:` with colon instead of proper marker format (parser compensates)
- WANDA pruning not implemented (BitNet b1.58, FP8, NVFP4 are live)
- No anti-regression techniques in SFT (no L2-SP, no token reweighting)
- fastsafetensors falls back to standard (missing DirectStorage DLLs on Windows)
