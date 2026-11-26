# RSI-Loop Research: Memory Topology — Train/Infer Decoupling on 12GB

Compiled 2026-11-23. Third in series: `RSI_LOOP_BASE_LORA_RESEARCH.md` (base
+ adapters) · `RSI_LOOP_TRAINING_OBJECTIVES.md` (objectives). This one
reconciles generic "QLoRA + GGUF + merge-restart" advice against ForgeAI's
actual engine paths, with measured numbers.

---

## 0. The pasted plan vs this stack

Generic advice assumes HF transformers + PEFT + GGUF/llama.cpp. ForgeAI
runs one custom engine for both rollout and training, with stronger
native equivalents for every component named:

| Generic advice | ForgeAI equivalent (already built) | Verdict |
|---|---|---|
| Unsloth kernels | Fused Triton GEMV/GEMM + CUDA-graph decode + chunked-CE + custom trainer | Superseded; Unsloth can't load LFM3 anyway |
| QLoRA NF4 base | ForgeQuant `_fq` int4+int8-outlier (~4.9bit) + `--qlora-lb2` + NF4Linear | Exists; but bf16-LoRA fits at 1.7B — quant base only when stacking engines |
| Paged optimizer | **CPUAdamW ZeRO-offload** (fp32 m/v + master in pinned CPU RAM; auto-enabled <15GB) + FlashAdamW 8-bit states + NvmeMuon4Bit | Stronger |
| Gradient checkpointing | `--grad-checkpoint` (default True) + **selective** recompute + elastic `--grad-accum` on VRAM pressure | Exists — prefer selective (~5% overhead) over full (+33–58%) |
| GGUF Q4_K_M serving | `_fq` packed ckpt + `prequant_forge_quant.py` offline convert | Exists — **but see §3 `_cache_w` caveat** |
| KV quantization | `kv_bits` 4/8 + `swa_ring_kv` + 20-strategy KV registry | Exists — **do NOT use on LFM3** (§2) |
| Draft-model speculative | MTP self-spec, dSpark, suffix/ngram, EAGLE-3, **CopySpan edit head** | Far ahead — CopySpan targets the actual workload |
| Merge→requantize→serve each round | `merge_lora_adapters` dequant→RTN-repack | **Actively harmful per-cycle — fix in §4** |

---

## 1. Corrected budgets (measured, RTX 5070)

### Training phase (LoRA hone / GRPO phase)

- Spark SFT arms: **peak 5.26GB**. GRPO phase: **~5GB total** after ref
  elimination (`infinite_loop.py:2744` docstring; ref was costing 3.4GB).
  Distill smoke seq2048: **7.78GB**. L2 LoRA-r8 consolidate @lr5e-4:
  **11.8GB tight** (134s/fit — an outlier, likely un-chunked logits or
  big batch).
- → **bf16-LoRA at 1.7B ≈ 5-7GB**, NOT "8-12GB QLoRA". The external
  measurements agree: Llama-3.2-1B r16 QLoRA = 5.35GiB, 3B = 6.69GiB
  (backpropagate estimator). QLoRA saves ~2-3GB at this scale — not
  needed unless running train+serve concurrently.
- **Hidden hog = 128K-vocab logits**: b4×4K×128K fp32 ≈ 8GB unchunked.
  `use_chunked_ce` is forced on (`sft_train.py:1726`) — keep it; it's
  the reason 5GB budgets work. GRPO path uses it too (`grpo_trainer.py:640`).
- Optimizer state on adapter-only params ≈ 0.3-0.6GB — paged/8-bit
  optimizers buy nothing here (bnb paged costs ~125ms/GB evicted/step
  when it does fire). CPUAdamW matters only if weights' grads/state go
  full-model.
- Skip full gradient checkpointing at this scale — selective recompute
  (~5% step cost) or none; full recompute taxes +33-58% step time.

### Inference phase (rollout/proposal)

Measured (scratchpad + bench notes):
- `lfm3_1p7b_cc_fq` static B=32 CUDA-graph: **1271 tok/s @ 5.80GB peak**;
  B=16 874-1053 tok/s. bf16 BCG on spark: 918.7 agg tok/s @B=32, 4.98GB.
- `spark sft10_fq` engine: 1.87-4.36GB resident.
- Granite-4.1-3B **unquantized BCG = 12.44GB — tight**; needs `_fq` for
  the challenger role.
- Decode is weight-bandwidth-bound at self-play batch sizes: int4 weights
  ≈ 3× less bytes/step than bf16. That is the speed lever — not KV quant.

### KV cache — the pasted "1.5GB @128K" framing is wrong for this loop

Loop bound: ~4K tokens/proposal (proposer max_new 448, solver 256;
`code_evolve.py:859` prompt ~2K).

| Model | KV bytes/token (fp16, computed) | 4K ctx | 16K ctx |
|---|---|---|---|
| `lfm3_1p7b` (6 full ×4kv×128 + 5 SWA cap ~2.6MB + convs ~0) | **12 KiB** | ~48MB | ~200MB |
| `spark_x26_1p7b` (7 full ×2kv×256 + SWA ring@512 ~44MB) | ~14 KiB | ~56MB | ~230MB |
| `parable_granite41_3b` (40 full ×8kv×64) | **80 KiB** | ~320MB | ~1.3GB |

- Measured `code_evolve.py:896` reports ~49KB/tok/row for LFM3 —
  higher than the 12KiB analytic figure (likely per-layer allocation for
  all attn types + layout overhead). **Verify**: check
  `PreAllocatedKVCache` allocation for SWA layers vs analytic. Even at
  49KB, 4K ctx = ~200MB — still small.
- **Do not quantize the KV on LFM3**: FP8-KV collapsed Gemma-3-1B whose
  5:1 local:global layout "carries long-range retrieval in few global
  layers with no redundant pathway" — LFM3's 6-of-24 global layers are
  exactly that shape. vLLM measured ~zero decode gain from FP8-KV on
  SWA/hybrid models anyway.
- `kv_bits` stays fp16; spend compression on weights.

---

## 2. Loop topology — already right, with two sharpenings

### What's correct in the pasted design

- Decoupled budgets, serialized phases: ForgeAI already does this —
  `_gen_lock`, `_park_engine` (sleep(1), host-RAM gate→sleep(2)),
  `release_engines(fast=True)`, weight-swap without reload (~5-10s,
  `param.data.copy_`).
- Merge in full precision: `merge_lora_adapters` dequant→merge→requant.
- Regression gate before promotion: `fast_eval` + canary quarantine.

### Sharpening 1 — always-awake colocate is feasible at 1.7B

int4 rollout weights (~1.4GB `_fq`) + bf16 train copy (3.4GB) + KV
(~1.5GB at b32×16K worst) + train activations (~3GB) ≈ **9-10GB < 12GB**.
Sleep/wake (L1 ~2-3s + H2D) can be skipped when running single-phase
colocate; keep sleep(1) only when GRPO-phase peaks demand it.
vLLM measured reference points: L1 wake-to-token ~0.6-0.7s, restart
tens of seconds — our weight-swap (~5-10s) already beats restart.

### Sharpening 2 — the off-policy gap from `_fq` rollouts

Serving `_fq` (~4.9bit) rollouts while the trainer computes bf16
logprobs = the quantized-rollout regime — now a named, measured problem:

- **TIS** (truncated IS): clamp `min(π_train/μ_rollout, C)` per token —
  merged into verl; SkyRL quantized-rollouts is exactly this pattern.
  Rollout-side μ logprobs are FREE in-process (record at gen time).
- **QaRL** (arXiv 2604.07853): w = π_bf16/π_quant drifts far from 1 on
  long outputs; fix = rollout-aligned QAT forward in trainer + trust-
  band dual clip (+5.5 vs naive on Qwen3-30B).
- **AIS** (2605.13907): mismatch is non-stationary — early = exploration
  bonus, later = destabilizer; interpolate correction strength.
- **Jet-RL**: make the inference graph a subgraph of the training graph.
  ForgeAI can approximate this nearly free: run the trainer forward
  through the same dequant path the engine uses (ForgeQuant dequant is
  differentiable through scales) — removes the weight-value component.
- Add a **quant-KL diagnostic**: log-ratio distribution per merge cycle
  = the known collapse early-warning metric.
- Same-dtype subtlety: kernel-order mismatch exists even at bf16
  (batch-variant reductions); FP16 trainer forward nearly eliminates
  the two-engine gap (arXiv 2510.26788) — cheap A/B if instability shows.

---

## 3. `_fq` serving caveat (measured in-repo)

`_cache_w=True` dequant-caches int4→bf16 at load → `_fq` saves DISK and
load-peak only; runtime VRAM ≈ bf16 on that path (scratchpad:1554).
The proposer path deliberately sets `_fused_gemv=False; _cache_w=True`
(cuBLAS decode 3.3× faster, costs ~3.4GB). Implication:

- int4's runtime-VRAM win only materializes on the fused-GEMV decode path
  (no `_cache_w`). Choose per engine role:
  proposer (throughput-critical) → `_fq` + fused GEMV, ~1.4GB weights;
  trainer-side eval engine → bf16 is fine.
- The 15+min single-core on-the-fly FQ convert is the known pain —
  `scripts/prequant_forge_quant.py` does it offline; fold into the
  merge-cycle tooling (materialize `_fq` copy after each canonical merge).

---

## 4. The real bug: merge-cycle integrity (sharpest finding)

Current path per loop iteration: `merge_lora` on quantized linears does
dequant→add-delta→**RTN re-pack** (`bitnet_lora.py:459-462`), then
`load_from_weight` **rebinds `param.data` to a new CPU tensor**
(`forge_quant.py:469-471`):

1. Each merge folds a fresh RTN error draw into "canonical" weights —
   **quant error compounds across rounds** (external evidence: all
   merge-into-4bit methods degrade; PEFT #1043).
2. `.data` rebind + CPU move **breaks captured CUDA graphs** (graphs hold
   addresses) and silently migrates weights off GPU — same bug class as
   the ATOM stale-graph crash.

**Target design:**
- Persistent **fp32/bf16 canonical master on CPU** (32GB RAM; 1.7B bf16
  = 3.4GB). `_save_original_weights` (`engine_activation.py:800-826`) is
  the existing skeleton — currently one-shot pre-quant, not per-cycle.
- Per round: apply LoRA delta to canonical (fp32) → re-quantize →
  **write packed buffers in place via `copy_`** into the same GPU
  storage (graphs stay valid).
- Better still for the RL loop: **don't merge at all** — keep the
  resident adapter and update A/B in place (adapter-only sync = the
  verl `lora.merge=False` pattern, zero quant error, zero sync). Merge
  only at epoch-end checkpointing, into the canonical master.
- LoftQ-style init if merge-in-place is kept: init new A/B from SVD of
  the current quant residual — absorbs rather than adds quant error
  (queued already in EVOLVE_ROADMAP).

---

## 5. Storage/scheduling notes

- DiskTokenCache on D: (packed single-file sequential) is right for the
  SMR drive; keep triton cache + spool on C: (already done).
- Kill+relaunch is never the right handoff (CUDA-graph recapture +
  kernel cache = tens of seconds) — sleep(1)/weight-swap always wins;
  consider pinned staging buffers for wake (vLLM hits ~24GB/s H2D;
  our L1 wake uses non-pinned `.to()` ~2-3s).
- No MPS/green-context needed while phases serialize; they matter only
  for concurrent train+infer (async off-policy — not planned).

## 6. Fix list (ordered)

1. **Canonical-master merge cycle** (§4): fp32/bf16 CPU master +
   in-place repack via `copy_`; stop `.data` rebind on quantized
   linears. Highest priority — silent drift today, graph break risk.
2. **Resident-adapter RL loop**: prefer in-place A/B update over
   per-round merge; merge only into canonical at epoch end.
3. **TIS + quant-KL metric** on the `_fq`-rollout/bf16-train boundary.
4. **KV stays fp16** on LFM3's 6 global-attn layers; disable KV-quant
   flags for `lfm3_*` presets serving at loop contexts.
5. **Verify 49KB-vs-12KiB KV discrepancy** in `PreAllocatedKVCache`
   (likely SWA layers allocated full-length — if so, ring-cap them too;
   saves ~40% KV at long context).
6. **Per-role `_cache_w`**: `_fq`+fused-GEMV for the proposer engine;
   bf16 for eval/train-side engines.
7. Sleep(1) wake path: pinned staging buffers; document wake latency.
