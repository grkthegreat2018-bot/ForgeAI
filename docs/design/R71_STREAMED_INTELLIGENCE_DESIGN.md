# R71 Program — "Weights as a Stream, Not a Residency"

Design doc for the R71+ breakthrough round. Goal: attack param cost, memory,
decode latency, and trainability **simultaneously** by changing the unit of
deployment from "resident weight blob" to "streamed compressed compute
substrate". All numbers below are **measured on this box** (RTX 5070, PCIe 5.0,
32GB RAM) unless marked paper/est.

Measured anchors (2026-11-09, `scripts/bench_airllm_prefetch.py`,
`scripts/bench_lossless_weights.py`, checkpoint `spark_x26_1p7b` 1.7B bf16):

| Fact | Value | Consequence |
|---|---|---|
| H2D pinned bandwidth | **53.1 GB/s** | layer streaming can be nearly free |
| H2D pageable bandwidth | 13.3 GB/s | why AirLLM is slow (see below) |
| AirLLM equivalent today | **4.4 tok/s** (pageable + sync, matches observed 1-5) | root cause = copy mode, not streaming itself |
| AirLLM from SMR disk | 0.35 tok/s | never stream from D: |
| int4 full-stream + prefetch | **65.9 tok/s** est | ~resident speed while streaming |
| int4 FFN-only stream + prefetch | **75.0 tok/s** est | compute-bound: streaming invisible |
| bf16 lossless ceiling | ideal **10.7 b/param** (H_hi=2.70, H_lo=7.97) | "ZIP for weights" caps ~1.5x |
| zlib / split-zlib on bf16 | 79.4% / **69.9%** (1.43x) | byte-plane split + ANS ≈ optimal |
| minmax int4 code entropy | **3.86/4 bits** (uniform by construction) | entropy coding needs *non-uniform* codebooks |
| minmax int4 + zlib floor | ~4.89 b/param | = ForgeQuant's measured 4.9 eff-bits |
| embedding dup/near-dup | 44 exact / 1.5% cos>0.999 | no free row-dedup win |

---

## B1 — StreamHead: async prefetched layer streaming (AirLLM overhaul)

**Problem**: `forge/engine/airllm_streamer.py` copies every layer CPU→GPU
*per token, synchronously, through pageable memory* → 4.4 tok/s. The idea
(stream only what you compute) is right; the execution is the slow part.

**Design**: rewrite the streamer around a pinned ring buffer:
- Weights live as **int4-packed (ForgeQuant format) in a pinned CPU arena**
  (~0.96GB for 1.7B vs 3.42GB bf16 — also makes more models RAM-resident).
- N pinned staging buffers; side CUDA stream issues H2D for layer i+1..i+N
  while the main stream computes layer i (double-buffer; layer order is a
  fixed cycle so prefetch is trivially perfect).
- **Partial residency**: embed/norm/attention-block params stay on GPU
  (~20% of bytes — measured mean non-attn frac 80%); only the big FFN/SSM
  matrices stream. Streaming then costs ~1.6ms/layer vs ~2.4ms full.
- Fuse dequant into the existing ForgeQuant GEMV path (already 41 tok/s on
  a bigger MoE — the GPU side exists).
- Optional: ANS-compress the int4 stream in RAM (needs non-uniform codebook,
  B2) → effective bus bandwidth ×~1.2 for streaming-bound configs.

**Expected**: 4.4 → **~65-75 tok/s** on a 1.7B (≈17x); enables 7-13B-class
models int4-streamed from 32GB RAM at usable speed. **Highest certainty x
impact in this round — do it first.**

**Files**: upgrade `forge/engine/airllm_streamer.py` (do NOT fork it —
directive E); reuse `forge/quant/paged_kv.py` pinned-buffer pattern and
`cpu_kv_offload.py` side-stream prefetch pattern (L222) as the template.

---

## B2 — ECQ-LLM: entropy-constrained vector quant + ANS ("ZIP done right")

**Problem**: lossless on bf16 caps at ~1.5x (measured ideal 10.67 b/param;
mantissa bytes are ~uniform). Lossy minmax int4 codes are *incompressible*
(3.86/4 bits — group-minmax uses all 16 levels uniformly by construction).
Both observations point the same way: **the code histogram must be
non-uniform for entropy coding to pay**.

**Design**:
1. **ECVQ packer** (offline, CPU): per-tensor Lloyd-Max codebook trained
   with entropy penalty `L = MSE + λ·H(indices)` — entropy-constrained
   vector quantization; λ sweeps the rate-distortion curve. Indices coded
   with rANS/ANS over the per-tensor histogram.
   Target: **3.2-3.6 b/param at AWQ-FP4-grade quality** (AWQ-FP4 already
   hits PPL parity at 4.53 eff-bits; giving it a non-uniform codebook +
   entropy coding trades a little quality headroom for ~25% more size cut).
   spark 1.7B: 3.42GB → **~0.75GB** (4.5x).
2. **Lossless mode** for archival: byte-plane split + order-0 ANS on hi
   bytes (H=2.70) + raw lo bytes → ~10.8-11.0 b/param ≈ **1.45-1.5x**
   guaranteed-lossless on any bf16 checkpoint.
3. Synergy with B1: stream the *compressed* codes, decode+dequant on GPU —
   weight storage AND transfer shrink by the same factor.

**Honesty note**: the 4.89 b/param measured floor already equals
ForgeQuant's 4.9 eff-bits — the win over plain int4 is ~20-30% size, not
2x. The *archival* lossless mode (~1.5x) is the other half of the "ZIP"
ask and is unconditional.

---

## B3 — LoRA-ES: gradient-free training at inference memory (new learning process)

**Problem**: no zeroth-order method exists in the codebase (confirmed).
All training paths need backward() → activations + grads + optimizer
states. On 12GB this caps full-param SFT.

**Design** — evolution-strategies fine-tuning on adapter subspace only:
- Perturb **LoRA A/B params** (r=64 → ~10-30M dims, not 1.7B) with
  antithetic Gaussian noise; score each perturbation on an eval batch
  (task CE or verifier-head P(pass) — both exist).
- 1+1-ES / NES update on the *flattened adapter vector*; keep the base
  model frozen. **Memory = inference + adapter deltas ≈ 4-6GB for 1.7B.**
- Machinery reuse: `engine_merging.evolve_merge` already implements GA
  selection/crossover/mutation over weight space; `consolidate_pairs`
  already folds trained adapters into weights with replay mixing. The
  missing piece is only the ZO scoring loop — ~150 lines.
- Novel twist (Forge-shaped): **layer-wise block-coordinate ES** —
  perturb one layer's adapter per generation (variance ∝ block dim, not
  model dim) + EF21-compressed update history (pattern exists in
  hybrid_offload.py int4 grad compression).
- Bonus: runs *during serving* on spare cycles — true background RSI.

**Expected**: slower convergence than SFT per-token, but full-param-
impossible configs become trainable on 12GB; pairs naturally with the
sft-N loop as a "can't-afford-backward" mode for larger models.

---

## B4 — FisherFold: post-hoc anti-forgetting merge (sftN collapse fix)

**Problem**: sft4-7 collapse was fixed by hard re-base (works, but
discards chained-round progress and costs a full retrain). **No
Fisher/EWC exists anywhere** in the codebase (grep-verified).

**Design**: diagonal-Fisher-weighted merge toward base:
```
F_i   = E_batch[ (∂L/∂w_i)^2 ]        # ~50-100 probe fwd+bwd, once/round
w'_i  = (F_i·w_new_i + λ·w_base_i) / (F_i + λ)
```
Params the SFT round actually *used* (high Fisher) keep their new value;
params that drifted unused snap back toward base — geometric interpolation
in importance space instead of weight space (strict superset of WiSE-FT /
model soup; soup = F_i uniform).

**Novel variant — "rescue merge"**: apply FisherFold between a *regressed*
sftN and base to extract whatever skills it did learn without its damage,
using Fisher computed on the base-eval set. Turns dead checkpoints
(sft5/6/7) into salvageable delta assets.

**Cost**: Fisher pass ≈ minutes on GPU for 1.7B; merge is CPU. Test plan:
FisherFold(sft6, base) vs 0.5-soup(sft3,sft6) vs sft8-rebase on the
240-task runnable eval.

---

## B5 — LoopThink: verifier-gated recurrent depth (reasoning per param)

**Problem**: 1.7B can't beat Qwen on params. But params only buy *width*;
reasoning needs *serial depth*. The lever is inference-time depth, and
the seeds already exist unwired: `forge/exp/loopdepth.py` (recurrent
middle-block looping, lossless LoopGates), `engine/gated.py` ForgeGate
(measured **+16.4pts acc, −34% tokens**), `model/edit_head.py`
VerifierHead, `forge/exp/decision.py` THINK/CONTINUE actions.

**Design**: wire loopdepth into `engine_generation` behind ForgeGate:
- On "hard" tokens (gate probe fires), re-run the middle block stack
  k times before emitting; KV for looped steps is discarded (loop is
  state-internal — matches existing ReplaySSM/discard-KV precedent in
  decision.py).
- Loop count k chosen by an EMA-calibrated entropy budget per step
  (conf_kv pattern) — "adaptive serial depth".
- Verifier head scores the post-loop hidden; if P(pass) < τ, loop again
  up to k_max → **test-time depth scaling without emitting tokens**
  (COCONUT-adjacent but lossless-gated, no latent-token training needed).
- Lossless integration: LoopGates init g=0 → bit-exact until trained.

**Expected**: the honest path to "reasons above its weight" — depth cost
is latency (~k× on gated tokens only), not params or VRAM.

---

## B6 — KV: promote the dormant winners; CopySpan-aware eviction

**Do not build strategy #29.** The live seam is `compact_past_kv` (only
beacon/anchor/metakv/flux reachable). Highest-value KV work this round:
1. **Promote `hisparse`** (exact GPU LRU + full DRAM history — already
   built, needs the fused kernel or a decode-path adapter) into the live
   seam → *exact* KV offload, no quality loss, bounded by DRAM not VRAM.
2. Novel eviction signal — **CopySpan-aware KV**: for the coder use-case,
   tokens the CopySpan pointer head can reach verbatim don't need live
   KV. Score = copy-head reachability × beacon score; est. 20-40%
   additional reclaim on edit/REF-heavy sessions.

---

## B7 — NTP overhaul: commit-ahead block decode

Existing: mtp_selfspec (5.37x claim, needs ckpt heads), CopySpan splices
verbatim spans (untrained head), uno Psi-Spec, peagle.

**Next increment**: **entropy-tiered commit** — merge verifier head +
MTP + CopySpan into one decode: the trunk runs once per block; inside the
block, tokens are sourced from the cheapest sufficient head:
`CopySpan (verbatim, 0 logits) → MTP head (1 head fwd) → full trunk
verify (only if verifier P(pass) < τ)`. Unlike spec-decode's bit-exact
verify, the verifier head is a *learned quality gate* — tunable
lossy/lossless via τ. For code (CopySpan's home turf) est. 3-8x on
edit-heavy sessions; neutral (~1x) on novel prose.

"Diffusion for boilerplate, AR for reasoning" is already the CopySpan
thesis — `forge/exp/diffdraft.py` is the embedding-space proposer seed if
we later want continuous-fill instead of copy-fill.

---

## B8 — Params: the composite answer

No single trick removes the param cost; the stack does:
- FFN factorization exists (NLRQ 12.8-16x, Monarch ~25x) — extend NLRQ
  coverage to attention projections in V15 key.
- ECQ (B2) → ~3.4 b/param storage.
- StreamHead (B1) → residency optional; params become a streamed asset.
- Net effect: 1.7B model = ~0.7GB streamed substrate + ~1-2GB resident
  hot set, or a ~10-13B-equivalent in the same 12GB envelope.

---

## Recommended order

1. **B1 StreamHead** — measured 17x, pure engineering, unblocks
   "memory cost" + "airllm" + partially "params" in one shot.
2. **B5 LoopThink** — reasoning lever; all seeds exist, needs the
   engine_generation wiring + gate calibration.
3. **B4 FisherFold** — cheap (~hours), directly attacks the sftN
   collapse the project just lived through; also enables sftN salvage.
4. **B2 ECQ packer** — offline CPU tool; pairs with B1 for full effect.
5. **B3 LoRA-ES** — new learning modality; medium effort, high novelty.
6. **B6/B7** — incremental KV + NTP work after the above lands.

Each lane gets: minimal test script first (done for B1/B2 anchors),
numbers vs the RIGHT incumbent baseline (s4r/standard for KV, ForgeQuant
4.9 for compression, muon_sf for training, standard decode for NTP),
and a `decode_live`-verified wiring path — no new dormant flags.

---

## Status after first session (2026-11-09)

**B2 ECQ codec — SHIPPED** (`forge/quant/ecq_codec.py`,
`scripts/bench_ecq_codec.py`, `tests/unit/test_ecq_codec.py` 9/9 green):
- Lossless bf16: byte-plane split + order-0 range coder → **10.73 b/param
  actual** on spark weights (ideal 10.67 — the coder is ~99.4% efficient;
  1.49x guaranteed-lossless on any bf16 ckpt)
- ECQ-16 λ=0: **3.84 b/param** vs minmax-int4 ~4.9 (22% smaller); λ dial
  reaches 2.57 b/param. Lloyd-Max codebook -> non-uniform index histogram
  -> range coding actually pays (minmax codes are uniform = incompressible)
- Rate-matched: ECQ-32 24.7dB@4.93 > g32-int4 22.1dB@4.9 (+2.6dB same bits)
- Next: ANS/rANS for speed (pure-Python range coder ~1-3M sym/s — offline
  OK), per-tensor quality check on eval, engine load path.

**B1 StreamHead — SHIPPED** (`forge/engine/airllm_streamer.py` rewrite,
`tests/unit/test_airllm_streamer.py` 6/6 green):
- Root cause found: old path copied every layer CPU→GPU per token through
  **pageable** memory (13.3 GB/s) AND copied everything back GPU→CPU per
  token + zero prefetch → ~4.4 tok/s model, **3.3-3.4 tok/s measured**.
- New: pinned shard cache + 3-deep ring of suffix-keyed GPU slots + side
  CUDA stream prefetching layer i+3 during compute of layer i + optional
  g32-int4 shards (FORGE_AIRLLM_INT4) with fused GPU dequant.
- **Measured 30-36 tok/s on spark-scale synthetic (28×50MB) = ~10x**, under
  live GPU contention. bf16 > int4 on this box (53 GB/s bus makes the
  dequant not pay yet — documented; Triton dequant is the phase-2 fix).
- VRAM bound unit-tested: GPU holds ~3 slots of weights, not the model.

**B4 FisherFold — SHIPPED core** (`research/merge_models.py`:
`merge_fisher` gate/interp modes + `fisher_diag`;
`scripts/fisher_merge.py` driver; `tests/unit/test_fisher_fold.py` 8/8):
- w' = w_b + Δ·F_n/(F_b+F_n+eps): new-task-critical deltas kept, unused
  drift + base-critical params revert to base. Rescue merge for sft5/6/7.
- Next: real eval — FisherFold(sft6, base) vs soup vs sft8-rebase on the
  240-task runnable eval (needs a free GPU window).

**Deferred**: B3 LoRA-ES, B5 LoopThink, B6 KV-promote, B7 commit-ahead.
**sft9 note**: selfgen resumed and finished slice-E (~+6 kept, ~43 total).

**B3 LoRA-ES + B5 LoopThink � SHIPPED same session**:
- B3: `forge/training/lora_es.py` (antithetic ZO ES, full/block/fisher
  perturb modes, forward-only => trains at inference memory),
  `scripts/lora_es_train.py`, `tests/unit/test_lora_es.py` 10/10.
- B5: `VerifierGate` + `looped_prefill_verified` in
  `forge/exp/loopdepth.py`, `scripts/bench_loopthink.py`,
  `tests/unit/test_loopthink_gate.py` 6/6. MEASURED: ungated recurrence
  diverges (6.58 vs 1.64); 60-step-trained LoopGates (12 params) +
  verifier early-stop => **1.529 vs 1.644 = -6.9% loss at +25% latency**
  on spark_x26_1p7b. Depth-at-inference confirmed on a frozen model.
