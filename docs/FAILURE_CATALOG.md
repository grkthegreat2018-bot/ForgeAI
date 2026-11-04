# Failure Catalog — what dead ends proved, and what survives (2026-11-04)

Each entry: what it tried → what it proved/disproved → what to salvage.

## 1. Learning-on-the-fly failures (FluxLM's ancestors)

- **Tay (Microsoft, 2016)** — learn-from-users live. Poisoned in <24h.
  Proved: unverified writes to learned state are an attack surface.
  Salvage: the gate, not the goal — writes must be admission-controlled
  (surprise/threshold) and consolidation must be verified-only. Our L1
  surprise-gate + L2 verified-replay design is exactly this fix.
- **BlenderBot 3 (Meta, 2022)** — web-grounded chatbot that learned
  from conversation. Same failure mode at LLM scale; taken offline.
  Reinforces: online learning ≠ online trusting.
- **TTT layers (Stanford)** — hidden state trained by inner-loop GD.
  Honest record: matched Mamba/Transformer at ≤1.3B with BETTER
  long-context (Mamba perplexity plateaus >16k, TTT doesn't), but
  TTT-MLP's sequential inner loop is memory-IO-bound and slow.
  ParaTTT (Oct 2026) parallelizes it 32x. Salvage: linear/fast-weight
  inner loop is the viable form at our scale — which is what GDN2
  already is. Deep-memory TTT is a future upgrade, not v1.
- **RWKV pure-RNN recall** — constant-size state can't retrieve
  arbitrary details (documented weakness). Proved our L1 architecture
  choice: bounded fuzzy state MUST be backed by an exact store
  (NGramMemory) — matches the exp_live_v2 numbers directly.
- **MemGPT/Letta** — OS-style paged memory. Heavy orchestration,
  niche adoption. Salvage: the *tier* concept (fast/slow memory
  hierarchy), not the runtime.

## 2. Distillation/imitation failures (the data-pipeline warnings)

- **"False Promise of Imitating Proprietary LLMs"** (Gudibande,
  ICLR'24) — finetuning 1.5B-13B on ChatGPT outputs: crowd raters
  fooled, but targeted evals show ~ZERO capability gap closed outside
  the imitation distribution. Mimics style, not factuality; more data
  doesn't bridge it. Proved: unverified teacher text teaches manner,
  not substance. Salvage: this is THE argument for our verified
  pipeline — teacher_logp scoring + oracle gates measure correctness,
  not style. Plain text-mimic SFT is the failure mode we're built
  to avoid.
- **Self-INSTRUCT / early Alpaca-Vicuna clones** — unverified synthetic
  instructions → fluent wrongness, benchmark inflation then collapse
  on targeted eval. Same lesson, earlier.
- **Sky-T1's <32B warning** — small models on long-CoT SFT "frequently
  generated repetitive content". Proved: reasoning-style distill
  without a loop-guard regresses generation. Our never-loops spec is
  a prerequisite for the distill plan, not an extra.
- **"Reasoning theater"** — small models produce CoT-shaped text that
  doesn't drive the answer. Verified-answer training (execution
  rewards, our evolve oracle) is the known fix.

## 3. Weight/knowledge editing failures (the WSP reality check)

- **ROME/MEMIT at scale** (ACL'24 findings) — sequential edits make the
  model progressively LESS editable, forget earlier edits, and degrade
  downstream tasks. Two-phase decay: gradual then abrupt catastrophic
  forgetting. Proved: fact injection into weights does not scale —
  you cannot store a knowledge base as edits.
- **Ripple effects** (TACL RippleEdits) — edits break logically related
  facts (change Rome's country → Colosseum answers break). And the
  killer result: a plain IN-CONTEXT baseline beat every editing method.
  Proved: memory belongs in context/state (our L1), not weights.
  Salvage: editing is still valid for *defect repair* (Astra's loop
  ablation = a defect, not a fact) and SVD task-vectors (Transformer²).

## 4. Architecture graves

- **Hyena → superseded by Mamba/DeltaNet lineage** — conv hierarchies
  lost to gated recurrences; then PURE SSM lost to hybrids on recall.
  Proved: hybrid (conv/SSM + real attention) is the surviving form —
  LFM3/Zamba2/Bamba/Flint all converged here. Salvage: our LFM3 +
  GDN2 stack is already the winner of that war.
- **KAN (Kolmogorov-Arnold)** — 2024 hype; didn't scale. Proved:
  activation cleverness doesn't beat scale+data.
- **MoE upcycling** — measured strictly worse at single-GPU scale.
  Proved: MoE must be trained native (Flint) or not at all.
- **Prefix/prompt-tuning, hypernetworks** — abandoned for LoRA/adapter
  family. Proved: param-efficient ≠ param-free; the adapter contract
  (small dense deltas, mergeable) won.
- **Sparse attention (BigBird/Longformer)** — superseded by plain
  sliding-window + occasional global (Flint's SWA-513+NoPE pattern).
- **PETALS/distributed inference** — latency killed interactivity.
  Proved: consumer-internet inference is a local problem.
- **Post-hoc ternarization** — fails; ternary must be train-aware
  (BitCPM/BitNet from scratch or QAT). ForgeQ-R exists precisely
  because int4 post-hoc loses less than ternary post-hoc.
- **Diffusion LMs** — declared dead early; revived (Mercury ~1000tps
  claims, LLaDA). Status: unproven for agentic quality, watch as a
  speed wildcard — parallel decode is a different axis than our stack.

## 5. Model/dataset flops

- **Galactica (Meta, 2022)** — science specialist pulled in 3 days:
  authoritative hallucination. Proved: specialization without
  grounding/verification produces confident wrongness.
- **OPT-175B logbook** — constant instability/crashes; Meta published
  the honest record. Salvage: checkpoint-everything + resume
  machinery (we have crash_recovery + journal patterns already).
- **Falcon-180B / BLOOM** — overtrained on RefinedWeb / multilingual
  dilution: quality below size-class. Proved: data curation > scale.
- **BabyLM/Cramming floors** — 24h/100M-token pretraining teaches
  grammar, not knowledge. Proved: the capability floor of minimal
  pretraining — our bootstrap must come via distill/port, not
  from-scratch runs.

## Meta-lessons (the pattern under all of it)

1. Unverified writes — to weights, memory, or training data — all
   fail the same way. Verification gates are the universal fix.
2. Bounded state always needs an exact-recall backup (RWKV, TTT,
   our own ngram-vs-titan numbers agree).
3. Style transfer is free; capability transfer requires verification
   (imitation paper, reasoning theater, Galactica).
4. Weights are the wrong place for facts — edits don't scale or
   ripple; context + consolidation is the working route.
5. Nothing clever survived alone — every winner is hybrid, verified,
   or merged. Single-mechanism designs die.

## 6. Own-house failures (ForgeAI, resolved — kept for the pattern)

- **The "default" tokenizer cache namespace** (distill CUDA assert):
  `_tokenize_cached` keyed the shared disk token cache under the literal
  string `"default"` — no tokenizer identity. A prior Qwen-vocab run had
  cached rendered ChatML texts as Qwen ids (up to 151645); the 128k-vocab
  LFM3 trainer was served them → `F.embedding` → `vectorized gather`
  device assert, async-misreported at whatever frame was in flight.
  Proven only by instrumenting `idx` bounds at the embed site — every
  isolated component (data, embed, loss, prefetcher, kernels) tested
  clean. **Fix**: hash = class+vocab+eos+pad at both cache layers +
  checkpoint-aware tokenizer selection + training-gated bounds check in
  `_forward_impl` (readable RuntimeError, no decode-path cost).
  Regression test: `tests/unit/test_sft_tokenizer_namespace.py`.
  Pattern: *shared caches must be namespaced by producer identity*;
  when assembled-path asserts survive clean component tests, the bug is
  in the glue — instrument the asserting op.
- **The whitespace-checkpoint illusion** (distill60 "degradation"): a
  60-step distill checkpoint generated only whitespace — looked like
  catastrophic training damage, but raw `model(ids)` logits were
  identical to base. Root cause: the bare `.safetensors` save carries no
  tokenizer, so `from_checkpoint` fell back to the canonical 65536-vocab
  tokenizer for a 128000-vocab model — every prompt mis-tokenized.
  **Fix**: `tokenizer_path` recorded in the `<ckpt>.meta.json` sidecar
  (sft_train) + `_tokenizer_from_meta` resolution in from_checkpoint —
  a saved file is now self-describing. Lesson: always discriminate
  weights-vs-wrapper with a raw-logits probe before blaming training.
- **0-dim tensors broke the pipelined loader**: `qscale` scalars saved
  by merged BitNet runs crashed `load_safetensors_pipelined`
  (`.view(torch.uint8)` requires dim>0) → silent 5x slower fallback.
  **Fix**: `reshape(-1)` before the uint8 view — load 4.9s → 0.8s.
- **Trainer/engine config split-brain**: `sft_train` mutated its own
  `cfg` (`use_chunked_ce`, `use_mtp`) while `ForgeEngine.from_checkpoint`
  builds the model on its own internal `ModelConfig` — the flags never
  arrived, so chunked CE was dead code and `[B,T,V]` logits materialized
  on every step. Combined with unconditional `sw = rw` (all-ones reward
  weights), the manual path was always taken and `teacher_logp` was
  silently ignored — `--distill` ran as plain CE. Fix: post-build flag
  sync onto `model.config` + drop uniform sample_weights.
  Pattern: *mutate the config the model actually holds, and treat
  "no-op" weights as absent — presence-based dispatch bites silently*.

## 2026-11-04 - 'OOM' load crashes on dense + ForgeQuant checkpoints were a compile hard-crash

Symptom: ForgeEngine.from_checkpoint on a 7GB merged distill save (and its packed _fq variant) died mid-load or right after activation � looked like OOM (GPU ~7.8-12.8GB at death), no Python traceback.

Root causes (two, layered):
1. _auto_activate_optimal ran OUTSIDE the OOM-guarded load try � activation extras (quant repack, KV prealloc, graph capture, warmup) could push a successfully-loaded model past free VRAM and crash with no fallback. Fixed: staged-degrade guard (drops graph/fusion/prefix extras, then quantize).
2. torch.compile (reduce-overhead) hard-crashes tracing the ForgeQuantLinear dequant path � native crash, no exception, bisected: compile alone kills generate on the packed ckpt; breakable CUDA graphs alone work. Fixed: _forgequant_prequant models now get use_compile/use_compile_autotune/use_block_fusion=False at auto-activate (same guard class as Int4TilePackedTo4dTensor); the vram_ratio>0.5 block no longer re-enables guarded-off features.

Lesson: 'OOM crash' claims need bisection � the actual OOM exception path was guarded; the killer was an inductor-generated kernel on packed weights. Diagnose by disabling features one at a time, not by reading the VRAM number.

Artifact note: 7GB dense merged saves OOM-pressure a 12GB card at activation; scripts/prequant_forge_quant.py converts to a packed _fq dir (distill250: 7.09GB -> 1.52GB) that loads in ~2-4GB with graphs still working.

