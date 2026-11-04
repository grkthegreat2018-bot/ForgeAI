# FluxLM v2 — Design (evidence-driven, 2026-11-04)

## Gold spec (user target) — honest physics per line

- >1000 tok/s single-gen: needs ≤~0.6GB weights/pass (5070 bandwidth)
  → ~1.2-1.4B int4 or ~2-2.5B ternary. LFM3-1.7B int4 caps ~400-500.
- ≤1GB total: CUDA ctx ~0.3G + weights ≈0.55-0.65G → ~1.1-1.3B int4
  or ~2.5B ternary + small KV. → **ternary ~2-2.5B dense is the
  design point** (BitNet-class already production here; BitCPM precedent).
- ms ingest + SFT-quality: split — L1 writes are ms-scale (measured);
  SFT-quality is L2 consolidation. Never one op.
- Qwen2.5-7B coding/agentic: in-domain only (parameter physics); gate =
  match 7B's valid-rate on the CC/evolve suite.
- rival-all-local: vs 7-8B resident → in-domain rival (R1-distill
  pattern: 1.5B hit MATH-500 83.9 via verified long-CoT SFT);
  vs 20-30B offloaded MoE (35-50tps on 12GB) → win on speed (5-10x)
  + domain, lose breadth permanently; vs everything → own outright
  the axes nobody else has (live-learn, never-loops, self-correct,
  ≤1GB, ~1000tps).
- self-correcting: verifier-in-loop decode — the real small-model
  equalizer (R1-Distill-1.5B AIME cons@64 52.7 vs pass@1 28.9).
- never loops: enforced not hoped — direction-ablation (Astra-style)
  + deterministic repetition-guard decode hook.
- dense→MoE upcycle: DOCUMENTED TRAP at single-GPU scale (moe-upcycle
  measured bigger/slower/worse than base; pays at ~40-50% original
  pretraining budget). MoE is future expert-offload for capacity,
  not a conversion.

## Indie-model evidence (Recursive Flint, Oct 2026)

40B-total/2.5B-active MoE built from scratch in <4mo by a small team +
agents ("Eureka Machine" automated research loop — ForgeEvolve's thesis,
vindicated at shipped-model scale). Leads <50B MoEs on agentic coding,
rivals dense 25-40B at >5x speed.

Stealable, in order of directness:
1. **Quant self-distill recovery**: they quantize lm_head to 4-bit too,
   then recover via offline self-distillation — quantized student trains
   on its own sampled responses (self-discovered coreset) with the bf16
   model as teacher. ~half QAT steps, within 0.7pt of bf16. Our
   score_teacher_logprobs.py + chunked_distill_ce IS this pipeline —
   add a self-distill recovery stage after ForgeQuant packing,
   complementing (or replacing) ForgeQ-R residuals.
2. **Native-MoE template** (future V15+ if MoE enters): 256 fine-grained
   experts x512-dim, top-6 sigmoid routing, ungated shared expert,
   attention sinks, QK-norm, SWA-513 + occasional NoPE full layers.
3. **TAP (tekkmaven/flint-1.2B)**: think/tool_call as the BASE LM
   objective from step 0 — agentic format is tokenizer+data, not a
   post-training template. Wide-shallow geometry (22L x 5504 FFN;
   1.5-Pints: wider MLP stores more knowledge/param at small scale).
   Pretrained on free Kaggle TPU v5e-8, 40h/~15B tokens — bootstrap
   compute does not require owned GPUs.

## Indie sweep #2 — the playbook the evidence actually supports

Standing: LFM3-1.7B Q4 is already the <=4B LiveCodeBench Pareto leader
(73.46 @ 1.1GB). The backbone is validated; the work is specialization.

Proven recipes (all indie/community scale):
- **Trajectory distill** — the small-agentic recipe: mini-coder-4b
  (400k mini-swe-agent trajectories + SWE-smith -> beats gpt-oss-120B
  on SWE-Verified; trains on ONE 80GB GPU); OpenThinker3-1.2M (QwQ-32B
  traces, SFT-only -> +15-20pts over R1-Distill-7B); xLAM-2-1b/3b +
  Katanemo Arch-1.5B -> GPT-4-class function calling at 1-3B.
  Our ForgeEvolve export + score_teacher_logprobs + chunked_distill_ce
  IS this pipeline; gap = trajectory corpus scale, not method.
- **Test-time compute is free capability**: s1K budget-forcing ("Wait"
  injection, 26min SFT), cons@64>>pass@1, verifier loops. Costs decode
  tokens, not params — the small-model equalizer.
- **Cheap-RL warning**: Sky-T1 found <32B models loop on long-CoT —
  the never-loops spec (direction ablation + decode rep-guard) is a
  prerequisite, not a nicety.
- **Architecture ingredients (small-team proven)**: Zamba2 shared
  attention x2 ABAB + LoRA on shared MLPs (param reuse); RWKV-7 =
  genuine test-time-training RNN, 1.6M-param state at L24-D1024 —
  bounded fast-weight memory is a shipped product; nanochat value
  embeddings (zero-FLOP params), ReLU^2, post-RoPE QK-norm, softcap;
  Transformer2 SVD task-vectors + router = self-adaptive experts
  (same SVD math as ForgeQ-R — residual and task-adapter are one
  mechanism family).
- **Compute ceiling is lifted**: Kaggle TPU v5e-8 free tier, DiLoCo/
  DisTrO community distributed (INTELLECT-1 10Bx1T over 14 nodes).
  A bootstrap-scale run is a coordination problem, not a purchase.
- SmolTalk recipe reference for SFT mix (Magpie-Ultra + constraint/
  rewrite subsets, ablated at 1.7B).

## AirLLM+MoE (user's old idea — vindicated, now a standard lane)

Layer/block weight-streaming + MoE = modern expert-offload inference.
AirLLM itself survived (v3+ hook-based, transformers-driven; does MoE,
671B on ~12GB, streamed-LoRA training <6GB). The proven refinements:
- MoE-Infinity: hot-expert cache + activation-trace prefetch at bs=1
  (3-17x over naive offload) — expert usage skews hard single-user.
- Fiddler: cold-expert inversion — send the ~KB activation to CPU,
  compute there, return — never stream cold weights.
- kTransformers: arithmetic-intensity split — attention + shared
  expert pinned on GPU, routed experts offload.
ForgeAI-specific advantages: domain-specialist routing concentrates
(cache hit rate should beat published), ternary experts = 0.2B/param
(200MB/token ~ 250tps ceiling even uncached), engine already has
prefetch I/O (load_safetensors_pipelined) + CPU/GPU split residency.
Spec reframe: "<=1GB" can mean <=1GB VRAM — ~8-12B-total ternary MoE,
~1.5B resident, experts in RAM. Capacity lane — BELOW backbone distill
on the critical path, but if V15 goes MoE it should be streaming-native
by design (separately-mappable packed experts, domain-clustered layout,
router-lookahead prefetch, Fiddler CPU-exec for cold experts).

## What FluxLM v1 was (and why it died)

`forge/model/flux/` — sparse associative-memory LM: hedged mixture of
unbounded n-gram cell tables + domain lanes + a teacher channel.
- Memory blew up: dict-of-dicts cell tables grow without bound on stream.
- No generalization: exact-suffix stores memorize surface continuations.
- Degeneration: mixture collapsed toward a near-bigram model (the
  teacher_channel floor exists because unfloored it decayed to ~0.001).

`forge/model/live.py` (LiveSession) is the *other*, better-directed
attempt that already exists: frozen substrate + journaled live surfaces.
v2 = finish that one, do not resurrect `flux/`.

## Measured (scripts/exp_live_v2.py on RTX 5070, CUDA, fp32)

Micro substrate (d128 L4, V=256, 8 fact pairs, 5-rep ingest). Columns:
verbatim gain (NLL drop on trained facts), post-512-noise gain,
overwrite CE on new values vs stale, alt-context CE (facts probed via
unseen separator), unseen-key CE (false memory check; base=log256=5.55).

| config | verb | dist | ow | stale | alt | unse | ing tps | vram |
|---|---|---|---|---|---|---|---|---|
| titan+gdn2+ngram | 5.03 | 3.88 | 0.89 | 1.99 | 5.56 | 5.64 | ~700 | 0.02G |
| titan+gdn2       | 2.02 | 0.43 | 4.13 | 5.43 | 5.58 | 5.64 | ~2800 | 0.04G |
| ngram only       | 3.67 | 3.50 | 2.35 | 2.36 | 5.66 | 5.51 | ~3000 | 0.05G |
| titan only (gd)  | 0.42 | 0.58 | 5.28 | 5.38 | 5.30 | 5.44 | ~6800 | 0.05G |
| d384 L6 (full)   | 4.92 | 4.06 | 0.88 | 1.96 | 5.99 | 6.10 | ~450  | 0.43G |

Findings:
- **NGramMemory carries recall.** Without it the verbatim gain halves and
  distractor wipes it out. TitanMemory alone (gd solver) learns almost
  nothing at 5 reps; `solve` helps but doesn't substitute.
- **Zero false memory anywhere** — unseen keys stay at base CE. The
  zero-init/zero-poison property holds; substrate is never corrupted.
  (This is the "vomit-impossible" guarantee.)
- **Overwrite is half-broken**: full config recalls new values (0.89) but
  stale co-fires (1.99); ngram-only stores old and new equally (2.35/2.36)
  — no recency precedence. Needs decay or explicit replace semantics.
- **NO surface generalizes**: alt-context CE ≈ base CE on every config.
  Ingested facts are verbatim-bound. This is the v1 disease persisting in
  miniature — live surfaces *memorize*; generalization is a different
  mechanism's job (L2), not a tuning problem (L1).
- **Cheap**: <0.5GB VRAM, hundreds-to-thousands of ingest tps even at
  d384/L6. Headroom is not the constraint.

## Architecture (three tiers, all bounded)

- **L0 context** — KV cache of the frozen backbone. Free.
- **L1 fast-write memory** (exists — `live.py`):
  - `NGramMemory`/`SuffixMemory` sidecar: exact recall, durable under
    distractor. FIX: add recency/decay or replace-on-write so
    overwritten keys stop co-firing (measured 2.35≈2.36).
  - `TitanMemory` fast weights: fuzzy partial-match, complement to exact
    store. Needs bootstrap to matter — at 5-rep cold-start it barely
    learns (gd 0.42 / solve ~2.0 vs ngram 3.7). Keep, but it is NOT the
    recall workhorse.
  - `GDN2` recurrent state: working memory across chunks.
  - Journaled/revertible (verified: `revert_exact`), gates default-closed.
- **L2 consolidation** (the missing piece): short LoRA/adapter fit over
  L1 traces + replay buffer, run idle/on-demand — the only tier that can
  produce *generalized* knowledge. This is where alt-context transfer
  lives. Reuses sft_train distillation machinery (reward weights,
  chunked CE) — no new training tech.

Backbone is read-only through all of it: memory reads modulate via
zero-init gates; writes never touch substrate params (measured clean —
unseen 5.5-6.1 ≈ base).

## Build order (each step independently shippable, kill gates explicit)

1. **NGramMemory overwrite fix** — DONE (EMA successor rows + row_mass;
   raw decay dilutes lift vs smoothing floor). ow 2.35→1.69 ngram-only,
   0.89→0.52 full; distractor durability 2.92→4.23. The stale>3.0 gate
   was mis-specified: different-suffix = different-fact under exact
   semantics, so old values legitimately persist under their original
   context keys. SuffixMemory (production R65) got the same fix.
2. **L1-on-real-backbone** — DONE (scripts/exp_l1_real.py, LFM3-CC int4):
   SuffixMemory sidecar works on a real model — recall CE 5.10→2.45,
   hit 0.25→0.50, distractor-immune, unseen-safe, 0.12ms/step (~0.3%).
   Fixed en route: inverted lift sign (now LLR vs prior), order-weight
   rank-vs-value bug, sentence-level ingest rule for BPE alignment.
   Next: LiveSession surfaces (titan/gdn2) on the real ckpt for fuzzy.
3. **L2 consolidation loop** — DONE (scripts/exp_l2_consolidate.py):
   full ingest→memory→reconstruct→LoRA(r8)→retention loop on the real
   1.7B. Operating point: lr 5e-4 + replay buffer → fact CE 7.22→5.42,
   drift +0.31 (vs +3.86 catastrophic forgetting at lr 3e-3 without
   replay — Sleeping-LLM failure mode reproduced and cured).
4. **TitanMemory bootstrap value-check** — DONE, dropped: fuzzy gain
   plateaus ~2.0n at reps 5/50/200 vs ngram ~4.3; gdn2-only = 0.00.
   Exact-suffix EMA memory is the L1 mechanism (recorded dead end).

Hard constraints (user): everything runs on RTX 5070 12GB; no step may
require >1 GPU-hour; generation speed must stay in the fused-quant
class (L1 adds one small matmul/layer — measured negligible).

## Explicitly not building

- Per-token gradient updates into backbone weights (the FluxLM v1
  memory blowup — backprop-per-token is ~3x forward cost).
- Append-only/unbounded memory stores (v1 again — everything bounded).
- Making L1 do L2's job (verbatim-bound memory cannot generalize;
  measured, not theorized).
