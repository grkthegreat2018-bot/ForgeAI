# RSI-Loop Research: Base Model Selection + LoRA-Family Methods for Non-SFT Training

Compiled 2026-11-23. Scope: (1) LFM3 vs Spark-X2.6 vs Granite as the RSI-loop
proposer/solver base; (2) QLoRA/rsLoRA/DoRA and adjacent adapter methods for
the loop's non-SFT update path (GRPO/self-generated data/verifier rewards).

Sources: codebase inventory + web research (cited URLs inline). All three
candidate bases are ALREADY ported into ForgeAI presets — this doc is a
decision aid, not a port proposal.

---

## 0. Current state (codebase ground truth)

| Base | Preset | Checkpoint dir | Status |
|---|---|---|---|
| LFM3-1.7B (LiquidAI, `mondk` CC fine-tune) | `lfm3_1p7b` (`forge/config.py:1218`) | `research/checkpoints/lfm3_1p7b_cc` (+`_fq`, `distill_v5–v10`) | Ported, bit-exact. Default CodeEvolve proposer fallback. |
| LFM3 + D1 grafts | `lfm3_1p7b_d1` (`config.py:1399`) | same ckpt | GDN2 side-paths + ValueResidual/LearnedSink/AttnRes + CopySpan/Verifier heads, gate=0 bit-exact. |
| Spark-X2.6-1.7B (XHToken) | `spark_x26_1p7b` (`config.py:1287`) | `spark_x26_1p7b` + `sft2–sft10` lineage | Longest SFT history; **sft10 showed no gain → 1.7B ceiling logged**. |
| Parable-Granite-4.1-3B (AnkitAI distill on ibm-granite/granite-4.1-3b) | `parable_granite41_3b` (`config.py:1354`) | `parable_granite41_3b`, `granite41_3b_base` | Ported, bit-exact vs HF (scratchpad:3724). Solver candidate. |
| Qwen2.5-Coder-1.5B/3B | `qwen25_coder_1p5b` (:949), `qwen25_coder_3b_mla_moe` (:1003) | both present | 3B is default proposer when cached (code_evolve.py:683-709). |
| ForgeLM V14-RSI (custom ~2.5B) | `forgelm_v14_rsi` (:849) | — | Design-complete; P0 heads verified. Long-run goal. |

Adapter machinery that ALREADY EXISTS (don't rebuild):
- `forge/training/bitnet_lora.py` — manual LoRA (α/r), fused-QKV aware,
  `merge_lora_adapters` (incl. NF4 dequant→merge→requant), `lora_disabled`
  ctx = free GRPO ref policy.
- R38 variants (implemented, test-covered, NOT CLI-wired into sft_train):
  `forge/training/dora.py` (DoRA), `dlora.py` (dynamic rank),
  `adapter_variants.py` (PiSSA, AdaLoRA, **rsLoRA**, MiCA minor-subspace),
  `forge_adapter.py` (entropy-guided multi-rank).
- `forge/training/lora_es.py` — zeroth-order ES on low-rank bypass (init
  modes incl. deterministic PiSSA). **Negative result logged**: halved
  pass@1 on a ZO-CE objective (scratchpad:2721).
- `forge/engine/quant/littlebit2.py` — LittleBit-2 QLoRA base
  (`--qlora-lb2`); NF4Linear QLoRA in bitnet_lora.
- `forge/training/sift.py` — weight-slice training (non-adapter), mutually
  exclusive with LoRA.
- `engine_lora.py` + `lora_registry.py` + `lora_router.py` — single-adapter
  residency, hot-swap, tagged-LoRA registry, aux-MLP tag router. Proven
  hone→hot-swap path (`code_evolve.py:971-1022`).

---

## 1. Base model decision: LFM3 over Spark/Granite — VERDICT: YES, with caveats

### LFM3-1.7B (released 2026-10-27, LiquidAI)

- Arch: 24L tri-hybrid — 13 gated short-conv (L=7) + 5 SWA(W=256,θ10K) +
  6 full attn (θ1M, partial-RoPE 0.5). GQA 16Q/4KV hd128, QK-norm, per-head
  sigmoid out-gate, sandwich norms, tied 128K vocab. 22T pretrain tokens.
- KV ≈ **12 KiB/token** (convs ~free, SWA capped) → ~1.5GiB @128K vs
  ~1.9GB Spark / ~3.6GB Qwen. Best decode economics of the three.
- License: LFM Open v1.0 (Apache-based, free commercial <$10M/yr). OK.
- Vendor numbers (self-reported): LCBv6 73.46, SWE-bench Verified 43.40
  (OpenCode harness), BFCLv4 71.39. **Card explicitly disclaims agentic
  coding at this size** — resolve with our own evolve-loop eval.
- Tool use: native `<tool_call>` XML via chat template.
- Ecosystem: official TRL LoRA path (`target_modules="all-linear"`);
  conv-block LoRA surface = `in_proj`/`out_proj`/`w1–w3` pointwise Linears
  (ordinary GEMMs — no fused-kernel bypass). Unsloth FT unsupported yet;
  native transformers `Lfm3Config` mid-transition (remote code today).
- Already ahead in-repo: D1 graft round designed, bit-exact-verified;
  `lfm3_1p7b_cc` itself is a self-RL fine-tune (mondk R3 = model generates
  its own QA) — philosophically aligned with the RSI loop.
- Sources: huggingface.co/LiquidAI/LFM3-1.7B, liquid.ai/blog/lfm3,
  docs.liquid.ai/lfm/fine-tuning/trl.

### Spark-X2.6-1.7B (XHToken, Apache 2.0)

- NOT a conv/SSM hybrid — pure transformer, "hybrid" = 21 SWA + 7 full
  attn only. KV still grows on 7 layers; head_dim 256, GeGLU, fused qkv.
- Vendor claims SWE-V 59.8 but at **thinking=high** (token-expensive
  proposals) with in-house benches — unverified at thinking=low/none.
- Remote-code only; vendor llama.cpp fork; weakest ecosystem of the three.
- In-repo verdict already emerging: sft3→sft10 chain plateaued
  ("1.7B ceiling"). Keep as legacy/reference, not the RSI default.
- Sources: huggingface.co/XHToken/Spark-X2.6-1.7B, ai.satyajit.si analysis.

### IBM Granite (two different animals)

- **granite-4.0-h family** (h-micro 3B / h-1b 1.5B): 36 Mamba-2 + 4 NoPE
  attn, muP scalars, Apache 2.0, native transformers (`granitemoehybrid`).
  Strong BASE-model code numbers (h-micro-base: HE 70.7/73.7, MBPP 74.9 —
  a good fine-tuning substrate signal) BUT: heaviest port (selective-scan
  op) and **documented LoRA trap** — standard `all-linear`/`target_modules`
  silently misses mamba internals (PEFT #2274/#2556/#2562; fused kernel
  bypasses `out_proj`). High port + adapter-targeting risk.
- **granite-4.1-3b** (dense, 40L llama-family, multipliers
  embed×12/residual×0.22/logit÷10, vocab 100352, Apache 2.0): already
  ported bit-exact via `parable_granite41_3b`. Easy LoRA (ordinary
  arch, zero fused-mixer bypass), cleanest license. Trade-off vs LFM3:
  full-attn KV growth (40 layers) and ~6GB bf16.
- Sources: huggingface.co/ibm-granite/granite-4.0-h-*,
  granite-4.1-language-models, HarmoniqOS/ssm-aware-lora-finetuning.

### Decision

| Role | Pick | Why |
|---|---|---|
| **RSI proposer/solver default** | `lfm3_1p7b_cc` | Best decode economics (KV ~free on 18/24 layers), official LoRA path with clean Linear targets, already ported + D1-grafted + quantized (`_fq`), self-RL lineage. |
| **Quality arm / A-B challenger** | `parable_granite41_3b` | Dense 3B headroom over the 1.7B plateau, trivially LoRA-able, Apache 2.0, ported. Slower + more VRAM — batch-4 evolve may need `_fq`. |
| Legacy | `spark_x26_1p7b` | Keep sft9 ckpt for regression A/B; no new spend. |
| Skip | granite-4.0-h-* | Mamba-2 port cost + silent-LoRA-miss traps; revisit only if a must-have capability appears. |

Watch items: (a) verify LFM3-cc on OUR evolve harness before retirement of
spark — vendor disclaims agentic coding; (b) LFM3 transformers-native load
is mid-transition — keep remote-code files vendored in the ckpt dir;
(c) `LFM3-D1` name collision: LiquidAI's `d1-*` = decision models on VL
backbone, unrelated to our `lfm3_1p7b_d1` graft preset.

---

## 2. LoRA-family methods for the non-SFT loop

Evidence split is important: nearly all LoRA-variant benchmarks are SFT.
RL/iterative evidence is called out; UNVERIFIED = no published RL result.

### 2a. Direct RL evidence (the load-bearing findings)

- **Thinking Machines "LoRA Without Regret"** (Schulman, Sep 2025):
  LoRA == full-FT on RL **even at rank 1** (RL info content ~O(1)
  bits/episode). Target ALL linear layers incl. MLP/MoE — attention-only
  underperforms at matched params. LoRA LR ≈ **10× full-FT LR**. LoRA
  degrades more at large batch — keep per-step batches moderate.
  Independently reproduced on Qwen3-1.7B GRPO (michaelbzhu/lora-without-regret).
- **MERL TR2026-092** (Jun 2026): policy-divergence bound ∝ rank under
  GRPO → **low rank is itself a KL stabilizer**; KL-free LoRA GRPO
  preserved eval accuracy on 1–3B while skipping ref-model evals. Matches
  our `ref_model=None` + `lora_disabled` design — already correct.
- **Token-Efficient RL** (arXiv 2504.20834): full-token GRPO under LoRA
  **failed to improve** Qwen2-1.5B; selective-token losses (S-GRPO) worked
  (46→70% SVAMP). Loss-mask sparsity = implicit regularizer at low rank.
- **Adaptive rank fails under GRPO** (arXiv 2605.07366): gradient-importance
  rank reallocation ~4.5 pts WORSE than uniform rank → kills AdaLoRA/DLoRA
  for the RL path (both are implemented but should stay SFT-only).
- **Forgetting**: *LoRA Learns Less and Forgets Less* (arXiv 2405.09673,
  ICLR'25) — LoRA forgets less than full-FT on code+math; r≥64 closes the
  code-quality gap. *Illusion of Equivalence* (arXiv 2410.21228) — LoRA
  adds "intruder dimensions" correlated with forgetting; higher α + lower
  rank → more intruders.

### 2b. Per-method verdicts

| Method | Mechanism | RSI verdict |
|---|---|---|
| **Vanilla LoRA** | W + BA, α/r | **DEFAULT**: r=16–32, all-linear incl. conv `in/out_proj`+`w1–3`, LR≈10×FT. Mergeable, hot-swappable, already canonical (`--manual-lora`). |
| **rsLoRA** | α/√r scaling | Free flag; implemented (`adapter_variants.py`). Enable when r>32 (code-gap ranks). No RL study — mechanism is scaling-only, low risk. |
| **QLoRA (NF4/LB2 base)** | 4-bit frozen base + bf16 adapters | arXiv 2511.15694: QLoRA ≈ good RL config (better than QA-RL or PTQ-then-full-RL). **But merge→retrain into NF4 compounds quant error** — prefer bf16 base at 1.7–3B/12GB; use `--qlora-lb2` only when VRAM-forced; never merge-cycle into a quantized canonical base (keep fp32 canonical). |
| **DoRA** | magnitude×direction decompose | Implemented (`dora.py`). +2–4 pts SFT; mergeable but unmerged serving ~139% slower (PEFT docs). **UPDATE (arXiv 2512.23165 RLVR bake-off): DoRA/AdaLoRA/MiSS consistently beat LoRA under RL — DoRA even beats full-FT.** Leading candidate for the GRPO path; merge-always, never serve unmerged. |
| **PiSSA** | top-r SVD init of A,B | Implemented + deterministic variant in `lora_es.py`. Faster convergence for ~100-step SFT hones. **UPDATE: SVD-inits (PiSSA/MiLoRA) COLLAPSE under RLVR (arXiv 2512.23165) — keep off the GRPO path.** |
| **MiCA / MiLoRA** | minor-singular-direction init | Implemented (MiCA). Anti-forgetting bias — right direction for iterative SFT loops. **Same RLVR-collapse caveat as PiSSA (SVD-family) — SFT-path only.** |
| **LoRA+** | η_B≈16·η_A | NOT implemented. ~1–2% + ~2× convergence; optimizer-level, trivial (~10 lines in param groups). Include in LR sweep. |
| **LoRA-GA / OLoRA** | grad-aligned / ortho init | Not implemented; small gains, cheap. Optional sweep arm. |
| **AdaLoRA / DLoRA** | adaptive/dynamic rank | **AVOID for RL** (2605.07366). DLoRA implemented — restrict to SFT stages if used at all. |
| **VeRA** | shared frozen A,B + per-layer vectors | Not implemented. ~KB-size adapters — interesting for a large adapter *library*, quality risk on hard tasks. Experiment tier. |
| **MoRA** | high-rank square update | Beats LoRA on memorization/continued-pretrain — candidate if the failure mode is "adapter can't absorb new knowledge" (relevant to ingest). Medium cost. |
| **(IA)³ / BitFit / Houlsby / prefix / ReFT** | activation/bias/soft-prompt | Skip as main adapter: too little capacity (RL) or unmergeable (Houlsby/ReFT needs hooks — wrong substrate for hot-swap weight evolution). |
| **LoRA-ES / EGGROLL / MeZO / ES-at-scale** | zeroth-order on adapter weights | `lora_es.py` exists — negative result was ZO-**CE objective** misaligned, not ES per se. EGGROLL (2511.16652): fresh rank-r perturbations, unbounded cumulative rank, GRPO-competitive, less reward-hacking → **retry ES on verifier/pass@1 reward, not CE**. |

### 2c. Hybrid-arch targeting (LFM3/Granite-h specifically)

- LoRA the **projections into/out of conv/SSM blocks**, never conv kernels
  or SSM internals: SSM-PEFT study (PMLR galim25a) — LoRA on SSM-internal
  modules fails; projection-LoRA + sparse-dim tuning wins. MambaPEFT
  (2411.03855): in/x/dt/out_proj all LoRA-able.
- LFM-family LoRA surface: attn `q/k/v/o_proj` + `gate_proj`, conv
  `in_proj`/`out_proj`, MLP `w1/w2/w3` (Liquid cookbook + community LFM2
  recipes). All are ordinary Linears in our port → `_resolve_lora_targets`
  must be extended to cover conv-block names, and **verify the regex
  actually hits** — PEFT silently no-ops on misses (Granite-h case:
  standard targets touched only 4/40 layers).
- TM all-linear finding ⇒ attention-only LoRA is wrong for RL; include
  conv/MLP projections.

### 2d. Iterative/lifelong strategy (merge→retrain vs adapter library)

- Merging LoRA into W is arithmetically lossless (fp rounding aside);
  long-cycle (>20 merge) degradation is UNVERIFIED. Nearest evidence
  (arXiv 2607.11020): sequential-write retention collapsed to 1% on narrow
  data, held at 46% with diverse data → **data diversity beats merge
  mechanics** — keep a replay/diversity buffer in every round's SFT set
  (our `consolidate_pairs` replay path already does this).
- Keep per-round adapters versioned in `lora_registry` (done); periodic
  TIES/DARE merge of adapters (PEFT `add_weighted_adapter` / mergekit —
  `research/merge_models.py` has merge tooling); gate merges on held-out
  eval. LoraHub-style composition is the non-destructive alternative.
- Keep canonical base in fp32; materialize bf16/`_fq` copies for training.

### 2e. Recommended stack (evidence × cost)

| Tier | Choice |
|---|---|
| 1 — default | Vanilla manual-LoRA, all-linear targets **extended to conv in/out_proj + w1–3**, r=16–32 (**r≤32–64 hard ceiling under GRPO: r=128 collapsed 73%→4.7%**, arXiv 2611.27148), `use_rslora` when r>32 (SFT only), LR≈10× FT LR, moderate batch. |
| 2 — cheap add-ons | Wire `rsLoRA`/`PiSSA`/`MiCA` (already implemented) behind `--lora-init`/`--lora-variant` flags in sft_train — **SFT-path only; SVD-inits banned from GRPO (RLVR collapse)**; add LoRA+ (λ≈16) param-group LR (robust under RL per same bake-off). |
| 3 — RL-path adapter | **DoRA for GRPO rounds** (beats LoRA+FFT under RLVR), vanilla-LoRA for SFT hones; merge-always, never serve unmerged. |
| 4 — loop hygiene | Replay buffer mandatory; uniform rank; drift metric per round (e.g. `_functional_fingerprint` ELO — already queued); adapter versioning + eval-gated TIES merge. |
| 5 — research arm | EGGROLL-style ES on verifier reward (replaces the failed ZO-CE lora_es objective); VeRA adapter library; MoRA for knowledge-ingest rounds. |
| Avoid | AdaLoRA/DLoRA under GRPO; attention-only targets; r>256; unmerged DoRA serving; merge-cycling into a quantized base. |

### 2f. Concrete wiring gaps (vs today)

1. `_resolve_lora_targets` (sft_train.py:118-147) matches fused
   `qkv_proj`/gate + `w_gate/w_up/w_down` — extend for LFM3 conv-block
   projection names; add a hit-count assert (target>0 matched modules).
2. R38 variants (rsLoRA/PiSSA/DoRA/MiCA) are modules+tests only — expose
   via `--lora-variant {vanilla,rslora,dora}` + `--lora-init {zero,pissa,mica}`.
3. GRPO path already correct on the two big RL findings (all-linear is the
   gap — it currently inherits whatever `add_lora_adapters` targets; check
   it covers MLP; ref-free via `lora_disabled`; low rank). Add rank-sweep
   + token-selective-loss arm (S-GRPO) per 2504.20834.
4. LoRA-ES: keep module, retarget reward to pass@1/verifier score
   (EGGROLL-style fresh-rank-r perturbations) — logged negative result was
   objective choice, not ES viability.
5. Proposer default: switch `code_evolve.py:683-709` ordering to prefer
   `lfm3_1p7b_cc_fq` first (drop Qwen-Coder-3B → challenger; note 3B is
   Qwen-Research non-commercial license, 1.5B is Apache).

---

## 3. Open/unverified items to instrument

- >20-cycle merge→retrain drift: no published data — add per-round
  held-out probe + logit-drift metric (cheap: NLL on a fixed eval set vs
  canonical base each merge).
- All RL-relevant variants (rsLoRA/PiSSA/DoRA/LoRA+) lack published RL
  results — A/B on single hone rounds before adopting as default.
- LFM3-cc real proposer quality on OUR tasks (vendor disclaims agentic
  coding) — run `scripts/bench_proposer.py` head-to-head vs spark sft9 and
  parable_granite41_3b before flipping the default.
- Repeated bf16 merge rounding drift — canonical fp32 base policy.
