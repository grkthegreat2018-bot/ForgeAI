# RSI-Loop Research: Beating LFM3-1.7B's Param Ceiling

Compiled 2026-11-23. Fourth in series: base+adapters → objectives → memory
→ this doc: **capacity**. Question: make the ported `lfm3_1p7b_cc` smarter
past 1.73B params — without retraining from scratch, without breaking the
pretrained weights (gate-zero/bit-exact-at-init), without killing decode
(~1k tok/s target) or the 12GB budget.

The honest framing: "param limit" splits into four separate capacities —
decode cost (keep it flat), retention (gate-zero preserves), reasoning
capability (loops/test-time-compute/training), knowledge capacity
(MoE/memory/adapters). Different levers per axis.

---

## 0. Ground truth (what exists)

- **D1 graft round** (`lfm3_1p7b_d1`, bit-exact verified): +73.9M dormant —
  GDN2@conv[1,5,9,13,17,21], ValueResidual, LearnedSink, AttnRes,
  CopySpan + Verifier heads. Open items logged (pad_mask, k-buffer,
  v0 donor, SWA ring pos-mod).
- **Queued next grafts** (scratchpad:1564): thin-MoE FFN, AttnRes-open,
  MTP-4 adapter, EAGLE-3 depth-3.
- **E-loop probe result** (scratchpad:126): untrained looping DEGRADES NLL
  monotonically (1it=2.99→8it=5.82); gate-training helps; "looped
  transformer needs training, not free compute" — now confirmed by the
  whole post-hoc-looping literature (§2).
- **Infrastructure**: `moe_dense_bypass` lossless MoE-at-init +
  router anneal; `use_titan_memory`, `use_mtp`; distill chain
  `lfm3_distill_v5–v10`; BT verifier head; LoRA registry/router;
  `probe_lfm3_loops.py` runtime loop dials already in forward.
- LFM3-1.7B is **~13k tokens/param** (22T pretrain) — the most
  overtrained donor anyone would upcycle; no larger LFM3 text ckpt
  exists, so surgery-vs-bigger-native is moot. Real baseline to beat:
  **continued-pretrain of the untouched base on the same tokens**.

---

## 1. Depth growth — best evidence is AT 1.7B scale

Controlled 1B→1.72B (+50% depth, 1.5B-token CPT, frozen trunk) results:

| Method | SFT avg | Notes |
|---|---|---|
| Continued-pretrain baseline | 49.25 | THE bar to beat |
| Naive SOLAR DUS (tail dup) | 49.49 | ≈ baseline — skip |
| LLaMA-Pro zero-init blocks | 52.15 | copy block, zero out-projs, train-only-new |
| LESA predicted-init | 53.66 | **fails at 1B** (aux net underfits <16L) — skip |
| OT-DUS fused adjacent init | 53.81 | +13% faster; insert top-half |
| MIDUS (dup + memory FFN) | 54.08 | best; maps to our TitanMemory pattern |

Recipes that work at this scale (TLI arXiv 2410.11654, OT-DUS 2508.08011,
LLaMA-Pro 2401.02415, µP depth theory 2511.04981):
- **Insert interleaved in the TOP HALF** (not contiguous tail — seam
  discontinuity); copy donor block; **zero only the OUTPUT projections**
  (`o_proj`/`down_proj`/conv `out_proj` — zeroing everything kills
  gradient flow); freeze trunk, train only new blocks.
- Budget honesty: published gains used ~1.5B-token CPT rounds. At
  ~100-1000-step hone, expect style/domain alignment, not capability.
  The graft is still the right *substrate* — schedule a CPT round to
  harvest it.
- For LFM3: copy **conv blocks for most insertions** (cheapest operator)
  + **1-2 full-attn blocks** (scarcest: only 6/24 — highest capability
  per grafted param, cross-operator transplant validated by
  E-Tamba/DiT-grafting at ~1B/≤2%-pretrain budgets).
- Optimizer hygiene for new params: fresh Adam state + short warmup,
  don't inherit the trunk's decayed LR (ReLoRA, µP-transfer papers).

## 2. Post-hoc looping — real now, but GATES are the load-bearing part

| Method | Recipe | Result | Budget |
|---|---|---|---|
| **UpLoop** (2610.25004) | layer-wise local loops + sigmoid gates on residual branches, gate→0 | **+6.17% avg/20b on Qwen3-4B @R=3** | **SFT-scale — the only recipe at our hone budget** |
| Dory-1.0-4B (2611.03998) | flat pretrain → ~1.5T looped continuation, sampled k, I-R-O reinjection | k=4: HumanEval +4, SWE +11.5; **1.3–2.5× fewer output tokens**; sats ~4 | production-scale |
| Retrofitted Recurrence (2511.07384) | P/R/C + `Linear([s;e],2h→h)` reinjection + curriculum | beats non-recurrent twin at matched FLOPs | ~50B tokens |
| LoopUS (2605.11011) | input-dependent gates vs hidden-state drift | stabilizes post-hoc looping | post-training |
| MixerLoop (2608.18230) | loops only the mixer (GDN) sub-block T×, FFN once; per-iter SSM states | −43% backbone compute at matched perf | post-training |
| **No Free Loop** (2610.42030, MSR) | training-free + 33-66M-tok LoRA | **null result above 0.8B** at equal params | — |

- Our zero-shot NLL degradation is the documented expected failure —
  untrained models drift; trained looped models converge to fixed
  points/cyclic trajectories (2604.11791). Stabilizers: gates≈0 (our
  machinery), input reinjection, LTI ρ<1 injection (Parcae 2604.12946),
  sampled-k training (fixed-k kills depth extrapolation — OpenMythos #28).
- **1.7B sits inside NFL's untested 0.8B–4B gap**; UpLoop's success at 4B
  (gated) vs NFL's null (ungated) says gating is the deciding variable.
- **KV/state**: `last_loop` KV for attention (k-independent memory,
  near-lossless — Dory-verified); **namespaced per-iteration conv states**
  (LFM3 conv state = 7-token window → nearly free). SWA inside a loop:
  window-cap keeps cost fixed.
- **Decode cost**: per-token time ≈ ×(N + m(k−1))/N layer-passes; batch-1
  looped decode is kernel-bound (Ouro: 62 tok/s @T=4 RTX5090) — fuse the
  loop body, capture per-k bucketed graphs (k∈{1,2,4}), and use
  **WaveFront self-speculation** (shallow-iter draft / deep-iter verify,
  2.4–4.8× on looped LMs, training-free) to defend the 1k tok/s target.
- **Novel twist for this arch** (directive C): loop only the CONV sub-
  block inside conv layers (MixerLoop recipe) — conv state is a 7-token
  window, per-iter namespacing ~free, keeps most of the pretrained layer
  function intact. Nobody has looped an LFM — first result is ours to
  publish internally.

## 3. MoE upcycling — via existing `moe_dense_bypass`, with a donor caveat

- Our `moe_dense_bypass` (fine-grained partition + bypassed router at
  init + anneal) is **strictly better than the standard softmax-topK
  init** (Nemotron upcycling paper 2410.07524 proves standard init isn't
  lossless when topK<N; ours is exact).
- **Caveat**: upcycling scaling law (arXiv 2502.03009) — "the more
  overtrained the donor, the smaller the improvement rate." LFM3 @22T
  tokens → cloned experts diversify slowly. Fixes: Drop-Upcycling
  partial re-init (2502.19261), DPSL noise (2510.01185), virtual-group
  init, or PESC-style per-expert LoRA differentiation (EMNLP'24 — the
  adapter-era upcycle, fits hone budget).
- Budgeted plan: **partial upcycle** — full-size clones (`moe_d_ff=7680`)
  on 4-8 FFNs (the 6 full-attn layers are the natural pick) ×2 experts +
  shared: +0.3-0.5B params, +0.6-1GB weights, bypass keeps init exact.
  Upcycle ALL 24 FFNs would blow the 12GB budget — skip.
- Zero-param variant: moefication PARTITION (`moe_d_ff<7680`, shared
  FFN split) — no capacity but cuts decode FLOPs once router activates;
  already-validated pattern from `qwen25_05b_mla_moe`.
- Qwen1.5-MoE upcycle precedent: fine-grained experts + shared expert +
  top-4 routing matched 7B dense at 2.7B active — the design target.

## 4. Param-free capability multipliers (parallel track)

These compose with every graft above and don't touch decode:

| Lever | Evidence | Fit |
|---|---|---|
| **Exec-feedback hone** (RLVR unit-test + lint reward) | +14pp @0.6B (2605.30478); >4σ @3B (2604.21950) — strongest small-model result | Already our loop; lint-only hurts — keep test+lint combined |
| **GenSelect RL selector** on BT verifier head | 1.7B selector ≈ large-model selection (2602.02143) | Upgrade the existing BT head into a trained selector → amplifies best-of-N = capability per decode |
| **Anchored OPSD distill** | teacher = same model + privileged info (zero extra VRAM); external teacher ≤8B or signal vanishes (2610.11247); anchor to avoid v5→v10 drift (2601.13100) | Fits hone budget; explains past distill rounds' plateau risk |
| **Repo RAG / kNN-LM datastore** | repo/API recall gains (RepoCoder +10%); **fails on algorithmic synthesis** — route by task type | NGramMemory → selective kNN injection; CPU datastore |
| **Per-task LoRA TTT** | ARC recipe: 1B 6.2→36.2% — biggest single number found | Offline/batch only — breaks decode budget; journal+discard adapter per task |
| **Product-key memory graft** | replaces 1-3 FFNs; factual-task gains ~0 FLOPs (berges25a) | Moderate graft risk; weaker for code than facts |
| MTP/EAGLE heads | speed only — **post-hoc MTP does not add capability** (Meta: gains need co-training; harmful at small scale) | Keep as decode lever, not capacity |
| MoD/early-exit | speed lever (Router-Tuning: 21% faster −0.2%) | Optional, stacks |
| CODI latent reasoning | self-distilled continuous thought at GPT-2 scale | Competes for same hone budget as looping — pick one |

**Honest ceilings** (don't retry): self-reflection/Reflexion without
external signal — all 18 comparisons negative at ≤7B (Reflexion never
fired at 1.5B); multi-turn self-driven agent chains — BFCL multi-turn
8-17% at 1.7B; unanchored recursive distillation → linear drift; RAG on
algorithmic reasoning → net-negative. External verifier stays ground
truth.

## 5. Recommended composite (evidence × cost × existing machinery)

| Round | Action | Expected | Cost |
|---|---|---|---|
| **D1.5 — finish open items** | pad_mask, k-buffer, v0 donor, SWA ring pos-mod | unlocks D1 gate-opening hone | small |
| **R-loop probe** | UpLoop-style gated layer-wise loop on 2-4 mid-stack layers (conv-majority + 1 attn), gates→0, k∈{1,2} sampled, ~200-step hone, last_loop KV + namespaced conv state | first published gated-loop on hybrid; k=1 bit-exact by construction | small; ~+0.3B×0 = ~0 params |
| **D2 graft** | TLI/OT-DUS injection +4-6 blocks top-half (3-4 conv + 1-2 full-attn), zero out-proj, train-only-new | +3.5-5 pts class result needs a ~1B-token round | +0.2-0.35B (~0.4-0.7GB) |
| **MoE-FFN graft** | partial upcycle: full-clone 2E+shared on 6 full-attn FFNs, DPSL noise, bypass-init | capacity at ~flat decode cost | +0.3-0.5B |
| **Selector upgrade** | GenSelect-style RL on verifier head | best-of-N multiplier | small |
| **Distill hygiene** | OPSD self-teacher + anchor to base each distill round | stops v5→v10 drift | free |
| **Later** | per-task TTT adapter for hard offline rounds; memory-FFN (MIDUS) inside D2 inserts; WaveFront decode once loops exist | — | — |
| **Skip** | naive SOLAR DUS, width growth, LESA, post-hoc MTP-as-capacity, pause/filler tokens, Soft Thinking, self-reflection chains, ungated re-run looping | — | — |

## 6. Measurement protocol (per F-directives)

- Lineage gate: every graft round must be bit-exact at gate=0
  (the `test_lfm3_d1_parity.py` pattern).
- Baseline: CPT/LoRA on the SAME tokens vs grafted variant — a graft
  that doesn't beat same-tokens-on-base is a scoring artifact
  (SOLAR-DUS lesson).
- Loops: NLL-vs-twin at k∈{1,2,4} + per-token ms batch-1 (kernel-bound
  floor) + end-to-end task latency (Dory's token-count reduction can
  make loops net-faster).
- Capability: fixed eval suite per round (HumanEval-class + evolve
  targets); pass@1 AND pass@8 (don't trade diversity for score).
- Drift: `_functional_fingerprint` ELO + canonical-base NLL delta per
  round.
