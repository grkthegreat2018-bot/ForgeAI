# RSI-Loop Research: Training Objectives — RL Variants + Preference Optimization

Compiled 2026-11-23. Companion to `RSI_LOOP_BASE_LORA_RESEARCH.md` (base model
+ adapter architecture). This doc covers the TRAINING-OBJECTIVE layer:
what to optimize on self-generated, verifier-scored data inside the
infinite_loop / code_evolve / ThinkingPipeline paths.

---

## 0. What already exists (don't re-build)

`forge/self_play/grpo_trainer.py`:
- MC-GRPO median baseline (G=2-4 small-group regime)
- CISPO (MiniMax asymmetric clip 0.2/0.28 + detached IS weight)
- GTPO (entropy control + conflict), ETR (entropy-trend reward)
- ExGRPO replay buffer (correctness × mean-token-entropy value)
- GRPO-λ dynamic length penalty, golden replay, repetition penalty,
  SC-GRPO/OM-GRPO/GVPO, zero-variance group pre-filtering
  (`infinite_loop.py:2820-2827`), behavior-log-probs fix (real PPO ratio),
  ref-free via `lora_disabled` forward.

`forge/training/runners/dpo_align.py`: DPO, ORPO (default), KTO (EMA
KL-reference — a variance-reducing variant of the paper's in-batch
estimate, unpublished but sound), self-reward LLM-judge mode.

`forge/training/runners/rlvr_train.py`: RLVR stage (binary verified
rewards, KL vs DPO ckpt, SPPO/PS-PPO option) — ThinkingPipeline
CPT→SFT→DPO→RLVR (staged pipeline, distinct from the live loop).

`forge/training/runners/dpo_data_gen.py`: judge-generated preference pairs.

`forge/model/edit_head.py` + verifier head: BT scalar trained on
evolve_pairs — already the loop's rerank/prior.

---

## 1. RL-path gaps (ranked, evidence-weighted)

### 1a. Cheap, high-value fixes

1. **Fixed-reference / sign advantage for small G** — at G=2-4 with binary
   rewards, all-pass and all-fail groups yield zero advantage → most
   groups are wasted (~0.69 degeneracy measured at G=4, not the i.i.d.
   prediction). Sign advantage **A = 2r − 1** scored 73.8% vs 28.4% GSM8K
   at G=4 vs group-mean (arXiv 2605.07689). NGRPO alternative: inject a
   virtual r_max sample into normalization for all-fail groups
   (arXiv 2509.18851). **~10 lines.**
2. **Dr. GRPO / Lite-PPO debias** — drop the 1/|o| length normalization +
   std-divide (constant-norm or group-mean + batch-level-std advantage);
   kills wrong-answer verbosity inflation on binary code rewards
   (arXiv 2503.20783, 2508.08221). Lite-PPO ablation: only group-mean +
   batch-std + token-level loss survive; beat GRPO/DAPO at 4B.
   **Trivial — advantage formula change.** Note the impossibility result:
   no estimator is both unbiased and length-invariant (arXiv 2607.23364).
3. **DAPO backfill (missing piece)** — we have asymmetric clip via CISPO
   and GRPO-λ length penalty; we do NOT have dynamic-sampling backfill
   (oversample prompts until the batch is filled with std>0 groups) or
   token-level loss aggregation (arXiv 2503.14476). **Low cost.**
4. **KL-Cov / CLIP-Cov entropy guard** — entropy change ≈ −Cov(log π,
   advantage); detach or KL-penalize the top ~0.02-0.2% covariance tokens
   (arXiv 2505.22617). Root-cause fix vs our reward-side ETR/GTPO.
   Entropy collapse lands in the **first ~50 GRPO steps** at small scale
   (arXiv 2605.11491) — instrument entropy telemetry first (already
   queued in scratchpad).

### 1b. Diversity preservation (the RSI-critical risk)

RLVR improves pass@1 but the **base model wins pass@k at large k**
(arXiv 2504.13837) — for a self-proposing loop, shrinking proposal
diversity is the silent killer. Mitigations, ranked by fit:
- **DPH-RL**: forward-KL/JS rehearsal against samples from the INITIAL
  policy — needs only a one-time base-sample dump (no live ref model);
  improves pass@1 AND pass@k, prevents forgetting (arXiv 2509.07430).
  Our `lora_disabled` forward makes this nearly free.
- **UCPO**: uniform-over-correct-solutions penalty, +10% pass@64 at
  1.5-7B (arXiv 2605.00365).
- **BBG**: update only on zero-success problems lifts pass@256 above base
  (arXiv 2606.15455) — dovetails with hardest-first curriculum.
- **Instrument pass@k tracking per round** — currently absent in `forge/`.
  Cheap monitor: per-round pass@1 AND pass@8 on a fixed eval set.

### 1c. Rollout efficiency (single-GPU dominant cost)

- G=2-4 validated: 2-GRPO retains 98.1% of 16-GRPO at 12.5% rollouts
  (arXiv 2510.00977) — our G=4 default is fine.
- **Prompt selection is the biggest lever**: bandit/Kalman difficulty
  scorers targeting p≈0.5 (Arcus −48-57% rollouts arXiv 2609.38018;
  KGPS −83% arXiv 2607.27610; MoPPS arXiv 2507.04632). Our loop
  pre-filters zero-variance groups but doesn't SELECT prompts — adding
  a difficulty-tracker on task stats is the largest remaining saving.
- Replay/staleness: ExGRPO/golden-replay are implicitly off-policy —
  add per-rollout age eviction, advantage-magnitude priority,
  negative-advantage veto, μ≈2-4 update epochs/rollout (μ-GRPO
  arXiv 2605.17570; replay practices arXiv 2606.04560).
- **TIS (truncated IS)**: only needed if rollouts run on a DIFFERENT/
  quantized engine than the trainer — relevant if proposing on `_fq`
  weights while training bf16.

### 1d. Reward shaping — what to skip

- **Skip fraction-of-tests partial credit**: denser but miscalibrated —
  partial-pass gradients conflict and don't reliably beat binary under
  critic-free RL (arXiv 2605.02944). Staged syntax→compile→run→tests is
  a viable aux only if all-fail rate stays high after #1a.
- Process rewards without a PRM: line-level execution feedback gives a
  free step signal (arXiv 2410.17621) — traceback line is already in our
  verifier output.
- Reward hacking: penalties beat defensive prompting (arXiv 2603.07084);
  hidden/rotated tests; our `persistent_executor` jail already confines
  fs writes — good posture. Monitors alone get evaded.
- **Self-judging at ≤3B is unreliable** (Qwen3-0.6B fails >50%;
  arXiv 2509.13332) — keep the external subprocess verifier as ground
  truth; the trainable alternative is a hidden-state probe
  ("Representation-as-a-Judge", arXiv 2601.22588) — i.e., exactly our
  verifier-head direction.

### 1e. Self-generated tasks

- **AZR** (arXiv 2505.03335): proposer generates code tasks verified by
  executor; proposer reward = learnability (~50% pass rate) — validates
  our infinite_curriculum proposer design; add learnability reward to
  task generation if absent.
- **R-Zero** (arXiv 2508.05004): challenger reward = solver uncertainty;
  majority-vote pseudo-labels when no gold answer exists.
- **RISE** (arXiv 2505.13445): joint solve+verify training improves both
  at 1.5-7B — candidate for the verifier head's training signal.

### 1f. GRPO+LoRA specifics (revises the LoRA doc)

- **Rank collapse**: DeepMath-Hard 73.1% @r=64 → 4.7% @r=128
  (arXiv 2611.27148). **Keep r ≤ 32-64; do NOT raise rank under small
  batches.** Conflicts with the earlier "rsLoRA frees high ranks"
  guidance — for the RL path, r16-32 is the ceiling, not the floor.
- **PEFT bake-off under RLVR** (arXiv 2512.23165): **DoRA/AdaLoRA/MiSS
  consistently beat LoRA (DoRA even beats full-FT); SVD-inits
  (PiSSA/MiLoRA) COLLAPSE under RL; rank-1 and VeRA collapse.**
  → Revises the adapter doc: DoRA moves from "A/B candidate" to the
  leading RL-path adapter; PiSSA/MiCA init must be kept OFF the GRPO
  path (fine for SFT hones).
- RL forgets less than SFT due to on-policy-ness, not KL
  (arXiv 2510.18874) — supports our ref-free design.
- Merge-vs-stack across GRPO rounds: UNVERIFIED — nearest evidence says
  merge periodically + fresh adapter; keep replay buffer.

---

## 2. Preference-optimization path

### 2a. Objective selection for verifier-scored data

- **Pairs when you have them, KTO for the residue.** KTO matches DPO at
  1-30B on unpaired data and survives 90% desirable-drop
  (arXiv 2402.01306), but on real same-task pairs, DPO extracts more
  (code: HumanEval 70.1 vs 62.8 naive-KTO, arXiv 2411.13611v3). KTO is
  the right objective for single-candidate verdicts and all-pass/
  all-fail groups — which the loop produces in bulk.
- **Soft margin-weighted labels**: convert verifier score delta →
  q = σ(k·Δ) as the DPO target instead of hard 0/1 — GPO soft labels
  (arXiv 2409.06691), MADPO margin-weighting (arXiv 2510.05342), ADPO
  soft-DPO (arXiv 2510.18913). Fits the BT-head scores directly.
- **Label-noise tolerance**: cDPO ε-smoothing (ericmitchell.ai/cdpo.pdf)
  caps margin-chasing at h* = (1/β)·log((1−ε)/ε); rDPO provably
  flip-robust (ICML'24). NOTE: one negative replication exists
  (insights-1.16) — sweep, don't trust.
- **20-variant controlled study at 1.5B** (arXiv 2603.19335): NO variant
  significantly beat vanilla DPO; ORPO ranked first n.s.; **SimPO
  significantly worse (−11.5pp)**. SimPO's gains are mostly length-norm,
  reproducible inside DPO (arXiv 2407.13709). Our ORPO default is fine
  but its "no ref model" advantage is moot — `lora_disabled` already
  gives a free ref.

### 2b. Iterative-DPO failure modes (the many-round regime)

- **Improve-then-saturate ~4-5 rounds** (Self-Improvement Reversal,
  incl. MBPP; gair-nlp.github.io/self-improvement-reversal/) — most gain
  is answer-reselection, not new capability. Diversity + OOD decline.
- **Likelihood displacement** (chosen logp falls too — arXiv 2410.08847,
  DPOP arXiv 2402.13228): worst on LOW-EDIT-DISTANCE pairs — exactly our
  regime (candidates to the same task share most tokens). Fix = NLL term
  on chosen (RPO, arXiv 2404.19733 — "crucial" per iterative study) or
  DPOP λ·max(0, ref−pol).
- **Data reuse**: 2nd epoch on the same pool redirects learning ~38°
  without held-out gain → fresh pairs each round, ≤1 epoch.
- **Continual-DPO degradation**: Scientific Amnesia (arXiv 2606.21089) —
  proposer chains lose pass@1 over rounds on HumanEval subdomains;
  conservative scheduling the only improver. Re-anchor to base
  periodically (iterated-RLHF overoptimization paper: re-anchoring is
  robust across rounds, arXiv 2505.18126).
- **Diversity collapse**: DPO adds modest collapse over SFT; monitor
  top-k entropy/mass as early warning (arXiv 2410.11677); EntroPO
  (arXiv 2509.12434) counters it specifically for multi-turn coding.

### 2c. Pipeline ordering — RL before preference, not after

- **G2D** (arXiv 2605.21266): short GRPO warm-up → offline DPO on the
  GENERATED pairs matched/beat full GRPO at ~4× less compute (7-8B MATH).
  Our loop already does this implicitly: GRPO rollouts produce
  verifier-scored candidates → evolve_pairs → DPO/verifier-head data.
  **RL→DPO is the validated ordering; DPO-before-RL is NOT established
  as beneficial at ≤3B** (head-to-head at 1.5B: SFT→DPO never beat its
  SFT start; NiteeshSK/math_square). ThinkingPipeline's SFT→DPO→RLVR is
  the conventional order — worth an A/B against SFT→GRPO→DPO.
- Recommended round shape: generate→verify→{SFT-on-verified (existing
  hone) | DPO on scored pairs}→optional GRPO polish. Alternate rather
  than fix stages.

### 2d. Verifier head's role (resolve the dual-use)

- Keep the BT head as: (i) soft-label/pair-weight source (§2a),
  (ii) best-of-n reranker for hone data + early-stop
  (BOND distillation arXiv 2407.14622; West-of-N improves the RM itself
  arXiv 2401.12086), (iii) drift monitor vs policy.
- **Don't feed it back as the RL reward** without guards — RM
  overoptimization on self-generated pairs is documented (Gao scaling
  laws; Coste/WCO ensembles at 25% noise; WARM weight-averaging
  arXiv 2401.12187).

---

## 3. Corrections to the pasted technique map (context matters)

| Pasted claim | Correction for ForgeAI |
|---|---|
| "RL/preference LR ~5e-6" | Full-FT figure. LoRA-RL: ~10× → ~5e-5, matching the spark-probe sweet spot (scratchpad). |
| "QLoRA+rsLoRA r16 + Unsloth" | Unsloth doesn't support LFM3 FT yet; QLoRA unnecessary at 1.7-3B/12GB unless stacking engines. Merge into **fp32** canonical (not just bf16). |
| "α=r default, 2r if underfit" | Under rsLoRA the α convention is α/√r — scaling flag replaces alpha tuning; high-α+low-rank produces more forgetting "intruder dims" (arXiv 2410.21228). |
| "AdaLoRA for strict param budgets" | Fails under GRPO (arXiv 2605.07366) — SFT-only. |
| "PiSSA for small models" | NEW: SVD-inits collapse under RLVR (arXiv 2512.23165) — keep PiSSA/MiCA OFF the GRPO path. |
| "VeRA extreme efficiency" | Collapses under RL; reasoning bottleneck — skip. |
| "SimPO for memory-constrained" | Significantly WORSE at 1.5B controlled (arXiv 2603.19335); ref is already free via `lora_disabled`. |
| "DPO needs ref in memory / ORPO saves it" | Already moot — LoRA-off forward = free rolling ref (recommended pattern). |
| "KTO for unpaired feedback" | Underplayed: binary pass/fail IS the loop's native signal — KTO handles all-same-outcome groups that zero out under DPO/GRPO-group-mean. |
| "GRPO needs strong SFT base" | Rank-collapse + pass@k findings refine: keep r≤32, add sign-advantage + diversity guards. |
| Frameworks (Unsloth/Axolotl/LF) | Reference implementations only — ForgeAI's own stack is ahead (CISPO/ExGRPO shipped); Spark's Llama-Factory suggestion is irrelevant post-port. |

---

## 4. Recommended training stack (ranked)

| Tier | Item | Cost |
|---|---|---|
| 1 — now | **Sign advantage A=2r−1** (or NGRPO virtual r_max) for degenerate groups + **Dr.GRPO/Lite-PPO debias** + **pass@k tracking**. G=4 binary-reward is exactly the starvation regime. | ~tens of lines |
| 2 — now | **DPO+NLL anchor + soft margin labels from verifier scores**, rolling ref via `lora_disabled`, ≤1 epoch, fresh pairs/round; **KTO for unpaired/all-same groups** with λ_D/λ_U per [1,4/3] rule. | small |
| 3 — soon | **DPH-RL forward-KL rehearsal** vs initial-policy samples (diversity preservation) or UCPO; dynamic-sampling backfill; difficulty-bandit prompt selection. | medium |
| 4 — soon | **DoRA on the RL path** (beats LoRA under RLVR), keep vanilla-LoRA for SFT hones; SVD-inits banned from GRPO. | low (adapter exists) |
| 5 — research | KL-Cov entropy guard; μ-GRPO replay epochs (2-4); AZR learnability reward for curriculum proposer; RISE joint solve+verify for verifier head. | medium |
| Avoid | GSPO (MoE-scale gains), fraction-of-tests rewards, self-judging at ≤3B (use probe/verifier head), adaptive rank under GRPO, r>64 under small batch, PiSSA/MiCA init on RL path, SimPO, 2nd-epoch pair reuse. |

## 5. Instrumentation checklist (cheap gates before scaling rounds)

- Per-round: pass@1 + pass@8 on fixed eval; chosen-absolute-logp
  (displacement watch); top-k entropy/mass (diversity); adapter
  magnitude vs base (drift); entropy trace for first ~50 GRPO steps.
- Verifier-head calibration vs subprocess verdict each round (it's the
  soft-label source — drift here corrupts pair weights).
- Replay fraction ≥ ~25% old data in every round (both SFT and DPO).
- Re-anchor to base policy every ~4 rounds (iterative-DPO saturation
  window).
