# ForgeLM V14-RSI — Design Document (R&D round V14)

Goal (user brief): a ~2.5B-param LLM that outperforms specialized indie coding
models (bar: Qwen2.5-Coder-3B, HumanEval 52.4 / HumanEval+ 42.7; aspirational:
Qwen3-4B-class), runs inside the RSI coding loop (CodeEvolve /
InfiniteSelfPlayLoop) at LFM2.5-class speed/memory or better, <3GB total
memory at max load, ~1k tok/s average generation, near-instant fine-tuning,
and fast knowledge ingestion.

## 1. Constraints → first-principles budget (RTX 5070, 12GB, ~672 GB/s)

Decode is HBM-bandwidth-bound: tokens/sec ≈ BW / bytes-read-per-step.

| Design point | Weights read/step | Max fwd/s (672 GB/s) | w/ MTP ×2 accept |
|---|---|---|---|
| Dense 2.5B bf16 | 5.0 GB | ~134 | ~270 |
| Dense 2.5B int4 | 1.25 GB | ~540 | ~1.1k |
| MoE 2.5B total / ~1.35B active int4 | ~0.69 GB | ~975 | ~1.9k |

=> **MoE + int4/NVFP4 + self-speculative decode is the only honest path to
~1k tok/s single-stream at 2.5B.** Batched RSI workers (generate_batch)
multiply aggregate throughput further.

Memory budget at max load: int4-packed 2.5B ≈ 1.3GB + GLA-latent KV
(~4 attn layers × compressed latent, ~50–150MB) + conv/SSM/GDN2 states
(<100MB) + CUDA context + activations ≈ 1.8–2.3GB < 3GB. Headroom holds.

## 2. Architecture — `forgelm_v14_rsi` (parent: forgelm_v13_gdn2)

Topology (deliberately non-lossless vs V13 — new model, documented in
`dropped_keys`):

- d_model=2560, n_layers=24, vocab=65536 (Jamba tokenizer — enables V2
  checkpoint partial reuse and the proven `port_qwen_mla_moe.py` path).
- layer_types (24): 15 `conv` (DoubleGatedConv, LFM2-exact), 5 `mamba`
  (Mamba-3, complex state), 4 `attention` (GLA latent-compressed, d_c=256).
  Conv/Mamba carry position → use_rope stays False (V2 lineage).
- MoE FFN on every layer: 8 routed experts (top-2) + shared expert,
  moe_d_ff≈704, shared d_ff=1408, aux_free router. `moe_dense_bypass=True`
  keeps dense-FFN lossless port-in; warmup steps anneal to routing.
- Carried V12/V13 keys: use_mamba3, use_kronecker_embed, use_pit, use_outro,
  use_gdn2 — with `gdn2_n_heads=4` (≈9.2M/layer vs 39M at H=20; live-learning
  surface kept on all blocks).
- Dropped keys (documented): `use_forge_hybrid` (explicit mamba3 layers make
  per-block SSM routing redundant; saves ~30M/layer of dead weight at decode).
- New subsetting fields: `gdn2_layers`, `titan_layers` (None = all blocks;
  list of idx = apply only there). TitanMemory placed on 4 blocks (stride 6)
  ≈ 105M params instead of 624M — enough LiveT capacity for code-domain
  ingest.
- use_mtp=True, mtp_n_heads=4, mtp_loss_weight=0.495 (evolution-tuned values).

Estimated params ≈ 2.4–2.6B total; ≈ 1.3–1.4B active/decode step
(mixers 686M + active FFN ~520M + head 168M; GDN2/Titan gate-zero-skipped).

## 3. Novel heads — "hyper acceleration"

Existing heads (MTP, Medusa, EAGLE-3, DSpark) accelerate *token* rate.
V14 adds heads that accelerate *work* rate inside the RSI loop.

### 3a. CopySpan head (`use_edit_head`) — novel
RSI proposers re-emit ~80–95% of the source file unchanged. CopySpan turns
unchanged regions into O(1) decode events:

- Emission format: the LM head can emit special op token `<copy>`; the next
  two "steps" are produced by a pointer head, not the vocab head:
  `P = softmax(h_t W_q (H_src W_k)^T / sqrt(d))` — a content pointer over the
  source span's final-layer hiddens (cached at prefill, no recompute).
- Driver splices `source[start:end]` into the output verbatim, then feeds the
  copied span through the model as ONE chunked T>1 forward (weights read
  once for K tokens) — commits state exactly as if decoded (respects the
  R60-4 conv/SSM left-context invariant).
- Cost of copying K tokens ≈ 2 pointer steps + amortized chunk forward ≈ 2–3
  weight-reads vs K. On a typical 5-line edit in a 300-line program: ~10–25×
  fewer sequential steps. Composes with MTP verify on the generated spans.
- Trained jointly (MTP-style auxiliary) on edit-format SFT rows derived from
  `code_programs` (child>parent pairs) — op supervision is cheap: diffs of
  parent→child give gold copy-spans automatically.
- Lossless fallback: if the model never emits `<copy>`, decode is standard
  autoregressive. Zero quality risk at init (head exists but op tokens are
  never emitted by an untrained model).

### 3b. Verifier head (`use_verifier_head`) — novel-in-house
- Scalar head on ln_f(hidden) → logit ≈ P(candidate passes target tests).
- Trained on evolve_pairs.jsonl / code_programs (chosen vs rejected scores)
  with a Bradley-Terry style loss — reuses DPO signal the loop already logs.
- Uses: best-of-n rerank of `generate_batch` samples before burning
  subprocess eval budget; early-stop when P(remaining samples win) < eps;
  feeds `reward_guard`/`surrogate` as a model-side prior. Minimal downtime =
  fewer doomed evals.

### 3c. Existing, carried: MTP-4 self-spec (`mtp_selfspec`), suffix decoding
for code repeat spans, megakernel CUDA-graph decode, sessions/prefix cache.

## 4. Fast fine-tune + knowledge ingest (no retraining wall)

- Weight side: tagged-LoRA hone (~100 steps, minutes) + hot-swap
  (`engine._load_lora_impl`, LoRARouter) — proven loop path.
- Live side: GDN2 recurrent states (working memory) + TitanMemory fast
  weights (4 blocks) + NGramMemory sidecar — `LiveSession.ingest` /
  `eval_nll` / journaled revert. Instant ingest of the target file, docs, or
  retrieved snippets without touching weights.
- Answer to "train from nothing vs port": **port + distill**. From-scratch
  2.5B needs ~1–3T tokens (weeks on one 5070). Port path:
  `port_qwen_mla_moe.py` on Qwen2.5-Coder-1.5B/3B → GLA+MoE (lossless) →
  anneal router → distill conv/mamba3/GDN2 substrate from the ported teacher
  (forge/distillation + `--distill` in sft_train) → SFT edit-format +
  verifier-head pairs. This gives code quality on day one while the hybrid
  substrate learns under distillation.

## 5. Validation gates (per F-directives)

1. `scripts/bench_v14.py`: param count (total + active), random-init forward
   smoke, decode tok/s + peak VRAM on CUDA at bf16 and int4-sim.
2. Bit-exactness: V14 is intentionally non-lossless vs V13 (new topology) —
   documented; but all NEW heads must not perturb a base forward (heads are
   read-only taps on `hidden`; op tokens unused at init).
3. tests/unit: copy-span head shape/pointer test, verifier head test,
   preset lineage test, subset-flag test (gdn2_layers/titan_layers).

## 6. Phased roadmap

- P0 (this round): preset + CopySpan/Verifier heads + subset flags + bench.
- P1: port Qwen-Coder → GLA+MoE base; distill hybrid substrate; MTP train.
- P2: edit-format SFT data pipeline from forge_evolve.db (parent→child diffs
  → copy-span supervision); verifier-head BT training on evolve_pairs.
- P3: wire `edit_decode` into engine decoding registry (`decoding=`), loop
  eval on CodeEvolve targets vs Qwen2.5-Coder-3B proposer baseline.
- P4: HumanEval/MBPP eval harness vs 3B-class baselines; iterate.

## 7. Measured (P0, random-init, RTX 5070)

- Total params **2.465B** (dedup): int4 ~1.23GB, bf16 4.93GB. ✓ ~2.5B target.
- Active/step ~1.26B (gate-zero GDN2/Titan skipped) → int4 ~0.63GB read →
  ~900 fwd/s ceiling at measured ~570 GB/s matvec bandwidth → ~1.8k tok/s
  with MTP ×2 accept. Dense eager: 19–37 tok/s (launch-overhead-bound —
  dense_bypass actually beats routed in eager; graphs flip this).
- Peak VRAM bf16 forward+decode: **5.12GB** → int4 model ≈ 1.6–1.9GB total.
- Build+load to CUDA: 8.6s. Prefill 256 tok: 2.1s eager (unoptimized).
- Copy-span driver verified end-to-end (pointer span splice + chunked
  commit, tiny model): `<copy>` → span (6,11) → 6 verbatim tokens.
- Bug found+fixed: Mamba3Block left fp32 after complex scan → dtype
  mismatch at bf16 out_proj (`mamba3.py` — cast at boundary; test added).
- Budget tuning hits: `gdn2_head_dim` MUST be explicit (default derives
  640-wide heads = 46M/layer, not 9.2M); GLA RoPE caches scale with
  max_seq_len (262k→16k saves ~800M buffers); MTP independent vocab heads
  cost 671M → adapter_heads (shared tied head) = 26M; Titan rank=128
  shrinks d×d fast-weight buffers 26M→~3M.
- torch.compile broken on this Windows env (Triton cache race + no
  complex-op codegen for mamba3 scan) — engine path should use the
  megakernel/graph-capture stack instead of inductor compile.
- **CUDA-graph decode VERIFIED on full V14**: BCG capture + replay works
  end-to-end → **108 tok/s bf16** (9.26ms/step) vs 19 eager (~5.7×).
  Required two capture-blocker fixes:
  (a) `MoELayer` routed path — nonzero/scatter loop is not capturable;
      added fixed-shape gather path (`_w1s/_w2s/_w3s` stacked caches,
      top-k weight gather via einsum, N≤64 tokens) — preserves the
      read-only-selected-experts bandwidth win inside the graph.
  (b) `KroneckerEmbedding._ids_to_bytes` allocated its byte table on CPU
      → cross-device copy inside capture; now device-local.
- Remaining speed headroom: bf16→int4/NVFP4 halves weight reads
  (~2× → ~200-220 tok/s/step), MTP-4 verify multiplies accepted tokens
  (~×2-3 → ~500+ tok/s single stream), CopySpan removes ~80-90% of
  sequential steps on edit outputs. Path to ~1k effective tok/s holds.
- Pre-existing fixes landed this round: activation catalog missing
  kv_v_bits/kv_boundary_v_bits entries; GUI hub tests updated for the
  bind_loop fan-out model. OPEN: test_flux cuda determinism (pre-existing,
  unrelated to V14 paths).
