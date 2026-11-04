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

## 2026-11-04 - Windows torch.compile failures are encoding + loader bugs, not model bugs

Symptom: inductor crashed `UnicodeEncodeError: 'charmap' ... '\u2192'` compiling the dense/base ckpt, and `OSError: could not get source code` on the packed one. Three layered bugs, all Windows-cp1252 / introspection:

1. `codecache.write_atomic` writes generated wrapper .py via `Path.open()` with locale default (cp1252) — U+2192 arrows in forge docstrings land in the generated source. **Fix**: replaced with a UTF-8 tmp-file + `os.replace` equivalent (`_patch_inductor_utf8` in engine_activation). Upstream torch main already grew an `encode_utf_8` kwarg for this — our patch is the backport.
2. `_reload_python_module` reads cached source via plain `open()` (cp1252 again) — decode side. And stale pre-patch files stay cp1252 → re-encode-on-miss fallback.
3. Its `ModuleType`+`exec` has no real loader → triton's `@jit` calls `inspect.getsourcelines` at exec time → 'could not get source' (this was the silent _fq warmup killer too). **Fix**: load via `importlib.util.spec_from_file_location` + sys.modules registration BEFORE exec — a real SourceFileLoader makes introspection work and decodes PEP-263 UTF-8 for free.

Lesson: three symptoms, one root class — Windows assumes locale encoding everywhere torch's internals don't. Patched once at compile-activation, covers codecache + all generated kernels.

## 2026-11-04 - Multi-generate crash: inference-tensor restore outside InferenceMode

Symptom: 2nd+ `generate()` died `RuntimeError: Inplace update to inference tensor outside InferenceMode` at `_defer_restore_recurrent`.

Root cause: conv/SSM state buffers are created inside `torch.inference_mode` (decode steps, graph capture) — the in-place snapshot restore ran wherever the caller left off. In-place writes to inference tensors are only legal inside inference mode; a blanket `inference_mode()` wrap would break the opposite case (normal buffers can't be written inside it).

Fix: `_state_copy_back_` decides per-tensor via `cur.is_inference()` — wraps the copy in inference_mode only for inference buffers.

## 2026-11-04 - 'Degenerate model' was mostly decode-settings, not weights

Symptom: base + distill250 both emitted verbatim loops ('arr: arr: arr:') on every prompt at temperature=0.

Root cause: LFM3-1.7B is an INSTRUCT model — raw prompts on the base format loop; and official generation_config wants sampling (temp=0.1, top_k=50, rep_penalty=1.1), not greedy. With the shipped chat_template.jinja + official sampling, both ckpts emit think-tagged reasoning + code blocks + tool-call structure.

Verdict after honest A/B (scripts/compare_distill250.py): base ≈ distill250 — no visible regression, no visible gain. 250 steps of r8 LoRA on 1412 rows is smoke-scale; the pipeline (score→train→merge→pack→load→generate) is what got validated. Quality deltas need a real run.

Lesson: eval an instruct model through its template + its generation_config before concluding the weights are broken. Raw-prompt looping is a format artifact.

## 2026-11-04 - Spec-decode and consolidation measurement findings

- **specbound self-speculation rejected on lfm3**: early-exit draft + full-forward verify ran 8x SLOWER than standard (6.79s vs 0.82s for 64 tok) and DIVERGED from greedy output on the conv/attn hybrid — early-exit + state rollback doesn't pay on this arch. Do not enable.
- **Dormant feature objects**: _adaptive_spec/_suffix_spec/_faser/_lookahead_gate/_hotprefix/_pod_attention attach at activation but are never consumed by the production decode loop (only debug/bench scripts touch them). "active" in activation logs means "attached", not "in the token path". Real spec speed must come via the `decoding=` strategy (suffix_draft/specbound/external), not these flags.
- **compile matters on dense**: base eager 33.2 tok/s vs compiled+fused+graphed 48.4 tok/s (+46%). Packed int4 still fastest at 93 tok/s without compile.
- **consolidate_pairs CE gating**: min_fit_ce + replay_topk shipped. Replay selection DIRECTION (high vs low CE anchors) is a wash at K=2 — replay SIZE remains the dominant drift control.
- **post_ce>pre_ce on ForgeQuant-packed models**: consolidate fits on the _fq packed ckpt made the fit set WORSE (5.19→7.5) with 2-anchor replay — LoRA over ForgeQuantLinear dequant is unstable at these settings vs stable fits on int4-load models. Investigate before trusting consolidation on packed checkpoints.


## 2026-11-04 - BCG decode graphs silently corrupted SWA layers (k_pos frozen at capture)

Symptom: Spark-X2.6-1.7B (21/28 sliding-window layers) generated ~40 coherent tokens then CJK garbage under breakable CUDA graphs; eager was perfect.

Root cause: `GroupedQueryAttention.forward` SWA mask in pos_t (graph-replay) mode built `k_pos` from `_past_len = preallocated_cache.position` � a python int FROZEN at graph-capture time. `q_pos` correctly tracked the device pos_t tensor, so the window mask anchored at the capture position and drifted further off the live keys every step. LFM3 was hit too � its 5 SWA layers degraded subtly instead of garbageing (conv layers carried the output).

Fix: pos_t path on a linear buffer now derives `k_pos = arange(total_len)` � slots are position-indexed, so the mask is fully device-derived and replays correctly. Spark _fq with BCG: coherent, 65-76 tok/s at 2.5GB.

Lesson: any python int read inside a captured step is a capture-time constant � verify each position/length read is tensor-derived before trusting graph replay on a new arch.

## 2026-11-04 - Spark-X2.6-1.7B port (spark2_5 arch)

Port of XHToken/Spark-X2.6-1.7B into preset `spark_x26_1p7b`. Parity vs HF reference (bundled modeling_spark.py, fp32): max|dlogit| = 2.1e-5, identical top-5 � PASS.

New arch support needed (all landed in shared code paths):
- Slice-mode partial RoPE (`partial_rotary_mode="slice"` on RotaryEmbedding): Spark rotates the FIRST rope_dim=64 head dims of 256 with freqs computed for rope_dim � pairs (j, j+32). LFM3's zero-tail layout pairs (j, j+dim/2) � NOT equivalent. Both modes coexist via rope_layer_params.
- GeGLU FFN (`ffn_type="geglu"` ? SwiGLUFFN act_mode="gelu", exact erf matching HF ACT2FN["gelu"]).
- Keymap: `model.embedding.weight`?embed (not embed_tokens), fused `q_k_v_proj` split to q/k/v by row, `g_proj`?`gate_proj`.
- Tokenizer: gigatoken can't parse Spark's CJK Split pre-tokenizer regexes � `_load_hf_tokenizer` now returns plain HF tokenizer when the gigatoken wrap raises (was only catching ImportError).

Measured on RTX 5070 (bench_spark_port.py): _fq packed (1.42GB, 196 linears) loads 2.5GB, decodes 65-76 tok/s with BCG+prefix+chunked; dense activation-quant path coherent too but ~4min inductor warmup on the dequant path. Coherent thinking-style output, official sampling temp1.0/top_p0.95.

## 2026-11-04 - suffix_draft rejected on Spark too; SuffixDraftModel logit(1.0) inf crash

- SuffixDraftModel._predict_logits emitted torch.logit(1.0)=+inf when a
  suffix row hit 100% confidence ? NaN probs ? multinomial crash
  (surfaced on spark_x26_1p7b_fq's first spec run). Clamped p to 1-1e-6.
- suffix_draft on Spark thinking text: 21.8 tok/s vs 62-76 standard �
  still not worth it (acceptance too low for the round overhead; same
  rejection class as LFM3). StandardDecoding stays default.

### Spark proposer: unbounded <	hink> starves code budget (fixed)
Spark thinks 2-4k tokens open-ended; inside evolve's max_gen_tokens the
reasoning never closes so _extract_code finds no fence - 0/10 gated on
fast_softmax (stages: missing 7, syntax 3). Fix: think_budget (default
512) two-pass generation - bound the think, force </think>+` ` `python
continuation, return tagged raw so extraction strips reasoning.
Bounded: 8/14 gated (57%%); think-off: 6/9 (67%%); both produce REAL
gate rejects (dup/incomplete) not extraction failures.

