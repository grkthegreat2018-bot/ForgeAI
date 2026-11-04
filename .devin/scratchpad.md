# Rocinante-X-12B → ForgeAI Port — Working Plan

## Goal
Port `D:/LMstudio/Models/TheDrummer/Rocinante-X-12B-v1-GGUF/Rocinante-X-12B-v1b-Q4_K_M.gguf`
(Mistral-Nemo arch: llama gguf arch, vocab 131072, d 5120, L40, 32Q/8KV, head_dim 128,
ffn 14336, theta 1e6, eps 1e-5, untied head, tekken tokenizer, ctx 131072) into ForgeAI:

- RP quality ≈ original (verified vs llama_cpp logits — ground truth)
- Speed/size → LFM2.5-class experience: packed int4 weights, compile, bounded decode cost
- Context-size independence (FluxLM tech): O(1) VRAM + O(1) RAM vs context length

## Key facts discovered
- `_map_gguf_to_forge` exists (engine_common.py:244) — name map only, NO rope un-permute
- `_load_gguf_checkpoint` (engine_checkpoints.py:815) is buggy stub: rope_base=10000 hardcoded,
  no head_dim, no q/k un-permute, fp16 full-load (OOM 12B), hand-dequant Q4_K "simplified" (wrong)
- HF-name checkpoints auto-remap via `ModelLoader._remap_hf_keys` (loader.py:346) — used by
  qwen/llama presets. So convert GGUF → HF-style safetensors once = canonical path.
- RoPE in layers.py:50 = HF rotate_half convention → GGUF interleaved q/k rows need
  inverse permute: w.reshape(h,2,hd//2,d).transpose(1,2).reshape
- `compact_past_kv` bridge (engine_generation._maybe_compact_past_kv :989) fires in
  _prefill/_decode_tokens/_chunked_prefill — NOT in StandardDecoding.generate loop.
  → attach via `model._forge_compact_hook`, 2-line hook in StandardDecoding.
- logits_processor + step_observer hooks exist in _decode_tokens (:1165, :1181) — sidecar
  lift/observe plug points; same model._forge_suffix_mem attribute read in StandardDecoding.
- BeaconKVCache (kv/beacon_kv.py) already = sink + window + scored-middle compact_past_kv.
- Tokenizer: no mistral_common. llama_cpp 0.3.34 vocab_only=True = exact tekken (0.6s load).
- Decoding zoo: ngram_speculative/uno (prompt-lookup drafts) exist; StandardDecoding default.
- quantize_model_w4 = torchao Int4WeightOnly g128 packed (stays packed, fused mm).
  embed+head NOT targeted (not nn.Linear-targeted names) — stays bf16 (2.7GB).

## Design decisions
- Preset `rocinante_12b` (root): vocab 131072, d 5120, L40, 32/8, head_dim=128,
  inter 14336, eps 1e-5, theta 1e6, untied, max_seq_len 131072.
- Converter: gguf pkg dequant → bf16 HF-name safetensors (`research/checkpoints/rocinante_x_12b/`).
  Converter ALSO emits the tokenizer? No — tokenizer = llama_cpp vocab_only wrapper
  `GGUFTekkenTokenizer` in forge_loader.py; `from_checkpoint(tokenizer_path="gguf:<path>")`.
- Context independence =
    a) `flux` KV strategy = BeaconKVCache subclass, sink=8 window=2048 budget=6144
       (O(1) VRAM; compressed RoPE positions stay in-range forever — StreamingLLM semantics)
    b) `SuffixMemory` (forge/model/ngram_mem.py extension): bounded sparse
       {order→{suffix_hash→{tok:count}}} + decay halving + observe()+lift_last()+draft()
       — verbatim recall over whole stream even after KV eviction (the FluxLM property).
    c) hook into StandardDecoding + _decode_tokens via model._forge_suffix_mem.
- Speed: load_quantize="int4" (packed W4G128 ≈ 5.9GB) + optional int8 head;
  activate(kv_cache="flux", use_compile=T, use_fused_qk_norm_rope_cache=T,
  use_chunked_prefill=T) + SuffixMemory(gate tuned ~0.5-1.0).
- Honest note: 12B dense can't literally reach LFM2.5-1.2B file size; target ~6-7GB
  resident + flat memory vs context + fastest decode w/o training.

## Files
1. forge/config.py — +rocinante_12b preset
2. scripts/convert_gguf_forge.py — NEW converter (gguf pkg, unpermute, HF names, safetensors)
3. forge/engine/forge_loader.py — +GGUFTekkenTokenizer (llama_cpp vocab_only adapter)
4. forge/model/ngram_mem.py — +SuffixMemory (bounded sparse, observe/lift_last/draft)
5. forge/engine/kv/flux_kv.py — NEW FluxKVCache(BeaconKVCache) tuned + register "flux"
6. forge/engine/engine_activation.py — set model._forge_compact_hook after kv build;
   "flux" in _KV_FALLBACK_CHAIN; kv_cache ctor kwargs passthrough for flux
7. forge/engine/engine_common.py — SamplingParams.suffix_gate
8. forge/engine/engine_generation.py — enable_suffix_memory API; _decode_tokens hooks;
   from_checkpoint gguf-tokenizer dispatch (engine_checkpoints.py:88 area)
9. forge/engine/decoding.py — StandardDecoding compact+suffix hooks
10. tests/unit/test_rocinante_port.py — permute, suffix mem, flux kv, preset
11. scripts/bench_rocinante.py — parity vs llama_cpp, tok/s, VRAM, long-ctx probe
12. AGENTS.md — document R65 port

## VRAM budget (RTX 5070 12GB)
weights int4 projs ≈5.9GB + bf16 embed/head 2.7GB = 8.6GB
KV flux budget 6144tok × 40L × 8kv × 128hd × 2 (k+v) × 2B = 1.0GB
sidecar ~50MB host RAM (bounded), activations ~0.5GB → ~10.3GB peak. Fits.

## PIVOT (2026-10-18): distill -> Gemma-4-12B-OBLITERATED student
User decision: instead of pure port, train Gemma-4-12B-OBLITERATED on Roc patterns.
Rationale (user): gemma4 decent at RP, lacks ctx issues (hybrid SWA), better stepping
stone, faster gen, smaller KV footprint. Plan: merge both strengths, then improve arch.

### Rocinante port status -> SHELVED (parity bug unresolved)
- Converted ckpt: research/checkpoints/rocinante_x_12b (22.8GB bf16 safetensors, 363 tensors)
- Unit tests pass (12), key/shape match perfect, packed W4 loads (5.08GB), tokenizer adapter works
- Parity FAIL: bf16 CPU forward vs llama.cpp => corr~-0.16 unpermuted, ~-0.77 re-permuted
- 1tok bisect (pos 0, RoPE=identity): embed/head rows match bf16 rounding; manual lazy
  forward in progress when shelved. Next step if resumed: layerwise hidden-state bisect
  (need llama.cpp internals or compare manual-forward logits - script .devin/parity_1tok.py)
- Suspects: gguf dequant layout detail, norm eps handling, or Mistral-Nemo quirk
- All port code stays: rocinante_12b preset, convert_gguf_forge.py, GGUFTekkenTokenizer,
  SuffixMemory, FluxKVCache, suffix_gate wiring - REUSABLE for any llama-arch GGUF.

### Gemma-4-12B-OBLITERATED spec (from GGUF + HF config.json)
- arch: gemma4_unified (multimodal: vision+audio+text towers); saved w/ transformers 5.10.0.dev0
- text: 48L, d=3840, 16 heads, kv=8, head_dim=256 (global_head_dim 512),
  attn_k_eq_v=true (8 full-attn layers share K==V -> only 40 attn_v tensors),
  40 sliding_attention (win=1024) + 8 full_attention (every 6th layer),
  dual rope (global: proportional partial_rotary 0.25 theta 1e6; swa: theta 1e4),
  gelu_pytorch_tanh MLP (NOT swiglu), inter 15360, softcap 30, rms_eps 1e-6,
  vocab 262144, tied embeds, QK-norm + post-attn/post-ffw norms + layer_output_scale
- HF repo OBLITERATUS/Gemma-4-12B-OBLITERATED has model.safetensors 23.92GB bf16
  + tokenizer.json + chat_template.jinja -> NO GGUF dequant needed for student
- transformers 4.57.6 (Forge venv) CANNOT load gemma4 -> separate training venv w/ 5.20.1
- D: free 2242GB

### Distillation plan
1. teacher = Roc GGUF in llama.cpp (GPU) -> RP transcripts JSONL from prompt bank
   (I write infra + mechanical prompt seeds; data authored by the model, not me)
2. student = OBLITERATUS safetensors, text submodel only (drop vision/audio)
3. train = QLoRA (bnb 4bit + peft LoRA) or unsloth on RTX 5070 12GB; ~4-8k ctx
4. merge LoRA -> bf16 -> convert_hf_to_gguf for LM Studio; keep HF for Forge later
5. eval = held-out prompts side-by-side + student ppl on Roc outputs
6. later: arch improvements (Forge port of gemma4 for flux-KV/suffix-mem, or transplant)

---

## EXP-1: Broad LLM architecture experiments on Qwen2.5-0.5B (ForgeEngine)

Session output: forge/exp/{harness,ncm,loopdepth,diffdraft,dynscale,abshead,ndp}.py + scripts/exp_*.py + .devin/exp_results.jsonl + tests/unit/test_exp_*.py

### Infra fixes (real bugs, kept)
- forge/config.py qwen25_05b: tie_word_embeddings=True (checkpoint has no lm_head; random head gave NLL=11.94)
- research/tokenizer_cache.py _resolve_path: HF repo-ids no longer redirect to 65k ForgeLM tokenizer (was NLL=9.99). Test: tests/unit/test_tokenizer_resolve.py
- Correct baseline: Forge NLL 2.9959, HF 3.0421, load ~3.7s, weights 0.94GB bf16

### Results (all logged to .devin/exp_results.jsonl)

MATRIX (quant x decode x KV): decode/KV cells 68-81 tok/s flat (dola|rotorquant 81.4 best); batched|snapkv anomaly 21.6. Quant all SLOWER at 0.5B on RTX5070 (small-model/kernel-launch bound): nvfp4 62.4, forge_quant 60.8, int4 38.1, int8 25.1, w8a8 22.6. Quality: int8/w8a8 dNLL +0.02, int4/nvfp4 +0.40, forge_quant +0.47. fp8 = no-op (0 layers quantized). ternary_ptq broken (missing collect_activations import in novel_quant_r48) -> awq_fp4 fallback. VERDICT: bf16+standard KV is fastest; quant only for memory.

TRAINESS (E-trainless): TitanMemory+NGramMemory on frozen Qwen, lossless attach delta=0.0. Held-out NLL: 3.6094->3.5913/3.5953 (24/8 mem layers, +0.014-0.018). Verbatim: +0.40-0.49 NLL (24 layers better). Ingest 1.3-2.3s. SuffixMemory engine path works. Takeaway: fast-weight memory gives real train-less verbatim recall; held-out gain marginal at 6 docs.

LOOP (E-loop): recurrent-depth over last 3 blocks. Zero-init gates = lossless (tested bit-exact; bug found: gate was wrongly applied to pass 0, deleting blocks). Untrained looping DEGRADES NLL: 1it=2.99, 2it=3.12, 4it=3.93, 8it=5.82 � deltas decay but plateau ~0.049 (no fixed point). Gate-trained (8.5s): 3.0352 vs 3.1178 ungated @2it. Decode ~21ms/tok either way (loop region small). Takeaway: looped-transformer needs training, not free compute.

NCM (E-ncm): CfC cells on last 4 blocks, lossless attach (delta=0.0, 1-line dtype fix: .to(p.dtype)). Trains to loss 0.004 on 4 docs but eval NLL +4.39 � catastrophic overfit. Timescales stay ~1.0. Takeaway: side-path overfits tiny data; needs regularization/data scale.

DIFFUSION+MEDUSA (E-diffusion/E-speed): forge/exp/diffdraft.py � DDIM denoiser proposes K=4 tokens in parallel, AR model spec-verifies (greedy-faithful: KV truncate to ctx+accepted, corrected token + fixup forward). Diffusion head: acc 27.6% per slot after 300 steps, accept_rate 3-8%, tok/fwd 0.53-0.59, fidelity match mostly True (1 bf16 near-tie flip /96 tok � inherent to batched verify). MedusaDraftHead (K delta-MLPs + shared frozen lm_head, same verify): acc->1.0 (overfit 6 docs), accept 15-28%, tok/fwd 0.65-0.75, match=True both. Speed numbers noisy (greedy measured 8.4-57 depending on run state). Takeaway: mechanism correct; draft quality is everything � medusa >> diffusion at this scale, still net-negative speed on frozen small model.

DYNSCALE (E-dynscale, subagent): HeadGate on out_proj input (B,T,H,Dh), EMA saliency, lossless tau=0. Free heads: 1.5% @dNLL-0.003, 4.5% @+0.13, 15% @+2.3, ~50% fatal. Live gating costs more than it saves: 57.8-71.8 tok/s vs 86 baseline (EMA tolist sync). EarlyExit: 4.5 depth=288 tok/s @dNLL+13; conservative exits slower than full depth (per-block check ~1ms over V=152k). DiagTempScaler (Adam not AdamW � decay killed gains): recovers +0.139 NLL post-gating. Takeaway: Qwen-0.5B attention heads have ~15% slack at NLL cost, but Python-side gating overhead kills the win � needs fused/kernel-level gating.

ABSHEAD (E-abshead, subagent): LatentThoughtHead predicts h_{t+4} � cos 0.668 vs 0.616 persistence (held-out, real signal). Zero-init gate lossless. Injecting as residual readout hint: gate->0.074, dNLL -0.0004 (wash); joint train +0.163 (overfit hurts). Takeaway: future-latent is learnable but frozen lm_head can't exploit it.

NDP (E-ndp): forge/exp/ndp.py � exact online-softmax split attention: host computes (m,s,acc) partials on far KV pages, GPU combines. Exact to bf16 (err 5e-4). Transfer/step: 3.7KB vs 2-8MB migrate (567-2270x less). BUT: wins vs migration only at T_far=4k (3.0 vs 4.0ms); loses at 16k (20.3 vs 10.7ms � CPU attention slower than PCIe). Takeaway: NDP break-even where host compute = transfer time; practical design = retrieve-subset + NDP on selected pages.

DECISION (E-decision, subagent): embedded decision head inside model � acc 0.54, controller works, ~14% overhead.

PREVTOK (E-prevtok): head training broken (acc 0.0, loss ~1e5 � likely target indexing/vocab bug); 544MB head = untied full-vocab linear, should tie to embed. Subagent fixing.

### Cross-cutting findings
- Side-path convention (zero-init gate + cached gate-zero check) WORKS: every attach verified lossless delta=0.0.
- bf16 model + fp32 aux modules = dtype traps everywhere (mse_loss backward rejects mixed inputs; Linear layers too). Pattern: aux modules fp32 internally, cast at boundary (CastTitan shim) or create with p.dtype.
- Frozen-model experiments that TRAIN auxiliary params on ~6 docs: massive overfit is the norm (ncm loss->0.004, medusa acc->1.0). Held-out eval mandatory; NCM/medusa need 10-100x more data.
- Engine tokenizer: ids are int32 � CE needs .long().
- NGramMemory dense rows (594KB/suffix @151936) + deep-copy snapshots = OOM at ~3GB; use snapshot-free eval (eval forwards don't mutate live state).

### Remaining
- E-cxl tiered KV pool (subagent running)
- E-prevtok fix (subagent running)
- ternary_ptq collect_activations bug (unfixed, fallback works)
- batched|snapkv 21.6 tok/s anomaly
- Qwen2.5-Coder-0.5B validation pass

PREVTOK (E-prevtok, fixed + measured): PrevTokenHead = tied-embedding backward readout (adapter(h)@E^T, ~1.4M params; replaces broken 544MB untied head + bnb-AdamW8bit that diverged). Trained 400 steps: held-out prev-NLL 8.354 (uniform=11.93), top1 19.8%. Key finding: greedy generated text is backward-INCONSISTENT (prev-NLL 9.11 vs 8.35 on real text). SelfCorrectingDecoder modes: threshold tau=0.01/0.05/0.15 corrects 63-79% of tokens (joint dNLL -2.4/-1.9/-1.6 but +0.65-0.8 forwards/tok); RERANK (batched top-K retry in ONE forward) = best: joint dNLL -3.26 at forwards/tok 1.016 � essentially free self-correction. Rollback thrashes (head too weak on gen text). Retry must mask BOTH failed argmax AND prev_tok (x_t==x_{t-1} trivially passes backward check � degenerate loophole).

CXL (E-cxl, subagent): forge/exp/cxlmem.py � TieredKV rolling GPU hot window (1024 tok) + pinned-CPU cold pool, CxLLink side-stream nonblocking copies, online-softmax streaming attend, persistent prefetch staging; plus WeightTier (FFN offload). Real Qwen decode ctx=8K: full-GPU 57.8 tok/s vs fetch-all 50.3 (-13%), stream 22.5 (-61%), prefetch 27.5 (-52%, H2D dedup -93%), +8-FFN-offload 25.4 frees 209MB. KV GPU-resident FLAT 12.6MB at any ctx (8x saving @8K, 32x @32K); page-miss 0.028ms launch-bound. Bit-exact argmax vs baseline. Found+fixed overlapping in-place CUDA tier-shift UB in production cpu_kv_offload.py too. Honest caveat: simulated tier over PCIe; streaming hurts throughput (-52%) � the win is capacity, not speed.

Qwen2.5-Coder-0.5B: snapshot was an empty stub (.no_exist markers) � engine silently built ALL-RANDOM weights (NLL 11.94, second silent-fallback trap after the tokenizer). Downloaded real weights (HF_HUB_DISABLE_SYMLINKS=1 on Windows); NLL(code)=0.617, fib generation coherent. TODO later: engine should refuse load when snapshot dir has no safetensors.

PREVTOK v2 (agent a942c50a, post-fix rerun): root cause of original failure = bnb AdamW8bit diverged (~1e5) AND 136M untied head memorized train docs (held-out top1 0.127). Fix: tied-embed head (2.3M params, ckpt 544MB->12MB) + early-layer taps (x_{t-1} most decodable from early blocks: L2 top1 ~0.45 vs final ~0.17; rms(L2)+rms(L9) concat -> 0.55). Final: held-out prev_top1=0.531, prev_nll=3.81 (was 8.35); rerank djoint -0.896 (stronger head -> less room to correct; still positive). Peak VRAM 1.84GB. Lesson: prev-token info lives in EARLY layers, not the final hidden.

FLUXSCALE (same held=2291 critique tail, Qwen-BPE):
- docs190k: NLL 6.82, top1 0.172 (consistent across runs)
- pool 3.1M mixed-domain: NLL 7.13, top1 0.145 - 16x corpus WORSE
- LEAKED first run (eval files in pool): NLL 1.12 - verbatim recall, discarded
- Attribution: docs_w 7.38 / unif_w 7.13 / pool_w 7.13 -> hedge w NOT the cause
- docs_last (re-ingest): 6.96 -> recency recovers only partially
- Mechanism: LOW-ORDER cells are domain-SHARED (code-'the' + prose-'the' same row)
- Fix built: domain lanes - key ^= dommask[k] for orders<=4; ingest(tag/domain=)
  selects write lane; reads route via EMA probe-hit at probe order. _dommix_g is
  a [O] device input tensor (graph-safe lane swap, zero re-capture).
- Bank disk tier shipped: rows_cap overflow seals .fluxbank shards behind
  per-shard bloom (bounded index); RAM+disk union in cell(); post-restart
  load_disk_index(). 9 bank tests + 6 domain tests green.

## Self-speculative decode design (queued, 2026-10-20)

Goal: thousands of tok/s decode. Current: ~200-990/s observe,
~70-240/s learn. Wall = per-step Python+sync (~1-3ms), NOT GPU work
(graph replay alone = 0.05ms).

Measured profile (100 steps, V=151936, cuda_primary):
- _fused_step 2.6ms: bookkeep 1.27 + tolist 0.9 + rl_write 0.77 +
  lift 0.26. Now pipelined (1 pass/token learn=True).
- predict+push was 2.9ms: sparse-lift dense[V] arrays + nonzero +
  5 knn D2H reads killed -> 1.0ms.

Speculation blocker: verification needs model logits AT each drafted
position = per-position forward. _fused_seq already batches T
learn-steps with ONE readback and returns [T,V] logits -> the
verification pass exists for learn=True.

learn=True spec design:
  1. draft K tokens cheaply (order-2 argmax via _gt probe ~0.15ms ea,
     or top-1 of last logits + bigram continuation)
  2. mark = journal pos; logits = _fused_seq(draft)  [K,V]
  3. accepted = longest prefix argmax(logit_j)==draft[j+1]; emit
     accepted + argmax at divergence
  4. revert_since(mark) unlearns the whole row, re-learn accepted
     prefix+corrected via _fused_seq
  Cost model K=8, accept a=0.8: emit ~7.4, cost ~18ms -> ~410/s
  (~1.6x over pipelined). The revert+relearn duplicate caps gains.

learn=False spec needs a write-free observe-row primitive
(_observe_seq: push row to ring, per-position logits, no writes) -
a new _step_dev variant (skip all write sections). Rejected tail =
ring-slot restore (bounded: save _count, ring/H slots, device
mirrors, dom EMA). Same math on logits.
  This is THE path to >1000/s: observe steps skip writes entirely,
  per-step device cost ~0.3ms, one readback per row.
  Est: draft 1.5ms + observe_seq(K) ~K*0.4+1ms + host 1ms -> 
  emit ~aK+1 in ~5ms at K=8,a=0.8 -> ~1500/s.

Risk notes: revert fidelity at block granularity (partial-row revert
undoes whole block -> re-learn accepted prefix); dom-EMA state must
restore with ring; _fused_seq's epi-eviction granularity note (stale
votes <=T steps) already documented.

Alternative 'free' speedup already shipped: pipeline fused_step for
learn=True + sparse-lift pair-collection + knn sync batching.

## Teacher channel ("tch") � hybrid Qwen floor + Flux recall (2026-10-2x)

Architecture: Qwen2.5-0.5B fp16 as a STATIC read-only Hedge channel.
- No training, nothing journaled (state derives from the token stream,
  which IS journaled). Lazy: _tch() builds on first query; sync_to
  feeds ring tokens the teacher missed (incremental KV; rebuild tail
  after revert/load/ring-overflow).
- Predict path: _tch_lift adds w_rel*(log p_tch - log p_uni), clipped
  4*lift_clip � same additive contract as gsyn/bank lifts; device-only
  (CPU predict + a HF forward per token is unusable).
- Rewards: teacher-forced bulk_logprobs over each ingest block ->
  real_r (fires everywhere -> teacher competes on its OWN predictions).
- tch stays in _pmask for the FUSED device step (no HF in-graph) but
  is excluded from the bulk loop's closed-form abstention fill when a
  real_r entry exists (pm_fill = _pmask & ~real_r).

Bugs found+fixed (test_flux_teacher.py, 6 tests):
1. bulk_logprobs stored LOG-probs into a probs array (missing .exp)
   -> real_r went negative, tch decayed to floor. The unit test caught
   this class of bug (w must RISE when teacher predicts 0.9 on noise).
2. _pmask closed-form fill ran AFTER real_r -> clobbered tch rewards
   every block. Fixed via pm_fill exclusion.
3. [8192,151936] fp32 logits materialization = 5GB OOM -> sliced
   logsumexp+gather at PB=1024.
4. transformers >=4.5x needs DynamicCache not tuple; crop() native.
5. SDPA math backend [H,S,S] at 32K chunks = 56GB -> chunk 8192.

Cost: teacher bulk-forced pass ~1 forward/8192 tok (seconds);
eval-time learn feeds 1 HF fwd/token (~30-50ms) � teacher decode is
~20-30 tok/s vs flux-only ~200-990. Disable tch lift for fast gen.

Open: hybrid NLL on docs190k running; teacher-only baseline ~= Qwen
4.30 (known); flux-only 6.69 (known). Success = hybrid < 5.0.

## Teacher channel � RESULTS (same-session followup)

Bug chain fixed to get the floor working (each measured):
1. bulk_logprobs stored log-probs where rewards need probs (.exp)
2. _pmask closed-form abstention fill overwrote tch's real_r every
   bulk block -> pm_fill exclusion
3. DynamicCache.crop() keeps the FIRST tokens (prefix!) not the tail -
   hand tail-slice required; alone worth ~3.4 nats (8.89->5.45)
4. crop breaks RoPE: model derives positions from cache length -> pass
   absolute position_ids so retained keys' rotations stay correct
5. sliding window needs sinks: keep first 	eacher_sink=8 positions +
   newest window-sink (StreamingLLM semantics); _has_sink flag disables
   pinning after mid-stream rebuilds
6. lift semantics: wt*(lp - row) row-relative geometric blend,
   NO clip, NOT len*w, NOT lp-uni reference. At wt->1 mixture=teacher.
   (w_rel clip flattened ordering -> NLL 62.9; lp-uni reference leaked
   mass -> mixture 6.47 vs teacher 5.45; row-relative -> mixture BEATS
   teacher: 5.42 vs 5.45 @2048, 4.3167 vs 4.3300 @4096)
7. _rebuild pos=ep_start left pos<count -> double-fed tail; now
   pos=count-n absolute.

MEASURED (60K-token train, 2000-token held docs slice, Qwen-BPE):
- teacher@w4096+sink8: NLL 4.3300 ~= standalone Qwen 4.298 baseline
- mixture:              NLL 4.3167  top1 .3022  (memory adds signal)
- docs190k full eval running.

teacher_decode=False flag shipped (generate bypasses the HF fwd/token
cap ~20-30 tok/s; memory-only decode stays ~200-990).
Open: pool_dom-scale ingest; whether tch dominates Hedge rewards so
hard memory channels starve (w_tch~0.98 on docs - correct but leaves
little share for recall channels; consider teacher_w0 cap or reward
normalization if recall quality suffers).

## 2026-10-20 � wt15m flux-only baseline + self-play wiring

### wt15m+docs (FLUX-ONLY � snapshot had teacher_channel=False)
- Run loaded a stale flux-only snapshot (n=-1 bug); evaluated honest flux-only@15M:
  held_wt NLL 7.2258 / top1 0.1238 | held_docs NLL 8.3676 / top1 0.1371
  vram 7.03GB, ingest 457s (13.77M tok � 30K tok/s)
- Flux@15M on wikitext is WORSE than docs190k 6.82 � wikitext is
  hapax-heavy; bloom_admit=2 + bank spilling lose one-shot contexts.
- THE generalization gap (7.2 vs Qwen 4.3) is exactly what tch covers �
  hybrid 15M re-ingest relaunched (dfac94) with teacher_channel=True.
- Artifact kept: .tmp\wt15m_fluxonly.flux (7.66GB).

### Self-play loop audit + wiring (commit afcc357)
- Gaps found: flux mode has NO self-prompting (goals = procedural QA +
  web-extractive builders only); no file read tool; no sandbox
  enforcement at loop level (ToolSecurityManager exists engine-side,
  never consulted); per-epoch gate is a no-regression ratchet, not an
  advancement proof; distill_from not wired into loop.
- Implemented: `file` read-only tool (component-exact denylist,
  256KB cap, offline OK), `note` sandbox-journal tool (fixed path,
  text-only => cannot redirect writes), flux_file_root/flux_note_file
  config, flux_teacher_channel/flux_teacher_decode wiring.
- Denylist bug found+fixed: substring matching denied any path under a
  dir containing "secrets" (pytest test dirs!) � now exact-component.
- Still missing vs spec: model self-prompting (AZR proposal machinery
  is transformer-only), write-to-sandbox file editing, stop-on-
  advancement, execution sandbox for model code.

### Self-speculative decode (commit 37b6fb0)
- generate(speculate=K): read-only draft rollout (_draft_tokens �
  virtual suffix hashes over order cells, deepest-hit argmax
  follower), learn row via ONE _fused_seq, longest verified prefix,
  revert+relearn tail. GPU+learn+argmax only.
- Bit-exactness VERIFIED on GPU: speculate=6 == sequential argmax
  (4 tests green). stream len exact (prompt+emitted, no orphan
  learns).
- Notable: bc blocks are all-or-nothing so rejected tails revert
  whole row + relearn accepted prefix (extra pass, bounded).
- Budget cap: row len <= max_new-_step prevents output overshoot.
- Throughput estimate ~aK+1 tokens/pass vs 1 � measures vs GPU when
  15M run frees the card.

### Sandbox-jailed write tool (commit 60955d3)
- `write <relpath>::<text>` appends under flux_sandbox_dir
  (status_dir/flux_sandbox) � resolved path must stay inside jail
  (absolute/.. rejected, deny components blocked).  Closes the
  read-anything/write-sandbox contract for code experiments.
- Remaining spec gaps (documented in AGENTS.md): code-execution
  sandbox for model code, stop-on-proven-advancement sentinel.
- NOTE: T:-phase budget is 14 tokens � long write calls truncate;
  short writes work, long ones need a bump if write usage grows.

### 15M hybrid: teacher-weight collapse + floor fix (commit a12be79)
- Hybrid ingest completed: 13,770,997 tok, 5867s (~2.3K tok/s with
  teacher bulk rewards), peak VRAM 13.26GB (over 12GB target).
- COLLAPSE: AdaHedge drove w_tch->0.001, o2->0.981.  Trained-stream
  rewards always favour memorized recall -> teacher starves ->
  mixture degenerates to ~bigram.  Numbers: wt 7.42/doc 8.53 ~=
  flux-only 7.23/8.37.
- Reweight probes on the SAME snapshot (cells intact, only w changed;
  MUST rescale _wg device weights + _wgR, not just _w host mirror):
  floor 0.5 -> doc 6.07 | 0.7 -> doc 5.87 (0.85 eval contaminated -
  re-measured ingested tokens; discarded).
- floor 0.6 clean eval (one process, fresh slices):
  wt 4.9277/0.2453 | doc 5.7063/0.2301.
  wt-held 4.93 approaches Qwen floor 4.30.  Logged variant
  wt15m+docs-hybrid-floor0.6.
- FIX: teacher_w0 now = persistent share floor via _teacher_floor_
  after every hedge update (device/host/bulk + cond_hedge rows).
  Exact: teacher=w0, others rescaled to 1-w0.  Above floor untouched.
  Init share also set to w0 directly (was w0/len(channels)).
- Driver sets teacher_w0=0.6 for future 15M runs.
- OPEN: full re-ingest WITH floor during training may differ slightly
  (surprise->write strengths depend on p_mix which the floor changes)
  - cells mostly identical, expect similar numbers.
- OPEN: 13.26GB peak still >12GB - teacher_bulk_chunk already 4096;
  next lever = chunk 2048 or KV-pinning the sink window.

### Spec-decode bench result (2026-10-20) - NET LOSS, shelved
- bench_fluxspec.py (learn=True, V=4096, mixed periodic+random):
  spec=0: 398 tok/s | spec=4: 98 | spec=8: 198 | spec=12: 183.
- Bit-exactness still verified (test_flux_spec 4/4) - the bench's
  'identical:False' is an artifact: each generate() learns emitted
  tokens into the stream, so the later spec=0 reference sees a
  different context.
- Why it loses: pipelined learn=True decode is already ~2.5ms/tok;
  _draft_tokens does K host-side read_row+virtual-hash passes, then
  fused_seq + prefix verify + revert+relearn adds more host work
  than the ~K saved device passes.  Spec pays when per-step cost is
  HIGH (large tables, teacher_decode=True) - revisit if decode
  crosses >10ms/tok or if drafting moves device-side.

### Decode-time teacher bugs + fixes (commit 47a70e1) - chat probe
- Generation probes on wt15m.flux exposed 3 eval-invisible bugs:
  1) pipeline learn=True path used raw _fused_step logits -> ALL
     post-graph lifts skipped (teacher never voted at decode).
  2) convex blend can't reach teacher argmax under +20..40-nat
     corpus lifts -> teacher_hard_arb: ungated row:=lp, mem_scale=0.
  3) teacher KV window = ingest tail (research-notes register) ->
     generate() rebuilds teacher as prompt-only context.
- Also: dom lane now routes at prompt ingest (16-tok cadence missed
  short prompts; dom=1 IS prose lane, was fine - earlier misread).
- RESULT (learn=False, hard_arb, prompt-scoped teacher):
  ' Paris, France. The tower is...' | ' The capital of France is
  Paris.' | ' 12-5+3 = 10 apples.' | real fibonacci structure.
  = base-Qwen-0.5B serving quality reached.
- learn=True decode self-poisons: emitted tokens write cells ->
  gate fires on own output -> corpus-echo cascade.  Serving =
  learn=False always.
- Corpus-echo caveat: facts NOT in corpus can't be recalled
  (4242 probe) - expected: gate only fires on trained contexts.
  Self-play ingest of chat data is the path to real recall+chat.

---

# 2026-10-21 � Technique Landscape Scan (user-requested research sweep)

## ForgeAI coverage matrix

COVERED (implemented):
- Packed sequences: PackedSequenceDataset + varlen attn (R14), emit_cu_seqlens
- Decision model heads: forge/exp/decision.py, forge/engine/decision_head.py, decide.py
- TurboQuant KV: LloydMaxRotatedKV (novel_quant_r45.py), rotorquant.py
- LoRA family: bitnet_lora.py (BitNet+LoRA ~ BitLoRA-adjacent), dora.py, dlora.py,
  adapter_variants.py (PiSSA, AdaLoRA, rsLoRA), engine_lora.py
- Diffusion decoding: exp/diffdraft.py, decoding/uno.py, DiffusionBlocks+AdaLN
- MoE: forge/moe/ (moe.py, routers.py, airmoe_infinite/hotswap), moe_optim.py
- Sparse-V-in-spirit: SparseResidualINT3 (R44) sparse outlier correction (weights)
- Live online learning: LiveT + NGramMemory suffix-hash sidecar (FLUX)

DOCS-ONLY (referenced, not implemented):
- RWKV-7 (rd_round_40/49 plan candidate � NO key exists in forge/keys/attention/)
- JEPA/LLM-JEPA (rd_round_40 refs list)

UNTOUCHED: everything below.

## Research findings (3-agent web sweep)

### PEFT / adapter cluster
- BitLoRA: no arXiv. ESWA 311 (2026) � BitNet-1.58b base + FP16 LoRA adapters,
  federated on-device; >85% mem cut. bitnet_lora.py already covers the idea.
- Aletheia LoRA (arXiv 2604.15351, Apr 2026, single-author, 0 citations):
  5-pass gradient probe ranks layers -> LoRA on top-50% only, asymmetric rank.
  15-28% train speedup. Cheap add-on; Mamba-layer probing untested.
- JIT-LoRA (NOT arXiv � self-pub, HF Ex0bit/jit-lora, MLX-only): online LoRA
  backprop after EVERY response, mid-conversation knowledge injection.
  58.1% fact recall @ Qwen3.5-2B, 180 steps ~= 70s M4 Max, needs 5e-4 LR +
  >=33% replay ratio. STRONG overlap w/ LiveT � same online-learning goal,
  different surface (LoRA weights vs journaled states).
- MiCA (arXiv 2604.01694, in HF PEFT as init_lora_weights="mica"):
  Minor-Component Adaptation � SVD W=USV^T, fix B to LAST-r singular vectors,
  train only A. Inverse of PiSSA (already impl). 5.9x better knowledge
  acquisition vs LoRA, 6-60% params. Trivial to add to adapter_variants.py.
- LIFT (arXiv 2506.00772, ICML 2025): sparse FT of top-5% magnitude
  "Principal Weights", no low-rank. Beats full FT on reasoning, +20% source
  retention, tiny optimizer state. Weight-space -> arch-agnostic.
- REAP (arXiv 2510.13999, ICLR 2026): one-shot MoE expert pruning, saliency =
  gate-value x activation-norm. Near-lossless @50% prune (Qwen3-Coder-480B).
  Applies to forge/moe/ � saves weight mem not KV.
- AoE (arXiv 2501.13074, ICML 2025): Autonomy-of-Experts � delete router;
  experts self-rank via low-rank pre-activation norms. Pretrain-time only.
- Per-Layer Embeddings (PLE): no standalone paper; Gemma 3n/4 technique.
  Per-layer 256-dim tables kept in host RAM, fetched per layer -> E2B 5.1B raw
  runs at ~1.9B effective. Train-from-scratch only; fits mixed CPU/GPU mandate.

### Architecture cluster
- SparseStack = WINA (arXiv 2505.19427, ICLR'26; SparseStack = internal name,
  same authors � unverified equivalence). Training-free activation sparsity:
  hidden-state mag x weight-col l2-norm gating. +2.94% avg over TEAL at equal
  sparsity. Needs sparse kernels for real speedup.
- LaminarNet (Zenodo Mar 2026, single-author, NO peer review): O(N) "Geometric
  Drift Field" + Cross-Stratum Routing multi-rate strata. 49M claims -38% ppl.
- R-CLA (arXiv 2604.22782, Apple, ICLR'26 TTU): random cross-layer KV during
  training -> robust to arbitrary depth-wise KV sharing at deploy. CLA base =
  2x KV cut vs MQA (2405.12981). Train-time; halves KV cache.
- MODUS (arXiv 2607.25948): 15B decoder-only any-to-any multimodal � SKIP.
- AION-1 (arXiv 2510.17960): astronomy omnimodal encoder � domain mismatch.
- TRM (arXiv 2510.04871, Samsung): single 2-layer net, recursive z/y refinement
  <=16 steps. 7M params -> 45% ARC-AGI-1, 8% ARC-AGI-2, beats o3-mini. Trains
  from ~1k examples. Trivially fits 12GB.
- OpenMythos (github kyegomez, no paper): community recurrent-depth transformer
  (prelude -> looped block -> coda, MLA<->GQA, sparse MoE FFN). Unproven; the
  recurrent-depth pattern itself is the same family as TRM.
- RWKV-8 ROSA (no arXiv; RWKV-LM RWKV-v8/): Rapid Online Suffix Automaton �
  PARAMETER-FREE neurosymbolic exact-match propagator. O(n) online suffix
  automaton predicts x[j+1] from longest suffix match. 100% MQAR/NIAH by
  construction; demos: 40K params 99.8% digit-reversal. DIRECT analog of
  ForgeAI n-gram suffix-hash sidecars � ROSA is the principled version.
- Stacked sparsity = likely same paper as SparseStack/WINA (unverified).

### Systems / misc cluster
- Packed-seq metadata caching (Unsloth+NVIDIA, PR unslothai/unsloth#4243):
  boundary metadata (cu_seqlens, max_seqlen) is per-BATCH not per-layer;
  caching it removes ~4N D2H syncs -> 14.3% per-batch speedup on QLoRA SFT.
  Check: does our varlen path rebuild cu_seqlens per layer? Cheap win.
- ESP32-S3: esp-dl/esp-nn int8/16 kernels; esp32-llm 260K @32.8 tok/s;
  slvDev/esp32-ai 28.9M @9.88 tok/s (4-bit, PLE-style flash streaming).
  Orthogonal deploy target; <=30M params.
- Infinite diffusion: (a) InfiniteDiffusion arXiv 2512.08309 (terrain, training-
  free unbounded); (b) BD3-LM arXiv 2503.09573 ICLR'25 Oral � block-diffusion
  LM, arbitrary length + KV cache + parallel sampling. Our Uno/diffdraft are
  diffusion-ASSISTED AR; BD3-LM is true block-diffusion LM.
- LLM-JEPA (arXiv 2509.14252, ICLR'26): aux loss predicting EMA target-encoder
  embed of an alternate view. +8-15pp GSM8K/TURK on Llama3-8B; ~2x fwd cost.
  DLLM-JEPA (2606.00091) for masked-diffusion LMs; ER-JEPA (2609.36952).
- Spiking transformers: Spikformer/Spike-driven (42-87x attn energy on neuro
  HW); SpikingBrain-7B (2509.05276) >100x TTFT @4M ctx, 69% spike sparsity.
  GPU-impractical (timestep unrolling); watch only.
- "Mojo beats CUDA": narrowly true � vendor-tuned B200 bf16 matmul 1772 vs
  cuBLAS 1763 TFLOPs; independent SC'25 shows ~87% of CUDA on stencil. NO
  Windows support (WSL only) � SKIP for this box.
- TurboQuant+ (community fork TheTom/turboquant_plus, not official):
  (i) V compresses nearly free � all loss is in K -> asymmetric K/V bits;
  (ii) protect first/last 2 layers recovers 37-91% gap;
  (iii) sparse-V dequant skip where attn<1e-6 -> +22.8% decode @32K, zero ppl
  delta, NIAH 9/9. DIRECT add-on to LloydMaxRotatedKV/rotorquant.
- Decentralized pretraining: INTELLECT-2 (2505.07291, 32B RL, TOPLOC verify,
  ~590Mb/s broadcast); Psyche/Consilience-40B (DisTrO/DeMo, Solana coord);
  DiLoCo = 500x less comm, ~14Gb/s for 8B. Hobbyist = contributor only.

## Highest-synergy picks for ForgeAI (ranked)
1. ROSA-style layer � we already built the machinery (ngram suffix-hash
   sidecar, FLUX cells). ROSA = exact-suffix-automaton channel; could be a
   new FLUX channel or an LLM-side near-free retrieval layer.
2. TurboQuant+ sparse-V dequant � drop-in on existing KV quant; +22.8% decode.
3. MiCA � PiSSAInitializer already does SVD; minor-subspace variant is ~30
   lines; PEFT shows 5.9x knowledge acquisition (relevant to proposer SFT).
4. Packed-seq metadata caching � check varlen path for per-layer cu_seqlens
   rebuilds; free ~14% SFT batch speedup.
5. TRM / recurrent-depth � separate small-net reasoning direction; complements
   the model-versioning chain only as an exp/, not a key.
6. REAP � applies iff forge/moe/ ships a trained MoE; one-shot 50% prune.
7. LIFT � sparse top-5% FT, tiny optimizer state; good paired w/ LiveT.
8. R-CLA � needs (continued) training but halves KV; candidate for a key.
9. LLM-JEPA � aux loss, ~2x fwd; plausible SFT/CPT add-on.
10. JIT-LoRA concepts � LiveT already does online learning better (journaled,
    exact-revert); the 5e-4 LR + 33% replay ratio recipe is worth noting for
    any future LoRA-surface live learning.
SKIP: MODUS (15B multimodal), AION (astro), Mojo (no Windows), spiking (needs
neuromorphic HW), ESP32 (<=30M deploy-only), decentralized pretraining
(contributor-only), OpenMythos (unproven code, but recurrent-depth motif
worth tracking), Aletheia (weak evidence, though cheap).

## Deep-dive details (2026-10-21, 3-agent second pass)

### PEFT cluster � algorithms
- MiCA: SVD W=USV^T, freeze B=U[:,-r:] (minor subspace), train A only, A=0 init.
  r=16-32, LR 5e-4-2e-3 (minor dirs are flat). q_proj/v_proj per paper.
  For KNOWLEDGE/CPT injection, NOT instruct/behavior. ~20 lines off
  PiSSAInitializer. HF PEFT: init_lora_weights="mica" freezes lora_B.
- LIFT: rank-128 SVD approx -> top-5% |W'| mask per tensor -> sparse AdamW on
  selected only (27GB->1.3GB opt states @7B). Mask RECOMPUTED every
  update_interval via fresh SVD. lm_head excluded. repo zihanghliu/LIFT
  sparseAdam.py. Port: persistent index buffers + flat moment vectors,
  scatter-add back. Open Q: moment carry-over across re-masking.
- Aletheia: 5 fwd+bwd probe batches, per-block grad-norm scores (chunks of 8
  blocks w/ requires_grad toggling), top-50% blocks get LoRA attn_r=16 mlp_r=64.
  Probe <2% of train time. Single-author, unvetted.
- REAP: S_j = mean over routed tokens of gate_j * ||expert_out_j|| (conditional
  mean � decouples firing freq). One-shot forward-only calibration ~1k samples.
  Post-prune: renorm surviving top-k gate weights to sum 1 (1.9% vs 2.6% drop).
  Slice expert weights + router cols. Grouped MoE: prune within groups.
  In llm-compressor as REAPPruningModifier.
- AoE: no router; gate branch factorized W_down(d x d_low, d_low~d/3) +
  W_up(d_low x d_wide). Per-token: C_i = x@W_down_i for ALL experts (one fused
  matmul), score = ||C_i||2, topk, weights = softmax(norms). Load balance
  emerges w/o aux loss. Pretrain-only; converts via SVD init of W_down/up.
- PLE (Gemma 3n): one Embedding(vocab=262144, L*256) gathered once per token
  (~15KB row bf16) -> + Linear(d -> L*256) context proj -> (ple+proj)/sqrt2.
  Per layer: gate=gelu(Linear(d->256)) * pli -> Linear(256->d) -> RMSNorm ->
  residual add. Table lives in host RAM; per-token row fetch is
  bandwidth-trivial at decode, heavy at long prefill. Scale-sign discrepancy:
  HF issue#45206 says x hidden^-0.5, vLLM uses hidden^+0.5 � check HF source.

### Architecture cluster � algorithms
- ROSA (RWKV-v8, no paper): online suffix automaton per discretized channel
  stream; y[i]=x[e+1] of longest prior suffix match. Demos: Emb_ROSA (embed
  match token), ROSA_1bit (C binary automata/layer, bit-flip-rescore
  pseudo-grad backward, slow), ROSA-QKV (SAM over k-stream, q walks suffix
  links, v retrieved at match end+1, learned scale e), ROSA_4bit (4ch->hex,
  C/4 automata, loss .25 vs .4). All impls pure-Python CPU � no official GPU
  kernel; wind_rosa/RASP community ports. ForgeAI fit: additive side-path key
  (gate=0 lossless) OR cheaper: NGramMemory IS the token-level analog � ROSA
  adds per-layer bit-stream retrieval. Sam state = dict, same per-seq state
  invariant as conv_state (R60-4).
- TRM: one 2L net, z<-f(x,y,z) x n=6 then y<-f(y,z); T=3 cycles, first T-1
  no_grad; deep-supervision + detach between sup steps (N_sup=16); halt=BCE
  scalar; EMA .999 (+7.5pts). 5M -> 87.4 Sudoku vs HRM-27M 55. CAVEAT
  2512.11847: ARC score collapses to 0% w/o per-puzzle-ID embed � transductive
  memorization is load-bearing. Fit: exp/ side-model or extend hyperloop_key
  (y,z dual-state + detach protocol); not a general-LM recipe.
- R-CLA: train-time per-layer Bernoulli(p~0.6): own KV vs uniform-random
  earlier layer's KV; layer1 always own. Inference: choose cache-set S
  (layer1 required), others read nearest cached predecessor -> arbitrary
  retention tiers, no retrain. FT F1 beats baseline even @100% (HotpotQA
  .203->.306). Pretrain degradation <2% @p=.75.
- OpenMythos (unproven code): prelude(2) -> looped block x T(16) -> coda(2);
  per-iter loop-index embed + per-iter LoRA + LTI h<-Ah+Be+out (spectral rho
  ~.357 stable) + ACT per-token halting + MoE FFN; KV namespaced loop_{t}
  (cache grows T x for loop block). Community repro: same-params looped beats
  dense (57.9 vs 70.0 ppl @8x flops) but DEPTH EXTRAPOL FAILS (U-shape @16
  loops). Skeleton = Huginn 2502.05171 (validated 3.5B/800B tokens). Mamba-in-
  loop subtlety: SSM state must be namespaced per iter or reset.
- BD3-LM (2503.09573, ICLR'25 oral): AR over blocks, mask-diffusion within
  block; block-causal attn (clean past + bidir current). Train: sample t_b per
  block, clean fwd caches KV, ONE vectorized concat [noisy;clean] fwd for all
  denoise logits; clipped noise schedule U[beta,omega] tuned per block size
  worth ~.3-1.0 ppl. KV cache = committed clean blocks only. Gen ppl 25.7-33.4
  (L'=4..16) vs AR 14.1 � best diffusion LM, still ~2x worse than AR. SSM
  can't denoise bidirectionally � Jamba variant = block-causal SSM + block-
  diag attn mask; pretrain-scale change.
- LaminarNet (Zenodo, unvetted): GDF = input-dependent unit-modulus rotation
  SSM (LRU-family, theta=tanh(Linear(x)), chunked cumsum-of-angles scan);
  v0.6.4 adds learned forget gates (not purely unit-modulus); CSR = parallel
  strata at rates (1,2,4), AvgPool down / nearest-up + dual gates, iterated
  per block. Claim -38.3% ppl @49M = ppl 44.2 vs unpublished baseline ~71 �
  plausible but unverified. Fit: "gdf" ltype/key ~1 file; CSR ~ attn_residual
  at multi-res. NOTE: forge/model/strata/ name collision � unrelated.

### Systems cluster � algorithms
- TurboQuant+ (fork TheTom/turboquant_plus; vLLM PR#38479 MERGED 2026-04-15,
  backend TurboQuantBackend, --kv-cache-dtype turboquant_k8v4; llama.cpp
  merged WHT/rotation kernels, codec still fork-only): WHT(x_hat*signflips)
  + norm + Lloyd-Max/PolarQuant idx. Findings: K precision dominates (turbo3K
  catastrophic ppl 3556 vs q8K+turbo3V 6.71); Sparse-V = skip V-dequant where
  attn<1e-6 per position in-kernel (GQA: skip iff ALL group weights<1e-6) ->
  +22.8% decode @32K, ~90% positions skipped, NIAH 7/9->9/9; BoundaryV =
  first2+last2 layers q8_0-V -> recovers 37-91% gap. head_dim=64 can break
  (WHT needs dims for CLT). Engine needs: separate K/V dtypes + per-layer
  dtype overrides + WHT sign vector consts. All CUDA-graph safe.
- Packed-metadata caching (Unsloth PR#4243, merged 2026-03-12): key on object
  identity of packed_seq_lengths per device; caches (lengths, cu_i32,
  max_seqlen, masks, seq_idx) -> 4N->4 syncs/fwd. DONE in ForgeAI: added
  _varlen_meta (weakref+_version keyed) in attention_ops.py � flash path now
  reuses cu_i32+max_seqlen; fallback reuses blocks for mask build. Tests in
  test_rd14_speedup.py::TestVarlenMetaCache (11 pass). Left-pad gotcha:
  position_ids-derived cu mis-segments � we use explicit cu_seqlens already.
- WINA (2505.19427 -> ICLR'26, microsoft/wina): gate |x_i|*||W[:,i]||2 topk
  per matrix; col norms precomputed once; uniform-K mode zero calibration.
  ~60-64% FLOP cut @65% sparsity; REAL speedup ~1.2-1.9x decode (bandwidth-
  bound), degrades at batch>=64. Triton GEMV in repo. Pairs w/ weight quant
  (skip dequant of unselected cols). CUDA-graph safe (static topk shape).
- LLM-JEPA: aux loss on paired views (NL<->regex, q<->SQL, problem<->soln);
  single shared encoder, last-token hidden = embed, NO EMA (conflicting
  secondary claim), predictor = special tokens in packed context, cosine
  dist, ~2x fwd. Needs paired-view data � no general text augmentation.
- JIT-LoRA recipe (MLX): r32 on q,v,out,down; Adam LR 5e-4; grad-clip 1;
  batch=1; <=15 epochs ~4; REPLAY >=33% of batch must be regularization data
  else catastrophic forgetting; r32 ~35 facts capacity; template-shared facts
  fail (18% recall). CUDA-graph note: update adapters between graphs or read
  adapter tensors inside graph.
- Decentralized: DisTrO 857x comm reduction; Psyche = full-replica + Solana;
  5070 verdict: INTELLECT-2 32B no (~19GB Q4), Psyche 40B no; feasible only
  for hypothetical <=1B runs (12GB borderline w/ 8-bit opt).
- ESP32-S3: PSRAM ~62MB/s measured => tok/s ~= 62MB / bytes-scanned; esp-dl
  has no fused attn/KV ops; winning pattern = hierarchical residency (flash
  XIP tables + PSRAM weights + SRAM activations), PLE-style = 28.9M @9.9-22.6
  tok/s. <=30-50M params practical.
- SpikingBrain-7B: GLA+SWA(4K) interleaved 1:1 converted from Qwen2.5-7B on
  ~150B tokens; adaptive-threshold spike coding on linear activations (69%
  sparsity); W8ASpike = fake-quant not true int8; no GGUF (GLAswa
  unsupported); CPU community ~14s/tok. Spike = energy play, skip on GPU;
  arch = linear-state + sliding-window KV strategies � both graph-friendly.

## Ranked candidates updated post-deep-dive
1. TurboQuant+ sparse-V + boundary-V + asymmetric K/V: drop-in on existing
   KVCacheStrategy; concrete rules verified; vLLM upstream merged = de-risked.
2. Packed-meta caching: DONE (this session, tests pass).
3. MiCA: ~20 lines via PiSSAInitializer; knowledge-injection SFT wins.
4. WINA uniform-K: topk(|x|*colnorm) gating on FFN GEMV; decode-only win,
   pairs w/ existing quant paths; needs sparse-gather GEMV to realize speedup.
5. ROSA-as-key: ngram_mem already does token-level; ROSA adds per-layer bit
   streams � CPU automata heavy; start as FLUX channel or 1-2 layer side-path.
6. REAP: one-shot prune for forge/moe � only matters once a trained MoE ckpt
   exists; llm-compressor impl reference.
7. TRM/hyperloop: recurrent-depth reasoning as exp/ model � bounded scope.
8. R-CLA: halve KV via train-time stochastic sharing � candidate for next
   model key (non-lossless vs prior preset; document delta per rule A).
9. LIFT: sparse top-5% FT; custom optimizer effort = medium.
10. PLE: train-scratch only; best mixed-CPU/GPU fit long-term.

## ForgeEvolve/CodeEvolve throughput teardown (2026-10-22)

Measured (forge_evolve.db, ~26h span):
- code_programs: 38701 rows; 35456 dup (91.6%), 1435 done, 1115 test-fail,
  361 missing, 303 syntax, 31 exec. 60 insights. ~25 rows/min, ~0.92 evals/min.
- Peak burst ~300 rows/min (proposer is NOT the whole problem; dedup is).

Loop stages + costs:
- LLMProposer: Qwen2.5-Coder-0.5B bf16 ~1GB; 448 max_gen serial under 1 engine
  lock; propose_batch only when room>=4 (burst<=12). ~5-15s/cand -> 4-12/min.
  Mutator docstring: 17/21 verbatim seed copies -> most LLM output = dup rows.
- Eval: subprocess.run(sys.executable _candidate_eval.py) PER candidate.
  Windows spawn ~0.5-1.5s + target import + test + bench + verify reps.
- Verify: r>=3 challenger + r incumbent re-evals = up to ~6 spawns/attempt.
- ForgeEvolve engine (mech6.log): 1000 prop/gen in ~0.03-0.1s (generators are
  NOT the bottleneck); score phase 8-14s per 200 evals (~20/s) dominates ~98%.
- evolver_tune domain: 20s/eval -> surrogate/quick-tier is the only fix.

Design target: 1k proposals/min evaluated, ~1 win/10min = 0.01% win rate.
Current yield: 60 insights/1435 evals ~ 4%/eval -> at 1k/min => ~40 wins/10min.
Constraint = eval throughput, NOT proposal quality or rate.

Architecture (propose -> screen -> evaluate, all async/batched/amortized):
1. Producer/consumer queue; LLM off critical path (async basin emitter).
2. Mutator fan-out: deterministic op x site enumeration keyed by
   (parent_hash, op, site) -> uniqueness by construction; dedup becomes free.
3. LLM fan-out amplification: 1 LLM parent -> 30-100 mutator variants.
   LLM budget ~20-100/min (batched B=32-64, prefix cache, ~300tok) not 1k.
4. Persistent eval worker pool: K spawned procs preload target once; feed
   candidates via pipe/file; kill+respawn on timeout (gpu-pool pattern).
   Removes per-candidate spawn+import. K=16 @ ~1.5s -> ~600/min evaluated.
5. Two-tier eval: screen in-parent (compile+names, ~ms); test-only in worker
   tier1; bench only test-passers; verify only near-incumbent, r=2, cached
   incumbent mean (re-bench incumbent only every ~N promos / load shift).
6. ForgeEvolve side: score only NEW frontier entries (surrogate changes
   slowly; rescore all every K gens); drop per-chunk empty_cache; keep
   quick_ratio/MCNS multi-fidelity as the screen tier.
7. VRAM: proposer int8 ~0.5GB resident; eval workers CPU-only unless torch
   detected (route to 1-2 GPU workers); total loop <2GB.

Consolidation option: make CodeEvolve a domain inside ForgeEvolve
(CodeDomain decode->program text) -> one frontier, one eval pool, one DB.
Qwen becomes an optional basin emitter on the shared frontier.

## ForgeAI Evolve Throughput Rebuild + RSI Loop (2026-10-22)

### Measured before/after (top_p_filter, fresh DB)
| Metric | Before | After |
|---|---|---|
| evals/min sustained | ~0.92 | ~23 (4-worker pool, CUDA bench ~1.7s) |
| dup-reject rate | 91.6% (35,456/38,701 rows) | ~6% |
| promoted | 60 insights / 26h | +23.2% verified win at eval 20 of 44 |
| eval path | subprocess-per-candidate (~1.5s spawn) | persistent workers, spawn amortized |

### What shipped
- `forge/evolution/eval_pool.py` — EvalPool: spawn-context persistent workers,
  per-worker 1-deep job queues, shared result queue, deadline kill+respawn,
  max_jobs self-recycle. `submit` nonblocking (None=saturated), `poll`,
  `eval_sync`, `eval_many` (parallel verify reps). CPU target -> 16 workers,
  CUDA target -> 4 (each torch-using worker ~1.5GB ctx).
- `Mutator.mutants_for(src, insp, max_family, seed_base)` — deterministic
  op×site enumeration. Per-site ops: literal.0-7 (site index into eligible
  float literals), funcify.{0,1}.{0-7} (direction × call site), masked_where.i,
  locals_bind rank 0-2, crossover seeded spans; singles; two-op compositions
  over top-5 singles. Seed-dependent ops fold `seed_base&7` into the sig so
  re-opened families yield fresh variants. Yields (op_sig, code); registers
  codes in _ops_by_code so note() credits ops identically to __call__.
  Canon-comparison suppresses unchanged trees.
- `CodeEvolveConfig.proposer="basin"`: `_basin_loop` thread emits batched
  LLM parents (LLMProposer.propose_batch, basin_batch=8) into _basin_q;
  `_propose_basin` drains (parent_h, op_sig)-keyed families, blocks 2.5s
  when dry and emitter alive. `_seen_ops` kills re-enumeration waste.
- `_verified` caches incumbent rep scores for `inc_stale` (48) evals.
- `scripts/rsi_evolve.py` — RSI campaign: round = all 7 LLM targets ×
  basin proposer → export_sft → finetune_proposer --mode lora (adapters
  + router) → next round hot-swaps them. --no-train, --min-new,
  --quiet-rounds, --minutes, --status .events.jsonl.

### Bugs found while building
- spawn workers re-import __main__ → test scripts need __main__ guard;
  fn_evolve.py/pytest already guarded.
- `_respawn` ValueError when worker replaced twice inside one deadline pass.
- Thin seeds (dry_penalties=0 variants, fast_softmax=2) have genuinely small
  enumerable space → mutator-only runs stall correctly; LLM basins are the
  fuel. Enumerable space per parent measured: min_k 11, top_p_filter 15,
  top_p_probs 12, others ~1-5.
- Seed eval needs 180s+ leash: worker boot (__main__ re-import chain) +
  per-worker CUDA context init while the basin LLM also loads.

### Remaining / next
- RSI driver not yet soak-tested end-to-end (LoRA hone step); smoke it
  with --no-train first, then short --lora-steps run.
- targets/ are all decode/quant micro-ops; for "scientific" scope, add
  targets wrapping forge/model attention, KV-cache, optimizer surfaces
  (each needs careful test()+bench()).
- Per-candidate VRAM cap: CUDA pool sized 4 by heuristic; _probe_load
  still throttles _limit via _cuda_cap.
- 1k proposals/min needs ~24+ eval workers on ~1.5s evals or cheaper
  benches; current honest ceiling ~100-300/min for CUDA targets at
  4-8 workers (VRAM-bound).


## Promotion ratchet � root cause + fix (session cont.)
- Bug: `_handle_result` gated on first-eval score vs incumbent_score, but stored
  verify challenger_mean as new incumbent. Under worker+LLM GPU contention the
  incumbent re-measure inflated ~26%; regressions promoted repeatedly
  (quantize_4bit 0.294 -> 0.345 -> 0.703). dry_penalties same (102 -> 88 us
  "wins" all worse than best-ever 84.9).
- Fix (code_evolve.py): challenger_mean must beat stored incumbent_score by
  margin before promotion -> bar monotonic within a run.
- Fix 2 (_resume): incumbent_score = max(fresh_remeasure, recorded score) ->
  inflated resume re-measure cannot lower bar across runs.
- Deleted 7 fake insight rows (id 61-67) from forge_evolve.db.
- Test: ratchet regression in test_code_evolve.py; 20/20 green.
- Booted 50-round campaign (shell f1a81e): rounds 1 targets all stagnating
  honestly (0 promos) vs prior fake ratchet � correct behavior on converged
  targets.

---

# 2026-10-22 � ENGINE_CRITIQUE_DB implementation pass + ranked research landed

## What shipped

- ~106 EC entries fixed (see DB statuses). Themes: spec-decode KV/recurrent
  rollback (shared `_spec_*` helpers in decoding.py, repair-replay vs
  pre-verify cache), strict strategy factories (EC-019), shared EOS set
  (EC-039), effective-state honesty (quantize/kv_bits/kv_cache record the
  APPLIED mode), cpu_offload placebo ? real compaction+context-cap (EC-007),
  GGUF metadata-driven arch params + act_kwargs passthrough + per-tensor
  streaming, FluxLM sleep/wake + memory_report diagnostics, chunked-prefill
  `_chunked_prefiller` wired into `_prefill`, session stop/skip_special
  plumbing, per-task serialization (EC-198), API-key middleware +
  /health+/metrics+/version/tokenize/detokenize (EC-058/065/086/subset-152),
  worker-thread SSE + producer-error propagation (EC-057/073), in-flight
  eviction guard (EC-199), batch cap + executor offload (EC-218),
  GUI/tool/web hardening (cmd allowlist, backup restore staging,
  SSRF private-host block, credential-read refusal, lorebook identity,
  path confinement).

## Bug of the pass (fixed)

- `send_task_message` per-task serialization used `asyncio.Lock` � waiters
  on a DIFFERENT event loop (starlette TestClient spawns a fresh portal
  loop per request) were stranded forever: `Lock.release()` wakes waiters
  via `call_soon`, which is not thread-safe and never wakes a foreign
  loop. Flaky ~80% of concurrent-post attempts; faulthandler showed the
  losing POST parked in `portal.call` with lock free + queue empty.
  Fix: `_task_locks` are `threading.Lock` now, contended acquire offloaded
  to `_task_lock_pool` (dedicated ThreadPoolExecutor) so waiters never
  block the loop or starve the default executor; cancellation rescue
  releases a lock that a parked worker wins after the awaiter is gone.
  Repro: `.devin/repro_task_race.py` � 0/15 ? 40/40 OK.
  Lesson: ANY asyncio primitive shared across loops is a landmine �
  FastAPI/TestClient/multi-portal hosts guarantee nothing about loop
  identity.

## Ranked research � done

- MiCA (rank 3): `MiCAInitializer` + `MiCALinear` in
  `forge/training/adapter_variants.py`. B = U[:, -r:] minor subspace,
  frozen as a BUFFER (not Parameter � can't leak into optimizer groups);
  A zero-init trainable; merge() folds back. 7 tests in
  test_r38_adapter_variants.py (37 total pass).
- TurboQuant+ (rank 1): `RotorQuantCache` gained `bits_v` (asymmetric
  K/V � K precision dominates), `layer_idx`/`n_layers`/`boundary_v_bits`
  (boundary-V for first/last 2 layers when instantiated per-layer), and
  `get_sparse_v(attn_weights, threshold=1e-6)` which genuinely SKIPS
  dequantizing V rows under threshold (returns zero rows ? attn@V
  unchanged within threshold�?v?). `RotorQuantKVCache` ctor takes
  bits/bits_v/boundary args; `build_kv_cache(**kwargs)` forwards ctor
  kwargs (unknown kwarg ? TypeError, no silent drop); ActivationConfig
  gains `kv_v_bits`/`kv_boundary_v_bits`; `_activate_kv_cache` finally
  passes `kv_bits` to rotorquant/hadamard_int4 (was a dead flag � the
  wrapper hardcoded bits=4). Also repaired latent interface bugs the
  wrapper had: `RotorQuantCache.append(k,v)` takes no position arg,
  `get()` no positions arg, and there was no `clear()` � the "rotorquant"
  strategy would have TypeError'd on first use (another dead-feature).
  tests/unit/test_turboquant_plus.py: 10/10 pass.
- Packed-meta caching: already DONE prior session (_varlen_meta).

## Not done / remaining

- EC-152 endpoint gaps: only subset landed (/version, /metrics, tokenize,
  detokenize, request-id, stream_options usage chunk). Remaining: /props,
  /apply-template, abort/drain, n>1 shared-prefill, penalties, logit_bias,
  response_format, backpressure, JIT load+TTL, GGUF hub pull, etc. � left
  `open`.
- EC-140 per-token host syncs: partially addressed (cumulative decode in
  generate_stream); remaining .item()/.tolist() syncs elsewhere � left
  open.
- ROSA (rank 5): NGramMemory already token-level analog; ROSA adds
  per-layer bit-stream automata � heavier, deferred.
- Fused dequant-GEMM (EC-145) still open � needs Triton/CUDA kernel work.
- Full pytest: 687+ pass; one flaky ordering test
  (test_evolution_frontier::test_no_duplicate_evals) passes standalone �
  shared-state pollution under load, not a regression from this pass.

## Verification anchors

- tests/unit/test_forge_server_fixes.py 23/23 (incl. concurrent POST
  ordering � repro script 40/40).
- tests/unit/test_tools_security_fixes.py 33/33;
  test_gui_hardening_ec.py 21/21.
- tests/unit/test_forge_engine_fixes.py + r39/r60 decoding: 93 pass.
- test_turboquant_plus.py 10/10; test_r38_adapter_variants.py 37/37.

---

# 2026-10-27 � ForgeEvolve RSI audit: LoRA hone was a silent no-op + research pass

## Model check (user asked: small Qwen, not ForgeLM V2)

- CodeEvolve proposer (`code_evolve.py:LLMProposer._model_id`): default =
  `forge/evolution/proposer_mla_moe` (Qwen2.5-Coder-0.5B ported to Forge
  GLA+MoE, `qwen25_05b_mla_moe`, ~994MB) ? HF `Qwen/Qwen2.5-Coder-0.5B`
  fallback. ForgeLM V2 only appears in forge_server.py defaults � NOT in
  the evolve path.
- `research/checkpoints/forgelm_v2_tokenizer` = LFM2.5 tokenizer only
  (vocab 65536) � used by the from-scratch 2-5M gen_model (opt-in
  `enable_compute_split`, off by default). No V2 weights involved.
- LLMDomainGenerator / CurriculumFineTuner / GenModelManager: dead-ish �
  never instantiated outside docstrings / opt-in flag.

## BUG FOUND + FIXED: RSI hone produced inert adapters

- `finetune_proposer --mode lora` trained via `sft_train --hf-model
  Qwen/Qwen2.5-Coder-0.5B` ? HF `Qwen2ForCausalLM` ? adapter keys
  `model.layers.N.self_attn.*.lora_adapter.*`.
- Deployed proposer = ported MLA+MoE model ? Forge keys
  `blocks.N.attn.*` (GLA: kv_down_proj/k_up_proj/v_up_proj/k_rope_proj;
  MoE experts w1/w2/w3).
- `engine_lora._load_lora_impl` copies `state[name]` on exact name match
  ? **0 tensors loaded**; lora_B zero-init ? adapter mathematically
  inert. Warning logged only. RSI "learning" was a placebo.
- Also: `LoRAManager` stale?retrain was unreachable (nothing ever called
  `reg.mark(stale=True)`); no val holdout / early stop in hone runs;
  LoRA lr was 2e-5 (merge-tuning scale, too low for 120-step adapters).

## Fixes (all verified)

- `sft_train.py`: NEW `--tokenizer` (points at port's Qwen tokenizer;
  canonical LFM2.5 default would have produced garbage token ids for the
  151936-vocab port) + `--lora-targets` (comma list; overrides the
  hardcoded GQA list).
- `finetune_proposer._lora_mode`: default base = same resolution as
  `LLMProposer._model_id` (ported dir else QWEN_CODER_ID);
  `_resolve_forge_base` ? sft_train runs `--checkpoint <dir> --config
  <forge_config> --tokenizer <dir>` (ForgeEngine path, never --hf-model)
  + `--lora-targets` per arch (mla_moe: q,out,kv_down,k_up,v_up,k_rope +
  experts w1/w2/w3; other: q,k,v,out,w_gate,w_up,w_down). Registry now
  records target_modules so deploy attaches the same set.
- Staleness wired: `_stale_adapter_tags(db)` � adapter-attributed
  children in code_programs with <5% child>parent win rate over =24 kids
  ? `reg.mark(stale=True)` ? manager emits "retrain" (fresh init).
- Overfit brakes: `--lora-val-every 15 --lora-val-size 0.15
  --lora-early-stop 2` passed to sft_train (held-out val + patience).
- Learn speed: `--lora-lr 1e-4` (was merge-mode 2e-5).
- `code_evolve._ensure_lora`: prints loud WARNING once per tag when
  `_load_lora_impl` returns 0 loaded � no more silent placebo.
- Smoke: qwen25_05b_mla_moe CPU build ? 432 adapters attached, router
  untouched; keys = `blocks.N.attn.*.lora_adapter.lora_{A,B}` = exactly
  what `_load_lora_impl` consumes.

## Research for the evolve process (user list)

- **OpenRSI / Frontis-MA1** (frontisai.github.io/OpenRSI, arXiv
  2607.28568): closest reference architecture � Draft/Improve/Debug/
  Crossover operators trained via execution-grounded SFT+RL, then
  composed into long-horizon search; runs on RTX 4090 12GB (same VRAM
  class). Transferables: (a) explicit operator split � we only have
  "Improve"; a **Debug** operator trained on failure?fix pairs could
  salvage near-miss candidates (we discard them); (b) eval-benchmark
  decontamination of SFT rows; (c) Evo-Max "benchmark-independent
  experience priors" � our code_insights � could be formalized.
- **AZR** (NeurIPS'25, LeapLabTHU): proposer+solver self-play with
  code-executor-verified rewards, zero external data; learnability
  reward = propose at the frontier of ability. Transferable: prefer SFT
  rows where the parent was HARD for the current proposer (recent
  near-misses) � frontier-of-ability weighting in export_sft.
- **PopuLoRA** (arXiv 2605.16727, vmax.ai): populations of LoRA adapters
  on a frozen shared base � literally our tagged-adapter registry.
  Transferable: (a) weight-space mutation + crossover between adapters
  (slerp/lerp of A/B matrices) = cheap diversity without retraining;
  (b) teacher/student asymmetry � separate proposer vs solver adapters.
- **NeoHorse-1** (TokenRhythm, 4B/9B, Qwen3.5 base): routing-harness ?
  capability-guided next-training-mixture. Our lora_router + tags are
  the same shape. NeoHorse-1-4B (~8GB bf16) fits 12GB � alternative
  proposer if we ever want a bigger one; repackaged key names though.
- **MiniCPM5-2B** (openbmb, Sep 2026): standard LlamaForCausalLM 2B
  dense, strong code/agent at size � viable proposer upgrade (~4x cost
  of 0.5B; still fits 12GB). Config `qwen`-style remap needed in
  ForgeEngine presets.
- **Flint** (Recursive SI, Oct 21 2026): 39.8B MoE/2.5B active, ~80GB
  BF16 / ~23GB NVFP4 � does NOT fit 12GB VRAM for the proposer role;
  notable only as AI-built-model existence proof (their Eureka Machine).
- **rsi_eval** (Danielxu0208): SDK/CLI that runs RSI candidates through
  real agent benchmarks (Terminal-Bench, SWE-bench Verified, LCB).
  Needs Docker+Codex � heavyweight; could serve as periodic OUTER
  honesty check that the honed proposer improves on tasks it never
  trained on. Not wired in.


## Round 2 research (EdgeRock / MemT / DuoAttention / MTP / warmup)

- **EdgeRock-4B** (rockmatrix-ai, arXiv 2610.10118): Qwen3.5-4B converted to bounded-memory arch — 24 recurrent (GDN) layers + 8 RMA-attention layers; local window 1024 + learned anchor bank 1024 records/KV-group committed every 256 tokens; fixed 72 MiB KV/seq (99.1% less than source), 82.9% AIME (96.3% retention), 6.42x batch throughput. Needs custom transformers fork + custom modeling — heavy to port, but the CONCEPT maps 1:1 to our beacon_kv/anchor strategies + GLA latent KV. Not a proposer candidate (4B, custom runtime).
- **MemT-1.0** (MemTensor, arXiv 2610.26825): memory-native LLM — Local Memory (dynamic matrix, per-turn GDN-2 token-level updates) + Hyper Memory (trainable qkv/erase/write/forget gates) + Mixture-of-Storage routing; updates in a single forward, no backprop. Nano=36B/3B-active on Qwen3.6-35B-A3B — too big, but the CONCEPT = exactly our LiveT (TitanMemory fast weights + GDN2 recurrent states + NGramMemory sidecar, journaled). We already implement this idea class.
- **DuoAttention** (ICLR'25, mit-han-lab): split heads into retrieval (full KV) vs streaming (constant-length KV); 2.55x MHA / 1.67x GQA mem cut, 2.18x/1.50x decode speedup. On GLA ports it is moot (latent KV already tiny); directly useful for any remaining full-GQA layers and for V2/V13 target models — implement as per-head KV budget in _maybe_compact_past_kv.
- **MTP**: already in codebase — forge/decoding/mtp.py MTPModule (shared lm_head, identity_init, n_heads=4, loss_weight~0.495); config use_mtp/mtp_n_heads/mtp_weight; sft_train has mtp arg plumbing. Gap = never enabled on the ported proposer.
- **Boot preload + cache warmup**: proposer_prewarm daemon thread exists (loads engine in background); use_chunked_prefix_cache=True already on. Gap = cache is cold at boot — first basin batch pays full prefill on the ~90%-shared prompt skeleton.

## Best recommended model: MiniCPM5-2B (proposer upgrade)

Architecture: standard LlamaForCausalLM — dense GQA + SwiGLU, 128K ctx, ~2.4B params, tied-ish head, BPE tokenizer. Best open code/agent model under 4B (beats Qwen2.5-Coder-7B on LiveCodeBench per vendor); fits 12GB with headroom; Llama-compat = ForgeEngine already has llama31/llama32 presets + _detect_config_from_header auto-pairing.

Plan: MiniCPM5-2B -> proposer_v2 (massive improvement recipe)
  A. Lossless port (reuse/extend scripts/port_qwen_mla_moe.py --src llama): GQA -> GLA latent-KV full-rank SVD (d_c=256), SwiGLU -> MoE-4 dense_bypass. Bit-exact parity test vs source. Preset minicpm5_2b_mla_moe + forge_convert.json.
  B. MTP: use_mtp=True on the port preset (MTPModule identity-init, n_heads=4, shared head); short MTP-only train pass on evolve SFT rows (freeze trunk, ~200 steps) -> spec-decode proposer gen ~1.5-2x.
  C. Memory-native proposer (MemT idea): wrap LLMProposer engine in LiveSession — ingest each promoted win into TitanMemory+NGram fast weights (bounded, journaled, revert-able). Proposer accumulates experience BETWEEN hone rounds; honesty preserved via mark/revert.
  D. Boot warmup: extend prewarm to prefill the basin prompt skeleton (shared system+template prefix) once -> first basin batch is all chunked-prefix-cache hits; pre-attach default LoRA tag to skip first-swap latency.
  E. VRAM: bf16 ~4.9GB weights + tiny latent KV + MTP ~15M -> ~5GB, leaves ~7GB for eval CUDA benches (vs 8-9GB needed for a 4B EdgeRock/NeoHorse option).
  F. A/B before default swap: acceptance rate + tok/s + proposal diversity vs proposer_mla_moe on same seeds; keep proposer_mla_moe as rollback.


## Overclock theory (proposer_v2 + evolve pipeline) — researched 2026-10-27

DECODE-SIDE SPEED (dominant cost: basin proposal generation)
- EAGLE-3 (SafeAILab, NeurIPS'25, arXiv 2503.01840): feature-level draft head fused across low/mid/high layers, direct token prediction (no feature loss), dynamic draft tree + tree-attn verify. Claims up to 6.5x, ~1.4x over EAGLE-2, 1.38x SGLang throughput @bs64; realistic 2-3x. Draft head ~1 decoder layer + head (~0.15-0.3B on 2B). Beats our MTPModule on acceptance; can coexist (MTP stays for loss signal, EAGLE head for draft). Train on OUR evolve SFT stream -> draft specializes to current campaign domain.
- NGram-lookahead / Jacobi decode: zero-train, zero-mem draft = reuse LiveT NGramMemory successor counts as lookahead candidates, verify in parallel. Code is n-gram-dense -> high acceptance. Self-improves as campaign ingests.
- Chain draft: keep old 0.5B ported proposer as standalone draft model for 2B target (+1GB, no training). Bridge until EAGLE-3 head is trained.
- CUDA-graph decode loop on ported arch (FluxLM already graphs; check transformer path) — at 0.5-2B the per-token launch overhead is the tax; ~1.2-1.4x.
- Sorted/continuous batching + shared-prefix dedup (prefix cache does dedup already; warmup prefills skeleton).
- Early stop at code fence — stop strings save decode tokens on every proposal.

COMPUTE-SIDE
- NVFP4 (SM120 native FP4 tensor cores): ~3.5x weight memory cut (4.9GB->~1.5GB), faster GEMMs; decode is bandwidth-bound so real speedup. CAVEAT: SM120 GeForce needs patched kernels/quant env (lna-lab blackwell-geforce-nvfp4-gemm repo; CUDA 13 + llm-compressor). We already list NVFP4 as a production path — verify SM120 kernel path works on 5070.
- SVDQuant trick: FP4 weights + bf16 low-rank (r 64-128) residual branch — quality ~bf16 at FP4 memory. We already own the SVD tooling from GLA port.
- MoE-bypass rethink: dense_bypass computes ALL 4 experts (4x FFN FLOPs). Options: (a) post-hone switch to top-1/2 routing -> 2-4x FFN cut; (b) fuse bypass into single kernel; (c) distill MoE->dense after specialization. Decide by measured speed/quality.
- 2:4 structured sparsity on FFN during hone — Blackwell sparse cores ~2x GEMM; quality risk, rank low.
- torch.compile single-token forward (HAS_TRITON gated; Windows Triton status uncertain).

MEMORY-SIDE
- NVFP4 weights (biggest): 4.9GB -> ~1.5GB.
- GLA latent KV: near-zero KV (planned). DuoAttention per-head budget for any residual GQA.
- lm_head/embed: MiniCPM vocab ~122K x d~2560 untied ~ 600MB bf16 — int8/fp8 head GEMM or hot-token low-rank.
- Prompt compression (LLMLingua-style) on the shared system skeleton at warmup — less prefill + less cached KV.
- LiveT surfaces + adapter pool: quantize TitanMemory fp16->int8 (minor, already small).
- Eval-worker arenas stay per-context; no MPS on Windows.

QUALITY-PER-BYTE
- On-policy distill (NeoHorse-style): teacher Qwen2.5-Coder-7B logits on evolve-domain prompts -> student MiniCPM5-2B. Offline, so teacher size irrelevant at deploy.
- RSI-draft co-evolution (novel): draft head trained on the same verified-win stream; draft+adapters co-adapt per tag — acceptance rises as campaign progresses. Possible per-tag draft LoRA.

END-STATE VRAM (proposer_v2, 12GB card): NVFP4 weights ~1.5GB + head/embed ~0.6GB + EAGLE head ~0.4GB + prefix cache 1GB + LiveT/adapters ~0.5GB = ~4GB -> ~8GB free for eval workers.
SPEED STACK (honest ranges, multiplicative where orthogonal): NVFP4 1.3-1.7x * EAGLE-3 2-3x * graphs 1.2-1.4x * prefill elimination -> ~4-8x proposal throughput vs bf16 vanilla.


## Probe run results (2026-10-27, scripts/overclock_probe.py)

BUGS FOUND+FIXED (verified):
1. forge/moe/moe.py dense_bypass: quantized experts (ForgeQuant/QuantizedLinear store q_weight+scales, no .weight) -> _get_weight None -> torch.stack crash. Fixed: generic per-expert forward fallback when weights unstackable. Test: test_moe.py::test_dense_bypass_quantized_experts.
2. forge/engine/quant/forge_quant.py _dequantize_dense: idx cropped to in_features BEFORE group reshape -> crash when in_features % group_size != 0 (MoE expert width 1216 = 4864/4). Fixed: crop moved to final line. Test: test_round32_quant.py::test_forge_quant_nondivisible_group_size.
3. forge/keys/attention/gla_key.py _identity: .item() per layer per forward = 24 GPU syncs + dynamo graph breaks per token; poisoned breakable-CUDA-graph capture. Fixed: _identity_cache in eval mode, invalidated by _load_from_state_dict + set_identity. Result: block fusion captures cleanly now.
4. forge/engine/engine_activation.py: ForgeQuant lazily mutates _cached_weight inside forward -> corrupts cudagraph tree manager. Fixed: warm all dequant caches post-quantize, pre-capture (steady-state mem unchanged — cache was always populated on first forward).

MEASURED (proposer_mla_moe, INT4+2bit sparse auto-quant, rotorquant 4-bit KV):
- Model: 441M params post-quant, 24 layers, d=896. Load 6.9-22s (compile-cache dependent).
- E2 layer-exit on code: top1 acc per layer — L0-15 ~0.004-0.02, L18 0.133, L21 0.333, L22 0.416, L23 0.463. Agreement w/final: L21 51%, L22 71%. => early-exit self-speculation DEAD END (predictions crystallize only in last ~3 layers). Final top-1 46.3% on code => base 0.5B quality IS the bottleneck; strengthens MiniCPM5-2B case.
- E3 ngram-lookahead (orders 2-8, greedy chain): mean 1.38 tok/forward, median 1, p90 2, max 9. => marginal; free add-on at best, not a draft substitute.
- E4 vocab: 5595/151936 unique tokens on repo code; top-4096 rows cover 97.4%, top-1024 80%. lm_head (136M = 30% of params!) is massively overprovisioned -> hot-head/small-vocab validated. MiniCPM head ~310M = 13% — same trick applies.
- KV math: GLA latent d_c=256 -> 24576 B/tok, 2x the GQA KV (12288). Latent dim is a parity-vs-memory tradeoff knob (d_c=128 halves KV at some parity cost).

REMAINING BUG (flagged, not fixed — engine infra):
- RoPE output + breakable-CUDA-graph tree manager conflict under torch.compile(reduce-overhead): 'accessing tensor output of CUDAGraphs overwritten' at layers.py:167 via gla_key.py:183. Breakable graph captured empty (decode_graphs: 0) but manager still asserts. Options: auto-disable breakable-cuda-graph when capture yields 0 graphs, or mark_step_begin around generate loop boundary. E1 decode tok/s still unmeasured — rerun probe after fix.
- Pre-existing unrelated failure: test_round32_quant.py::TestEngineDispatch::test_build_kv_cache_unknown_falls_back (build_kv_cache now raises on unknown strategy — stale test).


## RoPE/cudagraph fix + E1 decode number (2026-10-27)

ROOT CAUSE: engine compiled with mode=reduce-overhead -> cudagraph trees shared pool; decode loop feeds functional past_kv (run N outputs) into run N+1 -> 'overwritten by a subsequent run' at whatever tensor was read first (RoPE k, model output, etc.). ALSO stateful wrappers (seq_aware_split, rotorquant strategy) hold graph tensors.

FIXES:
- decoding.py: _clone_state_tree + _model_is_compiled; StandardDecoding clones logits+past_kv in-step after each compiled call (prefill + per-step). Protects any future cudagraph-on path; gated on compiled model only. Other strategies (NGram/Spec/generate_batch/stream/cascade) have the same pattern — flagged for the same treatment when used with compiled models.
- engine_activation.py: compile via options={triton.cudagraphs: False} (== reduce-overhead minus graph trees; mode+options are mutually exclusive). Kernel fusion kept; cudagraph-trees off engine-wide. Breakable-cuda-graph system still owns graph capture.
- engine_activation.py: ForgeQuantLinear dequant caches warmed at activation (prevents lazy _cached_weight mutation inside graphs + per-shape recompiles).
- gla_key.py: _identity cached in eval mode (killed 24 .item() graph breaks/token).
- forge_quant.py: pad-crop order fixed for in_features%group!=0 (MoE expert 1216).
- moe.py: dense_bypass per-expert fallback for quantized (non-stackable) expert weights.

RESULT: E1 decode 81.4 tok/s (was: hard crash; eager quantized fallback measured 39.0) — compile fusion ~2.1x. Still only 7% of the 672GB/s roofline -> decode is LAUNCH-OVERHEAD-bound. Next lever by far: static-KV + captured decode-step graph (breakable graph system captured 0 decode_graphs — that subsystem needs its own fix; with it, plausible 3-8x more).

REMAINING MINOR: ForgeQuantLinear.forward hits dynamo recompile_limit(8) — per-shape guards across 433 layers (896x896, 151936x896, 1216x896...). Wasted compiles then eager fallback. Fix candidate: @torch.compiler.disable on forward OR bump recompile_limit — deferred, harmless.


## V14-RSI round (2026-10-27)
- forgelm_v14_rsi preset: 2.465B params, ~1.26B active/step; conv16/mamba3-4/GLA-4 + MoE(8e top2+shared, d_ff=768) + GDN2(4h/128d all layers) + Titan(r128, [5,11,17,23]) + MTP-4-adapter + CopySpan + Verifier heads
- int4 ~1.23GB weights; bf16 peak 5.12GB; ~1.8k tok/s ceiling w/ MTPx2 @570GB/s (measured)
- Traps: gdn2_head_dim MUST be explicit (640-wide default=46M/layer); GLA rope bufs scale w/ max_seq_len; MTP vocab heads 671M->adapter 26M; Titan rank shrinks dxd buffer
- Bug fix: mamba3 out_proj fp32/bf16 mismatch; torch.compile unusable on this box (triton cache race + complex ops)
- Next: P1 port Qwen-Coder->GLA+MoE (port_qwen_mla_moe.py exists), distill substrate; P2 edit-SFT data from forge_evolve.db diffs + verifier BT on evolve_pairs; P3 engine decode registry + AirMoE expert prefetch (disk >200MB/s)



## BCG fix (2026-10-27, continued)

breakable_cuda_graph.capture_decode was structurally broken:
- Called model(input, position_ids=...) — ConfigurableResearchLLM.forward has no
  position_ids kwarg -> TypeError raised INSIDE torch.cuda.graph context ->
  cudaErrorStreamCaptureInvalidated corrupted the stream for the whole process
  (this is what killed block fusion + breakable capture + warmup at init).
- Even with correct kwargs it captured a CONTEXT-FREE forward — run_decode
  replay would emit tokens with empty context (silent wrongness, worse than 0).
Fixed: decode_step param now required (a static-KV one-token step fn); without
it capture logs why and skips — 0 graphs, zero stream corruption. Engine boots
clean: block fusion active, BCG honest no-op.
Note: run_decode was never wired — BCG was decorative + actively harmful.

REAL decode-graph path needs (the 'megakernel step'):
1. PreAllocatedKVCache.position -> device tensor; append -> index_copy_/scatter
   writes (python-int pos bakes into graph at capture).
2. Read full buffer + mask by pos tensor (paged-style) instead of [:pos] slices.
3. position_ids tensor routed to rope (rope already supports position_ids);
   _forward_impl computes it only from attention_mask today.
4. Wire decode_step into BCG.capture_decode + StandardDecoding replay branch.
Measured stake: eager/compiled decode 37-81 tok/s vs ~672GB/s roofline —
launch-overhead-bound; a captured step plausibly 3-8x.
- Pre-existing suite failures fixed along the way: activation_catalog missing kv_v_bits/kv_boundary_v_bits fields (added), gui_server tests broke under hub bind_loop model (test now drives a loop; watch fixtures bound). OPEN: test_flux.py::test_cuda_forward_deterministic fails deterministically � pre-existing, flux WIP uncommitted, NOT from V14 work (FluxLM doesn't use ConfigurableResearchLLM).

- V14 CUDA-graph decode WORKS: 108 tok/s bf16 BCG replay (vs 19 eager). Capture blockers fixed: MoE routed path -> fixed-shape gather (_w1s/_w2s/_w3s stacks, topk weight gather, N<=64); KroneckerEmbed._ids_to_bytes CPU alloc -> device-local. Headroom: int4 ~2x, MTP ~2-3x, CopySpan on edits.

- P1 started: qwen25_coder_3b_mla_moe preset added (d2048/L36/H16/KV2/GLA dc256/MoE 8x1376 dense_bypass), port_qwen_mla_moe.py gained --preset flag. Qwen2.5-Coder-3B downloading to .hf_cache (bg). TODO when done: run scripts/port_qwen_mla_moe.py --src <snapshot> --dst research/checkpoints/qwen25_coder_3b_mla_moe --preset qwen25_coder_3b_mla_moe --experts 8 --verify; VERIFY downloaded config.json matches preset (inter=11008, vocab=151936, tied=true) before port.


- R66 proposer static-KV decode graphs SHIPPED: PreAllocatedKVCache pos_t mode (enable_tensor_pos/sync/append_t index_copy_), GLA+GQA full-buffer reads under pos_t (masked by attention_bias), BCG capture_decode(decode_step=) refuses context-free captures, feature_registry wires cache+mask_buf at activation (compiled-step capture tried x3 then eager fallback), StandardDecoding._generate_static replays per token, gated on not _model_is_compiled. Capture fixes this session: torch.tensor(0.0,device=cuda) H2D-copies banned inside forward (moe.py:356 dense_bypass aux_loss -> x.new_zeros; llm.py:646 moe_aux_loss -> new_zeros); conv/SSM reset skipped when preallocated.position>0. Measured proposer_mla_moe RTX5070 int4+rotorquant: eager-KV 39, static-graph replay 76, compiled-functional 107.6 tok/s. Compiled-step capture blocked: fused mask+cumsum+index+masked_fill+SDPA kernel deterministically fails Triton codegen (tmp-dir FileNotFoundError in cache put; fresh cache dir + 3 retries same) � document as env-level dead end until triton version bump. Equivalence test: test_gqa_tensor_pos_equivalence (token-for-token + logits vs functional KV).
- BIG UNBLOCK: Windows MAX_PATH was killing Triton cache put() (133-char kernel names + long temp root). FIX: set TORCHINDUCTOR_CACHE_DIR=D:\tc\inductor and TRITON_CACHE_DIR=D:\tc\triton (short root). Verified: torch.compile+SDPA works. This unlocks the >107 tok/s compiled-step capture noted in AGENTS.md. NEXT: rerun V14 compiled_decode_step under these env vars (last attempt exited 1 with suppressed stderr � rerun WITHOUT 2>$null to see error; likely mamba3 complex-op unsupported by inductor -> may need eager-capture fallback for mamba layers or a real-scan triton kernel).
- PENDING: qwen25_coder_3b download to .hf_cache; then port w/ --preset qwen25_coder_3b_mla_moe --experts 8 --verify.


- R66 BREAKTHROUGH � bucketed fused decode graphs: 1258 tok/s vs 598 compiled-functional (2.1x, bit-identical, 256-tok decode, proposer_mla_moe int4+rotorquant). Two unlocks: (1) 'Triton cache race' was Windows MAX_PATH � ~200-char fused names + long cache base >260 -> deterministic tmp-dir FileNotFoundError in cache.put; configure.py now pins TRITON_CACHE_DIR to <=24-char drive-root (C:\tt, same as flux rule). (2) Full-8192-buffer attention in captured graph wasted ~30x KV reads at short ctx � KV-bucket graphs (512/1k/2k/4k/8k) read [:b] views of the SAME storage via cache.view_len; bucket switch = free (views share storage). decode_graphs keys now strings 'kv{b}'; stats() splits int/str keys. Dispatch prefers static only when ALL buckets captured compiled (compiled_step flag) else compiled-functional.
- P2 data pipeline landed: scripts/gen_edit_sft.py mines forge_evolve.db (code_programs ok=1, parent_hash->code join) -> difflib token-level ops -> edit_spans.jsonl (500 rows in .tmp test, 2211 verified candidates total). Format: src_ids + ops[{copy:[s,e]}|{tok:[ids]}] matching edit_decode.py semantics. NEXT: full export (no --limit), trainer hook for pointer+BT losses, HF download 0c6958 -> port.


- R66 batched static decode SHIPPED: model._ensure_static_batch(B,plen,need_len) lazily allocates batch-B cache (own pos_t/mask, B<=8) + captures only buckets covering (plen..need_len] (tags b{B}_kv{b}, same bcg dicts). BatchedDecoding.generate_batch routes through a step_fn seam � static prefill via _orig_mod into per-B cache, mask_buf[:, :T]=attn_mask (per-seq left-pads -> per-seq positions emerge from mask cumsum), per-step m_buf[:,pos]=1 + pos_t=pos + input copy + replay; finished rows keep replaying masked by `active`. Measured B=4 greedy 4x96tok: 599 tok/s agg vs 152 functional = 3.9x, 4/4 exact-match. 102/102 unit tests.
- E5 routed-MoE probe = dead end CONFIRMED: all 24 layers have divergent experts (earlier "identical L0-20" was truncated output � recount shows ALL divergent; maxdiff ~0.4-0.5). ForgeQuant is deterministic (quantize-twice = identical scales+indices) � divergence is checkpoint-native (port specialization or training), NOT a load bug. Router untrained: gating entropy 0.000 every layer; routed-k4 NLL 6.21 (+2.8), k2 13.55, k1 15.42 vs mean-of-4 3.41 -> dense_bypass mean-of-4 IS the intended semantics. The _experts_identical fast path is correct armor for fresh ports but never fires on this ckpt. Only way to cut 4x expert compute: distill ensemble->1 expert (proposer_v2 training step).
- E6 numbers context: static batch per-step slower than B=1 replay (4 rows: GEMM + 4x KV attn in one graph) but B tokens per replay. Campaigns generating N proposals per prompt get ~4x proposal throughput free.
- OOM root cause (3B port): post-load ForgeQuant path keeps bf16 model GPU-resident during quantize -> 'only 0 MB free' + RAM pressure (fp32 build = 16.5GB RSS; verify() fp32x2 ~25GB fixed to bf16 ~7GB peak). MEM-SAFE PATH: ForgeEngine.from_checkpoint(..., load_quantize='int4') packs weights during load, never materializes bf16 on GPU. Also fixed preset bug: qwen25_coder_3b_mla_moe gla_latent_dim 256->512 (d_c=2*kv_dim). Port parity verified: argmax-agree 100%, cosine 0.999946.

- MEM-SAFE RULES (32GB box, IDE ~10GB): (1) NO ForgeEngine.from_checkpoint on the 3B port without load_quantize='int4' � post-load forge_quant path OOMs (bf16-on-GPU + CPU pack + restore snapshot + IDE). Proven safe: load_quantize='int4' -> 3.4GB RSS, 5.26GB VRAM, 59.7 tok/s. (2) Model-level V14 tests: tiny configs or meta only. (3) verify() now bf16 (~7GB peak). (4) BCG capture does NOT engage on torchao Int4TilePackedTo4dTensor � forced flag gave no capture + slower decode (34.7). Graph speedup for 3B blocked on torchao-compatible capture OR ForgeQuant-int4 checkpoint pre-baked on disk.


## 2026-10-27 ForgeQuant prequant path � WORKING

Pre-baked checkpoint: `research/checkpoints/qwen25_coder_3b_mla_moe_fq` (2.35GB, 1080 packed linears).
- `scripts/prequant_forge_quant.py` � CPU-only state-dict packer (peak ~8GB RAM)
- Loader: `forge/model/loader.py` detects `*.dense_packed` ? swaps ForgeQuantLinear shells ? `model._forgequant_prequant=True`
- Activation: `engine_checkpoints.py` skips re-quantize, keeps BCG/compile available

Bugs fixed en route:
1. `ForgeQuantLinear.__init__` allocated `dense_packed` at (out,(in+1)//2) but `load_from_weight` pads to n_groups*gs ? shell/ckpt shape mismatch. Fixed: __init__ allocates padded width.
2. in_features recovery: `dense_packed.shape[1]*2` is PADDED � true in from `sparse_packed.shape[1]`.
3. `_has_int8` in loader was `any(int8 tensor)` ? ForgeQuant sparse_packed (int8) wrongly triggered BitNet path ? skipped head?embed re-tie ? head.weight stayed fp32 meta-remnant ? bf16�fp32 matmul crash in warmup. Fixed: only int8 `*.weight` keys count as BitNet.
4. `ForgeQuantLinear.forward` always `_dequantize_weight(cache=True)` ? caches full bf16 (6.2GB on 3B) ? 12.8GB VRAM, 0 free. Fixed: `_cache_w` flag; prequant shells get `cache_w=False` ? weights stay packed, dequant per forward (plain ops ? graph-capturable). Post-load `quantize_model_forge_quant` keeps cache default True (small models).

Measured (qwen25_coder_3b_mla_moe_fq on RTX 5070):
- load: ~31-36s | vram_alloc 4.53GB / res 4.84 | rss 4.84GB | sys_free ~19GB � SAFE
- BCG: 5 bucketed graphs captured (kv512..kv8192) � torchao CANNOT do this
- gen0: ~423s (one-time inductor compile of decode path, [1/5_1] variant)
- output correct (quicksort)

## Codebase touchup session 2026-10-29

Ruff cleanup: ~1600 autofixes (F401/I001/SIM/UP/RUF/C4) across forge + research
+ forge_gui + forge_gui_server + tests + scripts, EXCLUDING forge/evolution/**,
tests/evolution/**, evolve scripts (parallel agent owns those).

- TRAP HIT: ruff SIM118 unsafe-fix rewrote `for k in f.keys()` -> `for k in f`
  on safetensors `safe_open` (NOT iterable -> TypeError) in 9 files:
  checkpoint_io, model/loader, engine_checkpoints, flux/helpers,
  bench_weight_io, research/training_free/bake, _archive/train_r30_qlora.
  Restored `.keys()` everywhere. Lesson: never --unsafe-fix SIM118 blindly.
- Real bugs fixed: strata/blocks.py `dev`->`self.dev` NameError (dead branch),
  epoch_manager dead `self.db.query` -> `SELECT 1`, RUF006 dangling asyncio
  tasks (forge_gui_server engine_rt/procs), foundry.py dead `eval()` removed,
  model_loader stale KVCache alias shadow, engine mixin `import *` replaced
  with explicit torch/threading/SamplingParams + TYPE_CHECKING ForgeEngine,
  kv_cache `__all__` added (bounds the facade star re-export).
- Restored shadowed tests: `test_no_llama_models`, `test_no_gemma_models`
  (duplicate defs -> earlier versions never ran).
- session_manager request/output zips now `strict=True`.
- Junk file `None` (102MB sqlite, tracked since EXP-1): a script opens sqlite3
  with db_path=None -> literal filename "None". Root fix belongs to
  forge/evolution (other agent). Left alone; untrack with `git rm --cached None`
  when convenient.
- Accepted remaining debt: ~113 B905 (zip strict=), ~68 E702 + ~23 E701
  (one-line stmts — flux code style), ~61 E741 (`l` names), ~44 RUF012
  (intentional class-level constants/caches), ~47 I001 (evolution-adjacent
  files skipped).

Next: steady-state tok/s (gen1/gen2 in same proc), then evolve_all --proposer llm --model <fq_ckpt> end-to-end RSI proof.

## 2026-10-27 late � RSI loop proof + 3B evolve hone

RSI loop status:
- evolve_all.py + ported 3B (torchao int4): loop runs end-to-end mem-safe
  (6.8GB VRAM proposer + 2 eval workers).
- RAW 3B port on primed prompts: ~95%% "missing" stage (restarts with
  `import torch`, inline-continuation garbage, verbatim copies). The 0.5B
  proposer was SFT'd on this format; the 3B port was not ? same failure.
- FIX: QLoRA-NF4 evolve-format hone � sft_train --no-bitnet-everywhere
  --qlora-nf4 --lora, 300 steps, 2711 examples (3000 PIE + 61 evolve rows),
  9.44M trainable, VRAM 8.26GB. Saved merged ? dequantized NF4
  weight_packed/scales back to .weight offline ?
  research/checkpoints/qwen25_coder_3b_mla_moe_honed (6.39GB, 1443 tensors).
- HONED RESULT: "missing" drops from ~95%% to ~2/41 dry cycles � model now
  emits fenced body continuations. Promotion still elusive on mined-out
  incumbents; format lesson is proven.
- NOTE: --bitnet-everywhere defaults TRUE in sft_train � must pass
  --no-bitnet-everywhere for ported-quality models (ternary base is lossy).
  --tokenize-workers 0 on Windows (worker spawn appears to hang; disk cache
  makes serial fast anyway).
- Tokenizer note: trained-ckpt save has no dir � assemble_evolve_ckpt.py
  copies tokenizer+forge_convert.json into a loadable dir.

Still queued: V14 distill data (gen_distill_data.py � teacher-gen outputs
? student CE; vocab mismatch blocks logit-KD), CopySpan/Verifier trainer.

## 2026-10-27 late-2 - heads trainer + distill gen

- train_v14_heads.py verified on CPU (random-init trunk): finds
  edit_head+verifier_head = 20.32M trainable, fwd src_ids(432) OK,
  10 copy ops -> copy_span_loss -> backward+AdamW step OK.
  Bugs fixed: out[2] for return_hidden (out[1] is loss), pairs rows
  need chosen/rejected schema not src_ids, AutoTokenizer for _score.
- Data: edit_spans.jsonl 1894 rows; evolve_pairs.jsonl 61 pairs
  (chosen/rejected code strings + chosen_score/rejected_score).
- gen_distill_data.py RUNNING on honed 3B (torchao int4, ~5.9GB VRAM,
  PID 6448): 3045 unique prompts -> 400 rows target, ~0.15 rows/s
  (~40min). Output data/sft/v14_distill.jsonl.
- NEXT: when distill lands -> sft_train --preset forgelm_v14_rsi
  --checkpoint scratch on v14_distill (V14 trunk, vocab 65536 V2 tok),
  then train_v14_heads on the trained trunk, then evolve A/B.

## 2026-10-27 late-3 - fused ForgeQuant int4 GEMV (Triton)

- _fq_int4_gemv_kernel in forge/engine/quant/forge_quant.py: reads packed
  uint8 + group scales only, no bf16 weight materialization. Sparse INT8
  correction via index_add_ after. Folded /7 into scale multiply.
- Numerics vs dequant path: max rel ~0.4-0.5% (bf16 accum-order noise).
- BENCH per-layer (eager): 2048x2048 0.12ms vs 0.56ms dequant (4.6x),
  2560x2560 11x, 1376x11008 21x. Kernel 139 GB/s @BN128/w8 (still
  latency-bound at 2MB; under BCG only traffic matters: ~3GB packed
  vs ~27GB dequant path => ~8x traffic cut).
- CUDA-graph capture VERIFIED: bit-identical replay on captured step.
- _fused_gemv auto-fires in forward() when x cuda + M==1 + packed cuda;
  falls back to dequant for prefill/CPU. Guard: dense_packed.is_cuda.
- test_v14_rsi: 10/10 still green.
- END-TO-END fused GEMV bench (qwen25_coder_3b_mla_moe_fq, BCG 5 buckets):
  steady 1.55s/64tok = 41.3 tok/s (was 25.3 unfused = +63%). 4.52GB
  vram alloc / 4.99 res / rss 7.15. gen0 596s = one-time inductor
  compile (not decode cost). torchao still faster (59.7) but FQ now
  graph-capturable AND within 1.45x.

## 2026-10-27 late-4 - RSI pipeline bugfixes (THE missing-stage root cause)

_evidence: the "95% missing" failures were PIPELINE bugs, not model quality:_

1. _extract_code primed-continuation bug: model emits body + close-fence
   + restart blocks; rsplit("```") grabbed the LAST fence tail (the
   `import torch` restart) not the primed body. Rewrote extractor as
   ranked candidates: primed-body-before-first-fence > def-containing
   fenced blocks > bare tail > raw. Verified on all 3 failure modes.
2. Diversity-hint placement bug: hint was appended AFTER the primed
   signature line -> model continued the HINT SENTENCE (", sort once,
   cache.") not the body. _with_hint() now injects the hint BEFORE the
   primed ```python fence.
3. AST completeness gate: truncated gens (im_end mid-function) compiled
   but could only return None -> wasted subprocess evals scored -1e9.
   Gate checks each REQUIRED def has >=1 Return/Yield -> 'incomplete'
   stage, cheap-rejected. First prefix_hash run: 8 incompletes caught.
4. Multi-required priming bug: targets needing >1 public def (prefix_hash:
   3) were primed on ONE signature -> model emits only that def. Now
   unprimed (full fenced program) when len(required)>1.

- V14 from-scratch SFT: NOT viable here -- full-FT bf16 = 5GB params +
  ~5GB grads + activations OOMs 12GB (died at stream alloc); 296 rows
  can't train 2.5B anyway. V14 needs distill-scale compute; the honed
  3B (QLoRA) is the production proposer. sft_train gotchas: --lora
  defaults TRUE (--no-lora for full FT), CPUAdamW wants ~27GB states.
- Loop verdict: mut-proposer promotion +17.4% stands as the proof;
  honed-3B reaches eval reliably now (18-25 evals/run vs 1-4) but
  candidates fail semantically on hard seeds -- capability gap, not
  pipeline. Next lever: more/better hone data or softer-seed targets.

## 2026-10-28 hone-2 + gate/repair round
- hone-2: 500 steps QLoRA-NF4 on 560 verified evolve rows + PIE (1690 ex total),
  stacked on hone-1 -> .tmp/qwen3b_hone2.safetensors (72 NF4-packed linears)
- int4 end-to-end (user req): NF4 dequant (.tmp/fix_nf4.py, shapes from honed dir)
  -> assemble_evolve_ckpt -> prequant_forge_quant --device cuda (1081 linears, 2.55GB)
  -> research/checkpoints/qwen3b_hone2_fq. bf16 intermediates deleted.
- Verified: 41.3 tok/s fused GEMV decode, correct output, 4.52GB VRAM.
- evolve pipeline fixes in code_evolve.py:
  * undefined-name AST gate: names loaded at DEF TIME (defaults/decorators/bases/
    module stmts) must exist at module scope; nested scopes pruned. Saves eval spend.
  * _repair_seed_names(): undefined names the SEED defines (e.g. _TOP_P_CANDIDATES
    in a primed signature � model only writes the body, can never emit the
    constant) get spliced from seed source. Genuine unknowns still rejected.
  * Verified: seedconst candidate PASS after splice, _NOPE_X REJECT 'undefined'.
  * tests/unit/test_code_evolve.py: 27 passed.
- RSI runs on top_p_filter (fresh db .tmp/evolve_hone2.db):
  * run cev_e1a2c04e26: 20 'test' fails (execute-but-semantically-wrong) � repair
    converted undefined->evaluable; model can't hit torch.equal nucleus gate yet.
  * Recurring silent deaths of evolve proc mid-run; PYTHONFAULTHANDLER on now.
  * Engine warmup transiently spikes 12.79GB (aux features) then settles 5.77GB �
    use_* decode features now OFF in LLMProposer.from_checkpoint call.
- NEXT: basin proposer (LLM seeds + mutator fanout) on dequantize_4bit or
  easier target; DPO/reject-pairs from undefined/test fails is a future lever.

## leak fix (campaign-killer found)
- rsi_evolve rounds each build a fresh CodeEvolve -> ForgeEngine (~5.8GB).
  run() finally closed eval pool but NOT the proposer engine; caching
  allocator never returns it -> +5.8GB/round -> dies at ~12GB on round 3.
- Fix: run() finally now drops proposer._eng + gc.collect + empty_cache.
- Verified +29.3% promotion banked: dequantize_4bit 0.2027ms/call vs
  seed ~0.287 (run cev_f226ac3191, ok=1). Winner came through LLM
  proposal path w/ _repair_seed_names splicing _4BIT_MAGNITUDES.

## win-verification noise (important)
- dq +29.5% VERIFIED head-to-head twice. quantize_4bit (-55%) and
  top_p_probs (-11%) "wins" do NOT reproduce � _verified passed them in
  a lucky contention window (eval_many is sequential inside one pass;
  drift between challenger and incumbent blocks lets fake wins through).
- Fix applied: mean -> median in _verified (robust to load spikes).
  Consider reps>=4 + tighter rel_delta on GPU-contended targets later.
- For hone data trust: ok=1 rows are semantically correct programs
  regardless of score noise � safe for SFT. Incumbent-quality claims
  only trust re-verified wins.

## r64-campaign results (basin proposer, hone-2 int4 FQ base)
- Verified promotions banked (ok=1, promos=1):
  prefix_hash +6.1% | sample_pipeline +71.5% | stop_matcher +2.1%
  | min_k_filter +84.6% | dequantize_4bit ~+29% (re-benched real).
- Timing-luck "wins" (do NOT trust): quantize_4bit -55%, top_p_probs -11%.
- Engine transitions stable at ~11.8GB across 4+ targets � leak fix confirmed.
- dry_penalties starved; top_p_filter all 'test' fails (semantic gate too hard).

## int4 end-to-end (no more transcodes)
- qwen3b_hone2_fq: prequant_forge_quant --device cuda, 1081 linears, 2.55GB.
  41.3 tok/s, 4.52GB VRAM, correct output.
- Native NF4QuantLinear in forge_quant.py (NF4 LUT + fused GEMV variant);
  loader detects weight_packed/weight_scales -> NF4QuantLinear shells.
- manual-LoRA ON ForgeQuant base works in sft_train (4.44GB VRAM) �
  adapters merge into packed weights = int4-native hone path.

## self-hone status
- campaign auto-triggered sft_train on own wins; default.lora saved+registered.
- min_k_filter tag trainer died under RAM pressure; restarted standalone �
  slow packed backward (~40s/step), 3 train/1 val examples = smoke value only.
- NEXT: hone-3 = manual-LoRA on FQ base w/ v3(231)+v2(560)+pie;
  adapter hot-swap A/B in proposer; DPO pairs from 'test'-stage fails.

## ForgeQuant publishability audit (2026-10-29)
- Verdict: RTN-class engineering component, NOT a novel/SOTA format. Prior art: OWQ (identical skeleton), SpQR, LLM.int8, Atom/QUIK; GRINQH is the calibrated stronger version of the same idea.
- Math fix: ~4.9 bits/param NOT 3.6 (zeroed outlier rows still stored at int4). Compare vs W5 baselines.
- Mechanism fix: per-(row,group) scales mean outlier removal does NOT shrink dense dynamic range; only benefit is outlier rows at int8.
- Kernel uses no tensor cores (scalar dequant+FMA GEMV); SM120 benefit is bandwidth only.
- Measured: 3B _fq ~41 tok/s fused vs torchao int4 59.7; dNLL +0.47 @Qwen2.5-0.5B; packs lm_head (no exclusion).
- Docs corrected: forge_quant.py docstring, rd_round_32.md, rd_quant_master_list.md, AGENTS.md, rd_quant_plug_and_play.md.
- Cheap levers if revisiting: activation-aware row selection (1 calib batch), drop dead int4 rows (-0.4bpw), fused prefill GEMM; then honest Llama/Qwen PPL at matched bpw before any external claim.

## R66-quant: ForgeQuant rotation + batched GEMM (this session)
- use_rotation=True (opt-in): block-diag Hadamard R = D·H_gs per group on
  the input dim; dense packed in rotated basis, INT8 sparse rows stay
  ORIGINAL basis (in_f recovery + param layout unchanged). Synthetic
  128x256 gs=64 outlier weight: fwd rel-err direct 0.0857 -> rotated
  0.0366 (2.3x better); pure Gaussian neutral (0.984x). Pack:
  scripts/prequant_forge_quant.py --rotate; ckpt flags
  forge_convert.json {"fq_rotation": true} -> loader materializes
  rotated shells.
- _fq_int4_gemm_kernel: batched fused path M<=32 (BLOCK_M 16/32, tl.dot
  ieee; BLOCK_N halved to 64 when BLOCK_M=32 for SM120 smem). 2048x2048
  gs=128 RTX 5070: M=1 GEMV 1172us vs 1998us dequant (1.70x); M=8 336us
  vs 1332us (3.96x); M=32 424us vs 1212us (2.86x). Fused-vs-dequant rel
  diff ~3e-7 (fp32 accumulate order only); composes with rotation.
- code_evolve proposer tools: FETCH: <url> (jina->trafilatura, ~1.5KB
  inject, shares SEARCH budget+cache as fetch:<url> keys) + INSIGHT:
  <topic> (code_insights tag-overlap via mode=ro sqlite, free). Both
  share research_max_queries cap + single regen.
- Non-code domains: forge/evolution/domains/cev_math_solver.py
  (accuracy max, seed ~0.6) + cev_rle_codec.py (ratio min, seed ~0.61);
  Target.resolve checks targets/ then domains/; in rsi_evolve
  DEFAULT_TARGETS @60 evals.


## 2026-10-30 LFM3 proposer diagnosis (AFK watch)
- Campaign PID 9516: _fq + primed=true (forge_convert.json override set on BOTH lfm3_1p7b_cc + _fq)
- Root cause chain: adapter poison (fixed: canary+quarantine+scale1.0+min32rows) THEN unprimed ChatML -> think-soup at temp<=1.0 on long evolve prompts (95% fail)
- Probe facts: _fq chatml t0.8 = coherent think+code; fp16 chatml greedy = Step-loop derail; BOTH babble on raw prompts (chat-tuned ckpt)
- primed=true forces code-first (no think) -> verified clean output earlier; sacrifices optional reasoning but soup yields zero training data
- expandable_segments UNSUPPORTED on this Windows torch build (warns, no-op)
- workers 16 -> python311.dll 0xc0000005 native crash (paging exhaustion); run --workers 4
- _cache_w stays OFF on proposer (saves ~4GB VRAM for eval workers); fused Triton GEMM serves M<=32 decode
- Batch decode vectorized: batch_gen seeded from row-seed tuple (basin emitter seeds every row - was forcing per-row fallback), prealloc generated buffer, .tolist() stop check
- Still pending: ok-rate verdict on primed=true boot; split-K int4 GEMM for M=2-32; SFT export verified-rows-only; DPO pairs from judge

- primed=true VERDICT: soup 95%->29%, dup 60% (parseable but repeated - cheap rejects), evals flowing (15 evals on suffix_mem vs 2 prev), batch 70.8->52.9s, VRAM 9.5GB
- LFM3 native think toggle: template emits <think> always; enable_thinking=False -> <think></think> empty block. Dormant under primed=true; available via _wrap if unprimed prompts return
- Next bottleneck = proposal DIVERSITY not parse rate: model converges to same solutions, dedup kills them free


## Static-graph verdict (equality-harness verified)
- conv_state reuse fix killed the OOV panic; graphs now emit clean-but-drifted output at B=8/16
- Graph decode SLOWER than functional at short ctx: 567 vs 740 tok/s B=16 (full 512-bucket attention tax > launch savings). Kept OFF in code_evolve � revisit only if ctx >2k
- Remaining drift: argmax flips (~1 token), NOT corruption � replay picks diff SDPA/cuBLAS backends
- NPU path real: OpenVINO/LFM3-1.7B-int4-ov + LLMPipeline(path,'NPU'), needs OV 2026.5 nightly + optimum-intel PR#2087. Intel AI Boost 15.9GB shared. Offload = free whole GPU for evals
- LFM3 KV win queued: 5 SWA layers window=256 -> ring-buffer KV saves ~40% KV mem
- Live: B=16 cuBLAS functional ~740tok/s bench decode-side; prefill (1.5k-prompt x 16) now the wall
- Vendor sampling: temp=0.1/top_k=50 � our 0.7-1.25 sweep far hotter than spec
- PID 6848 campaign rebooted with verified stack

## 2026-10-30 LFM3 static-decode FIXED + verified (supersedes "graph decode SLOWER/drift" above)
Root causes found & fixed (was never a batch-size bug):
- conv pad-mask under static decode: llm.py read attention_mask[:,-T:] = bucket VIEW col -> always 0
  -> all 13 conv layers emitted zeros every replay. Fixed: index_select(1, preallocated_cache.pos_t).
- conv-state bank: _conv_state=None realloc on pos-0 reset freed captured-graph storage.
  Fixed: single owned buffer + per-B views; stale cross-B state guarded.
- LFM3 per-head sigmoid attn OUT GATE silently dropped by ALL fused attention wrappers
  (fused_qk_norm_rope_cache, seq_aware_split, atflash, compact_attention, flex_decoding, cosa,
  mosa, triroute): `self.out_proj(out)` bypassed `_finish_attn`. Functional path was the
  semantically-WRONG reference all along; static bails to the real forward (correct).
  Fixed: every wrapper now routes through `_finish_attn` (fallback to bare out_proj for
  GTA/GLA modules that lack it). Functional output CHANGED after the fix (gate is real).
- Fused-vs-unfused kernel numerics (~0.3-0.4 logit) flip greedy argmax near ties -> earlier
  "drift". bench --no-wrappers gives kernel parity -> B=8: 8/8, B=16: 16/16, bit-exact.

VERIFIED (scripts/bench_lfm3_evolve.py, lfm3_1p7b_cc_fq, RTX 5070):
- Equality: B=8 8/8, B=16 16/16, B=32 32/32 exact rows (wrapper parity, greedy, 96 tok)
- Static B=16 compile=True: replay ~9.9-12.7ms/step -> ~874-1053 tok/s agg, peak ~5.5GB
- Static B=32 compile=True: **1271 tok/s agg** (engine 1283), per-row 39.7, peak 5.80GB
- Weight path: int4-fused Triton GEMM M=16 = 42.1ms/step vs cached-bf16 cuBLAS 12.7ms ->
  _fused_gemv=False + _cache_w=True confirmed (3.3x faster; hand-rolled kernel no tensor cores)
- Sampling ~2.2ms, sync ~0.05ms -> replay is 75%+ of step cost; bandwidth-bound ~3.4GB/step
- Determinism: functional B=16 16/16 self-identical; static replay deterministic

code_evolve changes:
- _load_engine_obj: BCG+compile on, _fused_gemv=False/_cache_w=True forced (142 linears)
- basin_batch default 8->16; _propose_burst cap min(room,12)->min(room,24)
- release_engines: sleep(2)/close() (obj.unload() never existed)

Dead ends documented: int4-fused GEMM (no tensor cores), post-capture flag flips don't
change baked kernels (graph records kernels live at capture time).

Remaining headroom: replay ~13-18ms/step bandwidth-bound; INT8 tensor-core W-path or
batch-drain shrink (finished rows keep replaying = ~19% waste at EOS spread) are next levers.

## session: proposal-validity hardening (code_evolve.py)
Measured problem: ~29pct soup + ~60pct dup = invalid proposals burning slots.
Changes:
- _extract_code(raw, prime, required): strips closed think blocks + leaked
  ChatML markers, any-fence-lang tag, required-name block preference,
  leading-prose trim, dedent fallback.
- Basin emitter bug FIXED: per-prompt prime capture (was: last prompt's
  prime applied to every raw -> wrong-prime soup).
- _queue_gate_repair: syntax/missing/incomplete/undefined/dup rejects queue
  ONE regen w/ failure reason, marker blocks re-repair, shares
  repair_per_batch cap; basin mode exempt (mutator supplies volume).
- Same-batch dedup: dup_batch stage (burst) + gen_dup (basin) - no island
  penalty, no _dry.
- _tried_gists (deque 24): approach fingerprints of seed/resumed/basin/
  gated programs -> '# Already evaluated' line in _build_prompt (needs >=3).
- Telemetry: _gen_n/_gen_ok -> console line, run() result['proposals'],
  status JSON['proposals']; primed disabled for multi-required targets
  in burst too (was serial-only).
Tests: 8 new in tests/unit/test_code_evolve.py; 48/48 pass.
Next: run a real campaign and read status['stage_counts'] +
proposals.valid_pct to confirm dup/soup actually dropped.

## session 2: instruction-following / proposal validity
- _build_prompt: rules line now matches decode mode (primed -> continue
  body, do not restart; unprimed -> one fenced block). Mixed signal was
  teaching restarts.
- export_validity_pairs(db,out): DPO rows chosen=gated vs rejected=
  model''s own soup/dup for same parent (fallback same target); why=
  stage. Wired: fn_evolve --export-validity, rsi_evolve emits
  .validity.jsonl next to sft each round.
- export_sft: primed-prompt rows now emit BODY continuation as response
  (was fenced restart -> trained the prime-abandon failure).
- llm_domain_gen (self-play domains): ChatML-wrapped prompt + primed
  mid-class ending + prime splice + hardened _extract_code reuse;
  previously raw prompt + naive regex (same soup bug the proposer had).
Tests: 50/50 pass; domain-gen smoke: continuation + restart both recover.

## session 3: live-run observability
- EvalPool.in_flight(): jid/tag/worker/elapsed_s/over_s per pending job.
- status JSON gains inflight (epool + legacy pending); fn_evolve --status-file now exists (was cfg-only).
- evolve_watch --status FILE: in-flight jobs + OVER-deadline marker on live board.
- --eval-log per-eval lines + worker death/timeout prints = crash-locator.
- serial pinpoint: --no-eval-pool --workers 1.
- release_engines(fast=True): sleep(1) + wake + re-activation (stale graph pointer safety).

## session 4: self-knowledge + token efficiency
- _self_report(): last-60 rows -> reject-stage histogram + avg gen_tokens -> Self-check prompt line.
- FIXED dead telemetry: gen_meta rows stored metrics_json={} -> reasoning stats never fired. _log_program stores res when metrics empty.
- llm_domain_gen._sanity_check: exec + evaluate() on 2 configs, finite+non-constant required or GenericDomain fallback.

## 2026-11 evolve-loop audit+fix round (Devin continuation)

### Fixed this session (all confirmed by regression tests)
- code_evolve: finally-block called LLMProposer.release_engines(fast=True) — old _eng=None loop freed nothing (_ENGINES cache kept engine alive).
- export_pairs/export_sft: best-sibling fallback for never-evaluated basin parents; NULL-score guard; conn.close() leaks closed; primed-prompt+fenced-response SFT rows now skipped (train/inference mismatch).
- _ensure_lora: 0-tensor load no longer marks _evolve_lora_tag (was: short-circuit all later calls, no retry, no warning); missing adapter file now returns instead of FileNotFoundError inside engine lock.
- _propose: capture prime locally (shared self._prime race in pool mode).
- _judge_gen: skip Mutator-shaped proposers (was: mutated program written as "critique" insight).
- _norm_hash: ast.dump canonical hash; '#' inside strings no longer collides.
- _extract_code: primed raw starting with \"\"\"/\'\'\'/@ treated as restart (was: docstring glued under signature).
- _ENGINES key: +engine_config/proposer_quantize (was: second config silently got first config's engine).
- _lora_quarantine manifest write now atomic (tmp+os.replace).
- _seen_ops capped at 50k (unbounded growth).
- Dead _search_queries removed (superseded by _tool_calls); test updated.
- eval_pool: clean-exit race fixed (harvest-before-requeue — worker that evaluated then hit max_jobs used to double-eval + corrupt replacement slot); boot-death uses OLD worker.completed (was: new worker's 0 -> always counted); requeue preserves caller timeout; close() closes queues (Windows handle/thread leak); worker pins repo root on sys.path.
- mutator: _ctor returns v.clone() (alias bug: caller mutation poisoned cache); _SortReuse skips non-Name targets (self.x rebound to _sv); _DiffSwap axis-aware (x[:-1] is dim 0, x[..., :-1] is -1, mismatched axes skipped); _WhileFor requires loop-invariant bound.
- lora_router (prior): route() via AuxModelRunner.infer, full vector not [0]; fit() sigmoid+BCE.
- database: FindingsDB(None/""/"None") raises ValueError (was: file literally named "None", 119MB orphan — was TRACKED in git).
- finetune_proposer._run_sft: passes --tokenizer <hf base> (was: dataset tokenized with ForgeLM-v2 tokenizer before --hf-model override -> crash/vocab corruption).
- rsi_evolve: --lora-alpha default 64->32 (scale 2.0 produced the babble adapter).
- evolve_all: +cev_math_solver/cev_rle_codec defaults, --auto-targets collect_targets merge, --export-validity flag, --export-pairs default -> data/sft/.
- sandbox.json + tool_security DEFAULT_SANDBOX_CONFIG: post-migration forge/ paths (was: unmatched forge/* fell to READ_ONLY default).
- .gitignore: .tmp_*, *.err, *.jsonl, keycheck/tok_test, None, scratch/, logs/.
- Root cleanup: .tmp_* probes, None blob, stray logs/jsonls moved to scratch/, logs/, data/sft/ (nothing deleted).
- Tests added: norm_hash collision, primed restart docstring/decorator, diff_swap axis, sort_reuse attr target, ctor aliasing, stagnation hop overshoot+once-per-plateau, FindingsDB(None), ensure_lora zero-load retry.

### Test state
- test_code_evolve.py: 58/58. test_llm_domain_gen.py: 13/13. test_evolution_database+test_scout: 55/55.

### Known remaining audit items (lower severity, not yet fixed)
- ScoutConfig.n_elites dead field (scout uses its own constant); research error snippets in prompts; _seen_ops reset granularity; evolve_all/rsi_evolve/fn_evolve CLI drift (--parallel missing on rsi, --status vs --status-file); config.yaml host-local (kept at root); auto_slice_past_kv oracle can't detect clone-vs-view; .validity.jsonl has no consumer yet; mutator _ArithIdentity aliasing kept (gate+test cover); EvalPool in-flight seen-hash race (by design: spawn-fail consumed as reject).

### Broad suite result (22min): 3527 passed / 11 failed
- 8x test_evolution_frontier: MY FindingsDB(None) guard — db_path=None is the deliberate "no persistence" semantic (and the actual source of the tracked "None" file). Fixed: None -> :memory:.
- test_round32_quant::test_build_kv_cache_unknown_falls_back: pre-existing stale test — commit 65e7795 made build_kv_cache fail-closed. Fixed the test to expect ActivationError.
- test_flux::test_cuda_forward_deterministic + test_round32_quant::test_forge_quant_fused_gemm: GPU/heavy tests in user-edited files — flake category, unrelated to this pass.
- Earlier transient collection errors (batched_decoding import, IndentationError at 1111) were mid-write reads from parallel edits — all pass clean now.

## audit session state 20:07
- external_eval baseline running (pid 26412): tasks arc_easy,arc_challenge,hellaswag,mmlu(limit 200),gsm8k(50) -> results/external_eval_v2.json
- pytest full suite running shell 76008f -> results/pytest_full.log
- B re-audit done: response_format->json_schema wired (non-stream), /info, /health/ready, prometheus /metrics added to forge_server
- remaining: FluxLM subagent 1b512b4e, A5 proposer decision, plan approve/descope -> user

## A6 flux-vs-rag head-to-head (bench_flux_vs_rag.py, 2026-11)
- 500 docs/10qa: flux=0.0 rag=1.0; 2000 docs/40qa: flux=0.0 rag=0.9->1.0 (query-side stopword fix).
- DIAGNOSIS (CPU, R69/R70 knn-QA knobs, 200 docs): knn bank retrieves fact-TYPE-correct rows (population q -> population sentences) but entity-confused (Nimod vs Voromod - invented names share BPE pieces; idf weighting insufficient). knn_lift_topconf=0.4 blocks lift when votes dilute -> mixture falls to recitation.
- Hedge mixture collapses to ONE channel at corpus scale (h0=0.972 untrimmed; o3=0.982 with taps trimmed) -> thin-context recitation of the most recent/matched doc.
- Scaffold (teach_text on non-corpus entities) fixes output FORMAT (emits answer-type tokens, e.g. " cobalt") but answers are wrong-entity - entity->value binding is the broken link, not the format.
- w_boost {knn:0.4,deep:0.2} / {h0:0.02}: no change - the gate is vote-conf, not mixture weight.
- Open lever if revisited: entity-anchored fingerprints (per-entity row binding), scaffold on corpus-entity non-eval facts, or hybrid arm (flux provenance + rag coverage).

## 2026-10-31 LFM3-CC lossless-arch R&D (d1 graft VERIFIED bit-exact)
- test_lfm3_d1_parity.py: base lfm3_1p7b vs grafted (GDN2@conv[1,5,9,13,17,21] H8/dk64 + ValueResidual + LearnedSink + AttnRes + CopySpan/Verifier heads) on REAL lfm3_1p7b_cc fp16 ckpt, CPU fp32 seq48: max|dlogit|=0.000e+00, 100% argmax. 1.7288B -> 1.8029B (+73.9M). PASS = lineage gate.
- Placement theory: graft i%4==1 convs � each sits immediately BEFORE a full-attn layer (2,6,10,14,18,22) => delta-rule global-memory consult lands right before each global read. LFM3 motif is periodic-4: [conv,conv,FULL,conv,SWA].
- MTPModule @d=2048,vocab=128k: adapter_heads=True = 21.0M; independent heads = 1052.8M (tied 128k makes independent heads absurd). adapter_heads only.
- AttnRes gate=0: skips buffer entirely (attn_res_active=False) � zero compute, bit-exact.
- NGramMemory: gate=0 inert; .draft() = ready suffix-spec proposer for copy-heavy evolve outputs.
- Proposer bench verified (results/proposer_bench*): Qwen2.5-Coder-3B 100/100 vs lfm3_1p7b_cc 50/75 vs _fq 33/50. int4 costs ~17pts; LFM3-CC failure mode = verbose unterminated output (619-2367 chars vs Qwen 68-462) � partly behavioral, not just capability.
- _cache_w already dequants int4->bf16 => fp16 dir = SAME VRAM, no quant tax. _fq has NO fq_rotation flag (plain absmax RTN) => repack w/ --rotate or run fp16.
- NOT lossless (checked): sandwich-gamma folding (rms(gamma*z)!=rms(z)), conv->delta port (FIR != low-rank recurrence), SWA widening, vocab append, zero_init_residual.
- Open work items when gates open: GDN2 needs pad_mask on x0 writes (left-pad batches corrupt state) + static-graph-safe T==1 state write; AttnRes needs fixed k-buffer for capture; v0 donor fix (capture layer2 not conv layer0); SWA ring needs pos-mod write arithmetic under pos_t.

## 2026-10-31 LFM3-CC arch R&D round 2 � online research digest
- Kimi Linear (arXiv 2510.26692): 3:1 KDA:MLA ratio beats full-attn fair-compare; KDA = channel-wise decay delta rule = what our GDN2 graft adds. VALIDATES direction.
- Qwen3-Next: 3:1 GDN:GatedAttn, fused QKV+gate in ONE GEMM (qkv_proj emits (1+gate) query features � validates fused-QKV+gate idea), zero-centered RMSNorm (w->w-1 reparam = EXACT function-preserving, better WD behavior in hone), partial RoPE, MTP head.
- Falcon-H1/Hymba: PARALLEL SSM+attn within one block (concat before out_proj) � validates side-path graft topology; Hymba adds 128 meta tokens (learned KV init, fixes 'forced-to-attend') � our LearnedSink bias = the lossless equivalent (virtual register).
- Peri-LN (ICML'25, arXiv 2502.02732): norm before AND after module + embed norms = LFM3's sandwich_norm_mode='pre_residual' IS Peri-LN � already at norm-placement SOTA, nothing to add.
- Sparse upcycling (arXiv 2212.05055): identical expert copies => ANY routing gives same output => strictly function-preserving MoE-ify; cost FFN xE � thin-MoE graft (dense FFN + small zero-init routed experts) is the param-cheap variant.
- Zero/one-layer expansion (arXiv 2511.04981): random+copying best inits; ZERO-INIT SUBLAYER BLOCKS FEATURE LEARNING -> use LLaMA-PRO pattern: copy graft INTERNALS from neighbor layer, zero only exit (out_proj/gate). Expand at ~80% of sched, WSD.
- OT-DUS (arXiv 2508.08011): insert new layers at HIGHER positions = better perf + efficiency => capstone-block graft goes at END (before ln_f), not middle.
- DSA lightning indexer (DeepSeek V3.2) + GLM cross-layer top-k sharing: tiny FP8 indexer selects top-k keys; one indexer shared across layers � for CSA/long-ctx only, low priority at <=4k prompts.
- Model Stock (arXiv 2403.19522): principled 2-ckpt averaging formula for CC<->stock merge (non-lossless, documented-delta option).
- LFM3 vendor notes: head-wise out-gate suppresses activation outliers => quantization-friendly (int4 tax partly self-mitigated); conv K=3->K=7 was tuned => K=11 zero-pad graft continues their own direction losslessly; partial-RoPE unrotated half = position-agnostic content-matching dims.

## 2026-11-01 ForgeEngine proposer double-engine OOM (CONFIRMED BUG, open)
- bench_proposer.py on research/checkpoints/qwen25_coder_3b_mla_moe_fq dies at generation start EVERY time (3 repros): engine A init ok at 6.18GB -> generation begins -> a SECOND engine init block appears (paged KV, low-VRAM hotswap) while A stays resident -> 12.05GB -> silent native death, no traceback.
- Mechanism (from log): a full SECOND ForgeEngine activation block appears mid-run (paged KV, all 5 graph buckets recaptured) DURING/after eng.generate begins � engine-internal low-VRAM hotswap re-activation, NOT a second _ENGINES entry. Old config's buffers coexist with the new during the swap -> 12.05GB spike -> native OOM. The 3B-fq engine init alone (~6.2GB dequanted + KV/graph pools) sits close enough to the limit that ANY re-activation transients over.
- _fq notes confirmed: dequant-cached to bf16 at load -> fq dir saves DISK only, zero runtime-VRAM benefit. 12GB card cannot host two dequanted 3B engines.
- Production default UNAFFECTED: _model_id resolves to HF repo id 'Qwen/Qwen2.5-Coder-3B-Instruct' -> transformers path (~6GB, 100% valid). Bug only hits local converted dirs through ForgeEngine.
- ALSO: FastBuild on a device='cpu' config still runs a GPU staging pass ('GPU transfer: 248.8s' during a CPU fp32 load) � CPU loads must not run concurrently with GPU jobs.

## 2026-11-01 d2 strict-lossless progress (5/6 landed)
- share_rope (1d90891): dedupe identical RotaryEmbedding tables; exact-type check skips learnable variants. 11 copies -> 2 on LFM3.
- _warm_start_conv (abd09ca): conv_kernel_size>K_ckpt left-pads attn.conv.weight; bit-exact prefill + incremental decode.
- zero_centered_norm (fc54275): RMSNorm stores gamma-1, forward adds 1 (Qwen3-Next); _warm_start_zero_center ports by enumerating built-model zc modules; _scan_qk_norm_identity treats w==0 as identity under zc.
- fused QKV+gate + gate-up (5765c3d): use_fused_gemm was DEAD CODE (matrices built, never consumed, ctor init perturbed RNG). Forward reads them now; loader._refresh_fused_qkv re-packs post-load; fork_rng on ctors. Function-exact (fp32 kernel noise only).
- NGramMemory/use_suffix_spec: ALREADY EXISTS (ngram_mem.py, gate=0 + .draft() ready).
- REMAINING: SWA ring KV (ring-buffer sliding-window cache; touches PreAllocatedKVCache indexing + pos_t decode + bucket graphs � needs full-session care). Then zero-init grafts beyond d1 (capstone, thin-MoE FFN, AttnRes-open, MTP-4 adapter, EAGLE-3 3-depth).

## 2026-11-01 � Qwen-teacher distill + d2 strict-lossless complete

### Route (user-approved): Qwen teacher -> LoRA 'qwen_distill' on lfm3_1p7b_cc
- gen_qwen_distill.py: ~212 prompts (200 real logged code_programs prompts
  stratified per-target + 12 target-template) -> Qwen HF completions,
  compile-filtered. ~2.8 rows/min on RTX 5070, 100% usable rate.
- GOTCHA fixed mid-run: '`python' prefill means raw output lacks the
  fence opener � script now re-prepends it; existing rows need post-patch.
- Train: finetune_proposer --mode lora --data results/qwen_distill.jsonl
  --base research/checkpoints/lfm3_1p7b_cc --lora-root forge/evolution/loras_lfm3
  (bf16 cc base; _fq dequants anyway). alpha=32 (=rank, scale 1.0 � 64/32
  scale-2 + 17-row adapters are both quarantined in the manifest for babble).
- Eval: bench_proposer --lora-tag qwen_distill (new flags) vs Qwen 100% /
  LFM3-cc 50% baselines. Gate to flip back: >=90% of Qwen valid-rate.

### d2 strict-lossless set � ALL SIX LANDED
share_rope 1d90891 | conv K=7->11 zero-pad abd09ca | zero-centered norm
fc54275 | fused QKV+gate+gate-up 5765c3d+767d38a | NGramMemory/suffix-spec
(pre-existing, gate=0, .draft() ready) | swa_ring_kv 25bd153.

### Found en route
- use_fused_gemm was DEAD CODE: fused qkv_proj/gate_up built at init, never
  consumed in forward, and its un-forked ctor RNG shifted every later
  layer's init. Now wired + loader _refresh_fused_gemm repacks ckpt rows.
- SWA ring KV: keys are permutation-invariant -> slot=pos%W works; pos_t
  path uses pos_t%C index_copy_ (graph-safe); eager path cats the held
  window position-ordered. NOT bitwise vs linear (SDPA sees 9 vs 19 keys,
  ~5e-7 kernel tiling) � gate is argmax-trajectory equality.
- T>C forward caveat: a T>1 step needs last-(C-1)+T keys � more than the
  buffer. Solution: read held window + cat the new in-flight keys.

## 2026-11-01b � Round-1 poison found + Groq teacher path

### Semantic filter result: 51% of v1 teacher rows were BROKEN
- filter_distill_semantics.py: matches row -> target via keep-public-names,
  execs code, runs target's own test(ns) oracle.
- 212 rows -> 101 kept (99 ok + 2 no_target); 109 semantic_fail + 2 exec.
  Qwen-3B local teacher produced compilable-but-wrong code ~half the time.
- CONSEQUENCE: qwen_distill adapter trained on ~52% poisoned data ->
  DELETED (files + manifest + merged ckpt). Explains round-1 no-gain.
- LESSON: bench valid% = format only. The eval pool's test(ns) is the real
  semantic gate � teacher outputs must pass it before becoming SFT rows.

### Groq backend (gen_qwen_distill --backend groq)
- Teacher: openai/gpt-oss-120b (Apache-2.0 � distill-legal; avoid
  Llama-family: Llama license bars outputs improving other models).
- curl.exe transport REQUIRED: python urllib gets Cloudflare error 1010
  (browser-signature ban); curl works.
- gpt-oss is a reasoning model: reasoning_effort=low to skip the think.
- Org rate cap ~6-7 req/min on 4K-tok prompts -> 429 backoffs pace it.
- Inline oracle + feedback retry (teacher sees failure detail, 1 retry).

## Overnight distill run (2026-11-01)

CORRECTED BASELINE: on real evolve-format prompts LFM3-cc = 0% valid / 61.5% syntax
(the old 50% was a 3-seed-grid artifact). Failure axis = name-keeping.
Adapters v3 (think+code, 163 rows, 135 st): 3.8% / 46% — canary-quarantined on
raw-completion probes but real-path output just = base. v4 (code-only, 105 st):
3.8% / 38.5%. LoRA-r32 on ~150 rows does NOT install codegen capability.
Root confound found+fixed: bench prompts now emit real _build_prompt format.

IN FLIGHT: r2 corpus (deepseek-v4-pro, reasoning captured) ~110/400 kept;
3 API-proposer evolves (min_k/top_p/dry_penalties) harvesting verified
reasoning traces -> results/api_proposer_traces.jsonl + forge_evolve_api.db;
evolve gens logged via --log-gens.

NEXT: merge v5 corpus -> LoRA v5 (r=32/400st) -> bench; if flat try
full-FT w/ --optimizer cpu_offload (memory-guarded), else document
substrate ceiling.

## Overnight status ~21:00
- Full-FT v5 = 19.2% valid / 57.7% syntax on real-format bench (base 0/61.5).
  Failure profile: 0 no_code (format learned), 11 syntax-fail, 10 name-drop.
  Capability is the binding constraint -> more verified data only.
- LoRA r=32 dead end proven (3 adapters, ~4% all).
- RUNNING: 2-shard evolve sweep (26 targets, api proposer, CPU evals) ->
  forge_evolve_api.db + api_proposer_traces.jsonl (~162 traces so far).
- RUNNING: deepseek-v4-pro r3 gen (400 prompts, reasoning captured).
- merge_distill now caps 2 completions/prompt.
- NEXT: when harvest slows -> export_sft + merge all -> full-FT v6 -> bench.

## ~02:30 overnight results
- Sweep done: 347 programs / 30 ok wins / 329 reasoning traces.
- v6 corpus: 289 verified rows (114 w/ reasoning) via merge+ledger export.
- v6 full-FT (300 st, mixed think+code): 15.4% valid / 46.2% syntax —
  REGRESSED vs v5 (19.2/57.7). 40% think-rows dilute format signal.
- v7 = same corpus code-only -> training now.
- r3 gen: 361/406, 115 kept. -> v8 corpus candidate (+r3 rows).

## ~03:30 — plateau confirmed (n=52 bench)
- v5: 13.5% valid / 36.5% | v6: 11.5/46.2 | v8: 7.7/55.8
- corpus 182->323 rows moved nothing; all within noise of ~10%
- think-rows: v7 code-only COLLAPSED to 3.8% — reasoning helps, not hurts
- recipe ceiling ~10-15% valid on the strict target oracles
- RUNNING: r4 gen (deepseek-flash, 500 prompts) + sweep r2 (10 targets,
  4 gens) -> morning corpus v9 candidate; ledger keeps growing
- honest verdict: pipeline works (0->13%), 'near-flawless' unreachable
  on 1.7B + ~300 rows; needs 1000s rows or teacher-in-loop serving

## ~04:30 — DeepSeek balance dead -> failover to Mistral codestral
- 'Insufficient Balance' on all deepseek calls; r4 gen + sweeps killed
- codestral-latest works instantly; mistral backend added to gen script
- sweep now takes EVOLVE_API_MODEL/URL/KEY_ENV overrides
- relaunched: r5 gen (mistral, 400 prompts) + 2-shard sweep (codestral)
- ledger at ok=35 pre-failover; r5 at ~7/min

## OVERNIGHT FINAL — LFM3 distill campaign
GOAL: near-flawless codegen + reasoning + micro-mem + 1k tok + infinite
      evolve/self-play RSI loop. NOT fully met; honest state:

MEASURED (bench_proposer --from-targets, n=52 real prompts):
  base lfm3_1p7b_cc     :  0.0% valid / 61.5% syntax
  v5 full-FT (182 rows) : 13.5% valid / 36.5% syntax
  v6 (289r mixed think) : 11.5 / 46.2
  v7 (code-only)        :  3.8 / 46.2  <- reasoning rows HELP
  v8 (323r,46% think)   :  7.7 / 55.8
  v9 (486r,116 evolve)  : 13.5 / 53.8  <- BEST (ties v5, better syntax)

RSI LOOP: PROVEN WORKING END-TO-END
  - API proposer evolve: 699 programs, 171 verified wins (codestral)
  - v9 AS LOCAL PROPOSER: drove loop, 2 verified promotions, +2.2%
  - ledger -> merge --ledger -> SFT corpus -> retrain = closed flywheel
  - 807 reasoning traces captured in api_proposer_traces.jsonl

TEACHERS: qwen(51% poison caught) -> deepseek (balance died) ->
  codestral-latest (Mistral): 117/406 kept gen, 171/699 evolve wins —
  best teacher found. gpt-oss-120b blocked (Cloudflare/credits).

CEILING: ~13% single-shot valid on strict target oracles; LoRA=3.8%
  dead end; full-FT required; more data stopped helping at ~300 rows.
  'Near-flawless' needs bigger base or best-of-K in-loop usage —
  the loop itself IS the correct deployment (yield*evals=wins).

MEM-SAFE: all runs serialized GPU<->API; 4.3GB train / 5.4GB bench /
  0GB API proposer. No OOM events after serialization rule.

## ~11:00 — LLM-GENERATED GOALS SOLVED THE SATURATION WALL
- gen_llm_targets.py: teacher invents goals from 3 channels —
  code (real forge/ fn bodies scanned+injected), web (Tavily search
  snippets), topics (26x3x3 seed grid). Each JSON spec is AST-gated,
  subprocess-validated (determinism, runtime), self-oracle checked,
  written as llm_<name>.py in auto_targets format -> loads via the
  standard Target path for fn_evolve/gen/merge with zero changes.
- 111 llm_* targets live; evolve sweep auto-discovers llm_*.
- codestral sweep on expanded pool: 2188 programs / 816 verified wins
  across 66 distinct targets (was 171 wins / ~15 targets before)
- v10 corpus: 969 verified rows (602 from evolve ledger) — the
  diversity fix. Training now (500 steps, seq3072, cpu_offload).
- Commit: e5352fa (generator+54 goals), 67161d8 (+57 goals)

## ~12:00 — v10 BREAKS THE PLATEAU (same 137-prompt suite)
- v5: 19.7% valid / 43.1% | v9: 27.0 / 62.0 | v10: 34.3 / 65.0
- LLM-goal expansion (26 -> 137 targets) = the fix for corpus saturation
- v10 corpus: 969 verified rows, 602 from evolve ledger, 66 win-targets
- sweep still running on remaining llm targets -> v11 candidate ~1.5k rows
- next: keep harvesting -> v11 -> rebench; codestral remains best teacher

## ~12:30 stall + recovery
- 2 fn_evolve procs hung >40min on llm targets (0 CPU, API healthy) —
  killed; sweep resumed on the 60 unprocessed targets via
  EVOLVE_TARGET_LIST; +112 wins within 5min of restart.
- ledger: 2400 programs / 928 wins / 66+ targets and climbing
- v10 = 34.3% valid / 65.0% syntax on the SAME 137-prompt suite —
  LLM-goal diversity broke the ~13% plateau (v5 19.7 -> v9 27.0 -> v10 34.3)

## 2026-11-03 â€” Research: weight-space programming + faster distill (LFM3-CC -> 7B goal)

- WSP (MIT CSAIL, Takehi/Teoh/Zhang/Khattab, 2026-11-02): GPT-6 Astra edited Qwen3.5-0.8B
  weights directly. Removed a repetition direction from MLP down-proj layers 15+19 (64 cols ea,
  131072 w = 0.016%) -> MATH-500 7.9->48.1, GSM8K 13.8->67.1, IFEval 21->62.3. Model REPAIR not
  capability injection: failed on Gemma-3-270M (no loop defect). Baseline-sensitive (indep re-eval
  scored base ~44 under diff sampling). Tooling: weightspace pypi pkg + openwsp/openwsp GH
  (versioned sparse edit format, probe-budget harness, checkpoint/rollback).
- OUR failure mode (verbose unterminated output, 619-2367 chars) is same defect class ->
  scripted direction-ablation on MLP down-projs via contrastive acts (probe_lfm3_loops.py) is a
  cheap post-train complement; no frontier editor needed.
- Thinking Machines on-policy distill: student rollouts scored per-token by teacher logprobs,
  advantage = -reverseKL, IS-loss update. 9-30x cheaper than more SFT/RL for same gain (Qwen3 TR:
  AIME24 74.4 vs 67.6 at 1/10 GPU-hours). SEQUENCE-LEVEL => tokenizer-agnostic (sidesteps LFM3
  128K vs Qwen 151K vocab mismatch that makes top-k logit-KD impractical).
- CompactifAI (arXiv 2608.03796): cache teacher top-100 logits once -> same loss as online KD,
  ~29% faster iter, ~41% more tput; fused chunked-KL never materializes [B,T,V]. Megatron-LM
  cached_logits_loss.py is a ref impl. chunked_ce.py pattern extends to chunked-KL.
- sft_train --distill args are DEAD STUBS (963-975). Real impl plan: student rollouts via engine
  static-KV batch (1270 tok/s measured) -> teacher scores logprobs (forward-only, batched) ->
  per-token weights into compute_loss (sample_weights path exists). rlvr_train.py is a
  one-line-change distance from TM's method.
- Data: Magpie-style pre-query instruction extraction for diversity (diversity>size broke the
  v9->v10 plateau); AdvDistill reward-weighted multi-sample (k gens + rule-verifier weights ->
  --sample-weighting). Local teacher: qwen25_coder_7b preset exists + dl_qwen7b.py; int4 ~5.5GB
  fits w/ GPU<->API serialization rule; run via ForgeEngine not HF generate (2.8 rows/min HF).
- Qwen2.5-7B-Instruct targets: MMLU-Pro 56.3 / GSM8K ~91.6 / MATH ~75.5 / HumanEval ~84.8 /
  IFEval 71.2. Full parity unrealistic on 1.7B (knowledge capacity); scope = match on the
  CC/evolve suite (specialist > generalist in-domain per TM thesis).


## DeepInfer port plan (2026-11-03)
Source: deepseek-ai/DeepInfer (production engine for V4.1-Flash), cloned to D:\windsurf\deepinfer_ref.
Already in ForgeAI: DSpark paper impl (dspark.py confidence_schedule), _defer_sync launch-before-item (pure-attn only), ChunkedPrefixCache, xgrammar SO.

Deltas to port:
1. BOUNDED REPLAY on partial prefix hits: replay last R tokens of cached prefix to rebuild conv/SSM/dict state at hit boundary. Fixes: partial hits drop conv_snap -> zero-pad conv; dict presents keep state at entry_end (wrong position). R = sliding_window else 128.
2. CONFIDENCE-SCHEDULED VERIFY on ExternalDraftSpeculativeDecoding: draft softmax prob as c_k; survival cumprod; trim verify prefix + early draft stop.
3. ASYNC DECODE for hybrids: _defer_sync extended via _spec_snapshot_recurrent before speculative fwd + restore on break.
4. seq_seed deterministic sampling: SamplingParams.seed -> SplitMix64 hash -> CDF inversion.

## 2026-11-03 â€” DESIGN: quant-loss patch + real distillation (model-agnostic)

### A. "Patch the loss out" â€” chosen design: ForgeQ-R (residual-patched int4)

Honest info-theory first (measured literature, Qwen2.5-7B):
- bf16 weights, true lossless ceiling ~= 1.49x (Rotem IJCA 2026; ANS tile-decomp arXiv 2606.15789)
- already-quantized streams (Q4_K): ceiling ~= 1.04-1.08x â€” int4 output is near its entropy
  bound ALREADY -> "lossless codec at int4 size" is impossible. Drop it.
- Quant error is NOT low-rank (dense bounded noise) BUT the *functional* error concentrates on
  salient channels (AWQ result). So: low-rank residual + optional sparse-outlier set recovers
  ~all measurable loss at ~4-6% size overhead.

ForgeQ-R per-matrix plan:
  W_exact = dequant(int4 Q) + A@B  (+ optional fp16 sparse outlier set S)
  - A [r, in], B [out, r] fp16 from truncated SVD of E = W - dequant(Q), r=16 default
  - LoftQ-style alternating: re-quantize (W - A@B), recompute E, re-SVD, 2-3 iters (cheap, CPU ok)
  - Runtime: y = fq_gemm(x,Q) + (x@A.T)@B.T â€” rank-r GEMV ~= free at decode (r/din ~ 1%)
  - Size: r*(in+out)*2B vs 0.5*in*out B for int4 => +4-6% total (1.43GB -> ~1.5GB)
  - Optional stage-2: short QAT calib training ONLY A,B with quantized base in forward
    (bitnet_int8_training machinery exists) -> pushes to eval-parity ("effectively lossless")
  - Verify: max-abs logit diff + valid-rate re-bench _fq vs _fqr

### B. Real distillation â€” OFFLINE on-policy (sequence-level, tokenizer-agnostic)

Why sequence-level not top-k logits: LFM3 128K vs Qwen 151K vocab mismatch makes logit-KD
impractical; sequence scoring needs NO vocab alignment and works for ForgeLM custom students too.
Also matches TM result (9-30x compute-efficiency vs more SFT).

Loss (Thinking Machines / Agarwal on-policy distill):
  A_t = stopgrad(teacher_logp_t - student_logp_t)   # per response token
  L = sum_t w_t * CE_t,  w_t = clip(A_t, -clip, +clip) * alpha  (+1.0 base CE keeps floor)
  => reinforces tokens teacher likes more than student; suppresses student-overconfident tokens.
  Implementation: ChunkedDistillCE in chunked_ce.py (per-chunk student_logp = -ce_chunk;
  weight applied to (softmax-onehot) grad like entropy-weight does). Rides through
  model.config like entropy_alpha (pattern at llm.py:765 + sft_train.py:658) â€” zero signature
  changes. Manual-logits path gets same math for sample_weights compat.

Data flow (all model-agnostic):
  1. scripts/gen_student_rollouts.py: student samples k rollouts/prompt via ForgeEngine
     batched decode (~1270 tok/s measured on _fq) OR reuse evolve-ledger wins (they ARE
     on-policy verified rollouts â€” free data).
  2. scripts/score_teacher_logprobs.py: teacher (HF AutoModel int4 7B OR ForgeEngine port)
     forward-only scoring of the detokenized student text; teacher-token logps mapped back to
     student tokens by char-offset overlap (mean pool). Output row adds:
       teacher_logp: [float fp16-rounded, len = response tokens]
  3. sft_train --distill: dataset passes teacher_logp through collate -> per-token weights.
     --distill-alpha (default 1.0), --distill-clip (default 5.0), --distill-mix (CE floor 0-1).
  4. evolve export_sft: add "weight" = f(delta) -> collate reward path already consumes it
     (ex.get("reward")) â€” wire field name 'reward' directly.

### C. Application order (build -> verify)
  1. ChunkedDistillCE + sft_train --distill wiring   [core]
  2. score_teacher_logprobs.py                        [data]
  3. export_sft reward field                          [ForgeEvolve glue]
  4. ForgeQ-R: scripts/quant_resid_patch.py + runtime A@B in quantized Linear  [format]
  5. tests/unit for each; AGENTS.md + scratchpad update

ForgeEvolve note: evolve wins are already on-policy + oracle-verified; scoring them with the
teacher adds per-token dense signal on TOP of the binary win â€” strictly better data than the
current binary-valid SFT rows. DPO near-miss pairs stay on the export_pairs path.

## 2026-11-04 - Offline distill + ForgeQ-R residual: IMPLEMENTED & TESTED
All pytest-green (8 new in tests/unit/test_distill_quant_patch.py; sweep: 102 + 40 clean).
- ChunkedDistillCE (training/losses/chunked_ce.py): w = floor + (1-floor)*clip(alpha*(tlp-slp), -clip, clip); MISSING=-99 -> w=1. chunked_distill_ce(x, W, target, tlp, chunk_size, alpha, clip, floor). llm.py dispatches via config.teacher_logp + distill_alpha/clip/floor (entropy_alpha convention).
- sft_train: teacher_logp flows load_examples(jsonl+parquet) -> tokenize -> collate (per-completion list -> non--100 positions, -99 fill, 5-tuple) -> compute_loss. --distill gates at unpack; --distill-alpha/clip/floor live; packing path carries tlp.
- scripts/score_teacher_logprobs.py (~349 lines): HF int4/int8/bf16 or Forge-native (--teacher-forge); render_single_turn boundary; teacher logp -> student tokens via char-offset overlap (tokenizer-agnostic, LFM3<->Qwen works).
- code_evolve.export_sft: reward = 1-exp(-delta) per verified-improving row -> reward_weights.
- ForgeQ-R: ForgeQuantLinear(residual_rank=r) -> res_a[O,r]/res_b[r,I] fp16 params; _residual_apply after quant+LoRA in BOTH fused & dequant branches (original basis); merge_lora() refits via SVD. Loader reads rank from <mod>.res_a shape (self-describing). prequant_forge_quant.py --residual-rank/--residual-iters/--residual-max-elems; prints rel-Frobenius err before/after. LOSS-REDUCED, not mathematically lossless.
NOT DONE / next:
- No end-to-end distill run yet (scorer untested on a real checkpoint; needs GPU serial slot).
- Rank sweep r=8/16/32/64 dNLL on real lfm3_fq pack - needs GPU.
- engine public score() API (scorer script covers standalone for now).
- agentic_distill.py reward path still broken (tool_use_loop removed earlier).
- Tokenizer decision: reuse gigatoken base + PRIVATE special-token names (no im_/eot/tool_call conventions -> protects external harnesses/agents from parsing our markers as structure; reserved-ID block, never BPE-merged from text).
## 2026-11-04 - FluxLM v2 direction experiments (scripts/exp_live_v2.py)
KEY: v2 skeleton already exists - forge/model/live.py LiveSession (frozen substrate
+ GDN2 + TitanMemory + NGramMemory sidecar, journaled/revertible). Old flux/ =
hedged ngram mixture = the failure; do NOT resurrect.
Measured (micro d128L4 V256, CUDA): exact NGramMemory carries recall (verb 3.7-5.0,
survives 512-tok distractor). TitanMemory fuzzy learns almost nothing cold (0.42-2.0)
- needs bootstrap. OVERWRITE BROKEN: ngram stores old+new equally (2.35/2.36).
NO surface generalizes (alt-separator CE = base). Zero false memory (unseen=base).
VRAM <0.5G, ingest 450-3000tps. Design: docs/FLUXLM_V2_DESIGN.md - L1=episodic
scratchpad (exact+fuzzy), L2=LoRA consolidation (ONLY path to generalization -
verbatim-bound memory cannot transfer by tuning). Build order: ngram overwrite
fix -> L1 on real ckpt -> L2 loop -> titan bootstrap check.
## 2026-11-04 - Gold-spec research (rival all local on 5070)
Local field on 12GB: resident 7-8B int4 ~60-100tps; offloaded MoE GPT-OSS-20B ~35-42tps,
Qwen3-30B-A3B ~36-50tps (VRAM follows TOTAL params, speed follows ACTIVE - expert path
crosses ~50GB/s RAM bus every token).
Evidence: R1-Distill-Qwen-1.5B = MATH-500 83.9 (>GPT-4o 74.6) via verified long-CoT SFT -
domain-specialist distill works at small scale; breadth does not transfer (LCB 16.9).
cons@64 >> pass@1 (28.9->52.7 AIME) = test-time compute is the real sub-7B equalizer.
SmolLM3/MiniCPM4 quality = 8-11T tokens datacenter scale - NOT replicable; BitCPM proves
ternary production-grade. dense->MoE upcycle = trap at single-GPU scale (moe-upcycle
measured strictly worse; Sparse Upcycling pays at ~40-50% orig pretraining cost).
CONCLUSION: ternary ~2-2.5B dense backbone (BitNet-class) + distill + verifier loop +
L1/L2. Rival = in-domain + speed + live-learn axes, never breadth.
## 2026-11-04 - FLINT indie-model research
Recursive Flint (RecursiveSI): 40B/2.5B-active MoE, small team + agents, <4mo from scratch.
Claims: best <50B MoE on SWE-Bench Pro v2/Terminal-Bench 2.1/SWE-Verified; rivals dense
25-40B coding at >5x speed. ~18GB int4 = still RAM-offload on 5070.
STEALABLE: (1) Eureka Machine = ForgeEvolve thesis vindicated - automated search found
arch+quant recipe. (2) QUANT RECIPE: quantize lm_head 4-bit too, recover via OFFLINE
SELF-DISTILL (quantized student trains on own samples, bf16 teacher, self-discovered
coreset) - ~half QAT steps, within 0.7pt. Our score_teacher_logprobs+chunked_distill_ce
IS this pipeline - add self-distill recovery stage to ForgeQuant flow. (3) Native-MoE
template: 256 fine-grained experts x512, top-6 sigmoid, shared expert, sinks, QK-norm,
SWA-513+NoPE hybrid.
tekkmaven/flint-1.2B: TAP (Thought-Action Pretraining) - think/tool_call as BASE
objective from step 0; wide-shallow 22Lx5504 (1.5-Pints: wider MLP = more
knowledge/param at small scale); pretrained on FREE Kaggle TPU v5e-8 40h/15B tok.
## 2026-11-04 - Indie-model sweep #2 (all findings)
LEADERBOARD TRUTH (localaistats): LFM3-1.7B Q4_K_M IS the <=4B Pareto leader on
LiveCodeBench v6 (73.46 @ 1.1GB, best score/GB) - backbone choice confirmed.
Competition: K2 Horizon (0.9/3.7B), MiniCPM5-2B, Granite-4.2-3B, Qwen3.5-2B.
TRAJECTORY DISTILL = the proven small-agentic recipe: ricdomolm/mini-coder-4b
(400k mini-swe-agent trajectories + SWE-smith -> beats gpt-oss-120B on SWE-Verified
Bash, 26.8 vs 26.0; 1.7b variant 18.6/50.4 pass@100). Trains on ONE 80GB GPU.
OpenThinker3-7B: SFT-only on OpenThoughts3-1.2M (QwQ-32B traces, 850k math/250k code/
100k sci) -> +15-20pts over R1-Distill-7B. SmolTalk recipe = Magpie-Ultra 400k +
constraints/rewrite subsets ablated at 1.7B.
AGENTIC SPECIALISTS: xLAM-2-1b/3b-fc-r + Katanemo Arch-Function-1.5B = GPT-4-class
BFCL at 1-3B, ~12x faster; Arch uses sub-1B models for routing/jailbreak roles.
CHEAP REASONING: s1K (1,000 curated + budget-forcing "Wait" injection -> o1-preview
level at 32B, 26min train); TinyZero <$30 RL self-verification at 3B; Sky-T1 $450
BUT warns <32B loops on long-CoT (never-loops spec matters).
ARCH INGREDIENTS: Zamba2 (shared-attention x2 ABAB + LoRA on shared MLP + concat
embeddings; 6x less KV); RWKV-7 Goose = TRUE test-time-training RNN (in-context GD
per token, state only 1.6M params L24-D1024 - bounded fast-weight memory exists!);
nanochat: value-embeddings (zero-FLOP params), ReLU^2, QK-norm-post-RoPE, softcap,
vanilla REINFORCE works; Transformer2 (Sakana): SVD task-vectors + router dispatch
= self-adaptive experts, same math as ForgeQ-R.
COMPUTE UNLOCK: Kaggle TPU v5e-8 (flint-1.2B 40h/15Btok); INTELLECT-1 10Bx1T over
14 community nodes (DiLoCo+int8); Nous DisTrO/DeMo trained 15B+40B over internet.
CAUTION: OrionLLM GRM2-3b claims 3B>Qwen3-32B - unverified, low adoption.
## 2026-11-04 - Failure catalog (docs/FAILURE_CATALOG.md)
VERIFIED via sources: ROME/MEMIT at scale = progressive un-editability + gradual-then-
CATASTROPHIC forgetting; RippleEdits: in-context baseline BEAT all editing methods
(facts don't belong in weights - L1/L2 split confirmed). TTT: MLP IO-bound/slow, linear
viable at <=1.3B, ParaTTT fixed parallelism 32x. False-Promise-of-Imitation: teacher
text teaches style not capability - VERIFIED-scored distill (our pipeline) is the
documented fix. Sky-T1: <32B loops on long-CoT SFT (loop-guard is prerequisite).
KNOWN failures recorded: Tay/BlenderBot3 (unverified writes=poison), RWKV recall
(bounded state needs exact backup - our ngram choice), Hyena>KAN>prefix-tuning>sparse-
attn>PETALS>posthoc-ternary>MoE-upcycle graves, Galactica (specialist w/o grounding),
OPT logbook, Falcon-180B/BLOOM (curation>scale), BabyLM floor (distill not pretrain).
META: all failures = unverified writes, missing exact-recall backup, style-vs-capability
confusion, facts-in-weights, single-mechanism designs. Our stack already encodes 4/5.
## 2026-11-04 - AirLLM+MoE verdict (user's old idea - vindicated)
AirLLM alive: v3+ hook-based (transformers drives fwd -> new archs auto-work), does MoE
(671B on ~12GB) + streamed-LoRA train <6GB. Lane is now standard: MoE-Infinity (hot-expert
cache + activation-trace prefetch, bs=1 usage skews, 3-17x), Fiddler (cold-expert: send
KB activation to CPU, compute there - never stream cold weights), kTransformers
(arithmetic-intensity split: attn+shared-expert GPU, routed experts CPU).
ForgeAI edge: domain-specialist routing concentrates (better hit rates), ternary experts
= 0.2B/param (~250tps ceiling even uncached), load_safetensors_pipelined IS prefetch I/O.
Spec reframe: <=1GB VRAM != <=1GB weights - 8-12B ternary MoE total, 1.5B resident,
experts in RAM. CAPACITY lane - below backbone distill; if V15 goes MoE: streaming-native
design (mappable packed experts, domain-clustered layout, router-lookahead prefetch,
Fiddler CPU-exec fallback).
## 2026-11-04 - NGramMemory overwrite fix DONE (step 1 of build order)
Bug: pure +=1 accumulate -> rewrite blends 50/50 forever (ow=2.35/stale=2.36).
Attempt 1 (raw *=decay) REGRESSED ow to 2.86: smoothing floor adds ~28 absolute
mass; small counts dilute lift. FIX: EMA successor rows (row=(1-a)row+a*onehot,
normalized) + row_mass=8.0 effective-count in lift(). SuffixMemory same EMA +
overflow prune for float masses. write_decay=1.0 = last-write-wins.
Probe (exp_live_v2): ngram-only ow 2.35->1.69, dist 2.92->4.23, verb 3.24->4.29;
full ow 0.89->0.52, dist 3.36->4.51. stale==ow is CORRECT (different-suffix=
different-fact under exact semantics). 15/15 livet tests pass incl 3 new.
Research: "Sleeping LLM" (vbario) = our L2 design on consumer hw (wake-log ->
curate -> LoRA -> validation gate + rollback + spaced replay); failure modes:
catastrophic forgetting at high LR, over-aggressive curation -> empty trains.
fla-org ships production gated-delta-rule Triton kernels (GDN2 family reference).
## 2026-11-04 - Step 2 DONE: L1 suffix memory on REAL backbone (exp_l1_real.py)
LFM3-1.7B-CC int4 (1.4GB weights, 2.2GB vram): recall CE 5.10->2.45, hit
0.25->0.50; distractor-immune (3261tok zero change); unseen-suffix lift=0;
cost 0.12ms/step (~0.3%). THREE BUGS FIXED: (1) SuffixMemory.lift_last sign
inverted (log p + uniform floor suppresses observed vs unseen) -> LLR vs
unigram+uniform prior; (2) order_weight**(max_o-o) rank-vs-value bug ->
**(len(orders)-1-i) in BOTH classes (was masking shallow orders at 5e-5);
(3) deployment rule: ingest sentence/segment-level so BPE boundaries match
fresh queries (doc-as-one-string merges across boundaries -> 0 key matches).
EMA note: single overwrite = 50/50 row; dominance needs 2-3 rewrites.

## 2026-11-03 � LLM memory/capability probes (R&D exploration)

Scripts added: `scripts/probe_llm_memory_math.py`, `scripts/probe_exact_decode_tradeoff.py`, `scripts/probe_weight_entropy.py`, `scripts/probe_kv_manifold.py`

### Measured results
- Memory accounting (1.5B-class): AdamW = 8x weights (fp16w+g + fp32 m,v,master). KV = 48KiB/tok (24L x 2kvh x 64hd); KV > weights at T=62k.
- Chinchilla: halving loss-excess = x7.68 params -> memory ~ loss^-2.94.
- softmax rank inflation: rank(QK^T)=64 -> rank(softmax)=512 (T=512, dh=64). Attention matrix can't be exactly low-rank; O=PV has rank<=dh.
- Exact decode tradeoff (fp16 cuda, T=2048, mini 4L model): KV 16.78MB->0MB linear in kept-fraction; latency 1.52->3.48ms (2.3x at f=0). Teacher-forced: max_dlogit=1.95e-3, argmax 126-127/128 on RANDOM init (worst case). fp64 CPU: frac=0 vs 1 = bitwise logits (torch.equal=True), K recompute bitwise. => recompute-for-memory is mathematically exact; fp16 noise = kernel reduction order only.
- Weights (proposer_mla_moe 994MB bf16): byte-entropy 6.29/8 -> lossless floor 1.27x; lzma 1.27-1.64x; mantissa 6.97/7 bits (99.6% max = near-random); H(first-diff)=8.0/8 (decorrelated, delta coding useless); zeros 0.2%. => ~30% lossless savings is ALL that exists on weights.
- KV (real model, real text, 24 GLA layers): stored 512B/pos/layer (post n_kv=2 dedup; latent+rope would be 640B = LARGER here). H(K)=H(V)~6.25-6.33/8 -> ~1.3x lossless. H(dK)~6.0 (delta ~5% gain). cos(K_t,K_{t-1})=0.73-0.99. rank99(K)~64-90/128, rank99(V)~65-120/128. int8 past-KV E2E: cos=0.9995, argmax same; int4: cos=0.953, argmax same (1 sample).
- GLA note: gla_key.py:247 caches reconstructed K,V not latent; for THIS cfg (nkv=2,dc=256) reconstructed is smaller. Latent-caching only wins for large n_heads*n_kv*hd.

### Interpretations
- "identical output" tiers: bit-exact (recompute only / reduction-order control) > argmax-exact (int8/int4 KV easily, ~4-8x smaller) > distribution-exact (lossy zoo).
- Three exact-relief axes: (a) time<->space: recompute/vT-checkpoint (linear for attn, sqrt for sequential state); (b) architecture-time factorization (GLA/GQA/MLA latents � must be trained in, not discovered); (c) entropy coding (~1.3x, weak everywhere).
- Novel direction: per-layer memory-type optimizer � conv/SSM O(1) already, SWA O(W), GQA gets fractional-cache+recompute; plus top-attended-position fp16 residual correction over int4 base for argmax-exactness at ~4x.
## Step 4 verdict (2026-11-04): fuzzy fast-weights FAIL the recall gate
titan+gdn2 gain plateaus ~2.0n at reps 5/50/200 (vs ngram ~4.3); gdn2-only
= 0.00 at ALL reps. Exact-suffix EMA memory is the L1 mechanism; titan
dropped for fact-recall role (recorded dead end; may revisit for fuzzy
context matching, but alt-ctx also showed ~0 transfer).
## Loop research: detection-as-stop is the ONLY zero-cost loop ender �
penalties/DRY forbid forms not behaviors (vllm-mlx verified live). Ref impl:
exllamav3 LoopDetector (circular buf, per-period streaks, heap). Budget-
forcing = suppress end-token + append "Wait" to extend / force close-tag.
## Step 3 DONE (2026-11-04): L2 consolidation works E2E on real 1.7B
exp_l2_consolidate.py: ingest->SuffixMemory->draft-reconstruct->LoRA(r8,
q/o/gate/up/down, 360K params)->QA retention eval. Operating point found:
lr5e-4 + replay buffer(4 rows): fact CE 7.22->5.42 (-1.8), ctrl drift
+0.31 (vs lr3e-3 no-replay: -1.5 but +3.86 CATASTROPHIC forgetting =
Sleeping-LLM failure mode reproduced+fixed). 134s/fit, VRAM 11.8GB(tight).
Draft hit-rate was LOW (mem draft reconstructs only ~1-2/8 fact answers �
exact store carries continuation not QA-form; rows fell back to doc pairs).
NEXT: consolidation loop needs multi-fact docs + more steps/rank for full
recall; periodic (sleep-cycle) schedule.
## 2026-11-04 overnight batch results
SCORER: fixed cuda/cpu gather bug (tgt.to(lp.device)); E2E run 6 rows,
0 unmapped, 2.6s scoring (Qwen2.5-Coder-3B int4 teacher -> LFM3 student,
char-offset mapping works).
AGENTIC_DISTILL: tool_use_loop.compute_reward was deleted in AZR merge ->
implemented _local_compute_reward heuristic (answered .30/tool_use .25/
clean_stop .20/substance .15/novelty .10); path functional again.
FORGEQ-R r=16 pack (lfm3_1p7b_cc_fqr16, 1.46GB): rel Frobenius
0.1164->0.1149 = ~1% reduction. VERDICT: int4-groupwise error is
near-white -> rank-16 SVD captures ~9% flat spectrum. Residual patching
is the wrong repair for diffuse error; prefer FLINT self-distill recovery.
(Device-mismatch bug in residual path also fixed.)
TERNARY KERNELS (V15 ammo): bitnet-tc = ternaryxINT8 tensor-core GEMM,
4.8-9.4x decode GEMMs, ~150TFLOPS; TernaInfer true-1.58b @1.53x; spbitnet
adds 2:4 sparsity on Ampere consumer. MoE offload practice confirmed:
llama.cpp --cpu-moe/--override-tensor "ffn_.*_exps" - validates the
streaming-native expert-packing design.

## 2026-11 session 2 (post-summary continuation)

### Research findings banked
- MEGa (arXiv 2504.21239): per-memory gated LoRA adapters — QA 0.98,
  MMLU preserved (61.8 vs 62.5 base), recall cosine 0.95 on fictional-
  char + Wikipedia-event datasets. Weakness: top-1 gate rate 85-88% +
  softmax dilution blends all adapters; per-memory adapters grow
  linearly. ForgeAI takeaway: L2 could shard per-document gated LoRA
  instead of one consolidated adapter — better isolation; middle ground
  = one adapter per sleep-cycle batch (current design) + replay.
- ParaTTT (arXiv 2610.38255): fixed-point iteration parallelizes deep
  TTT — 2 iters suffice, 32x faster inner trajectory at 4K ctx, 340M
  model trains 4.9x faster, matches GDN/Mamba2 with better recall.
  Deep-memory TTT is a viable L1-upgrade lane (GDN2 kernel already
  exists via fla-org reference).
- Sleeping-LLM failure confirmed live in exp_l2_consolidate: lr 3e-3
  gave +3.86 ctrl drift; lr 5e-4 + replay -> +0.31. Documented fix works.

### Shipped this session
- SuffixMemory.save/load: pickle persistence, version-tagged, exact
  state parity (test_suffixmem_save_load_parity — 16/16 livet).
- Loop detector verified end-to-end? NOT yet — lfm3_1p7b_cc_fq exits
  via EOS too fast on generic loop-priming prompts (2-20 tok outputs).
  Unit coverage is good; E2E still open.
- Distill data-path: scorer 150/150 rows 0 unmapped ->
  data/sft/evolve_sft_v3.scored.jsonl. tokenize_example drops are purely
  seq_len overflow (1024->43/150, 2048->144, 4096->150).
- AGENTS.md gained 'Live-Learning Stack Status' section.
- sft_train smoke (earlier): 4-step --distill ran end-to-end, merged-
  LoRA checkpoint saved, VRAM 7.79GB. MuonSFBlockwise auto-selected.

### Running now
- b762ed: 60-step --distill on 150 scored rows, seq2048, b4 ga4, lr1e-4
  -> research/checkpoints/lfm3_cc_distill_v1.safetensors
  SUPERSEDED: that run died on the cache-poison bug below; relaunched as
  lfm3_1p7b_cc.distill60.safetensors on the fixed code.

## Distill CUDA assert — RESOLVED (3 independent bugs, proven by instrumentation)

Symptom: `Assertion 'ind >=0 && ind < ind_dim_size' vectorized gather` device
assert; async-misreported at embed/rms_norm frames; every isolated component
passed when probed directly.

Bug 1 (THE crash): `_tokenize_cached` keyed DiskTokenCache under literal
"default" — no tokenizer identity in the key. A prior Qwen-vocab run had
cached rendered ChatML texts (im_start/im_end markers) as Qwen ids up to
151645; the LFM3 run (128000-vocab embed) was served those ids -> embed
gather OOB. Verified: 102 hits under "default", 65 with ids >= 128000.
Fix: hash = class_vocabsize_eos_pad at BOTH the disk layer and the
in-memory LRU (same collision existed there). sft_train.py:476-518.

Bug 2 (silent wrong-vocab): `get_tokenizer()` default = ForgeLM V2 tokenizer
(65536 vocab) whenever --tokenizer absent — wrong vocab for any foreign
checkpoint (would silently garbage-tokenize, no crash). Fix: prefer
args.checkpoint's own tokenizer.json; canonical only for from-scratch.
sft_train.py:1310-1320.

Bug 3 (chunked CE was dead code + distill silently dropped):
  a) cfg.use_chunked_ce/mtp set on trainer cfg, but ForgeEngine builds its
     own ModelConfig -> model.config never saw them. Synced post-build.
     sft_train.py:1494-1505.
  b) `sw = rw` unconditionally (reward_weights all-ones for reward-less
     data) forced the manual [B,T,V] logits path every step — chunked CE
     unreachable AND teacher_logp ignored there (distill degraded to plain
     CE). Fix: uniform-1.0 sample_weights dropped to None. sft_train.py:669+.

Permanent guard added: llm.py _forward_impl checks idx bounds vs
num_embeddings when self.training -> readable RuntimeError instead of an
async gather assert. Skipped in decode hot path.

Verification: 4-step real run clean (loss 36.5 finite, VRAM 7.78GB, merged-
LoRA ckpt saved+verified); 46/46 engine tests; 2/2 new regression tests
(tests/unit/test_sft_tokenizer_namespace.py).

Lesson learned: when every component passes in isolation but the assembled
path asserts, the delta is in the GLUE — here, a cache layer no probe
exercised. Runtime instrumentation (bounds-check at the asserting op)
beat another hour of code reading.

## Session: distill60 validated + tokenizer self-description (continuation)

- distill60 ckpt VERIFIED: logits ~= base (top8 same, std 2.234); whitespace gen was
  a wrong-tokenizer artifact, NOT training damage. from_checkpoint gave the bare
  .safetensors the canonical 65536-vocab tokenizer (HFCompat eos=519) for a
  128000-vocab model -> mis-tokenized prompt -> degenerate output.
- Fix chain now complete: trainer meta records "tokenizer_path"; engine
  _tokenizer_from_meta() resolves file ckpts before vocab dispatch. distill60
  meta.json patched manually (tokenizer_path -> research/checkpoints/lfm3_1p7b_cc).
- After fix: vocab 124893, eos 124900, generate('def add(a, b):') -> clean code
  matching base style.
- Pipelined loader fixed for 0-dim qscale scalars (reshape(-1) then uint8 view):
  distill60 load 4.9s fallback -> 0.8s pipelined, total 2.2s.
- Tests: test_sft_tokenizer_namespace.py now 3/3 (added _tokenizer_from_meta cases);
  test_livet 17/17.
- Lesson banked: raw-logits probe FIRST before blaming training — decode-path
  artifacts (tokenizer, offload, templates) mimic degradation.
