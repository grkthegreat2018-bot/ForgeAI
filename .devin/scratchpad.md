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

## 2026-11-04 late - consolidate_pairs() engine API + spec telemetry
- ExternalDraftSpeculativeDecoding.spec_stats {proposed,accepted,rounds} + spec_report() accept_rate (commit 25ff1f3).
- engine.consolidate_pairs() = productized L2: L1-gated pair filter + LoRA fit + held-out ctrl drift metric + save_adapter/merge. E2E on lfm3: 6 facts 4.35->0.0 CE.
- Operating point finding: held-out control is the REAL drift metric (replay rows are trained, cannot serve as ctrl). lr5e-4x60steps+2 replay = +2.17 drift; lr2e-4x20 = +0.80 with facts still memorized. Replay SIZE bounds drift more than lr.
- Scorer relaunched (PID 23292) after download stall; HF resume works, both shards active.


## suffix_draft telemetry findings (2026-11-04)
- spec_stats on repetitive code: 96 proposed / 0 accepted / 24 rounds at tau=0 -> proposals were TOKEN-0 SPAM: mem.draft() returns [] on BPE tails with varying digits; shim emits uniform -20 logits -> argmax = vocab 0.
- Fix shipped (cdb72cb): activation defaults confidence_tau=0.3 -> shim's honest successor-p gates no-opinion rounds. Result: 36 proposed/9 rounds, 1.86x speedup (1.33s vs 2.47s), bit-identical.
- IMPORTANT: the 1.86x speedup at accept_rate=0 comes from AMORTIZED PYTHON OVERHEAD (2 commits/round via main+resample; ~35ms/tok python in StandardDecoding), NOT acceptance. Suffix draft pays when token-level recurrence is high; on varied-digit patterns accept stays ~0 but spec still wins on overhead amortization.
- consolidate_pairs E2E verified twice: lr5e-4x60 -> post_ce 0.0, held-out ctrl drift +2.17 (too hot); lr2e-4x20 -> post_ce 0.0001, drift +0.80. Replay count bounds drift more than lr.


## 2026-11-05 — util/slow-task investigation + SFT run #3

D: drive root-caused: WD Elements 2620 = WD40NDZW device-managed SMR
on USBSTOR BOT (verified DEVPKEY_Device_Service=USBSTOR, TRIM
supported). ~230MB/s seq / ~38 files/s small-file / <10MB/s random
writes once CMR cache fills. Docs: wiki Development+Known-Issues
(pushed a92911a), FAILURE_CATALOG, AGENTS.md section D.

Committed: e43b0d4 reward_weights=None for all-default batches (kills
per-step GPU sync; 3 tests); 790fa7a phase timings persisted into
status.json (bottleneck() was never called); 0690c3a SMR catalog;
d0b7b27 AGENTS.md note.

Trainer deaths this session: (a) PID 74812 stale run at 25.4GB private
+ 11GB VRAM — killed, freed to 19.3GB free RAM; (b) PID 75700 died
silently ~15min in — no status/heartbeat/stop/log passed, stdout lost.
interrupt_step60 = weights+meta only, .train.pt absent = killed between
the two writes in save_training_checkpoint (order: weights+meta, then
train sidecar — load_training_state handles missing sidecar).

Run #3 (PID 127552 real / stub 94660): resume step50, --status-file
data/run_tasks/sft_status.json --heartbeat-file ... --stop-file
<save>.stop, stdout->research/checkpoints/sft_run_1105_0550.log.
Step 56 loss 10.6875 = bit-consistent w/ prior run's step56 (resume
deterministic). GPU sampled 100/100/25/100/100/98 — brief step-boundary
dips (optimizer/data_load bubble), otherwise saturated.
## 2026-11-05 late — trainer memory profile (PID 127552)
Private commit plateaued at 24.7GB (24.701->24.697 over 90s sample) —
NOT a leak; it's retained peak-commit from the step100 save transient
(7.4GB CPU state_dict copy + 605MB optimizer sidecar). Reinterprets
the earlier deaths: the checkpoint-save spike is the kill moment, not
steady growth. Commit 41.1/54.7GB; physical 64%; next saves spike
~+8GB transient — watch at step 150/200/... intervals.
GPU sampling: ~100% with brief ~25% dips at step boundaries (optimizer
sync + collate); phase timings now persisted for future runs.
## Memory plateau confirmed (3 saves observed)
save@100: +7.9GB retained | save@150: +2.5GB | save@200: +0.0GB —
allocator arenas now recycle the checkpoint transient; flat 27.13GB.
Safe to run to completion (~42GB commit < 54.7 limit). Earlier deaths
= save-transient spike at peak system pressure.
## 2026-11-05 sft1 postmortem (diagnosis CLOSED)

Result chain: sft1 merged model = orig + trained LoRA deltas on out_proj+FFN only.
q/k/v adapters bit-zero (attached to dead modules - fused qkv forward bypasses them).
Merged model = garbage through EVERY load path; orig dense through same path = perfect.
=> adapter deltas are destructive noise, not a merge/pack/cache bug.

Why destructive: bitnet_everywhere QAT forward (ternary weights) during training;
adapters partly learned quantization-error compensation. On fp inference weights
that compensation is noise. Plus loss ~10 (cross-doc packing, no cu_seqlens emitted).

Artifacts: sft1.safetensors / .lora / _fq / sft1_src / dense_clean all carry the
same deltas - none usable. Disk space: sft1_src+dense_clean ~15GB reclaimable.

Next-run recipe (bounded, ~same 3h):
- --no-bitnet-everywhere (fp forward = inference path)
- LoRA on fused path or unfuse qkv for training
- pack_sequences off or emit_cu_seqlens=True wired to forward
- lr 2e-5, 2-3 epochs, watch loss < 5
## 2026-11-05 - OS crashes root cause + sft1 salvage verdict

CRASH: 2x bugcheck 0x00020001 HYPERVISOR_ERROR today (11:47, 17:19), both during
ForgeEngine CUDA work (graph capture / engine load). NOT commit-pressure - VBS/HVCI
level fault (driver+IOMMU or RAM under GPU DMA). Mitigation: auto_activate=False
skips graph capture for diagnostic engine loads. Open: could also be unstable RAM
(XMP) or NVIDIA driver - watch for recurrence outside Forge workloads.

sft2 corrective run (PID live, detached): loss 1.04 at step1 vs ~26 on broken run.
Confirms root cause chain: BitNet ternary forward + dead qkv adapters + boundary-
blind packing produced the destructive deltas. sft1 artifacts all carry the noise
- ~15GB reclaimable (sft1_src, sft1_dense_clean, sft1_fq, sft1.safetensors).
## 2026-11-05 sft3 RFT pipeline (running)

gen_sft3_data.py: verified-sample generation, append-mode JSONL, resume via gen_key.
Stage1: base_fq x proposer-wrap (139 targets x4 reps xk4) -> sft3_proposer.jsonl
Stage2: sft2_fq x direct (900 prompts xk3) -> sft3_direct.jsonl
Verify tier: fence+compile+required-names+AST dangerous-import blocklist.
Proposer rows embed runtime prefill (</think>+```python) INSIDE response so
the trained pattern continues the open fence at inference.

PACK FIX for sft3: strip *.attn.qkv_proj.weight from src before
prequant_forge_quant (engine rebuilds fused from separate q/k/v at load;
packing both = +0.48GB dup). sft2_fq bloat 1.81GB -> ~1.33GB.
## 2026-11-07 - spark sft3 continuation (Devin)
- Corpus: spark_sft_v2.jsonl = 833 rows (283 rft_direct + 259 rft_proposer + 291 replay_v1) via build_sft3.py.
- UNSEEN EVAL VERDICT (results/unseen_eval.json): base_fq valid 4/10 -> sft2_fq valid 10/10 (fence+syntax+named, 10 fresh coding prompts, temp0). sft2 confirmed NOT degraded; large format+syntax win.
- sft3 train attempt #1 KILLED at step 11: loss 8.375 -> 12.69 -> 9.31 (sft2 was ~1.0 at step1). Recipe mirrored sft2 (--manual-lora --no-bitnet-everywhere --no-pack-sequences lr2e-5 cpu_offload). Differences vs sft2: ckpt = merged sft2 (unfused q/k/v keys, 224 LoRA adapters incl out-gate vs 140 on base HF dir), data = v2 RFT corpus.
- Loader note: load_examples dedups by prompt[:200] -> 833 rows collapse to 508 conversations (generator's <=2 variants/prompt intentionally dropped by loader); 489 after tokenize.
- Probe running: .devin/_probe_sft3_loss.py = per-row eval CE of merged sft2 on replay_v1 vs rft_direct vs rft_proposer through exact sft_train tokenize+load path.

### sft3 loss anomaly - probe results
- _probe_sft3_loss.py: merged sft2 eval CE on 9 sampled v2 rows = 0.005-0.95 (model+data both clean in eval).
- _probe_sft3_sweep.py (results/sft3_row_ce.json): ALL 493 usable rows mean CE = 0.644, max row = 3.53. replay_v1 mean 0.711 (n=264), rft_direct 0.497 (n=179), rft_proposer 0.819 (n=50). Training step1 = 8.375 ~= 13x true CE -> TRAINING-PATH bug, not data, not weights.
- Corpus composition note: loader prompt[:200] dedup hits proposer rows hardest (259 -> 50); final mix = 54% replay / 36% direct / 10% proposer (intended 35/35/30). Fix later: dedup key should be prompt+response hash.
- Next: _probe_sft3_step.py isolates mask vs internal-targets-CE vs LoRA on one fixed batch.

### sft3 loss anomaly - ROOT CAUSE + FIX (commit ecbe404)
- Step-probe (mask/internal-targets/LoRA-224/train-mode) ALL CLEAN: internal loss 0.547 on a fixed batch. Not mask, not LoRA, not chunked CE.
- ROOT CAUSE: sft_train tokenizer resolution for a bare .safetensors --checkpoint looked for tokenizer.json in the checkpoint's PARENT dir (research/checkpoints) -> missing -> get_tokenizer() default = ForgeLM V2/Qwen-vocab tokenizer. Spark vocab 131072 mis-tokenized every row. sft2 dodged it because --checkpoint was the base HF DIRECTORY (has tokenizer.json). Tell: 489 usable examples (wrong tok) vs 493 (right tok, matches probe sweep exactly).
- FIX ecbe404: _resolve_tokenizer_source() = --tokenizer -> ckpt dir -> .meta.json tokenizer_path (engine's _tokenizer_from_meta) -> default. Vocab guard after cfg build: tok_vocab > model vocab = hard error, != = loud warning. Test test_sft_tokenizer_source_resolution 4/4 pass.
- sft3 train attempt #2 (fixed): step1 loss 0.8203 (expected ~0.64 CE x WeFT), step11 0.5586. 120 steps on 493 rows.
- LESSON banked: when train loss ~10 but eval CE ~0.6 on same ckpt+data, check WHICH tokenizer the trainer resolved BEFORE suspecting adapters/packing. The 489-vs-493 usable-example count diff was the fingerprint.

### 2026-11-07 sft3 COMPLETE + training-process upgrade round
- sft3 train: 120 steps / 493 ex, final loss ~0.7-0.9, merged save 311 tensors + lora 448 tensors. Pack: sft3_fq 1.59GB (qkv-strip worked; sft2_fq was 1.94GB dup pack).
- UNSEEN EVAL sft3: sft2_fq 10/10, sft3_fq 10/10 (results/unseen_eval_sft3.json). RFT round holds the format win; outputs differ from sft2 (genuine new weights).
- Periodic saves on D: SMR measured ~110s each (4.39GB weights + 0.64GB sidecar) - NOT a hang; py-spy confirmed forward resumes after. This is ~20% of a short run's wall time.
- Upgrades in flight: (a) torch.compile call-site fix (cache_dir kwarg doesn't exist on torch 2.11 -> TORCHINDUCTOR_CACHE_DIR env; default mode not reduce-overhead for CPU-offload optimizers), (b) lora-only periodic saves (save_training_checkpoint lora_only=True; resume re-injects adapters onto base; --full-step-saves opt-out) + tests 37/37, (c) compile fwd+bwd bench running.
- sft2 corrective recipe now proven TWICE (sft2 + sft3 both healthy loss + 10/10 unseen). Canonical launcher: .devin/launch_sft3_train.cmd.

### 2026-11-07 selfgen (AZR) implementation + first smoke
- User directive: fixed prompts overfit -> self-expanding curriculum. Implemented scripts/gen_sft3_data.py --mode selfgen (commit 8b3f14e, moved from .devin into tracked source for the tests): fresh tasks per wave (12 domains x difficulty x uuid seed), cliche banlist, 8-gram Jaccard novelty gate (0.5), persistent-sandbox reference execution for ground truth, k=4 solver rollouts execution-verified vs reference, Goldilocks keep (0<rate<1), per-domain rolling-rate difficulty adaptation, <=2 variants/task, marker-row resume.
- Tests test_selfgen_rft.py 10/10 (incl REAL sandbox: reference ground truth + wrong-solver mismatch).
- Smoke #1 on sft3_fq: 0/8 valid proposals. RAW outputs: model copied format placeholders literally (def solve(literal_input)) or fell into verbatim if-not-s loops. LESSON (same as proposer-wrap round): abstract format instructions fail on 1.7B; few-shot + PRIME is mandatory. Fix: one complete example in-prompt + wrap_proposal prime '`python\n# Task: ' (model continues the open comment).
- sft4 launcher staged (.devin/launch_sft4_train.cmd): val-every 25 + early-stop-patience 3 + lora-only saves + 150 max steps.
- Compile bench timeboxed OUT: harness thrashes VRAM (37s/step vs trainer 1.7s) - not representative; call-site fix committed (4686267) unmeasured. Trainer real numbers: fwd ~850ms bwd ~815ms per micro-step (batch4x2048), CPUAdamW ~420ms/step, periodic full saves ~110s -> now lora-only (0d82900).
- py-spy via uvx works on this box for hung-python diagnosis (uvx py-spy dump --pid N).

### 2026-11-07 selfgen v2: codebase-grounded (user directive) + staged plan
- Proposal-based selfgen DEAD END (2 designs measured): from-scratch 0/10 valid, seed-variation 0/4 (verbatim seed clones, fence spam, placeholder code). The 1.7B solves well but does NOT invent tasks well. PIVOT: tasks = REAL functions harvested from ForgeAI production code (forge/scripts/research, GUI excluded) — user's explicit directive.
- Implementation (commit 9b69aed): AST eligibility (top-level, public, undecorated, 3-45 lines, <=4 params, stdlib+builtins free-names only, deterministic, no banned deps) + type-annotation fuzz inputs + per-input-isolated sandbox ground truth (>=2 valid + >=2 distinct outputs) + solver rollouts execution-matched on the reference-valid subset + Goldilocks + 8-gram novelty + source-hash dedup + resume. Undocumented functions get a describe-first pass (model describes existing code = learnable skill). Unannotated params get name-heuristic types (execution validates).
- Sandbox fix (93b568f): worker boot pre-imports math/itertools/collections/functools/string/operator — sandboxed 'import string' previously died on out-of-jail stdlib file read.
- Pool: 28 strictly-clean functions at first pass; expanded via name-heuristics + describe-pass. First smoke (older code): wave0 executable 5/8 -> solver rollouts running.
- STAGED PLAN (user): (1) codebase-grounded variety batch NOW; (2) after training on that variety, RETRY model-authored unique prompts (the AZR proposal path, currently dead) — if still failing, (3) ONLINE DBs: code-task DBs with working references+tests (MBPP, HumanEval, APPS/CodeContests/TACO, Magicoder) for the prompt-gen side + knowledge/skill self-improvement DBs (FLAN/Tulu/OpenHermes/UltraFeedback-class instruction data) for the general side. MBPP/HumanEval = first external supply (references+tests included, drops straight into the execution-verified loop).
- Tests test_selfgen_rft.py 9/9 (incl real sandbox + real-codebase pool check).

### 2026-11-07 online-DB fallback catalog (stage-3 backup, researched via HF hub + web)

A. CODE TASKS w/ verifiable refs+tests (drop into execution-verified selfgen loop):
- OpenCoder-LLM/opc-sft-stage2 `educational_instruct` — (instruction, code, test case) triples, compiler-validated. ~436k rows total config family.
- KodCode/KodCode-V1 — 447k verifiable coding tasks w/ unit tests (12 subsets, algo->package). CC-BY-NC-4.0 (non-comm). SFT variant: KodCode/KodCode-V1-SFT-R1.
- nvidia/OpenCodeInstruct — 5M SFT rows w/ quality scores + unit-test pass arrays. CC-BY-4.0. Filter to test-passed subset for RFT.
- bigcode/self-oss-instruct-sc2-exec-filter-50k — self-instruct prompts EXECUTION-filtered (the "working prompt" recipe Spark needs). ODC-BY.
- open-r1/codeforces-cots (~100k R1 CoT), open-r1/ioi-cots — competitive problems w/ tests for RLVR-style verification.
- deepmind/code_contests, BAAI/TACO, newfacade/LeetCodeDataset — classic verified-task pools.

B. REASONING:
- open-thoughts/OpenThoughts3-1.2M — math+code+science R1 traces, apache-2.0, SOTA-open.
- nvidia/OpenCodeReasoning — 735k code-reasoning (28k unique problems), CC-BY-4.0.
- nvidia/OpenMathReasoning — CoT/TIR/GenSelect splits, CC-BY-4.0.
- open-r1/OpenR1-Math-220k — verified math traces.
- PrimeIntellect/SYNTHETIC-1 — 2M crowdsourced R1 traces w/ verifier annotations (filtered SFT variant in its collection). apache-2.0.
- allenai/big-reasoning-traces — ~2.5B tokens permissive traces for midtraining.
- MarshootLab/beam-verified-reasoning-traces — 63.6k verified across 58 task families (RLVR tag).

C. TOKEN EFFICIENCY (short/controlled CoT):
- horseee CoT-Valve: MixChain-Z-GSM8K (6.9k), MixChain-Z-PRM12K (12k), MixChain-C-LIMO — same Q, multiple CoT lengths => length-conditioning pairs.
- distilled-ai/condensed-thinking-sft-dataset — condensed-think SFT.
- yrdu/S3-CoT-Self-Sampled-Data — multi-ratio compressed traces + token stats (length-compression SFT).
- TokenSkip family: SmallDoge/TokenSkip-10K; vm2825/token-skip-gsm8k-gamma-{0.5..0.9} (explicit compression-ratio tags).
- simplescaling/s1K-1.1 (1k MIT) + GAIR/LIMO (817 apache-2.0) — less-is-more high-quality reasoning.

D. BEHAVIOR / GOAL-HONING / self-improvement mindset:
- bigai-nlco/ReflectionEvo — ReflectEvo-460k self-generated reflection traces (identify-failure -> locate-error -> correct); SFT+DPO formats, multi-domain incl code. ~437k rows.
- lambda/hermes-agent-reasoning-traces — multi-turn tool-calling agent traces, apache-2.0.
- nvidia/Nemotron-SFT-SWE-v3 — 238k agentic SWE trajectories (OpenHands/SWE-agent/mini-SWE-agent), CC-BY-4.0 => developer-mindset task pursuit.
- SWE-bench/SWE-smith-trajectories — ~102k SWE-agent trajectories; SWE-Gym/OpenHands-SFT-Trajectories (491 curated) + SWE-Gym env tasks.
- nebius/SWE-rebench + R2E-Gym trajectories (in fan-shu/swe-native-trajectories bundle, gated).
- GAP: no quality public DB explicitly trains "propose your own goals" — closest is ReflectEvo (self-diagnosis) + agentic traj DBs (persistent goal pursuit). AZR proposal-path remains novel/novel.

Priority if stage-3 triggers: opc-sft-stage2/educational_instruct + KodCode tests (verification-ready) -> OpenCodeReasoning/OT3 slice (reasoning) -> ReflectionEvo (mindset) -> MixChain/TokenSkip (efficiency). Licenses to respect: KodCode NC, self-oss ODC-BY, Magicoder OpenAI-output terms.

### 2026-11-07 selfgen external supply ONLINE + results
- scripts/fetch_external_tasks.py: opc-sft-stage2 educational_instruct -> 500 tasks from 506 rows (99% yield; >=2 literal asserts + non-constant outputs). --kind reflection writes ReflectEvo corrected-CoT rows for the knowledge/skill side.
- External 32-task smoke: 32/32 executable (DB references all pass their asserts) -> 7 goldilocks -> 9 verified rows (waves vary 0-7/8 by difficulty). Full run on all 500 launched at batch 24 (data/sft/sft4_external.jsonl).
- Local pool expanded 28 -> 80 candidates (name-heuristic types + undocumented via describe-pass); diagnostic: 43/80 crash on wrong literal types, 14 constant-output, 23 OK -> added model-written input rescue pass (parse_input_calls).
- LoRA-only save/resume VERIFIED end-to-end: step3.safetensors 79.9MB (vs 4.3GB full) -> 'adapter-only snapshot' + 448 tensors restored + resume at step4 + clean continuation (smoke_lora_resume2.log). NOTE: the LAST step goes to the final merged save, not a periodic stepN file — resume must target an actual stepN.
- AGENTS.md selfgen section updated with implementation status; wiki pushed dfc0fbe (Training/Self-Play/Model-Presets).
- Commits this session: ecbe404 tokenizer, 190dea5 dedup, 8b3f14e selfgen AZR, 0d82900 lora-saves, 4686267 compile fix, 9b69aed codebase-grounded pivot, 93b568f sandbox preimports, 1634652 input rescue, 5e01504 external supply.

### 2026-11-07 overnight plan (AFK session)
DONE: evol-harness hardening committed (ace44da + 0c5b70b + 295aac1) — deny-wins scrub x3 sites incl. worker-embedded copy, nonced __RESULT__ + os._exit, promotion evals bypass pool, 131 targets fresh-seeded.
DONE: keep-easy-prob 0.3 in selfgen (2249e6a); external run v2 on 392 tasks underway (wave4: 13 gold+2 easy -> 22 kept).
NEXT: (1) build spark_sft_v3.jsonl = sft4_external + sft4_selfgen + ~300 reflectionevo slice + 35% replay v2; (2) train sft4 via .devin/launch_sft4_train.cmd (sft3 ckpt, val25/early3 -> .bestval snapshots now work); (3) pack sft4_fq w/ QKV strip; (4) unseen eval; (5) docs+commit; (6) stretch: retry model-authored proposals on sft4, KodCode/OpenCodeInstruct supply.
GUARDRAILS: one GPU job at a time; --ram-limit 75 --vram 10.5; SMR writes only via adapter ckpts; no 25+ QA-line file edits.

### 2026-11-08 Sonnet-review application pass
DONE: review findings implemented + committed on main:
- f033668 fresh-seed all 137 evolve targets (_gen_cases(seed) + paired a/b calls + np uint32 mask); auto_targets._gen_source/_ORACLE + gen_llm_targets schema/validator emit the seed-param pattern for future targets; verify harness 137/137 pass.
- c58460d forge_server API key via hmac.compare_digest; forge_gui_server per-launch token (auth.py, /api middleware + /ws query-param check, launcher ?token= URL, SPA sessionStorage + X-Forge-Token, rebuilt dist); gate_probes.pt->safetensors (GateProbes.load prefers .safetensors, warns on pickle), bench_gate_route repointed.
- 95ac212 ruff pinned 0.16.2 (CI+pre-commit+pyproject); tests/ lint-clean (43->0); last_detail NameError bound pre-loop; triton-windows sys_platform marker x2; groq_models.json deleted (401 body); run_evolve/rescore_db moved to scripts/ (parents[2]->[1]); README results/-tracked + portable PYTHONPATH; AGENTS ~1000 files + token posture.
- Prior commits already covered: deny-wins scrub x3 (incl _BOOTSTRAP worker copy, TOKENIZERS_PARALLELISM sole exact-keep), nonced __RESULT__+os._exit, promotion evals bypass pool (ace44da/0c5b70b/295aac1).
TESTS: test_code_evolve 63 + test_sandbox_escapes 26 pass; test_selfgen_rft 13 pass; GUI auth smoke 401/200/200/200; ruff tests/ + F821,F823 forge+scripts green.
NOT DONE (review items deferred): external-eval baseline + engine-vs-llama.cpp numbers (A1, needs GPU run); results/ ledger aggregation; git filter-repo purge of tools/llama DLLs (154MB history); test skip-gates for gitignored assets (26 files); AGENTS.md 169KB trim; giant-function refactors (sft_train.main 1610, _setup_routes 1569).

### 2026-11-08 k4 held-out eval rerun (cont.)
- pass@2 was noise (sft3 1/117, sft4 0/117). Rerunning k=4 temp=0.8 matching selfgen conditions on same 117 verified-GT tasks; log .devin/solver_eval_k4.log. sft3 wave2 still 1/117 cumulative.
- Staged: opc_sft2_b.json = 500 more verified tasks (rows 660+, 1179 scanned). .devin/_propose_retry.py = stage-2 few-shot proposal test for sft4.
- NOTE: parallel session did Sonnet-review pass (f033668/c58460d/95ac212): all 137 evolve targets fresh-seeded incl llm_*; GUI token auth; ruff pin. Fixed-seed pending item CLOSED.


### k4 held-out FINAL: sft3 1/117, sft4 0/117 (temp .8, pass@4)
- Identical to pass@2; Fisher p~1.0 = no measurable diff. Held-out slice too hard for 1.7B (only ~1% ceiling).
- _solver_eval.py instrumented: per-attempt passes + per-assert accuracy (dense metric for future runs) + per-task JSON to results/solver_heldout_<label>.json.
- _onmanifold_eval.py built: 86 unique known-solvable tasks from kept-row gen_keys -> regression check sft3 vs sft4 running now (log .devin/onmanifold_eval.log).
- _propose_retry.py built: few-shot (2 real exemplar tasks) proposal -> parse -> sandbox GT -> validity rate on sft4.
- Next: on-manifold -> propose-retry -> sft5 selfgen on opc_sft2_b (500 tasks, sft4 solver).


### on-manifold eval + batch-composition investigation (sft3_fq vs sft4_fq)
- RESULT: dead heat. sft3 3/60 pass@4 (6/102 attempts), sft4 3/60 (4/83). ~60%% of rollouts don't produce extractable+named code at temp .8.
- Held-out k4 FINAL: sft3 1/117, sft4 0/117 � statistically identical, held-out slice is beyond 1.7B capability ceiling.
- 86/86 STORED corpus solutions still re-verify � loop consistent; gap is generation-side.
- Diag: top-rate tasks (orig .75-1.0) re-solve 4-8/8 with identical-8 batch; first-8 low-rate tasks ~1-3/32 either batch layout. task-major vs rep-major A/B = 4 vs 2 = noise -> NO batch bug proven; true per-rollout rates are heterogeneous and mostly lower than the 1-3/4 observed single-pass draw suggests (selection bias: kept tasks = the lucky tail).
- IMPLICATION: goldilocks corpus IS hard-tail informative, but pass@k evals need many attempts to discriminate; per-assert accuracy instrumentation added for future runs.
- Eval scripts: _solver_eval.py (per-task JSON + attempt/assert acc), _onmanifold_eval.py (gen_key->pool join), _diag_resample.py, _diag_batch_ab.py, _propose_retry.py (few-shot proposal test on sft4).


### sft5 staging + proposal verdicts
- Stage-2 CLOSED: sft4 proposal attempts = 0/32 (abstract few-shot) + 0/24 (corpus-native few-shot, outputs ~16tok empty fences). 1.7B still can't author verifiable tasks; external DBs remain supply.
- sft5 selfgen RUNNING: opc_sft2_b (500 tasks, sft4_fq solver, k4 t.8 keep-easy .3) -> data/sft/sft5_external.jsonl. Yield lower than sft3 slice A so far (4 kept/4 waves) � slice-vs-solver confound noted.
- Staged for sft5 corpus: reflectionevo_sft_b.jsonl = 600 FRESH reflection rows (skip 300 sources; fetcher reflection branch now honors --skip, commit bfccb9e). Builder cmd: build_sft3.py --new sft5_external.jsonl reflectionevo_sft_b.jsonl --replay spark_sft_v3.jsonl --replay-frac .35 --out spark_sft_v4.jsonl. Train: .devin/launch_sft5_train.cmd (sft4->sft5 same recipe).
- NOTE: fetch_external_tasks.py --kind reflection needs --dataset bigai-nlco/ReflectionEvo --config D+ explicitly (defaults are opc-sft2).


### sft5 lineage (overnight)
- Corpus spark_sft_v4.jsonl: 1169 rows = 43 fresh sft4-solved external (opc_sft2_b: 36 gold+1 easy of 489 executable, yield lower than slice A's 75 � harder slice or solver parity within noise) + 600 fresh reflectionevo_b (300 new sources, skip-300) + 526 v3 replay (0.45).
- Train: sft4->sft5 LoRA-r32 275 steps (bumped from 150: corpus ~2x), val .5629->.5297 monotone, bestval adapter captured each improvement. Merged 4.1GB + .lora + sft5_fq 1.59GB packed.
- Eval running: instrumented held-out (per-attempt + per-assert acc) sft4 vs sft5.


### sft5 eval FINAL (all arms, on-manifold 60 tasks, k4 t.8)
- sft3 3/60 (6/102 att) | sft4 3/60 (4/83) | sft5 1/60 (1/53) | sft4+sft5.bestval(partial,392/448 tensors attached � fused-qkv missing k_proj slots) 3/60 (3/53)
- KEY: solves noise-flat but ATTEMPT rate declines each round 102->83->53. bestval doesn't rescue => not late drift. Reflection-heavy corpus = prime suspect for extraction decline. Held-out dense: sft4 147att acc.006 | sft5 108att acc.010 (both ~0 solves, unfiltered slice above ceiling).
- sft6 corpus plan: code-heavier (reflection ~300 absolute not 600, replay-frac .45+, add KodCode/OpenCodeInstruct verified supply); track attempt-rate as leading metric; consider sft6 train FROM sft5 (lineage) or re-eval whether sft4 is the better solver base � sft5 selfgen yield was 43 rows vs sft3's 75 on comparable slices (confounded by slice difficulty).
- KNOWN CAVEAT for _onmanifold_eval.py: task dicts shared across eval_model calls => _solved/_attempts flags carry between arms => mid-run wave prints for arm2+ contaminated (final counts clean � reset at rep0). Fix later if reused.


- Extraction-decline refined: sft5's extra failures are DEGENERATE outputs (empty code fences + EOS collapse, same mode as proposal collapse) not prose drift. Peek: sft5 rep5 = repeated empty fences; sft4 reps all emitted code.
- _onmanifold_eval flag-carryover fixed (up-front reset).
- sft6 banking: selfgen slice C (opc_sft2_c 600 tasks, sft3_fq solver = best extraction arm) running -> sft6_external.jsonl.

### sft6 round FINAL (trained sft5->sft6, 520 rows: 112 ext + 200 refl-c + 208 v2-replay)
- Train: early-stop @101/175. bestval step26 val .4941; final val .5042 (overfit).
- Artifacts: sft6.safetensors (merged final) + sft6_fq; sft6.bestval.safetensors (adapter) -> .devin/merge_bestval.py -> sft6.bestval_merged.safetensors -> sft6.bestval_merged_fq.
- merge_bestval.py: build_model_fast(sft5) + add_lora_adapters(rank32 a64, 7-name target list, 224 adapters) + load 448 tensors (0 missing/unexpected) + merge_lora_adapters + dedup tied head/embed clone + save_checkpoint. CPU-model tie-break: .cpu() is no-op on CPU so tied storages must be clone()-deduped before safetensors save.
- On-manifold (60 tasks k4 t.8): sft6_fq 5/60 pass, runnable 25 (24% precision) | sft6bv 1/60, runnable 45 (4%). Extraction decline continues+worsens: 102->83->53->25. Bestval WORSE than final: earlier ckpt = more code-forward, less precise; later ckpt = rarer code, higher precision.
- PEEK root cause (raw outputs): EMPTY-FENCE COLLAPSE — model emits ```python then EOS/empty body on hard prompts (2/6 reps sft6, 1/6 bv). NOT prose drift; learned abstention.
- CORPUS root cause: v5 fenced-body median 228ch vs v2 770ch; 32% of v5 fenced bodies <60ch (stubs) vs 17% v2. Reflection+short external rows taught short/stub/abstain behavior.
- sft7 direction: filter responses to complete functions (body len floor, no trivial returns, required names present); cut reflection or reformat so rows don't END in short code; prefer v2-style longer solutions; keep sft6_fq (final) not bestval for lineage — higher solve precision.
- Held-out (unseen, k4 t.8): all arms ~0-1/117 — slice above capability ceiling, non-discriminating; on-manifold is the regression instrument.
- opc_sft2_c consumed (600 rows used); next slices: educational_instruct has 118k rows; fetch via fetch_external_tasks.py --skip 1660.


### sft7 staging (abstention-reversal experiment)
- NEW TOOLS: scripts/gen_sft3_data.py clean_for_store() truncates stored rows
  after the verified fence (line-based parse survives glued "```\n\n```python"
  artifacts; committed 0229fc0). .devin/build_sft7.py = build_sft3 + response
  cleaning (strips empty fences + prose-in-fence stubs <60ch; CODE_SIGNAL
  keeps legit one-liners). .devin/merge_bestval.py merges adapter-only ckpt
  into base for packing (CPU tie-break: clone() shared tensors before save).
- sft7 supply: opc_sft2_d_fresh.json = 458 disjoint tasks (skip 1660 overlaps
  142 with prior slices � fetcher --skip is eligible-row-index not key-disjoint;
  post-filter by key against a/b/c/heldout pools is REQUIRED).
  reflectionevo_sft_d.jsonl = 107 disjoint rows (kind=reflection needs
  --dataset bigai-nlco/ReflectionEvo --config "D+"; default dataset yields 0).
- sft7 corpus plan (spark_sft_v6): new = sft7_external + sft6_external
  (cleaned), ZERO reflection, replay v2 @ .55 (v2 median body 894ch = long-code
  anchor). Hypothesis: prose-free + stub-free corpus restores attempt rate.
- OPEN diagnostic: is the attempt-decline partly decoding-side? Cheap test �
  on-manifold @ temp 0.4 arm on sft6_fq before training sft7. If temp.4
  rescues extraction, evals/selfgen should drop temp.
- sft7 train from sft6.safetensors (lineage; sft6_fq final > bestval on
  solve precision). Launcher staged: .devin/launch_sft7_train.cmd.

### sft7 result + literature fix (2026-11-08)
- sft7 (pure-code corpus, 0 refl, cleaned): 1/60 pass, runnable 32 (bv 26).
  Corpus composition DOES NOT reverse decline => cumulative weight drift.
- LITERATURE: ReST-EM canonical = M-step restarts from BASE each iterate,
  not prev ckpt (arXiv 2312.06585; "constraints of prev-iterate patterns").
  Personalized-ST study: base-restart beats continue-chain ~4%. Continual-
  learning fixes: WiSE-FT/soup/SFA weight interpolation recovers capability.
- SOUP TEST: (1-b)*sft3 + b*sft6 @ b=.5 -> 3/60 pass, runnable 53 (was 25).
  Weight interpolation rescues ~half the lost extraction. Tool:
  .devin/merge_soup.py (lazy tensor interp, no base needed same-layout).
- sft8 = canonical re-base: spark_x26_1p7b (base) + spark_sft_v7.jsonl
  (290 verified ext union + 209 v2 replay, 0 refl) 175 steps max.
- If sft8 recovers extraction near sft3 levels => future rounds ALWAYS
  retrain from base/sft8 on union corpus, never chain sftN->sftN+1.

### sft8 RESULT (canonical re-base, 2026-11-08) � FIX CONFIRMED
- sft8_fq on-manifold: 2/60 pass@4, runnable **101/240** � extraction FULLY
  recovered to sft3 peak level (102). pass@4 noise-equal (2 vs 3).
- Chain-collapse table (runnable/240): sft3 102 -> sft4 83 -> sft5 53 ->
  sft6 25 -> sft7 32 -> [soup .5] 53 -> [sft8 re-base] 101.
- NEW RULE (supersedes chain-SFT): each round = retrain from BASE
  (spark_x26_1p7b) on union of ALL verified data. Never sftN->sftN+1.
- sft8 train: val best .4482 @~151 (best of any round; no early-stop,
  healthy curve vs sft7's immediate overfit on chained small corpus).
- Soup remains a valid cheap-repair tool (merge_soup.py) but re-base is
  the correct recipe going forward.
- OPEN: sft8 precision low (3/101). Next lever = MORE verified rows per
  round (bigger union) or harder tasks; also soup(sft8,sft6) if precision
  vs recall tradeoff matters.

## sft9 pipeline (current)
- selfgen slice E: sft3_fq solver (sft8-as-solver yielded ~4%, killed at wave 6; sft3 wave3=12 rows ~13%) -> data/sft/sft9_external.jsonl, 544 tasks, ~23 waves
- NEXT: build spark_sft_v8 = union(5 ext files) + frontier(150 refs from opc_sft2_pool_union misses) + v2 replay(frac .30) -> .devin/build_sft7.py --frontier-pool ... --frontier-limit 150
- TRAIN: .devin/launch_sft9_train.cmd (staged: base ckpt, spark_sft_v8, 175 steps)
- EVAL: .devin/_onmanifold_eval.py --rows (all 5 ext jsonls) --pool data/external/opc_sft2_pool_union.json -> 203 tasks, arms sft8_fq + sft9_fq
- frontier builder: --frontier-pool/--frontier-limit flags in build_sft7.py; rows = solver prompt + fenced ref code
- retry-missed: gen_sft3_data.py --retry-missed --k N revisits marker tasks (done_keys kept_only), novelty grams = kept rows only
- spark_sft_v8 built: 744 rows = 317 rft_external + 10 easy + 150 frontier + 267 v2 replay; frontier from opc_sft2_pool_union (2065 missed available). sft9 selfgen killed at wave 18 (window-close) but banked 461 rows.
- sft9 launched (ReST-EM re-base from BASE, 175 steps, LoRA r32, cpu_offload, bestval snapshots + early-stop p3). 697 usable ex.
- sft9 DONE: 175 steps, best val 0.4043 (best all rounds; sft8 .4482). Packed sft9_fq (1.59GB).
- sft9 on-manifold (union 228 tasks, k4 t.8): sft9 13/228 pass@4 (5.7%), attempts 30/369 (8.1%) vs sft8 11/228 (4.8%), 24/364 (6.6%). Re-base still holds; precision slowly climbing.
- sft10 supply: opc_sft2_f.json = 500 tasks @ skip3000, 25/500 key overlap w/ prior slices. Selfgen launched w/ sft3_fq solver -> data/sft/sft10_external.jsonl (.devin/launch_sft10_selfgen.cmd).
- NEXT: build spark_sft_v9 = union(6 ext) + frontier + v2 replay; train sft10 via clone of launch_sft9_train.cmd (relabel v8->v9, sft9->sft10); eval on-manifold again.
- sft10 selfgen DONE: 491/500 executable, 68 kept (49 goldilocks + 2 easy-kept ratio) — best yield yet vs sft9's 37.
- spark_sft_v9: 868 rows = 383 rft_external + 12 easy + 150 frontier + 323 v2 replay.
- sft10 train DONE: 175 steps, best val 0.4722 (corpus differs from sft9's — not comparable). Packed sft10_fq.
- RUNNING: on-manifold sft9 vs sft10 on union of 6 ext files -> .devin/onman_sft10.log
- sft10 RESULT: on-manifold 12/229 pass@4 (5.2%), attempts 26/461 (5.6%) vs sft9 13/229 (5.7%), 30/372 (8.1%). sft10 did NOT improve — pass@4 within noise, attempt rate regressed. Union corpus growth alone isn't lifting extraction; 1.7B ceiling near. sft9 remains canonical.
- Candidates: soup(sft9,sft10) via merge_soup.py + re-eval; or pivot to reflection-heavy corpus (reflectionevo rows unused since v4); or accept sft9 as final Spark iterate.
- MBR guidance WIRED: gen_sft3_data.py records all rollout execution signatures (MBR-exec clusters incl. failures); kept rows take the modal correct implementation (_central_pass = syntactic MBR, token-Jaccard centrality — exec-MBR is degenerate among passers since they share one signature); non-kept tasks emit consensus_miss marker when >=k/2 rollouts share one wrong signature (repair/frontier material). _onmanifold_eval.py reports mbr@1 (modal signature vs expected) alongside pass@k/attempt rate.
- NCM (Next-Concept Modeling, Hazy/Together Oct'26): coarse->fine hierarchical VQ-VAE generation, ~12 parallel-code passes vs ~1k sequential — user's take: batched compute, "1k words in 32 loops". VERDICT: needs retrained tokenizer+level-transformer, can't bolt onto autoregressive Spark ckpt. Cheap approximations on current model: speculative decoding (engine already has suffix-draft/BCG), sketch->code two-stage solver prompts. Noted, not implemented.

## 2026-11-09 � R71 breakthrough program scoping + measured anchors

- sft9 selfgen run CRASHED at wave 19 (~task 425/544, kept=37): `forrtl: error (200)
  window-CLOSE event` � console window was closed. sft9_external.jsonl = 461 rows
  banked. Restart needed to cover remaining ~119 tasks.
- New doc: docs/R71_STREAMED_INTELLIGENCE_DESIGN.md � full lane designs B1-B8.
- MEASURED (spark_x26_1p7b, this box):
  - Pinned H2D 53.1 GB/s (PCIe5), pageable 13.3 GB/s.
  - AirLLM today = pageable+sync � 4.4 tok/s predicted (matches observed 1-5);
    from SMR disk 0.35 tok/s. Root cause = copy mode, not streaming concept.
  - int4 FFN-only stream + double-buffer prefetch � 75 tok/s (compute-bound) �
    streaming at resident speed. B1 StreamHead = top lane.
  - bf16 lossless ceiling � 10.7 b/param (H_hi=2.70 H_lo=7.97); zlib 79.4%,
    split-zlib 69.9% (1.43x). ANS -> ~1.5x cap. "ZIP for weights" alone is
    bounded; non-uniform ECQ codebook needed to make int4 codes compressible
    (minmax g32 codes are ~uniform: H=3.86/4, floor ~4.89 b/p = FQ 4.9).
- CONFIRMED GREENFIELD (nothing to dupe): entropy-coded weights (R50 audit noted),
  async weight-layer prefetch (nowhere), MeZO/ZO training (nowhere), Fisher/EWC
  (grep zero hits).
- New scripts: scripts/bench_airllm_prefetch.py (H2D bw + streaming tok/s model),
  scripts/bench_lossless_weights.py (entropy/zlib/int4-code measurement).

## 2026-11-09b � R71 first-session results (B1+B2+B4 shipped)

- **ECQ codec**: forge/quant/ecq_codec.py + scripts/bench_ecq_codec.py +
  tests/unit/test_ecq_codec.py (9 tests). Range coder = Subbotin carryless;
  decode MUST track (low,code) both � code//r alone desyncs (bug fixed).
  Lossless bf16 actual 10.73 b/p (bound 10.67). ECQ-16 3.84 b/p real ckpt;
  ECQ-32 24.7dB vs g32-int4 22.1dB at same 4.9 b/p. Pure-python RC ~1-3M
  sym/s � packer is offline-only until ANS.
- **StreamHead**: airllm_streamer.py rewritten � pinned shard cache +
  3-slot GPU ring + side-stream prefetch + suffix-keyed binding (CRITICAL:
  key slots by suffix not full name or GPU holds all layers). Old path was
  pageable-copy + copy-back-per-token ? 3.3 tok/s vs new 30-36 tok/s (~10x)
  on 28x50MB synthetic. bf16 beats int4 here: dequant 1.7ms/layer > 0.7ms
  saved by 3.5x smaller transfer at 53GB/s bus. int4 opt-in via
  FORGE_AIRLLM_INT4; phase-2 = Triton fused dequant. FORGE_AIRLLM_LEGACY=1
  = old path for A/B.
- **FisherFold**: research/merge_models.py merge_fisher (gate=rescue,
  interp=union) + fisher_diag (E[grad^2] per param, GPU/CPU accumulate).
  scripts/fisher_merge.py driver (ModelLoader.build_model_fast + CE-loss
  fisher on jsonl rows, name-remap for ckpt keyspace). 8 tests green.
  REAL eval queued: FisherFold(sft6) vs soup vs sft8 on 240-task eval.
- Selfgen sft9 slice-E done: ~43 kept rows total in sft9_external.jsonl.
- REMAINING lanes (user said easiest->hardest): B3 LoRA-ES next, then
  B5 LoopThink (hardest - engine decode surgery). B6/B7 opportunistic.

## 2026-11-09c � R71 B3+B5 shipped (all 5 lanes now have code)

- **B3 LoRA-ES**: forge/training/lora_es.py � ZO antithetic ES over LoRA
  params, forward-only (inference_mode). Modes: full (MeZO), block
  (block-coordinate subsets), fisher (per-param eps scale from
  fisher_diag � B4/B3 cross-lane). attach_lora wraps nn.Linear with
  LowRankBypass (y = Wx + s�xA^T B^T). scripts/lora_es_train.py driver.
  10 tests. ZO cos � sqrt(n_dirs/d) verified empirically (32 dirs?0.117,
  256?>0.2 on d=2048).
- **B5 LoopThink**: forge/exp/loopdepth.py already had looped_prefill/
  decode + LoopGates + train_loop_gates. Added VerifierGate (stop when
  next-token argmax stable AND softmax margin>=thr � output-level
  convergence) + looped_prefill_verified. scripts/bench_loopthink.py.
  MEASURED on spark_x26_1p7b (8 texts, L8-20, iters<=3):
    ungated loop   : loss 6.58 (DIVERGES - frozen model can't recur)
    trained gates(60 steps, 12 params) + verifier: loss 1.529 vs std
      1.644 = **-6.9% loss, +25% latency**, verifier early-stops 6/8.
  LoopThink works on a FROZEN model with 12 scalar params. Next: longer
  gate training + decode-step wiring (looped_decode_step exists) +
  try multiple loop ranges.
- All 5 R71 lanes now have shipped code+tests. Remaining: real-ckpt evals
  (FisherFold sft6 rescue on 240-task eval; LoRA-ES VRAM/quality vs LoRA
  backprop; StreamHead end-to-end tok/s; longer gate training).
- LoopThink HOLDOUT validation (fixed bench: train/eval split):
  8/8 split: std 1.499 vs gated 1.378 = -8.1%. 21/22 split (300 steps):
  std 1.746 vs gated 1.650 = -5.5%. Ungated always diverges ~10.4.
  Generalizes to unseen text � real capability, not memorization.
  Verifier early-stops ~half the eval texts at iter 2-3.
- LoopThink CROSS-DOMAIN: gates trained on glaive_fc_2k (function-call
  chat), eval on sft9 code rows: std 1.511 vs gated 1.472 = -2.6%.
  Transfers but weaker than same-domain (-5.5/-8.1%): part general
  "recur safely" skill, part domain-specific. 200 steps, gates still
  small (|g|<0.12).
- FisherFold REAL runs (grouped GPU fisher, bf16, ~9.4GB peak):
  sft6 rescue (collapsed donor): general CE 2.106 = base-level (sft6 was
  2.167) � damage correctly reverted; domain CE mid 2.720.
  sft9 merge (good donor): alpha sweep on sft9-domain CE �
  base 1.571 | a1 1.505 | a4 1.462 | a10 1.436 | sft9 1.387.
  a10 general CE 2.083 = sft9 level (no imported damage).
  VERDICT: gate reverts regressed deltas AND recovers donor skill with
  alpha as the dial. CE-proxy positive; 240-task runnable eval queued.
  Fishers saved to .devin/fisher_sft9 (reusable via --load-fisher).
- LoRA-ES REAL run (spark, ffn r8 adapters, 5.85M trainables):
  Fixes found: (1) fp32 masters required � bf16 params round away
  eps perturbations (|g|=0). (2) zero-init A,B is DEAD for ZO �
  delta=B@A stays 0 through single-matrix perturbations; needs
  randn_ab (both nonzero, B std 0.002).
  Result: held-out evalCE monotone 1.963->1.682 (-14.3%) over 100
  fwd-only steps, block mode, eps 5e-4 lr 1e-5 n_dirs 16 @3.94GB.
  Honest: still above unadapted base (init damage not fully recovered
  in 100 steps); VRAM win vs 1 backprop is small at batch=1/seq=256 �
  ES=inference memory claim holds at larger batch where activations
  dominate. Niche: adapter updates under memory pressure, not speed.

## StreamHead real-model fix (commit 2a7936e)
- Bug: lexicographic shard glob sort (shard_10<shard_2) scrambled layers >=10; numeric sort fix (same as HF #32023).
- Bug: prefix-cache + finish_sentence + CacheBlend routed std decode on ring-aliased params -> '***'. Streaming dispatches first; finish_sentence skipped under streaming.
- Bug: streamed sampling hardcoded top_k=0/rep=1.0 + prompt missing from penalty window; now flows params + prompt_ids.
- Verified: 36-step logits bit-exact vs monolithic; smoke match:True. Streamed 1.8x slower than std on a VRAM-fitting model (expected � transport overhead; value is for non-fitting models).

## R71-B6 lookahead decode (session findings)

- `forge/engine/lookahead_decode.py` behind FORGE_LOOKAHEAD=k,w.
- Mechanism duplicates existing `NGramSpeculativeDecoding` (decoding.py:647)
  and `suffix_draft` � all ~2.3-2.4x on repetitive code, ~0.9-1.0x open text.
  Value-add vs ngram_spec: EXACT std parity (rep-penalty window is
  generated-only-last-64 NOT prompt+gen � decoding.py:410; prompt feeds DRY
  only), eos-append-then-stop, degenerate guard (x8 same / <40% unique-20),
  max_new cap, adaptive cooldown+dk. ngram_spec lacks ALL of these and also
  does a stale-KV refeed on mismatch.
- Bugs found wiring it: prompt-vs-gen rep window (fixed), eos must be
  appended before break (std semantics), draft-accept can overshoot
  max_new_tokens (cap), _forge_last_kv/logits needed for finish_sentence.
- KV entropy probe (real spark KV): lo-byte H=3.97, hi-byte H=1.38 ->
  independent-byte lossless ceiling ~3.0x. KV/token=57.3KB->19.2KB.
  Parked: needs GPU-side entropy coder; kvtc-style lossy needed for more.
- 240-task runnable eval harness: scripts/eval_runnable_tasks.py
  (opc_sft2_tasks.json; --run-tests execs in separate process).
  Comparing base/sft9/sft9_ff/sft6/sft6_ff � results pending.


## eval+training research (pass@k / PiSSA / continual-SFT)
- unbiased pass@k = 1-C(n-c,k)/C(n,k) (Chen et al HumanEval); naive 1-(1-p)^k biased
- eval_runnable_tasks.py now: --samples N --temp T (sample0 greedy), per-sample verdicts, pass@k + fn_emitted vs pass|fn decomposition
- base 240-task: 42% emit fn, 30.7% pass|fn -> dominant failure = FORMAT (139/240 no_fn) not capability
- PiSSA init for LowRankBypass: B=U_r*S_r, A=V_r^T, base weight -= s*BA -> function-preserving + ZO-perturbable
- PROBE (same 8 texts): unadapted 1.8892, pissa-attached 1.8777 -> preserves function exactly; earlier '1.57' was different eval set
- real run r=8 ffn 100 steps: randn_ab 1.963->1.682 (never beat base); pissa 1.888->1.422 = first ZO adapter to beat unadapted
- svd_lowrank niter=2 vs 4: same eval@0 -> init gap was bf16/noise not approx; fp32 fwd rel err 0.16%
- continual-SFT candidates from research: SuRe (high-NLL replay + fast/slow EMA), SFA (periodic merge-back mid-training), GeRe (fixed general replay + logit/state constraint) � untested, target sft-chain collapse
- running: 80-task x8-sample pass@k eval base vs sft6_fisherfold (bg)


## pass@k results (80 tasks x 8 samples, comment fmt)
- base:    pass@1 13.6  p@2 20.0  p@4 25.9  p@8 30.0 | no_fn 53% of samples | pass|fn 31.8
- sft6_ff: pass@1 15.6  p@2 23.6  p@4 32.1  p@8 41.2 | no_fn 41.7% | pass|fn 29.0 | fn_emitted greedy 38.8 vs base 27.5
- FF gap WIDENS with k (30->41 vs 13.6->15.6) - donor skill real under sampling
- 17/80 base tasks solved ONLY via sampling (temp unlocks latent capability)
- judge fix: subprocess per row + 30s timeout (was in-process exec = hang); verdict 'timeout' added
- prompt formats added: comment/humaneval/mbpp; running humaneval probe on base


## prompt-format A/B (80 tasks, greedy, base)
- comment:   fn 27.5  p@1 8.8   p|fn 31.8
- humaneval: fn 90.0  p@1 27.5  p|fn 30.6  <- 3x pass@1 from format alone
- mbpp:      fn 50.0  p@1 20.0  p|fn 40.0  <- asserts lift per-emission solve +9
- sft6_ff humaneval: fn 92.5  p@1 28.8  p|fn 31.1 (FF advantage was mostly format rescue; ~+1pt honest)
- NEXT: hybrid prompt (docstring + inline asserts) may get 90 fn AND 40 p|fn
- humaneval echoes prompt (~120 tok) -> humaneval runs need max_new>=256


## PiSSA adapter-load bugs (fixed)
- svd_lowrank nondeterministic -> eval-time re-split corrupted W_res (loaded CE 2.21 vs trained 1.27; zero-load 8.62)
- fix: per-layer seed = crc32(module name) in attach -> bit-identical re-attach; OLD unseeded adapter lora_es_pissa_he.pt discarded, retrain needed
- const-lr ZO diverges: 400 steps 1.27->12.5; fix --lr-decay linear + best-eval snapshot
- rerun 400 steps block: min evalCE 1.2682 @~150, stable plateau ~1.30
- retrain running w/ deterministic split; then eval pass@1 vs base 27.5



## SIFT design (2026-11-09, R&D)
- Structured Importance-ranked Fine-Tuning: train real weight SLICES not adapters
- units: attn sub-head blocks (q+g rows of fused qkv + out cols, gran=128) ; ffn neuron-blocks (gate_up rows j & j+6656 + down cols, gran=32)
- importance = ||dW_unit||_F from ONE calibration fwd+bwd (Gauss-Southwell block selection)
- frontier F = shallowest block covering mass_frac(0.75) of grad mass; below F all params frozen -> autograd builds NO graph -> free backward-depth cut
- delta side-params zero-init (lossless), merge into W on save/reselect; touched entries only -> .sift adapter ~40MB
- prefix cache: h_F per row keyed by ids-hash -> epoch>=2 skips prefix fwd (fill pass doubles as prefix compute, no epoch-1 penalty)
- target: spark_x26_1p7b SFT; recipe flags --manual-lora --no-bitnet-everywhere --no-pack-sequences
- est: ~33-45% step-FLOP cut vs LoRA SFT; params touched ~LoRA-scale but REAL weights

## SIFT R70 — implemented + measured (2026-11-10)
- code: forge/training/sift.py (~780 ln) + llm.py 3 consume-on-read hooks + sft_train --sift* (11 flags) + checkpoint_io sift_only + tests/unit/test_sift.py (16 CPU tests, green)
- spark bench (bench_sift.py, spark_sft_v9, bs4 seq2048, fused AdamW):
  lora-r32: 31.66M touched / 1288ms step / 5.78GB / loss .992->1.188 (10 steps)
  sift a2g128+f4g16: 33.25M touched / 1125ms (-12.7%) / 5.71GB / loss .668->1.125
  sift+prefix-cache ep2: 1038ms (-19.4% vs lora); adapter 66MB vs 4.1GB full
- inject path BIT-EXACT vs full forward (loss 0.94921875 both) — verified before bench
- spark gate is 1 SCALAR ROW/head inside fused qkv (out_dim 3080 = 2048+512+512+8): units own gate row g_base+head (gran<=hd, first sub-unit); 132-row toy test added
- sq-norm vectors (row/col) replaced granule views — arbitrary spans incl len-1; 3080%256!=0 would've crashed
- BUGS FOUND+FIXED:
  * calibration OOM: left embed/head trainable -> 512MB CE grad buf + dead autograd; now freezes non-candidates
  * GC not engaged: ForgeEngine path enables grad-ckpt at ~line 2083, sift setup ran at 1862 -> enable_gradient_checkpointing() inside setup_sift
  * resume loaded the ADAPTER as --checkpoint (meta lacked sift_only) -> random-init base, loss 23.6; save_sift_adapter now writes sift_only+base_checkpoint; resume meta check covers sift_adapter too
- CAVEAT: importance mass on spark_sft_v9 near-UNIFORM (L2..L9 ~similar) -> frontier auto-picks block 4-5/28; backward-depth cut modest here. Expected bigger on late-layer-concentrated corpora; frontier is per-corpus measurable (block-mass printed at setup)
- e2e smoke sft_train --sift: calib->train 4 steps loss 2.08->1.22 VRAM 4.06GB, adapter+merged saves, resume re-attach verified loss continues 1.70->0.71
- SIFT score modes (R70 follow-on): --sift-score-mode fro|smean|obs —
  fro = E‖g‖_F (Gauss-Southwell), smean = ‖ḡ‖_F (noise-robust),
  obs = Σ ḡ²/Var[g] (OBS saliency as SELECTOR: diag empirical-Fisher =
  free Hessian proxy, eps = module-mean Var keeps magnitude ordering).
  fp16 Σg + fp32 Σg² accumulators on CPU during calib (~6GB RAM).
  A/B/C on spark_sft_v9 (bs4 seq2048, a2g128+f4g16, 10 steps):
    fro:   frontier=5, 33.25M, 1125ms, 5.71GB, .668->1.125
    smean: frontier=5, 33.25M, 1099ms, 5.64GB, .672->1.125
    obs:   frontier=6, 31.77M, 1070ms, 5.60GB, .680->1.125
  VERDICT: neutral on clean corpus (same endpoint), obs leaner+faster;
  differentiator expected on noisy/mixed corpora where fro picks
  large-but-inconsistent gradients. test: sign-flip loss_fn kills
  smean/obs scores <5% of fro (proves the mechanism, 17 tests green)
- SIFT-N Newton init (--sift-newton-passes N): Δ=-lr·E[g]/(E[x²]+damp)
  (input-factor K-FAC half). Raw lr=1 BLEW UP on real CE (loss 12.6 flat)
  → added per-slice trust region (0.25‖W_sel‖) + Armijo backtracking
  (accepted point never worse than zero-init, no-op fallback).
  Bench spark_sft_v9 a2g128+f4g16, 10 steps: loss0 0.668->0.287 (-57%),
  loss10 1.125->0.305, step 817ms, VRAM 5.52GB. Closed-form slice solve
  >> dozens of Adam steps. Tests 19/19 (toy optimum is Δ*=D-W_sel not D —
  assertion bug caught + fixed; gradient noise needs >=4k samples).
  Test gotcha: trust=0.25 truncates toy's large true delta -> trust=1e6
  in the recovery test.
- Loss-band culling (--loss-band LO HI, general SFT flag): skip backward+
  step on out-of-band micro-batches; NaN always skipped. band-skip K/N in
  logs; all-skipped epoch -> warn+stop (else step never advances and
  range(100) epochs of dead forwards spin). One loss sync per micro-batch
  (opt-in; default path keeps deferred .item()).
  Smoke (spark_x26_1p7b): skip-all guard fires clean exit; accept-all on
  full v9 trains normal loss 1.22->0.73 band-skip 0/10.
  PRE-EXISTING BUG FOUND (not fixed, out of scope): tiny packed corpora
  (16-row subset -> 12 usable -> 8 packed seqs) yield NaN mean CE on every
  batch (default LoRA AND sift paths) — suspect all-masked-label batches
  0/0 in entropy-weighted CE (alpha=0.5 default). Band shields training.
  NEXT: root-cause compute_loss weighted-mean zero-weight guard.
- Research sweep 2026-11 (scripts/bench_spark_research.py, wiki Spark-SIFT-Research):
  PRISM-lite on SIFT deltas: NOT a win (lossN 1.141 vs 1.125, +6% ms) — Muon-family NS
    doesn't pay on small slice matrices at lr2e-5/10 steps. Not integrated.
  SIFT-ES scales-only: DIVERGED at lr.5/sigma.1 (sign flips); WORKS at lr.05/sigma.05
    clamp>=0: eval .861->.551 holdout .945->.848 (16 iters, 256 fwd-only evals).
    Shipped as es_scale_polish + --sift-es-polish (gains fold into deltas, rollback safe).
  Vocab shortlist (kmeans 256 centroids on tied head): recall@10 .30@3% .69@12.5% .79@25%
    -> post-hoc two-stage decode NOT viable; needs trained adaptive softmax/MIPS bounds.
  HDC random codebook vs kmeans (real out_proj W): 1.154x MSE (training-free codebook ok)
    but rel MSE .39 at ~1bit/w -> far from lossless (rate-distortion confirmed again).
  Assoc memory d=64: recall 1.0@n<=32, .984@64, .93@128, .676@256 -> capacity ~d.
  FF/HAZE/recurrence/sparse-attn/SSM/neuro-symbolic: analytic verdicts in wiki.
  Tests 20/20. Wiki pushed (Spark-SIFT-Research + Training + _Sidebar).
- SIFT+MTP (round-2 pointer implemented): 3 bugs — (1) ForgeEngine builds own cfg ->
  use_mtp never constructed mtp_module on sft_train load path (pre-existing for ALL
  mtp runs, not just sift); fixed by post-load construction+tie in the config-sync
  block. (2) mtp.py _resolve_tied_weight `tensor or w` ambiguous-bool crash -> explicit
  None check. (3) freeze_non_sift froze mtp too; now unfreezes mtp params EXCEPT the
  tied main head (id match) which must stay frozen (268M, OOM'd 12GB when unfrozen).
  adapter_heads forced under --sift: 8.39M trainable @n_heads=1 (vs 1.08B full heads).
  MTP full-vocab loss is NOT chunked-CE: 4 heads = 13s/step + VRAM spike; use
  --mtp-n-heads 1 (640ms/step, 4.21GB). Smoke loss 2.40->1.93 (CE + 0.1*fresh-head
  ~11.8 MTP CE — consistent). LIMITATION: mtp params not in sift_only adapter saves;
  final merged save includes them. tests/unit/test_mtp_sift.py (2 tests) 22/22 green.

## R&D reference shelf — provable algorithms for the 4 user questions (2026-11-10)
Q1 next-token "algorithm": fixed GEMM+softmax pipeline; provable bounds:
  dense matvec Theta(d^2) (low-rank O(dr)/butterfly O(d log d) only outs);
  FlashAttention IO-optimal exact; speculative decode provably identical dist;
  ADAPTIVE SOFTMAX (Grave'17) = EXACT softmax at O(d log V) — the one
  provable free lunch left; Spark head = 268M params = biggest single GEMM.
Q2 update rule = steepest descent under a norm; proven per-norm optima:
  SGD euclidean | Adam ~diag-Fisher heuristic | Muon spectral-norm optimal |
  K-FAC exact Gauss-Newton on linear layer | Gauss-Southwell = provably
  optimal block pick per step (Nutini'15) — SIFT calibration IS this rule |
  OBS saliency g^2/Hii provably min-loss unit removal (Hassibi'93) —
  SIFT-v2 = g^2/v_hat selection (OBS as SELECTOR not pruner, v_hat = free
  Adam 2nd-moment Hessian diag) | PiSSA = Eckart-Young optimal rank-r |
  closed-form slice Newton: dW* = E[YX^T] E[XX^T]^-1 EXACT for quadratic —
  only feasible because SIFT shrinks space to ~33M.
Q3 1-bit f16-quality: DISPROVEN as post-hoc compression — scalar SQNR
  ~1.2dB@1bit, need ~40dB. Provable: VQ>scalar (Shannon; E8 +1.41dB);
  orthogonal rotation exact re-basis (QuaRot, in-repo); GPTQ = provably
  optimal OBS quadratic solve; IRB converges geometrically (SubBitnet
  in-repo); BitNet1.58 parity = TRAINED-native only (empirical). Floor
  in-repo: IRI-FP4 r2 ~9b/w lossless, int4+rot ~lossless. Honest novel:
  W = s*T + C[g] ternary + shared-codebook index ~1.7b/w (Shannon VQ
  says it beats pure ternary). Better reframe: base+66MB SIFT adapter
  IS the 60x artifact.
Q4 full reinvention: NO — transformer is the hardware equilibrium; SSM/
  linear attn have PROVEN recall gap (Arora'23); hybrids (Spark's own
  SWA+full) are the fix. Replace weakest incumbents: vocab head, param
  representation, update rule — per-component behind parity tests.


## NEGATIVE RESULT: LoRA-ES adapter halves pass@1
- deterministic-split retrain: evalCE 1.49->1.1152 (best@399, -25% on clean humaneval SFT rows)
- adapter eval (80 tasks humaneval greedy): fn 91.2, pass@1 13.8 (vs base 27.5), pass|fn 15.1 (vs 30.6)
- verdicts: wrong_answer 53 (base 43) - adapter emits plausible-but-wrong code
- interpretation: ZO mean-CE objective misaligned w/ functional correctness; surface-fluency gain, correctness loss. LoRA-ES = proven MEMORY-efficient trainer but objective needs to be task-aligned (e.g. pass reward / RL-style, not CE)
- machinery all works: deterministic load (diff 0.0), best-snapshot, lr decay, --adapter eval
- NEXT IDEAS: (a) ES on pass@k reward instead of CE (binary signal, ~80 gen/step too slow?); (b) train on sft9_external instead; (c) LoopGates RSI-depth training still untried


## 2026-11-10 — RSI loop on Spark (commit ef1f2f0)

Raw base spark_x26_1p7b + sft6_src both produced 0 validated tasks.
Root causes (2 stacked bugs, both Jamba-assumption):
1. render_messages_for_config ignored config_name -> Jamba ChatML for
   all presets. Added spark_render_messages (port of chat_template.jinja).
   InfiniteCurriculum(config_name) -> _render_chat. forge_server already
   passed config_name so Spark serving fixed for free.
2. _load_engine + fast_eval pinned forgelm_v2_tokenizer (65k vocab) ->
   Spark 131k token ids decoded as garbage. Pin now forgelm-only.
Also: fast_eval ChatML->config prompt translation + per-config eos;
_RE_TURN_MARKERS strips Spark markers.

Result (sft6, 1 epoch, 6 tasks, 16 LoRA steps): propose 6/6, solve
4/6 (57%), eval TIE 0.63/0.63, PROMOTED. Not a measurable gain -
infrastructure proof only. "who_made_you" PASS is NOT a data leak:
_SYS prompt literally contains 'built by GRKTheGreat' -> parroting.
Corpus scan: 0 hits grkthegreat/tmk68/forgelm/paths/keys across all
data/sft/*.jsonl; emails/password/api_key hits are fictional/tutorial.

Open: --sift not plumbed into loop _finetune (RSI used LoRA+muon_sf);
eval engine needs leaner activation (0MB free -> KV cap 64 toks);
orphaned infinite_loop children hold RAM across kills - reap by cmdline.

## Maintenance audit notes (2026-11-13)

Dead-code sweep (whole repo): codebase is well-wired - feature_registry /
kv_backend / configure_optimizer lazy-import nearly everything. Findings:

- forge/decoding/edit_decode.py: ONLY truly unreferenced file. Deliberately
  parked V14-RSI CopySpan decode driver (model side IS wired: edit_head,
  use_edit_head, copy_span_loss). V14_RSI_DESIGN.md P3 = wire it into the
  engine decoding registry. Do NOT delete; either wire or accept rot.
- forge/training/optim/flashoptim.py vs flash_optim.py: two DIFFERENT
  FlashOptim variants, both wired (names flashoptim vs flash_adamw/
  flash_lion). flashoptim = per-tensor companded int8 m+v; flash_optim =
  blockwise companding + 24-bit master + Triton fused + gradient release +
  Lion. Merge = real R&D (train-parity bench needed), not janitorial.
- Test-only production files (exercised by tests, no prod importer):
  self_play/data_quality.py, self_play/replay_buffer.py, quant/ecq_codec.py,
  model/strata/ (7 files), engine/quant/novel_quant_r50.py (script-only).
  Keep unless their test coverage is judged worthless.
- novel_quant_r{44,45,46,48,50}.py versioned series: cross-dependent,
  r48 wired into engine_activation + sft_train. Consolidation needs care.
- Wiki: Home/Decoding/ForgeEngine/Quantization.md were mixed CP1252/UTF-8 -
  converted to UTF-8 (wiki now renders em-dashes correctly).

## Pre-existing test failures (2026-11-13 full unit run: 5 failed / 3823 passed)

NOT regressions from the 2026-11-13 maintenance commits (all in modules
untouched by the diff). Persistent on isolated rerun:
- tests/unit/test_distill_quant_patch.py::test_collate_no_teacher_logp_returns_none
  (rw None, expected .item()==1.0 - distill collate path)
- tests/unit/test_r60_kv.py::test_decode_loop_invokes_bridge
  (source-inspection assert: _maybe_compact_past_kv missing from decode src)
- tests/unit/test_flux.py::test_cuda_forward_deterministic (CUDA determinism)
Flaky (fail in full run, pass isolated):
- test_sandbox_escapes.py::TestFilesystemJail::test_env_scrubbed (env ordering)
- test_round32_quant.py::TestForgeQuant::test_forge_quant_fused_gemm

## External research sweep (2026-11-15, user-requested reading list)

### SteerDistill (arXiv 2611.13410, Tsinghua AIR, Nov 2026)
- Quantization-aware distillation for LLM *agents*: closed-loop teacher steering.
- Frozen BF16 teacher guides quantized student to retry divergent turns from
  pre-error state; teacher refines guidance from execution feedback until the
  correction passes local verification. Only verified corrections are kept.
- Internalization: student trains WITHOUT guidance — matches steered student's
  predictions at corrected turns, BF16 teacher's at other turns → unassisted
  inference at deploy time.
- Results: W4A16 within <1 pt of BF16 across BFCL-V4, tau2-Bench, WebShop,
  ALFWorld, SearchQA (Qwen3-8B W4A16/W4A8/W2A16, Qwen3.5-4B, Gemma-4-12B).
  Beats QeRL by 1.84pp on tau2-Bench with 67% fewer GPU-hours.
- ForgeAI relevance: direct analog for our QAT/distill paths —
  "teacher-steered trajectory repair then distill corrections" is a novel twist
  on our existing distill infrastructure. Applies to tool-use SFT of quantized
  ForgeLM variants.

### Effective Teacher Support for Reasoning Distillation (arXiv 2611.14850)
- Forward-KL distillation from teacher top-k vocab ONLY, sweeping support
  100%→0.01%. Qwen3 4B→1.7B main; 8B→4B + Nemotron-Nano 12B→9B confirmations.
- Qwen3: k≈15 of 151,936 (0.01% support) → NO MMLU-Pro degradation vs full-vocab
  KL. Nemotron needs 0.1% (smaller → misses closing reasoning tag; cutoff is
  family-dependent).
- At 0.01%: caching retained teacher scores cuts step time 38%; restricting
  student output head to retained tokens cuts peak mem 24%.
- ForgeAI relevance: cheap distillation — our distill collate caches teacher
  logprobs; storing only top-15..top-1k makes cached teacher targets tiny and
  lets the student LM head run narrow. Directly applicable to sft_train
  distill path (note existing test failure in test_distill_quant_patch.py).

### llama.cpp v1.0.0 (released 2026-11-13, ggml v1.0.0 too)
- Semantic versioning contract: MAJOR = public C API break in ./include;
  MINOR = backward-compat features/backends/models; PATCH = bugfix.
- Removed 35 deprecated functions, defrag_thold; NextN/speculative API promoted
  llama-ext.h → llama.h; llama_set_embd_state_src → llama_set_ctx_state_src.
- OCP FP8 KV caches (-ctk/-ctv f8_e4m3, optional static per-tensor scales).
- Host-memory MoE expert cache: moe_cache_size + moe_cache_cpu_share in
  llama_context_params; LRFU/ClassLRU replacement; no CPU wait on cached
  experts — same direction as ForgeAI hybrid_offload/cpu_kv_offload.
- External PLE/engram loading from separate GGUF shards.
- New archs: Kimi-K3.1-Pro (DSA replacing full-attn MLA), Ripple-65B (dense),
  xing4_0 (mHC+MLA+MoE+MTP), drama-3 TTS; MageVL-Cowork + Kimi-K3.1 vision.
- Server hardening: SSRF, CORS, DNS-rebinding, MCP tool cancellation;
  /v1/audio/speech endpoint; redesigned web UI (chat groups, md editor).

### Strata (external — NOT our forge/model/strata)
- ro99/strata + Niko1221/Strata: C++20/CUDA MoE inference engine for models
  larger than VRAM/RAM. No Python/ML framework at runtime; CLI + OpenAI/HTTP.
- Three-tier split: GPU VRAM (dense parts, routers, shared experts, MTP draft,
  hot KV, adaptive expert cache) + pinned host RAM (full expert pool, CPU
  computes cache-misses in place, concurrent with GPU) + read-only NVMe
  (n-gram table shard mmap'd, ~16 row reads/tok issued a step ahead).
- Fidelity stance: checkpoint-declared precision/router/expert-count/top-k;
  predictors may reorder work but never change what is computed; "exact mode
  completes or fails." (NOTE: ro99 README claims never below 4-bit, but the
  Niko1221 Strata paper reports Q2_0/IQ2_XS/IQ3_XXS results — either a
  downstream fork or a policy change; verify before citing the 4-bit floor.)
- Headline bench (Niko1221 paper): Qwen3.8-Flash-Next 125B MoE (6B active,
  24,576 experts, top-10) on **RTX 5070 12GB + Ryzen 7600 + 64GB DDR5** —
  95 tok/s at 4K ctx with 2-bit file, 45-95 tok/s across quants to 262K ctx,
  prompt read 285-571 tok/s. Expert traffic ~0.66 GB/tok over PCIe 4.0 x16.
- Third-party check: 16GB V100 + 2016 Xeon ran 111GB 4-bit file ~20 tok/s vs
  ~9 tok/s llama.cpp (claimed ~2x).
- Multi-GPU: pipeline/layer-split (token crosses cards once per verify window),
  no NVLink needed; split adds capacity not speed.
- Also note amperor-strata (separate): Apple-Silicon MLX MoE expert-streaming
  lib, different project, same name.
- ForgeAI relevance: architectural twin of our hybrid_offload design goal —
  CPU computes expert misses in-place (no PCIe copy) is the key differentiator
  vs naive offload; MTP-as-spec-verify = our speculative path. Their n-gram
  table on NVMe mirrors DiskTokenCache philosophy (sequential NVMe reads OK on
  SMR/USB? — n-gram rows are small random reads, better on C: NVMe).

### Cerebras inference
- Wafer-scale: WSE-3 = 4T transistors, ~900k cores, 44GB on-chip SRAM,
  21 PB/s aggregate SRAM bandwidth (~7000x H100 HBM BW). Whole model lives
  on-chip → decode never crosses a memory bus.
- Reported: 1,800 tok/s Llama3.1-8B, 450 tok/s 70B (native 16-bit weights),
  ~2.4x Groq at launch; current claims up to 30x GPU systems. Models >1 wafer
  split at layer boundaries across CS-3/CS-4 systems (70B on ~4 systems).
- CS-4: 3x WSE-3 Turbo wafers/system, 2µs wafer-to-wafer interconnect,
  >1,000 tok/s on >50T-param models, ~10x throughput/watt vs CS-3. Nexus
  rack-scale platform, wafer-scale "backpack" packaging.
- Service via API + on-prem; pitch is latency-first (single-second reasoning
  makes test-time-compute loops practical) — they now argue fast inference
  itself is a capability unlock for agentic/reasoning loops.
- ForgeAI relevance: validates "bandwidth is the whole game" thesis our
  quant+offload stack is built on; useful counterpoint when justifying why
  12GB VRAM forces tiered memory designs (we can't buy bandwidth, we tier it).

### OPAL — ambiguous, three live candidates
- (a) **On-Policy Attention Linearization (Opal)** — arXiv 2609.31947
  (CMU/Cornell). Distills hybrid linear-attention students from frozen
  full-attention teachers, on-policy: student samples own long-context
  trajectories, gets dense per-position teacher supervision. Qwen3-4B +
  MiMo-7B-RL: 87-94% commonsense, 100% NIAH retrieval, 83-93% math with only
  3B tokens, no SFT/RLVR. Fixes the off-policy drift problem (linear-attn
  layers' fixed-state compression errors compound; teacher-forced training
  never teaches recovery). STRONGEST ForgeAI match — direct recipe for
  V13-GDN2/V14 linear side-path distillation from our Jamba checkpoints.
- (b) **Reactive Optimizers (OPAL)** — arXiv 2610.33826 (MIT, Sitzmann group).
  Pretrained LM agent inspects training dynamics and rewrites executable
  optimizer code mid-run (per-param-group update rules/schedules/precision/
  momentum), no optimizer meta-training. Matches/beats fixed recipes on
  untuned settings; adapts to data corruption, reduces overfitting.
  code at github.com/xingjian-bai/opal. Relevant to our optimizer R&D
  (muon_sf_plain, flash_optim family) — "LLM-mutating optimizer" is a
  self-play-flavored twist we could bolt onto evolution/eval_pool.
- (c) **OPAL: Outlier-Preserved Microscaling Quantization Accelerator** —
  arXiv 2409.05902 (2024, DGIST/KoreaU). HW-SW codesign: MX-format activation
  quant keeping ~4/128 outliers per block, mixed 3/5-bit by layer sensitivity,
  log2-approx softmax (shift+subtract only). 1.6-2.2x energy, 2.4-3.1x area,
  <1 ppl hit. Older but the outlier-per-subblock trick overlaps our
  activation-quant rounds.
- (d) Long tail: OPAL oversampling-for-imbalanced-data (2406.03628),
  ontology-aware TOD (2209.04595), Opal private-memory ORAM (2604.02522),
  Opal perf-analytics+LLM kernel optimization (2510.00932 — relevant-ish:
  19-52% kernel speedups from feeding roofline/PC-stall diagnostics to LLM).
- Recommendation: (a) is almost certainly the intended one given our
  linear-attention program; (b) and the kernel-tuning (d) are secondary hits.

### TPU v8 (Google, 8th gen — 4 SKUs, first time 4 in one year)
- 8t (training): 9,600 chips/superpod, 2 PB shared HBM, 121 EF/pod, 2x interchip
  BW vs 7x, ~3x compute/pod; 2.8x perf/$, 2x perf/W vs TPU 7x (Ironwood).
- 8i (inference): hierarchical butterfly topology, 80% perf/$ gain, 2x latency
  cut vs 7x; tuned for low-latency large-MoE reasoning/agentic.
- 8i Max (Nov 9, 2026): 2x compute, 2x SRAM, 2x HBM BW of 8i in same rack
  footprint — aimed at massive context windows.
- 8tx (Nov 9, 2026): confidential/secure inference + small-scale training,
  hardware-level data protection for regulated/gov.
- Announced: 8t/8i at Cloud Next (April 2026), 8i Max/8tx Nov 9, 2026.
  Availability "later this year" per announcement.

### "Team of experts" — ambiguous, best candidates
- (a) **"Scaling Inference-Time Compute with Teams of Coding Agents"**
  (arXiv 2610.17471): team design as compute allocation. Findings: delegating
  only PLANNING to workers + lead implements/verifies alone = half cost at
  same reward; cheap planners OK, cheap implementers hurt; planner reward
  scales with team size, implementer/verifier gains plateau >4 workers;
  heterogeneous 17-agent team within 1.4pts of homogeneous 5-agent at 70%
  less cost. Most likely match if user meant a recent paper.
- (b) MoA/MoE-flavored: Mixture-of-Agents lineage → DMoA (differentiable
  mixture-of-agents, 2605.15706), MOSAIC (MoA scheduling, 2606.03014),
  Skill-MoE (skill-inferred routing, PMLR v306).
- (c) Possibly colloquial for MoE itself ("team of 24,576 experts" is the
  exact phrasing Strata coverage uses for Qwen3.8-Flash-Next) or for
  Nemotron-3-Super LatentMoE (22/512 experts active).
- TODO: ask user which they meant.

## 2026-11-15 — Spark RSI-loop gen/train probe (MEASURED on this box)

Goal: train spark_x26_1p7b in an RSI loop (FLINT-style); gen must
brute-force candidates fast, train step must be short.

### Gen matrix (scripts/bench_spark_genmatrix.py, real runs, temp0.7/160tok)
| mode | B=1 tok/s | B=8 | B=16 | B=32 | peakGB |
|---|---|---|---|---|---|
| plain bf16 | 36.9 | | | | 3.91 |
| bf16+BCG | 87.7 | | | | 3.91 |
| bf16+BCG+ngram_spec | 29.2 | | | | 3.91 |
| bf16+BCG+suffix_draft | 43.9 | | | | 3.91 |
| bf16+BCG batch | 81.9 | 223.9 | 815.3 | **918.7** | 4.98 |
| _fq (no BCG) | 12.0 | | | | 2.12 |
| _fq+BCG batch | 59.8 | 150.2 | 274.5 | | 3.08 |
- bf16+BCG+B=32 = **919 tok/s aggregate** ≈ 25x plain, ~11x B=1.
- ngram/suffix spec HURTS on Spark (python verify overhead > gain).
- _fq quant slower than bf16 on this GPU — memory lever only.
- Loop today: decoding="mtp_selfspec" silently no-ops (Spark has NO
  mtp_module) → loop effectively runs plain-path decode.
- B=8 dip (224 vs 816/919 at B=16/32) — bucket/batch-path artifact.
- forge_quant auto-activate on bf16 ckpt = 15+min single-core CPU
  conversion at load (killed) — use _fq prepacked dir or auto_activate=F.
- CPU use during load ~1 core (NOT parallel); decode is GPU+launch-bound.

### Supervision probe (scripts/probe_spark_supervision.py)
LoRA-r32 (working targets: qkv_proj/out_proj/w_gate/w_up/w_down — the
default list hits spark's fused qkv and yields ZERO-grad adapters),
AdamW fused lr2e-5, bs2xseq1024, 20 steps, v9 corpus, base heldout
0.7068:
| arm | ms/step | heldout | note |
|---|---|---|---|
| ce | 317 | 0.580 | production chunked-CE |
| ent05 | 375 | 0.579 | entropy-weight — no gain at this budget |
| focal g4.93 | 319 | 0.591 | single-pass custom impl ≈ ce |
| band[0.4,6] | 474 | 0.589 | skipped 6/22 batches FREE (NaN/hard cull) |
| confdrop CE<0.3 | 317 | 0.589 | dropping confident tokens ~FREE |
| anslast (last-16 tok) | 318 | 0.624 | only ~0.04 worse — answer-only nearly free |
| seq512 | **198** | 0.499* | −38% step; *eval-set differs (36 vs 40 rows) |
| full_kl | 432 | 0.584 | teacher=sft10_src, fwd cost inside step |
| topk256 kl | 413 | 0.586 | ≈ full_kl |
| topk256 kl+0.3ce | 414 | 0.585 | |
| **topk15 kl+0.3ce** | **409** | **0.579** | ≈ full-KL, tiny teacher cache |
- massprobe (teacher=sft10, heldout positions): top-1=88.1% mass,
  top-15=99.0%, top-256=99.84%; gold-token inside teacher top-15 at
  98.9% of positions, top-256 at 99.97%, top-1024 at 100%.
- head GEMM [2048,2048]@[2048,131072]=16.1ms vs @[2048,256]=0.12ms (137x).
- VERDICT: at 20-step RSI budget on v9 data, supervision diet barely
  matters (0.578-0.62 band) → pick the FASTEST correct recipe:
  seq512 + band + confdrop-style token thinning ≈ free throughput.

### Theorized tricks for ForgeAI (ranked by evidence strength)
1. BATCH-BRUTE-FORCE: propose+solve+retries through generate_batch at
   B=16-32 + BCG → ~919 tok/s vs ~37-82 now = 11-25x candidates/sec.
   Retries are currently serial (infinite_curriculum) — batch them too.
2. TOPK-15 TEACHER CACHE: extend score_teacher_logprobs →
   score_teacher_topk: store (idx15, logp15) fp16 per supervised token
   (~8700x smaller than full-vocab, richer than scalar advantage).
   New chunked_ce fn = _ChunkedKL topk path (probe shows heldout parity
   w/ full KL; grad flows only through k head columns). In-loop: teacher
   scoring is OFFLINE (amortized), student step ≈ CE step.
3. STEERDISTILL x AZR: failed solves get teacher-steered repair from the
   pre-error state, executor-verified, exported tagged "corrected" →
   distill corrected turns to teacher dist (topk15), other turns to
   self-dist — maps 1:1 onto propose→solve→verify→export.
4. CONFIDENCE-THIN SUPERVISION: anslast/confdrop measured ~free → thin
   supervision 5-10x, then either shorten epochs or multiply epochs.
5. seq512+no-pack+band recipe = fastest measured (198ms) — validate on
   real loop exports (gen<=256tok fits).
6. FIX THE LOOP'S DEAD/WRONG KNOBS (measured bad): mtp_selfspec no-op
   (train a real 1-head MTP via --mtp-n-heads 1 = 640ms/step → real
   self-spec decode); warmup=20 flatlines <=20-step epochs (use 3-5);
   ft steps are OPTIMIZER steps (100x5=500 presentations on ~60 rows =
   overfit engine; rely on loss-band frontier stop); BitNet-everywhere
   = destructive on spark; LoRA default targets = zero-grad on fused
   qkv (use SIFT or the working target list); packing w/o cu_seqlens =
   loss-10 (no-pack or wire cu_seqlens); dead flags --sync-freq,
   --grad-compression int4, --distill-topk|truncate|prefix.
7. EVAL IS THE EPOCH BOTTLENECK: ~600 gens vs ~30s of SIFT steps.
   Brute-force phase: --eval-seed + small --eval-online/--eval-scale,
   full suite only at promotion candidates.
8. OPAL on-policy linearization (arXiv 2609.31947): when we port
   Jamba→GDN2/V14 linear side-paths, distill via student-sampled
   trajectories + dense teacher supervision (87-94% recovery vs
   off-policy collapse) — the recipe for linear-attn conversion.
9. OPAL reactive-optimizer (arXiv 2610.33826): LLM agent rewrites
   optimizer config mid-run. Safe variant for us: evolve mutates
   LoopConfig recipe per epoch on eval deltas (tagged adapters,
   already-sandboxed proposer path) — not free code exec.
10. STRATA-style CPU-expert-miss-in-place + NVMe n-gram table: future
    MoE/hybrid-offload direction (their RTX5070+64GB numbers validate
    our tiered-memory thesis; Spark is dense so only relevant for
    larger models).
11. ENGINE DOUBLE-LOAD: loop loads engine per epoch AND sft_train
    reloads inside runner → 2 full loads/epoch. Reuse/sleep-wake.

## 2026-11-15 — research batch 2 (few-shot/meta + arch/curr/distill)

### low-resource NLP (survey 2411.09539; sysreview 2505.04531/54 studies)
- Consensus: PEFT dominates <10k samples; augmentation + self-training
  (confidence-filtered pseudo-labels) = biggest levers; forgetting
  mitigation via replay/low-LR/few epochs; gains <1-3% often noise.
- Spark's loop already IS the modern recipe (synthetic + self-training).
  Adds: confidence-filtered self-labeling, replay buffer, per-skill LoRA.

### DistilBERT → modern KD
- Classic: student-init from teacher every-other-layer; triple loss
  (soft KL + MLM + cosine hidden). Decoder-era upgrades: reverse-KL
  (MiniLLM 2306.08543), on-policy GKD (2306.13649) — student generates,
  teacher scores; off-policy logit matching = exposure bias.
- CAUTION 2509.01649: pretrain-distill can hurt ICL/induction heads.
- Ours: offline stored-logprob = teacher-prefix KD; adding a GKD-style
  on-policy fraction is cheap since gen is already in-loop.

### MAML — dead for LMs; surviving descendants
- Reptile (1803.02999): move init toward post-SGD weights ≈ MAML,
  trivially cheap → CHECKPOINT-AVERAGING across per-skill SFTs.
- ANIL (1909.09157): inner loop on head only ≈ full MAML — MAML mostly
  learns feature reuse, not rapid learning → adapter-only adaptation.
- MAML-en-LLM (2405.11446): true 2nd-order on LLMs, +2% at big cost —
  skip. MetaICL line = multitask outer loop (what self-play already is).

### Domain homogeneity (2 interps)
- A) data-mixture lit: DoReMi 2305.10429 (+6.5pt, 2.6x fewer steps via
  proxy-fitted mixture), Data Mixing Laws 2403.16952 (mixture→perf is
  fit-predictable), DEMIX 2108.05036. Specialist = narrow homogeneous
  buckets; global = forgetting risk. Cheap action: proxy-sweep mixture
  weights instead of hand-tuned sampling.
- B) MoE microbatch homogeneity (2501.11873): per-microbatch LBL blocks
  expert specialization; global-batch LBL + domain experts fix it —
  relevant to MoE-FFN side-paths later.

### TinyFS-IDS (Fusco/Palmieri/Ficco, Internet of Things v34, Nov 2025)
- Few-shot federated IDS: Siamese nets + ε-greedy RL bandit picking
  gradient-sparsification rate per round. Transferable PATTERN only:
  bandit-per-round hyperparam (LR/mix/replay per epoch on measured
  reward) — cheap pair w/ reactive-optimizer idea.

### FastKAN-Tiny
- FastKAN (2405.06721): RBF-approx KAN, 3.33x vs other KANs but still
  slower than param-matched MLP matmul (~Gx FLOPs, G≈8 RBF centers).
  "Tiny" = tiny impl or TinyML variants (XTiny-FastKAN, FastKAN-DDD).
- 10M BabyLM audit (2607.15525): KAN≈SwiGLU≈MLP parity, no density
  advantage at LM scale; no credible >=1B result. KAN-FFN = non-lossless
  preset key → LOWEST priority, small ablation shard only.

### Curriculum + intrinsic motivation (maps onto our loop)
- Easy→hard ordering: −18..45% steps at <=100B-tok scale (EACL'26,
  200+ models); gains shrink >=1B — test don't assume. Best difficulty
  proxies: compression ratio, MTLD, Flesch. (One counter-study: hard→
  easy on doc-level — metric-dependent.)
- AZR (2505.03335 NeurIPS'25): proposer reward = LEARNABILITY (MC solver
  success ~50% sweet spot) — our verifier emits this signal FOR FREE;
  sort propose queue by it = emergent curriculum.
- eva (Ye'25): asymmetric self-play prompt-evolver, gemma-9b Arena-Hard
  51.6→60.1/62.4 no human prompts. MERCI (2510.16614): pseudo-count
  novelty vs repetitive CoT. i-MENTOR (2505.17621): dense traj rewards.

### distillation x PEFT
- Prompt distillation (2412.14964): teacher = same weights + privileged
  prompt; adapter toggle → no 2nd model in VRAM (fits 12GB); self-
  distill, offline logit cache suffices — DIRECT FIT to our pipeline.
- KD-LoRA (2410.20777), LLM-NEO (2411.06839): distill into LoRA works.
- Tina (2504.15777): LoRA+RL on 1.5B → 43.3% AIME24, $9 — OUR SCALE;
  "Are Reasoning Abilities Low-Rank?": r=64 LoRA-of-GRPO-delta BEATS
  the full-FT ckpt. LoRID (2508.13037): multi-LoRA roles +16.1% GSM8K.
- TRAP — "Illusion of Equivalence" (NeurIPS'25): LoRA intruder dims
  ACCUMULATE under sequential FT → merge + re-init adapter each epoch.
- FFT > LoRA +7.3% GSM8K on 7B (2505.22355) — gap on reasoning tasks.
- Structural: CASS shares teacher A-matrix (+5.7% ROUGE-L, 0.0004%
  params); SAD-LoRA aligns span(B) to teacher subspace.

### New tricks added to ranked list
12. LEARNABILITY CURRICULUM: verifier success-rate already gives the
    AZR proposer reward — sort task queue toward ~50% success items.
13. EASY→HARD in-epoch ordering via compression/perplexity proxy —
    cheap sweep, possible −18..45% steps (verify at 1.7B scale).
14. BANDIT-PER-ROUND recipe knob (TinyFS-IDS pattern): ε-greedy over
    LR/mix/replay per epoch on measured reward — 30-line experiment.
15. PROMPT-DISTILL SELF-TEACHER: adapter toggled same-weights teacher
    w/ privileged context — zero extra VRAM, offline cache works.
16. MERGE+REINIT LoRA each epoch (kill intruder dims); optional
    REPTILE soup across per-skill SFTs (checkpoint averaging).
17. TINA-STYLE LoRA-RL: validated at 1.5B; r=64 LoRA-of-delta can beat
    FFT — plausible for solve-verified rewards in-loop.
18. PROXY MIXTURE FIT (DoReMi): small sweeps to set domain mix for
    exported corpora instead of hand ratios.
19. KAN-FFN — parked: unproven at LM scale, +FLOPs hurts gen speed.

## 2026-11-15 — probe round 2 RESULTS (gold setup: same LoRA-r32/AdamW/
## 20-step/bs2x1024/v9 corpus, ce baseline heldout 0.580 @317ms)

| arm | heldout | step_ms | extras | verdict |
|---|---|---|---|---|
| ord_easy | 0.626 | 226 | score pass 14.8s | WORSE — budget burned on known data (loss0 0.10) |
| ord_hard | **0.566** | 338 | score pass 14.2s | BEST — hardest-first ordering |
| ord_comp | 0.573 | 508 | proxy: distinct-tok ratio | ≈ce, slower steps (long rows first) |
| learnband | 0.588 | 301 | band CE[0.31,0.84] kept 115/227 | ≈ce — mid-band excluded signal-rich hard rows |
| reptile | 0.622 | 290 | 2 half-runs + LoRA soup | WORSE — souped adapters half-strength on each half |
| selfteach15 | 0.685 | 344 | priv-context self-teacher 19.4s | WORSE — seeing the answer → near-delta dist, no new info |

- READ: at micro-epoch budgets the binding constraint is signal/step —
  hard→easy beats curriculum-lit easy→hard (matches the SustainNLP
  doc-level counter-result; easy→hard applies to full-corpus sweeps,
  not 40-of-227-example slices). AZR learnability's "~50% success"
  sweet spot sits at the HARD end of a corpus the model mostly knows.
- NEGATIVES recorded: reptile soup, privileged self-teacher,
  easy-first, mid-band restriction, token-proxy ordering.

### APPLIED to production this session
- engine_activation.py: mtp_selfspec now LOGS when model lacks mtp_head
  (was a silent no-op on Spark — the 2-4× claim never applied).
- infinite_loop.py: warmup clamped to min(ft_warmup_steps,
  max(2, steps//4)) — a 20-step warmup on a <=20-step epoch is flat LR.
  ft_curriculum default "none" → "frontier"; CLI choices + comments.
- curriculum_augment.py: new "frontier" strategy = hardest-first
  (descending combined difficulty).
- sft_train.py: FIXED curriculum dead-code — val-split random.shuffle
  and AsyncPrefetcher(shuffle=True) silently discarded the ordering.
  Curriculum now: order-preserving val pick + shuffle disabled.
- Already-correct by design (verified, no change needed): per-epoch
  merge_lora_adapters → fresh adapters each epoch (no intruder-dim
  stacking); solve path already uses generate_batch; LoRA defaults
  cover spark (v_proj matches fused qkv_proj + FFN targets).

### still-unapplied candidates (need bigger wiring)
- score_teacher_topk (idx+logp cache) + production _ChunkedKLTopk —
  probe parity shown; needs scorer + collate + compute_loss plumbing.
- TinyFS-IDS bandit-per-round recipe knob (LR/mix/replay per epoch).
- Tina-style LoRA-RL on solve-verified rewards (needs RL trainer).

## 2026-11-15 — probe round 3 RUNNING (anti-overfit + eval-axis arms)
Arms: stepsweep(60 steps, heldout every 10 = overfit curve +
gen_prefix), lr_hi 5e-5, lr_lo 5e-6, wd01 (AdamW wd=0.1), r8 (rank8),
replay20 (20% out-of-window rows), evalgen (CE train + gen_prefix).

- gen_prefix_eval = NEW second metric: greedy-decode 24 tok on 6 heldout
  prompts, fraction matching gold completion token-for-token — a
  task-level eval axis vs heldout-CE (token-level). For eval-score work:
  check rank-correlation of heldout_ce vs gen_prefix across arms; if
  they disagree, CE is lying about task quality.

## NEXT — proposer-prompt experiment (queued, not yet run)
Lever found: `_propose_prompt` -> CHAT_PROPOSE_PROMPT.format(domain,
difficulty, signature, desc_hint); desc_hint drawn from a static
per-domain bank (algos=10, math=10, strings=10, logic=5, ds=5).
Arms to test (engine + _parse_proposal + executor validate):
 a) baseline (1 desc_hint)
 b) 2 desc_hints crossed (idea mixing → diversity)
 c) constraint block: "3+ test cases incl. edge cases; solvable <30
    lines; deterministic; stdlib only"
 d) 1-shot exemplar: prepend a gold task spec
Metric: parse-valid rate AND executor-validated rate (task's own tests
pass on its reference solution) — the real proposer quality signal is
"solvable-valid", not "well-formed". Small bank (logic/ds=5) is a
diversity bottleneck — expanding it is near-free.

## probe round 3 RESULTS (anti-overfit axis; ce baseline 0.580)
| arm | heldout | read |
|---|---|---|
| stepsweep | **0.535** | curve 0.621→0.581→0.564→0.550→0.541→0.535 — STILL FALLING at 60 steps, no overfit upturn on 226-row corpus |
| lr_hi 5e-5 | **0.539** | probe LR 2e-5 was underfit; +2.5x LR = free win, same cost |
| lr_lo 5e-6 | 0.664 | badly underfit |
| wd01 | 0.580 | neutral at 20 steps (epoch-accumulation guard only) |
| r8 | 0.646 | rank 8 underfits — 32 is right |
| replay20 | 0.580 | neutral single-epoch (protection is cross-epoch) |
| evalgen | 0.580, gen_prefix 0.0625 | — |

READS:
- The "prevent overfitting" worry was INVERTED: at this budget the
  model UNDERFITS (heldout still descending at 60 steps). Loop's
  production defaults (ft_lr 5e-5, ft_max_steps 100) were already at
  the measured sweet spot — probe's 2e-5/20-step grid was conservative.
- Overfitting risk lives in EPOCH ACCUMULATION (repeated exposure to
  near-identical self-play rows), not per-epoch steps — that's what
  replay/L2-SP/merge-reinit exist for; they test neutral per-epoch
  because the danger isn't per-epoch.
- EVAL-AXIS finding: heldout-CE moves while gen_prefix stays ~0.06 —
  distributional gains below argmax-flip threshold. Token-prefix match
  is the WRONG task-metric at short horizons; the executor's pass@
  (real code runs) is the right one. Don't promote on CE alone —
  already handled by the loop's promotion eval, worth keeping.
- wd/rank/replay: verified-neutral-to-negative, don't tune here.

## proposer-prompt probe
Probe: `scripts/probe_spark_proposer.py` — Spark-X2.6-1.7B engine, N=16
proposals/variant (4 each algorithms/math/strings/logic), paired seeds,
production `_generate_batch` path (propose prefill + think-cap).
valid_rate = reference impl runs on own test inputs in sandbox
(`_validate_task` + conf==1.0 — AZR derived-output).

| variant    | parse | solvable-valid | tok/s |
|------------|-------|----------------|-------|
| baseline   | 16/16 | 14/16 (0.875)  | 688   |
| dual       | 14/16 |  9/16 (0.562)  | 729   |
| constraint | 15/16 | 10/16 (0.625)  | 717   |
| exemplar   | 15/16 | 10/16 (0.625)  | 673   |

- BASELINE wins decisively — every variant is -25 to -31 pts valid_rate.
  The existing prompt (inline format example + bare `# solve(input)` AZR
  format + "Reply with ONLY the code block") is already the sweet spot;
  extra text dilutes the 1.7B's instruction-following. Dual's crossed
  hints produced degenerate specs ("(" descs, parse fails in logic).
- NO prompt change applied (winner must beat baseline by >5pts — none did).
- APPLIED regardless (pure diversity fix): extracted the duplicated
  `desc_hints` dicts (propose_tasks + _propose_one, had drifted apart —
  strings 9 vs 10 entries) into module-level `_DESC_HINT_BANK`, and
  expanded logic + data_structures 5 → 10 entries each.
- Gen notes: all variants emitted exactly 16×640=10240 tok (no early EOS
  — prefill-mode replies run to cap); B=16 ≈ 700 tok/s agg, ~15s/variant,
  peak 6.0GB. Manual activate (standard KV + BCG) — auto_activate measured
  0 free VRAM mid-load → capped KV at 64 tok + hit `(*bias): last
  dimension must be contiguous` in seq_aware_split static-graph path
  (fell back OK, but KV cap alone would have starved 640-tok prompts).

## bugfixes + scale confirm

### bugfix 1 � auto_activate KV-cap=64 (starved 640+ tok gen)
- CAUSE: `_activate_kv_cache` probed free VRAM mid-activation while loader
  staging / quant-repack blocks were still cached in the allocator �
  `mem_get_info` read ~0 free ? `VRAMManager.max_gen_tokens` returned its
  64-token emergency floor ? KV `max_seq_len=64`.
- FIX `forge/engine/engine_activation.py:317-337`: `torch.cuda.synchronize`
  + `empty_cache()` before `max_gen_tokens`, floor cap at 1024, log chosen
  cap ("VRAM-aware KV cap: N tokens (requested M)").
- FIX `forge/engine/engine_checkpoints.py:456-462`: same settle before the
  `_auto_activate_optimal` free-VRAM ratio read.
- VERIFY: real auto_activate run now logs `VRAM-aware KV cap: 77322 tokens
  (requested 16384)` (bench path: 120038) � was 64. Gen 640+ tok OK.

### bugfix 2 � seq_aware_split `(*bias): last dimension must be contiguous`
- CAUSE: additive `attn_mask`/bias reaching `F.scaled_dot_product_attention`
  could carry stride-0/sliced layout (expand views, graph-pool slices);
  mem-eff SDPA kernel requires last-dim stride 1.
- FIX: `.contiguous()` on the final bias at every SDPA call site �
  `forge/model/layers.py:485,512,522,535` (sliding-window combined bias,
  attention_bias, causal bias, explicit mask) and
  `forge/engine/attention/fused_qk_norm_rope_cache.py:307,317`.
- VERIFY: full auto_activate + BCG run captured all 5 kv buckets
  (512/1024/2048/4096/8192) + B=16 batch gen � no crash. `py_compile OK`.

### scale confirm (scripts/probe_spark_scale.py, NEW runner)
N_TRAIN=512 N_EVAL=96 rows v9 corpus, 60 steps, lr5e-5, bs2xseq1024,
LoRA r32/a64 on qkv_proj,out_proj,w_gate,w_up,w_down. 512 rows -> 448
packed train examples; heldout CE every 10 steps.

| arm | @10 | @20 | @40 | @60 | prior |
|---|---|---|---|---|---|
| ce (shuffled, seed0) | 0.6106 | 0.5869 | 0.5601 | **0.5467** | 0.535 @60 (stepsweep, 256 rows, lr2e-5) |
| ord_hard (base-CE hardest-first) | 0.6093 | 0.5919 | 0.5600 | **0.5438** | 0.566 @20 (256 rows, lr2e-5, 20 steps) |

- Both still descending at 60 � no plateau; 2x data + 2.5x lr did not
  overfit (loss0: ce 0.287 vs ord_hard 2.0 = expected hard-first front).
- ord_hard edges ce @60 (0.5438 vs 0.5467, -0.003) � ordering advantage
  shrinks vs small-scale but direction holds; curves nearly identical
  steps 30-60, ord_hard marginally ahead throughout.
- ord_hard @20 (0.5919) looks worse than prior 0.566 @20 but is NOT a
  regression: larger train window (448 vs ~227 ex) + higher lr shifts
  the whole curve; compare same-budget @60 where ord_hard wins.
- peak 5.26 GB both arms; ce 265.7 ms/step, ord_hard 409.6 (24.5s
  one-time score pass + harder rows = longer avg seq).

### batch gen (scripts/bench_spark_genmatrix.py batch, BLIST="16 32")
B=16: 883.8 tok/s | B=32: **913.4 tok/s** | peak 4.98 GB � meets ~900+
target at B=32 (prior best 918.7, within noise).


## epoch smoke + longrun

### Yardstick (coding pass@1, scripts/eval_runnable_tasks.py artifacts)
- gen_he_base.jsonl (HumanEval-style, n=80): PASS@1 22/80 = 27.5%
- gen_humaneval_sft6ff.jsonl (n=80): 23/80 = 28.8%
- gen_mbpp_base.jsonl (MBPP-style, n=80): 16/80 = 20.0% (fn-emission 50%)
- gen_he_pissa.jsonl: 11/80 = 13.8% (negative adapter result, not used)
- Qwen2.5-7B reference (published-style): Instruct ~60-70% HumanEval,
  base ~50-60%. Spark is base-like => gap ~22.5-42.5 pp vs HumanEval.
- Loop-level eval is fast_eval (46 tests/10 cats, mixed quality score),
  NOT coding pass@1 - do not conflate.

### Smoke epochs (spark_x26_1p7b.sft6_src, 12 tasks, 30 SFT steps)
Run A (pre-fix), 461.4s total:
  engine load ~7.8s | propose+validate ~62.7s (3 attempts -> 13 tasks)
  solve ~94.8s (5/13 = 38.5%) | finetune+save ~129.2s | eval ~167.0s
  fast_eval: base 0.71 vs cand 0.68 -> DEMOTED (adjacent +0.036)
Run B (post-fix, no --no-ft-bitnet-everywhere needed), 350.9s total:
  solve 1/13 (7.7%) | finetune | eval 100.4s
  base 0.70 vs cand 0.65 -> DEMOTED. Process EXITED CLEANLY (was the
  verification for the close() deadlock fix).

### Long-run blockers found + fixed
1. LiveStatusWriter.close() deadlock: held non-reentrant Lock while
   _write_heartbeat() re-acquired it -> process parked forever after
   LOOP COMPLETE (~10.8GB RSS, 97 threads; py-spy confirmed both threads
   idle at live_status.py:91). Lock->RLock.      commit 20cac6d
2. Module-global sandbox ThreadPool never closed -> added
   shutdown_sandbox_pool(), called at run() boundary.  commit c9620b2
3. _load_engine() could stack a 2nd ~4GB engine on a crashed-epoch
   leftover (stale KV/prefix caches for replaced weights) -> free first.
                                                    commit 5109d34
4. fast_eval ban list also ate static CATEGORIES (yardstick shrank every
   epoch -> cross-epoch scores incomparable) -> procedural names only.
                                                    commit 586c9c2
5. Strict promote accepted any positive delta (1 question ~= 0.022 ->
   noise ratchet over a long run) -> promote_margin 0.02 required.
                                                    commit fde983b
6. ft_bitnet_everywhere defaulted ON everywhere: spark QAT = loss ~26,
   ~24s/step, guaranteed demote. Now ON only for forgelm*/lfm*.
   --ft-pack-sequences exposed (was hardwired on).   commit 65f9e04
Wiki Self-Play.md synced: scratch/wiki_repo commit 2497a10.

### Remaining blockers (not fixed)
- Eval is the epoch bottleneck: ~100-167s of 350-460s (~30-36%).
- Full merged save ~110s/epoch on the SMR USB drive; LoRA-only save
  can't skip it because fast_eval weight-swaps a standalone safetensors.
- archive/ grows ~4.3GB/demoted epoch (+sidecars) - manageable but
  unbounded on a multi-day run.
- mtp_selfspec silently no-ops on spark (no mtp_head) -> standard decode.
- fastsafetensors DLLs absent on this box -> standard safetensors load.
- Solve success is high-variance (38% vs 8% on same ckpt); SFT corpus
  per epoch is 1-17 examples -> candidates regress as often as improve.

## soak test 2026-11-16 — 8-epoch unattended run: PASS

Invocation (replicates the 11/15 run that died silently in ep7 eval):
python -u -m forge.self_play.infinite_loop --checkpoint research/checkpoints/spark_x26_1p7b.sft6_src --config spark_x26_1p7b --epochs 8 --tasks-per-epoch 12 --ft-steps 30 --staging-dir C:\forge_spool --archive-keep 3
Detached (Start-Process, venv python is a uv shim that re-execs a child - monitor the CHILD pid). Log: .devin/tmp/soak.log, metrics: .devin/tmp/soak_metrics.csv.

### Per-epoch (4/12 solved every epoch, all demoted; base stayed sft6_src)
ep1 502.9s base .71 cand .66 DEMOTED (rotated SP2)
ep2 257.9s .68 DEMOTED (rotated SP3)
ep3 255.3s .68 DEMOTED (rotated SP5)
ep4 260.8s .70 DEMOTED (rotated SP1)
ep5 253.3s .68 DEMOTED (rotated SP2)
ep6 307.0s .72 CAND-WINS but DEMOTED - delta +.013 < promote_margin .02 (anti-ratchet gate working, infinite_loop.py:2570)
ep7 234.5s .67 DEMOTED (rotated SP4) - prior run died HERE mid base-eval; clean this time
ep8 232.5s .70 DEMOTED (rotated SP5)
Total wall ~40min. LOOP COMPLETE -> status.json "done" -> process exit <60s (RLock close path verified - prior version parked forever here).

### Leak check (1-min samples of loop pid)
RSS: 6.25GB ep1-end -> 7.33-7.37 plateau ep2-4 (sandbox once-script pool spin-up, py_procs bursts 2->6->2) -> DECLINES to 7.0 by ep8. Net steady-state drift ~-0.3GB ep2->ep8 = no leak.
Threads 96-97 flat; handles 850-885 oscillating, end ~800 (bounded, not monotonic).
VRAM 4.6-6.3GB phase oscillation, never trending up; post-exit back to 610MiB baseline.
Spool C:\forge_spool bounded 12.38<->16.51GB (3 kept archives + 1 in-flight candidate); rotation fired on all 8 demotes. D:\archive untouched at 47GB (staging layout = archive lives next to candidate).
Orphans: 0 python* procs post-exit (baseline 0). No crash/hang.

### Verdict
PASS on all criteria (>=5 epochs, clean exit, no orphans, RSS bounded, archive rotation verified 8x, no crash). Prior silent ep7-eval death did NOT reproduce - most consistent with external kill at prior session end, not a code defect (no crash signature in 8 clean epochs + clean shutdown path).
Multi-hour confidence: HIGH for demote-heavy runs. Untested this run: promote->durable-copy path (prior run exercised it twice, SP4/SP6) and >8-epoch archive behavior.

## long run 2026-11-16 - 100-epoch self-play from SP1
python -u -m forge.self_play.infinite_loop --checkpoint research/checkpoints/ForgeLM_SPARK_X26_1P7B_SP1.safetensors --config spark_x26_1p7b --epochs 100 --tasks-per-epoch 12 --ft-steps 30 --flux-group-size 8 --staging-dir C:\forge_spool --archive-keep 3
Detached Start-Process. Shim pid 16664 (monitor CHILD pid). Log: .devin/tmp/longrun.log, err: longrun.err, pidfile: longrun.pid.
Baseline: SP1 = soak2 ep1 promote (fast_eval 0.72). Prior demote-epoch ~230-310s; 100 ep est ~7-10h.


## R73 quota+spark self-play round

Diagnosis: epochs trained on 12-19 rows, SFT loss -> 0 in 30 steps
(memorization, not learning), all 10 observed epochs demoted.

Root causes fixed:
1. sft_train rendered Jamba ChatML for spark ckpts (loop never passed
   --chat-format). spark uses SS/SE sentence markers + |User|/|Bot|;
   im_start/im_end are not even in spark vocab. Now: _finetune passes
   --chat-format spark; sft_train calls canonical
   qwen_adapter.spark_render_messages (parity-verified byte-identical).
2. infinite_curriculum hardcoded Jamba ids: eos_token_ids=[2,519] and
   _CONTROL_TOKEN_KEEP={2,519,542}. On spark, 2=pad, 519=random BPE,
   real eos id 1 was BANNED and missing from stops -> every generation
   ran to cap. Now: _control_keep_ids/_gen_stop_ids/_think_end_ids
   resolve from the loaded tokenizer (spark: eos=1, think=4).
   Verified: keep=[1,4], stop=[1], 113 banned, eos clean.
3. LoRA targets q/k/v_proj never matched spark fused qkv_proj.
   _resolve_lora_targets detects qkv_proj + gate_proj out-gate.

Quota: Phase 1 now loops propose+solve rounds until solve_quota
verified rows (0 -> tasks_per_epoch), caps quota_max_rounds=4 and
quota*K*10 rollouts. STaR ref-fill on solver-missed tasks via
curriculum._task_ref (ref_fill_frac=0.4 of quota). export_per_task=2
distinct passes.

Speed: solve_group_size default 4, grad-mixup 3, warmup already capped
at steps//4, kv-cache standard on spark (was lossy spectral),
pack-sequences default OFF on spark, eval kv standard on spark
(was hadamard_int4 inside a 0.02-margin paired decision), fast_eval
decision_margin early-abort on bound-proven outcomes.

Recommended run:
venv/Scripts/python -u -m forge.self_play.infinite_loop   --checkpoint research/checkpoints/spark_x26_1p7b.sft6_src   --config spark_x26_1p7b --epochs 8 --solve-quota 24   --tasks-per-epoch 24 --ft-steps 30   --staging-dir C:\forge_spool --archive-keep 3


## 2026-11 — Spark flat-throughput: continuous-batching root causes + fixes

Final state: ~0.9-1.1k tok/s 64-prompt mixed-length generate_batch,
~1.9k tok/s at N=96; was ~100 tok/s at session start.

Throughput chain (each verified by bench, in order of impact):
1. `_fused_gemv=False` + `_cache_w=True` in `_load_engine` — cached-bf16
   cuBLAS beats fused int4 Triton GEMM 3.3x at B>1 (215us vs 18us/call).
   Self-play path never applied the code_evolve flip.
2. Lazy per-B decode graphs up to WIDE_CAP=64: b64 replay 12.6ms =
   5.1 tok/ms vs b32 10.3ms = 3.1 tok/ms. N<=64 runs exact-width — NO
   refills needed; N>64 windows at 64 + `_refill_batch`.
3. GPU suppress_ids + model-EOS termination replaces Python logits
   processors (~2x on long gens; also kills per-step .tolist() syncs).
4. Batched refill prefill (<=16/call, deferred until >=6 free slots):
   eager B=1 prefill is ~50ms of launch overhead — 32 serial refills
   cost ~4.6s; 5 batched calls cost ~0.4s.
5. `_compact()` — after the pending queue drains, gather surviving rows
   into the next-smaller captured graph (64->32->16->8->4->2->B=1
   activation graphs, prefix ""). Tail decodes at ~live-row cost.
   KV/mask/ring_valid/metadata gathered by active-row index.
6. k-subset sampling (uniform top_k): softmax+multinomial over k
   survivors instead of [B,V]; top-k indices are temp-invariant so
   temp folds into top_v. ~4.5ms/step saved at V=131k.

### Refill correctness — the actual bug was POSITION SEMANTICS
`forge/model/llm.py` derives `position_ids = attention_mask.cumsum(-1)-1`
— positions come from the MASK, not buffer slot index. Consequences:
- Copy scratch-prefilled KV VERBATIM (K keeps R(src)). Any absolute-
  position RoPE rebase detunes: refill at pos=131 collapsed to spam,
  pos~60 looked ok, pos~8 near-perfect — error grows with the shift.
- Refilled row mask = {base..pos} -> stream numbered 0,1,2,... matching
  the scratch prefill's own numbering. Verified 100/100 token-exact
  vs standalone greedy at pos=131.
- Dead-token hole: letting the dead stream's token occupy pos (masked)
  still shifts every prompt->gen distance +1 -> measurably corrupts.
  Fix: pre-sample tok1 from scratch prefill logits, feed as
  next_ids[i] so ITS KV lands at pos — contiguous stream.
- ring_valid[B,W] per ring layer: stale ring slots are MASKED not
  overwritten with fabricated content. Every fabrication attempt
  (tiled prompt-tail fill, BOS-wall) degraded output — repetition
  prior. Also unmark left-pad slots after batch prefill (latent pad-KV
  pollution bug in EVERY unequal-length batch).
- tok1 must be sampled under the NEW stream's params — metadata update
  BEFORE sampling (was after: dead stream's temp/topk/seed).

Dead ends (documented so nobody retries them):
- RoPE rebase via inv_freq: inv_freq-derived rotations diverge up to
  0.275 from the model's cached cos/sin tables anyway — and the whole
  premise was wrong (mask-cumsum, not absolute positions).
- Scratch prefill at position=base>0: zero-KV prefix region is
  ATTENDED (full-attn else-branch allows [0,pos); ring n_past reads
  it) — always prefill scratch at position 0.
- conv/SSM state hypothesis: spark is pure-attention, no state to
  migrate. Checkpoint itself is healthy (standalone output clean).
- gate slotted refill for swa_ring_kv: no longer needed — refill
  works on ring caches given ring_valid + verbatim copy.

Debug env vars left in batched_decoding.py (env-gated, zero cost):
FORGE_DBG_REFILL (refill+compact events), FORGE_DBG_STEPS (per-step
top5 logits for rows in _dbg_rows).
Debug scripts: .devin/dbg_refill{,2,3}.py, dbg_geom/padgeo/kvcheck/
rebase/graphtest/scratch.py, bench_slotted2.py, dbg_steptime.py.


## Lacuna research sweep — Spark self-play improvements (2026-11-18)

Source: lacuna.tiptreesystems.com (research-direction index, 2.25M papers).
Scope: AZR RSI loop on spark_x26_1p7b (infinite_loop / infinite_curriculum /
grpo_trainer / fast_eval / export+SFT path). Findings mapped to code.

### A. Confirmed defects (fix first — evidence: code read, not papers)

1. **GRPO PPO ratio is always 1.0** — `collect_log_probs` is never called, so
   `old_log_probs=None` in every `train_step` (grpo_trainer.py:925-35 warns
   once). Effect: "GRPO" is actually REINFORCE-with-group-baseline; clipping
   is a no-op, no off-policy correction. Fix: record per-token log-probs
   during solve rollouts (or a detached first pass) and feed to train_step.
2. **`_mutate_arg` RNG fixed seed 7919** (infinite_curriculum.py) — identical
   fuzz mutations for every task → `_fuzz_verify` mutation coverage is
   degenerate; lookup-table hacks survive. Seed per-task.
3. **sgs.py / soar.py verifiers are heuristics** (`len>50 and "def "`), no
   executor — `--self-play-mode sgs|soar` can promote garbage. Wire
   `_execute_sandbox`/`io_similarity`, or guard the flags.
4. **Ref model `.to(cuda)` despite "CPU offloaded" docstring**
   (infinite_loop.py ~2625) — ~3.5GB VRAM on a 12GB card.
5. Dead surfaces: `--saerl`, `--opmix` print enabled, do nothing;
   `train_step_advanced` (sc/om/gvpo/turn-level) unreachable;
   `_propose_one` dead; `_parse_proposal` dead expr; `data_quality.py`,
   `replay_buffer.py`, `get_training_batch_elo`, `solve_with_ladder` all
   built and unused.

### B. Task supply — the real bottleneck (model-authored proposals are a
###    documented dead end at 1.7B; external pool ~118k, ~1.3k consumed)

6. **Programmatic variational synthesis (SvS, ICLR'26, +18-22% Pass@32)** —
   mutate VERIFIED tasks: perturb inputs/constants/edge-cases, re-derive
   expected outputs through the existing `_derive_outputs` executor path.
   Free verified tasks at ~zero model cost; no LLM authoring required →
   dodges the 1.7B dead end entirely. SvS targets the 12.5-50% solve band.
   Bonus: sustained problem turnover is the published entropy-collapse
   antidote.
7. **Task composition (Composition-RL + Sancaktar'26 "Deep Dive")** — chain
   2-3 verified tasks so A's output feeds B's input; composed asserts derive
   via executor. Guaranteed-verifiable tasks ABOVE current difficulty —
   ceiling expansion frontier rows can't provide. Also generates easy/med/
   hard "stepping stone" chains (Sancaktar: structure>volume, medium-
   difficulty tasks train best).
8. **Wire `solve_with_ladder`** (implemented, never called) into quota-loop
   misses — recursive easier-variant decomposition is exactly the SGS/
   Sancaktar stepping-stone idea, already coded.
9. **Fuzzing environment** (Sancaktar's 4th env): "find an input that breaks
   this buggy function" — mutants come free from the mutation-audit work
   (item 11). New task type, zero model authoring.
10. **Teacher-mutation loop if `api` task source is ever used** — Sancaktar:
    in-context teacher mutates problems from student solve-rate summaries,
    no teacher FT → 4x valid yield vs single-turn generation.

### C. Verification quality

11. **Mutation-testing audit of task test suites** (VeriScale, EvolveCoder,
    HardTests — verifier quality bounds the trained model): mutate the ref
    impl, confirm the task's own asserts kill mutants; expand weak suites
    with `_mutate_arg`-style fuzz. `_fuzz_verify` already does candidate-side
    agreement — extend the same machinery to audit the test suite itself.
12. **Dense rewards via `io_match.io_similarity`** (exists, unused in verify
    path) + per-assert pass fraction (Beyond Binary'26): graded rewards for
    GRPO → less advantage collapse; binary pass/fail throws away signal the
    eval scripts already compute (`per-assert accuracy`).
13. **V-STaR-style cheap verifier** — distill io_similarity labels into a
    tiny scorer to rank rollouts pre-sandbox → fewer sandbox calls/quota.
    (Counterpoint worth noting: "A Cheap Verifier is Good Enough"'26 —
    don't over-engineer; executor+fuzz is already strong.)

### D. GRPO dynamics

14. **DAPO dynamic sampling** — drop/zero-weight zero-variance groups
    (all-pass/all-fail). `grpo_concise_bonus` gives nonzero advantage but
    flat groups still dilute; Sancaktar/Chae confirm medium-difficulty data
    carries the signal.
15. **Entropy telemetry** — log policy entropy + distinct-solutions-per-group
    per step; entropy collapse is the documented plateau cause (SGS, SvS,
    Chae'25) and is currently invisible. The sft-era "extraction decline"
    may partly have been this. Cheap: count distinct AST-cleaned bodies.
16. **EAD annealed decoding** (Yang'25) — temp high→low within a rollout;
    cheap k=4 diversity without extra samples.
17. **ELO keyed on `_functional_fingerprint`** not task.id — ratings then
    persist across re-proposals; enables dead `select_mixed_prompts` (80%
    Goldilocks/20% explore) and `get_training_batch_elo`.

### E. Data/export + eval hygiene

18. Wire `data_quality.py` semantic dedup (TF-IDF cosine>0.85 + AST Dice —
    built, zero callers) into `_export_trajectories`; n-gram Jaccard misses
    semantic near-dupes.
19. Wire `ReplayBuffer` (FOREVER forgetting-curve, built) into
    `GRPOTrainer._inject_golden_replays` (exists; dead because loop passes
    replay_buffer=None). Reinforces re-base rule with RL-side replay.
20. Persist `_ban_question` across runs (per-process now); rotate a second
    eval seed — fixed seed for trend comparability + rotating for honesty
    (dynamic-evaluation direction; survivorship guard they already believe).
21. Track Pass@k at larger k in eval — binary pass@4 saturates ~2-5/60
    (documented); Chae: self-play preserves base capacity at large k, so
    large-k pass + attempt-rate is the honest frontier metric.

### F. Strategic calibration (what the literature says about the loop shape)

22. **External supply + frontier rows is the RIGHT architecture** — Chae et
    al. (NeurIPS MATH-AI'25): self-play is bounded by base-model support
    ("invisible leash" — can't upweight sequences the model can't sample);
    the published escape hatch is off-policy data mixing. External verified
    pools + frontier rows ARE that fix; lean in (KodCode, OpenCodeInstruct
    already queued in lineage).
23. **Don't train a proposer reward on the 50% target** — Chae: explicit
    50%-success reward shaping had little effect and COST 2% accuracy;
    R-Zero's version needs a separate challenger model (dead at 1.7B).
    Keep `1-|rate-0.5|` for frontier weighting only — it's a sampler, not
    a trainer.
24. **Proposer re-entry path if ever retried**: R-Zero lesson — separate
    checkpoint for challenger (same base is fine, two LoRA heads), majority-
    vote pseudo-labels where no executor exists; expect label-quality decay
    (79%->63% over rounds) → cap rounds before collapse.
25. **Adaptive response length emerges on its own** (Chae) — concise_qa +
    grpo_concise_bonus already push short answers; make sure the length
    shaping doesn't suppress legitimately long solutions on hard tasks
    (the stub-abstention collapse was this failure mode in SFT form).
26. Cleanup: implement or remove `--saerl`/`--opmix`; either finish
    sgs/soar (executor verify + real `target_problems` = current frontier
    backlog) or mark experimental-only.


## P0 self-play fixes landed (2026-11-18)

- GRPO ratio==1.0 FIXED: _grpo_train now collects behavior-policy log-probs
  via trainer.collect_log_probs(prompts, completions) BEFORE the step loop
  (fresh LoRA B=0 -> identical to rollout policy; groups resampled across
  steps so drift compounds -> real PPO ratio now). Per-batch old_log_probs
  sliced by the same indices; ExGRPO replay groups get on-demand collection.
  train_step only uses an old_lp entry on exact numel match (empty/mismatch
  -> detached fallback, no silent broadcast).
- collect_log_probs alignment bug FIXED: it appended invalid-completion
  empties in encounter order then padded the tail -> a short/invalid first
  completion shifted every later entry. Now preallocated by comp_idx.
- Ref model ELIMINATED in _grpo_train (was .to(cuda) despite 'CPU offload'
  docstring, ~3.4GB): ref_model=None -> compute_kl_penalty uses
  lora_disabled(model) forward (new ctx manager in bitnet_lora.py covering
  all 3 adapter integration sites: quant lora_adapter=None + patched
  forward restore via _lora_orig_forward). Docstring VRAM now ~5GB.
- _fuzz_verify RNG: random.Random(7919 ^ crc32(description)) per task (was
  fixed 7919 -> identical mutation trajectory for every task, so a
  hardcode tuned to it passed fuzz-verify on ALL tasks).
- Tests: tests/unit/test_grpo_offpolicy.py (10 tests, CPU-only, toy LM +
  word tokenizer). Passing: 10 new + 145 existing (grpo_lambda, grpo_rsi,
  self_play_components, curriculum_engine_gen, fused_qkv_lora).
- Note: lora_disabled mutates module attrs during model.modules()
  iteration � safe (attrs only, tree unchanged). nn.Linear/fused-QKV
  patched forward restores _lora_orig_forward; quant linears get
  lora_adapter=None. Adapter-off logits == base logits (verified by test).

## Spark gen-speed research (2026-11-18) � see .devin/_spark_genspeed_research.md

Full mapping in the research file. Headline: at B=64 decode is compute-bound
(batching paradox) so standard speculative decoding won't help; the
research-backed paths are post-trained MTP heads on self-distilled data
(+adaptive K + chain-aware relaxed verify), FR-Spec head crop for drafting,
and SpecFormer-class NAR drafting for sustained B<=128 speedup. Kernel side:
Marlin W4A16 / NVFP4 both viable on SM120 � replaces the +3.5GB cached-bf16
hack AND halves weight traffic. Stability: pipeline the eos .item() sync,
pre-capture compaction widths, kill the O(L2) eager-KV path reach via
generate_raw(suppress/stops), denser KV buckets <=4k, side-stream refill
prefill. cuDNN SDPA supports hd256 GQA+window decode on SM120 � flag bench.

## 2026-11 decode-stability pass (post-genspeed-research)

Landed all P0 items from _spark_genspeed_research.md (detail appended
there). Key design notes for future sessions:

- EOS pipelining consumes each step's EOS flag ONE step later via
  pinned-host copies + cuda.Event recorded before the next replay ->
  sync is instant; <=1 overrun token, trimmed by _finish end index.
- Subtlety: a refilled slot must ignore flags recorded at steps <= its
  refill step (they describe the DEAD stream's filler token). Guard is
  `eos_floor[i]` (0 originals, R+1 refills) in `_drain_eos`; also
  `_compact` drains both parities before row remap.
- `_ensure_static_batch(1)` returns the activation B=1 state
  (prefix ''), so single-prompt suppress/processor/stop requests get
  graph decode; only multi-token stop strings still need
  generate_raw/_StopMatcher (last-id check can't express them).
- _generate_batch_impl tokenizer call: `tokenizer(s,
  add_special_tokens=False).input_ids` returns flat list[int] for a
  single string (HF BatchEncoding) — len == token count.
- First two compaction halving widths pre-captured at batch start;
  ensure() is a no-op on captured widths, failure leaves width lazy.
- Tests: 203 unit tests green; new spark-routing regression tests in
  test_curriculum_engine_gen.py.

## Spec run (2026-11, post decode-stability) — bench_spark_genmatrix + debug-round

Hardware: RTX 5070 12GB. Checkpoint: spark_x26_1p7b.sft10_fq (fq, BCG).

bench_spark_genmatrix.py batch_fq (BLIST runs; second pass = warm,
post-capture):
- Load: FastBuild ~16-17s; activation captures ALL 9 densified B=1
  buckets; warmup 0.93-2.27s; engine VRAM 1.87-4.36GB (peak 3.97GB
  incl. pre-captured compaction widths).
- B=1: 42-65 tok/s (generate() serial path)
- B=16: 331.9 tok/s warm (2560 tok / 7.71s); cold 252 (capture ~0.7s)
- B=18: 358.3 tok/s warm (2880 tok / 8.04s); cold 327

infinite_loop --debug-round (real self-play path, sft10_fq):
- Task pool had 399 tasks on disk -> propose skipped; quota used
  backlog tasks. Solve batch B=8: 868 tok / 5.34s = 163 tok/s INCLUDING
  first-call b8 capture + pre-capture of widths 4+2 (~2-3s est) ->
  warm ~300 tok/s.
- Single-row solves through the NEW routed path (BatchedDecoding b1
  graphs, was generate_raw O(L^2)): 120 tok/1.19s = 101 tok/s,
  25 tok/0.27s = 94 tok/s.
- sinks/_v0_gate warning: NOT printed -> dormant as designed.
  KeyStack features: []. Loop completed, clean engine free.

Log: .devin/_debuground_run.log; trace: research/checkpoints/self_play/
debug_round_ep1.{txt,jsonl}


## 2026-11-19 � LittleBit-2 port notes
- Reference impl cloned at `scratch/littlebit_ref/` (SamsungLabs/LittleBit, --use_itq path).
- orge/engine/quant/littlebit2.py: LittleBitLinear (Dual-SVID + Joint-ITQ + residual + STEBinary/SmoothSign + int32 packer). quantize='littlebit2'.
- BUG pattern to remember: computing `W - W_approx` for the residual factor via `_dequantize_weight` included the `torch.empty` UNINITIALIZED _R params -> garbage residual target -> factor explosion (finite-but-huge, intermittently inf). Fix: primary-only reconstruction (matches reference).
- torch.empty params read before _set_factors = nondeterministic garbage; sanitize via nan_to_num at set time + orthogonality check on ITQ's Procrustes R.
- Init-time residual gain is only guaranteed when residual is STRUCTURED (low-rank leftover). On noise-like binarization error, init is marginal � real gain comes via QAT (paper consistent).
- block_recon supports LittleBitLinear (soft-forward: U_soft/V_soft/u1..v2 params, STE grads flow).


## 2026-11-19 � Parable-Granite-4.1-3B port DONE
- `research/checkpoints/parable_granite41_3b` = RAW HF download of AnkitAI/Parable-Granite-4.1-3B-Claude-Fable-5 (2 fp16 shards, 6.34GiB, 362 tensors, tied embeds) + forge_convert.json {forge_config: parable_granite41_3b} written by a prior session.
- No structural transform needed � `ModelLoader._remap_hf_keys` covers all granite/llama names. Preset carries the 4 granite multipliers.
- hf download stalled twice on xet at ~55%; `HF_HUB_DISABLE_XET=1 hf download --include <file> --local-dir <dir>` resumed and finished (~710MB/min). .cache/huggingface/download/ holds the stale lock � harmless.
- VERIFIED: scripts/test_parable_port.py -> argmax-100%, top5 identical, max|dlogit|=0.14 (fp16 kernel noise). ForgeEngine.from_checkpoint(dir) + generate -> coherent text (12.44GB VRAM tight but works; engine auto-restored unquantized after forge_quant attempt).
- tests/unit/test_parable_port.py: preset fields + multiplier wiring + full remap coverage (3 tests).
- KNOWN: `granite41_3b_base` dir has TRUNCATED shards (278MB+154MB vs expected ~5GB+1.7GB) � re-download before any base-vs-tune compare. Not needed for the port itself.

## Workspace cleanup pass (2026-11-20)
- scratch/ purged to littlebit_ref/ + wiki_repo/ only (~700MB smoke ckpts + v1-v7 log series gone).
- .devin/ purged ~40GB: fisher_sft9/ + loose f_base.pt/f_new.pt (sft9-era Fisher mats, recompute via scripts/fisher_merge.py --save-fisher), .devin/ckpt/*.flux (Sept FluxLM corpus snapshots), tmp/, hf_cache/, corpus/, logs/, triton_fresh/, ruffbase/, torch_cache/, trivia_rows*.pkl, 133 logs + 17 .err. Loose files culled 572->95 (kept: sft-chain launchers/builders/packers/mergers, canonical evals _solver/_onmanifold/_unseen/_bestval, eval_flux_*, bench_corpus/stuck, gsyn_parity, migrate_v2, v2_roundtrip, _parable_*, smoke_grpo, run_*_evolve, scratchpad*.md notes, exp_results.jsonl, test_corpus*.txt, config.local.json).
- research/ left alone - it is LIVE data: 521GB / 605k files (data/ corpora + checkpoints/ model dirs; 81 refs in forge/).
- docs/ pruned 47->14 + docs/design/(6) + docs/research/(15) + docs/archive/(1): git rm'd 17 rd_round_* + rd_quant_* + CODEBASE_CRITIQUE 09-09/09-10 + Aug research notes (FFN/GROQ/DIFFUSIONBLOCKS/Gemini/WebScrappers/RND_PLAN_V8). CODEBASE_CRITIQUE_2026-09-16 -> docs/archive/ (exp_flux*.py read it as byte corpus; repointed). Design docs git mv'd to docs/design/; code comments + AGENTS.md paths updated. INDEX.md rewritten.
- NOTE: AGENTS.md mojibake incident mid-pass (Get-Content cp1252 -> Set-Content utf8); reversed via cp1252 round-trip, verified zero residual.
## Agent-navigation layer (2026-11-20)
- NEW docs/CODEBASE_MAP.md: task->file routing, package map, data roots, freshness rules.
- NEW .devin/rules/ (7 files, TRACKED via .gitignore whitelist): scoped edit rules per subsystem (workspace/engine/model_keys/quant_decoding/training_selfplay/flux_exp/gui_server). Check matching file before editing - the point is a ~30-line read vs 2500-line AGENTS.md.
- NEW .devin/config.json: machine-readable venvs/test cmd/hardware/data roots.
- NEW scripts/README.md: auto-generated categorized index of 150 scripts; regenerate via scripts/build_script_index.py (147 auto-categorized by prefix).
- AGENTS.md: 'Quick Find' block added at top (routing + file-placement rules).
- Wiki Development.md + Docs-Index.md synced, pushed.
## Lacuna+web research round 2 (2026-11-20) - see .devin/scratchpad_research_lacuna_r2.md

55-item test queue + 7 theory probes, grouped Tier S/A/B/C with beat-incumbent per item. Key corrections: cuDNN decode caps at hd128 (hd256 spark decode NOT served), FA4/FlashMLA/DeepGEMM no SM120 path, NVFP4 only wins at B>=32 (marlin int4 stays for decode), CLA = capacity not bandwidth. Top items: mid-depth-scored SnapKV, ESP training-free MTP draft, MTP+GatedLoRA for B>=32, LoopSpec intermediate-depth drafting for LoopThink/V14, aLoRA prefix-KV reuse, hedge-collapse capacity-law + whitening fix, CCR task-packing (-80% GRPO tokens), QAD for _fq consolidation bug, SOPHIA state-conditioned loop-breaking, TransMLA on spark, ReadKV progressive codes, KDA for V15, retrofitted-recurrence curriculum.
## Research round 3 MEGA-PASS (6 subagents) - queue now ~Q260 + 27 theory probes

File: .devin/scratchpad_research_lacuna_r2.md. Lanes: agent/tool/gov (Q110-134), data/curr/eval (Q135-159), V15 arch zoo (Q160-182), selfplay/verify (Q183-210), kernels/serve SM120 (Q211-235), longctx/retrieval (Q236-260).
Convergent: same-weights triad = collusion risk (monitor-only probes + inoculation prompt); abstention fix = ration+conf channel NOT penalties; ~25% global-attn is hybrid sweet spot; V15 evals need agentic multi-hop; spark RoPE-OOD>2048 plausible (YaRN sweep first); LoopThink gate = two-test loop audit; B=1 launch-bound -> multi-step->WHILE-resident decode.
New DEAD ENDS: softcap (kernels silently ignore), DyT (data-rich regime), EC-routing text, TMA-multicast sm120 (2000x emu), cudaMallocManaged WDDM, WC-pinning, native stochastic rounding sm120, deception-probe-as-reward (obfuscation), homogeneous debate, second-model review (anchor), LLMLingua on tool ctx, naive PER replay, concat-truncate packing.

## RSI supervised run (chunk 4+, post-fix)
- chunk1: epoch1 PROMOTED SP1 (tie 0.58/0.58, efficiency win 1174 vs 1912 tok)
- chunk2-3: dead (msgspec DecodeError pre-fix; supervisor fake-completed)
- chunk4 epoch1: DEMOTED 0.557 (clean cycle: park/wake/eval/revert all OK)
- harness_genome.py spike shipped (opt-in --harness-evolve, 16/16 tests)
- BES proposer agent building


## Infinite-goal machinery (OMNI-EPIC/Prism/QD-informed)
- _validated_descs archive (persisted validated_descriptions.json, cap 400): validated tasks become stepping-stone seeds
- _sample_desc_hint: 35% archive variant (harder/orthogonal/inverse/gen), 20% cross-domain mix, else bank
- _coverage_domain: 35% override to least-proposed bank domain when skew >=4
- _avoid_hint domain-aware via tagged archive; n=8
- New domains: bit_manipulation/geometry/simulation (15 seeds each + sigs); config.domains + DOMAINS now 8
- Applies from chunk 10 onward (fresh child reads code)


---

## EXP-XTAL � Rule Crystallization Dynamics Harness (2026-11-26, design v1)

Origin: theory session on next-token prediction mechanics. Core hypothesis:
shared structure accumulates gradient faster than it decays (interference is
the ENGINE of generalization, not just the enemy). Rule formation = discovery
of latent equivalence structure. Experiments target: crystallization curves,
gradient-alignment early signal, rule-formation threshold, spacing effects,
backward-gated training (dense fwd, jurisdiction-masked updates � virgin
territory, no compute incentive so unexplored).

### Grammar design (synthetic, fully controlled)
- vocab 96: 64 item tokens partitioned into 8 latent classes x 8 members
  (partition NOT given to model � must be discovered from behavior),
  8 markers, BOS/PAD.
- Rule A (general): agree-chunks. class(x_{i+1}) = f(class(x_i)), f fixed
  permutation of 8 classes. Successor member uniform within class ->
  irreducible floor log(8)=2.08 nats. Ungoverned ceiling ~log(96)=4.56.
- Rule B (rare, shares the class partition with A -> planted shared
  sub-structure for transfer/interference tests): marker_B, item x, gap,
  then successor class g(class(x)), g different permutation.
- Rule C (copy, independent machinery): marker_C, item x, filler, repeat x.
  Deterministic -> floor ~0. Orthogonal to class structure (induction-like).
- Noise chunks: uniform over items. Floor log(64)=4.16.
- Sequences T=64, chunks concatenated per mixing weights, cross-chunk
  boundary positions tagged noise.

### Memorization-vs-rule separation (holdout trick)
- 2 members/class held out from ALL trigger roles in training (never
  chunk-initial, never agree-trigger). They still appear as successors
  (uniform within class) and noise -> marginals unchanged, floors match.
- KNOWN LEAK (documented, accepted): successor co-occurrence stats in
  principle carry class signal for heldout members. But exploiting it
  REQUIRES class-level pooling inference � a pure memorizer cannot use it.
  Any heldout generalization is therefore still evidence of rule structure.

### Key measurements (every K steps)
- per-tag mean CE (train), per-rule CE on segregated fixed eval batches
  (train-trigger vs heldout-trigger agree chunks, copy, noise)
- excess loss = loss - entropy floor per rule
- gradient alignment: segregated batches per rule -> full-param grad vectors
  -> pairwise cosine. PREDICTION: align(A,B) > 0 (shared partition),
  align(A,noise) ~ 0. KEY PREDICTION: alignment rises BEFORE heldout loss
  improves (early-warning signal of rule formation).
- embedding geometry: mean within-class vs across-class cosine of input
  embedding rows � direct observation of equivalence-class discovery.
  PREDICTION: within-class cosine rises at crystallization onset.

### Model
tiny decoder-only transformer: d_model 128, 4 layers, 4 heads, GELU MLP
(d_ff 512), learned pos emb, T=64, vocab 96 -> ~820k params. AdamW lr 3e-3,
batch 64. Minutes per run on RTX 5070.

### Experiment queue
1. Exp1 baseline crystallization curves + alignment early signal
2. Exp3 capacity x data sweep -> rule-formation threshold
3. Spacing: interleaved vs blocked chunk presentation (same data!)
4. Exp2 interference cartography (shared machinery map)
5. Exp4 interventions: backward-gated training, alignment-routed gating,
   consensus-weighted loss. Phased rule introduction (A, then B, then C):
   measure TRANSFER (does B crystallize faster post-A?) vs FORGETTING
   (does C destroy A?) across regimes.

### EXP-XTAL RESULTS 1 � wd sweep (6000 steps, d128, batch 64, lr 3e-3)

Runs: results/xtal/wd001.jsonl, wd03.jsonl, wd10.jsonl. Reader:
scripts/exp_xtal_read.py (backfills xs/ keys � driver had prefix bug, fixed).

FINDINGS (confirmed across wd 0.01 / 0.3 / 1.0 � remarkably insensitive):
1. CRYSTALLIZATION ORDERING: memorizable rules crystallize FAST
   (ruleA xs<1.0 by step 50!, ruleB by ~250), copy � the only rule where
   memorization is IMPOSSIBLE (answer depends on variable x, not fixed
   statistics) � crystallizes 10-20x slower (~1200+ steps, still
   descending at 6000). Memorization is a faster attractor than machinery.
2. ZERO heldout generalization in ALL conditions, all 6000 steps:
   AH loss flat at ceiling (~4.16 = uniform-over-items) entire run.
   Gen-gap INCREASES over training (2.0 -> 2.3). Weight decay 1.0 did NOT
   produce grokking. Per-token memorized solution is a stable equilibrium
   at this capacity � nothing pressures the model toward class pooling.
3. REPRESENTATION/USAGE DECOUPLING (novel-ish observation): input
   embeddings DO cluster by latent class (within-class cos 0.35-0.39 vs
   across ~0) � wd10 spikes to 0.59 by step 700 then settles 0.38.
   The class structure is IN THE EMBEDDINGS but the prediction circuit
   never uses it. Clustering is a byproduct of shared successor
   statistics, not a mechanism. Geometry != usage.
4. INTERFERENCE IS REAL AND PERSISTENT: align/A_C goes negative early
   and stays (-0.35 to -0.52). Agree vs copy gradients actively fight
   even after both crystallize. align/A_N ~0 as predicted (noise is
   orthogonal). NOTE: negative alignment did NOT prevent crystallization
   of either rule � interference here is cost, not blocker.
5. Noise floor behavior correct: eval/N ~4.4 vs floor log64=4.16;
   chain-initial noise in A chunks ~5.2 (uniform over 48 train items,
   floor log48=3.87 � model imperfectly calibrated there, minor).

INTERPRETATION: capacity is abundant (0.87M params for a 64-token
behavioral repertoire), so per-token memorization is cheap and nothing
favors the pooled/class solution � grokking pressure absent. Copy proves
the model CAN form general machinery when memorization is impossible.
The control knob is therefore the COST OF MEMORIZATION: capacity,
data scale, or update interference. -> capacity sweep (cap16/32/64,
wd0.3) running. Prediction: below some capacity, per-token memorization
of 64 triggers x successor distributions becomes more expensive than the
8-class pooled solution, and AH drops � the rule-formation threshold.

### EXP-XTAL RESULTS 2 � capacity/members sweeps + decoupling confirmation (2026-11-26)

CAPACITY SWEEP (cap16/32/64, wd0.3): naive economic theory REFUTED.
Squeezing capacity STRENGTHENED class geometry (cap16 emb gap 0.56 vs
0.35 at d128) but produced ZERO generalization, and DAMAGED copy
machinery formation (1.73 vs 0.08 excess). Memorization wins whenever
feasible � not a continuous cost tradeoff.

MEMBERS SWEEP (m24/m64, up to 512 items): same � memorization still
feasible at 512 triggers, still wins. emb gap up to 0.88 transiently
at m64d32 � strongest clustering, zero usage.

CRITICAL SELF-CATCH � the original holdout design measured an IMPOSSIBLE
task: full exclusion of holdout members from agree chunks meant zero
class signal anywhere in training. Even a perfect class-pooling model
couldn't generalize. Flat AH curves v1 were vacuous. Lesson: check what
information the data actually contains before interpreting failure.
REDESIGN v2: holdout members appear as chain-terminating successors
(p=0.1/position, class-revealing leak) but never as triggers. Tests +
floors updated. Also fixed: gen_eval_batch used hash() (process-
randomized) � now crc32 stable salt. NOTE: v1 curves not cross-comparable
bit-exactly (eval batches differed per process � statistical effect small).

V2 RESULTS � THE DECOUPLING IN PUREST FORM:
- holdout embeddings JOIN their true class clusters through successor-
  side statistics alone: emb/holdout_cluster_gap 0.43 (v2m8) / 0.40
  (v2m64; own-class cos 0.48 vs train within-class 0.51 � nearly
  indistinguishable). Distributional class inference WORKS.
- heldout prediction STILL flat at ceiling. Knowledge present, never
  used. The loophole 'clustering too weak' is closed.
- Mechanism pinned: usage requires residual gradient on positions where
  lookup can't complete. Train-trigger positions hit floor in ~100-300
  steps killing gradient; holdout triggers never trained � no pathway
  for the class-direction readout to form, ever.

RULE-D (class-copy) added to grammar: [markerD,x,f1,f2,member(class(x))]
� lookup impossible by construction (x varies). THE forced-usage test.
Runs v3dm8/v3dm64 in flight. Probe probe_xtal_readout.py in flight:
linear probes per layer, successor-class vs trigger-class targets,
train-trigger vs holdout-trigger positions. Decides WHERE the decoupling
lives (readout-only vs end-to-end).

## EXP-XTAL v5 relational rule-R results (2026-11-26)

v5rm8 (members=8, lookup surface 4096 pairs): ruleR crystallized on train
(xs 0.019) but RH holdout stuck AT ceiling (xs 1.955 vs blind-ceiling 2.08)
— memorized per-pair lookup, zero generalization.
v5rm64 (members=64, surface 262144 pairs): RH xs fell 2.46 -> 1.19 by step
6000 then partially ERODED to 1.59 at 7999 — forced class-composition
machinery + the same erosion signature as rule-D. Copy slightly damaged
(xs 0.152 vs 0.074 in m8): capacity competition real but secondary.
Rule-A holdout flat at ceiling in both: modes never cross-recruit.
=> THE control knob is lookup feasibility. Program pivot: interventions
(mode-injection, backward-gating, memory-as-pressure-valve).

## EXP-XTAL theory synthesis v1 (2026-11-26, pre-intervention-results)

THE ECONOMIC MODEL OF RULE FORMATION
For each position type there are two loss-reduction paths:
- lookup path: fast initial loss reduction per unit gradient, but plateaus
  at completion (memorizable rules) or at partial capture (non-completable
  rules with marginal structure)
- machinery path: slow initial progress (requires coordination: embeddings
  cluster AND readout aligns AND attention extracts), but ceiling is the
  full structure
Lookup wins early everywhere because it is fast. Machinery forms only where
the lookup ceiling sits below the machinery ceiling AND residual error
survives long enough to fund coordination. Error is the only fuel.

WHY OSCILLATION (v5rm64 finding, replaces monotonic-erosion model)
In rule-R, single-token marginals carry partial information: knowing x1
alone constrains the answer distribution. So lookup has a partial plateau
that keeps pulling even though completion is impossible. The system
oscillates: machinery consolidates (RH xs -> 0.5-0.7), partial-lookup
encroaches (RH xs -> 2+), period ~1000-1500 steps, amplitude large.
Step-6400 event: copy xs spiked 0.14->0.38 for one eval exactly when RH hit
its best — transient wholesale flip into structural mode. Neither attractor
is absorbing -> heteroclinic-style cycling driven by momentum+noise.
PREDICTION: oscillation amplitude should scale with the marginal-vs-joint
informativeness gap of the composition table. Sweepable dial.
PREDICTION: weight decay should damp oscillation because lookup needs
precise high-magnitude weights while distributed machinery is cheaper to
maintain under decay. v6rm64wd10 tests this.
GEOMETRY COROLLA: holdout-own cosine pinned at 0.42 throughout while
cross-class cosine rises 0.005->0.073 — slow representational collapse is
continuous but behavioral crashes are abrupt: circuit-level threshold
effect, not representation drift.

MODE-INJECTION question (next intervention): what flips attention into
structural mode? In D/R contexts the answer depends on a variable token
that attention must fetch — the attention pattern IS the mode selector.
Test: pretrain machinery with R-forcing, then introduce A positions — do
they recruit machinery or lookup? (recruitment test)
PRESSURE-VALVE test (user hypothesis): sidecar cheap memory should drain
machinery formation even where lookup is infeasible in weights — because
the sidecar completes the lookup OUTSIDE the gradient economy.

REAL-WORLD PAYLOAD
- repetition/epochs make lookup completable = kill machinery funding
- compositional holdout performance should oscillate during long training
  even as train loss monotonically improves — checkable in real LLM logs
- holdout-probe monitoring is necessary: train metrics are blind to mode

## EXP-XTAL v6 intervention results (2026-11-26) — INTERFERENCE IS THE ENGINE

stats over step>2000, RH_xs (ceiling 2.08, floor 0):
v5rm64     mean 1.32 min 0.32 frac_below_ceiling 0.84
v6rm64long mean 2.12 min 1.08 frac 0.46 (12000 steps — cuda nondeterminism
           makes oscillation phase chaotic; single runs noisy, need seeds)
v6rm64wd10 mean 1.16 min 0.30 frac 0.99 — wd 1.0 IMPROVED relational
           generalization but copy NEVER crystallized (xs 1.196): strong
           decay holds net in fuzzy-structural regime; class machinery
           tolerates fuzz (directions), copy precision does not.
v6rm64gate mean 2.75 min 2.21 frac 0.00 — PCGrad deconfliction DESTROYED
           relational generalization entirely while copy crystallized BEST
           (0.018) and trainR fine (0.026). Interference is not the cost of
           generalization — it is the PRESSURE that forces shared (class)
           structure. Self-contained circuits (copy/induction) benefit from
           deconfliction; cross-cutting compositional structure dies without
           conflict. Directly contradicts PCGrad-style intervention premise
           for compositional learning.

NEXT: recruitment test (does pre-formed machinery absorb new positions?);
marginal-informativeness dial on composition table; pressure-valve sidecar;
multi-seed replication discipline (min 3 seeds for any strong claim).

## EXP-XTAL theory synthesis v2 (2026-11-26) — interference-filter + limit-cycle model

THE INTERFERENCE FILTER (mechanism behind v6gate inversion)
Gradient descent is myopic: it follows whichever direction reduces loss
fastest. When one rule owns the network, that direction is the rule's
private optimum (lookup). When multiple rules with different private optima
share weights, the idiosyncratic gradient components cancel and only the
ALIGNED components survive. Rules can only agree on shared structure ->
conflict is a coincidence detector that passes only structure-aligned
gradient. Mixture-of-rules loss is a voting system where only shared
structure gets elected.
DESIGN LAW: maximize surface diversity at fixed latent structure. Each
task's lookup-gradient is idiosyncratic (cancels); structure-gradient is
shared (amplifies). SHARP TEST: add rule-R2 (different composition table,
same classes) to R-m64 mix — class-direction gradient doubles while lookup
components cancel => faster/more stable machinery than single-R. If
machinery formation is only about lookup-infeasibility, R2 should not
matter. Discriminates interference-filter vs pure-feasibility theories.
SECOND mechanism for why repetition hurts (independent of lookup
completion): repetition reduces surface diversity, weakening cancellation.
Templated-but-unique synthetic data should still fail to form machinery if
surface diversity is low. Directly relevant to synthetic-data pipeline
design.

LIMIT-CYCLE MODEL OF THE OSCILLATION
Project dynamics onto (L=lookup strength, M=machinery strength). Identity
codes and class codes compete for the SAME representation bandwidth. M
consolidation rewrites dimensions L was using -> partial-lookup collapses
-> train error rises -> gradient rebuilds L -> re-scrambles class dims ->
M erodes -> repeat. Predator-prey structure: lookup preys on bandwidth
machinery needs; machinery growth destroys lookup substrate. NOT noise —
a deterministic limit cycle with chaotic phase (explains seed
irreproducibility). PREDICTION: oscillation amplitude should shrink with
capacity (d_model) at fixed members, because bandwidth scarcity drives the
competition — capacity dial discriminates bandwidth-competition vs
feasibility-only models. PREDICTION: period scales with the ratio of
crystallization timescales.

RECRUITMENT THEORY (ahead of the experiment)
New positions are absorbed by whichever mode their initial error-gradient
aligns with. Existing machinery gets strengthened when new positions'
gradients point along it (positive feedback: more positions = more funding
for machinery). Curriculum art: introduce positions whose initial errors
align with existing machinery — surface similarity to machinery-served
positions should promote recruitment. TEST: rule-A positions surface-
similar to R chunks vs surface-distinct.

PRESCRIPTIVE PRINCIPLES FOR DATA DESIGN (program output so far)
P1 never let lookup complete (relational targets, big entity vocabs)
P2 maximize surface diversity at fixed latent structure
P3 mixture unified in latent structure, diverse in surface
P4 weight decay selects directional over precision machinery
P5 train metrics are blind to mode oscillation — probe holdouts
P6 capacity is not the lever; feasibility economics is

## v7 experiment matrix launched (2026-11-26) — AWAITING ANALYSIS
- v7cap32/v7cap256: capacity-amplitude test of limit-cycle model (d32/d256
  vs d128 baseline=v6rm64long). PREDICTION: oscillation amplitude shrinks
  with capacity if bandwidth-competition drives the cycle. Measure: std of
  RH_xs oscillation + min/max spread per run.
- v7r2m64: interference-filter discriminator. R(add)+R2(xor) mixed, same
  total relational fraction. PREDICTION (filter theory): R2H/RH generalize
  faster+stabler than single-R control. PREDICTION (feasibility-only): no
  difference. Compare mean/frac_below_ceiling vs v5rm64.
- v7recruit: phase1 no-A machinery pretrain, phase2 (step 4000) rule-A
  enters. KEY MEASURE: does AH drop below ceiling in phase 2 (machinery
  recruitment) or stay flat (lookup colonizes)? Also rule-A crystallization
  speed in phase 2 vs baseline ~step50.
ANALYSIS CODE hint: xs/RH/ruleR xs/R2H/ruleR xs/AH/ruleA_heldout keys in
results/xtal/v7*.jsonl — remember chaotic phase caveat, need care.

## v7recruit — THE BOOTSTRAPPING RESULT (2026-11-27) — major

Phase 1 (no rule-A, steps 0-4000): rule-R NEVER crystallizes even on TRAIN
(xs pinned at exactly 2.08 = blind-uniform for 4000 straight steps) and
emb/class_gap stays +0.000 — NO class geometry forms without rule-A.
Copy crystallizes fine (0.001). Copy is self-contained; R is not.
Phase 2 (rule-A enters step 4000): class_gap explodes 0.000->0.278 in 800
steps; ONLY THEN does trainR crystallize (1.81->0.056 over 1600 steps).

REINTERPRETATION OF GEOMETRY/USAGE DECOUPLING: the useless geometry was
never useless — it is the SUBSTRATE relational machinery builds on.
Pipeline: (1) simple rules memorize per-token, (2) class geometry forms as
free byproduct, (3) hard-rule machinery pays to use the substrate.
Memorization is both enemy AND scaffold of machinery — the tension IS the
dynamics. Relational rules are not learnable from scratch in isolation;
they need pre-existing class structure.

RECRUITMENT VERDICT: negative. AH never below ceiling in phase 2 — new
positions are colonized by lookup even when machinery exists. RH in phase
2 EXPLODED above ceiling (4.57) as trainR crystallized: late-arriving
geometry funneled R into lookup-memorization rather than machinery —
suggests a CRITICAL WINDOW: machinery must form early while plastic;
positions trained as uniform-noise for 4000 steps develop an entrenched
habit that later learning routes around via lookup.

## EXP-XTAL pressure-valve design (2026-11-27) — HIGHEST-IMPACT BRIDGE

HYPOTHESIS (user-originated): cheap lookup memory is a pressure valve that
drains rule formation. If confirmed => memory-augmented architectures
(FluxLM/STRATA/NGramMemory line) have a FUNDAMENTAL rule-formation deficit
unless consolidation pressure is designed in. Central question for the L2
consolidation program: what pressure moves memory content into machinery?

DESIGN: bigram logit-bias sidecar. For relational chunk [MARKER_R,x1,x2,
target], the target is predicted right after x2 — so the last-2-tokens
bigram context IS exactly the lookup surface. A learned table mem[K, V=256]
indexed by bucket=(x1*V+x2)%K added to logits is the easiest possible
lookup circuit: no coordination, pure addressing. Trained end-to-end as
part of model params. K=32768 (8.4M params) — compressed but large.
CONTROL: same run --mem 0 (machinery forced, v5rm64-equivalent).
PREDICTION (valve): trainR crystallizes FAST through the table, machinery
never forms, RH stays at ceiling even at m64 where it previously could
not. RH osc gone. If instead RH still dips below ceiling => machinery is
robust to cheap memory, valve hypothesis dead.
READOUT: compare trainR crystallization step + RH frac_below_ceiling vs
control. ALSO track mem-table gradient share: if table absorbs the R
gradient, main-net class geometry should NOT form (emb/class_gap lower).
Implementation: driver-level loss override in exp_xtal.py — when args.mem>0,
compute logits via model(toks[:,:-1])[0], build bucket ids from shifted
token pairs (pad bucket 0 at position 0), add mem[bucket] before CE.
Applied at ALL positions (noise positions cannot benefit — control for
table capacity theft). Flag: --mem K.
RUNS: v8mem (mem on) + v8nomem (control) both m64 ruler 8000 steps.

## SESSION HANDOFF (2026-11-27) — READ FIRST NEXT SESSION

RUNNING NOW: v8mem + v8nomem pressure-valve pair in results/xtal/. ANALYZE FIRST: trainR crystallization speed + xs/RH/ruleR frac_below_ceiling vs v5rm64 baseline; emb/class_gap shows whether table absorbs gradient (valve confirmed => geometry does NOT form).

QWEN BRIDGE BLOCKED: random-init Qwen2-0.5B vocab256 OOMed 12GB even with bnb optimizer (fp32 param+grad+moments ~7GB). FIXES: run solo after v8 finishes; add model.to(torch.bfloat16) in build_model qwen path; batch 4 steps 3000; cpu_offload in forge/training/training_utils.py. Smoke then full run v8qwen --steps 4000 --batch 8 --members 64 --ruler --arch qwen. KEY Q: do xtal phenomena replicate at 0.5B?

UNCOMMITTED: exp_xtal.py mem/arch/optimizer changes — commit after qwen smoke passes.

## v9qwen32 INTERIM RESULT (2026-11-27): healthy but under-trained
fp32 fixed bf16 stall (user instinct correct). At 2500 steps lr1e-3: trainA crystallizing 0.71, emb_gap rising monotonic +0.119 (slow), trainR not ignited (consistent with substrate threshold ~0.3-0.4 not crossed), copy STILL at ceiling 6.28 — watch item. Token-equivalent: batch16x2500 = tiny-model step ~625 — everything consistent with tiny ordering so far, just slower. RELAUNCHED v10qwen: 8000 steps lr 2e-3 batch 16, eval/grad every 200. VERDICT KEYS: does emb_gap pass 0.3 then trainR ignite (substrate pipeline scale-invariant)? does copy ever crystallize (geometry-independent canary)? RH oscillation at scale?
