# Rocinante-X-12B â†’ ForgeAI Port â€” Working Plan

## Goal
Port `D:/LMstudio/Models/TheDrummer/Rocinante-X-12B-v1-GGUF/Rocinante-X-12B-v1b-Q4_K_M.gguf`
(Mistral-Nemo arch: llama gguf arch, vocab 131072, d 5120, L40, 32Q/8KV, head_dim 128,
ffn 14336, theta 1e6, eps 1e-5, untied head, tekken tokenizer, ctx 131072) into ForgeAI:

- RP quality â‰ˆ original (verified vs llama_cpp logits â€” ground truth)
- Speed/size â†’ LFM2.5-class experience: packed int4 weights, compile, bounded decode cost
- Context-size independence (FluxLM tech): O(1) VRAM + O(1) RAM vs context length

## Key facts discovered
- `_map_gguf_to_forge` exists (engine_common.py:244) â€” name map only, NO rope un-permute
- `_load_gguf_checkpoint` (engine_checkpoints.py:815) is buggy stub: rope_base=10000 hardcoded,
  no head_dim, no q/k un-permute, fp16 full-load (OOM 12B), hand-dequant Q4_K "simplified" (wrong)
- HF-name checkpoints auto-remap via `ModelLoader._remap_hf_keys` (loader.py:346) â€” used by
  qwen/llama presets. So convert GGUF â†’ HF-style safetensors once = canonical path.
- RoPE in layers.py:50 = HF rotate_half convention â†’ GGUF interleaved q/k rows need
  inverse permute: w.reshape(h,2,hd//2,d).transpose(1,2).reshape
- `compact_past_kv` bridge (engine_generation._maybe_compact_past_kv :989) fires in
  _prefill/_decode_tokens/_chunked_prefill â€” NOT in StandardDecoding.generate loop.
  â†’ attach via `model._forge_compact_hook`, 2-line hook in StandardDecoding.
- logits_processor + step_observer hooks exist in _decode_tokens (:1165, :1181) â€” sidecar
  lift/observe plug points; same model._forge_suffix_mem attribute read in StandardDecoding.
- BeaconKVCache (kv/beacon_kv.py) already = sink + window + scored-middle compact_past_kv.
- Tokenizer: no mistral_common. llama_cpp 0.3.34 vocab_only=True = exact tekken (0.6s load).
- Decoding zoo: ngram_speculative/uno (prompt-lookup drafts) exist; StandardDecoding default.
- quantize_model_w4 = torchao Int4WeightOnly g128 packed (stays packed, fused mm).
  embed+head NOT targeted (not nn.Linear-targeted names) â€” stays bf16 (2.7GB).

## Design decisions
- Preset `rocinante_12b` (root): vocab 131072, d 5120, L40, 32/8, head_dim=128,
  inter 14336, eps 1e-5, theta 1e6, untied, max_seq_len 131072.
- Converter: gguf pkg dequant â†’ bf16 HF-name safetensors (`research/checkpoints/rocinante_x_12b/`).
  Converter ALSO emits the tokenizer? No â€” tokenizer = llama_cpp vocab_only wrapper
  `GGUFTekkenTokenizer` in forge_loader.py; `from_checkpoint(tokenizer_path="gguf:<path>")`.
- Context independence =
    a) `flux` KV strategy = BeaconKVCache subclass, sink=8 window=2048 budget=6144
       (O(1) VRAM; compressed RoPE positions stay in-range forever â€” StreamingLLM semantics)
    b) `SuffixMemory` (forge/model/ngram_mem.py extension): bounded sparse
       {orderâ†’{suffix_hashâ†’{tok:count}}} + decay halving + observe()+lift_last()+draft()
       â€” verbatim recall over whole stream even after KV eviction (the FluxLM property).
    c) hook into StandardDecoding + _decode_tokens via model._forge_suffix_mem.
- Speed: load_quantize="int4" (packed W4G128 â‰ˆ 5.9GB) + optional int8 head;
  activate(kv_cache="flux", use_compile=T, use_fused_qk_norm_rope_cache=T,
  use_chunked_prefill=T) + SuffixMemory(gate tuned ~0.5-1.0).
- Honest note: 12B dense can't literally reach LFM2.5-1.2B file size; target ~6-7GB
  resident + flat memory vs context + fastest decode w/o training.

## Files
1. forge/config.py â€” +rocinante_12b preset
2. scripts/convert_gguf_forge.py â€” NEW converter (gguf pkg, unpermute, HF names, safetensors)
3. forge/engine/forge_loader.py â€” +GGUFTekkenTokenizer (llama_cpp vocab_only adapter)
4. forge/model/ngram_mem.py â€” +SuffixMemory (bounded sparse, observe/lift_last/draft)
5. forge/engine/kv/flux_kv.py â€” NEW FluxKVCache(BeaconKVCache) tuned + register "flux"
6. forge/engine/engine_activation.py â€” set model._forge_compact_hook after kv build;
   "flux" in _KV_FALLBACK_CHAIN; kv_cache ctor kwargs passthrough for flux
7. forge/engine/engine_common.py â€” SamplingParams.suffix_gate
8. forge/engine/engine_generation.py â€” enable_suffix_memory API; _decode_tokens hooks;
   from_checkpoint gguf-tokenizer dispatch (engine_checkpoints.py:88 area)
9. forge/engine/decoding.py â€” StandardDecoding compact+suffix hooks
10. tests/unit/test_rocinante_port.py â€” permute, suffix mem, flux kv, preset
11. scripts/bench_rocinante.py â€” parity vs llama_cpp, tok/s, VRAM, long-ctx probe
12. AGENTS.md â€” document R65 port

## VRAM budget (RTX 5070 12GB)
weights int4 projs â‰ˆ5.9GB + bf16 embed/head 2.7GB = 8.6GB
KV flux budget 6144tok Ã— 40L Ã— 8kv Ã— 128hd Ã— 2 (k+v) Ã— 2B = 1.0GB
sidecar ~50MB host RAM (bounded), activations ~0.5GB â†’ ~10.3GB peak. Fits.

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

LOOP (E-loop): recurrent-depth over last 3 blocks. Zero-init gates = lossless (tested bit-exact; bug found: gate was wrongly applied to pass 0, deleting blocks). Untrained looping DEGRADES NLL: 1it=2.99, 2it=3.12, 4it=3.93, 8it=5.82 — deltas decay but plateau ~0.049 (no fixed point). Gate-trained (8.5s): 3.0352 vs 3.1178 ungated @2it. Decode ~21ms/tok either way (loop region small). Takeaway: looped-transformer needs training, not free compute.

NCM (E-ncm): CfC cells on last 4 blocks, lossless attach (delta=0.0, 1-line dtype fix: .to(p.dtype)). Trains to loss 0.004 on 4 docs but eval NLL +4.39 — catastrophic overfit. Timescales stay ~1.0. Takeaway: side-path overfits tiny data; needs regularization/data scale.

DIFFUSION+MEDUSA (E-diffusion/E-speed): forge/exp/diffdraft.py — DDIM denoiser proposes K=4 tokens in parallel, AR model spec-verifies (greedy-faithful: KV truncate to ctx+accepted, corrected token + fixup forward). Diffusion head: acc 27.6% per slot after 300 steps, accept_rate 3-8%, tok/fwd 0.53-0.59, fidelity match mostly True (1 bf16 near-tie flip /96 tok — inherent to batched verify). MedusaDraftHead (K delta-MLPs + shared frozen lm_head, same verify): acc->1.0 (overfit 6 docs), accept 15-28%, tok/fwd 0.65-0.75, match=True both. Speed numbers noisy (greedy measured 8.4-57 depending on run state). Takeaway: mechanism correct; draft quality is everything — medusa >> diffusion at this scale, still net-negative speed on frozen small model.

DYNSCALE (E-dynscale, subagent): HeadGate on out_proj input (B,T,H,Dh), EMA saliency, lossless tau=0. Free heads: 1.5% @dNLL-0.003, 4.5% @+0.13, 15% @+2.3, ~50% fatal. Live gating costs more than it saves: 57.8-71.8 tok/s vs 86 baseline (EMA tolist sync). EarlyExit: 4.5 depth=288 tok/s @dNLL+13; conservative exits slower than full depth (per-block check ~1ms over V=152k). DiagTempScaler (Adam not AdamW — decay killed gains): recovers +0.139 NLL post-gating. Takeaway: Qwen-0.5B attention heads have ~15% slack at NLL cost, but Python-side gating overhead kills the win — needs fused/kernel-level gating.

ABSHEAD (E-abshead, subagent): LatentThoughtHead predicts h_{t+4} — cos 0.668 vs 0.616 persistence (held-out, real signal). Zero-init gate lossless. Injecting as residual readout hint: gate->0.074, dNLL -0.0004 (wash); joint train +0.163 (overfit hurts). Takeaway: future-latent is learnable but frozen lm_head can't exploit it.

NDP (E-ndp): forge/exp/ndp.py — exact online-softmax split attention: host computes (m,s,acc) partials on far KV pages, GPU combines. Exact to bf16 (err 5e-4). Transfer/step: 3.7KB vs 2-8MB migrate (567-2270x less). BUT: wins vs migration only at T_far=4k (3.0 vs 4.0ms); loses at 16k (20.3 vs 10.7ms — CPU attention slower than PCIe). Takeaway: NDP break-even where host compute = transfer time; practical design = retrieve-subset + NDP on selected pages.

DECISION (E-decision, subagent): embedded decision head inside model — acc 0.54, controller works, ~14% overhead.

PREVTOK (E-prevtok): head training broken (acc 0.0, loss ~1e5 — likely target indexing/vocab bug); 544MB head = untied full-vocab linear, should tie to embed. Subagent fixing.

### Cross-cutting findings
- Side-path convention (zero-init gate + cached gate-zero check) WORKS: every attach verified lossless delta=0.0.
- bf16 model + fp32 aux modules = dtype traps everywhere (mse_loss backward rejects mixed inputs; Linear layers too). Pattern: aux modules fp32 internally, cast at boundary (CastTitan shim) or create with p.dtype.
- Frozen-model experiments that TRAIN auxiliary params on ~6 docs: massive overfit is the norm (ncm loss->0.004, medusa acc->1.0). Held-out eval mandatory; NCM/medusa need 10-100x more data.
- Engine tokenizer: ids are int32 — CE needs .long().
- NGramMemory dense rows (594KB/suffix @151936) + deep-copy snapshots = OOM at ~3GB; use snapshot-free eval (eval forwards don't mutate live state).

### Remaining
- E-cxl tiered KV pool (subagent running)
- E-prevtok fix (subagent running)
- ternary_ptq collect_activations bug (unfixed, fallback works)
- batched|snapkv 21.6 tok/s anomaly
- Qwen2.5-Coder-0.5B validation pass

PREVTOK (E-prevtok, fixed + measured): PrevTokenHead = tied-embedding backward readout (adapter(h)@E^T, ~1.4M params; replaces broken 544MB untied head + bnb-AdamW8bit that diverged). Trained 400 steps: held-out prev-NLL 8.354 (uniform=11.93), top1 19.8%. Key finding: greedy generated text is backward-INCONSISTENT (prev-NLL 9.11 vs 8.35 on real text). SelfCorrectingDecoder modes: threshold tau=0.01/0.05/0.15 corrects 63-79% of tokens (joint dNLL -2.4/-1.9/-1.6 but +0.65-0.8 forwards/tok); RERANK (batched top-K retry in ONE forward) = best: joint dNLL -3.26 at forwards/tok 1.016 — essentially free self-correction. Rollback thrashes (head too weak on gen text). Retry must mask BOTH failed argmax AND prev_tok (x_t==x_{t-1} trivially passes backward check — degenerate loophole).

CXL (E-cxl, subagent): forge/exp/cxlmem.py — TieredKV rolling GPU hot window (1024 tok) + pinned-CPU cold pool, CxLLink side-stream nonblocking copies, online-softmax streaming attend, persistent prefetch staging; plus WeightTier (FFN offload). Real Qwen decode ctx=8K: full-GPU 57.8 tok/s vs fetch-all 50.3 (-13%), stream 22.5 (-61%), prefetch 27.5 (-52%, H2D dedup -93%), +8-FFN-offload 25.4 frees 209MB. KV GPU-resident FLAT 12.6MB at any ctx (8x saving @8K, 32x @32K); page-miss 0.028ms launch-bound. Bit-exact argmax vs baseline. Found+fixed overlapping in-place CUDA tier-shift UB in production cpu_kv_offload.py too. Honest caveat: simulated tier over PCIe; streaming hurts throughput (-52%) — the win is capacity, not speed.

Qwen2.5-Coder-0.5B: snapshot was an empty stub (.no_exist markers) — engine silently built ALL-RANDOM weights (NLL 11.94, second silent-fallback trap after the tokenizer). Downloaded real weights (HF_HUB_DISABLE_SYMLINKS=1 on Windows); NLL(code)=0.617, fib generation coherent. TODO later: engine should refuse load when snapshot dir has no safetensors.

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

## Teacher channel ("tch") — hybrid Qwen floor + Flux recall (2026-10-2x)

Architecture: Qwen2.5-0.5B fp16 as a STATIC read-only Hedge channel.
- No training, nothing journaled (state derives from the token stream,
  which IS journaled). Lazy: _tch() builds on first query; sync_to
  feeds ring tokens the teacher missed (incremental KV; rebuild tail
  after revert/load/ring-overflow).
- Predict path: _tch_lift adds w_rel*(log p_tch - log p_uni), clipped
  4*lift_clip — same additive contract as gsyn/bank lifts; device-only
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
eval-time learn feeds 1 HF fwd/token (~30-50ms) — teacher decode is
~20-30 tok/s vs flux-only ~200-990. Disable tch lift for fast gen.

Open: hybrid NLL on docs190k running; teacher-only baseline ~= Qwen
4.30 (known); flux-only 6.69 (known). Success = hybrid < 5.0.

## Teacher channel — RESULTS (same-session followup)

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
