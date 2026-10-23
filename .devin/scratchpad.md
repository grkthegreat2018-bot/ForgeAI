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

## 2026-10-20 — wt15m flux-only baseline + self-play wiring

### wt15m+docs (FLUX-ONLY — snapshot had teacher_channel=False)
- Run loaded a stale flux-only snapshot (n=-1 bug); evaluated honest flux-only@15M:
  held_wt NLL 7.2258 / top1 0.1238 | held_docs NLL 8.3676 / top1 0.1371
  vram 7.03GB, ingest 457s (13.77M tok ˜ 30K tok/s)
- Flux@15M on wikitext is WORSE than docs190k 6.82 — wikitext is
  hapax-heavy; bloom_admit=2 + bank spilling lose one-shot contexts.
- THE generalization gap (7.2 vs Qwen 4.3) is exactly what tch covers —
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
  dir containing "secrets" (pytest test dirs!) — now exact-component.
- Still missing vs spec: model self-prompting (AZR proposal machinery
  is transformer-only), write-to-sandbox file editing, stop-on-
  advancement, execution sandbox for model code.

### Self-speculative decode (commit 37b6fb0)
- generate(speculate=K): read-only draft rollout (_draft_tokens —
  virtual suffix hashes over order cells, deepest-hit argmax
  follower), learn row via ONE _fused_seq, longest verified prefix,
  revert+relearn tail. GPU+learn+argmax only.
- Bit-exactness VERIFIED on GPU: speculate=6 == sequential argmax
  (4 tests green). stream len exact (prompt+emitted, no orphan
  learns).
- Notable: bc blocks are all-or-nothing so rejected tails revert
  whole row + relearn accepted prefix (extra pass, bounded).
- Budget cap: row len <= max_new-_step prevents output overshoot.
- Throughput estimate ~aK+1 tokens/pass vs 1 — measures vs GPU when
  15M run frees the card.

### Sandbox-jailed write tool (commit 60955d3)
- `write <relpath>::<text>` appends under flux_sandbox_dir
  (status_dir/flux_sandbox) — resolved path must stay inside jail
  (absolute/.. rejected, deny components blocked).  Closes the
  read-anything/write-sandbox contract for code experiments.
- Remaining spec gaps (documented in AGENTS.md): code-execution
  sandbox for model code, stop-on-proven-advancement sentinel.
- NOTE: T:-phase budget is 14 tokens — long write calls truncate;
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

# 2026-10-21 — Technique Landscape Scan (user-requested research sweep)

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
- RWKV-7 (rd_round_40/49 plan candidate — NO key exists in forge/keys/attention/)
- JEPA/LLM-JEPA (rd_round_40 refs list)

UNTOUCHED: everything below.

## Research findings (3-agent web sweep)

### PEFT / adapter cluster
- BitLoRA: no arXiv. ESWA 311 (2026) — BitNet-1.58b base + FP16 LoRA adapters,
  federated on-device; >85% mem cut. bitnet_lora.py already covers the idea.
- Aletheia LoRA (arXiv 2604.15351, Apr 2026, single-author, 0 citations):
  5-pass gradient probe ranks layers -> LoRA on top-50% only, asymmetric rank.
  15-28% train speedup. Cheap add-on; Mamba-layer probing untested.
- JIT-LoRA (NOT arXiv — self-pub, HF Ex0bit/jit-lora, MLX-only): online LoRA
  backprop after EVERY response, mid-conversation knowledge injection.
  58.1% fact recall @ Qwen3.5-2B, 180 steps ~= 70s M4 Max, needs 5e-4 LR +
  >=33% replay ratio. STRONG overlap w/ LiveT — same online-learning goal,
  different surface (LoRA weights vs journaled states).
- MiCA (arXiv 2604.01694, in HF PEFT as init_lora_weights="mica"):
  Minor-Component Adaptation — SVD W=USV^T, fix B to LAST-r singular vectors,
  train only A. Inverse of PiSSA (already impl). 5.9x better knowledge
  acquisition vs LoRA, 6-60% params. Trivial to add to adapter_variants.py.
- LIFT (arXiv 2506.00772, ICML 2025): sparse FT of top-5% magnitude
  "Principal Weights", no low-rank. Beats full FT on reasoning, +20% source
  retention, tiny optimizer state. Weight-space -> arch-agnostic.
- REAP (arXiv 2510.13999, ICLR 2026): one-shot MoE expert pruning, saliency =
  gate-value x activation-norm. Near-lossless @50% prune (Qwen3-Coder-480B).
  Applies to forge/moe/ — saves weight mem not KV.
- AoE (arXiv 2501.13074, ICML 2025): Autonomy-of-Experts — delete router;
  experts self-rank via low-rank pre-activation norms. Pretrain-time only.
- Per-Layer Embeddings (PLE): no standalone paper; Gemma 3n/4 technique.
  Per-layer 256-dim tables kept in host RAM, fetched per layer -> E2B 5.1B raw
  runs at ~1.9B effective. Train-from-scratch only; fits mixed CPU/GPU mandate.

### Architecture cluster
- SparseStack = WINA (arXiv 2505.19427, ICLR'26; SparseStack = internal name,
  same authors — unverified equivalence). Training-free activation sparsity:
  hidden-state mag x weight-col l2-norm gating. +2.94% avg over TEAL at equal
  sparsity. Needs sparse kernels for real speedup.
- LaminarNet (Zenodo Mar 2026, single-author, NO peer review): O(N) "Geometric
  Drift Field" + Cross-Stratum Routing multi-rate strata. 49M claims -38% ppl.
- R-CLA (arXiv 2604.22782, Apple, ICLR'26 TTU): random cross-layer KV during
  training -> robust to arbitrary depth-wise KV sharing at deploy. CLA base =
  2x KV cut vs MQA (2405.12981). Train-time; halves KV cache.
- MODUS (arXiv 2607.25948): 15B decoder-only any-to-any multimodal — SKIP.
- AION-1 (arXiv 2510.17960): astronomy omnimodal encoder — domain mismatch.
- TRM (arXiv 2510.04871, Samsung): single 2-layer net, recursive z/y refinement
  <=16 steps. 7M params -> 45% ARC-AGI-1, 8% ARC-AGI-2, beats o3-mini. Trains
  from ~1k examples. Trivially fits 12GB.
- OpenMythos (github kyegomez, no paper): community recurrent-depth transformer
  (prelude -> looped block -> coda, MLA<->GQA, sparse MoE FFN). Unproven; the
  recurrent-depth pattern itself is the same family as TRM.
- RWKV-8 ROSA (no arXiv; RWKV-LM RWKV-v8/): Rapid Online Suffix Automaton —
  PARAMETER-FREE neurosymbolic exact-match propagator. O(n) online suffix
  automaton predicts x[j+1] from longest suffix match. 100% MQAR/NIAH by
  construction; demos: 40K params 99.8% digit-reversal. DIRECT analog of
  ForgeAI n-gram suffix-hash sidecars — ROSA is the principled version.
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
  free unbounded); (b) BD3-LM arXiv 2503.09573 ICLR'25 Oral — block-diffusion
  LM, arbitrary length + KV cache + parallel sampling. Our Uno/diffdraft are
  diffusion-ASSISTED AR; BD3-LM is true block-diffusion LM.
- LLM-JEPA (arXiv 2509.14252, ICLR'26): aux loss predicting EMA target-encoder
  embed of an alternate view. +8-15pp GSM8K/TURK on Llama3-8B; ~2x fwd cost.
  DLLM-JEPA (2606.00091) for masked-diffusion LMs; ER-JEPA (2609.36952).
- Spiking transformers: Spikformer/Spike-driven (42-87x attn energy on neuro
  HW); SpikingBrain-7B (2509.05276) >100x TTFT @4M ctx, 69% spike sparsity.
  GPU-impractical (timestep unrolling); watch only.
- "Mojo beats CUDA": narrowly true — vendor-tuned B200 bf16 matmul 1772 vs
  cuBLAS 1763 TFLOPs; independent SC'25 shows ~87% of CUDA on stencil. NO
  Windows support (WSL only) — SKIP for this box.
- TurboQuant+ (community fork TheTom/turboquant_plus, not official):
  (i) V compresses nearly free — all loss is in K -> asymmetric K/V bits;
  (ii) protect first/last 2 layers recovers 37-91% gap;
  (iii) sparse-V dequant skip where attn<1e-6 -> +22.8% decode @32K, zero ppl
  delta, NIAH 9/9. DIRECT add-on to LloydMaxRotatedKV/rotorquant.
- Decentralized pretraining: INTELLECT-2 (2505.07291, 32B RL, TOPLOC verify,
  ~590Mb/s broadcast); Psyche/Consilience-40B (DisTrO/DeMo, Solana coord);
  DiLoCo = 500x less comm, ~14Gb/s for 8B. Hobbyist = contributor only.

## Highest-synergy picks for ForgeAI (ranked)
1. ROSA-style layer — we already built the machinery (ngram suffix-hash
   sidecar, FLUX cells). ROSA = exact-suffix-automaton channel; could be a
   new FLUX channel or an LLM-side near-free retrieval layer.
2. TurboQuant+ sparse-V dequant — drop-in on existing KV quant; +22.8% decode.
3. MiCA — PiSSAInitializer already does SVD; minor-subspace variant is ~30
   lines; PEFT shows 5.9x knowledge acquisition (relevant to proposer SFT).
4. Packed-seq metadata caching — check varlen path for per-layer cu_seqlens
   rebuilds; free ~14% SFT batch speedup.
5. TRM / recurrent-depth — separate small-net reasoning direction; complements
   the model-versioning chain only as an exp/, not a key.
6. REAP — applies iff forge/moe/ ships a trained MoE; one-shot 50% prune.
7. LIFT — sparse top-5% FT, tiny optimizer state; good paired w/ LiveT.
8. R-CLA — needs (continued) training but halves KV; candidate for a key.
9. LLM-JEPA — aux loss, ~2x fwd; plausible SFT/CPT add-on.
10. JIT-LoRA concepts — LiveT already does online learning better (journaled,
    exact-revert); the 5e-4 LR + 33% replay ratio recipe is worth noting for
    any future LoRA-surface live learning.
SKIP: MODUS (15B multimodal), AION (astro), Mojo (no Windows), spiking (needs
neuromorphic HW), ESP32 (<=30M deploy-only), decentralized pretraining
(contributor-only), OpenMythos (unproven code, but recurrent-depth motif
worth tracking), Aletheia (weak evidence, though cheap).

## Deep-dive details (2026-10-21, 3-agent second pass)

### PEFT cluster — algorithms
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
  mean — decouples firing freq). One-shot forward-only calibration ~1k samples.
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
  HF issue#45206 says x hidden^-0.5, vLLM uses hidden^+0.5 — check HF source.

### Architecture cluster — algorithms
- ROSA (RWKV-v8, no paper): online suffix automaton per discretized channel
  stream; y[i]=x[e+1] of longest prior suffix match. Demos: Emb_ROSA (embed
  match token), ROSA_1bit (C binary automata/layer, bit-flip-rescore
  pseudo-grad backward, slow), ROSA-QKV (SAM over k-stream, q walks suffix
  links, v retrieved at match end+1, learned scale e), ROSA_4bit (4ch->hex,
  C/4 automata, loss .25 vs .4). All impls pure-Python CPU — no official GPU
  kernel; wind_rosa/RASP community ports. ForgeAI fit: additive side-path key
  (gate=0 lossless) OR cheaper: NGramMemory IS the token-level analog — ROSA
  adds per-layer bit-stream retrieval. Sam state = dict, same per-seq state
  invariant as conv_state (R60-4).
- TRM: one 2L net, z<-f(x,y,z) x n=6 then y<-f(y,z); T=3 cycles, first T-1
  no_grad; deep-supervision + detach between sup steps (N_sup=16); halt=BCE
  scalar; EMA .999 (+7.5pts). 5M -> 87.4 Sudoku vs HRM-27M 55. CAVEAT
  2512.11847: ARC score collapses to 0% w/o per-puzzle-ID embed — transductive
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
  (L'=4..16) vs AR 14.1 — best diffusion LM, still ~2x worse than AR. SSM
  can't denoise bidirectionally — Jamba variant = block-causal SSM + block-
  diag attn mask; pretrain-scale change.
- LaminarNet (Zenodo, unvetted): GDF = input-dependent unit-modulus rotation
  SSM (LRU-family, theta=tanh(Linear(x)), chunked cumsum-of-angles scan);
  v0.6.4 adds learned forget gates (not purely unit-modulus); CSR = parallel
  strata at rates (1,2,4), AvgPool down / nearest-up + dual gates, iterated
  per block. Claim -38.3% ppl @49M = ppl 44.2 vs unpublished baseline ~71 —
  plausible but unverified. Fit: "gdf" ltype/key ~1 file; CSR ~ attn_residual
  at multi-res. NOTE: forge/model/strata/ name collision — unrelated.

### Systems cluster — algorithms
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
  _varlen_meta (weakref+_version keyed) in attention_ops.py — flash path now
  reuses cu_i32+max_seqlen; fallback reuses blocks for mask build. Tests in
  test_rd14_speedup.py::TestVarlenMetaCache (11 pass). Left-pad gotcha:
  position_ids-derived cu mis-segments — we use explicit cu_seqlens already.
- WINA (2505.19427 -> ICLR'26, microsoft/wina): gate |x_i|*||W[:,i]||2 topk
  per matrix; col norms precomputed once; uniform-K mode zero calibration.
  ~60-64% FLOP cut @65% sparsity; REAL speedup ~1.2-1.9x decode (bandwidth-
  bound), degrades at batch>=64. Triton GEMV in repo. Pairs w/ weight quant
  (skip dequant of unselected cols). CUDA-graph safe (static topk shape).
- LLM-JEPA: aux loss on paired views (NL<->regex, q<->SQL, problem<->soln);
  single shared encoder, last-token hidden = embed, NO EMA (conflicting
  secondary claim), predictor = special tokens in packed context, cosine
  dist, ~2x fwd. Needs paired-view data — no general text augmentation.
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
  arch = linear-state + sliding-window KV strategies — both graph-friendly.

## Ranked candidates updated post-deep-dive
1. TurboQuant+ sparse-V + boundary-V + asymmetric K/V: drop-in on existing
   KVCacheStrategy; concrete rules verified; vLLM upstream merged = de-risked.
2. Packed-meta caching: DONE (this session, tests pass).
3. MiCA: ~20 lines via PiSSAInitializer; knowledge-injection SFT wins.
4. WINA uniform-K: topk(|x|*colnorm) gating on FFN GEMV; decode-only win,
   pairs w/ existing quant paths; needs sparse-gather GEMV to realize speedup.
5. ROSA-as-key: ngram_mem already does token-level; ROSA adds per-layer bit
   streams — CPU automata heavy; start as FLUX channel or 1-2 layer side-path.
6. REAP: one-shot prune for forge/moe — only matters once a trained MoE ckpt
   exists; llm-compressor impl reference.
7. TRM/hyperloop: recurrent-depth reasoning as exp/ model — bounded scope.
8. R-CLA: halve KV via train-time stochastic sharing — candidate for next
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
- `forge/evolution/eval_pool.py` â€” EvalPool: spawn-context persistent workers,
  per-worker 1-deep job queues, shared result queue, deadline kill+respawn,
  max_jobs self-recycle. `submit` nonblocking (None=saturated), `poll`,
  `eval_sync`, `eval_many` (parallel verify reps). CPU target -> 16 workers,
  CUDA target -> 4 (each torch-using worker ~1.5GB ctx).
- `Mutator.mutants_for(src, insp, max_family, seed_base)` â€” deterministic
  opÃ—site enumeration. Per-site ops: literal.0-7 (site index into eligible
  float literals), funcify.{0,1}.{0-7} (direction Ã— call site), masked_where.i,
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
- `scripts/rsi_evolve.py` â€” RSI campaign: round = all 7 LLM targets Ã—
  basin proposer â†’ export_sft â†’ finetune_proposer --mode lora (adapters
  + router) â†’ next round hot-swaps them. --no-train, --min-new,
  --quiet-rounds, --minutes, --status .events.jsonl.

### Bugs found while building
- spawn workers re-import __main__ â†’ test scripts need __main__ guard;
  fn_evolve.py/pytest already guarded.
- `_respawn` ValueError when worker replaced twice inside one deadline pass.
- Thin seeds (dry_penalties=0 variants, fast_softmax=2) have genuinely small
  enumerable space â†’ mutator-only runs stall correctly; LLM basins are the
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


## Promotion ratchet — root cause + fix (session cont.)
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
  honestly (0 promos) vs prior fake ratchet — correct behavior on converged
  targets.

---

# 2026-10-22 — ENGINE_CRITIQUE_DB implementation pass + ranked research landed

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

- `send_task_message` per-task serialization used `asyncio.Lock` — waiters
  on a DIFFERENT event loop (starlette TestClient spawns a fresh portal
  loop per request) were stranded forever: `Lock.release()` wakes waiters
  via `call_soon`, which is not thread-safe and never wakes a foreign
  loop. Flaky ~80% of concurrent-post attempts; faulthandler showed the
  losing POST parked in `portal.call` with lock free + queue empty.
  Fix: `_task_locks` are `threading.Lock` now, contended acquire offloaded
  to `_task_lock_pool` (dedicated ThreadPoolExecutor) so waiters never
  block the loop or starve the default executor; cancellation rescue
  releases a lock that a parked worker wins after the awaiter is gone.
  Repro: `.devin/repro_task_race.py` — 0/15 ? 40/40 OK.
  Lesson: ANY asyncio primitive shared across loops is a landmine —
  FastAPI/TestClient/multi-portal hosts guarantee nothing about loop
  identity.

## Ranked research — done

- MiCA (rank 3): `MiCAInitializer` + `MiCALinear` in
  `forge/training/adapter_variants.py`. B = U[:, -r:] minor subspace,
  frozen as a BUFFER (not Parameter — can't leak into optimizer groups);
  A zero-init trainable; merge() folds back. 7 tests in
  test_r38_adapter_variants.py (37 total pass).
- TurboQuant+ (rank 1): `RotorQuantCache` gained `bits_v` (asymmetric
  K/V — K precision dominates), `layer_idx`/`n_layers`/`boundary_v_bits`
  (boundary-V for first/last 2 layers when instantiated per-layer), and
  `get_sparse_v(attn_weights, threshold=1e-6)` which genuinely SKIPS
  dequantizing V rows under threshold (returns zero rows ? attn@V
  unchanged within threshold·?v?). `RotorQuantKVCache` ctor takes
  bits/bits_v/boundary args; `build_kv_cache(**kwargs)` forwards ctor
  kwargs (unknown kwarg ? TypeError, no silent drop); ActivationConfig
  gains `kv_v_bits`/`kv_boundary_v_bits`; `_activate_kv_cache` finally
  passes `kv_bits` to rotorquant/hadamard_int4 (was a dead flag — the
  wrapper hardcoded bits=4). Also repaired latent interface bugs the
  wrapper had: `RotorQuantCache.append(k,v)` takes no position arg,
  `get()` no positions arg, and there was no `clear()` — the "rotorquant"
  strategy would have TypeError'd on first use (another dead-feature).
  tests/unit/test_turboquant_plus.py: 10/10 pass.
- Packed-meta caching: already DONE prior session (_varlen_meta).

## Not done / remaining

- EC-152 endpoint gaps: only subset landed (/version, /metrics, tokenize,
  detokenize, request-id, stream_options usage chunk). Remaining: /props,
  /apply-template, abort/drain, n>1 shared-prefill, penalties, logit_bias,
  response_format, backpressure, JIT load+TTL, GGUF hub pull, etc. — left
  `open`.
- EC-140 per-token host syncs: partially addressed (cumulative decode in
  generate_stream); remaining .item()/.tolist() syncs elsewhere — left
  open.
- ROSA (rank 5): NGramMemory already token-level analog; ROSA adds
  per-layer bit-stream automata — heavier, deferred.
- Fused dequant-GEMM (EC-145) still open — needs Triton/CUDA kernel work.
- Full pytest: 687+ pass; one flaky ordering test
  (test_evolution_frontier::test_no_duplicate_evals) passes standalone —
  shared-state pollution under load, not a regression from this pass.

## Verification anchors

- tests/unit/test_forge_server_fixes.py 23/23 (incl. concurrent POST
  ordering — repro script 40/40).
- tests/unit/test_tools_security_fixes.py 33/33;
  test_gui_hardening_ec.py 21/21.
- tests/unit/test_forge_engine_fixes.py + r39/r60 decoding: 93 pass.
- test_turboquant_plus.py 10/10; test_r38_adapter_variants.py 37/37.
