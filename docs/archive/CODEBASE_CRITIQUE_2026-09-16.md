# ForgeAI Critique — 2026-09-16 (Performance, Memory & Adapter-Portability Audit)

Scope: this is the third critique, but the first aimed specifically at the
serving/runtime surface rather than code hygiene. Prior docs
(`CODEBASE_CRITIQUE_2026-09-09.md`, `_2026-09-10_followup.md`) covered god
objects, lint, tests, commit hygiene, and the custom-LoRA-vs-PEFT question.
This audit covers what they did not:

1. ForgeEngine model **loading and holding** efficiency (speed + memory)
2. ForgeLM V2 **architecture** (the Jamba-3B port)
3. **Hot-swappable topic adapters** that survive base-model retraining and
   architecture changes — this is the deep section
4. Model load-to-memory time
5. Memory cost **beyond params/quant** (CUDA context, allocator, state pools)
6. How vLLM / SGLang / TensorRT-LLM / llama.cpp / Ollama / ServerlessLLM /
   LoRAX / PEFT handle the same features — where we are ahead, behind, and
   where we are reinventing wheels
7. New R&D areas worth a round
8. Training-free **effective-context extension** (merged from
   `.devin/scratchpad.md` research track — §6)
9. Second sweep (§7.5): training/self-play/merging, decode execution,
   structured output, model multiplexing, next-gen arch, tokenizer

Method: direct code verification (loader, engine_lora, engine_lifecycle,
kv_backend, config, hotswap read line-by-line), the measured boot report in
`.devin/boot_v2_jamba_report_bf16.json`, the prior-session scratchpad
research (`.devin/scratchpad.md`), and ~65 external sources (papers +
serving-system docs) cited inline.

---

## 0. Measured baseline (from `.devin/boot_v2_jamba_report_bf16.json`)

| Metric | Value | Note |
|--------|-------|------|
| Model load | **3.9 s** | bf16, `fastsafetensors` nogds path or fallback |
| Strategy activation | 1.4 s | rotorquant KV + ~15 features on |
| Params reported | 2,127 M | (≈2.1B; true total ≈3.20B — §3) |
| VRAM after activate | 7.30 GB alloc / 7.63 GB reserved | 8.93 GB total used incl. context |
| Model weights on GPU | **4.35 GB** | vs 6.1 GB file → cast/pack happening |
| **Non-weight VRAM** | **~2.95 GB** | alloc − weights; ledger in §2 |
| Fragmentation (reserved−alloc) | ~330 MB | expandable_segments already on |
| Generation | 16–29 s / ~450 tok | ≈ 16–20 tok/s at bs=1, StandardDecoding |
| LoRA load | 0.3 s, 232 tensors, 36.8 M params, r=32 | single-adapter, rebuilds modules |
| Idle state | rotorquant KV 4-bit, chunked prefill, prefix cache, seq_split, suffix_spec, fused_qk_norm_rope, adaptive_spec, quamba2, replay_ssm, avmp, virtual_tensor, triton_conv, cache_blend all ON | impressive default stack |

The single most useful framing: **at idle-after-activation, ~40% of allocated
VRAM is not weights.** For a "3B model on 12 GB" the real budget is
4.35 GB weights + ~3 GB everything-else.

> **Caveat discovered during this audit**: the reported KV strategy
> (`rotorquant`, 4-bit) is a *bookkeeping* object — the model's attention
> path never reads `engine.kv_cache` (§2.1). Treat all KV-strategy numbers
> in this report as "activated but not on the hot path."

---

## 1. Load path — what's actually there (verified)

`forge/model/loader.py` is better than the codebase's reputation suggests:

- `build_model_fast` (loader.py:~640) runs **weight loading in a background
  thread** overlapping **meta-init** architecture build — correct design.
- `_load_safetensors_mmap` (loader.py:109-155) tries `fastsafetensors`
  (`nogds=True`, pinned-memory + async DMA) and falls back to
  `safe_open(..., device="cuda")` direct-to-GPU loading with
  `SAFETENSORS_FAST_CUDA=1` (set in `forge/runtime/configure.py`). On
  Windows, GDS is unavailable so `nogds` async I/O is the right mode.
- Pre-quantized paths exist and avoid bf16 intermediates: BitNet int8 direct
  (`load_prequantized`, loader.py:723-783) and IRI-FP4 packed
  (`IRIFP4Linear` swap-in, loader.py:785+). This is the correct pattern —
  the peak-RAM note in the code ("4.7 GB instead of 3×") matches best
  practice.
- `ProgressiveLoader` (`engine/loader/progressive_loader.py`) exists for
  TTFT-first streaming (essential 25% then background fill) — but the boot
  report shows `use_progressive_load: false`.
- `ForgeLoader` (`engine/forge_loader.py`) does true zero-copy GGUF mmap —
  genuinely llama.cpp-style; this is ahead of what most PyTorch engines do.
- Sleep/wake exists (`engine_lifecycle.py`): L1 = `model.to("cpu")` +
  cache clear; L2 = discard + reload. Parity with vLLM sleep levels.

### 1.1 Critique — remaining load-time gaps

**L1. Per-tensor `.clone()` in the fastsafetensors loop** (loader.py:132,
180). `get_tensor(key).clone()` allocates + copies each tensor
individually on-device. ServerlessLLM (OSDI'24) and the fastsafetensors
paper both get their 4.8–7.5× by treating groups of tensors as *contiguous
buffer batches* — one DMA per group, not per tensor. With ~500+ tensors in
V2, per-tensor launch overhead + allocator churn is real. Fix: request a
bulk copy — `fastsafetensors` supports loading contiguous frame ranges;
instantiate tensors as views into a persistent buffer instead of cloning
out of a transient one.

**L2. The fallback `safe_open(device="cuda")` still materializes per-tensor
on CPU-side metadata + individual H2D copies.** It is strictly better than
`torch.load`, but it is the "vLLM default path" that the 2026 cold-start
measurements (vllm-coldstart-probe, eBPF trace) show is *syscall- and
per-tensor-bound, not bandwidth-bound*: kernel I/O was ~7% of wall time;
the bottleneck was per-tensor CPU work + first-kernel-launch. Our 3.9 s for
6.1 GB ≈ 1.56 GB/s effective — consistent with that profile (NVMe here can
do ~5 GB/s sequential). **There is likely ~2× left on the table** with
batched contiguous reads + a pinned staging pool.

**L3. `load_state_dict(assign=True)` on meta-init model is right, but the
int8/FP4 pre-quantized branches re-enter per-tensor Python loops**
(loader.py:749-783: `for k, t in state.items()` with `isinstance` module
lookup per tensor). Fine at 3B, but this is O(tensors × dict lookups) on
the critical path — the BitNet path should group by module first.

**L4. No load-time instrumentation is emitted.** The boot script measures
3.9 s externally, but the loader itself doesn't record per-phase timings
(header parse / state load / assign / quantize / first forward). vLLM logs
`Loading weights took Xs, Y GB`. We should emit the same; the data exists
(`t_arch`, `t_weights` locals) but is only partially printed.

**L5. `wake()` L2 calls `build_model_fast` then `model.to(device)`**
(engine_lifecycle.py:91-94) — `build_model_fast` already loads to
`config.device`; the extra `.to()` is at best a no-op, at worst a redundant
pass. Also L1 sleep uses pageable `.to("cpu")` — a **pinned CPU staging
pool** (we already have pinned-memory infra in `hybrid_offload.py` /
`cpu_kv_offload.py`) would cut wake time meaningfully: pageable D2H/H2D
goes through an internal staging copy; pinned is a straight DMA. vLLM's
sleep L1 is the same idea but they keep the CUDA context and only move
weights — ours does too, so this is a ~cheap upgrade to hit their
"wake in seconds" numbers rather than our "~2-3 s" comment claim.

**L6. ProgressiveLoader is built but off by default** — `use_progressive_load:
false` in the measured config. TTFT on cold start could drop ~2× (only
embed + first-quarter layers needed for first token). Either wire it or
delete it (rule E).

### 1.2 Where the industry's load-time edge actually is (and what transfers to Windows)

| System | Technique | Transfers to us? |
|--------|-----------|------------------|
| ServerlessLLM (OSDI'24) | Custom contiguous format + O_DIRECT + pinned DMA pool + multi-threaded | **Yes** — we already have the file locally; the win is format/layout, not network. Their `sllm-store` gives 6-10× vs SafeTensors, Linux-only, but the design (sequential-layout file + pinned pool) is portable. |
| fastsafetensors (arXiv 2505.23072) | Batched lazy tensor instantiation, GPU-side type conversion, GDS | **Already wired** — but see L1: we clone per-tensor out of its buffers, partially defeating it. GDS itself is Linux-only; `nogds` on Windows is correct. |
| NVIDIA Dynamo Snapshot (CRIU + cuda-checkpoint) | Whole-process GPU+CPU state snapshot → ~10 s warm start for 120B | **No on Windows** — cuda-checkpoint is Linux/x64 only, no UVM/IPC. Closest Windows equivalent is our L1 sleep + pinned CPU pool. |
| vLLM sleep L1/L2 | weights→CPU / discard; wake ~0.1-6 s; 18-20× faster than restart | **Already have it** (sleep L1/L2). Gap: ours doesn't keep weights *pinned*, and L2 wake re-runs full build+`.to()`. |
| Ollama/llama.cpp | lazy mmap — "instant" because nothing is loaded until touched | Partially — our GGUF path already does this; safetensors path deliberately doesn't (we want weights resident). Could offer a lazy/`mmap`-resident mode for low-VRAM multitenant. |

Bottom line for load time: we're at ~1.56 GB/s effective. Best-case on this
hardware (Gen4 NVMe, PCIe 5.0 x16 GPU): a contiguous-format + pinned-pool
loader should hit 4-6 GB/s → **6.1 GB in ~1.0-1.5 s**. That is the R&D
target, and it is strictly an I/O pipeline problem — no model work needed.

---

## 2. ForgeEngine "holding" cost — the non-parameter memory ledger

Measured: 7.30 GB allocated vs 4.35 GB weights → **~2.95 GB of non-weight
allocations**. Breakdown, verified against code:

### 2.1 KV cache — **the strategy zoo is not on the hot path** (verified)

This is the audit's most consequential finding, confirmed in source:

- The engine decode path passes `(k, v)` **tuple `presents`** through the
  model. `GroupedQueryAttention.forward` does
  `torch.cat([past_key_value[0], k], dim=-2)` per attention layer **per
  generated token** (layers.py:271-272) — an O(n) copy each step →
  **O(n²) total** over a generation, plus a fresh allocation each step
  (allocator churn).
- The ~20 `KVCacheStrategy` backends (`kv_backend.py`, `engine/kv/*` —
  s4r, paged, rotorquant, snapkv, cpu_offload…) are activated in
  `engine_activation.py` and sized in `forge_engine.py`, but **the model's
  attention never reads `engine.kv_cache`/`_forge_kv_state`.** They are
  side-band bookkeeping objects: `rotorquant` "4-bit KV" in the boot
  report almost certainly did not compress the real cache.
- `PreAllocatedKVCache` (`kv_cache.py:24-197`, O(1) indexed append)
  exists and is correct — but is used **only** by
  `ModelLoader.generate_text` (loader.py:1238-1253), never by the engine.
- Downstream breakage: `crash_recovery.py:173-203` reads `kv.k_cache`/
  `kv.v_cache` (silently wrong on compressed strategies); the OOM-recovery
  chain (`engine_generation.py:1136-1158`) "switches to S4R / CPU offload"
  — i.e. swaps the bookkeeping object while the tuple-cat hot path
  continues unchanged. Recovery likely doesn't reduce real KV memory.
- `StandardKVCache._ensure_allocated` (kv_backend.py:70-75) also
  preallocates `[B, 1, 262144, 128]` at the full 262 K ceiling
  (~268 MB across 2 attn layers) and `clear()` zero-fills per generation —
  waste, but secondary to the decoupling above.

**Fix direction** (this is the canonical Jamba-shape design): one unified
paged pool — vLLM V1's hybrid KV cache manager is the blueprint
(docs.vllm.ai/design/hybrid_kv_cache_manager): attention-KV slots and
Mamba-state slots share a page size; "align mode" (PR #30877)
block-aligns Mamba state checkpoints so prefix caching works for SSM
layers too. Concretely for us: route engine decode through
`PreAllocatedKVCache` (or a paged variant) and make `KVCacheStrategy`
backends *wrap that same storage* so the strategy choice actually gates
the hot path.

### 2.2 SSM/conv state — small per seq, unmanaged as a pool, and fp32

Verified shapes (`mamba_probe.py`, d_inner = 5120, d_state = 16, d_conv = 4):
- SSM state: (1, 5120, 16) **fp32** = 327.7 KB/layer — fp32 is deliberate
  (bf16 state drifts ~8%/layer, noted at mamba_probe.py:158-160)
- conv state: (1, 5120, 3) bf16 = 30.7 KB/layer
- **Total ≈ 9.3 MiB per sequence across 26 layers, fixed vs context**

~9 MB/seq is small — until multiplied by concurrent sessions, or until you
notice it lives in per-layer ad-hoc buffers with no pool, no eviction, and
no offload. `cpu_kv_offload.py` handles KV only; **SSM state has no
offload path** — an idle session pins ~9 MB/seq forever. Also note
`_last_prefill_recurrent` (llm.py:578-591) is empty for V2: the
`MambaLayer._conv_state`/`_ssm_state` attrs exist but are never written —
state is carried in the `presents` dicts, so the snapshot machinery that
would enable state persistence/migration is dead for the production model.

### 2.3 CUDA context + libraries — the silent ~0.5-1 GB

`vram_alloc_gb=7.3` is torch-visible; `used_gb=8.93` (driver view) →
**~1.6 GB is invisible to the torch allocator**: CUDA context, loaded
cubins (Triton JIT cache — `use_triton_conv` + fused_qk_norm_rope +
quamba2 + replay_ssm kernels), cuBLAS/cuDNN workspaces, UVM driver
structures. On a 12 GB card that's 13% gone before we start. Note the
known Blackwell/sm_120 + `expandable_segments` interaction: PyTorch issue
#182286 documents expandable_segments **crashing the CUDA driver** on
sm_120 under WSL2 — we're native Windows so likely fine, but sm_120
allocator bugs are active upstream; worth a `torch.cuda.memory_stats()`
regression check after torch upgrades (see also the torch 2.11+cu130
~23 GiB-at-init regression, pytorch#182941).

### 2.4 Everything else

- **Logits**: vocab 65536 → fp32 logits 256 KB/seq/step; sampling buffers +
  top-k workspaces are small but per-step allocated unless pooled.
- **Activation peak**: chunked prefill (512) bounds it — good. Prefill
  workspace for Mamba chunk-scan is O(chunk² · d_inner) — bounded.
- **CUDA graphs**: `_graph_runner` exists; graphs pin their capture
  buffers. If multiple shapes are captured, each is a permanent
  reservation. Check the count.
- **Pinned host buffers**: hybrid_offload + cpu_kv_offload pin RAM on the
  CPU side — that's system RAM (32 GB) not VRAM, but it counts toward the
  "holding cost" story; ~6 GB CPU-resident weights on L1-sleep is the
  right trade.

### 2.5 The ledger, summarized

| Bucket | Est. size | Managed? |
|--------|-----------|----------|
| Weights (bf16, some cast) | 4.35 GB | yes |
| CUDA ctx + cubins + workspaces | ~1.3-1.6 GB | no (driver) |
| KV pool (standard, max_seq) | up to 268 MB | yes, but over-prealloc'd at 262 K ceiling |
| SSM+conv state | ~9 MB/seq | partially — no pool/evict/offload |
| RotorQuant/other strategy buffers | 100s MB | per-strategy |
| Allocator fragmentation | ~330 MB measured | expandable_segments helps |
| Logits/sampling/graph capture | 10s-100s MB | partial |

**The real finding**: on 12 GB, the model fits with ~3.9 GB free — but the
*headroom* is what the non-weight ledger eats first. Every 100 MB of
per-tensor clones, full-window KV prealloc, and un-pooled SSM state is
context length you can't sell.

---

## 3. ForgeLM V2 architecture critique

Verified spec (config.py:590-643, not just AGENTS.md):

- 28 layers: **26 Mamba + 2 attention at indices 7 and 21** — matches HF
  `attn_layer_offset=7, attn_layer_period=14`.
- **The "Mamba-2" is actually a Mamba-1-style mixer** (verified, not just a
  naming nit): `ssm_type="mamba2"` dispatches to `MambaLayer` in
  **`forge/keys/architecture/mamba_probe.py`** — a probe/reference file that
  *is* the production runtime. It is HF `JambaMambaMixer`-equivalent
  (Mamba-1 semantics): `in_proj` 2560→10240 (x,z), depthwise conv k=4,
  `x_proj` 5120→192 (dt_rank 160 + B16 + C16, single shared group),
  `A_log` per-channel (5120,16), Jamba norms on dt/B/C. Faithful to the
  Jamba-3B source — but `ssm_type`/AGENTS.md saying "Mamba-2" is wrong,
  and the file's own docstring calls the scan "not intended for production
  scale."
- d_model 2560, n_heads 20, **n_kv_heads=1 (MQA)**, head_dim 128, no RoPE
  (Jamba attention is position-free; Mamba carries position).
- vocab 65536, **untied embed+head** → 2 × 65536 × 2560 × 2 B = **671 MB
  (≈11% of the 6.1 GB file)** spent on vocabulary. For a 3B that is heavy
  (Llama-3-8B is ~6% embed share). Embedding int8 and/or a factored embed
  is one of the highest-ROI byte cuts available.
- Param math (computed): SSM ≈ 41.2 M/layer (in_proj dominates at 26.2 M),
  FFN 62.9 M/layer → ~104.2 M × 26 = 2.71 B; attn block ≈ 76.7 M × 2 =
  153 M; embed+head 335.5 M → **total ≈ 3.20 B** — consistent with the
  "~3.2B" claim. (The boot report's "2127M" undercounts — likely excludes
  embed/head and/or some projections.)
- `max_seq_len=262144` with only-2-attn layers is exactly the regime where
  MQA + tiny KV shines: KV/token ≈ 2 KB vs 28 KB for a full-attention
  28-layer model → AI21's "8× smaller KV" holds by construction.

### 3.1 Implementation findings — these are the real architecture risks

Ordered by severity; all verified in source:

1. **The SSM scan is a pure-Python per-token loop.**
   `MambaLayer` falls back to `_selective_scan_ref` (mamba_probe.py:135-189)
   — a `for t in range(L)` fp32 loop whose own docstring says "slow but
   correct… not intended for production scale." It is the production path:
   no SSD chunked-matmul formulation, no Triton/CUDA kernel
   (`triton_conv.py` only patches `DoubleGatedConvLayer` — dead for V2).
   Prefill is O(T) sequential Python steps × 26 layers of ~8 kernel
   launches each; decode pays the same Python overhead per token. **This
   alone plausibly explains the ~16-20 tok/s decode and makes the 262K
   context claim aspirational** — at 100K+ tokens prefill is dominated by
   a Python loop, not bandwidth. Priority: implement the SSD chunked scan
   (mamba2-style, works for any d_state) or a Triton selective-scan; HF's
   `mamba_ssm`/`causal_conv1d` aren't imported anywhere (`grep` confirms
   only 4 files reference them, none in the hot path).
2. **All-T logits materialization on prefill** (llm.py:656): `logits =
   self.head(hidden)` runs on the full [B,T,2560] hidden even when only
   the last token is needed → [1,262144,65536] bf16 = **34.4 GB → OOM**
   at the advertised context. Even an 8K prompt materializes ~1.1 GB
   transient. Fix: `self.head(hidden[:, -1:])` on the generation path.
3. **conv_state dropped on T>1 resumes** (mamba_probe.py:~223): chunked
   prefill and prefix-cache suffix hits lose the conv left-context — the
   first `d_conv-1`=3 tokens of every resumed segment get zero-left-pad
   convolutions in all 26 Mamba layers. SSM state continues via `h_init`,
   so this is a *silent quality bug*, not a crash — and chunked prefill is
   ON by default.
4. **Per-step GPU→CPU syncs**: `(next_token == eos).any().item()` per
   token (loader.py:1267) plus per-sequence `.item()` stop checks — each
   stalls the pipeline. Standard fix: check EOS every k steps or keep a
   device-side flag.
5. **MQA 1-KV-head × quantized KV**: a single KV head is a known recall
   cliff at long context; if/when the KV strategies are actually wired in
   (§2.1), 4-bit KV on top of MQA needs the **query-agnostic eval
   protocol** — the 2026 matched-budget audit (arXiv 2607.11942) and the
   ACL'26 pitfalls paper show eviction compression (SnapKV-class) loses to
   trivial keep-start+recent baselines under query-agnostic scoring and
   degrades multi-instruction following unevenly. Re-benchmark our 20+
   strategies under that protocol before trusting the leaderboard.
6. **Spec-decode on SSM state**: `use_replay_ssm: true` (boot config) —
   the TRT-LLM "replay" rollback (state 2 steps back + step inputs) is the
   right design; the known failure mode is the lemon-mlx bug (KV rolls
   back, Mamba state doesn't → corrupted state → degenerate output). Needs
   a draft-rejection regression test asserting spec output ≡ non-spec.
7. **Mamba weight-quant fragility**: MambaQuant (ICLR'25) / Quamba2
   (arXiv 2503.22879) show SSM blocks are more quant-fragile than
   attention (outliers in gate proj + PScan output amplified by scan;
   Quarot loses 21% acc at W8A8). `use_quamba2` is already on — but the
   same caution applies to BitNet int8/W8A8 on `in_proj`/`out_proj`;
   per-state-group quant for B/C/dt is the literature answer.
8. **Smaller items**: `cudagraph_mark_step_begin()` every forward
   (llm.py:409) — fixed per-step overhead; `rope_base`/`conv_kernel_size`
   are stored-but-dead config for V2; prefix-cache partial hits pass Mamba
   state dicts through `_slice_past_kv` unsliced (only position-correct at
   exact boundaries — interacts with finding 3).

---

## 4. The adapter problem — "topic LoRAs that survive retraining and arch changes"

This is the user's sharpest question and the one with the richest research
landscape. First, the honest current state:

### 4.1 What we have today (verified — `engine_lora.py` + `sft_train.py` + `forge_server.py`)

- `load_lora()`: **single adapter only** — `self.unload_lora()` first
  (line 67). Swap = delete all `lora_adapter` submodules + re-patch
  forwards + copy weights. It works (0.3 s, 142 adapters, 36.8 M params in
  the boot run) but it is *structural* swapping, not weight hotswap:
  every swap rebuilds the monkey-patched forwards.
- **Two incompatible adapter key conventions exist**: the manual
  `bitnet_lora.add_lora_adapters` path writes `…<mod>.lora_adapter.lora_A`
  (engine-loadable), while `--save-lora-adapter` under PEFT writes
  `base_model.model.blocks.{i}…lora_A.weight` — **saved but not loadable
  by `engine.load_lora`** (zero name matches → all-warning silent no-op).
  Adapters produced by `self_play/discovery/finetune.py` use the manual
  convention (good); `sft_train.py --save-lora-adapter` under the default
  non-BitNet path can produce the unloadable kind.
- **Target-list asymmetry between train and serve**: engine defaults
  `{w_gate,w_up,w_down,q_proj,v_proj,out_proj,in_proj}` (omits `k_proj`,
  `x_proj`, `dt_proj`, `head`); SFT non-BitNet hardcodes
  `{q_proj,k_proj,v_proj,out_proj,w_gate,w_up,w_down}` (no `in_proj`); SFT
  default (bitnet-everywhere → all Linears) trains `x_proj`/`dt_proj`/
  `head` tensors the engine can't attach under its default targets →
  warn-and-drop. Same adapter file, different subset applied depending on
  who loaded it.
- **Nothing inside SSM dynamics is adapted** (A_log, D, dt, B/C paths) —
  consistent with MambaPEFT (ICLR'25) finding LoRA-on-linears "fails on
  SSM modules"; SDT (ICML'25, Sparse Dimension Tuning) is the current
  best answer (LoRA on linears + sparse-dim tuning on SSM state dims).
- **Non-atomic, unlocked load**: `load_lora`/`unload_lora` are *not* under
  `_gen_lock`; per-request `lora_adapter` in `forge_server.py:719-755`
  swaps→generates→restores on the **non-streaming path only**
  (`_stream_chat` has no such logic) and races with in-flight
  generations. A shape mismatch inside `param.data.copy_` raises
  RuntimeError **mid-load → partially applied adapter**.
- **No manifest**: adapter files have no rank/alpha/targets/base-hash/
  config-fingerprint sidecar; `load_lora` trusts caller-supplied `rank`.
  The GUI `lora_store` infers rank from header shapes for display only.
- Checkpoint keys = exact `named_parameters()` names → bound to layer
  index + attribute name + dims. Any change that permutes `layer_types`,
  renames modules, or changes a module's dims invalidates the file — and
  the only guard is a warning log.
- **Closest existing infra for topic routing**: AirMoE
  (`moe/routers.py` SemanticRouter/KeywordRouter + `airmoe_infinite.py`
  manifest + LRU disk cache) already does *semantic topic → weights*
  hot-injection — for experts, not LoRA. Also already in the tree:
  `training/forge_adapter.py`, `dlora.py`, `dora.py`,
  `adapter_variants.py` (PiSSA/AdaLoRA/rsLoRA), `self_play/opmix.py` —
  per rule E, any LoRA-XS/DoRA work should extend these, not spawn new
  files. `discovery_loop.py` already hot-swaps epoch LoRAs
  (unload→load each winning epoch) — real usage of the current path.

### 4.2 What "survives retraining / new arch" actually requires

The binding constraint is *where* the adaptation lives:

| Adapter space | Survives weight retrain (same arch) | Survives arch change (same dims) | Survives dim/arch change | Mechanism |
|---|---|---|---|---|
| LoRA on weights (ours) | ⚠️ partially — deltas assume the W they were trained on | ✗ module-bound | ✗ dim-bound | W + BA |
| **LoRA-XS / LoRA-X** (arXiv 2405.17604, 2501.16559) | ✅ by construction | ⚠️ needs same module set | ⚠️ needs dim compat | Store only r×r matrix R between frozen SVD bases of W. **On base retrain: recompute truncated SVD of the new W, keep R.** LoRA-X explicitly demonstrates transfer across model *versions* data-free. |
| **Cross-LoRA** (arXiv 2508.05232) | ✅ | ✅ heterogeneous models | ✅ dim mismatch handled | SVD subspace alignment + Frobenius-optimal projection of source LoRA into target weight space. Data-free, ~20 min on commodity GPU. |
| **LoRASuite** (NeurIPS'25) | ✅ designed exactly for "base model upgraded" | ✅ layer/head remap via CKA + cosine | partial | transfer matrix from old+new params + small skillful finetune; beats full LoRA retrain on MiniCPM/Qwen (+1.4/+6.6 pts). |
| **Proxy-tuning** (Liu et al. 2024, arXiv 2401.08565) | ✅✅ | ✅✅ any arch | ✅ only needs shared **vocab** | logit_base_large + (logit_small_tuned − logit_small_untuned). The "adapter" is a small-model delta applied at decode. 88% of true-tuning gain on Llama2-70B. **This is the only fully arch-agnostic option** — and it's decode-time, hot-swappable per-request, and survives ANY base change as long as the tokenizer stays 65536. |
| Prompt/prefix tuning | ⚠️ degrade on weight updates | ⚠️ tied to d_model | ✗ | activation-space; transferable cross-model only via trained projectors (THUNLP prompt-transferability, SPoT). Cheap to retrain though. |
| kNN-LM datastore | ✅ (keys are hidden states — but must re-encode on weight change: forward-only, no training) | ⚠️ needs same hidden space, or external embedder | ✅ with external embedder | retrieval + logit interpolation; +17% ppl over zero-shot (Bhardwaj'23). |
| Text-to-LoRA / Doc-to-LoRA (Sakana, ICML'25) | regenerate from description → arch-bound but *regeneration is free* | regenerate for new arch | regenerate | hypernetwork emits LoRA weights in one forward pass. Meta-train once per base family; when base retrains, re-emit adapters instead of retraining them. |
| LoraHub composition | inherits base-adapter limits | — | — | gradient-free coefficient mixing of existing LoRAs on ~5 examples; the *routing* layer for topic adapters. |
| **Trans-LoRA** (NeurIPS'24, IBM) | ✅ | ✅ cross-family, cross-PEFT (LoRA↔DoRA) | ✅ | nearly data-free: source base+LoRA generates synthetic data → fresh adapter trained on new base. Fits our self-play/synthetic-data infra directly. |
| **TransFusion task-vector re-basin** (arXiv 2505.22697) | ✅ | ✅ | partial | training/data-free transfer of τ = θ_ft − θ_base via permutation alignment. |
| **ReLoRA-v2** (arXiv 2606.02606 — NOT the 2023 ReLoRA) | ✅ | — | — | Bayesian-opt fusion init (old adapter + base delta) → short FT w/ scheduled reg: 8.9× faster rollout, +4.6% acc. The "warm-start touch-up" recipe. |
| **Exact delta rebase (closed-form)** | ✅ trivially | ✗ | ✗ | ΔW' = (W_old + ΔW_lora) − W_new reproduces the merged model bit-exactly but *cancels* new-base gains — only for behavior freezing; SVD-truncate to stay rank-r. |

**Change-type verdicts** (from the scratchpad's analysis, cross-checked
against the code path): a zero/identity-init *lossless port* carries
adapters **for free** (shared weights bit-identical, new modules no-op —
breaks only on renames (partial apply) or resizes (crash)). *Same-arch
weight drift* (merged self-play epoch, continued pretrain) loads fine and
decays gracefully with ‖W_new − W_old‖. *Structural change* (dims, layer
order, renames, mixer swap) requires transplant tooling that doesn't exist
today.

**Important calibration from ICML'26 "Trivial Baselines"**: for *related*
bases, plain direct-copy of LoRA weights beats elaborate transfer schemes
(CrossLoRA/ProLoRA); success tracks weight similarity — MCQA transfers
easily, open generation degrades most. So the right policy ordering is:
**direct copy first → warm-start touch-up (ReLoRA-style, ~10-20% of
original steps) → transplant methods (LoRASuite/Cross-LoRA) only when
shapes actually break → Trans-LoRA distillation for large jumps.**

**ForgeAI-specific wrinkle**: our adapter targets include `in_proj`
(Mamba SSM input projection) — the entire transplant literature above is
transformer-only. For Mamba-resident adapters, prefer zero-shot copy or
distillation; SVD-subspace methods have no published validation on SSM
projections.

### 4.3 Recommended design — a 3-tier adapter stack

The research points at a layered answer, and each tier maps to code we
mostly have:

**Tier 1 — Weight-space, same-arch, hot (today's LoRA, fixed up).**
Keep current LoRA but change the *swap mechanics* to PEFT-style hotswap:
keep the patched modules permanently, maintain a dict of named adapter
state-dicts, and `param.data.copy_` into a *slot* (that's what
`peft hotswap=True` does — in-place weight replacement, no module rebuild,
torch.compile-safe). Then add llama.cpp-style `scale` — per-request
`adapter scale` multiply is ~free and gives **LoraHub composition for
free** (a mixture `Σ wᵢ·AᵢBᵢ` is just per-adapter scales). Storage keying
must move from positional names to **(layer_type, module_kind, shape)
triples** so a future layer re-order doesn't silently misbind.

Tier 1 hardening detail (from the scratchpad audit, all cheap):

- **Fingerprint at save time**: safetensors supports a `metadata=` dict —
  stamp `{"base_preset", "base_hash", "arch_fingerprint"}` into
  `*.lora.safetensors` (the `lora_store.py` header parser is already
  stdlib-only), warn in `load_lora` on mismatch. No manifest format needed
  for this subset.
- **`LORA_NAME_REMAP` dict in `load_lora`** — one-line insurance for the
  day a key renames `w_gate` → `ffn.gate` under a future derived preset.
- **Zero-init tolerates structural gaps for free**: today's missing-name
  path already tolerates partial transplants — new layers added by a
  lossless port just get zero-init adapters. Exploit this deliberately.

**Tier 2 — Version-portable weight adapters (the LoRA-XS pivot).**
Store topic adapters as the small **R matrix between SVD-frozen bases**
(LoRA-XS) instead of raw A/B. Consequence: when the parent model retrains
or a derived preset changes a weight's content but keeps its shape,
**recompute the SVD of the new W, reattach R — the trained knowledge is
preserved with zero retraining.** For true arch changes (Mamba-2→Mamba-3 module swap),
apply **Cross-LoRA/LoRASuite**: SVD-align source/target subspaces and
project the adapter into the new module's weight space (data-free, ~20 min
on this GPU — or fall back to LoRASuite's CKA layer-matching + small
finetune when shapes genuinely differ). This turns "every arch bump kills
all adapters" into a mechanical conversion pass.

**Tier 3 — Fully arch-agnostic decode-time adapters (the strategic escape).**
**Proxy-tuning**: train topic deltas as *small-model* logit differences
against a shared-vocab proxy pair (e.g. a `forgelm_tiny`-scale model on the
same 65536 tokenizer — trained tuned vs untuned). At decode:
`logits = base + α·(expert − antiexpert)`. Properties: survives *any* base
retrain, any arch, any quant; per-request selectable; composable by adding
multiple deltas; the only hard dependency is the tokenizer. Cost: a second
forward through a small model per step (~2-5% latency for a 100-300M
proxy at bs1 — and the proxy forward can run on CPU/off-GPU stream, or be
replaced by an N-gram/retrieval head for lexical tasks). This is the
"adapter that can't break" floor of the design — and it doubles as an
evaluation harness: measure a weight-space adapter's transfer quality by
comparing its logit delta to the proxy delta it was supposed to implement.

**Routing**: topic→adapter(s) selection can start as embedding-kNN over
adapter descriptions (we have `topic_scan_cache.json` infrastructure
already) and graduate to a T2L-style hypernetwork if the adapter count
justifies it. LoraHub shows ~5-example gradient-free coefficient fitting
beats zero-shot and approaches ICL on BBH — a strong baseline router.

### 4.4 What the big platforms do for this exact feature

- **vLLM**: `--lora-modules`, `max_loras` GPU slots + `max_cpu_loras`
  CPU pool, LRU evict (~30-50 ms swap from CPU pool measured), Punica
  SGMV grouped-GEMM for heterogeneous-adapter batching, per-request
  `model=<adapter>` addressing, dual-stream MoE-LoRA. Throughput findings:
  `max_loras=16` optimal under skewed traffic; uniform traffic over 1000
  adapters ~halves throughput (grouped GEMM vs dense).
- **S-LoRA** (MLSys'24): all adapters in CPU RAM, unified paging pool
  shared by adapter weights AND KV blocks — thousands of adapters, ~4×
  vLLM-naive. **Unified paging of adapters+KV is the single most
  transferable idea for our 12 GB budget.**
- **LoRAX** (Predibase): JIT adapter load from HF mid-request without
  blocking; async prefetch/offload scheduler; per-request *merge* of
  adapters.
- **llama.cpp**: `--lora`, `--lora-scaled`, `--lora-init-without-apply`,
  `POST /lora-adapters` to re-scale at runtime, per-request `lora:
  [{id, scale}]`, `llama_set_adapter_lora(ctx, lora, scale)` — i.e., the
  *scale-mixing* primitive; different LoRA configs don't co-batch (we could
  beat that with SGMV-style gather).
- **PEFT**: `hotswap=True` in-place weight swap (compile-safe), adapter
  naming normalization, `modules_to_save`.
- **TensorRT-LLM**: `LoraConfig(max_lora_rank, max_loras, max_cpu_loras)`,
  task_id adapter cache + LRU, fused MoE-LoRA CUTLASS kernel.
- **ServerlessLLM**: `load_lora` through the same multi-tier fast store —
  adapter load ~ms.

None of them solve cross-architecture adapter survival — that's still
research-tier (LoRA-XS/Cross-LoRA/LoRASuite are 2025 papers, not shipped
features). **This is a genuine gap we could lead in**: ForgeAI's KeyStack
makes arch churn routine, which makes adapter portability a bigger pain for
us than for anyone shipping a fixed HF arch.

---

## 5. Platform comparison — where we're ahead / behind / equal

| Capability | ForgeAI | vLLM | SGLang | TRT-LLM | llama.cpp/Ollama | ServerlessLLM | LoRAX/S-LoRA |
|---|---|---|---|---|---|---|---|
| Weight load speed | 3.9 s / 1.56 GB/s eff | safetensors (slow default) or fastsafetensors | same loaders | engine rebuild | mmap-instant (GGUF) | 6-10× safetensors | same as base |
| Direct-to-GPU load | ✅ fastsafetensors nogds | ✅ load_format | ✅ | n/a | mmap | ✅ O_DIRECT pool | — |
| Sleep/wake | ✅ L1/L2 | ✅ L1/L2 (18-20× faster than restart) | partial | — | keep_alive | ✅ GPU multiplexing | — |
| Multi-LoRA serving | ✗ single adapter | ✅ slots+CPU pool+SGMV | ✅ | ✅ plugin+LRU | ✅ scale-mix | ✅ load_lora | ✅✅ thousands |
| Adapter-per-request | ✗ | ✅ | ✅ | ✅ | ✅ (no co-batch) | ✅ | ✅ |
| Paged KV | ⚠️ strategy exists but off hot path (§2.1) | ✅✅ unified | ✅ | ✅ | contiguous | — | unified w/ adapters |
| SSM-state pool mgmt | partial | ✅ hybrid mgr + align-mode prefix cache | partial | ✅ replay rollback | — | — | — |
| Prefix cache | ✅ + radix sched | ✅ hash blocks | ✅✅ HiCache 3-tier (GPU/host/L3) | ✅ | prompt cache | — | — |
| Spec decode + SSM | ✅ replay_ssm | ✅ ReplaySSM (2.3× @bs512) | ✅ | ✅ replay | — | — | — |
| Chunked prefill | ✅ | ✅ | ✅ | ✅ | partial | — | — |
| Quant at load (prequantized) | ✅✅ int8/FP4 direct | ✅ | ✅ | ✅ | ✅ GGUF-native | — | — |
| CPU offload (weights/KV) | ✅ hybrid_offload + cpu_kv | ✅ (KV off) | ✅✅ HiCache L2 | partial | ✅ mmap/offload | ✅ | ✅ adapter off |
| CPU pool for adapters | ✗ | ✅ max_cpu_loras | ✅ | ✅ | RAM-resident | ✅ | ✅ main store |
| torch.compile/graphs | ✅ (off in measured cfg) | ✅ piecewise | ✅ | ✅ engine graphs | ggml graphs | — | — |

**Where we're genuinely ahead or equal**: format coverage (GGUF mmap +
safetensors + prequantized direct-load is broader than vLLM's),
Windows-native (everything above assumes Linux), and the KeyStack
hot-config system (HotSwapManager) is more dynamic than any of their
runtime reconfiguration. Feature breadth is *real but with an asterisk* —
rotorquant/quamba2/replay_ssm are on by default, yet the KV-strategy
checkmarks apply to bookkeeping until §2.1's decoupling is fixed.

**Where we're behind**: (a) multi-adapter *serving* (single-adapter only);
(b) unified paging of heterogeneous state (KV + SSM + adapters in one
pool); (c) SSM-state lifecycle management (evict/offload/align-checkpoint);
(d) batched contiguous loading (we still do per-tensor clones);
(e) adapter CPU pool + scheduling.

**Where the comparison is unfair-in-our-favor**: they're multi-GPU serving
stacks; we're a single-12GB-card engine. The right baseline is "features
per GB on one consumer card" — and on that metric the gap list above is
achievable precisely *because* our scale is small (a single-GPU unified
pool is much simpler than their distributed version).

---

## 6. Massively extending effective context (no training)

From `.devin/scratchpad.md`'s second research track, cross-checked against
the code. Reframe first: **KV cache is not the context bottleneck on this
model** — 2 attention layers × MQA ≈ 1 KB/token → 1M tokens ≈ 1 GB, and
Mamba state is a fixed ~9.3 MB regardless of length. The real ceiling is
the **Mamba effective receptive field (ERF)**: the SSM state is a lossy
fixed-size compression and recall saturates long before the advertised
262K. So context-extension work should aim at *recall*, not capacity.

Already-built pieces (do NOT rebuild): `hotswap.set_infinite_context`
(1M ctx + auto eviction), the KV-strategy zoo + `auto_context`
meta-manager, `ChunkedPrefiller`/`HybridChunkedPrefiller`, and — important
— **Marconi-style recurrent-state snapshot machinery already exists**:
`prefix_cache.capture_recurrent_state`/`apply_recurrent_state_prefix`, and
`llm.py:575-591` writes `_last_prefill_recurrent` at prefill end.
*Caveat from §2.2*: that snapshot only captures layers exposing
`_conv_state`/`_ssm_state` attrs — i.e. `DoubleGatedConvLayer`, **not**
V2's `MambaLayer` (its state lives in `presents` dicts). The plumbing
exists; it must be wired to extract state from the Mamba presents path.

Ranked options (all training-free):

| Approach | Cost | Value |
|---|---|---|
| **ERF probe first** — passkey/depth curve at 4k/16k/64k/256k | afternoon | tells you where recall actually breaks before spending effort; `mamba_probe.py` exists for this |
| **MambaExtend ∆t calibration** (ICLR'25) | ~26 zeroth-order scalars, afternoon-scale | 32× extension (2k→64k) with minimal PPL cost — Mamba long-context failure = OOD discretization steps; calibrate per-layer ∆t scaling. Highest leverage/effort in the table. github.com/ArminAzizi98/LongContextMamba |
| **SSM-state document library** — ingest doc once → snapshot conv+SSM state (~10-30 MB) → restore + query later | medium; extends existing snapshot code | Mamba-native RAG with zero re-prefill; Megatron-LM shipped prod Mamba prefix caching (PR #3225). Nobody has productized per-doc state libraries — genuinely novel. Needs the presents-path extraction noted above; ~100+ docs fit trivially in 32 GB RAM. |
| **Query-at-end prompt policy + LongLLMLingua compression** | prompt-layer only; LLMLingua-2 runs pure-CPU | Mamba is recency-biased (not lost-in-middle) → put query/key facts at the END; 4-6× compression, +21% on RAG tasks |
| **Chain-of-Agents / ReadAgent / MemGPT-style paging** | orchestration only | sequential worker passes + running note (+10% over RAG), gist memories (3.5-20× effective ext), self-paging via our existing tool-use infra |
| **InfLLM block memory** (NeurIPS'24) | heavier engine work | past context in CPU-RAM units, retrieve per step; validated at 1024K; composes with `cpu_kv_offload` |
| **DeciMamba** (ICLR'25) | heaviest — needs a port from the Mamba-1 implementation | token decimation inside the SSM via ∆t-norm importance: longer ERF *and* faster inference |

Suggested order: ERF probe → MambaExtend → SSM-state doc library →
prompt policy → orchestration. Open items from the scratchpad: verify the
∆t hook point (post-softplus) in our Mamba impl, confirm state snapshots
under Quamba2's quantized `_ssm_state` handling, size the RAM doc library.

## 7. New R&D areas (ranked by expected value on this hardware)

(These assume the §8 P0-P4 engineering fixes land first — the SSM scan
kernel and KV-strategy wiring are prerequisites, not R&D. Context-extension
R&D — ERF probe, MambaExtend, SSM-state doc library — is §6.)

1. **LoRA-XS adapter storage** — store topic adapters as r×r R matrices
   between SVD bases of the current weights. On any retrain: re-SVD, keep
   R. Zero-retrain portability for same-shape modules; combines with
   Cross-LoRA for shape changes. Directly answers the user's #3 ask and
   it's ~a storage-format change + an SVD pass, not a training pipeline.
2. **Unified state pool** — one paged allocator serving KV blocks, SSM
   state slots (block-aligned, align-mode), *and* adapter weights. vLLM's
   hybrid manager + S-LoRA's unified paging are the blueprints; ours is
   simpler (single GPU). Highest memory-structural ROI.
3. **Contiguous-format loader** — repack the 6.1 GB safetensors into a
   load-optimized contiguous layout (ServerlessLLM-style), pinned DMA
   staging pool, batched DLPack instantiation. Target: 6.1 GB in ≤1.5 s
   (from 3.9 s). Pure I/O engineering, measurable, high certainty.
4. **Proxy-tuning topic adapter** — small same-tokenizer proxy pair +
   logit-delta decode hook. Arch-immune adapters; also doubles as the
   transfer-quality metric for Tier-2 adapters. Needs one small model
   trained on the ForgeLM tokenizer.
5. **SDT for Mamba internals** — sparse-dimension tuning on SSM state dims
   (ICML'25) layered on existing LoRA-on-linears. We're already LoRA'ing
   `in_proj`; the literature says the SSM dynamics are where Mamba
   adaptation actually lives. Natural extension of `bitnet_lora.py`.
6. **SSM-state lifecycle** — evictable/offloadable/checkpointed Mamba
   state pool (align-mode block-boundary checkpoints à la vLLM PR #30877).
   Enables prefix-reuse for the 26 Mamba layers (currently only attention
   KV gets prefix caching) — big multi-turn win, and the substrate for the
   §6 SSM-state document library.
7. **Query-agnostic KV eval harness** — re-benchmark our 20+ KV strategies
   under the matched-budget query-agnostic protocol (arXiv 2607.11942)
   + multi-instruction (IFEval-style) suite (ACL'26 pitfalls paper).
   Cheap, high information, likely reorders our KV leaderboard — SnapKV-
   class methods lose to trivial baselines under this protocol.
8. **LoraHub-style adapter mixing** — once per-request adapter scale
   exists (Tier-1 fix), gradient-free coefficient search on ~5 examples.
   Near-free capability on top of the scale primitive.
9. **Text-to-LoRA hypernetwork** — meta-train an emitter over our adapter
   library so "new topic" = one forward pass, not a training run. Only
   worth it once we have ≥dozens of topic adapters; shelve as phase 2.
10. **d_state schedule sweep** — per-layer SSM state-size allocation (deep
    layers bigger). Math-checkable parameter-efficiency knob nobody has
    swept on a 26-Mamba model; a 20-line sweep script per directive F.

---

## 7.5 Second research sweep — lanes outside the serving surface

The first sweep covered loading, memory, adapters, KV, and context. This
one covers everything else the codebase touches: training, self-play,
merging, decoding, scheduling, tokenizer, and the next arch generation.

### 7.5.1 Training & self-improvement (sft_train, infinite_loop, ForgeEvolve)

- **Muon — ours is structurally correct (verified).**
  `build_muon_sf_plain` (muon_sf_blockwise.py:308-359) does the right
  param grouping: 2-D hidden weights → Muon+NS orthogonalization with
  weight_decay, embed/head + <2-D params → ScheduleFree-AdamW. The
  remaining gaps vs the Moonlight recipe (arXiv 2502.16982, ~2× AdamW
  efficiency): **per-shape update scaling** (their "match-RMS" trick is
  what makes LR transferable across widths — our LR is globally scaled
  off `max_lr` instead) and NS in bf16 stability. Worth a diff-level audit,
  not a rewrite. Memory upside stands: one momentum buffer vs Adam's two.
- **Sequence packing for SFT — partially built, with a Mamba-specific
  caveat.** `varlen_attention` (attention_ops.py:50-143, R&D round 14)
  already does packing-correct attention via `flash_attn_varlen_func` with
  a block-diagonal SDPA fallback — worth up to 2× SFT throughput
  (HF/IBM measurements). **But** the easy HF path — inferring boundaries
  from `position_ids` — does not work for linear-attention/conv models
  (HF docs warn GDN/conv models ignore `position_ids` boundaries). For V2,
  packing must also **reset conv+SSM state at each document boundary**
  (`_conv_state_reset` mechanism exists at llm.py:523-526 but nothing
  sets it mid-batch) — otherwise packed docs leak state into each other,
  a silent quality regression exactly like P3.
- **GRPO/RLVR infra already exists — upgrade it.** `rlvr_train.py` and
  `self_play/grpo_trainer.py` are in-tree; replicated results show GRPO
  works at 3B scale (smol-reason R1-Zero reproduction). Take the DAPO
  deltas: token-level loss, **drop the KL term** (kills the reference
  model → large memory saving), overlong filtering. Two failure modes to
  engineer around: **diversity collapse** (STaR loops plateau;
  Multiagent-Finetuning's answer is specialized generator/critic
  populations — ForgeEvolve already has populations) and task exhaustion —
  **Absolute Zero** (model proposes its own tasks under a learnability
  reward) is the natural next step for `infinite_loop.py` (which already
  has SGS/SAERL modes aimed at exactly this).
- **On-policy distillation is likely the single biggest quality lever
  available.** V2 is a *port*, so "train a better V2" mostly means
  post-training. GKD/GOLD (TRL; verl OPD): student samples its own
  trajectories, teacher supplies top-k log-probs, minimize sparse KL.
  Beats RL at a fraction of the compute (Thinking Machines' result; HF
  showed GOLD > ULD > GRPO on multi-step reasoning). GOLD removes the
  same-tokenizer constraint, so the teacher can be anything. On this
  hardware the teacher can even run on CPU/llama.cpp asynchronously —
  rollout generation on GPU, teacher scoring on CPU.
- **Model merging is free capability composition — check coverage.**
  `engine_merging` should be audited against the standard toolkit:
  task vectors (τ = θ_ft − θ_base), TIES (trim+elect-sign+merge) for
  multi-model interference, **DARE** (randomly drop ~90% of deltas and
  rescale — fine-tune deltas are *that* redundant), SLERP for 2-model
  blends. mergekit is the reference. Merging also shares math with the
  adapter-portability problem (§4) — task vectors and LoRA deltas are the
  same object at different ranks.
- **MTP vs EAGLE-3: our built-in head may already be the right design.**
  A careful Thoughtworks replication (vLLM, H200/B200) found **native
  multi-token-prediction heads beat a retrofitted EAGLE-3 head at bs=1
  and at high concurrency**, while EAGLE-3 wins mid-range. ForgeAI already
  trains MTP (n_heads=4, validated weight 0.495). Implication: invest in
  MTP head quality/data rather than retrofitting EAGLE-style feature
  prediction. If more acceptance is wanted cheaply and training-free,
  **Lookahead decoding** (Jacobi n-gram verification, ~1.8× on MT-bench,
  no draft model) generalizes the suffix-speculation already in-tree.

### 7.5.2 Decode-path execution (the Python-overhead tax)

- **CUDA-graph the decode step.** vLLM's experience with hybrid models is
  directly applicable: enabling piecewise CUDA graphs for Mamba layers
  gave "a pretty big performance boost" (PR #21194) and the residual gap
  was CPU overhead *in the Mamba layer itself*. Our decode step is
  Python-loop scan + `torch.cat` KV + eager everything — at bs=1 the
  per-step launch/Python overhead plausibly dominates the measured
  16-20 tok/s. Decode shapes are static (T=1), so a captured graph over
  the whole step is the classic fix; vLLM's `FULL_AND_PIECEWISE` mode
  (full graph for uniform decode, piecewise elsewhere) is the template.
  Watch the known hybrid gotcha: FULL-decode graphs must be capped to
  available Mamba state slots (vLLM #34571).
- **P1 has a concrete implementation path now.** `mamba-ssm` no longer
  builds its CUDA extension by default (needs `MAMBA_KEEP_CUDA_BUILD=TRUE`,
  painful on Windows) — but **flash-linear-attention (FLA) ships pure-Triton
  chunk-scan + causal-conv1d for Mamba-1/2** that runs anywhere Triton
  does. Vendoring/porting FLA's `chunk_scan` + `causal_conv1d` for our
  exact Jamba-1-style mixer is cheaper than writing one from scratch and
  carries no CUDA-extension build problem.
- **Name collision warning**: vLLM's `ReplaySSM` (PR #48018) is *not* our
  `replay_ssm.py`. Theirs caches recent SSM inputs to accelerate the
  Mamba-2 state-update kernel at serving batch sizes (at bs=1 the kernel
  is latency-bound at 13-21% of HBM BW — explicitly not their target).
  Ours reconstructs state by input replay for prefix caching. Rename ours
  or note the distinction before anyone imports the wrong mental model.
- **Structured output exists but isn't wired to the server.**
  `self_play/discovery/qwen_adapter.py:234-333` already implements
  two-phase xgrammar constrained decoding (unconstrained until
  TOOL_CALL_START, then bitmask-constrained) — for the self-play Qwen
  adapter only. `forge_server.py` exposes no grammar/json_schema/
  response_format field at all. The production gap is small: route a
  GrammarMatcher into the engine sampling loop (vocab 65536 → masks are
  cheap; llguidance ~50µs/mask, xgrammar <8µs after precompute). This is
  the recurring pattern — the R&D side-path has the feature, the serving
  path doesn't get it.
- **Sampler stack gaps + a real per-step perf bug.** `StandardDecoding`
  (decoding.py:81-177) has temperature/top-p/top-k/min-p/min-k and a
  last-64 repetition penalty — but: (a) the rep penalty is a Python loop
  doing `next_logits[:, tid] /= penalty` per unique recent token — **up
  to ~64 separate GPU kernel launches per decode step**; one
  `index_fill_`/`scatter` does it in one; (b) it's applied *after*
  temperature division — modern pipelines (llama.cpp default order:
  `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`)
  apply penalties to raw logits first; (c) no **DRY** — the only sampler
  that actually stops verbatim loops (rep penalty can't; it's
  sequence-level not token-level — suffix-match scan, oobabooga PR #5677,
  arXiv 2608.22761); no XTC/top-n-sigma/presence-freq. Cheap adds, real
  quality wins — and CREATIVE_SAMPLING is a promoted preset, so sampler
  quality is user-facing.
- **Lookahead + AirLLM already in-tree**: `decoding/lookahead_gate.py`
  (Jacobi n-gram verification) and `engine/airllm_streamer.py` exist —
  verify they're reachable from engine paths, not orphans (rule E).
- **Multi-model multiplexing is a natural niche.** ServerlessLLM's
  checkpoint format + pinned pool = our §1 roadmap; their multiplexing
  (10 models/GPU, sleep/wake) matches our HotSwapManager direction. llmux
  shows the trivial version (proxy + sleep/wake hooks over any backend).
  With 12 GB VRAM, "several quantized models + adapter zoo, fast switching"
  is a differentiator no big platform optimizes for.
- **PowerInfer-style activation locality** — hot neurons on GPU, cold on
  CPU, 11.7× over llama.cpp — needs ReLU-style activation sparsity; SwiGLU
  is denser, so direct gains are uncertain. Still worth a probe run (measure
  activation sparsity of V2's MLPs) before dismissing, per directive D.
- **kTransformers' lesson: split by tensor role, not by layer.** Their
  SOSP'25 result — 671B MoE on one 24 GB GPU (~14 GB VRAM + DRAM),
  27.8× prefill / ~3× decode vs llama.cpp — works because attention+KV
  (small, latency-critical, touched every token) stays on GPU while huge
  rarely-touched expert FFNs live in CPU RAM with async CUDA-graph
  scheduling. llama.cpp exposes the same thing as `--cpu-moe`. For us this
  generalizes beyond MoE: **the role-split applies to adapters and state
  too** — cold LoRA banks / SSM-state snapshots / doc libraries belong in
  the CPU tier (§6), not competing for VRAM. Our `hybrid_offload.py` and
  `cpu_kv_offload.py` should expose role-based placement, not per-layer.
- **bitnet.cpp vs our "BitNet" path.** Our int8 BitNet path is an
  approximation; Microsoft's bitnet.cpp runs *lossless* ternary (1.58-bit)
  mpGEMM — 2.4-6.2× over fp16 on x86 CPU, 100B model at human-reading
  speed on one CPU, GPU kernel now exists too. If we adopt real ternary
  weights for V2-derivative quants, the CPU becomes a genuine second
  inference device (directive D), e.g. running the §4.3 proxy model or
  serving the model when the GPU is busy training.
- **Attention kernels on sm120 (low priority — only 2 attn layers).**
  FA-2 runs on sm120 (mma.sync path, reduced throughput); FA-3 has no
  workstation-Blackwell port; the FA4 CuTeDSL sm120 PR adds paged-KV,
  split-KV decode, and fp8 KV decode (~1.6-1.9× at GQA≤4). PyTorch SDPA
  already auto-dispatches cuDNN/FA2 on this GPU — our `flash_attention()`
  wrapper (attention_ops.py:20-47) is fine; the interesting future option
  is **FlexAttention with the FA4 backend** for the custom attention keys
  (differential/GLA/GTA/CSA) since mask_mod/score_mod express their
  variants without bespoke kernels.

### 7.5.3 Architecture succession — where the Mamba line went

- The 2025 successor family is **Gated DeltaNet → Kimi Delta Attention**:
  GDN (NVIDIA, ICLR'25) adds a forget gate + delta rule, beats Mamba-2 on
  retrieval and long-context; KDA (Kimi Linear, arXiv 2510.26692) adds
  channel-wise gating and **beats full attention outright** in matched
  recipes — 6× decode throughput at 1M ctx, 75% less KV. Both have FLA
  Triton kernels and vLLM support. When the preset chain next evolves the
  SSM layers, GDN/KDA — not Mamba-3 speculation — is the evidence-backed
  target, and per directive A the port must be lossless-staged.
- **SuperBPE for the next tokenizer.** Superword tokens bridge whitespace:
  ~33% fewer tokens for the same text, +4.0% avg across 30 tasks (+8.2
  MMLU) at fixed model+compute, ~27% less inference compute. It's a
  permanent inference speedup baked into the tokenizer — but it changes
  the vocab, so it can only land with a full retrain/embedding re-init,
  never as a port. Log it for the next ground-up training round.

### 7.5.4 Third sweep — training-data hazards, eval, and frontier kernels

- **"Attention amnesia" — the biggest new threat to our SFT pipeline.**
  CoT-style SFT systematically degrades long-context recall in hybrid
  linear-attention models by up to ~58% (HypeNet-9B NIAH-S2@256K: 67.2% →
  9.4% after reasoning SFT, arXiv 2606.11052). The fix is training-free:
  **QK-Restore** restores pre-SFT query/key projections, recovering up to
  +19.8% while keeping reasoning gains. For ForgeAI this is squarely on
  the critical path — every reasoning-SFT run on V2 is likely quietly
  eroding the 262K-context claim. Action: add a post-SFT long-context
  probe to the training loop (cheap NIAH at 32K/64K), and keep a QK-restore
  recipe in the toolbox (snapshot W_q/W_k pre-SFT, restore after).
- **RULER validates the hybrid bet — and gives us the eval to run.**
  Jamba-1.5-large took **#1 on RULER** (>128K effective length, beating
  Gemini-1.5-pro and GPT-4). NVIDIA's controlled 8B study (arXiv
  2406.07887) shows *pure* Mamba lags transformers ~15 pts on 5-shot
  MMLU / in-context recall — the 2 attention layers in V2 are exactly the
  published fix. So the arch choice is right; what's missing is the
  measurement: run RULER (13 tasks, configurable depth) — or at minimum
  the passkey/ERF probe from §6 — before claiming any context length.
- **sm120 NVFP4 trap — verify which machine our "FP4" runs on.** vLLM on
  sm120 silently falls back to Marlin weight-only kernels (16-bit
  activations, FP4 tensor cores unused) with only a stderr warning
  (vllm#47749; "NVFP4 on a 5090 names two different machines"). Our packed
  IRI-FP4 loader is fine on the *storage* side — the question is whether
  the dequant/matmul path uses real FP4 tensor-core kernels or a
  dequantize-then-bf16 fallback (same class of issue as NC6's int8
  dequant-in-hot-path). A `torch._int_mm`-style kernel check + one
  throughput probe answers it.
- **Preference post-training: ORPO/KTO fit us better than our DPO.**
  `dpo_align.py` carries classic DPO — needs paired data *and* a frozen
  reference model (a second ~6 GB resident). On a 12 GB card the
  reference-free variants dominate: **ORPO** folds SFT+preference into one
  stage (no reference, half the pipeline); **KTO** learns from unpaired
  binary good/bad labels — which is *literally the shape of our self-play
  outcome data*; **SimPO** adds length-normalization, tightest VRAM of the
  paired methods. KTO is the natural fit for `infinite_loop` telemetry.
- **Megakernel numbers justify our direction.** Hazy Research: full-fusion
  hits ~78% HBM BW at bs=1 where vLLM/SGLang get ~50% → 1.5× end-to-end;
  even CUDA-graph launches retain ~1.3µs/teardown. Our `megakernel.py`
  (whole-step graph capture + torch.compile intra-layer fusion) is the
  pragmatic first rung; the true megakernel (interpreter + instruction
  schedule, their 70B-TP follow-up beat SGLang by 22%) is the R&D
  ceiling if P1 lands a real SSM kernel to fuse.
- **PDL (programmatic dependent launch)** — lets the next kernel start
  loading weights while the previous finishes — is a middle point between
  per-op kernels and megakernel; CUDA 12.0+, works inside graphs.

## 8. Priority fixes (the confirm-then-fix list)

| # | Finding | Sev | Effort |
|---|---------|-----|--------|
| P0 | Engine decode KV path is O(n²) `torch.cat`-per-token AND the ~20-strategy KV zoo never reaches the hot path (layers.py:271-272; strategies activated but unread; OOM-recovery + crash_recovery assumptions broken) | **Critical** | Med — route engine decode through `PreAllocatedKVCache`; make `KVCacheStrategy` backends wrap that real storage |
| P1 | SSM scan is a pure-Python per-token loop — the decode/prefill bottleneck (mamba_probe.py:135-189, self-described "not intended for production scale") | **Critical** | High — SSD chunked scan (mamba2 formulation works at any d_state) or Triton selective-scan kernel |
| P2 | Prefill materializes [B,T,65536] logits — 34.4 GB at 262K ctx → OOM; ~1.1 GB transient even at 8K (llm.py:656) | **Critical** | Trivial — `head(hidden[:, -1:])` on generation path |
| P3 | conv_state dropped on T>1 resumes — silent quality bug with default-on chunked prefill / prefix-cache suffix hits (mamba_probe.py:~223) | High | Low-Med — carry conv window across chunk boundary |
| P4 | LoRA stack: two key conventions (manual vs PEFT — PEFT files unloadable), non-atomic `copy_` mid-load, no `_gen_lock`, streaming path has no adapter support, no manifest, train/serve target-list asymmetry | High | Med — canonical key schema + manifest sidecar + slot-swap under lock + unify target lists |
| P5 | Per-token `.item()` EOS sync stalls pipeline every step (loader.py:1267) | Med | Trivial — batched/deferred stop checks |
| P6 | `StandardKVCache` preallocates full 262144 window + `clear()` zero-fills per generation (kv_backend.py:70-95) — moot if P0 lands paged storage | Med | Trivial |
| P7 | Per-tensor `.clone()` defeats fastsafetensors batching (loader.py:132,180) | Med | Low — persistent-buffer views |
| P8 | SSM state has no pool/evict/offload (~9.3 MB/seq pinned); `_last_prefill_recurrent` snapshot machinery dead for V2 | Med | Med — extend cpu_kv_offload to state slots |
| P9 | `wake()` L2 redundant `model.to(device)`; L1 sleep uses pageable CPU copies (engine_lifecycle.py:42,91-94) | Low | Trivial — pinned staging pool |
| P10 | `use_progressive_load` built but off; no per-phase load timing emitted | Low | Trivial — wire or delete (rule E); log t_arch/t_weights/assign |
| P11 | Missing regression test: spec-decode draft-rejection must verify SSM state rollback (`use_replay_ssm` default-on) | High | Med — lemon-mlx failure mode |
| P12 | sm_120 + `expandable_segments` / torch-2.11-cu130 memory regressions — add `torch.cuda.memory_stats()` assert to boot script | Low | Trivial |
| P13 | CUDA-graph + megakernel decode **exist and are wired** (`acceleration="cuda_graph"/"megakernel"`, engine_activation.py:367-381; `decoding/megakernel.py` captures the whole step) but are opt-in and off in the measured boot — and compatibility with the `presents`-dict KV path (growing list breaks static shapes) is unverified | High | Med — make graph capture work with the real KV path (preallocated cache), default-on for bs=1; cap capture batch to Mamba state slots (vLLM #34571 gotcha). Hazy Research shows the ceiling: even graphed launches cost ~1.3µs each — a true fused megakernel hits 78% HBM BW vs ~50% for split kernels |
| P14 | `muon_sf_plain` param grouping verified correct (hidden→Muon, embed/head+scalars→SF-AdamW, wd on); remaining delta vs Moonlight recipe: no per-shape update scaling (global LR scale instead) → LR doesn't transfer across widths | Low | Low — add match-RMS scaling, re-sweep LR once |
| P15 | Repetition penalty issues up to ~64 individual GPU kernel launches per decode step (Python loop over recent ids, decoding.py:121-123) AND runs after temperature division (wrong order vs modern pipeline) | Med | Trivial — single `index_fill_`, move before temp |
| P16 | Packing/varlen exists (attention_ops.py:50) but nothing resets conv/SSM state at packed doc boundaries → cross-doc state leak (same class of bug as P3) | High | Med — set `_conv_state_reset` at cu_seqlens boundaries |
| P17 | xgrammar constrained decoding implemented in `qwen_adapter.py` only — unreachable from forge_server (no grammar/response_format field) | Med | Low-Med — plumb GrammarMatcher into engine sampling loop |
| P18 | No post-SFT long-context recall check — attention amnesia (arXiv 2606.11052) predicts reasoning-SFT silently degrades hybrid NIAH up to ~58%; QK-Restore is the training-free fix | High | Low — NIAH probe at 32K/64K in eval loop + W_q/W_k pre-SFT snapshot for restore recipe |

Cross-references to still-open prior findings that intersect this audit:
NC6 (quant 3.5× — naive dequant-then-linear still in the decode hot path),
NC4 (preset lineage unenforced — matters directly for adapter binding
since `parent` is how adapters should inherit compatibility), NC5
(GPU-gated tests — P9's regression test belongs there but won't run on
CPU CI).
