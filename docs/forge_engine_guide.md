# ForgeEngine — Agent Guide

> **File**: `forge/engine/forge_engine.py`
> **Purpose**: Unified inference engine for ForgeAI models. Orchestrates
> KV cache strategies, decoding strategies, quantization, acceleration,
> and 40+ opt-in inference optimizations.

This document is the **single source of truth** for what ForgeEngine
actually contains. Read it before modifying the engine or adding new
features.

---

## Quick Start

```python
from forge.engine.forge_engine import ForgeEngine

engine = ForgeEngine.from_checkpoint(
    checkpoint="research/checkpoints/ForgeLM_V2.safetensors",
    config_name="forgelm_v2",
    tokenizer_path="research/checkpoints/forgelm_v2_tokenizer",
)
engine.activate_optimal()  # best defaults for RTX 5070
output = engine.generate("def fibonacci(n):", max_new_tokens=50)
```

---

## Public API (what agents should call)

| Method | Purpose |
|---|---|
| `from_checkpoint(checkpoint, config_name, tokenizer_path, device)` | Classmethod. Builds engine from a safetensors checkpoint. Auto-detects BitNet pre-quant, VRAM capacity, and KeyStack features. |
| `activate_optimal(**overrides)` | Activate best-practice strategy combo for current hardware. Calls `activate()` with curated defaults. |
| `activate(kv_cache=, decoding=, quantize=, acceleration=, ...)` | Configure runtime strategies. See **activate() parameters** below. |
| `generate(prompt, max_new_tokens, temperature, top_p, top_k, ...)` | High-level generation with prefix caching + finish-to-stop extension. Returns decoded string. |
| `generate_raw(prompt, max_new_tokens, ..., logits_processor=, ...)` | Low-level generation for self-play/agentic loops. Supports constrained decoding via `logits_processor`. No prefix cache, no finish extension. |
| `generate_stream(prompt, ...)` | Generator yielding decoded text chunks token-by-token for SSE streaming. |
| `benchmark(prompt, max_new_tokens, n_runs)` | Runs N timed generations, returns `{tokens_per_sec, latency_ms, ...}`. |
| `stats()` | Returns dict with generation count, KV cache info, decoding name, VRAM usage. |
| `diagnose()` | Full health report: stats + VRAM + warnings + recent errors. Non-invasive. |
| `bottleneck(prompt, max_new_tokens)` | Per-layer profiling to identify slowest transformer blocks. |
| `read_log(n, level, source)` | Read recent engine events (structured). |
| `read_output(n)` | Read recent generation outputs with metadata. |
| `sleep(level=1)` | Offload model to CPU (L1) or discard (L2) to free VRAM. |
| `wake()` | Restore model from sleep. |
| `is_awake` | Property. True if model is on GPU. |
| `vram_usage()` | Returns `{total_gb, free_gb, used_gb, model_weights_gb}`. |

### Convenience Properties

| Property | Returns |
|---|---|
| `config` | `getattr(model, 'config', None)` — the ModelConfig dataclass. |
| `dtype` | Dtype of the model's first parameter (bf16 by default). |

---

## `activate()` Parameters

### Core strategies (commonly used)

| Parameter | Type | Default | Values |
|---|---|---|---|
| `kv_cache` | str | `"paged"` | `standard`, `paged`, `rotorquant`, `hadamard_int4`, `compressed`, `streaming`, `snapkv`, `snapkv_4bit`, `paged_eviction`, `xquant`, `cpu_offload`, `s4r`, `hqe_kv` |
| `decoding` | str | `"standard"` | `standard`, `speculative`, `medusa`, `dspark`, `eagle3`, `mtp_selfspec` |
| `quantize` | str\|None | `None` | `None`, `int8`, `int4`, `fp8`, `w8a8`, `nvfp4` |
| `acceleration` | str\|None | `None` | `None`, `cuda_graph`, `airllm_streaming`, `megakernel`, `flex_decoding` |
| `kv_bits` | int | `4` | 4 or 8 (KV cache quantization bits) |
| `kv_cache_tokens` | int\|None | `None` | Limit KV cache allocation to N tokens (saves VRAM) |
| `mrl_keep_ratio` | float\|None | `None` | If <1.0, truncate to that fraction of dims (MRL) |
| `warmup` | bool | `True` | Pre-run dummy token to JIT-compile CUDA kernels |

### Innovation toggles (KeyStack features)

| Parameter | Default | Effect |
|---|---|---|
| `use_v0_warm` | `False` | V0 warm-start for KV cache (requires `value_residual` feature) |
| `use_progressive_kv` | `False` | Progressive KV (anchor + residual streams) |
| `use_spec_attn` | `False` | L1 Speculative Attention (57% attn cut, lossless) |

### Inference optimization toggles (opt-in, all default `False`)

These are advanced techniques wired into `activate()`. Each imports its
module lazily and applies the optimization. **All modules exist and are
fully implemented** — none are stubs.

#### Kernel fusion / compilation

| Parameter | Module | Effect |
|---|---|---|
| `use_compile` | `torch.compile` | 1.3-2x decode speedup (reduce-overhead mode) |
| `use_triton_conv` | `forge/decoding/triton_conv.py` | Fused Triton conv kernel (89% conv bottleneck cut) |
| `use_fused_qk_norm_rope_cache` | `forge/engine/attention/fused_qk_norm_rope_cache.py` | Fuse QK-Norm+RoPE+KV-cache-write (5-10% decode) |
| `use_block_fusion` | `forge/engine/graphs/block_fusion.py` | Per-block CUDA graph + compile (1.3-1.5x) |
| `use_compile_autotune` | `forge/engine/graphs/compile_autotune.py` | Auto-benchmark compile modes, pick best |
| `use_breakable_cuda_graph` | `forge/engine/graphs/breakable_cuda_graph.py` | Segmented graph capture (1.7-1.93x) |
| `use_foundry` | `forge/engine/graphs/foundry.py` | Template-based CUDA graph (10x faster startup) |

#### Attention optimizations

| Parameter | Module | Effect |
|---|---|---|
| `use_wavelength_pruning` | `forge/engine/attention/atflash.py` | ATFlash per-RoPE-wavelength pruning (37-48% attn) |
| `use_pod_attention` | `forge/engine/attention/pod_attention.py` | POD prefill-decode overlap (28% for hybrid batches) |
| `use_cosa` | `forge/engine/attention/cosa.py` | CoSA sparse attention (4.93x at 128K) |
| `use_seq_split` | `forge/engine/attention/seq_aware_split.py` | Seq-aware split for low-head-count (21-24% SM) |
| `use_compact_attn` | `forge/engine/attention/compact_attention.py` | CompactAttention for chunked prefill (2.72x at 128K) |
| `use_fa4` | `forge/engine/attention/fa4_attention.py` | FlashAttention-4 for Blackwell (1.3x prefill) |
| `use_aoh` | `forge/engine/kv/aoh_retmask.py` | Autonomy-of-Heads sparse attention (66% decode cut) |

#### KV cache compression / eviction

| Parameter | Module | Effect |
|---|---|---|
| `use_kara_kv` | `forge/engine/kv/kara_kv.py` | KARA sliding-window compression + Token2Chunk |
| `use_moment_kv` | `forge/engine/kv/moment_kv.py` | MomentKV moment-statistics eviction |
| `use_kvpop` | `forge/engine/kv/kvpop.py` | KVpop predictive online pruning |
| `use_conf_kv` | `forge/engine/kv/conf_kv.py` | CONF-KV confidence-aware eviction + mixed-precision |
| `use_mosa` | `forge/engine/kv/mosa.py` | MoSA mixture of sparse attention (27% better ppl) |

#### Prefix caching / scheduling

| Parameter | Module | Effect |
|---|---|---|
| `use_prefix_cache` | (built-in dict) | Cache KV for repeated prompt prefixes |
| `use_chunked_prefill` | `forge/engine/prefill/` (merged `__init__`) | Split long prompts into 512-token chunks |
| `use_hybrid_prefill` | `forge/engine/prefill/` (merged `__init__`) | Adaptive chunked prefill (+2-5% throughput) |
| `use_learned_prefix_cache` | `forge/engine/kv/learned_prefix_cache.py` | ML-guided prefix cache eviction (18-47% size cut) |
| `use_hotprefix` | `forge/engine/scheduler/hotprefix.py` | Hotness-aware GPU/CPU prefix promotion |
| `use_unified_radix` | `forge/engine/scheduler/unified_radix.py` | Hybrid prefix caching (HiCache L1/L2/L3) |

#### Speculative decoding extensions

| Parameter | Module | Effect |
|---|---|---|
| `use_adaptive_spec` | `forge/decoding/adaptive_speculative.py` | n-gram + EAGLE-3 combo (up to 4.9x on code) |
| `use_suffix_spec` | `forge/decoding/suffix_decoding.py` | Suffix decoding (training-free, code/RAG) |
| `use_peagle` | `forge/decoding/peagle.py` | P-EAGLE parallel speculative (1.69x over EAGLE-3) |
| `use_lookahead_gate` | `forge/decoding/lookahead_gate.py` | Lookahead quality gate (2.6-7.9x block acceptance) |
| `use_faser` | `forge/engine/attention/faser.py` | FASER dynamic spec length + early exit + frontier |

#### Scheduling / serving

| Parameter | Module | Effect |
|---|---|---|
| `use_triroute` | `forge/engine/scheduler/triroute.py` | Unified attention mode + KV bits routing |
| `use_corun` | `forge/engine/scheduler/corun.py` | CoRun deterministic inference (padding + fixed graph) |
| `use_fastserve` | `forge/engine/scheduler/fastserve.py` | FastServe skip-join MLFQ (6.1x throughput) |
| `use_libra` | `forge/engine/scheduler/libra.py` | Libra micro-request partitioning (1.91x goodput) |
| `use_kairos` | `forge/engine/scheduler/kairos.py` | Kairos SLO-aware prefill+decode scheduling |

#### Position encoding

| Parameter | Module | Effect |
|---|---|---|
| `use_jet_long` | `forge/engine/scheduler/jet_long.py` | Jet-Long dynamic bifocal RoPE (zero-shot 128K) |
| `use_rope_id` | `forge/engine/position/` (merged `__init__`) | RoPE-ID in-distribution high-freq rotation |
| `use_lerope` | `forge/engine/position/` (merged `__init__`) | LeRoPE learnable frequencies (3.4% less compute) |
| `use_lampe` | `forge/engine/scheduler/lampe.py` | LaMPE length-aware multi-grained positional encoding |

#### MoE optimizations

| Parameter | Module | Effect |
|---|---|---|
| `use_elbow_moe` | `forge/engine/scheduler/moe_optim.py` | Elbow dynamic top-k routing (5.3% latency cut) |
| `use_alloc_moe` | `forge/engine/scheduler/moe_optim.py` | Alloc-MoE budget-aware expert activation (1.34x) |
| `use_lda_moe` | `forge/engine/scheduler/moe_optim.py` | LDA distribution-consistent routing |

#### Quantization extensions

| Parameter | Module | Effect |
|---|---|---|
| `use_adamx` | `forge/quant/adaptive_quant.py` | AdaMX adaptive microscaling (83% MXFP4 loss removed) |
| `use_sharq` | `forge/quant/adaptive_quant.py` | SharQ sparse-dense FP4 activation quant (2.2-2.4x) |
| `use_mosaic_quant` | `forge/quant/adaptive_quant.py` | MosaicQuant inlier-outlier disaggregation 4-bit |

---

## Internal Methods (do NOT call from outside)

| Method | Purpose |
|---|---|
| `_detect_keystack_features()` | Reads checkpoint metadata to detect KeyStack transforms |
| `_warmup()` | Pre-compiles CUDA kernels with dummy forward pass |
| `_setup_airllm_smart()` | Sets up AirLLM layer-streaming for models exceeding VRAM |
| `_apply_quantization(mode)` | Applies weight-only quantization (int8/int4/fp8/w8a8/nvfp4) |
| `_sample_next_token(logits, temp, top_k, top_p, rep_penalty, gen_ids)` | Centralised sampling: greedy or top-k/top-p/rep-penalty |
| `_decode_with_kv(ids, logits, past_kv, ...)` | Decode from existing KV cache (prefix cache fast path) |
| `_get_stop_tokens()` | Token IDs for natural sentence/code boundaries |
| `_finish_to_stop(output_ids, ...)` | Continue generation to natural stopping point |
| `_generate_streaming(ids, max_new_tokens, temperature)` | AirLLM streaming generation (shard-per-forward) |

---

## What Was Removed (refactor log)

The following dead/wasteful code was removed in the 2026-08-20 refactor:

1. **`_materialize_meta_params()`** — module-level function, never called
   anywhere. ModelLoader handles meta→real materialization.
2. **`_convert_to_int8_storage()`** — method, never called.
   `from_checkpoint()` uses `convert_model_to_int8()` directly from
   `bitnet_b158_key.py`.
3. **`compare_strategies()`** — method, never called externally. It
   mutated engine state by calling `activate()` with hardcoded configs,
   which would silently override the user's chosen strategies.
4. **`use_flex_decoding` parameter + block** — no-op: imported
   `FlexDecodingAttention` but never instantiated or applied it. The
   real FlexDecoding is via `acceleration="flex_decoding"` which uses
   `FlexDecodingWrapper` (lines ~553-562).
5. **Redundant imports** — `import os` inside eagle3 block (os already
   imported at module top); `from pathlib import Path as _Path` inside
   `_detect_keystack_features` (Path already imported at module top).

## Bugs Fixed

1. **`self.config`** — 12 references in `activate()` used
   `getattr(self.config, ...)` but `self.config` was never set in
   `__init__`. Would raise `AttributeError` if any of these opt-in
   features were enabled. Fixed by adding a `config` property that
   returns `getattr(self.model, 'config', None)`.
2. **`self.dtype`** — referenced in CoRun scheduler setup but never
   set. Would raise `AttributeError` if `use_corun=True`. Fixed by
   adding a `dtype` property that returns the model's first parameter
   dtype.
3. **`_decode_with_kv` / `_finish_to_stop` ignored `top_p`** — both
   accepted a `top_p` parameter but never applied it in sampling. Now
   fixed via the shared `_sample_next_token` helper which always
   applies top-p when <1.0.
4. **`_decode_with_kv` / `_finish_to_stop` used `temperature == 0`**
   instead of `temperature <= 0` — negative temperatures would
   incorrectly enter sampling path. Fixed via shared helper using
   `<= 0`.

## Code Dedup

The sampling logic (temperature scaling, repetition penalty, top-k
filtering, top-p filtering, multinomial sampling) was duplicated 4
times across `generate_raw`, `generate_stream`, `_decode_with_kv`, and
`_finish_to_stop`. Extracted into a single `_sample_next_token()`
helper. All 4 call sites now use it, ensuring identical filtering
semantics.

---

## Architecture Notes

### Loading paths in `from_checkpoint()`

```
checkpoint size < VRAM free * 0.77?
├── YES → build_model_fast() → normal load (fast path)
└── NO  → build_model(meta) → _needs_streaming=True → AirLLM shard loading

Pre-quantized BitNet? (metadata _bitnet_prequant=1)
└── YES → build_model_fast() → convert_model_to_int8() → int8 storage
```

### Strategy application order in `activate()`

1. Quantization → 2. MRL → 3. QuaRot-KV → 4. V0 warm → 5. ProgressiveKV
→ 6. KV cache → 7. Decoding → 8. Acceleration → 9. torch.compile
→ 9b. Triton conv → 10. Prefix cache → 10b-10aq. Opt-in features
→ 11. Speculative attention → 12. Warmup

### KeyStack feature auto-detection

`_detect_keystack_features()` reads safetensors metadata and tensor keys:
- `value_residual_v0` in keys → `value_residual`
- `rotorquant_rotations` in keys → `rotorquant`
- `mtp_head.heads.0.weight` in keys → `mtp`
- `_airllm_streamable` in keys → `airllm`
- `_bitnet_prequant == "1"` in metadata → `bitnet_prequant`
- `quarot` and `mrl` are assumed applied by pipeline (always appended)

---

## Adding a New Inference Technique

To add a new opt-in technique to `activate()`:

1. **Create the module** under the appropriate subdirectory
   (`forge/engine/attention/`, `forge/engine/kv/`, etc.).
2. **Add a `use_<name>: bool = False` parameter** to `activate()`.
3. **Add a lazy import + application block** following the existing
   pattern (e.g. `10r`, `10s`, etc.):
   ```python
   self._my_feature = None
   if use_my_feature:
       from research.inference.attention.my_feature import MyFeatureWrapper
       self._my_feature = MyFeatureWrapper(...)
       self._my_feature.apply(self.model)
       print(f"  [ForgeEngine] MyFeature: active (...)")
   ```
4. **Add a docstring entry** for the parameter.
5. **Update this guide doc** — add the parameter to the appropriate
   table above.
6. **Use `self.config` / `self.dtype` properties** (not
   `getattr(self.model, 'config', None)` — the property is cleaner
   and already handles None).

### Anti-patterns to avoid

- **Do NOT** add a `use_*` parameter that just prints without applying
  anything (this was the `use_flex_decoding` bug).
- **Do NOT** duplicate sampling logic — use `_sample_next_token()`.
- **Do NOT** reference `self.config` or `self.dtype` without ensuring
  they're set (they are now properties, so this is safe).
- **Do NOT** add methods that mutate engine state as a side effect of
  a "comparison" or "benchmark" call (this was the `compare_strategies`
  bug).
- **Do NOT** re-import modules already imported at the top of the file.
