# ForgeEngine Critique DB

A living catalogue of weaknesses, dead code, bloat, and improvement
opportunities in `forge/engine/` (and adjacent surface: `forge_server.py`,
`scripts/`, `research/` dependency edges).

Each entry: `EC-NNN` id, severity, category, location, evidence, suggested
fix. Status: `open` | `fixed` | `wontfix` | `disputed`.

**Severities** — `high`: correctness/safety/memory risk or large dead
weight. `med`: maintainability drag, duplicated effort, missing hardening.
`low`: polish, naming, docs drift.

**Categories** — `dead-code`, `arch` (layering/structure), `weak-spot`
(fragile logic), `bug` (verified misbehavior), `dirty` (style/dup/magic
numbers), `bloat`, `missing-feature`, `test-gap`, `docs-drift`,
`security`.

Generated: 2026-10-20. Sources: manual audit + 6 parallel codebase
exploration agents covering engine core, subsystems (kv/quant/scheduler/
graphs/decoding), server/loader/scripts, and the GUI server surface
(`forge_gui_server`, `forge_gui/api`), plus online comparison against
vLLM/TensorRT-LLM/hipEngine and OWASP LLM/MCP guidance.

---

## Executive summary

The single biggest systemic problem: **the engine advertises far more than
it runs.** ~21 of 24 selectable KV-cache strategies are constructed,
logged, and never see a byte of KV. Whole subpackages (`safety/`,
`reasoning/`, `loader/`, `compat/`) are production-unwired. Innovation
slots, scheduler features, ~30 `use_*` flags, chunked prefill, hotswap
request overrides, session KV reuse, CUDA-graph captures, and most
spec-decode strategies are constructed-but-unconsumed or
dead-on-arrival — yet `diagnose()`/`info()` report them as active.
Second: **every remote surface is a shell** — the GUI server ships
`/tasks/launch` (raw argv exec), `/agent/runs` (caller-chosen workspace +
`approval_mode="none"` + interpreter-allowlisted `run_cmd`), merge/LoRA
endpoints (arbitrary read/write paths), a `..`-bypassable `DELETE
/models`, and a wipe-then-`extractall` backup restore — all
unauthenticated (EC-058). Third: **silent degradation is the house
style** — `.get(name, Standard)` factory fallbacks, ~200 broad `except`
sites, quant/KV fallback chains that change numerics while the config
still reports the requested mode, telemetry crediting speculative runs
that never speculated, and a five-layer GGUF failure stack. Fourth:
**the seams nobody exercised carry the bugs** — spec-decode KV rollback
missing, session KV reuse that never hits, `embed()` returning logits,
tool handlers that always error, `embed()`/`logprobs.bytes` violating
API contracts, route↔implementation drift killing whole GUI flows
(finetune, LoRA training, evolve-merge, lorebook) behind `{"ok":True}`,
`.env` readable by the model, a leaked HF token in a committed script.
Fifth: the servers are async shells around a synchronous engine — no
auth, no metrics, blocking work on the event loop, cancel tokens never
wired, orphaned workers.

---

## Summary table

| ID | Sev | Category | Location | Finding |
|----|-----|----------|----------|---------|
| EC-001 | high | arch | `forge/` ↔ `research/` | Incomplete `research/`→`forge/` migration; production imports legacy tree |
| EC-002 | med | weak-spot | `forge_engine.py:262` | Heavy `__del__` does CUDA work at GC time |
| EC-003 | med | arch | `forge/engine/__init__.py` | Package facade exports nothing |
| EC-004 | low | docs-drift | `forge_engine.py:12` | Docstring says "42 flags", actual 44; stale research/ paths |
| EC-005 | med | dirty | `activation.py` vs `engine_activation.py` | Confusing near-identical filenames |
| EC-006 | high | dead-code | `kv_backend.py:354-439`, `engine_generation.py:995-1039` | ~21/24 KV strategies runtime-inert |
| EC-007 | high | bug | `engine_activation.py:936-966` | Low-VRAM `cpu_offload` rescue is a placebo |
| EC-008 | high | bug | `decoding.py:879` | `self_speculative_sparse` crashes — `sparse_k` not a model arg |
| EC-009 | high | bug | `decoding.py:367-409` | `ngram_speculative` never rolls back past_kv on rejection |
| EC-010 | high | bug | `decoding.py:476-510` | `external_draft_speculative` same + wrong next-step logits index |
| EC-011 | high | dirty | `prefix_cache.py:163 vs 236` | `cache_prompt_prefix` defined twice; first is dead |
| EC-012 | high | dirty | `engine_activation.py:903-920` | `ternary_ptq` runs calibration twice |
| EC-013 | high | dead-code | `safety/`, `reasoning/`, `loader/`, `compat/` | Four subpackages production-unwired |
| EC-014 | high | dead-code | `activation.py:53,106`, `session_manager.py:77` | `use_spec_attn`/`use_progressive_load` flags with no consumer |
| EC-015 | high | bug | `innovations.py:289-354` | `ProgressiveKV` discards residual scale; never invoked |
| EC-016 | high | dead-code | `engine_activation.py:197-217` | `quarot_kv`/`v0_warm`/`progressive_kv` slots are dashboard props |
| EC-017 | high | dead-code | `feature_registry.py:444-503` | Many registry handlers write attrs nothing reads |
| EC-018 | high | weak-spot | `crash_recovery.py` | Signal-handler deadlock risk; KV snapshot write-only |
| EC-019 | high | weak-spot | `kv_backend.py:436-438`, `decoding.py:1706` | Unknown strategy names silently become `standard` |
| EC-020 | high | bug | `session_manager.py:591-617` | B==1 batch path drops `top_k`/`repetition_penalty`/`stop` |
| EC-021 | high | dead-code | `session_manager.py:400-405` | `use_feather_scheduler` builds scheduler never consulted |
| EC-022 | med | weak-spot | `engine_generation.py:1017-1039` | KV compaction no-ops on preallocated cache; errors swallowed |
| EC-023 | med | bug | `kv/metakv.py:127-131` | `MetaKVCache.append` drops `token_ids` |
| EC-024 | med | dirty | `engine_generation.py` vs `decoding.py` | Two parallel decode loops kept in sync by convention |
| EC-025 | med | dirty | `engine_common.py` vs `decoding.py` | `_min_k_filter`/`_dry_penalties` duplicated (claimed cycle doesn't exist) |
| EC-026 | med | arch | `session_manager.py` vs `session_cache.py` | Two session systems, overlapping names, dead fields |
| EC-027 | med | security | `tool_security.py:414-502` | File blacklist is write-only; reads of `.env` allowed; protected match over-broad |
| EC-028 | med | weak-spot | `forge_engine.py:194-208` | `__init__` does disk I/O + installs signal handlers unconditionally |
| EC-029 | med | arch | `forge/engine/quant/` vs `forge/quant/` | Two quantization trees, split authority |
| EC-030 | med | dirty | `engine/quant/novel_quant_rNN.py` | Versioned siblings violate derive-don't-fork; cross-version imports |
| EC-031 | med | weak-spot | `engine_activation.py:615-699` | Quant fallback chain silently downgrades numerics; `self.quantize` lies |
| EC-032 | med | dirty | `decoding.py:346,377,382,841` | Discarded expressions (dead work) |
| EC-033 | med | bug | `decoding.py:413-426` | `_ngram_lookup` mixes 0-dim tensor into int list |
| EC-034 | med | bug | `library.py:236-240,376-389` | Pre-tokenization assumes flat `input_ids`; budget accounting diverges |
| EC-035 | med | dirty | `engine_common.py:116-150` | `_min_k_filter` sorts vocab twice per token |
| EC-036 | med | arch | `forge/engine/mamba3.py` | Model layer lives in engine package (layering inversion) |
| EC-037 | med | dirty | `prefix_cache.py` + `kv/learned_prefix_cache.py` | Three prefix-cache impls + legacy dict path |
| EC-038 | med | security | `crash_recovery.py:333-336` | `torch.load(weights_only=False)` fallback — ACE surface |
| EC-039 | low | dirty | `engine_common.py:40` vs `decoding.py` | EOS sets hardcoded twice, already drifted |
| EC-040 | low | dirty | `session_manager.py:666-673` | Dead `pass` block for chunked prefill |
| EC-041 | low | weak-spot | `session_manager.py:509` | Batch window slices before grouping — one model can starve others |
| EC-042 | low | weak-spot | `kv/beacon_kv.py:232-270` | Per-layer Python beacon loop fires even under budget |
| EC-043 | low | weak-spot | engine-wide | ~200 broad `except` sites convert real bugs into warn logs |
| EC-044 | low | dirty | `cascade.py:102-145` | Magic-number heuristic routing, no sweep backing |
| EC-045 | low | dirty | `kv/` generally | Strategy params scattered as magic literals; no defaults module |
| EC-046 | low | weak-spot | `engine_generation.py:1764` | `kv_cache.info()` is the only interface the engine uses — status lies |
| EC-047 | low | dirty | `forge_engine.py` + mixins | `# noqa: F401` re-export sprawl hides dead imports |
| EC-048 | low | dead-code | `airllm_streamer.py` | Third parallel decode implementation |
| EC-049 | low | missing-feature | `session_manager.py:159-161` | `TaskSession.kv_cache`/`kv_seq_len` stub fields never populated |
| EC-050 | high | security | `scripts/distill/extract_gemma4_text.py:25` | **Live HF API token hardcoded in committed file** |
| EC-051 | high | bug | `forge_loader.py:335-340` | Unknown GGUF dtypes → wrong-format dequant or silent `torch.zeros` |
| EC-052 | high | bug | `forge_loader.py:193-214` | Q4_0 dequant interleaves nibbles — scrambled weights, loads "succeed" |
| EC-053 | high | bug | `forge_loader.py:230-266` | Q4_K drops min term, unpacks scales wrong |
| EC-054 | high | bug | `forge_loader.py:269-301` | Q6_K reads fields at wrong offsets — garbage output |
| EC-055 | high | bug | `forge_loader.py:316-321` | Unmapped dtypes return raw bytes reshaped as weights |
| EC-056 | high | bug | `airllm_streamer.py:237-264` | AirLLM decode is context-free + reloads all shards per token |
| EC-057 | high | weak-spot | `forge_server.py:1885-1899,2001-2015` | Async SSE handlers synchronously iterate blocking generator — one stream stalls all endpoints |
| EC-058 | high | security | `forge_server.py:566-573` | 0.0.0.0 + CORS `*` + zero auth on mutation endpoints |
| EC-059 | high | weak-spot | `engine_checkpoints.py:933-956` | GGUF load swallows per-tensor failures + `strict=False` |
| EC-060 | high | weak-spot | `forge_loader.py:600-604` | Unbounded `-GGUF` suffix recursion on HF fallback |
| EC-061 | high | bug | `model_registry.py:179,228` | Token stats double-count quadratically |
| EC-062 | high | bug | `forge_server.py:1644-1663` | `/v1/engine/kv-cache-type` resets whole activation; ignores `cache_type_v` |
| EC-063 | high | weak-spot | `forge_loader.py:484-507` | `detect_format` misroutes local `qwen_*`/`meta_*` paths; checks magic before `.yaml` |
| EC-064 | high | weak-spot | `forge_loader.py:509-563` | `load()` returns `model=None` for GGUF — violates contract |
| EC-065 | high | missing-feature | `forge_server.py` | No request-ID tracing, no `/metrics`, shallow `/health` |
| EC-066 | high | missing-feature | `forge_server.py:1941-2032` | No client-disconnect/streaming-abort handling |
| EC-067 | med | dead-code | `hotswap.py:300-312` | Per-request override system never invoked (getter pops) |
| EC-068 | med | dead-code | `hotswap.py` | Six convenience setters never called |
| EC-069 | med | dirty | `hotswap.py:348-351` | `apply_pending` assigns `_current` twice |
| EC-070 | med | dead-code | `engine_generation.py`, `engine_diagnostics.py` | Public APIs with no production callers (`generate_gated`, `apply_flux_packets`, `attach_head`, `profile_bandwidth`) |
| EC-071 | med | dirty | `hotswap.py` `update_from_dict` | `key.replace("set_","")` not prefix-anchored — mangles `reset_*` |
| EC-072 | med | dirty | `forge_server.py:746` | `/health` reaches into `BatchQueue._pending` |
| EC-073 | med | weak-spot | `session_manager.py` | Batched "streaming" delivers whole response as one chunk |
| EC-074 | med | missing-feature | `requirements.txt`, `pyproject.toml` | Serving deps (fastapi/uvicorn/pydantic/gguf) undeclared |
| EC-075 | med | docs-drift | `forge_loader.py:1-9` | Docstring promises mmap zero-copy / <100ms; code eagerly dequantizes |
| EC-076 | med | test-gap | `forge_server.py` | Entire HTTP surface untested except `/v1/systemone` |
| EC-077 | med | dirty | `forge_gui/api/process_manager.py:42-58` | GUI presets launch archived/nonexistent scripts |
| EC-078 | med | dirty | `scripts/download_datasets.py:23`, `download_github_code.py:7` | Data written into `scripts/`; `.env` resolved from wrong depth |
| EC-079 | med | bloat | `scripts/generate_synthetic_data.py` | 905 lines, only script >800; stale `-m research.data` docstring |
| EC-080 | med | bloat | `scripts/download_*.py` (9 files) | Nine downloaders reimplement shard rolling/resume/token loading |
| EC-081 | med | bloat | `scripts/bench_*.py` | 8 active + 6 archived benches with overlapping scope |
| EC-082 | med | bloat | `scripts/exp_*.py` (17 files) | 8 experiment drivers unreferenced by tests/docs |
| EC-083 | med | dead-code | `scripts/smoke_r50_*.py` | Superseded by `tests/unit/test_r50_rd_features.py` |
| EC-084 | med | dirty | `scripts/test_*.py` (3 files) | pytest-named scripts never collected (`testpaths=["tests"]`) |
| EC-085 | med | test-gap | engine-wide | airllm_streamer, forge_loader, model_registry, hotswap, lora, merging, library, engine_tools, innovations, forge_server untested |
| EC-086 | low | missing-feature | `forge_server.py:799` | Request-ID header on exactly one route |
| EC-087 | low | test-gap | `pyproject.toml:98-102` | gpu/integration/slow markers registered, never applied |
| EC-088 | low | docs-drift | `docs/BUG_LOG.md:509-510` | Claims airllm_streamer deleted — it's wired and reachable |
| EC-089 | low | docs-drift | `forge_server.py:736` | `/health` docstring claims Prometheus output — returns plain JSON |
| EC-090 | low | docs-drift | `scripts/*` docstrings | Usage strings reference pre-move `research/data/` paths |
| EC-091 | low | dirty | `scripts/distill/gen_roc_data.py:25` etc. | Machine-specific absolute paths hardcoded |
| EC-092 | low | dirty | repo root | Stray artifacts: `parity_*.txt`, file named `None`, committed smoke LoRA ckpt |
| EC-093 | high | bug | `engine_tools.py:1103-1126,1145,1157` | `self._ast`/`self._random` never assigned — math_eval/calc/random_number/chance always error |
| EC-094 | high | bug | `session_cache.py:201-214`, `engine_sessions.py:80-115` | Session KV reuse never hits; generated tokens lost from history |
| EC-095 | high | bug | `engine_checkpoints.py:1102-1116,426-428` | Streaming fallback produces an engine that cannot generate |
| EC-096 | high | bug | `engine_generation.py:1310-1318` | `embed()` returns logits, not hidden states — rerank/semantic callers silently wrong |
| EC-097 | high | security | `engine_tools.py:815-870` | `file_read`/`file_list` bypass all security checks; `.env` readable → key exfil |
| EC-098 | high | weak-spot | `engine_activation.py:90-91,201-204` | `quarot`/`mrl` unconditionally marked present → wrong KV semantics on unqualified ckpts |
| EC-099 | high | bug | `engine_checkpoints.py:73-74,155-157` | Activation kwargs silently dropped for GGUF checkpoints |
| EC-100 | high | weak-spot | `hotswap.py:316-353,389-393` | Failed reactivation retries forever on every `generate()` |
| EC-101 | high | bug | `engine_generation.py:817-819` | Batch path counts characters, not tokens — stats inflated ~4x |
| EC-102 | high | weak-spot | `engine_generation.py:1570,1622-1629` | `generate_stream` RLock released in generator `finally` — cross-thread GC → engine deadlock |
| EC-103 | med | bug | `engine_activation.py:387-398` | `_suffix_mem_for` writes `mem.gate` on the shared sidecar — per-request override leaks globally |
| EC-104 | med | weak-spot | `engine_checkpoints.py:891-958` | GGUF load hardcodes rope/eps/bias; stages whole fp16 model twice → OOM |
| EC-105 | med | weak-spot | `engine_checkpoints.py:764-773,799-801` | int8 head dequantizes entire weight every forward; dead local; stale fp weight re-tied |
| EC-106 | med | weak-spot | `engine_checkpoints.py:1042,856` | `_load_quantized`/`_load_prequant` clobber detected `keystack_features` |
| EC-107 | med | weak-spot | `engine_checkpoints.py:218-236` | Broad `RuntimeError` → streaming fallback masks real load bugs |
| EC-108 | med | dead-code | `engine_activation.py:544-551` | Chunked prefill fully wired but never invoked |
| EC-109 | med | bug | `engine_generation.py:884,953-957` | `generate_with_tools` NameError on `max_tool_rounds<=0`; no budget/timeout; re-prefills transcript each round |
| EC-110 | med | bug | `engine_sessions.py:73-78`, `session_cache.py:172-174` | Session path skips BOS, mutates state outside `_gen_lock`, unbound `logits` on empty delta, double `_record_generation` |
| EC-111 | med | bug | `engine_merging.py:111-146,45-51` | 1-parent crossover IndexError; `_swap_weights` strict=False hides mismatched keys |
| EC-112 | med | weak-spot | `engine_lora.py:76-106,120-139` | Random-init adapters stay attached when load matches 0 tensors; unload mutates module tree mid-iteration |
| EC-113 | med | bug | `engine_generation.py:155-168` | `mem_bias` silently dropped when `learn=False` |
| EC-114 | med | weak-spot | `engine_generation.py:1711-1745` | OOM recovery leaves inconsistent state; dead `getattr`; permanent silent downgrade |
| EC-115 | med | weak-spot | `hotswap.py:355-393` | Reactivation resets unspecified feature flags but leaves stale instances active |
| EC-116 | med | bug | `engine_sessions.py:32,46` vs `forge_gui_server/routes/core.py:318-343` | GUI calls `ttl_s=`/`max_tokens=` — mixin takes `ttl`/`max_new_tokens` → TypeError; SessionState not JSON-serializable |
| EC-117 | med | weak-spot | `crash_recovery.py` | Swallowed handler-install/snapshot failures; no `snapshots_failed` counter |
| EC-118 | med | dirty | `engine_generation.py:355,212-214` | OutputHistory records the library-injected prompt, not the user's |
| EC-119 | med | weak-spot | `engine_tools.py:1364-1377,1144-1154,1101-1132,1452-1488` | Tool handlers trust model args — unbounded `range(count)`, `9**9**9`, uncapped `rglob`, `args={}` on malformed JSON |
| EC-120 | med | weak-spot | `engine_lifecycle.py:59-64,82-86,141-143` | Flux sleep leaks GPU tables / no-ops; swallowed unlink; wake failure wedges `_awake` |
| EC-121 | low | bug | `engine_generation.py:251-268,1196` | EOS counted in `n_gen` on schema/logprobs path, excluded on standard path |
| EC-122 | low | weak-spot | `engine_generation.py:1142-1155,347-352` | `eos` custom ids replace defaults; `stop=[""]` → empty output; `context_limit=0` treated unset; session SamplingParams omits fields |
| EC-123 | low | dirty | `engine_checkpoints.py:598-634` | Dead `list(shapes)` expression; sparse GGUF config validation |
| EC-124 | low | dirty | all mixins | `from .engine_common import *` + explicit re-imports; most names unused (lora uses 0 of 13) |
| EC-125 | low | dead-code | `engine_activation.py:971-974` | `CREATIVE_SAMPLING` never referenced; docstring claims S4R default, code uses rotorquant |
| EC-126 | low | weak-spot | `engine_activation.py:599-611` | Warmup leaves mutable conv/KV residue visible to live-state readers |
| EC-127 | low | docs-drift | `session_cache.py:12-14,47-59,251-275` | Docstring claims "radix tree prefix matching" — it's a dict + linear check; TTL eviction throttled to every 8th call |
| EC-128 | low | weak-spot | `engine_lifecycle.py:173-198` | `vram_usage` reports 0 GB for FluxLM table weights |
| EC-129 | low | weak-spot | `engine_common.py:241`, `engine_checkpoints.py:118`, `model_registry.py:118` | Hardcoded relative tokenizer path — breaks when cwd isn't repo root |
| EC-130 | low | weak-spot | `engine_diagnostics.py:27-59` | `benchmark()` doesn't hold `_gen_lock` across the measurement loop |
| EC-131 | low | weak-spot | `forge_engine.py:246-250` | `_log` falls back to raw `print()` when no root handlers — bypasses log pipeline |
| EC-132 | low | bug | `engine_generation.py:1747-1760` | `_safe_decode_ids` doesn't clamp negative ids |
| EC-133 | low | weak-spot | `engine_diagnostics.py:61-79` | `stats()`/`diagnose()` omit suffix_mem, hotswap, session cache, tools |
| EC-134 | low | weak-spot | `engine_checkpoints.py:839-843` | QK-norm identity scan assumes `model.blocks` — AttributeError on non-Forge archs |
| EC-135 | high | bug | `engine_generation.py:301-312,343-352` | `stop=` never stops decode — post-hoc `find()` truncation only; `decoding.generate` never receives it (plan 0.4) |
| EC-136 | high | bug | `batched_decoding.py:97-127` | Left-pad batch prefill pollutes Mamba/conv state — mask guards attention only (plan 0.5) |
| EC-137 | med | bug | `prefix_cache.py:373-385` | `_slice_past_kv` passes SSM dict state untouched — wrong position on any mid-entry hit |
| EC-138 | high | missing-feature | `structured/xgrammar.py` | "xgrammar" is a char-level first-char FSM — incorrect constrained decoding; no real grammar dep |
| EC-139 | high | missing-feature | `mamba_probe.py:164-189`, `llm.py:350`, `scheduler/async_d2h.py` | Launch-bound decode: Python SSM scan, ~800 launches/token; `compile_decode_step` + `async_d2h` exist, never wired |
| EC-140 | med | weak-spot | `engine_generation.py:1195,1590` etc. | Per-token `.item()`/`.tolist()`/`decode` syncs; no device-resident sampling |
| EC-141 | med | bug | `engine_generation.py:1747-1760` | Per-token `decode` emits mojibake for multi-byte UTF-8 chars; no incremental detokenizer |
| EC-142 | high | missing-feature | prefix cache (plan Phase 3) | Chunked mamba-state prefix snapshots — novel capability vLLM lacks |
| EC-143 | high | missing-feature | decode path (plan Phase 4) | Unified state pool + single-graph decode — ~800 launches → 1 replay |
| EC-144 | med | missing-feature | `session_manager.py` (plan Phase 5) | Continuous batching with per-row state slots |
| EC-145 | high | missing-feature | `forge/quant/`, `engine/quant/` | No fused dequant-GEMM kernels — all paths dequantize-then-matmul (open since Sept-9); Marlin/GemLite/`_scaled_mm` run on SM120 |
| EC-146 | med | missing-feature | attention path (plan 8B) | No split-KV/FlashDecoding — SMs idle at bs≤8+MQA decode |
| EC-147 | high | bug | `attention_ops.py:50` + conv/SSM layers | Varlen packing: conv/SSM state never resets at doc boundaries → cross-doc leak (P16) |
| EC-148 | med | weak-spot | nvfp4 quant path | sm120 NVFP4 trap — must verify tensor cores engaged, not silent dequant (vLLM #47749) |
| EC-149 | low | missing-feature | scheduler (plan 8C) | Cheap wins absent: recompute-preemption, `max_num_batched_tokens`, prefix-aware admission, `stream_interval` |
| EC-150 | low | missing-feature | runtime/platform (plan 8D) | `expandable_segments` Windows check, ReBAR verify, `cudaMallocAsync` probe — unverified |
| EC-151 | med | missing-feature | sampling/determinism | No batch-invariance/determinism mode (Thinking Machines ops exist; vLLM ships `VLLM_BATCH_INVARIANT`; +18.1% via invariant BMM) — matters for self-play RL rollouts |
| EC-152 | high | missing-feature | `forge_server.py` | Server-endpoint gap list: /tokenize, /detokenize, /props, /apply-template, /version, /metrics, ready-vs-live health, latency headers, abort, graceful drain, request logging, stream_options, prompt_token_ids, n>1 shared-prefill, min_tokens, stop_token_ids, presence/frequency_penalty, logit_bias, response_format, priority/timeout/backpressure, HF JIT load+TTL, GGUF hub pull, revision pinning, response cache, batch file runner, bench_serving, chat_template_kwargs, aliases, env config, OTel |
| EC-153 | med | docs-drift | `docs/ENGINE_IMPROVEMENT_PLAN.md` | Plan (9/25, "awaiting execution") overlaps ~20 EC entries — sync statuses; some DONE markers stale |
| EC-154 | med | missing-feature | `engine_generation.py:1310` | No pooling option/normalize on `embed()`; no `/rerank` (and EC-096 makes embeds wrong anyway) |
| EC-155 | low | weak-spot | `session_cache.py`, `prefix_cache.py` | Caches are entry-count/uncapped, not MiB-budgeted (llama.cpp `--cache-ram` model) |
| EC-156 | low | weak-spot | engine/server | Per-request `seed` plumbed partially only; no end-to-end determinism story |
| EC-157 | low | bug | `engine_generation.py:1914,1920,1931,1938,1962,1968` | `logprobs[].bytes` returns token id, not UTF-8 bytes (OpenAI spec violation) |
| EC-158 | med | security | `engine_tools.py`, `tool_security.py` | Tool design vs OWASP/MCP guidance: no read/write annotation tiers, secrets reachable via model tools, `sandbox.json` in a writable path |
| EC-159 | low | weak-spot | `engine_generation.py:1943-1969` | Legacy logprobs path re-runs full decode step-by-step for per-step logits |
| EC-160 | med | security | `tool_security.py` + plan 6.2 | `sandbox.json` under writable `research/data`, not in `PROTECTED_ENGINE_FILES` — model can edit its own security config |
| EC-161 | low | weak-spot | `model_registry.py` | VRAM accounting uses declared 1.5×ckpt budget not measured `mem_get_info` delta; `is_awake` misses external sleeps (plan 6.5) |
| EC-162 | med | weak-spot | `crash_recovery.py` | Snapshot writes not atomic (no temp+rename+checksum) — torn state on the crash it exists for (plan 6.6) |
| EC-163 | med | security | `engine_tools.py` + tool loop | Untrusted tool outputs (web/file contents) fed back untagged — indirect prompt-injection path into tool calls |
| EC-164 | high | bug | `kv/mosa.py:205-212`, `scheduler/triroute.py:~275-293` | Sparse-attn wrappers drop `past_key_value` on both branches — decode attends to wrong/stale positions (plan 0.6 residual) |
| EC-165 | high | bug | `decoding.py:636-644,676-685`, `mtp.py:377`, `engine_checkpoints.py:80` | `mtp_selfspec` dead-on-arrival: `hasattr(model,"mtp_head")` never true (attr is `mtp_module`); `heads[0]=None` breaks state_dict; keystack detection un-fireable |
| EC-166 | high | bug | `decoding.py:1742-1792` | EAGLE-3 drops accepted draft tokens from output, never rolls back KV, wrong fused index on rejection, single hardcoded EOS |
| EC-167 | high | bug | `decoding.py:516-528,534-546` | `medusa` crashes at activation (required arg); `dspark` crashes mid-generation (`dspark_head=None`) |
| EC-168 | high | bug | `forge/decoding/medusa.py:240-299`, `dspark.py:789-814` | `medusa_generate` never runs the Medusa heads (`hidden=out[0]` is logits); both helpers O(T²) re-forward per step, no use_cache |
| EC-169 | high | bug | `forge/model/kv_cache.py:208-211` | `unpack_output_with_kv` returns `loss`/`hidden` as `past_kv` for 2/3-tuples — contract bug at 49 call sites |
| EC-170 | high | bug | `graphs/breakable_cuda_graph.py:122-227` | Can never capture (`position_ids` kwarg doesn't exist — TypeError swallowed) but reports `_captured=True`; prefill capture is `pass` |
| EC-171 | high | bug | `graphs/foundry.py:200-270`, `scheduler/corun.py:80-136` | Same `position_ids` TypeError; "materialization" still captures; dead `eval(dtype_str)` |
| EC-172 | high | weak-spot | `engine_activation.py:463`, `engine_generation.py:1764` | Telemetry reports requested strategy, not the fresh `StandardDecoding` that actually ran |
| EC-173 | high | bug | `test_time_scaling.py:140-177`, `engine_common.py:312` | FirstFinishSearch can't short-circuit (`with` waits all futures); serial under `_gen_lock` anyway; `seed()` is a no-op |
| EC-174 | med | weak-spot | `engine_generation.py:1082-1096,1184-1194`, `engine_common.py:182-201` | DRY penalty O(n·last_n) per token; rep-penalty does per-token `set()`+H2D transfer |
| EC-175 | med | bug | `engine_generation.py:1657-1659,1119-1125`, `decoding.py:164-167` | `top_k=0` rejected by validation but dead `else` branch exists; `top_k>vocab` crashes strategy path, clamped on engine path |
| EC-176 | med | weak-spot | `decide.py`, `decision_head.py:375-410` | decide()/PRM materialize full [B,T,V] logits to read hidden states — ~150GB transient at batch scale → OOM |
| EC-177 | med | bug | `engine_generation.py:1238-1241` | `_forge_last_kv`/`_forge_last_logits` written on every generator exit incl. aborts → cross-request state leak |
| EC-178 | med | dead-code | `runtime/vram_manager.py`, `engine_activation.py:298` | 6/7 VRAMManager methods dead; `profile_after_warmup` is a third O(n²) decode loop; fresh instance per activation |
| EC-179 | med | weak-spot | `decoding.py:1504-1533` | SpecBound re-forwards growing suffix per draft token (O(d²)); reinstalls conv prefix each iter; 2 `.item()` syncs/cycle |
| EC-180 | low | weak-spot | `decision_head.py:353-372` | PRM step boundaries assume incremental tokenization == joint tokenization (off-by-one risk) |
| EC-181 | low | dead-code | `feature_registry.py:78-83`, `engine_checkpoints.py:364-371` | `_adaptive_spec` write-only; `_load_quantize` bulk-enables ~8 inert features "on by default" |
| EC-182 | low | dead-code | `graphs/block_fusion.py`, `engine_checkpoints.py:399` | `torch.compile(block)` result discarded; `use_block_fusion` auto-enabled but does nothing |
| EC-183 | low | bug | `decoding.py:633,665` | MTPSelfSpec EOS check reads `model.eos_token_id` — attr doesn't exist → would never stop |
| EC-184 | high | security | `forge_gui_server/routes/chat.py:231-280`, `api/agent_tools.py` | `/agent/runs` accepts caller `workspace` + `approval_mode="none"` → unauthenticated arbitrary file write + `python -c` exec |
| EC-185 | high | security | `forge_gui_server/routes/system.py:38-48`, `procs.py:90-97` | `/tasks/launch` runs caller `command` via `create_subprocess_exec` — unauthenticated arbitrary exec |
| EC-186 | high | security | `forge_gui_server/services/support.py:547-588` | Backup restore: wipes project then `extractall` with no member validation — zip-slip writes anywhere |
| EC-187 | high | security | `forge_gui_server/routes/core.py:582-601` | `DELETE /models` `..` traversal → delete any model-extension file on host; sidecar names never match real files |
| EC-188 | high | bug | `routes/core.py:480-506,664-668`, `services/lora.py:139-168` | Merge endpoints allow arbitrary read/write; `evolve_merge` always TypeErrors (wrong kwargs); `hot_swap=` silently swallowed by `**kwargs` |
| EC-189 | high | weak-spot | `engine_rt.py:141-181` | `load()`→`unload()` runs `sleep(2)`+`empty_cache` on the event loop — model switch stalls all endpoints + /ws |
| EC-190 | high | weak-spot | `gen.py:78`, `chat_loop.py:534`, `routes/chat.py` | Orphan generation: futures discarded, `StreamCancel`/`ChatCancel` never wired — work (incl. tool exec) outlives dead clients |
| EC-191 | med | bug | `forge_gui/api/lora_training_trigger.py:161-234` | `train_lora` dead 3 ways: wrong CLI flags, path passed as `conv_ids`, nonexistent `proc_mgr.processes()` |
| EC-192 | med | bug | `forge_gui_server/routes/train.py:108-151` | `/finetune/*` points at `research/training/runners/sft_train.py` — file moved to `forge/` → every launch crashes |
| EC-193 | med | bug | `forge_gui_server/routes/system.py:176-184` | `POST /lorebook` always 500 — calls `add(tags=, description=)` kwargs the signature lacks |
| EC-194 | med | bug | `routes/chat.py:178-182`, `api/lorebook.py` | Lorebook injection computed then discarded (`str` extended as messages); still `save()`s full JSON per message |
| EC-195 | med | weak-spot | `routes/core.py:116-119`, `engine_rt.py:346-389` | GUI `reactivate` rebuilds full config from partial dict (EC-062-class #3); `active_config` reports requested not applied |
| EC-196 | med | bug | `session_manager.py:80-114` | `TaskBootConfig.differs_from` checks 4/14 fields, can't see revert-to-default; `apply_to` wipes other flags (EC-062-class #4) |
| EC-197 | med | weak-spot | `session_manager.py:419-551`, `forge_server.py:1474` | BatchQueue.stop doesn't drain/resolve pending; `_push_stream` futures dropped; HTTP timeout leaves orphaned history append |
| EC-198 | med | bug | `forge_server.py:1402-1433`, `session_manager.py` | No per-task serialization — concurrent messages corrupt session history order/context |
| EC-199 | med | weak-spot | `model_registry.py:333-395` | Registry lock held across `sleep()`/`wake()` → one in-flight decode freezes all registry ops; LRU evicts busy engines |
| EC-200 | med | weak-spot | `engine_rt.py:394-489` | `learn()` monopolizes the shared 1-worker pool for hours; per-slice lease gaps expose half-learned state |
| EC-201 | med | bug | `engine_rt.py:499-505` | `shutdown()` awaits `concurrent.futures.Future` — TypeError swallowed; load/learn/engine never stopped |
| EC-202 | med | bug | `deps.py:102-126`, `support.py:661-695` | `make_harness` captures `get_event_loop()` on a pool thread → LoRA tools hang 180s via dead loop |
| EC-203 | med | weak-spot | `support.py:453-534` | Periodic full-project zip runs on event loop; change hash `md5(size+first4KB)` misses edits past 4KB |
| EC-204 | med | weak-spot | `services/lora.py:30,207-252`, `routes/core.py:686` | `scan_lora_adapters()` rglob+header-parse on event loop; single LoRA worker starved by merges → 60s lease timeouts |
| EC-205 | med | bug | `services/lora.py:67-87,195-246` | Auto-swap has unadapted window (unload→load as two actions); lock-free `_current`/`_pinned` state across REST calls |
| EC-206 | med | security | `routes/core.py:639-678`, `services/lora.py:99-104` | LoRA load/pin/merge accept arbitrary host paths + unclamped rank/alpha |
| EC-207 | med | weak-spot | `routes/core.py:64-67`, `api/status_reader.py` | `read_log_tail` reads whole file on loop; `status_file` param reads any dir's `log.txt` |
| EC-208 | med | weak-spot | `services/procs.py:43-145`, `routes/core.py:37` | `all_tasks()` deep-copies 5000-line buffers per poll; per-line log file I/O on the loop |
| EC-209 | med | bug | `forge_gui/api/mcp_client.py:185-396`, `deps.py:229-242` | stdio MCP: unserialized transport, no response-id check, no read timeout, leaked subprocess on failed connect; `connect_all`/`disconnect_all` never called |
| EC-210 | med | weak-spot | `routes/system.py:212-215` | `mcp_disconnect` runs `proc.wait(timeout=5)` on the event loop |
| EC-211 | med | weak-spot | `api/chat_store.py:65-70`, `api/lorebook.py` | Whole-file JSON save per mutation, no lock → lost updates + N+1 rewrites per chat turn; `lorebook_update` allows overwriting `id`/`created` |
| EC-212 | med | bug | `support.py:34-62` | `ApprovalChannel._counter += 1` non-atomic → duplicate ids orphan waiters (300s denied-by-timeout) |
| EC-213 | med | security | `forge/web_primitives.py:52-78`, `api/web_tools.py:186` | `web_fetch` validates scheme only → SSRF (server fetches localhost/169.254/intranet) |
| EC-214 | med | bug | `services/boot.py:48-71` | Boot test compares resolved path to raw request → always reloads, killing resident activation state |
| EC-215 | low | bug | `procs.py:67-72` | Task ids restart at `task_0001` → collide with old dirs; logs append into stale runs |
| EC-216 | low | bug | `hub.py:36-71` | `_fanout` races outside lock; `put_nowait` on asyncio.Queues from a foreign thread → lost wakeups |
| EC-217 | low | dead-code | `session_manager.py:158` | `TaskSession.total_tokens` initialized, never incremented → `/v1/tasks` reports 0 forever |
| EC-218 | low | weak-spot | `forge_server.py:1058-1114` | `/v1/batch/completions` + `/v1/tools/execute` synchronous on loop; unbounded prompts; word-count reported as tokens |
| EC-219 | low | missing-feature | `forge_server.py` | OpenAI server missing engine subsystems the GUI exposes (merge/evolve/diagnose/benchmark/learn/blend/recover) |
| EC-220 | low | weak-spot | `routes/train.py:186-195`, `api/events_reader.py` | Selfplay status endpoints parse files on the loop every poll; `latest_status` rglobs per call |

---

## Detailed findings

### Engine core & activation

**EC-006 — ~21/24 KV strategies are runtime-inert** [high, dead-code] — `open`
`KVCacheStrategy` (`init/append/get/clear`) is never called from any
generation path. Repo-wide grep finds only `cache.init` in
`_activate_kv_cache` and `.info()` calls. Only `beacon`, `anchor`,
`metakv` implement `compact_past_kv` (the R60 decode bridge);
`auto_context` works only via monkey-patching
`_generate_with_oom_recovery`. `paged` (**the default**), `rotorquant`,
`s4r`, `hadamard_int4`, `compressed`, `streaming`, `snapkv`,
`snapkv_4bit`, `2bit`, `paged_eviction`, `xquant`, `cpu_offload`,
`disk_offload`, `hqe_kv`, `kvzip`, `spectral`, `residual_stream`,
`hyquant`, `hisparse`, `evo_sparse`, `vegas`, `capture`, `vtoken`,
`matryoshka`, `filler` are constructed, `.init()`ed (some eagerly
allocate GPU+pinned buffers — `CPUKVCache.init` allocates an 8K-token GPU
buffer), logged via `info()`… and never see a KV byte.
`kv_cache="snapkv"` is a silent no-op that still logs a convincing
status line.
*Fix:* delete inert strategies or implement `compact_past_kv` on each;
make `info()` report `wired: false`; document the two real mechanisms.

**EC-007 — low-VRAM rescue is a placebo** [high, bug] — `fixed`
`engine_activation.py:936-966` — `_check_vram_and_offload_if_needed`
switches to `kv_cache="cpu_offload"` below 500 MB free VRAM, but
`CPUKVCache` is in the inert set (EC-006) — generation OOMs anyway; only
the log line changes.
*Fix:* real `compact_past_kv`/preallocated-tier offload, or replace with
`kv_cache_tokens` shrink + `PreAllocatedKVCache` resize.

**EC-008 — `self_speculative_sparse` crashes on first use** [high, bug] — `fixed`
`decoding.py:879` — `SelfSpeculativeSparse.generate` calls
`model(draft_tokens[-1], past_key_values=..., sparse_k=self.sparse_k)`;
`ConfigurableResearchLLM.forward` has no `sparse_k` param and no
`**kwargs`. No model in the repo accepts it.
*Fix:* delete the kwarg (implement sparse attention inside the layer) or
remove the strategy from `build_decoding`.

**EC-009 — `ngram_speculative` never rolls back past_kv** [high, bug] — `fixed`
`decoding.py:367-409` — verify pass writes all draft positions into
`past_kv`; on partial rejection the loop continues — rejected draft
tokens remain in cache permanently (wrong K/V, wrong positions for all
later tokens). Contrast SpecBound's `_slice_kv` + ReplaySSM rollback.
*Fix:* snapshot `past_len` before verify; `past_kv = self._slice_kv(new_past_kv, past_len + n_accepted + 1)`.

**EC-010 — `external_draft_speculative` same corruption + wrong logits index** [high, bug] — `fixed`
`decoding.py:476-510` — same missing rollback; also `verify_logits[:,
-1:, :]` always takes the last verify position — after rejection that
position follows the last *rejected* token, so the next main token is
conditioned on discarded context.
*Fix:* `_slice_kv` rollback; `verify_logits[:, n_accepted:n_accepted+1]`.

**EC-011 — `cache_prompt_prefix` defined twice** [high, dirty] — `fixed`
`prefix_cache.py:163` vs `:236` — second definition silently shadows the
first; the dead copy has a different contract in its docstring. Anyone
editing the first edits dead code.
*Fix:* delete lines 163-197.

**EC-012 — `ternary_ptq` runs calibration twice** [high, dirty] — `fixed`
`engine_activation.py:903-920` — calls
`novel_quant_r48.collect_activations` into `acts`, then immediately
re-imports `novel_quant_r46.collect_activations` and overwrites it. One
full calibration forward pass wasted.
*Fix:* drop the r48 call; pick the intended collector.

**EC-013 — four subpackages production-unwired** [high, dead-code] — `open`
`forge.engine.safety` (trilens, siren, agent_corrector, vram_monitor),
`reasoning` (budget, shortcut, syncthink), `loader` (ProgressiveLoader),
`compat` (arch_adapters) are imported **only** by `tests/` and
`scripts/_archive/` — zero references from `forge_engine.py`,
`forge_server.py`, `feature_registry.py`, or any production path.
*Fix:* wire via feature registry or move to `scripts/`/`research/`.

**EC-014 — dead activation flags** [high, dead-code] — `open`
`use_spec_attn` (ActivationConfig, TaskBootConfig, forge_server CLI,
self_play configs) and `use_progressive_load` have no registry handler
and no `activate` branch reads them — silently do nothing.
*Fix:* implement handlers or delete flags end-to-end.

**EC-015 — `ProgressiveKV` broken and unused** [high, bug] — `open`
`innovations.py:289-354` — `split()` computes `residual_scale` then
doesn't return it; `reconstruct()` fabricates it
(`anchor_scale/residual_qmax*anchor_qmax`) — wrong scale contradicts the
"equivalent to full-precision" claim. `info()` reports `total_bits=16` =
zero compression. `split`/`reconstruct` have zero call sites.
*Fix:* return `residual_scale` from split; wire or delete.

**EC-016 — innovation slots are dashboard props** [high, dead-code] — `open`
`engine_activation.py:197-217` — `quarot_kv`, `v0_warm`, `progressive_kv`
constructed, logged, surfaced in `diagnose()` — their methods
(`QuaRotKV.quantize_k/v`, `V0WarmStart` warming, `ProgressiveKV.split`)
have zero call sites. Only `mrl_adapter` mutates the model.
*Fix:* route through model/`past_kv` path or mark "detection only".

**EC-017 — registry handlers write attrs nothing reads** [high, dead-code] — `open`
`feature_registry.py:444-503` — `_h_kara_kv/_h_moment_kv/_h_kvpop/
_h_conf_kv/_h_matryoshka_kv` set `eng._*` attrs found only in registry +
diagnostics. Scheduler handlers (`fastserve/libra/kairos/hotprefix/
triroute/unified_radix/corun`) similarly assign `eng._*` —
`BatchQueue`/`_dispatch_loop` never reads them. `mosa`/`aoh`/
`learned_prefix` are live.
*Fix:* name the consumer per handler or mark `experimental_dead`.

**EC-018 — crash recovery half-works and can deadlock** [high, weak-spot] — `open`
`crash_recovery.py` — (a) `_cleanup` takes a plain `Lock` inside a
signal handler → deadlock if SIGINT lands while
`checkpoint_generation`/`snapshot_kv_cache` holds it; (b)
`snapshot_kv_cache` requires `kv.k_cache` — absent on every strategy
except Standard/SnapKV4Bit → silently skips under the default `paged`;
(c) `recover()` returns `kv_snapshot` but **nothing reinstalls it** —
write-only feature.
*Fix:* `RLock`; restore path in `engine_lifecycle.recover`; iterate
`past_kv`-style entries or drop KV snapshotting.

**EC-019 — unknown strategy names silently become `standard`** [high, weak-spot] — `fixed`
`kv_backend.py:436-438`, `decoding.py:1706` —
`strategies.get(strategy, Standard*)` — `kv_cache="rotorqunat"` (typo)
silently runs standard KV; no warning, no error. Exactly the "silent
regression" AGENTS.md bans.
*Fix:* raise `ActivationError` with the valid-name list.

**EC-020 — B==1 batch path drops sampling params** [high, bug] — `fixed`
`session_manager.py:591-617, 776-799` — `submit()` collects
`top_k`/`repetition_penalty`/`stop` into `PendingRequest`, but the
single-request branch passes only `max_new_tokens/temperature/top_p`;
`_run_single_stream` same. A `top_k=1` or `stop=[...]` is silently
ignored whenever the window collects one request — the common case.
*Fix:* pass all fields through both paths.

**EC-021 — feather scheduler never consulted** [high, dead-code] — `fixed`
`session_manager.py:400-405` — `use_feather_scheduler` builds
`FeatherScheduler`; `_run_batched`'s docstring claims it uses it;
`self._feather` is never referenced again.
*Fix:* call it before dispatch or remove flag + import.

**EC-022 — compaction bridge no-ops on preallocated cache** [med, weak-spot] — `open`
`engine_generation.py:1017-1039` — `isinstance(past_kv, (list, tuple))`
gate silently skips compaction under `PreAllocatedKVCache` (CUDA-graph
path); `except Exception → warn + return` degrades a broken strategy to
unbounded KV.
*Fix:* support the preallocated container or warn once; count failures
in `stats()`.

**EC-023 — `MetaKVCache.append` drops `token_ids`** [med, bug] — `open`
`kv/metakv.py:127-131` — signature accepts `token_ids=None`, forwards
only `attention_weights`. Latent today (append never called — EC-006)
but corrupts any inner strategy needing ids once resurrected.
*Fix:* forward `token_ids=token_ids`.

**EC-024 — two parallel decode loops** [med, dirty] — `open`
`generate()` routes through `self.decoding.generate`
(`StandardDecoding`), while `_finish_to_stop`/`_decode_with_kv`/sessions/
prefix-cache/graph paths use the engine's own `_decode_tokens`/
`_sample_next_token`. Identical semantics duplicated; SpecBound comments
admit they must "mirror StandardDecoding exactly" — drift inevitable.
*Fix:* single implementation or shared `_decode_step`.

**EC-025 — sampling helpers duplicated** [med, dirty] — `fixed`
`_min_k_filter`, `_dry_penalties`, `_apply_dry` copied between
`engine_common.py` and `decoding.py`; decoding.py's "cycle" justification
is false — `engine_common` has no forge imports at runtime.
*Fix:* import from `engine_common`.

**EC-026 — two session systems** [med, arch] — `open`
`SessionManager`/`TaskSession` (history + boot config;
`kv_cache`/`kv_seq_len` fields "not yet populated") vs `_session_cache`/
`SessionCacheManager` (KV reuse). The module docstring's "layer map"
exists precisely because the naming collision confuses readers.
*Fix:* rename one (`ConversationManager`); populate or delete the dead
fields.

**EC-027 — file blacklist is write-only** [med, security] — `fixed`
`tool_security.py:414-502` — `check_file_read` blocks only 'denied'
rules + outside-workspace; `DEFAULT_FILE_BLACKLIST_PATTERNS` (`.env`,
`*.pem`, `**/id_rsa*`) applies only to writes — nested `.env`/`*.key`
readable. Conversely `PROTECTED_ENGINE_FILES` uses `endswith` —
`config.py`/`paths.py` anywhere are write-refused.
*Fix:* apply blacklist to reads; match protected files by `Path.name` or
explicit paths.

**EC-028 — `__init__` side effects** [med, weak-spot] — `open`
`forge_engine.py:194-208` — every `ForgeEngine(...)` creates
`.forge_recovery/` (cwd-relative), registers SIGINT/SIGTERM + atexit,
constructs `Library` → mkdir + load of `research/data/library` — even in
unit tests. Makes tests non-hermetic.
*Fix:* lazy-init recovery/library; opt-in flags.

**EC-029 — two quantization trees** [med, arch] — `open`
`_apply_quantization_single` reaches into both `forge.quant.*` and
`forge.engine.quant.*`; overlapping domains (`kv_compress`/`kv_2bit`/
`paged_kv` in `forge/quant`, strategy wrappers in `kv_backend`) with no
documented boundary.
*Fix:* consolidate into one canonical tree + re-export shim.

**EC-030 — `novel_quant_rNN` versioned siblings** [med, dirty] — `open`
`novel_quant.py` + `r44`/`r45`/`r46`/`r48`/`r50` coexist; `r48` imports
from `r44` AND `r46` (skips r45 — violates §A derive-chain); `r45`/`r50`
unreachable from `activate()`.
*Fix:* fold into one chain or move off the engine path.

**EC-031 — quant fallback chain silently downgrades** [med, weak-spot] — `fixed`
`engine_activation.py:615-699` — `nvfp4→w8a8→fp8→int8→int4→None` etc.; a
failed `nvfp4` can land on int8/bf16 — speed AND quality change, only a
warn log marks it, `self.quantize` still reports the requested mode.
*Fix:* set `self.quantize` to the applied mode; report effective mode in
`activate()`'s return.

**EC-032 — discarded expressions** [med, dirty] — `fixed`
`decoding.py:346,377,382,841` — `input_ids.shape[1]`,
`main_token.squeeze()`, `torch.tensor([draft_tok])`, an `eos_tensor`
built then thrown away.
*Fix:* delete or complete intended use.

**EC-033 — `_ngram_lookup` mixes tensor into int list** [med, bug] — `fixed`
`decoding.py:413-426` — `seq = ids[0].cpu().tolist() + [last_token]`
where `last_token` is a 0-dim tensor → `[int,…,tensor]`; works by
accident via `tensor.__eq__`; O(seq_len) rescan + GPU sync per step.
*Fix:* `.item()`; rolling n-gram index.

**EC-034 — library pre-tokenization assumes flat ids** [med, bug] — `open`
`library.py:236-240,376-389` — `tokenizer(content).input_ids` returns
`[[ids…]]` → `token_count=1`, nested list fed to `decode`/`extend`;
tokenizer=None fallback charges 0 tokens against budget.
*Fix:* flatten `ids[0]`; count estimate when falling back.

**EC-035 — `_min_k_filter` sorts vocab twice** [med, dirty] — `fixed`
`engine_common.py:116-150` — `torch.sort` at :116 for values, again at
:147 for indices. 2× O(V log V) per decode step on 65K vocab.
*Fix:* one `torch.sort`, use both returns.

**EC-036 — model layer in engine package** [med, arch] — `open`
`forge/engine/mamba3.py` — `forge/model/layers.py:629` lazily imports
`forge.engine.mamba3.Mamba3Block` — model→engine dependency inversion.
*Fix:* move to `forge/model/mamba3.py`, re-export shim.

**EC-037 — three prefix-cache impls + dict path** [med, dirty] — `open`
`LRUPrefixCache`, `ChunkedPrefixCache`, `LearnedPrefixCache` (registry
replaces `eng._prefix_cache`), plus plain-dict compat in
`get_cached_prefix` — four branching storage-format cases.
*Fix:* one class with `mode=` knob.

**EC-038 — `torch.load(weights_only=False)` fallback** [med, security] — `open`
`crash_recovery.py:333-336` — after `weights_only=True` fails, retries
with `False` — ACE surface on `.forge_recovery/` contents (cwd-relative,
writable by tools).
*Fix:* drop the fallback.

### Engine KV/decoding/misc low-severity

- **EC-039** — EOS sets hardcoded in two places, already drifted
  (`engine_common._DEFAULT_EOS_TOKEN_IDS` vs inline set in
  `SelfSpeculativeSparse` missing 128001/128009/128039/128040).
- **EC-040** — `session_manager.py:666-673` dead `pass` block for chunked
  prefill — looks like wiring, does nothing.
- **EC-041** — `session_manager.py:509` slices `max_batch_size` globally
  before grouping by `(model, boot_key)` — one model's flood starves
  others.
- **EC-042** — `beacon_kv.py:232-270` per-layer Python beacon-update loop
  runs on every bridge call even when `T ≤ budget`.
- **EC-043** — ~200 broad `except` sites engine-wide; resilience layers
  double as regression-hiders. Needs per-site narrowing + a `strict`
  flag.
- **EC-044** — `cascade.py:102-145` undocumented magic constants
  (0.1/0.05/0.2/0.8, log interpolation) with no sweep backing (§F).
- **EC-045** — `kv/` strategy params scattered as literals
  (`sink=4,window=128,budget=1024,n_beacons=16,ema=0.9`…); no defaults
  module.
- **EC-046** — `kv_cache.info()` is the *only* interface the engine uses;
  status lines report decorative objects (companion to EC-006).
- **EC-047** — `# noqa: F401` re-export sprawl in `forge_engine.py`/
  mixins makes dead-import detection unreliable; move re-exports to
  `__init__.py` `__all__` (companion to EC-003).
- **EC-048** — `airllm_streamer.py` is a third parallel decode loop
  (after `_decode_tokens` and `DecodingStrategy`); technically reachable
  via `acceleration="airllm_streaming"` — see EC-056 for why it
  shouldn't be.
- **EC-049** — `TaskSession.kv_cache`/`kv_seq_len` declared "not yet
  populated" — shipped feature stubs.

### GGUF loader (`forge_loader.py`) — all verified, all silent

**EC-051 — unknown dtypes → wrong-format dequant or zeros** [high, bug]
`forge_loader.py:335-340` — the `else` branch runs `_dequant_q4_0` on
Q4_1/Q5_*/Q2_K/Q3_K/Q5_K/Q8_K/IQ*/TQ*, and on failure returns
`torch.zeros(n_elem)` — silent all-zero weights.
*Fix:* raise on unimplemented dtypes; delegate to a real GGUF dequant.

**EC-052 — Q4_0 dequant scrambles nibble order** [high, bug]
`forge_loader.py:193-214` — Q4_0 stores lo→elems 0-15, hi→16-31; code
writes `lo→vals[0::2], hi→vals[1::2]` (interleaved). Every 32-elem block
re-ordered; loads "succeed" with wrong weights.
*Fix:* `vals[:16]=lo; vals[16:]=hi` + parity test vs llama.cpp.

**EC-053 — Q4_K drops min term, mis-unpacks scales** [high, bug]
`forge_loader.py:230-266` — `dmin` read and discarded; 12-byte
scales+mins decoded as 8 bare 6-bit scales; nibble interleave bug too.
*Fix:* implement real Q4_K scheme (8 sub-blocks, two-level 6-bit unpack,
`d*sc*q − dmin*m`).

**EC-054 — Q6_K reads wrong offsets** [high, bug]
`forge_loader.py:269-301` — real layout `ql[128], qh[64], scales[16],
d@208`; code reads `d` first then 16B scales + 16B ql + 16B qh. Header
says "simplified" — output is garbage.
*Fix:* implement the real layout or refuse Q6_K.

**EC-055 — unmapped dtypes return raw bytes as weights** [high, bug]
`forge_loader.py:316-321` — when `_GGUF_BLOCK_BYTES` lacks the dtype
(all IQ*, TQ*, Q3_K — Q3_K is in `_GGUF_BLOCK_SIZE` but missing from
`_GGUF_BLOCK_BYTES`), slices `n_elem` raw bytes and reshapes.
*Fix:* raise `NotImplementedError`.

**EC-059 — per-tensor swallow + strict=False** [high, weak-spot]
`engine_checkpoints.py:933-956` — `except Exception: logger.warning(
"skipping tensor")` in load loop + `load_state_dict(strict=False)` —
combined with EC-051…055 produces silently part-random-init models.
*Fix:* count missing/unexpected; fail below a mapped fraction.

**EC-060 — unbounded `-GGUF` recursion** [high, weak-spot]
`forge_loader.py:600-604` — on failure retries `model_id + "-GGUF"`
recursively → `-GGUF-GGUF`… → `RecursionError` after ~1000 network calls.
*Fix:* loop with max 1 fallback; narrower exception types.

**EC-063 — `detect_format` misroutes local paths** [high, weak-spot]
`forge_loader.py:484-507` — `source.startswith(("http","hf://","liquid",
"qwen","meta"))` — a local dir named `qwen_*` is treated as an HF repo
ID; `.yaml` whose first bytes decode to a u64<1e8 returns "safetensors".
*Fix:* `Path(source).exists()` first; conservative `org/model` check.

**EC-064 — `load()` returns `model=None` for GGUF** [high, weak-spot]
`forge_loader.py:509-563` — `_load_gguf` returns `(None, config)` meant
for ModelLoader, but ModelLoader never consumes `_gguf_info`; callers get
a None model. Also reads only `llama.*` metadata — qwen2/gemma GGUFs get
garbage dims.
*Fix:* build the model here or remove the branch + document.

**EC-075 — loader docstring lies** [med, docs-drift]
`forge_loader.py:1-9` — promises "mmap zero-copy", "<100ms total load";
`get_tensor` copies full bytes then dequantizes to fp16 copies;
`_load_gguf_checkpoint` eagerly loads everything.

### AirLLM streaming

**EC-056 — context-free decode + per-token shard reload** [high, bug]
`airllm_streamer.py:237-264` — `_forward_with_kv` calls
`engine.model.blocks[li](x)` without kv_cache/past; k_cache/v_cache
written but never fed back — decode attends to the current token only.
`generate()` calls `_load_all_shards()` before prefill AND inside the
decode loop → full-model H2D per token. BUG_LOG claims this path was
deleted — it wasn't (EC-088).
*Fix:* delete the path or pass committed cache into blocks; keep prefill
shards resident.

### Server (`forge_server.py`) & session manager

**EC-057 — blocking generator inside async SSE handlers** [high, weak-spot]
`forge_server.py:1885-1899,2001-2015` — `_stream_chat`/`_stream_completion`
iterate `self.registry.generate_stream(...)` directly in the coroutine —
every decode step runs on the event loop; one streaming request stalls
ALL endpoints for seconds.
*Fix:* worker thread + `asyncio.Queue` (BatchQueue already has the
pattern) or `asyncio.to_thread`.

**EC-058 — no auth on mutation endpoints** [high, security]
`forge_server.py:566-573` — binds 0.0.0.0, `CORSMiddleware(allow_origins=
["*"], allow_credentials=True)`, no API key anywhere — while
`/v1/engine/*` mutates KV/decoding/quant, `/v1/library/*` writes entries.
*Fix:* `FORGE_API_KEY` Bearer middleware (default-on for non-loopback);
restrict mutation endpoints to localhost.

**EC-061 — registry token stats double-count quadratically** [high, bug]
`model_registry.py:179,228` — `entry.total_tokens +=
engine.total_tokens_generated` adds the lifetime cumulative counter per
request.
*Fix:* add `after − before` delta. (ENGINE_IMPROVEMENT_PLAN already lists
this; still unfixed.)

**EC-062 — kv-cache-type endpoint resets whole activation** [high, bug]
`forge_server.py:1644-1663` — `engine.activate(kv_bits=k_bits)` builds a
default `ActivationConfig` → kv_cache reverts to `paged`, decoding to
`standard`, ~50 feature flags off, quantize/acceleration cleared.
`v_bits` computed and echoed but never applied; bf16/f16 map identically.
*Fix:* route through `engine.hotswap` partial update; apply or reject
`v_bits` honestly.

**EC-065 — no observability** [high, missing-feature]
No request-ID tracing, no `/metrics`, `/health` returns static counts —
no per-model state, queue latency, or error surface.
*Fix:* request-ID middleware, `/metrics` endpoint, deepen `/health`.

**EC-066 — no disconnect/abort handling** [high, missing-feature]
No `request.is_disconnected()` checks, no `CancelledError` handling in
any SSE generator — a dropped client leaves generation running to
`max_new_tokens` on a serialized engine (`_gen_lock`), blocking everyone.
*Fix:* poll disconnect between chunks; propagate a stop flag into
`generate_stream`.

**EC-067 — hotswap request overrides never invoked** [med, dead-code]
`hotswap.py:300-312` — `set_request_override`/`get_request_overrides`
zero callers; the getter *pops* — a getter that destroys state.
*Fix:* delete or wire into BatchQueue.

**EC-068 — six dead hotswap setters** [med, dead-code]
Called: `set_kv_cache`, `set_decoding`, `set_context_limit`,
`set_infinite_context`, `set_generation_defaults`, `set_feature`,
`update_from_dict`, `apply_pending`. Dead: `set_quantize`,
`set_acceleration`, `set_features`, `set_vram_margin`, `set_auto_offload`,
`set_batch_config`.
*Fix:* delete or expose via settings endpoints.

**EC-069 — `apply_pending` double-assign** [med, dirty]
`hotswap.py:348-351` — `_current` set inside the branch then
unconditionally again.

**EC-070 — dead public engine APIs** [med, dead-code] — `open` (partially disputed)
Narrowed by the mixin audit: `generate_gated` and
`attach_head`/`detach_head`/`head_report` ARE wired (gated pipeline /
FluxLM persistence). Still unreferenced: `apply_flux_packets`
(docstring-only) and `profile_bandwidth`.
*Fix:* delete `apply_flux_packets` and `profile_bandwidth` or wire them.

**EC-071 — `update_from_dict` key mangling** [med, dirty]
`key.replace("set_","")` strips the substring anywhere — `reset_x`
mangles; silently-ignored fields produce no error.
*Fix:* `removeprefix` or explicit allowlist.

**EC-072 — `/health` reaches into private member** [med, dirty]
`forge_server.py:746` — `self.batch_queue._pending`.
*Fix:* `BatchQueue.pending_count` property.

**EC-073 — batched "streaming" is one blob** [med, weak-spot]
Batched requests wait for completion then push the whole text through
the stream queue — defeats token streaming.
*Fix:* per-request token streaming or reject stream=True for batched.

**EC-074 — serving deps undeclared** [med, missing-feature]
`requirements.txt`/`pyproject` lack fastapi/uvicorn/pydantic/gguf/
llama-cpp/zstandard; `[serving]` extra lists only `gradio` (unused by
forge_server).
*Fix:* fix the extra.

### Scripts & repo hygiene

**EC-050 — hardcoded HF token** [high, security] — `fixed` ⚠ **ACTION REQUIRED**
`scripts/distill/extract_gemma4_text.py:25` —
`TOK = "hf_...REDACTED_REVOKED"` — a real-format HF user
token in plaintext, committed.
*Fix:* **revoke/rotate immediately**, purge from git history, read from
`os.environ["HF_TOKEN"]`.

**EC-077 — GUI presets launch archived/nonexistent scripts** [med, dirty]
`forge_gui/api/process_manager.py:42-58` — "Ablation Benchmark" runs
`scripts/ablation_benchmark.py` (exists only in `scripts/_archive/`,
whose README forbids execution); "Extract Vocab Packs" runs
`scripts/extract_vocab_packs.py` (doesn't exist).
*Fix:* repoint presets; startup check that preset paths exist.

**EC-078 — download scripts write into scripts/** [med, dirty]
`download_datasets.DATA_DIR = Path(__file__).parent` → datasets land in
`scripts/pretrain|sft/`; `download_github_code` `.env` uses `parents[3]`
→ resolves to `D:\` (repo `.env` is `parents[1]`).
*Fix:* resolve via project-root paths module; write under `research/data/`.

**EC-079 — `generate_synthetic_data.py` 905 lines** [med, bloat]
Only script >800 lines; docstring says `python -m
research.data.generate_synthetic_data` — not a package module.
*Fix:* fix docstring; consider module + thin CLI.

**EC-080 — nine downloaders, duplicated infra** [med, bloat]
Each `download_*.py` reimplements shard rolling/HF_TOKEN loading/resume.
`download_gaps` explicitly overlaps FineWeb-Edu/Numina/Glaive coverage.
*Fix:* shared `scripts/_hf_download.py` helper → one registry-driven
downloader.

**EC-081 — bench script proliferation** [med, bloat]
8 active `bench_*.py` + 6 archived; `bench_load_time` vs
`bench_weight_io` overlap I/O timing; `bench_r50_features` vs
`smoke_r50_features` vs `test_r50_rd_features` triple-cover R50.
*Fix:* keep `bench_load_time` + `bench_weight_io` + one feature-bench.

**EC-082 — `exp_*.py` orphans** [med, bloat]
17 files; `exp_baseline_qwen`, `exp_matrix`, `exp_trainless`,
`exp_fluxgv/rev/scale/scale_wt/tok` have no test/doc/AGENTS reference.
*Fix:* archive per `_archive` policy.

**EC-083 — `smoke_r50_*` superseded** [med, dead-code]
`test_r50_rd_features.py` imports the same symbols under pytest.
*Fix:* archive both.

**EC-084 — pytest-named scripts never collected** [med, dirty]
`test_forgelm_v2_generate.py`, `test_hermes_w4_generate.py`,
`test_xparam_r50.py` live in `scripts/` — `testpaths=["tests"]` means
pytest never sees them.
*Fix:* rename `check_*`/`bench_*` or move under `tests/` with a gpu mark.

**EC-090 — script docstrings reference pre-move paths** [low, docs-drift]
`filter_wikipedia.py`, `download_stack_v2.py`, `download_datasets.py`
tell users to run `research/data/...` paths that no longer exist there.

**EC-091 — machine-specific absolute paths** [low, dirty]
`gen_roc_data.py:25` hardcodes `D:\LMstudio\Models\...`; same pattern in
`download_stack_v2.py`.
*Fix:* env vars / CLI args.

**EC-092 — stray repo-root artifacts** [low, dirty]
`parity_ab.txt`, `parity_cpu.txt`, a binary file literally named `None`,
committed `adapters/smoke_lora/checkpoint-1`, vendored dataset-card dirs
inside `scripts/`.

### Test gaps

**EC-076** — entire HTTP surface except `/v1/systemone` untested
(`test_decide.py:294` is the only `TestClient`).
**EC-085** — engine modules with zero test imports: `airllm_streamer`,
`forge_loader`, `model_registry`, `hotswap`, `engine_lora`,
`engine_merging`, `library`, `engine_tools`, `innovations`,
`forge_server`. (Explains why EC-051…056, EC-061 went unnoticed.)
**EC-087** — `gpu`/`integration`/`slow` markers registered under
`--strict-markers` but applied nowhere — `-m "not gpu"` deselects
nothing.

### Docs drift

**EC-004** — "42 flags" vs actual 44; `research/checkpoints/` paths in
docstring.
**EC-088** — `BUG_LOG.md:509-510` claims airllm_streamer deleted; it's
wired (`engine_activation.py:496-497`, `engine_generation.py:297-298`,
`forge_server.py:266`).
**EC-089** — `/health` docstring claims Prometheus output — returns
plain JSON.

---

### Core mixin audit (engine_generation / checkpoints / tools / hotswap / sessions / lifecycle)

**EC-093 — tool handlers always error** [high, bug] — `fixed`
`engine_tools.py:1103-1126,1145,1157` — `self._ast` and `self._random`
are never assigned (`__init__` sets only engine/_handlers/security; only
a module-level `import ast as _ast` exists, no `random`). `_safe_math_eval`,
`_eval_node`, `_random_number`, `_chance` hit `AttributeError` → caught by
`execute()` → `{"status":"error"}` on every call. Zero test coverage.
*Fix:* `self._ast`→`_ast`, module-level `import random`; smoke-test every
registered handler.

**EC-094 — session KV reuse never hits** [high, bug] — `fixed`
`session_cache.py:201-214` + `engine_sessions.py:80-115` —
`continue_session` sets `session.token_ids = full_ids` (prompts only);
`update_session_kv` stores `past_kv` covering prompt+generated but never
appends generated ids. Next turn `kv_len != len(token_ids)` → mismatch →
`past_kv=None` → `_full_generate` every turn, AND the prefill uses
prompts-only ids so prior assistant replies drop from context. The
delta-prefill branch is unreachable.
*Fix:* append generated ids to `session.token_ids`; add a 2-turn test
asserting `_prefix_hits > 0`.

**EC-095 — streaming fallback → unusable engine** [high, bug] — `open`
`engine_checkpoints.py:1102-1116` + `426-428` — `_load_streaming` builds a
meta model, sets `_needs_streaming`, returns;
`_auto_activate_optimal` early-returns for streaming so
`AirLLMStreamer.setup` is never called; `engine.acceleration` stays None
→ `_generate_impl` skips `AirLLMStreamer.generate` and decodes meta
tensors. Only a manual `activate(acceleration="airllm_streaming")` saves
it; `model_registry.py:119` uses auto_activate default.
*Fix:* call `AirLLMStreamer.setup(engine)` inside `_load_streaming`.

**EC-096 — `embed()` returns logits** [high, bug] — `fixed`
`engine_generation.py:1310-1318` — `model(ids, return_hidden=True)`
returns a 3-tuple `(logits, loss, hidden)` (llm.py:707-710); the code
checks `len(out) >= 4` → false → fallback `hidden = out[0]` = logits.
`rerank()` and every semantic caller get wrong vectors silently.
*Fix:* accept `len(out) >= 3` (hidden is last) or pass `use_cache=True`.

**EC-097 — file_read/list bypass security; .env exfil** [high, security] — `fixed`
`engine_tools.py:815-870` — `_security_check` gates only
file_write/edit/delete/move/rename + web; `file_read`/`file_list` fall
through to `SecurityDecision(allowed=True)`.
`ToolSecurityManager.check_file_read` (tool_security.py:414) exists and
would deny ACCESS_DENIED paths but is never called; the docstring
explicitly permits reading blacklisted files. Workspace root = repo root
→ model can read `.env` containing the same TAVILY/EXA keys
`_get_api_key` reads → exfiltration into generated text. (Read-side of
EC-027.)
*Fix:* route file_read/file_list through `check_file_read`; add
`.env`/credential globs to the read deny-list.

**EC-098 — quarot/mrl unconditionally marked present** [high, weak-spot] — `fixed`
`engine_activation.py:90-91,201-204` — `features.append("quarot")` /
`("mrl")` "assume applied by pipeline"; `_activate_core_innovations`
then enables `QuaRotKV(has_quarot=True)` whenever kv_cache ∈
{hadamard_int4, rotorquant} regardless of actual checkpoint content →
wrong KV semantics when QuaRot wasn't applied.
*Fix:* detect from checkpoint keys/metadata, don't assume.

**EC-099 — activation kwargs dropped for GGUF** [high, bug] — `fixed`
`engine_checkpoints.py:73-74,155-157` — `act_kwargs` popped at :73; the
`.gguf` early-return at :156 forwards only `**kwargs` →
`engine._user_activation_overrides` never set → e.g. `kv_cache_tokens`
passed to `from_checkpoint` silently ignored on GGUF.
*Fix:* forward `act_kwargs` into `_load_gguf_checkpoint` before
`_auto_activate_optimal`.

**EC-100 — failed reactivation retries forever** [high, weak-spot] — `fixed`
`hotswap.py:316-353,389-393` — `_apply_reactivation` raises before
`_current` updates and `_pending` stays set → `apply_pending()` re-runs
the known-failing config on every subsequent `generate()`. (Includes the
double-`_current`-assign at :349 — EC-069.) `set_feature`/`update`
silently ignore unknown flag names.
*Fix:* mark pending failed/clear it, or roll back + surface a persistent
error.

**EC-101 — batch path counts characters as tokens** [high, bug] — `fixed`
`engine_generation.py:817-819` — `total_tokens = sum(len(r) for r in
results)` where `results` are decoded strings → `total_tokens_generated`
inflated ~4x on the batch path only (compounds EC-061's registry
double-count).
*Fix:* sum generated token ids instead.

**EC-102 — RLock cross-thread release → deadlock** [high, weak-spot] — `open`
`engine_generation.py:1570,1622-1629` — `_gen_lock.acquire()` at :1570,
released in `finally` at :1629. If a caller abandons the generator
without `close()`, finalization may run on another thread;
`RLock.release()` from a non-owning thread raises inside GC → lock stays
held → all subsequent `generate()`/`sleep()`/LoRA calls hang.
*Fix:* token-guarded release (only if owned), or hold the lock via a
dedicated owner thread.

**EC-103 — per-request gate leaks globally** [med, bug] — `fixed`
`engine_activation.py:387-398` — `_suffix_mem_for` writes
`mem.gate = gate` into the shared `SuffixMemory` — concurrent/later
requests inherit the mutated gate.
*Fix:* pass a request-local gate into the lift computation.

**EC-104 — GGUF hardcoded arch + double fp16 staging** [med, weak-spot] — `fixed`
`engine_checkpoints.py:891-958` — `rope_base=10000.0`, `attn_bias=False`,
`use_qk_norm=False`, `norm_eps=1e-5` hardcoded — ignores
`{arch}.rope.freq_base` metadata → silent wrong RoPE; `vocab_size`
defaults 32000 when absent. Every tensor staged in `state_dict` on
device then `.to(device).to(fp16)` — a 6GB Q4 GGUF needs ~12GB+ free →
OOM, the very problem GGUF avoids.
*Fix:* read rope/eps/bias from metadata; stream per-tensor into model.

**EC-105 — int8 head dequantizes whole weight per forward** [med, weak-spot] — `fixed`
`engine_checkpoints.py:764-773,799-801` — `_int8_forward` calls
`_w.to(x.dtype)` (full int8→bf16, ~250MB+/token on a 65k×2048 head) every
forward — defeats int8 storage; `_orig_forward` captured but unused;
:801 re-ties `model.head.weight` after the prequantized head had weight
deleted → stale fp weight registered-but-unused.
*Fix:* fused int8 matmul; remove dead local; skip re-tie when
prequantized.

**EC-106 — `keystack_features` clobbered** [med, weak-spot] — `fixed`
`engine_checkpoints.py:1042,856` — `_load_quantized` sets
`keystack_features = [f"load_quant_{mode}"]`, `_load_prequant` sets
`["bitnet_prequant","quarot","mrl"]` — discarding detected `mtp`/
`value_residual` → `_auto_activate_optimal` misses `mtp_selfspec`/
`use_v0_warm`.
*Fix:* append load-path flags; don't replace the detected list.

**EC-107 — broad RuntimeError → streaming fallback** [med, weak-spot] — `open`
`engine_checkpoints.py:218-236` — `except (OutOfMemoryError,
RuntimeError)` — any RuntimeError (corrupt weights, construction bug)
silently falls into AirLLM streaming — which per EC-095 yields an
unusable engine anyway.
*Fix:* restrict to OOM; re-raise others with context.

**EC-108 — chunked prefill wired but never invoked** [med, dead-code] — `fixed`
`engine_activation.py:544-551` + `session_manager.py:664-673` +
`engine_generation.py:1776` — flag → `ChunkedPrefiller` stored on
`self._chunked_prefill`; zero call sites for `prefill_chunked`/
`prefill_interleaved`; session_manager detects long prompts then
`pass` (EC-040). Mixin method also shadowed by the attribute.
*Fix:* dispatch long prompts through the prefiller or remove the
plumbing.

**EC-109 — `generate_with_tools` edge crashes + unbounded** [med, bug] — `open`
`engine_generation.py:884,953-957` — `content`/`round_idx` bound only
inside the loop → `NameError` on `max_tool_rounds=0`; no cancellation,
per-round timeout, or context budget — each round re-prefills the whole
transcript.
*Fix:* validate ≥1; bound total tokens/rounds; reuse prefix cache across
rounds.

**EC-110 — session path skips BOS, mutates outside lock, double-counts** [med, bug] — `fixed`
`engine_sessions.py:73-78` + `session_cache.py:172-174` — tokenizer uses
`add_special_tokens=False` while `generate()` requires BOS ("raw prompts
tokenize to garbage") → session quality silently worse;
`continue_session` mutates session state before `_gen_lock` → TOCTOU vs
`end_session`/evict; `logits` unbound on empty-delta continue;
`_record_generation` called inside `_full_generate` AND again at :114 →
`generation_count` ~2x on misses.
*Fix:* `add_special_tokens=True`; mutate under lock; guard empty delta;
remove double count.

**EC-111 — merging edge crashes + silent swaps** [med, bug] — `open`
`engine_merging.py:111-146,45-51` — `len(parents) < 1` admits 1 parent →
`states[1]` IndexError in blockwise/uniform/slerp/block_swap;
`_swap_weights` uses `strict=False` + ignores missing/unexpected → a
mismatched candidate scores as "clean".
*Fix:* validate parent count per method; raise on missing+unexpected.

**EC-112 — LoRA load/unload hazards** [med, weak-spot] — `open`
`engine_lora.py:76-106,120-139` — adapters attach before loading;
`loaded==0` produces no warning and leaves random LoRA corrupting
output; unload deletes `module.lora_adapter` while iterating
`named_modules()`.
*Fix:* load state first, then attach; warn/fail on `loaded==0`; collect
then delete after iteration.

**EC-113 — `mem_bias` dropped when `learn=False`** [med, bug] — `fixed`
`engine_generation.py:155-168` — the `params.learn is False` early-return
fires before the `mem_bias` branch → `generate(learn=False, mem_bias=x)`
silently drops it.
*Fix:* apply both gates in one block; restore in `finally`.

**EC-114 — OOM recovery leaves inconsistent state** [med, weak-spot] — `fixed`
`engine_generation.py:1711-1745` — dead `getattr` at :1713; on failed
retry only `kv_cache` is restored while `_active_kv_bits`/
`_active_kv_cache_name` keep failed values; on success the engine
silently stays on the fallback permanently (no re-upgrade;
`hotswap.current` unchanged).
*Fix:* remove dead expr; restore all related state on failure; record
downgrade or auto-retry original later.

**EC-115 — reactivation drops flags, keeps stale instances** [med, weak-spot] — `fixed`
`hotswap.py:355-393` — bare `ActivationConfig` (12 fields) → every
previously-enabled registry feature (avmp, learned_prefix_cache, …)
dropped from `feature_flags` while instances (`self.avmp`,
`_prefix_cache`) never cleared → half-stale strategy state.
*Fix:* seed config from `active_config`, overlay changes, disable stale
instances explicitly.

**EC-116 — GUI session endpoints kwarg mismatch → TypeError** [med, bug] — `open`
`forge_gui_server/routes/core.py:318-343` calls
`begin_session(ttl_s=...)` / `continue_session(max_tokens=...)`; mixin
signatures are `ttl=`/`max_new_tokens=` → TypeError on every session
endpoint; `begin_session` returns `SessionState` (not JSON-serializable)
→ serialization failure even if called correctly.
*Fix:* align kwarg names or `**kwargs` map; return a dict.

**EC-117 — recovery failures invisible** [med, weak-spot] — `open`
`crash_recovery.py` — signal-install/`atexit.unregister`/snapshot
failures pass silently or degrade to a warn line; no
`snapshots_failed` counter surfaced in `stats()`/`diagnose()`.
*Fix:* counter + surface. (Extends EC-018.)

**EC-118 — OutputHistory records injected prompt** [med, dirty] — `open`
`engine_generation.py:355` vs `212-214` —
`prompt = self.library.inject(...)` mutates the variable before
`_record_output(prompt, …)` → history stores lorebook-injected content.
*Fix:* keep `original_prompt` separate.

**EC-119 — tool handlers trust model args** [med, weak-spot] — `fixed`
`engine_tools.py:1364-1377,1144-1154,1101-1132,1452-1488` —
`_random_number`/`_chance` accept unbounded `count` (`range(10^9)` →
hang); `_safe_math_eval` no depth/size cap (`9**9**9` CPU burn);
`_file_list` recursive `rglob` no entry cap; `execute_calls` silently
substitutes `args={}` on malformed JSON.
*Fix:* clamp counts/list sizes; AST depth + bigint guards; error on
malformed args.

**EC-120 — Flux sleep leaks / wedges** [med, weak-spot] — `fixed`
`engine_lifecycle.py:59-64,82-86,141-143` — level-1 on cuda_primary
FluxLM only logs a warning, leaves all GPU tables resident; temp `.flux`
unlink failure swallowed; failed `FluxLM.load` on wake leaves
`self.model` unset with `_awake` already False → wedged.
*Fix:* free tables at level 1 or refuse; log unlink failures; fix
`_awake` semantics on wake failure.

**EC-121 — EOS counted on one path, excluded on another** [low, bug] — `fixed`
`engine_generation.py:251-268,1196` — `gen_tokens.append(next_token)`
unconditional on schema/logprobs path (includes EOS) vs `_decode_with_kv`
excludes EOS → `n_gen` off-by-one between paths; same in
`generate_stream`'s `generated_ids`.
*Fix:* skip appending when `should_stop`.

**EC-122 — sampling edge-case params misbehave** [low, weak-spot] — `fixed`
`_eos_token_ids(custom_ids)` replaces the default set entirely; `[]`
falls back (can't disable EOS); `stop=[""]` → `find("")`→0 → empty
output; `context_limit=0` treated as unset; session `SamplingParams`
omits `stop`/`skip_special_tokens`/`learn`.
*Fix:* merge custom ids; reject empty stop; treat 0 as explicit.

**EC-123 — dead expression + sparse GGUF validation** [low, dirty] — `open`
`engine_checkpoints.py:598-634` — `list(shapes)` at :621 discarded;
validation checks only embed/d_model/depth — rope/layer-type/
mixed-precision mismatches pass silently.
*Fix:* delete dead line; extend validation.

**EC-124 — wildcard + explicit double imports** [low, dirty] — `open`
Every mixin does `from .engine_common import *` then re-imports ~14
names; in `engine_lora` all 13 named imports are unused; same pattern in
diagnostics/sessions/generation/checkpoints/lifecycle.
*Fix:* drop wildcards; keep only used names.

**EC-125 — `CREATIVE_SAMPLING` dead + docstring drift** [low, dead-code] — `fixed`
`engine_activation.py:971-974` — dict referenced only in AGENTS.md, never
in code; docstring claims S4R default while `_auto_activate_optimal`
uses rotorquant; lists `hqe_kv, 2bit` absent from `_KV_FALLBACK_CHAIN`.
*Fix:* delete or wire; correct docstring (extends EC-004).

**EC-126 — warmup leaves conv/KV residue** [low, weak-spot] — `fixed`
`engine_activation.py:599-611` — `_warmup` runs `model(dummy,
use_cache=True)` filling per-layer conv/`_conv_state` with garbage-token
state; overwritten by next real prefill but observable to live-state
readers (gated probes, snapshots).
*Fix:* reset conv state after warmup or use a scratch forward.

**EC-127 — session_cache "radix tree" doesn't exist** [low, docs-drift] — `fixed`
`session_cache.py:12-14,47-59,251-275` — implementation is a dict of
`SessionState` + linear KV-length check; `_maybe_evict` runs only every
8th begin/continue → expired sessions linger.
*Fix:* fix docstring or implement real prefix matching; evict per access
or on a timer.

**EC-128 — `vram_usage` reports 0 for FluxLM** [low, weak-spot] — `fixed`
`engine_lifecycle.py:173-198` — sums `parameters()` only; FluxLM tables
are plain tensor attrs → reports 0 while holding GBs.
*Fix:* `model.mem_bytes()`-style hook.

**EC-129 — hardcoded relative tokenizer path** [low, weak-spot] — `fixed`
`engine_common.py:241` + `engine_checkpoints.py:118` +
`model_registry.py:118` — `"research/checkpoints/forgelm_v2_tokenizer"`
is cwd-dependent; a GUI launched elsewhere resolves it wrong.
*Fix:* package-relative resource / paths module.

**EC-130 — `benchmark()` doesn't hold the lock** [low, weak-spot] — `fixed`
`engine_diagnostics.py:27-59` — each inner `generate()` acquires per
call; another request can interleave and skew timings.
*Fix:* hold `_gen_lock` for the whole loop.

**EC-131 — `_log` falls back to raw `print()`** [low, weak-spot] — `fixed`
`forge_engine.py:246-250` — `logging.getLogger().handlers` check — with
only named-logger handlers configured, engine logs go to stdout
unfiltered.
*Fix:* `logger.hasHandlers()` or always emit via logger.

**EC-132 — `_safe_decode_ids` no negative clamp** [low, bug] — `fixed`
`engine_generation.py:1747-1760` — `t if t < tok_vocab else tok_vocab-1`
passes negatives through to `tokenizer.decode` → decode error.
*Fix:* clamp `0 <= t < tok_vocab`.

**EC-133 — `stats()`/`diagnose()` miss subsystems** [low, weak-spot] — `fixed`
`engine_diagnostics.py:61-79` — omits `_suffix_mem.info()`,
`hotswap.get_settings()`, `session_stats()`, tools registry size.
*Fix:* add the blocks.

**EC-134 — QK-norm scan assumes `model.blocks`** [low, weak-spot] — `fixed`
`engine_checkpoints.py:839-843` — `for block in model.blocks:
block.attn` — AttributeError on non-Forge archs (GGUF/HF naming) inside a
supposedly generic load path.
*Fix:* `getattr(model, "blocks", ())` + per-block `hasattr`.

**Verified live — NOT dead** (corrections to over-eager dead-code calls):
`_chunked_prefill` attr (dynamic wiring), `generate_with_scaling`,
`generate_cascade`, `attach_head`/`detach_head`/`head_report` (FluxLM
persistence), `library_*` wrappers (tool handlers), `has_lora`/`lora_info`/
`load_lora`/`unload_lora` (forge_server:819-853 + GUI lora services),
`generate_with_tools` (forge_server:1151), `decide`/`generate_gated`/
`load_gate_probes` (gated pipeline), CrashRecoveryManager. All 177 mixin
methods have unique names — no mixin collisions. This narrows EC-070:
`generate_gated` and the head-attach trio are wired; `apply_flux_packets`
and `profile_bandwidth` remain unreferenced.

### Plan cross-reference & production-parity findings (EC-135+)

Round-2 pass: fresh code verification + cross-reference with
`docs/ENGINE_IMPROVEMENT_PLAN.md` (2026-09-25, "awaiting execution") +
online research on vLLM 0.12 / llama.cpp / SGLang / Thinking Machines.
Items noted "(plan N.N)" were already documented open in the plan —
confirmed still open by fresh reads.

**EC-135 — `stop=` never stops decode** [high, bug] — `fixed`
`engine_generation.py:301-312` passes no `stop` to
`self.decoding.generate` (it isn't in the signature); `:343-352`
truncates `result` at the first `find()` hit **after** full generation —
the model decodes to EOS/`max_new_tokens` regardless. Wasted compute,
wrong `n_gen`, and on infinite-context runs potentially unbounded.
*Fix:* incremental suffix match inside the decode loop (plan's
ring-buffer suggestion) or pass `stop` through `SamplingParams` to the
strategy loop. (Plan 0.4.)

**EC-136 — left-pad pollutes Mamba/conv state** [high, bug] — `open`
`batched_decoding.py:97-127` — zero left-pads + `attention_mask` into
prefill; the mask guards the 2 attention layers only — the 26
Mamba/conv layers integrate pad tokens into SSM/conv state. Every
batched generation with unequal prompt lengths produces polluted state.
The decode-time mask fix at :124-127 acknowledges the problem for
attention KV, not for SSM.
*Fix:* per-row sequential prefill through state slots, or zero the pad
contribution at the SSM layer (mask-aware conv/state update). (Plan 0.5.)

**EC-137 — SSM state passed through on mid-entry prefix hits** [med, bug] — `fixed`
`prefix_cache.py:373-385` — `_slice_past_kv` slices `(k,v)` to `length`
but passes dict SSM entries untouched. Correct only when the hit lands
exactly on a stored entry boundary; any mid-entry KV slice pairs
truncated attention KV with full-length SSM state → wrong position.
*Fix:* reject partial hits on stateful layers until chunked SSM
snapshots land (EC-142), or store per-chunk `(ssm, conv)` snapshots.
(Plan 0.1.)

**EC-138 — "xgrammar" is not constrained decoding** [high, missing-feature] — `open`
`structured/xgrammar.py` — compiles JSON schema to a **character-level**
FSM and masks tokens by *first character only* — a token whose first
char is legal but whose tail violates the schema still passes
(e.g. `":garbage`). Plus a 65k-iteration Python mask per step. No
xgrammar/llguidance/outlines dependency in requirements.
*Fix:* real xgrammar (`TokenizerInfo` + `GrammarMatcher` bitmask
overlapped with forward — ~36 µs/token, Windows wheels exist) or
llguidance. (Plan 2.3.)

**EC-139 — launch-bound decode; kernels exist unwired** [high, missing-feature] — `open`
Python SSM scan at `forge/model/mamba_probe.py:164-189` (~800 kernel
launches/token on V2's 26 mamba layers + 2 attn); `compile_decode_step`
(`llm.py:350`) never invoked; `scheduler/async_d2h.py` (pinned+event
readback) imported only by self_play. Vendor vLLM's Triton
`causal_conv1d_update` + `selective_state_update` (Apache-2.0, runs on
triton-windows) — prerequisite for graph capture.
*Fix:* plan 1.2/1.8/1.9 + Phase 4. (EC-143.)

**EC-140 — per-token host syncs** [med, weak-spot] — `open`
`.item()` at engine_generation.py:1195/:1590, `ids[0].tolist()` at
:260/:1496/:1587/:1834/:2023, per-token `tokenizer.decode` — each is a
GPU pipeline drain per decode step. vLLM 0.12's GMR-V2 Triton-native
sampler (device-resident, per-request seeds) is the direction.
*Fix:* preallocated GPU int32 `generated_ids` buffer; EOS via
`torch.isin`; sync only on stop. (Plan 2.2.)

**EC-141 — streaming emits mojibake for multi-byte chars** [med, bug] — `fixed`
`_safe_decode_ids` (engine_generation.py:1747-1760) decodes each token
id individually — a UTF-8 char split across tokens yields `\ufffd`
replacement chars in SSE chunks.
*Fix:* HF `DecodeStream` (tokenizers ≥0.22) per request; hold back
`max_stop_len-1` chars for stop matching (plan 2.6).

**EC-142 — chunked mamba-state prefix snapshots** [high, missing-feature] — `open`
Plan Phase 3: store `(ssm, conv)` per 256-token chunk alongside KV —
enables true partial-prefix reuse on a hybrid SSM (vLLM disables prefix
caching on mamba layers entirely — this would exceed production
engines). Mamba state is fixed-size (~35-70 MB) — cheap snapshots.
Fixes the EC-137 class of bug properly.
*Effort:* M. Precedent: vLLM Mamba2 APC (#30877), SGLang MambaRadixCache
(#11214), Marconi (MLSys'25).

**EC-143 — unified state pool + single-graph decode** [high, missing-feature] — `open`
Plan Phase 4: per-layer `[num_slots, dims]` conv+SSM pools +
`state_indices_tensor` indirection → capture the whole decode step as
one CUDA graph (~800 launches → 1 replay; est. 3-8× at bs=1 on this
launch-bound Windows box — WDDM makes launch overhead *worse*). vLLM
ships full mamba graphs by default since v0.11.0 — copy the layout.
Gotcha: `#slots ≥ max_captured_batch_size`. PDL available on SM120.

**EC-144 — continuous batching** [med, missing-feature] — `open`
Plan Phase 5: per-row mamba-state slots + row scheduler evicting
finished rows / admitting queued between replays — replaces the static
left-padded `BatchedDecoding` (which also has EC-136's pollution bug).
Pairs with cooperative cancellation (EC-066).

**EC-145 — no fused dequant-GEMM kernels** [high, missing-feature] — `open`
All quant paths (forge_quant, quamba2 W4A8, w8a8, bitnet b1.58, nvfp4)
dequantize-then-matmul — flagged as NC6 in the Sept-9 critique, still
open. Marlin runs on SM120 (vLLM's 5090 fallback); GemLite is
pure-Triton with explicit sm_120 focus (covers W2A8 b1.58 + NVFP4);
`torch._scaled_mm` gives native FP4 on sm120 for free.
*Fix:* vendor Marlin or GemLite for W4A16/W8A16 paths (plan 8A).

**EC-146 — no split-KV decode attention** [med, missing-feature] — `open`
At bs≤8 + MQA, decode attention leaves most of the 5070's ~48 SMs idle;
split-KV/FlashDecoding parallelizes over KV length — biggest long-ctx
decode win after graphs (plan 8B).

**EC-147 — varlen packing leaks conv/SSM state across docs** [high, bug] — `open`
`attention_ops.py:50` has cu_seqlens varlen support but conv/SSM state
never resets at document boundaries → cross-document context leak in
packed sequences (P16).
*Fix:* reset per-doc state on `cu_seqlens` boundaries in the conv/SSM
layers.

**EC-148 — sm120 NVFP4 trap** [med, weak-spot] — `open`
vLLM #47749: "FP4" on 5090 silently resolves to Marlin W4A16 — FP4
tensor cores idle. Our nvfp4 path needs kernel-selection logging to
prove it isn't dequant-only.
*Fix:* log kernel selection; assert tensor-core path in a test.

**EC-149 — scheduler cheap wins** [low, missing-feature] — `open`
Absent per plan 8C: recompute-preemption + LCF victim
(−20-59% wasted recompute), `max_num_batched_tokens` tuning
(~1-2k on 12GB), prefix-aware admission (`lpm` policy — sort queue by
prefix-hit length), `stream_interval N` output coalescing (57% e2e in
I/O-bound tests).

**EC-150 — platform verification items** [low, missing-feature] — `open`
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` now works on Windows
single-process (torch PR #189163 — check our torch version); ReBAR must
be verified enabled (CPU→GPU streamed weights/KV benefit);
`cudaMallocAsync` is flaky on WDDM — probe
`cudaDevAttrMemoryPoolsSupported` first.

**EC-151 — no batch-invariance/determinism mode** [med, missing-feature] — `open`
Thinking Machines' `batch_invariant_ops` (torch.Library swap) +
vLLM's `VLLM_BATCH_INVARIANT` ship batch-size-independent outputs —
vLLM measured **+18.1% throughput** from the invariant BMM alone
(#29345). Relevant twice: (a) self-play/RL rollouts need deterministic
generation for reproducibility, (b) batch-position-dependent outputs
compound EC-136's pollution. No seed plumbing end-to-end either
(EC-156).

**EC-152 — server endpoint gaps** [high, missing-feature] — `open`
Verified vs llama.cpp + vLLM + plan 6b (updated with post-plan items):
- **Debug/utility**: `/tokenize`, `/detokenize`, `/props`
  (model+runtime defaults), `/apply-template` dry-run, `/version`.
- **Ops**: `/metrics`, ready-vs-live `/health` split, per-request
  latency headers (`x-queue-time`/`x-inference-time`/`x-time-per-token`),
  `/abort_request`, graceful SIGTERM drain, request logging +
  X-Request-ID (one route only today — EC-086), OTel (nice-to-have).
- **OpenAI-compat fields**: `stream_options.include_usage`,
  `prompt_token_ids`/`input_ids`, `n>1` (vLLM 0.12's shared-prefill
  delay-split — one prefill serves n decode children), `min_tokens`,
  `stop_token_ids`, `presence/frequency_penalty`, `logit_bias`,
  `response_format` (blocked by EC-138), `chat_template_kwargs`, model
  aliases.
- **Queue mgmt**: per-request `priority`/`timeout`,
  `max_concurrent_requests` backpressure + queue timeout.
- **Model mgmt**: HF-hub JIT download + `keep_alive`/TTL unload, GGUF
  hub pull (loader exists, no fetch), revision pinning
  (`hf://id@rev`), response cache (exact-match dedup — trivial on
  single GPU), batch JSONL runner (`vllm run-batch` shape),
  `bench_serving` script, env-var config mirroring.
- **llama.cpp-specific worth copying**: `/slots`-equivalent session/state
  visibility, per-request LoRA scale/selection (`/lora-adapters` get+set),
  context checkpoints per slot (usable version of our write-only
  `kv_snapshot` — EC-018), `--cache-ram` byte-budgeted prompt cache.

**EC-153 — plan↔critique status sync** [med, docs-drift] — `open`
`ENGINE_IMPROVEMENT_PLAN.md` (9/25, "research-complete, awaiting
execution approval") overlaps ~20 EC entries — our audit independently
re-confirmed plan items 0.4/0.5/0.8/0.10, 1.1/1.2/1.7-1.9, 2.x, 4.x,
6.x. The plan's DONE markers and this DB's `open`/`fixed` statuses
should be kept in sync (e.g. 0.2/0.6/0.7/0.9/1.3-1.7 marked DONE there;
EC-020's B==1 param drop shows plan 1.7 is only partially true). The
plan sat unexecuted for a month — EC entries now carry the fresher
evidence.

**EC-154 — embedding surface** [med, missing-feature] — `open`
`/v1/reranking` and `/v1/embeddings` endpoints exist, but `embed()` has
no `pooling={mean,cls,last}` option and no `normalize` flag (llama.cpp
ships `--pooling {none,mean,cls,last,rank}`). Moot until EC-096 (embed
returns logits) is fixed — then pooling/normalize become the contract
question.

**EC-155 — caches lack byte budgets** [low, weak-spot] — `open`
`session_cache`/`prefix_cache` are entry-count or uncapped; llama.cpp's
`--cache-ram` MiB budget is the safer model for a 12GB box.
*Fix:* byte-budgeted eviction (pairs with Phase-3 tiering).

**EC-156 — seed plumbing partial** [low, weak-spot] — `fixed`
`seed` reaches `registry.generate_stream` (plan 1.7) but not all
sampling paths; combined with EC-136/EC-151 there's no determinism
story even at temperature=0 (batch position affects results).

**EC-157 — `logprobs[].bytes` is a token id, not bytes** [low, bug] — `fixed`
`engine_generation.py:1914,1920,1931,1938,1962,1968` — every logprobs
entry sets `"bytes": token_id` — an int — while the OpenAI spec defines
`bytes` as the token's UTF-8 byte values (list of ints). Clients that
parse `bytes` break.
*Fix:* `"bytes": list(token_str.encode("utf-8"))` or drop the field.

**EC-158 — tool design vs security best-practice** [med, security] — `fixed`
OWASP LLM01 + Anthropic MCP guidance for local engines: (a) secrets must
be handled in code, never reachable by model tools — violated via
`.env` read path (EC-097); (b) read/write tool annotations driving
auto-approve tiers — our approval flow exists but `file_read`/`file_list`
aren't even in the checked set; (c) the security config itself must be
outside tool-writable space (EC-160).
*Fix:* shared read/write secret deny-list; `readOnlyHint`-style tiers
(auto-approve reads, confirm destructive); `sandbox.json` moved to a
protected path.

**EC-159 — legacy logprobs re-runs decode** [low, weak-spot] — `fixed`
`engine_generation.py:1943-1969` — when `captured_steps` is None, the
logprobs path re-executes the entire generation one token at a time for
per-step logits (O(n) extra forwards). Fallback path exists; no warning.
*Fix:* warn + prefer the captured path; consider deleting legacy.

**EC-160 — `sandbox.json` is model-writable** [med, security] — `fixed`
Plan 6.2: the tool-security config lives under `research/data` (a
read_write zone) and isn't in `PROTECTED_ENGINE_FILES` — the model can
rewrite its own sandbox policy, then exercise the loosened rules.
*Fix:* move out of writable paths; deny-list it; auth-gate
`approve_request` (also EC-058).

**EC-161 — registry VRAM accounting is declared, not measured** [low, weak-spot] — `fixed`
Plan 6.5: registry budgets use `1.5×ckpt` declared VRAM instead of a
`torch.cuda.mem_get_info` delta measured at load; `is_awake` doesn't
reflect sleeps triggered outside the registry.
*Fix:* measure at load; sync `is_awake` from engine state.

**EC-162 — crash snapshots aren't atomic** [med, weak-spot] — `open`
Plan 6.6: snapshot writes lack temp+rename+checksum — a crash mid-write
leaves a torn file that `recover()` may read back (with the
weights_only=False fallback, EC-038).
*Fix:* atomic write; checksum; validate on load.

**EC-163 — untrusted tool outputs enter context untagged** [med, security] — `fixed`
Web/file tool results are concatenated into the transcript with no
untrusted-content tagging — an indirect prompt-injection path: hostile
page/file content can steer the model into dangerous tool calls (the
approval flow only covers configured-risk tools — and `file_read` is
free per EC-097). The unwired `safety/` subpackage (EC-013) is where
this belongs.
*Fix:* wrap tool outputs in a tagged block; optional content scan;
wire `safety/` or delete it.

**EC-164 — sparse-attn wrappers drop `past_key_value` entirely** [high, bug] — `fixed`
`kv/mosa.py:205-212` (and same code shape at `scheduler/triroute.py`): in
`mosa_forward`, the non-preallocated branch sets `k_cache, v_cache = k,
v` — `past_key_value` received but never concatenated; `new_kv` returns
only the current step's K/V, so every decode step after prefill attends
to nothing but the latest token. The preallocated branch is also wrong:
`k_caches[layer_idx][:, :, :T]` reads the first T positions of the pool
instead of `[:cur_len]` — T=1 decode attends to position 0 only. RoPE
`offset=0` hardcoded (harmless on no-RoPE V2, wrong generally).
*Fix:* `torch.cat([past[0], k])` / `cat([past[1], v])` on the tuple
path; `[:, :, :cur_len]` on the pool path; `offset=past_len`. Same fix
in triroute. (Plan 0.6 — the part it left open.)

### Decode-path deep audit (decoding strategies, graphs, scaling)

**EC-165 — `mtp_selfspec` dead-on-arrival** [high, bug] — `fixed`
Three stacked failures: (a) `_activate_decoding` and
`MTPSelfSpecDecoding.generate` check `hasattr(model, "mtp_head")` — the
model registers `self.mtp_module` (llm.py:141) → always falls back
silently; (b) `mtp.py:377` sets `self.heads[0] = None` inside a
`ModuleList` → `state_dict` can never contain `mtp_module.heads.0.weight`
→ the keystack "mtp" probe (engine_checkpoints.py:80, looks for
`mtp_head.heads.0.weight`) can never fire → auto-activation of
`mtp_selfspec` is dead code; (c) even if reached,
`mtp(hidden[:, -1:])` calls `MTPModule.forward(hidden, token_embeds,
targets)` with 1 arg → TypeError, and `mtp_out` is `(loss, logits_list)`
so `.argmax` would hit the scalar loss. Also `decode_kwargs["k"]=4`
contradicts the advertised k=7.
*Fix:* pass `model.mtp_module`; give `MTPModule` a `draft_logits(hidden)`
API; tie `heads[0].weight` instead of nulling the module; align the
keystack probe key.

**EC-166 — EAGLE-3 emits wrong output + no rollback** [high, bug] — `fixed`
`decoding.py:1742-1792` — accepted drafts go only to the Python
`generated` list; `_eagle_tokens` (the tensor actually returned) only
gets `next_token` → output sequence missing accepted drafts. `presents`
accumulates KV for all drafted tokens, reused next round un-sliced →
attends to rejected tokens. `cur_fused = fused[:, -1:]` should be
`[:, n_accepted]` on rejection. EOS = single hardcoded id.
*Fix:* append accepted drafts to `_eagle_tokens`; `_slice_kv` rollback
(like SpecBound); `fused[:, n_accepted:n_accepted+1]`; `_eos_token_ids()`.

**EC-167 — `medusa`/`dspark` crash paths** [high, bug] — `fixed`
`build_decoding("medusa")` → `MedusaDecoding()` with required positional
`medusa_heads` → TypeError at activation. `dspark` builds, then
`dspark_generate` calls `dspark_head.eval()` on `self.dspark=None` →
AttributeError mid-run.
*Fix:* `medusa_heads=None` + StandardDecoding fallback; same guard in
dspark; or remove both names.

**EC-168 — medusa/dspark helpers are no-cache O(T²) stubs** [high, bug] — `fixed`
`medusa_generate`: `hidden = out[0]` is *logits* (vocab-dim) → the
`hidden.size(-1) == vocab` check is true → heads unreachable; the loop
is an O(T²) no-cache greedy. `dspark_generate` calls
`model(input_ids, return_hidden=True)` with no `past_key_values` →
O(T²·block) per step.
*Fix:* `return_hidden=True` + `past_key_values`/`use_cache`, or delete
the helpers.

**EC-169 — `unpack_output_with_kv` contract bug** [high, bug] — `fixed`
`forge/model/kv_cache.py:208-211` — `out[2] if len(out) > 2 else out[1]`:
for `use_cache=False` the model returns `(logits, loss)` → **loss tensor
returned as past_kv**; for `return_hidden=True` without cache → hidden
states returned as past_kv. 49 call sites use this helper; any caller
feeding the second element back as `past_key_values` corrupts the next
forward.
*Fix:* return `past_kv` only when the tuple's last element is a
presents-list (or plumb a `has_cache` flag).

**EC-170 — BreakableCudaGraph never captures, reports captured** [high, bug] — `fixed`
`graphs/breakable_cuda_graph.py` — every call passes
`position_ids=static_pos` but `ConfigurableResearchLLM.forward` has no
`position_ids` param → TypeError; warmup and capture swallow it
(`except: break`/`continue`) leaving `_decode_graphs` empty while
`_captured=True` → `stats()` reports success. `capture_prefill_segments`
is literally `pass`. `run_decode` would slice the presents *list* not
the batch dim.
*Fix:* drop `position_ids` (model derives it); report real capture
status; delete the stub or implement it.

**EC-171 — Foundry/CoRun same TypeError + fake materialization** [high, bug] — `fixed`
`graphs/foundry.py:227,235`, `scheduler/corun.py:80-136` — same
`position_ids` TypeError → `materialize` returns False, eager fallback
raises. `eval(dtype_str)` result discarded; dtype guessed by buffer
*name*. "Reconstruct in <1s without capture" still runs a full
`torch.cuda.graph()` capture. `_h_foundry` logs success regardless.
*Fix:* drop `position_ids`; fix fallback call; honest status.

**EC-172 — telemetry reports requested, not effective, strategy** [high, weak-spot] — `open`
`_log(f"Decoding: {self.decoding.name}")` + `_record_output` record the
requested name, but Eagle3/MTP/SpecBound-b>1 silently instantiate and
run a fresh `StandardDecoding` — events/history/diagnostics credit
speculative speedups to runs that never speculated.
*Fix:* `effective_name` recorded at fallback time + warning.

**EC-173 — FirstFinishSearch can't short-circuit** [high, bug] — `open`
`test_time_scaling.py:140-177` — `return out` sits inside
`with ThreadPoolExecutor` → `shutdown(wait=True)` waits for ALL N
futures; and every sample serializes on `_gen_lock` anyway → N×
wall-time for a pick-first policy. `model.seed(seed)` is a documented
no-op on `_ScalingModelAdapter` → samples differ only by RNG order.
*Fix:* drop the pool (sequential loop + early exit) or implement real
parallelism; implement or remove `seed()`.

**EC-174 — DRY penalty is quadratic per request** [med, weak-spot] — `open`
`_decode_tokens` builds `dry_ctx + generated_ids` fresh each token;
`_dry_penalties` rescans `last_n` positions per token → O(n²·last_n)
host compute; `repetition_penalty` builds `set(generated_ids[-64:])` +
H2D tensor per step.
*Fix:* incremental n-gram index; device-resident last-64 window.

**EC-175 — `top_k` contract split** [med, bug] — `fixed`
`_validate_generation_params` rejects `top_k <= 0` while
`_sample_next_token` has an explicit `top_k == 0` (disabled) branch —
unreachable dead code. `_sample_from_logits` (decoding.py) does
`torch.topk(logits, top_k)` unclamped → `top_k > vocab` (validation
allows) → RuntimeError; engine path clamps.
*Fix:* `top_k <= 0` = disabled; clamp identically in both samplers.

**EC-176 — decide()/PRM materialize full-vocab logits for hidden reads** [med, weak-spot] — `open`
`decide.py` `_candidate_hidden`/`_continuation_scores` and
`decision_head.py:379` call `model(batch, return_hidden=True)` — runs
lm_head over the full padded batch; at max_batch=32 × 8k ctx × 151k
vocab that's ~150GB transient logits → guaranteed OOM. Same tuple-arity
fragility as EC-169.
*Fix:* `logits_last_only=True` or `layer_indices`-only forwards; skip
the head when only hidden is needed.

**EC-177 — `_forge_last_*` written on aborted exits** [med, bug] — `fixed`
`engine_generation.py:1238-1241` — `finally` writes `_forge_last_kv`/
`_forge_last_logits` on every exit including `GeneratorExit`/
exceptions → a later request reading them sees a partially-generated
sequence's KV.
*Fix:* write only on normal completion (flag) or record token count.

**EC-178 — VRAMManager mostly dead + third decode loop** [med, dead-code] — `open`
`runtime/vram_manager.py` — only `max_gen_tokens` is called
(engine_activation.py:298); 6/7 methods dead incl.
`check_during_generation` ("abort before OOM" — dead).
`profile_after_warmup` re-implements an O(n²) `torch.cat` decode loop.
Manager instantiated fresh per activation — profiles discarded.
*Fix:* wire `check_during_generation` into the decode loop or delete;
keep one instance on the engine.

**EC-179 — SpecBound draft is O(d²) + conv churn** [med, weak-spot] — `fixed`
`decoding.py:1504-1533` — `_draft_logits` re-forwards the growing
`suffix` with `use_cache=False` per draft token (O(d²) partial
forwards); `_install_conv_prefix(snap)` mutates live conv state every
iteration; 2 `.item()` syncs per cycle. Correct but wasteful.
*Fix:* chain cached recurrent state per draft step; hoist conv install.

**EC-180 — PRM boundary tokenization fragility** [low, weak-spot] — `open`
`decision_head.py:353-372` — `_step_end_positions` encodes each step
separately; merges across boundaries can shift positions vs a jointly
tokenized transcript.
*Fix:* tokenize the joined text once; locate delimiter ids.

**EC-181 — `_adaptive_spec` write-only + bulk inert auto-enable** [low, dead-code] — `open`
`feature_registry.py:78-83` assigns `eng._adaptive_spec`, never read.
`_load_quantize` profile setdefaults ~8 features "on by default"
(`adaptive_spec`, `suffix_spec`, `faser`, `lookahead_gate`,
`pod_attention`, `hybrid_prefill`, `hotprefix`, `learned_prefix_cache`)
— most have no generation-path consumer; the "everything safe ON"
comment misleads.
*Fix:* route `self.decoding` through `_adaptive_spec` or delete; trim
the auto-enable list to verified consumers.

**EC-182 — block_fusion discards compiled blocks** [low, dead-code] — `open`
`graphs/block_fusion.py` — `block = torch.compile(block)` assigned to a
local, never written back to `model.blocks`; zero generation call sites;
auto-enabled at engine_checkpoints.py:399 when VRAM is ample → reports
"BlockFusion active" doing nothing.
*Fix:* write compiled blocks back; wire `run_model` or remove flag.

**EC-183 — MTPSelfSpec EOS check reads missing attr** [low, bug] — `fixed`
`decoding.py:633,665` — `getattr(model, "eos_token_id", None)` —
`ConfigurableResearchLLM` has no such attr → always falsy → MTP loop
(moot per EC-165) would run to `max_new_tokens`.
*Fix:* same eos_set construction as StandardDecoding.

Merge notes into earlier entries:
- **EC-114** — add: on success the engine *permanently* swaps to the
  fallback KV strategy for the session (`_activate_kv_cache("s4r")` then
  `"cpu_offload"` at engine_generation.py:1716,1727) with only a log
  line; restore the user's strategy or warn explicitly.
- **EC-018** — add: `snapshot_kv_cache` reads `kv.k_cache`/`v_cache`/
  `seq_len` — attrs no strategy exposes (`hasattr` always False) → the
  per-token call at engine_generation.py:1219 never writes a byte.
- **EC-032** — add: also `decoding.py:841` discarded `eos_tensor`.
- **EC-113** — add: `learn=False`/`mem_bias` are also silent no-ops on
  non-memory models with no warning (engine_generation.py:154-181).

### GUI server & remote-surface audit (forge_gui_server, forge_gui/api)

**EC-184 — `/agent/runs` is unauthenticated RCE** [high, security] — `fixed`
`routes/chat.py:231-280` + `agent_tools.py` — `AgentStartRequest` takes
`workspace` verbatim (no confinement under the project) and
`approval_mode` accepts `"none"` → `_needs_approval` False for
everything. Agent tools (`write_file`/`delete_file`/`run_python`/
`run_cmd`) then jail to whatever dir the caller picked; `run_cmd`'s
allowlist contains `python`/`pip`/`git` (`python -c` = arbitrary code)
and the basename check passes `..\x\python.exe`.
Net: `POST /agent/runs {"workspace":"C:\\","approval_mode":"none"}` =
remote write + exec.
*Fix:* whitelist approval_mode; `resolve()`+`is_relative_to` workspace;
drop interpreters from `CMD_ALLOWLIST` or resolve `parts[0]` first.

**EC-185 — `/tasks/launch` unauthenticated exec** [high, security] — `fixed`
`routes/system.py:38-48` → `procs.launch` →
`create_subprocess_exec(*cmd)` on caller `command`. `/launch/preset`
passes caller `overrides` into `build_command` → arg injection into
"blessed" presets.
*Fix:* auth; restrict to preset list; overrides as values for declared
args only.

**EC-186 — backup restore: wipe-then-zip-slip** [high, security] — `fixed`
`support.py:547-588` — `_restore_backup` deletes every non-skipped file
under project_root FIRST, then `zf.extractall()` on a caller-supplied
zip with no member validation (`../`/absolute members write anywhere);
corrupt zip = wiped project + failed restore. `backup_path` isn't
confined to `backup_dir`.
*Fix:* validate members (reject absolute/`..`/drive-letters), extract
to staging then swap; require zip under `data/backups/`.

**EC-187 — `DELETE /models` path traversal** [high, security] — `fixed`
`routes/core.py:582-601` — `relative_to()` matches segments lexically:
`path="research/checkpoints/../../../Windows/Temp/x.pt"` passes the
prefix check, `is_file()` resolves `..`, suffix passes → delete any
`.pt`/`.safetensors` on the host. Also `p.with_suffix(".meta.json")`
computes `model.meta.json` but real sidecars are
`model.safetensors.meta.json` → orphaned on every delete.
*Fix:* `resolve()` then `is_relative_to`; iterate `str(p)+".meta.json"`.

**EC-188 — merge endpoints: arbitrary paths + dead calls** [high, bug] — `fixed`
`merge_checkpoints(parents=[any], out_path=any)` reads/writes anywhere;
`/lora/merge` same. `engine_merge` passes `hot_swap=` — engine kwarg is
`load_result=` — swallowed by `**kwargs` → hot-swap never happens.
`evolve_merge` called with `out_path=,generations=,population=...` —
real signature is `n_generations=,population_size=,out_dir=` with no
`**kwargs` → TypeError on every call.
*Fix:* confine paths; rename kwarg; align call signature; one route
test would have caught both.

**EC-189 — model switch stalls the event loop** [high, weak-spot] — `fixed`
`engine_rt.py:141-181` — `load()`→`unload()` runs `engine.sleep(2)`
(acquires `_gen_lock` — waits for in-flight decode) + `del engine` +
`empty_cache()` synchronously on the loop thread.
*Fix:* offload `unload()` into the pool / `run_in_executor` the whole
`load()`.

**EC-190 — orphan generation; cancel tokens never wired** [high, weak-spot] — `fixed`
`gen.py:78`, `chat_loop.py:534`, `routes/chat.py:188-220,345-355` —
`run_in_executor` futures discarded; no `StreamCancel`/`ChatCancel`
created or passed. On disconnect the consumer dies but the worker keeps
generating (holds the engine lease) and keeps executing file/python/LoRA
tools for a dead client; unbounded queue buffers all output.
*Fix:* `finally`/`GeneratorExit` → `cancel.cancel()`; set cancel on
`request.is_disconnected()`.

**EC-191 — `train_lora` dead three ways** [med, bug] — `fixed`
`lora_training_trigger.py:161-234` — uses `--lora-rank`/`--epochs`/
`--output` but `sft_train.py` defines `--lora-r`/`--max-steps`/`--save`
→ argparse exit 2. `_export_chat_data` passes the path as `conv_ids`
(char-set → empty export) then compares `(Path,int)` tuple to `0`.
`_check_training` calls `proc_mgr.processes()` — doesn't exist.
*Fix:* fix flags; `export_training_data(out_path=)`; `all_tasks()`.

**EC-192 — `/finetune` points at the migrated-away path** [med, bug] — `fixed`
`routes/train.py:108-151` — builds `research/training/runners/
sft_train.py`; file lives at `forge/training/runners/` (EC-001's
migration strikes again).
*Fix:* `-m forge.training.runners.sft_train`; `is_file()` check at
launch.

**EC-193 — `POST /lorebook` always 500s** [med, bug] — `fixed`
`routes/system.py:176-184` — `add(tags=..., description=...)` kwargs
`Lorebook.add()` lacks.
*Fix:* drop kwargs or extend the signature.

**EC-194 — lorebook injection discarded; full JSON save per message** [med, bug] — `fixed`
`routes/chat.py:178-182` — `inject()` returns `str`; route does
`history.extend(injected if isinstance(injected, list) else prior)` —
never a list → memory text thrown away, while `inject()` still bumps
counts and rewrites the whole file.
*Fix:* append to system prompt; `save()` only when counts changed.

**EC-195 — GUI reactivate wipes config from partial dict** [med, weak-spot] — `fixed`
`routes/core.py:116-119` + `engine_rt.py:346-389` —
`ActivationConfig.from_kwargs` drops unknown keys silently;
`activate(**partial)` rebuilds all ~50 flags; route returns `ok:True`
even when reactivate refused; `active_config` reports *requested* not
*applied* params (compounds EC-172).
*Fix:* validate+warn; result object; record applied names.

**EC-196 — `TaskBootConfig.differs_from` checks 4/14 fields** [med, bug] — `fixed`
`session_manager.py:80-114` — compares only kv_cache/decoding/quantize/
acceleration, gated behind `!= "standard"` → `paged→standard` reverts
undetected; `apply_to` → `engine.activate(**14)` wipes all other flags.
*Fix:* diff full `to_dict()` vs `active_config`; merge-apply.

**EC-197 — queue/future lifecycle leaks** [med, weak-spot] — `fixed`
`BatchQueue.stop()` doesn't drain `_pending` or resolve futures (routes
wait the full 120s on a dead queue); `_push_stream`'s
`run_coroutine_threadsafe` future dropped (closed loop → silent);
task-message timeout returns 504 while the `PendingRequest` keeps
running → history mutated after the call returned.
*Fix:* fail pending on stop; cancel on timeout; skip history append for
cancelled requests.

**EC-198 — no per-task serialization** [med, bug] — `fixed`
`forge_server.py:1402-1433` — history rendered before `append_message`,
assistant appended later in dispatcher order → two concurrent POSTs to
one task interleave `user2,user1,assistantA,assistantB`.
*Fix:* per-task asyncio lock; render at dispatch time.

**EC-199 — registry lock across sleep/wake** [med, weak-spot] — `fixed`
`model_registry.py:333-395` — `_evict_lru`→`sleep(1)` under
`self._lock`; sleep waits on `_gen_lock` → one in-flight decode freezes
`generate()`/`stats()`/`list_models()` for ALL models. No in-flight
accounting — busy engines are eviction candidates (they just block).
*Fix:* in-flight counter; pick victims under lock, sleep outside.

**EC-200 — `learn()` monopolizes the shared worker** [med, weak-spot] — `open`
`engine_rt.py:394-489` — `_learn_blocking` runs on the 1-worker pool
shared with load/reactivate → hours-long ingest queues model ops; each
8K slice acquires/releases the lease → generations interleave mid-corpus
observing half-learned state.
*Fix:* dedicated executor (or 2 workers); doc-level resets inside the
ingest lease.

**EC-201 — `shutdown()` stops nothing** [med, bug] — `fixed`
`engine_rt.py:499-505` — `asyncio.wait_for` on a
`concurrent.futures.Future` → TypeError → swallowed → load never
awaited; `_learn_task` never cancelled; `unload()` never called;
`_pool.shutdown(wait=False)` leaves work running during teardown.
*Fix:* `asyncio.wrap_future`; cancel+await learn; `unload()`;
`shutdown(wait=True, cancel_futures=True)`.

**EC-202 — `make_harness` captures a dead loop** [med, bug] — `fixed`
`deps.py:102-126` + `support.py:661-695` — harness factory invoked on
the `agent` pool thread → `get_event_loop()` binds a dead/absent loop →
`run_coroutine_threadsafe` against it → every LoRA tool hangs 180s.
*Fix:* capture the loop once in `services.start()`; pass explicitly.

**EC-203 — periodic backup zips on the loop; weak change hash** [med, weak-spot] — `fixed`
`support.py:453-534` — `create_backup()` (walk + zipfile) runs on the
event loop every 60s when changes exist; `_hash_file = md5(size +
first 4096 bytes)` → edits past 4KB invisible → skipped backups.
*Fix:* offload like the route does; hash mtime+size+sampled blocks.

**EC-204 — LoRA scans + single-worker starvation** [med, weak-spot] — `fixed`
`services/lora.py` — `scan_lora_adapters()` (rglob + header parse) runs
inline in async handlers; the shared 1-worker `_pool` hosts
multi-minute merges → `load_on_engine`'s 60s lease times out.
*Fix:* `run_in_executor` scans; second worker for merges.

**EC-205 — LoRA auto-swap window + lock-free state** [med, bug] — `fixed`
`services/lora.py:67-87,195-246` — unload and load are separate worker
actions → engine runs unadapted between them while `status()` claims a
mode; `_current`/`_pinned`/`_mode`/`_busy` mutated lock-free →
interleaved REST calls reorder state; failed reload leaves `_current`
disagreeing with the engine.
*Fix:* service-level lock across the swap; restore `_current` on load
failure.

**EC-206 — LoRA endpoints: arbitrary paths + unclamped rank** [med, security] — `fixed`
`routes/core.py:639-678` — adapter path falls back to raw input →
absolute/`../` grafts any `.safetensors` on the host into the live
engine (read + weight-poison); `rank`/`alpha` uncapped → VRAM blowup.
*Fix:* confine to checkpoints/lora dirs; clamp rank/alpha; confine out.

**EC-207 — `read_log_tail` whole-file + `status_file` unvalidated** [med, weak-spot] — `fixed`
`routes/core.py:64-67` — `readlines()` entire log on the loop (multi-GB
stall); `status_file` query param → read any dir's `log.txt`
(unauthenticated file read).
*Fix:* seek-from-end tail; confine path; offload.

**EC-208 — `all_tasks()` deep-copies buffers; per-line log I/O** [med, weak-spot] — `fixed`
`services/procs.py` — `asdict(self)` copies up to 5000 lines per task
per `/status` poll; `_emit_line` opens/appends/closes `log.txt` per
output line on the loop; `kill` stamps "killed" on already-finished
tasks.
*Fix:* lazy tail; buffered writes; only kill live procs.

**EC-209 — stdio MCP: races, leaks, dead wiring** [med, bug] — `fixed`
`mcp_client.py` — `_send_request` writes+reads with no lock and never
matches `resp["id"]` → concurrent calls return the wrong response; no
read timeout → stalled server hangs forever; failed `connect()` leaks
the subprocess; `connect_all`/`disconnect_all` never invoked (dead
config).
*Fix:* transport lock + id match + read timeout; `try/finally close()`;
wire connect/disconnect into start/stop.

**EC-210 — `mcp_disconnect` blocks the loop** [med, weak-spot] — `fixed`
`routes/system.py:212-215` — `proc.wait(timeout=5)` inline.
*Fix:* `run_in_executor` like `mcp_connect`.

**EC-211 — JSON stores rewrite whole files per mutation** [med, weak-spot] — `fixed`
`chat_store.py`/`lorebook.py` — every mutation = full-file temp+replace
with no lock → lost updates on concurrent mutations; `/chat/send` does
N+1 rewrites per turn; `lorebook_update` lets callers overwrite
`id`/`created`/`keys`.
*Fix:* lock around mutate+save; batch saves; field whitelist.

**EC-212 — `ApprovalChannel._counter` non-atomic** [med, bug] — `fixed`
`support.py:34-62` — concurrent `+= 1` → duplicate `req_id` → second
registration overwrites the waiter → first caller blocks 300s and
denies-by-timeout even if approved.
*Fix:* `itertools.count` or lock; include kind/detail in `pending()`.

**EC-213 — `web_fetch` SSRF** [med, security] — `fixed`
`web_primitives.py:52-78` — scheme-only validation → the server fetches
`http://localhost:8765/`, `http://169.254.169.254/`, intranet URLs for
any caller (or injected prompt — EC-163).
*Fix:* resolve+reject private/loopback/link-local; host allowlist.

**EC-214 — boot test always reloads** [med, bug] — `fixed`
`boot.py:48-71` — compares resolved `info["checkpoint"]` to the raw
request path → never equal → every boot unload+loads → destroys
activation/LoRA state, stalls on `_gen_lock`.
*Fix:* compare against `_resolve_ckpt(checkpoint)`.

**EC-215 — task id collisions** [low, bug] — `fixed`
`procs.py:67-72` — counter resets per process → `task_0001` collides;
`log.txt` opened `"a"` appends into the stale run.
*Fix:* timestamp/uuid ids or clean the dir.

**EC-216 — hub cross-thread fanout** [low, bug] — `fixed`
`hub.py:36-71` — when `publish` runs with no running loop it calls
`_fanout` on the publisher's thread → `put_nowait` on asyncio.Queues
cross-thread → lost wakeups/RuntimeError; `subscribe` iterates
`_recent` lock-free.
*Fix:* guard with the same lock; skip fanout when no loop.

**EC-217 — `TaskSession.total_tokens` dead field** [low, dead-code] — `open`
`session_manager.py:158` — initialized 0, read for serialization, never
incremented → `/v1/tasks` reports 0 forever.
*Fix:* accumulate in `append_message` or drop the field.

**EC-218 — batch/tools endpoints synchronous on loop** [low, weak-spot] — `fixed`
`forge_server.py:1058-1114` — `generate_batch` + `execute_tool` inline
in async handlers; `prompts` unbounded; `total_tokens =
sum(len(r.split()))` counts WORDS as tokens.
*Fix:* `run_in_executor`; cap batch size; count real tokens.

**EC-219 — forge_server missing engine subsystems the GUI exposes** [low, missing-feature] — `open`
`merge_checkpoints`/`evolve_merge`/`benchmark`/`bottleneck`/`diagnose`/
`profile_bandwidth`/CacheBlend/`learn`/`recover`/engine `session_*` —
implemented on the engine, unreachable via the 45 OpenAI routes (GUI
exposes some). Two divergent surfaces for one engine — inverse of
EC-152.
*Fix:* `/v1/engine/{merge,diagnose,benchmark,blend,learn,session}` or
document the split.

**EC-220 — selfplay status file I/O per poll** [low, weak-spot] — `fixed`
`routes/train.py:186-195` + `events_reader.py` — `all_events()`/
`heartbeat_age()`/`heartbeat_stalled()` open+parse files on the loop
every poll; `latest_status` rglobs per call when the preferred path is
absent.
*Fix:* batch into one executor call; cache the resolved path.

Merge updates to earlier entries:
- **EC-020** — extend: `_run_single_stream` passes `stop` but also drops
  `top_k`/`repetition_penalty`/`seed`; the registry signature supports
  all fields, so sampling silently depends on batch timing.
- **EC-073** — upgrade severity: errors before `_push_stream` never send
  the sentinel → `_stream_task` blocks on `await queue.get()` with no
  timeout → **the HTTP request hangs indefinitely**; always push `None`
  in a `finally` + `asyncio.wait_for`.
- **EC-156** — extend: `model_registry.generate(seed=)`/
  `generate_stream(seed=)` call `torch.manual_seed` WITHOUT `fork_rng`
  and outside `_gen_lock` → concurrent seeds clobber each other AND leak
  global RNG state (model_registry.py:157-161,199-202).

## Cross-referenced systemic issues

1. **Over-advertised activation surface** (EC-006/013/014/016/017/021/
   046/067/068/070): ~30 `use_*` flags + 24 KV names + 13 decoding names,
   yet whole subpackages, innovation slots, scheduler features, and KV
   feature attrs are constructed-but-unconsumed — and reported as
   active. Fix direction: a `wired` bit in `info()`/`diagnose()` output;
   registry entries marked `experimental_dead`; delete-or-wire sweep.
2. **Silent degradation as house style** (EC-007/019/022/031/043/051-
   055/059): factory `.get(name, Standard)`, quant fallback chains that
   don't update `self.quantize`, ~200 broad excepts, five-layer silent
   GGUF failure stack. Fix direction: strict-name validation, effective-
   mode reporting, `strict=True` re-raise flag, load-integrity counters.
3. **Duplicate decode/sampling/session implementations** (EC-024/025/
   026/037/048/073): two decode loops + a third AirLLM loop, two session
   systems, three prefix caches, duplicated sampling helpers. Fix
   direction: pick canonical implementations, delete compat branches.
4. **research/ is the de-facto shared-utils + data root under a
   misleading name** (EC-001/078/090/091): move utils into `forge/`,
   re-export shims, or rename the directory; either way kill the
   inversion documented in AGENTS.md.
5. **Tests don't cover the seams where bugs live** (EC-076/085/087):
   the unwired/broken paths (GGUF dequant, spec-decode rollback, B==1
   param drop, session KV reuse, tool handlers) are exactly the untested
   ones. Cheapest wins: `forge_loader` dequant parity vs llama.cpp,
   `session_manager` param pass-through, `model_registry` delta
   accounting, a per-tool-handler smoke test, a 2-turn session test.
6. **Per-request state leaks into engine-global state** (EC-100/102/103/
   110/115): `_suffix_mem_for` writes `mem.gate`, session mutation runs
   outside `_gen_lock`, hotswap `_pending` persists through failures,
   the streaming RLock can strand on cross-thread GC — correctness
   depends entirely on lock discipline that isn't enforced at the
   session boundary.
7. **Public-surface contract drift** (EC-096/116/127/122): `embed()`
   assumes a 4-tuple the model never returns, GUI routes use kwarg names
   the mixins don't accept, `begin_session` returns a dataclass an HTTP
   route can't serialize, `session_cache` claims a radix tree that
   doesn't exist — the engine's public surface needs an API conformance
   test.
8. **Every remote surface is a trusted-local primitive over HTTP**
   (EC-184/185/186/187/188/206/213): raw argv exec, caller-chosen
   workspace + `approval_mode="none"`, arbitrary read/write paths in
   merge/LoRA/delete, zip-slip restore, SSRF — all unauthenticated. The
   fix isn't just "add auth" — the endpoints were designed as local
   primitives and expose raw power verbatim; each needs confinement
   (resolve+is_relative_to, member validation, allowlists) independent
   of the auth fix.
9. **Route↔implementation drift kills whole user flows behind
   `{"ok":True}`** (EC-188/191/192/193/194/116): wrong script paths,
   wrong CLI flags, nonexistent methods/kwargs, `**kwargs`-swallowed
   renames — finetune, LoRA training, evolve-merge, lorebook, memory
   injection all dead or no-op. A single per-route smoke test would
   have caught all of them.
10. **The event loop is used as a blocking worker** (EC-057/189/203/204/
    207/208/210/218/220): `sleep(2)`/`empty_cache` in `load()`, whole-
    project zips, rglob+whole-file reads, `proc.wait`, 5000-line
    `asdict` deep-copies per poll — each freezes every endpoint plus the
    `/ws` hub; `run_in_executor` usage is inconsistent, hiding stall
    sites next to correctly-offloaded neighbors.
11. **Work outlives its request on both servers** (EC-190/197/198/156):
    futures discarded, `StreamCancel`/`ChatCancel` defined but never
    wired, `BatchQueue.stop` abandons pending futures, dispatch errors
    strand stream consumers forever, history mutated after HTTP
    timeout, no per-task serialization, seeds clobber global RNG —
    "cancel" is a dead flag everywhere.
12. **Config is rebuilt-from-partial in four places** (EC-062/195/196):
    `ActivationConfig.from_kwargs`/`TaskBootConfig.from_dict` drop
    unknown keys silently, `activate()`/`apply_to()`/`_reactivate_
    blocking` rebuild ~50 flags from partial dicts, `differs_from`
    watches 4 fields and misses revert-to-default, `active_config`
    reports requested not applied — activation state resets, typos
    vanish, `/activation/diff` lies.

## Recommended fix order

Highest value-per-effort, based on confirmed bugs vs. effort:

1. **EC-050** — revoke the leaked HF token (minutes, security).
2. **EC-184/185** — kill/gate `/tasks/launch` + the `/agent/runs`
   workspace/approval_mode override (small diffs; closes two
   unauthenticated RCE vectors).
3. **EC-186/187/188** — confine backup restore, `DELETE /models`, and
   merge endpoints (resolve+is_relative_to + member validation).
4. **EC-097/027/206/213** — route `file_read`/`file_list` through
   `check_file_read`; confine LoRA paths; SSRF host filter.
5. **EC-094** — append generated ids to `session.token_ids` (one-line
   class of fix; unbreaks session KV reuse entirely).
6. **EC-096** — `embed()` tuple-length fix (one line; unbreaks rerank).
7. **EC-019** — strict strategy-name validation (small diff; kills a
   whole class of silent regressions).
8. **EC-020** — pass `top_k`/`rep_penalty`/`stop`/`seed` through the
   B==1 and stream paths.
9. **EC-191/192/193/188** — route↔impl fixes (one-line each): correct
   sft_train path/flags, lorebook kwargs, `hot_swap`→`load_result`,
   `evolve_merge` signature; add one smoke test per route to keep them
   dead-proof.
10. **EC-190/073/197** — wire cancel tokens + stream sentinels +
    queue drain (unbreaks disconnect/cancel semantics on both servers).
11. **EC-008/009/010/165/166/167** — fix or delete the dead spec-decode
    strategies (mtp_selfspec, eagle3, medusa, dspark are DOA; ngram/
    external draft corrupt KV).
12. **EC-006/007** — the big one: decide the canonical KV contract
    (bridge-hook vs full interface), then delete-or-wire the 21 inert
    strategies — verify `compact_past_kv`/`PreAllocatedKVCache`
    interaction on a CUDA-graph run first (EC-022 hedge).

## Status legend

- `open` — verified, not yet addressed
- `fixed` — landed; note commit
- `wontfix` — reviewed, intentionally kept (note why)
- `disputed` — evidence contested; needs re-verification
