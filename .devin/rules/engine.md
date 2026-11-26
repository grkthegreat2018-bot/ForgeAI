# Engine Rules — forge/engine/

## Entry points
- `forge_engine.py` — ForgeEngine: core init + 8 mixins
  (`engine_checkpoints/activation/generation/diagnostics/merging/lora/
  lifecycle/sessions`; shared helpers in `engine_common.py`).
- `forge_server.py` — OpenAI-compatible server (127.0.0.1, FORGE_API_KEY).
- Subdirs: `kv/` (32 cache strategies), `quant/` (engine quant),
  `scheduler/` (batching), `attention/`, `graphs/`, `decoding helpers`.

## Invariants (full list: AGENTS.md "Engine Invariants")
- **Report effective state, not requested** — after any fallback,
  `self.quantize`/`kv_bits`/diagnostics show what actually applied.
- **No asyncio primitives across loops** — per-task locking is
  `threading.Lock` + `_task_lock_pool` in forge_server.py.
- **Unknown strategy names fail loudly** — `build_kv_cache` /
  `build_decoding` raise; never silently substitute `standard`.
- **Spec KV rollback = snapshot + repair** — reuse `_spec_snapshot` /
  `_spec_truncate` / `_spec_repair` in `decoding.py`; don't roll your own.
- **No half-dead flags** — a `use_*` flag must wire to a real path or
  refuse to activate (this is how ENGINE_CRITIQUE_DB hit 220 entries).

## Gotchas
- `.item()` host syncs per layer/step are the classic decode-speed kill —
  pipeline flag copies via cuda.Event (see `_drain_eos` pattern).
- Batched decode: position_ids from attention_mask.cumsum(-1)-1; KV is
  copied verbatim on refill — never RoPE-rebase.
- Many features gate on `FORGE_*` env vars — check `feature_registry.py`
  before assuming a path is dead.
