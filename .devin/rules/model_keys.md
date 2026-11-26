# Model/Keys Rules — forge/model/, forge/keys/, forge/config.py

## Hard rules (directive A — highest priority in this area)
- **Version chain is linear:** `forgelm_v2` → `v12_jamba` → `v13_gdn2` →
  `v14_rsi`. New presets derive from the immediately previous one —
  carry all keys forward, add/replace only what's new.
- **Port-first, train-second:** every new arch key/attention variant
  ships its lossless checkpoint-conversion path (`XxxKey` class,
  identity/zero-init warm start, bit-exact load test) BEFORE any
  training experiment.
- **No silent key drops** — a preset dropping a prior key documents WHY
  in the preset line + round notes. Silent drops are bugs.
- **Preset lineage check** — bit-exact forward-pass vs the prior preset's
  checkpoint on the BSP base; max logit diff 0.0 unless intentionally
  non-lossless (document the delta in `dropped_keys`).

## Entry points
- `forge/config.py` — ModelConfig dataclass + all presets.
- `forge/model_loader.py` — facade over `forge/model/` (kv_cache,
  attention_ops, layers, builders, llm, loader/ModelLoader).
- `forge/keys/` — 45 files: per-arch key classes (attention/, etc.).
- Adding an HF model: follow `docs/ADAPTER_AUTHORING.md` — preset +
  key map + `forge_convert.json` + bit-exact port test (see
  `scripts/test_parable_port.py` for the reference parity harness).
