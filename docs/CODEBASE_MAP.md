# Codebase Map — find things fast

Routing layer over the 1300-file tree. Pair with `docs/INDEX.md` (docs),
`scripts/README.md` (script index), `.devin/rules/<area>.md` (scoped
edit rules), `.devin/config.json` (machine-readable env/test facts).

## "I want to…" → go to

| Task | Start at | Verify with |
|---|---|---|
| Change inference behavior | `forge/engine/forge_engine.py` (+ its `engine_*` mixin) | `tests/unit/` + `scripts/bench_load_time.py` |
| Add/change a KV-cache strategy | `forge/engine/kv/` (32 impls) — reuse `build_kv_cache` registry | unknown names must raise (EC-019) |
| Add a decoding strategy | `forge/decoding/` + `build_decoding` registry | spec rollback via `_spec_snapshot/_repair` |
| Add a quant method | `forge/engine/quant/` (pattern: `littlebit2.py`) + `feature_registry.py` | `quantize="<name>"` config key |
| Add a model preset | `forge/config.py` — derive from the previous chain version | bit-exact lineage check (directive A) |
| Port a new HF model | `docs/ADAPTER_AUTHORING.md` recipe | `scripts/test_parable_port.py` (reference harness) |
| Add an arch key (attention/SSM variant) | `forge/keys/` (45 impls) — lossless port class FIRST | key load test before training |
| Touch SFT/GRPO training | `forge/training/runners/` — sft_train.py is canonical | `.devin/rules/training_selfplay.md` |
| Touch the self-play/RSI loop | `forge/self_play/infinite_{loop,curriculum}.py` | GRPO old_log_probs rule (rules file) |
| Add an evolve target | `forge/evolution/` (204 files) + `scripts/fn_evolve.py` | `forge_evolve.db` ledger |
| Sandbox model-generated code | `research/persistent_executor.py` — the ONLY jail | `tests/unit/test_sandbox_escapes.py` |
| Work on FluxLM/STRATA | `forge/model/flux/` · `forge/model/strata/` | `.devin/rules/flux_exp.md` |
| GUI/server work | `forge_gui*/`, `forge_server.py` | `.devin/rules/gui_server.md` trust rules |
| Check a known bug/failure | `docs/BUG_LOG.md` · `docs/FAILURE_CATALOG.md` | `log-bug` skill to add entries |
| Find prior art on a technique | `docs/research/` atlas + `.devin/scratchpad.md` | wiki `tried/` pages (R32–70) |

## Package map (`forge/`, ~545 py)

| Dir | Files | Role |
|---|---|---|
| `engine/` | 134 | ForgeEngine + mixins; subdirs: `kv/` 32, `quant/` 15, `scheduler/` 13, `attention/` 10, `graphs/` 5, `safety/` 5, `reasoning/` 4 |
| `evolution/` | 204 | ForgeEvolve/CodeEvolve: domains, eval pool, proposers, ledger |
| `training/` | 47 | runners (sft/grpo/rlvr/dpo/cpt), optim (muon_sf…), bitnet_lora |
| `keys/` | 45 | arch-key classes — port-first lossless conversions |
| `model/` | 35 | ConfigurableResearchLLM, layers, kv_cache, loader; + `flux/`, `strata/` |
| `self_play/` | 28 | infinite loop/curriculum, GRPO trainer, discovery, sandbox |
| `decoding/` | 16 | decoding strategies incl. V14 CopySpan edit-mode |
| `quant/` | 11 | standalone quant (ECQ codec, …) |
| `exp/` | 11 | EXP experiment harnesses (log → `.devin/exp_results.jsonl`) |
| `runtime/` `evaluation/` `moe/` `distillation/` | 20 | support packages |

Sibling roots: `scripts/` (150, indexed in `scripts/README.md`) ·
`tests/unit/` (191 CPU tests) + `tests/{evolution,fixtures,integration}` ·
`research/` (live legacy package + ~500GB data roots — see
`.devin/rules/workspace.md`) · `forge_gui/` `forge_gui_server/`
`forge_ui/` · `tools/llama/` (ggml DLLs).

## Data & artifact roots

| Path | Contents | Status |
|---|---|---|
| `research/data/` | corpora, tok_cache, self-play task stores (604k files) | live |
| `research/checkpoints/` | ported model dirs (granite/hermes/lfm/flux…) | live |
| `checkpoints/`, `data/`, `logs/` | run artifacts | ignored |
| `results/` | RSI run summaries + external evals | tracked |
| `forge_evolve*.db` | per-domain evolve ledgers | tracked (root) |
| `.devin/` | scratch: scripts, notes, `exp_results.jsonl`, `qwen25-05b/` | ignored except `scratchpad.md`, `rules/`, `config.json` |
| `scratch/` | `wiki_repo/` (live wiki clone), reference clones | ignored |

## Freshness rules

- New script → run `scripts/build_script_index.py` (regenerates
  `scripts/README.md`).
- New file/feature/R&D round → update `AGENTS.md` + relevant wiki page
  in the SAME session (directive G).
- New doc → add a line to `docs/INDEX.md` and to this map's tables if
  it changes the routing answer.
- Shelved idea → `.devin/scratchpad.md`, not a stale file on disk.
