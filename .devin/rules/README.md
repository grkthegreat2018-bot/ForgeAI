# Scoped Rules

Check the rule file matching your work area **before editing code**.
Each file is deliberately short (~30 lines): hard rules + entry points +
the right test command. Deep context stays in `AGENTS.md` — these files
point into it instead of duplicating it.

| File | Covers |
|---|---|
| `workspace.md` | disks, `.devin/` conventions, `research/` data roots, caches |
| `engine.md` | `forge/engine/` — ForgeEngine invariants, KV/decoding, env vars |
| `model_keys.md` | `forge/model/`, `forge/keys/`, `forge/config.py` — presets, ports |
| `quant_decoding.md` | `forge/quant/`, `forge/engine/quant/`, `forge/decoding/` |
| `training_selfplay.md` | `forge/training/`, `forge/self_play/`, sft/grpo/rsi loops |
| `flux_exp.md` | `forge/model/flux/`, `forge/model/strata/`, `forge/exp/`, EXP scripts |
| `gui_server.md` | `forge_gui*/`, `forge/engine/forge_server.py` — trust boundaries |

Rule for rule files: keep under ~50 lines; a rule that needs a paragraph
belongs in `AGENTS.md` — link the section instead.
