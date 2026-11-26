# Workspace Rules — disks, scratch, data roots

## Hard rules
- **D: is SMR over USB-BOT** — sequential ~230MB/s, random writes <10MB/s
  once the CMR cache fills. Checkpoints/datasets/packed caches on D:;
  many-small-file IO belongs on C:. See `docs/FAILURE_CATALOG.md`.
- **`research/` is LIVE, not legacy** — 19 tracked modules
  (`persistent_executor` sandbox, `tokenizer_cache`, `paths`,
  `merge_models`, `training_free/`) + ~500GB data/checkpoint roots,
  46 import sites. Never `rm` or "clean" it; new code goes in `forge/`.
- **`.devin/` is scratch** — everything gitignored except
  `scratchpad.md`, `rules/`, `config.json`. Anything worth keeping must
  be promoted out (tracked file, docs note, or scratchpad entry).
- **`scratch/` is untracked temp** — `wiki_repo/` (the live wiki clone)
  and reference clones only. Treat everything else there as deletable.
- Caches (`torch_cache`, `hf_cache`, `__pycache__`, triton dirs) are
  regenerable — deleting them is safe, just costs recompile/redownload.

## Conventions
- Working notes, shelved approaches, dead ends → `.devin/scratchpad.md`
  (append; it's the only tracked `.devin` file).
- Experiment results → `.devin/exp_results.jsonl` via `log_result`
  (`forge/exp/harness.py`), not ad-hoc prints.
- One-off probes/debug scripts → `.devin/` — never `scripts/` or repo root.
- Env-var caches are set to `.devin/torch_cache` etc. to dodge the
  Windows MAX_PATH bug in triton/inductor (see AGENTS.md ~line 573).
