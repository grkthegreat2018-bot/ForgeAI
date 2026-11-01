# ForgeAI

Local-first LLM research stack for an RTX 5070 / 12GB box: ForgeEngine
inference (KV-cache strategies, quantization, graph decode, continuous
batching), ForgeLM model presets (V2 Jamba base → V14-RSI), training
runners (SFT/DPO/GRPO, LiveT online learning), and the ForgeEvolve /
CodeEvolve evolutionary optimization + RSI self-improve loop.

## Layout

- `forge/` — production code (engine, model, training, evolution, quant)
- `scripts/` — standalone drivers (evolve campaigns, benches, converters)
- `tests/unit/` — CPU-runnable unit tests (`tests/integration/` needs GPU)
- `docs/` — documentation + R&D round notes (`docs/INDEX.md` is the index)
- `research/` — legacy root still holding live modules (paths, tokenizer
  cache, port scripts) and checkpoints — new code goes in `forge/`
- `data/`, `checkpoints/`, `results/`, `logs/` — local artifacts (ignored)
- `.devin/scratchpad.md` — running R&D working notes

## Docs

Start with **AGENTS.md** (canonical layout, operating directives,
per-subsystem notes, build/test commands) and **docs/INDEX.md**.
High-level tour: `docs/project_overview.md`. Engine guide:
`docs/forge_engine_guide.md`. Evolve roadmap: `docs/EVOLVE_ROADMAP.md`.

## Test

```powershell
$env:PYTHONPATH="D:\windsurf\ForgeAI"
.\venv\Scripts\python.exe -m pytest tests/ --tb=short -q
```

## Quickstart

```powershell
# pre-flight: verify local checkpoints/tokenizer exist before any run
.\venv\Scripts\python.exe scripts\check_assets.py   # add --all for optional assets

# evolve a single engine hot-path with the LFM3 proposer (default model)
.\venv\Scripts\python.exe scripts\fn_evolve.py --target prefix_hash `
    --evals 60 --status-file .devin\run.json --eval-log --log-gens

# watch it live
.\venv\Scripts\python.exe scripts\evolve_watch.py --status .devin\run.json

# system-level meta-target (e2e decode-step us/token)
.\venv\Scripts\python.exe scripts\fn_evolve.py --target decode_step --evals 100

# RSI campaign: all targets -> SFT export -> LoRA hone -> next round
.\venv\Scripts\python.exe scripts\rsi_evolve.py --rounds 1 --minutes 30

# promotion-ledger audit (what % of discoveries ever got wired in)
.\venv\Scripts\python.exe scripts\ledger_audit.py forge_evolve.db
```

Useful proposer flags: `--proposer mut|mix|llm|basin|none`
(mut = free AST mutations, no model), `--proposer-think` (LFM3
reasoning prefill), `--proposer-compile` (torch.compile; decode already
replays CUDA graphs, so this only speeds prefill), `--lora-root` /
`--lora-tag` (tagged adapters; defaults to `forge/evolution/loras_lfm3`),
`--no-eval-pool --workers 1` (serial subprocess-per-eval debug mode).

### Server env vars

- `FORGE_HOST` / `FORGE_PORT` — bind address/port (`forge_server.py`)
- `FORGE_API_KEY` — bearer key gating mutating `/v1` methods
- `FORGE_MODEL_ALIASES` — `fast=forgelm-v2-jamba,code=lfm3` style aliases
- `FORGE_LOG_REQUESTS=1` — one JSON line per request to stderr
- `FORGE_NO_CACHE=1` — disable the exact-match response cache
- `FORGE_LORA_ROOTS` — `os.pathsep`-separated allowed LoRA directories
- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` — refuse HF downloads;
  cached artifacts only (load fails loudly on a cache miss)
- `FORGE_HF_REVISION` — default HF revision for hub-resolved models

### Secondary environments

- `requirements_ov.txt` — OpenVINO GenAI venv (`venv_ov\Scripts\python.exe`) for NPU/CPU proposer offload.
- `requirements_train.txt` — training venv (`venv_train\Scripts\python.exe`) for SFT/LoRA hone rounds.
