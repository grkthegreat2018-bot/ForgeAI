"""ForgeEvolve: evolutionary candidate discovery via tiny generators + surrogate filtering.

Architecture:
  Phase 1: GENERATE  — N tiny MLP generators produce candidate configs (CPU, ~1s)
  Phase 2: FILTER    — trained surrogate predicts scores, only top-K evaluated (CPU, ~1s)
  Phase 3: SCORE     — real evaluation of K candidates (GPU/domain-specific, ~minutes)
  Phase 4: TRAIN     — update generators (REINFORCE) + surrogate (online) + archive (CPU, ~10s)
  Phase 5: REPEAT

Key insight: 1000 candidates generated, only 50 evaluated → 20:1 compression.
Surrogate learns which configs tend to work, so its top-50 predictions improve over time.

Lazy exports (PEP 562): the names below resolve on first attribute access.
Eager importing used to pull torch + engine + gen-model chains into EVERY
spawn-context eval worker (eval_pool._worker_main is resolved by importing
this package) — ~335MB RSS and multi-second boot per worker for code the
worker never touches. Submodules (eval_pool, code_evolve, mutator, ...) are
still importable directly; ``forge.evolution.<submodule>`` attribute access
also lazily resolves.
"""
from __future__ import annotations

import importlib

# Public name -> owning submodule (resolved on first access).
_EXPORTS = {
    "MapElitesArchive": "archive",
    "HeuristicChecker": "checker_model",
    "SharedCheckerModel": "checker_model",
    "get_checker": "checker_model",
    "reset_checker": "checker_model",
    "CurriculumFineTuner": "curriculum_finetuner",
    "FindingsDB": "database",
    "ForgeEvolve": "engine",
    "ForgeEvolveConfig": "engine",
    "GenModelManager": "gen_model_manager",
    "BatchedGenerator": "generators",
    "GeneratorPopulation": "generators",
    "TemplateGenerator": "generators",
    "LLMGenModel": "llm_gen_model",
    "SurrogateModel": "surrogate",
    "GeneratorTrainer": "trainer",
}

__all__ = [
    "BatchedGenerator",
    "CurriculumFineTuner",
    "FindingsDB",
    "ForgeEvolve",
    "ForgeEvolveConfig",
    "GenModelManager",
    "GeneratorPopulation",
    "GeneratorTrainer",
    "HeuristicChecker",
    "LLMGenModel",
    "MapElitesArchive",
    "SharedCheckerModel",
    "SurrogateModel",
    "TemplateGenerator",
    "get_checker",
    "reset_checker",
]


def __getattr__(name: str):
    sub = _EXPORTS.get(name)
    if sub is not None:
        mod = importlib.import_module(f".{sub}", __name__)
        val = getattr(mod, name)
        globals()[name] = val          # one lazy hit, then direct attribute
        return val
    try:
        return importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError as e:
        if e.name != f"{__name__}.{name}":
            raise   # a real nested-import failure — surface it
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return sorted(set(globals()) | set(__all__))
