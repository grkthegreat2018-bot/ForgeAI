"""Meta-evolution domain (R65): evolve ForgeEvolve's own search knobs.

Candidates are ForgeEvolveConfig settings — generator population shape,
mutation rate, surrogate filter ratio, novelty pulse schedule, elite
mutants, learning rates. evaluate() runs REAL short inner ForgeEvolve
runs on SyntheticDomain (known optimum, microsecond evals) and scores:

  progress   = best_score / optimal_score      (proposal quality)
  efficiency = expected_evals / actual_evals    (proposal speed)

Inner runs are isolated: fixate=True (no cross-domain seeding, no
refinement jumps), warm_start=False (cold search — measures the
proposal machinery itself, not priors), scratch DB, verbose off.

This is literally evolution optimizing evolution: the outer archive
converges on the engine configuration that finds better proposals
quicker. The winning config is then applied to production runs.
"""
from __future__ import annotations

import contextlib
import io
import math
import time

import numpy as np
import torch

from . import BaseDomain

_N_GEN_OPTS = (50, 100, 200, 300, 500)
_HIDDEN_OPTS = (32, 64, 128, 256)
_NOISE_OPTS = (8, 16, 32, 64)
_CTX_OPTS = (16, 32, 64)
_FILT_OPTS = (4, 8, 12, 20, 30)
_EVAL_OPTS = (15, 30, 60, 100)
_PULSE_OPTS = (1, 2, 3, 5)
_MUT_OPTS = (0, 4, 8, 16)
_SURR_OPTS = ("mlp", "gp")
_ISL_OPTS = (1, 2, 3, 4)
_BW_OPTS = (0, 1, 2, 3, 5)
_BK_OPTS = (0, 2, 3, 4, 6)

_INNER_GENS = 10
_INNER_REPS = 2
_INNER_DIM = 8
# Inner runs need zero persistence — a private in-memory DB per run
# kills the WAL/connect/schema-migration disk traffic (was the inner
# loop's dominant cost at ~24 inner runs per outer gen).
_SCRATCH_DB = ":memory:"


class EvolverTune(BaseDomain):
    """Evolve the evolver: ForgeEvolveConfig search on inner runs."""

    # evaluate() builds a nested ForgeEvolve — inner engines must not
    # share side-stream CUDA state with sibling evals. Serial only.
    eval_thread_safe = False
    # Inner runs are Python/CPU-bound only when the generator nets are
    # small — heavy candidates (500 gens × hidden 256 ≈ 30M params)
    # want CUDA for forward+backward. GPU-pool workers each own a CUDA
    # context so nested engines run at full speed in parallel; CPU-pool
    # remains as the process-safe fallback.
    # Measured on this box: inner engines are small-op-bound — CPU
    # beats contended CUDA (8.3s vs 11.9s full solo, and no kernel-
    # launch serialization under parallel workers). CPU pool wins.
    cpu_eval_ok = True
    gpu_pool_ok = False

    def gpu_worker_kwargs(self) -> dict:
        return {"seed": 43, "device": "cuda"}

    def __init__(self, seed: int = 42, device=None, inner_gens=_INNER_GENS,
                 reps=_INNER_REPS, scratch_db=_SCRATCH_DB):
        super().__init__()
        if device is not None:
            self._device = torch.device(device)
        self._seed = seed
        self._gens = inner_gens
        self._reps = reps
        self._db = scratch_db
        self._eval_no = 0

    def name(self) -> str:
        return "evolver_tune"

    def output_dim(self) -> int:
        return 24

    def discrete_choices(self) -> dict[str, list]:
        # 5*5*2*4 = 200 cells: pop × filter × novelty × islands
        return {
            "n_generators": list(_N_GEN_OPTS),
            "filter_ratio": list(_FILT_OPTS),
            "enable_novelty": [False, True],
            "n_islands": list(_ISL_OPTS),
        }

    def decode(self, params: torch.Tensor) -> dict:
        p = params.detach().cpu().numpy()

        def pick(i, seq):
            return seq[min(int(p[i] * (len(seq) - 1e-9)), len(seq) - 1)]

        def interp(i, lo, hi):
            return float(np.interp(np.clip(p[i], 0, 1), [0, 1], [lo, hi]))

        def log_interp(i, lo, hi):
            return float(10 ** np.interp(np.clip(p[i], 0, 1), [0, 1],
                                         [math.log10(lo),
                                          math.log10(hi)]))

        return {
            "n_generators": pick(0, _N_GEN_OPTS),
            "hidden_dim": pick(1, _HIDDEN_OPTS),
            "noise_dim": pick(2, _NOISE_OPTS),
            "context_dim": pick(3, _CTX_OPTS),
            "mutation_rate": interp(4, 0.02, 0.40),
            "filter_ratio": pick(5, _FILT_OPTS),
            "max_evaluate": pick(6, _EVAL_OPTS),
            "generator_lr": log_interp(7, 3e-4, 6e-3),
            "surrogate_lr": log_interp(8, 3e-4, 6e-3),
            "exploration": interp(9, 0.05, 0.60),
            "enable_novelty": bool(p[10] > 0.5),
            "novelty_pulse_len": pick(11, _PULSE_OPTS),
            "novelty_bonus": interp(12, 0.5, 3.0),
            "elite_mutants": pick(13, _MUT_OPTS),
            "adaptive_population": bool(p[14] > 0.5),
            "surrogate_mode": pick(15, _SURR_OPTS),
            # ── emitter machinery (the new stack — wilds, gates,
            # bursts, demes, crossover, explore fraction) ──
            "wild_gen_frac": interp(16, 0.0, 0.30),
            "wild_frac": interp(17, 0.0, 0.15),
            "mc_floor_q": interp(18, 0.0, 0.40),
            "crossover_frac": interp(19, 0.0, 0.40),
            "n_islands": pick(20, _ISL_OPTS),
            "burst_winners": pick(21, _BW_OPTS),
            "burst_k": pick(22, _BK_OPTS),
            "explore_frac": interp(23, 0.05, 0.40),
        }

    def encode(self, config: dict) -> torch.Tensor:
        def binof(val, seq):
            try:
                return seq.index(val) / max(len(seq) - 1, 1)
            except (ValueError, TypeError):
                return 0.5

        def un(val, lo, hi):
            return float(np.clip((float(val) - lo) / (hi - lo), 0, 1))

        def unlog(val, lo, hi):
            v = max(float(val), 1e-12)
            return float(np.clip(
                (math.log10(v) - math.log10(lo))
                / (math.log10(hi) - math.log10(lo)), 0, 1))

        c = config
        return torch.tensor([
            binof(c.get("n_generators", 300), _N_GEN_OPTS),
            binof(c.get("hidden_dim", 64), _HIDDEN_OPTS),
            binof(c.get("noise_dim", 16), _NOISE_OPTS),
            binof(c.get("context_dim", 32), _CTX_OPTS),
            un(c.get("mutation_rate", 0.1), 0.02, 0.40),
            binof(c.get("filter_ratio", 20), _FILT_OPTS),
            binof(c.get("max_evaluate", 60), _EVAL_OPTS),
            unlog(c.get("generator_lr", 1e-3), 3e-4, 6e-3),
            unlog(c.get("surrogate_lr", 1e-3), 3e-4, 6e-3),
            un(c.get("exploration", 0.3), 0.05, 0.60),
            1.0 if c.get("enable_novelty", True) else 0.0,
            binof(c.get("novelty_pulse_len", 3), _PULSE_OPTS),
            un(c.get("novelty_bonus", 1.5), 0.5, 3.0),
            binof(c.get("elite_mutants", 0), _MUT_OPTS),
            1.0 if c.get("adaptive_population", True) else 0.0,
            binof(c.get("surrogate_mode", "mlp"), _SURR_OPTS),
            un(c.get("wild_gen_frac", 0.10), 0.0, 0.30),
            un(c.get("wild_frac", 0.04), 0.0, 0.15),
            un(c.get("mc_floor_q", 0.15), 0.0, 0.40),
            un(c.get("crossover_frac", 0.15), 0.0, 0.40),
            binof(c.get("n_islands", 1), _ISL_OPTS),
            binof(c.get("burst_winners", 2), _BW_OPTS),
            binof(c.get("burst_k", 3), _BK_OPTS),
            un(c.get("explore_frac", 0.20), 0.05, 0.40),
        ], dtype=torch.float32)

    def behavioral_dims(self):
        return [("progress", 10, 0.0, 1.0),
                ("log_evals", 10, 3.0, 6.0)]

    def seed_configs(self) -> list[dict]:
        prod = self.decode(self._mid())
        cheap = dict(prod, n_generators=100, filter_ratio=8,
                     max_evaluate=30)
        heavy = dict(prod, n_generators=500, filter_ratio=30,
                     max_evaluate=100)
        nonov = dict(prod, enable_novelty=False)
        # Pass-2 winner family (score 80.9): big pop, tight filter,
        # mlp surrogate — the progress-max corner.
        winner = dict(prod, n_generators=500, filter_ratio=8,
                      max_evaluate=60)
        # Hybrid: winner recipe + elevated demes/burst — the corner
        # the search drifted toward but under-sampled.
        hybrid = dict(winner, n_islands=4, burst_winners=5,
                      burst_k=6, crossover_frac=0.25)
        return [prod, cheap, heavy, nonov, winner, hybrid]

    def _mid(self) -> torch.Tensor:
        v = np.array([
            (3 + .5) / 5, (1 + .5) / 4, (1 + .5) / 4, (1 + .5) / 3,
            (0.10 - .02) / .38, (3 + .5) / 5, (2 + .5) / 4,
            0.38, 0.38, (0.30 - .05) / .55,
            0.9, (2 + .5) / 4, (1.5 - .5) / 2.5,
            0.05, 0.9, 0.1,
            # new emitter dims at their proven defaults
            0.10 / 0.30, 0.04 / 0.15, 0.15 / 0.40, 0.15 / 0.40,
            0.0, (2 + .5) / 5, (2 + .5) / 5, (0.20 - 0.05) / 0.35,
        ])
        return torch.tensor(np.clip(v, 0, 1), dtype=torch.float32)

    # ── inner run ───────────────────────────────────────────────────────

    def _inner_run(self, knobs: dict, rep: int, gens: int = None) -> dict:
        from forge.evolution import ForgeEvolve, ForgeEvolveConfig
        from forge.evolution.domains.synthetic import SyntheticDomain

        # Vary the landscape per rep (8, 9, ... dims) — a config that
        # only wins on one landscape is an overfit, not a recipe.
        domain = SyntheticDomain(dim=_INNER_DIM + rep,
                                 seed=self._seed + rep * 7919)
        # Random-guess floor: score scale is offset (+50) so most of the
        # landscape is negative — progress must be measured from where
        # blind guessing lands, not from 0.
        rng = np.random.RandomState(self._seed + rep * 104729)
        floor = float(np.median([
            domain.evaluate({"x": rng.rand(domain.output_dim())})["score"]
            for _ in range(64)]))
        self._eval_no += 1
        cfg = ForgeEvolveConfig(
            domain=domain,
            n_generators=int(knobs["n_generators"]),
            hidden_dim=int(knobs["hidden_dim"]),
            noise_dim=int(knobs["noise_dim"]),
            context_dim=int(knobs["context_dim"]),
            mutation_rate=float(knobs["mutation_rate"]),
            filter_ratio=int(knobs["filter_ratio"]),
            max_evaluate=int(knobs["max_evaluate"]),
            generator_lr=float(knobs["generator_lr"]),
            surrogate_lr=float(knobs["surrogate_lr"]),
            surrogate_mode=str(knobs["surrogate_mode"]),
            exploration=float(knobs["exploration"]),
            enable_novelty=bool(knobs["enable_novelty"]),
            novelty_pulse_len=int(knobs["novelty_pulse_len"]),
            novelty_bonus=float(knobs["novelty_bonus"]),
            elite_mutants=int(knobs["elite_mutants"]),
            adaptive_population=bool(knobs["adaptive_population"]),
            wild_gen_frac=float(knobs["wild_gen_frac"]),
            wild_frac=float(knobs["wild_frac"]),
            mc_floor_q=float(knobs["mc_floor_q"]),
            crossover_frac=float(knobs["crossover_frac"]),
            n_islands=int(knobs["n_islands"]),
            burst_winners=int(knobs["burst_winners"]),
            burst_k=int(knobs["burst_k"]),
            explore_frac=float(knobs["explore_frac"]),
            generations=gens or self._gens,
            enable_refinement=False,
            fixate=True,
            warm_start=False,
            db_path=self._db,
            run_id=f"meta_inner_{self._eval_no}_{rep}",
            device=self._device.type,
            verbose=False,
            log_every=10_000,
        )
        t0 = time.time()
        # Silence inner engine output — worker inner "[Generators]"
        # prints were flooding the outer log (~1K lines/gen).
        with contextlib.redirect_stdout(io.StringIO()):
            res = ForgeEvolve(cfg).run()
        wall = time.time() - t0
        return {"best": float(res["best_score"]),
                "optimal": float(domain.optimal_score),
                "floor": floor,
                "evals": int(res["total_evaluations"]),
                "discoveries": int(res["discoveries"]),
                "wall_s": wall}

    def quick_eval(self, config: dict) -> dict:
        """Successive-halving screen (Hyperband): 1 rep × 5 inner gens
        ranks candidates at ~30% of full-eval cost. Promotion to the
        full tier (2 reps × 10 gens) is decided by the engine."""
        return self._evaluate(config, reps=1, gens=5)

    def evaluate(self, config: dict) -> dict:
        return self._evaluate(config, reps=self._reps,
                              gens=self._gens)

    def _evaluate(self, config: dict, reps: int, gens: int) -> dict:
        try:
            runs = [self._inner_run(config, r, gens)
                    for r in range(reps)]
        except Exception as e:
            return {"score": -1e9, "behavioral": (0, 0),
                    "metadata": {"fail": f"inner: {e}"}}

        expected_evals = (6 + gens * min(
            int(config["max_evaluate"]),
            max(10, int(config["n_generators"])
                // max(int(config["filter_ratio"]), 1))
            + int(config["elite_mutants"])))

        prog, eff, evals, wall, disc = [], [], 0, 0.0, 0
        for r in runs:
            span = max(r["optimal"] - r["floor"], 1e-9)
            p = float(np.clip((r["best"] - r["floor"]) / span, 0.0, 1.0))
            prog.append(p)
            eff.append(expected_evals / max(r["evals"], 1))
            evals += r["evals"]
            wall += r["wall_s"]
            disc += r["discoveries"]
        prog_m = float(np.mean(prog))
        eff_m = float(np.clip(np.mean(eff), 0.0, 4.0))

        score = 100.0 * prog_m + 4.0 * math.log2(1.0 + eff_m)
        return {"score": float(score),
                "behavioral": (prog_m,
                               float(np.clip(math.log10(max(evals, 1)),
                                             3, 6))),
                "metadata": {
                    "progress": prog_m, "efficiency": eff_m,
                    "evals": evals, "wall_s": wall,
                    "discoveries": disc,
                    "inner_bests": [r["best"] for r in runs],
                    "inner_optimal": runs[0]["optimal"]}}
