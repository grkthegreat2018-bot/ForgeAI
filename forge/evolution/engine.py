"""ForgeEvolve engine: the phase loop.

  Phase 1: GENERATE  — N generators produce candidates (batched GPU forward)
  Phase 2: FILTER    — surrogate predicts scores, top-K selected (GPU)
  Phase 3: SCORE     — real evaluation of K candidates (domain-specific)
  Phase 4: TRAIN     — update generators + surrogate + archive (GPU)
  Phase 5: REPEAT

CUDA: generators + surrogate run on GPU when available. Domain evaluation
may use GPU independently (e.g., quant domain does FP4 ops on GPU).

Usage:
  from forge.evolution import ForgeEvolve, ForgeEvolveConfig
  from forge.evolution.domains.synthetic import SyntheticDomain

  cfg = ForgeEvolveConfig(
      domain=SyntheticDomain(...),
      n_generators=500,
      filter_ratio=20,
      generations=50,
  )
  engine = ForgeEvolve(cfg)
  results = engine.run()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

from .archive import MapElitesArchive
from .database import FindingsDB
from .domains import BaseDomain
from .generators import GeneratorConfig, GeneratorPopulation, TemplateGenerator
from .surrogate import SurrogateModel
from .trainer import GeneratorTrainer

# ── CPU worker globals (set by _cpu_worker_init in each worker process) ──
_CPU_DOMAIN = None


def _cpu_worker_init(domain_cls, seq_len: int):
    """Initialize a CPU worker process with its own domain copy.

    Runs once per worker at pool creation. Creates a CPU-only domain
    so the worker can evaluate configs without touching the GPU.
    Handles both heavy domains (seq_len, seed, device args) and
    lightweight domains (no args).
    """
    global _CPU_DOMAIN
    import os
    import torch
    torch.set_num_threads(2)
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_k, "2")
    # Try constructor variants: (seq_len, seed, device) -> (seed, device) -> ()
    try:
        _CPU_DOMAIN = domain_cls(seq_len=seq_len, seed=43, device=torch.device("cpu"))
    except TypeError:
        try:
            _CPU_DOMAIN = domain_cls(seed=43, device=torch.device("cpu"))
        except TypeError:
            _CPU_DOMAIN = domain_cls()


def _cpu_eval(job) -> dict:
    """Evaluate in a CPU worker. job = config dict, or (kind, config)
    tuple — 'quick' routes to domain.quick_eval when present. Uncaught
    errors become a fail-score so the worker stays alive."""
    global _CPU_DOMAIN
    kind, config = ("full", job) if not isinstance(job, tuple) else job
    try:
        if kind == "quick" and hasattr(_CPU_DOMAIN, "quick_eval"):
            return _CPU_DOMAIN.quick_eval(config)
        return _CPU_DOMAIN.evaluate(config)
    except Exception as e:
        try:
            n_beh = len(_CPU_DOMAIN.behavioral_dims())
        except Exception:
            n_beh = 1
        return {"score": -1e9, "behavioral": (0.0,) * n_beh,
                "metadata": {"fail": f"cpu-worker: {e}"}}


# ── GPU worker globals (spawn context — each process gets its own
#    CUDA context + domain instance; no CUDA fork) ──
_GPU_DOMAIN = None


def _gpu_worker_init(domain_mod: str, domain_cls_name: str,
                     worker_kwargs: dict):
    """Build the domain inside a spawned GPU worker process.

    The domain instance (and its FluxLM builds) lives per-process —
    evals across workers share the device but never a CUDA context,
    graph capture, or RNG state. Safe exactly because FluxTune evals
    run eager + graphless per candidate.
    """
    global _GPU_DOMAIN
    import importlib
    import os
    import torch
    # Cap intra-op threads — N workers each defaulting to ALL cores
    # oversubscribes 5-10x (100 threads on 20 cores). PyTorch docs
    # prescribe floor(cores/workers); 2 matches the CPU worker. Env
    # caps cover BLAS/OpenMP pools that set_num_threads misses —
    # set before first compute so lazy init picks them up.
    torch.set_num_threads(2)
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_k, "2")
    mod = importlib.import_module(domain_mod)
    cls = getattr(mod, domain_cls_name)
    _GPU_DOMAIN = cls(**worker_kwargs)


def _gpu_eval(job) -> dict:
    """Evaluate one config in the GPU worker. job = (kind, config);
    kind 'quick' routes to domain.quick_eval when present. Any uncaught
    error becomes a fail-score — the worker process stays alive."""
    global _GPU_DOMAIN
    kind, config = job
    try:
        if kind == "quick" and hasattr(_GPU_DOMAIN, "quick_eval"):
            return _GPU_DOMAIN.quick_eval(config)
        return _GPU_DOMAIN.evaluate(config)
    except Exception as e:
        try:
            n_beh = len(_GPU_DOMAIN.behavioral_dims())
        except Exception:
            n_beh = 1
        return {"score": -1e9, "behavioral": (0.0,) * n_beh,
                "metadata": {"fail": f"gpu-worker: {e}"}}


@dataclass
class ForgeEvolveConfig:
    """Configuration for a ForgeEvolve run."""
    # Domain (provides evaluate + config space)
    domain: Any  # BaseDomain

    # Generator population
    n_generators: int = 500
    noise_dim: int = 16
    context_dim: int = 32
    hidden_dim: int = 64
    mutation_rate: float = 0.1

    # Filtering
    filter_ratio: int = 20        # N_generated / N_evaluated
    min_evaluate: int = 10        # minimum candidates to evaluate per gen
    max_evaluate: int = 150       # maximum candidates to evaluate per gen (raised for throughput)
    elite_mutants: int = 0        # per-gen domain-space mutants of archive elites
                                  # (0=off; domains define propose_elite_mutants)

    # Proposal frontier (async submit→eval decoupling): every generated
    # candidate is decoded + deduped + queued; each gen evaluates the
    # surrogate's current best-ranked pending proposals instead of a
    # gen-local top-K. Best-first over a growing frontier — no proposal
    # is wasted, eval slots always spend on the global best-so-far.
    propose_mult: int = 1         # proposals/gen = n_generators * propose_mult
    frontier_cap: int = 4000      # pending-queue bound (evict lowest-scored)
    quick_ratio: int = 2          # quick_evals per full eval (needs
                                  # domain.quick_eval; quick results rank the
                                  # promotion set only — never archived)
    crossover_frac: float = 0.15  # fraction of neural submissions replaced by
                                  # two-elite blended proposals (recombination)
    n_islands: int = 1            # >1 = shard archive into demes (FunSearch
                                  # islands); generator g feeds island g%K
    island_migrate_gens: int = 10 # every N gens, weakest islands import elites
    gpu_eval_workers: int = 0     # >1 = spawn-process GPU eval pool (only for
                                  # domains declaring gpu_pool_ok + kwargs)
    # Pattern-breaking guarantees (the "always some random directions"
    # rule): a slice of generators gets ZERO context every gen (their
    # proposals aren't conditioned on any elite), plus raw uniform-random
    # decodes bypassing generators entirely. Without these, the whole
    # population orbits discovered cells.
    wild_gen_frac: float = 0.10   # last fraction of generators: ctx=0
    wild_frac: float = 0.04       # fraction of submissions: uniform-random
    explore_frac: float = 0.20    # first fraction of generators are
                                  # novelty-fitness explore emitters
    # Minimal-criterion novelty search (MCNS): unconditioned emitters
    # (coverage -3, wild -4) must beat this quick-score quantile before
    # spending a full-eval slot — novelty requires viability, nature's
    # "survive to reproduce" floor. 0 = no gate.
    mc_floor_q: float = 0.15
    # Winner burst (Go-Explore "return then explore"): when an eval
    # takes its archive cell, queue this many mutations of the winner
    # for next gen — exploit new winners before drifting on.
    burst_winners: int = 2        # winning configs per gen to burst
    burst_k: int = 3              # mutations queued per winner

    # Training
    generator_lr: float = 1e-3
    surrogate_lr: float = 1e-3
    surrogate_mode: str = "mlp"   # "mlp" or "gp"

    # Search
    generations: int = 50
    exploration: float = 0.3      # 0=exploit, 1=explore
    time_budget_s: float = 0      # 0 = no limit
    convergence_patience: int = 0  # 0 = disabled; >0 = stop early if best
                                   # score doesn't improve for N consecutive
                                   # generations (evolution convergence detector)
    # Adaptive population: grow generators when plateauing, shrink when stable
    adaptive_population: bool = True  # auto-tune n_generators per domain
    pop_growth_rate: float = 0.2  # grow by 20% on plateau
    pop_shrink_rate: float = 0.1  # shrink by 10% when improving steadily
    pop_min: int = 50             # minimum population size
    pop_max: int = 2000           # maximum population size
    # Dynamic domain spawning: when a domain converges, spawn a refinement
    # domain that narrows the search around the best config. This enables
    # infinite-depth search without the 57-domain limit.
    enable_refinement: bool = True
    refinement_max_depth: int = 5  # max recursive refinement depth
    refinement_narrowing: float = 0.2  # each level narrows to ±20% of parent range
    # Fixation: pin the entire run to this single domain/task.
    # - disables cross-domain transfer seeding (no foreign-topic priors)
    # - forces enable_refinement off (never jump to a spawned domain)
    # - on convergence, re-diversifies in place instead of stopping —
    #   the run keeps grinding the same topic until steps/budget end
    fixate: bool = False
    # Novelty search: alternate between novelty (explore) and quality (exploit)
    # phases to prevent plateaus. Based on Lehman & Stanley novelty search +
    # composite novelty pulsation (GECCO 2020).
    enable_novelty: bool = True
    novelty_k_neighbors: int = 5
    novelty_pulse_len: int = 3   # generations of novelty search per cycle
    quality_pulse_len: int = 5   # generations of quality search per cycle
    novelty_bonus: float = 1.5   # bonus added to surrogate during novelty phase

    # Device
    device: str = "auto"          # "auto", "cuda", "cpu"

    # Database
    db_path: str = "forge_evolve.db"
    run_id: str = ""              # empty = auto-generate
    warm_start: bool = True       # load past findings for this domain

    # Parallelism
    parallel_eval: int = 1        # GPU eval parallelism
    cpu_eval_frac: float = 0.6    # fraction of eval batch routed to the
                                  # CPU pool (rest evals in-process)

    # ── Compute split: foreground gen models + background score checks ──
    # When the domain uses an LLM gen model (gen_model_type="llm" in JSON spec),
    # the engine batches gen-model inference on the GPU (foreground) and runs
    # the SharedCheckerModel scoring asynchronously (background) to overlap
    # compute.  Set enable_compute_split=True to activate.
    enable_compute_split: bool = False
    gen_model_manager: Any = None   # GenModelManager instance (optional)
    checker_model: Any = None       # SharedCheckerModel instance (optional)
    # Background checker uses a ThreadPoolExecutor so the LLM-as-judge calls
    # overlap with the next generation's gen-model inference.
    checker_n_workers: int = 2

    # Logging
    verbose: bool = True
    log_every: int = 1


class ForgeEvolve:
    """The main evolutionary search engine."""

    def __init__(self, cfg: ForgeEvolveConfig):
        self.cfg = cfg
        try:
            import multiprocessing as _mp
            if _mp.parent_process() is not None:
                print("[ForgeEvolve] WARNING: engine instantiated inside a "
                      "spawned worker — driver is missing an "
                      "`if __name__ == '__main__'` guard; every pool worker "
                      "will recursively launch its own engine",
                      flush=True)
        except Exception:
            pass
        if cfg.fixate:
            cfg.enable_refinement = False
        self.domain = cfg.domain
        self.output_dim = self.domain.output_dim()

        # Enable TF32 tensor cores for faster matmul on Ampere+ (RTX 5070)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Device selection
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif cfg.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build generator population (batched, on device)
        gen_cfg = GeneratorConfig(
            noise_dim=cfg.noise_dim,
            context_dim=cfg.context_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=self.output_dim,
            n_generators=cfg.n_generators,
            mutation_rate=cfg.mutation_rate,
        )

        template = None
        discrete_choices = self.domain.discrete_choices()
        if discrete_choices:
            template = TemplateGenerator(discrete_choices)

        self.population = GeneratorPopulation(gen_cfg, template, device=self.device)

        # Surrogate (on device)
        self.surrogate = SurrogateModel(
            input_dim=self.output_dim,
            mode=cfg.surrogate_mode,
            lr=cfg.surrogate_lr,
            device=self.device,
        )

        # Archive
        self.archive = MapElitesArchive(self.domain.behavioral_dims())

        # Domain factory for dynamic refinement spawning
        from forge.evolution.domain_factory import DomainFactory
        self.domain_factory = DomainFactory(
            max_depth=cfg.refinement_max_depth,
            narrowing=cfg.refinement_narrowing,
        )
        self.pending_refinements: list[BaseDomain] = []

        # Novelty search: behavioral diversity to prevent plateaus
        from forge.evolution.novelty_search import NoveltyConfig, NoveltySearch
        novelty_cfg = NoveltyConfig(
            enabled=cfg.enable_novelty,
            k_neighbors=cfg.novelty_k_neighbors,
            pulse_novelty_len=cfg.novelty_pulse_len,
            pulse_quality_len=cfg.quality_pulse_len,
            novelty_bonus=cfg.novelty_bonus,
        )
        self.novelty = NoveltySearch(novelty_cfg)

        # Trainer (uses batched generator, on device)
        self.trainer = GeneratorTrainer(
            self.population.batched_gen, lr=cfg.generator_lr,
            device=self.device,
        )

        # State
        self.generation = 0
        self.all_results: list[dict] = []
        self.max_results: int = 10000  # cap to prevent OOM on long runs
        self.discoveries: list[dict] = []
        # Proposal frontier: pending decoded candidates awaiting eval.
        # Entries: {"key", "params" (cpu row), "config", "generator_idx",
        #           "noise"}. Keys live in _frontier_keys while queued and
        # move to _eval_cache when popped for eval.
        self._frontier: list[dict] = []
        self._frontier_keys: set = set()
        self._submitted: int = 0
        self.start_time = 0.0

        # Config dedup cache: prevents re-evaluating identical configs within
        # a run AND prevents duplicate discoveries being saved to DB.
        # Key = canonical config string, value = (score, behavioral, metadata)
        self._eval_cache: dict[str, tuple] = {}
        # Param vectors of evaluated/committed candidates — drives the
        # novelty-weighted frontier pop (cap memory at 1024 rows).
        self._evald_params: np.ndarray | None = None
        # Evaluated (params, score) pairs — ground-truth set for
        # local-incumbent thresholds (MAE-style Δ ranking) and NSLC
        # local-competition fitness (cap 2048 rows).
        self._scored_params: np.ndarray | None = None
        self._scored_scores: np.ndarray | None = None
        # Wild explorer generators (last wild_gen_frac of population):
        # zero context + novelty/local-quality fitness — never trained
        # on raw score (PGA-ME: diversity emitters aren't quality-
        # punished, or they collapse into exploiters).
        self._wild_gens: set[int] = set()
        # Per-axis bin eval counts for the coverage emitter.
        self._bin_counts: dict[str, dict] = {}

        # CPU/GPU pair processing: separate processes for true parallelism
        # GPU process: generators + surrogate + GPU-side evals
        # CPU processes: CPU-side evals (bypass GIL entirely)
        self.cpu_pool = None
        self.n_cpu_workers = 0
        if cfg.parallel_eval > 1 and self.device.type == "cuda":
            try:
                import multiprocessing as mp
                # Use spawn to avoid CUDA fork issues
                ctx = mp.get_context("spawn")
                self.n_cpu_workers = cfg.parallel_eval - 1
                self.cpu_pool = ctx.Pool(
                    processes=self.n_cpu_workers,
                    initializer=_cpu_worker_init,
                    initargs=(type(self.domain), self.domain.seq_len if hasattr(self.domain, "seq_len") else 2048),
                )
            except Exception as e:
                if cfg.verbose:
                    print(f"[CPU pool] Failed to start: {e}, falling back to GPU-only")
                self.cpu_pool = None
                self.n_cpu_workers = 0

        # GPU worker pool: spawn processes, each with own CUDA context
        # + domain instance. Opt-in ONLY for domains declaring
        # gpu_pool_ok + gpu_worker_kwargs (FluxTune: eager+graphless
        # per-candidate evals are safe to run on the shared device).
        self.gpu_pool = None
        self.n_gpu_workers = 0
        if (cfg.gpu_eval_workers > 1
                and getattr(self.domain, "gpu_pool_ok", False)
                and self.device.type == "cuda"):
            try:
                import multiprocessing as mp
                ctx = mp.get_context("spawn")
                self.n_gpu_workers = cfg.gpu_eval_workers
                self.gpu_pool = ctx.Pool(
                    processes=self.n_gpu_workers,
                    initializer=_gpu_worker_init,
                    initargs=(type(self.domain).__module__,
                              type(self.domain).__name__,
                              self.domain.gpu_worker_kwargs()),
                    # Recycle workers — nested-engine evals leave
                    # CUDA-context + allocator residue per eval
                    # (observed ~1GB/40 evals creep); respawn bounds
                    # RSS without touching correctness.
                    maxtasksperchild=40)
                if cfg.verbose:
                    print(f"[GPU pool] {self.n_gpu_workers} workers "
                          f"for {self.domain.name()}")
            except Exception as e:
                if cfg.verbose:
                    print(f"[GPU pool] Failed to start: {e}, "
                          f"falling back to serial GPU eval")
                self.gpu_pool = None
                self.n_gpu_workers = 0

        # Database
        self.db = FindingsDB(cfg.db_path)
        self.run_id = cfg.run_id or f"{self.domain.name()}_{int(time.time())}"
        self._warm_started = False
        self._pending_discoveries = []  # batched DB saves

        # ── Compute split: gen model + checker wiring ──
        self._gen_mgr = cfg.gen_model_manager
        self._checker = cfg.checker_model
        self._checker_pool = None
        if cfg.enable_compute_split:
            import concurrent.futures
            self._checker_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=cfg.checker_n_workers,
                thread_name_prefix="checker",
            )
            # Auto-create defaults if not provided
            if self._gen_mgr is None:
                from .gen_model_manager import GenModelManager
                self._gen_mgr = GenModelManager(db=self.db)
            if self._checker is None:
                from .checker_model import get_checker
                self._checker = get_checker()
            # Wire gen model into domain if it supports set_gen_model
            if hasattr(self.domain, "set_gen_model") and self._gen_mgr is not None:
                gm = getattr(self._gen_mgr, "_model", None)
                if gm is not None:
                    self.domain.set_gen_model(gm)
                    self._log(f"[ComputeSplit] gen model attached to domain "
                              f"({gm.param_count()} params)")
            self._log(f"[ComputeSplit] enabled — "
                      f"checker={'yes' if self._checker else 'no'}, "
                      f"gen_mgr={'yes' if self._gen_mgr else 'no'}")

        # ── Canonical load: ALWAYS load the best-ever generators + surrogate
        # for this domain, regardless of warm_start. This is the permanent
        # knowledge layer — findings from all past runs (any profile, any
        # DB) accumulate here. The search starts from the best-known point.
        # Skip canonical load for Generic domains (heuristic evaluator)
        is_generic = "Generic_" in self.domain.name()
        canonical_best = self.db.get_canonical_best_score(self.domain.name())
        if not is_generic and self.db.load_canonical_generators(self.domain.name(),
                                             self.population.batched_gen):
            self._warm_started = True
            self._log(f"Loaded canonical generators for '{self.domain.name()}'"
                      + (f" (past best={canonical_best:.4f})" if canonical_best else ""))
        if not is_generic and self.db.load_canonical_surrogate(self.domain.name(), self.surrogate):
            if not self._warm_started:
                self._warm_started = True
            self._log(f"Loaded canonical surrogate for '{self.domain.name()}'")

        # Warm-start from past findings (per-run generators + discoveries)
        if cfg.warm_start:
            n_seeded = self.db.seed_from_past(
                self.domain.name(),
                self.population.batched_gen,
                self.surrogate,
            )
            if n_seeded > 0:
                self._warm_started = True
                self._log(f"Warm-started from {n_seeded} past findings "
                          f"(generators + surrogate loaded)")
            else:
                # Cross-domain transfer: if no same-domain history, borrow
                # generators from a different domain with matching output_dim.
                # Skip for Generic domains (heuristic, not real performance)
                # and fixated runs (single-topic focus: no foreign priors).
                if not is_generic and not cfg.fixate:
                    od = self.domain.output_dim()
                    n_xfer = self.db.seed_cross_domain(
                        self.domain.name(),
                        self.population.batched_gen,
                        self.surrogate,
                        od,
                    )
                    if n_xfer > 0:
                        self._warm_started = True
                        self._log(f"Cross-domain transfer from {n_xfer} source domain(s) "
                                  f"(output_dim={od})")
            # Also load past discoveries as archive seeds
            # Cross-run cache: use stored scores from DB instead of re-evaluating
            past = self.db.query_best_configs(self.domain.name(), limit=20)
            n_cached = 0
            n_reevaled = 0
            for p in past:
                key = self._config_key(p["config"])
                if key in self._eval_cache:
                    continue  # already evaluated as a seed
                # Use stored score from DB (cross-run cache) — avoids
                # re-evaluating configs that were already scored in past runs.
                stored_score = p.get("score")
                stored_behavioral = p.get("behavioral", (0,))
                stored_metadata = p.get("metadata", {})
                if stored_score is not None:
                    self._eval_cache[key] = (stored_score,
                                             stored_behavioral,
                                             stored_metadata)
                    self.archive.add(p["config"], stored_score,
                                     stored_behavioral, -1)
                    n_cached += 1
                else:
                    result = self.domain.evaluate(p["config"])
                    self._eval_cache[key] = (result["score"],
                                             result.get("behavioral", (0,)),
                                             result.get("metadata", {}))
                    self.archive.add(p["config"], result["score"],
                                     result.get("behavioral", (0,)), -1)
                    n_reevaled += 1
            if past:
                self._log(f"Loaded {len(past)} past discoveries into archive "
                          f"({n_cached} cached, {n_reevaled} re-evaluated)")

        # Seed from domain's known-good configs
        seed_configs = self.domain.seed_configs()
        if seed_configs:
            self._log(f"Evaluating {len(seed_configs)} seed configs...")
            seed_params = []
            seed_scores = []
            for sc in seed_configs:
                key = self._config_key(sc)
                if key in self._eval_cache:
                    continue  # already evaluated (e.g., from past discoveries)
                result = self.domain.evaluate(sc)
                self._eval_cache[key] = (result["score"],
                                         result.get("behavioral", (0,)),
                                         result.get("metadata", {}))
                self.archive.add(sc, result["score"],
                                 result.get("behavioral", (0,)), -1)
                self.all_results.append({
                    "generation": -1,
                    "config": sc,
                    "score": result["score"],
                    "behavioral": result.get("behavioral", (0,)),
                    "metadata": result.get("metadata", {}),
                })
                # Persist seeds so query_best_configs reflects the archive's
                # best (dedup updates the row if a better score comes later)
                self._pending_discoveries.append({
                    "generation": -1,
                    "config": sc,
                    "score": result["score"],
                    "behavioral": result.get("behavioral", (0,)),
                    "metadata": result.get("metadata", {}),
                })
                # Encode for surrogate training
                params = self.domain.encode(sc)
                if params is not None:
                    seed_params.append(params)
                    seed_scores.append(result["score"])

            # Train surrogate on seeds immediately (not random anymore)
            if seed_params:
                param_tensor = torch.stack(seed_params).to(self.device)
                score_tensor = torch.tensor(seed_scores, dtype=torch.float32)
                try:
                    self.surrogate.train(param_tensor, score_tensor, epochs=5)
                    self._log(f"Surrogate trained on {len(seed_params)} seeds "
                              f"(best seed score={max(seed_scores):.4f})")
                except Exception as e:
                    self._log(f"Surrogate seed training skipped: {e}")

            # Update context with best seed
            if seed_scores:
                best_idx = int(np.argmax(seed_scores))
                self.population.update_context(seed_params[best_idx])

    # ── Config dedup ──────────────────────────────────────────────────

    @staticmethod
    def _config_key(config: dict) -> str:
        """Canonical hashable key for a config dict.
        Sorts keys and normalizes values to handle int/str equivalence
        (e.g., 32 and '32' from JSON round-trip)."""
        normalized = {}
        for k in sorted(config.keys()):
            v = config[k]
            # Normalize numeric strings to their numeric form
            if isinstance(v, str):
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            # Numpy scalar → python scalar
            if hasattr(v, 'item') and not hasattr(v, '__len__'):
                v = v.item()
            # Round floats to 6 decimals to avoid hash misses from FP noise
            if isinstance(v, float):
                v = round(v, 6)
            # Numpy arrays / lists → tuple of rounded floats
            # (but not strings — they have no numeric values to round)
            if isinstance(v, str):
                pass  # keep as-is
            elif hasattr(v, 'tolist'):
                v_list = v.tolist()
                if isinstance(v_list, (list, tuple)):
                    v = tuple(round(float(x), 6) for x in v_list)
                else:
                    v = round(float(v_list), 6)
            elif isinstance(v, (list, tuple)):
                v = tuple(round(float(x), 6) if isinstance(x, (int, float))
                          else x for x in v)
            normalized[k] = v
        return json.dumps(normalized, sort_keys=True, default=str)

    def _eval_with_dedup(self, config: dict) -> dict | None:
        """Evaluate a config, returning cached result if we've seen it before.
        Returns None if this is a duplicate (caller should skip)."""
        key = self._config_key(config)
        if key in self._eval_cache:
            return None  # duplicate — skip
        result = self._safe_eval(config)
        self._eval_cache[key] = (result["score"],
                                 result.get("behavioral", (0,)),
                                 result.get("metadata", {}))
        return result

    # ── Proposal frontier ────────────────────────────────────────────

    def _frontier_scores(self) -> np.ndarray:
        """Surrogate-score all pending frontier entries (batched)."""
        params = torch.stack([e["params"] for e in self._frontier]
                             ).to(self.device)
        return self.surrogate.predict(params).detach().cpu().numpy()

    def _frontier_evict(self) -> None:
        """Bound the frontier to frontier_cap, keeping best-scored."""
        if len(self._frontier) <= self.cfg.frontier_cap:
            return
        scores = self._frontier_scores()
        order = np.argsort(-scores)[:self.cfg.frontier_cap]
        keep = set(order.tolist())
        evicted = [e for i, e in enumerate(self._frontier)
                   if i not in keep]
        for e in evicted:
            # Never evaluated — free the key so it may re-enter later
            self._frontier_keys.discard(e["key"])
        self._frontier = [self._frontier[i] for i in order]

    def _frontier_pop(self, n: int, gen: int) -> list[dict]:
        """Pop n pending proposals: surrogate best-first + epsilon tail."""
        if n <= 0 or not self._frontier:
            return []
        scores = self._frontier_scores()
        order = np.argsort(-scores)
        n = min(n, len(order))
        if self.cfg.enable_novelty and self.novelty.phase == "novelty":
            eps = max(0.1, 0.5 - gen * 0.005)
        else:
            eps = max(0.1, 0.3 - gen * 0.005)
        n_rand = int(n * eps)
        n_best = n - n_rand
        # Improvement-weighted best slice (CMA-MAE-style): rank by
        # predicted Δ over the LOCAL incumbent — the score of the
        # nearest evaluated point — not raw surrogate score. A mediocre
        # prediction that beats its region's best outranks a strong
        # prediction in a maxed region; the distance term still covers
        # virgin space. Pure-surrogate pops converge onto near-dupes of
        # the evaluated region (the new=0 collapse on mech_search).
        if n_best > 0 and self._evald_params is not None:
            pool = order[:min(2 * n_best, len(order))]
            dmin = self._min_dist_to_evald(pool)
            s_pool = scores[pool]
            if self._scored_params is not None:
                t_loc = self._nearest_scored(pool)
                d_pool = s_pool - t_loc
                d_n = (d_pool - d_pool.min()) / (np.ptp(d_pool) + 1e-9)
                sel = d_n + 0.3 * dmin
            else:
                s_n = (s_pool - s_pool.min()) / (np.ptp(s_pool) + 1e-9)
                sel = s_n + 0.3 * dmin
            idxs = list(pool[np.argsort(-sel)[:n_best]])
        else:
            idxs = list(order[:n_best])
        rest = [i for i in order if i not in set(idxs)]
        if n_rand > 0 and len(rest) > 0:
            idxs += np.random.choice(
                rest, size=min(n_rand, len(rest)), replace=False).tolist()
        picked_set = set(idxs)
        picked = [self._frontier[i] for i in idxs]
        self._frontier = [e for i, e in enumerate(self._frontier)
                          if i not in picked_set]
        for e in picked:
            self._frontier_keys.discard(e["key"])
            # Popped = committed to eval — permanently deduped
            self._eval_cache[e["key"]] = None
        self._record_evald(picked)
        return picked

    def _record_evald(self, entries: list[dict]) -> None:
        """Append param vectors of committed candidates (cap 1024 rows)."""
        rows = [e["params"].detach().float().cpu().numpy()
                for e in entries if e.get("params") is not None]
        if not rows:
            return
        new = np.stack(rows)
        self._evald_params = new if self._evald_params is None else \
            np.concatenate([self._evald_params, new])
        if len(self._evald_params) > 1024:
            self._evald_params = self._evald_params[-1024:]

    def _min_dist_to_evald(self, idxs) -> np.ndarray:
        """Min L2 distance from frontier entries idxs to evaluated set,
        min-max normalized to [0,1]. Vectorized pairwise distance."""
        P = np.stack([self._frontier[i]["params"].detach().float()
                      .cpu().numpy() for i in idxs])
        E = self._evald_params
        # ||p-e||^2 = |p|^2 + |e|^2 - 2 p·e
        d2 = (P * P).sum(1)[:, None] + (E * E).sum(1)[None, :] - 2 * P @ E.T
        dmin = np.sqrt(np.maximum(d2.min(1), 0.0))
        dmax = dmin.max() if len(dmin) else 0.0
        return dmin / (dmax + 1e-9)

    def _entry_params(self, entries) -> np.ndarray:
        """Param matrix for entries, coerced to output_dim rows —
        template/decoded entries can carry short encodings."""
        dim = self.domain.output_dim()
        rows = []
        for e in entries:
            p = e["params"]
            if torch.is_tensor(p):
                p = p.detach().float().cpu().numpy()
            p = np.asarray(p, dtype=np.float64).ravel()
            if len(p) < dim:
                p = np.pad(p, (0, dim - len(p)),
                           constant_values=0.5)
            rows.append(p[:dim])
        return np.stack(rows)

    def _record_scored(self, entries, scores) -> None:
        """Track (params, score) of evaluated candidates — the ground-
        truth set for local-incumbent thresholds and NSLC fitness."""
        if not entries:
            return
        try:
            P = self._entry_params(entries)
        except Exception:
            return
        S = np.asarray(scores, dtype=np.float64)[:len(P)]
        self._scored_params = P if self._scored_params is None else \
            np.concatenate([self._scored_params, P])
        self._scored_scores = S if self._scored_scores is None else \
            np.concatenate([self._scored_scores, S])
        if len(self._scored_params) > 2048:
            self._scored_params = self._scored_params[-2048:]
            self._scored_scores = self._scored_scores[-2048:]

    def _nearest_scored(self, idxs):
        """Per frontier idx: (normalized distance, incumbent score) of
        the nearest EVALUATED point — a Voronoi-local incumbent used
        for CMA-MAE-style improvement ranking."""
        P = self._entry_params([self._frontier[i] for i in idxs])
        E, S = self._scored_params, self._scored_scores
        d2 = (P * P).sum(1)[:, None] + (E * E).sum(1)[None, :] - 2 * P @ E.T
        nn = np.argmin(d2, axis=1)
        # min_f floor (CMA-MAE's empty-cell threshold): a fail-score
        # incumbent (-1e9) would otherwise make every neighbor look
        # like a +1e9 improvement. Weak regions are opportunities, not
        # infinite ones.
        min_f = np.quantile(S, 0.10)
        return np.maximum(S[nn], min_f)

    def _local_quality(self, entries, scores, k: int = 15) -> list:
        """NSLC local competition: fraction of each candidate's k
        nearest evaluated neighbors it outperforms. <3 neighbors → 1.0
        (virgin region is locally best by definition — keeps regional
        champions alive against globally-better crowded basins)."""
        if self._scored_params is None or len(self._scored_params) < 3:
            return [1.0] * len(entries)
        P = self._entry_params(entries)
        E, S = self._scored_params, self._scored_scores
        d2 = (P * P).sum(1)[:, None] + (E * E).sum(1)[None, :] - 2 * P @ E.T
        kk = min(k, len(S))
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        return [float(np.mean(S[nn[i]] < scores[i]))
                for i in range(len(P))]

    def _init_islands(self) -> None:
        """Lazily create K island archives (FunSearch-style demes)."""
        if getattr(self, "_islands", None) is None:
            if self.cfg.n_islands > 1:
                self._islands = [MapElitesArchive(
                    self.domain.behavioral_dims())
                    for _ in range(self.cfg.n_islands)]
                self._weak_islands: set[int] = set()
            else:
                self._islands = []

    def _contexts_for_generation(self, gen: int) -> torch.Tensor | None:
        """Archive-conditioned emitter contexts: one sampled parent per
        generator, instead of broadcasting a single elite to everyone.

        Mix per generator: 60% UCB elite, 15% uniform elite, 15% blend of
        two elites (iso-line recombination proxy), 10% global best.
        With islands: generator g samples from island g%K's archive —
        weak islands (lowest best_score) import global elites instead
        (elite migration). Returns (n_generators, context_dim) or None."""
        if not self.archive.get_all_elites():
            return None
        n = self.population.cfg.n_generators
        ctx_dim = self.cfg.context_dim
        out = torch.zeros(n, ctx_dim, device=self.device)
        n_isl = len(getattr(self, "_islands", []) or [])
        # Wild generators: last wild_gen_frac fraction keep ZERO context —
        # proposals not conditioned on any elite (permanent explorers).
        n_wild = int(n * self.cfg.wild_gen_frac)
        if n_wild:
            self._wild_gens = set(range(n - n_wild, n))
        for g in range(n):
            if g >= n - n_wild:
                continue  # leave zeros — unconditioned proposal
            # Per-generator source archive (island or global)
            src = self.archive
            if n_isl:
                isl = g % n_isl
                if (isl not in self._weak_islands
                        and self._islands[isl].get_all_elites()):
                    src = self._islands[isl]
            elites = src.get_all_elites()
            if not elites:
                continue
            r = np.random.random()
            try:
                if r < 0.60:
                    e = src.sample_elite_ucb(current_gen=gen)
                    p = self.domain.encode(e.config) if e is not None else None
                elif r < 0.75:
                    e = src.sample_elite()
                    p = self.domain.encode(e.config) if e is not None else None
                elif r < 0.90 and len(elites) >= 2:
                    e1, e2 = np.random.choice(elites, 2, replace=False)
                    p1 = self.domain.encode(e1.config)
                    p2 = self.domain.encode(e2.config)
                    p = (p1 + p2) / 2 if (p1 is not None
                                          and p2 is not None) else p1
                else:
                    e = src.best_entry
                    p = self.domain.encode(e.config) if e is not None else None
                if p is not None:
                    k = min(p.shape[0], ctx_dim)
                    out[g, :k] = p[:k].to(self.device)
            except Exception:
                pass  # leave zeros — unconditioned proposal
        return out

    def _island_migration(self, gen: int) -> None:
        """Every island_migrate_gens: rank demes by best_score; the
        weakest half import global elites next interval (elite gene
        flow — FunSearch restarts worst islands from best clones)."""
        if not self._islands:
            return
        if gen == 0 or gen % self.cfg.island_migrate_gens != 0:
            return
        ranked = sorted(range(len(self._islands)),
                        key=lambda i: self._islands[i].best_score)
        n_weak = max(1, len(self._islands) // 2)
        self._weak_islands = set(ranked[:n_weak])
        if self.cfg.verbose:
            scores = [f"{self._islands[i].best_score:.2f}"
                      for i in ranked]
            self._log(f"  [Islands] migration — weak={sorted(self._weak_islands)}"
                      f" (ranked best: {', '.join(scores)})")

    def _eval_configs(self, configs: list[dict],
                      eval_fn=None) -> tuple[list[dict], list[dict]]:
        """Dispatch evals -> (results, configs). Compute-split can remap
        configs, so it returns its own copy. Quick-tier eval_fn always
        takes the plain GPU/serial path — cheap evals must not split."""
        if (eval_fn is None and self.cfg.enable_compute_split
                and hasattr(self.domain, "set_gen_model")):
            return self._compute_split_eval(configs)
        if (self.gpu_pool is not None
                and getattr(self.domain, "gpu_pool_ok", False)):
            # Spawned GPU workers: each process owns its CUDA context +
            # domain; evals share the device, never a context. Quick
            # tier routes through quick_eval inside workers.
            kind = "quick" if eval_fn is not None else "full"
            try:
                jobs = [(kind, c) for c in configs]
                results = self.gpu_pool.map(
                    _gpu_eval, jobs,
                    chunksize=max(1, len(jobs)
                                  // (self.n_gpu_workers * 4)))
                return results, configs
            except Exception as e:
                self._log(f"  [GPUPool] failed ({e}) — serial fallback")
                self.gpu_pool = None
                self.n_gpu_workers = 0
                return self._gpu_batch_eval(configs, eval_fn), configs
        if (self.cpu_pool is not None
                and getattr(self.domain, "cpu_eval_ok", True)
                and len(configs) >= 8):
            # Split: cpu_eval_frac to the pool, rest in-process. Serial-
            # eval domains (eval_thread_safe=False) should push ~all
            # to workers — in-process serial is their worst path.
            # Quick-tier evals route through ("quick", cfg) jobs.
            kind = "quick" if eval_fn is not None else "full"
            n_cpu = int(len(configs) * self.cfg.cpu_eval_frac)
            gpu_configs = configs[:len(configs) - n_cpu]
            cpu_configs = configs[len(configs) - n_cpu:]
            chunksize = max(1, len(cpu_configs) // (self.n_cpu_workers * 2))
            cpu_result_async = self.cpu_pool.map_async(
                _cpu_eval, [(kind, c) for c in cpu_configs],
                chunksize=chunksize)
            gpu_results = self._gpu_batch_eval(gpu_configs, eval_fn)
            cpu_results = cpu_result_async.get()
            return gpu_results + cpu_results, gpu_configs + cpu_configs
        return self._gpu_batch_eval(configs, eval_fn), configs

    def _gpu_batch_eval(self, configs: list[dict], eval_fn=None) -> list[dict]:
        """Evaluate configs on GPU with Triton batched kernels + CUDA stream overlap.

        Priority:
        1. Triton batched eval (if domain supports it) — single GPU launch for all configs
        2. Threaded eval with CUDA streams — overlap kernel launches across configs
        3. Sequential eval — fallback for small batches
        """
        import torch

        if len(configs) <= 2:
            return [self._safe_eval(cfg) for cfg in configs]

        # ── Try Triton batched evaluation first ──
        # This processes all configs in a single GPU operation, eliminating
        # Python/kernel-launch overhead entirely for supported domains.
        if self.device.type == "cuda" and not hasattr(self, '_batched_eval'):
            try:
                from forge.evolution.triton_batched import BatchedEvaluator
                self._batched_eval = BatchedEvaluator(self.domain)
            except Exception:
                self._batched_eval = None

        if (getattr(self, "_batched_eval", None) is not None
                and self._batched_eval.can_batch()
                and len(configs) >= 4):
            try:
                results = self._batched_eval.batch_evaluate(configs)
                if all(r is not None for r in results):
                    return results
            except Exception:
                logger.debug("Batched eval failed, falling through to threaded eval", exc_info=True)

        # Fixed chunk size: each eval uses ~100MB, reserve 2GB for generators
        # On 12GB GPU: (12GB - 2GB) / (100MB * 1.5) = ~66 concurrent evals
        # Cap at 8 for thread safety
        # VRAM-aware: reduce chunk if VRAM is already heavily used
        if not hasattr(self, '_eval_chunk'):
            if self.device.type == "cuda":
                total = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                available = total - allocated
                # Reserve 1GB headroom for eval workspace
                safe_budget = max(available - 1 * 1024**3, 256 * 1024**2)
                self._eval_chunk = max(1, min(8, int(safe_budget / (150 * 1024**2))))
            else:
                self._eval_chunk = len(configs)

        chunk = self._eval_chunk

        # For small batches, just run sequentially (no thread overhead)
        if len(configs) <= chunk:
            return [self._safe_eval(cfg) for cfg in configs]

        # Chunked evaluation with threaded eval for CUDA stream overlap
        results = []
        for i in range(0, len(configs), chunk):
            batch = configs[i:i + chunk]
            # Use threaded eval within each chunk for kernel overlap.
            # Domains whose evaluate() is not CUDA-stream-safe (concurrent
            # graph captures, nested engines, large per-eval VRAM) opt out
            # via eval_thread_safe=False -> serial _safe_eval on the
            # default stream.
            if (self.device.type == "cuda" and len(batch) > 2
                    and getattr(self.domain, "eval_thread_safe", True)):
                results.extend(self._threaded_eval(batch, eval_fn))
            else:
                results.extend(self._safe_eval(cfg, eval_fn)
                               for cfg in batch)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return results

    def _safe_eval(self, config: dict, eval_fn=None) -> dict:
        """Evaluate a single config with OOM recovery."""
        fn = eval_fn or self.domain.evaluate
        try:
            return fn(config)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                # Return a penalty score so the search doesn't crash
                return {"score": -1e9, "behavioral": (0.0,) * len(self.domain.behavioral_dims()),
                        "metadata": {"oom": True}}
            raise

    def _compute_split_eval(self, configs: list[dict]) -> tuple[list[dict], list[dict]]:
        """Compute-split evaluation for LLM gen-model domains.

        Foreground: gen model generates answers (batched on GPU).
        Background: SharedCheckerModel scores answers asynchronously.

        The domain.evaluate() already calls the gen model internally (via
        set_gen_model).  This method wraps the evaluation so that:
        1. Gen-model inference (foreground) runs in a thread pool to overlap
           CUDA kernel launches.
        2. After each eval completes, if the checker is available and the
           domain's spec has checker_type="llm_judge", the checker score is
           submitted to the background pool and merged when ready.

        Returns (results, configs) in the same order as input.
        """
        # Phase A: foreground gen-model inference (domain.evaluate calls gen model)
        if self.device.type == "cuda" and len(configs) > 2:
            raw_results = self._threaded_eval(configs)
        else:
            raw_results = [self._safe_eval(cfg) for cfg in configs]

        # Phase B: background checker scoring (LLM-as-judge, optional)
        if self._checker is None or self._checker_pool is None:
            return raw_results, configs

        # Submit checker calls to background pool
        futures = []
        for i, (cfg, res) in enumerate(zip(configs, raw_results)):
            meta = res.get("metadata", {})
            problem = meta.get("problem", "")
            model_answer = meta.get("model_answer", "")
            real_answer = meta.get("real_answer")
            if not problem or not model_answer:
                futures.append(None)
                continue
            req_text = (f"Problem: {problem}\n"
                        f"Expected: {real_answer}\n"
                        f"Answer: {model_answer}")
            req = {
                "prompt": req_text,
                "requirements": str(real_answer) if real_answer is not None else "",
                "expected": str(real_answer) if real_answer is not None else "",
            }
            fut = self._checker_pool.submit(self._checker.check, req)
            futures.append(fut)

        # Merge checker scores (non-blocking — if not ready, skip)
        for i, fut in enumerate(futures):
            if fut is None:
                continue
            try:
                cscore = fut.result(timeout=0.1)
                raw_results[i]["metadata"]["checker_score"] = cscore
            except Exception:
                logger.debug("Checker score not ready, heuristic score stands", exc_info=True)

        return raw_results, configs

    def _threaded_eval(self, configs: list[dict], eval_fn=None) -> list[dict]:
        """Evaluate configs using threads with CUDA streams for kernel overlap.

        Each thread gets its own CUDA stream so kernels from multiple evals
        can overlap on the GPU, keeping utilization high. The GIL is released
        during CUDA calls, so threads achieve real parallelism.
        """
        from concurrent.futures import ThreadPoolExecutor

        import torch

        # More threads = more kernel overlap. Cap at len(configs) and 8.
        n_threads = min(8, len(configs))
        results = [None] * len(configs)

        fn = eval_fn or self.domain.evaluate

        def _eval_one(idx_cfg):
            idx, cfg = idx_cfg
            try:
                if self.device.type == "cuda":
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        result = fn(cfg)
                    stream.synchronize()
                else:
                    result = fn(cfg)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    result = {"score": -1e9, "behavioral": (0.0,) * len(self.domain.behavioral_dims()),
                              "metadata": {"oom": True}}
                else:
                    raise
            return idx, result

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for idx, result in pool.map(_eval_one, enumerate(configs)):
                results[idx] = result

        return results

    def _spawn_refinement(self):
        """Spawn a refinement domain from the converged parent.

        Creates a narrowed search domain centered on the best config,
        registers it in the factory, and queues it for the next loop.
        """
        if not self.archive.best_entry:
            return
        best = self.archive.best_entry
        best_config = best.config
        best_params = self.domain.encode(best_config)

        # Determine refinement depth from parent name
        parent_name = self.domain.name()
        depth = 1
        if "_refine_d" in parent_name:
            depth = int(parent_name.split("_refine_d")[-1]) + 1

        child = self.domain_factory.spawn_refinement(
            parent=self.domain,
            best_config=best_config,
            best_params=best_params,
            depth=depth,
        )
        if child is not None:
            self.domain_factory.register(child)
            self.pending_refinements.append(child)
            self._log(
                f"  [Refinement] Spawned {child.name()} "
                f"(depth={depth}, narrowing=±{child.narrowing:.1%})")

    def _fixate_rediversify(self):
        """Fixate-mode plateau escape: stay on the same domain, re-diversify.

        Forces a long novelty phase + re-mutates the bottom third of the
        generator population from elite parents at a high rate, then
        re-centers the generator context on a RANDOM archive elite —
        recentering on global best collapses the burst back into the
        converged basin (mech run: best never moved across plateaus).
        Rotating elites rotates the basin being explored.
        """
        if self.cfg.enable_novelty:
            self.novelty.phase = "novelty"
            self.novelty.phase_counter = 0
            self.novelty._current_novelty_len = self.novelty.cfg.max_novelty_len
        bg = self.population.batched_gen
        fitness = bg.fitness_ema.detach().cpu().numpy()
        ranked = np.argsort(fitness)[::-1]
        n = len(ranked)
        n_re = max(1, n // 3)
        n_top = max(1, n // 5)
        for i, bad in enumerate(ranked[-n_re:]):
            parent = ranked[i % n_top]
            bg.mutate_generator(int(bad), int(parent), rate=0.35)
        # Re-center on a RANDOM archive elite, not global best — best
        # pulls the burst back into the converged basin (mech run:
        # best never moved across 2 re-diversifications).
        elites = self.archive.get_all_elites()
        if elites:
            pick = elites[int(np.random.randint(len(elites)))]
            enc = self.domain.encode(pick.config)
            if enc is not None:
                self.population.update_context(enc.to(self.device))
            self._log(f"  [Fixate] Re-mutated bottom {n_re}/{n} generators, "
                      f"context re-centered on elite score={pick.score:.2f} "
                      f"(of {len(elites)} cells)")
        else:
            self._log(f"  [Fixate] Re-mutated bottom {n_re}/{n} generators")

    def _log(self, msg: str):
        if self.cfg.verbose:
            elapsed = time.time() - self.start_time if self.start_time else 0
            line = f"[Gen {self.generation:3d} | {elapsed:6.1f}s] {msg}"
            try:
                print(line)
            except UnicodeEncodeError:
                # cp1252 consoles can't encode →/—/± etc.
                print(line.encode("ascii", "replace").decode())

    def run(self) -> dict:
        """Run the full evolutionary search loop."""
        self.start_time = time.time()
        self._prev_best_score = float('-inf')
        self._convergence_counter = 0
        self._pop_stable_count = 0
        self._pop_plateau_count = 0
        self._init_islands()
        self._log(f"Starting ForgeEvolve: {self.cfg.n_generators} generators, "
                  f"filter_ratio={self.cfg.filter_ratio}, "
                  f"domain={self.domain.name()}, device={self.device}"
                  + (f", CPU_pool={self.n_cpu_workers} workers" if self.cpu_pool else ""))

        # ── Pre-loop: if unapplied discoveries exist in DB, train on them first ──
        # This ensures that discoveries from previous runs are incorporated into
        # the gen model's knowledge before starting a new search. After training,
        # the discoveries are flagged as applied so they won't be re-trained on.
        if self._gen_mgr is not None:
            try:
                n_unapplied = self.db.count_unapplied()
                if n_unapplied > 0:
                    self._log(f"[TrainFirst] {n_unapplied} unapplied discoveries in DB — "
                              f"training gen model before search")
                    unapplied = self.db.get_unapplied_discoveries(limit=200)
                    solutions = []
                    applied_ids = []
                    for d in unapplied:
                        meta = d.get("metadata", {}) or {}
                        problem = meta.get("problem", "")
                        answer = meta.get("model_answer", "")
                        if problem and answer:
                            solutions.append({
                                "input": problem,
                                "output": answer,
                                "score": d.get("score", 0),
                            })
                        applied_ids.append(d["id"])
                    if solutions:
                        self._log(f"[TrainFirst] fine-tuning on {len(solutions)} solutions")
                        try:
                            stats = self._gen_mgr.fine_tune_on_solutions(solutions)
                            self._log(f"[TrainFirst] done: "
                                      f"{stats.get('n_examples', 0)} examples, "
                                      f"final_loss={stats.get('loss_history', [0])[-1]:.4f}")
                        except Exception as e:
                            self._log(f"[TrainFirst] fine-tune failed: {e}")
                    # Mark all unapplied discoveries as applied (even those
                    # without problem/answer text — they don't need training)
                    if applied_ids:
                        self.db.mark_applied(applied_ids)
                        self._log(f"[TrainFirst] marked {len(applied_ids)} discoveries as applied")
                else:
                    self._log("[TrainFirst] no unapplied discoveries — starting fresh search")
            except Exception as e:
                self._log(f"[TrainFirst] error checking unapplied: {e}")

        try:
            for gen in range(self.cfg.generations):
                self.generation = gen

                # Clear CUDA cache periodically to prevent fragmentation OOM
                if gen > 0 and gen % 5 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()

                # ── Novelty search: update phase (novelty ↔ quality pulsation) ──
                if self.cfg.enable_novelty and gen > 0:
                    self.novelty.update_phase(self.archive.best_score)
                    if gen == 1 or self.novelty.phase_counter == 1:
                        self._log(f"  [Novelty] {self.novelty.status()}")

                if self.cfg.time_budget_s > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > self.cfg.time_budget_s:
                        self._log(f"Time budget exceeded ({elapsed:.0f}s)")
                        break

                # ── Convergence detection: stop early if no improvement ──
                if self.cfg.convergence_patience > 0 and gen > 0:
                    if self.archive.best_score > self._prev_best_score:
                        self._convergence_counter = 0
                        self._prev_best_score = self.archive.best_score
                    else:
                        self._convergence_counter += 1
                        if self._convergence_counter >= self.cfg.convergence_patience:
                            if self.cfg.fixate:
                                # Fixated: never leave the topic. Plateau
                                # means re-diversify in place and keep going.
                                self._log(
                                    f"Fixate: plateau at "
                                    f"best={self.archive.best_score:.4f} — "
                                    f"re-diversifying population in place")
                                self._convergence_counter = 0
                                self._fixate_rediversify()
                            else:
                                self._log(
                                    f"Converged: no improvement for "
                                    f"{self.cfg.convergence_patience} generations "
                                    f"(best={self.archive.best_score:.4f})")
                                # ── Dynamic domain spawning ──
                                # When a domain converges, spawn a refinement
                                # domain that narrows the search around the
                                # best config. This enables infinite-depth
                                # search.
                                if self.cfg.enable_refinement:
                                    self._spawn_refinement()
                                break

                # ── PHASE 1: GENERATE (batched GPU forward) ──
                t0 = time.time()
                candidates = self.population.generate(
                    n_per_gen=self.cfg.propose_mult,
                    contexts=self._contexts_for_generation(gen))
                n_generated = len(candidates)
                t_gen = time.time() - t0

                # ── PHASE 1.5: DOMAIN-SPACE ELITE MUTANTS ──
                # Some domains (e.g. program synthesis) can mutate elite
                # configs directly in their own representation — a much
                # stronger operator than generator-weight noise when the
                # decode map is rugged. Opt-in via domain.propose_elite_mutants.
                if self.cfg.elite_mutants > 0 and gen > 0:
                    _mut_hook = getattr(self.domain, "propose_elite_mutants", None)
                    if _mut_hook is not None:
                        try:
                            for mc in _mut_hook(self.archive,
                                                self.cfg.elite_mutants, gen) or []:
                                candidates.append({"params": None,
                                                   "template_config": mc,
                                                   "generator_idx": -1})
                        except Exception as e:
                            self._log(f"  elite-mutant hook failed (gen {gen}): {e}")

                # ── PHASE 2: SUBMIT — transfer, dedup, decode, enqueue ──
                # Proposals are generated every gen regardless of eval
                # progress: unique ones land on the pending frontier and are
                # popped for eval later. Submitted rate >> eval rate.
                t0 = time.time()
                neural_cands = [c for c in candidates
                                if c.get("params") is not None]
                template_cands = [c for c in candidates
                                  if c.get("template_config") is not None]

                # Template proposals bypass the frontier (few, full configs).
                # Dedup keys are bare config keys — a template that decodes
                # to a config already queued/evaluated by another emitter
                # would otherwise be evaluated twice (mixed "t"/"c"
                # namespaces let the same config slip past dedup).
                templates_new = []
                for cand in template_cands:
                    key = self._config_key(cand["template_config"])
                    if (key in self._eval_cache
                            or key in self._frontier_keys):
                        continue
                    self._eval_cache[key] = None  # committed — block re-entry
                    templates_new.append(cand)

                n_sub_new = 0
                if neural_cands:
                    # Vectorized dedup: ONE D2H transfer + decode +
                    # config-key for all candidates (~100x faster than
                    # per-candidate .cpu()). Key is the DECODED config —
                    # discrete-heavy domains bin many param vectors into
                    # the same config, so param-level keys let duplicate
                    # configs waste real evals.
                    bp = torch.stack(
                        [c["params"] for c in neural_cands]).detach().cpu()
                    for i, cand in enumerate(neural_cands):
                        config = self.domain.decode(bp[i])
                        key = self._config_key(config)
                        if (key in self._eval_cache
                                or key in self._frontier_keys):
                            continue
                        self._frontier_keys.add(key)
                        self._frontier.append({
                            "key": key,
                            "params": bp[i],
                            "config": config,
                            "generator_idx": cand["generator_idx"],
                            "noise": cand.get("noise"),
                        })
                        n_sub_new += 1

                # Crossover emitter: blend pairs of archive elites into
                # direct proposals (recombination pressure — bypasses the
                # generators entirely, generator_idx=-2 carries no credit).
                n_x = int(len(neural_cands) * self.cfg.crossover_frac)
                if n_x > 0:
                    elites = self.archive.get_all_elites()
                    if len(elites) >= 2:
                        for _ in range(n_x):
                            try:
                                e1, e2 = np.random.choice(
                                    elites, 2, replace=False)
                                p1 = self.domain.encode(e1.config)
                                p2 = self.domain.encode(e2.config)
                                if p1 is None or p2 is None:
                                    continue
                                t = float(np.random.uniform(0.2, 0.8))
                                row = (t * p1 + (1 - t) * p2
                                       + torch.randn_like(p1) * 0.02)
                                row = row.clamp(0, 1).cpu()
                                config = self.domain.decode(row)
                                key = self._config_key(config)
                                if (key in self._eval_cache
                                        or key in self._frontier_keys):
                                    continue
                                self._frontier_keys.add(key)
                                self._frontier.append({
                                    "key": key, "params": row,
                                    "config": config,
                                    "generator_idx": -2, "noise": None})
                                n_sub_new += 1
                            except Exception:
                                continue

                # Coverage emitter: targets least-evaluated discrete bins.
                # Generator output distributions collapse onto a few bins
                # (mech run: d128 had the best mean but only 8/692
                # evals) — this guarantees every bin gets sampled each
                # gen. generator_idx=-3 carries no fitness credit.
                disc = self.domain.discrete_choices()
                if disc and self.archive.get_all_elites():
                    n_cov = min(8, max(2, int(len(neural_cands) * 0.03)))
                    axes = list(disc.keys())
                    for _ in range(n_cov):
                        try:
                            elite = self.archive.get_all_elites()[
                                int(np.random.randint(
                                    len(self.archive.get_all_elites())))]
                            config = dict(elite.config)
                            ax = axes[int(np.random.randint(len(axes)))]
                            counts = self._bin_counts.get(ax, {})
                            least = min(disc[ax],
                                        key=lambda v: counts.get(str(v), 0))
                            config[ax] = least
                            key = self._config_key(config)
                            if (key in self._eval_cache
                                    or key in self._frontier_keys):
                                continue
                            enc = self.domain.encode(config)
                            if enc is None:
                                continue
                            self._frontier_keys.add(key)
                            self._frontier.append({
                                "key": key, "params": enc.cpu(),
                                "config": config,
                                "generator_idx": -3, "noise": None})
                            n_sub_new += 1
                        except Exception:
                            continue

                # Wild emitter: uniform-random param vectors decoded
                # straight to configs — bypasses generators AND the
                # archive. Permanent "random directions" emitter so a
                # fraction of evals always probe off-manifold space.
                n_w = int(len(neural_cands) * self.cfg.wild_frac)
                for _ in range(n_w):
                    try:
                        row = torch.rand(self.domain.output_dim())
                        config = self.domain.decode(row)
                        key = self._config_key(config)
                        if (key in self._eval_cache
                                or key in self._frontier_keys):
                            continue
                        self._frontier_keys.add(key)
                        self._frontier.append({
                            "key": key, "params": row,
                            "config": config,
                            "generator_idx": -4, "noise": None})
                        n_sub_new += 1
                    except Exception:
                        continue

                self._submitted += n_sub_new
                if len(self._frontier) > self.cfg.frontier_cap:
                    self._frontier_evict()
                t_filter = time.time() - t0

                # ── PHASE 3: SELECT + EVAL ──
                t0 = time.time()
                scores = []
                behavioral_list = []
                metadata_list = []

                n_eval = max(self.cfg.min_evaluate,
                             min(self.cfg.max_evaluate,
                                 n_generated // self.cfg.filter_ratio))
                n_neural_eval = max(0, n_eval - len(templates_new))

                # Multi-fidelity: pop quick_ratio× more candidates, run the
                # cheap tier on all, promote the top n_neural_eval to full.
                quick_fn = getattr(self.domain, "quick_eval", None)
                n_pop = n_neural_eval * (self.cfg.quick_ratio
                                         if quick_fn else 1)
                picked = self._frontier_pop(n_pop, gen)

                if quick_fn and len(picked) > n_neural_eval:
                    qres, _ = self._eval_configs(
                        [e["config"] for e in picked], eval_fn=quick_fn)
                    order = np.argsort([r["score"] for r in qres])[::-1]
                    # Minimal-criterion gate (MCNS): unconditioned
                    # emitters (coverage -3, wild -4) must clear the
                    # quick-score floor before spending a full-eval
                    # slot — novelty requires viability. Guided sources
                    # (neural, templates, crossover) are exempt.
                    if self.cfg.mc_floor_q > 0:
                        qs = np.array([r["score"] for r in qres])
                        ok = qs[np.isfinite(qs) & (qs > -1e6)]
                        if len(ok):
                            floor = np.quantile(ok, self.cfg.mc_floor_q)
                            order = [i for i in order
                                     if picked[i].get("generator_idx", -1)
                                     > -3 or qs[i] >= floor] or order
                    # Promote ~70% by quick score, 30% by param-space
                    # distance — pure-score promotion re-collapses onto
                    # near-duplicate configs (same new=0 failure mode).
                    n_top = max(1, int(n_neural_eval * 0.7))
                    top = list(order[:n_top])
                    rem = list(order[n_top:])
                    n_far = min(len(rem), n_neural_eval - n_top)
                    if n_far > 0 and self._evald_params is not None:
                        P = np.stack([picked[i]["params"].detach()
                                      .float().cpu().numpy()
                                      for i in rem])
                        E = self._evald_params
                        d2 = ((P * P).sum(1)[:, None]
                              + (E * E).sum(1)[None, :]
                              - 2 * P @ E.T)
                        far = np.argsort(-np.sqrt(
                            np.maximum(d2.min(1), 0.0)))[:n_far]
                        top += [rem[i] for i in far]
                    else:
                        top += rem[:n_far]
                    picked = [picked[i] for i in top]

                # Template configs: decode via encode-merge fallback.
                # The RESOLVED config differs from template_config, so
                # dedup must commit the resolved key too — otherwise a
                # later proposal decoding to the same resolved config
                # slips past _eval_cache and gets evaluated twice.
                templates_resolved = []
                for cand in templates_new:
                    tc = cand["template_config"]
                    try:
                        config = self.domain.decode(self.domain.encode(tc))
                        config.update(tc)
                    except Exception:
                        try:
                            default_params = torch.full(
                                (self.domain.output_dim(),), 0.5,
                                device=self.domain.device)
                            config = self.domain.decode(default_params)
                            config.update(tc)
                        except Exception:
                            config = tc
                    rkey = self._config_key(config)
                    if rkey in self._eval_cache:
                        continue  # resolved config already committed
                    self._eval_cache[rkey] = None
                    cand["config"] = config
                    templates_resolved.append(cand)

                selected = picked + templates_resolved
                configs = [e["config"] for e in selected]
                # config-key -> generator_idx map (survives eval reordering)
                key2gen = {e["key"]: e.get("generator_idx", -1)
                           for e in picked}
                for cand in templates_resolved:
                    key2gen[self._config_key(cand["config"])] = \
                        cand.get("generator_idx", -1)
                results, configs = self._eval_configs(configs)

                # ── Collect results (no post-eval dedup needed — pre-filtered) ──
                disc_axes = self.domain.discrete_choices() or {}
                for config, result in zip(configs, results):
                    scores.append(result["score"])
                    behavioral_list.append(result.get("behavioral", (0,)))
                    metadata_list.append(result.get("metadata", {}))
                    for ax in disc_axes:
                        if ax in config:
                            bc = self._bin_counts.setdefault(ax, {})
                            k = str(config[ax])
                            bc[k] = bc.get(k, 0) + 1

                    self.all_results.append({
                        "generation": gen,
                        "config": config,
                        "score": result["score"],
                        "behavioral": result.get("behavioral", (0,)),
                        "metadata": result.get("metadata", {}),
                    })

                # Cap results to prevent OOM on long runs
                if len(self.all_results) > self.max_results:
                    self.all_results = self.all_results[-self.max_results:]

                # Behavioral-grid calibration guard: if a large share of
                # evals saturate a dim's declared range, the archive can't
                # differentiate them and coverage is fake (mech run: 65
                # cells because live_gain 0-3 clipped real 5.4-5.7 vals).
                if not getattr(self, "_behav_calib_warned", False):
                    try:
                        dims = self.domain.behavioral_dims()
                        for d_i, (dn, _, lo, hi) in enumerate(dims):
                            vals = [b[d_i] for b in behavioral_list
                                    if len(b) > d_i]
                            if not vals:
                                continue
                            sat = sum(1 for v in vals
                                      if v >= hi or v <= lo) / len(vals)
                            if sat > 0.4:
                                self._log(
                                    f"  [Calib] {sat*100:.0f}% of evals "
                                    f"saturate behavioral '{dn}' "
                                    f"range [{lo},{hi}] — widen the dim "
                                    f"or coverage is fake")
                                self._behav_calib_warned = True
                                break
                    except Exception:
                        pass

                t_score = time.time() - t0

                # Ground-truth set for Δ-ranking + NSLC local quality
                self._record_scored(picked, scores[:len(picked)])

                # ── PHASE 4: TRAIN (GPU updates) ──
                t0 = time.time()

                # Island demotion/migration bookkeeping for this gen
                self._island_migration(gen)

                # Update archive + save new discoveries to DB
                new_discoveries = 0
                new_discovery_list = []
                novelty_list = []
                burst_cands = []
                n_isl = len(self._islands) if self._islands else 0
                for config, score, behavioral, metadata in zip(configs, scores, behavioral_list, metadata_list):
                    # Island archive add — deme of the generator that
                    # emitted it (crossover/templates -> random deme)
                    if n_isl:
                        gidx = key2gen.get(
                            self._config_key(config), -1)
                        isl = (gidx % n_isl) if gidx >= 0 else \
                            int(np.random.randint(n_isl))
                        self._islands[isl].add(config, score,
                                               behavioral, gen)
                    # Novelty search: compute novelty + track behavioral
                    novelty_score = 0.0
                    if self.cfg.enable_novelty:
                        novelty_score = self.novelty.compute_novelty(behavioral)
                        self.novelty.add_behavioral(behavioral)
                    novelty_list.append(novelty_score)

                    # Novelty-aware archive acceptance
                    if self.cfg.enable_novelty:
                        # During novelty phase, accept novel configs even if
                        # not the best in their cell (fills diverse cells)
                        if self.novelty.phase == "novelty" and novelty_score > self.novelty.cfg.min_novelty:
                            added = self.archive.add(config, score, behavioral, gen)
                        else:
                            added = self.archive.add(config, score, behavioral, gen)
                    else:
                        added = self.archive.add(config, score, behavioral, gen)
                    if added:
                        new_discoveries += 1
                        burst_cands.append((score, config))
                        self.discoveries.append({
                            "generation": gen,
                            "config": config,
                            "score": score,
                            "behavioral": behavioral,
                        })
                        new_discovery_list.append({
                            "generation": gen,
                            "config": config,
                            "score": score,
                            "behavioral": behavioral,
                            "metadata": metadata,
                        })

                # Save new discoveries to database (batched — flush every 10 gens)
                if new_discovery_list:
                    self._pending_discoveries.extend(new_discovery_list)
                    if (gen + 1) % 10 == 0 or gen == self.cfg.generations - 1:
                        self.db.save_discoveries(self.run_id, self.domain.name(),
                                                 self._pending_discoveries)
                        self._pending_discoveries = []

                # Winner burst (Go-Explore "return then explore"):
                # configs that just took their archive cell get a ring
                # of local mutations queued for next gen — exploit new
                # winners immediately, before the population drifts on.
                if burst_cands and self.cfg.burst_winners > 0:
                    winners = sorted(burst_cands, key=lambda w: -w[0])[
                        :self.cfg.burst_winners]
                    for _, w_cfg in winners:
                        try:
                            base = self.domain.encode(w_cfg)
                            if base is None:
                                continue
                            base = base.detach().float().cpu()
                            for _ in range(self.cfg.burst_k):
                                mut = (base + torch.randn_like(base)
                                       * 0.05).clamp(0, 1)
                                cfg_m = self.domain.decode(mut)
                                key = self._config_key(cfg_m)
                                if (key in self._eval_cache
                                        or key in self._frontier_keys):
                                    continue
                                self._frontier_keys.add(key)
                                self._frontier.append({
                                    "key": key, "params": mut,
                                    "config": cfg_m,
                                    "generator_idx": -5,
                                    "noise": None})
                        except Exception:
                            continue

                # Update surrogate (on the proposals actually evaluated)
                if picked:
                    eval_params = torch.stack(
                        [e["params"] for e in picked]).detach()
                    eval_scores = torch.tensor(
                        scores[:len(picked)], dtype=torch.float32
                    )
                    try:
                        self.surrogate.train(eval_params, eval_scores)
                    except Exception as e:
                        self._log(f"Surrogate training skipped (gen {gen}): {e}")
                    self.surrogate.generation = gen

                # Update generators (REINFORCE) — pass context.
                # Rank-shape scores first: raw scores carry -1e9 failure
                # spikes and domain-dependent scale; centered ranks make
                # updates monotone-invariant and outlier-robust
                # (OpenAI-ES fitness shaping). Archive/surrogate keep raw.
                shaped = np.zeros(len(scores), dtype=np.float64)
                if len(scores) > 1:
                    shaped[np.argsort(scores)] = (
                        np.arange(len(scores)) / (len(scores) - 1) - 0.5)
                shaped_scores = shaped.tolist()
                gen_indices = [c.get("generator_idx", -1) for c in selected]

                # Explore objective (NSLC): novelty blended with LOCAL
                # competition — each candidate's score ranked against
                # its k-NN evaluated neighbors, not the global batch.
                # Regional champions get fitness even when globally
                # mediocre (a bacterium doesn't compete with a bear).
                local_q = self._local_quality(selected, scores)
                explore_raw = [0.5 * n + 0.5 * q
                               for n, q in zip(novelty_list, local_q)]
                ex_shaped = np.zeros(len(explore_raw), dtype=np.float64)
                if len(explore_raw) > 1:
                    ex_shaped[np.argsort(explore_raw)] = (
                        np.arange(len(explore_raw))
                        / (len(explore_raw) - 1) - 0.5)
                explore_fit = ex_shaped.tolist()

                # Fitness routing by emitter role (PGA-ME): wild
                # generators are pure explorers — REINFORCE and fitness
                # come from the explore objective, never raw score,
                # or they'd be trained INTO exploiters.
                train_scores = [
                    ef if gi in self._wild_gens else sc
                    for sc, ef, gi in zip(shaped_scores, explore_fit,
                                          gen_indices)]
                self.trainer.update(selected, train_scores,
                                    self.population.context)

                mutated = self.population.evolve(
                    shaped_scores, gen_indices,
                    novelties=explore_fit,
                    explore_frac=self.cfg.explore_frac,
                    extra_explore=self._wild_gens)

                # Notify trainer of mutations so it can reset optimizer state
                for idx in mutated:
                    self.trainer.notify_mutation(idx)

                # Update context: UCB-selected parent (not just best)
                # 70% UCB selection, 30% best (exploit known good)
                if np.random.random() < 0.7:
                    parent = self.archive.sample_elite_ucb(current_gen=gen, c=1.0)
                else:
                    parent = self.archive.best_entry
                if parent is None:
                    parent = self.archive.best_entry
                if parent:
                    parent_params = self.domain.encode(parent.config)
                    if parent_params is not None:
                        self.population.update_context(parent_params)

                t_train = time.time() - t0

                # Incremental generator/surrogate checkpoint every 10 gens
                if (gen + 1) % 10 == 0:
                    self.db.save_generators(self.run_id, self.domain.name(),
                                            self.population.batched_gen)
                    self.db.save_surrogate(self.run_id, self.domain.name(), self.surrogate)

                # ── LOG ──
                # Always log gen 0, final gen, and every log_every gens
                is_last_gen = (gen == self.cfg.generations - 1)
                if gen == 0 or gen % self.cfg.log_every == 0 or is_last_gen:
                    # Track phase timings for GPU utilization analysis
                    phase_info = (f"gen={t_gen:.3f}s filter={t_filter:.3f}s "
                                  f"score={t_score:.2f}s train={t_train:.3f}s")
                    self._log(
                        f"{phase_info} | "
                        f"evaluated={len(selected)}/{n_generated} "
                        f"submitted={self._submitted} "
                        f"frontier={len(self._frontier)} | "
                        f"new={new_discoveries} | "
                        f"{self.archive.summary()}"
                    )
                    # Detect if score phase dominates (good = GPU busy)
                    # vs gen/filter phase dominates (bad = GPU idle on Python)
                    total_t = t_gen + t_filter + t_score + t_train
                    if total_t > 0 and t_score / total_t < 0.3 and gen > 2:
                        # Score phase < 30% of total = GPU underutilized.
                        # Increase eval batch to keep GPU busy longer.
                        old_max = self.cfg.max_evaluate
                        self.cfg.max_evaluate = min(old_max * 2, 200)
                        if self.cfg.max_evaluate != old_max:
                            self._log(f"  [GPUKeepBusy] max_eval {old_max}→{self.cfg.max_evaluate} "
                                     f"(score phase only {t_score/total_t*100:.0f}% of time)")

                # ── ADAPTIVE POPULATION SIZING ──
                # Grow population when plateauing (more explorers needed),
                # shrink when steadily improving (efficiency gain).
                if self.cfg.adaptive_population and gen > 0:
                    if self.archive.best_score > self._prev_best_score + 1e-6:
                        self._pop_stable_count += 1
                        if self._pop_stable_count >= 3:
                            # Steady improvement → shrink (save compute)
                            new_n = max(self.cfg.pop_min,
                                        int(self.cfg.n_generators * (1 - self.cfg.pop_shrink_rate)))
                            if new_n != self.cfg.n_generators:
                                self.cfg.n_generators = new_n
                                self._log(f"  [AdaptivePop] shrinking to {new_n} (steady improvement)")
                            self._pop_stable_count = 0
                    else:
                        self._pop_plateau_count += 1
                        if self._pop_plateau_count >= 2:
                            # Plateau → grow (need more exploration)
                            new_n = min(self.cfg.pop_max,
                                        int(self.cfg.n_generators * (1 + self.cfg.pop_growth_rate)))
                            if new_n != self.cfg.n_generators:
                                self.cfg.n_generators = new_n
                                self._log(f"  [AdaptivePop] growing to {new_n} (plateau detected)")
                            self._pop_plateau_count = 0

                # ── COMPUTE SPLIT: gen model lifecycle (grow/shrink/fine-tune) ──
                if self.cfg.enable_compute_split and self._gen_mgr is not None:
                    # Record this round's performance
                    best_score = self.archive.best_score
                    gm = getattr(self._gen_mgr, "_model", None)
                    gm_size = None
                    if gm is not None:
                        if hasattr(gm, "get_size_config"):
                            gm_size = gm.get_size_config()
                        elif hasattr(gm, "param_count"):
                            gm_size = {"params": gm.param_count()}
                    self._gen_mgr.record_round(self.domain.name(), best_score, gm_size)

                    # Grow if plateauing
                    if self._gen_mgr.should_grow():
                        self._log(f"  [GenModel] growing (plateau {self._gen_mgr._plateau_count} rounds)")
                        self._gen_mgr.grow()
                        if gm is not None and hasattr(self.domain, "set_gen_model"):
                            self.domain.set_gen_model(self._gen_mgr._model)
                    elif self._gen_mgr.should_shrink():
                        self._log("  [GenModel] shrinking (overperforming)")
                        self._gen_mgr.shrink()
                        if hasattr(self.domain, "set_gen_model"):
                            self.domain.set_gen_model(self._gen_mgr._model)

                    # Fine-tune on successful solutions every 5 rounds
                    if (gen + 1) % 5 == 0 and self._gen_mgr is not None:
                        solutions = []
                        for d in self.discoveries[-20:]:
                            meta = d.get("metadata", {})
                            problem = meta.get("problem", "")
                            answer = meta.get("model_answer", "")
                            if problem and answer:
                                solutions.append({"prompt": problem, "output": answer})
                        if solutions and self._gen_mgr is not None:
                            self._log(f"  [GenModel] fine-tuning on {len(solutions)} solutions")
                            try:
                                self._gen_mgr.fine_tune_on_solutions(solutions)
                            except Exception as e:
                                self._log(f"  [GenModel] fine-tune failed: {e}")

            # Final summary
            elapsed = time.time() - self.start_time
            self._log(f"Done: {self.generation+1} generations, "
                      f"{len(self.all_results)} evaluations, "
                      f"{len(self.discoveries)} discoveries, "
                      f"{elapsed:.1f}s total")

            # Save final state to database
            # Flush any remaining pending discoveries
            if self._pending_discoveries:
                n_saved, n_deduped = self.db.save_discoveries(
                    self.run_id, self.domain.name(),
                    self._pending_discoveries)
                if n_deduped > 0:
                    self._log(f"DB dedup: saved {n_saved}, skipped {n_deduped} duplicates")
                self._pending_discoveries = []

            # Build final results dict (includes refinement domain info)
            results = {
                "generations": self.generation + 1,
                "total_evaluations": len(self.all_results),
                "discoveries": len(self.discoveries),
                "best_score": self.archive.best_score,
                "best_config": self.archive.best_entry.config if self.archive.best_entry else None,
                "device": str(self.device),
                "pending_refinements": [d.name() for d in self.pending_refinements],
                "spawned_domains": self.domain_factory.get_spawned_names(),
            }
            self.db.save_run(self.run_id, self.domain.name(),
                             {"n_generators": self.cfg.n_generators,
                              "filter_ratio": self.cfg.filter_ratio,
                              "generations": self.cfg.generations},
                             results, self.start_time)
            self.db.save_generators(self.run_id, self.domain.name(),
                                    self.population.batched_gen)
            self.db.save_surrogate(self.run_id, self.domain.name(), self.surrogate)

            # ── Canonical save: update permanent knowledge ONLY if this run
            # produced new discoveries. If the run found nothing new (all
            # configs were dupes or archive didn't improve), skip the save
            # to avoid overwriting canonical weights with identical ones.
            # Skip canonical save for Generic domains (heuristic evaluator,
            # not real performance data — would pollute cross-domain transfer)
            is_generic = "Generic_" in self.domain.name()
            best = self.archive.best_score
            n_new = len(self.discoveries)
            canonical_best = self.db.get_canonical_best_score(self.domain.name())
            if best > float('-inf') and n_new > 0 and not is_generic:
                # Only save if we actually found new discoveries AND beat
                # the canonical best (the DB method checks the score)
                updated_gen = self.db.save_canonical_generators(
                    self.domain.name(), self.population.batched_gen,
                    best, self.run_id)
                updated_surr = self.db.save_canonical_surrogate(
                    self.domain.name(), self.surrogate, best, self.run_id)
                if updated_gen or updated_surr:
                    self._log(f"Updated canonical knowledge for '{self.domain.name()}' "
                              f"(best={best:.4f}) — future runs will start from here")
            elif best > float('-inf') and n_new == 0:
                canon_str = f"{canonical_best:.4f}" if canonical_best is not None else "none"
                self._log(f"No new discoveries for '{self.domain.name()}' — "
                          f"canonical unchanged (best={best:.4f}, "
                          f"canonical={canon_str})")

            self._log(f"Saved to database: {self.cfg.db_path} (run_id={self.run_id})")

            results = {
                "generations": self.generation + 1,
                "total_evaluations": len(self.all_results),
                "discoveries": len(self.discoveries),
                "archive_coverage": self.archive.coverage(),
                "best_score": self.archive.best_score,
                "best_config": self.archive.best_entry.config if self.archive.best_entry else None,
                "pareto_front": [
                    {"config": e.config, "score": e.score, "behavioral": e.behavioral,
                     "generation": e.generation, "metadata": e.metadata}
                    for e in self.archive.get_pareto_front()
                ],
                "time_s": elapsed,
                "all_results": self.all_results,
                "discoveries_list": self.discoveries,
                "archive": self.archive,
                "device": str(self.device),
                "run_id": self.run_id,
                "warm_started": self._warm_started,
                "pending_refinements": [d.name() for d in self.pending_refinements],
                "spawned_domains": self.domain_factory.get_spawned_names(),
            }
        finally:
            # Clean up CPU worker pool
            if self.cpu_pool is not None:
                self.cpu_pool.close()
                self.cpu_pool.join()
                self.cpu_pool = None
            # Clean up GPU worker pool
            if self.gpu_pool is not None:
                self.gpu_pool.close()
                self.gpu_pool.join()
                self.gpu_pool = None
            # Clean up checker thread pool
            if self._checker_pool is not None:
                self._checker_pool.shutdown(wait=False, cancel_futures=True)
                self._checker_pool = None

        return results

    def guide(self, direction: str = "explore",
              seed_configs: list[dict] | None = None,
              constraints: dict | None = None,
              time_budget: str | float | None = None):
        """Steer the search mid-run (applies on next generation)."""
        if direction == "explore":
            self.cfg.exploration = 0.8
            self.cfg.mutation_rate = 0.3
        elif direction == "exploit":
            self.cfg.exploration = 0.1
            self.cfg.mutation_rate = 0.02

        if seed_configs:
            for config in seed_configs:
                result = self.domain.evaluate(config)
                self.archive.add(config, result["score"],
                                 result.get("behavioral", (0,)), self.generation)

        if time_budget:
            if isinstance(time_budget, str):
                import re
                m = re.match(r"(\d+(?:\.\d+)?)\s*(s|m|h)", time_budget)
                if m:
                    val, unit = float(m.group(1)), m.group(2)
                    self.cfg.time_budget_s = val * {"s": 1, "m": 60, "h": 3600}[unit]
            else:
                self.cfg.time_budget_s = time_budget
