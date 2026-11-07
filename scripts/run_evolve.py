"""ForgeEvolve runner — single entry point for all evolution runs.

Driven by JSON configs in forge/evolution/configs/.
Replaces boot_evolve.py, deep_evolve.py, test_multi_domain.py, test_long.py.

Usage:
    # Boot run (short, all domains)
    python run_evolve.py --profile boot

    # Deep run (long, all domains)
    python run_evolve.py --profile deep

    # Specific category
    python run_evolve.py --profile deep --category quantization

    # Specific domains
    python run_evolve.py --profile deep --domains QuantDomain,AaacQuant

    # Custom overrides
    python run_evolve.py --profile deep --gens 200 --gen-pop 1000

    # Smoke test
    python run_evolve.py --profile smoke --domains SyntheticDomain
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import argparse
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch

from forge.evolution import ForgeEvolve, ForgeEvolveConfig
from forge.evolution.domains import DOMAINS, list_domains
from forge.evolution.llm_domain_gen import LLMDomainGenerator
from forge.evolution.revisit_scheduler import DomainRevisitScheduler

# Pre-load simulators before any threading to avoid race conditions
from forge.evolution.simulators import _ensure_loaded
from forge.evolution.topic_scanner import TopicScanner

_ensure_loaded()

CONFIG_DIR = Path(__file__).resolve().parents[2] / "forge" / "evolution" / "configs"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "research" / "results"


# ---------------------------------------------------------------------------
# GPU utilization monitor — samples GPU util every 0.5s for adaptive parallelism
# ---------------------------------------------------------------------------
class GPUMonitor:
    """Background thread that samples GPU utilization via NVML.

    Used by the adaptive parallelism controller to increase/decrease
    the number of concurrent domains to keep GPU at target utilization.
    """

    def __init__(self, target_util: float = 0.85, sample_interval: float = 0.5):
        self.target_util = target_util
        self.sample_interval = sample_interval
        self._samples = []
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._nvml_ok = False
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml

                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._pynvml = pynvml
                self._nvml_ok = True
        except Exception:
            self._nvml_ok = False

    def start(self):
        if not self._nvml_ok or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _sample_loop(self):
        while self._running:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                with self._lock:
                    self._samples.append(util.gpu)
                    if len(self._samples) > 60:  # keep last 30s
                        self._samples = self._samples[-60:]
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def get_avg_util(self) -> float:
        """Return average GPU utilization (0-100) over recent samples."""
        with self._lock:
            if not self._samples:
                return 0.0
            return float(np.mean(self._samples[-20:]))  # last 10s

    def get_vram_usage(self) -> float:
        """Return VRAM usage fraction (0-1)."""
        if not self._nvml_ok:
            return 0.0
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return info.used / info.total
        except Exception:
            return 0.0

    def recommend_parallelism(self, current_parallel: int, max_parallel: int = 12, min_parallel: int = 2) -> int:
        """Recommend new parallelism level based on GPU util + VRAM.

        If GPU < 50%: increase parallelism (more domains needed)
        If GPU > 95% or VRAM > 85%: decrease parallelism (too much contention/OOM risk)
        Otherwise: keep current
        """
        util = self.get_avg_util()
        vram = self.get_vram_usage()
        if util > 95 or vram > 0.85:
            # GPU overloaded or VRAM pressure — reduce concurrency
            new_p = max(current_parallel - 2, min_parallel)
        elif util < self.target_util * 100 * 0.6 and vram < 0.75:
            # GPU starved + VRAM available — add more concurrent domains
            new_p = min(current_parallel + 2, max_parallel)
        elif vram > 0.70:
            # VRAM getting tight — don't increase, maybe decrease
            new_p = max(current_parallel - 1, min_parallel)
        else:
            new_p = current_parallel
        return new_p


DEFAULT_FOCUS_FILE = CONFIG_DIR / "default_focus.txt"


def load_json(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return json.load(f)


def resolve_domains(category: str | None, domain_list: str | None, focus: str | None = None) -> list[str]:
    """Resolve which domains to run from category, explicit list, or focus profile.

    Priority: domain_list > focus > category > all
    """
    if domain_list:
        names = [d.strip() for d in domain_list.split(",")]
        for n in names:
            if n not in DOMAINS:
                raise KeyError(f"Unknown domain '{n}'. Available: {list_domains()}")
        return names

    if focus:
        return resolve_focus_domains(focus)

    if category is None or category == "all":
        return list_domains()

    cats = load_json("domain_categories.json")["categories"]
    if category not in cats:
        raise KeyError(f"Unknown category '{category}'. Available: {list(cats.keys())}")
    domains = cats[category]["domains"]
    return [d for d in domains if d in DOMAINS]


def resolve_focus_domains(focus: str) -> list[str]:
    """Resolve domains from a focus profile (memory/speed/quality/training/all)."""
    profiles = load_json("focus_profiles.json")["focus_profiles"]
    if focus not in profiles:
        raise KeyError(f"Unknown focus '{focus}'. Available: {list(profiles.keys())}")
    fp = profiles[focus]

    # Start with all domains, then filter
    if focus == "all":
        return list_domains()

    cats = load_json("domain_categories.json")["categories"]
    domains = set()

    # Add domains from specified categories
    for cat_name in fp.get("categories", []):
        if cat_name in cats:
            for d in cats[cat_name]["domains"]:
                if d in DOMAINS:
                    domains.add(d)

    # Add extra domains
    for d in fp.get("extra_domains", []):
        if d in DOMAINS:
            domains.add(d)

    # Remove excluded domains
    for d in fp.get("exclude_domains", []):
        domains.discard(d)

    return sorted(domains)


def get_focus_weights(focus: str | None) -> dict:
    """Get score weights for a focus profile (empty dict if no focus)."""
    if not focus or focus == "all":
        return {}
    profiles = load_json("focus_profiles.json")["focus_profiles"]
    if focus not in profiles:
        return {}
    return profiles[focus].get("score_weights", {})


def build_config(
    profile: str, overrides: dict, domain_name: str, db_path: str, domain_instance=None
) -> ForgeEvolveConfig:
    """Build ForgeEvolveConfig from a JSON profile + CLI overrides.

    If domain_instance is provided, use it directly (for refinement domains).
    """
    profiles = load_json("run_profiles.json")
    if profile not in profiles["profiles"]:
        raise KeyError(f"Unknown profile '{profile}'. Available: {list(profiles['profiles'].keys())}")

    p = profiles["profiles"][profile]
    defaults = profiles.get("defaults", {})

    cfg_kwargs = {
        "domain": domain_instance if domain_instance is not None else DOMAINS[domain_name](),
        "n_generators": overrides.get("gen_pop", p["n_generators"]),
        "filter_ratio": p["filter_ratio"],
        "min_evaluate": p["min_evaluate"],
        "max_evaluate": overrides.get("max_eval", p["max_evaluate"]),
        "generations": overrides.get("gens", p["generations"]),
        "exploration": p["exploration"],
        "parallel_eval": defaults.get("parallel_eval", 1),
        "db_path": db_path,
        "run_id": f"{domain_name}_{profile}",
        "warm_start": p["warm_start"],
        "verbose": defaults.get("verbose", False),
        "log_every": p["log_every"],
        "time_budget_s": overrides.get("time_budget_s", p.get("time_budget_s", 0)),
        "convergence_patience": p.get("convergence_patience", 0),
        "enable_compute_split": overrides.get("enable_compute_split", False),
        "fixate": overrides.get("fixate", p.get("fixate", False)),
    }
    return ForgeEvolveConfig(**cfg_kwargs)


def run_one_domain(name: str, profile: str, overrides: dict, db_path: str, domain_instance=None) -> tuple:
    """Run ForgeEvolve on one domain. Returns (results, time_s, error).

    If domain_instance is provided (a pre-instantiated domain, e.g. a
    refinement domain), use it instead of looking up DOMAINS[name].
    """
    try:
        if domain_instance is not None:
            cfg = build_config(profile, overrides, name, db_path, domain_instance=domain_instance)
        else:
            cfg = build_config(profile, overrides, name, db_path)
    except Exception as e:
        return None, 0, str(e)

    t0 = time.time()
    try:
        engine = ForgeEvolve(cfg)
        results = engine.run()
        elapsed = time.time() - t0
        torch.cuda.empty_cache()
        return results, elapsed, None
    except Exception as e:
        torch.cuda.empty_cache()
        return None, time.time() - t0, str(e)[:200]


def extract_top_discoveries(db_path: str, top_n: int = 20) -> dict:
    """Extract top discoveries per domain from the DB."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT domain, config_json, score, behavioral_json, metadata_json,
               run_id, generation
        FROM discoveries WHERE score IS NOT NULL
        ORDER BY domain, score DESC
    """)
    by_domain = {}
    for row in c.fetchall():
        domain = row[0]
        if domain not in by_domain:
            by_domain[domain] = []
        try:
            by_domain[domain].append(
                {
                    "config": json.loads(row[1]),
                    "score": row[2],
                    "behavioral": json.loads(row[3]) if row[3] else [],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "run_id": row[5],
                    "generation": row[6],
                }
            )
        except (json.JSONDecodeError, TypeError):
            continue
    conn.close()
    return {d: sorted(items, key=lambda x: x["score"], reverse=True)[:top_n] for d, items in by_domain.items()}


def generate_ideas_report(top: dict, profile: str) -> str:
    """Generate markdown report of best optimization ideas."""
    L = [
        f"# ForgeEvolve '{profile}' Run: Top Optimization Ideas\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Domains: {len(top)}\n\n",
    ]

    # Tier 1: highest scores overall
    all_items = [(d, i) for d, items in top.items() for i in items]
    all_items.sort(key=lambda x: x[1]["score"], reverse=True)

    L.append("## Tier 1: Top 50 Configurations\n\n")
    L.append("| Rank | Domain | Score | Config |\n|------|--------|-------|--------|\n")
    for idx, (domain, item) in enumerate(all_items[:50]):
        cfg = json.dumps(item["config"], default=str)
        if len(cfg) > 80:
            cfg = cfg[:77] + "..."
        L.append(f"| {idx + 1} | {domain} | {item['score']:.2f} | `{cfg}` |\n")

    # Tier 2: best per domain
    L.append("\n## Tier 2: Best Per Domain\n\n")
    for domain in sorted(top.keys()):
        items = top[domain]
        if not items:
            continue
        best = items[0]
        L.append(f"### {domain} (score={best['score']:.2f})\n")
        L.append(f"- **Config**: `{json.dumps(best['config'], default=str)}`\n")
        L.append(f"- **Metadata**: `{json.dumps(best['metadata'], default=str)}`\n\n")

    # Tier 3: cross-domain patterns
    L.append("## Tier 3: Cross-Domain Parameter Patterns\n\n")
    key_patterns = {}
    for domain, items in top.items():
        for item in items[:3]:
            for key, val in item["config"].items():
                key_patterns.setdefault(key, []).append((domain, val, item["score"]))
    for key in sorted(key_patterns.keys()):
        entries = key_patterns[key]
        if len(entries) < 3:
            continue
        L.append(f"### `{key}`\n")
        val_counts = {}
        for d, v, s in entries:
            vs = str(v)
            vc = val_counts.setdefault(vs, {"n": 0, "scores": [], "domains": set()})
            vc["n"] += 1
            vc["scores"].append(s)
            vc["domains"].add(d)
        for val, info in sorted(val_counts.items(), key=lambda x: -x[1]["n"])[:5]:
            L.append(
                f"- `{val}`: {info['n']}x, avg={np.mean(info['scores']):.1f}, "
                f"domains={','.join(sorted(info['domains'])[:5])}\n"
            )
        L.append("")

    return "".join(L)


def print_summary(all_results: list, total_t: float, profile: str):
    """Print summary table."""
    valid = [(n, r, t, e) for n, r, t, e in all_results if r is not None]
    failed = [(n, r, t, e) for n, r, t, e in all_results if r is None]

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {len(all_results)} domains in {total_t:.1f}s ({total_t / 60:.1f}m) [{profile}]")
    print(f"{'=' * 70}")
    print(f"  {'Domain':<30s} {'Best Score':>12s} {'Disc':>5s} {'Archive':>8s} {'Time':>7s}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 5} {'-' * 8} {'-' * 7}")

    for name, results, t, _ in sorted(valid, key=lambda x: x[1]["best_score"], reverse=True):
        print(
            f"  {name:<30s} {results['best_score']:12.4f} "
            f"{results['discoveries']:5d} {results['archive_coverage'] * 100:7.0f}% {t:6.1f}s"
        )

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for name, _, _t, err in failed:
            print(f"    {name}: {err[:80]}")

    print(f"\n  Valid: {len(valid)}/{len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description="ForgeEvolve runner")
    parser.add_argument("--profile", default="boot", help="Run profile from config (boot/deep/ultra/smoke)")
    parser.add_argument("--category", default=None, help="Domain category from config")
    parser.add_argument(
        "--focus",
        default=None,
        help="Focus profile: memory (minimize param/VRAM), speed, quality, training, all. "
        "Overrides category. Use --list-focus to see options. "
        "Set persistent default with --set-focus.",
    )
    parser.add_argument(
        "--set-focus",
        default=None,
        help="Set persistent default focus profile (saved to configs/default_focus.txt). "
        "Use 'none' to clear. Example: --set-focus memory",
    )
    parser.add_argument("--domains", default=None, help="Comma-separated domain names (overrides focus+category)")
    parser.add_argument("--list-focus", action="store_true", help="List available focus profiles and exit")
    parser.add_argument("--gens", type=int, default=None, help="Override generations")
    parser.add_argument("--gen-pop", type=int, default=None, help="Override generator population")
    parser.add_argument("--no-ideas", action="store_true", help="Skip ideas report generation")
    parser.add_argument(
        "--parallel", type=int, default=4, help="Number of domains to run concurrently on GPU (default: 4)"
    )
    parser.add_argument(
        "--adaptive-parallel",
        action="store_true",
        default=True,
        help="Adaptively tune parallelism based on GPU utilization (default: on). "
        "Monitors GPU via NVML and adjusts concurrent domain count to hit target.",
    )
    parser.add_argument(
        "--no-adaptive-parallel",
        dest="adaptive_parallel",
        action="store_false",
        help="Disable adaptive parallelism (use fixed --parallel count)",
    )
    parser.add_argument(
        "--gpu-target",
        type=float,
        default=0.85,
        help="Target GPU utilization (0-1) for adaptive parallelism (default: 0.85 = 85%%)",
    )
    parser.add_argument(
        "--min-parallel", type=int, default=2, help="Minimum concurrent domains for adaptive parallelism (default: 2)"
    )
    parser.add_argument(
        "--max-parallel", type=int, default=10, help="Maximum concurrent domains for adaptive parallelism (default: 10)"
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Number of consecutive loops (each builds on canonical knowledge from the last)",
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        default=True,
        help="Auto-discover new optimization topics + generate domains (default: on)",
    )
    parser.add_argument(
        "--no-auto-discover",
        dest="auto_discover",
        action="store_false",
        help="Disable auto-discovery of new optimization topics",
    )
    parser.add_argument(
        "--revisit-stale",
        action="store_true",
        default=True,
        help="Re-run stale domains whose code may have changed (default: on)",
    )
    parser.add_argument(
        "--no-revisit", dest="revisit_stale", action="store_false", help="Disable stale domain revisiting"
    )
    parser.add_argument(
        "--max-new-domains", type=int, default=10, help="Max new domains to auto-generate per run (default: 10)"
    )
    parser.add_argument(
        "--max-gpu",
        action="store_true",
        default=False,
        help="Maximize GPU utilization: removes time budgets, increases eval batch sizes, "
        "sets high gen-pop. Equivalent to --gpu-target 0.95 --gen-pop 1000 "
        "--no-time-budget --max-eval 200",
    )
    parser.add_argument(
        "--compute-split",
        action="store_true",
        default=False,
        help="Enable compute split: foreground gen model + background checker. "
        "Also enables train-first on unapplied DB discoveries.",
    )
    parser.add_argument(
        "--fixate",
        action="store_true",
        default=False,
        help="Fixate on ONE topic: run only the first resolved domain "
        "(use --domains or --focus to pick it), disables "
        "auto-discover/revisit/parallel waves, and pins the "
        "engine: no cross-domain transfer, no refinement "
        "spawning, plateaus re-diversify in place.",
    )
    args = parser.parse_args()

    # --set-focus: save persistent default and exit
    if args.set_focus is not None:
        if args.set_focus.lower() == "none":
            DEFAULT_FOCUS_FILE.unlink(missing_ok=True)
            print("Cleared default focus profile.")
        else:
            profiles = load_json("focus_profiles.json")["focus_profiles"]
            if args.set_focus not in profiles:
                print(f"ERROR: Unknown focus '{args.set_focus}'. Available: {list(profiles.keys())}")
                sys.exit(1)
            DEFAULT_FOCUS_FILE.write_text(args.set_focus)
            print(f"Set default focus profile to '{args.set_focus}'.")
            print(f"Saved to: {DEFAULT_FOCUS_FILE}")
            print("Future runs will use this focus unless --focus or --domains overrides.")
        sys.exit(0)

    # --list-focus: print available focus profiles and exit
    if args.list_focus:
        profiles = load_json("focus_profiles.json")["focus_profiles"]
        print("Available focus profiles:")
        for name, fp in profiles.items():
            domains = resolve_focus_domains(name) if name != "all" else list_domains()
            print(f"\n  {name:12s} ({len(domains)} domains)")
            print(f"    {fp['description']}")
            if name != "all":
                print(f"    Domains: {', '.join(domains[:8])}{'...' if len(domains) > 8 else ''}")
        # Show current default
        if DEFAULT_FOCUS_FILE.exists():
            print(f"\n  Current default: {DEFAULT_FOCUS_FILE.read_text().strip()}")
        else:
            print("\n  No default set (use --set-focus to set one)")
        sys.exit(0)

    # Load persistent default focus if --focus not specified
    effective_focus = args.focus
    if effective_focus is None and DEFAULT_FOCUS_FILE.exists():
        effective_focus = DEFAULT_FOCUS_FILE.read_text().strip()
        if effective_focus:
            print(f"  [Focus] Using default focus: '{effective_focus}' (from configs/default_focus.txt)")

    names = resolve_domains(args.category, args.domains, focus=effective_focus)
    if args.fixate:
        if len(names) > 1:
            print(
                f"  [Fixate] {len(names)} domains resolved — pinning to "
                f"'{names[0]}' only (narrow with --domains/--focus)"
            )
        names = names[:1]
        args.auto_discover = False
        args.revisit_stale = False
        args.parallel = 1
        args.adaptive_parallel = False
        overrides_pre = {"fixate": True}
    else:
        overrides_pre = {}
    get_focus_weights(effective_focus)
    profiles = load_json("run_profiles.json")
    db_suffix = profiles["profiles"][args.profile]["db_suffix"]
    db_path = str(RESULTS_DIR / f"forge_evolve{db_suffix}.db")

    overrides = dict(overrides_pre)
    if args.gens is not None:
        overrides["gens"] = args.gens
    if args.gen_pop is not None:
        overrides["gen_pop"] = args.gen_pop

    # --max-gpu: override settings to maximize GPU utilization
    if args.max_gpu:
        if args.gen_pop is None:
            overrides["gen_pop"] = 1000  # more generators = more GPU work
        if args.gpu_target == 0.85:  # only override if user didn't set it
            args.gpu_target = 0.95
        if args.max_parallel == 10:
            args.max_parallel = 12
        overrides["max_eval"] = 200  # bigger eval batches = more GPU kernels
        overrides["time_budget_s"] = 0  # no time limit — let domains run fully
        print(
            f"  [MaxGPU] Overrides: gen_pop={overrides['gen_pop']}, "
            f"max_eval=200, time_budget=0, gpu_target={args.gpu_target * 100:.0f}%"
        )

    # --compute-split: enable gen model + checker compute split
    if args.compute_split:
        overrides["enable_compute_split"] = True
        print("  [ComputeSplit] Enabled — gen model + checker + train-first on unapplied DB entries")

    # ── Auto-discovery: scan codebase for new optimization topics ──
    project_root = str(Path(__file__).resolve().parents[2])
    revisit_scheduler = DomainRevisitScheduler(project_root=project_root)
    if args.auto_discover:
        print("\n  [AutoDiscover] Scanning codebase for optimization topics...")
        scanner = TopicScanner(project_root)
        scanner.scan_all()
        summary = scanner.get_summary()
        print(
            f"  [AutoDiscover] Found {summary['total_topics']} topics "
            f"({summary['covered']} covered, {summary['uncovered']} uncovered)"
        )
        uncovered = scanner.get_uncovered_topics()
        # Skip domain generation if we already have enough domains registered
        # (avoids slow LLM generation on repeat boots). Only generate if we
        # have fewer than max_new_domains Generic_* domains already registered.
        existing_generic = sum(1 for d in DOMAINS if d.startswith("Generic_"))
        if uncovered and existing_generic < args.max_new_domains:
            remaining = args.max_new_domains - existing_generic
            print(
                f"  [AutoDiscover] {len(uncovered)} uncovered topics — "
                f"generating {remaining} domains (skip {existing_generic} existing)..."
            )
            # Generate domains for uncovered topics (GenericDomain fallback)
            gen = LLMDomainGenerator(max_domains_per_run=remaining)
            generated = gen.generate_for_uncovered_topics(uncovered)
            registered = gen.register_domains(generated)
            if registered:
                print(
                    f"  [AutoDiscover] Registered {len(registered)} new domains: "
                    f"{', '.join(registered[:5])}{'...' if len(registered) > 5 else ''}"
                )
                # Add new domains to the run list
                names.extend(registered)
        print()

    print(f"{'=' * 70}")
    print(
        f"  ForgeEvolve [{args.profile}] — {len(names)} domains" + (f" x {args.loops} loops" if args.loops > 1 else "")
    )
    if effective_focus:
        print(f"  Focus: {effective_focus}")
    print(f"  DB: {db_path}")
    print(f"  Warm start: {profiles['profiles'][args.profile]['warm_start']}")
    if args.adaptive_parallel:
        print(
            f"  Parallel: adaptive (target GPU={args.gpu_target * 100:.0f}%, "
            f"range={args.min_parallel}-{args.max_parallel})"
        )
    else:
        print(f"  Parallel: fixed ({args.parallel})")
    print(f"{'=' * 70}\n")

    grand_start = time.time()
    loop_summaries = []

    for loop in range(args.loops):
        if args.loops > 1:
            print(f"\n{'-' * 70}")
            print(f"  LOOP {loop + 1}/{args.loops}")
            print(f"{'-' * 70}\n")

        all_results = []
        t_start = time.time()

        if args.parallel > 1 and len(names) > 1:
            # ── Concurrent domain execution with adaptive parallelism ──
            # Each domain uses ~100MB VRAM; with 12GB we can run 4-8 concurrently.
            # ThreadPoolExecutor works because CUDA calls release the GIL.
            # Adaptive mode: GPU monitor adjusts concurrency to hit target util.

            def _run_domain(idx_name):
                idx, name = idx_name
                time.time()
                results, t, err = run_one_domain(name, args.profile, overrides, db_path)
                return idx, name, results, t, err

            # Start GPU monitor for adaptive parallelism
            gpu_mon = None
            if args.adaptive_parallel:
                gpu_mon = GPUMonitor(target_util=args.gpu_target)
                gpu_mon.start()
                current_parallel = max(args.min_parallel, args.parallel)
                print(
                    f"  [AdaptiveParallel] GPU monitor started "
                    f"(target={args.gpu_target * 100:.0f}%, range={args.min_parallel}-{args.max_parallel})"
                )
            else:
                current_parallel = args.parallel

            # Submit domains in waves: fill the pool, as each completes,
            # submit the next pending domain. Adjust pool size between waves.
            pending = list(enumerate(names))
            done = {}
            domain_start_times = {}  # idx -> start_time for heartbeat

            with ThreadPoolExecutor(max_workers=current_parallel) as pool:
                active = {}
                # Submit initial wave
                for _ in range(min(current_parallel, len(pending))):
                    idx, name = pending.pop(0)
                    future = pool.submit(_run_domain, (idx, name))
                    active[future] = (idx, name)
                    domain_start_times[idx] = time.time()
                    print(f"[{idx + 1:3d}/{len(names)}] {name}... STARTED", flush=True)

                while active:
                    # Wait for at least one to complete (with timeout for heartbeat)
                    done_futures, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                    if not done_futures:
                        # Heartbeat: no domain finished in 15s, show progress
                        now = time.time()
                        active_names = []
                        for idx, st in domain_start_times.items():
                            elapsed = now - st
                            active_names.append(f"{names[idx]}({elapsed:.0f}s)")
                        print(
                            f"  [Heartbeat] {len(active)} running: "
                            f"{', '.join(active_names[:4])}"
                            f"{'...' if len(active_names) > 4 else ''}",
                            flush=True,
                        )
                        continue
                    for future in done_futures:
                        idx, name = active.pop(future)
                        _, _, results, t, err = future.result()
                        done[idx] = (name, results, t, err)
                        domain_start_times.pop(idx, None)
                        if err or results is None:
                            err_msg = err if err else "results is None"
                            print(f"[{idx + 1:3d}/{len(names)}] {name}... ERROR ({t:.1f}s): {err_msg[:60]}", flush=True)
                        else:
                            best = results["best_score"]
                            disc = results["discoveries"]
                            cov = results["archive_coverage"]
                            flag = " <<< FREEZE" if t > 30 else (" << slow" if t > 15 else "")
                            print(
                                f"[{idx + 1:3d}/{len(names)}] {name}... "
                                f"best={best:10.4f}, {disc:4d} disc, archive={cov * 100:.0f}%, {t:.1f}s{flag}",
                                flush=True,
                            )

                        # Submit next pending domain if any
                        if pending:
                            # Adaptive: check GPU util and adjust
                            if gpu_mon and len(pending) > 2:
                                util = gpu_mon.get_avg_util()
                                vram = gpu_mon.get_vram_usage()
                                new_p = gpu_mon.recommend_parallelism(
                                    current_parallel, max_parallel=args.max_parallel, min_parallel=args.min_parallel
                                )
                                if new_p != current_parallel:
                                    current_parallel = new_p
                                    pool._max_workers = current_parallel
                                    print(
                                        f"  [AdaptiveParallel] GPU={util:.0f}%, VRAM={vram * 100:.0f}% "
                                        f"→ parallelism {current_parallel}",
                                        flush=True,
                                    )
                            idx_next, name_next = pending.pop(0)
                            future_next = pool.submit(_run_domain, (idx_next, name_next))
                            active[future_next] = (idx_next, name_next)
                            domain_start_times[idx_next] = time.time()
                            print(f"[{idx_next + 1:3d}/{len(names)}] {name_next}... STARTED", flush=True)
                    # Process all completed futures in this batch before re-entering while

            if gpu_mon:
                avg_util = gpu_mon.get_avg_util()
                print(f"  [AdaptiveParallel] Avg GPU util this loop: {avg_util:.0f}%")
                gpu_mon.stop()

            # Collect in original order
            for i in range(len(names)):
                if i in done:
                    all_results.append(done[i])
        else:
            # ── Sequential execution (fallback) ──
            for i, name in enumerate(names):
                print(f"[{i + 1:3d}/{len(names)}] {name}...", end=" ", flush=True)
                torch.cuda.empty_cache()
                results, t, err = run_one_domain(name, args.profile, overrides, db_path)
                if err:
                    print(f"ERROR ({t:.1f}s): {err[:60]}")
                    all_results.append((name, None, t, err))
                else:
                    best = results["best_score"]
                    disc = results["discoveries"]
                    cov = results["archive_coverage"]
                    flag = " <<< FREEZE" if t > 30 else (" << slow" if t > 15 else "")
                    print(f"best={best:10.4f}, {disc:4d} disc, archive={cov * 100:.0f}%, {t:.1f}s{flag}")
                    all_results.append((name, results, t, None))

        # ── Run refinement domains spawned during this loop ──
        # When a domain converges, the engine spawns a refinement domain
        # that narrows the search around the best config. Run these now.
        refinement_results = []
        for _name, results, _t, _err in all_results:
            if results and results.get("pending_refinements"):
                for ref_name in results["pending_refinements"]:
                    if ref_name in DOMAINS:
                        print(f"  [Refinement] Running {ref_name}...")
                        ref_domain = DOMAINS[ref_name]()
                        ref_results, ref_t, ref_err = run_one_domain(
                            ref_name, args.profile, overrides, db_path, domain_instance=ref_domain
                        )
                        if ref_err:
                            print(f"    ERROR: {ref_err[:60]}")
                        else:
                            print(
                                f"    best={ref_results['best_score']:.4f}, "
                                f"{ref_results['discoveries']} disc, {ref_t:.1f}s"
                            )
                            refinement_results.append((ref_name, ref_results, ref_t, None))

        # Add refinement results to the summary
        if refinement_results:
            all_results.extend(refinement_results)
            print(f"  [Refinement] {len(refinement_results)} refinement domains completed")

        # ── Revisit scheduler: re-run stale domains whose code may have changed ──
        # Based on Dynamic QD: when the environment changes (code updates),
        # previously-converged domains may find new optima.
        if args.revisit_stale and loop > 0:
            # Update scheduler with results from this loop
            for name, results, _t, _err in all_results:
                if results:
                    revisit_scheduler.update_after_run(
                        name, loop, results["best_score"], converged=results.get("pending_refinements") is not None
                    )

            # Check for stale domains to re-run
            stale = revisit_scheduler.get_stale_domains(current_loop=loop + 1)
            if stale:
                print(f"  [Revisit] {len(stale)} stale domains to re-run")
                revisit_results = []
                for name, priority in stale[:5]:  # cap at 5 per loop
                    if name in DOMAINS:
                        print(f"  [Revisit] Re-running {name} (priority={priority:.2f})...")
                        r_results, r_t, r_err = run_one_domain(name, args.profile, overrides, db_path)
                        if r_err:
                            print(f"    ERROR: {r_err[:60]}")
                        else:
                            print(
                                f"    best={r_results['best_score']:.4f}, {r_results['discoveries']} disc, {r_t:.1f}s"
                            )
                            revisit_results.append((name, r_results, r_t, None))
                if revisit_results:
                    all_results.extend(revisit_results)
                    print(f"  [Revisit] {len(revisit_results)} stale domains re-run")

        total_t = time.time() - t_start
        print_summary(all_results, total_t, args.profile)

        # Save JSON summary (per-loop, last loop overwrites)
        valid = [(n, r, t) for n, r, t, e in all_results if r is not None]
        summary = {
            "profile": args.profile,
            "loop": loop + 1,
            "total_loops": args.loops,
            "total_domains": len(names),
            "total_time_s": total_t,
            "total_time_m": total_t / 60,
            "valid_runs": len(valid),
            "failed_runs": len(all_results) - len(valid),
            "results": [
                {
                    "domain": n,
                    "best_score": r["best_score"],
                    "discoveries": r["discoveries"],
                    "archive_coverage": r["archive_coverage"],
                    "time_s": t,
                    "best_config": r.get("best_config"),
                }
                for n, r, t in valid
            ],
        }
        summary_path = RESULTS_DIR / f"evolve{db_suffix}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Summary: {summary_path}")

        loop_summaries.append(
            {
                "loop": loop + 1,
                "time_s": total_t,
                "valid": len(valid),
                "failed": len(all_results) - len(valid),
                "best_scores": {n: r["best_score"] for n, r, t in valid},
            }
        )

        # Ideas report (only on last loop to save time)
        if loop == args.loops - 1 and not args.no_ideas and valid:
            print("  Extracting discoveries...")
            top = extract_top_discoveries(db_path, top_n=20)
            ideas = generate_ideas_report(top, args.profile)
            ideas_path = RESULTS_DIR / f"evolve{db_suffix}_ideas.md"
            with open(ideas_path, "w") as f:
                f.write(ideas)
            print(f"  Ideas: {ideas_path}")

        print(f"\n  Loop {loop + 1} done in {total_t / 60:.1f}m")

    # ── Grand summary across all loops ──
    grand_t = time.time() - grand_start
    if args.loops > 1:
        print(f"\n{'=' * 70}")
        print(f"  GRAND SUMMARY: {args.loops} loops in {grand_t:.1f}s ({grand_t / 60:.1f}m)")
        print(f"{'=' * 70}")
        print(f"  {'Loop':>4s} {'Time':>8s} {'Valid':>6s} {'Top Score':>12s} {'Domain':>20s}")
        print(f"  {'-' * 4} {'-' * 8} {'-' * 6} {'-' * 12} {'-' * 20}")
        for ls in loop_summaries:
            if ls["best_scores"]:
                top_domain = max(ls["best_scores"], key=ls["best_scores"].get)
                top_score = ls["best_scores"][top_domain]
                print(f"  {ls['loop']:4d} {ls['time_s']:7.1f}s {ls['valid']:6d} {top_score:12.4f} {top_domain:>20s}")
            else:
                print(f"  {ls['loop']:4d} {ls['time_s']:7.1f}s {ls['valid']:6d} {'N/A':>12s} {'(no valid)':>20s}")

        # Show score progression per domain across loops
        print("\n  Score progression (top 10 domains):")
        all_domains = set()
        for ls in loop_summaries:
            all_domains.update(ls["best_scores"].keys())
        # Rank by final loop score
        final = loop_summaries[-1]["best_scores"]
        top_domains = sorted(all_domains, key=lambda d: final.get(d, -1e9), reverse=True)[:10]
        header = f"  {'Domain':<25s}" + "".join(f" {'L' + str(i + 1):>8s}" for i in range(args.loops))
        print(header)
        for d in top_domains:
            row = f"  {d:<25s}"
            for ls in loop_summaries:
                s = ls["best_scores"].get(d, float("nan"))
                row += f" {s:8.2f}"
            print(row)
        print(f"\n  Total: {grand_t:.1f}s ({grand_t / 60:.1f}m)")
    else:
        print(f"\n  Done in {grand_t / 60:.1f}m")


if __name__ == "__main__":
    main()
