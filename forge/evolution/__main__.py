"""CLI entry point for the ForgeEvolve evolutionary optimizer.

Usage:
    # Run evolution on a domain for N generations (steps)
    python -m forge.evolution --domain synthetic --steps 10

    # List all available domains
    python -m forge.evolution --list-domains

    # Run with custom parameters
    python -m forge.evolution --domain w8a8_quant --steps 50 --generators 200 --filter-ratio 20

    # Run on CPU only
    python -m forge.evolution --domain synthetic --steps 5 --device cpu

    # Export best configs so sft_train.py picks them up (live link)
    python -m forge.evolution --apply-best

    # Quiet mode (minimal logging)
    python -m forge.evolution --domain synthetic --steps 5 --quiet

Domain names can be either:
  - JSON spec names (lowercase, e.g. "synthetic", "w8a8_quant", "mtp_config")
  - Python class names (CamelCase, e.g. "SyntheticDomain")

Results are persisted to the SQLite database (default: forge_evolve.db).
"""
from __future__ import annotations

import argparse
import sys

from .engine import ForgeEvolve, ForgeEvolveConfig


def _resolve_domain(name: str):
    """Resolve a domain name to a BaseDomain instance.

    Tries JSON spec name first (lowercase), then Python class name (CamelCase).
    """
    from .domain_spec import JSONSpecDomain, list_specs, load_spec
    from .domains import DOMAINS, get_domain

    # Try JSON spec name (lowercase with underscores)
    if name in list_specs():
        spec = load_spec(name)
        return JSONSpecDomain(spec=spec)

    # Try Python class name (CamelCase)
    if name in DOMAINS:
        return get_domain(name)

    # Try case-insensitive match on JSON specs
    specs = list_specs()
    lower_map = {s.lower(): s for s in specs}
    if name.lower() in lower_map:
        spec = load_spec(lower_map[name.lower()])
        return JSONSpecDomain(spec=spec)

    # Try case-insensitive match on class names
    class_map = {k.lower(): k for k in DOMAINS}
    if name.lower() in class_map:
        return get_domain(class_map[name.lower()])

    raise SystemExit(
        f"Unknown domain '{name}'.\n"
        f"Available JSON specs: {sorted(specs)}\n"
        f"Available classes: {sorted(DOMAINS.keys())}\n"
        f"Use --list-domains to see all options."
    )


def _list_domains() -> None:
    """Print all available domain names."""
    from .domain_spec import list_specs
    from .domains import DOMAINS

    json_specs = list_specs()
    class_names = sorted(DOMAINS.keys())

    print("Available domains:")
    print()
    print("JSON spec names (use with --domain):")
    for s in json_specs:
        print(f"  {s}")
    print()
    print("Python class names (use with --domain):")
    for c in class_names:
        print(f"  {c}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m forge.evolution",
        description="ForgeEvolve — evolutionary candidate discovery engine.",
    )
    parser.add_argument(
        "--domain", "-d",
        help="Domain to evolve (JSON spec name or Python class name). "
             "Use --list-domains to see options.",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int, default=50,
        help="Number of evolution generations to run (default: 50).",
    )
    parser.add_argument(
        "--generators", "-g",
        type=int, default=500,
        help="Number of generators in the population (default: 500).",
    )
    parser.add_argument(
        "--filter-ratio",
        type=int, default=20,
        help="Generate/filter ratio: N_generated / N_evaluated (default: 20).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device (default: auto = cuda if available).",
    )
    parser.add_argument(
        "--db-path",
        default="forge_evolve.db",
        help="Path to the SQLite findings database (default: forge_evolve.db).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-generation logging.",
    )
    parser.add_argument(
        "--fixate",
        action="store_true",
        help="Pin the run to this single domain: no cross-domain transfer, "
             "no refinement spawning, plateaus re-diversify in place "
             "instead of stopping.",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List all available domain names and exit.",
    )
    parser.add_argument(
        "--apply-best",
        nargs="?",
        const="forge/evolution/best_configs.json",
        default=None,
        metavar="OUT_JSON",
        help="Export the best config per domain from the findings DB to "
             "best_configs.json (default path inside the package) and exit. "
             "sft_train.py reads this file to override evolution-discovered "
             "defaults — the live link between evolution and production.",
    )
    args = parser.parse_args(argv)

    if args.list_domains:
        _list_domains()
        return 0

    if args.apply_best:
        from .best_configs import export_best_configs
        exported = export_best_configs(db_path=args.db_path,
                                       out_path=args.apply_best)
        print(f"Exported best configs for {len(exported)} domains "
              f"from {args.db_path} -> {args.apply_best}")
        for name, entry in sorted(exported.items()):
            print(f"  {name}: score={entry.get('score')} "
                  f"config={entry.get('config')}")
        try:
            import json as _json
            skipped = _json.loads(
                open(args.apply_best, encoding="utf-8").read()).get("skipped")
            if skipped:
                print(f"  ({skipped} discovery rows rejected by the "
                      f"verified-delta>0 score floor)")
        except Exception:
            pass
        return 0

    if not args.domain:
        parser.error("--domain is required (or use --list-domains/--apply-best)")

    domain = _resolve_domain(args.domain)

    cfg = ForgeEvolveConfig(
        domain=domain,
        n_generators=args.generators,
        filter_ratio=args.filter_ratio,
        generations=args.steps,
        device=args.device,
        db_path=args.db_path,
        verbose=not args.quiet,
        fixate=args.fixate,
    )

    engine = ForgeEvolve(cfg)
    results = engine.run()

    # Print summary
    print()
    print("=" * 60)
    print(f"ForgeEvolve complete — domain: {domain.name()}")
    print(f"  Generations:      {results['generations']}")
    print(f"  Evaluations:      {results['total_evaluations']}")
    print(f"  Discoveries:      {results['discoveries']}")
    print(f"  Best score:       {results['best_score']:.4f}")
    print(f"  Best config:      {results['best_config']}")
    print(f"  Device:           {results['device']}")
    print(f"  Time:             {results.get('time_s', 0):.1f}s")
    if results.get("spawned_domains"):
        print(f"  Spawned domains:  {results['spawned_domains']}")
    print(f"  Database:         {args.db_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
