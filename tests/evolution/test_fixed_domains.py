"""Quick test of the 5 fixed domains."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["PYTHONUTF8"] = "1"

import torch

from forge.evolution.domain_spec import JSONSpecDomain

domains_to_test = [
    "quant_domain",
    "cross_layer_kv",
    "gla_attention",
    "cpu_kv_offload",
    "flashoptim_config",
]

for name in domains_to_test:
    domain = JSONSpecDomain(spec_name=name, device=torch.device("cpu"))

    print(f"\n=== {name} ===")
    seeds = domain.seed_configs()
    for _i, config in enumerate(seeds[:3]):
        result = domain.evaluate(config)
        score = result["score"]
        print(f"  config={config}")
        print(f"  score={score:.4f}")
        print()

    # Also test a random config
    import random

    params = torch.tensor([random.random() for _ in range(domain.output_dim())])
    config = domain.decode(params)
    result = domain.evaluate(config)
    print(f"  random config={config}")
    print(f"  score={result['score']:.4f}")
