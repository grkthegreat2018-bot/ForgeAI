"""Smoke test: verify all 57 domains can instantiate and evaluate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest
import torch

from forge.evolution.domains import DOMAINS, list_domains


@pytest.mark.parametrize("name", list_domains())
def test_domain(name):
    """Test one domain: instantiate, decode, evaluate, check output."""
    cls = DOMAINS[name]
    domain = cls()
    # Check output_dim
    od = domain.output_dim()
    assert od > 0, f"output_dim={od}"
    # Check behavioral_dims
    bd = domain.behavioral_dims()
    assert len(bd) >= 1, "behavioral_dims empty"
    # Decode random params
    params = torch.rand(od)
    config = domain.decode(params)
    assert isinstance(config, dict), f"decode returned {type(config)}"
    # Evaluate
    result = domain.evaluate(config)
    assert "score" in result, "missing score"
    assert "behavioral" in result, "missing behavioral"
    assert "metadata" in result, "missing metadata"
    assert isinstance(result["score"], float), "score not float"
    assert isinstance(result["behavioral"], tuple), "behavioral not tuple"
    # Check behavioral dims match
    assert len(result["behavioral"]) == len(bd), f"behavioral len {len(result['behavioral'])} != dims {len(bd)}"
    # Check encode round-trip
    encoded = domain.encode(config)
    assert encoded.shape[0] == od, f"encode shape {encoded.shape} != {od}"


if __name__ == "__main__":
    names = list_domains()
    print(f"Testing {len(names)} domains...\n")
    passed = 0
    failed = 0
    scores = []
    for name in names:
        ok, info = test_domain(name)
        if ok:
            passed += 1
            scores.append((name, info))
            print(f"  OK  {name:30s} score={info:10.4f}")
        else:
            failed += 1
            print(f"  FAIL {name:30s} {info}")
    print(f"\n{'=' * 60}")
    print(f"Passed: {passed}/{len(names)}, Failed: {failed}")
    if scores:
        best = max(scores, key=lambda x: x[1])
        worst = min(scores, key=lambda x: x[1])
        print(f"Best score: {best[0]} = {best[1]:.4f}")
        print(f"Worst score: {worst[0]} = {worst[1]:.4f}")
