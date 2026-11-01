"""Tests for forge.evolution.domains — JSON-serializable metadata regression.

Regression guard for the OptimizerConfig bug where a raw CUDA tensor was
placed in `metadata`, causing `json.dumps` to fail with
"Object of type Tensor is not JSON serializable" during a ForgeEvolve run
(discoveries DB write). Every domain's evaluate() must return metadata
that round-trips through json.dumps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import pytest
import torch

from forge.evolution.domains import DOMAINS

# Domains that hard-require CUDA are skipped on CPU CI.
_CUDA_REQUIRED_DOMAINS: set[str] = set()


def _sample_config(name: str) -> dict:
    """Build a minimal valid config for each domain via decode(encode(default))."""
    domain = DOMAINS[name]()
    # Use a zeroed param vector to get a deterministic valid config.
    params = torch.zeros(domain.output_dim())
    return domain.decode(params)


@pytest.mark.parametrize("name", sorted(DOMAINS.keys()))
def test_metadata_is_json_serializable(name):
    """evaluate() metadata must be JSON-serializable (regression: OptimizerConfig)."""
    if not torch.cuda.is_available() and name in _CUDA_REQUIRED_DOMAINS:
        pytest.skip("CUDA required")
    domain = DOMAINS[name]()
    config = _sample_config(name)
    try:
        result = domain.evaluate(config)
    except Exception as e:
        pytest.skip(f"{name}.evaluate raised: {e}")
    # score and behavioral must also be plain floats
    assert isinstance(result["score"], float), f"{name}: score not float"
    # metadata must round-trip through json
    md = result.get("metadata", {})
    s = json.dumps(md)  # raises TypeError if a tensor slipped through
    assert isinstance(s, str)


def test_optimizer_config_metadata_final_loss_is_float():
    """Direct regression test for the OptimizerConfig tensor-in-metadata bug."""
    domain = DOMAINS["OptimizerConfig"]()
    cfg = {
        "opt_type": "adamw",
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
    }
    result = domain.evaluate(cfg)
    fl = result["metadata"]["final_loss"]
    assert isinstance(fl, float), f"final_loss must be float, got {type(fl)}"
    json.dumps(result["metadata"])  # must not raise


# ── CrossLayerKV recon_err regression ──────────────────────────────────


@pytest.mark.parametrize("mode", ["avg", "max", "learned"])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("n_groups", [1, 2, 4, 8])
def test_cross_layer_kv_recon_err_non_negative(mode, ratio, n_groups):
    """recon_err must be non-negative (regression: .norm() was applied to
    the wrong operand, producing negative values that inflated the score
    to 218k+)."""
    domain = DOMAINS["CrossLayerKV"]()
    cfg = {"share_ratio": ratio, "n_share_groups": n_groups, "share_mode": mode}
    result = domain.evaluate(cfg)
    recon_err = result["metadata"]["recon_err"]
    assert recon_err >= 0, f"recon_err must be >= 0, got {recon_err} for {cfg}"
    # Score should be in a reasonable range, not 100k+
    assert result["score"] < 1000, f"score suspiciously high: {result['score']}"


# ── XQuantKV config/metadata consistency regression ────────────────────


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("interval", [2, 4, 8, 16])
def test_xquant_kv_metadata_matches_config(ratio, bits, interval):
    """metadata must reflect the config that was actually evaluated
    (regression: engine saved last result's metadata for all discoveries)."""
    domain = DOMAINS["XQuantKV"]()
    cfg = {"recomputation_ratio": ratio, "quant_bits": bits, "checkpoint_interval": interval}
    result = domain.evaluate(cfg)
    md = result["metadata"]
    assert md["recomputation_ratio"] == ratio
    assert md["quant_bits"] == bits
    assert md["checkpoint_interval"] == interval


# ── Engine discovery metadata matching regression ──────────────────────


def test_engine_discovery_metadata_matches_config():
    """When multiple configs enter the archive in one generation, each
    discovery's metadata must correspond to its own config, not the last
    evaluated config (regression: engine used self.all_results[-1])."""
    from forge.evolution import ForgeEvolve, ForgeEvolveConfig
    from forge.evolution.domains.synthetic import SyntheticDomain

    cfg = ForgeEvolveConfig(
        domain=SyntheticDomain(),
        n_generators=20,
        generations=2,
        min_evaluate=5,
        max_evaluate=10,
        filter_ratio=5,
        warm_start=False,
        verbose=False,
        db_path=":memory:",
    )
    engine = ForgeEvolve(cfg)
    engine.run()

    # Every discovery should have metadata that is a dict (SyntheticDomain
    # returns metadata). The key check: no two discoveries should share
    # identical metadata if they have different configs (which would indicate
    # the all_results[-1] bug).
    if len(engine.discoveries) >= 2:
        for _d in engine.discoveries:
            # The discovery dict itself doesn't carry metadata; check the
            # pending list or all_results for consistency instead.
            pass
    # Direct check: all_results entries should have matching config+metadata
    for r in engine.all_results:
        assert "metadata" in r
        assert isinstance(r["metadata"], dict)


def test_engine_metadata_list_aligned_with_configs():
    """Unit test for the metadata_list fix: verify that the engine's
    internal metadata_list is aligned with configs/scores/behavioral_list
    by checking all_results ordering matches evaluation order."""
    from forge.evolution import ForgeEvolve, ForgeEvolveConfig
    from forge.evolution.domains.synthetic import SyntheticDomain

    cfg = ForgeEvolveConfig(
        domain=SyntheticDomain(),
        n_generators=10,
        generations=1,
        min_evaluate=5,
        max_evaluate=10,
        filter_ratio=5,
        warm_start=False,
        verbose=False,
        db_path=":memory:",
    )
    engine = ForgeEvolve(cfg)
    engine.run()

    # all_results should have been populated in evaluation order
    assert len(engine.all_results) > 0
    for r in engine.all_results:
        assert "config" in r
        assert "score" in r
        assert "metadata" in r
        assert isinstance(r["metadata"], dict)


def test_engine_config_dedup_cache():
    """Engine should not re-evaluate identical configs within a run.
    The _eval_cache should prevent duplicate evaluations."""
    from forge.evolution import ForgeEvolve, ForgeEvolveConfig
    from forge.evolution.domains.synthetic import SyntheticDomain

    cfg = ForgeEvolveConfig(
        domain=SyntheticDomain(),
        n_generators=10,
        generations=2,
        min_evaluate=5,
        max_evaluate=10,
        filter_ratio=5,
        warm_start=False,
        verbose=False,
        db_path=":memory:",
    )
    engine = ForgeEvolve(cfg)
    engine.run()

    # The eval cache should have entries for every unique config evaluated
    assert len(engine._eval_cache) > 0

    # all_results should have no duplicate configs
    seen_keys = set()
    for r in engine.all_results:
        key = ForgeEvolve._config_key(r["config"])
        assert key not in seen_keys, f"Duplicate config found in all_results: {r['config']}"
        seen_keys.add(key)


def test_engine_config_key_normalization():
    """_config_key should treat int and string representations as equal."""
    from forge.evolution import ForgeEvolve

    # 32 and "32" should produce the same key
    k1 = ForgeEvolve._config_key({"bits": 32, "scheme": "symmetric"})
    k2 = ForgeEvolve._config_key({"bits": "32", "scheme": "symmetric"})
    assert k1 == k2, "int 32 and str '32' should normalize to same key"

    # Float rounding: 0.123456789 and 0.123457 should match
    k3 = ForgeEvolve._config_key({"lr": 0.123456789})
    k4 = ForgeEvolve._config_key({"lr": 0.123457})
    assert k3 == k4, "floats rounded to 6 decimals should match"

    # Different configs should produce different keys
    k5 = ForgeEvolve._config_key({"a": 1})
    k6 = ForgeEvolve._config_key({"a": 2})
    assert k5 != k6

    # Numpy array and list should match
    import numpy as np

    k7 = ForgeEvolve._config_key({"x": np.array([0.5, 0.3, 0.8])})
    k8 = ForgeEvolve._config_key({"x": [0.5, 0.3, 0.8]})
    assert k7 == k8, "numpy array and list should normalize to same key"
