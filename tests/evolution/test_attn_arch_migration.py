"""Bit-exact verification test for attention + arch domain migration to JSON spec.

For each domain, compares the original Python domain class evaluate() output
against the JSONSpecDomain evaluate() output. Scores must match within 1e-4.
"""

import os
import sys

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

# --- Domain configs to test (one per domain) ---
TEST_CONFIGS = {
    # Attention domains
    "rope_config": (
        "RopeConfig",
        "rope_config",
        {"theta": 500000, "scaling_type": "yarn", "scaling_factor": 2.0},
    ),
    "diff_attn": (
        "DiffAttnConfig",
        "diff_attn",
        {"lambda_init": 0.5, "n_heads": 16, "softmax_sep": 0.5},
    ),
    "csa_attention": (
        "CsaAttention",
        "csa_attention",
        {"top_k": 256, "pattern_type": "csa", "block_size": 32},
    ),
    "gla_attention": (
        "GlaAttention",
        "gla_attention",
        {"latent_dim": 256, "n_heads": 16, "compression_ratio": 2.0},
    ),
    "gta_attention": (
        "GtaAttention",
        "gta_attention",
        {"v_k_mix": 0.5, "n_kv_heads": 8, "tie_strength": 0.5},
    ),
    "qk_norm": (
        "QkNormConfig",
        "qk_norm_config",
        {"norm_type": "rmsnorm", "epsilon": 1e-5, "scale_init": 1.0},
    ),
    "attn_residual": (
        "AttnResidual",
        "attn_residual",
        {"k_layers": 4, "gate_init": 0.5, "retrieval_dim": 256},
    ),
    "mhc_config": (
        "MhcConfig",
        "mhc_config",
        {"rank": 256, "gate_init": 0.5, "n_connections": 4},
    ),
    "sliding_window": (
        "SlidingWindow",
        "sliding_window",
        {"window_size": 1024, "stride": 512, "overlap_ratio": 0.5},
    ),
    "local_global": (
        "LocalGlobal",
        "local_global",
        {"local_window": 512, "global_ratio": 0.2, "n_global_heads": 4},
    ),
    # Arch domains
    "moe_routing": (
        "MoeRouting",
        "moe_routing",
        {"n_experts": 8, "top_k": 2, "router_mode": "switch", "load_balance_weight": 0.01, "shared_expert": True},
    ),
    "factorized_embed": (
        "FactorizedEmbed",
        "factorized_embed",
        {"rank": 256, "init_mode": "svd", "tie_factor": 0.5, "vocab_size": 65536},
    ),
    "titan_memory": (
        "TitanMemory",
        "titan_memory",
        {"memory_rank": 256, "gate_init": 0.1, "n_memory_slots": 4, "update_freq": 5},
    ),
    "ffn_skip": (
        "FfnSkip",
        "ffn_skip",
        {"skip_threshold": 0.3, "n_eval_layers": 8, "skip_strategy": "cosine", "min_keep": 0.8},
    ),
    "conv_config": (
        "ConvConfig",
        "conv_config",
        {"kernel_size": 5, "stride": 1, "dilation": 2, "groups": 4, "n_conv_layers": 3},
    ),
}

# Additional configs to test edge cases (trivial solutions, etc.)
EDGE_CONFIGS = {
    "rope_config": [
        {"theta": 10000, "scaling_type": "none", "scaling_factor": 1.0},
        {"theta": 1000000, "scaling_type": "none", "scaling_factor": 1.0},
        {"theta": 10000000, "scaling_type": "linear", "scaling_factor": 3.0},
    ],
    "csa_attention": [
        {"top_k": 2048, "pattern_type": "standard", "block_size": 128},  # sparsity < 0.1
        {"top_k": 64, "pattern_type": "csa_hca_hybrid", "block_size": 8},
    ],
    "gla_attention": [
        {"latent_dim": 512, "n_heads": 4, "compression_ratio": 1.0},  # compression < 1.5
        {"latent_dim": 64, "n_heads": 32, "compression_ratio": 8.0},
    ],
    "gta_attention": [
        {"v_k_mix": 0.9, "n_kv_heads": 4, "tie_strength": 1.0},  # high deviation
        {"v_k_mix": 0.1, "n_kv_heads": 16, "tie_strength": 0.0},
    ],
    "sliding_window": [
        {"window_size": 4096, "stride": 1, "overlap_ratio": 0.5},  # memory_ratio > 0.9
    ],
    "local_global": [
        {"local_window": 2048, "global_ratio": 0.9, "n_global_heads": 16},  # compute > 0.9
    ],
    "moe_routing": [
        {
            "n_experts": 8,
            "top_k": 1,
            "router_mode": "switch",
            "load_balance_weight": 0.01,
            "shared_expert": False,
        },  # top_k=1
        {"n_experts": 4, "top_k": 3, "router_mode": "aux_free", "load_balance_weight": 0.05, "shared_expert": True},
    ],
    "factorized_embed": [
        {"rank": 64, "init_mode": "random", "tie_factor": 0.0, "vocab_size": 32000},
    ],
    "titan_memory": [
        {"memory_rank": 64, "gate_init": 0.4, "n_memory_slots": 8, "update_freq": 10},
    ],
    "ffn_skip": [
        {"skip_threshold": 0.5, "n_eval_layers": 16, "skip_strategy": "norm", "min_keep": 1.0},
    ],
}


def run_test():
    from forge.evolution.domain_spec import JSONSpecDomain
    from forge.evolution.domains.arch_domains import (
        ConvConfig,
        FactorizedEmbed,
        FfnSkip,
        MoeRouting,
        TitanMemory,
    )
    from forge.evolution.domains.attention_domains import (
        AttnResidual,
        CsaAttention,
        DiffAttnConfig,
        GlaAttention,
        GtaAttention,
        LocalGlobal,
        MhcConfig,
        QkNormConfig,
        RopeConfig,
        SlidingWindow,
    )

    DOMAIN_CLASSES = {
        "RopeConfig": RopeConfig,
        "DiffAttnConfig": DiffAttnConfig,
        "CsaAttention": CsaAttention,
        "GlaAttention": GlaAttention,
        "GtaAttention": GtaAttention,
        "QkNormConfig": QkNormConfig,
        "AttnResidual": AttnResidual,
        "MhcConfig": MhcConfig,
        "SlidingWindow": SlidingWindow,
        "LocalGlobal": LocalGlobal,
        "MoeRouting": MoeRouting,
        "FactorizedEmbed": FactorizedEmbed,
        "TitanMemory": TitanMemory,
        "FfnSkip": FfnSkip,
        "ConvConfig": ConvConfig,
    }

    passed = 0
    failed = 0
    failures = []

    for domain_name, (old_cls_name, spec_name, base_config) in TEST_CONFIGS.items():
        configs_to_test = [base_config]
        if domain_name in EDGE_CONFIGS:
            configs_to_test.extend(EDGE_CONFIGS[domain_name])

        for i, config in enumerate(configs_to_test):
            label = f"{domain_name}" + (f"/edge_{i}" if i > 0 else "")

            # Original domain
            torch.manual_seed(42)
            np.random.seed(42)
            old_domain = DOMAIN_CLASSES[old_cls_name]()
            old_result = old_domain.evaluate(dict(config))

            # JSON spec domain
            torch.manual_seed(42)
            np.random.seed(42)
            new_domain = JSONSpecDomain(spec_name)
            new_result = new_domain.evaluate(dict(config))

            old_score = old_result["score"]
            new_score = new_result["score"]
            diff = abs(old_score - new_score)

            if diff < 1e-4:
                passed += 1
                print(f"  PASS  {label:40s}  old={old_score:.6f}  new={new_score:.6f}  diff={diff:.2e}")
            else:
                failed += 1
                failures.append((label, old_score, new_score, diff, config))
                print(f"  FAIL  {label:40s}  old={old_score:.6f}  new={new_score:.6f}  diff={diff:.2e}")

    print(f"\n{'=' * 80}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failures:
        print("\nFailures:")
        for label, old_s, new_s, diff, cfg in failures:
            print(f"  {label}: old={old_s:.6f} new={new_s:.6f} diff={diff:.2e}")
            print(f"    config: {cfg}")
    return failed == 0


def test_attn_arch_migration():
    """Pytest entry: bit-exact attention + arch domain migration check."""
    assert run_test(), "attention/arch migration mismatches (see FAIL lines above)"


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
