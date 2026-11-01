"""Bit-exact match test: old Python domains vs new JSON spec + simulator.

For each of the 13 training domains, evaluate the same config with both
the original Python domain class and the new JSONSpecDomain, then assert
the scores match within 1e-4.
"""

import numpy as np
import pytest
import torch

from forge.evolution.domain_spec import JSONSpecDomain
from forge.evolution.domains.training_domains import (
    ApolloConfig,
    BreadConfig,
    CpuAdamwConfig,
    FlashOptimConfig,
    Fp8TrainingConfig,
    GradAccumConfig,
    LossConfig,
    ModConfig,
    MuonConfig,
    OptimizerConfig,
    SchedulerConfig,
    TritonKernelConfig,
    VarlenConfig,
)

# (old_class, spec_name, config_dict, description)
TESTS = [
    (
        OptimizerConfig,
        "optimizer_config",
        {"opt_type": "adamw", "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.01},
        "optimizer adamw",
    ),
    (
        SchedulerConfig,
        "scheduler_config",
        {"sched_type": "cosine", "warmup_steps": 100, "min_lr_ratio": 0.1, "decay_steps": 1000},
        "scheduler cosine w/ warmup",
    ),
    (
        LossConfig,
        "loss_config",
        {"loss_type": "ce", "label_smoothing": 0.1, "focal_gamma": 2.0, "temperature": 1.0},
        "loss ce",
    ),
    (MuonConfig, "muon_config", {"momentum": 0.95, "nesterov": True, "weight_decay": 0.001, "ns_steps": 3}, "muon"),
    (
        CpuAdamwConfig,
        "cpu_adamw_config",
        {"offload_ratio": 0.5, "prefetch_depth": 4, "compression": "int8", "update_freq": 4},
        "cpu_adamw int8",
    ),
    (
        GradAccumConfig,
        "grad_accum_config",
        {"accum_steps": 8, "micro_batch": 4, "grad_clip": 1.0, "sync_freq": 4},
        "grad_accum",
    ),
    (
        Fp8TrainingConfig,
        "fp8_training_config",
        {"autocast_mode": "e4m3", "smooth_swiglu": True, "mu_scaling": True, "loss_scale": 1024},
        "fp8 e4m3",
    ),
    (
        ModConfig,
        "mod_config",
        {"keep_fraction": 0.8, "router_type": "mlp", "aux_loss_weight": 0.001, "n_skip_layers": 4},
        "mod mlp",
    ),
    (ApolloConfig, "apollo_config", {"rank": 8, "scale": "channel", "lr_scale": 2.0}, "apollo channel"),
    (BreadConfig, "bread_config", {"correction_mode": "partial", "sgd_lr_scale": 5.0}, "bread partial"),
    (FlashOptimConfig, "flashoptim_config", {"bits": 8, "companding": "sqrt"}, "flashoptim 8bit sqrt"),
    (
        TritonKernelConfig,
        "triton_kernel_config",
        {"rms_block_size": 4096, "swiglu_block_size": 16384},
        "triton kernel matched",
    ),
    (VarlenConfig, "varlen_config", {"use_varlen": True}, "varlen true"),
]

# Config key remapping for domains where flag handlers expect different keys
KEY_REMAPS = {}


def _remap_config(spec_name, config):
    """Remap config keys for JSONSpecDomain if needed."""
    remap = KEY_REMAPS.get(spec_name, {})
    new_config = {}
    for k, v in config.items():
        new_key = k
        for new_k, old_k in remap.items():
            if k == old_k:
                new_key = new_k
                break
        new_config[new_key] = v
    return new_config


@pytest.mark.parametrize("old_cls,spec_name,config,desc", TESTS, ids=[t[3] for t in TESTS])
def test_bit_exact_match(old_cls, spec_name, config, desc):
    torch.manual_seed(42)
    np.random.seed(42)
    old = old_cls()
    ro = old.evaluate(config)

    torch.manual_seed(42)
    np.random.seed(42)
    new_config = _remap_config(spec_name, config)
    new = JSONSpecDomain(spec_name)
    rn = new.evaluate(new_config)

    diff = abs(ro["score"] - rn["score"])
    assert diff < 1e-4, f"{desc}: old={ro['score']:.6f} new={rn['score']:.6f} diff={diff:.2e}"
