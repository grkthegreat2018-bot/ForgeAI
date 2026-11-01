"""Bit-exact migration test for training + attention + arch domains."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forge.evolution.domain_spec import JSONSpecDomain

training_domains = [
    ("OptimizerConfig", "optimizer_config", "forge.evolution.domains.training_domains"),
    ("SchedulerConfig", "scheduler_config", "forge.evolution.domains.training_domains"),
    ("LossConfig", "loss_config", "forge.evolution.domains.training_domains"),
    ("MuonConfig", "muon_config", "forge.evolution.domains.training_domains"),
    ("CpuAdamwConfig", "cpu_adamw_config", "forge.evolution.domains.training_domains"),
    ("GradAccumConfig", "grad_accum_config", "forge.evolution.domains.training_domains"),
    ("Fp8TrainingConfig", "fp8_training_config", "forge.evolution.domains.training_domains"),
    ("ModConfig", "mod_config", "forge.evolution.domains.training_domains"),
    ("ApolloConfig", "apollo_config", "forge.evolution.domains.training_domains"),
    ("BreadConfig", "bread_config", "forge.evolution.domains.training_domains"),
    ("FlashOptimConfig", "flashoptim_config", "forge.evolution.domains.training_domains"),
    ("TritonKernelConfig", "triton_kernel_config", "forge.evolution.domains.training_domains"),
    ("VarlenConfig", "varlen_config", "forge.evolution.domains.training_domains"),
]

attention_domains = [
    ("RopeConfig", "rope_config", "forge.evolution.domains.attention_domains"),
    ("DiffAttnConfig", "diff_attn", "forge.evolution.domains.attention_domains"),
    ("CsaAttention", "csa_attention", "forge.evolution.domains.attention_domains"),
    ("GlaAttention", "gla_attention", "forge.evolution.domains.attention_domains"),
    ("GtaAttention", "gta_attention", "forge.evolution.domains.attention_domains"),
    ("QkNormConfig", "qk_norm_config", "forge.evolution.domains.attention_domains"),
    ("AttnResidual", "attn_residual", "forge.evolution.domains.attention_domains"),
    ("MhcConfig", "mhc_config", "forge.evolution.domains.attention_domains"),
    ("SlidingWindow", "sliding_window", "forge.evolution.domains.attention_domains"),
    ("LocalGlobal", "local_global", "forge.evolution.domains.attention_domains"),
]

arch_domains = [
    ("MoeRouting", "moe_routing", "forge.evolution.domains.arch_domains"),
    ("FactorizedEmbed", "factorized_embed", "forge.evolution.domains.arch_domains"),
    ("TitanMemory", "titan_memory", "forge.evolution.domains.arch_domains"),
    ("FfnSkip", "ffn_skip", "forge.evolution.domains.arch_domains"),
    ("ConvConfig", "conv_config", "forge.evolution.domains.arch_domains"),
]

all_domains = training_domains + attention_domains + arch_domains


def test_training_arch_migration():
    n_pass = 0
    n_fail = 0

    for cls_name, spec_name, mod_path in all_domains:
        try:
            mod = importlib.import_module(mod_path)
            OldCls = getattr(mod, cls_name)
            old = OldCls()
            seeds = old.seed_configs()
            if not seeds:
                params = torch.tensor([0.5] * old.output_dim())
                seeds = [old.decode(params)]
            test_cfg = seeds[0]

            torch.manual_seed(42)
            np.random.seed(42)
            ro = old.evaluate(test_cfg)
            torch.manual_seed(42)
            np.random.seed(42)
            new = JSONSpecDomain(spec_name)
            rn = new.evaluate(test_cfg)

            match = abs(ro["score"] - rn["score"]) < 1e-3
            status = "OK" if match else "MISMATCH"
            if match:
                n_pass += 1
            else:
                n_fail += 1
            print(f"{cls_name:25s} old={ro['score']:10.4f} new={rn['score']:10.4f} {status}")
        except Exception as e:
            n_fail += 1
            print(f"{cls_name:25s} ERROR: {e}")

    print(f"\n{n_pass}/{n_pass + n_fail} passed")
    assert n_fail == 0, (
        f"{n_fail}/{n_pass + n_fail} training/attention/arch domains failed "
        f"bit-exact migration (see MISMATCH/ERROR lines above)"
    )
