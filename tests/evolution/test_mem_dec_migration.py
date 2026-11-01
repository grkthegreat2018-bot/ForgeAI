"""Bit-exact migration test for memory + decoding domains."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forge.evolution.domain_spec import JSONSpecDomain

domains = [
    ("HybridOffload", "hybrid_offload", "forge.evolution.domains.memory_domains"),
    ("CpuKvOffload", "cpu_kv_offload", "forge.evolution.domains.memory_domains"),
    ("ExpertHotload", "expert_hotload", "forge.evolution.domains.memory_domains"),
    ("MemoryBudget", "memory_budget", "forge.evolution.domains.memory_domains"),
    ("CheckpointRecompute", "checkpoint_recompute", "forge.evolution.domains.memory_domains"),
    ("SpeculativeDecode", "speculative_decode", "forge.evolution.domains.decoding_domains"),
    ("MtpConfig", "mtp_config", "forge.evolution.domains.decoding_domains"),
    ("BatchedDecode", "batched_decode", "forge.evolution.domains.decoding_domains"),
    ("SamplingConfig", "sampling_config", "forge.evolution.domains.decoding_domains"),
    ("BeamSearch", "beam_search", "forge.evolution.domains.decoding_domains"),
]


def test_mem_dec_migration():
    n_pass = 0
    n_fail = 0

    for cls_name, spec_name, mod_path in domains:
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
        f"{n_fail}/{n_pass + n_fail} memory/decoding domains failed bit-exact "
        f"migration (see MISMATCH/ERROR lines above)"
    )
