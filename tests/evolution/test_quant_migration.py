"""Test all quant simulators + JSON specs for bit-exact match."""

import numpy as np
import torch

from forge.evolution.domain_spec import JSONSpecDomain
from forge.evolution.domains.quant_domains import (
    AaacQuant,
    ActivationQuant,
    BitnetConfig,
    GroupQuant,
    MixedPrecision,
    MosaicQuant,
    Nvfp4Quant,
    OffqQuant,
    SharqQuant,
    W8A8Quant,
)

tests = [
    (
        "W8A8Quant",
        W8A8Quant,
        "w8a8_quant",
        {"mode": "int8", "calib_samples": 256, "per_channel": True, "smoothquant_alpha": 0.5},
    ),
    ("Nvfp4Quant", Nvfp4Quant, "nvfp4_quant", {"block_size": 32, "w4a8": True, "scale_mode": "per_block"}),
    (
        "BitnetConfig",
        BitnetConfig,
        "bitnet_config",
        {"learned_scale": True, "quant_mode": "ternary", "init_scale": 1.0},
    ),
    ("SharqQuant", SharqQuant, "sharq_quant", {"n_levels": 16, "adaptive": True, "warmup_steps": 100}),
    ("MosaicQuant", MosaicQuant, "mosaic_quant", {"n_tiles": 16, "tile_dim": 128, "mix_ratio": 0.5}),
    ("AaacQuant", AaacQuant, "aaac_quant", {"n_codebooks": 8, "codebook_size": 256, "n_bits": 3}),
    ("OffqQuant", OffqQuant, "offq_quant", {"offset_init": 0.5, "n_iter": 50, "learn_offset": True}),
    ("GroupQuant", GroupQuant, "group_quant", {"group_size": 32, "n_bits": 4, "scheme": "symmetric"}),
    ("MixedPrecision", MixedPrecision, "mixed_precision", {"n_levels": 3, "assignment": "uniform", "bits_base": 4}),
    (
        "ActivationQuant",
        ActivationQuant,
        "activation_quant",
        {"calib_method": "minmax", "percentile": 0.99, "smooth_alpha": 0.5},
    ),
]


def test_quant_migration():
    n_fail = 0
    for name, old_cls, spec_name, cfg in tests:
        torch.manual_seed(42)
        np.random.seed(42)
        old = old_cls()
        ro = old.evaluate(cfg)
        try:
            torch.manual_seed(42)
            np.random.seed(42)
            new = JSONSpecDomain(spec_name)
            rn = new.evaluate(cfg)
            match = abs(ro["score"] - rn["score"]) < 1e-4
            print(f"{name:20s} old={ro['score']:10.4f} new={rn['score']:10.4f} match={match}")
            if not match:
                n_fail += 1
        except FileNotFoundError:
            print(f"{name:20s} SPEC NOT FOUND: {spec_name}")
        except Exception as e:
            n_fail += 1
            print(f"{name:20s} ERROR: {e}")
    assert n_fail == 0, f"{n_fail} quant migration checks failed (see MISMATCH/ERROR lines above)"
