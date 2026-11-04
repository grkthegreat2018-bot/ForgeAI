"""Spark-X2.6-1.7B port parity check — HF reference vs ForgeAI port.

Run:
    venv\\Scripts\\python.exe scripts\\test_spark_port.py

Loads XHToken/Spark-X2.6-1.7B from research/checkpoints/spark_x26_1p7b
via its bundled configuration_spark/modeling_spark modules, runs one
forward, then the ForgeAI port on the same input, and reports max |dlogit|.
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "research" / "checkpoints" / "spark_x26_1p7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if "--fp32" in sys.argv else torch.bfloat16


def hf_model():
    """Load the HF Spark2_5 model from the checkpoint dir."""
    pkg_name = "spark_ckpt_pkg"
    import types
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(CKPT)]
        sys.modules[pkg_name] = pkg
    for mod_name in ("configuration_spark", "modeling_spark"):
        qual = f"{pkg_name}.{mod_name}"
        spec = importlib.util.spec_from_file_location(qual, CKPT / f"{mod_name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qual] = mod
        spec.loader.exec_module(mod)
    cfg_mod = sys.modules[f"{pkg_name}.configuration_spark"]
    mdl_mod = sys.modules[f"{pkg_name}.modeling_spark"]
    import json
    cfg = cfg_mod.Spark2_5Config(**json.loads((CKPT / "config.json").read_text()))
    cfg._attn_implementation = "eager"
    model = mdl_mod.Spark2_5ForCausalLM(cfg)
    from safetensors.torch import load_file
    import json as _json
    index = _json.loads((CKPT / "model.safetensors.index.json").read_text())
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update(load_file(str(CKPT / shard)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected: {unexpected}"
    assert [k for k in missing if k != "lm_head.weight"] == [], missing
    return model.to(DEVICE).to(DTYPE).eval()


def forge_model():
    from forge.config import get_config
    from forge.model.loader import ModelLoader
    cfg = get_config("spark_x26_1p7b", device=DEVICE)
    model = ModelLoader.build_model_fast(cfg, str(CKPT), dtype=DTYPE)
    return model.eval()


def main():
    torch.manual_seed(0)
    ids = torch.tensor([[0, 1234, 5678, 9012, 34567, 890, 123, 456, 78901,
                         2345, 6789, 12, 345, 678, 901, 1]],
                       device=DEVICE)

    hf = hf_model()
    with torch.no_grad():
        hf_out = hf(ids)
    hf_logits = hf_out.logits.float()

    fm = forge_model()
    with torch.no_grad():
        f_out = fm(ids)
    f_logits = (f_out[0] if isinstance(f_out, tuple) else f_out).float()

    diff = (hf_logits - f_logits).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    top_hf = hf_logits[0, -1].topk(5).indices.tolist()
    top_fm = f_logits[0, -1].topk(5).indices.tolist()
    print(f"HF   logits {tuple(hf_logits.shape)}  top5={top_hf}")
    print(f"FORGE logits {tuple(f_logits.shape)}  top5={top_fm}")
    print(f"max|dlogit|={max_d:.6f}  mean={mean_d:.6f}")
    exact = max_d == 0.0
    close = max_d < 0.05
    print("BIT-EXACT" if exact else ("PASS (kernel-level fp diff)" if close
                                     else "FAIL — investigate"))
    return 0 if (exact or close) else 1


if __name__ == "__main__":
    sys.exit(main())
