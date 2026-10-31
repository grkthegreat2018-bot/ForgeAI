"""LFM3 port parity check — HF reference (transformers ≥5.12) vs ForgeAI port.

Run with the TRAIN env (transformers 5.20.x):
    venv_train\\Scripts\\python.exe scripts\\test_lfm3_port.py

Loads mondk/LFM3-1.7B-Claude-Code-Safetensors via its bundled
configuration_lfm3/modeling_lfm3 modules (no trust_remote_code needed —
files live in the checkpoint dir), runs one forward, then the ForgeAI port
on the same input, and reports max |Δlogit|.
"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "research" / "checkpoints" / "lfm3_1p7b_cc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_hf_module(name: str, path: Path):
    """Import a module file from the checkpoint dir by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DTYPE = torch.float32 if "--fp32" in sys.argv else torch.float16


def hf_model():
    """Load the HF LFM3 model from the checkpoint dir."""
    # The bundled modeling file does `from .configuration_lfm3 import ...`
    # — package-relative imports need a package context, so we create one.
    pkg_name = "lfm3_ckpt_pkg"
    import types
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(CKPT)]
        sys.modules[pkg_name] = pkg
    for mod_name in ("configuration_lfm3", "lfm3_conv", "modeling_lfm3"):
        qual = f"{pkg_name}.{mod_name}"
        spec = importlib.util.spec_from_file_location(qual, CKPT / f"{mod_name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qual] = mod
        spec.loader.exec_module(mod)
    cfg_mod = sys.modules[f"{pkg_name}.configuration_lfm3"]
    mdl_mod = sys.modules[f"{pkg_name}.modeling_lfm3"]
    cfg = cfg_mod.Lfm3Config()
    cfg._attn_implementation = "eager"
    model = mdl_mod.Lfm3ForCausalLM(cfg)
    from safetensors.torch import load_file
    state = load_file(str(CKPT / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected: {unexpected}"
    assert [k for k in missing if k != "lm_head.weight"] == [], missing
    return model.to(DEVICE).to(DTYPE).eval()


def forge_model():
    from forge.config import get_config
    from forge.model.loader import ModelLoader
    cfg = get_config("lfm3_1p7b", device=DEVICE)
    model = ModelLoader.build_model_fast(
        cfg, str(CKPT), dtype=DTYPE)
    return model.eval()


def main():
    torch.manual_seed(0)
    # "<|im_start|>user\nWrite fizzbuzz in Python.<|im_end|>\n<|im_start|>assistant\n"
    ids = torch.tensor([[124899, 5922, 207, 3838, 23639, 34806, 304, 3901,
                         13, 124900, 198, 124899, 100645, 198]],
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
