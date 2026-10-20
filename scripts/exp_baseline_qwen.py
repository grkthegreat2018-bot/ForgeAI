"""R&D baseline: Qwen2.5-0.5B (+Coder) — load time, VRAM, tok/s, NLL.

Usage: venv\\Scripts\\python.exe scripts\\exp_baseline_qwen.py [--coder]
"""
import glob, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

SNAP = glob.glob(os.path.expandvars(
    r"%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2.5%s-0.5B\snapshots\*"))
MODEL_KIND = "-Coder" if "--coder" in sys.argv else ""


def snap_dir(kind=""):
    hits = glob.glob(os.path.expandvars(
        rf"%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2.5{kind}-0.5B\snapshots\*"))
    return hits[0] if hits else None


def hf_nll(model_id, texts, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16).to(device).eval()
    nll_sum, n_tok = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.to(device)
            out = model(ids, labels=ids)
            nll_sum += out.loss.item() * (ids.shape[1] - 1)
            n_tok += ids.shape[1] - 1
    del model
    torch.cuda.empty_cache()
    return nll_sum / max(n_tok, 1)


def main():
    kind = MODEL_KIND
    snap = snap_dir(kind)
    assert snap, f"no snapshot for Qwen2.5{kind}-0.5B"
    model_id = f"Qwen/Qwen2.5{kind}-0.5B"
    print(f"[baseline] {model_id}  snap={snap}")

    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="test")
        texts = [t for t in ds["text"] if len(t) > 200][:16]
    except Exception as e:
        print(f"[baseline] wikitext load failed ({e}); using canned text")
        texts = ["The quick brown fox jumps over the lazy dog. " * 32]

    # HF reference NLL
    nll = hf_nll(model_id, texts)
    print(f"[baseline] HF NLL={nll:.4f} PPL={2.718281828 ** nll:.2f} "
          f"({len(texts)} docs)")

    # ForgeEngine path
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    from forge.engine.forge_engine import ForgeEngine
    eng = ForgeEngine.from_checkpoint(
        snap, config_name="qwen25_05b", device="cuda", quantize=None,
        kv_cache="standard", use_compile=False,
        use_breakable_cuda_graph=False, use_block_fusion=False)
    load_s = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[baseline] engine load={load_s:.2f}s peakVRAM={peak:.2f}GB")

    res = eng.benchmark("The quick brown fox", max_new_tokens=128, n_runs=5)
    res.update({"model": model_id, "load_s": load_s, "peak_vram_gb": peak,
                "hf_nll": nll})
    out = os.path.join(ROOT, ".devin", "exp_baseline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
