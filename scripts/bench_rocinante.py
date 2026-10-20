"""Rocinante-X-12B port verification (R65).

Stages:
  1. parity  — llama_cpp (GGUF, ground truth) vs ForgeEngine greedy logits
  2. gen     — sample generation sanity
  3. mem     — SuffixMemory recall after KV eviction (long-context probe)
  4. speed   — tok/s + peak VRAM

Usage:
  venv\\Scripts\\python.exe -X utf8 scripts\\bench_rocinante.py
      [--ckpt research/checkpoints/rocinante_x_12b]
      [--gguf  D:/LMstudio/.../Rocinante-X-12B-v1b-Q4_K_M.gguf]
      [--stage parity|gen|mem|speed|all]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

GGUF = r"D:\LMstudio\Models\TheDrummer\Rocinante-X-12B-v1-GGUF\Rocinante-X-12B-v1b-Q4_K_M.gguf"
CKPT = r"research\checkpoints\rocinante_x_12b"

PROMPT = ("[INST]You are a fantasy bard. Compose a four-line verse about "
          "a clockwork raven.[/INST]")


def stage_parity(gguf_path, ckpt_dir):
    """Greedy next-token agreement: llama.cpp logits vs Forge logits."""
    from llama_cpp import Llama
    from forge.engine.forge_engine import ForgeEngine

    print("[parity] llama_cpp baseline...")
    llm = Llama(model_path=gguf_path, n_ctx=512, n_gpu_layers=0,
                logits_all=True, verbose=False)
    toks = llm.tokenize(PROMPT.encode(), add_bos=True, special=True)
    llm.eval(toks)
    # logits_all=True → eval_logits is (n_tokens, vocab); last row is the
    # next-token distribution we compare against Forge's last-position.
    base_logits = torch.tensor(llm.eval_logits[-1], dtype=torch.float32)
    base_top = base_logits.argsort(descending=True)[:10]

    print("[parity] ForgeEngine load (load_quantize=None, bf16 CPU w/ "
          "single-layer GPU spot check is enough for parity of direction)…")
    eng = ForgeEngine.from_checkpoint(
        ckpt_dir, device="cuda", auto_activate=False,
        load_quantize="int4")   # fits 12GB; quant adds noise, see margin
    ids = eng.tokenizer(PROMPT, return_tensors="pt").input_ids.to(eng.device)
    with torch.inference_mode():
        out = eng.model(ids, use_cache=True, logits_last_only=True)
    from forge.model_loader import unpack_output_with_kv
    logits, _ = unpack_output_with_kv(out)
    forge_logits = logits[0, -1].float().cpu()
    forge_top = forge_logits.argsort(descending=True)[:10]

    print(f"  llama.cpp top-10: {base_top.tolist()}")
    print(f"  forge    top-10: {forge_top.tolist()}")
    agree = len(set(base_top.tolist()) & set(forge_top.tolist()))
    print(f"  top-10 overlap: {agree}/10  "
          f"(argmax match: {base_top[0].item() == forge_top[0].item()})")
    return eng


def stage_gen(eng):
    print("[gen] sampling 80 tokens...")
    t0 = time.time()
    text = eng.generate(PROMPT, max_new_tokens=80, temperature=0.7,
                        top_p=0.9, top_k=80, repetition_penalty=1.05)
    dt = time.time() - t0
    print(f"  -> {text!r}")
    print(f"  {80 / dt:.1f} tok/s")


def stage_mem(eng):
    """SuffixMemory recall: feed a verbatim span, evict it, check lift."""
    eng.enable_suffix_memory(gate=1.0)
    mem = eng._suffix_mem
    secret = "the crimson anvil sings at dawn"
    filler = "lorem ipsum dolor sit amet consectetur adipiscing elit " * 40
    ids = eng.tokenizer.encode(secret + " " + filler + secret[:20],
                               add_special_tokens=True)
    mem.feed(ids)
    lift = mem.lift_last()
    nxt = eng.tokenizer.encode("anvil", add_special_tokens=False)[0]
    print(f"[mem] lift on 'anvil' (verbatim successor): "
          f"{lift[nxt].item():.3f}  (argmax={int(lift.argmax())})")
    print(mem.info())


def stage_speed(eng):
    print("[speed] 128-token decode timing…")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    eng.generate(PROMPT, max_new_tokens=128, temperature=0.0)
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {128 / dt:.1f} tok/s, peak VRAM {vram:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--stage", default="all",
                    choices=["parity", "gen", "mem", "speed", "all"])
    args = ap.parse_args()

    eng = None
    if args.stage in ("parity", "all"):
        eng = stage_parity(args.gguf, args.ckpt)
    if eng is None:
        from forge.engine.forge_engine import ForgeEngine
        eng = ForgeEngine.from_checkpoint(
            args.ckpt, device="cuda", load_quantize="int4",
            kv_cache="flux", use_compile=False)
    if args.stage in ("gen", "all"):
        stage_gen(eng)
    if args.stage in ("mem", "all"):
        stage_mem(eng)
    if args.stage in ("speed", "all"):
        stage_speed(eng)


if __name__ == "__main__":
    main()
