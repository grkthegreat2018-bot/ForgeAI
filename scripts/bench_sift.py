"""Bench SIFT vs manual-LoRA SFT on the spark_x26_1p7b base.

Measures, on identical real SFT batches (spark_sft_v9 corpus):
  - params touched (trainable entries) + % of model
  - chosen frontier + block importance profile
  - per-step wall time (fwd+bwd+opt), peak VRAM
  - loss trajectory (first/last loss — SIFT starts lossless so first-step
    loss must match the LoRA baseline within fp noise)
  - prefix-cache epoch-2 step time (repeat views skip the frozen prefix)

Run:
  venv/Scripts/python scripts/bench_sift.py --steps 12 --n-data 192
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.config import get_config
from forge.model.loader import ModelLoader
from forge.training.runners import sft_train as S
from forge.training import sift as sift_mod
from forge.training.bitnet_lora import add_lora_adapters
from research.tokenizer_cache import get_tokenizer

CKPT = "research/checkpoints/spark_x26_1p7b"
DATA = "data/sft/spark_sft_v9.jsonl"


def build_batches(tokenizer, n_rows, bs, seq_len, device, seed=0):
    S._CHAT_FORMAT = "spark"
    import random
    rows = S.load_examples([DATA])[:n_rows]
    dataset = []
    for ex in rows:
        dataset.extend(S.tokenize_example(ex, tokenizer, seq_len))
    random.Random(seed).shuffle(dataset)
    batches = []
    for i in range(0, len(dataset) - bs + 1, bs):
        batches.append(S.collate_batch(
            dataset[i:i + bs], pad_id=2, device=device))
    return dataset, batches


def build_model(device):
    cfg = get_config("spark_x26_1p7b", device=device)
    cfg.use_gradient_checkpointing = True
    cfg.selective_gradient_checkpointing = "all"
    cfg.use_chunked_ce = True
    cfg.ce_chunk_size = 128
    return ModelLoader.build_model_fast(cfg, checkpoint_path=CKPT,
                                        dtype=torch.bfloat16)


def ce_loss(model, ids, labels, mask):
    return S.compute_loss(model, ids, labels, mask)


def run_steps(model, optimizer, batches, n_steps, loss_out,
              pre=None, post=None):
    model.train()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(n_steps):
        ids, labels, mask, _, _ = batches[i % len(batches)]
        if pre is not None:
            pre(ids, mask)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = ce_loss(model, ids, labels, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if post is not None:
            post()
        loss_out.append(float(loss.detach().float()))
    return times, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--n-data", type=int, default=192)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--attn-units", type=int, default=4)
    ap.add_argument("--ffn-units", type=int, default=8)
    ap.add_argument("--attn-gran", type=int, default=128)
    ap.add_argument("--ffn-gran", type=int, default=32)
    ap.add_argument("--frontier-mass", type=float, default=0.75)
    ap.add_argument("--newton-passes", type=int, default=0)
    ap.add_argument("--newton-lr", type=float, default=1.0)
    ap.add_argument("--score-mode", default="fro",
                    choices=["fro", "smean", "obs"])
    ap.add_argument("--skip-lora", action="store_true")
    ap.add_argument("--skip-pc", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    tokenizer = get_tokenizer(CKPT)
    dataset, batches = build_batches(tokenizer, args.n_data, args.bs,
                                     args.seq, device)
    print(f"Data: {len(dataset)} examples -> {len(batches)} batches "
          f"(bs{args.bs} x seq<={args.seq})")
    n_model_params = None
    results = {}

    # ── LoRA baseline (canonical recipe: manual lora r32 a64) ──
    if not args.skip_lora:
        model = build_model(device)
        n_model_params = sum(p.numel() for p in model.parameters())
        n_ad, lora_p = add_lora_adapters(
            model, rank=32, alpha=64,
            target_modules=["qkv_proj", "out_proj", "w_gate", "w_up", "w_down"])
        ids = {id(p) for p in lora_p}
        for p in model.parameters():
            if id(p) not in ids:
                p.requires_grad_(False)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(lora_p, lr=args.lr, fused=True)
        losses = []
        run_steps(model, opt, batches, 2, losses)  # warmup
        losses = []
        times, vram = run_steps(model, opt, batches, args.steps, losses)
        results["lora"] = dict(step_ms=1000 * sum(times) / len(times),
                               vram_gb=vram, touched=n_train,
                               loss0=losses[0], lossN=losses[-1],
                               n_adapters=n_ad)
        del model, opt
        torch.cuda.empty_cache()

    # ── SIFT ──
    model = build_model(device)
    n_model_params = sum(p.numel() for p in model.parameters())

    def calib_iter():
        yield from ((b[0], b[1], b[2]) for b in batches)

    st = sift_mod.setup_sift(
        model, calib_iter, lambda i, l, m: ce_loss(model, i, l, m),
        attn_units=args.attn_units, ffn_units=args.ffn_units,
        attn_gran=args.attn_gran, ffn_gran=args.ffn_gran,
        frontier="auto", frontier_mass=args.frontier_mass, calib_batches=4,
        score_mode=args.score_mode,
        newton_passes=args.newton_passes, newton_lr=args.newton_lr)
    opt = torch.optim.AdamW(st.delta_params, lr=args.lr, fused=True)
    bm = st.block_mass
    results["sift_profile"] = {
        "frontier": st.frontier,
        "block_mass": {b: round(m, 4) for b, m in sorted(bm.items())},
    }
    losses = []
    run_steps(model, opt, batches, 2, losses)
    losses = []
    times, vram = run_steps(model, opt, batches, args.steps, losses)
    results["sift"] = dict(step_ms=1000 * sum(times) / len(times),
                           vram_gb=vram, touched=st.n_touched,
                           loss0=losses[0], lossN=losses[-1],
                           frontier=st.frontier,
                           n_slices=len(st.selection))
    del model, opt, st
    torch.cuda.empty_cache()

    # ── SIFT + prefix cache: measure epoch-1 vs epoch-2 step time ──
    if not args.skip_pc:
        model = build_model(device)
        st = sift_mod.setup_sift(
            model, calib_iter, lambda i, l, m: ce_loss(model, i, l, m),
            attn_units=args.attn_units, ffn_units=args.ffn_units,
            attn_gran=args.attn_gran, ffn_gran=args.ffn_gran,
            frontier="auto", frontier_mass=args.frontier_mass,
            calib_batches=4, prefix_cache_gb=6.0,
            score_mode=args.score_mode,
            newton_passes=args.newton_passes, newton_lr=args.newton_lr)
        opt = torch.optim.AdamW(st.delta_params, lr=args.lr, fused=True)
        losses = []
        run_steps(model, opt, batches, 2, losses)  # warmup
        losses = []
        times, vram = run_steps(
            model, opt, batches, args.steps, losses,
            pre=lambda i, m: st.prepare_prefix(i, m, device),
            post=lambda: st.clear_prefix())
        ep1 = 1000 * sum(times) / len(times)
        times2, vram2 = run_steps(
            model, opt, batches, args.steps, losses,
            pre=lambda i, m: st.prepare_prefix(i, m, device),
            post=lambda: st.clear_prefix())
        ep2 = 1000 * sum(times2) / len(times2)
        pc = st.prefix_cache
        results["sift_pc"] = dict(step_ms_ep1=ep1, step_ms_ep2=ep2,
                                  vram_gb=vram, fills=pc.fills, hits=pc.hits,
                                  cache_mb=pc.bytes / 1e6,
                                  loss0=losses[0], lossN=losses[-1],
                                  touched=st.n_touched)
        del model, opt, st
        torch.cuda.empty_cache()

    # ── report ──
    print("\n" + "=" * 68)
    print(f"SIFT bench on spark_x26_1p7b ({n_model_params/1e9:.2f}B params) — "
          f"{args.steps} timed steps, bs{args.bs}, lr {args.lr}")
    print("=" * 68)
    if "sift_profile" in results:
        sp = results["sift_profile"]
        print(f"frontier picked: block {sp['frontier']}/28")
        tops = sorted(sp["block_mass"].items(), key=lambda kv: -kv[1])[:8]
        print("block mass (top8): "
              + ", ".join(f"L{b}={m:.3f}" for b, m in tops))
    for k in ("lora", "sift", "sift_pc"):
        if k not in results:
            continue
        r = results[k]
        if k == "lora":
            print(f"\n[{k}] {r['n_adapters']} adapters")
            print(f"  touched: {r['touched']/1e6:.2f}M "
                  f"({100*r['touched']/n_model_params:.2f}% of model)")
            print(f"  step: {r['step_ms']:.0f} ms | peak VRAM {r['vram_gb']:.2f} GB")
            print(f"  loss: {r['loss0']:.4f} -> {r['lossN']:.4f}")
        elif k == "sift":
            print(f"\n[{k}] {r['n_slices']} slices, frontier={r['frontier']}")
            print(f"  touched: {r['touched']/1e6:.2f}M "
                  f"({100*r['touched']/n_model_params:.2f}% of model)")
            print(f"  step: {r['step_ms']:.0f} ms | peak VRAM {r['vram_gb']:.2f} GB")
            print(f"  loss: {r['loss0']:.4f} -> {r['lossN']:.4f}")
        else:
            print(f"\n[{k}] frontier-cached prefix")
            print(f"  step ep1 (fill): {r['step_ms_ep1']:.0f} ms | "
                  f"ep2 (hits): {r['step_ms_ep2']:.0f} ms")
            print(f"  cache: {r['cache_mb']:.0f} MB, "
                  f"fills={r['fills']} hits={r['hits']}")
            print(f"  loss: {r['loss0']:.4f} -> {r['lossN']:.4f}")
    if "lora" in results and "sift" in results:
        d = 100 * (1 - results["sift"]["step_ms"] / results["lora"]["step_ms"])
        print(f"\nSIFT vs LoRA step time: {d:+.1f}%")
        dl = results["sift"]["lossN"] - results["lora"]["lossN"]
        print(f"loss delta at step {args.steps}: {dl:+.4f}")


if __name__ == "__main__":
    main()
