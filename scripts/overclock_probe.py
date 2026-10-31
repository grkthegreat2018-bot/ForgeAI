"""Overclock discovery probes — measure headroom before proposer_v2.

E1: decode tok/s vs bandwidth ceiling (proposer_mla_moe, bf16)
E2: per-layer top-1 accuracy + agreement-vs-final on code (early-exit viability)
E3: ngram-lookahead acceptance simulation on held-out code (zero-train draft)
E4: vocab hot-token coverage (lm_head quant case)

Run: python scripts/overclock_probe.py [--tokens 96]
"""
import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROPOSER = ROOT / "forge" / "evolution" / "proposer_mla_moe"


def load_engine():
    from forge.engine.forge_engine import ForgeEngine
    t0 = time.time()
    eng = ForgeEngine.from_checkpoint(str(PROPOSER), device="cuda")
    print(f"[load] {time.time()-t0:.1f}s")
    return eng


def e1_decode_speed(eng, max_new=96):
    prompt = "def quicksort(arr):\n    \"\"\"Sort a list.\"\"\"\n"
    try:
        eng.generate(prompt, max_new_tokens=8, temperature=0.0)
    except TypeError:
        eng.generate(prompt, max_new_tokens=8)
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        out = eng.generate(prompt, max_new_tokens=max_new, temperature=0.0)
    except TypeError:
        out = eng.generate(prompt, max_new_tokens=max_new)
    torch.cuda.synchronize()
    dt = time.time() - t0
    model_bytes = sum(p.numel() * p.element_size() for p in eng.model.parameters())
    tps = max_new / dt
    eff_bw = model_bytes * tps / 1e9
    spec_bw = 672.0  # RTX 5070 GDDR7 ~672 GB/s
    print(f"[E1] decode {tps:.1f} tok/s | weights {model_bytes/1e9:.2f}GB | "
          f"effective BW {eff_bw:.0f} GB/s vs ~{spec_bw:.0f} spec "
          f"({eff_bw/spec_bw*100:.0f}% of roofline)")
    print(f"     out: {out[:120]!r}")
    return tps


def e2_layer_exit(eng, test_text, n_tok=256):
    tok = eng.tokenizer
    ids = tok(test_text, return_tensors="pt").input_ids[:, :n_tok].cuda()
    m = eng.model
    outs = {}
    hooks = []
    for i, blk in enumerate(m.blocks):
        def mk(j):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                outs[j] = o.detach()
            return h
        hooks.append(blk.register_forward_hook(mk(i)))
    with torch.no_grad():
        m(ids[:, :-1])
    for h in hooks:
        h.remove()
    H = torch.stack([outs[i] for i in sorted(outs)])  # [L,1,T,D]
    L = H.shape[0]
    norm = m.ln_f if m.ln_f is not None else (lambda x: x)
    truth = ids[:, 1:]
    print(f"[E2] {L} layers, T={truth.shape[1]}")
    print("     layer | top1-acc | agree-final")
    for i in range(L):
        logits = m.head(norm(H[i]))
        am = logits.argmax(-1)
        acc = (am == truth).float().mean().item()
        fin = m.head(norm(H[-1])).argmax(-1)
        agr = (am == fin).float().mean().item()
        if i % 3 == 0 or i >= L - 3:
            print(f"     {i:5d} |  {acc:.3f}  |   {agr:.3f}")
    return H


def e3_ngram_lookahead(train_ids, test_ids, orders=(2, 3, 4, 6, 8), kmax=8):
    tab = {o: defaultdict(Counter) for o in orders}
    for o in orders:
        for i in range(len(train_ids) - o):
            tab[o][tuple(train_ids[i:i + o])][train_ids[i + o]] += 1
    accepts = []
    i = 0
    while i < len(test_ids) - 1:
        chain, ctx = [], list(test_ids[max(0, i - 32):i + 1])
        for _ in range(kmax):
            pred = None
            for o in sorted(orders, reverse=True):
                key = tuple(ctx[-o:])
                if key in tab[o] and tab[o][key]:
                    pred = tab[o][key].most_common(1)[0][0]
                    break
            if pred is None:
                break
            chain.append(pred)
            ctx.append(pred)
        acc = 0
        for j, p in enumerate(chain):
            if i + 1 + j < len(test_ids) and p == test_ids[i + 1 + j]:
                acc += 1
            else:
                break
        accepts.append(acc + 1)  # +1 = the verified token itself
        i += 1
    import statistics as st
    mean = st.mean(accepts)
    print(f"[E3] ngram-lookahead: mean {mean:.2f} tok/forward "
          f"(median {st.median(accepts)}, p90 {sorted(accepts)[int(len(accepts)*.9)]}, "
          f"max {max(accepts)}) over {len(accepts)} positions")
    return mean


def e4_vocab_coverage(all_ids, vocab_size):
    c = Counter(all_ids)
    tot = sum(c.values())
    print(f"[E4] unique tokens {len(c)}/{vocab_size}")
    cum = 0
    for n in (256, 1024, 4096, 16384):
        cum = sum(v for _, v in c.most_common(n))
        print(f"     top-{n:5d} rows cover {cum/tot*100:.1f}% of tokens")


def e5_moe_routed(eng, test_ids, n_tok=256):
    """dense_bypass mean-of-4 vs routed top-k: is the MoE natively routed?

    If the port carried a trained router, dense_bypass is actively degrading
    quality AND spending 4x FFN compute — routed top-k would win both ways.
    """
    from forge.moe.moe import MoELayer
    m = getattr(eng.model, "_orig_mod", eng.model)
    ids = torch.as_tensor(test_ids[:n_tok], dtype=torch.long).unsqueeze(0).cuda()
    moes = [blk.ffn for blk in m.blocks if isinstance(blk.ffn, MoELayer)]
    if not moes:
        print("[E5] no MoE layers")
        return
    g = moes[0].router.gate.weight.detach().float()
    gn = g.norm(dim=-1)
    print(f"[E5] {len(moes)} MoE layers, n_experts={moes[0].n_experts}, "
          f"top_k={moes[0].top_k}, gate row-norm {gn.mean():.3f}+-{gn.std():.3f}")

    # Per-layer routing stats (capture gating weights on real tokens)
    gw_seen = {}
    hks = []
    def mk(i):
        def h(mod, inp, out):
            gw_seen[i] = out[1].detach()  # (N, n_experts)
        return h
    for i, moe in enumerate(moes):
        hks.append(moe.router.register_forward_hook(mk(i)))

    def _fwd():
        out = m(ids)
        if isinstance(out, tuple):
            return out[0]
        return out.logits if hasattr(out, "logits") else out

    def _nll(lg):
        lp = F.log_softmax(lg[:, :-1].float(), dim=-1)
        return -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).mean().item()

    results = {}
    with torch.no_grad():
        logits_mean = _fwd()  # dense_bypass as loaded
        results["mean-of-4"] = (_nll(logits_mean), logits_mean)
        for moe in moes:
            moe.dense_bypass = False
        for k in (4, 2, 1):
            for moe in moes:
                moe.router.top_k = moe.top_k = k
            lg = _fwd()
            results[f"routed-k{k}"] = (_nll(lg), lg)
        for moe in moes:  # restore
            moe.dense_bypass = True
            moe.router.top_k = moe.top_k = 4
    for h in hks:
        h.remove()

    base_nll, base_lg = results["mean-of-4"]
    print(f"     mean-of-4 NLL {base_nll:.4f}  (baseline)")
    for name, (nll, lg) in results.items():
        if name == "mean-of-4":
            continue
        agree = (base_lg.argmax(-1) == lg.argmax(-1)).float().mean().item()
        print(f"     {name:9s} NLL {nll:.4f} (d={nll-base_nll:+.4f}) "
              f"argmax-agree {agree*100:.1f}%")

    # routing concentration: collapsed or discriminative? (k=4 capture)
    for i in (0, len(moes) // 2, len(moes) - 1):
        gw = gw_seen.get(i)
        if gw is None:
            continue
        ent = -(gw.clamp_min(1e-9) * gw.clamp_min(1e-9).log()).sum(-1).mean().item()
        print(f"     L{i:02d} gating entropy {ent:.3f} (uniform-4={1.386:.3f})")


def e6_batched_static(eng, n_tok=96):
    """Batched static-graph decode vs batched functional — greedy equality
    + aggregate throughput. First call lazily captures the batch graphs
    (excluded from timing via a warmup run)."""
    prompts = [
        "def fibonacci(n):\n    \"\"\"Return nth fib.\"\"\"\n",
        "Write a python function to reverse a linked list.\n",
        "# TODO: implement binary search over a sorted array\n",
        "class LRUCache:\n    def __init__(self, capacity):\n",
    ]
    B = len(prompts)
    try:
        eng.generate_batch(prompts, max_new_tokens=16, temperature=0.0)
    except Exception as e:
        print(f"[E6] warmup batch failed: {e}")
    bs_map = (getattr(eng.model, '_decode_state', None) or {}).get('batch', {})
    captured = {b: bool(s) for b, s in bs_map.items()}

    torch.cuda.synchronize()
    t0 = time.time()
    outs_s = eng.generate_batch(prompts, max_new_tokens=n_tok,
                                temperature=0.0)
    torch.cuda.synchronize()
    dt_s = time.time() - t0

    m = eng.model
    saved = getattr(m, '_ensure_static_batch', None)
    m._ensure_static_batch = lambda *a, **k: None
    try:
        torch.cuda.synchronize()
        t0 = time.time()
        outs_f = eng.generate_batch(prompts, max_new_tokens=n_tok,
                                    temperature=0.0)
        torch.cuda.synchronize()
        dt_f = time.time() - t0
    finally:
        m._ensure_static_batch = saved

    agree = sum(a == b for a, b in zip(outs_s, outs_f))
    toks = n_tok * B
    print(f"[E6] batch={B} static {toks/dt_s:.0f} tok/s agg vs functional "
          f"{toks/dt_f:.0f} tok/s | exact-match {agree}/{B} | "
          f"batch-states {captured}")
    for i, (a, b) in enumerate(zip(outs_s, outs_f)):
        if a != b:
            # first divergence point
            j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]),
                     min(len(a), len(b)))
            print(f"     row{i} diverges @char{j}: {a[j:j+40]!r} vs {b[j:j+40]!r}")


def kv_math(cfg):
    l, dc = cfg.n_layers, 256
    gla_kv = l * dc * 2 * 2  # d_c K and V, fp16
    gqa_kv = l * cfg.n_kv_heads * (cfg.d_model // cfg.n_heads) * 2 * 2
    print(f"[KV] GLA latent {gla_kv} B/tok vs GQA {gqa_kv} B/tok "
          f"({gqa_kv/gla_kv:.0f}x smaller)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=96)
    args = ap.parse_args()

    files = [ROOT / "forge" / "evolution" / "code_evolve.py",
             ROOT / "forge" / "training" / "runners" / "sft_train.py",
             ROOT / "forge" / "model" / "llm.py"]
    test_file = ROOT / "forge" / "engine" / "engine_lora.py"

    eng = load_engine()
    m = eng.model
    print(f"[model] {sum(p.numel() for p in m.parameters())/1e6:.0f}M params, "
          f"{m.config.n_layers} layers, d={m.config.d_model}")
    kv_math(m.config)

    import traceback
    for name, fn in [
        ("E1 decode", lambda: e1_decode_speed(eng, args.tokens)),
        ("E2 layer-exit", lambda: e2_layer_exit(
            eng, test_file.read_text(encoding="utf-8", errors="ignore"))),
    ]:
        try:
            fn()
        except Exception:
            print(f"[{name}] FAILED"); traceback.print_exc(limit=2)

    tok = eng.tokenizer
    test_text = test_file.read_text(encoding="utf-8", errors="ignore")
    train_ids = []
    for f in files:
        train_ids += tok(f.read_text(encoding="utf-8", errors="ignore")).input_ids
    test_ids = tok(test_text).input_ids
    e3_ngram_lookahead(train_ids, test_ids)
    e4_vocab_coverage(train_ids + test_ids, m.config.vocab_size)
    try:
        e5_moe_routed(eng, test_ids)
    except Exception:
        print("[E5 routed] FAILED"); traceback.print_exc(limit=2)
    try:
        e6_batched_static(eng, args.tokens)
    except Exception:
        print("[E6 batched] FAILED"); traceback.print_exc(limit=2)


if __name__ == "__main__":
    main()
