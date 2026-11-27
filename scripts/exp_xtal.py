"""EXP-XTAL driver â€” rule crystallization dynamics on the synthetic grammar.

Trains a vanilla tiny transformer (ConfigurableResearchLLM via livet_config,
GDN2/Titan disabled) on XtalGrammar streams, logging per-rule loss
crystallization curves, gradient alignment between rules, and embedding
class-geometry over training.

Curve records append to results/xtal/<run>.jsonl; the run summary logs via
forge.exp.harness.log_result (.devin/exp_results.jsonl) per repo convention.

Usage:
  python scripts/exp_xtal.py --steps 3000 --run baseline
  python scripts/exp_xtal.py --steps 200 --run smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
torch.backends.cuda.matmul.fp32_precision = "tf32"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.exp.xtal_grammar import VOCAB, XtalGrammar
from forge.exp.xtal_probe import (embedding_geometry, eval_fixed_batches,
                                  grad_alignment, batch_loss)
from forge.exp.harness import log_result

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "xtal")


def build_model(args, vocab=None):
    if getattr(args, "arch", "tiny") == "qwen":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=vocab or VOCAB, hidden_size=896,
                          intermediate_size=4864, num_hidden_layers=24,
                          num_attention_heads=14, num_key_value_heads=2,
                          max_position_embeddings=512)
        model = Qwen2ForCausalLM(cfg)
        if getattr(args, "grad_ckpt", False):
            model.gradient_checkpointing_enable()
        model.to(device=args.device)
        return model
    from forge.model.live import livet_config
    from forge.model_loader import ConfigurableResearchLLM
    cfg = livet_config(
        vocab_size=vocab or VOCAB, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, max_seq_len=128, device=args.device,
        dtype="float32",
        use_gdn2=False, use_titan_memory=False,   # vanilla transformer
    )
    model = ConfigurableResearchLLM(cfg)
    from forge.training.runners.train_8b_all import (
        initialize_weights_, normalize_logit_scale_)
    initialize_weights_(model, args.n_layers)
    model.to(args.device)
    normalize_logit_scale_(model, torch.device(args.device), cfg)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--n-classes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--grad-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--ruled", action="store_true",
                    help="add rule-D (class-copy) chunks to the mix")
    ap.add_argument("--ruler", action="store_true",
                    help="add rule-R (relational composition) chunks to the mix")
    ap.add_argument("--ruler2", action="store_true",
                    help="split relational fraction between add-table R and xor-table R2")
    ap.add_argument("--arch", default="tiny", choices=["tiny", "qwen"],
                    help="qwen = random-init Qwen2-0.5B architecture (scale-transfer bridge)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing (qwen arch) — memory for larger batches")
    ap.add_argument("--mem", type=int, default=0,
                    help="bigram logit-bias sidecar buckets (pressure-valve test); 0=off")
    ap.add_argument("--recruit", action="store_true",
                    help="phase 1 (first half): no rule-A in mix (machinery-only pretrain); phase 2: full ruler mix")
    ap.add_argument("--gate", action="store_true",
                    help="PCGrad-style: project out negative-cosine components "
                         "between per-tag group gradients each step")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save resume checkpoint every N steps (0=off)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from results/xtal/<run>.ckpt.pt if present")
    ap.add_argument("--run", default="baseline")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    curve_path = os.path.join(OUT_DIR, f"{args.run}.jsonl")
    torch.manual_seed(args.seed)

    mix = None
    if args.ruled:
        mix = {"A": 0.45, "B": 0.05, "C": 0.2, "D": 0.15, "N": 0.15}
    elif args.ruler2:
        mix = {"A": 0.4, "B": 0.05, "C": 0.2, "R": 0.1, "R2": 0.1, "N": 0.15}
    elif args.ruler:
        mix = {"A": 0.4, "B": 0.05, "C": 0.2, "R": 0.2, "N": 0.15}
    if args.recruit:
        mix = {"C": 0.35, "R": 0.45, "N": 0.2}
    grammar = XtalGrammar(seed=args.seed, device=args.device,
                          n_classes=args.n_classes, members=args.members,
                          mix=mix)
    model = build_model(args, vocab=grammar.vocab)
    if args.arch == "qwen":
        class _Ad(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.base = m
                self.embed = m.model.embed_tokens
            def forward(self, x):
                return (self.base(x).logits,)
        model = _Ad(model)
        if getattr(args, "compile", True):
            model.base = torch.compile(model.base, mode="reduce-overhead")
    if args.mem > 0:
        V = grammar.vocab
        K = args.mem
        class _Mem(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.mem = torch.nn.Parameter(torch.zeros(K, V, device=args.device))
            def __getattr__(self, k):
                try:
                    return super().__getattr__(k)
                except AttributeError:
                    return getattr(self._modules["base"], k)
            def forward(self, x):
                logits = self.base(x)[0]
                x1 = torch.roll(x, 1, dims=1)
                x1[:, 0] = 0
                bucket = (x1.long() * V + x.long()) % K
                return (logits + self.mem[bucket],)
        model = _Mem(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[xtal] params={n_params/1e6:.2f}M device={args.device}")

    # fixed eval batches â€” identical across all runs
    kinds = ("A", "AH", "B", "C", "N")
    if args.ruled:
        kinds += ("D", "DH")
    if args.ruler:
        kinds += ("R", "RH")
    if args.ruler2:
        kinds += ("R", "RH", "R2", "R2H")
    eval_sets = {k: grammar.gen_eval_batch(k, batch_size=args.eval_batches)
                 for k in kinds}
    floors = grammar.floors()

    if args.arch == "qwen":
        from forge.training.training_utils import configure_optimizer
        opt = configure_optimizer(model, args.lr, args.wd, optimizer_name="bnb")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.95), weight_decay=args.wd,
                                fused=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr

    # pause/resume: full training state — weights, optimizer, grammar RNG,
    # step counter. Grammar batches are reproducible given the generator
    # state, so a resumed run continues the identical data stream.
    ckpt_path = os.path.join(OUT_DIR, f"{args.run}.ckpt.pt")
    start_step = 0
    tokens_seen0 = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=args.device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        grammar._train_rng.set_state(ck["train_rng"].cpu())
        start_step = ck["step"] + 1
        tokens_seen0 = ck["tokens_seen"]
        print(f"[xtal] resumed from step {start_step}")

    def save_ckpt(step, tokens_seen):
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "train_rng": grammar._train_rng.get_state(),
            "step": step,
            "tokens_seen": tokens_seen,
        }, ckpt_path)

    curve = open(curve_path, "a")

    def emit(step, rec):
        rec["step"] = step
        rec["run"] = args.run
        curve.write(json.dumps(rec) + "\n")
        curve.flush()

    t0 = time.time()
    tokens_seen = tokens_seen0
    model.train()
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if args.recruit and step == args.steps // 2:
            grammar.mix = {"A": 0.4, "B": 0.05, "C": 0.2, "R": 0.2, "N": 0.15}
            print(f"[xtal] step {step}: recruitment phase 2 — rule-A enters mix")
        toks, tags = grammar.gen_batch(args.batch)
        opt.zero_grad(set_to_none=True)
        if args.gate:
            # per-tag group losses sharing one forward; PCGrad-project pairs
            # with negative cosine, then apply the summed modified gradient
            from forge.exp.xtal_grammar import (TAG_A, TAG_B, TAG_C, TAG_D,
                                                TAG_R, TAG_NOISE)
            tags_present = [TAG_A, TAG_B, TAG_C, TAG_NOISE]
            if args.ruled:
                tags_present.append(TAG_D)
            if args.ruler:
                tags_present.append(TAG_R)
            logits = model(toks[:, :-1])[0]
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                toks[:, 1:].reshape(-1), reduction="none").view(toks.size(0), -1)
            npos = ce.numel()
            params = [p for p in model.parameters() if p.requires_grad]
            group_grads = {}
            loss = torch.zeros((), device=toks.device)
            for i, tg in enumerate(tags_present):
                mask = (tags[:, 1:] == tg).float()
                lg = (ce * mask).sum() / npos
                loss = loss + lg.detach()
                retain = i < len(tags_present) - 1
                gs = torch.autograd.grad(lg, params, retain_graph=retain,
                                         allow_unused=True)
                group_grads[tg] = [g.detach().clone() if g is not None
                                   else torch.zeros_like(p)
                                   for g, p in zip(gs, params)]
            # PCGrad: for each pair with negative cosine, remove the
            # conflicting component from both directions (symmetric variant)
            keys = list(group_grads)
            for i in range(len(keys)):
                for j in range(len(keys)):
                    if i == j:
                        continue
                    gi = group_grads[keys[i]]
                    gj = group_grads[keys[j]]
                    dot = sum((a * b).sum() for a, b in zip(gi, gj))
                    nj = sum((b * b).sum() for b in gj).clamp_min(1e-12)
                    if dot < 0:
                        for a, b in zip(gi, gj):
                            a.sub_((dot / nj) * b)
            with torch.no_grad():
                for k, p in enumerate(params):
                    p.grad = sum(group_grads[tg][k] for tg in keys).clone()
        else:
            if args.arch == "qwen":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = batch_loss(model, toks, tags)
            else:
                loss = batch_loss(model, toks, tags)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += toks.numel()

        if args.ckpt_every and step % args.ckpt_every == 0:
            save_ckpt(step, tokens_seen)
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            if args.arch == "qwen":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    res = eval_fixed_batches(model, eval_sets)
            else:
                res = eval_fixed_batches(model, eval_sets)
            rec = {"kind": "eval", "train_loss": float(loss.detach())}
            for k, v in res.items():
                rec[f"eval/{k}"] = round(v, 4)
            # excess loss above analytic floor for the governed tags
            from forge.exp.xtal_grammar import (TAG_A, TAG_AH, TAG_B, TAG_C,
                                                TAG_D, TAG_R, TAG_R2)
            for tag, key in ((TAG_A, "A/ruleA"), (TAG_AH, "AH/ruleA_heldout"),
                             (TAG_B, "B/ruleB"), (TAG_C, "C/ruleC"),
                             (TAG_D, "D/ruleD"), (TAG_D, "DH/ruleD"),
                             (TAG_R, "R/ruleR"), (TAG_R, "RH/ruleR"),
                             (TAG_R2, "R2/ruleR"), (TAG_R2, "R2H/ruleR")):
                if key in res:
                    rec[f"xs/{key}"] = round(res[key] - floors[tag], 4)
            emit(step, rec)
            model.train()

        if step % args.grad_every == 0 or step == args.steps - 1:
            model.eval()
            ga = grad_alignment(model, {k: eval_sets[k] for k in ("A", "C", "N")})
            eg = embedding_geometry(model, n_classes=grammar.n_classes,
                                    members=grammar.members)
            rec = {"kind": "grad"}
            rec.update({k: round(v, 5) for k, v in ga.items()})
            rec.update({k: round(v, 5) for k, v in eg.items()})
            emit(step, rec)
            model.train()

    dt = time.time() - t0
    final = eval_fixed_batches(model, eval_sets)
    summary = dict(
        run=args.run, steps=args.steps, params=n_params,
        tokens=tokens_seen, wall_s=round(dt, 1),
        tok_s=int(tokens_seen / max(dt, 1e-9)),
        final_A=round(final["A/ruleA"], 4),
        final_AH=round(final.get("AH/ruleA_heldout", float("nan")), 4),
        final_B=round(final.get("B/ruleB", float("nan")), 4),
        final_C=round(final.get("C/ruleC", float("nan")), 4),
        final_D=round(final.get("D/ruleD", float("nan")), 4),
        final_DH=round(final.get("DH/ruleD", float("nan")), 4),
        final_R=round(final.get("R/ruleR", float("nan")), 4),
        final_RH=round(final.get("RH/ruleR", float("nan")), 4),
        final_N=round(final["N"], 4),
    )
    log_result("exp_xtal", **summary)
    print(f"[xtal] done in {dt:.0f}s â€” curve at {curve_path}")


if __name__ == "__main__":
    main()
