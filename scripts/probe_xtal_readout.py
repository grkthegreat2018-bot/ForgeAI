"""Probe the decoupling: does the network INTERNALLY compute successor
class for holdout triggers, or is the class knowledge confined to the
embedding geometry and never used in the forward pass?

Protocol:
  1. Train a tiny transformer on the v2 xtal grammar (members 8, d64) —
     same conditions as the v2m8d64 run where holdout embeddings clustered
     (0.42 own-class cos) yet heldout loss stayed at ceiling.
  2. Freeze. Capture per-layer hidden states at trigger positions in
     agree chains (train-member triggers) and holdout chains.
  3. Train linear probes per layer on train-trigger positions; test on
     held-out train-trigger AND holdout-trigger positions.
     - probe target A: successor class (the computation prediction needs)
     - probe target B: trigger's own class (pure geometry/accessibility)

Interpretation:
  - If successor-class probes succeed on holdout positions at depth:
    the network computes the answer internally but the readout doesn't
    use it — decoupling is final-stage only.
  - If they fail everywhere above the embedding layer: the memorized
    circuit bypasses class structure end-to-end; decoupling total.

Usage: python scripts/probe_xtal_readout.py [--steps 3000]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.exp.xtal_grammar import TAG_A, TAG_AH, XtalGrammar
from forge.exp.xtal_probe import batch_loss

CHANCE = 1.0 / 8.0


def capture_hidden(model, tokens, device):
    """Per-layer hidden states for a batch. Returns list of [B,T,d]
    tensors: index 0 = input embeddings, 1..L = block outputs."""
    with torch.no_grad():
        emb = model.embed(tokens)
        out = model(tokens, return_hidden_states=True)
    hs_list = out[2]
    layers = [emb.detach().float()]
    for h in hs_list:
        if isinstance(h, torch.Tensor) and h.dim() == 3:
            layers.append(h.detach().float())
    return layers


def collect_positions(model, toks, tags, want_tag, device, members=8):
    """For every position with tag==want_tag, gather the hidden state at
    the TRIGGER position (i-1 in token coords) per layer, plus the
    successor class and trigger class labels.

    Note: prediction of token i is made from hidden state at position
    i-1 — which IS the trigger token position in agree chains."""
    inp = toks[:, :-1]  # model input positions 0..T-1
    layers = capture_hidden(model, inp, device)
    xs_per_layer = [[] for _ in layers]
    succ_cls, trig_cls = [], []
    B = toks.shape[0]
    for b in range(B):
        for i in range(1, toks.shape[1]):
            if tags[b, i].item() != want_tag:
                continue
            trig_pos = i - 1  # token coords == input coords for pos < T
            if trig_pos < 1:
                continue
            for li, h in enumerate(layers):
                xs_per_layer[li].append(h[b, trig_pos])
            succ_cls.append(toks[b, i].item())
            trig_cls.append(toks[b, trig_pos].item())
    item0 = 16
    succ_cls = torch.tensor([(t - item0) // members for t in succ_cls], device=device)
    trig_cls = torch.tensor([(t - item0) // members for t in trig_cls], device=device)
    xs_per_layer = [torch.stack(x) if x else torch.zeros(0, 1, device=device)
                    for x in xs_per_layer]
    return xs_per_layer, succ_cls, trig_cls


def train_probe(Xtr, ytr, n_classes=8, steps=400, lr=1e-2):
    probe = torch.nn.Linear(Xtr.shape[1], n_classes).to(Xtr.device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(probe(Xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probe


def acc(probe, X, y):
    if len(X) == 0:
        return float("nan")
    with torch.no_grad():
        return float((probe(X).argmax(-1) == y).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--wd", type=float, default=0.3)
    ap.add_argument("--ruled", action="store_true",
                    help="train with rule-D in the mix + probe D-context positions")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device
    torch.manual_seed(0)

    from scripts.exp_xtal import build_model

    class MArgs:
        d_model, n_layers, n_heads = args.d_model, 4, 4
    MArgs.device = device

    mix = {"A": 0.45, "B": 0.05, "C": 0.2, "D": 0.15, "N": 0.15} if args.ruled else None
    grammar = XtalGrammar(seed=0, device=device, members=args.members, mix=mix)
    model = build_model(MArgs(), vocab=grammar.vocab)

    # ---- train (same protocol as v2 runs) ----
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3,
                            betas=(0.9, 0.95), weight_decay=args.wd)
    model.train()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = 3e-3 * min(1.0, (step + 1) / 100)
        toks, tags = grammar.gen_batch(64)
        loss = batch_loss(model, toks, tags)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"[probe-train] step {step} loss {float(loss):.3f}")
    model.eval()

    # quick replication sanity: heldout should be at ceiling
    ah_toks, ah_tags = grammar.gen_eval_batch("AH", batch_size=128)
    a_toks, a_tags = grammar.gen_eval_batch("A", batch_size=64)

    # ---- collect probe data ----
    xsA, succA, trigA = collect_positions(model, a_toks, a_tags, TAG_A, device,
                                          members=args.members)
    xsH, succH, trigH = collect_positions(model, ah_toks, ah_tags, TAG_AH, device,
                                          members=args.members)

    # rule-D context positions: hidden at the prediction position inside
    # D chunks (target -1), label = class of x (the copied class source)
    xsD, yD = None, None
    if args.ruled:
        from forge.exp.xtal_grammar import MARKER_D, TAG_D
        d_toks, d_tags = grammar.gen_eval_batch("D", batch_size=128)
        inp = d_toks[:, :-1]
        layers = capture_hidden(model, inp, device)
        per_layer = [[] for _ in layers]
        labels = []
        for b in range(d_toks.shape[0]):
            for j in range(d_toks.shape[1] - 4):
                if d_toks[b, j].item() == MARKER_D and d_tags[b, j + 4].item() == TAG_D:
                    x_tok = d_toks[b, j + 1].item()
                    for li, h in enumerate(layers):
                        per_layer[li].append(h[b, j + 3])
                    labels.append((x_tok - 16) // args.members)
        xsD = [torch.stack(x) if x else torch.zeros(0, 1, device=device)
               for x in per_layer]
        yD = torch.tensor(labels, device=device)
        print(f"[probe] ruleD-context positions={len(labels)}")
    n_layers = len(xsA)
    print(f"[probe] layers={n_layers} train-positions={len(succA)} "
          f"holdout-positions={len(succH)}")

    # split train-trigger positions 90/10
    n = len(succA)
    perm = torch.randperm(n, device=device)
    cut = int(n * 0.9)
    tr, te = perm[:cut], perm[cut:]

    hdr = f"{'layer':>6} {'succA_tr':>9} {'succA_te':>9} {'succHOLD':>9} {'trigA_te':>9} {'trigHOLD':>9}"
    if xsD is not None:
        hdr += f" {'ruleDctx':>9}"
    print(hdr + f"   (chance={CHANCE:.3f})")
    for li in range(n_layers):
        X = xsA[li]
        if len(X) == 0:
            continue
        Xtr, ytr = X[tr], succA[tr]
        probe_s = train_probe(Xtr, ytr)
        probe_t = train_probe(Xtr, trigA[tr])
        row = [
            acc(probe_s, Xtr, ytr),
            acc(probe_s, X[te], succA[te]),
            acc(probe_s, xsH[li], succH),
            acc(probe_t, X[te], trigA[te]),
            acc(probe_t, xsH[li], trigH),
        ]
        extra = ""
        if xsD is not None and len(xsD[li]) > 0:
            nd = len(yD)
            perm_d = torch.randperm(nd, device=device)
            cutd = int(nd * 0.9)
            dtr, dte = perm_d[:cutd], perm_d[cutd:]
            probe_d = train_probe(xsD[li][dtr], yD[dtr])
            extra = f" {acc(probe_d, xsD[li][dte], yD[dte]):9.3f}"
        name = "embed" if li == 0 else f"L{li}"
        print(f"{name:>6} " + " ".join(f"{v:9.3f}" for v in row) + extra)

    from forge.exp.harness import log_result
    # final-layer numbers as the headline
    log_result("probe_xtal_readout",
               steps=args.steps, members=args.members,
               succA_te=round(acc(probe_s, X[te], succA[te]), 4),
               succHOLD_final=round(acc(probe_s, xsH[n_layers - 1], succH), 4),
               trigHOLD_final=round(acc(probe_t, xsH[n_layers - 1], trigH), 4))


if __name__ == "__main__":
    main()
