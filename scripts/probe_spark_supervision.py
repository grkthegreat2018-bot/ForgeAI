"""Spark supervision probe — what does 20-step LoRA-SFT actually need?

Uses PRODUCTION loss paths where they exist (S.compute_loss ->
chunked_linear_cross_entropy / chunked_entropy_weighted_ce) and custom
autograd Functions cloned from the same manual-grad pattern for the
non-production arms (focal, confident-drop, top-k teacher KL) — no
[N,V] autograd graphs anywhere. Label shift follows sft_train:
hidden[t] predicts labels[t+1].

One subprocess per arm (VRAM isolation):
  ce        production chunked-CE baseline
  focal     focal CE gamma=4.93 (single-pass, own-logit weighting)
  ent05     production entropy-weighted CE alpha=0.5
  band      production CE + batch-skip outside mean-CE [0.4,6.0]
  confdrop  drop supervised tokens whose own CE < 0.3 (single-pass)
  anslast   keep only last 16 supervised tokens/row
  seq512    plain CE at seq_len=512
  full_kl   forward-KL to teacher (sft10_src) full-vocab dist
  topk256   forward-KL renorm'd on teacher top-256, narrow-head grad
  topk256ce topk256 + 0.3*CE on gold
  topk15ce  top-15 KL + 0.3*CE (paper's extreme-support claim, OUR model)
  massprobe no training: teacher top-k cumulative mass + gold-in-topk

Each arm: fresh model, LoRA-r32 (working targets), AdamW fused lr2e-5,
20 timed steps bs2 x seq1024, heldout CE on disjoint rows.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from forge.config import get_config
from forge.model.loader import ModelLoader
from forge.training.runners import sft_train as S
from forge.training.bitnet_lora import add_lora_adapters
from research.tokenizer_cache import get_tokenizer

STU_CKPT = "research/checkpoints/spark_x26_1p7b"
TEA_CKPT = "research/checkpoints/spark_x26_1p7b.sft10_src"
DATA = "data/sft/spark_sft_v9.jsonl"
LORA_TARGETS = ["qkv_proj", "out_proj", "w_gate", "w_up", "w_down"]
N_TRAIN, N_EVAL, BS, SEQ, STEPS, LR = 256, 48, 2, 1024, 20, 2e-5


# ── custom chunked Functions (same manual-grad pattern as production) ──

class _ChunkedFocalOrDrop(torch.autograd.Function):
    """Single-pass weighted CE; weight derived from own logits per chunk.
    mode="focal": w = (1 - p_gold)^gamma
    mode="drop":  w = 1[ce >= thresh]
    grad = w * (softmax - onehot) / n_kept_valid  (weight is stop-grad)
    """
    @staticmethod
    def forward(ctx, x, weight, target, mode, param, chunk_size=512):
        N, V = x.shape[0], weight.shape[0]
        weight_cast = weight.to(x.dtype)
        grad_x = torch.zeros_like(x) if x.requires_grad else None
        grad_w = torch.zeros_like(weight) if weight.requires_grad else None
        total, n_kept = 0.0, 0
        for s in range(0, N, chunk_size):
            xc, tc = x[s:s + chunk_size], target[s:s + chunk_size]
            m = tc != -100
            lg = (xc @ weight_cast.t()).float()
            lp = F.log_softmax(lg, dim=-1)
            ce = F.nll_loss(lp, tc.clamp(0), reduction="none")
            if mode == "focal":
                p = lp.gather(1, tc.clamp(0).unsqueeze(1)).squeeze(1).exp()
                w = ((1 - p).clamp(min=0) ** param) * m.float()
            else:
                w = ((ce >= param) & m).float()
            total += float((ce * w).sum())
            n_kept += int((w > 0).sum())
            if x.requires_grad:
                with torch.no_grad():
                    pr = lp.exp()
                    pr.scatter_add_(1, tc.clamp(0).unsqueeze(1),
                                    torch.full_like(pr[:, :1], -1.0))
                    pr = pr * w.unsqueeze(1)
                grad_x[s:s + chunk_size] = pr.to(x.dtype) @ weight_cast
                if grad_w is not None:
                    grad_w += pr.to(x.dtype).t() @ xc
        ctx.save_for_backward(grad_x, grad_w)
        ctx.n = max(n_kept, 1)
        return torch.tensor(total / max(n_kept, 1), device=x.device,
                            dtype=x.dtype)

    @staticmethod
    def backward(ctx, gout):
        gx, gw = ctx.saved_tensors
        g = gout / ctx.n
        return (gx * g if gx is not None else None,
                gw * g if gw is not None else None,
                None, None, None, None)


class _ChunkedKL(torch.autograd.Function):
    """Forward-KL to a teacher distribution.

    full:  KL over full vocab; grad = (p_s - p_t) @ W
    topk:  KL renorm'd on teacher's top-k support; grad flows only
           through the k columns: (q_s - q_t) @ W_sel  (narrow head)
    ce_mix adds 0.3 * gold-token CE on top.
    Teacher log-probs tlp are fp32 [N,V] computed upstream.
    """
    @staticmethod
    def forward(ctx, x, weight, target, tlp, k, ce_mix, chunk_size=512):
        N, V = x.shape[0], weight.shape[0]
        weight_cast = weight.to(x.dtype)
        grad_x = torch.zeros_like(x) if x.requires_grad else None
        grad_w = torch.zeros_like(weight) if weight.requires_grad else None
        total, n_val = 0.0, 0
        for s in range(0, N, chunk_size):
            xc, tc = x[s:s + chunk_size], target[s:s + chunk_size]
            tl = tlp[s:s + chunk_size]
            m = tc != -100
            nv = int(m.sum())
            if nv == 0:
                continue
            n_val += nv
            if k is None:                                    # full-vocab KL
                sl = (xc @ weight_cast.t()).float()
                sp = F.log_softmax(sl, dim=-1)
                tp = tl
                kl = (tp.exp() * (tp - sp)).sum(-1)
                loss_tok = kl * m.float()
                if ce_mix > 0:
                    ce = F.nll_loss(sp, tc.clamp(0), reduction="none")
                    loss_tok = loss_tok + ce_mix * ce * m.float()
                total += float(loss_tok.sum())
                if x.requires_grad:
                    with torch.no_grad():
                        diff = sp.exp() - tp.exp()          # [cs,V]
                        if ce_mix > 0:
                            diff = diff * (1 - ce_mix)
                            ce_g = sp.exp()
                            ce_g.scatter_add_(
                                1, tc.clamp(0).unsqueeze(1),
                                torch.full_like(ce_g[:, :1], -1.0))
                            diff = diff + ce_mix * ce_g
                        diff = diff * m.unsqueeze(1).float()
                    grad_x[s:s + chunk_size] = diff.to(x.dtype) @ weight_cast
                    if grad_w is not None:
                        grad_w += diff.to(x.dtype).t() @ xc
            else:                                            # top-k support
                topv, topi = tl.topk(k, dim=-1)              # [cs,k]
                tq = F.log_softmax(topv, dim=-1)
                Wsel = weight_cast[topi.reshape(-1)].view(
                    topi.size(0), k, -1)                     # [cs,k,d]
                sq = torch.einsum("cd,ckd->ck", xc, Wsel).float()
                sq = F.log_softmax(sq, dim=-1)
                kl = (tq.exp() * (tq - sq)).sum(-1)
                loss_tok = kl * m.float()
                extra = None
                if ce_mix > 0:
                    # cheap gold-CE path: gold logit via dot product
                    wg = weight_cast[tc.clamp(0)]            # [cs,d]
                    gl = (xc * wg).sum(-1).float()
                    lse = torch.logsumexp(sq, dim=-1)        # over k only
                    # approximate p(gold) using renorm'd support when gold
                    # in topk; else tiny floor. Simple, honest surrogate.
                    gold_in = (topi == tc.unsqueeze(1)).float().sum(-1)
                    ce = -(gl - lse)
                    ce = ce * gold_in + (~gold_in.bool()).float() * 3.0
                    loss_tok = loss_tok + ce_mix * ce * m.float()
                    extra = (ce * m.float()).sum()
                total += float(loss_tok.sum())
                if x.requires_grad:
                    with torch.no_grad():
                        diff = sq.exp() - tq.exp()           # [cs,k]
                        diff = diff * m.unsqueeze(1).float()
                    grad_x[s:s + chunk_size] = torch.einsum(
                        "ck,ckd->cd", diff.to(x.dtype), Wsel)
                    if grad_w is not None:
                        gw = torch.zeros(
                            k * topi.size(0), weight.size(1),
                            device=x.device, dtype=x.dtype)
                        gw += diff.unsqueeze(-1).to(x.dtype) * xc.unsqueeze(1)
                        grad_w.index_add_(
                            0, topi.reshape(-1),
                            gw.view(-1, weight.size(1)))
        ctx.save_for_backward(grad_x, grad_w)
        ctx.n = max(n_val, 1)
        return torch.tensor(total / ctx.n, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, gout):
        gx, gw = ctx.saved_tensors
        g = gout / ctx.n
        return (gx * g if gx is not None else None,
                gw * g if gw is not None else None,
                None, None, None, None)


class _ChunkedKLTopk(torch.autograd.Function):
    """Forward-KL on pre-stored teacher top-k support (cached distillation).

    tval [N,k] fp32 teacher log-probs (raw, renorm'd inside); tidx [N,k]
    int64 token ids. Grad flows only through the k head columns — this is
    the in-training consumption half of a score_teacher_topk cache.
    """
    @staticmethod
    def forward(ctx, x, weight, target, tval, tidx, ce_mix, chunk_size=512):
        N, V = x.shape[0], weight.shape[0]
        weight_cast = weight.to(x.dtype)
        grad_x = torch.zeros_like(x) if x.requires_grad else None
        grad_w = torch.zeros_like(weight) if weight.requires_grad else None
        total, n_val = 0.0, 0
        for s in range(0, N, chunk_size):
            xc, tc = x[s:s + chunk_size], target[s:s + chunk_size]
            tv, ti = tval[s:s + chunk_size], tidx[s:s + chunk_size]
            m = tc != -100
            nv = int(m.sum())
            if nv == 0:
                continue
            n_val += nv
            k = tv.size(1)
            tq = F.log_softmax(tv, dim=-1)
            Wsel = weight_cast[ti.reshape(-1)].view(ti.size(0), k, -1)
            sq = torch.einsum("cd,ckd->ck", xc, Wsel).float()
            sq = F.log_softmax(sq, dim=-1)
            kl = (tq.exp() * (tq - sq)).sum(-1)
            loss_tok = kl * m.float()
            if ce_mix > 0:
                wg = weight_cast[tc.clamp(0)]
                gl = (xc * wg).sum(-1).float()
                lse = torch.logsumexp(sq, dim=-1)
                gold_in = (ti == tc.unsqueeze(1)).float().sum(-1)
                ce = -(gl - lse)
                ce = ce * gold_in + (~gold_in.bool()).float() * 3.0
                loss_tok = loss_tok + ce_mix * ce * m.float()
            total += float(loss_tok.sum())
            if x.requires_grad:
                with torch.no_grad():
                    diff = sq.exp() - tq.exp()
                    diff = diff * m.unsqueeze(1).float()
                grad_x[s:s + chunk_size] = torch.einsum(
                    "ck,ckd->cd", diff.to(x.dtype), Wsel)
                if grad_w is not None:
                    gw = diff.unsqueeze(-1).to(x.dtype) * xc.unsqueeze(1)
                    grad_w.index_add_(
                        0, ti.reshape(-1),
                        gw.view(-1, weight.size(1)))
        ctx.save_for_backward(grad_x, grad_w)
        ctx.n = max(n_val, 1)
        return torch.tensor(total / ctx.n, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, gout):
        gx, gw = ctx.saved_tensors
        g = gout / ctx.n
        return (gx * g if gx is not None else None,
                gw * g if gw is not None else None,
                None, None, None, None)


# ── data / model helpers ──

def build_data(seq_len, n_train, n_eval):
    """Returns (train_examples, train_batches, eval_batches)."""
    S._CHAT_FORMAT = "spark"
    tok = get_tokenizer(STU_CKPT)
    rows = S.load_examples([DATA])
    train_rows, eval_rows = rows[:n_train], rows[n_train:n_train + n_eval]

    def pack(rr):
        ds = []
        for ex in rr:
            ds.extend(S.tokenize_example(ex, tok, seq_len))
        return ds

    tr, ev = pack(train_rows), pack(eval_rows)
    mk = lambda ds: [S.collate_batch(ds[i:i + BS], pad_id=2, device="cuda")
                     for i in range(0, len(ds) - BS + 1, BS)]
    return tr, mk(tr), ev, mk(ev)


def score_examples(model, ds, W_head):
    """Per-example mean CE over supervised positions (difficulty signal)."""
    out = []
    model.eval()
    with torch.no_grad():
        for ex in ds:
            ids, labels, mask, *_ = S.collate_batch(
                [ex], pad_id=2, device="cuda")
            h = hidden_of(model, ids, mask).view(-1, model.config.d_model)
            tgt = shift_labels(labels).view(-1)
            m = tgt != -100
            tot, cnt = 0.0, 0
            for s in range(0, tgt.numel(), 512):
                lp = F.log_softmax(
                    h[s:s + 512].float() @ W_head.float().t(), dim=-1)
                ce = F.nll_loss(lp, tgt[s:s + 512].clamp(0),
                                reduction="none")
                tot += float((ce * m[s:s + 512].float()).sum())
                cnt += int(m[s:s + 512].sum())
            out.append(tot / max(cnt, 1))
    model.train()
    return out


def example_proxy(ex):
    """Cheap difficulty proxy: distinct-token ratio (compression-ish)."""
    ids = ex["input_ids"] if isinstance(ex, dict) else ex[0]
    ids = list(ids) if not torch.is_tensor(ids) else ids.tolist()
    return len(set(ids)) / max(len(ids), 1)


def reorder_batches(ds, order):
    ordered = [ds[i] for i in order]
    return [S.collate_batch(ordered[i:i + BS], pad_id=2, device="cuda")
            for i in range(0, len(ordered) - BS + 1, BS)]


def build_model(ckpt):
    cfg = get_config("spark_x26_1p7b", device="cuda")
    cfg.use_gradient_checkpointing = True
    cfg.selective_gradient_checkpointing = "all"
    cfg.use_chunked_ce = True
    cfg.ce_chunk_size = 128
    return ModelLoader.build_model_fast(cfg, checkpoint_path=ckpt,
                                        dtype=torch.bfloat16)


def shift_labels(labels):
    return F.pad(labels[:, 1:], (0, 1), value=-100)


def hidden_of(model, ids, mask):
    out = model(ids, attention_mask=mask, return_hidden=True,
                logits_last_only=True)
    return out[-1]


def eval_ce(student, eval_b, W_head):
    student.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for ids, labels, mask, *_ in eval_b:
            h = hidden_of(student, ids, mask)
            tgt = shift_labels(labels).view(-1)
            m = tgt != -100
            N, V = tgt.numel(), W_head.shape[0]
            hf = h.view(-1, h.shape[-1])
            for s in range(0, N, 512):
                lp = F.log_softmax(
                    hf[s:s + 512].float() @ W_head.float().t(), dim=-1)
                ce = F.nll_loss(lp, tgt[s:s + 512].clamp(0),
                                reduction="none")
                tot += float((ce * m[s:s + 512].float()).sum())
            cnt += int(m.sum())
    student.train()
    return tot / max(cnt, 1)


def teacher_logp(teacher, ids, mask):
    with torch.no_grad():
        h = hidden_of(teacher, ids, mask)
        W = teacher.head.weight
        N = h.numel() // h.shape[-1]
        hf = h.view(-1, h.shape[-1])
        out = torch.empty(N, W.shape[0], device=h.device,
                          dtype=torch.float32)
        for s in range(0, N, 512):
            out[s:s + 512] = F.log_softmax(
                hf[s:s + 512].float() @ W.float().t(), dim=-1)
        return out


def teacher_topk_priv(model, ds, k=15):
    """Privileged-context SELF-teacher (prompt-distillation style): the
    base model scores the second copy of each duplicated example — the
    first copy is the privileged context (full solution). Zero extra
    VRAM: no separate teacher weights. Returns per-example
    (tval [T,k] fp32, tidx [T,k] int64) aligned to student positions."""
    model.eval()
    W = model.head.weight
    out = []
    with torch.no_grad():
        for ex in ds:
            ids, labels, mask, *_ = S.collate_batch(
                [ex], pad_id=2, device="cuda")
            ids2 = torch.cat([ids, ids], dim=1)
            mask2 = torch.cat([mask, mask], dim=1)
            h = hidden_of(model, ids2, mask2)          # [1,2T,d]
            L = ids.size(1)
            h2 = h[0, L:]                              # [T,d] ~ 2nd-copy dist
            tgt = shift_labels(labels).view(-1)
            tvl = torch.zeros(tgt.numel(), k, device="cuda")
            tid = torch.zeros(tgt.numel(), k, dtype=torch.long,
                              device="cuda")
            hs = h2.reshape(-1, h2.shape[-1])
            for s in range(0, tgt.numel(), 512):
                lp = F.log_softmax(
                    hs[s:s + 512].float() @ W.float().t(), dim=-1)
                v, i = lp.topk(k, dim=-1)
                tvl[s:s + 512] = v
                tid[s:s + 512] = i
            out.append((tvl, tid))
    model.train()
    return out


def attach_lora(model, lr=LR, wd=0.0, rank=32, alpha=64):
    n_ad, lora_p = add_lora_adapters(model, rank=rank, alpha=alpha,
                                   target_modules=LORA_TARGETS)
    ids_g = {id(p) for p in lora_p}
    for p in model.parameters():
        if id(p) not in ids_g:
            p.requires_grad_(False)
    return torch.optim.AdamW(lora_p, lr=lr, weight_decay=wd, fused=True)


def gen_prefix_eval(model, eval_ds, n_rows=6, n_tok=24):
    """Generative eval: greedy-decode n_tok tokens on eval prompts,
    report fraction matching the gold completion token-for-token
    (prefix accuracy — a task-level metric, unlike heldout CE)."""
    model.eval()
    W = model.head.weight
    matches, total = 0.0, 0
    with torch.no_grad():
        for ex in eval_ds[:n_rows]:
            ids, labels, mask, *_ = S.collate_batch(
                [ex], pad_id=2, device="cuda")
            keep = (labels[0] == -100)
            if not bool(keep.any()):
                continue
            plen = int(keep.sum())  # left-pad => prompt = masked prefix
            ids = ids[:, -plen:]
            mask = torch.ones_like(ids)
            gold = labels[0, -plen:][labels[0, -plen:] != -100]
            gold = labels[0][labels[0] != -100][:n_tok]
            cur = ids.clone()
            cm = mask.clone()
            hit = 0
            for t in range(min(n_tok, gold.numel())):
                h = hidden_of(model, cur, cm)
                nxt = (h[0, -1].float() @ W.float().t()).argmax()
                if int(nxt) == int(gold[t]):
                    hit += 1
                cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
                cm = torch.cat([cm, cm.new_ones(1, 1)], dim=1)
            matches += hit / max(min(n_tok, gold.numel()), 1)
            total += 1
    model.train()
    return matches / max(total, 1)


def train_loop(opt, train_b, batch_loss_fn, steps=STEPS, eval_fn=None,
               eval_every=0):
    losses, times, skipped = [], [], 0
    curve = []
    for i in range(steps + 2):
        batch = train_b[i % len(train_b)]
        ids, labels, mask = batch[0], batch[1], batch[2]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, skip = batch_loss_fn(ids, labels, mask, batch)
        if skip:
            skipped += 1
            opt.zero_grad()
            continue
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - t0)
            losses.append(float(loss.detach().float()))
            if eval_fn and eval_every and (len(losses) % eval_every == 0):
                curve.append(round(eval_fn(), 4))
    if eval_fn and eval_every and len(losses) % eval_every != 0:
        curve.append(round(eval_fn(), 4))
    return losses, times, skipped, curve


def run_arm(arm):
    seq = 512 if arm == "seq512" else SEQ
    tr_ds, train_b, ev_ds, eval_b = build_data(seq, N_TRAIN, N_EVAL)
    res = {"arm": arm, "n_train": len(train_b) * BS,
           "n_eval": len(eval_b) * BS}

    student = build_model(STU_CKPT)
    W_head = student.head.weight
    h_dim = student.config.d_model

    # ── difficulty scoring / ordering / teacher-cache passes (pre-LoRA) ──
    scores = None
    if arm in ("ord_easy", "ord_hard", "learnband"):
        t0 = time.perf_counter()
        scores = score_examples(student, tr_ds, W_head)
        res["score_s"] = round(time.perf_counter() - t0, 1)
        res["n_examples"] = len(tr_ds)
    if arm == "ord_easy":
        order = sorted(range(len(tr_ds)), key=lambda i: scores[i])
        train_b = reorder_batches(tr_ds, order)
    elif arm == "ord_hard":
        order = sorted(range(len(tr_ds)), key=lambda i: -scores[i])
        train_b = reorder_batches(tr_ds, order)
    elif arm == "learnband":
        qs = sorted(scores)
        lo, hi = qs[len(qs) // 4], qs[3 * len(qs) // 4]
        order = [i for i in range(len(tr_ds)) if lo <= scores[i] <= hi]
        train_b = reorder_batches(tr_ds, order)
        res["band_lo_hi"] = [round(lo, 3), round(hi, 3)]
        res["n_kept"] = len(order)
    elif arm == "ord_comp":
        order = sorted(range(len(tr_ds)),
                       key=lambda i: example_proxy(tr_ds[i]))
        train_b = reorder_batches(tr_ds, order)

    topk_cache = None
    if arm == "selfteach15":
        t0 = time.perf_counter()
        topk_cache = teacher_topk_priv(student, tr_ds, k=15)
        res["teach_s"] = round(time.perf_counter() - t0, 1)
        tb = []
        for i in range(0, len(tr_ds) - BS + 1, BS):
            ids, labels, mask, *_ = S.collate_batch(
                tr_ds[i:i + BS], pad_id=2, device="cuda")
            B, T = ids.shape
            k = topk_cache[i][0].size(1)
            tvl = torch.zeros(B * T, k, device="cuda")
            tid = torch.zeros(B * T, k, dtype=torch.long, device="cuda")
            for j in range(B):  # left-pad alignment: block at row end
                tj = topk_cache[i + j][0].size(0)
                tvl[j * T + (T - tj):(j + 1) * T] = topk_cache[i + j][0]
                tid[j * T + (T - tj):(j + 1) * T] = topk_cache[i + j][1]
            tb.append((ids, labels, mask, tvl, tid))
        train_b = tb

    # ── arm-specific training-config overrides ──
    lr = {"lr_hi": 5e-5, "lr_lo": 5e-6}.get(arm, LR)
    wd = 0.1 if arm == "wd01" else 0.0
    rank, alpha = (8, 16) if arm == "r8" else (32, 64)
    opt = attach_lora(student, lr=lr, wd=wd, rank=rank, alpha=alpha)

    if arm == "replay20":
        # 20% replay: rows from outside the train/eval window act as
        # prior-distribution anchors (anti-forgetting).
        tok = get_tokenizer(STU_CKPT)
        replay_rows = S.load_examples([DATA])[N_TRAIN + N_EVAL:
                                            N_TRAIN + N_EVAL + 48]
        rp = []
        for ex in replay_rows:
            rp.extend(S.tokenize_example(ex, tok, seq))
        mixed = list(tr_ds)
        for j in range(len(mixed)):
            if j % 5 == 4 and j // 5 < len(rp):
                mixed[j] = rp[j // 5]
        train_b = [S.collate_batch(mixed[i:i + BS], pad_id=2,
                                   device="cuda")
                   for i in range(0, len(mixed) - BS + 1, BS)]
        res["n_replay"] = len(rp)

    teacher = None
    if arm.startswith(("full_kl", "topk", "massprobe")):
        teacher = build_model(TEA_CKPT).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    if arm == "massprobe":
        K = (1, 4, 8, 15, 64, 256, 1024, 4096)
        mass = {kk: 0.0 for kk in K}
        gold_in = {kk: 0 for kk in K}
        npos = 0
        for ids, labels, mask, *_ in eval_b[:12]:
            tlp = teacher_logp(teacher, ids, mask)
            tgt = shift_labels(labels).view(-1)
            m = tgt != -100
            topv, topi = tlp.topk(max(K), dim=-1)
            cum = topv.exp().cumsum(-1)
            gold = tgt[m].unsqueeze(1)
            for kk in K:
                mass[kk] += float(cum[m, kk - 1].sum())
                gold_in[kk] += int((topi[m, :kk] == gold).any(-1).sum())
            npos += int(m.sum())
        res["mass_at_k"] = {kk: round(mass[kk] / max(npos, 1), 4)
                            for kk in K}
        res["gold_in_topk"] = {kk: round(gold_in[kk] / max(npos, 1), 4)
                               for kk in K}
        res["n_positions"] = npos
        h = torch.randn(2048, h_dim, device="cuda", dtype=torch.bfloat16)
        Wf = student.head.weight
        for name, Wsel in (("full", Wf), ("k256", Wf[:256]),
                           ("k15", Wf[:15])):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20):
                _ = h @ Wsel.t()
            torch.cuda.synchronize()
            res[f"head_ms_{name}"] = round(
                1000 * (time.perf_counter() - t0) / 20, 3)
        print("RESULT " + json.dumps(res), flush=True)
        return

    def batch_loss(ids, labels, mask, batch=None):
        sl = shift_labels(labels)
        if arm in ("ce", "seq512", "ent05", "band", "ord_easy", "ord_hard",
                   "ord_comp", "learnband", "reptile", "stepsweep",
                   "lr_hi", "lr_lo", "wd01", "r8", "replay20", "evalgen"):
            loss = S.compute_loss(
                student, ids, labels, mask,   # compute_loss shifts internally
                entropy_alpha=0.5 if arm == "ent05" else 0.0)
            skip = arm == "band" and not (
                0.4 <= float(loss.detach()) <= 6.0)
            return loss, skip
        h = hidden_of(student, ids, mask).view(-1, h_dim)
        tgt = sl.view(-1)
        if arm == "focal":
            return _ChunkedFocalOrDrop.apply(
                h, W_head, tgt, "focal", 4.93), None
        if arm == "confdrop":
            return _ChunkedFocalOrDrop.apply(
                h, W_head, tgt, "drop", 0.3), None
        if arm == "anslast":
            sl2 = sl.clone()
            for r in range(sl2.size(0)):
                pos = (sl2[r] != -100).nonzero().flatten()
                if pos.numel() > 16:
                    sl2[r, pos[:-16]] = -100
            t2 = sl2.view(-1)
            return _ChunkedFocalOrDrop.apply(
                h, W_head, t2, "focal", 0.0), None
        if arm == "selfteach15":
            tvl = batch[3].view(-1, batch[3].shape[-1])
            tid = batch[4].view(-1, batch[4].shape[-1])
            return _ChunkedKLTopk.apply(h, W_head, tgt, tvl, tid, 0.3), \
                None
        tlp = teacher_logp(teacher, ids, mask)
        if arm == "full_kl":
            return _ChunkedKL.apply(h, W_head, tgt, tlp, None, 0.0), None
        if arm == "topk256":
            return _ChunkedKL.apply(h, W_head, tgt, tlp, 256, 0.0), None
        if arm == "topk256ce":
            return _ChunkedKL.apply(h, W_head, tgt, tlp, 256, 0.3), None
        if arm == "topk15ce":
            return _ChunkedKL.apply(h, W_head, tgt, tlp, 15, 0.3), None
        raise ValueError(arm)

    if arm == "notrain":
        res.update(heldout_ce=round(eval_ce(student, eval_b, W_head), 4),
                   peak_gb=round(
                       torch.cuda.max_memory_allocated() / 2**30, 2))
        print("RESULT " + json.dumps(res), flush=True)
        return

    if arm == "reptile":
        half = len(tr_ds) // 2
        student.train()
        torch.cuda.reset_peak_memory_stats()
        lA, tA, skA, _ = train_loop(opt, reorder_batches(
            tr_ds, list(range(0, half))), batch_loss)
        loraA = {n: p.detach().to("cpu", torch.float32).clone()
                 for n, p in student.named_parameters()
                 if "lora_" in n}
        del opt, student
        torch.cuda.empty_cache()
        student = build_model(STU_CKPT)
        W_head = student.head.weight
        opt = attach_lora(student)
        student.train()
        lB, tB, skB, _ = train_loop(opt, reorder_batches(
            tr_ds, list(range(half, len(tr_ds)))), batch_loss)
        for n, p in student.named_parameters():
            if "lora_" in n and n in loraA:
                p.data.mul_(0.5).add_(
                    loraA[n].to(p.device, p.dtype).mul_(0.5))
        losses = lA + lB
        times = tA + tB
        skipped = skA + skB
        res["split_ce"] = [round(lA[-1], 4) if lA else None,
                           round(lB[-1], 4) if lB else None]
        res.update(step_ms=round(
            1000 * sum(times) / max(len(times), 1), 1),
            loss0=round(losses[0], 4) if losses else None,
            lossN=round(losses[-1], 4) if losses else None,
            heldout_ce=round(eval_ce(student, eval_b, W_head), 4),
            peak_gb=round(
                torch.cuda.max_memory_allocated() / 2**30, 2),
            skipped=skipped)
        print("RESULT " + json.dumps(res), flush=True)
        return

    student.train()
    torch.cuda.reset_peak_memory_stats()
    n_steps = 60 if arm == "stepsweep" else STEPS
    eval_fn = (lambda: eval_ce(student, eval_b, W_head)
               ) if arm == "stepsweep" else None
    losses, times, skipped, curve = train_loop(
        opt, train_b, batch_loss, steps=n_steps, eval_fn=eval_fn,
        eval_every=10)
    res.update(step_ms=round(1000 * sum(times) / max(len(times), 1), 1),
               loss0=round(losses[0], 4) if losses else None,
               lossN=round(losses[-1], 4) if losses else None,
               heldout_ce=round(eval_ce(student, eval_b, W_head), 4),
               peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               skipped=skipped)
    if curve:
        res["heldout_curve"] = curve
    if arm in ("stepsweep", "evalgen"):
        res["gen_prefix"] = round(gen_prefix_eval(student, ev_ds), 4)
    print("RESULT " + json.dumps(res), flush=True)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "ce"
    if arm == "all3":
        import subprocess
        for a in ("stepsweep", "lr_hi", "lr_lo", "wd01", "r8",
                  "replay20", "evalgen"):
            env = dict(os.environ)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            p = subprocess.run([sys.executable, __file__, a],
                               env=env, capture_output=True,
                               text=True, timeout=2400)
            hit = [l for l in p.stdout.splitlines()
                   if l.startswith("RESULT ")]
            print(hit[-1] if hit else
                  f"RESULT {{\"arm\":\"{a}\",\"error\":\""
                  f"{(p.stderr or p.stdout)[-300:].replace(chr(34), chr(39))}\"}}",
                  flush=True)
        return
    if arm == "all2":
        import subprocess
        for a in ("ord_easy", "ord_hard", "ord_comp", "learnband",
                  "reptile", "selfteach15"):
            env = dict(os.environ)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            p = subprocess.run([sys.executable, __file__, a],
                               env=env, capture_output=True,
                               text=True, timeout=1800)
            hit = [l for l in p.stdout.splitlines()
                   if l.startswith("RESULT ")]
            print(hit[-1] if hit else
                  f"RESULT {{\"arm\":\"{a}\",\"error\":\""
                  f"{(p.stderr or p.stdout)[-300:].replace(chr(34), chr(39))}\"}}",
                  flush=True)
        return
    if arm == "all":
        import subprocess
        for a in ("massprobe", "ce", "focal", "ent05", "band", "confdrop",
                  "anslast", "seq512", "full_kl", "topk256", "topk256ce",
                  "topk15ce"):
            env = dict(os.environ)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            p = subprocess.run([sys.executable, __file__, a],
                               env=env, capture_output=True,
                               text=True, timeout=1800)
            hit = [l for l in p.stdout.splitlines()
                   if l.startswith("RESULT ")]
            print(hit[-1] if hit else
                  f"RESULT {{\"arm\":\"{a}\",\"error\":\""
                  f"{(p.stderr or p.stdout)[-300:].replace(chr(34), chr(39))}\"}}",
                  flush=True)
        return
    run_arm(arm)


if __name__ == "__main__":
    main()
