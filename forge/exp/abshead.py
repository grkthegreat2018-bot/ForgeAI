"""E-abshead: "abstract thinking" latent-thought head on a frozen AR model.

A small MLP side-head reads the frozen final hidden state h_t and learns
to predict a FUTURE representation — by default h_{t+k} (the post-norm
hidden k steps ahead, i.e. multi-step hidden-state prediction = a learned
"latent reasoning" direction).  mode="embed" instead predicts the mean
embedding of the next k tokens (a softer, input-space future summary).

The head never writes into the base model, so attach is lossless by
construction.  A second zero-init scalar gate controls an optional
residual "hint" at readout:

    logits = lm_head(h_t + gate * (pred_t - h_t))

gate=0 -> bit-exact baseline logits; gate learned -> asks whether the
future-latent carries signal usable for next-token NLL (honest dNLL).

Pieces:
  * LatentThoughtHead      — Linear(d, hid) -> GELU -> Linear(hid, d),
                             output layer zero-init (~1.8M params @896).
  * future_pairs/collect   — slice (h_t, target_{t+k}) pairs from one
                             frozen forward (no grad).
  * train_latent/eval      — MSE on the future target; eval reports
                             cos-sim + MSE vs the persistence baseline
                             (predict h_t itself).
  * train_hint/hint_nll    — CE through the gated hint readout (gate +
                             head trained, latent-anchor optional).
  * decode_ms_per_tok      — greedy decode timing with/without the hint
                             recompute (honest overhead number).
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentThoughtHead(nn.Module):
    """h_t -> predicted future latent.  Zero-init output layer (and a
    zero-init scalar hint gate) -> lossless at attach.

    forward(h): (..., d) -> (..., d) predicted future representation.
    hint(h):    gated residual to add to h before lm_head —
                gate * (pred - h), 0 at init.
    """

    def __init__(self, d_model: int, hidden: int = 1024, k: int = 4):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.inp = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Head can live in fp32 while the frozen model is bf16 — compute
        # in the head's dtype, callers cast the result as needed.
        return self.out(self.act(self.inp(h.to(self.inp.weight.dtype))))

    def hint(self, h: torch.Tensor) -> torch.Tensor:
        """Residual readout hint: gate * (pred - h).  Zero at gate=0."""
        return self.gate * (self.forward(h) - h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ───────────────────────── targets / data ──────────────────────────


def _final_hidden(model, ids: torch.Tensor) -> torch.Tensor:
    """Frozen-model post-norm final hidden states (B, T, d)."""
    out = model(ids, return_hidden=True, logits_last_only=True)
    hidden = out[-1]
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(1)
    return hidden


def future_pairs(ids: torch.Tensor, hidden: torch.Tensor,
                 embed_w: torch.Tensor | None, k: int,
                 mode: str = "hidden"):
    """(src, tgt) for predicting the future from each position t.

    src = h_t for t in [0, T-k).  tgt =
      mode="hidden": h_{t+k}        (latent reasoning target)
      mode="embed":  mean embedding of tokens t+1..t+k
    Both (B, T-k, d).  tgt is returned detached.
    """
    T = ids.shape[1]
    src = hidden[:, :T - k]
    if mode == "embed":
        assert embed_w is not None, "embed mode needs the embedding matrix"
        tgt_ids = torch.stack([ids[:, j:j + T - k]
                               for j in range(1, k + 1)], dim=2)
        tgt = F.embedding(tgt_ids, embed_w).mean(dim=2)
    else:
        tgt = hidden[:, k:]
    return src, tgt.detach()


@torch.no_grad()
def collect_pairs(model, ids: torch.Tensor, k: int,
                  mode: str = "hidden"):
    """One frozen forward -> (src, tgt) pair tensors for a doc."""
    embed_w = (model.embed.weight if mode == "embed" else None)
    hidden = _final_hidden(model, ids)
    return future_pairs(ids, hidden, embed_w, k, mode)


# ───────────────────── phase A: latent prediction ──────────────────


def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)


def train_latent(model, head: LatentThoughtHead, ids_list,
                 steps: int = 300, lr: float = 1e-3,
                 mode: str = "hidden") -> dict:
    """Train only the head on MSE(pred, future target); model frozen.

    Returns {"losses": [...], "cos": [...]} sampled per step."""
    _freeze(model)
    model.eval()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    losses, coss = [], []
    step = 0
    while step < steps:
        for ids in ids_list:
            if step >= steps:
                break
            ids = ids.to(next(model.parameters()).device)
            src, tgt = collect_pairs(model, ids, head.k, mode)
            pred = head(src)
            loss = F.mse_loss(pred.float(), tgt.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            coss.append(float(F.cosine_similarity(
                pred.float().reshape(-1, pred.size(-1)).detach(),
                tgt.float().reshape(-1, tgt.size(-1)),
                dim=-1).mean()))
            step += 1
    _unfreeze(model)
    return {"losses": losses, "cos": coss}


@torch.no_grad()
def eval_latent(model, head: LatentThoughtHead, ids_list,
                mode: str = "hidden") -> dict:
    """Held-out latent-prediction quality vs the persistence baseline
    (predict h_t itself).  Cos-sim and MSE averaged over positions."""
    k = head.k
    cos_h = cos_p = mse_h = mse_p = n = 0.0
    for ids in ids_list:
        ids = ids.to(next(model.parameters()).device)
        src, tgt = collect_pairs(model, ids, k, mode)
        pred = head(src)
        pf, tf = pred.float(), tgt.float()
        sf = src.float()
        B, L, D = pf.shape
        cos_h += F.cosine_similarity(pf.reshape(-1, D),
                                     tf.reshape(-1, D), dim=-1).sum()
        cos_p += F.cosine_similarity(sf.reshape(-1, D),
                                     tf.reshape(-1, D), dim=-1).sum()
        mse_h += F.mse_loss(pf, tf, reduction="sum") / D
        mse_p += F.mse_loss(sf, tf, reduction="sum") / D
        n += B * L
    n = max(n, 1)
    return {"cos_head": float(cos_h / n), "cos_persist": float(cos_p / n),
            "mse_head": float(mse_h / n), "mse_persist": float(mse_p / n),
            "n_positions": int(n)}


# ───────────────────── phase B: residual hint dNLL ─────────────────


def hint_logits(model, head: LatentThoughtHead,
                hidden: torch.Tensor) -> torch.Tensor:
    """lm_head(hidden + gate * (pred - hidden)).  == base at gate=0."""
    hinted = (hidden + head.hint(hidden)).to(hidden.dtype)
    return model.head(hinted)


@torch.no_grad()
def hint_nll(model, head: LatentThoughtHead, ids_list) -> float:
    """Next-token NLL with the gated hint injected at readout."""
    tot, n = 0.0, 0
    for ids in ids_list:
        ids = ids.to(next(model.parameters()).device)
        hidden = _final_hidden(model, ids[:, :-1])
        logits = hint_logits(model, head, hidden)
        # CE on raw-dtype logits — matches the model's internal loss
        # path so gate=0 gives a bit-exact baseline NLL.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1))
        tot += float(loss) * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return tot / max(n, 1)


def train_hint(model, head: LatentThoughtHead, ids_list,
               steps: int = 150, lr: float = 1e-3,
               latent_w: float = 0.1, mode: str = "hidden",
               gate_only: bool = False) -> list:
    """Train the gated hint on next-token CE through the hint readout.

    gate_only=True trains just the scalar gate (head weights frozen at
    their latent-trained values) — isolates whether the *learned latent*
    itself helps NLL.  Otherwise gate + head train jointly.

    latent_w > 0 keeps a light MSE anchor to the phase-A future target so
    the head stays a *latent* (not a free-floating readout bias).
    gate=0 start: dL/dgate != 0 once the head produces nonzero output.
    """
    _freeze(model)
    model.eval()
    params = [head.gate] if gate_only else list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    embed_w = (model.embed.weight if mode == "embed" else None)
    losses = []
    step = 0
    while step < steps:
        for ids in ids_list:
            if step >= steps:
                break
            ids = ids.to(next(model.parameters()).device)
            hidden = _final_hidden(model, ids[:, :-1]).detach()
            logits = hint_logits(model, head, hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                ids[:, 1:].reshape(-1))
            if latent_w > 0:
                src, tgt = future_pairs(ids[:, :-1], hidden, embed_w,
                                        head.k, mode)
                loss = loss + latent_w * F.mse_loss(
                    head(src).float(), tgt.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            step += 1
    _unfreeze(model)
    return losses


# ───────────────────── decode overhead ─────────────────────────────


@torch.inference_mode()
def decode_ms_per_tok(model, ids: torch.Tensor, n_tokens: int = 64,
                      head: LatentThoughtHead | None = None,
                      n_runs: int = 3) -> float:
    """Greedy decode ms/tok.  head=None -> plain; else each step
    recomputes logits as lm_head(hidden + hint) (extra MLP + 1 lm_head
    matmul on the last position = honest overhead).  One untimed warmup
    run first so kernel-launch/JIT noise doesn't pollute the mean."""

    def run(n: int) -> float:
        out = model(ids, use_cache=True, return_hidden=True)
        logits, past, hidden = out[0], out[2], out[3]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _i in range(n):
            if head is None:
                tok = logits[:, -1].argmax(-1, keepdim=True)
            else:
                tok = hint_logits(model, head,
                                  hidden[:, -1:]).argmax(-1)
            out = model(tok, past_key_values=past, use_cache=True,
                        return_hidden=True)
            logits, past, hidden = out[0], out[2], out[3]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
        del past
        return el

    dev = next(model.parameters()).device
    ids = ids.to(dev)
    run(16)  # warmup (untimed)
    rates = [n_tokens / run(n_tokens) for _ in range(n_runs)]
    return 1000.0 / (sum(rates) / len(rates))
