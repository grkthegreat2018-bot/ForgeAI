"""E-loop: recurrent-depth (looped transformer) on a pretrained AR model.

Runs the middle block range of ConfigurableResearchLLM for multiple
iterations at prefill (use_cache=True only commits KV from the LAST
iteration — decode then sees the converged representation).

Optional learned LoopGates: h <- h + g_i * (B_i(h) - h) per looped layer.
g=0 init -> loop is a no-op (lossless); a few dozen scalar params trained
on text tell us whether *any* recurrent depth helps a frozen model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _embed(model, ids):
    x = model.embed(ids)
    if getattr(model, "embed_norm", None) is not None:
        x = model.embed_norm(x)
    return x


@torch.no_grad()
def looped_prefill(model, ids, loop_lo: int, loop_hi: int,
                   max_iters: int = 0, tol: float = 0.0,
                   gates: torch.Tensor | None = None,
                   collect_deltas: bool = False):
    """Prefill with middle blocks [loop_lo, loop_hi) iterated.

    Returns (logits_last, pasts, stats). pasts is a list[(k,v)|None]
    matching the model's decode-step convention.
    """
    dev = next(model.parameters()).device
    ids = ids.to(dev)
    x = _embed(model, ids)
    pasts: list = [None] * len(model.blocks)
    for i in range(loop_lo):
        x, pasts[i] = model.blocks[i](x, past_key_value=None,
                                      use_cache=True, layer_idx=i)

    deltas = []
    it = 0
    for it in range(max(1, max_iters)):
        h = x
        for i in range(loop_lo, loop_hi):
            out, pasts[i] = model.blocks[i](
                h, past_key_value=None, use_cache=True, layer_idx=i)
            # Iteration 0 is the normal block pass; only EXTRA iterations
            # are gated so g=0 == standard transformer.
            if it > 0 and gates is not None:
                h = h + gates[i - loop_lo] * (out - h)
            else:
                h = out
        d = (h - x).norm() / x.norm().clamp_min(1e-8)
        deltas.append(float(d))
        x = h
        if tol > 0 and float(d) < tol:
            break

    for i in range(loop_hi, len(model.blocks)):
        x, pasts[i] = model.blocks[i](x, past_key_value=None,
                                      use_cache=True, layer_idx=i)
    hidden = model.ln_f(x) if model.ln_f is not None else x
    logits = model.head(hidden)
    stats = {"iters_run": it + 1, "deltas": deltas}
    return logits, pasts, stats


@torch.no_grad()
def looped_decode_step(model, tok, pasts, loop_lo: int, loop_hi: int,
                       max_iters: int, gates=None):
    """One decode step; looped blocks iterate with the committed context
    (past attends, no append on early iterations; last iter appends)."""
    x = _embed(model, tok)
    for i in range(loop_lo):
        x, pasts[i] = model.blocks[i](x, past_key_value=pasts[i],
                                      use_cache=True, layer_idx=i)
    for it in range(max(1, max_iters)):
        for i in range(loop_lo, loop_hi):
            out, new_kv = model.blocks[i](
                x, past_key_value=pasts[i],
                use_cache=True, layer_idx=i)
            if it > 0 and gates is not None:
                x = x + gates[i - loop_lo] * (out - x)
            else:
                x = out
            if it == max(1, max_iters) - 1:
                pasts[i] = new_kv
    for i in range(loop_hi, len(model.blocks)):
        x, pasts[i] = model.blocks[i](x, past_key_value=pasts[i],
                                      use_cache=True, layer_idx=i)
    hidden = model.ln_f(x) if model.ln_f is not None else x
    return model.head(hidden[:, -1:]), pasts


class VerifierGate:
    """Output-level early-stop for looped blocks (R71/B5).

    Stops the loop when the next-token distribution has CONVERGED —
    the model stopped changing its mind — rather than when hidden-state
    drift is small (the old `tol` criterion, which can fire while logits
    still move). Cheaper than a learned verifier head: head+ln_f on one
    position is ~0.3% of a layer's FLOPs.

    Rule: stop after `patience` consecutive iterations where
      argmax(logits_t) == argmax(logits_{t-1})  AND
      softmax margin (p1 - p2) >= min_margin.
    """

    def __init__(self, min_margin: float = 0.15, patience: int = 1):
        self.min_margin = min_margin
        self.patience = patience
        self._prev_argmax = None
        self._stable = 0

    def update(self, logits: torch.Tensor) -> bool:
        """logits: [vocab] or [1, 1, vocab] for the last position."""
        lg = logits.reshape(-1).float()
        probs = torch.softmax(lg, dim=-1)
        top2 = probs.topk(2)
        am = int(top2.indices[0])
        margin = float(top2.values[0] - top2.values[1])
        if self._prev_argmax is not None and am == self._prev_argmax \
                and margin >= self.min_margin:
            self._stable += 1
        else:
            self._stable = 0
        self._prev_argmax = am
        return self._stable >= self.patience

    def reset(self):
        self._prev_argmax = None
        self._stable = 0


@torch.no_grad()
def looped_prefill_verified(model, ids, loop_lo: int, loop_hi: int,
                            max_iters: int = 4, gates=None,
                            gate: VerifierGate | None = None):
    """looped_prefill with VerifierGate early-stop.

    Identical math to looped_prefill except the loop exits early when
    `gate` fires (output distribution converged). Returns
    (logits_last, pasts, stats) with stats['stop_reason'].
    """
    dev = next(model.parameters()).device
    ids = ids.to(dev)
    x = _embed(model, ids)
    pasts: list = [None] * len(model.blocks)
    for i in range(loop_lo):
        x, pasts[i] = model.blocks[i](x, past_key_value=None,
                                      use_cache=True, layer_idx=i)
    deltas = []
    stop = "max_iters"
    it = -1
    for it in range(max(1, max_iters)):
        h = x
        for i in range(loop_lo, loop_hi):
            out, pasts[i] = model.blocks[i](
                h, past_key_value=None, use_cache=True, layer_idx=i)
            if it > 0 and gates is not None:
                h = h + gates[i - loop_lo] * (out - h)
            else:
                h = out
        deltas.append(float((h - x).norm() / x.norm().clamp_min(1e-8)))
        x = h
        if gate is not None and it >= 1:
            hh = model.ln_f(h) if model.ln_f is not None else h
            lg = model.head(hh[:, -1:])
            if gate.update(lg):
                stop = f"verified@{it}"
                break
    for i in range(loop_hi, len(model.blocks)):
        x, pasts[i] = model.blocks[i](x, past_key_value=None,
                                      use_cache=True, layer_idx=i)
    hidden = model.ln_f(x) if model.ln_f is not None else x
    logits = model.head(hidden)
    stats = {"iters_run": it + 1, "deltas": deltas, "stop_reason": stop}
    return logits, pasts, stats


class LoopGates(nn.Module):
    """Per-loop-layer scalar gates. g=0 => loop is identity."""

    def __init__(self, n_layers: int, device="cuda"):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(n_layers, device=device))

    def forward(self):
        return self.g


def train_loop_gates(model, ids_list, loop_lo: int, loop_hi: int,
                     max_iters: int = 2, steps: int = 300, lr: float = 3e-2):
    """Train only the loop gates (frozen model) — asks 'does any recurrent
    depth help?' Gates init 0 -> lossless start; optimizer decides."""
    gates = LoopGates(loop_hi - loop_lo, device=ids_list[0].device)
    opt = torch.optim.Adam(gates.parameters(), lr=lr)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def fwd(ids):
        x = _embed(model, ids)
        for i in range(loop_lo):
            x, _ = model.blocks[i](x, past_key_value=None,
                                   use_cache=False, layer_idx=i)
        for it in range(max(1, max_iters)):
            for i in range(loop_lo, loop_hi):
                out, _ = model.blocks[i](x, past_key_value=None,
                                         use_cache=False, layer_idx=i)
                x = (x + gates.g[i - loop_lo] * (out - x)
                     if it > 0 else out)
        for i in range(loop_hi, len(model.blocks)):
            x, _ = model.blocks[i](x, past_key_value=None,
                                   use_cache=False, layer_idx=i)
        h = model.ln_f(x) if model.ln_f is not None else x
        return model.head(h)

    losses = []
    step = 0
    while step < steps:
        for ids in ids_list:
            if step >= steps:
                break
            ids = ids.to(next(model.parameters()).device)
            logits = fwd(ids[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                ids[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            step += 1
    for p in model.parameters():
        p.requires_grad_(True)
    return gates, losses
