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
