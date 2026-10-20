"""E-ncm: Neural Circuit Model (closed-form continuous cell) as a
ModularBlock-style side-path on a frozen transformer.

CfC recurrence (Liquid AI family): time-evolving state driven by learned
gates — an ODE-dynamics layer next to attention/FFN:

    d_t = sigmoid(-softplus(f(x_t)) * tau)          # learned decay
    g_t = tanh(g(x_t))                            # candidate write
    h_t = d_t * h_{t-1} + (1 - d_t) * g_t         # continuous-time update
    y_t = o(h_t)

Attach with gate=0 => lossless; train ~3d^2 params on text to see if ODE
dynamics help a frozen AR model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NCMLayer(nn.Module):
    """Closed-form continuous cell over the sequence dim.

    gate is a scalar zero-init parameter — output is gate * y so attach
    is lossless (matches the repo's side-path convention).
    """

    def __init__(self, d_model: int, tau: float = 1.0):
        super().__init__()
        self.f_decay = nn.Linear(d_model, d_model, bias=False)
        self.g_cand = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        # per-channel timescale (learnable), init near `tau`
        self.log_tau = nn.Parameter(torch.full((d_model,), float(tau)).log())
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.eye_(self.o_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model) gated output."""
        B, T, D = x.shape
        dec = torch.sigmoid(
            -torch.nn.functional.softplus(self.f_decay(x))
            * self.log_tau.exp())                       # (B,T,D) in (0,1)
        cand = torch.tanh(self.g_cand(x))
        h = torch.zeros(B, D, dtype=x.dtype, device=x.device)
        outs = []
        for t in range(T):
            h = dec[:, t] * h + (1.0 - dec[:, t]) * cand[:, t]
            outs.append(h)
        y = self.o_proj(torch.stack(outs, dim=1))
        return self.gate * y

    # live-surface-ish contract for tests
    def reset_state(self):
        pass


def attach_ncm(model, layers: list[int], tau: float = 1.0) -> list[int]:
    """Attach an NCMLayer as `block._ncm` to given block indices and wrap
    the block forward so `x += _ncm(x0)` happens post-block (same residual
    convention as _memory). Lossless at attach (gate=0)."""
    p = next(model.parameters())
    attached = []
    for i in layers:
        blk = model.blocks[i]
        if getattr(blk, "_ncm", None) is not None:
            continue
        blk._ncm = NCMLayer(model.config.d_model, tau=tau).to(
            device=p.device, dtype=p.dtype)
        blk._ncm_gate_zero = None
        if not getattr(blk, "_ncm_wrapped", False):
            _wrap_block_forward(blk)
            blk._ncm_wrapped = True
        attached.append(i)
    return attached


def _wrap_block_forward(blk):
    orig = blk.forward

    def fwd(x, *a, **k):
        x0 = x
        x, present = orig(x, *a, **k)
        ncm = getattr(blk, "_ncm", None)
        if ncm is not None:
            if blk._ncm_gate_zero is None:
                blk._ncm_gate_zero = (ncm.gate.item() == 0.0)
            if not blk._ncm_gate_zero:
                x = x + ncm(x0)
        return x, present

    blk.forward = fwd


def train_ncm(model, ids_list, layers, steps=200, lr=3e-3):
    """Train only the NCM params (frozen model) on next-token CE.
    Gates open during training so gradients flow."""
    params = []
    for i in layers:
        ncm = model.blocks[i]._ncm
        ncm.gate.data.fill_(1.0)
        model.blocks[i]._ncm_gate_zero = None
        params += list(ncm.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    losses = []
    step = 0
    while step < steps:
        for ids in ids_list:
            if step >= steps:
                break
            ids = ids.to(next(model.parameters()).device)
            logits = model(ids[:, :-1])[0]
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
    return losses
