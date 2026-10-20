"""E-dynscale: dynamic live-scaling of parameters / heads at inference.

Two mechanisms measured independently on ConfigurableResearchLLM
(Qwen2.5-0.5B: d_model=896, 24 blocks, 14 Q heads / 2 KV heads, head_dim=64):

1. ``HeadGateController`` — runtime gating of attention heads by EMA
   saliency.  ``GroupedQueryAttention.forward`` flattens its per-head
   output to (B,T,H*Dh) and feeds ``self.out_proj`` — we wrap that linear
   with ``HeadGate``, which views the input back to (B,T,H,Dh), scores
   each head by output-contribution norm (EMA-smoothed), and zero-gates
   heads below threshold tau.  Gating the out_proj *input* in head_dim
   blocks is mathematically identical to gating per-head attention
   outputs, and needs no model surgery.

2. ``EarlyExitController`` — live depth scaling during greedy decode.
   The block loop is run manually; after each block the last-position
   hidden is pushed through ``model.ln_f + model.head`` and a confidence
   metric (top-1 prob / top1-top2 margin / logit margin) is compared to
   tau.  If it passes, remaining blocks are skipped for this token —
   their KV caches simply keep a shorter history (skipped-token
   semantics).  Absolute ``position_ids`` are passed to every block so
   RoPE stays correct even for layers whose cache lagged behind.

3. ``DiagTempScaler`` (bonus) — per-channel diagonal "temperature" gains
   on each block's attention-out and FFN-out, briefly re-optimized on
   wikitext after gating, then folded into the ``out_proj`` / ``w_down``
   weight rows so recalibration is free at runtime.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "HeadGate", "HeadGateController", "EarlyExitController",
    "OutGain", "DiagTempScaler", "find_linear", "confidence",
]


# ── helpers ──────────────────────────────────────────────────────────────

def find_linear(mod):
    """Walk ``.linear``/``.inner`` wrapper chains down to the nn.Linear.

    Returns None if the chain bottoms out on a non-Linear (e.g. a
    quantized or kernel-specialized projection).
    """
    seen = set()
    while mod is not None and not isinstance(mod, nn.Linear):
        if id(mod) in seen:
            return None
        seen.add(id(mod))
        nxt = getattr(mod, "linear", None)
        if nxt is None:
            nxt = getattr(mod, "inner", None)
        mod = nxt
    return mod if isinstance(mod, nn.Linear) else None


def confidence(logits: torch.Tensor, metric: str) -> torch.Tensor:
    """(…, V) logits -> (…,) confidence in [0,1]-ish range.

    metric:
      "prob"         — top-1 softmax probability
      "margin"       — top1 - top2 softmax-probability margin
      "logit_margin" — top1 - top2 raw-logit margin (cheapest)
    """
    z = logits.float()
    if metric == "prob":
        return z.softmax(dim=-1).amax(dim=-1)
    if metric == "margin":
        top2 = z.softmax(dim=-1).topk(2, dim=-1).values
        return top2[..., 0] - top2[..., 1]
    if metric == "logit_margin":
        top2 = z.topk(2, dim=-1).values
        return top2[..., 0] - top2[..., 1]
    raise ValueError(f"unknown confidence metric {metric!r}")


# ── 1. HeadGate ──────────────────────────────────────────────────────────

class HeadGate(nn.Module):
    """Per-head zero-gating wrapper for a GQA ``out_proj``.

    Sits where ``attn.out_proj`` used to live: the input is the flattened
    per-head attention output (B,T,H*Dh).  Each call computes the per-head
    L2 norm (the head's output-contribution magnitude), folds it into an
    EMA saliency score, and multiplies head_dim-blocks of the input by a
    0/1 gate before the wrapped linear runs.

    EMA is kept as a plain Python list: one small ``.tolist()`` D2H read
    per update, and the gate tensor is rebuilt from host floats per call —
    so no inference-mode tensor can ever leak into a later grad-tracking
    forward (inference tensors saved for backward are a hard error).

    Saliency is measured on the *raw* (ungated) input, so a gated head
    whose output becomes important again can recover on the next step —
    gating is live, not sticky.
    """

    def __init__(self, linear, n_heads: int, head_dim: int,
                 ema_alpha: float = 0.9, tau: float = 0.0,
                 tau_mode: str = "rel_mean", update_every: int = 1,
                 enabled: bool = True):
        super().__init__()
        self.linear = linear          # the module called after gating
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.ema_alpha = float(ema_alpha)
        self.tau = float(tau)
        self.tau_mode = tau_mode      # abs | rel_mean | rel_max | frac | off
        self.update_every = max(1, int(update_every))
        self.enabled = enabled
        self.ema: list[float] | None = None
        self.last_gate: list[float] | None = None
        self.last_sal: list[float] | None = None
        self.calls = 0

    # -- nn.Linear passthroughs (stack code reads .weight / .bias / dims) --
    @property
    def weight(self):
        lin = find_linear(self.linear)
        return lin.weight if lin is not None else None

    @property
    def bias(self):
        lin = find_linear(self.linear)
        return getattr(lin, "bias", None) if lin is not None else None

    @property
    def in_features(self):
        return self.n_heads * self.head_dim

    @property
    def out_features(self):
        lin = find_linear(self.linear)
        return lin.out_features if lin is not None else -1

    def reset_ema(self):
        self.ema = None

    def _update_ema(self, sal: torch.Tensor):
        vals = sal.tolist()                       # one small D2H sync
        self.last_sal = vals
        if self.ema is None:
            self.ema = list(vals)
        else:
            a = self.ema_alpha
            self.ema = [a * e + (1.0 - a) * v
                        for e, v in zip(self.ema, vals)]

    def gate_list(self) -> list[float] | None:
        """0/1 gate per head from the current EMA.  Never gates every
        head — the top-EMA head is always kept so a layer can't collapse.
        Returns None when gating is a no-op (cold EMA / tau<=0 / off)."""
        if self.ema is None or self.tau <= 0.0 or self.tau_mode == "off":
            return None
        ema, H = self.ema, len(self.ema)
        if self.tau_mode == "abs":
            thr = self.tau
            gate = [0.0 if e < thr else 1.0 for e in ema]
        elif self.tau_mode == "rel_mean":
            thr = self.tau * (sum(ema) / H)
            gate = [0.0 if e < thr else 1.0 for e in ema]
        elif self.tau_mode == "rel_max":
            thr = self.tau * max(ema)
            gate = [0.0 if e < thr else 1.0 for e in ema]
        elif self.tau_mode == "frac":
            # Gate exactly the lowest-saliency floor(tau*H) heads.
            n_gate = int(self.tau * H)
            gate = [1.0] * H
            for h in sorted(range(H), key=lambda j: ema[j])[:n_gate]:
                gate[h] = 0.0
        else:
            raise ValueError(f"unknown tau_mode {self.tau_mode!r}")
        if all(g == 0.0 for g in gate):
            gate[max(range(H), key=lambda j: ema[j])] = 1.0
        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.linear(x)
        B, T, C = x.shape
        xr = x.reshape(B, T, self.n_heads, self.head_dim)
        self.calls += 1
        if self.ema is None or self.calls % self.update_every == 0:
            with torch.no_grad():
                sal = xr.float().norm(dim=-1).mean(dim=(0, 1))   # (H,)
            self._update_ema(sal)
        gate = self.gate_list()
        self.last_gate = gate
        if gate is None:
            return self.linear(x)
        g = xr.new_tensor(gate).view(1, 1, self.n_heads, 1)
        return self.linear((xr * g).reshape(B, T, C))


class HeadGateController:
    """Attaches/manages ``HeadGate`` wrappers on every GQA-style block.

    Duck-typed attach: a block is wrapped iff ``block.attn`` exposes
    ``n_heads``, ``head_dim`` and an ``out_proj`` chain ending in an
    nn.Linear with ``in_features == n_heads*head_dim`` (Qwen2.5 GQA:
    14×64=896=d_model).  Non-GQA blocks are skipped untouched.
    """

    def __init__(self, model, ema_alpha: float = 0.9, tau: float = 0.0,
                 tau_mode: str = "rel_mean", update_every: int = 1):
        self.model = model
        self.ema_alpha = ema_alpha
        self.tau = tau
        self.tau_mode = tau_mode
        self.update_every = update_every
        self.gates: dict[int, HeadGate] = {}
        self._origs: dict[int, nn.Module] = {}

    @property
    def attached(self) -> bool:
        return bool(self.gates)

    def attach(self) -> int:
        n = 0
        for i, block in enumerate(self.model.blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            op = getattr(attn, "out_proj", None)
            if isinstance(op, HeadGate):           # already wrapped: re-arm
                op.tau, op.tau_mode = self.tau, self.tau_mode
                op.ema_alpha, op.update_every = self.ema_alpha, self.update_every
                self.gates[i] = op
                continue
            H = getattr(attn, "n_heads", None)
            Dh = getattr(attn, "head_dim", None)
            lin = find_linear(op)
            if (lin is None or H is None or Dh is None
                    or lin.in_features != int(H) * int(Dh)):
                continue
            w = HeadGate(op, H, Dh, ema_alpha=self.ema_alpha, tau=self.tau,
                         tau_mode=self.tau_mode, update_every=self.update_every)
            attn.out_proj = w
            self.gates[i] = w
            self._origs[i] = op
            n += 1
        return n

    def detach(self):
        """Restore every wrapped out_proj to its original module."""
        for i, op in self._origs.items():
            self.model.blocks[i].attn.out_proj = op
        self.gates.clear()
        self._origs.clear()

    def set_tau(self, tau: float, mode: str | None = None):
        self.tau = tau
        if mode is not None:
            self.tau_mode = mode
        for g in self.gates.values():
            g.tau = tau
            if mode is not None:
                g.tau_mode = mode

    def set_enabled(self, flag: bool):
        for g in self.gates.values():
            g.enabled = flag

    def reset_ema(self):
        for g in self.gates.values():
            g.reset_ema()

    def stats(self) -> dict:
        """Fraction of heads currently zero-gated + saliency summary."""
        total = gated = 0
        per_layer = {}
        ema_all: list[float] = []
        for i, g in self.gates.items():
            H = g.n_heads
            total += H
            gate = g.last_gate or [1.0] * H
            ng = sum(1 for v in gate if v == 0.0)
            gated += ng
            per_layer[i] = ng
            if g.ema:
                ema_all.extend(g.ema)
        out = {
            "frac_gated": (gated / total) if total else 0.0,
            "heads_gated": gated,
            "heads_total": total,
            "per_layer_gated": per_layer,
        }
        if ema_all:
            out["ema_min"] = round(min(ema_all), 4)
            out["ema_max"] = round(max(ema_all), 4)
        return out


# ── 2. EarlyExitController ───────────────────────────────────────────────

class EarlyExitController:
    """Confidence-gated per-token early exit over ``model.blocks``.

    ``decode()`` replicates the model's own decode loop block-by-block.
    After each block (once ``depth >= min_depth``, then every
    ``check_every`` blocks, and always at the last block) the hidden is
    pushed through ln_f + head; if the confidence metric clears ``tau``
    the step stops early — remaining blocks don't run and their KV caches
    keep a shorter history (skipped-token semantics).  ``position_ids``
    are always passed explicitly so RoPE uses absolute positions even on
    layers whose cache lagged behind.

    ``exit_nll()`` is the teacher-forced twin for held-out NLL: every
    position independently exits at the first depth where confidence >=
    tau; positions that never pass use final-depth logits.
    """

    def __init__(self, eng=None, model=None):
        self.eng = eng
        self.model = model if model is not None else eng.model

    # -- internals --------------------------------------------------------

    def _head_logits(self, x_last: torch.Tensor) -> torch.Tensor:
        m = self.model
        h = m.ln_f(x_last) if getattr(m, "ln_f", None) is not None else x_last
        return m.head(h)

    def _conf(self, logits: torch.Tensor, metric: str) -> torch.Tensor:
        return confidence(logits, metric)

    @staticmethod
    def _check_now(i: int, n_blocks: int, min_depth: int, check_every: int) -> bool:
        d = i + 1
        return (d >= min_depth
                and (d % check_every == 0 or d == n_blocks))

    def _reset_recurrent(self):
        """Mirror _forward_impl's new-sequence reset for conv/SSM/side-path
        layers (no-ops on pure-GQA stacks like Qwen2.5)."""
        for block in self.model.blocks:
            attn = getattr(block, "attn", None)
            if attn is not None:
                if hasattr(attn, "_conv_state"):
                    attn._conv_state_reset = True
                if hasattr(attn, "_ssm_state"):
                    attn._ssm_state = None
                if hasattr(attn, "_ssm_state_real"):
                    attn._ssm_state_real = None
                    attn._ssm_state_imag = None
            kda = getattr(block, "_kda", None)
            if kda is not None:
                kda._state_reset = True
            g2 = getattr(block, "_gdn2", None)
            if g2 is not None:
                g2._state_reset = True
            fh = getattr(block, "_forge_hybrid_ssm", None)
            if fh is not None:
                if hasattr(fh, "_ssm_state"):
                    fh._ssm_state = None
                if hasattr(fh, "_conv_state"):
                    fh._conv_state_reset = True

    def _embed(self, tok: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = m.embed(tok)
        if getattr(m, "embed_norm", None) is not None:
            x = m.embed_norm(x)
        if getattr(m, "ngram_embed", None) is not None:
            x = m.ngram_embed(tok, x)
        return x

    def _attn_res_active(self) -> bool:
        m = self.model
        ar = getattr(m, "_attn_res", None)
        return ar is not None and not getattr(m, "_attn_res_gate_zero", True)

    # -- decode -----------------------------------------------------------

    @torch.inference_mode()
    def decode(self, logits: torch.Tensor, past, start_pos: int,
               n_tokens: int, tau: float | None = None, metric: str = "margin",
               min_depth: int = 1, check_every: int = 1) -> dict:
        """Greedy decode ``n_tokens`` steps from (logits, past) post-prefill.

        logits: (B,1,V) or (B,T,V) — only [:, -1] is read.
        past:   list of per-layer (k,v) tuples (None entries allowed).
        tau:    None → plain full-depth manual loop (no exit checks).

        Returns dict(tokens, depths, mean_depth, logits, past).
        """
        m = self.model
        B = logits.shape[0]
        n_blocks = len(m.blocks)
        toks, depths = [], []
        pos = start_pos
        use_ar = self._attn_res_active()
        for _step in range(n_tokens):
            tok = logits[:, -1].argmax(-1, keepdim=True)
            toks.append(int(tok[0, 0]))
            x = self._embed(tok)
            pos_ids = torch.full((B, 1), pos, dtype=torch.long,
                                 device=x.device)
            new_past = list(past)
            past_outputs = [] if use_ar else None
            lg = None
            used = n_blocks
            for i, block in enumerate(m.blocks):
                x, present = block(x, past_key_value=past[i], use_cache=True,
                                   layer_idx=i, position_ids=pos_ids)
                if present is not None:
                    new_past[i] = present
                if use_ar:
                    x = x + m._attn_res(x, i, past_outputs)
                    past_outputs.append(x)
                if tau is not None and self._check_now(
                        i, n_blocks, min_depth, check_every):
                    lg = self._head_logits(x[:, -1:])
                    if bool((self._conf(lg, metric) >= tau).all()):
                        used = i + 1
                        break
            if lg is None:
                lg = self._head_logits(x[:, -1:])
            depths.append(used)
            logits = lg
            past = new_past
            pos += 1
        return {"tokens": toks, "depths": depths,
                "mean_depth": sum(depths) / max(len(depths), 1),
                "logits": logits, "past": past}

    # -- teacher-forced NLL under the same first-exit policy -------------

    @torch.inference_mode()
    def exit_nll(self, ids: torch.Tensor, tau: float | None,
                 metric: str = "prob", min_depth: int = 1,
                 check_every: int = 1) -> tuple[float, float]:
        """First-exit NLL on ``ids`` (B,T).

        Each position exits at the first block whose confidence >= tau —
        the same policy decode applies per token.  Positions that never
        pass are scored at final depth.  Returns (nll, mean_exit_depth).
        ``tau=None`` → plain full-depth NLL via model() (exact parity).
        """
        m = self.model
        ids = ids.long()
        n_blocks = len(m.blocks)
        if tau is None:
            out = m(ids[:, :-1], targets=ids[:, 1:])
            return float(out[1]), float(n_blocks)
        self._reset_recurrent()
        x = self._embed(ids[:, :-1])
        tgt = ids[:, 1:]
        B, T = x.shape[:2]
        done = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        use_ar = self._attn_res_active()
        past_outputs = [] if use_ar else None
        ce_sum, dep_sum = 0.0, 0.0
        lg = None
        for i, block in enumerate(m.blocks):
            x, _ = block(x, past_key_value=None, use_cache=False, layer_idx=i)
            if use_ar:
                x = x + m._attn_res(x, i, past_outputs)
                past_outputs.append(x)
            if not self._check_now(i, n_blocks, min_depth, check_every):
                continue
            lg = self._head_logits(x)
            c = self._conf(lg, metric)                 # (B,T)
            take = (~done) & (c >= tau)
            if bool(take.any()):
                ce_sum += float(F.cross_entropy(
                    lg[take].float(), tgt[take], reduction="sum"))
                dep_sum += float(take.sum().item()) * (i + 1)
                done |= take
            if bool(done.all()):
                lg = None
                break
        left = ~done
        if bool(left.any()):
            # Never passed the threshold — scored at the deepest depth run.
            if lg is None:
                lg = self._head_logits(x)
            ce_sum += float(F.cross_entropy(
                lg[left].float(), tgt[left], reduction="sum"))
            dep_sum += float(left.sum().item()) * n_blocks
        n = B * T
        return ce_sum / n, dep_sum / n


# ── 3. DiagTempScaler (bonus) ────────────────────────────────────────────

class OutGain(nn.Module):
    """Diagonal output gain: ``y = inner(x) ⊙ gain`` over d_model channels.

    Gain is fp32 for stable AdamW, cast to x's dtype at apply time.
    """

    def __init__(self, inner, d_model: int, device=None):
        super().__init__()
        self.inner = inner
        self.gain = nn.Parameter(
            torch.ones(d_model, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) * self.gain.to(x.dtype)


class DiagTempScaler:
    """Per-block diagonal "temperature" gains, folded into weights after
    a short recalibration.

    One gain vector on each block's attention out_proj output and one on
    each FFN w_down output (both produce d_model channels).  With the
    substrate frozen, the gains are AdamW-recalibrated on wikitext to
    recover NLL lost to head gating, then ``fold()`` multiplies them into
    the Linear weight *rows* (out channel j ← W[j,:] * gain[j]) and
    unwraps — zero runtime cost.
    """

    def __init__(self, model):
        self.model = model
        self.wraps: list[dict] = []

    def attach(self) -> int:
        d = self.model.config.d_model
        for block in self.model.blocks:
            attn = getattr(block, "attn", None)
            op = getattr(attn, "out_proj", None) if attn is not None else None
            lin = find_linear(op)
            if (lin is not None and lin.out_features == d
                    and not isinstance(op, OutGain)):
                g = OutGain(op, d, device=lin.weight.device)
                attn.out_proj = g
                self.wraps.append({"parent": attn, "attr": "out_proj",
                                   "gain_mod": g, "inner": op, "linear": lin})
            ffn = getattr(block, "ffn", None)
            wd = getattr(ffn, "w_down", None) if ffn is not None else None
            if isinstance(wd, nn.Linear) and wd.out_features == d:
                g = OutGain(wd, d, device=wd.weight.device)
                ffn.w_down = g
                self.wraps.append({"parent": ffn, "attr": "w_down",
                                   "gain_mod": g, "inner": wd, "linear": wd})
        return len(self.wraps)

    def params(self) -> list[nn.Parameter]:
        return [w["gain_mod"].gain for w in self.wraps]

    def detach(self):
        """Unwrap without folding (gains discarded)."""
        for w in self.wraps:
            setattr(w["parent"], w["attr"], w["inner"])
        self.wraps.clear()

    def fold(self):
        """Multiply each gain into its Linear's weight rows, then unwrap."""
        for w in self.wraps:
            lin = w["linear"]
            g = w["gain_mod"].gain.detach()
            with torch.no_grad():
                lin.weight.mul_(g.to(lin.weight.dtype).unsqueeze(1))
                if getattr(lin, "bias", None) is not None:
                    lin.bias.mul_(g.to(lin.bias.dtype))
            setattr(w["parent"], w["attr"], w["inner"])
        self.wraps.clear()

    def recalibrate(self, next_ids, steps: int = 200, lr: float = 5e-3,
                    log_every: int = 0) -> list[float]:
        """Adam on the gain vectors only; everything else frozen.

        Plain Adam, NOT AdamW: decoupled weight decay would pull the
        gains toward 0 (collapsing residual contributions) — the correct
        prior is staying at the 1.0 init, so no decay at all.

        next_ids: callable(step:int) -> (1,T) long tensor on the model's
        device.  Returns the per-step loss list.
        """
        m = self.model
        gain_ids = {id(p) for p in self.params()}
        flags = [(p, p.requires_grad) for p in m.parameters()]
        for p in m.parameters():
            p.requires_grad_(id(p) in gain_ids)
        opt = torch.optim.Adam(self.params(), lr=lr, betas=(0.9, 0.95))
        losses: list[float] = []
        try:
            for step in range(steps):
                ids = next_ids(step)
                out = m(ids[:, :-1].long(), targets=ids[:, 1:].long())
                loss = out[1]
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
                if log_every and (step % log_every == 0 or step == steps - 1):
                    print(f"  [diagtemp] step {step} loss={float(loss):.4f}")
        finally:
            for p, f in flags:
                p.requires_grad_(f)
        return losses
