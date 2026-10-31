"""EXP-prevtok: previous-token prediction head + self-correcting decode.

Round E-prevtok on Qwen2.5-0.5B (Forge arch: d_model=896, 24 blocks,
vocab=151936).

Pieces:
  * PrevTokenHead — tied-embedding backward readout: predict token
    t-1 from the final hidden state h_t via
    scale * (adapter(h_t) @ E^T) + bias, where E is the model's own
    embedding matrix (no free d x V projection — see class docstring
    for why the 136M-param original memorized instead of learning).
  * train_prev_head / eval_prev_head — AdamW trainer on frozen-model
    hidden states, CE against ids[:, :-1] from hidden[:, 1:].
  * SelfCorrectingDecoder — greedy decode with a backward-consistency
    check: after committing token x_t, prev_head(h_t) must assign
    enough probability to the actual previous token x_{t-1}.  On
    failure the top-K alternate candidates are re-decoded in ONE
    batched forward (KV expanded to K rows), scored by
    logp_next + alpha * logp_prev, and the winner replaces x_t.
    Optional single-token rollback when even the best candidate fails.
  * mode="rerank" — always-on variant: every step decodes the top-K
    candidates batched and picks argmax of the combined score.
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────────── PrevTokenHead ─────────────────────────────


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-8)


class PrevTokenHead(nn.Module):
    """Backward LM head: h_t -> logits over token t-1.

    Design (post-mortem fix).  The original head was a free
    nn.Linear(d_model -> vocab) = 136M params in fp32 (~545MB weight +
    ~1.1GB AdamW state).  On a handful of wikitext docs it memorized
    the train set — train acc -> 1.0 while held-out prev_top1 stayed
    ~0.1 — and under bnb AdamW8bit the run diverged to ~1e5 loss.

    Two measured fixes:

    1. TIED READOUT — the output projection is the model's embedding
       matrix E [V, d] (the idiomatic LM-head approach), not a free
       linear.  Learned params drop to ~2.3M and the readout basis is
       frozen, so scoring well requires finding REAL structure instead
       of memorizing (each adapter row is constrained by thousands of
       tokens).
    2. EARLY-LAYER TAPS — a layer sweep showed x_{t-1} is linearly
       decodable best from EARLY block outputs (layer 2 eval top1
       ~0.45 vs ~0.17 at the final post-norm hidden; deeper layers
       overfit).  With tap_layers=(2, 9) — the driver setting — the
       head reads RMSNorm'd per-layer hiddens concatenated -> din = 2d.

    Feature:  f_t = cat[rms(h_t^l) for l in tap_layers]
    Adapter:  z = f @ A^T + W2 @ gelu(W1 @ f)   (A init ~= mean-over-
              taps identity; W1 random, W2 zero -> pass-through at init)
    Logits:   scale * (z @ E^T) + bias

    At init the head emits (approx) the model's own next-token logits —
    a sane prior.  Params: din*d + din*512 + 512*d + V + 1 ~ 2.3M
    (~9MB checkpoint instead of 545MB).

    tap_layers=() (default) consumes a single [*, d] hidden state
    directly — used by the synthetic CPU tests; the driver passes
    tap_layers=(2, 9).
    """

    def __init__(self, d_model: int, vocab_size: int, bias: bool = True,
                 init_from: torch.Tensor | None = None,
                 out_weight: torch.Tensor | None = None,
                 tap_layers: tuple[int, ...] = (),
                 mlp_hidden: int = 512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tap_layers = tuple(tap_layers)
        self.din = d_model * max(1, len(self.tap_layers))
        # Linear path + zero-init residual MLP (pass-through at init).
        self.adapter = nn.Linear(self.din, d_model, bias=False)
        self.mlp1 = nn.Linear(self.din, mlp_hidden, bias=False)
        self.mlp2 = nn.Linear(mlp_hidden, d_model, bias=False)
        with torch.no_grad():
            w = torch.zeros(d_model, self.din)
            ntap = max(1, len(self.tap_layers))
            for i in range(ntap):
                w[:, i * d_model:(i + 1) * d_model] = (
                    torch.eye(d_model) / ntap)
            self.adapter.weight.copy_(w)
            self.mlp2.weight.zero_()
        self.scale = nn.Parameter(torch.ones(()))
        self.bias = nn.Parameter(torch.zeros(vocab_size)) if bias else None
        w = out_weight if out_weight is not None else init_from
        if w is not None:
            # Tied readout: register as a NON-persistent buffer so the
            # embedding isn't duplicated in the checkpoint.  The buffer
            # is detached — gradients never reach the embedding.
            self.tied = True
            self.register_buffer("out_w", w.detach(), persistent=False)
        else:
            self.tied = False
            self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def _out_weight(self) -> torch.Tensor:
        return self.out_w if self.tied else self.proj.weight

    def featurize(self, hidden_or_states):
        """Build the probe input.

        hidden_or_states: either a single [*, d] tensor (when
        tap_layers is empty — returned unchanged) or a list/tuple of
        per-layer hidden states [*, d] as returned by the model when
        return_hidden_states=True.  Returns [*, din].
        """
        if not self.tap_layers:
            h = hidden_or_states
            if isinstance(h, (list, tuple)):
                h = h[-1]
            return h
        if torch.is_tensor(hidden_or_states):
            raise ValueError(
                "PrevTokenHead has tap_layers but got a single hidden "
                "tensor — pass the per-layer hidden_states list "
                "(model(..., return_hidden_states=True))")
        parts = [_rms(hidden_or_states[l].float())
                 for l in self.tap_layers]
        return torch.cat(parts, dim=-1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Featurized input [*, din] -> [*, vocab] fp32 logits."""
        f = feat.float()
        z = self.adapter(f) + self.mlp2(F.gelu(self.mlp1(f)))
        w = self._out_weight()
        logits = F.linear(z.to(w.dtype), w).float()
        logits = logits * self.scale
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    def prev_log_probs(self, feat: torch.Tensor) -> torch.Tensor:
        """log p(x_{t-1} | f_t): [..., din] -> [..., vocab] fp32."""
        return F.log_softmax(self(feat), dim=-1)

    def prev_prob_of(self, feat: torch.Tensor,
                     prev_id: int) -> tuple[float, float]:
        """(prob, log_prob) that the head assigns to the actual previous
        token.  feat: featurized [din] or [1, din]."""
        lp = self.prev_log_probs(feat.reshape(1, -1))[0]
        return lp[prev_id].exp().item(), lp[prev_id].item()


def prev_ce_loss(head: PrevTokenHead, hidden: torch.Tensor,
                 ids: torch.Tensor) -> tuple[torch.Tensor, float]:
    """CE of predicting ids[:, :-1] from hidden[:, 1:].  Returns
    (loss, top1_acc).  ids: [B, T] long, hidden: [B, T, d_model]."""
    logits = head(hidden[:, 1:].float())               # [B, T-1, V]
    tgt = ids[:, :-1].long()                            # [B, T-1]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    acc = (logits.argmax(-1) == tgt).float().mean().item()
    return loss, acc


# ───────────────────────────── training / eval ───────────────────────────


def _forward_feats(model, head: PrevTokenHead, ids: torch.Tensor,
                   use_autocast: bool) -> torch.Tensor:
    """Frozen-model probe features [B, T, din].

    With tap_layers set, requests per-layer hidden states
    (return_hidden_states=True) and applies head.featurize; otherwise
    returns the final post-norm hidden state.
    """
    dev = ids.device
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_autocast and dev.type == "cuda"
           else torch.autocast(device_type="cpu", enabled=False))
    with torch.no_grad(), ctx:
        out = model(ids, return_hidden=True, logits_last_only=True,
                    return_hidden_states=bool(head.tap_layers))
    # out[-1] is hidden_states_list when requested, else hidden.
    return head.featurize(out[-1])


def train_prev_head(model, head: PrevTokenHead,
                    doc_ids: list[torch.Tensor], steps: int = 400,
                    lr: float = 3e-4, weight_decay: float = 0.01,
                    optimizer: str = "adamw",
                    log_every: int = 50, shuffle_seed: int = 0,
                    use_autocast: bool = True, ce_chunk: int = 64,
                    time_budget_s: float = 280.0) -> dict:
    """AdamW on the head only; model weights frozen (no_grad forward).

    doc_ids: list of [1, T_i] long tensors on the model device.  Steps
    cycle over a seeded permutation of the docs.

    optimizer: "adamw" (fp32 moments, ~1.1GB at vocab=152k) |
    "cpu_adamw" (forge.training.optim.hybrid_offload.CPUAdamW —
    fp32 moments + master in CPU pinned RAM, GPU sees only weight+grad;
    keeps train peak under 4GB on the RTX 5070; identical math) |
    "sgdm" (momentum SGD).  NOTE: bnb AdamW8bit diverges on this
    workload (loss explodes ~1e5) — do not use.
    """
    import random
    rng = random.Random(shuffle_seed)
    order = list(range(len(doc_ids)))
    rng.shuffle(order)

    for p in model.parameters():
        p.requires_grad_(False)
    if optimizer == "cpu_adamw":
        from forge.training.optim.hybrid_offload import CPUAdamW
        opt = CPUAdamW(head.parameters(), lr=lr,
                       weight_decay=weight_decay, verbose=False)
    elif optimizer == "sgdm":
        opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=lr,
                                weight_decay=weight_decay)
    hist: list[dict] = []
    loss_v = acc = 0.0
    t0 = time.perf_counter()
    step = 0
    while step < steps:
        ids = doc_ids[order[step % len(order)]]
        hidden = _forward_feats(model, head, ids, use_autocast)
        # Chunked backward: one [chunk, V] fp32 logit block alive at a
        # time (~39MB at chunk=64) instead of the full [T, V] block
        # (~310MB at T=256) — keeps train peak VRAM under 4GB.
        opt.zero_grad(set_to_none=True)
        n_pairs = hidden.shape[1] - 1
        tot, hit, seen = 0.0, 0, 0
        for s in range(0, n_pairs, ce_chunk):
            e = min(s + ce_chunk, n_pairs)
            lc = head(hidden[:, 1 + s:1 + e].float())
            tc = ids[:, s:e].long().reshape(-1)
            lsum = F.cross_entropy(
                lc.reshape(-1, lc.size(-1)), tc, reduction="sum")
            lsum.backward()
            tot += lsum.item()
            hit += (lc.argmax(-1).reshape(-1) == tc).sum().item()
            seen += tc.numel()
        opt.step()
        loss_v, acc = tot / max(seen, 1), hit / max(seen, 1)
        step += 1
        if step % log_every == 0 or step == 1:
            hist.append({"step": step, "loss": round(loss_v, 4),
                         "acc": round(acc, 4),
                         "elapsed_s": round(time.perf_counter() - t0, 1)})
        if time.perf_counter() - t0 > time_budget_s:
            break
    return {"steps": step, "history": hist,
            "train_s": round(time.perf_counter() - t0, 1),
            "final_loss": round(loss_v, 4),
            "final_acc": round(acc, 4)}


@torch.no_grad()
def eval_prev_head(model, head: PrevTokenHead,
                   doc_ids: list[torch.Tensor],
                   use_autocast: bool = True) -> dict:
    """Held-out backward NLL + top-1 accuracy."""
    nll_sum, n_tok, hit, p_sum = 0.0, 0, 0, 0.0
    p_all = []
    for ids in doc_ids:
        hidden = _forward_feats(model, head, ids, use_autocast)
        lp = head.prev_log_probs(hidden[:, 1:])          # [1, T-1, V]
        tgt = ids[:, :-1].long()
        nll_sum += F.nll_loss(
            lp.reshape(-1, lp.size(-1)), tgt.reshape(-1),
            reduction="sum").item()
        hit += (lp.argmax(-1) == tgt).sum().item()
        p = lp.exp().gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        p_all.append(p.flatten().float().cpu())
        p_sum += p.sum().item()
        n_tok += tgt.numel()
    out = {"prev_nll": nll_sum / max(n_tok, 1),
           "prev_top1": hit / max(n_tok, 1), "n_tokens": n_tok,
           "mean_prev_prob": p_sum / max(n_tok, 1)}
    if p_all:
        pc = torch.cat(p_all)
        for q in (0.05, 0.15, 0.30, 0.50):
            out[f"pprev_q{int(q * 100):02d}"] = float(
                pc.quantile(q).item())
    return out


@torch.inference_mode()
def seq_prev_nll(model, head: PrevTokenHead, ids: torch.Tensor,
                 use_autocast: bool = True) -> float:
    """Backward NLL of a full sequence: -log p(x_{t-1} | h_t) averaged
    over all adjacent pairs.  One prefill forward; used to score how
    "self-consistent" a generated sequence is after the fact."""
    hidden = _forward_feats(model, head, ids, use_autocast)
    lp = head.prev_log_probs(hidden[:, 1:])
    tgt = ids[:, :-1].long()
    return F.nll_loss(
        lp.reshape(-1, lp.size(-1)), tgt.reshape(-1)).item()


def save_head(head: PrevTokenHead, path: str,
              extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": head.state_dict(),
               "d_model": head.d_model, "vocab_size": head.vocab_size,
               "tied": head.tied, "tap_layers": head.tap_layers,
               "mlp_hidden": head.mlp2.in_features}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_head(path: str, device: str = "cuda",
              out_weight: torch.Tensor | None = None) -> PrevTokenHead:
    """Reload a saved head.  For tied heads (tied=True in the payload —
    the embedding rows are NOT stored) pass
    ``out_weight=model.embed.weight``."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    tied = payload.get("tied", False)
    if tied and out_weight is None:
        raise ValueError(
            f"{path} holds a tied PrevTokenHead — pass "
            f"out_weight=model.embed.weight to reconstruct the readout")
    head = PrevTokenHead(payload["d_model"], payload["vocab_size"],
                         out_weight=out_weight if tied else None,
                         tap_layers=payload.get("tap_layers", ()),
                         mlp_hidden=payload.get("mlp_hidden", 512))
    head.load_state_dict(payload["state_dict"])
    return head.to(device)


# ───────────────────────────── KV cache helpers ──────────────────────────


def _is_kv_entry(e) -> bool:
    return (isinstance(e, tuple) and len(e) == 2
            and torch.is_tensor(e[0]) and e[0].dim() >= 3)


def _kv_map(past, fn):
    """Apply fn(k, v) to each (k, v) tuple entry; pass everything else
    through (None, dict conv/SSM state, ...)."""
    if past is None:
        return None
    out = []
    for e in past:
        out.append(fn(e[0], e[1]) if _is_kv_entry(e) else e)
    return type(past)(out) if isinstance(past, tuple) else out


def kv_truncate(past, n: int):
    """Drop the last n cached positions (T is dim -2 in [B, H, T, D])."""
    if n <= 0:
        return past
    return _kv_map(past, lambda k, v: (k[..., :-n, :].contiguous(),
                                       v[..., :-n, :].contiguous()))


def kv_expand(past, n_rows: int):
    """B=1 -> B=n_rows by repeating the batch dim."""
    return _kv_map(past, lambda k, v: (
        k.repeat(n_rows, *([1] * (k.dim() - 1))).contiguous(),
        v.repeat(n_rows, *([1] * (v.dim() - 1))).contiguous()))


def kv_select(past, j: int):
    """Keep only batch row j (B stays 1)."""
    return _kv_map(past, lambda k, v: (k[j:j + 1].contiguous(),
                                       v[j:j + 1].contiguous()))


# ───────────────────────── SelfCorrectingDecoder ─────────────────────────


class SelfCorrectingDecoder:
    """Greedy decode with backward-consistency self-correction.

    model must implement the Forge forward contract:
      model(ids[B,T], return_hidden=True, logits_last_only=True)
          -> (logits, loss, hidden)
      model(tok[B,1], past_key_values=past, use_cache=True,
            return_hidden=True)
          -> (logits, loss, presents, hidden)
      model(tok[B,1], past_key_values=past, use_cache=True)
          -> (logits, loss, presents)

    B=1 only.  All decode under inference_mode.
    """

    def __init__(self, model, prev_head: PrevTokenHead | None,
                 threshold: float = 0.05, n_cands: int = 4,
                 alpha: float = 1.0, check_every: int = 1,
                 rollback: bool = False, max_rollbacks: int = 8):
        self.model = model
        self.head = prev_head
        self.threshold = threshold
        self.n_cands = max(2, n_cands)
        self.alpha = alpha
        self.check_every = max(1, check_every)
        self.rollback = rollback
        self.max_rollbacks = max_rollbacks

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, v: float):
        self._threshold = v
        self.log_threshold = math.log(max(v, 1e-30))

    # -- low-level steps --------------------------------------------------

    def _want_hs(self) -> bool:
        return bool(self.head is not None and self.head.tap_layers)

    def _step(self, tok: torch.Tensor, past):
        """One cached decode step. tok [B,1] ->
        (logits [B,1,V], feat [B,1,din], past')."""
        out = self.model(tok, past_key_values=past, use_cache=True,
                         return_hidden=True,
                         return_hidden_states=self._want_hs())
        return out[0], self.head.featurize(out[-1]), out[2]

    def _prefill(self, ids: torch.Tensor):
        out = self.model(ids, use_cache=True, return_hidden=True,
                         logits_last_only=True)
        return out[0], out[-1], out[2]

    def _prev_logp(self, h: torch.Tensor, prev_id: int) -> torch.Tensor:
        """log p(prev_id | h); h [B,d] or [B,1,d] -> [B]."""
        lp = self.head.prev_log_probs(h.reshape(h.shape[0], -1))
        return lp[:, prev_id]

    # -- batched candidate pick -------------------------------------------

    def _batched_pick(self, lp: torch.Tensor, past_ctx, prev_tok: int,
                      exclude: set[int]):
        """Score top-K candidates in ONE batched forward.

        lp: [V] next-token log-probs for the position being picked.
        past_ctx: KV state at the position BEFORE the new token (the
        token being replaced must already be truncated out).
        Returns (next_logits [1,1,V], past', x, acc_prev_logp).
        """
        dev = lp.device
        K = min(self.n_cands + len(exclude), lp.numel())
        cands = lp.topk(K).indices.tolist()
        cands = [c for c in cands if c not in exclude][: self.n_cands]
        if not cands:                       # nothing left — keep argmax
            cands = [int(lp.argmax().item())]
        ct = torch.tensor(cands, device=dev, dtype=torch.long).view(-1, 1)
        pastK = kv_expand(past_ctx, len(cands))
        out = self.model(ct, past_key_values=pastK, use_cache=True,
                         return_hidden=True,
                         return_hidden_states=self._want_hs())
        logitsK, hK, pastK = out[0], self.head.featurize(out[-1]), out[2]
        plp_c = self._prev_logp(hK.squeeze(1), prev_tok)   # [K]
        lp_c = lp[torch.tensor(cands, device=dev)]         # [K]
        score = lp_c + self.alpha * plp_c
        j = int(score.argmax().item())
        return (logitsK[j:j + 1], kv_select(pastK, j), cands[j],
                float(plp_c[j].item()))

    # -- generation --------------------------------------------------------

    @torch.inference_mode()
    def generate_baseline(self, ids: torch.Tensor, n_tokens: int) -> dict:
        """Plain greedy decode — same instrumentation, no checks."""
        from forge.model_loader import unpack_output_with_kv
        dev = ids.device
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model(ids, use_cache=True, logits_last_only=True)
        logits, past = unpack_output_with_kv(out)
        forwards = 1
        gen, nlls = [], []
        while len(gen) < n_tokens:
            lp = F.log_softmax(logits[:, -1].float(), dim=-1)[0]
            x = int(lp.argmax().item())
            out = self.model(torch.tensor([[x]], device=dev),
                             past_key_values=past, use_cache=True)
            logits, past = unpack_output_with_kv(out)
            forwards += 1
            gen.append(x)
            nlls.append(-lp[x].item())
        if dev.type == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
        return self._stats(gen, nlls, el, forwards)

    @torch.inference_mode()
    def generate(self, ids: torch.Tensor, n_tokens: int,
                 mode: str = "threshold") -> dict:
        """Self-correcting greedy decode.

        mode="threshold": commit argmax, check p_prev every check_every
        steps, batched top-K retry on failure, optional rollback.
        mode="rerank": every step decodes top-K candidates in one
        batched forward and picks argmax(logp_next + a*logp_prev).
        """
        assert ids.shape[0] == 1, "B=1 only"
        assert mode in ("threshold", "rerank")
        assert self.head is not None, "self-correcting decode needs a head"
        dev = ids.device
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, _h, past = self._prefill(ids)
        forwards = 1
        gen, nlls = [], []
        prev_tok = int(ids[0, -1].item())
        lp_prev_pos = None          # lp of previous position (rollback)
        checks = corrections = resamples = extra_fw = rollbacks = 0
        accepted_plp_sum, n_acc = 0.0, 0
        while len(gen) < n_tokens:
            lp = F.log_softmax(logits[:, -1].float(), dim=-1)[0]
            x = int(lp.argmax().item())
            if mode == "rerank":
                # Always-on: score top-K in one batched forward.
                checks += 1
                logits, past, x, acc_plp = self._batched_pick(
                    lp, past, prev_tok, exclude=set())
                forwards += 1
                resamples += self.n_cands - 1
                if x != int(lp.argmax().item()):
                    corrections += 1
                accepted_plp_sum += acc_plp
                n_acc += 1
            elif len(gen) % self.check_every == 0:
                # Threshold path: commit argmax first, verify.
                lg, h, past_new = self._step(
                    torch.tensor([[x]], device=dev), past)
                forwards += 1
                plp = float(self._prev_logp(h, prev_tok)[0].item())
                checks += 1
                if plp >= self.log_threshold:
                    logits, past = lg, past_new
                    accepted_plp_sum += plp
                    n_acc += 1
                else:
                    # Retry: top-K alternates in one batched forward; KV
                    # of the failed token truncated.  The failed argmax
                    # AND the previous token are masked — x_t == x_{t-1}
                    # trivially satisfies the backward check (the head
                    # sees its own embedding in h) and is a degenerate
                    # escape hatch, not a real consistency fix.
                    logits, past, x, acc_plp = self._batched_pick(
                        lp, kv_truncate(past_new, 1), prev_tok,
                        exclude={x, prev_tok})
                    forwards += 1
                    extra_fw += 1
                    resamples += 1
                    corrections += 1      # x is masked — pick always swaps
                    accepted_plp_sum += acc_plp
                    n_acc += 1
                    if (self.rollback and acc_plp < self.log_threshold
                            and len(gen) >= 1
                            and rollbacks < self.max_rollbacks):
                        # Single-token rollback: drop the retried token
                        # AND x_{t-1} from the cache (positions t, t-1),
                        # then resample position t-1 with its old lp.
                        past = kv_truncate(past, 2)
                        bad_prev = gen.pop()
                        nlls.pop()
                        prev_tok = (gen[-1] if gen
                                    else int(ids[0, -1].item()))
                        lp = lp_prev_pos.clone()
                        lp[bad_prev] = float("-inf")
                        x = int(lp.argmax().item())
                        lg, h, past = self._step(
                            torch.tensor([[x]], device=dev), past)
                        logits = lg
                        forwards += 1
                        rollbacks += 1
                        checks += 1
            else:
                lg, _h, past = self._step(
                    torch.tensor([[x]], device=dev), past)
                logits = lg
                forwards += 1
            lp_prev_pos = lp
            gen.append(x)
            nlls.append(-lp[x].item())
            prev_tok = x
        if dev.type == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
        s = self._stats(gen, nlls, el, forwards)
        s.update({"checks": checks, "corrections": corrections,
                  "correction_rate": corrections / max(checks, 1),
                  "resamples": resamples,
                  "extra_forwards": extra_fw,
                  "retry_rate": resamples / max(len(gen), 1),
                  "rollbacks": rollbacks,
                  "mean_accepted_prev_logp":
                      accepted_plp_sum / max(n_acc, 1)})
        return s

    # -- stats -------------------------------------------------------------

    @staticmethod
    def _stats(gen, nlls, el, forwards):
        n = max(len(gen), 1)
        return {"ids": list(gen), "n_tokens": len(gen),
                "nll": sum(nlls) / n, "elapsed_s": round(el, 3),
                "tok_s": round(n / max(el, 1e-6), 1),
                "forwards": forwards}
