"""E-decision: in-model per-decode-step decision head.

A small head on the final-block hidden state picks an action each step:

    CONTINUE=0  normal greedy step
    THINK=1     extra compute: re-run the LAST block once on the current
                hidden (its KV writes are discarded — `use_cache=False`
                only reads the past, it never mutates it) and re-derive
                logits from the refined hidden via ln_f + head.
    STOP=2      end generation
    VERIFY=3    re-check the emitted token: resample between the top-2
                logit candidates with a temperature.

Labels for training come from a teacher-forced decode trace of the frozen
model on wikitext train docs:

    STOP     ground-truth next token is sentence-end / punct-only / EOS
    THINK    base top1-prob below the trace p25 (uncertain step)
    VERIFY   top1-top2 probability margin < verify_delta (contested step)
    CONTINUE otherwise

Priority order (as specified): STOP > THINK > VERIFY > CONTINUE.

Model plumbing (Qwen2.5-0.5B, forge/model/llm.py):
    model(idx, use_cache=True, return_hidden=True)
        -> (logits, loss, presents, hidden)   hidden = post-ln_f [B,T,D]
    blocks[-1](x, past_key_value=past[-1], use_cache=False, layer_idx=L-1,
               position_ids=pos) -> (x', None)   # no KV mutation
"""
from __future__ import annotations

import string
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTINUE, THINK, STOP, VERIFY = 0, 1, 2, 3
ACTIONS = ("CONTINUE", "THINK", "STOP", "VERIFY")
N_ACTIONS = 4

_PUNCT = set(string.punctuation)


def is_sentence_end(tok_text: str, token_id: int, eos_id: int | None = None) -> bool:
    """Heuristic: token ends a sentence or is punctuation-only."""
    if eos_id is not None and token_id == eos_id:
        return True
    s = tok_text.strip()
    if not s:
        return False
    if s[-1] in ".!?":
        return True
    return all(c in _PUNCT for c in s)


class DecisionHead(nn.Module):
    """Linear(896,256) -> SiLU -> Linear(256,4) on last-position hidden."""

    def __init__(self, d_model: int = 896, hidden: int = 256,
                 n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.float())

    def act(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.float()).argmax(-1)


def label_positions(
    top1_prob: torch.Tensor,
    margin: torch.Tensor,
    sent_end: torch.Tensor,
    think_thresh: float,
    verify_delta: float = 0.05,
) -> torch.Tensor:
    """Heuristic per-position labels. Priority STOP > THINK > VERIFY >
    CONTINUE (later assignments win)."""
    labels = torch.full_like(top1_prob, CONTINUE, dtype=torch.long)
    labels[margin < verify_delta] = VERIFY
    labels[top1_prob < think_thresh] = THINK
    labels[sent_end] = STOP
    return labels


@torch.inference_mode()
def collect_trace(eng, docs: list[str], max_len: int = 512,
                  verify_delta: float = 0.05,
                  think_quantile: float = 0.25,
                  device: str = "cuda") -> dict:
    """Teacher-forced trace over docs -> per-position features + labels.

    Runs the frozen model once per doc (no KV), records the post-ln_f
    hidden at each position, top1/top2 probs, margin, entropy, and the
    ground-truth next token. Labels use a think threshold estimated as
    the `think_quantile` of THIS trace's top1-prob distribution — callers
    re-apply the returned `think_thresh` on held-out data.
    """
    model = eng.model
    tok = eng.tokenizer
    eos_id = getattr(tok, "eos_token_id", None)
    feats, p1s, margins, ents, sent_ends, next_ids_all = [], [], [], [], [], []
    _decode_cache: dict[int, str] = {}

    for d in docs:
        ids = tok(d, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 9:
            continue
        idx, targets = ids[:, :-1], ids[:, 1:]
        logits, _, hidden = model(idx, return_hidden=True)
        probs = F.softmax(logits[0].float(), dim=-1)          # [T,V]
        v2, _ = probs.topk(2, dim=-1)
        p1, p2 = v2[:, 0], v2[:, 1]
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        nxt = targets[0]
        se = torch.tensor(
            [is_sentence_end(
                _decode_cache.setdefault(int(t), tok.decode([int(t)])),
                int(t), eos_id) for t in nxt],
            dtype=torch.bool)
        feats.append(hidden[0].float().cpu())
        p1s.append(p1.cpu())
        margins.append((p1 - p2).cpu())
        ents.append(ent.cpu())
        sent_ends.append(se)
        next_ids_all.append(nxt.cpu())
        del logits, probs, hidden

    feats = torch.cat(feats)
    p1s = torch.cat(p1s)
    margins = torch.cat(margins)
    ents = torch.cat(ents)
    sent_ends = torch.cat(sent_ends)
    think_thresh = float(torch.quantile(p1s, think_quantile).item())
    labels = label_positions(p1s, margins, sent_ends,
                             think_thresh=think_thresh,
                             verify_delta=verify_delta)
    return {
        "hidden": feats, "labels": labels, "top1": p1s, "margin": margins,
        "entropy": ents, "sent_end": sent_ends,
        "next_ids": torch.cat(next_ids_all),
        "think_thresh": think_thresh, "verify_delta": verify_delta,
        "n": int(feats.shape[0]),
    }


def relabel(trace: dict, think_thresh: float, verify_delta: float = 0.05):
    """Re-label a trace with a fixed threshold (held-out protocol)."""
    return label_positions(trace["top1"], trace["margin"], trace["sent_end"],
                           think_thresh=think_thresh,
                           verify_delta=verify_delta)


def train_head(head: DecisionHead, feats: torch.Tensor,
               labels: torch.Tensor, steps: int = 400,
               batch: int = 512, lr: float = 2e-3, wd: float = 0.01,
               device: str = "cuda", seed: int = 0) -> dict:
    """AdamW on class-weighted CE, random minibatches."""
    head.to(device).train()
    X, Y = feats.to(device), labels.to(device)
    counts = torch.bincount(Y, minlength=N_ACTIONS).float()
    weights = (counts.sum() / counts.clamp_min(1.0))
    weights = weights / weights.mean()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = X.shape[0]
    last_loss = 0.0
    for _ in range(steps):
        sel = torch.randint(0, n, (batch,), generator=g).to(device)
        loss = F.cross_entropy(head(X[sel]), Y[sel], weight=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    head.eval()
    return {"final_loss": round(last_loss, 4),
            "class_counts": counts.long().tolist(),
            "class_weights": [round(float(w), 3) for w in weights]}


@torch.inference_mode()
def eval_head(head: DecisionHead, feats: torch.Tensor,
              labels: torch.Tensor, device: str = "cuda") -> dict:
    """Overall + per-class accuracy and 4x4 confusion (rows=true)."""
    head.eval()
    X, Y = feats.to(device), labels.to(device)
    pred = head.act(X)
    correct = pred == Y
    conf = torch.zeros(N_ACTIONS, N_ACTIONS, dtype=torch.long)
    for t in range(N_ACTIONS):
        for p in range(N_ACTIONS):
            conf[t, p] = int(((t == Y) & (pred == p)).sum())
    per_class = {}
    for a, name in enumerate(ACTIONS):
        m = a == Y
        per_class[name] = round(float(correct[m].float().mean()), 4) \
            if int(m.sum()) else None
    return {"acc": round(float(correct.float().mean()), 4),
            "per_class": per_class,
            "confusion": conf.tolist(),
            "pred_dist": torch.bincount(pred, minlength=N_ACTIONS).tolist(),
            "true_dist": torch.bincount(Y, minlength=N_ACTIONS).tolist()}


@dataclass
class ControllerResult:
    tokens: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    think_logit_shift: list[float] = field(default_factory=list)
    stop_suppressed: int = 0
    stopped_early: bool = False
    elapsed_s: float = 0.0

    @property
    def action_hist(self) -> dict:
        h = torch.bincount(torch.tensor(self.actions or [0]),
                           minlength=N_ACTIONS).tolist()
        return {ACTIONS[i]: h[i] for i in range(N_ACTIONS)}

    @property
    def tok_s(self) -> float:
        return len(self.tokens) / max(self.elapsed_s, 1e-9)


class DecisionController:
    """Manual decode loop driven by a DecisionHead.

    Works against a ForgeEngine-like object: needs `.model`, `.tokenizer`,
    `._tokenize(prompt)`. `generate` also accepts a raw LongTensor of ids
    (skips the tokenizer) so CPU tests can run without one.
    """

    def __init__(self, eng, head: DecisionHead,
                 verify_temp: float = 0.7,
                 min_new_tokens: int = 8,
                 seed: int = 0):
        self.eng = eng
        self.model = eng.model
        self.head = head
        self.verify_temp = verify_temp
        self.min_new_tokens = min_new_tokens
        dev = next(self.model.parameters()).device
        self._gen = torch.Generator(device=dev).manual_seed(seed)

    def _logits_hidden_past(self, ids, past):
        out = self.model(ids, past_key_values=past, use_cache=True,
                         return_hidden=True, logits_last_only=True)
        logits, _, past, hidden = out
        return logits[:, -1], past, hidden[:, -1]      # [1,V], past, [1,D]

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens: int = 64) -> ControllerResult:
        res = ControllerResult()
        ids = prompt if torch.is_tensor(prompt) else self.eng._tokenize(prompt)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        dev = next(self.model.parameters()).device
        ids = ids.to(dev).long()

        t0 = time.perf_counter()
        out = self.model(ids, use_cache=True, return_hidden=True,
                         logits_last_only=True)
        logits, _, past, hidden = out
        logits = logits[:, -1]
        h = hidden[:, -1]

        last_block = self.model.blocks[-1]
        ln_f = self.model.ln_f
        lm_head = self.model.head
        L = len(self.model.blocks)

        for _ in range(max_new_tokens):
            ent = float(-(F.softmax(logits.float(), -1)
                          * F.log_softmax(logits.float(), -1)).sum())
            action = int(self.head.act(h).item())
            res.actions.append(action)
            res.entropies.append(ent)

            if action == STOP:
                if len(res.tokens) >= self.min_new_tokens:
                    res.stopped_early = True
                    break
                res.stop_suppressed += 1
                action = CONTINUE

            if action == THINK:
                # Re-run last block once on current hidden. past[-1] is
                # only READ (use_cache=False -> present=None); torch.cat
                # inside attention never mutates the cache. position_ids
                # pins RoPE to the hidden token's true position.
                past_len = past[-1][0].shape[-2] if past[-1] is not None else 0
                if past_len > 0:
                    pos = torch.full((1, 1), past_len - 1,
                                     dtype=torch.long, device=dev)
                    x2, _ = last_block(
                        h.unsqueeze(1), past_key_value=past[-1],
                        use_cache=False, layer_idx=L - 1,
                        position_ids=pos)
                    h2 = ln_f(x2) if ln_f is not None else x2
                    logits2 = lm_head(h2)[:, -1]
                    res.think_logit_shift.append(
                        float((logits2 - logits).abs().max()))
                    logits = logits2
                nxt = logits.argmax(-1)
            elif action == VERIFY:
                v, i = logits.float().topk(2, dim=-1)
                p = F.softmax(v / self.verify_temp, dim=-1)
                pick = torch.multinomial(p, 1, generator=self._gen)
                nxt = i.gather(-1, pick).squeeze(-1)
            else:
                nxt = logits.argmax(-1)

            tok = nxt.view(1, 1)
            res.tokens.append(int(nxt.item()))
            logits, past, h = self._logits_hidden_past(tok, past)

        res.elapsed_s = time.perf_counter() - t0
        return res


@torch.inference_mode()
def greedy_generate(eng, prompt, max_new_tokens: int = 64,
                    eos_id: int | None = None) -> ControllerResult:
    """Plain greedy decode baseline (same manual loop, no head)."""
    res = ControllerResult()
    model = eng.model
    ids = prompt if torch.is_tensor(prompt) else eng._tokenize(prompt)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    dev = next(model.parameters()).device
    ids = ids.to(dev).long()
    t0 = time.perf_counter()
    from forge.model_loader import unpack_output_with_kv
    out = model(ids, use_cache=True, logits_last_only=True)
    logits, past = unpack_output_with_kv(out)
    logits = logits[:, -1]
    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1)
        res.tokens.append(int(nxt.item()))
        if eos_id is not None and int(nxt.item()) == eos_id:
            res.stopped_early = True
            break
        out = model(nxt.view(1, 1), past_key_values=past, use_cache=True)
        logits, past = unpack_output_with_kv(out)
        logits = logits[:, -1]
    res.elapsed_s = time.perf_counter() - t0
    return res


@torch.inference_mode()
def gen_nll(eng, prompt_ids: torch.Tensor, gen_ids: list[int],
            device: str = "cuda") -> float:
    """Teacher-forced mean NLL of `gen_ids` continuing `prompt_ids`."""
    if not gen_ids:
        return float("nan")
    model = eng.model
    gen = torch.tensor(gen_ids, dtype=torch.long, device=device).view(1, -1)
    ids = torch.cat([prompt_ids.to(device).long(), gen], dim=1)
    plen = prompt_ids.shape[1]
    logits, _ = model(ids[:, :-1])
    lp = F.log_softmax(logits[0, plen - 1:].float(), dim=-1)
    return float(-lp.gather(-1, gen[0].unsqueeze(-1)).mean())
