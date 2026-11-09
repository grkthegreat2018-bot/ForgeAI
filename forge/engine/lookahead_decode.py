"""Lookahead decode: n-gram draft + batched verification (lossless parallel
decode, no draft model). Enabled via FORGE_LOOKAHEAD=k[,w] (default 8,3).

Mechanism (same core as Lookahead Decoding, Fu et al. arXiv:2402.02057,
minus the Jacobi tree): at each step, find the most recent non-overlapping
occurrence of the last w tokens in the context and draft its continuation.
A single batched forward verifies all draft positions — memory-bound cost
≈ one token since weight reads dominate. Accepted prefix is kept; the
corrected token falls out of the verification logits for free.

Measured on spark_x26_1p7b (RTX 5070, bf16) vs StandardDecoding:
2.2-2.4x decode speedup on repetitive code output, ~0.85-1.0x on
open-ended text (drafts miss; ~3ms/step python overhead remains).
Emitted tokens match standard greedy decode exactly on all bench
prompts — rep-penalty window (generated-only, last 64), eos-append,
degenerate-guard (same-token x8 / <40% unique in last 20) and
max_new_tokens cap all mirror decoding.py semantics. Batched-verify
can flip a bf16 near-tie vs single-token forwards (same caveat as
any speculative scheme) — observed only at degenerate-output tails.

Greedy only (temperature==0): acceptance is argmax equality. Sampled
acceptance needs distribution-level accept/reject (future work).
"""
from __future__ import annotations

import os

import torch

_spec = os.environ.get("FORGE_LOOKAHEAD", "")


def lookahead_config() -> tuple[int, int] | None:
    """Parse FORGE_LOOKAHEAD=k[,w] -> (draft_len, ngram_w). None = disabled."""
    if not _spec:
        return None
    try:
        parts = [int(x) for x in _spec.split(",")]
        k = parts[0]
        w = parts[1] if len(parts) > 1 else 3
        return (k, w) if k >= 1 and w >= 1 else None
    except ValueError:
        return None


def _find_draft(context: list[int], w: int, k: int) -> list[int] | None:
    """Most recent non-overlapping (w)-gram match -> its continuation."""
    ctx = context[-w:]
    for j in range(len(context) - w - 1, -1, -1):
        if context[j:j + w] == ctx:
            draft = context[j + w:j + w + k]
            return draft or None
    return None


def _pick(logits_row: torch.Tensor, gen: list[int],
          rep: float) -> int:
    """Greedy pick under the engine's sampling chain for temperature==0:
    rep-penalty over the last-64 GENERATED window (prompt excluded —
    decoding.py:generated_ids, not prompt_ids) then argmax. Top-k/top-p/
    min-p/min-k/DRY cannot change argmax — only rep does."""
    nl = logits_row.clone()
    if rep != 1.0 and gen:
        toks = torch.tensor(sorted(set(gen[-64:])), device=nl.device,
                            dtype=torch.long)
        nl.index_copy_(-1, toks, nl.index_select(-1, toks) / rep)
    return nl.argmax(-1).item()


def _degenerate(out: list[int]) -> bool:
    """Mirror StandardDecoding's degeneration guard (decoding.py): same
    token x8, or <40% unique over the last 20 generated tokens."""
    n = len(out)
    if n >= 8 and len(set(out[-8:])) == 1:
        return True
    return n >= 20 and len(set(out[-20:])) < 8


def _trunc_kv(p, keep: int):
    if p is None:
        return None
    k_, v_ = p
    return k_[:, :, :keep], v_[:, :, :keep]


STATS = {"fw": 0, "try": 0, "acc": 0, "single": 0}


@torch.no_grad()
def lookahead_generate(engine, ids: torch.Tensor, max_new_tokens: int,
                       k: int = 8, w: int = 3,
                       repetition_penalty: float = 1.0) -> torch.Tensor:
    """Greedy decode with n-gram drafts + batched verification.

    Costs at most one forward per draft (the correction token is emitted
    by the next iteration's argmax — no refeed). Adaptive: poor drafts
    double a backoff cooldown and shrink the next draft.
    """
    from forge.engine.engine_generation import unpack_output_with_kv

    model = engine.model
    device = engine.device
    eos_set = engine._eos_token_ids()
    logits, past = unpack_output_with_kv(
        model(ids, use_cache=True, logits_last_only=True))
    seq_len = ids.shape[1]
    out: list[int] = []
    hist = ids[0].tolist()
    cooldown, backoff, dk = 0, 8, 2

    while len(out) < max_new_tokens:
        draft = (_find_draft(hist, w, dk)
                 if len(hist) > w * 2 and cooldown == 0 else None)
        if not draft:
            cooldown = max(0, cooldown - 1)
            tid = _pick(logits[0, -1], out, repetition_penalty)
            # std appends the token first, then checks eos/degen — mirror.
            out.append(tid)
            hist.append(tid)
            if tid in eos_set or _degenerate(out):
                break
            logits, past = unpack_output_with_kv(
                model(torch.tensor([[tid]], device=device),
                      past_key_values=past, use_cache=True))
            seq_len += 1
            STATS["single"] += 1
            STATS["fw"] += 1
            continue

        STATS["try"] += 1
        STATS["fw"] += 1
        inp = torch.tensor([draft], device=device)
        d_logits, d_past = unpack_output_with_kv(
            model(inp, past_key_values=past, use_cache=True))
        # chain[k] = penalized argmax at draft position k — the rep window
        # is generated-only: out + draft[:k], exactly what std would emit.
        chain = [_pick(logits[0, -1], out, repetition_penalty)]
        g = list(out)
        for t in range(len(draft) - 1):
            g.append(draft[t])
            chain.append(_pick(d_logits[0, t], g, repetition_penalty))
        acc = 0
        for want, got in zip(draft, chain):
            if got != want:
                break
            acc += 1

        STATS["acc"] += acc
        if acc >= 2:
            cooldown, backoff = 0, 8
            dk = min(acc + 1, k)
        else:
            cooldown, backoff = backoff, min(backoff * 2, 64)
            dk = 2

        n_before = len(out)
        eos_hit = False
        for t in draft[:acc]:
            out.append(t)
            hist.append(t)
            if (t in eos_set or _degenerate(out)
                    or len(out) >= max_new_tokens):
                eos_hit = True
                break
        emitted = len(out) - n_before
        if emitted == len(draft):
            logits = d_logits[:, -1:, :]
        elif emitted > 0:
            logits = d_logits[:, emitted - 1:emitted, :]
        past = [_trunc_kv(p, seq_len + emitted) for p in d_past]
        seq_len += emitted
        if eos_hit:
            break

    # Same post-state StandardDecoding exposes: _finish_to_stop consumes
    # last-KV + last-logits to continue without a re-prefill.
    model._forge_last_kv = past
    model._forge_last_logits = logits
    if not out:
        return ids
    gen = torch.tensor(out, device=device).reshape(1, -1)
    return torch.cat([ids, gen], dim=-1)
