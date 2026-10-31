"""CPU tests for forge/exp/prevtok.py — synthetic fake model, no downloads."""
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from forge.exp.prevtok import (
    PrevTokenHead,
    SelfCorrectingDecoder,
    eval_prev_head,
    kv_expand,
    kv_select,
    kv_truncate,
    load_head,
    prev_ce_loss,
    save_head,
    seq_prev_nll,
    train_prev_head,
)

V = D = 48                       # orthogonal embeddings need d == vocab
A, B, C, E = 0, 1, 2, 3          # token ids used in transition tests


class FakeLM(nn.Module):
    """Deterministic bigram LM with a KV-cache-shaped state.

    past = [(k, v)] with k = one-hot history [B, 1, T, V] (dim>=3, T at
    -2 — same slicing contract as the real [B, H, T, D] cache).
    logits(x_{t+1}) = logit_table[x_t].
    h_t = emb[x_t] + mix * gate[x_{t-1}, x_t] * emb[x_{t-1}], where
    gate marks "consistent" transitions — the prev token is only
    recoverable from h_t for good pairs, mimicking how a real hidden
    state encodes likely predecessors.
    """

    def __init__(self, logit_table, pair_gate=None, prev_mix=1.0):
        super().__init__()
        v = logit_table.shape[0]
        # Orthonormal-ish embeddings: sqrt(D) * I — dot(u,u)=D, dot(u,v)=0.
        self.embed = nn.Embedding(v, D)
        with torch.no_grad():
            self.embed.weight.copy_((D ** 0.5) * torch.eye(D)[:v])
        self.logit_table = nn.Parameter(logit_table.float(),
                                        requires_grad=False)
        if pair_gate is None:
            pair_gate = torch.ones(v, v)
        self.register_buffer("pair_gate", pair_gate.float())
        self.prev_mix = prev_mix

    def forward(self, idx, targets=None, past_key_values=None,
                use_cache=False, return_hidden=False,
                logits_last_only=False, **_kw):
        _B, T = idx.shape
        Vv = self.logit_table.shape[0]
        oh = F.one_hot(idx, Vv).float().unsqueeze(1)          # [B,1,T,V]
        hist = (past_key_values[0][0]
                if past_key_values is not None else None)
        k = oh if hist is None else torch.cat([hist, oh], dim=2)
        v = torch.zeros_like(k)
        ids_full = k.argmax(-1).squeeze(1)                    # [B,Tt]
        emb = self.embed(ids_full)                            # [B,Tt,D]
        prev_ids = torch.zeros_like(ids_full)
        prev_ids[:, 1:] = ids_full[:, :-1]
        prev_emb = self.embed(prev_ids)
        g = torch.zeros_like(ids_full, dtype=emb.dtype)
        g[:, 1:] = self.pair_gate[ids_full[:, :-1], ids_full[:, 1:]]
        hidden = (emb + self.prev_mix * g.unsqueeze(-1)
                  * prev_emb)[:, -T:]
        logits = self.logit_table[idx]                        # [B,T,V]
        if logits_last_only:
            logits = logits[:, -1:]
        loss = (F.cross_entropy(logits[:, :-1].reshape(-1, Vv),
                                idx[:, 1:].reshape(-1))
                if targets is not None else None)
        if use_cache and return_hidden:
            return logits, loss, [(k, v)], hidden
        if use_cache:
            return logits, loss, [(k, v)]
        if return_hidden:
            return logits, loss, hidden
        return logits, loss


def _table(argmax_map):
    """V x V logit table: row i peaks at argmax_map[i] (else ~uniform)."""
    t = torch.full((V, V), -2.0)
    for i, j in argmax_map.items():
        t[i, j] = 3.0
    return t


def _head_for(fake, scale=1.0):
    """Prev head = scaled orthogonal embedding readout (untied proj).
    With the fake's mix=4 prev component, a good pair gives
    p(prev)~1, bad pair ~0."""
    h = PrevTokenHead(D, V)          # out_weight=None -> untied .proj
    with torch.no_grad():
        h.proj.weight.copy_(fake.embed.weight * scale)
    return h


def _gate(good_pairs):
    g = torch.zeros(V, V)
    for a_, b_ in good_pairs:
        g[a_, b_] = 1.0
    return g


# ─────────────────────────── head / loss ───────────────────────────


def test_head_shapes():
    head = PrevTokenHead(D, V)
    logits = head(torch.randn(2, 5, D))
    assert logits.shape == (2, 5, V)
    p, lp = head.prev_prob_of(torch.randn(D), 7)
    assert 0.0 <= p <= 1.0
    assert abs(torch.exp(torch.tensor(lp)).item() - p) < 1e-5


def test_prev_ce_loss_grads():
    head = PrevTokenHead(D, V)
    hidden = torch.randn(1, 8, D)
    ids = torch.randint(0, V, (1, 8))
    loss, acc = prev_ce_loss(head, hidden, ids)
    assert loss.ndim == 0 and 0.0 <= acc <= 1.0
    loss.backward()
    assert head.proj.weight.grad is not None
    assert head.proj.weight.grad.abs().sum() > 0


def test_loss_goes_down_synthetic():
    """Cyclic sequences: prev token is learnable -> loss must drop."""
    torch.manual_seed(0)
    fake = FakeLM(_table({}), prev_mix=1.0)      # gate=None -> all good
    docs = [(torch.arange(1, 33) % V).view(1, -1) for _ in range(4)]
    head = PrevTokenHead(D, V)
    first = prev_ce_loss(head, fake(docs[0], return_hidden=True)[-1],
                         docs[0])[0].item()
    r = train_prev_head(fake, head, docs, steps=80, lr=1e-2,
                        log_every=80, use_autocast=False)
    assert r["final_loss"] < first * 0.8
    ev = eval_prev_head(fake, head, docs, use_autocast=False)
    assert ev["prev_top1"] > 0.9       # cyclic -> prev fully learnable
    assert ev["prev_nll"] < 0.5
    pn = seq_prev_nll(fake, head, docs[0], use_autocast=False)
    assert 0.0 <= pn == pytest.approx(ev["prev_nll"], abs=0.6)


# ─────────────────────────── KV helpers ─────────────────────────────


def test_kv_helpers():
    k = torch.randn(1, 2, 7, 4)
    v = torch.randn(1, 2, 7, 4)
    past = [(k, v), None, {"state": 1}]
    t = kv_truncate(past, 2)
    assert t[0][0].shape == (1, 2, 5, 4)
    assert t[1] is None and t[2] == {"state": 1}
    e = kv_expand(past, 3)
    assert e[0][0].shape == (3, 2, 7, 4)
    s = kv_select(e, 1)
    assert s[0][0].shape == (1, 2, 7, 4)
    assert torch.equal(s[0][0][0], e[0][0][1])


# ─────────────────────────── decoding ───────────────────────────────


def test_baseline_greedy_fake():
    fake = FakeLM(_table({A: C, C: E, E: C}))
    dec = SelfCorrectingDecoder(fake, None)
    r = dec.generate_baseline(torch.tensor([[A]]), 5)
    assert r["ids"] == [C, E, C, E, C]
    assert r["forwards"] == 6 and r["n_tokens"] == 5


# Chain used below: prompt A.  Baseline greedy walks A -> C -> E -> C...
# "Good" pairs: (A,B), (B,E), (E,C), (C,E).  After a correction to B the
# walk B -> E -> C -> E is fully consistent, so exactly 1 correction fires.
GOOD = {(A, B), (B, E), (E, C), (C, E)}


def _chain_table():
    t = _table({A: C, B: E, E: C, C: E})
    t[A, B] = 2.0                       # B = runner-up after A
    return t


def test_threshold_correction_swaps_argmax():
    """argmax C after A is backward-inconsistent (gate[A,C]=0); B is
    second-best and consistent (gate[A,B]=1) -> swap C for B."""
    fake = FakeLM(_chain_table(), pair_gate=_gate(GOOD), prev_mix=4.0)
    dec = SelfCorrectingDecoder(fake, _head_for(fake), threshold=0.5,
                                n_cands=4)
    r = dec.generate(torch.tensor([[A]]), 4)
    assert r["ids"] == [B, E, C, E]     # consistent alternate picked
    assert r["corrections"] == 1        # only first step needed fixing
    assert r["resamples"] >= 1
    assert r["forwards"] > r["n_tokens"] + 1   # extra batched retry
    base = dec.generate_baseline(torch.tensor([[A]]), 4)
    assert base["ids"] == [C, E, C, E]  # baseline keeps argmax chain


def test_rerank_mode():
    fake = FakeLM(_chain_table(), pair_gate=_gate(GOOD), prev_mix=4.0)
    dec = SelfCorrectingDecoder(fake, _head_for(fake), n_cands=3)
    r = dec.generate(torch.tensor([[A]]), 4, mode="rerank")
    assert r["ids"][0] == B
    assert r["checks"] == 4
    assert r["corrections"] >= 1


def test_rollback_path():
    """Every transition inconsistent (gate=0) -> checks fail, retried
    alternates still fail -> single-token rollback fires."""
    fake = FakeLM(_chain_table(), pair_gate=torch.zeros(V, V),
                  prev_mix=4.0)
    dec = SelfCorrectingDecoder(fake, _head_for(fake), threshold=0.5,
                                n_cands=3, rollback=True,
                                max_rollbacks=4)
    r = dec.generate(torch.tensor([[A]]), 4)
    assert r["n_tokens"] == 4 and len(r["ids"]) == 4
    assert r["rollbacks"] >= 1


def test_check_every():
    fake = FakeLM(_table({A: B, B: B}))
    dec = SelfCorrectingDecoder(fake, _head_for(fake), threshold=0.0,
                                check_every=2)
    r = dec.generate(torch.tensor([[A]]), 8)
    assert r["checks"] == 4             # every 2nd step


def test_save_load_head(tmp_path):
    head = PrevTokenHead(D, V)
    p = str(tmp_path / "h.pt")
    save_head(head, p, extra={"steps": 1})
    h2 = load_head(p, device="cpu")
    assert torch.equal(h2.proj.weight, head.proj.weight)
