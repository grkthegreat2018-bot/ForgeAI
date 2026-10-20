"""E-diffusion: hybrid next-token + diffusion decoding.

A small continuous-diffusion denoiser head proposes the next K tokens in
PARALLEL (joint sample), conditioned on the last hidden state h_T.  The
AR model then spec-verifies the draft in a single forward — accepted
prefix is committed, so output is greedy-identical to plain AR decode.

Denoiser: per-position k in 1..K, input [h_T ; pos_k ; z_k] -> e_hat_k
(embedding-space DDIM).  Train loss = CE(head(e_hat), tok_{T+k}) +
mse(e_hat, embed(tok_{T+k})).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    return torch.cos((t + s) / (1 + s) * math.pi / 2).clamp(0, 1) ** 2


class DiffusionDraftHead(nn.Module):
    """Position-conditional embedding denoiser for K parallel tokens."""

    def __init__(self, d_model: int, k: int = 4, hidden: int = 512,
                 steps: int = 4):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.steps = steps
        self.pos = nn.Parameter(torch.randn(k, d_model) * 0.02)
        self.t_embed = nn.Linear(1, d_model)
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_model))

    def denoise(self, h: torch.Tensor, z: torch.Tensor,
                t) -> torch.Tensor:
        """h: (B, D) context; z: (B, K, D) noisy embeds;
        t: float or (B,) tensor in [0,1].  Computes in the module's
        param dtype (fp32) regardless of input dtype."""
        B, K, D = z.shape
        pdtype = self.t_embed.weight.dtype
        z = z.to(pdtype)
        h = h.to(pdtype)
        if not torch.is_tensor(t):
            t = torch.full((B,), float(t), device=z.device,
                           dtype=pdtype)
        tt = t.to(z.device).view(B, 1).to(pdtype)
        te = self.t_embed(tt).unsqueeze(1)                 # (B,1,D)
        inp = torch.cat([
            h.unsqueeze(1).expand(-1, K, -1),
            self.pos.unsqueeze(0).expand(B, -1, -1) + te,
            z], dim=-1)                                    # (B,K,3D)
        return self.net(inp)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, embed_weight: torch.Tensor,
               temperature: float = 1.0, **_kw) -> torch.Tensor:
        """DDIM sample -> (B, K) token ids (nearest embed)."""
        B = h.shape[0]
        z = torch.randn(B, self.k, self.d_model,
                        device=h.device, dtype=h.dtype)
        ts = torch.linspace(1.0, 0.0, self.steps + 1)
        for i in range(self.steps):
            t_cur, t_nxt = float(ts[i]), float(ts[i + 1])
            e_hat = self.denoise(h, z, t_cur)
            a_cur, a_nxt = _cosine_alpha_bar(
                torch.tensor(t_cur)), _cosine_alpha_bar(torch.tensor(t_nxt))
            eps_hat = (z - a_cur.sqrt() * e_hat) / (1 - a_cur).sqrt().clamp_min(1e-4)
            z = a_nxt.sqrt() * e_hat + (1 - a_nxt).sqrt() * eps_hat
            if temperature > 0 and t_nxt > 0:
                z = z + temperature * (1 - a_nxt).sqrt() * 0.1 * torch.randn_like(z)
        logits = z.to(embed_weight.dtype) @ embed_weight.T
        return logits.argmax(-1)


class MedusaDraftHead(nn.Module):
    """K parallel classification heads — same (h -> K tokens) interface
    as DiffusionDraftHead but direct argmax proposals (no sampling).
    Head k predicts token t+k+1 via a shared delta MLP + frozen lm_head."""

    def __init__(self, d_model: int, k: int = 4, hidden: int = 512):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.delta = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU(),
                          nn.Linear(hidden, d_model))
            for _ in range(k))

    def draft_logits(self, h: torch.Tensor, head_w: torch.Tensor,
                     ln_f=None) -> torch.Tensor:
        """h: (B, D) -> (B, K, V)."""
        outs = []
        for mlp in self.delta:
            x = h + mlp(h.float()).to(h.dtype)
            if ln_f is not None:
                x = ln_f(x.unsqueeze(1)).squeeze(1)
            outs.append(x.to(head_w.dtype) @ head_w.T)
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, embed_weight: torch.Tensor,
               head_w=None, ln_f=None, **_kw) -> torch.Tensor:
        """(B, D) -> (B, K) argmax draft (sample-interface compat)."""
        w = head_w if head_w is not None else embed_weight
        return self.draft_logits(h, w, ln_f).argmax(-1)


def train_medusa_head(model, head: MedusaDraftHead, ids_list,
                      steps: int = 300, lr: float = 1e-3):
    """CE on each slot's target token; frozen model, fp32 head."""
    head_w = model.head.weight
    ln_f = getattr(model, "ln_f", None)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    losses, accs = [], []
    step = 0
    while step < steps:
        for ids in ids_list:
            if step >= steps:
                break
            ids = ids.to(next(model.parameters()).device)
            k = head.k
            with torch.no_grad():
                hidden = model(ids, return_hidden=True)[-1]
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(1)
                h = hidden[:, :ids.shape[1] - k].reshape(-1, hidden.shape[-1])
                tgt = torch.stack(
                    [ids[:, j:j + ids.shape[1] - k]
                     for j in range(1, k + 1)], dim=2).reshape(-1, k)
            logits = head.draft_logits(h.float(), head_w, ln_f)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            accs.append(float((logits.argmax(-1) == tgt).float().mean()))
            step += 1
    for p in model.parameters():
        p.requires_grad_(True)
    return losses, accs


def collect_hidden_targets(model, embed_w, ids, k: int):
    """h_t for every t plus next-k token embeds/ids. Returns
    (h, tgt_emb, tgt_ids) with h (B,T-k,D)."""
    out = model(ids, return_hidden=True)
    hidden = out[-1] if isinstance(out, tuple) else out
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(1)
    B, T = ids.shape
    h = hidden[:, :T - k].contiguous()                    # h_t predicts t+1..t+k
    tgt_ids = torch.stack([ids[:, j:j + T - k]
                           for j in range(1, k + 1)], dim=2)   # (B,T-k,K)
    tgt_emb = F.embedding(tgt_ids, embed_w)
    return h.detach(), tgt_emb.detach(), tgt_ids


def train_draft_head(model, head: DiffusionDraftHead, ids_list,
                     steps: int = 300, lr: float = 1e-3, mse_w: float = 0.1):
    embed_w = model.embed.weight
    head_w = model.head.weight
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    losses, accs = [], []
    step = 0
    while step < steps:
        for ids in ids_list:
            if step >= steps:
                break
            ids = ids.to(next(model.parameters()).device)
            with torch.no_grad():
                h, tgt_emb, tgt_ids = collect_hidden_targets(
                    model, embed_w, ids, head.k)
            B, T, K, D = tgt_emb.shape
            t = torch.rand(B, device=ids.device)
            a = _cosine_alpha_bar(t).sqrt().view(B, 1, 1, 1)
            eps = torch.randn_like(tgt_emb)
            z = a * tgt_emb + (1 - a * a).sqrt() * eps
            t_rep = t.unsqueeze(1).expand(B, T).reshape(-1)   # (B*T,)
            e_hat = head.denoise(h.reshape(-1, D),
                                 z.reshape(-1, K, D), t_rep
                                 ).reshape(B, T, K, D)
            logits = e_hat @ head_w.T.float()             # (B,T,K,V)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                tgt_ids.reshape(-1))
            mse = F.mse_loss(e_hat, tgt_emb.float())
            loss = ce + mse_w * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            accs.append(float((logits.argmax(-1) == tgt_ids).float().mean()))
            step += 1
    for p in model.parameters():
        p.requires_grad_(True)
    return losses, accs


def _truncate_pasts(pasts, keep_total: int):
    """Keep only the first `keep_total` sequence positions of (k,v)
    tuple pasts (drops the unaccepted draft tail)."""
    out = []
    for p in pasts:
        if p is None:
            out.append(None)
        elif isinstance(p, tuple) and len(p) == 2:
            out.append((p[0][:, :, :keep_total],
                        p[1][:, :, :keep_total]))
        else:
            out.append(p)  # non-tuple cache (strategy objects): leave
    return out


@torch.no_grad()
def spec_verify(model, pasts, last_logits, draft_ids: torch.Tensor):
    """Greedy spec-decode verify of a K-token parallel draft.

    pasts: context KV (list[(k,v)|None]); last_logits: logits for the
    last committed position (predicts draft_ids[0]); draft_ids: (1,K).

    Returns dict(committed, kv, logits_next, hidden_next, n_accepted,
    all_accepted).  On rejection, logits_next/hidden_next are None and
    the caller must re-forward the last committed token.
    """
    ctx_len = pasts[0][0].shape[-2] if pasts and pasts[0] is not None else 0
    out = model(draft_ids, past_key_values=pasts, use_cache=True,
                return_hidden=True)
    logits, kv_full, hidden = out[0], out[2], out[3]
    K = draft_ids.shape[1]
    committed = []
    expected = int(last_logits[0, -1].argmax())
    accepted = 0
    for j in range(K):
        d = int(draft_ids[0, j])
        if d == expected:
            committed.append(d)
            accepted += 1
            expected = int(logits[0, j].argmax())
        else:
            committed.append(expected)     # corrected token (model's pick)
            break
    all_acc = accepted == K
    kv = _truncate_pasts(kv_full, ctx_len + accepted)
    return {
        "committed": committed,
        "kv": kv,
        "logits_next": logits[:, -1:] if all_acc else None,
        "hidden_next": hidden[:, -1:] if all_acc else None,
        "n_accepted": accepted,
        "all_accepted": all_acc,
    }


@torch.no_grad()
def fixup_step(model, pasts, last_token_id: int):
    """Re-forward the corrected token to obtain fresh next-step state."""
    tok = torch.tensor([[last_token_id]], device=next(
        model.parameters()).device)
    out = model(tok, past_key_values=pasts, use_cache=True,
                return_hidden=True)
    return out[0], out[2], out[3]   # logits, kv, hidden
