"""V14-RSI work-acceleration heads (docs/V14_RSI_DESIGN.md §3).

Two read-only taps on the model's final hidden states. They never perturb
the base forward: an untrained model simply never emits the <copy> op and
the verifier output is ignored, so both heads are lossless at init.

CopySpanPointerHead — copy-span ("edit mode") decoding. RSI proposers
re-emit ~80-95% of a source file unchanged. When the LM head emits the
<copy> op token, this head produces (start, end) pointer distributions
over the SOURCE span's cached final-layer hiddens instead of vocab
logits. The driver splices source[start:end] verbatim and commits it with
ONE chunked T>1 forward — K copied tokens cost ~2 pointer steps + one
amortized weight-read instead of K sequential decode steps.

VerifierHead — scalar P(candidate passes target tests) from ln_f(hidden).
Bradley-Terry trained on evolve pairs (chosen/rejected). Used for
best-of-n rerank before subprocess eval spend and early-stop gating.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CopySpanPointerHead(nn.Module):
    """Content-based pointer over cached source hiddens.

    start_logits, end_logits = head(h_t, src_k) where src_k =
    head.key_cache(H_src) is precomputed once per source span (H_src are
    the source positions' post-ln_f hidden states, captured at prefill —
    no recompute during decode).
    """

    def __init__(self, d_model: int, ptr_dim: int | None = None):
        super().__init__()
        self.ptr_dim = ptr_dim or d_model
        self.q_start = nn.Linear(d_model, self.ptr_dim, bias=False)
        self.q_end = nn.Linear(d_model, self.ptr_dim, bias=False)
        self.k_src = nn.Linear(d_model, self.ptr_dim, bias=False)
        self.scale = self.ptr_dim ** -0.5

    def key_cache(self, src_hiddens: torch.Tensor) -> torch.Tensor:
        """Precompute pointer keys for a source span.

        src_hiddens: (S, d_model) or (B, S, d_model) — final-layer hiddens
        of the source positions. Returns src_k with matching batch rank.
        """
        return self.k_src(src_hiddens)

    def forward(
        self,
        h_t: torch.Tensor,
        src_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pointer logits for span start and end.

        h_t: (B, d_model) or (B, 1, d_model) — last-token hidden.
        src_k: (S, ptr_dim) or (B, S, ptr_dim) — from key_cache().
        Returns (start_logits, end_logits), each (B, S).
        """
        if h_t.dim() == 3:
            h_t = h_t[:, -1, :]
        if src_k.dim() == 2:
            src_k = src_k.unsqueeze(0)
        start = torch.einsum("bd,bsd->bs", self.q_start(h_t), src_k) * self.scale
        end = torch.einsum("bd,bsd->bs", self.q_end(h_t), src_k) * self.scale
        return start, end


class VerifierHead(nn.Module):
    """Scalar P(pass) head on ln_f(hidden).

    Returns a logit (unnormalized); use torch.sigmoid for probability.
    Trained with verifier_bt_loss on (chosen, rejected) candidate pairs.
    """

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_model) -> (...,) score logits."""
        return self.net(h).squeeze(-1)


def verifier_bt_loss(
    score_chosen: torch.Tensor,
    score_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise loss: chosen should outscore rejected."""
    return -F.logsigmoid(score_chosen - score_rejected).mean()


def copy_span_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    gold_start: torch.Tensor,
    gold_end: torch.Tensor,
) -> torch.Tensor:
    """CE over gold (start, end) pointer targets.

    start_logits/end_logits: (B, S); gold_start/gold_end: (B,) indices.
    """
    return (
        F.cross_entropy(start_logits, gold_start)
        + F.cross_entropy(end_logits, gold_end)
    ) * 0.5
