"""Uno: diffusion-augmented lossless block decoding (arXiv:2609.04010).

Uno defines two weight sets: the frozen AR weights (quality) and lightweight
draft weights (speed). This module implements the inference side — the
Psi-Spec family of samplers that draw token blocks in parallel from the AR
distribution and verify them losslessly:

- Greedy mode: accept a drafted token iff it equals the AR argmax. Output is
  bit-exact vs standard greedy decoding regardless of draft quality.
- Sampling mode: standard speculative rejection sampling when the proposer
  supplies draft log-probs; match-based acceptance otherwise (near-lossless,
  prompt-lookup convention).

Block drafting uses a pluggable proposer:
- NgramProposer (default, training-free): suffix-match drafting from the
  prompt + generated context.
- Any callable ``propose(context_ids, k) -> (ids, logprobs | None)``.

Entropy-bounded early stop (DiffusionGemma-style): when the AR entropy over
the verified block drops below ``entropy_stop``, drafting is disabled for the
rest of the sequence (pure AR from there).

Lossless fallback: ``proposer=None`` (or an empty draft every block) reduces
exactly to standard autoregressive decoding.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import torch
import torch.nn.functional as F

from forge.engine.decoding import StandardDecoding
from forge.model_loader import unpack_output_with_kv as unpack_kv


class NgramProposer:
    """Training-free block proposer via n-gram suffix matching."""

    def __init__(self, n: int = 3):
        self.n = max(2, int(n))
        self.table: dict[tuple, Counter] = {}

    def update(self, ids) -> None:
        ids = list(ids)
        for i in range(len(ids) - self.n + 1):
            key = tuple(ids[i:i + self.n - 1])
            self.table.setdefault(key, Counter())[ids[i + self.n - 1]] += 1

    def propose(self, context_ids, k: int):
        ids = list(context_ids)
        draft: list[int] = []
        logprobs: list[float] = []
        for _ in range(k):
            key = tuple(ids[len(ids) - self.n + 1:])
            counter = self.table.get(key)
            if not counter:
                break
            token, count = counter.most_common(1)[0]
            total = sum(counter.values())
            draft.append(int(token))
            logprobs.append(float(torch.log(torch.tensor(count / total))))
            ids.append(int(token))
        if not draft:
            return None, None
        return (torch.tensor(draft, dtype=torch.long),
                torch.tensor(logprobs, dtype=torch.float32))


class UnoDecoding(StandardDecoding):
    """Psi-Spec block decoding: propose blocks, verify against the AR model."""

    def __init__(self, proposer: Callable | None = None, block_size: int = 4,
                 entropy_stop: float | None = None):
        self.proposer = proposer
        self.block_size = max(1, int(block_size))
        self.entropy_stop = entropy_stop
        self._drafting_enabled = True

    @property
    def name(self) -> str:
        return "UnoDecoding"

    def _propose(self, ids: list[int], k: int):
        if not self._drafting_enabled or self.proposer is None:
            return None, None
        if isinstance(self.proposer, NgramProposer):
            return self.proposer.propose(ids, k)
        out = self.proposer(ids, k)
        if out is None:
            return None, None
        draft, logprobs = out
        if draft is None or len(draft) == 0:
            return None, None
        if not torch.is_tensor(draft):
            draft = torch.tensor(draft, dtype=torch.long)
        return draft, logprobs

    @staticmethod
    def _truncate(past, length: int):
        if past is None:
            return None
        return (past[0][:, :, :length], past[1][:, :, :length])

    @torch.inference_mode()
    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0, top_k=80,
                 repetition_penalty=1.05, **kwargs):
        if self.proposer is None:
            return super().generate(model, input_ids, max_new_tokens,
                                    temperature, top_p, top_k,
                                    repetition_penalty, **kwargs)
        greedy = temperature <= 0
        device = input_ids.device
        context: list[int] = input_ids[0].tolist()
        n_ctx = len(context)

        out = model(input_ids, use_cache=True)
        logits, past_kv = unpack_kv(out)
        past_len = n_ctx

        if isinstance(self.proposer, NgramProposer):
            self.proposer.update(context)

        # Merge with the shared default EOS set — hardcoding {7, ...}
        # missed Jamba/Qwen EOS ids (EC-039).
        from forge.engine.engine_common import _DEFAULT_EOS_TOKEN_IDS
        eos_set = set(_DEFAULT_EOS_TOKEN_IDS)
        eos = getattr(model, "eos_token_id", None)
        if eos is None:
            cfg = getattr(model, "config", None)
            eos = getattr(cfg, "eos_token_id", None) if cfg else None
        if isinstance(eos, (list, tuple, set, frozenset)):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)

        generated = 0
        while generated < max_new_tokens:
            k = min(self.block_size, max_new_tokens - generated)
            draft, draft_logprobs = self._propose(context, k)

            if draft is None:
                next_logits = logits[:, -1, :]
                if not greedy and context:
                    for tid in set(context[-64:]):
                        next_logits[:, tid] /= repetition_penalty
                token = int(next_logits.argmax()) if greedy else sample_token(
                    next_logits, top_p, top_k)
                context.append(token)
                generated += 1
                if token in eos_set:
                    break
                out = model(torch.tensor([[token]], device=device),
                            use_cache=True, past_key_value=past_kv)
                logits, past_kv = unpack_kv(out)
                past_len += 1
                continue

            draft = draft.to(device)
            n_draft = draft.shape[0]
            block_out = model(draft.unsqueeze(0), use_cache=True,
                              past_key_value=past_kv)
            block_logits, block_past = unpack_kv(block_out)

            ar_logits = torch.cat([logits[:, -1:, :], block_logits[:, :-1, :]],
                                  dim=1).float()
            ar_log_probs = F.log_softmax(ar_logits, dim=-1)
            ar_entropy = -(ar_log_probs.exp() * ar_log_probs).sum(dim=-1)[0]

            accepted = 0
            correction: int | None = None
            for j in range(n_draft):
                row = ar_log_probs[0, j]
                if greedy:
                    match = int(row.argmax()) == int(draft[j])
                elif draft_logprobs is not None:
                    ar_lp = row[draft[j]]
                    d_lp = float(draft_logprobs[j])
                    u = torch.rand(1).item()
                    match = u < torch.exp((ar_lp - d_lp).clamp(max=0.0)).item()
                else:
                    match = int(row.argmax()) == int(draft[j])
                if match:
                    accepted += 1
                else:
                    correction = int(row.argmax())
                    break

            for j in range(accepted):
                context.append(int(draft[j]))
            generated += accepted

            if (self.entropy_stop is not None and accepted >= 2
                    and ar_entropy[:accepted].mean().item() < self.entropy_stop):
                self._drafting_enabled = False

            if correction is not None:
                context.append(correction)
                generated += 1
                past_kv = self._truncate(block_past, past_len + accepted)
                out = model(torch.tensor([[correction]], device=device),
                            use_cache=True, past_key_value=past_kv)
                logits, past_kv = unpack_kv(out)
                past_len += accepted + 1
            else:
                past_len += n_draft
                logits = block_logits[:, -1:, :]

            if context[-1] in eos_set:
                break
            if isinstance(self.proposer, NgramProposer):
                self.proposer.update(context[-self.proposer.n:])

        return torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)


def sample_token(logits, top_p: float = 1.0, top_k: int = 80) -> int:
    l = logits[0].float().clone()
    if top_k > 0 and top_k < l.shape[-1]:
        thresh = torch.topk(l, top_k)[0][-1]
        l[l < thresh] = float("-inf")
    if top_p < 1.0:
        sorted_l, sorted_idx = torch.sort(l, descending=True)
        cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        mask = cum - F.softmax(sorted_l, dim=-1) > top_p
        sorted_l[mask] = float("-inf")
        l = torch.zeros_like(l).scatter(0, sorted_idx, sorted_l)
    return int(F.softmax(l, dim=-1).multinomial(1))
