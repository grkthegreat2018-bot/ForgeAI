"""System One-style typed decisions (TypeSafe AI API-compatible).

Implements the TypeSafe "System One" request/response contract
(``POST /v1/systemone``: ``state`` + typed ``questions`` → probabilistic
answers) on top of a causal LM, with **no generation and no parsing**.

Mechanism — candidate-continuation scoring:
  For each question we build a chat-formatted prompt ending at the
  assistant turn, then score every candidate answer string as a
  continuation:

  * fast path   — all candidates are single tokens: ONE forward pass,
                  read the softmax over the candidate token ids at the
                  last prompt position.
  * continuation — otherwise: one right-padded batched forward over
                  ``prompt + candidate`` rows; score = sum of candidate
                  token logprobs (teacher-forced).  Shared candidate
                  prefixes cancel exactly in the softmax.

  Scores → softmax → per-question probability distribution.  These are
  *raw LM probabilities*: consistent across candidates within a question
  but NOT calibrated.  When a trained DecisionScorer is attached
  (``scorer=`` or ForgeEngine.load_decision_scorer), candidate
  probabilities come from the verifier head instead — Tier-1 calibrated
  (see forge/engine/decision_head.py).

Wire format (matches typesafe_sdk ``_schemas/models.py``):
  answers are discriminated by a ``type`` tag:
    noul   → {"type": "noul",  "noul": p_yes}
    choice → {"type": "choice", "choice": label, "confidence": c,
              "probabilities": {label: p}}
    score  → {"type": "score", "score": expected_index, "confidence": c,
              "legend": {str(i): desc}, "probabilities": {str(i): p}}

Usage:
    evaluator = SystemOneEvaluator(model, tokenizer, device)
    result = evaluator.evaluate(state, questions, model_id="forgelm-v2-jamba")

    # or via the engine (holds _gen_lock):
    result = engine.decide(state, questions)
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "QuestionValidationError",
    "SystemOneEvaluator",
]

MAX_CHOICE_OPTIONS = 255
MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
DEFAULT_CONTEXT_LIMIT = 8192
DEFAULT_MAX_BATCH = 32

_NOUL_LABELS = ("yes", "no")

# ForgeLM V2 is a reasoning model: the canonical chat template primes the
# assistant turn with an open <think> block (see qwen_adapter).  For
# single-token decisions we must close that block ourselves and give the
# model an explicit answer field — otherwise the first generated token is
# reasoning text and answer-token probabilities are buried ~15 logits deep.
_THINK_HINT = (
    "Begin by thinking about the reasoning process in the mind "
    "within <think> </think> tags and then proceed to give your "
    "response.\n"
)
_ASSISTANT_ANSWER_SLOT = "<|im_start|>assistant\n<think>\n</think>\nAnswer:"


class QuestionValidationError(ValueError):
    """Raised when a System One question dict fails schema validation."""


@dataclass
class _Question:
    """Parsed + validated question spec."""
    key: str
    qtype: str                      # "noul" | "choice" | "score"
    instructions: str               # flattened instructions text
    labels: list[str]               # wire labels aligned with candidates
    candidates: list[str]           # answer strings scored as continuations
    legend: dict[str, str] = field(default_factory=dict)   # score: str(i) → desc
    option_descs: dict[str, str] = field(default_factory=dict)  # choice: label → desc
    noul_descs: dict[str, str] = field(default_factory=dict)    # noul: true/false → desc


def _jsonable(content) -> str:
    """Flatten JSONContent (str | dict | list | None) to prompt text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def _parse_question(key: str, spec) -> _Question:
    """Validate one question dict and normalize to candidates/labels."""
    if not isinstance(spec, Mapping):
        raise QuestionValidationError(
            f"question '{key}': must be an object, got {type(spec).__name__}")
    qtype = spec.get("type")
    instructions = _jsonable(spec.get("instructions"))

    if qtype == "noul":
        descs: dict[str, str] = {}
        criteria = spec.get("criteria")
        if isinstance(criteria, Mapping):
            for side in ("true", "false"):
                d = _jsonable(criteria.get(side))
                if d:
                    descs[side] = d
        return _Question(key=key, qtype="noul", instructions=instructions,
                         labels=list(_NOUL_LABELS),
                         candidates=[" " + l for l in _NOUL_LABELS],
                         noul_descs=descs)

    if qtype == "choice":
        criteria = spec.get("criteria")
        if not isinstance(criteria, Mapping):
            raise QuestionValidationError(
                f"question '{key}': choice requires 'criteria' object "
                f"mapping option labels to descriptions")
        if not (MIN_CHOICE_OPTIONS <= len(criteria) <= MAX_CHOICE_OPTIONS):
            raise QuestionValidationError(
                f"question '{key}': choice needs {MIN_CHOICE_OPTIONS}-"
                f"{MAX_CHOICE_OPTIONS} options, got {len(criteria)}")
        labels = [str(l) for l in criteria]
        descs = {l: _jsonable(v) for l, v in zip(labels, criteria.values())}
        return _Question(key=key, qtype="choice", instructions=instructions,
                         labels=labels,
                         candidates=[" " + l for l in labels],
                         option_descs=descs)

    if qtype == "score":
        criteria = spec.get("criteria")
        if not isinstance(criteria, (list, tuple)):
            raise QuestionValidationError(
                f"question '{key}': score requires 'criteria' list of "
                f"level descriptions")
        if not (MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS):
            raise QuestionValidationError(
                f"question '{key}': score needs {MIN_SCORE_LEVELS}-"
                f"{MAX_SCORE_LEVELS} levels, got {len(criteria)}")
        labels = [str(i) for i in range(len(criteria))]
        legend = {str(i): _jsonable(d) for i, d in enumerate(criteria)}
        return _Question(key=key, qtype="score", instructions=instructions,
                         labels=labels,
                         candidates=[" " + l for l in labels],
                         legend=legend)

    raise QuestionValidationError(
        f"question '{key}': unknown type {qtype!r}; "
        f"expected 'noul', 'choice', or 'score'")


def _peaked_confidence(probs: list[float]) -> float:
    """Normalized max-probability confidence in [0, 1].

    0 when the distribution is uniform over K candidates, 1 when it is a
    point mass on the winner.
    """
    k = len(probs)
    if k <= 1:
        return 1.0
    p_max = max(probs)
    return max(0.0, min(1.0, (p_max - 1.0 / k) / (1.0 - 1.0 / k)))


class SystemOneEvaluator:
    """Evaluate typed questions against a state in single forward passes.

    Parameters
    ----------
    model : ConfigurableResearchLLM (or compatible) — returns
        ``(logits, loss, ...)`` from ``model(ids, attention_mask=...)``.
    tokenizer : HF-style tokenizer (callable, .decode, .pad_token_id).
    device : torch.device | str
    max_batch : max candidate rows per continuation forward pass.
    """

    def __init__(self, model, tokenizer, device,
                 max_batch: int = DEFAULT_MAX_BATCH, scorer=None):
        self.model = model
        self.tok = tokenizer
        self.device = torch.device(device)
        self.max_batch = max(1, max_batch)
        # Optional trained DecisionScorer (decision_head.py).  When set,
        # candidate probabilities come from the verifier head instead of
        # raw LM token probabilities — calibrated, Tier 1.
        self.scorer = scorer

    # ── public API ─────────────────────────────────────────────────────

    def evaluate(self, state, questions: Mapping, model_id: str = "",
                 context_limit: int | None = None) -> dict:
        """Evaluate all questions against ``state``.

        Returns a TypeSafe-compatible response dict:
            {"model": ..., "answers": {...},
             "usage": {"billing_units": n, "input_tokens": n,
                       "output_tokens": 0}}
        """
        if state is None:
            raise QuestionValidationError("'state' is required (text or JSON)")
        if not isinstance(questions, Mapping) or not questions:
            raise QuestionValidationError(
                "'questions' must be a non-empty object keyed by question name")

        parsed = [_parse_question(k, q) for k, q in questions.items()]
        state_text = _jsonable(state)
        if not state_text.strip():
            raise QuestionValidationError("'state' must not be empty")

        limit = context_limit or DEFAULT_CONTEXT_LIMIT
        input_tokens = 0
        answers: dict[str, dict] = {}
        with torch.inference_mode():
            for q in parsed:
                probs, n_tok = self._score_question(q, state_text, limit)
                input_tokens += n_tok
                answers[q.key] = self._answer_dict(q, probs)
        return {
            "model": model_id,
            "answers": answers,
            "usage": {
                "billing_units": int(input_tokens),
                "input_tokens": int(input_tokens),
                "output_tokens": 0,
            },
        }

    # ── prompt construction ────────────────────────────────────────────

    @staticmethod
    def _prompt_parts(q: _Question) -> tuple[str, str]:
        """Return (prefix_before_state, suffix_after_state) for the prompt."""
        prefix = f"<|im_start|>user\n{_THINK_HINT}State:\n"
        if q.qtype == "noul":
            body = f"Statement: {q.instructions}\n"
            if q.noul_descs.get("true"):
                body += f"If yes: {q.noul_descs['true']}\n"
            if q.noul_descs.get("false"):
                body += f"If no: {q.noul_descs['false']}\n"
            body += 'Is the statement true? Answer with only "yes" or "no".'
        elif q.qtype == "choice":
            lines = [f"{q.instructions}\nOptions:"]
            for label in q.labels:
                desc = q.option_descs.get(label) or ""
                lines.append(f"- {label}: {desc}" if desc else f"- {label}")
            lines.append("Answer with only the option label.")
            body = "\n".join(lines)
        else:  # score
            hi = len(q.labels) - 1
            lines = [f"{q.instructions}\nScale:"]
            for _i, label in enumerate(q.labels):
                lines.append(f"{label}: {q.legend.get(label, '')}")
            lines.append(f"Answer with only the level number (0-{hi}).")
            body = "\n".join(lines)
        # Prompt ends at "Answer:" (no trailing space) — candidates carry
        # the leading space (" yes", " finance", " 3") so the scored token
        # is what the model would actually emit next.
        suffix = f"\n\n{body}<|im_end|>\n{_ASSISTANT_ANSWER_SLOT}"
        return prefix, suffix

    def _encode(self, text: str, special: bool) -> list[int]:
        enc = self.tok(text, add_special_tokens=special)
        ids = getattr(enc, "input_ids", None)
        if ids is None:
            ids = enc["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = list(ids)
        # gigatoken returns a flat list for a single string, HF returns [1, T]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = list(ids[0])
        return ids

    def _fit_state(self, state_text: str, prefix: str, suffix: str,
                   limit: int) -> str:
        """Truncate state (in tokens) so the full prompt fits ``limit``."""
        overhead = len(self._encode(prefix + suffix, special=True))
        budget = max(64, limit - overhead)
        sids = self._encode(state_text, special=False)
        if len(sids) > budget:
            state_text = self.tok.decode(sids[:budget])
        return state_text

    # ── scoring ────────────────────────────────────────────────────────

    def _forward_logits(self, ids: torch.Tensor,
                        attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.model(ids, attention_mask=attention_mask)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return logits

    def _score_question(self, q: _Question, state_text: str,
                        limit: int) -> tuple[list[float], int]:
        """Return (probabilities over q.candidates, input_tokens_used)."""
        prefix, suffix = self._prompt_parts(q)
        state_text = self._fit_state(state_text, prefix, suffix, limit)
        prompt_ids = self._encode(prefix + state_text + suffix, special=True)
        cand_ids = [self._encode(c, special=False) for c in q.candidates]
        cand_ids = [c if c else [0] for c in cand_ids]

        if self.scorer is not None:
            scores, n_tok = self._head_scores(prompt_ids, cand_ids)
            return self.scorer.probs(scores), n_tok

        if all(len(c) == 1 for c in cand_ids):
            scores = self._first_token_scores(prompt_ids, cand_ids)
            n_tok = len(prompt_ids)
        else:
            scores, n_tok = self._continuation_scores(prompt_ids, cand_ids)

        probs = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=-1)
        return probs.tolist(), n_tok

    def _candidate_hidden(self, prompt_ids: list[int],
                          cand_ids: list[list[int]],
                          pad_id: int) -> torch.Tensor:
        """Last-token hidden for each ``prompt + candidate`` row.

        Rows are right-padded and batched up to ``max_batch``; the hidden
        is read at each row's final real token (position L-1) — padding
        follows it, so a causal model is unaffected.  Returns
        (n_candidates, d_model) on ``self.device``.
        """
        rows = [prompt_ids + c for c in cand_ids]
        out_rows: list[torch.Tensor] = []
        for start in range(0, len(rows), self.max_batch):
            chunk = rows[start:start + self.max_batch]
            width = max(len(r) for r in chunk)
            batch = torch.full((len(chunk), width), pad_id,
                               dtype=torch.long, device=self.device)
            mask = torch.zeros((len(chunk), width),
                               dtype=torch.long, device=self.device)
            lens = []
            for i, row in enumerate(chunk):
                batch[i, :len(row)] = torch.tensor(row, dtype=torch.long)
                mask[i, :len(row)] = 1
                lens.append(len(row))
            out = self.model(batch, attention_mask=mask,
                             return_hidden=True)
            hidden = out[-1]
            out_rows.extend(hidden[i, L - 1] for i, L in enumerate(lens))
        return torch.stack(out_rows)

    def _head_scores(self, prompt_ids: list[int],
                     cand_ids: list[list[int]]) -> tuple[list[float], int]:
        """Verifier-head score for each candidate, batched right-padded."""
        pad_id = getattr(self.tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tok, "eos_token_id", None) or 0
        h = self._candidate_hidden(prompt_ids, cand_ids, pad_id)
        n_tok = sum(len(prompt_ids) + len(c) for c in cand_ids)
        return self.scorer.score(h).tolist(), n_tok

    def _first_token_scores(self, prompt_ids: list[int],
                            cand_ids: list[list[int]]) -> list[float]:
        """One forward; read softmax logits at each candidate's first token."""
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        logits = self._forward_logits(ids)
        last = logits[0, -1].float()
        idx = torch.tensor([c[0] for c in cand_ids], device=last.device)
        return last[idx].tolist()

    def _continuation_scores(self, prompt_ids: list[int],
                             cand_ids: list[list[int]]) -> tuple[list[float], int]:
        """Teacher-forced logprob of each candidate, batched right-padded."""
        pad_id = getattr(self.tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tok, "eos_token_id", None) or 0
        rows = [prompt_ids + c for c in cand_ids]
        p_len = len(prompt_ids)
        scores: list[float] = []
        n_tok = 0
        for start in range(0, len(rows), self.max_batch):
            chunk = rows[start:start + self.max_batch]
            width = max(len(r) for r in chunk)
            batch = torch.full((len(chunk), width), pad_id,
                               dtype=torch.long, device=self.device)
            mask = torch.zeros((len(chunk), width),
                               dtype=torch.long, device=self.device)
            for i, row in enumerate(chunk):
                batch[i, :len(row)] = torch.tensor(row, dtype=torch.long)
                mask[i, :len(row)] = 1
                n_tok += len(row)
            logits = self._forward_logits(batch, attention_mask=mask).float()
            logp = torch.log_softmax(logits, dim=-1)
            for i, row in enumerate(chunk):
                c_len = len(row) - p_len
                # positions p_len-1 .. len(row)-2 predict tokens p_len .. len(row)-1
                tgt = batch[i, p_len:p_len + c_len]
                lp = logp[i, p_len - 1:p_len + c_len - 1].gather(-1, tgt.unsqueeze(-1))
                scores.append(float(lp.sum()))
        return scores, n_tok

    # ── answer shaping ─────────────────────────────────────────────────

    @staticmethod
    def _answer_dict(q: _Question, probs: list[float]) -> dict:
        if q.qtype == "noul":
            # labels are ("yes", "no"); probs[0] = P(yes)
            return {"type": "noul", "noul": float(probs[0])}
        if q.qtype == "choice":
            win = max(range(len(probs)), key=lambda i: probs[i])
            return {
                "type": "choice",
                "choice": q.labels[win],
                "confidence": _peaked_confidence(probs),
                "probabilities": {l: float(p) for l, p in zip(q.labels, probs)},
            }
        # score
        expected = sum(i * p for i, p in enumerate(probs))
        return {
            "type": "score",
            "score": float(expected),
            "confidence": _peaked_confidence(probs),
            "legend": q.legend,
            "probabilities": {str(i): float(p) for i, p in enumerate(probs)},
        }
