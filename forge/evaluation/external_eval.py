"""External benchmark evaluation (COMPETITIVE_AUDIT A1).

The credibility gap: every prior performance claim was measured on
internal corpora and internal baselines. This module runs standard
public benchmarks — ARC-Easy/ARC-Challenge, HellaSwag, MMLU, GSM8K —
against any logits-producing model, using the lm-eval-harness scoring
conventions (sum-loglikelihood argmax for multiple choice, acc_norm =
loglik per output byte for HellaSwag, greedy + "####"-extraction for
GSM8K) so numbers are comparable to published results.

Data is NOT vendored: `scripts/download_eval_data.py` fetches the HF
datasets once and caches normalized JSONL under ``data/eval/`` (the
``data/`` tree is gitignored). This module reads only those caches,
so it works offline after the first download.

Usage:
    from forge.evaluation.external_eval import run_eval, load_task
    report = run_eval(model, tokenizer, device, tasks=["arc_easy"],
                      limit=200)
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "eval"

#: Multiple-choice tasks scored by sum-loglikelihood argmax.
MC_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "mmlu")
#: acc_norm (loglik-per-byte) is the headline metric lm-eval reports for
#: HellaSwag; ARC reports raw acc. We compute both for every MC task.
NORM_TASKS = ("hellaswag",)
GEN_TASKS = ("gsm8k",)
ALL_TASKS = MC_TASKS + GEN_TASKS

_LETTERS = "ABCDEFGH"


# ── Data loading ──────────────────────────────────────────────────────────


def load_task(name: str, limit: int | None = None,
              data_dir: str | Path = DEFAULT_DATA_DIR) -> list[dict]:
    """Load normalized rows for *name* from the local JSONL cache.

    Row schemas (written by scripts/download_eval_data.py):
      mc:   {"context": str, "choices": [str], "answer": int}
      gsm8k: {"question": str, "answer": float}
    """
    path = Path(data_dir) / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/download_eval_data.py first")
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    return rows


# ── Model interface ───────────────────────────────────────────────────────


def _forward_logits(model: Any, ids: torch.Tensor) -> torch.Tensor:
    """Return (T, V) logits for a (1, T) id row from any engine model."""
    out = model(ids)
    logits = out[0] if isinstance(out, tuple) else (
        out.logits if hasattr(out, "logits") else out)
    return logits[0]


def _encode(tokenizer: Any, text: str, device: torch.device) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return list(ids)


def loglikelihood(model: Any, tokenizer: Any, context: str,
                  continuation: str, device: torch.device,
                  max_seq_len: int = 8192) -> tuple[float, int, bool]:
    """Sum-logprob of *continuation* given *context*.

    Returns (sum_logprob, n_cont_tokens, all_greedy). Mirrors the
    lm-eval ``loglikelihood`` request: tokens are scored teacher-forced
    over a single concatenated forward; the context/continuation split
    is taken on the token level so BPE-boundary artifacts are avoided.
    """
    ctx_ids = _encode(tokenizer, context, device)
    cont_ids = _encode(tokenizer, continuation, device)
    if not cont_ids:
        return 0.0, 0, False
    ids = ctx_ids + cont_ids
    if len(ids) > max_seq_len:  # keep the tail: continuation must survive
        overflow = len(ids) - max_seq_len
        ids = ids[overflow:]
        ctx_len = max(len(ctx_ids) - overflow, 0)
    else:
        ctx_len = len(ctx_ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = _forward_logits(model, x)            # (T, V)
    logp = F.log_softmax(logits[:-1].float(), dim=-1)
    tgt = x[0, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    cont_lp = tok_lp[ctx_len - 1:]                    # predicts cont tokens
    greedy = (logp[ctx_len - 1:].argmax(-1) == tgt[ctx_len - 1:])
    return float(cont_lp.sum().item()), int(cont_lp.numel()), bool(
        greedy.all().item())


def generate(model: Any, tokenizer: Any, prompt: str,
             device: torch.device, max_new_tokens: int = 256,
             stop_ids: set[int] | None = None,
             max_seq_len: int = 8192) -> str:
    """Greedy generation; returns decoded continuation text only."""
    ids = _encode(tokenizer, prompt, device)[-max_seq_len:]
    prompt_len = len(ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    stop = set(stop_ids or ())
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stop.add(eos)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = _forward_logits(model, x[:, -max_seq_len:])
            nxt = int(logits[-1].argmax().item())
            if nxt in stop:
                break
            x = torch.cat(
                [x, torch.tensor([[nxt]], dtype=torch.long,
                                 device=device)], dim=1)
    return tokenizer.decode(x[0, prompt_len:].tolist(),
                            skip_special_tokens=True)


# ── Prompt formats (lm-eval conventions) ─────────────────────────────────


def _mc_prompt(context: str, choices: list[str] | None = None) -> str:
    return f"Question: {context.strip()}\nAnswer:"


def _mmlu_prompt(question: str, choices: list[str]) -> str:
    body = question.strip() + "\n"
    for i, ch in enumerate(choices):
        body += f"{_LETTERS[i]}. {ch}\n"
    return body + "Answer:"


def _gsm8k_prompt(question: str) -> str:
    return ("Question: " + question.strip()
            + "\nAnswer: Let's think step by step.")


_GSM_RE = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")
_NUM_RE = re.compile(r"[-+]?[\d,]*\.?\d+")


def extract_gsm8k_answer(text: str) -> float | None:
    """Pull the final numeric answer from generated GSM8K text."""
    m = _GSM_RE.search(text)
    raw = m.group(1) if m else (
        _NUM_RE.findall(text) or [None])[-1]
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


# ── Task runners ──────────────────────────────────────────────────────────


def run_mc_task(model: Any, tokenizer: Any, rows: list[dict],
                device: torch.device, task: str,
                max_seq_len: int = 8192,
                progress: Callable[[int, int], None] | None = None
                ) -> dict[str, Any]:
    """Score a multiple-choice task by sum-loglikelihood argmax."""
    correct = correct_norm = 0
    n = 0
    for i, row in enumerate(rows):
        choices = row["choices"]
        gold = int(row["answer"])
        if task == "mmlu":
            ctx = _mmlu_prompt(row["context"], choices)
            conts = [f" {_LETTERS[j]}" for j in range(len(choices))]
        else:
            ctx = _mc_prompt(row["context"])
            conts = [f" {c.strip()}" for c in choices]
        lps, norms = [], []
        for cont in conts:
            lp, ntok, _ = loglikelihood(
                model, tokenizer, ctx, cont, device, max_seq_len)
            lps.append(lp)
            norms.append(lp / max(len(cont.encode("utf-8")), 1))
        correct += int(int(np_argmax(lps)) == gold)
        correct_norm += int(int(np_argmax(norms)) == gold)
        n += 1
        if progress and (i % 25 == 0 or i == len(rows) - 1):
            progress(i + 1, len(rows))
    return {"task": task, "n": n,
            "acc": correct / max(n, 1),
            "acc_norm": correct_norm / max(n, 1)}


def run_gsm8k(model: Any, tokenizer: Any, rows: list[dict],
              device: torch.device, max_new_tokens: int = 256,
              max_seq_len: int = 8192,
              progress: Callable[[int, int], None] | None = None
              ) -> dict[str, Any]:
    """Greedy-generate GSM8K answers; exact-match the final number."""
    correct = 0
    n = 0
    for i, row in enumerate(rows):
        text = generate(model, tokenizer, _gsm8k_prompt(row["question"]),
                        device, max_new_tokens=max_new_tokens,
                        max_seq_len=max_seq_len)
        pred = extract_gsm8k_answer(text)
        gold = float(row["answer"])
        correct += int(pred is not None
                       and math.isclose(pred, gold, rel_tol=1e-4,
                                        abs_tol=1e-4))
        n += 1
        if progress and (i % 10 == 0 or i == len(rows) - 1):
            progress(i + 1, len(rows))
    return {"task": "gsm8k", "n": n, "acc": correct / max(n, 1)}


def np_argmax(xs: list[float]) -> int:
    best, bi = xs[0], 0
    for i, x in enumerate(xs):
        if x > best:
            best, bi = x, i
    return bi


def run_eval(model: Any, tokenizer: Any, device: torch.device | str,
             tasks: list[str] | tuple[str, ...] = ALL_TASKS,
             limit: int | None = None,
             data_dir: str | Path = DEFAULT_DATA_DIR,
             max_seq_len: int = 8192,
             gsm8k_max_new: int = 256,
             verbose: bool = True) -> dict[str, Any]:
    """Run the external benchmark suite; returns a report dict."""
    device = torch.device(device)
    model.eval()

    def _prog(name: str):
        def f(i: int, n: int) -> None:
            if verbose:
                print(f"  {name}: {i}/{n}", flush=True)
        return f

    results: dict[str, Any] = {}
    for task in tasks:
        rows = load_task(task, limit=limit, data_dir=data_dir)
        if task in GEN_TASKS:
            results[task] = run_gsm8k(
                model, tokenizer, rows, device,
                max_new_tokens=gsm8k_max_new, max_seq_len=max_seq_len,
                progress=_prog(task))
        elif task in MC_TASKS:
            results[task] = run_mc_task(
                model, tokenizer, rows, device, task,
                max_seq_len=max_seq_len, progress=_prog(task))
        else:
            raise ValueError(f"unknown task {task!r}; have {ALL_TASKS}")
    headline = {t: (r["acc_norm"] if t in NORM_TASKS else r["acc"])
                for t, r in results.items()}
    return {"tasks": results, "headline": headline,
            "metric_note": "acc_norm=loglik/byte (hellaswag headline); "
                           "gsm8k=greedy exact-match"}


def save_report(report: dict[str, Any], path: str | Path,
                model_tag: str, meta: dict[str, Any] | None = None
                ) -> Path:
    """Persist a report JSON — the regression-gate artifact for A1."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_tag, "benchmark": "external_v1",
               **(meta or {}), **report}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def compare_reports(base: dict[str, Any], cand: dict[str, Any],
                    tol: float = 0.01) -> dict[str, Any]:
    """Regression-gate comparison: cand must not drop any headline metric
    by more than *tol* absolute. Returns {"ok", "deltas"}."""
    deltas, ok = {}, True
    for task, hb in base.get("headline", {}).items():
        hc = cand.get("headline", {}).get(task)
        if hc is None:
            continue
        deltas[task] = round(hc - hb, 4)
        ok &= (hc - hb) >= -tol
    return {"ok": ok, "deltas": deltas}
