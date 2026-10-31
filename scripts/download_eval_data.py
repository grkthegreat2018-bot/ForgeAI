"""Download + normalize external eval data (COMPETITIVE_AUDIT A1).

Fetches public benchmark splits from HuggingFace `datasets` once and
writes normalized JSONL caches under data/eval/ (gitignored) that
forge/evaluation/external_eval.py consumes offline thereafter.

    .\\venv\\Scripts\\python.exe scripts\\download_eval_data.py
    .\\venv\\Scripts\\python.exe scripts\\download_eval_data.py --tasks gsm8k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "eval"

_LETTER_IDX = {c: i for i, c in enumerate("ABCDEFGH")}


def _write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} rows -> {path}")


def _arc(config: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config, split="test")
    rows = []
    for r in ds:
        key = r["answerKey"]
        idx = _LETTER_IDX.get(key, int(key) - 1 if key.isdigit() else -1)
        if idx < 0 or idx >= len(r["choices"]["text"]):
            continue
        rows.append({"context": r["question"],
                     "choices": list(r["choices"]["text"]),
                     "answer": idx})
    return rows


def _hellaswag() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rows = []
    for r in ds:
        if not r["label"].isdigit():
            continue
        rows.append({"context": r["ctx"], "choices": list(r["endings"]),
                     "answer": int(r["label"])})
    return rows


def _mmlu() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = [{"context": r["question"], "choices": list(r["choices"]),
             "answer": int(r["answer"])} for r in ds]
    return rows


def _gsm8k() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for r in ds:
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"question": r["question"], "answer": float(final)})
    return rows


_FETCHERS = {
    "arc_easy": lambda: _arc("ARC-Easy"),
    "arc_challenge": lambda: _arc("ARC-Challenge"),
    "hellaswag": _hellaswag,
    "mmlu": _mmlu,
    "gsm8k": _gsm8k,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(_FETCHERS))
    args = ap.parse_args()
    for name in args.tasks:
        _write(name, _FETCHERS[name]())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
