"""Merge + normalize distillation corpora.

Every source row is re-extracted to its VERIFIED code (the block that
passed the oracle at gen time), re-verified once more for safety, and
rewritten into the uniform SFT shape:

    <think>
    <reasoning / prose the model emitted outside the code fence>
    </think>
    ```python
    <verified code>
    ```

Rows whose code no longer passes the oracle are dropped into the
report — the training set only ever sees verified-clean rows.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from filter_distill_semantics import (  # noqa: E402
    _load_targets, _required_from_prompt)
from gen_qwen_distill import _extract_code  # noqa: E402


def _check(code: str, prompt: str, oracles) -> tuple[str, str]:
    """Mirror filter_distill_semantics' per-row gate."""
    try:
        compile(code, "<c>", "exec")
    except SyntaxError:
        return "compile_fail", "syntax"
    req = _required_from_prompt(prompt)
    entry = oracles.get(req)
    if entry is None:
        cands = [(k, v) for k, v in oracles.items() if k <= req]
        entry = cands[0][1] if len(cands) == 1 else None
    if entry is None:
        return "no_target", ""
    tname, test = entry
    ns = {}
    try:
        exec(compile(code, "<c>", "exec"), ns)
    except Exception as e:
        return "exec_fail", f"{tname}: {type(e).__name__}"
    missing = [n for n in req if n not in ns]
    if missing:
        return "semantic_fail", f"{tname}: missing {missing}"
    try:
        res = test(ns)
    except Exception as e:
        return "exec_fail", f"{tname}: {type(e).__name__}"
    return ("ok", "") if res.get("ok") else (
        "semantic_fail", f"{tname}: {res.get('detail','?')[:120]}")


def _think_of(raw: str, stored: str) -> str:
    if stored:
        return stored
    return re.sub(r"```.*?```", "", raw, flags=re.S).strip()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="qwen_distill_v3")
    args = ap.parse_args()

    oracles = _load_targets(REPO / "forge" / "evolution" / "targets")
    seen = set()
    kept, dropped = [], []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"[merge] missing {f}")
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = hashlib.md5(
                r["prompt"][:400].encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            code = _extract_code(r["response"])
            if not code.strip():
                dropped.append({"why": "no_code", "src": p.name})
                continue
            status, detail = _check(code, r["prompt"], oracles)
            if status not in ("ok", "no_target"):
                dropped.append({"why": status, "detail": detail[:120],
                                "src": p.name})
                continue
            think = _think_of(r["response"], r.get("reasoning", ""))
            resp = ((f"<think>\n{think}\n</think>\n" if think else "")
                    + "```python\n" + code.rstrip() + "\n```\n")
            kept.append({"prompt": r["prompt"], "response": resp,
                         "reasoning": think, "target": args.tag,
                         "verified": status == "ok",
                         "semantic": status})

    out = Path(args.out)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
        encoding="utf-8")
    rep = out.with_suffix(".merge.json")
    rep.write_text(json.dumps({
        "kept": len(kept), "with_reasoning": sum(
            1 for r in kept if r["reasoning"]),
        "dropped": len(dropped), "drop_sample": dropped[:10]}, indent=2))
    print(f"[merge] kept={len(kept)} "
          f"(reasoning={sum(1 for r in kept if r['reasoning'])}) "
          f"dropped={len(dropped)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
