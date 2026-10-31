"""Generate CopySpan supervision rows from evolve DB parent->child diffs.

For each verified code_programs row with a resolvable parent, we align
parent->child at token level (difflib opcodes) and emit:

  {"target":..., "src_ids":[parent tokens], "prompt": <evolve prompt>,
   "ops":[{"copy":[s,e]} | {"tok":[ids]} ...]}

op semantics (matches forge/decoding/edit_decode.py):
  copy [s,e] -> emit <copy> then verbatim src_ids[s:e+1]
  tok  [ids] -> emit the literal token run

Usage:
  venv\\Scripts\\python.exe scripts\\gen_edit_sft.py --db forge_evolve.db \
      --out data/sft/edit_spans.jsonl --tokenizer research/checkpoints/forgelm_v2_tokenizer
"""
from __future__ import annotations

import argparse
import difflib
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=REPO / "forge_evolve.db")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path,
                    default=REPO / "research/checkpoints/forgelm_v2_tokenizer")
    ap.add_argument("--min-span", type=int, default=8,
                    help="copy ops shorter than this emit as literals")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))

    con = sqlite3.connect(str(args.db))
    rows = con.execute(
        "SELECT code_hash, parent_hash, code, prompt, target FROM "
        "code_programs WHERE ok=1 AND parent_hash IS NOT NULL").fetchall()
    by_hash = dict(con.execute("SELECT code_hash, code FROM code_programs"))

    n_out = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for code_hash, parent_hash, code, prompt, target in rows[:args.limit or None]:
            parent = by_hash.get(parent_hash)
            if not parent or parent == code:
                continue
            s_ids = tok(parent, add_special_tokens=False).input_ids
            t_ids = tok(code, add_special_tokens=False).input_ids
            sm = difflib.SequenceMatcher(None, s_ids, t_ids, autojunk=False)
            ops = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal" and (i2 - i1) >= args.min_span:
                    ops.append({"copy": [i1, i2 - 1]})
                elif tag != "equal" or (i2 - i1) > 0:
                    lit = t_ids[j1:j2]
                    if lit:
                        ops.append({"tok": lit})
            if not ops or not any("copy" in o for o in ops):
                continue
            f.write(json.dumps({
                "target": target, "code_hash": code_hash,
                "parent_hash": parent_hash, "prompt": prompt or "",
                "src_ids": s_ids, "ops": ops}) + "\n")
            n_out += 1
    print(f"wrote {n_out} edit-span rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
