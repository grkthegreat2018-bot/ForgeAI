"""Build the Spark SFT mix — data/sft/spark_sft_v1.jsonl.

Sources (all normalized to {prompt, response} — sft_train --chat-format
spark renders them into Spark's DeepSeek-style turns):

1. evolve-contract verified code — export_sft() over every evolve DB +
   existing evolve_sft_v2/v3/rsi_sft/distill_v10 rows. Responses are the
   verified fenced programs (the exact proposer output contract).
2. API think distill — forge_evolve_api.gens.jsonl rows whose closed
   <think> contains a fenced block: the LAST fence becomes the answer,
   the think tail (last ~1200 chars) becomes the reasoning. Teaches the
   bounded think -> ```python contract the proposer now enforces.
3. openthoughts3 — R1-style think->answer traces ending in a code fence.
4. magicoder_evol — code instruction -> fenced solution.
5. hermes_fc — tool_use prompts -> <tool_call> emissions (Spark's
   native tool format).

Usage: venv\\Scripts\\python.exe scripts\\gen_spark_sft.py
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "sft" / "spark_sft_v1.jsonl"
MAX_CHARS = 7800            # ~2048-token proxy; longer rows truncate badly
THINK_TAIL = 1400           # keep the conclusion of long reasoning
rng = random.Random(7)

DBS = ["forge_evolve.db", "forge_evolve_api.db", "forge_evolve_sin.db",
       "forge_evolve_spectrum.db", "forge_evolve_tanh44.db",
       "forge_evolve_mech.db", "forge_evolve_livefmt.db",
       "forge_evolve_meta.db", "forge_evolve_mla.db",
       "forge_evolve_v9.db", "forge_evolve_selfplay.db",
       "forge_evolve_flux.db", "forge_evolve_mink.db"]

_fence = re.compile(r"```[^\n`]*\n(.*?)```", re.S)


def _fenced(code: str) -> str:
    code = code.strip()
    if "```" in code:
        return code
    return "```python\n" + code + "\n```"


def emit(rows, prompt, response, source, tlp=None, reward=None):
    prompt = (prompt or "").strip()
    response = (response or "").strip()
    if not prompt or not response:
        return
    if len(prompt) + len(response) > MAX_CHARS:
        return
    r = {"prompt": prompt, "response": response, "source": source}
    if tlp is not None:
        r["teacher_logp"] = tlp
    if reward is not None:
        r["reward"] = reward
    rows.append(r)


def evolve_rows():
    """Verified winners exported per-DB + existing evolve SFT files."""
    from forge.evolution.code_evolve import export_sft
    rows, seen = [], set()
    tmp = ROOT / "data" / "sft" / "_spark_export_tmp.jsonl"
    for db in DBS:
        p = ROOT / db
        if not p.exists():
            continue
        try:
            n = export_sft(str(p), str(tmp))
        except Exception as e:
            print("export_sft failed", db, e)
            continue
        for line in open(tmp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            code = r.get("completion") or r.get("response") or ""
            key = (r.get("prompt", "")[:160], code[:160])
            if key in seen:
                continue
            seen.add(key)
            emit(rows, r.get("prompt", ""), _fenced(code), "evolve_db")
    tmp.unlink(missing_ok=True)
    for f in ["evolve_sft_v2.jsonl", "evolve_sft_v3.scored.jsonl",
              "forge_evolve.rsi_sft.jsonl"]:
        p = ROOT / "data" / "sft" / f
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            code = r.get("response") or r.get("completion") or ""
            key = (r.get("prompt", "")[:160], code[:160])
            if key in seen:
                continue
            seen.add(key)
            emit(rows, r.get("prompt", ""), _fenced(code), f[:-6],
                 tlp=r.get("teacher_logp"))
    # distill_v10: verified rows, reasoning -> bounded-think variant
    p = ROOT / "data" / "sft" / "distill_v10.scored.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("verified") is False:
                continue
            code = r.get("response") or ""
            key = (r.get("prompt", "")[:160], code[:160])
            if key in seen:
                continue
            seen.add(key)
            resp = _fenced(code)
            think = (r.get("reasoning") or "")[-THINK_TAIL:]
            if think.strip():
                resp = "<think>" + think.strip() + "</think>\n" + resp
            emit(rows, r.get("prompt", ""), resp, "distill_v10",
                 tlp=r.get("teacher_logp"))
    return rows


def api_think_rows():
    """Closed-think API gens -> move the last think-internal fence out."""
    rows = []
    p = ROOT / "forge_evolve_api.gens.jsonl"
    if not p.exists():
        return rows
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        raw = r.get("raw") or ""
        i = raw.find("</think>")
        if i < 0:
            continue
        think, after = raw[:i], raw[i + 8:]
        think = re.sub(r"</?think>", "", think)
        blocks = _fence.findall(after)
        if not blocks:
            blocks = _fence.findall(think)
            if not blocks:
                continue
            think = _fence.sub("", think)
        code = blocks[-1]
        think = think.strip()[-THINK_TAIL:]
        resp = ("<think>" + think + "</think>\n" if think else "")
        resp += "```python\n" + code.strip() + "\n```"
        emit(rows, r.get("prompt", ""), resp, "api_think")
    return rows


def hf_rows(name, want, cap, think_keep=True):
    rows = []
    p = ROOT / "data" / "hf_datasets" / (name + ".jsonl")
    if not p.exists():
        return rows
    pool = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        sol = r.get("solution") or ""
        if not want(sol):
            continue
        pool.append(r)
        if len(pool) >= cap * 4:
            break
    rng.shuffle(pool)
    for r in pool[:cap]:
        sol = (r.get("solution") or "").strip()
        # bound thinking inside the row too (same contract)
        if "<think>" in sol and "</think>" in sol:
            head, rest = sol.split("<think>", 1)
            think, tail = rest.split("</think>", 1)
            sol = (head + "<think>" + think.strip()[-THINK_TAIL:]
                   + "</think>" + tail) if think_keep else head + tail
        emit(rows, r.get("prompt", ""), sol, name)
    return rows


def main():
    rows = []
    ev = evolve_rows()
    print("evolve:", len(ev))
    rows += ev
    api = api_think_rows()
    print("api_think:", len(api))
    rows += api
    ot = hf_rows("openthoughts3",
                 lambda s: "<think>" in s and "```" in s
                 and s.rstrip().endswith("```"), cap=900)
    print("openthoughts3:", len(ot))
    rows += ot
    mc = hf_rows("magicoder_evol", lambda s: "```" in s, cap=1500)
    print("magicoder_evol:", len(mc))
    rows += mc
    fc = hf_rows("hermes_fc", lambda s: "<tool_call>" in s, cap=600)
    print("hermes_fc:", len(fc))
    rows += fc
    rng.shuffle(rows)
    seen, out = set(), []
    for r in rows:
        k = r["prompt"][:160]
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(out), "->", OUT)


if __name__ == "__main__":
    main()
