"""Test: can LFM3-v10 emit parseable goal specs (JSON) for self-expansion?

Same spec contract gen_llm_targets uses: {name, prompt, seed_code,
gen_cases}. The AST gate + subprocess validation drops bad specs, so
only parse-and-validate matters — measure raw spec validity.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.engine.forge_engine import ForgeEngine  # noqa: E402

SPEC_PROMPT = """Invent one NEW optimization target for a code-evolution
benchmark, about: TOPIC_HERE
Output ONLY a JSON object on one line with keys:
- "name": short snake_case id (e.g. "topk_histogram")
- "prompt": the task text a coder will receive (must ask to optimize a
  pure Python function; may import torch/numpy)
- "seed_code": complete naive reference implementation (correct, slow)
- "gen_cases": code that returns a list of test-case arg dicts
JSON only, no prose.

Example output:
{"name": "windowed_max", "prompt": "Optimize this sliding-window max:
def windowed_max(xs, k): naive double loop. Make it O(n).",
 "seed_code": "def windowed_max(xs, k):\n    return [max(xs[i:i+k]) for i in range(len(xs)-k+1)]",
 "gen_cases": "return [{'xs': [1,3,2,5,4], 'k': 2}]"}"""

# ChatML wrapper — built by concat so the tokens survive file tooling
S = "<" + "|im_start|>"
E = "<" + "|im_end|>"


def wrap(user_text: str) -> str:
    return S + "user\n" + user_text + E + "\n" + S + "assistant\n"


def main():
    eng = ForgeEngine.from_checkpoint(
        "research/checkpoints/lfm3_distill_v10_full")
    topics = ["tensor reshape", "string join"]
    ok = 0
    dump = Path("results/lfm3_goal_gens.jsonl")
    dump.parent.mkdir(exist_ok=True)
    with dump.open("w", encoding="utf-8") as df:
        for i, t in enumerate(topics):
            prompt = wrap(SPEC_PROMPT.replace("TOPIC_HERE", t))
            out = eng.generate(prompt, max_new_tokens=700,
                               temperature=0.7)
            txt = out if isinstance(out, str) else out.get("text", "")
            df.write(json.dumps({"topic": t, "raw": txt}) + "\n")
            m = re.search(r"\{.*\}", txt, re.S)
            spec = None
            if m:
                try:
                    spec = json.loads(m.group(0))
                except Exception:
                    spec = None
            good = (spec and all(k in spec for k in
                                 ("name", "prompt", "seed_code",
                                  "gen_cases")))
            ok += bool(good)
            print(f"[{i}] topic={t} parse={'OK' if good else 'BAD'}")
    print(f"\nspec-valid: {ok}/{len(topics)}")


if __name__ == "__main__":
    main()
