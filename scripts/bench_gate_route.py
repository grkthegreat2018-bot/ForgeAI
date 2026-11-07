"""Measure ForgeGate route-probe scores on real chat prompts.

The route head was trained on the gate_r template
(``<|im_start|>user\n{THINK_HINT}{q}<|im_end|>\n{REASONED_TAIL}``).
chat_loop scores the fully-rendered conversation WITHOUT the <tools>
block — measured on V2 the ~17-schema tools block collapses h_mean and
drags borderline prompts under threshold ("Hello": 0.355 tools-free vs
0.098 with tools), so the production probe input is tools-free by design
and bare greetings/acks bypass the probe entirely (_is_trivial_turn).
This bench prints p_easy for the full-render vs question-only variants
side-by-side so a route threshold can be chosen from numbers, not vibes.

Usage:
    python scripts/bench_gate_route.py          # p_easy table
    python scripts/bench_gate_route.py --e2e    # + direct/capped decode check
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = [
    "hello",
    "what's 2+2",
    "thanks!",
    "yes",
    "Explain what this project does, at a high level",
    "Write a Python function that checks whether a number is prime",
    "Prove that sqrt(2) is irrational",
    "My quicksort is O(n^2) on sorted input — why, and how do I fix it?",
]


def main() -> None:
    from forge.engine.forge_engine import ForgeEngine
    from forge.engine.gated import REASONED_TAIL, THINK_HINT
    from forge.self_play.discovery.qwen_adapter import render_messages_for_config
    from forge_gui.api.master_prompt import get_default_prompt_for_config
    from forge_gui_server.services.chat_loop import (
        _DIRECT_SUFFIX,
        _FORCE_ANSWER_SUFFIX,
        _THINK_MAX_TOKENS,
        _route_p_easy,
        _think_cap_processor,
    )

    engine = ForgeEngine.from_checkpoint(
        checkpoint=str(ROOT / "research/checkpoints/ForgeLM_V2.safetensors"),
        config_name="forgelm_v2")
    engine.load_gate_probes(
        str(ROOT / "research/checkpoints/gate_probes.safetensors"))

    system = get_default_prompt_for_config("forgelm_v2")

    def q_prompt(q: str) -> str:
        return (f"<|im_start|>user\n{THINK_HINT}{q}<|im_end|>\n"
                + REASONED_TAIL)

    print(f"{'p_full':>7} {'p_qonly':>7}  question")
    for q in QUESTIONS:
        conv = [{"role": "system", "content": system},
                {"role": "user", "content": q}]
        full = render_messages_for_config(
            conv, config_name="forgelm_v2", tools=None,
            add_generation_prompt=True, thinking=True)
        p_full = _route_p_easy(engine, full)
        p_q = _route_p_easy(engine, q_prompt(q))
        fs = f"{p_full:.3f}" if p_full is not None else "  —"
        qs = f"{p_q:.3f}" if p_q is not None else "  —"
        print(f"{fs:>7} {qs:>7}  {q[:58]}")

    if "--e2e" not in sys.argv:
        return

    # Direct path: thinking=False render + closed-think suffix. Expect a
    # clean greeting with no reasoning text.
    conv = [{"role": "system", "content": system},
            {"role": "user", "content": "hello"}]
    direct = render_messages_for_config(
        conv, config_name="forgelm_v2", tools=None,
        add_generation_prompt=True, thinking=False) + _DIRECT_SUFFIX
    out = engine.generate_raw(direct, max_new_tokens=48, temperature=0.0,
                              skip_special_tokens=True)
    print("\n[direct 'hello']\n" + out.strip()[:300])

    # Think path with the cap: </think> must appear right after the budget
    # and the answer must follow.
    conv = [{"role": "system", "content": system},
            {"role": "user", "content": "Prove that sqrt(2) is irrational"}]
    think = render_messages_for_config(
        conv, config_name="forgelm_v2", tools=None,
        add_generation_prompt=True, thinking=True)
    out = engine.generate_raw(
        think, max_new_tokens=140, temperature=0.0,
        logits_processor=_think_cap_processor(
            48, engine.tokenizer.encode(
                _FORCE_ANSWER_SUFFIX, add_special_tokens=False)),
        skip_special_tokens=False)
    print("\n[capped think 'sqrt(2)' — cap=48]\n" + out.strip()[:900])

    # Repro of the reported failure: "hello" on the think path used to
    # cap mid-ramble, print "</think>\nAnswer:", then REOPEN <think> and
    # loop. With the 541 ban it must answer and stop.
    conv = [{"role": "system", "content": system},
            {"role": "user", "content": "hello"}]
    think = render_messages_for_config(
        conv, config_name="forgelm_v2", tools=None,
        add_generation_prompt=True, thinking=True)
    out = engine.generate_raw(
        think, max_new_tokens=200, temperature=0.0,
        logits_processor=_think_cap_processor(
            _THINK_MAX_TOKENS,
            engine.tokenizer.encode(
                _FORCE_ANSWER_SUFFIX, add_special_tokens=False)),
        skip_special_tokens=False)
    print("\n[capped think 'hello' — cap=160]\n" + out.strip()[:900])


if __name__ == "__main__":
    main()
