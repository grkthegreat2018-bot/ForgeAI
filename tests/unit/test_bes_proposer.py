"""Tests for the BES bidirectional-search proposer
(forge/evolution/bes_proposer.py).

CPU-only: the inner LLM is a scripted fake — no model loads, no GPU.
"""

from __future__ import annotations

import textwrap

from forge.evolution.bes_proposer import BESProposer
from forge.evolution.code_evolve import (
    CodeEvolve,
    CodeEvolveConfig,
    _extract_code,
)

# Mirror of the unprimed _build_prompt shape (primed=False tail).
PROMPT = (
    "# Rewrite this Python program to be much better.\n"
    "# Candidate: abc12345\n"
    "# Direction: ms_per_call MINIMIZE\n"
    "# Hard budgets (must satisfy): none\n"
    "# Rules: keep public names (fast_sum); stdlib+numpy+torch only;\n"
    "# output ONLY one python code block; identical semantics.\n"
    "#\n"
    "# Current best (ms_per_call=1.234, metrics={}):\n"
    "```python\n"
    "def fast_sum(xs):\n"
    "    return sum(xs)\n"
    "```\n"
    "Reply with ONLY one ```python code block containing the complete "
    "improved implementation. Skip <think> for small changes; use a "
    "short <think> only when the optimization is subtle."
)

CRITIQUE = ("# Two implementations of the same function.\n"
            "# CHALLENGER beat INCUMBENT by 0.01 on ms_per_call.\n"
            "INCUMBENT:\n```python\ndef fast_sum(xs):\n    return 0\n```\n"
            "In 2-3 sentences: what changed?")

CODE = ("```python\nimport torch\n\n\ndef fast_sum(xs):\n"
        "    return float(torch.as_tensor(xs, dtype=torch.float64).sum())\n"
        "```")

DUMMY = textwrap.dedent("""\
    NAME = "fast_sum"
    GOAL = "min"
    METRIC = "ms_per_call"
    TAGS = ["test"]
    BUDGETS = {}
    REQUIRED = ["fast_sum"]
    SEED_CODE = "def fast_sum(xs):\\n    return sum(xs)\\n"
    def test(ns):
        return {"ok": True}
    def bench(ns):
        return {"ms_per_call": 1.0}
    """)


class _FakeLLM:
    """Scripted LLMProposer stand-in: pops one queued raw per call and
    records every prompt it saw (batch calls record the prompt list)."""

    primed = False

    def __init__(self, queue=None):
        self.queue = list(queue or [])
        self.calls = []
        self.notes = []
        self._last_adapter = None

    def _pop(self):
        return self.queue.pop(0) if self.queue else CODE

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._pop()

    def propose_batch(self, prompts):
        prompts = list(prompts)
        self.calls.append(prompts)
        return [self._pop() for _ in prompts]

    def note(self, code, credit, delta, prompt=None):
        self.notes.append((code, credit, delta, prompt))


class _FakeSerialLLM(_FakeLLM):
    """No propose_batch — exercises the serial fallback path."""
    propose_batch = None


def _prop(llm=None, **cfg_kw):
    cfg = CodeEvolveConfig(target="x", verbose=False, **cfg_kw)
    fake = llm or _FakeLLM()
    return BESProposer(cfg, llm=fake), fake


# ── prompt transforms ────────────────────────────────────────────────

def test_decompose_prompt_replaces_reply_contract():
    prop, _ = _prop()
    p = prop._decompose_prompt(PROMPT)
    assert "BACKWARD PASS" in p
    assert "Sub-goals:" in p
    # The code-fence contract is replaced, not stacked underneath.
    assert "Reply with ONLY one ```python code block" not in p
    assert "8" in p                       # max_subgoals bound stated
    # Parent program context survives for decomposition.
    assert "def fast_sum" in p


def test_parse_subgoals_basic_and_bounded():
    prop, _ = _prop(bes_subgoals=3)
    raw = ("<think>plan: decompose first</think>\n"
           "Here is the list:\n"
           "1. keep the fast_sum signature and semantics\n"
           "2) vectorize the loop with a torch reduction\n"
           "# 3. avoid python-level temporaries\n"
           "4. drop float() casts until the end\n"
           "5. this one is over the bound\n")
    goals = prop._parse_subgoals(raw)
    assert goals == [
        "keep the fast_sum signature and semantics",
        "vectorize the loop with a torch reduction",
        "avoid python-level temporaries",
    ]


def test_parse_subgoals_rejects_code_and_trivia():
    prop, _ = _prop()
    # Fenced code is never a sub-goal; unclosed fence drops the tail.
    raw = ("1. real goal\n"
           "```python\n"
           "2. x = compute()\n"
           "3. fake goal\n"
           "```\n"
           "4. post-fence goal\n"
           "```python\n"
           "5. trailing code that never closed\n"
           "6. lost\n")
    goals = prop._parse_subgoals(raw)
    assert goals == ["real goal", "post-fence goal"]
    # Trivia: sub-4-char and empty items are dropped.
    assert prop._parse_subgoals("1. x\n2. \n3. ok goal\n") == ["ok goal"]
    assert prop._parse_subgoals("") == []
    assert prop._parse_subgoals("no list at all") == []


def test_inject_subgoals_before_reply_contract():
    prop, _ = _prop()
    solve = prop._inject_subgoals(
        PROMPT, ["vectorize with torch", "keep semantics"])
    gi = solve.find("vectorize with torch")
    ri = solve.rfind("Reply with ONLY")
    assert 0 < gi < ri
    assert "#   1. vectorize with torch" in solve
    assert "#   2. keep semantics" in solve


# ── proposer contract ────────────────────────────────────────────────

def test_call_two_phase_flow():
    fake = _FakeLLM([
        "1. keep signature\n2. vectorize with torch.sum\n",
        CODE,
    ])
    prop, fake = _prop(llm=fake)
    out = prop(PROMPT)
    assert out == CODE
    assert len(fake.calls) == 2
    dec, solve = fake.calls
    assert "BACKWARD PASS" in dec and "Sub-goals:" in dec
    assert "keep signature" in solve
    assert "vectorize with torch.sum" in solve
    assert prop._solve_of[PROMPT] == solve
    # The emitted raw still parses through the standard extraction.
    code = _extract_code(out, "", ["fast_sum"])
    assert "def fast_sum" in code
    compile(code, "<cand>", "exec")


def test_call_forward_fallback_on_empty_decomposition():
    fake = _FakeLLM(["sorry, I cannot help with that", CODE])
    prop, fake = _prop(llm=fake)
    out = prop(PROMPT)
    assert out == CODE
    # Pass 1 = decompose; pass 2 = the ORIGINAL prompt, unsullied.
    assert len(fake.calls) == 2
    assert fake.calls[1] == PROMPT
    assert PROMPT not in prop._solve_of


def test_call_non_proposal_is_single_pass():
    prop, fake = _prop()
    out = prop(CRITIQUE)
    assert len(fake.calls) == 1
    assert fake.calls[0] == CRITIQUE
    assert out == CODE  # scripted default


def test_call_llm_error_falls_back_forward():
    class _Flaky(_FakeLLM):
        def __call__(self, prompt):
            self.calls.append(prompt)
            if len(self.calls) == 1:
                raise RuntimeError("engine boom")
            return self._pop()

    prop, fake = _prop(llm=_Flaky())
    out = prop(PROMPT)
    assert out == CODE
    assert len(fake.calls) == 2
    assert fake.calls[1] == PROMPT


def test_propose_batch_two_batched_passes():
    p2 = PROMPT.replace("abc12345", "def67890")
    fake = _FakeLLM([
        # decompose batch — one raw per proposal prompt
        "1. goal one\n2. goal two\n",
        "garbage, no numbered items\n",
        # solve batch — one raw per input prompt
        CODE,
        CODE.replace("torch.float64", "torch.float32"),
        CODE,
    ])
    prop, fake = _prop(llm=fake)
    out = prop.propose_batch([PROMPT, p2, CRITIQUE])
    assert len(out) == 3
    assert len(fake.calls) == 2
    dec_ps, solve_ps = fake.calls
    # Decompose batch only saw the two proposal prompts.
    assert len(dec_ps) == 2
    assert all("BACKWARD PASS" in p for p in dec_ps)
    # Solve batch: row 0 got sub-goals, row 1 fell back to the original
    # (empty decomposition), row 2 (critique) passes through unchanged.
    assert len(solve_ps) == 3
    assert "#   1. goal one" in solve_ps[0]
    assert "#   2. goal two" in solve_ps[0]
    assert solve_ps[1] == p2
    assert solve_ps[2] == CRITIQUE


def test_propose_batch_serial_fallback_without_batch_method():
    prop, fake = _prop(llm=_FakeSerialLLM([
        "1. goal one\n",
        CODE,
    ]))
    out = prop.propose_batch([PROMPT])
    # decompose (serial) + solve (serial)
    assert len(out) == 1 and out[0] == CODE
    assert len(fake.calls) == 2


def test_propose_batch_empty():
    prop, _ = _prop()
    assert prop.propose_batch([]) == []


def test_note_maps_credit_to_solve_prompt():
    fake = _FakeLLM(["1. goal one\n", CODE])
    prop, fake = _prop(llm=fake)
    prop(PROMPT)
    solve_prompt = prop._solve_of[PROMPT]
    assert "goal one" in solve_prompt and solve_prompt != PROMPT
    prop.note("prog", 1.0, 0.5, prompt=PROMPT)
    # Credit lands on the SOLVE prompt (hint attribution), not the
    # original — and the mapping is consumed.
    assert fake.notes[-1] == ("prog", 1.0, 0.5, solve_prompt)
    assert PROMPT not in prop._solve_of


def test_contract_attrs_proxy():
    prop, fake = _prop()
    assert prop.primed is False
    fake._last_adapter = "tag7"
    assert prop.last_adapter == "tag7"
    fake._lfm3 = True
    assert prop._lfm3 is True


def test_bes_max_subgoals_config():
    prop, _ = _prop(bes_subgoals=2)
    assert prop.max_subgoals == 2
    raw = "1. a-one\n2. b-two\n3. c-three\n"
    assert prop._parse_subgoals(raw) == ["a-one", "b-two"]
    # Invalid/zero config degrades to the class bound, not a crash.
    prop2, _ = _prop(bes_subgoals=0)
    assert prop2.max_subgoals == BESProposer.MAX_SUBGOALS


# ── dispatch wiring ──────────────────────────────────────────────────

def test_dispatch_builds_bes(tmp_path):
    """cfg.proposer='bes' -> CodeEvolve wires a BESProposer (opt-in;
    other modes untouched)."""
    tp = tmp_path / "t.py"
    tp.write_text(DUMMY, encoding="utf-8")
    cfg = CodeEvolveConfig(
        target=str(tp), db_path=str(tmp_path / "d.db"),
        proposer="bes", verbose=False)
    ce = CodeEvolve(cfg)
    assert isinstance(ce._proposer, BESProposer)
    assert ce._basin_mode is False
    # The inner delegate is a real LLMProposer (engine lazy, no load).
    from forge.evolution.code_evolve import LLMProposer
    assert isinstance(ce._proposer._llm, LLMProposer)
    # Default proposer stays mut.
    cfg2 = CodeEvolveConfig(
        target=str(tp), db_path=str(tmp_path / "d2.db"),
        verbose=False)
    ce2 = CodeEvolve(cfg2)
    from forge.evolution.mutator import Mutator
    assert isinstance(ce2._proposer, Mutator)
