"""BES proposer — bidirectional-search candidate proposals (RSI-loop
whitespace recon doc §E.2 / action note 2).

All existing proposers are FORWARD-only: mutate-or-regenerate the parent
program and hope the rollout lands in a better basin (mut | mix | llm |
basin | api). BES adds the backward half: before any code is written, the
LLM DECOMPOSES the goal into bounded sub-goals (the contract, the
algorithmic steps, the edge cases, the performance property that wins),
then a forward joint-solve pass emits ONE program conditioned on the
sub-goal list — the same fenced-code contract the evolve loop already
parses via ``code_evolve._extract_code``.

Two LLM passes per proposal (decompose + solve) — batched as two
generate_batch calls in propose_batch, so the GPU cost stays one extra
pass per burst, not per sub-goal. The decomposition is the cheap part:
the search escapes the rollout entropy shell because each candidate is
conditioned on an explicit plan instead of the same parent+direction
context every forward proposer shares.

Call contract (same as LLMProposer/Mutator, so CodeEvolve consumes it
transparently):

    prop(prompt) -> raw text            # unprimed prompt -> fenced code
    prop.propose_batch(prompts) -> raws # burst path, same contract
    prop.note(code, credit, delta, prompt=...)  # outcome feedback
    prop.primed -> False                # instruction-style prompts
    prop.last_adapter                   # LoRA provenance passthrough

Design choices:
  - Wraps an LLMProposer (lazy-constructed or injected) so the engine
    cache, LoRA router, hint bandit, SEARCH tools, and HF/forge backend
    dispatch all stay exactly once — BES is a prompt-pipeline layer, not
    a second model.
  - ``primed=False``: the solve pass needs a COMPLETE fenced program
    (sub-goals span the whole file), not a signature continuation.
  - Non-proposal prompts (judge/critique/repair-shaped prose that does
    not open with the evolve header) pass through as a single forward
    call — decomposing a critique request would just waste a pass.
  - Decompose failure (empty/garbage list, engine error) degrades to the
    plain forward call — BES never emits fewer proposals than "llm".
"""
from __future__ import annotations

import re

# Numbered sub-goal line: "1. x", "2) x", "# 3. x", "-4. x" — the model
# occasionally answers in the evolve prompt's own comment style. [ \t]
# (not \s) everywhere: a bare "2." line must NOT swallow the next line's
# text as its item.
_SG_RE = re.compile(
    r"(?m)^[ \t]*#?[ \t]*-?[ \t]*(\d{1,2})[ \t]*[.)\]:-][ \t]*"
    r"(\S[^\n]*)")

# The evolve prompt header (_build_prompt line 1). Prompts that lack it
# are judge/critique calls — BES must not decompose those.
_PROPOSAL_MARK = "# Rewrite this Python program"

# Trailing reply-contract line _build_prompt appends when primed=False.
_REPLY_MARK = "Reply with ONLY"


class BESProposer:
    """Backward goal-decomposition + forward joint-solve proposer.

    See module docstring. ``llm`` injection point exists for tests and
    for callers that already hold a live LLMProposer; otherwise one is
    constructed lazily from cfg (shared engine table — no extra VRAM).
    """

    primed = False                     # _build_prompt -> unprimed tail
    MAX_SUBGOALS = 8

    def __init__(self, cfg, db=None, tag_sig: str = "", llm=None) -> None:
        self.cfg = cfg
        self.db = db
        self.tag_sig = tag_sig
        if llm is None:
            # Lazy import: code_evolve imports BESProposer for the
            # dispatch table — a top-level import here would cycle.
            from .code_evolve import LLMProposer
            llm = LLMProposer(cfg, db=db, tag_sig=tag_sig)
        self._llm = llm
        self.max_subgoals = max(
            1, int(getattr(cfg, "bes_subgoals", 0) or self.MAX_SUBGOALS))
        # orig evolve prompt -> the solve prompt actually sent to the
        # LLM. note() maps outcome credit back through it so the hint
        # bandit attributes wins to the SOLVE pass (which emitted the
        # code), not the decompose pass.
        self._solve_of: dict[str, str] = {}

    # ── pass 1: backward goal decomposition ──────────────────────────
    def _is_proposal(self, prompt: str) -> bool:
        """Only evolve proposal prompts get the two-phase pipeline."""
        return _PROPOSAL_MARK in prompt[:400]

    def _decompose_prompt(self, prompt: str) -> str:
        """Same task context, different output contract: the trailing
        "Reply with ONLY one ```python block" line is replaced by a
        bounded numbered-list contract (the "Sub-goals:" tail primes the
        list the way the primed fence primes code continuations)."""
        base = re.sub(r"(?m)^Reply with ONLY[^\n]*\n?", "", prompt)
        return (
            base.rstrip()
            + "\n# BACKWARD PASS — decompose the goal, do NOT write code."
            "\n# Break the optimization goal above into the concrete"
            "\n# sub-goals the improved program must satisfy: the public"
            "\n# contract to keep, the algorithmic steps, the edge cases,"
            "\n# and the performance property that wins the metric."
            "\n# Work backward from the goal — the next pass solves them"
            "\n# jointly. Reply with ONLY a numbered list of at most"
            f"\n# {self.max_subgoals} sub-goals, one per line, no code,"
            "\n# no fences."
            "\nSub-goals:\n")

    def _parse_subgoals(self, raw: str) -> list[str]:
        """Numbered-list extraction — same cleaning order as
        _extract_code (think regions out first, then fenced code which
        can never be a sub-goal anyway), then numbered lines, deduped,
        bounded to max_subgoals."""
        if not raw:
            return []
        work = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        work = re.sub(r"</?think>", "", work)
        work = re.sub(r"```[^\n`]*\n?.*?```", "", work, flags=re.S)
        # Unclosed fence -> the model leaked into code; drop the tail.
        work = work.split("```", 1)[0]
        goals: list[str] = []
        for _num, item in _SG_RE.findall(work):
            g = " ".join(item.split()).rstrip(".;")
            # Trivia guard: empty, one-char, or pure-code "goals" add
            # noise to the solve context.
            if len(g) < 4:
                continue
            if g not in goals:
                goals.append(g[:200])
            if len(goals) >= self.max_subgoals:
                break
        return goals

    # ── pass 2: forward joint solve ──────────────────────────────────
    def _inject_subgoals(self, prompt: str, subgoals: list[str]) -> str:
        """Sub-goal block goes BEFORE the final reply-contract line —
        same insertion rule as _with_hint/_queue_repair (conditions in
        the middle, the output contract last)."""
        block = (
            "# Backward decomposition — the improved program must"
            "\n# satisfy EVERY sub-goal jointly (solve them in one"
            "\n# pass, not as separate functions):"
            + "".join(f"\n#   {i}. {g}"
                      for i, g in enumerate(subgoals, 1)))
        i = prompt.rfind(_REPLY_MARK)
        if i >= 0:
            return prompt[:i] + block + "\n" + prompt[i:]
        return prompt + "\n" + block

    def _record_solve(self, orig: str, solve: str) -> None:
        self._solve_of[orig] = solve
        if len(self._solve_of) > 512:          # same FIFO cap as hints
            self._solve_of.pop(next(iter(self._solve_of)))

    def _batch(self, prompts: list[str]) -> list[str]:
        """LLM pass: propose_batch when supported (one GPU pass),
        serial calls otherwise — mirrors MixProposer's fallback."""
        if not prompts:
            return []
        fn = getattr(self._llm, "propose_batch", None)
        if fn is not None:
            try:
                return fn(prompts)
            except Exception:
                pass
        return [self._llm(p) for p in prompts]

    # ── proposer contract ────────────────────────────────────────────
    def __call__(self, prompt: str) -> str:
        if not self._is_proposal(prompt):
            return self._llm(prompt)           # judge/critique: 1 pass
        subgoals: list[str] = []
        try:
            subgoals = self._parse_subgoals(
                self._llm(self._decompose_prompt(prompt)))
        except Exception:
            subgoals = []
        if not subgoals:
            return self._llm(prompt)           # forward fallback
        solve = self._inject_subgoals(prompt, subgoals)
        self._record_solve(prompt, solve)
        return self._llm(solve)

    def propose_batch(self, prompts: list[str]) -> list[str]:
        """Two batched passes: one decompose batch for all proposal
        prompts, then one solve batch. Non-proposal rows and failed
        decompositions ride the solve batch with their prompt unchanged
        — identical cost profile to 'llm' plus one shared decompose
        pass."""
        n = len(prompts)
        if n == 0:
            return []
        sg: dict[int, list[str]] = {}
        todo = [i for i, p in enumerate(prompts)
                if self._is_proposal(p)]
        if todo:
            try:
                raws = self._batch(
                    [self._decompose_prompt(prompts[i]) for i in todo])
                for i, r in zip(todo, raws):
                    g = self._parse_subgoals(r or "")
                    if g:
                        sg[i] = g
            except Exception:
                sg = {}
        solve: list[str] = []
        for i, p in enumerate(prompts):
            sp = self._inject_subgoals(p, sg[i]) if i in sg else p
            self._record_solve(p, sp)
            solve.append(sp)
        return self._batch(solve)

    def note(self, code: str, credit: float, delta: float,
             prompt: str | None = None) -> None:
        """Outcome -> the wrapped LLM's hint bandit. `prompt` arrives as
        the ORIGINAL evolve prompt; the bandit keyed its hint under the
        SOLVE prompt, so map through _solve_of (the decompose pass's
        hint is unrecorded noise — it emitted no code)."""
        if prompt is not None:
            prompt = self._solve_of.pop(prompt, prompt)
        n = getattr(self._llm, "note", None)
        if n is None:
            return
        try:
            n(code, credit, delta, prompt=prompt)
        except TypeError:
            try:
                n(code, credit, delta)
            except Exception:
                pass
        except Exception:
            pass

    @property
    def last_adapter(self) -> str | None:
        """LoRA provenance for code_programs.adapter — the tag the inner
        proposer routed the last (solve) generation through."""
        return getattr(self._llm, "_last_adapter", None)

    @property
    def _lfm3(self) -> bool:
        """Prompt-shape flag passthrough — _build_prompt reads it off
        the TOP-LEVEL proposer to add the verbatim-code clause."""
        return bool(getattr(self._llm, "_lfm3", False))
