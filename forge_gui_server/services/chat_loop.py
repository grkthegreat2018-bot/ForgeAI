"""Chat generation loop — async port of pages/chat.py::_ChatWorker.

Streams raw model chunks to the SSE consumer (the React client does the
<think>/<tool_call> marker splitting for display, same role the Qt
_StreamParser played). After each generation the accumulated text is
parsed for tool calls; calls execute via ToolHarness and the loop
continues, mirroring the Qt worker's max_rounds=6 behavior.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

import torch

logger = logging.getLogger(__name__)

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat")

# Matches the Qt chat worker's hardcoded stop ids (im_end + eos for the
# Jamba-derived tokenizer).
_EOS_IDS = [2, 519]

# Jamba structural token ids — when tools are enabled, generation must
# stop at the end of a <tool_call> block (532) or the start of a
# <tool_response> block (539). Without these stops the model continues
# past its call and hallucinates the tool's output itself.
_TOOL_CALL_END_ID = 532       # </tool_call>
_TOOL_RESP_START_ID = 539     # <tool_response>
_TOOL_CALL_START = "<tool_call>"
_RE_NAME_HINT = re.compile(r'\{"name"\s*:\s*"')

# ForgeGate route threshold — P(direct-safe) >= this routes the turn to a
# closed-think direct answer instead of an open <think> reasoning pass.
# Calibrated on chat prompts via scripts/bench_gate_route.py: gate_r's
# 0.5 almost never fires on real chat turns (easy cluster ~0.22-0.55,
# reasoning cluster ~0.02-0.12), so 0.2 keeps the feature alive.
_GATE_T_ROUTE = 0.2

# Whole-message greeting/ack/closing matcher for the pending user turn.
# These prompts are out-of-distribution for the route probe — the gate_r
# corpus has no chit-chat class, so "hey"/"thanks!" score under the
# threshold even on a clean render — and there is objectively nothing to
# reason about, so they get a deterministic direct route.
_TRIVIAL_RE = re.compile(
    r"^(?:hello+|hi+|hey+|yo|hiya|howdy|sup|wassup|what'?s up|"
    r"good\s+(?:morning|afternoon|evening|night)|greetings|"
    r"thanks?|thank\s+you|thx|ty|cheers|"
    r"ok(?:ay)?|k+|sure|yes|yeah|yep|yup|no|nope|"
    r"bye+|good\s*bye|see\s+(?:ya|you)|later|"
    r"nice|great|cool|awesome|perfect|got\s+it|understood|"
    r"lol|haha+|lmao|lmfao)"
    r"[\s.!?~]*$", re.IGNORECASE)

_TRIVIAL_MAX_CHARS = 48


def _is_trivial_turn(conv: list[dict]) -> bool:
    """True when the pending user turn is a bare greeting/ack/closing."""
    if not conv or conv[-1].get("role") != "user":
        return False
    c = (conv[-1].get("content") or "").strip()
    return 0 < len(c) <= _TRIVIAL_MAX_CHARS and bool(_TRIVIAL_RE.match(c))

# Appended to a thinking=False render when the gate routes to direct:
# empty closed think + the "Answer:" anchor — ForgeGate's validated
# direct-mode tail (gated.py). The anchor matters: a bare </think> leaves
# the model musing in think-voice instead of answering.
_DIRECT_SUFFIX = "<think>\n</think>\nAnswer:"

# Token-injected when the think budget fires — gated.py's FORCE_SUFFIX.
_FORCE_ANSWER_SUFFIX = "\n</think>\nAnswer:"

_THINK_START_ID = 541         # <think>  (Jamba structural token)
_THINK_END_ID = 542           # </think>
_THINK_END_STR = "</think>"

_IM_START_ID = 518            # <|im_start|> — never emitted mid-generation
_TOOL_CALL_START_ID = 531     # <tool_call>
_TOOL_RESP_END_ID = 540       # </tool_response>
# Structural ids the model must never emit in generated text: 518 would
# fake a new turn boundary, 539/540 would fake a tool result. (<tool_call>
# is handled separately — legal, but only after </think>.)
_BANNED_IDS = (_IM_START_ID, _TOOL_RESP_START_ID, _TOOL_RESP_END_ID)

# Literal-marker stop strings — belt-and-suspenders for the case where the
# model emits a marker as ordinary text tokens instead of the special id
# (id-based EOS never fires, and the model would hallucinate the tool
# result itself). Checked on a rolling tail of decoded text.
_TEXT_STOP_TAIL = 24
_TEXT_STOPS = ("</tool_call>", "<tool_response>")

# Fed-back tool results are capped so a big page/file can't drown the
# synthesis turn's context.
_TOOL_RESULT_MAX_CHARS = 4000

# Hard think budget — when the model takes the think path, inject the
# force-answer suffix after this many generated think tokens so a
# reasoning pass can't run away on trivial prompts. This is the BACKSTOP;
# the conv probe below usually exits earlier.
_THINK_MAX_TOKENS = 160

# Gate C convergence probe — learned early exit for think mode. The conv
# head scores P(force-answer-now is correct) on per-step hidden states
# (feature: cat(h_last, min(step/budget, 1)) — same as gated.py's
# monitored loop). _CONV_K consecutive scores above _CONV_T, no earlier
# than _CONV_MIN_STEP tokens in, inject the force-answer suffix.
_CONV_T = 0.75
_CONV_K = 2
_CONV_MIN_STEP = 32

# Special tokens that must never be persisted in stored message content —
# they are structural markers, not user-visible text.
_SPECIAL_TOKENS = (
    "<|im_start|>", "<|im_end|>", "<|startoftext|>", "<|endoftext|>",
    "<tool_response>", "</tool_response>",
)


def _clean_content(text: str) -> str:
    for t in _SPECIAL_TOKENS:
        text = text.replace(t, "")
    return text


def _route_p_easy(engine, rendered: str) -> float | None:
    """Score the ForgeGate route probe on a rendered chat prompt.

    One uncached forward over the prompt (hidden states only); returns
    P(direct-safe), or None when the probe bundle isn't loaded — chat then
    falls back to the caller's thinking flag unchanged.
    """
    probes = getattr(engine, "_gate_probes", None)
    if probes is None:
        return None
    try:
        ids = engine._tokenize(rendered)
        with engine._gen_lock, torch.inference_mode():
            out = engine.model(ids, use_cache=False, return_hidden=True)
        h = out[-1][0].float()
        return probes.score("route", torch.cat([h[-1], h.mean(0)]))
    except Exception:
        logger.debug("gate route probe failed", exc_info=True)
        return None


def _conv_exit_observer(probes, budget: int, emit=None):
    """Per-step hidden-state observer driving Gate C early exit in chat.

    Returns ``(observer, fired)``: ``observer(h, generated_ids)`` is fed
    each decode step's last-token hidden by ``generate_stream`` and flips
    the shared state once the conv probe clears the threshold _CONV_K
    times running (past _CONV_MIN_STEP). Stops scoring after ``</think>``
    appears — post-think tokens are the answer, not reasoning.
    """
    state = {"fired": False, "run": 0}

    def observer(h, generated_ids):
        if state["fired"] or _THINK_END_ID in generated_ids:
            return
        step = len(generated_ids)
        if step < _CONV_MIN_STEP:
            return
        feat = torch.cat([h.float(), torch.tensor(
            [min(step / max(budget, 1), 1.0)], device=h.device)])
        s = probes.score("conv", feat)
        state["run"] = state["run"] + 1 if s > _CONV_T else 0
        if state["run"] >= _CONV_K:
            state["fired"] = True
            if emit is not None:
                emit(("gate", {"p_easy": round(s, 3),
                               "mode": "conv-exit"}))

    def fired() -> bool:
        return state["fired"]

    return observer, fired


def _call_sig(call: dict) -> tuple:
    """(name, canonical-args) signature for repeat-call detection."""
    args = call.get("arguments") or call.get("args") or {}
    try:
        return (str(call.get("name", "")),
                json.dumps(args, sort_keys=True, default=str))
    except Exception:
        return (str(call.get("name", "")), str(args))


def _strip_direct_musing(content: str) -> str:
    """Direct-mode (closed-think) answers sometimes still open with
    think-voice musing terminated by a stray ``</think>`` — the model
    re-enacts the think block in plain text (the documented "direct mode
    emits ~96tok derivations" limitation). When the marker is present the
    real answer is what follows it."""
    if _THINK_END_STR in content:
        tail = content.rsplit(_THINK_END_STR, 1)[-1].lstrip("\n")
        if tail.strip():
            return tail
    return content


def _think_cap_processor(budget: int | None, suffix_ids: list[int],
                         exit_flag=None, tool_calls_allowed: bool = True,
                         track_think: bool | None = None):
    """Per-step logits guard for chat generation.

    - ``track_think`` (default ``budget is not None``): the prompt left a
      ``<think>`` block open. While it stays open (no ``</think>``
      generated), ``<tool_call>`` (531) is masked so a call can only start
      after the reasoning pass closes — the trained Jamba order is
      think → ``</think>`` → text → ``<tool_call>``. This is the fix for
      thinking escaping into tool-call turns: a mid-think call used to
      leave the block unclosed, so the reasoning ended up persisted and
      displayed as the visible reply.
    - ``budget``: after `budget` generated think tokens without a
      ``</think>``, inject ``suffix_ids`` — the force-answer tail
      ``"\\n</think>\\nAnswer:"`` — one forced token per step. The
      "Answer:" anchor flips the model into answer voice; a bare </think>
      leaves it musing. ``budget=None`` (direct path — the prompt already
      carries a closed think) disables the budget check.
    - ``exit_flag``: optional callable -> bool (the conv-probe observer's
      ``fired``) that triggers the same suffix injection before the
      budget — the learned convergence exit.
    - ``tool_calls_allowed``: False (tools disabled, or defs dropped by
      the repeat-call guard) bans ``<tool_call>``/``</tool_call>``
      outright — the model can't spend tokens on a call that will never
      execute, and stray markers can't leak into the visible reply.
    - Always banned in generated text: ``<think>`` (541 — a generated one
      can only re-open a reasoning pass; observed on real output: after
      the cap fired the model emitted `Answer:` then re-opened <think>
      and started reasoning all over again), ``<|im_start|>`` (518 — fake
      turn boundary), ``<tool_response>``/``</tool_response>`` (539/540 —
      fake tool result; 539 used to be an EOS that truncated the turn).
    """
    queue: list[int] = []
    track = (budget is not None) if track_think is None else track_think

    def proc(logits, generated_ids):
        think_open = track and _THINK_END_ID not in generated_ids
        forced = ((budget is not None and len(generated_ids) >= budget)
                  or (exit_flag is not None and exit_flag()))
        if think_open and not queue and forced and suffix_ids:
            queue.extend(suffix_ids)
        if queue:
            tid = queue.pop(0)
            masked = torch.full_like(logits, float("-inf"))
            masked[..., tid] = logits[..., tid]
            return masked
        masked = logits.clone()
        for tid in _BANNED_IDS + (_THINK_START_ID,):
            masked[..., tid] = float("-inf")
        if think_open or not tool_calls_allowed:
            masked[..., _TOOL_CALL_START_ID] = float("-inf")
        if not tool_calls_allowed:
            masked[..., _TOOL_CALL_END_ID] = float("-inf")
        return masked
    return proc


def _split_reasoning(content: str) -> tuple[str, str]:
    """Split a think-path completion into ``(reasoning, visible_body)``.

    The prompt opens the ``<think>`` block, so raw output starts in
    reasoning voice. A well-formed completion contains ``</think>``: text
    before it is reasoning, text after is the answer. When the marker
    never arrived — max-token truncation, an EOS inside the block, or a
    bare-JSON tool call that bypassed the 531 mask — the whole completion
    is reasoning and the visible body is empty (it must NOT be persisted
    as the reply — that was the "thinking escapes" bug).
    """
    if _THINK_END_STR in content:
        reasoning, _, body = content.partition(_THINK_END_STR)
        # stray extra closers in the body are noise — drop them
        return (reasoning.strip(),
                body.replace(_THINK_END_STR, "").strip())
    return content.strip(), ""


_ANSWER_ANCHOR_RE = re.compile(
    r"^\s*(?:final\s+)?answer\s*[:：]\s*", re.IGNORECASE)


def _strip_answer_anchor(text: str) -> str:
    """Drop a leading ``Answer:`` echo — the injected force-answer suffix
    ends with the anchor and the model often repeats it verbatim."""
    if not text:
        return text
    stripped = _ANSWER_ANCHOR_RE.sub("", text, count=1)
    return stripped if stripped.strip() else text


class ChatCancel:
    def __init__(self) -> None:
        self._flag = threading.Event()

    def cancel(self) -> None:
        self._flag.set()

    @property
    def cancelled(self) -> bool:
        return self._flag.is_set()


async def run_chat(runtime, harness, messages: list[dict],
                   config_name: str = "", max_new_tokens: int = 512,
                   temperature: float = 0.7, top_p: float = 0.95,
                   top_k: int = 80, repetition_penalty: float = 1.05,
                   tools_enabled: bool = True, thinking: bool = True,
                   min_p: float = 0.0, dry_multiplier: float = 0.0,
                   dry_base: float = 1.75,
                   max_rounds: int = 6,
                   cancel: ChatCancel | None = None,
                   think_budget: int | None = None,
                   ) -> AsyncIterator[dict]:
    """Yield chat events: gate / token / tool_call / tool_result / done /
    error."""
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancel = cancel or ChatCancel()

    def emit(item):
        loop.call_soon_threadsafe(q.put_nowait, item)

    def worker():
        from forge.self_play.discovery.qwen_adapter import (
            qwen_parse_tool_calls, render_messages_for_config)
        try:
            if not runtime.is_ready():
                emit(("error", "No model resident — load one on the "
                      "Models page first."))
                return
            defs = (harness.chat_tool_defs()
                    if (harness and tools_enabled) else None)
            rounds = max_rounds if tools_enabled else 1
            conv = list(messages)
            new_messages: list[dict] = []
            seen_sigs: set = set()

            for _round in range(rounds):
                if cancel.cancelled:
                    break
                rendered = render_messages_for_config(
                    conv, config_name=config_name, tools=defs,
                    add_generation_prompt=True, thinking=thinking)
                chunks: list[str] = []
                eos_ids = (_EOS_IDS + [_TOOL_CALL_END_ID, _TOOL_RESP_START_ID]
                           if defs else _EOS_IDS)
                with runtime.acquire(timeout_s=120.0) as engine:
                    # Route/exit gating applies to fresh user turns only —
                    # tool-continuation rounds (conv ends with a tool
                    # result) go straight to think + budget backstop: both
                    # probes were trained on user questions and are OOD on
                    # synthesis turns (observed: conv-exit fired ~35 tok
                    # in, the forced "Answer:" restated the plan, and the
                    # model re-called the same tool 5x).
                    fresh_turn = (bool(conv)
                                  and conv[-1].get("role") == "user")
                    think_active = thinking
                    if thinking and fresh_turn:
                        # ForgeGate route probe: easy turns get a closed
                        # <think> block so the model answers directly instead
                        # of burning tokens on a reasoning pass.
                        # Score a tools-free render — gate_r was harvested
                        # without a <tools> block, and the ~20-schema block
                        # the production render injects collapses h_mean
                        # and drags borderline prompts under the threshold
                        # (measured on V2: "Hello" 0.355 tools-free vs
                        # 0.098 with tools). Generation still uses the
                        # tools render; only the probe input is stripped.
                        if _is_trivial_turn(conv):
                            p_easy, direct = 1.0, True
                        else:
                            probe_rendered = rendered if defs is None else \
                                render_messages_for_config(
                                    conv, config_name=config_name,
                                    tools=None, add_generation_prompt=True,
                                    thinking=True)
                            p_easy = _route_p_easy(engine, probe_rendered)
                            direct = (p_easy is not None
                                      and p_easy >= _GATE_T_ROUTE)
                        if p_easy is not None:
                            emit(("gate", {"p_easy": round(p_easy, 3),
                                           "mode": "direct" if direct
                                           else "think"}))
                        if direct:
                            rendered = render_messages_for_config(
                                conv, config_name=config_name, tools=defs,
                                add_generation_prompt=True,
                                thinking=False)
                            rendered += _DIRECT_SUFFIX
                            think_active = False
                    # budget active only on the open-think path; the
                    # <think> re-open ban applies to every mode
                    budget = _THINK_MAX_TOKENS
                    if think_budget is not None:
                        # 0 = no forced exit (bounded by max_new_tokens);
                        # otherwise clamp to a sane range.
                        budget = (max_new_tokens if think_budget <= 0
                                  else max(8, min(think_budget, 4096)))
                    # Gate C: when probes are loaded and we're thinking,
                    # attach the conv-probe observer — learned early exit,
                    # with the token budget as backstop.
                    hidden_obs, exit_flag = None, None
                    probes = getattr(engine, "_gate_probes", None)
                    if think_active and fresh_turn and probes is not None:
                        hidden_obs, exit_flag = _conv_exit_observer(
                            probes, budget, emit)
                    cap = _think_cap_processor(
                        budget if think_active else None,
                        engine.tokenizer.encode(
                            _FORCE_ANSWER_SUFFIX,
                            add_special_tokens=False),
                        exit_flag=exit_flag,
                        tool_calls_allowed=defs is not None)
                    tail = ""
                    for tok in engine.generate_stream(
                            rendered, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            skip_special_tokens=False,
                            eos_token_ids=eos_ids,
                            logits_processor=cap,
                            min_p=min_p, dry_multiplier=dry_multiplier,
                            dry_base=dry_base,
                            hidden_observer=hidden_obs):
                        if cancel.cancelled:
                            break
                        chunks.append(tok)
                        emit(("token", tok))
                        # Literal-marker stop: catches markers emitted as
                        # ordinary text tokens (the id-based EOS only sees
                        # the special ids). Rolling tail keeps this O(1).
                        tail = (tail + tok)[-_TEXT_STOP_TAIL:]
                        if any(s in tail for s in _TEXT_STOPS):
                            break
                raw = "".join(chunks)
                tool_calls, content = qwen_parse_tool_calls(raw)
                if tool_calls:
                    # anything the model emitted after its first call is a
                    # hallucinated continuation (a simulated tool response)
                    # — keep only the musing that precedes the call
                    cut = raw.find(_TOOL_CALL_START)
                    if cut < 0:
                        m = _RE_NAME_HINT.search(raw)
                        cut = m.start() if m else -1
                    if cut >= 0:
                        content = raw[:cut]
                if not think_active:
                    if tool_calls:
                        # pre-call text on a closed-think turn is
                        # reasoning voice — the tool call speaks for
                        # itself, don't persist it as a "reply"
                        content = ""
                    else:
                        # closed-think renders can still open with
                        # think-voice musing followed by a stray
                        # </think> — the real answer is the tail.
                        content = _strip_direct_musing(content or "")
                elif tool_calls and defs is None:
                    # tools were dropped by the repeat guard — treat a
                    # stray call as plain text and end the turn
                    tool_calls = None
                reasoning = ""
                if think_active:
                    # Interleaved-thinking contract: reasoning rides along
                    # as reasoning_content (re-rendered by the template on
                    # the next tool round, dropped once a new user turn
                    # starts), never as visible body text.
                    reasoning, content = _split_reasoning(content or "")
                content = _strip_answer_anchor(content or "")
                # A literal-marker stop can leave a few chars of a faked
                # tool response in the tail — cut at the marker itself so
                # the payload never persists as visible text.
                for marker in ("<tool_response>", "</tool_response>",
                               "</tool_call>", "<tool_call>"):
                    idx = content.find(marker)
                    if idx >= 0:
                        content = content[:idx]
                msg = {"role": "assistant", "content": _clean_content(content or ""),
                       "tool_calls": tool_calls or None}
                if reasoning:
                    msg["reasoning_content"] = _clean_content(reasoning)
                conv.append(msg)
                new_messages.append(msg)

                if not tool_calls or not harness:
                    break

                # Repeat-call guard: if every call this round duplicates
                # one already made (same name + args), the model is looping
                # on its plan instead of reading results — drop tool defs
                # so the next round must synthesize an answer.
                sigs = {_call_sig(c if isinstance(c, dict)
                                  else {"name": str(c)})
                        for c in tool_calls}
                if sigs <= seen_sigs:
                    defs = None
                seen_sigs |= sigs

                for tc in tool_calls:
                    if cancel.cancelled:
                        break
                    call = tc if isinstance(tc, dict) else {"name": str(tc)}
                    emit(("tool_call", call))
                    rec = harness.execute_calls([call])[0]
                    emit(("tool_result", rec))
                    from forge_gui.api.agent_tools import (
                        tool_results_to_text)
                    ttext = tool_results_to_text(rec)
                    if len(ttext) > _TOOL_RESULT_MAX_CHARS:
                        ttext = (ttext[:_TOOL_RESULT_MAX_CHARS]
                                 + "...[truncated]")
                    tmsg = {"role": "tool",
                            "name": call.get("name", "tool"),
                            "content": ttext}
                    conv.append(tmsg)
                    new_messages.append(tmsg)

            emit(("done", {"messages": new_messages}))
        except Exception as e:
            logger.warning("chat loop failed: %s", e, exc_info=True)
            emit(("error", f"{type(e).__name__}: {e}"))
        finally:
            emit(("end", None))

    fut = loop.run_in_executor(_pool, worker)
    try:
        while True:
            kind, data = await q.get()
            if kind == "end":
                break
            yield {"type": kind, "data": data}
    finally:
        # Consumer gone — stop the generation loop + any in-flight tool
        # execution instead of orphaning the worker for a dead client.
        cancel.cancel()
        fut.cancel()
