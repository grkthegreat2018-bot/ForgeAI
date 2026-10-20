"""Jamba-format tool-call adapter for the discovery self-play loop.

ForgeLM V2 uses Jamba Reasoning 3B as parent, with ChatML format and
JSON tool calls wrapped in Jamba's native special tokens (ids 531/532):
  <|im_start|>assistant
  <tool_call>
  {"name": "tool_name", "arguments": {...}}
  </tool_call>
  <|im_end|>

Jamba also supports thinking mode with <think>/</think> tags (ids 541/542)
and <tool_response>/</tool_response> result markers (ids 539/540).

This module provides:
  - qwen_render_messages: render a conversation in canonical Jamba ChatML
    (delegates to chat_template.apply_chat_template — a port of the
    tokenizer's built-in Jinja template)
  - qwen_parse_tool_calls: parse JSON tool calls from model output
  - QwenDiscoveryAdapter: wraps the discovery loop to use Jamba format

Legacy LFM2.5 format (Pythonic tool calls, ids 10/11) is still supported
for backward compatibility in the parser only.
"""
from __future__ import annotations

import json
import re
from typing import Any

import torch

from .chat_template import (  # noqa: F401  (re-exported constants)
    TOOL_RESP_END,
    TOOL_RESP_START,
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# Jamba Reasoning 3B token IDs
EOS_IDS = (2, 519)  #  and <|im_end|>
BOS_ID = 1  # <|startoftext|>

# Jamba tool call markers (single tokens: 531/532)
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_CALL_START_ID = 531
TOOL_CALL_END_ID = 532

# Aliases for the constrained-decoding helpers below. In the legacy LFM2.5
# tokenizer these were the first BPE ids of the <|tool_call_start|> text
# markers (10/11) and EOS_ID was <|im_end|> (7). With the Jamba tokenizer
# the markers are single special tokens, so the aliases point at 531/532
# and <|im_end|> (519).
TOOL_CALL_START_FIRST_ID = TOOL_CALL_START_ID
TOOL_CALL_END_FIRST_ID = TOOL_CALL_END_ID
EOS_ID = EOS_IDS[1]  # <|im_end|>

# Jamba tool result markers (single tokens: 539/540). Single source of
# truth lives in chat_template — TOOL_RESP_START/END above alias it.


def qwen_render_messages(messages: list[dict],
                         tools: list[dict] | None = None,
                         add_generation_prompt: bool = True,
                         thinking: bool = True) -> str:
    """Render a conversation in canonical Jamba ChatML format.

    Delegates to ``chat_template.apply_chat_template`` — a faithful port of
    the tokenizer's built-in Jinja chat_template. ``bos=False`` because the
    engine tokenizes prompts with add_special_tokens=True (BOS auto-added).

    Messages: [{"role": "system"|"user"|"assistant"|"tool", "content": str,
                "tool_calls": [...], "reasoning_content": str, "name": str}]
    """
    from .chat_template import apply_chat_template
    return apply_chat_template(
        messages, tools=tools, add_generation_prompt=add_generation_prompt,
        bos=False, thinking=thinking)


# Back-compat alias — the fake <|startofsegment|> format was removed; Jamba
# is ChatML. Kept so stale callers still get the canonical renderer.
def jamba_render_messages(messages: list[dict],
                          tools: list[dict] | None = None,
                          add_generation_prompt: bool = True) -> str:
    """Deprecated alias for qwen_render_messages (canonical Jamba ChatML)."""
    return qwen_render_messages(messages, tools, add_generation_prompt)


def render_messages_for_config(messages: list[dict],
                               config_name: str | None = None,
                               tools: list[dict] | None = None,
                               add_generation_prompt: bool = True,
                               thinking: bool = True) -> str:
    """Render messages in the canonical ChatML format for a config preset.

    ForgeLM V2-family configs (forgelm_v2, forgelm_v12_jamba) share the
    Jamba-Reasoning-3B chat_template — including its trained thinking
    prefix + open ``<think>`` block in the generation prompt.

    Llama/Hermes-family presets (``llama31_8b``) use plain ChatML — the
    same im_start/im_end skeleton but NO thinking prefix and NO open
    think block (Hermes-3 was not trained on Jamba's thinking protocol).
    """
    if config_name and config_name.startswith("llama"):
        thinking = False
    return qwen_render_messages(messages, tools, add_generation_prompt,
                                thinking=thinking)


# ── Tool-call parsing ─────────────────────────────────────────────────────

# The training format uses the native special tokens as tool call markers:
#   {start}\n{"name": "...", "arguments": {...}}\n{end}
# We parse these markers first, then fall back to bare JSON detection.

_RE_NAME_HINT = re.compile(r'\{"name"\s*:\s*"([^"]+)"')


def _find_balanced_json(text: str, start: int) -> str | None:
    """Return the substring of the first balanced {...} starting at start.

    Handles nested braces and strings with escaped quotes.
    """
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def qwen_parse_tool_calls(text: str) -> tuple[list[dict] | None, str]:
    """Parse tool calls from model output.

    Returns (tool_calls | None, musing_text).

    Primary format (training): {start}\n{json}\n{end}
    Fallback: bare {"name": ...} JSON objects
    Fallback: ForgeLM V10 Pythonic format
    """
    calls = []
    consumed_spans = []  # (start, end) ranges of parsed content

    # Phase 1: Parse start-marker ... end-marker wrapped tool calls
    pos = 0
    while True:
        start_idx = text.find(TOOL_CALL_START, pos)
        if start_idx == -1:
            break
        end_idx = text.find(TOOL_CALL_END, start_idx + 1)
        if end_idx == -1:
            # No end marker — try to parse from start to end of text
            end_idx = len(text)
        # Extract JSON between markers
        json_region = text[start_idx + len(TOOL_CALL_START):end_idx].strip()
        raw = _find_balanced_json(json_region, 0)
        if raw:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    import json_repair
                    obj = json_repair.loads(raw)
                except Exception:
                    obj = None
            if isinstance(obj, dict) and "name" in obj:
                name = obj["name"]
                args = obj.get("arguments", obj.get("args", {}))
                if not isinstance(args, dict):
                    args = {}
                calls.append({"name": name, "args": args, "arguments": args})
                consumed_spans.append((start_idx, min(end_idx + len(TOOL_CALL_END), len(text))))
        pos = end_idx + 1

    # Phase 2: If no marker-wrapped calls found, try bare JSON
    if not calls:
        for hint in _RE_NAME_HINT.finditer(text):
            brace_start = hint.start()
            if any(s <= brace_start < e for s, e in consumed_spans):
                continue
            raw = _find_balanced_json(text, brace_start)
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    import json_repair
                    obj = json_repair.loads(raw)
                except Exception:
                    continue
            if isinstance(obj, dict) and "name" in obj:
                name = obj["name"]
                args = obj.get("arguments", obj.get("args", {}))
                if not isinstance(args, dict):
                    args = {}
                calls.append({"name": name, "args": args, "arguments": args})
                consumed_spans.append((brace_start, brace_start + len(raw)))

    if calls:
        # Build musing from text outside consumed spans
        musing_parts = []
        last = 0
        for s, e in sorted(consumed_spans):
            if s > last:
                musing_parts.append(text[last:s].strip())
            last = e
        if last < len(text):
            musing_parts.append(text[last:].strip())
        musing = " ".join(musing_parts).strip()
        return calls, musing

    # Fallback: try ForgeLM V10 Pythonic format
    from forge.self_play.discovery.chat_template import parse_tool_calls as _lfm_parse
    return _lfm_parse(text)


def qwen_generate(model, tokenizer, prompt: str, max_new_tokens: int = 256,
                  temperature: float = 0.2, device: str = "cuda",
                  top_k: int = 80, repetition_penalty: float = 1.05,
                  grammar_matcher=None, bitmask=None) -> str:
    """Generate text from the model, stopping at <|im_end|> (EOS_ID=7).

    Returns only the generated tokens (not the prompt), preserving special
    tokens so tool-call JSON is visible to the parser.

    Uses ForgeLM V10-recommended generation defaults:
      - temperature: 0.2 (low randomness for reliable tool calls)
      - top_k: 80 (limits sampling to top-k logits)
      - repetition_penalty: 1.05 (discourages token repetition)

    If grammar_matcher is provided, uses two-phase constrained decoding:
    Phase 1: Generate freely (for musings/reasoning text).
    Phase 2: When the model emits the start marker (token 10), switch to
             constrained mode — only valid JSON with correct tool names allowed.
    Structural tokens (end marker, newlines) are skipped in constrained mode.
    This prevents hallucinated tool names while allowing free-form reasoning.
    """
    import torch
    import torch.nn.functional as F

    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
    prompt_len = ids.shape[1]

    # Constrained decoding: trigger on the start marker (token 531).
    # In constrained mode, skip the end marker (token 532) and newline tokens
    # (structural tokens between markers and JSON, not part of the JSON grammar).
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))
    skip_token_ids = {TOOL_CALL_END_FIRST_ID} | newline_ids

    constrained = False  # start in free-text mode
    generated_ids: list[int] = []  # track generated token ids for repetition penalty

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(ids)
            logits = out[0] if isinstance(out, tuple) else out
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Apply xgrammar bitmask only when in constrained mode.
            # Structural tokens (end marker, newlines) are ONLY un-masked when
            # the JSON grammar is complete. If they're always un-masked, the
            # model emits <|tool_call_end|> or newline spam mid-JSON and the
            # tool call becomes unparseable (epoch-3 collapse: 50/50 tasks
            # with 0 tools because the JSON was garbage).
            if constrained and grammar_matcher is not None and bitmask is not None:
                import xgrammar as xgr
                grammar_matcher.fill_next_token_bitmask(bitmask)
                # Save skip-list logits before masking — only when grammar done
                skip_logits = {}
                grammar_done = grammar_matcher.is_terminated()
                if grammar_done:
                    for sid in skip_token_ids:
                        if 0 <= sid < next_logits.shape[-1]:
                            skip_logits[sid] = next_logits[:, sid].clone()
                xgr.apply_token_bitmask_inplace(next_logits, bitmask)
                # Restore skip-list logits (un-mask structural tokens)
                for sid, val in skip_logits.items():
                    next_logits[:, sid] = val

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                # Repetition penalty: penalize tokens already generated
                # (look at last 64 tokens to limit compute)
                if generated_ids:
                    for tid in set(generated_ids[-64:]):
                        next_logits[:, tid] /= repetition_penalty

                # Top-k filtering: keep only top_k logits before softmax
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(
                        next_logits, top_k)[0][..., -1, None]
                    next_logits.masked_fill_(indices_to_remove, float('-inf'))

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tok_id = next_token.item()
            generated_ids.append(tok_id)
            ids = torch.cat([ids, next_token], dim=1)

            if tok_id == EOS_ID:
                break

            # Switch to constrained mode when the start marker is emitted
            if not constrained and grammar_matcher is not None and tok_id == TOOL_CALL_START_FIRST_ID:
                constrained = True
                continue

            # Accept token in grammar matcher when constrained.
            # Skip structural tokens (end marker + newlines) — not part of JSON.
            if constrained and grammar_matcher is not None:
                if tok_id in skip_token_ids:
                    if tok_id == TOOL_CALL_END_FIRST_ID:
                        constrained = False  # back to free text after end marker
                    continue
                try:
                    grammar_matcher.accept_token(tok_id)
                except Exception:
                    break

    gen_ids = ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=False)


# ── xgrammar constrained decoding helpers ─────────────────────────────────

# Lazy import — xgrammar is optional (only needed for constrained decoding)
xgr = None

# Cache tokenizer info + compiler (expensive to create, reuse across calls)
_xgr_tokenizer_info = None
_xgr_compiler = None


def _ensure_xgr():
    """Lazy-load xgrammar on first use (avoids ImportError when unused)."""
    global xgr
    if xgr is None:
        import xgrammar as _xgr
        xgr = _xgr
    return xgr


def _get_xgr_compiler(tokenizer):
    """Get or create a cached xgrammar compiler for the tokenizer."""
    global _xgr_tokenizer_info, _xgr_compiler
    xgr = _ensure_xgr()
    if _xgr_compiler is None or _xgr_tokenizer_info is None:
        # Load a fresh HF tokenizer for xgrammar (gigatoken wrapper not supported)
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained("research/checkpoints/forgelm_v2_tokenizer")
        # Pass vocab_size=65536 to match model config (tokenizer has 64416)
        # This ensures the bitmask covers all model logits
        _xgr_tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            hf_tok, vocab_size=65536, stop_token_ids=[7])
        _xgr_compiler = xgr.GrammarCompiler(_xgr_tokenizer_info)
    return _xgr_compiler


def create_tool_grammar(tokenizer, tools: list[dict]) -> tuple[Any, Any]:
    """Create an xgrammar matcher that constrains tool call JSON to valid names.

    The grammar enforces:
      - "name" must be one of the registered tool names (enum)
      - "arguments" must be a JSON object
      - Valid JSON structure

    Used with qwen_generate() in two-phase mode: free text first, then
    constrained JSON when the model starts emitting a tool call.

    Returns (matcher, bitmask) for use with qwen_generate().
    """
    compiler = _get_xgr_compiler(tokenizer)

    # Unwrap OpenAI-style {"type": "function", "function": {"name": ...}}
    unwrapped = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            unwrapped.append(t["function"])
        else:
            unwrapped.append(t)
    tool_names = [t["name"] for t in unwrapped]

    # Schema for a single tool call object
    tool_call_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": tool_names},
            "arguments": {"type": "object"}
        },
        "required": ["name", "arguments"]
    }

    compiled = compiler.compile_json_schema(json.dumps(tool_call_schema))
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, _xgr_tokenizer_info.vocab_size)

    return matcher, bitmask


def make_grammar_logits_processor(grammar_matcher, bitmask, tokenizer):
    """Build a logits_processor callback for ForgeEngine.generate_raw().

    Implements the same two-phase constrained decoding as qwen_generate():
      - Phase 1 (free text): no masking. When the start marker
        (<tool_call>, id 531) is emitted, switch to constrained mode.
      - Phase 2 (constrained): apply xgrammar bitmask to enforce valid
        tool-call JSON. Skip structural tokens (end marker, newlines).
        When the end marker (</tool_call>, id 532) is emitted, switch
        back to free text.

    The processor is stateful (tracks constrained mode + accepted tokens).
    """
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))
    skip_token_ids = {TOOL_CALL_END_FIRST_ID} | newline_ids
    state = {"constrained": False}

    def logits_processor(next_logits: torch.Tensor,
                         generated_ids: list[int]) -> torch.Tensor:
        # FIRST accept the previous token into the grammar matcher, so the
        # bitmask for the NEXT token reflects the full state. Doing this
        # after fill_next_token_bitmask leaves the mask one step stale and
        # lets the model re-emit tokens it already generated (garbage like
        # {"{"namename"") — only visible when the model emits the real
        # <tool_call> special token (id 531).
        if state["constrained"] and generated_ids and grammar_matcher is not None:
            last_tok = generated_ids[-1]
            if last_tok == TOOL_CALL_END_FIRST_ID:
                # End marker → back to free text
                state["constrained"] = False
            elif last_tok not in skip_token_ids and last_tok != TOOL_CALL_START_FIRST_ID:
                try:
                    grammar_matcher.accept_token(last_tok)
                except Exception:
                    pass  # grammar mismatch — let it slide

        # Check if we just entered a tool call (last token was start marker)
        if generated_ids and not state["constrained"]:
            if generated_ids[-1] == TOOL_CALL_START_FIRST_ID:
                state["constrained"] = True

        if not state["constrained"]:
            return next_logits

        # In constrained mode: apply grammar bitmask (state is now current)
        if grammar_matcher is not None and bitmask is not None:
            grammar_matcher.fill_next_token_bitmask(bitmask)
            # Move bitmask to same device as logits (xgrammar allocates on CPU)
            bitmask_dev = bitmask.to(next_logits.device)
            # Save skip-list logits before masking — only when grammar done.
            # If always un-masked, model emits newline spam or the end marker
            # mid-JSON, making the tool call unparseable.
            skip_logits = {}
            grammar_done = grammar_matcher.is_terminated()
            if grammar_done:
                for sid in skip_token_ids:
                    if 0 <= sid < next_logits.shape[-1]:
                        skip_logits[sid] = next_logits[:, sid].clone()
            xgr.apply_token_bitmask_inplace(next_logits, bitmask_dev)
            # Restore skip-list logits
            for sid, val in skip_logits.items():
                next_logits[:, sid] = val

        return next_logits

    return logits_processor
