"""Convert glaive-function-calling-v2 to ForgeLM V10 SFT format.

Glaive format:
  {"system": "SYSTEM: ...", "chat": "USER: ...\n\nASSISTANT: <functioncall> {...}\n\nFUNCTION RESPONSE: {...}\n\nASSISTANT: ..."}

ForgeLM V10 SFT format (what sft_train.py expects):
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [{"name": "...", "args": {...}}]},
    {"role": "tool", "name": "...", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}

Tool calls are converted to Pythonic format for rendering by sft_train.py:
  <|tool_call_start|>[func_name(arg='val')]<|tool_call_end|>
"""
import json
import re
from pathlib import Path

_FUNCTIONCALL_RE = re.compile(
    r'<functioncall>\s*(.+?)(?=\n\n|\Z)',
    re.DOTALL,
)


def _parse_functioncall(text: str) -> tuple[list[dict], str]:
    """Parse <functioncall> {...} from assistant text.

    Glaive format:
      <functioncall> {"name": "func", "arguments": '{"key": "val"}'}

    The arguments value is a JSON string wrapped in single quotes.

    Returns (tool_calls, remaining_text).
    tool_calls: [{"name": "...", "args": {...}}]
    """
    tool_calls = []

    for m in _FUNCTIONCALL_RE.finditer(text):
        raw = m.group(1).strip()
        # The raw text is like: {"name": "func", "arguments": '{"key": "val"}'}
        # The outer JSON has arguments as a single-quoted JSON string.
        # Try parsing directly first.
        obj = None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # The single-quoted arguments string breaks JSON parsing.
            # Extract name and arguments manually.
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
            args_match = re.search(
                r'"arguments"\s*:\s*\'(\{.*?\})\'', raw)
            if name_match:
                name = name_match.group(1)
                if args_match:
                    try:
                        args = json.loads(args_match.group(1))
                    except json.JSONDecodeError:
                        args = {}
                else:
                    # Try double-quoted arguments
                    args_match2 = re.search(
                        r'"arguments"\s*:\s*(\{.*?\})\s*\}', raw)
                    if args_match2:
                        try:
                            args = json.loads(args_match2.group(1))
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = {}
                obj = {"name": name, "arguments": args}

        if obj and isinstance(obj, dict) and "name" in obj:
            name = obj["name"]
            args = obj.get("arguments", obj.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append({"name": name, "args": args})

    # Remove functioncall blocks from remaining text
    remaining = _FUNCTIONCALL_RE.sub("", text).strip()
    return tool_calls, remaining


def _parse_chat(chat_str: str) -> list[dict]:
    """Parse glaive chat string into messages list.

    Format: "USER: ...\n\nASSISTANT: ...\n\nFUNCTION RESPONSE: ...\n\nASSISTANT: ..."
    """
    messages = []

    # Split on double-newline, keeping the role markers.
    # Use a non-capturing lookahead to avoid duplicate parts.
    parts = re.split(r'\n\n(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)', chat_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("USER:"):
            messages.append({"role": "user",
                             "content": _clean_content(part[5:])})
        elif part.startswith("ASSISTANT:"):
            content = part[10:].strip()
            tool_calls, remaining = _parse_functioncall(content)
            msg = {"role": "assistant",
                   "content": _clean_content(remaining)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
        elif part.startswith("FUNCTION RESPONSE:"):
            messages.append({"role": "tool", "name": "function",
                             "content": _clean_content(part[18:])})

    return messages


def _make_msg(role: str, content: str) -> dict:
    """Create a message dict, parsing functioncalls from assistant messages."""
    if role == "assistant":
        tool_calls, remaining = _parse_functioncall(content)
        msg = {"role": "assistant", "content": remaining}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg
    elif role == "tool":
        return {"role": "tool", "name": "function", "content": content}
    else:
        return {"role": role, "content": content}


def _clean_system(system_str: str) -> str:
    """Clean the system prompt — remove the 'SYSTEM: ' prefix and
    strip non-LFM special tokens."""
    if system_str.startswith("SYSTEM:"):
        system_str = system_str[7:]
    # Strip non-LFM special tokens (glaive uses <|endoftext|> etc.)
    system_str = system_str.replace("<|endoftext|>", "")
    return system_str.strip()


def _clean_content(content: str) -> str:
    """Strip non-LFM special tokens from message content."""
    # <|endoftext|> is not an LFM token — LFM uses <|im_end|> (id 7)
    content = content.replace("<|endoftext|>", "")
    return content.strip()


def convert(input_path: str, output_path: str, max_examples: int = 0) -> int:
    """Convert glaive JSON to ForgeLM V10 SFT JSONL.

    Args:
        input_path: path to glaive-function-calling-v2.json
        output_path: output .jsonl file
        max_examples: 0 = all, >0 = limit

    Returns: number of examples written.
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from {input_path}")

    n_written = 0
    n_skipped = 0
    n_with_tools = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for _i, item in enumerate(data):
            if max_examples > 0 and n_written >= max_examples:
                break

            system = item.get("system", "")
            chat = item.get("chat", "")

            if not chat or not system:
                n_skipped += 1
                continue

            messages = []
            messages.append({"role": "system", "content": _clean_system(system)})
            messages.extend(_parse_chat(chat))

            # Only keep examples that have at least one tool call
            has_tools = any(m.get("tool_calls") for m in messages)
            if not has_tools:
                n_skipped += 1
                continue

            n_with_tools += 1

            # Validate: must have at least user + assistant
            roles = [m["role"] for m in messages]
            if "user" not in roles or "assistant" not in roles:
                n_skipped += 1
                continue

            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Written: {n_written} (with tools: {n_with_tools}, skipped: {n_skipped})")
    return n_written


def main():
    input_path = "scripts/sft/tool_use/glaive-fc-v2/glaive-function-calling-v2.json"
    output_path = "data/sft/glaive_fc_pythonic.jsonl"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n = convert(input_path, output_path)
    print(f"\nOutput: {output_path} ({n} examples)")

    # Show a sample
    if n > 0:
        with open(output_path, encoding="utf-8") as f:
            first = json.loads(f.readline())
        print("\n=== Sample (first 500 chars) ===")
        print(json.dumps(first, ensure_ascii=False, indent=1)[:500])


if __name__ == "__main__":
    main()
