"""ChatStore — persistent chat conversations with quality ratings + SFT export.

Pure-python (no Qt) so it is trivially unit-testable and reusable from
worker threads. Conversations live in ``data/chats/conversations.json``;
training-data exports are written to ``data/sft/*.jsonl`` in the exact
format ``forge/training/runners/sft_train.py:load_examples`` consumes
(``{"messages": [{role, content}, ...]}`` per line).

Rating model (drives what becomes training data):
  - every assistant message carries ``rating``: "good" | "bad" | None
  - export emits, per good-rated assistant turn, the conversation prefix
    up to and including that turn (system + user + assistant), so the
    trainer learns to reproduce exactly the approved behavior.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

RATING_GOOD = "good"
RATING_BAD = "bad"


def _now() -> float:
    return time.time()


def _slug(title: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return (s[:limit] or "chat")


class ChatStore:
    """JSON-persisted conversation list with per-message ratings."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            root = Path(__file__).resolve().parents[2] / "data"
        self.root = Path(root)
        self.chats_dir = self.root / "chats"
        self.sft_dir = self.root / "sft"
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self.sft_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.chats_dir / "conversations.json"
        self.conversations: list[dict[str, Any]] = []
        # Mutate+save under one lock — the SSE stream persists on the
        # loop thread while REST/imports mutate from executor threads.
        self._lock = threading.RLock()
        self.load()

    # ── persistence ───────────────────────────────────────────────────
    def load(self) -> None:
        if not self.path.is_file():
            self.conversations = []
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.conversations = data.get("conversations", [])
        except Exception:
            self.conversations = []

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": 1,
                           "conversations": self.conversations},
                          f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)

    # ── conversation CRUD ─────────────────────────────────────────────
    def create(self, title: str = "New chat", model: str = "") -> dict[str, Any]:
        with self._lock:
            conv = {
                "id": uuid.uuid4().hex[:12],
                "title": title,
                "model": model,
                "created_at": _now(),
                "updated_at": _now(),
                "messages": [],
            }
            self.conversations.insert(0, conv)
            self.save()
            return conv

    def get(self, conv_id: str) -> dict[str, Any] | None:
        for c in self.conversations:
            if c["id"] == conv_id:
                return c
        return None

    def delete(self, conv_id: str) -> bool:
        with self._lock:
            before = len(self.conversations)
            self.conversations = [c for c in self.conversations
                                  if c["id"] != conv_id]
            changed = len(self.conversations) != before
            if changed:
                self.save()
            return changed

    def rename(self, conv_id: str, title: str) -> None:
        with self._lock:
            conv = self.get(conv_id)
            if conv is not None:
                conv["title"] = title or "Untitled"
                conv["updated_at"] = _now()
                self.save()

    def touch(self, conv_id: str, model: str = "") -> None:
        with self._lock:
            conv = self.get(conv_id)
            if conv is None:
                return
            conv["updated_at"] = _now()
            if model:
                conv["model"] = model
            # auto-title from the first user message
            if conv["title"] in ("New chat", "Untitled"):
                for m in conv["messages"]:
                    if m.get("role") == "user" and m.get("content"):
                        conv["title"] = \
                            m["content"].strip().splitlines()[0][:60]
                        break
            self.save()

    # ── messages ──────────────────────────────────────────────────────
    def append_message(self, conv_id: str, role: str, content: str,
                       rating: str | None = None,
                       image: str = "",
                       tool_calls: list[dict] | None = None,
                       name: str = "",
                       reasoning_content: str | None = None) -> int:
        conv = self.get(conv_id)
        if conv is None:
            raise KeyError(conv_id)
        msg = {
            "role": role, "content": content,
            "rating": rating, "ts": _now(),
        }
        if image:
            msg["image"] = image
        # Function-calling fields (canonical Jamba chat_template shape):
        # assistant messages keep their tool_calls + reasoning so history
        # re-renders exactly what the model produced; tool messages carry
        # the tool name alongside the JSON result content.
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if name:
            msg["name"] = name
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        with self._lock:
            conv["messages"].append(msg)
            conv["updated_at"] = _now()
            self.save()
        return len(conv["messages"]) - 1

    def append_messages(self, conv_id: str,
                        messages: list[dict]) -> int:
        """Append a batch of produced messages with ONE save — the chat
        stream used to N+1-rewrite the whole conversations file for a
        tool-call turn (assistant + tool + assistant …)."""
        conv = self.get(conv_id)
        if conv is None:
            raise KeyError(conv_id)
        with self._lock:
            for m in messages:
                msg = dict(m)
                msg.setdefault("rating", None)
                msg.setdefault("ts", _now())
                conv["messages"].append(msg)
            conv["updated_at"] = _now()
            self.save()
        return len(conv["messages"])

    def truncate_messages(self, conv_id: str, index: int) -> dict | None:
        """Drop messages[index:] — backs regenerate/edit-and-resubmit."""
        with self._lock:
            conv = self.get(conv_id)
            if conv is None or not (0 <= index <= len(conv["messages"])):
                return None
            conv["messages"] = conv["messages"][:index]
            conv["updated_at"] = _now()
            self.save()
            return conv

    def rate_message(self, conv_id: str, msg_idx: int,
                     rating: str | None) -> str | None:
        """Set/clear a message rating ('good' | 'bad' | None). Toggles."""
        with self._lock:
            conv = self.get(conv_id)
            if conv is None or not (0 <= msg_idx < len(conv["messages"])):
                return None
            cur = conv["messages"][msg_idx].get("rating")
            new = None if (rating is None or cur == rating) else rating
            conv["messages"][msg_idx]["rating"] = new
            conv["updated_at"] = _now()
            self.save()
            return new

    @staticmethod
    def count_ratings(conv: dict[str, Any]) -> tuple[int, int]:
        good = sum(1 for m in conv["messages"] if m.get("rating") == RATING_GOOD)
        bad = sum(1 for m in conv["messages"] if m.get("rating") == RATING_BAD)
        return good, bad

    # ── training-data export ──────────────────────────────────────────
    def export_training_data(self, conv_ids: list[str] | None = None,
                             out_path: Path | None = None,
                             include_system: bool = True) -> tuple[Path, int]:
        """Export good-rated turns as sft_train-compatible JSONL.

        Each good-rated assistant message becomes one line:
          {"messages": [system?], {"role": "user", ...}, {"role": "assistant", ...}}
        The prefix is truncated at the good turn, so multi-turn conversations
        yield one example per approved reply. Returns (path, n_examples).
        """
        ids = set(conv_ids) if conv_ids else {c["id"] for c in self.conversations}
        examples: list[dict] = []
        for conv in self.conversations:
            if conv["id"] not in ids:
                continue
            prefix: list[dict] = []
            for m in conv["messages"]:
                role, content = m.get("role"), m.get("content", "")
                if role == "system":
                    if include_system and content.strip():
                        prefix.append({"role": "system", "content": content})
                    continue
                if role == "user":
                    prefix.append({"role": "user", "content": content})
                elif role == "tool":
                    # Tool results feed back as user-turn <tool_response>
                    # blocks in the canonical template — keep them so
                    # tool-using conversations export faithfully.
                    prefix.append({"role": "tool",
                                   "name": m.get("name", ""),
                                   "content": content})
                elif role == "assistant":
                    stored = {"role": "assistant", "content": content}
                    if m.get("tool_calls"):
                        stored["tool_calls"] = m["tool_calls"]
                    if m.get("reasoning_content"):
                        stored["reasoning_content"] = m["reasoning_content"]
                    if (m.get("rating") == RATING_GOOD
                            and (content.strip() or m.get("tool_calls"))):
                        messages = [p for p in prefix
                                    if p["content"].strip()
                                    or p.get("tool_calls")]
                        messages.append(dict(stored))
                        examples.append({"messages": messages})
                    prefix.append(stored)
        if out_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = self.sft_dir / f"forge_chats_{stamp}.jsonl"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return out_path, len(examples)

    def list_exports(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.sft_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime,
                        reverse=True):
            try:
                n = sum(1 for line in p.read_text(encoding="utf-8").splitlines()
                        if line.strip())
            except OSError:
                n = 0
            out.append({"path": str(p), "name": p.name, "examples": n,
                        "size": p.stat().st_size, "mtime": p.stat().st_mtime})
        return out

    @staticmethod
    def preview_title(conv: dict[str, Any]) -> str:
        good, bad = ChatStore.count_ratings(conv)
        n = len(conv["messages"])
        return f"{conv['title']}  ·  {n} msg" + (f"  ·  ★{good}" if good else "")


# ── transcript import ─────────────────────────────────────────────
# Paste-import for the Chat page: parse a copied conversation back into
# [{role, content}] turns.

_IM_START_RE = re.compile(
    r"<\|im_start\|>\s*(\w+)\s*\n(.*?)<\|im_end\|>", re.DOTALL)

_ROLE_ALIASES = {
    "system": "system", "tool": "tool",
    "user": "user", "human": "user", "you": "user", "me": "user",
    "question": "user", "q": "user", "prompt": "user",
    "assistant": "assistant", "ai": "assistant", "model": "assistant",
    "chatgpt": "assistant", "claude": "assistant", "gemini": "assistant",
    "a": "assistant", "answer": "assistant", "forgelm": "assistant",
}

# Marker-only line: "**User:**", "ChatGPT said:", "## Assistant", "[You]"
_MARKER_LINE_RE = re.compile(
    r"^\s*[#*>\[\(\-]*\s*"
    r"(system|tool|user|human|you|me|question|q|prompt|assistant|ai|"
    r"model|chatgpt|claude|gemini|a|answer|forgelm)"
    r"\s*(?:said|says)?\s*[:\]\)\*#]*\s*$", re.IGNORECASE)

# Inline marker: "User: hello" — requires a colon so prose lines that
# merely start with a role word don't false-trigger.
_INLINE_RE = re.compile(
    r"^\s*[#*>\[\(\-]*\s*"
    r"(system|tool|user|human|question|q|prompt|assistant|ai|model|"
    r"chatgpt|claude|gemini|a|answer|forgelm)"
    r"\s*:\s*(.*\S.*)$", re.IGNORECASE)


def parse_transcript(text: str) -> list[dict[str, str]]:
    """Parse a pasted chat transcript into ``[{role, content}]`` turns.

    Formats tried in order:
      1. JSON — ``[{"role": ..., "content": ...}]`` or ``{"messages": []}``
      2. ChatML — ``<|im_start|>role`` ... ``<|im_end|>`` blocks
      3. Role markers — "User:", "**Assistant:**", "ChatGPT said:", ...
         (marker-only lines, or inline "role: content")
      4. Blank-line paragraphs — alternating user/assistant, starting user
    """
    text = (text or "").strip()
    if not text:
        return []
    msgs = (_parse_json_transcript(text) or _parse_chatml(text)
            or _parse_role_marked(text) or _parse_paragraphs(text))
    return [{"role": m["role"], "content": m["content"].strip()}
            for m in msgs if m["content"].strip()]


def _parse_json_transcript(text: str) -> list[dict[str, str]]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("messages")
    if not isinstance(data, list):
        return []
    out = []
    for m in data:
        if not isinstance(m, dict):
            return []
        role = str(m.get("role", "")).lower()
        content = m.get("content")
        if isinstance(content, list):      # OpenAI-style content parts
            content = "".join(str(p.get("text", ""))
                              for p in content if isinstance(p, dict))
        if role not in ("system", "user", "assistant", "tool") \
                or not isinstance(content, str):
            return []
        out.append({"role": role, "content": content})
    return out


def _parse_chatml(text: str) -> list[dict[str, str]]:
    if "<|im_start|>" not in text:
        return []
    out = []
    for m in _IM_START_RE.finditer(text):
        role = _ROLE_ALIASES.get(m.group(1).lower())
        if role:
            out.append({"role": role, "content": m.group(2).strip()})
    return out


def _parse_role_marked(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    for line in text.splitlines():
        mm = _MARKER_LINE_RE.match(line)
        if mm:
            cur = {"role": _ROLE_ALIASES[mm.group(1).lower()],
                   "content": ""}
            out.append(cur)
            continue
        im = _INLINE_RE.match(line)
        if im:
            cur = {"role": _ROLE_ALIASES[im.group(1).lower()],
                   "content": im.group(2)}
            out.append(cur)
            continue
        if cur is not None:
            cur["content"] += "\n" + line
    # needs at least two turns to count as a transcript — a lone marker
    # falls through to the paragraph fallback
    return out if len(out) >= 2 else []


def _parse_paragraphs(text: str) -> list[dict[str, str]]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    roles = ("user", "assistant")
    return [{"role": roles[i % 2], "content": p}
            for i, p in enumerate(paras)]
