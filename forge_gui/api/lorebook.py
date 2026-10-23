"""Lorebook memory database — keyword-triggered persistent memory for ForgeAI.

Stores JSON entries with keyword triggers, constant flags, and priority
ordering. Entries are injected into the system prompt before each
generation cycle using a hybrid strategy:

1. **Constant entries** — always injected (up to token budget)
2. **Keyword-triggered entries** — auto-inject when trigger keywords appear
   in the recent message history (scan_depth messages)
3. **Explicit recall** — the model can call the `recall_memory` tool to
   search the database for specific entries

Entries are stored in ``data/memory/lorebook.json`` and managed via the
``Lorebook`` class. The ``MemoryTools`` class provides tool definitions
(``remember``, ``recall_memory``, ``forget``) for the agentic harness.

Entry structure (lorebook-style):
    {
        "id": "uuid",
        "keys": ["trigger", "words"],
        "content": "Text injected when active",
        "constant": false,
        "priority": 100,
        "category": "user_pref" | "project" | "skill" | "feedback",
        "enabled": true,
        "created": "2026-09-01T12:00:00",
        "last_triggered": null,
        "trigger_count": 0
    }
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ── entry dataclass ─────────────────────────────────────────────────────

CATEGORIES = ("user_pref", "project", "skill", "feedback", "reference", "note")


@dataclass
class LoreEntry:
    id: str
    keys: list[str]
    content: str
    constant: bool = False
    priority: int = 100          # lower = higher priority (injected first)
    category: str = "note"
    enabled: bool = True
    created: str = ""
    last_triggered: str | None = None
    trigger_count: int = 0

    def matches(self, text: str) -> bool:
        """Check if any key appears in the text (case-insensitive)."""
        text_lower = text.lower()
        return any(k.lower() in text_lower for k in self.keys if k)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LoreEntry:
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            keys=d.get("keys", []),
            content=d.get("content", ""),
            constant=d.get("constant", False),
            priority=d.get("priority", 100),
            category=d.get("category", "note"),
            enabled=d.get("enabled", True),
            created=d.get("created", ""),
            last_triggered=d.get("last_triggered"),
            trigger_count=d.get("trigger_count", 0),
        )


# ── lorebook ────────────────────────────────────────────────────────────

DEFAULT_SCAN_DEPTH = 20       # messages to scan for triggers
DEFAULT_TOKEN_BUDGET = 2000   # max tokens for injected lore


class Lorebook:
    """Persistent lorebook with hybrid injection (constant + triggered).

    Stored as JSON at ``data/memory/lorebook.json``. Pure-stdlib (no Qt)
    so it can be unit-tested anywhere and used from worker threads.
    """

    # Fields a caller may change via update() — id/created/keys are the
    # entry's identity + provenance (keys double as the trigger contract;
    # letting an update rewrite them silently re-scopes who the memory
    # fires for), and the trigger stats are internal bookkeeping.
    _MUTABLE_FIELDS = {"content", "constant", "priority", "category",
                       "enabled"}

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root else Path("data/memory")
        self._path = self._root / "lorebook.json"
        self._entries: list[LoreEntry] = []
        # Mutate+save under one lock — inject() runs on the loop while
        # remember/update run on harness threads; the old unguarded
        # read-modify-write could interleave a dropped JSON write.
        self._lock = threading.RLock()
        self._dirty = False          # trigger-stat changes pending save
        self._last_save = 0.0
        self.load()

    # ── persistence ───────────────────────────────────────────────────
    # inject() fires once per chat send; throttle its stat-only save so a
    # chatty session doesn't rewrite the whole file every message.
    _SAVE_DEBOUNCE_S = 30.0

    def load(self) -> None:
        if not self._path.is_file():
            self._entries = []
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = [LoreEntry.from_dict(e)
                             for e in data.get("entries", [])]
        except Exception as e:
            logger.warning("lorebook load failed: %s", e)
            self._entries = []

    def save(self) -> None:
        with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            data = {"version": 1,
                    "entries": [e.to_dict() for e in self._entries]}
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self._path)
            self._dirty = False
            self._last_save = time.monotonic()

    def _save_debounced(self) -> None:
        """Persist pending trigger-stat updates at most every
        _SAVE_DEBOUNCE_S — callers must hold the lock."""
        self._dirty = True
        if time.monotonic() - self._last_save >= self._SAVE_DEBOUNCE_S:
            self.save()

    def flush(self) -> None:
        """Force-write pending trigger-stat updates (server shutdown)."""
        with self._lock:
            if self._dirty:
                self.save()

    # ── CRUD ──────────────────────────────────────────────────────────
    @property
    def entries(self) -> list[LoreEntry]:
        return self._entries

    def get(self, entry_id: str) -> LoreEntry | None:
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    def add(self, keys: list[str], content: str,
            constant: bool = False, priority: int = 100,
            category: str = "note") -> LoreEntry:
        """Add a new entry and persist."""
        with self._lock:
            entry = LoreEntry(
                id=str(uuid.uuid4()),
                keys=keys,
                content=content,
                constant=constant,
                priority=priority,
                category=category if category in CATEGORIES else "note",
                created=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            self._entries.append(entry)
            self.save()
            return entry

    def update(self, entry_id: str, **kwargs) -> LoreEntry | None:
        """Update mutable fields on an entry. Returns the entry or None.
        Immutable fields (id/created/keys/trigger stats) are silently
        dropped — a caller can't retarget or rewrite an entry's identity."""
        with self._lock:
            e = self.get(entry_id)
            if e is None:
                return None
            for k, v in kwargs.items():
                if k in self._MUTABLE_FIELDS:
                    setattr(e, k, v)
            self.save()
            return e

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries
                             if e.id != entry_id]
            if len(self._entries) < before:
                self.save()
                return True
            return False

    def forget(self, keys: list[str]) -> int:
        """Delete all entries matching any of the given keys. Returns count."""
        with self._lock:
            to_remove = {e.id for e in self._entries
                         if any(k.lower() in " ".join(e.keys).lower()
                                for k in keys if k)}
            if not to_remove:
                return 0
            self._entries = [e for e in self._entries
                             if e.id not in to_remove]
            self.save()
            return len(to_remove)

    # ── search ────────────────────────────────────────────────────────
    def search(self, query: str, limit: int = 10) -> list[LoreEntry]:
        """Search entries by keyword (in keys or content)."""
        q = query.lower()
        scored: list[tuple[float, LoreEntry]] = []
        for e in self._entries:
            if not e.enabled:
                continue
            score = 0.0
            for k in e.keys:
                if k.lower() in q or q in k.lower():
                    score += 2.0
            if q in e.content.lower():
                score += 1.0
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: (-x[0], x[1].priority))
        return [e for _, e in scored[:limit]]

    # ── injection ─────────────────────────────────────────────────────
    def inject(self, recent_messages: list[dict],
               scan_depth: int = DEFAULT_SCAN_DEPTH,
               token_budget: int = DEFAULT_TOKEN_BUDGET) -> str:
        """Build the lore injection text for the system prompt.

        Hybrid strategy:
        1. All constant entries (sorted by priority)
        2. Keyword-triggered entries from recent message scan
        3. Truncated to token_budget (rough char estimate: 4 chars/token)
        """
        if not self._entries:
            return ""

        # build scan text from recent messages
        scan_msgs = recent_messages[-scan_depth:] if scan_depth > 0 else recent_messages
        scan_text = " ".join(
            m.get("content", "") for m in scan_msgs
            if m.get("role") in ("user", "assistant"))

        # collect entries — lock while reading/mutating entry state;
        # add/delete/update run on other threads.
        with self._lock:
            triggered: list[LoreEntry] = []
            for e in self._entries:
                if not e.enabled:
                    continue
                if e.constant:
                    triggered.append(e)
                elif e.matches(scan_text):
                    triggered.append(e)
                    # update trigger stats
                    e.last_triggered = time.strftime("%Y-%m-%dT%H:%M:%S")
                    e.trigger_count += 1

        if not triggered:
            return ""

        # sort by priority (lower = first)
        triggered.sort(key=lambda e: e.priority)

        # build text with budget
        char_budget = token_budget * 4
        lines: list[str] = ["", "=== Memory ==="]
        for e in triggered:
            line = f"[{e.category}] {e.content}"
            if len("\n".join(lines)) + len(line) > char_budget:
                break
            lines.append(line)

        if len(lines) <= 1:  # only the header
            return ""
        lines.append("=== End Memory ===")
        text = "\n".join(lines)

        # Trigger stats changed — persist via debounce, not a full JSON
        # rewrite on every chat send.
        with self._lock:
            self._save_debounced()
        return text

    # ── stats ─────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "total": len(self._entries),
            "enabled": sum(1 for e in self._entries if e.enabled),
            "constant": sum(1 for e in self._entries if e.constant),
            "by_category": {c: sum(1 for e in self._entries if e.category == c)
                            for c in CATEGORIES},
        }


# ── memory tools (for the agentic harness) ──────────────────────────────

def memory_tool_defs() -> list[dict]:
    """Tool definitions for the model to interact with the lorebook."""
    return [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": (
                    "Save a fact, preference, or note to long-term memory. "
                    "The content will be recalled when the trigger keys "
                    "appear in future conversations. Use this when the user "
                    "says 'remember this' or when you discover something "
                    "worth persisting."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Trigger keywords that activate this "
                                "memory (e.g. ['python', 'naming'])"),
                        },
                        "content": {
                            "type": "string",
                            "description": "The memory text to save",
                        },
                        "category": {
                            "type": "string",
                            "enum": list(CATEGORIES),
                            "description": "Memory category",
                        },
                        "constant": {
                            "type": "boolean",
                            "description": (
                                "If true, always inject (not just on "
                                "keyword match). Use for critical rules."),
                        },
                    },
                    "required": ["keys", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall_memory",
                "description": (
                    "Search long-term memory for entries matching a query. "
                    "Returns matching entries with their content."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (keywords)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default 10)",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget",
                "description": (
                    "Delete memory entries matching the given keys. "
                    "Use when the user says 'forget about X' or when "
                    "information is outdated."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys to match for deletion",
                        },
                    },
                    "required": ["keys"],
                },
            },
        },
    ]


class MemoryTools:
    """Wraps a Lorebook with execute() for the agentic harness.

    Each method returns a dict with {"ok": bool, "result": ...} matching
    the agent_tools.py convention.
    """

    def __init__(self, lorebook: Lorebook) -> None:
        self.lore = lorebook

    def execute(self, name: str, arguments: dict) -> dict:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return {"ok": False, "result": {"error": f"unknown memory tool: {name}"}}
        try:
            return fn(arguments)
        except Exception as e:
            return {"ok": False, "result": {"error": f"{type(e).__name__}: {e}"}}

    def tool_remember(self, args: dict) -> dict:
        keys = args.get("keys", [])
        content = args.get("content", "")
        category = args.get("category", "note")
        constant = args.get("constant", False)
        if not keys or not content:
            return {"ok": False, "result": {"error": "keys and content required"}}
        entry = self.lore.add(keys, content, constant=constant,
                              category=category)
        return {"ok": True, "result": {
            "id": entry.id, "keys": entry.keys, "content": entry.content,
            "category": entry.category, "constant": entry.constant,
            "message": f"Remembered: {content[:80]}"}}

    def tool_recall_memory(self, args: dict) -> dict:
        query = args.get("query", "")
        limit = args.get("limit", 10)
        if not query:
            return {"ok": False, "result": {"error": "query required"}}
        results = self.lore.search(query, limit=limit)
        return {"ok": True, "result": {
            "count": len(results),
            "entries": [{"keys": e.keys, "content": e.content,
                         "category": e.category, "priority": e.priority}
                        for e in results],
        }}

    def tool_forget(self, args: dict) -> dict:
        keys = args.get("keys", [])
        if not keys:
            return {"ok": False, "result": {"error": "keys required"}}
        n = self.lore.forget(keys)
        return {"ok": True, "result": {
            "deleted": n,
            "message": f"Forgot {n} entries matching {keys}"}}
