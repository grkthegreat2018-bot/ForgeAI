"""ForgeAI Library — persistent knowledge base with lorebook-style injection.

A massive I/O-focused infinite context system. The model can save its
failures, wins, research, and common data. Entries are pre-tokenized on
insertion for instant injection. Auto-trims and optimizes each entry.

Architecture:
    Library/
    ├── entries/          # One .json + .tokens per entry
    │   ├── {id}.json     # Content + metadata
    │   └── {id}.tokens   # Pre-tokenized token IDs (numpy .npy)
    ├── index.json        # Tag → entry IDs, keyword → entry IDs
    └── meta.json         # Library metadata (size, config, stats)

Key features:
1. **Pre-tokenization cache**: Entries are tokenized on insertion. Token
   IDs are saved alongside the original text. Injection is instant — no
   re-tokenization needed at generation time.

2. **Lorebook-style injection**: Entries have trigger keywords. When the
   prompt contains a trigger, the matching entry is injected into the
   context (up to a token budget). This is I/O-focused infinite context —
   the model effectively has access to terabytes of data, with only the
   relevant portions loaded into context.

3. **Auto-trim + optimization**: The library maintains a max size. When
   full, it trims least-relevant entries (LRU + priority + access count).
   Similar entries can be merged. Entries are re-ranked by relevance.

4. **Model self-write**: The model can save entries via library.save().
   This creates a feedback loop — the model learns from its own history.
   Categories: "failure", "win", "research", "common_data", "custom".

5. **Fast lookup**: In-memory tag index + keyword index for O(1) tag
   lookup and O(k) keyword lookup. Disk-backed for persistence.

6. **Tags + descriptions**: Every entry has tags, a description, and
   metadata for easy filtering and search.

Usage:
    from forge.engine.library import Library

    lib = Library(tokenizer, path="research/data/library")
    lib.save(content="Solution to bug X...", category="win",
             tags=["bug", "fix"], description="Bug X fix recipe",
             triggers=["bug X", "error X"])
    lib.save(content="Failed approach Y...", category="failure",
             tags=["approach", "failed"], triggers=["approach Y"])

    # Inject relevant entries into a prompt
    augmented = lib.inject(prompt, max_tokens=2048)

    # Lookup
    results = lib.lookup(tags=["bug"], category="win")
    results = lib.search("memory leak")
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

# ── Entry categories ──────────────────────────────────────────────────────────

CATEGORIES = {"failure", "win", "research", "common_data", "custom"}

# Category retention weights (higher = kept longer during trim)
CATEGORY_RETENTION = {
    "failure": 0.8,      # failures are valuable lessons
    "win": 1.0,          # wins are most valuable
    "research": 0.9,     # research findings are valuable
    "common_data": 0.5,  # common data is easily replaceable
    "custom": 0.7,       # user-defined, moderate retention
}


@dataclass
class LibraryEntry:
    """A single library entry — one piece of knowledge.

    The content is pre-tokenized on creation, and the token IDs are
    cached alongside the original text for instant injection.
    """
    id: str
    content: str                    # original text
    token_ids: list[int] = field(default_factory=list)  # pre-tokenized
    token_count: int = 0            # len(token_ids)
    tags: list[str] = field(default_factory=list)
    description: str = ""           # short description for lookup
    category: str = "custom"        # failure/win/research/common_data/custom
    triggers: list[str] = field(default_factory=list)  # lorebook activation keys
    priority: int = 0               # higher = injected first
    max_tokens: int = 2048          # max tokens this entry can use in context
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    enabled: bool = True
    source: str = "model"           # "model" or "user" or "auto"

    def touch(self):
        """Update last_accessed and increment access_count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def relevance_score(self) -> float:
        """Compute relevance score for trim ranking.

        Higher = more relevant = kept longer.
        Combines: access_count, recency, priority, category retention.
        """
        now = time.time()
        age_s = now - self.last_accessed
        age_days = age_s / 86400

        # Recency decay: recent accesses weigh more
        recency = 1.0 / (1.0 + age_days * 0.1)

        # Access frequency
        frequency = min(self.access_count / 10.0, 1.0)

        # Category retention weight
        cat_weight = CATEGORY_RETENTION.get(self.category, 0.5)

        # Priority bonus
        prio = self.priority / 100.0

        return (recency * 0.3 + frequency * 0.3 + cat_weight * 0.3 + prio * 0.1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for JSON storage). Excludes token_ids (saved separately)."""
        d = asdict(self)
        d.pop("token_ids", None)  # saved as .tokens file
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LibraryEntry:
        """Deserialize from dict."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


class Library:
    """Persistent knowledge base with lorebook-style injection.

    Stores entries with pre-tokenized content for instant context injection.
    Auto-trims based on relevance scoring. Thread-safe.

    Args:
        tokenizer: HuggingFace tokenizer for pre-tokenization
        path: directory path for persistent storage
        max_entries: maximum number of entries (auto-trim when exceeded)
        max_total_tokens: maximum total token count across all entries
        injection_budget: default max tokens to inject per generate() call
    """

    def __init__(
        self,
        tokenizer=None,
        path: str | Path = "research/data/library",
        max_entries: int = 10_000,
        max_total_tokens: int = 5_000_000,
        injection_budget: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.path = Path(path)
        self.max_entries = max_entries
        self.max_total_tokens = max_total_tokens
        self.injection_budget = injection_budget

        self._lock = threading.RLock()
        self._entries: dict[str, LibraryEntry] = {}
        self._tag_index: dict[str, set[str]] = {}      # tag → entry IDs
        self._keyword_index: dict[str, set[str]] = {}   # keyword → entry IDs
        self._total_tokens: int = 0

        # Ensure directory exists
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "entries").mkdir(exist_ok=True)

        # Load from disk
        self._load()

    # ── Core operations ─────────────────────────────────────────────────

    def save(
        self,
        content: str,
        category: str = "custom",
        tags: list[str] | None = None,
        description: str = "",
        triggers: list[str] | None = None,
        priority: int = 0,
        max_tokens: int = 2048,
        source: str = "model",
        entry_id: str | None = None,
    ) -> str:
        """Save a new entry to the library.

        The content is pre-tokenized immediately and cached. The entry
        is persisted to disk.

        Args:
            content: the text content to store
            category: "failure", "win", "research", "common_data", "custom"
            tags: list of tags for categorization and lookup
            description: short description for search
            triggers: keywords that activate lorebook-style injection
            priority: higher = injected first (0-100)
            max_tokens: max tokens this entry can use in context
            source: "model", "user", or "auto"
            entry_id: optional custom ID (auto-generated if None)

        Returns:
            entry_id (str)
        """
        if category not in CATEGORIES:
            category = "custom"

        entry_id = entry_id or str(uuid.uuid4())[:12]
        if not self._entry_id_safe(entry_id):
            raise ValueError(f"invalid library entry_id: {entry_id!r}")
        tags = tags or []
        triggers = triggers or []

        # Pre-tokenize content
        token_ids: list[int] = []
        if self.tokenizer is not None:
            token_ids = self.tokenizer(
                content, add_special_tokens=False
            ).input_ids
        token_count = len(token_ids)

        entry = LibraryEntry(
            id=entry_id,
            content=content,
            token_ids=token_ids,
            token_count=token_count,
            tags=tags,
            description=description,
            category=category,
            triggers=triggers,
            priority=priority,
            max_tokens=max_tokens,
            source=source,
        )

        with self._lock:
            # If entry already exists, remove old version from indices
            if entry_id in self._entries:
                self._remove_from_indices(entry_id)
                old = self._entries[entry_id]
                self._total_tokens -= old.token_count

            self._entries[entry_id] = entry
            self._total_tokens += token_count
            self._add_to_indices(entry)

            # Auto-trim if over limits
            self._maybe_trim()

            # Persist to disk
            self._save_entry(entry)

        return entry_id

    def get(self, entry_id: str) -> LibraryEntry | None:
        """Get an entry by ID. Updates access tracking."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                # Try loading from disk
                entry = self._load_entry(entry_id)
                if entry is None:
                    return None
                self._entries[entry_id] = entry
                self._add_to_indices(entry)
                self._total_tokens += entry.token_count
            entry.touch()
            return entry

    def delete(self, entry_id: str) -> bool:
        """Delete an entry from the library."""
        with self._lock:
            if entry_id not in self._entries:
                return False
            self._remove_from_indices(entry_id)
            entry = self._entries.pop(entry_id)
            self._total_tokens -= entry.token_count
            # Remove from disk
            self._delete_entry_files(entry_id)
            return True

    # ── Lorebook-style injection ────────────────────────────────────────

    def inject(
        self,
        prompt: str,
        max_tokens: int | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        format: str = "system",
    ) -> str:
        """Inject relevant library entries into a prompt (lorebook-style).

        Scans the prompt for trigger keywords. Matching entries are
        injected into the context up to the token budget. This is the
        I/O-focused infinite context system — the model has access to
        the entire library, but only relevant entries are loaded.

        Args:
            prompt: the user's prompt
            max_tokens: max tokens to inject (default: self.injection_budget)
            category: filter by category (None = all)
            tags: filter by tags (None = all)
            format: injection format — "system" (prepend as system msg),
                    "prefix" (prepend to prompt), "suffix" (append to prompt)

        Returns:
            augmented prompt with relevant entries injected
        """
        budget = max_tokens or self.injection_budget
        if budget <= 0 or not self._entries:
            return prompt

        # Find matching entries via trigger keywords
        prompt_lower = prompt.lower()
        matched: list[LibraryEntry] = []

        with self._lock:
            for entry in self._entries.values():
                if not entry.enabled:
                    continue
                if category and entry.category != category:
                    continue
                if tags and not set(tags) & set(entry.tags):
                    continue

                # Check trigger keywords
                triggered = False
                for trigger in entry.triggers:
                    if trigger.lower() in prompt_lower:
                        triggered = True
                        break

                # If no triggers defined, use tags as triggers
                if not entry.triggers and entry.tags:
                    for tag in entry.tags:
                        if tag.lower() in prompt_lower:
                            triggered = True
                            break

                if triggered:
                    matched.append(entry)

        # Sort by priority (desc) then relevance score (desc)
        matched.sort(key=lambda e: (e.priority, e.relevance_score()), reverse=True)

        # Inject up to token budget
        injected_texts: list[str] = []
        tokens_used = 0

        for entry in matched:
            if tokens_used >= budget:
                break

            remaining = budget - tokens_used
            entry_tokens = entry.token_ids[:remaining]

            if not entry_tokens and entry.content:
                # No pre-tokenized cache, use content directly
                text = entry.content[:remaining * 4]  # rough char estimate
            elif self.tokenizer is not None:
                # Decode from cached tokens
                text = self.tokenizer.decode(entry_tokens, skip_special_tokens=True)
            else:
                text = entry.content[:remaining * 4]

            if text:
                injected_texts.append(f"[{entry.category.upper()}] {entry.description}\n{text}")
                tokens_used += len(entry_tokens)
                entry.touch()

        if not injected_texts:
            return prompt

        # Format injection
        injection_block = "\n\n---\n\n".join(injected_texts)

        if format == "system":
            return f"<|library_context|>\n{injection_block}\n<|/library_context|>\n\n{prompt}"
        elif format == "suffix":
            return f"{prompt}\n\n<|library_context|>\n{injection_block}\n<|/library_context|>"
        else:  # prefix
            return f"{injection_block}\n\n{prompt}"

    def get_injection_tokens(
        self,
        prompt: str,
        max_tokens: int | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[int]:
        """Get pre-tokenized injection tokens (no re-tokenization needed).

        Returns token IDs for relevant entries, ready to prepend to the
        prompt's token IDs. This is the fastest injection path — no
        string manipulation or re-tokenization.
        """
        budget = max_tokens or self.injection_budget
        if budget <= 0 or not self._entries:
            return []

        prompt_lower = prompt.lower()
        matched: list[LibraryEntry] = []

        with self._lock:
            for entry in self._entries.values():
                if not entry.enabled:
                    continue
                if category and entry.category != category:
                    continue
                if tags and not set(tags) & set(entry.tags):
                    continue
                for trigger in entry.triggers:
                    if trigger.lower() in prompt_lower:
                        matched.append(entry)
                        break
                else:
                    if not entry.triggers and entry.tags:
                        for tag in entry.tags:
                            if tag.lower() in prompt_lower:
                                matched.append(entry)
                                break

        matched.sort(key=lambda e: (e.priority, e.relevance_score()), reverse=True)

        result: list[int] = []
        for entry in matched:
            if len(result) >= budget:
                break
            remaining = budget - len(result)
            result.extend(entry.token_ids[:remaining])
            entry.touch()

        return result

    # ── Lookup and search ───────────────────────────────────────────────

    def lookup(
        self,
        tags: list[str] | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[LibraryEntry]:
        """Lookup entries by tags and/or category."""
        with self._lock:
            candidates: set[str] = set()

            if tags:
                for tag in tags:
                    if tag in self._tag_index:
                        candidates.update(self._tag_index[tag])
                if not candidates:
                    return []
            else:
                candidates = set(self._entries.keys())

            results = []
            for eid in candidates:
                entry = self._entries.get(eid)
                if entry is None or not entry.enabled:
                    continue
                if category and entry.category != category:
                    continue
                results.append(entry)

            results.sort(key=lambda e: e.relevance_score(), reverse=True)
            return results[:limit]

    def search(self, query: str, limit: int = 20) -> list[LibraryEntry]:
        """Full-text search across content, descriptions, and tags."""
        query_lower = query.lower()
        with self._lock:
            scored: list[tuple[float, LibraryEntry]] = []
            for entry in self._entries.values():
                if not entry.enabled:
                    continue
                score = 0.0
                # Search in description (highest weight)
                if query_lower in entry.description.lower():
                    score += 3.0
                # Search in tags
                for tag in entry.tags:
                    if query_lower in tag.lower():
                        score += 2.0
                # Search in content
                if query_lower in entry.content.lower():
                    score += 1.0
                # Search in triggers
                for trigger in entry.triggers:
                    if query_lower in trigger.lower():
                        score += 1.5
                if score > 0:
                    scored.append((score + entry.relevance_score() * 0.5, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [e for _, e in scored[:limit]]

    # ── Auto-trim and optimization ──────────────────────────────────────

    def _maybe_trim(self):
        """Auto-trim if over max_entries or max_total_tokens."""
        trimmed = False

        # Trim by entry count
        while len(self._entries) > self.max_entries:
            self._trim_least_relevant()
            trimmed = True

        # Trim by total tokens
        while self._total_tokens > self.max_total_tokens and len(self._entries) > 1:
            self._trim_least_relevant()
            trimmed = True

        if trimmed:
            self._save_index()

    def _trim_least_relevant(self):
        """Remove the least relevant entry."""
        if not self._entries:
            return

        # Find least relevant
        worst_id = min(
            self._entries.keys(),
            key=lambda eid: self._entries[eid].relevance_score()
        )
        self._remove_from_indices(worst_id)
        entry = self._entries.pop(worst_id)
        self._total_tokens -= entry.token_count
        self._delete_entry_files(worst_id)

    def optimize(self) -> dict:
        """Optimize the library: merge similar entries, re-index, clean up.

        Returns stats about the optimization.
        """
        with self._lock:
            n_before = len(self._entries)
            tokens_before = self._total_tokens

            # Remove disabled entries
            disabled = [eid for eid, e in self._entries.items() if not e.enabled]
            for eid in disabled:
                self._remove_from_indices(eid)
                entry = self._entries.pop(eid)
                self._total_tokens -= entry.token_count
                self._delete_entry_files(eid)

            # Merge entries with identical tags + category if content is similar
            merged = self._merge_similar()

            # Rebuild indices
            self._rebuild_indices()

            # Save index
            self._save_index()

            return {
                "entries_before": n_before,
                "entries_after": len(self._entries),
                "merged": merged,
                "removed_disabled": len(disabled),
                "tokens_before": tokens_before,
                "tokens_after": self._total_tokens,
            }

    def _merge_similar(self) -> int:
        """Merge entries with same tags + category. Returns merge count."""
        # Group by (category, frozenset(tags))
        groups: dict[tuple, list[str]] = {}
        for eid, entry in self._entries.items():
            key = (entry.category, frozenset(entry.tags))
            groups.setdefault(key, []).append(eid)

        merged = 0
        for key, eids in groups.items():
            if len(eids) < 2:
                continue
            # Merge entries with same key, keeping the most relevant
            eids.sort(
                key=lambda eid: self._entries[eid].relevance_score(),
                reverse=True
            )
            keeper = eids[0]
            keep_entry = self._entries[keeper]

            for eid in eids[1:]:
                other = self._entries[eid]
                # Only merge if descriptions are similar (simple check)
                if (keep_entry.description and other.description and
                    keep_entry.description.lower() == other.description.lower()):
                    # Merge content
                    keep_entry.content += "\n\n" + other.content
                    keep_entry.token_ids.extend(other.token_ids)
                    keep_entry.token_count = len(keep_entry.token_ids)
                    keep_entry.access_count += other.access_count
                    # Merge triggers
                    for t in other.triggers:
                        if t not in keep_entry.triggers:
                            keep_entry.triggers.append(t)

                    self._remove_from_indices(eid)
                    self._total_tokens -= other.token_count
                    self._entries.pop(eid)
                    self._delete_entry_files(eid)
                    merged += 1

            # Re-save the merged keeper
            self._save_entry(keep_entry)

        return merged

    # ── Indexing ────────────────────────────────────────────────────────

    def _add_to_indices(self, entry: LibraryEntry):
        """Add entry to tag and keyword indices."""
        for tag in entry.tags:
            self._tag_index.setdefault(tag, set()).add(entry.id)
        for trigger in entry.triggers:
            # Index each word in the trigger
            for word in re.findall(r'\w+', trigger.lower()):
                self._keyword_index.setdefault(word, set()).add(entry.id)

    def _remove_from_indices(self, entry_id: str):
        """Remove entry from all indices."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return
        for tag in entry.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(entry_id)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
        for trigger in entry.triggers:
            for word in re.findall(r'\w+', trigger.lower()):
                if word in self._keyword_index:
                    self._keyword_index[word].discard(entry_id)
                    if not self._keyword_index[word]:
                        del self._keyword_index[word]

    def _rebuild_indices(self):
        """Rebuild all indices from scratch."""
        self._tag_index.clear()
        self._keyword_index.clear()
        for entry in self._entries.values():
            self._add_to_indices(entry)

    # ── Disk persistence ────────────────────────────────────────────────

    def _save_entry(self, entry: LibraryEntry):
        """Save a single entry to disk."""
        entry_dir = self.path / "entries"
        entry_dir.mkdir(exist_ok=True)

        # Save metadata + content as JSON
        json_path = entry_dir / f"{entry.id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)

        # Save pre-tokenized token IDs as .npy for fast loading
        if entry.token_ids:
            tokens_path = entry_dir / f"{entry.id}.tokens"
            np.save(tokens_path, np.array(entry.token_ids, dtype=np.int32))

    @staticmethod
    def _entry_id_safe(entry_id: str) -> bool:
        """entry_id reaches the filesystem — reject anything that could
        traverse out of the entries dir (HTTP-supplied IDs included)."""
        import re
        return bool(re.fullmatch(r'[A-Za-z0-9_-]{1,128}', entry_id))

    def _load_entry(self, entry_id: str) -> LibraryEntry | None:
        """Load a single entry from disk."""
        if not self._entry_id_safe(entry_id):
            return None
        json_path = self.path / "entries" / f"{entry_id}.json"
        if not json_path.exists():
            return None

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            # Load pre-tokenized tokens
            tokens_path = self.path / "entries" / f"{entry_id}.tokens"
            if tokens_path.exists():
                token_ids = np.load(tokens_path).tolist()
                data["token_ids"] = token_ids
            else:
                data["token_ids"] = []

            return LibraryEntry.from_dict(data)
        except Exception:
            return None

    def _delete_entry_files(self, entry_id: str):
        """Delete entry files from disk."""
        if not self._entry_id_safe(entry_id):
            return
        for suffix in [".json", ".tokens"]:
            p = self.path / "entries" / f"{entry_id}{suffix}"
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    logger.debug("Failed to delete library entry file %s", p, exc_info=True)

    def _save_index(self):
        """Save the tag/keyword index to disk."""
        index_data = {
            "tag_index": {k: list(v) for k, v in self._tag_index.items()},
            "keyword_index": {k: list(v) for k, v in self._keyword_index.items()},
            "total_tokens": self._total_tokens,
            "entry_count": len(self._entries),
            "updated_at": time.time(),
        }
        index_path = self.path / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    def _save_meta(self):
        """Save library metadata."""
        meta = {
            "max_entries": self.max_entries,
            "max_total_tokens": self.max_total_tokens,
            "injection_budget": self.injection_budget,
            "total_tokens": self._total_tokens,
            "entry_count": len(self._entries),
            "categories": list(CATEGORIES),
            "updated_at": time.time(),
        }
        meta_path = self.path / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load(self):
        """Load the entire library from disk."""
        # Load index
        index_path = self.path / "index.json"
        if index_path.exists():
            try:
                with open(index_path, encoding="utf-8") as f:
                    idx = json.load(f)
                self._tag_index = {k: set(v) for k, v in idx.get("tag_index", {}).items()}
                self._keyword_index = {k: set(v) for k, v in idx.get("keyword_index", {}).items()}
            except Exception:
                logger.debug("Failed to load library index", exc_info=True)

        # Load all entries
        entries_dir = self.path / "entries"
        if entries_dir.exists():
            for json_file in entries_dir.glob("*.json"):
                entry_id = json_file.stem
                entry = self._load_entry(entry_id)
                if entry is not None:
                    self._entries[entry_id] = entry
                    self._total_tokens += entry.token_count

        # Rebuild indices if they were empty
        if not self._tag_index and self._entries:
            self._rebuild_indices()

    def flush(self):
        """Save all state to disk."""
        with self._lock:
            self._save_index()
            self._save_meta()
            for entry in self._entries.values():
                self._save_entry(entry)

    # ── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get library statistics."""
        with self._lock:
            by_category: dict[str, int] = {}
            for entry in self._entries.values():
                by_category[entry.category] = by_category.get(entry.category, 0) + 1

            return {
                "total_entries": len(self._entries),
                "total_tokens": self._total_tokens,
                "max_entries": self.max_entries,
                "max_total_tokens": self.max_total_tokens,
                "injection_budget": self.injection_budget,
                "by_category": by_category,
                "tag_count": len(self._tag_index),
                "keyword_count": len(self._keyword_index),
                "path": str(self.path),
            }

    def list_entries(
        self,
        category: str | None = None,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List entries with optional filtering. Returns summary dicts."""
        with self._lock:
            results = []
            for entry in self._entries.values():
                if category and entry.category != category:
                    continue
                if tag and tag not in entry.tags:
                    continue
                results.append({
                    "id": entry.id,
                    "description": entry.description,
                    "category": entry.category,
                    "tags": entry.tags,
                    "token_count": entry.token_count,
                    "priority": entry.priority,
                    "access_count": entry.access_count,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "enabled": entry.enabled,
                })

            results.sort(key=lambda e: e["last_accessed"], reverse=True)
            return results[offset:offset + limit]
