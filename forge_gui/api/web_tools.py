"""Keyless web tools for the ForgeAI agent harness.

Exposes ``web_search``, ``web_fetch``, ``wikipedia_search``, and
``arxiv_search`` to the model so the agent can do real-time research
without any API key. All HTTP is stdlib ``urllib`` GET-only — no
side-effecting requests, no auth, no dependencies.

The search/fetch primitives are shared with
``forge/self_play/discovery/discovery_tools.py`` via
``forge/web_primitives.py`` (critique F20 — extracted to avoid duplication).

Safety:
- Only ``http``/``https`` URLs are accepted for ``web_fetch`` —
  ``javascript:``, ``file:``, ``data:``, ``ftp:`` etc. are rejected
  before any request is made.
- SSRF guard (EC-213): ``fetch_url`` resolves the target host and refuses
  private/loopback/link-local/reserved addresses (localhost, 127.*,
  10.*, 172.16-31.*, 192.168.*, 169.254.*, ::1, metadata endpoints),
  and every redirect hop is re-validated.
- All requests are GET with a fixed browser User-Agent and a hard timeout.
- Output is capped (``MAX_FETCH_CHARS`` / ``MAX_SNIPPET``) to keep the
  tool result inside the engine's KV-cache budget, and results are
  wrapped in a tagged untrusted-content block (EC-163) so page text is
  treated as data, not instructions.
"""
from __future__ import annotations

from typing import Any

# NOTE: ``forge.web_primitives`` is stdlib-only, but importing it triggers
# ``forge/__init__.py`` → runtime configure → ``import torch`` (~1.3 s).
# The import is deferred into ``WebTools.execute()`` so GUI startup never
# pays the torch cost just to enumerate tool definitions.

# ── tool definitions (OpenAI function-calling shape) ────────────────────
_WEB_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the whole web — multi-engine, no API key. Returns "
                "{url, title, snippet} per result. Use web_fetch to read "
                "any result url (it returns full page text + links). For "
                "current news/headlines prefer news_search."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "n": {"type": "integer",
                          "description": "max results (default 5, max 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "news_search",
            "description": ("Search Google News for current headlines — use "
                            "this for 'the news', 'today's news', or any "
                            "current-events request. Returns real article "
                            "title, url, published date, and source. An "
                            "empty query returns today's top headlines."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": ("News topic to search; leave "
                                              "empty for today's top "
                                              "headlines")},
                    "n": {"type": "integer",
                          "description": "Max results (1-10, default 5)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Open any page on the internet and read it. Works on ANY "
                "http(s) URL — articles, docs, search results, link "
                "targets. Returns {title, text, links, final_url}: text "
                "is the page content with [label](url) links kept inline; "
                "links lists every link name + url on the page so you can "
                "follow them. Bare domains are OK (auto-https). Long pages "
                "page through with offset/next_offset."),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string",
                            "description": "URL to fetch — any http(s) page or bare domain"},
                    "max_chars": {"type": "integer",
                                  "description": "max chars to return (default 4000, max 16000)"},
                    "offset": {"type": "integer",
                               "description": "skip this many chars — page through long docs via next_offset (default 0)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Search Wikipedia (free API). Returns {title, snippet, url} "
                "summaries. Best for encyclopedic / factual background."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "n": {"type": "integer",
                          "description": "max results (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": (
                "Search arXiv for academic papers (free API). Returns "
                "{title, summary, url, published}. Best for ML/AI/math "
                "research papers."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "n": {"type": "integer",
                          "description": "max results (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
]


def web_tool_defs() -> list[dict]:
    """Return the OpenAI-style tool definition list for the web tools."""
    return [dict(d) for d in _WEB_TOOL_DEFS]


# ── manager ─────────────────────────────────────────────────────────────
class WebTools:
    """Stateless holder for the web tool implementations.

    Mirrors the ``TimeManager`` / ``BackupManager`` pattern so the
    :class:`ToolHarness` can hold a single instance and dispatch to it.
    All methods are safe to call from any thread (each opens its own
    ``urlopen`` with a hard timeout).
    """

    NAMES = frozenset(d["function"]["name"] for d in _WEB_TOOL_DEFS)

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a web tool call. Returns the raw result dict.

        On success the ``error`` key (which the primitives set to ``None``)
        is removed so the harness ``"error" in result`` ok-check works.
        """
        if not self.enabled:
            return {"error": "web tools disabled"}
        # Deferred import — see module docstring note (avoids the
        # forge/__init__ → torch chain at GUI startup).
        from forge.web_primitives import (
            DEFAULT_N,
            MAX_FETCH_CHARS,
            arxiv_search,
            ddg_search,
            fetch_url,
            google_news_search,
            wikipedia_search,
            wrap_untrusted,
        )
        n = int(args.get("n", DEFAULT_N))
        # clamp n to a sane range to avoid huge responses
        n = max(1, min(n, 10))
        if name == "web_search":
            res = ddg_search(args.get("query", ""), n=n)
        elif name == "news_search":
            res = google_news_search(args.get("query", ""), n=n)
        elif name == "web_fetch":
            max_chars = int(args.get("max_chars", MAX_FETCH_CHARS))
            max_chars = max(200, min(max_chars, 16000))
            offset = max(0, int(args.get("offset", 0) or 0))
            res = fetch_url(args.get("url", ""), max_chars=max_chars,
                            offset=offset)
        elif name == "wikipedia_search":
            res = wikipedia_search(args.get("query", ""), n=n)
        elif name == "arxiv_search":
            res = arxiv_search(args.get("query", ""), n=n)
        else:
            return {"error": f"unknown web tool: {name}"}
        # normalize: drop a falsy error so the harness ok-check passes
        if not res.get("error"):
            res.pop("error", None)
        else:
            return res
        # Page/search content is untrusted external data — tag it before
        # it enters the transcript (EC-163).
        return wrap_untrusted(res, source=name)
