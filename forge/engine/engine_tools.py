"""Built-in tool registry and execution engine for ForgeEngine.

Provides the LLM with tools to use ForgeEngine features during generation:
  - Library: save, search, lookup, get, delete, update, stats, optimize, config
  - Hot-swap: change KV cache, decoding, context limit, generation params
  - Batch generation: generate for multiple prompts
  - Engine info: get settings, stats, VRAM info
  - Adaptive thinking: generate with think/no-think decision
  - Math: safe expression evaluation (math_eval, calc)
  - Random: random numbers, chance operations (coin flip, dice, choice)
  - Web: Tavily search, Exa semantic search, Firecrawl scrape
  - Files: read, write, edit, move, rename, list, delete (workspace-scoped)

Tools use Qwen-format tool calls (JSON wrapped in special tokens).
The execution engine runs tools server-side and feeds results back to
the model in a multi-turn agentic loop.

Note: This is the INFERENCE tool registry, used during model serving.
The self-play discovery loop has its own separate tool registry in
forge/self_play/discovery/discovery_tools.py — the two are not mixed.

Usage:
    from forge.engine.engine_tools import EngineToolRegistry

    registry = EngineToolRegistry(engine)
    tools = registry.get_tool_defs()  # pass to model as tool definitions
    result = registry.execute(name, arguments)  # execute a tool call
"""
from __future__ import annotations

import ast as _ast
import json
import math as _math
import operator as _op
import random as _random
from collections.abc import Callable
from typing import Any

from forge.engine.library import CATEGORIES
from forge.engine.tool_security import SecurityDecision, ToolSecurityManager
from forge.web_primitives import wrap_untrusted

# ── Tool tiers & annotations (EC-158) ────────────────────────────────────────
# MCP-style read/write tiers drive the approval policy in _security_check:
#   "read"        — read-only; auto-approvable EXCEPT blacklisted/credential
#                   paths and 'denied' zones (checked by ToolSecurityManager)
#   "write"       — mutating; gated by sandbox access rules + content scans
#   "destructive" — irreversible-ish; always requires permission unless
#                   auto_mode == "allow" (delete/move/rename)
#   "network"     — reaches outside the workspace; domain allow/deny list
TOOL_TIERS: dict[str, str] = {
    # library
    "library_save": "write", "library_update": "write",
    "library_delete": "destructive",
    "library_search": "read", "library_lookup": "read",
    "library_get": "read", "library_stats": "read",
    "library_optimize": "write", "library_set_config": "write",
    # hot-swap / engine info / generation
    "engine_set_kv_cache": "write", "engine_set_decoding": "write",
    "engine_set_context_limit": "write",
    "engine_enable_infinite_context": "write",
    "engine_set_generation_params": "write", "engine_set_feature": "write",
    "engine_apply_changes": "write",
    "engine_get_settings": "read", "engine_get_stats": "read",
    "engine_get_pending": "read",
    "engine_batch_generate": "write", "engine_generate_adaptive": "write",
    # math / random
    "math_eval": "read", "calc": "read",
    "random_number": "read", "chance": "read",
    # web
    "web_search": "network", "web_search_semantic": "network",
    "web_scrape": "network",
    # files
    "file_read": "read", "file_list": "read",
    "file_write": "write", "file_edit": "write",
    "file_move": "destructive", "file_rename": "destructive",
    "file_delete": "destructive",
    # security introspection
    "scan_script": "read", "security_get_config": "read",
    "security_get_pending": "read",
}

# Tools whose output embeds untrusted external content (web pages, file
# contents). Results are wrapped in a tagged untrusted-content block before
# entering the transcript so embedded instructions are treated as data,
# not commands (EC-163 — indirect prompt-injection mitigation).
UNTRUSTED_OUTPUT_TOOLS = {
    "web_search", "web_search_semantic", "web_scrape",
    "file_read", "file_list",
}

# ── Tool definition schema ───────────────────────────────────────────────────

def _make_tool_def(
    name: str,
    description: str,
    parameters: dict[str, Any],
    annotations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a tool definition in the format the model expects."""
    tool_def = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "tier": TOOL_TIERS.get(name, "read"),
    }
    if annotations:
        tool_def["annotations"] = annotations
    return tool_def


# ── Built-in tool definitions ────────────────────────────────────────────────

def get_builtin_tool_defs() -> list[dict[str, Any]]:
    """Get all built-in tool definitions.

    These are the tools the LLM can call to interact with ForgeEngine
    features during generation.
    """
    return [

        # ── Library tools ──────────────────────────────────────────────

        _make_tool_def(
            "library_save",
            "Save an entry to the Library knowledge base. Use this to record "
            "failures, wins, research findings, or common data. Content is "
            "pre-tokenized for instant injection later.",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The text content to store"},
                    "category": {
                        "type": "string",
                        "enum": list(CATEGORIES),
                        "description": "Entry category: 'failure' (failed approaches), "
                                       "'win' (successful solutions), 'research' (findings), "
                                       "'common_data' (reference data), 'custom' (other)"
                    },
                    "tags": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Tags for categorization and lookup"
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description (1-2 sentences) for search"
                    },
                    "triggers": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Keywords that activate lorebook injection. "
                                       "When a future prompt contains these keywords, "
                                       "this entry will be injected into context."
                    },
                    "priority": {
                        "type": "integer", "default": 0,
                        "description": "Higher = injected first (0-100)"
                    },
                },
                "required": ["content", "category"],
            }
        ),

        _make_tool_def(
            "library_search",
            "Search the Library for relevant entries by full-text query. "
            "Returns matching entries with previews.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10,
                              "description": "Max results to return"}
                },
                "required": ["query"],
            }
        ),

        _make_tool_def(
            "library_lookup",
            "Lookup Library entries by tags and/or category. "
            "Returns entries sorted by relevance.",
            {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Tags to filter by"
                    },
                    "category": {
                        "type": "string", "enum": list(CATEGORIES),
                        "description": "Filter by category"
                    },
                    "limit": {"type": "integer", "default": 20},
                },
            }
        ),

        _make_tool_def(
            "library_get",
            "Get a single Library entry by ID, including full content.",
            {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "string", "description": "Entry ID"}
                },
                "required": ["entry_id"],
            }
        ),

        _make_tool_def(
            "library_delete",
            "Delete a Library entry by ID.",
            {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "string", "description": "Entry ID to delete"}
                },
                "required": ["entry_id"],
            }
        ),

        _make_tool_def(
            "library_stats",
            "Get Library statistics: total entries, tokens, categories, etc.",
            {"type": "object", "properties": {}}
        ),

        _make_tool_def(
            "library_optimize",
            "Optimize the Library: merge similar entries, remove disabled, "
            "rebuild indices. Returns optimization stats.",
            {"type": "object", "properties": {}}
        ),

        _make_tool_def(
            "library_set_config",
            "Configure Library injection: enable/disable, set token budget.",
            {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean",
                                "description": "Enable/disable lorebook injection"},
                    "injection_budget": {"type": "integer",
                                         "description": "Max tokens to inject per request"}
                },
            }
        ),

        # ── Hot-swap tools ─────────────────────────────────────────────

        _make_tool_def(
            "engine_set_kv_cache",
            "Hot-swap the KV cache strategy at runtime. Takes effect on "
            "the next generation. Options: standard, paged, streaming, "
            "snapkv, snapkv_4bit, kvzip, s4r, cpu_offload, hadamard_int4.",
            {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string",
                                 "description": "KV cache strategy name"}
                },
                "required": ["strategy"],
            }
        ),

        _make_tool_def(
            "engine_set_decoding",
            "Hot-swap the decoding strategy at runtime. Options: standard, "
            "speculative, eagle3, medusa, mtp_selfspec, dspark.",
            {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string",
                                 "description": "Decoding strategy name"}
                },
                "required": ["strategy"],
            }
        ),

        _make_tool_def(
            "engine_set_context_limit",
            "Set the maximum context window in tokens. Use a large number "
            "(e.g. 1000000) for effectively infinite context with eviction.",
            {
                "type": "object",
                "properties": {
                    "max_tokens": {"type": "integer",
                                   "description": "Max context tokens"}
                },
                "required": ["max_tokens"],
            }
        ),

        _make_tool_def(
            "engine_enable_infinite_context",
            "Enable infinite context mode with KV cache eviction. "
            "The engine maintains unbounded context within the VRAM budget.",
            {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "budget": {"type": "integer", "default": 100000,
                               "description": "KV cache token budget before eviction"}
                },
            }
        ),

        _make_tool_def(
            "engine_set_generation_params",
            "Set default generation parameters (temperature, max_tokens, "
            "top_p, top_k, repetition_penalty). Overridable per-request.",
            {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number",
                                    "description": "0.0-2.0, 0=greedy"},
                    "max_tokens": {"type": "integer",
                                   "description": "Default max generation length"},
                    "top_p": {"type": "number", "description": "Nucleus sampling"},
                    "top_k": {"type": "integer", "description": "Top-k sampling"},
                    "repetition_penalty": {"type": "number",
                                           "description": "1.0=none, >1=penalize"}
                },
            }
        ),

        _make_tool_def(
            "engine_set_feature",
            "Toggle a runtime feature flag (e.g. use_prefix_cache, "
            "use_compile, use_chunked_prefill, use_triton_conv).",
            {
                "type": "object",
                "properties": {
                    "flag": {"type": "string",
                             "description": "Feature flag name (e.g. 'use_prefix_cache')"},
                    "enabled": {"type": "boolean"}
                },
                "required": ["flag", "enabled"],
            }
        ),

        _make_tool_def(
            "engine_apply_changes",
            "Force-apply pending hot-swap changes immediately (normally "
            "applied lazily on next generation).",
            {"type": "object", "properties": {}}
        ),

        # ── Engine info tools ──────────────────────────────────────────

        _make_tool_def(
            "engine_get_settings",
            "Get current engine settings (KV cache, decoding, context limit, "
            "generation params, feature flags, etc.).",
            {"type": "object", "properties": {}}
        ),

        _make_tool_def(
            "engine_get_stats",
            "Get engine statistics: generation count, tokens generated, "
            "VRAM usage, etc.",
            {"type": "object", "properties": {}}
        ),

        _make_tool_def(
            "engine_get_pending",
            "Check if there are pending hot-swap changes not yet applied.",
            {"type": "object", "properties": {}}
        ),

        # ── Batch generation ───────────────────────────────────────────

        _make_tool_def(
            "engine_batch_generate",
            "Generate text for multiple prompts in a single batched forward "
            "pass (3-5x faster than serial). Useful for parallel exploration.",
            {
                "type": "object",
                "properties": {
                    "prompts": {
                        "type": "array", "items": {"type": "string"},
                        "description": "List of prompts to generate"
                    },
                    "max_tokens": {"type": "integer", "default": 256},
                    "temperature": {"type": "number", "default": 0.0},
                },
                "required": ["prompts"],
            }
        ),

        # ── Adaptive thinking ──────────────────────────────────────────

        _make_tool_def(
            "engine_generate_adaptive",
            "Generate with adaptive thinking: the model decides whether to "
            "think (reason step-by-step) or answer directly based on "
            "problem difficulty. Saves ~50% tokens on easy problems.",
            {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt"},
                    "think_max_tokens": {"type": "integer", "default": 512},
                    "no_think_max_tokens": {"type": "integer", "default": 256},
                    "temperature": {"type": "number", "default": 0.0},
                },
                "required": ["prompt"],
            }
        ),

        # ── Math / calculation tools ───────────────────────────────────

        _make_tool_def(
            "math_eval",
            "Evaluate a mathematical expression safely. Supports +, -, *, /, "
            "**, %, //, parentheses, sqrt(), sin(), cos(), tan(), log(), "
            "exp(), abs(), round(), min(), max(), and constants pi, e. "
            "Example: math_eval('sqrt(144) + 2**10') -> 1036.0",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string",
                                   "description": "Math expression to evaluate"}
                },
                "required": ["expression"],
            }
        ),

        _make_tool_def(
            "calc",
            "Quick calculator: evaluate an expression and return the result. "
            "Same as math_eval but with a shorter name for simple arithmetic. "
            "Example: calc('2*3+4') -> 10",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string",
                                   "description": "Expression to calculate"}
                },
                "required": ["expression"],
            }
        ),

        _make_tool_def(
            "random_number",
            "Generate random number(s). Supports integers, floats, and ranges. "
            "Example: random_number(min=1, max=100, count=5) -> [42, 7, 83, 15, 91]",
            {
                "type": "object",
                "properties": {
                    "min": {"type": "number", "default": 0,
                            "description": "Minimum value (inclusive)"},
                    "max": {"type": "number", "default": 1,
                            "description": "Maximum value (inclusive for int, exclusive for float)"},
                    "count": {"type": "integer", "default": 1,
                              "description": "Number of values to generate"},
                    "integer": {"type": "boolean", "default": True,
                                "description": "True for integers, False for floats"},
                    "seed": {"type": "integer",
                             "description": "Optional seed for reproducibility"},
                },
            }
        ),

        _make_tool_def(
            "chance",
            "Probability and random choice operations. Modes: "
            "'coin_flip' (heads/tails), 'dice' (roll N-sided dice), "
            "'choice' (pick one from list), 'shuffle' (shuffle a list), "
            "'weighted' (weighted random pick). "
            "Example: chance(mode='dice', sides=20) -> {roll: 14}",
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["coin_flip", "dice", "choice", "shuffle", "weighted"],
                        "description": "Operation mode"
                    },
                    "sides": {"type": "integer", "default": 6,
                              "description": "Dice sides (for 'dice' mode)"},
                    "count": {"type": "integer", "default": 1,
                              "description": "Number of dice/coins"},
                    "items": {
                        "type": "array",
                        "description": "List to pick from (for 'choice'/'shuffle'/'weighted')"
                    },
                    "weights": {
                        "type": "array", "items": {"type": "number"},
                        "description": "Weights for 'weighted' mode (must match items length)"
                    },
                    "seed": {"type": "integer",
                             "description": "Optional seed for reproducibility"},
                },
                "required": ["mode"],
            }
        ),

        # ── Web search / scrape tools ──────────────────────────────────

        _make_tool_def(
            "web_search",
            "Search the web using Tavily API (agent-native search). Returns "
            "content snippets ready for reasoning. Best for real-time info, "
            "documentation, news, and general queries.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5,
                                    "description": "Max results (1-10)"},
                    "search_depth": {
                        "type": "string", "enum": ["basic", "advanced"],
                        "default": "basic",
                        "description": "'advanced' for deeper results (slower)"
                    },
                    "include_answer": {"type": "boolean", "default": True,
                                       "description": "Include synthesized answer"},
                },
                "required": ["query"],
            },
            annotations={"readOnlyHint": True, "openWorldHint": True},
        ),

        _make_tool_def(
            "web_search_semantic",
            "Search the web using Exa API (neural/semantic search). Finds "
            "pages by meaning, not just keywords. Best for finding obscure "
            "documentation, code examples, or conceptually similar content. "
            "Supports 'auto', 'fast', 'deep' search types.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5},
                    "search_type": {
                        "type": "string",
                        "enum": ["auto", "fast", "instant", "deep-lite", "deep"],
                        "default": "auto",
                        "description": "Search depth (auto=balanced, deep=thorough)"
                    },
                    "include_domains": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Restrict to these domains (e.g. ['arxiv.org'])"
                    },
                },
                "required": ["query"],
            },
            annotations={"readOnlyHint": True, "openWorldHint": True},
        ),

        _make_tool_def(
            "web_scrape",
            "Scrape a single URL and return clean text content using Firecrawl. "
            "Best for reading full articles, documentation pages, or any URL "
            "you found via web_search. Returns markdown-formatted text.",
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "max_chars": {"type": "integer", "default": 10000,
                                  "description": "Max characters to return"},
                    "format": {
                        "type": "string", "enum": ["markdown", "html", "text"],
                        "default": "markdown",
                        "description": "Output format"
                    },
                },
                "required": ["url"],
            },
            annotations={"readOnlyHint": True, "openWorldHint": True},
        ),

        # ── File operation tools ───────────────────────────────────────

        _make_tool_def(
            "file_read",
            "Read the contents of a file. Path is relative to the project "
            "workspace root. Returns file content as text.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "File path (relative to workspace or absolute within workspace)"},
                    "max_chars": {"type": "integer", "default": 50000,
                                  "description": "Max characters to return"}
                },
                "required": ["path"],
            },
            annotations={"readOnlyHint": True},
        ),

        _make_tool_def(
            "file_write",
            "Write content to a file (creates or overwrites). Path is relative "
            "to the project workspace root. Creates parent directories if needed.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                    "append": {"type": "boolean", "default": False,
                               "description": "Append to file instead of overwriting"}
                },
                "required": ["path", "content"],
            },
            annotations={"readOnlyHint": False, "destructiveHint": False,
                         "idempotentHint": True},
        ),

        _make_tool_def(
            "file_edit",
            "Edit a file by replacing old_string with new_string. The "
            "old_string must be unique in the file. Use file_read first to "
            "see the exact content.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_string": {"type": "string",
                                   "description": "Exact text to find (must be unique)"},
                    "new_string": {"type": "string",
                                   "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            annotations={"readOnlyHint": False, "destructiveHint": False,
                         "idempotentHint": True},
        ),

        _make_tool_def(
            "file_move",
            "Move a file from source to destination. Creates parent "
            "directories at the destination if needed.",
            {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source file path"},
                    "destination": {"type": "string", "description": "Destination path"},
                },
                "required": ["source", "destination"],
            },
            annotations={"readOnlyHint": False, "destructiveHint": True},
        ),

        _make_tool_def(
            "file_rename",
            "Rename a file or directory. The parent directory stays the same; "
            "only the name changes.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Path to file or directory"},
                    "new_name": {"type": "string",
                                 "description": "New name (not full path, just the name)"},
                },
                "required": ["path", "new_name"],
            },
            annotations={"readOnlyHint": False, "destructiveHint": True},
        ),

        _make_tool_def(
            "file_list",
            "List files and directories in a given path. Returns names, "
            "types (file/dir), and sizes.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": ".",
                             "description": "Directory path (default: workspace root)"},
                    "recursive": {"type": "boolean", "default": False,
                                  "description": "List recursively"},
                    "pattern": {"type": "string",
                                "description": "Glob pattern filter (e.g. '*.py')"},
                },
            },
            annotations={"readOnlyHint": True},
        ),

        _make_tool_def(
            "file_delete",
            "Delete a file. Use with caution — this is irreversible. "
            "Cannot delete directories.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to delete"},
                    "confirm": {"type": "boolean", "default": False,
                                "description": "Must be true to actually delete"},
                },
                "required": ["path", "confirm"],
            },
            annotations={"readOnlyHint": False, "destructiveHint": True},
        ),

        # ── Library update tool ────────────────────────────────────────

        _make_tool_def(
            "library_update",
            "Update an existing Library entry. Can change content, tags, "
            "description, triggers, priority, or category. Only provided "
            "fields are updated; omitted fields keep their current value.",
            {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "string", "description": "Entry ID to update"},
                    "content": {"type": "string", "description": "New content (optional)"},
                    "category": {"type": "string", "enum": list(CATEGORIES),
                                 "description": "New category (optional)"},
                    "tags": {"type": "array", "items": {"type": "string"},
                             "description": "New tags (optional)"},
                    "description": {"type": "string",
                                    "description": "New description (optional)"},
                    "triggers": {"type": "array", "items": {"type": "string"},
                                 "description": "New trigger keywords (optional)"},
                    "priority": {"type": "integer",
                                 "description": "New priority 0-100 (optional)"},
                },
                "required": ["entry_id"],
            }
        ),

        # ── Security / script scan tools ───────────────────────────────

        _make_tool_def(
            "scan_script",
            "Pre-scan a Python script for dangerous imports, risky command "
            "patterns, and escape attempts before writing it to disk. "
            "Returns a verdict (allow/needs_permission/refuse) and detailed "
            "findings. Always call this before file_write on .py files to "
            "avoid permission prompts.",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string",
                                "description": "Python script content to scan"}
                },
                "required": ["content"],
            }
        ),

        _make_tool_def(
            "security_get_config",
            "Get the current security configuration: write whitelist, "
            "write blacklist, file blacklist, website whitelist/blacklist, "
            "auto mode, and pending permission requests.",
            {"type": "object", "properties": {}}
        ),

        _make_tool_def(
            "security_get_pending",
            "Get pending permission requests (operations that need user "
            "approval before proceeding).",
            {"type": "object", "properties": {}}
        ),
    ]


# ── Tool execution engine ────────────────────────────────────────────────────

class EngineToolRegistry:
    """Registry of built-in tools + execution engine.

    Wraps a ForgeEngine and provides:
    - get_tool_defs(): tool definitions for the model
    - execute(name, arguments): run a tool and return the result
    - execute_calls(tool_calls): batch-execute parsed tool calls

    The tools give the LLM direct access to Library, hot-swap, batch
    generation, and engine introspection.
    """

    def __init__(self, engine, security: ToolSecurityManager | None = None):
        self.engine = engine
        self._handlers: dict[str, Callable] = {}
        # Security manager — controls file writes, website access, script scanning
        if security is not None:
            self.security = security
        else:
            from pathlib import Path
            workspace = str(Path(__file__).resolve().parents[2])
            self.security = ToolSecurityManager(workspace_root=workspace)
        self._register_handlers()

    def _register_handlers(self):
        """Register all built-in tool handlers."""
        h = self._handlers

        # ── Library ──
        h["library_save"] = self._library_save
        h["library_search"] = self._library_search
        h["library_lookup"] = self._library_lookup
        h["library_get"] = self._library_get
        h["library_delete"] = self._library_delete
        h["library_stats"] = self._library_stats
        h["library_optimize"] = self._library_optimize
        h["library_set_config"] = self._library_set_config
        h["library_update"] = self._library_update

        # ── Hot-swap ──
        h["engine_set_kv_cache"] = self._set_kv_cache
        h["engine_set_decoding"] = self._set_decoding
        h["engine_set_context_limit"] = self._set_context_limit
        h["engine_enable_infinite_context"] = self._enable_infinite_context
        h["engine_set_generation_params"] = self._set_generation_params
        h["engine_set_feature"] = self._set_feature
        h["engine_apply_changes"] = self._apply_changes

        # ── Engine info ──
        h["engine_get_settings"] = self._get_settings
        h["engine_get_stats"] = self._get_stats
        h["engine_get_pending"] = self._get_pending

        # ── Batch generation ──
        h["engine_batch_generate"] = self._batch_generate

        # ── Adaptive thinking ──
        h["engine_generate_adaptive"] = self._generate_adaptive

        # ── Math / calculation ──
        h["math_eval"] = self._math_eval
        h["calc"] = self._calc
        h["random_number"] = self._random_number
        h["chance"] = self._chance

        # ── Web search / scrape ──
        h["web_search"] = self._web_search
        h["web_search_semantic"] = self._web_search_semantic
        h["web_scrape"] = self._web_scrape

        # ── File operations ──
        h["file_read"] = self._file_read
        h["file_write"] = self._file_write
        h["file_edit"] = self._file_edit
        h["file_move"] = self._file_move
        h["file_rename"] = self._file_rename
        h["file_list"] = self._file_list
        h["file_delete"] = self._file_delete

        # ── Security / script scan ──
        h["scan_script"] = self._scan_script
        h["security_get_config"] = self._security_get_config
        h["security_get_pending"] = self._security_get_pending

    def get_tool_defs(self) -> list[dict[str, Any]]:
        """Get tool definitions for the model."""
        return get_builtin_tool_defs()

    def get_tool_names(self) -> set[str]:
        """Get set of available tool names."""
        return set(self._handlers.keys())

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call.

        Args:
            name: tool name (e.g. "library_save")
            arguments: tool arguments as a dict

        Returns:
            dict with "status" ("success", "error", "needs_permission", or "refused")
            and result data
        """
        handler = self._handlers.get(name)
        if handler is None:
            return {"status": "error", "error": f"Unknown tool: {name}"}

        # Pre-execution security check (for file/web tools)
        sec_check = self._security_check(name, arguments)
        if sec_check.refused:
            return {"status": "refused", "error": sec_check.reason,
                    "details": sec_check.details}
        if sec_check.needs_permission:
            return {"status": "needs_permission", "error": sec_check.reason,
                    "details": sec_check.details}

        try:
            result = handler(arguments)
        except Exception as e:
            return {"status": "error", "error": str(e)}

        # Tag outputs that embed untrusted external content before they
        # are serialized into the transcript (EC-163).
        if name in UNTRUSTED_OUTPUT_TOOLS:
            result = wrap_untrusted(result, source=name)
        return {"status": "success", "result": result}

    def tool_tier(self, name: str) -> str:
        """Return the read/write annotation tier for a tool (EC-158)."""
        return TOOL_TIERS.get(name, "read")

    def _security_check(self, name: str, args: dict) -> SecurityDecision:
        """Run security checks before executing a tool.

        Returns a SecurityDecision. If refused or needs_permission,
        the tool is not executed.
        """
        sec = self.security

        # File read operations — gated like everything else (EC-097):
        # 'denied' zones, credential/secret blacklist, outside-workspace.
        # Auto-approve reads only AFTER these checks pass.
        if name == "file_read":
            path = args.get("path", "")
            return sec.check_file_read(path)

        if name == "file_list":
            path = args.get("path", ".")
            return sec.check_file_read(path)

        # File write operations
        if name == "file_write":
            path = args.get("path", "")
            content = args.get("content", "")
            return sec.check_file_write(path, content)

        if name == "file_edit":
            path = args.get("path", "")
            new_str = args.get("new_string", "")
            # Check write permission + scan new_string for risky patterns
            decision = sec.check_file_write(path, new_str)
            return decision

        if name == "file_delete":
            path = args.get("path", "")
            return sec.check_file_delete(path)

        if name == "file_move":
            src = args.get("source", "")
            dst = args.get("destination", "")
            return sec.check_file_move(src, dst)

        if name == "file_rename":
            path = args.get("path", "")
            new_name = args.get("new_name", "")
            return sec.check_file_rename(path, new_name)

        # Web operations — check target URL/domain
        if name == "web_scrape":
            url = args.get("url", "")
            return sec.check_website(url)

        if name == "web_search":
            # Tavily search — check if query contains a domain we should filter
            # Search queries themselves are allowed; results are filtered post-hoc
            return SecurityDecision(allowed=True)

        if name == "web_search_semantic":
            # Exa search — check include_domains if specified
            include_domains = args.get("include_domains", [])
            for domain in include_domains:
                decision = sec.check_website(domain)
                if not decision.allowed:
                    return decision
            return SecurityDecision(allowed=True)

        # All other tools (library, engine, math, random) — no security check needed
        return SecurityDecision(allowed=True)

    def execute_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute multiple tool calls and return results.

        Args:
            tool_calls: list of {"name": str, "arguments": dict}

        Returns:
            list of {"name": str, "status": str, "result"/"error": ...}
        """
        results = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            if isinstance(args, str):
                # Model emitted raw JSON text — parse it; malformed JSON is
                # an error, NOT a silent {} fallback (EC-119).
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as e:
                    results.append({
                        "name": name,
                        "status": "error",
                        "error": f"Malformed JSON arguments for '{name}': {e}",
                    })
                    continue
            if not isinstance(args, dict):
                results.append({
                    "name": name,
                    "status": "error",
                    "error": f"Tool arguments for '{name}' must be a JSON object",
                })
                continue
            result = self.execute(name, args)
            result["name"] = name
            results.append(result)
        return results

    # ── Library handlers ──────────────────────────────────────────────

    def _library_save(self, args: dict) -> dict:
        entry_id = self.engine.library_save(
            content=args["content"],
            category=args.get("category", "custom"),
            tags=args.get("tags"),
            description=args.get("description", ""),
            triggers=args.get("triggers"),
            priority=args.get("priority", 0),
        )
        return {"entry_id": entry_id, "saved": True}

    def _library_search(self, args: dict) -> dict:
        entries = self.engine.library_search(
            args["query"], limit=args.get("limit", 10))
        return {"results": [
            {"id": e.id, "description": e.description,
             "category": e.category, "tags": e.tags,
             "content_preview": e.content[:300],
             "token_count": e.token_count, "priority": e.priority}
            for e in entries
        ]}

    def _library_lookup(self, args: dict) -> dict:
        entries = self.engine.library_lookup(
            tags=args.get("tags"), category=args.get("category"),
            limit=args.get("limit", 20))
        return {"entries": [
            {"id": e.id, "description": e.description,
             "category": e.category, "tags": e.tags,
             "token_count": e.token_count, "priority": e.priority,
             "access_count": e.access_count}
            for e in entries
        ]}

    def _library_get(self, args: dict) -> dict:
        entry = self.engine.library.get(args["entry_id"])
        if entry is None:
            return {"error": "Entry not found"}
        return {
            "id": entry.id, "content": entry.content,
            "description": entry.description, "category": entry.category,
            "tags": entry.tags, "triggers": entry.triggers,
            "priority": entry.priority, "token_count": entry.token_count,
            "access_count": entry.access_count,
        }

    def _library_delete(self, args: dict) -> dict:
        deleted = self.engine.library.delete(args["entry_id"])
        return {"deleted": deleted, "entry_id": args["entry_id"]}

    def _library_stats(self, args: dict) -> dict:
        return self.engine.library_stats()

    def _library_optimize(self, args: dict) -> dict:
        return self.engine.library_optimize()

    def _library_set_config(self, args: dict) -> dict:
        if "enabled" in args:
            self.engine.library_set_enabled(args["enabled"])
        if "injection_budget" in args:
            self.engine.library_set_budget(args["injection_budget"])
        return {
            "enabled": self.engine._library_enabled,
            "injection_budget": self.engine._library_injection_budget,
        }

    # ── Hot-swap handlers ─────────────────────────────────────────────

    def _set_kv_cache(self, args: dict) -> dict:
        self.engine.hotswap.set_kv_cache(args["strategy"])
        return {"kv_cache": args["strategy"], "pending": True}

    def _set_decoding(self, args: dict) -> dict:
        self.engine.hotswap.set_decoding(args["strategy"])
        return {"decoding": args["strategy"], "pending": True}

    def _set_context_limit(self, args: dict) -> dict:
        self.engine.hotswap.set_context_limit(args["max_tokens"])
        return {"max_context_tokens": args["max_tokens"], "pending": True}

    def _enable_infinite_context(self, args: dict) -> dict:
        enabled = args.get("enabled", True)
        budget = args.get("budget", 100_000)
        self.engine.hotswap.set_infinite_context(enabled, budget)
        return {"infinite_context": enabled, "budget": budget, "pending": True}

    def _set_generation_params(self, args: dict) -> dict:
        self.engine.hotswap.set_generation_defaults(
            temperature=args.get("temperature"),
            max_tokens=args.get("max_tokens"),
            top_p=args.get("top_p"),
            top_k=args.get("top_k"),
            repetition_penalty=args.get("repetition_penalty"),
        )
        return {"updated": True, "pending": True}

    def _set_feature(self, args: dict) -> dict:
        self.engine.hotswap.set_feature(args["flag"], args["enabled"])
        return {args["flag"]: args["enabled"], "pending": True}

    def _apply_changes(self, args: dict) -> dict:
        applied = self.engine.hotswap.apply_pending()
        return {"applied": applied, "settings": self.engine.hotswap.get_settings()}

    # ── Engine info handlers ──────────────────────────────────────────

    def _get_settings(self, args: dict) -> dict:
        return self.engine.hotswap.get_settings()

    def _get_stats(self, args: dict) -> dict:
        return {
            "generation_count": self.engine.generation_count,
            "total_tokens_generated": self.engine.total_tokens_generated,
            "library": self.engine.library_stats(),
            "decoding": getattr(self.engine.decoding, "name", "unknown"),
            "acceleration": self.engine.acceleration,
            "quantize": self.engine.quantize,
        }

    def _get_pending(self, args: dict) -> dict:
        return {
            "has_pending": self.engine.hotswap.has_pending(),
            "pending_changes": self.engine.hotswap.get_pending_changes(),
        }

    # ── Batch generation ──────────────────────────────────────────────

    def _batch_generate(self, args: dict) -> dict:
        results = self.engine.generate_batch(
            args["prompts"],
            max_new_tokens=args.get("max_tokens", 256),
            temperature=args.get("temperature", 0.0),
        )
        return {"results": results, "count": len(results)}

    # ── Adaptive thinking ─────────────────────────────────────────────

    def _generate_adaptive(self, args: dict) -> dict:
        result, did_think = self.engine.generate_adaptive(
            args["prompt"],
            think_max_tokens=args.get("think_max_tokens", 512),
            no_think_max_tokens=args.get("no_think_max_tokens", 256),
            temperature=args.get("temperature", 0.0),
        )
        return {"result": result, "did_think": did_think}

    # ── Library update ────────────────────────────────────────────────

    def _library_update(self, args: dict) -> dict:
        entry_id = args["entry_id"]
        entry = self.engine.library.get(entry_id)
        if entry is None:
            return {"error": f"Entry not found: {entry_id}"}
        lib = self.engine.library
        with lib._lock:
            # Update provided fields
            if "content" in args:
                entry.content = args["content"]
                if lib.tokenizer is not None:
                    entry.token_ids = lib.tokenizer(
                        args["content"], add_special_tokens=False).input_ids
                    entry.token_count = len(entry.token_ids)
                else:
                    entry.token_count = len(args["content"]) // 4
            if "category" in args:
                entry.category = args["category"] if args["category"] in CATEGORIES else "custom"
            if "tags" in args:
                entry.tags = args["tags"]
            if "description" in args:
                entry.description = args["description"]
            if "triggers" in args:
                entry.triggers = args["triggers"]
            if "priority" in args:
                entry.priority = args["priority"]
            # Rebuild indices (tags/triggers may have changed)
            lib._remove_from_indices(entry_id)
            lib._add_to_indices(entry)
            # Save to disk
            lib._save_entry(entry)
        return {"updated": True, "entry_id": entry_id}

    # ── Math / calculation handlers ───────────────────────────────────

    _MATH_FUNCS = {
        'sqrt': _math.sqrt, 'sin': _math.sin, 'cos': _math.cos,
        'tan': _math.tan, 'log': _math.log, 'log2': _math.log2,
        'log10': _math.log10, 'exp': _math.exp, 'abs': abs,
        'round': round, 'min': min, 'max': max, 'pow': pow,
        'floor': _math.floor, 'ceil': _math.ceil,
        'factorial': _math.factorial, 'gcd': _math.gcd,
    }
    _MATH_CONSTS = {'pi': _math.pi, 'e': _math.e, 'tau': _math.tau, 'inf': _math.inf}

    _BINOPS = {
        _ast.Add: _op.add, _ast.Sub: _op.sub, _ast.Mult: _op.mul,
        _ast.Div: _op.truediv, _ast.FloorDiv: _op.floordiv,
        _ast.Mod: _op.mod, _ast.Pow: _op.pow,
    }
    _UNARYOPS = {
        _ast.UAdd: _op.pos, _ast.USub: _op.neg,
    }

    # EC-119: bounds so a hostile/buggy expression can't burn CPU or memory.
    _MAX_EXPR_LEN = 500
    _MAX_AST_NODES = 200
    _MAX_AST_DEPTH = 40
    _MAX_POW_EXP = 10_000        # reject 9**9**9-style exponent towers
    _MAX_FACTORIAL_ARG = 10_000  # factorial() bigint bomb guard
    _MAX_COUNT = 1_000           # random_number / chance 'count' cap
    _MAX_ITEMS = 10_000          # chance 'items' list cap
    _MAX_LIST_ENTRIES = 10_000   # file_list entry cap

    def _safe_math_eval(self, expr: str) -> float:
        """Safely evaluate a math expression using AST."""
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("Empty expression")
        if len(expr) > self._MAX_EXPR_LEN:
            raise ValueError("Expression too long")
        tree = _ast.parse(expr, mode='eval')
        if sum(1 for _ in _ast.walk(tree)) > self._MAX_AST_NODES:
            raise ValueError("Expression too complex")
        if self._ast_depth(tree) > self._MAX_AST_DEPTH:
            raise ValueError("Expression too deeply nested")
        return self._eval_node(tree.body)

    @classmethod
    def _ast_depth(cls, node) -> int:
        """Max nesting depth of an AST subtree (iterative, recursion-safe)."""
        best = 0
        stack = [(node, 1)]
        while stack:
            n, d = stack.pop()
            if d > best:
                best = d
                if best > cls._MAX_AST_DEPTH:
                    return best  # already over the cap — bail early
            stack.extend((c, d + 1) for c in _ast.iter_child_nodes(n))
        return best

    def _eval_node(self, node):
        if isinstance(node, _ast.Constant):
            if not isinstance(node.value, (int, float, complex)):
                raise ValueError(f"Unsupported literal: {node.value!r}")
            return node.value
        elif isinstance(node, _ast.Num):  # py < 3.8 compat
            return node.n
        elif isinstance(node, _ast.Name):
            if node.id in self._MATH_CONSTS:
                return self._MATH_CONSTS[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, _ast.BinOp):
            op_func = self._BINOPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            if op_func is _op.pow:
                # Exponent first: reject exponent towers like 9**9**9 that
                # would materialize astronomical bigints (EC-119).
                right = self._eval_node(node.right)
                if not isinstance(right, (int, float)) or abs(right) > self._MAX_POW_EXP:
                    raise ValueError("Exponent too large")
                return op_func(self._eval_node(node.left), right)
            return op_func(self._eval_node(node.left), self._eval_node(node.right))
        elif isinstance(node, _ast.UnaryOp):
            op_func = self._UNARYOPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return op_func(self._eval_node(node.operand))
        elif isinstance(node, _ast.Call):
            func_name = node.func.id if isinstance(node.func, _ast.Name) else None
            if func_name not in self._MATH_FUNCS:
                raise ValueError(f"Unknown function: {func_name}")
            args = [self._eval_node(a) for a in node.args]
            if func_name == "factorial":
                a = args[0]
                if not isinstance(a, int) or a < 0 or a > self._MAX_FACTORIAL_ARG:
                    raise ValueError(f"factorial() argument out of range (0..{self._MAX_FACTORIAL_ARG})")
            elif func_name == "pow" and len(args) >= 2:
                if not isinstance(args[1], (int, float)) or abs(args[1]) > self._MAX_POW_EXP:
                    raise ValueError("Exponent too large")
            return self._MATH_FUNCS[func_name](*args)
        else:
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    def _math_eval(self, args: dict) -> dict:
        expr = args["expression"]
        result = self._safe_math_eval(expr)
        return {"expression": expr, "result": result}

    def _calc(self, args: dict) -> dict:
        expr = args["expression"]
        result = self._safe_math_eval(expr)
        return {"expression": expr, "result": result}

    def _clamp_count(self, args: dict) -> int:
        """Clamped 'count' arg — unbounded range(count) hangs (EC-119)."""
        try:
            count = int(args.get("count", 1))
        except (TypeError, ValueError):
            count = 1
        return max(0, min(count, self._MAX_COUNT))

    def _random_number(self, args: dict) -> dict:
        rng = _random.Random(args["seed"]) if "seed" in args else _random
        min_val = args.get("min", 0)
        max_val = args.get("max", 1)
        count = self._clamp_count(args)
        integer = args.get("integer", True)
        if integer:
            values = [rng.randint(int(min_val), int(max_val)) for _ in range(count)]
        else:
            values = [rng.uniform(min_val, max_val) for _ in range(count)]
        return {"values": values, "count": count}

    def _chance(self, args: dict) -> dict:
        rng = _random.Random(args["seed"]) if "seed" in args else _random
        mode = args["mode"]
        if mode == "coin_flip":
            count = self._clamp_count(args)
            flips = [rng.choice(["heads", "tails"]) for _ in range(count)]
            return {"flips": flips, "count": count}
        elif mode == "dice":
            sides = args.get("sides", 6)
            count = self._clamp_count(args)
            rolls = [rng.randint(1, sides) for _ in range(count)]
            return {"rolls": rolls, "sides": sides, "total": sum(rolls)}
        elif mode == "choice":
            items = args.get("items", [])
            if not items:
                return {"error": "items required for choice mode"}
            if len(items) > self._MAX_ITEMS:
                return {"error": f"items too large (max {self._MAX_ITEMS})"}
            return {"choice": rng.choice(items)}
        elif mode == "shuffle":
            items = list(args.get("items", []))
            if not items:
                return {"error": "items required for shuffle mode"}
            if len(items) > self._MAX_ITEMS:
                return {"error": f"items too large (max {self._MAX_ITEMS})"}
            rng.shuffle(items)
            return {"shuffled": items}
        elif mode == "weighted":
            items = args.get("items", [])
            weights = args.get("weights", [])
            if not items or not weights or len(items) != len(weights):
                return {"error": "items and weights (same length) required for weighted mode"}
            if len(items) > self._MAX_ITEMS:
                return {"error": f"items too large (max {self._MAX_ITEMS})"}
            choice = rng.choices(items, weights=weights, k=1)[0]
            return {"choice": choice}
        return {"error": f"Unknown mode: {mode}"}

    # ── Web search / scrape handlers ──────────────────────────────────

    def _get_api_key(self, name: str) -> str | None:
        """Load API key from environment or .env file."""
        import os
        key = os.environ.get(name)
        if key:
            return key
        # Try loading from .env
        from pathlib import Path
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == name:
                        return v.strip()
        return None

    def _web_search(self, args: dict) -> dict:
        """Tavily web search."""
        import urllib.parse
        import urllib.request
        api_key = self._get_api_key("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY not set"}
        query = args["query"]
        max_results = min(args.get("max_results", 5), 10)
        search_depth = args.get("search_depth", "basic")
        include_answer = args.get("include_answer", True)
        payload = json.dumps({
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": f"Tavily search failed: {e}"}
        results = []
        filtered = 0
        for r in data.get("results", []):
            url = r.get("url", "")
            # Filter by website blacklist
            if url:
                decision = self.security.check_website(url)
                if not decision.allowed:
                    filtered += 1
                    continue
            results.append({
                "title": r.get("title", ""),
                "url": url,
                "content": r.get("content", "")[:2000],
            })
        return {
            "query": query,
            "answer": data.get("answer", ""),
            "results": results,
            "count": len(results),
            "filtered_by_blacklist": filtered,
        }

    def _web_search_semantic(self, args: dict) -> dict:
        """Exa semantic search."""
        import urllib.request
        api_key = self._get_api_key("EXA_API_KEY")
        if not api_key:
            return {"error": "EXA_API_KEY not set"}
        query = args["query"]
        num_results = args.get("num_results", 5)
        search_type = args.get("search_type", "auto")
        include_domains = args.get("include_domains")
        payload_dict = {
            "query": query,
            "type": search_type,
            "numResults": num_results,
            "contents": {"highlights": True},
        }
        if include_domains:
            payload_dict["includeDomains"] = include_domains
        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            "https://api.exa.ai/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": f"Exa search failed: {e}"}
        results = []
        filtered = 0
        for r in data.get("results", []):
            url = r.get("url", "")
            if url:
                decision = self.security.check_website(url)
                if not decision.allowed:
                    filtered += 1
                    continue
            results.append({
                "title": r.get("title", ""),
                "url": url,
                "highlights": r.get("highlights", []),
            })
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "filtered_by_blacklist": filtered,
        }

    def _web_scrape(self, args: dict) -> dict:
        """Firecrawl page scrape."""
        import urllib.request
        api_key = self._get_api_key("FIRECRAWL_API_KEY")
        if not api_key:
            return {"error": "FIRECRAWL_API_KEY not set"}
        url = args["url"]
        max_chars = args.get("max_chars", 10000)
        fmt = args.get("format", "markdown")
        payload = json.dumps({
            "url": url,
            "formats": [fmt],
            "onlyMainContent": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.firecrawl.dev/v1/scrape",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": f"Firecrawl scrape failed: {e}"}
        # Extract content from response
        content = ""
        if data.get("success") and data.get("data"):
            d = data["data"]
            if fmt == "markdown":
                content = d.get("markdown", "")
            elif fmt == "html":
                content = d.get("html", "")
            else:
                content = d.get("text", d.get("markdown", ""))
        return {
            "url": url,
            "content": content[:max_chars],
            "truncated": len(content) > max_chars,
            "title": data.get("data", {}).get("metadata", {}).get("title", ""),
        }

    # ── File operation handlers ───────────────────────────────────────

    def _workspace_root(self) -> str:
        """Get the workspace root directory."""
        return str(self.security.workspace_root)

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to workspace root, ensuring it stays within."""
        from pathlib import Path
        root = Path(self._workspace_root()).resolve()
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        # Security: ensure path is within workspace
        try:
            p.relative_to(root)
        except ValueError as e:
            raise ValueError(f"Path outside workspace: {path}") from e
        return str(p)

    def _file_read(self, args: dict) -> dict:
        path = self._resolve_path(args["path"])
        try:
            max_chars = max(1, min(int(args.get("max_chars", 50000)), 1_000_000))
        except (TypeError, ValueError):
            max_chars = 50000
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars + 1)
        except FileNotFoundError:
            return {"error": f"File not found: {args['path']}"}
        except IsADirectoryError:
            return {"error": f"Path is a directory: {args['path']}"}
        return {
            "path": args["path"],
            "content": content[:max_chars],
            "truncated": len(content) > max_chars,
            "size": len(content),
        }

    def _file_write(self, args: dict) -> dict:
        path = self._resolve_path(args["path"])
        content = args["content"]
        append = args.get("append", False)
        from pathlib import Path
        # Create parent dirs
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        result = {"path": args["path"], "written": True, "bytes": len(content),
                  "mode": "append" if append else "write"}
        # Include script scan report for Python files
        if path.endswith(".py") and not append:
            scan = self.security.scan_script(content)
            result["script_scan"] = scan
        return result

    def _file_edit(self, args: dict) -> dict:
        path = self._resolve_path(args["path"])
        old_str = args["old_string"]
        new_str = args["new_string"]
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return {"error": f"File not found: {args['path']}"}
        count = content.count(old_str)
        if count == 0:
            return {"error": "old_string not found in file"}
        if count > 1:
            return {"error": f"old_string appears {count} times; must be unique"}
        new_content = content.replace(old_str, new_str, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return {"path": args["path"], "edited": True, "replacements": 1}

    def _file_move(self, args: dict) -> dict:
        import shutil
        src = self._resolve_path(args["source"])
        dst = self._resolve_path(args["destination"])
        from pathlib import Path
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        return {"source": args["source"], "destination": args["destination"],
                "moved": True}

    def _file_rename(self, args: dict) -> dict:
        import os
        path = self._resolve_path(args["path"])
        new_name = args["new_name"]
        parent = os.path.dirname(path)
        new_path = os.path.join(parent, new_name)
        os.rename(path, new_path)
        return {"path": args["path"], "new_path": new_name, "renamed": True}

    def _file_list(self, args: dict) -> dict:
        import os
        from pathlib import Path
        path = self._resolve_path(args.get("path", "."))
        recursive = args.get("recursive", False)
        pattern = args.get("pattern")
        entries = []
        truncated = False
        cap = self._MAX_LIST_ENTRIES
        if recursive:
            base = Path(path)
            # Cap the walk — an uncapped rglob over the whole workspace
            # (venv/, .git/, node_modules/) can enumerate 100k+ entries (EC-119).
            for p in base.rglob(pattern or "*"):
                if len(entries) >= cap:
                    truncated = True
                    break
                entries.append({
                    "name": str(p.relative_to(base)),
                    "type": "dir" if p.is_dir() else "file",
                    "size": p.stat().st_size if p.is_file() else 0,
                })
        else:
            for name in os.listdir(path):
                if len(entries) >= cap:
                    truncated = True
                    break
                full = os.path.join(path, name)
                if pattern:
                    import fnmatch
                    if not fnmatch.fnmatch(name, pattern):
                        continue
                entries.append({
                    "name": name,
                    "type": "dir" if os.path.isdir(full) else "file",
                    "size": os.path.getsize(full) if os.path.isfile(full) else 0,
                })
        return {"path": args.get("path", "."), "entries": entries,
                "count": len(entries), "truncated": truncated}

    def _file_delete(self, args: dict) -> dict:
        import os
        path = self._resolve_path(args["path"])
        confirm = args.get("confirm", False)
        if not confirm:
            return {"error": "Set confirm=true to delete"}
        if os.path.isdir(path):
            return {"error": "Cannot delete directories with file_delete; use file_move to relocate"}
        os.remove(path)
        return {"path": args["path"], "deleted": True}

    # ── Security / script scan handlers ───────────────────────────────

    def _scan_script(self, args: dict) -> dict:
        """Pre-scan a Python script for dangerous content."""
        return self.security.scan_script(args["content"])

    def _security_get_config(self, args: dict) -> dict:
        """Get current security configuration."""
        return self.security.get_config()

    def _security_get_pending(self, args: dict) -> dict:
        """Get pending permission requests."""
        return {"pending": self.security.get_pending_requests(),
                "count": len(self.security.pending_requests)}
