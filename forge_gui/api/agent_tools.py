"""Sandboxed coding tools for the ForgeAI agent loop.

Pure-python tool implementations (no Qt, no torch) executed inside the
agent's tool rounds. Every path-taking tool is jailed to a workspace root:
absolute paths and ``..`` escapes are rejected. Command execution uses a
deny-by-default allowlist and a hard timeout.

Tool schema matches the OpenAI function-calling shape the engine's
``qwen_parse_tool_calls`` emits: {"name", "arguments": {...}}.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 4000
MAX_FILE_BYTES = 512_000

# Directories to skip when walking the project tree
SKIP_DIRS_WALK = {".git", "__pycache__", ".venv", "venv", "node_modules",
                  ".pytest_cache", ".ruff_cache"}

# Commands the agent may run via run_cmd (prefix match on argv[0]).
# Interpreters/package managers/VCS are deliberately absent: an
# allowlisted 'python'/'pip'/'git' is arbitrary code/data exfiltration
# with extra steps — the dedicated run_python/git_* tools cover those
# needs. (sys.executable was also a dead entry: the basename check below
# could never match a full path anyway.)
CMD_ALLOWLIST = {
    "pytest", "dir", "type", "echo",
}

_TOOL_DEFS: list[dict] = [
    {"name": "list_dir", "description": "List files/directories under a workspace-relative path.",
     "parameters": {"path": {"type": "string", "description": "Workspace-relative directory ('.' for root)"}}},
    {"name": "read_file", "description": "Read a text file from the workspace.",
     "parameters": {"path": {"type": "string"}, "max_chars": {"type": "integer", "description": "optional truncation limit"}}},
    {"name": "write_file", "description": "Create or overwrite a text file in the workspace (parent dirs auto-created).",
     "parameters": {"path": {"type": "string"}, "content": {"type": "string"}}},
    {"name": "append_file", "description": "Append text to an existing workspace file.",
     "parameters": {"path": {"type": "string"}, "content": {"type": "string"}}},
    {"name": "delete_file", "description": "Delete a file inside the workspace.",
     "parameters": {"path": {"type": "string"}}},
    {"name": "run_python", "description": "Execute a python snippet in the workspace and capture stdout/stderr.",
     "parameters": {"code": {"type": "string"}, "timeout_s": {"type": "number", "description": "optional, default 30"}}},
    {"name": "run_cmd", "description": "Run an allowlisted shell command (pytest, dir, type, echo) in the workspace. Use run_python for python code.",
     "parameters": {"command": {"type": "string"}, "timeout_s": {"type": "number", "description": "optional, default 60"}}},
    {"name": "grep_project", "description": "Regex search across workspace text files; returns matching path:line:text.",
     "parameters": {"pattern": {"type": "string"}, "glob": {"type": "string", "description": "optional filename glob, e.g. '*.py'"}}},
    # ── Git tools (read-only + revert) ──
    {"name": "git_status", "description": "Show git working tree status (modified, staged, untracked files).",
     "parameters": {}},
    {"name": "git_diff", "description": "Show git diff of unstaged changes, or diff of a specific file.",
     "parameters": {"path": {"type": "string", "description": "optional file path to diff"}, "staged": {"type": "boolean", "description": "show staged changes"}}},
    {"name": "git_log", "description": "Show recent git commit log.",
     "parameters": {"n": {"type": "integer", "description": "number of commits (default 10)"}}},
    {"name": "git_revert", "description": "Revert a file to its last committed state (git checkout -- <file>). Does NOT revert git history.",
     "parameters": {"path": {"type": "string", "description": "file to revert to last committed version"}}},
    # ── Precise edit tools ──
    {"name": "search_replace", "description": "Replace exact text in a file. Fails if old_text is not found or not unique. Safer than write_file for small edits.",
     "parameters": {"path": {"type": "string"}, "old_text": {"type": "string", "description": "exact text to find"}, "new_text": {"type": "string", "description": "replacement text"}, "replace_all": {"type": "boolean", "description": "replace all occurrences (default false)"}}},
    {"name": "create_file", "description": "Create a new file. Fails if the file already exists (use write_file to overwrite).",
     "parameters": {"path": {"type": "string"}, "content": {"type": "string"}}},
    # ── File management ──
    {"name": "rename_file", "description": "Rename or move a file within the workspace.",
     "parameters": {"old_path": {"type": "string"}, "new_path": {"type": "string"}}},
    {"name": "create_dir", "description": "Create a directory (and parent dirs) in the workspace.",
     "parameters": {"path": {"type": "string"}}},
    {"name": "file_info", "description": "Get file metadata: size, mtime, line count, extension.",
     "parameters": {"path": {"type": "string"}}},
    {"name": "dir_tree", "description": "Show a recursive directory tree (up to 3 levels deep). Useful for understanding project structure.",
     "parameters": {"path": {"type": "string", "description": "directory to tree (default '.')"}, "max_depth": {"type": "integer", "description": "max depth (default 3)"}}},
    # ── Code intelligence ──
    {"name": "find_references", "description": "Find all references to a symbol (function/class/variable name) across the project. Returns file:line:text matches.",
     "parameters": {"symbol": {"type": "string"}, "glob": {"type": "string", "description": "optional file glob, e.g. '*.py'"}}},
    {"name": "find_definitions", "description": "Find function/class/method definitions in a file or across the project. Returns file:line:signature.",
     "parameters": {"path": {"type": "string", "description": "file to search (empty = all .py files)"}, "symbol": {"type": "string", "description": "optional name to filter by"}}},
    {"name": "find_todos", "description": "Find TODO, FIXME, HACK, XXX comments across the project. Returns file:line:text.",
     "parameters": {"glob": {"type": "string", "description": "optional file glob"}}},
    {"name": "line_count", "description": "Count lines of code in a file or across the project. Returns total, blank, comment, and code lines.",
     "parameters": {"path": {"type": "string", "description": "file or directory to count"}}},
    {"name": "syntax_check", "description": "Check Python syntax of a file without executing it. Returns syntax errors if any.",
     "parameters": {"path": {"type": "string"}}},
    # ── Project-wide search & replace ──
    {"name": "project_search_replace", "description": "Search and replace text across all files in the project. Use carefully — affects multiple files. Shows count of files modified.",
     "parameters": {"old_text": {"type": "string"}, "new_text": {"type": "string"}, "glob": {"type": "string", "description": "file glob to limit scope, e.g. '*.py'"}}},
    # ── Git extras ──
    {"name": "git_branch", "description": "List, create, or switch git branches. action: 'list' (default), 'create', 'switch'.",
     "parameters": {"action": {"type": "string", "description": "'list', 'create', or 'switch'"}, "name": {"type": "string", "description": "branch name (for create/switch)"}}},
    {"name": "git_stash", "description": "Git stash operations. action: 'save' (default), 'pop', 'list', 'drop'.",
     "parameters": {"action": {"type": "string"}, "message": {"type": "string", "description": "stash message (for save)"}}},
    # ── Test runner ──
    {"name": "run_tests", "description": "Run pytest on a specific test file or directory. Returns pass/fail counts and output.",
     "parameters": {"path": {"type": "string", "description": "test file or directory (default 'tests/')" }, "args": {"type": "string", "description": "extra pytest args, e.g. '-k test_name'"}}},
    # ── Undo ──
    {"name": "undo_edit", "description": "Undo the last file edit made by the agent (reverts to the content before the last write_file/append_file/search_replace). Only keeps 1 level of history.",
     "parameters": {"path": {"type": "string", "description": "file to undo (must be the last edited file)"}}},
]


def tool_defs() -> list[dict]:
    """OpenAI-style tool definition list for the renderer."""
    return [
        {"type": "function", "function": {"name": t["name"],
                                          "description": t["description"],
                                          "parameters": {"type": "object",
                                                         "properties": t["parameters"],
                                                         "required": [
                                                             k for k, v in t["parameters"].items()
                                                             if k in ("path", "content", "code", "command", "pattern",
                                                                      "old_text", "new_text", "old_path", "new_path",
                                                                      "symbol", "action")]}}}
        for t in _TOOL_DEFS
    ]


class ToolSandbox:
    """Executes agent tools against a workspace root."""

    def __init__(self, workspace: str | Path,
                 allow_delete: bool = True,
                 default_timeout_s: float = 30.0) -> None:
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.allow_delete = allow_delete
        self.default_timeout_s = default_timeout_s
        self.calls: list[dict[str, Any]] = []
        self._undo_buffer: dict[str, str] = {}  # path → previous content

    # ── path jail ─────────────────────────────────────────────────────
    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute():
            raise PermissionError(f"absolute paths not allowed: {rel}")
        target = (self.workspace / p).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as e:
            raise PermissionError(f"path escapes workspace: {rel}") from e
        return target

    # ── dispatch ──────────────────────────────────────────────────────
    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            fn = getattr(self, f"tool_{name}", None)
            if fn is None:
                raise ValueError(f"unknown tool: {name}")
            result = fn(**args)
            ok = not (isinstance(result, dict) and "error" in result)
        except TypeError as e:
            # Include expected parameters in error for wrong-arg failures
            import inspect
            fn = getattr(self, f"tool_{name}", None)
            expected = ""
            if fn is not None:
                try:
                    sig = inspect.signature(fn)
                    params = []
                    for pname, p in sig.parameters.items():
                        if p.default is inspect.Parameter.empty:
                            params.append(f"{pname} (required)")
                        else:
                            params.append(f"{pname}={p.default!r}")
                    expected = f". Expected parameters: {', '.join(params)}"
                except (ValueError, TypeError):
                    pass
            result = {"error": f"{type(e).__name__}: {e}{expected}"}
            ok = False
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
            ok = False
        rec = {"name": name, "args": args, "ok": ok,
               "elapsed_s": round(time.perf_counter() - t0, 3),
               "result": result}
        self.calls.append(rec)
        return rec

    def execute_calls(self, calls: list[dict]) -> list[dict]:
        return [self.execute(c.get("name", ""), c.get("arguments", {}) or {})
                for c in calls]

    # ── file tools ────────────────────────────────────────────────────
    def tool_list_dir(self, path: str = ".") -> dict:
        d = self.resolve(path or ".")
        if not d.is_dir():
            return {"error": f"not a directory: {path}"}
        entries = []
        for e in sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            try:
                st = e.stat()
                entries.append({"name": e.name, "dir": e.is_dir(),
                                "size": 0 if e.is_dir() else st.st_size})
            except OSError:
                continue
        return {"path": str(d), "entries": entries[:500]}

    def tool_read_file(self, path: str, max_chars: int = 20_000) -> dict:
        f = self.resolve(path)
        if not f.is_file():
            return {"error": f"not a file: {path}"}
        if f.stat().st_size > MAX_FILE_BYTES:
            return {"error": f"file too large ({f.stat().st_size} bytes)"}
        text = f.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        return {"path": path, "content": text[:max_chars],
                "truncated": truncated}

    def tool_write_file(self, path: str, content: str = "") -> dict:
        f = self.resolve(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        # save undo state
        if f.is_file():
            self._undo_buffer[str(f)] = f.read_text(encoding="utf-8", errors="replace")
        f.write_text(content, encoding="utf-8")
        return {"path": path, "bytes": len(content.encode("utf-8")),
                "written": True}

    def tool_append_file(self, path: str, content: str = "") -> dict:
        f = self.resolve(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.is_file():
            self._undo_buffer[str(f)] = f.read_text(encoding="utf-8", errors="replace")
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(content)
        return {"path": path, "appended_bytes": len(content.encode("utf-8"))}

    def tool_delete_file(self, path: str) -> dict:
        if not self.allow_delete:
            return {"error": "delete_file disabled by policy"}
        f = self.resolve(path)
        if not f.is_file():
            return {"error": f"not a file: {path}"}
        f.unlink()
        return {"path": path, "deleted": True}

    # ── execution tools ───────────────────────────────────────────────
    def tool_run_python(self, code: str, timeout_s: float = 0) -> dict:
        if not code.strip():
            return {"error": "empty snippet"}
        script = self.workspace / ".agent_snippet.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(self.workspace), capture_output=True, text=True,
                timeout=(timeout_s or self.default_timeout_s),
                encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return self._proc_result(proc)
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout_s or self.default_timeout_s}s",
                    "stdout": "", "stderr": ""}
        finally:
            try:
                script.unlink()
            except OSError:
                pass

    def tool_run_cmd(self, command: str, timeout_s: float = 0) -> dict:
        parts = command.strip().split()
        if not parts:
            return {"error": "empty command"}
        # argv[0] must be a bare command name resolved via PATH — a
        # relative/absolute path like '..\x\pytest.exe' used to pass the
        # basename check while executing an out-of-workspace binary.
        argv0 = parts[0]
        if ("/" in argv0 or "\\" in argv0 or ":" in argv0
                or argv0 in (".", "..")):
            return {"error": f"command '{argv0}' must be a bare name "
                             "(no path separators)"}
        exe = os.path.splitext(argv0)[0].lower()
        if exe not in CMD_ALLOWLIST:
            return {"error": f"command '{argv0}' not in allowlist "
                             f"({', '.join(sorted(CMD_ALLOWLIST))})"}
        try:
            proc = subprocess.run(
                parts, cwd=str(self.workspace), capture_output=True, text=True,
                timeout=(timeout_s or self.default_timeout_s * 2),
                encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return self._proc_result(proc)
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout_s or self.default_timeout_s * 2}s",
                    "stdout": "", "stderr": ""}
        except FileNotFoundError as e:
            return {"error": f"executable not found: {e.filename}"}

    def _walk_files(self) -> list[Path]:
        """Walk project files, skipping standard dirs and oversized files."""
        results = []
        for dirpath, dirnames, filenames in os.walk(self.workspace):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS_WALK]
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    if p.stat().st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                results.append(p)
        return results

    def tool_grep_project(self, pattern: str, glob: str = "**/*") -> dict:
        import re as _re
        try:
            rx = _re.compile(pattern)
        except _re.error as e:
            return {"error": f"bad regex: {e}"}
        matches: list[dict] = []
        root = self.workspace
        for p in self._walk_files():
            if not _fnmatch_rel(p, root, glob):
                continue
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            matches.append({
                                "path": str(p.relative_to(root)).replace("\\", "/"),
                                "line": i, "text": line.strip()[:200]})
                            if len(matches) >= 80:
                                return {"matches": matches, "truncated": True}
            except OSError:
                continue
        return {"matches": matches, "truncated": False}

    # ── git tools ─────────────────────────────────────────────────────
    def _git(self, args: list[str], timeout_s: float = 15) -> dict:
        try:
            proc = subprocess.run(
                ["git"] + args, cwd=str(self.workspace),
                capture_output=True, text=True,
                timeout=timeout_s, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return self._proc_result(proc)
        except subprocess.TimeoutExpired:
            return {"error": "git timeout"}
        except FileNotFoundError:
            return {"error": "git not installed"}

    def tool_git_status(self) -> dict:
        return self._git(["status", "--porcelain"])

    def tool_git_diff(self, path: str = "", staged: bool = False) -> dict:
        args = ["diff"]
        if staged:
            args.append("--staged")
        if path:
            args += ["--", path]
        return self._git(args)

    def tool_git_log(self, n: int = 10) -> dict:
        return self._git(["log", f"-{n}", "--oneline"])

    def tool_git_revert(self, path: str) -> dict:
        f = self.resolve(path)
        rel = str(f.relative_to(self.workspace)).replace("\\", "/")
        return self._git(["checkout", "--", rel])

    # ── precise edit tools ────────────────────────────────────────────
    def tool_search_replace(self, path: str, old_text: str,
                            new_text: str, replace_all: bool = False) -> dict:
        f = self.resolve(path)
        if not f.is_file():
            return {"error": f"not a file: {path}"}
        content = f.read_text(encoding="utf-8")
        self._undo_buffer[str(f)] = content  # save undo state
        count = content.count(old_text)
        if count == 0:
            return {"error": f"old_text not found in {path}"}
        if count > 1 and not replace_all:
            return {"error": f"old_text not unique ({count} matches) — "
                             f"set replace_all=true to replace all"}
        if replace_all:
            new_content = content.replace(old_text, new_text)
        else:
            new_content = content.replace(old_text, new_text, 1)
        f.write_text(new_content, encoding="utf-8")
        return {"path": path, "replacements": count if replace_all else 1,
                "done": True}

    def tool_create_file(self, path: str, content: str = "") -> dict:
        f = self.resolve(path)
        if f.exists():
            return {"error": f"file already exists: {path} (use write_file to overwrite)"}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        return {"path": path, "bytes": len(content.encode("utf-8")),
                "created": True}

    # ── file management ───────────────────────────────────────────────
    def tool_rename_file(self, old_path: str, new_path: str) -> dict:
        src = self.resolve(old_path)
        dst = self.resolve(new_path)
        if not src.exists():
            return {"error": f"source not found: {old_path}"}
        if dst.exists():
            return {"error": f"destination exists: {new_path}"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return {"old_path": old_path, "new_path": new_path, "renamed": True}

    def tool_create_dir(self, path: str) -> dict:
        d = self.resolve(path)
        d.mkdir(parents=True, exist_ok=True)
        return {"path": path, "created": True}

    def tool_file_info(self, path: str) -> dict:
        f = self.resolve(path)
        if not f.exists():
            return {"error": f"not found: {path}"}
        st = f.stat()
        info = {"path": path, "size": st.st_size, "size_human": _human_size(st.st_size),
                "mtime": st.st_mtime, "is_dir": f.is_dir(), "ext": f.suffix}
        if f.is_file():
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                info["lines"] = len(lines)
                info["blank_lines"] = sum(1 for l in lines if not l.strip())
                info["comment_lines"] = sum(1 for l in lines if l.strip().startswith("#"))
            except OSError:
                pass
        return info

    def tool_dir_tree(self, path: str = ".", max_depth: int = 3) -> dict:
        d = self.resolve(path or ".")
        if not d.is_dir():
            return {"error": f"not a directory: {path}"}
        tree = _build_tree(d, self.workspace, 0, max_depth)
        return {"path": path, "tree": tree}

    # ── code intelligence ─────────────────────────────────────────────
    def tool_find_references(self, symbol: str, glob: str = "**/*.py") -> dict:
        if not symbol:
            return {"error": "symbol required"}
        import re as _re
        # word-boundary search for the symbol
        try:
            rx = _re.compile(r"\b" + _re.escape(symbol) + r"\b")
        except _re.error:
            return {"error": "bad symbol pattern"}
        matches = []
        for p in self._walk_files():
            if not _fnmatch_rel(p, self.workspace, glob):
                continue
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            matches.append({
                                "path": str(p.relative_to(self.workspace)).replace("\\", "/"),
                                "line": i, "text": line.strip()[:200]})
                            if len(matches) >= 100:
                                return {"matches": matches, "truncated": True}
            except OSError:
                continue
        return {"matches": matches, "count": len(matches)}

    def tool_find_definitions(self, path: str = "", symbol: str = "") -> dict:
        import re as _re
        # Python def/class patterns
        patterns = [
            (r"^\s*(def\s+(\w+))", "function"),
            (r"^\s*(class\s+(\w+))", "class"),
            (r"^\s*(async\s+def\s+(\w+))", "async_function"),
        ]
        rxs = [(_re.compile(p), t) for p, t in patterns]
        sym_rx = _re.compile(r"\b" + _re.escape(symbol) + r"\b") if symbol else None
        defs = []
        files = []
        if path:
            p = self.resolve(path)
            if p.is_file():
                files = [p]
            elif p.is_dir():
                files = [f for f in self._walk_files() if f.suffix == ".py"]
            else:
                return {"error": f"not found: {path}"}
        else:
            files = [f for f in self._walk_files() if f.suffix == ".py"]
        for p in files:
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        for rx, kind in rxs:
                            m = rx.match(line)
                            if m:
                                name = m.group(2)
                                if sym_rx and not sym_rx.search(name):
                                    continue
                                defs.append({
                                    "path": str(p.relative_to(self.workspace)).replace("\\", "/"),
                                    "line": i, "kind": kind, "name": name,
                                    "signature": line.strip()[:200]})
                                break
            except OSError:
                continue
        return {"definitions": defs[:100], "count": len(defs)}

    def tool_find_todos(self, glob: str = "**/*") -> dict:
        import re as _re
        rx = _re.compile(r"(?i)\b(TODO|FIXME|HACK|XXX|BUG|NOTE)\b")
        todos = []
        for p in self._walk_files():
            if not _fnmatch_rel(p, self.workspace, glob):
                continue
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        m = rx.search(line)
                        if m:
                            todos.append({
                                "path": str(p.relative_to(self.workspace)).replace("\\", "/"),
                                "line": i, "tag": m.group(1).upper(),
                                "text": line.strip()[:200]})
                            if len(todos) >= 100:
                                return {"todos": todos, "truncated": True}
            except OSError:
                continue
        return {"todos": todos, "count": len(todos)}

    def tool_line_count(self, path: str = ".") -> dict:
        p = self.resolve(path)
        if p.is_file():
            return _count_lines(p, path)
        if p.is_dir():
            total = {"files": 0, "total": 0, "blank": 0, "comment": 0, "code": 0}
            for f in self._walk_files():
                if f.suffix not in (".py", ".js", ".ts", ".jsx", ".tsx",
                                    ".c", ".cpp", ".h", ".hpp", ".rs", ".go",
                                    ".java", ".rb", ".sh", ".sql", ".html", ".css"):
                    continue
                lc = _count_lines(f, str(f.relative_to(self.workspace)))
                for k in ("total", "blank", "comment", "code"):
                    total[k] += lc.get(k, 0)
                total["files"] += 1
            return total
        return {"error": f"not found: {path}"}

    def tool_syntax_check(self, path: str) -> dict:
        import ast as _ast
        f = self.resolve(path)
        if not f.is_file():
            return {"error": f"not a file: {path}"}
        if f.suffix != ".py":
            return {"error": "syntax_check only supports .py files"}
        try:
            content = f.read_text(encoding="utf-8")
            _ast.parse(content, filename=path)
            return {"path": path, "valid": True, "errors": []}
        except SyntaxError as e:
            return {"path": path, "valid": False,
                    "errors": [{"line": e.lineno, "msg": e.msg,
                                "offset": e.offset}]}

    def tool_project_search_replace(self, old_text: str, new_text: str,
                                    glob: str = "**/*") -> dict:
        if not old_text:
            return {"error": "old_text required"}
        modified = []
        for p in self._walk_files():
            if not _fnmatch_rel(p, self.workspace, glob):
                continue
            if p.suffix in (".pyc", ".so", ".dll", ".bin"):
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if old_text in content:
                    count = content.count(old_text)
                    new_content = content.replace(old_text, new_text)
                    p.write_text(new_content, encoding="utf-8")
                    modified.append({
                        "path": str(p.relative_to(self.workspace)).replace("\\", "/"),
                        "replacements": count})
            except OSError:
                continue
        return {"modified_files": modified, "total_files": len(modified),
                "total_replacements": sum(m["replacements"] for m in modified)}

    def tool_run_tests(self, path: str = "tests/", args: str = "") -> dict:
        cmd_parts = ["pytest", path]
        if args:
            cmd_parts.extend(args.split())
        try:
            proc = subprocess.run(
                cmd_parts, cwd=str(self.workspace), capture_output=True,
                text=True, timeout=120, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return self._proc_result(proc)
        except subprocess.TimeoutExpired:
            return {"error": "pytest timeout (120s)"}
        except FileNotFoundError:
            return {"error": "pytest not installed"}

    def tool_undo_edit(self, path: str) -> dict:
        f = self.resolve(path)
        if not f.is_file():
            return {"error": f"not a file: {path}"}
        prev = self._undo_buffer.get(str(f))
        if prev is None:
            return {"error": f"no undo history for {path}"}
        f.write_text(prev, encoding="utf-8")
        del self._undo_buffer[str(f)]
        return {"path": path, "restored": True}

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _proc_result(proc: subprocess.CompletedProcess) -> dict:
        out = (proc.stdout or "")[-MAX_OUTPUT_CHARS:]
        err = (proc.stderr or "")[-MAX_OUTPUT_CHARS:]
        return {"exit_code": proc.returncode, "stdout": out, "stderr": err}

    def summary(self) -> dict:
        return {"workspace": str(self.workspace), "n_calls": len(self.calls),
                "tools_used": sorted({c["name"] for c in self.calls})}


def _fnmatch_rel(p: Path, root: Path, glob: str) -> bool:
    if glob in ("**/*", "*", ""):
        return True
    from fnmatch import fnmatch
    rel = str(p.relative_to(root)).replace("\\", "/")
    name = p.name
    if "/" in glob or "**" in glob:
        # for **/*.py style globs, also match by the last segment (filename)
        last_seg = glob.split("/")[-1]
        return (fnmatch(rel, glob) or fnmatch(rel, f"**/{glob}")
                or fnmatch(name, last_seg))
    return fnmatch(name, glob)


def tool_results_to_text(rec: dict) -> str:
    """Compact JSON text of a tool record for feeding back to the model."""
    payload = {"ok": rec["ok"], **rec["result"]}
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"ok": rec["ok"], "result": str(rec["result"])},
                          ensure_ascii=False)


# ── helper functions ────────────────────────────────────────────────────

def _human_size(size: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _build_tree(path: Path, root: Path, depth: int, max_depth: int) -> dict:
    """Build a recursive directory tree dict."""
    node = {"name": path.name or str(path), "type": "dir" if path.is_dir() else "file"}
    if path.is_file():
        try:
            node["size"] = path.stat().st_size
        except OSError:
            node["size"] = 0
        return node
    if depth >= max_depth:
        node["truncated"] = True
        return node
    children = []
    try:
        for child in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if child.name in SKIP_DIRS_WALK:
                continue
            children.append(_build_tree(child, root, depth + 1, max_depth))
    except OSError:
        pass
    node["children"] = children[:200]  # cap at 200 entries
    if len(children) > 200:
        node["truncated"] = True
    return node


def _count_lines(path: Path, rel: str = "") -> dict:
    """Count lines in a file: total, blank, comment, code."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"error": "cannot read file"}
    lines = text.splitlines()
    total = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    comment = sum(1 for l in lines if l.strip().startswith("#"))
    code = total - blank - comment
    return {"path": rel, "total": total, "blank": blank,
            "comment": comment, "code": code}
