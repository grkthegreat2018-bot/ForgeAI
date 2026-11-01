"""Security manager for ForgeEngine tool execution.

Sandbox-file-based security model:
  - A sandbox config file (<workspace>/sandbox.json) defines all access rules
  - Everything OUTSIDE the sandbox rules is blacklisted for writes
  - Reading is allowed within the workspace EXCEPT 'denied' paths and
    credential/secret files matched by the file blacklist (.env, *.pem,
    id_rsa, *.key, .ssh, sandbox.json, ...)
  - Per-path access rules: 'read_write', 'read_only', 'denied'
  - Website whitelist/blacklist for web tools
  - Script pre-scan for dangerous imports and escape attempts

Access rule precedence (highest to lowest):
  1. 'denied' — no access at all (can't read or write)
  2. File blacklist patterns — can't read/write/delete (credentials stay
     out of model context entirely)
  3. Protected engine files — can't write/delete (but CAN read)
  4. 'read_write' — full access
  5. 'read_only' — can read but not write/delete
  6. No rule — default: read-only (can read, can't write)

Security decisions return one of:
  - "allow": proceed with the operation
  - "needs_permission": flag for user approval (operation is risky but not blocked)
  - "refuse": hard block, operation is denied

Usage:
    from forge.engine.tool_security import ToolSecurityManager

    security = ToolSecurityManager(workspace_root="/path/to/workspace")
    # Loads sandbox.json automatically
    decision = security.check_file_write("/path/to/file.py")
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Security decision ────────────────────────────────────────────────────────

@dataclass
class SecurityDecision:
    """Result of a security check."""
    allowed: bool
    needs_permission: bool = False
    reason: str = ""
    severity: str = "info"  # info, warning, critical
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return not self.allowed and not self.needs_permission


# ── Access levels ────────────────────────────────────────────────────────────

ACCESS_READ_WRITE = "read_write"
ACCESS_READ_ONLY = "read_only"
ACCESS_DENIED = "denied"
VALID_ACCESS_LEVELS = {ACCESS_READ_WRITE, ACCESS_READ_ONLY, ACCESS_DENIED}


# ── Default file blacklist patterns ──────────────────────────────────────────
# These are always blocked for reads AND writes/deletes regardless of sandbox
# rules — credentials/secrets must never enter the model's context (EC-027).

DEFAULT_FILE_BLACKLIST_PATTERNS = [
    ".env",
    "*.env",
    ".git/config",
    ".git/hooks/*",
    "**/.git/**",
    "**/credentials*",
    "**/secrets*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/.ssh",
    "**/.ssh/**",
    "**/__pycache__/**",
    "*.pyc",
    # The sandbox policy itself — the model must not read or rewrite its
    # own security configuration (EC-160).
    "sandbox.json",
    "**/sandbox.json",
]

# Protected engine files — can read but never write/delete.
# Exact repo-relative paths (EC-027): the previous bare-name endswith()
# match over-blocked *any* file named config.py/paths.py anywhere in the
# workspace (e.g. research/data/run1/config.py).
PROTECTED_ENGINE_FILES = [
    "forge/engine/tool_security.py",
    "forge/engine/engine_tools.py",
    "forge/engine/forge_engine.py",
    "forge/engine/forge_server.py",
    "forge/engine/hotswap.py",
    "forge/engine/library.py",
    "forge/engine/model_registry.py",
    "forge/engine/kv_backend.py",
    "forge/engine/session_manager.py",
    "forge/engine/session_cache.py",
    "forge/model_loader.py",
    "forge/checkpoint_io.py",
    "forge/config.py",
    "research/paths.py",
    # A18: the self-improvement surface — model-facing tools can READ the
    # evolution stack but must never rewrite the eval machinery or the
    # promotion gate (that is the part an RSI loop is most motivated to
    # modify). sandbox.json grants forge/evolution read_only; these are
    # the files where a silent rewrite would corrupt every future eval.
    "forge/evolution/code_evolve.py",
    "forge/evolution/eval_pool.py",
    "forge/evolution/mutator.py",
    "forge/evolution/database.py",
    "forge/evolution/best_configs.py",
    # The promotion gate's output file — a forged best_configs.json would
    # inject attacker-chosen "evolved" values into sft_train defaults.
    "forge/evolution/best_configs.json",
]

# Protected engine file GLOBS — same rule as PROTECTED_ENGINE_FILES
# (read allowed, write/delete refused) but matched with fnmatch so whole
# generated-file families can be covered without enumerating them.
# Threat model (A18/EC-027): forge/evolution/targets/*.py are the eval
# targets — canonical implementations AND auto-generated challengers.
# Rewriting a target rewrites what every future candidate is measured
# against, which is the cheapest way to fake a "verified promotion".
PROTECTED_ENGINE_GLOBS = [
    "forge/evolution/targets/*.py",
]

# ── Risky command patterns ───────────────────────────────────────────────────
# These trigger needs_permission (flag for user) rather than outright refusal

RISKY_COMMAND_PATTERNS = [
    # Shell execution
    r"os\.system\s*\(",
    r"subprocess\.(call|run|Popen|check_output|check_call)\s*\(",
    r"subprocess\..*shell\s*=\s*True",
    r"pty\.spawn\s*\(",
    r"commands\.(getoutput|getstatusoutput)\s*\(",
    # Code execution
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"__import__\s*\(",
    r"importlib\.import_module\s*\(",
    # File system destruction
    r"shutil\.rmtree\s*\(",
    r"os\.remove\s*\(",
    r"os\.unlink\s*\(",
    r"os\.rmdir\s*\(",
    r"Path.*\.unlink\s*\(",
    r"os\.walk.*os\.remove",
    # Network exfiltration
    r"urllib.*\.urlopen\s*\(",
    r"requests\.(get|post|put|delete|patch)\s*\(",
    r"httpx\.(get|post|put|delete|patch)\s*\(",
    r"socket\.socket\s*\(",
    r"webbrowser\.open\s*\(",
    # Process control
    r"os\.(fork|execv?|execvp?|kill|killpg)\s*\(",
    r"os\._exit\s*\(",
    r"signal\.(SIGKILL|SIGTERM|SIGSTOP)\s*",
    r"multiprocessing\.Process\s*\(",
    r"threading\.Thread\s*\(.*daemon",
    # Environment / config manipulation
    r"os\.environ\s*\[",
    r"os\.chmod\s*\(",
    r"os\.chown\s*\(",
    r"sys\.path\.(append|insert)",
    # Privilege escalation
    r"sudo\b",
    r"chmod\s+\+x",
    r"chmod\s+777",
    # Data exfiltration patterns
    r"base64\.b64encode\s*\(.*(?:environ|secret|key|password|token)",
    r"pickle\.dumps\s*\(",
    r"marshal\.dumps\s*\(",
    # Registry / system config (Windows)
    r"winreg\.",
    r"ctypes\.windll",
    # Escape attempt patterns
    r"\.\./\.\./\.\.",
    r"\.\.\\\\\.\.\\\\",
    r"/etc/passwd",
    r"/etc/shadow",
    r"C:\\Windows\\System32",
    r"\\\\\\\\\\.",
]

# Patterns that are ALWAYS refused (not just flagged)
HARD_REFUSAL_PATTERNS = [
    # Attempting to disable security
    r"security\.(disable|set_.*blacklist.*\[\]|clear_.*whitelist)",
    r"ToolSecurityManager\s*\(",
    # Self-modification of the tool registry
    r"_handlers\s*\[",
    r"_register_handlers",
]

# Python imports that indicate escape attempts in generated scripts
DANGEROUS_IMPORTS = {
    "ctypes": "Direct FFI — can bypass Python sandbox",
    "subprocess": "Process spawning — can execute arbitrary commands",
    "multiprocessing": "Process spawning — can escape sandbox",
    "pty": "Pseudo-terminal — can spawn interactive shells",
    "socket": "Raw network access — can exfiltrate data",
    "socketserver": "Network server — can open backdoors",
    "http.server": "Network server — can open backdoors",
    "ftplib": "Network file transfer — can exfiltrate data",
    "smtplib": "Email sending — can exfiltrate data",
    "telnetlib": "Network protocol — can exfiltrate data",
    "paramiko": "SSH client — can access remote systems",
    "winreg": "Windows registry — can modify system config",
    "msvcrt": "Windows runtime — can bypass security",
    "_winapi": "Windows API — can bypass security",
    "shutil": "File operations — can delete/move files outside workspace",
}


# ── Default sandbox config ───────────────────────────────────────────────────

DEFAULT_SANDBOX_CONFIG = {
    "auto_mode": "ask",
    "access_rules": {
        # Writable dirs (model can create/modify files here) — post
        # research/->forge/ migration layout; research/* kept for the
        # data/results dirs that still exist on disk.
        "research/data": ACCESS_READ_WRITE,
        "research/results": ACCESS_READ_WRITE,
        "forge/data": ACCESS_READ_WRITE,
        "forge/results": ACCESS_READ_WRITE,
        # A18: forge/evolution is read_only for model-facing tools — it
        # holds the eval machinery + promotion gate (code_evolve,
        # eval_pool, mutator, targets/) that a self-improvement loop is
        # most motivated to rewrite. Evolution outputs are written by
        # internal engine code (not the tool layer) so no writable
        # carve-out is needed; forge/results stays the results drop.
        "forge/evolution": ACCESS_READ_ONLY,
        "sandbox": ACCESS_READ_WRITE,
        ".devin": ACCESS_READ_WRITE,
        # Read-only dirs (model can read but not modify)
        "forge/engine": ACCESS_READ_ONLY,
        "forge/training": ACCESS_READ_ONLY,
        "forge/self_play": ACCESS_READ_ONLY,
        "forge/distillation": ACCESS_READ_ONLY,
        "forge/evaluation": ACCESS_READ_ONLY,
        "forge/decoding": ACCESS_READ_ONLY,
        "forge/quant": ACCESS_READ_ONLY,
        "forge/keys": ACCESS_READ_ONLY,
        "forge/moe": ACCESS_READ_ONLY,
        "forge/model": ACCESS_READ_ONLY,
        "forge/runtime": ACCESS_READ_ONLY,
        "forge/exp": ACCESS_READ_ONLY,
        "research/checkpoints": ACCESS_READ_ONLY,
        "research/architecture": ACCESS_READ_ONLY,
        "AGENTS.md": ACCESS_READ_ONLY,
        "docs": ACCESS_READ_ONLY,
        # Denied (no access at all)
        ".env": ACCESS_DENIED,
        ".git": ACCESS_DENIED,
        ".ssh": ACCESS_DENIED,
        "venv": ACCESS_DENIED,
        ".venv": ACCESS_DENIED,
        "node_modules": ACCESS_DENIED,
        "__pycache__": ACCESS_DENIED,
        # The sandbox policy lives at the workspace root (EC-160) and is
        # denied + blacklisted so the model cannot read or rewrite it.
        "sandbox.json": ACCESS_DENIED,
    },
    "website_whitelist": [],
    "website_blacklist": [],
}


# ── ToolSecurityManager ──────────────────────────────────────────────────────

class ToolSecurityManager:
    """Sandbox-file-based security manager for LLM tool execution.

    Security model:
    - A sandbox config file defines access rules for paths
    - Everything OUTSIDE the sandbox rules defaults to read-only (can read, can't write)
    - Reading is allowed within workspace unless path is 'denied' or
      matches a credential/secret blacklist pattern (EC-027)
    - Writing/deleting requires 'read_write' access rule
    - File blacklist patterns always block reads AND writes
    - Protected engine files always block writes (but reading is OK)
    - Website whitelist/blacklist controls web tool access

    Attributes:
        workspace_root: The root directory all paths are relative to
        sandbox_path: Path to the sandbox.json config file
        access_rules: Dict of {path_pattern: access_level}
        auto_mode: "allow", "deny", or "ask" for risky operations
        website_whitelist: Set of allowed domains (empty = all allowed)
        website_blacklist: Set of blocked domains
    """

    def __init__(
        self,
        workspace_root: str,
        sandbox_path: str | None = None,
        auto_mode: str = "ask",
    ):
        self.workspace_root = Path(workspace_root).resolve()

        # Sandbox config file path. EC-160: the default used to live under
        # research/data — a read_write zone — so the model could rewrite its
        # own sandbox policy. The default now sits at the workspace root
        # (default read-only, 'denied' + blacklisted by pattern). An
        # existing legacy config is migrated once (see load_sandbox).
        self._legacy_sandbox_path: Path | None = None
        if sandbox_path is not None:
            self.sandbox_path = Path(sandbox_path)
        else:
            self.sandbox_path = self.workspace_root / "sandbox.json"
            legacy = self.workspace_root / "research" / "data" / "sandbox.json"
            if legacy.exists() and not self.sandbox_path.exists():
                self._legacy_sandbox_path = legacy

        # Access rules: {path_pattern: access_level}
        self.access_rules: dict[str, str] = {}

        # Website controls
        self.website_whitelist: set[str] = set()
        self.website_blacklist: set[str] = set()

        # Auto mode for risky operations
        self.auto_mode = auto_mode

        # File blacklist (always blocked for writes, reading OK)
        self.file_blacklist: list[str] = list(DEFAULT_FILE_BLACKLIST_PATTERNS)

        # Command patterns
        self.risky_patterns = [re.compile(p, re.IGNORECASE) for p in RISKY_COMMAND_PATTERNS]
        self.refusal_patterns = [re.compile(p, re.IGNORECASE) for p in HARD_REFUSAL_PATTERNS]

        # Pending permission requests (for "ask" mode)
        self.pending_requests: list[dict] = []

        # Load sandbox config
        self.load_sandbox()

    # ── Sandbox file I/O ──────────────────────────────────────────────

    def load_sandbox(self):
        """Load sandbox configuration from the sandbox.json file.

        If the file doesn't exist, creates it with default config.
        A legacy config left over in research/data (EC-160) is migrated
        to the current sandbox_path once, then saved there.
        """
        load_path = self.sandbox_path
        migrated = False
        if (not load_path.exists() and self._legacy_sandbox_path is not None
                and self._legacy_sandbox_path.exists()):
            load_path = self._legacy_sandbox_path
            migrated = True

        if load_path.exists():
            try:
                with open(load_path, encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # Fall back to defaults on parse error
                config = dict(DEFAULT_SANDBOX_CONFIG)
                config["_parse_error"] = str(e)
        else:
            config = dict(DEFAULT_SANDBOX_CONFIG)
            # Create the file with defaults
            self.save_sandbox(config)

        self.access_rules = config.get("access_rules", {})
        # Validate access levels
        self.access_rules = {
            k: v for k, v in self.access_rules.items()
            if v in VALID_ACCESS_LEVELS
        }

        # Hard floor: these paths are always denied regardless of what a
        # stale (or hand-edited) config file says — the policy file itself
        # and the usual credential locations.
        for floor in ("sandbox.json", ".env", ".git", ".ssh"):
            self.access_rules[floor] = ACCESS_DENIED

        self.website_whitelist = set(config.get("website_whitelist", []))
        self.website_blacklist = set(config.get("website_blacklist", []))

        if "auto_mode" in config:
            self.auto_mode = config["auto_mode"]

        if migrated:
            # Persist the migrated policy at the protected location.
            self._save_current_sandbox()
            self._legacy_sandbox_path = None

    def save_sandbox(self, config: dict | None = None):
        """Save sandbox configuration to the sandbox.json file.

        Args:
            config: full config dict. If None, saves current state.
        """
        if config is None:
            config = self.get_config()
            # Remove non-serializable fields
            config.pop("workspace_root", None)
            config.pop("pending_requests", None)
            config.pop("risky_pattern_count", None)
            config.pop("refusal_pattern_count", None)
            config.pop("dangerous_import_count", None)
            config.pop("sandbox_path", None)

        # Ensure parent dir exists
        self.sandbox_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.sandbox_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def reload_sandbox(self):
        """Reload sandbox config from disk (picks up external changes)."""
        self.load_sandbox()

    # ── Path resolution ───────────────────────────────────────────────

    def _resolve(self, path: str) -> str:
        """Resolve a path relative to workspace root."""
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_root / p
        return str(p.resolve())

    def _rel_path(self, path: str) -> str:
        """Get path relative to workspace root (forward slashes)."""
        resolved = self._resolve(path)
        p = Path(resolved)
        try:
            rel = p.relative_to(self.workspace_root)
        except ValueError:
            return str(p)
        return str(rel).replace("\\", "/")

    def _match_glob(self, path: str, pattern: str) -> bool:
        """Check if a path matches a glob pattern."""
        from fnmatch import fnmatch
        norm_path = path.replace("\\", "/")
        norm_pattern = pattern.replace("\\", "/")
        return fnmatch(norm_path, norm_pattern)

    # ── Access rule lookup ────────────────────────────────────────────

    def _get_access_level(self, rel_path: str) -> str:
        """Get the access level for a relative path.

        Checks access_rules in order. The most specific match wins.
        If no rule matches, defaults to read_only.

        Returns: 'read_write', 'read_only', or 'denied'
        """
        norm_path = rel_path.replace("\\", "/")

        # Find all matching rules, pick the most specific (longest pattern)
        best_match = None
        best_length = 0

        for pattern, level in self.access_rules.items():
            norm_pattern = pattern.replace("\\", "/")

            # Check if path is under this pattern
            if norm_path == norm_pattern or norm_path.startswith(norm_pattern + "/"):
                # Direct prefix match
                if len(norm_pattern) > best_length:
                    best_match = level
                    best_length = len(norm_pattern)
            elif self._match_glob(norm_path, norm_pattern):
                # Glob match
                if len(norm_pattern) > best_length:
                    best_match = level
                    best_length = len(norm_pattern)

        return best_match or ACCESS_READ_ONLY

    # ── File path checks ──────────────────────────────────────────────

    def check_file_read(self, path: str) -> SecurityDecision:
        """Check if a file can be read.

        Reading is allowed within workspace EXCEPT for:
        - paths with 'denied' access level
        - paths matching the file blacklist (.env, *.pem, id_rsa, *.key,
          .ssh, sandbox.json, ...) — credentials must never enter the
          model's context (EC-027/EC-097)
        """
        resolved = self._resolve(path)
        p = Path(resolved)

        # Must be within workspace
        try:
            p.relative_to(self.workspace_root)
        except ValueError:
            return SecurityDecision(
                allowed=False,
                reason=f"Path outside workspace: {path}",
                severity="warning",
            )

        rel_path = self._rel_path(path)
        access = self._get_access_level(rel_path)

        if access == ACCESS_DENIED:
            return SecurityDecision(
                allowed=False,
                reason=f"Path is denied in sandbox: {rel_path}",
                severity="warning",
                details={"access_level": access, "path": rel_path},
            )

        # Credential/secret blacklist applies to reads too (EC-027).
        for pattern in self._effective_blacklist():
            if self._match_glob(rel_path, pattern) or self._match_glob(str(p), pattern):
                return SecurityDecision(
                    allowed=False,
                    reason=f"File matches blacklist pattern: {pattern}",
                    severity="critical",
                    details={"pattern": pattern, "path": rel_path},
                )

        # Reading is allowed for both read_only and read_write
        return SecurityDecision(allowed=True, details={"access_level": access})

    def _is_protected_engine_file(self, rel_path: str) -> str | None:
        """Return the matching PROTECTED_ENGINE_FILES/_GLOBS entry, else None.

        Exact repo-relative match only — the previous ``endswith`` match
        refused writes to *any* ``config.py``/``paths.py``/``myconfig.py``
        anywhere in the tree (EC-027). Globs use fnmatch against the
        normalized relative path (A18: covers generated file families
        like forge/evolution/targets/auto_*.py).
        """
        norm = rel_path.replace("\\", "/")
        for protected in PROTECTED_ENGINE_FILES:
            if norm == protected:
                return protected
        for pattern in PROTECTED_ENGINE_GLOBS:
            if self._match_glob(norm, pattern):
                return pattern
        return None

    def check_file_write(self, path: str, content: str = "") -> SecurityDecision:
        """Check if a file can be written/created.

        Security model:
        1. Path must be within workspace
        2. Path must have 'read_write' access in sandbox (outside sandbox = blacklisted)
        3. Path must not match file blacklist patterns (secrets, .env, etc.)
        4. Path must not be a protected engine file
        5. Content must not match hard refusal patterns
        6. Content is scanned for risky command patterns (→ needs_permission)
        7. Python scripts are scanned for dangerous imports
        """
        resolved = self._resolve(path)
        p = Path(resolved)

        # 1. Must be within workspace
        try:
            p.relative_to(self.workspace_root)
        except ValueError:
            return SecurityDecision(
                allowed=False,
                reason=f"Path outside workspace: {path}",
                severity="critical",
            )

        rel_path = self._rel_path(path)

        # 2. Check sandbox access level — must be read_write
        access = self._get_access_level(rel_path)
        if access != ACCESS_READ_WRITE:
            return SecurityDecision(
                allowed=False,
                reason=f"Path not in sandbox writable area (access={access}): {rel_path}",
                severity="warning",
                details={"access_level": access, "path": rel_path},
            )

        # 3. Check file blacklist patterns (always blocked for writes)
        for pattern in self._effective_blacklist():
            if self._match_glob(rel_path, pattern) or self._match_glob(str(p), pattern):
                return SecurityDecision(
                    allowed=False,
                    reason=f"File matches blacklist pattern: {pattern}",
                    severity="critical",
                    details={"pattern": pattern, "path": rel_path},
                )

        # 4. Check protected engine files (exact repo-relative match)
        protected = self._is_protected_engine_file(rel_path)
        if protected is not None:
            return SecurityDecision(
                allowed=False,
                reason=f"Cannot modify protected engine file: {protected}",
                severity="critical",
                details={"protected_file": protected, "path": rel_path},
            )

        # 5. Check content for hard refusal patterns
        if content:
            for regex in self.refusal_patterns:
                match = regex.search(content)
                if match:
                    return SecurityDecision(
                        allowed=False,
                        reason=f"Content matches forbidden pattern: {regex.pattern}",
                        severity="critical",
                        details={"pattern": regex.pattern, "match": match.group()[:100]},
                    )

        # 6. Check content for risky command patterns
        risky_findings: list[dict] = []
        if content:
            for regex in self.risky_patterns:
                matches = regex.findall(content)
                if matches:
                    risky_findings.append({
                        "pattern": regex.pattern,
                        "count": len(matches),
                        "sample": matches[0] if isinstance(matches[0], str) else str(matches[0])[:100],
                    })

        # 7. Scan Python scripts for dangerous imports
        is_python = str(p).endswith(".py")
        dangerous_imports: list[dict] = []
        if is_python and content:
            dangerous_imports = self._scan_python_imports(content)
            if dangerous_imports:
                for imp in dangerous_imports:
                    risky_findings.append({
                        "pattern": f"dangerous_import:{imp['module']}",
                        "count": 1,
                        "sample": imp["reason"],
                    })

        # If there are risky findings, decide based on auto_mode
        if risky_findings:
            if self.auto_mode == "allow":
                return SecurityDecision(
                    allowed=True,
                    reason=f"Risky patterns detected but auto-approved (mode=allow): {len(risky_findings)} findings",
                    severity="warning",
                    details={"risky_findings": risky_findings},
                )
            elif self.auto_mode == "deny":
                return SecurityDecision(
                    allowed=False,
                    reason=f"Risky patterns detected and auto-denied (mode=deny): {len(risky_findings)} findings",
                    severity="warning",
                    details={"risky_findings": risky_findings},
                )
            else:  # "ask"
                request_id = f"perm_{len(self.pending_requests)}"
                req = {
                    "id": request_id,
                    "type": "file_write",
                    "path": rel_path,
                    "risky_findings": risky_findings,
                    "content_preview": content[:500],
                }
                self.pending_requests.append(req)
                return SecurityDecision(
                    allowed=False,
                    needs_permission=True,
                    reason=f"Risky patterns detected: {len(risky_findings)} findings. Permission required.",
                    severity="warning",
                    details={"risky_findings": risky_findings, "request_id": request_id},
                )

        return SecurityDecision(allowed=True, details={"access_level": access})

    def check_file_delete(self, path: str) -> SecurityDecision:
        """Check if a file can be deleted.

        Same rules as write, plus always needs permission (unless auto_mode=allow).
        """
        resolved = self._resolve(path)
        p = Path(resolved)

        # Must be within workspace
        try:
            p.relative_to(self.workspace_root)
        except ValueError:
            return SecurityDecision(
                allowed=False,
                reason=f"Path outside workspace: {path}",
                severity="critical",
            )

        rel_path = self._rel_path(path)

        # Check sandbox access — must be read_write
        access = self._get_access_level(rel_path)
        if access != ACCESS_READ_WRITE:
            return SecurityDecision(
                allowed=False,
                reason=f"Path not in sandbox writable area (access={access}): {rel_path}",
                severity="warning",
                details={"access_level": access},
            )

        # Check file blacklist
        for pattern in self._effective_blacklist():
            if self._match_glob(rel_path, pattern) or self._match_glob(str(p), pattern):
                return SecurityDecision(
                    allowed=False,
                    reason=f"File matches blacklist pattern: {pattern}",
                    severity="critical",
                )

        # Check protected engine files (exact repo-relative match)
        protected = self._is_protected_engine_file(rel_path)
        if protected is not None:
            return SecurityDecision(
                allowed=False,
                reason=f"Cannot delete protected engine file: {protected}",
                severity="critical",
            )

        # Deletes always need permission (unless auto_mode=allow)
        if self.auto_mode == "allow":
            return SecurityDecision(
                allowed=True,
                reason="Delete auto-approved (mode=allow)",
                severity="info",
            )

        request_id = f"perm_{len(self.pending_requests)}"
        req = {
            "id": request_id,
            "type": "file_delete",
            "path": rel_path,
        }
        self.pending_requests.append(req)
        return SecurityDecision(
            allowed=False,
            needs_permission=True,
            reason="File deletion requires permission",
            severity="warning",
            details={"request_id": request_id},
        )

    def check_file_move(self, source: str, destination: str) -> SecurityDecision:
        """Check if a file can be moved. Validates both source and destination."""
        # Source must be readable
        read_check = self.check_file_read(source)
        if not read_check.allowed:
            return read_check

        # Source must not be a protected engine file or blacklisted
        rel_src = self._rel_path(source)
        for pattern in self._effective_blacklist():
            if self._match_glob(rel_src, pattern):
                return SecurityDecision(
                    allowed=False,
                    reason=f"Source file matches blacklist pattern: {pattern}",
                    severity="critical",
                )
        protected = self._is_protected_engine_file(rel_src)
        if protected is not None:
            return SecurityDecision(
                allowed=False,
                reason=f"Cannot move protected engine file: {protected}",
                severity="critical",
            )

        # Destination must be writable
        write_check = self.check_file_write(destination)
        if not write_check.allowed:
            return write_check

        # Moves/renames are destructive to the source path — require the
        # same approval tier as deletes (EC-158) unless auto-approved.
        if self.auto_mode == "allow":
            return SecurityDecision(
                allowed=True,
                reason="Move auto-approved (mode=allow)",
                severity="info",
            )

        request_id = f"perm_{len(self.pending_requests)}"
        req = {
            "id": request_id,
            "type": "file_move",
            "path": rel_src,
            "destination": self._rel_path(destination),
        }
        self.pending_requests.append(req)
        return SecurityDecision(
            allowed=False,
            needs_permission=True,
            reason="File move/rename requires permission",
            severity="warning",
            details={"request_id": request_id},
        )

    def check_file_rename(self, path: str, new_name: str) -> SecurityDecision:
        """Check if a file/dir can be renamed."""
        resolved = self._resolve(path)
        p = Path(resolved)
        new_path = p.parent / new_name
        try:
            rel_new = str(new_path.relative_to(self.workspace_root)).replace("\\", "/")
        except ValueError:
            return SecurityDecision(
                allowed=False,
                reason="Renamed path would be outside workspace",
                severity="critical",
            )
        return self.check_file_move(path, rel_new)

    # ── Script pre-scan ───────────────────────────────────────────────

    def _scan_python_imports(self, content: str) -> list[dict]:
        """Scan Python code for dangerous imports.

        Returns list of {module, reason, line} for each dangerous import found.
        """
        findings: list[dict] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return findings

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in DANGEROUS_IMPORTS:
                        findings.append({
                            "module": module,
                            "reason": DANGEROUS_IMPORTS[module],
                            "line": getattr(node, "lineno", 0),
                        })
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split(".")[0]
                if module in DANGEROUS_IMPORTS:
                    findings.append({
                        "module": module,
                        "reason": DANGEROUS_IMPORTS[module],
                        "line": getattr(node, "lineno", 0),
                    })

        # Also check for string-based dynamic imports
        dynamic_patterns = [
            (r"__import__\s*\(['\"](\w+)", "Dynamic __import__ call"),
            (r"importlib\.import_module\s*\(['\"](\w+)", "Dynamic importlib call"),
            (r"importlib\.__import__\s*\(['\"](\w+)", "Dynamic importlib call"),
        ]
        for pattern, reason in dynamic_patterns:
            for match in re.finditer(pattern, content):
                module = match.group(1)
                if module in DANGEROUS_IMPORTS:
                    findings.append({
                        "module": module,
                        "reason": f"{reason}: {DANGEROUS_IMPORTS[module]}",
                        "line": content[:match.start()].count("\n") + 1,
                    })

        return findings

    def scan_script(self, content: str) -> dict:
        """Full script scan. Returns detailed report.

        Returns:
            {
                "dangerous_imports": list[dict],
                "risky_patterns": list[dict],
                "refusal_patterns": list[dict],
                "verdict": "allow" | "needs_permission" | "refuse",
                "summary": str,
            }
        """
        dangerous_imports = self._scan_python_imports(content)

        risky_found: list[dict] = []
        for regex in self.risky_patterns:
            matches = regex.findall(content)
            if matches:
                risky_found.append({
                    "pattern": regex.pattern,
                    "count": len(matches),
                })

        refusal_found: list[dict] = []
        for regex in self.refusal_patterns:
            matches = regex.findall(content)
            if matches:
                refusal_found.append({
                    "pattern": regex.pattern,
                    "count": len(matches),
                })

        if refusal_found:
            verdict = "refuse"
            summary = f"Script contains {len(refusal_found)} forbidden pattern(s)"
        elif risky_found or dangerous_imports:
            if self.auto_mode == "allow":
                verdict = "allow"
                summary = f"Script has {len(risky_found)} risky pattern(s), {len(dangerous_imports)} dangerous import(s) — auto-approved"
            elif self.auto_mode == "deny":
                verdict = "refuse"
                summary = f"Script has {len(risky_found)} risky pattern(s), {len(dangerous_imports)} dangerous import(s) — auto-denied"
            else:
                verdict = "needs_permission"
                summary = f"Script has {len(risky_found)} risky pattern(s), {len(dangerous_imports)} dangerous import(s) — needs permission"
        else:
            verdict = "allow"
            summary = "Script appears safe"

        return {
            "dangerous_imports": dangerous_imports,
            "risky_patterns": risky_found,
            "refusal_patterns": refusal_found,
            "verdict": verdict,
            "summary": summary,
        }

    # ── Website checks ────────────────────────────────────────────────

    def check_website(self, url: str) -> SecurityDecision:
        """Check if a website URL/domain is allowed.

        Rules:
        - If whitelist is empty, all sites are allowed (unless blacklisted)
        - If whitelist is non-empty, only whitelisted domains are allowed
        - Blacklist always takes precedence
        """
        domain = self._extract_domain(url)

        # Check blacklist first
        for blocked in self.website_blacklist:
            if self._domain_matches(domain, blocked):
                return SecurityDecision(
                    allowed=False,
                    reason=f"Domain '{domain}' is blacklisted",
                    severity="warning",
                    details={"domain": domain, "blocked_by": blocked},
                )

        # Check whitelist (if non-empty)
        if self.website_whitelist:
            allowed = any(
                self._domain_matches(domain, allowed_domain)
                for allowed_domain in self.website_whitelist
            )
            if not allowed:
                return SecurityDecision(
                    allowed=False,
                    reason=f"Domain '{domain}' not in whitelist",
                    severity="warning",
                    details={"domain": domain, "whitelist": list(self.website_whitelist)},
                )

        return SecurityDecision(allowed=True, details={"domain": domain})

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = parsed.netloc or parsed.path.split("/")[0]
        domain = domain.split(":")[0]
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()

    def _domain_matches(self, domain: str, pattern: str) -> bool:
        """Check if a domain matches a pattern (supports wildcards)."""
        domain = domain.lower()
        pattern = pattern.lower().strip()
        if pattern.startswith("*."):
            suffix = pattern[2:]
            return domain == suffix or domain.endswith(f".{suffix}")
        return domain == pattern or domain.endswith(f".{pattern}")

    # ── Permission management ─────────────────────────────────────────

    def get_pending_requests(self) -> list[dict]:
        """Get all pending permission requests."""
        return list(self.pending_requests)

    def approve_request(self, request_id: str) -> bool:
        """Approve a pending permission request."""
        for i, req in enumerate(self.pending_requests):
            if req["id"] == request_id:
                req["approved"] = True
                self.pending_requests.pop(i)
                return True
        return False

    def deny_request(self, request_id: str) -> bool:
        """Deny a pending permission request."""
        for i, req in enumerate(self.pending_requests):
            if req["id"] == request_id:
                req["approved"] = False
                self.pending_requests.pop(i)
                return True
        return False

    def clear_pending(self):
        """Clear all pending permission requests."""
        self.pending_requests.clear()

    # ── Sandbox config management ─────────────────────────────────────

    def set_access_rule(self, path: str, level: str):
        """Set or update an access rule for a path.

        Args:
            path: path pattern (relative to workspace root)
            level: 'read_write', 'read_only', or 'denied'
        """
        if level not in VALID_ACCESS_LEVELS:
            raise ValueError(f"Invalid access level: {level}. Must be one of {VALID_ACCESS_LEVELS}")
        self.access_rules[path] = level
        self._save_current_sandbox()

    def remove_access_rule(self, path: str):
        """Remove an access rule for a path."""
        self.access_rules.pop(path, None)
        self._save_current_sandbox()

    def get_access_rules(self) -> dict[str, str]:
        """Get all access rules."""
        return dict(self.access_rules)

    def get_access_level(self, path: str) -> str:
        """Get the access level for a path (public API)."""
        rel_path = self._rel_path(path)
        return self._get_access_level(rel_path)

    def _effective_blacklist(self) -> list[str]:
        """Blacklist actually enforced: hard-coded secret defaults are a
        non-removable floor — remote/local config mutation can only ADD
        patterns, never drop the credential/.git/sandbox.json protections.
        """
        merged = list(dict.fromkeys(
            list(DEFAULT_FILE_BLACKLIST_PATTERNS) + list(self.file_blacklist)))
        return merged

    def add_to_file_blacklist(self, pattern: str):
        """Add a file pattern to the blacklist."""
        if pattern not in self.file_blacklist:
            self.file_blacklist.append(pattern)

    def remove_from_file_blacklist(self, pattern: str):
        """Remove a file pattern from the blacklist.

        Refuses to drop DEFAULT_FILE_BLACKLIST_PATTERNS entries — they are
        the hard deny floor (EC-027): /v1/security/config PATCH could
        otherwise un-blacklist .env/.ssh/sandbox.json remotely.
        """
        if pattern in DEFAULT_FILE_BLACKLIST_PATTERNS:
            raise ValueError(
                f"'{pattern}' is a default protected pattern and cannot "
                "be removed")
        self.file_blacklist = [p for p in self.file_blacklist if p != pattern]

    def add_to_website_whitelist(self, domain: str):
        """Add a domain to the website whitelist."""
        self.website_whitelist.add(domain.lower())
        self._save_current_sandbox()

    def remove_from_website_whitelist(self, domain: str):
        """Remove a domain from the website whitelist."""
        self.website_whitelist.discard(domain.lower())
        self._save_current_sandbox()

    def add_to_website_blacklist(self, domain: str):
        """Add a domain to the website blacklist."""
        self.website_blacklist.add(domain.lower())
        self._save_current_sandbox()

    def remove_from_website_blacklist(self, domain: str):
        """Remove a domain from the website blacklist."""
        self.website_blacklist.discard(domain.lower())
        self._save_current_sandbox()

    def set_auto_mode(self, mode: str):
        """Set auto-approval mode for risky operations.

        Args:
            mode: "allow" (auto-approve), "deny" (auto-deny), "ask" (flag for user)
        """
        if mode not in ("allow", "deny", "ask"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'allow', 'deny', or 'ask'")
        self.auto_mode = mode
        self._save_current_sandbox()

    def _save_current_sandbox(self):
        """Save current state to sandbox.json."""
        config = {
            "auto_mode": self.auto_mode,
            "access_rules": dict(self.access_rules),
            "website_whitelist": list(self.website_whitelist),
            "website_blacklist": list(self.website_blacklist),
        }
        self.save_sandbox(config)

    def get_config(self) -> dict:
        """Get the current security configuration."""
        return {
            "sandbox_path": str(self.sandbox_path),
            "workspace_root": str(self.workspace_root),
            "auto_mode": self.auto_mode,
            "access_rules": dict(self.access_rules),
            "file_blacklist": self.file_blacklist,
            "website_whitelist": list(self.website_whitelist),
            "website_blacklist": list(self.website_blacklist),
            "pending_requests": len(self.pending_requests),
            "risky_pattern_count": len(self.risky_patterns),
            "refusal_pattern_count": len(self.refusal_patterns),
            "dangerous_import_count": len(DANGEROUS_IMPORTS),
            "protected_engine_files": list(PROTECTED_ENGINE_FILES),
            "protected_engine_globs": list(PROTECTED_ENGINE_GLOBS),
        }
