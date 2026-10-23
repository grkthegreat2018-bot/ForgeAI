"""Regression tests for tool-security fixes (EC-027/093/097/119/158/160/163/213).

Covers:
- EC-093: math/calc/random/chance handlers actually run (self._ast/_random bug)
- EC-097: file_read/file_list are gated by check_file_read; .env/creds denied
- EC-027: read-side blacklist + exact-path PROTECTED_ENGINE_FILES matching
- EC-119: count/AST/rglob caps, malformed-JSON args -> error (not silent {})
- EC-158: read vs destructive tiers — moves require permission in "ask" mode
- EC-160: sandbox.json lives at workspace root (not tool-writable research/data)
- EC-163: untrusted tool outputs wrapped in a tagged block
- EC-213: web_fetch SSRF — private/loopback/link-local hosts refused

CPU-only; no network access required (all SSRF cases are literal IPs or
loopback names that never reach urlopen).
"""
from __future__ import annotations

import threading
import types

import pytest

from forge.engine.engine_tools import EngineToolRegistry
from forge.engine.tool_security import ToolSecurityManager
from forge.web_primitives import fetch_url, is_public_host


# ── fakes ──────────────────────────────────────────────────────────────────

class _FakeHotswap:
    def set_kv_cache(self, s): pass
    def set_decoding(self, s): pass
    def set_context_limit(self, n): pass
    def set_infinite_context(self, e, b): pass
    def set_generation_defaults(self, **kw): pass
    def set_feature(self, f, e): pass
    def apply_pending(self): return ["kv_cache"]
    def get_settings(self): return {"kv_cache": "standard"}
    def has_pending(self): return False
    def get_pending_changes(self): return {}


class _FakeLibrary:
    def __init__(self):
        self._lock = threading.Lock()
        self.tokenizer = None
        self.entry = types.SimpleNamespace(
            id="e1", content="c", description="d", category="custom",
            tags=[], triggers=[], priority=0, token_count=1,
            token_ids=[1], access_count=0)

    def get(self, entry_id): return self.entry if entry_id == "e1" else None
    def delete(self, entry_id): return True
    def _remove_from_indices(self, eid): pass
    def _add_to_indices(self, e): pass
    def _save_entry(self, e): pass


class _FakeEngine:
    def __init__(self):
        self.library = _FakeLibrary()
        self.hotswap = _FakeHotswap()
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.decoding = types.SimpleNamespace(name="standard")
        self.acceleration = None
        self.quantize = None
        self._library_enabled = True
        self._library_injection_budget = 100

    def library_save(self, **kw): return "e1"
    def library_search(self, q, limit=10): return []
    def library_lookup(self, tags=None, category=None, limit=20): return []
    def library_stats(self): return {"entries": 0}
    def library_optimize(self): return {"merged": 0}
    def library_set_enabled(self, v): self._library_enabled = v
    def library_set_budget(self, v): self._library_injection_budget = v
    def generate_batch(self, prompts, **kw): return ["ok"] * len(prompts)
    def generate_adaptive(self, prompt, **kw): return ("text", False)


@pytest.fixture
def sec(tmp_path):
    (tmp_path / "research" / "data" / "notes").mkdir(parents=True)
    return ToolSecurityManager(workspace_root=str(tmp_path), auto_mode="allow")


@pytest.fixture
def reg(sec):
    r = EngineToolRegistry(_FakeEngine(), security=sec)
    # web handlers must never hit the network in tests
    r._get_api_key = lambda name: None
    return r


# ── EC-093: every registered handler runs (no 'has no attribute') ───────────

_SMOKE_ARGS = {
    "library_save": {"content": "x", "category": "custom"},
    "library_search": {"query": "x"},
    "library_lookup": {},
    "library_get": {"entry_id": "e1"},
    "library_delete": {"entry_id": "e1"},
    "library_stats": {},
    "library_optimize": {},
    "library_set_config": {"enabled": False, "injection_budget": 5},
    "library_update": {"entry_id": "e1", "content": "new"},
    "engine_set_kv_cache": {"strategy": "standard"},
    "engine_set_decoding": {"strategy": "standard"},
    "engine_set_context_limit": {"max_tokens": 4096},
    "engine_enable_infinite_context": {"enabled": True},
    "engine_set_generation_params": {"temperature": 0.5},
    "engine_set_feature": {"flag": "use_prefix_cache", "enabled": True},
    "engine_apply_changes": {},
    "engine_get_settings": {},
    "engine_get_stats": {},
    "engine_get_pending": {},
    "engine_batch_generate": {"prompts": ["hi"]},
    "engine_generate_adaptive": {"prompt": "hi"},
    "math_eval": {"expression": "sqrt(144) + 2**10"},
    "calc": {"expression": "2*3+4"},
    "random_number": {"min": 1, "max": 10, "count": 3, "seed": 7},
    "chance": {"mode": "dice", "sides": 6, "count": 2, "seed": 7},
    "web_search": {"query": "x"},
    "web_search_semantic": {"query": "x"},
    "web_scrape": {"url": "https://example.com"},
    "file_read": {"path": "research/data/notes/a.txt"},
    "file_write": {"path": "research/data/notes/a.txt", "content": "hello"},
    "file_edit": {"path": "research/data/notes/a.txt",
                  "old_string": "hello", "new_string": "bye"},
    "file_move": {"source": "research/data/notes/a.txt",
                  "destination": "research/data/notes/b.txt"},
    "file_rename": {"path": "research/data/notes/b.txt", "new_name": "c.txt"},
    "file_list": {"path": "research/data/notes"},
    "file_delete": {"path": "research/data/notes/c.txt", "confirm": True},
    "scan_script": {"content": "print(1)"},
    "security_get_config": {},
    "security_get_pending": {},
}


def test_every_registered_handler_smoke(reg, sec):
    """Every registered handler must execute without an AttributeError —
    EC-093 shipped handlers that always failed on self._ast/self._random."""
    missing = set(_SMOKE_ARGS) - reg.get_tool_names()
    assert not missing, f"smoke args missing for: {missing}"
    notes = sec.workspace_root / "research/data/notes"
    # file ops are independent — recreate each op's target so call order
    # doesn't matter
    prep = {
        "file_read": ("a.txt", "hello"), "file_edit": ("a.txt", "hello"),
        "file_move": ("a.txt", "hello"), "file_rename": ("b.txt", "x"),
        "file_delete": ("c.txt", "x"),
    }
    for name in sorted(reg.get_tool_names()):
        if name in prep:
            fn, content = prep[name]
            (notes / fn).write_text(content)
        res = reg.execute(name, _SMOKE_ARGS[name])
        err = res.get("error", "") or ""
        assert "has no attribute" not in err, f"{name}: {err}"
        assert res["status"] in ("success", "needs_permission"), \
            f"{name}: {res['status']} {err}"


def test_math_eval_and_random_actually_work(reg):
    assert reg.execute("math_eval", {"expression": "2+3*4"})["result"]["result"] == 14
    assert reg.execute("calc", {"expression": "10//3"})["result"]["result"] == 3
    rn = reg.execute("random_number", {"count": 5, "seed": 1})
    assert len(rn["result"]["values"]) == 5
    ch = reg.execute("chance", {"mode": "coin_flip", "count": 4, "seed": 1})
    assert len(ch["result"]["flips"]) == 4


# ── EC-119: model-arg hardening ──────────────────────────────────────────────

def test_math_eval_pow_bomb_rejected(reg):
    for expr in ["9**9**9", "pow(9, 9**9)", "factorial(10**9)",
                 "-" * 50 + "1", "1" + "+1" * 2000, "1" + "+1" * 150]:
        res = reg.execute("math_eval", {"expression": expr})
        assert res["status"] == "error", f"not rejected: {expr[:30]}"


def test_count_clamped(reg):
    rn = reg.execute("random_number", {"count": 10**9, "seed": 1})
    assert rn["result"]["count"] == reg._MAX_COUNT == 1000
    ch = reg.execute("chance", {"mode": "dice", "count": 10**9, "seed": 1})
    assert len(ch["result"]["rolls"]) == 1000


def test_malformed_json_args_is_error(reg):
    res = reg.execute_calls([{"name": "calc", "arguments": "{not json"}])
    assert res[0]["status"] == "error"
    res = reg.execute_calls([{"name": "calc", "arguments": [1, 2]}])
    assert res[0]["status"] == "error"


def test_file_list_entry_cap(reg, sec):
    notes = sec.workspace_root / "research/data/notes"
    for i in range(8):
        (notes / f"f{i}.txt").write_text("x")
    reg._MAX_LIST_ENTRIES = 5
    res = reg.execute("file_list", {"path": "research/data/notes"})
    data = res["result"]["untrusted_content"]["data"]
    assert data["count"] == 5
    assert data["truncated"] is True


# ── EC-027 + EC-097: reads are gated too ─────────────────────────────────────

def test_env_and_credentials_reads_refused(reg, sec):
    for rel in (".env", "research/data/notes/.env",
                "research/data/notes/id_rsa", "research/data/notes/k.pem",
                "research/data/notes/api.key", ".ssh/id_ed25519",
                "sandbox.json"):
        p = sec.workspace_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("SECRET")
        res = reg.execute("file_read", {"path": rel})
        assert res["status"] == "refused", f"read allowed: {rel}"


def test_file_list_gated_by_security(reg):
    assert reg.execute("file_list", {"path": ".env"})["status"] == "refused"
    assert reg.execute("file_list", {"path": "research/data"})["status"] == "success"


def test_protected_files_exact_path_match(sec):
    # Previously: any file ending in 'config.py'/'paths.py' was refused.
    assert sec.check_file_write(
        "research/data/notes/config.py", "x=1").allowed
    assert sec.check_file_write(
        "research/data/notes/myconfig.py", "x=1").allowed
    assert sec.check_file_write(
        "research/data/notes/paths.py", "x=1").allowed
    # The real protected files are still unwritable (exact rel-path match).
    assert not sec.check_file_write(
        "forge/engine/tool_security.py", "x=1").allowed
    assert not sec.check_file_write("forge/config.py", "x=1").allowed


# ── EC-158: destructive ops require permission in 'ask' mode ─────────────────

def test_destructive_ops_need_permission(tmp_path):
    notes = tmp_path / "research/data/notes"
    notes.mkdir(parents=True)
    (notes / "a.txt").write_text("x")
    sec = ToolSecurityManager(workspace_root=str(tmp_path), auto_mode="ask")
    for name, args in (
        ("file_delete", {"path": "research/data/notes/a.txt", "confirm": True}),
        ("file_move", {"source": "research/data/notes/a.txt",
                       "destination": "research/data/notes/b.txt"}),
        ("file_rename", {"path": "research/data/notes/a.txt", "new_name": "b.txt"}),
    ):
        r = EngineToolRegistry(_FakeEngine(), security=sec)
        assert r.execute(name, args)["status"] == "needs_permission", name
    # Reads are auto-approved — but only after the credential/deny checks.
    r = EngineToolRegistry(_FakeEngine(), security=sec)
    assert r.execute("file_read", {"path": "research/data/notes/a.txt"})["status"] == "success"


# ── EC-160: sandbox.json out of tool-writable space ──────────────────────────

def test_sandbox_config_at_root_and_unwritable(tmp_path):
    sec = ToolSecurityManager(workspace_root=str(tmp_path), auto_mode="allow")
    assert sec.sandbox_path == tmp_path / "sandbox.json"
    assert "research" not in sec.sandbox_path.parts[-3:]
    assert not sec.check_file_write("sandbox.json", "{}").allowed
    assert not sec.check_file_write("research/data/sandbox.json", "{}").allowed
    assert not sec.check_file_read("sandbox.json").allowed


def test_legacy_sandbox_migrated(tmp_path):
    legacy = tmp_path / "research/data/sandbox.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        '{"access_rules": {"research/data": "read_write", "custom/zone": "read_write"}}')
    sec = ToolSecurityManager(workspace_root=str(tmp_path))
    assert sec.sandbox_path == tmp_path / "sandbox.json"
    assert sec.access_rules.get("custom/zone") == "read_write"   # carried over
    assert sec.access_rules["sandbox.json"] == "denied"           # hard floor
    assert (tmp_path / "sandbox.json").exists()                   # persisted


# ── EC-163: untrusted tool outputs are tagged ────────────────────────────────

def test_untrusted_output_wrapped(reg, sec):
    (sec.workspace_root / "research/data/notes/a.txt").write_text("hello")
    res = reg.execute("file_read", {"path": "research/data/notes/a.txt"})
    assert res["status"] == "success"
    block = res["result"]["untrusted_content"]
    assert block["source"] == "file_read"
    assert "UNTRUSTED" in block["notice"]
    # trusted tools are NOT wrapped
    res2 = reg.execute("calc", {"expression": "1+1"})
    assert "untrusted_content" not in res2["result"]


# ── EC-213: SSRF ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("host", [
    "127.0.0.1", "127.53.0.9", "0.0.0.0", "::1",
    "10.0.0.1", "172.16.0.1", "172.31.255.1", "192.168.1.1",
    "169.254.169.254", "100.64.0.1",
    "localhost", "metadata.google.internal", "foo.local",
])
def test_private_hosts_rejected(host):
    assert not is_public_host(host)


def test_public_host_allowed():
    assert is_public_host("172.15.0.1")   # just outside 172.16/12
    assert is_public_host("8.8.8.8")


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8080/admin",
    "http://169.254.169.254/latest/meta-data", "http://10.0.0.1",
    "http://[::1]/", "192.168.0.1",
])
def test_fetch_url_blocks_private(url):
    res = fetch_url(url)
    assert res["error"] is not None
    assert "blocked host" in res["error"] or "scheme" in res["error"]
