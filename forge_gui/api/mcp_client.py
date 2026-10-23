"""MCP (Model Context Protocol) client support for ForgeAI.

Implements a lightweight MCP client that connects to external MCP servers
via stdio or HTTP/SSE transport. Discovered tools are exposed as ForgeAI
tool definitions that the unified harness can dispatch.

Architecture:
- ``MCPClient`` — base client with JSON-RPC 2.0 protocol
- ``StdioMCPClient`` — launches a subprocess and communicates via stdin/stdout
- ``HTTPMCPClient`` — connects to an HTTP/SSE endpoint
- ``MCPManager`` — manages multiple MCP servers, aggregates tools

The MCP protocol is JSON-RPC 2.0 over stdio (line-delimited) or HTTP.
Key methods:
- initialize: handshake with server capabilities
- tools/list: discover available tools
- tools/call: invoke a tool with arguments

Configuration is stored in ``data/mcp/servers.json``:
    {
        "servers": [
            {
                "name": "github",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "..."}
            },
            {
                "name": "linear",
                "transport": "http",
                "url": "http://localhost:3001/mcp"
            }
        ]
    }
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── JSON-RPC types ──────────────────────────────────────────────────────

@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)
    server_name: str = ""       # which MCP server provides this tool

    def to_openai_def(self) -> dict:
        """Convert to OpenAI function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or f"MCP tool from {self.server_name}",
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


@dataclass
class MCPServerConfig:
    """Configuration for one MCP server."""
    name: str
    transport: str = "stdio"     # "stdio" | "http"
    command: str = ""            # stdio: executable
    args: list[str] = field(default_factory=list)  # stdio: args
    env: dict[str, str] = field(default_factory=dict)  # stdio: env vars
    url: str = ""                # http: endpoint URL
    enabled: bool = True
    auto_start: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> MCPServerConfig:
        return cls(
            name=d.get("name", "unnamed"),
            transport=d.get("transport", "stdio"),
            command=d.get("command", ""),
            args=d.get("args", []),
            env=d.get("env", {}),
            url=d.get("url", ""),
            enabled=d.get("enabled", True),
            auto_start=d.get("auto_start", True),
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ── Base MCP client ─────────────────────────────────────────────────────

class MCPClient:
    """Base MCP client — JSON-RPC 2.0 protocol."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.tools: list[MCPTool] = []
        self._connected = False
        self._request_id = 0
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connected

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and return the result. Override in subclasses."""
        raise NotImplementedError

    def initialize(self) -> dict:
        """Perform MCP handshake."""
        result = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "forgeai",
                "version": "1.0",
            },
        })
        self._connected = True
        # send initialized notification
        try:
            self._send_notification("notifications/initialized", {})
        except Exception:
            logger.debug("Failed to send initialized notification to MCP server", exc_info=True)
        return result

    def list_tools(self) -> list[MCPTool]:
        """Discover tools from the server."""
        result = self._send_request("tools/list", {})
        tools = []
        for t in result.get("tools", []):
            tools.append(MCPTool(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                server_name=self.config.name,
            ))
        self.tools = tools
        return tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Invoke a tool on the server."""
        result = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        return result

    def close(self) -> None:
        """Disconnect from the server. Override in subclasses."""
        self._connected = False

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a notification (no response expected). Override in subclasses."""
        pass


# ── Stdio transport ─────────────────────────────────────────────────────

# Sentinel pushed onto the rx queue when the server's stdout closes.
_PROC_EOF = object()



class StdioMCPClient(MCPClient):
    """MCP client over stdio — launches a subprocess and communicates via pipes.

    Reads go through a dedicated stdout pump thread feeding a queue:
    ``_send_request`` then waits for the response whose JSON-RPC ``id``
    matches its request (notifications/responses for other ids are
    skipped) under ``_io_lock``, with a hard read timeout — a hung or
    chatty server used to block the caller on ``readline()`` forever and
    could return someone else's response.
    """

    # Max time to wait for OUR response after writing a request.
    READ_TIMEOUT_S = 30.0
    # Max response lines to scan past while looking for our id.
    MAX_SKIP_LINES = 100

    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._proc: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._rx: queue.Queue = queue.Queue()
        self._io_lock = threading.Lock()   # serialize write→matched read

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise ConnectionError("MCP subprocess not running")
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            req["params"] = params
        line = json.dumps(req) + "\n"
        with self._io_lock:
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise ConnectionError(
                    f"MCP subprocess stdin broken: {e}") from e
            # Read lines until our id answers — the server may interleave
            # notifications or log lines on stdout.
            deadline = time.monotonic() + self.READ_TIMEOUT_S
            skipped = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"MCP '{method}' timed out after "
                        f"{self.READ_TIMEOUT_S}s")
                try:
                    resp = self._rx.get(timeout=min(remaining, 5.0))
                except queue.Empty:
                    if self._proc.poll() is not None:
                        raise ConnectionError(
                            "MCP subprocess exited while awaiting "
                            "response")
                    continue
                if resp is _PROC_EOF:
                    raise ConnectionError("MCP subprocess closed stdout")
                if not isinstance(resp, dict):
                    continue
                if resp.get("id") != req["id"]:
                    skipped += 1
                    if skipped > self.MAX_SKIP_LINES:
                        raise RuntimeError(
                            "MCP response never arrived (skipped "
                            f"{skipped} unrelated messages)")
                    logger.debug("MCP skipping non-matching message "
                                 "(id=%s)", resp.get("id"))
                    continue
                if "error" in resp:
                    raise RuntimeError(f"MCP error: {resp['error']}")
                return resp.get("result", {})

    def _send_notification(self, method: str, params: dict) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        notif = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        line = json.dumps(notif) + "\n"
        try:
            with self._io_lock:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
        except Exception:
            logger.debug("Failed to write to MCP server stdin",
                         exc_info=True)

    def start(self) -> None:
        """Launch the MCP server subprocess."""
        env = os.environ.copy()
        env.update(self.config.env)
        self._proc = subprocess.Popen(
            [self.config.command] + self.config.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # drain stderr in background to prevent pipe deadlock
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        # stdout pump — the ONLY reader of proc.stdout.
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout, daemon=True)
        self._stdout_thread.start()

    def _pump_stdout(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._rx.put(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("MCP non-JSON stdout [%s]: %s",
                                 self.config.name, line[:200])
        except Exception:
            logger.debug("MCP stdout pump failed", exc_info=True)
        finally:
            self._rx.put(_PROC_EOF)

    def _drain_stderr(self) -> None:
        if self._proc is None:
            return
        for line in self._proc.stderr:
            logger.debug("MCP stderr [%s]: %s", self.config.name, line.strip())

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    logger.debug("Failed to kill MCP server process",
                                 exc_info=True)
            self._proc = None
        self._connected = False
        super().close()


# ── HTTP transport ──────────────────────────────────────────────────────

class HTTPMCPClient(MCPClient):
    """MCP client over HTTP — sends JSON-RPC POST requests."""

    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._session = None

    def _get_session(self):
        if self._session is None:
            import urllib.request
            self._session = urllib.request
        return self._session

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            req["params"] = params
        data = json.dumps(req).encode("utf-8")
        url = self.config.url

        sess = self._get_session()
        request = sess.Request(
            url, data=data,
            headers={"Content-Type": "application/json"})
        with sess.urlopen(request, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        result = json.loads(body)
        if "error" in result:
            raise RuntimeError(f"MCP error: {result['error']}")
        return result.get("result", {})

    def start(self) -> None:
        """HTTP clients don't need to start a subprocess."""
        pass

    def close(self) -> None:
        self._connected = False
        super().close()


# ── MCP manager ─────────────────────────────────────────────────────────

class MCPManager:
    """Manages multiple MCP servers and aggregates their tools.

    Configuration is stored in ``data/mcp/servers.json``.
    Pure-stdlib (no Qt) so it can be unit-tested anywhere.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root else Path("data/mcp")
        self._path = self._root / "servers.json"
        self._servers: list[MCPServerConfig] = []
        self._clients: dict[str, MCPClient] = {}
        self.load()

    # ── persistence ───────────────────────────────────────────────────
    def load(self) -> None:
        if not self._path.is_file():
            self._servers = []
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._servers = [MCPServerConfig.from_dict(s)
                             for s in data.get("servers", [])]
        except Exception as e:
            logger.warning("MCP config load failed: %s", e)
            self._servers = []

    def save(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        data = {"version": 1,
                "servers": [s.to_dict() for s in self._servers]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    # ── server management ─────────────────────────────────────────────
    @property
    def servers(self) -> list[MCPServerConfig]:
        return self._servers

    def add_server(self, config: MCPServerConfig) -> None:
        # replace if same name exists
        self._servers = [s for s in self._servers if s.name != config.name]
        self._servers.append(config)
        self.save()

    def remove_server(self, name: str) -> bool:
        before = len(self._servers)
        self._servers = [s for s in self._servers if s.name != name]
        if len(self._servers) < before:
            self.save()
            return True
        return False

    def get_server(self, name: str) -> MCPServerConfig | None:
        for s in self._servers:
            if s.name == name:
                return s
        return None

    # ── connection management ─────────────────────────────────────────
    def connect(self, name: str) -> bool:
        """Connect to a specific MCP server."""
        config = self.get_server(name)
        if config is None or not config.enabled:
            return False
        try:
            if config.transport == "stdio":
                client = StdioMCPClient(config)
                client.start()
            elif config.transport == "http":
                client = HTTPMCPClient(config)
                client.start()
            else:
                logger.warning("unknown transport: %s", config.transport)
                return False
            client.initialize()
            client.list_tools()
            self._clients[name] = client
            logger.info("MCP server '%s' connected: %d tools",
                        name, len(client.tools))
            return True
        except Exception as e:
            # Kill the spawned subprocess — a failed handshake used to
            # orphan the server process until GC/finalizers got to it.
            try:
                client.close()
            except Exception:
                pass
            logger.warning("MCP connect '%s' failed: %s", name, e)
            return False

    def disconnect(self, name: str) -> None:
        client = self._clients.pop(name, None)
        if client is not None:
            client.close()

    def connect_all(self) -> dict[str, bool]:
        """Connect to all enabled servers. Returns name→success map."""
        results = {}
        for s in self._servers:
            if s.enabled and s.auto_start:
                results[s.name] = self.connect(s.name)
        return results

    def disconnect_all(self) -> None:
        for name in list(self._clients.keys()):
            self.disconnect(name)

    # ── tool aggregation ──────────────────────────────────────────────
    def all_tools(self) -> list[MCPTool]:
        """Get all tools from all connected servers."""
        tools = []
        for client in self._clients.values():
            tools.extend(client.tools)
        return tools

    def tool_defs(self) -> list[dict]:
        """Get OpenAI-style tool definitions from all connected servers."""
        return [t.to_openai_def() for t in self.all_tools()]

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool on the appropriate server."""
        for client in self._clients.values():
            for tool in client.tools:
                if tool.name == name:
                    return client.call_tool(name, arguments)
        return {"error": f"MCP tool '{name}' not found on any connected server"}

    # ── status ────────────────────────────────────────────────────────
    def status(self) -> list[dict]:
        """Get status of all servers."""
        out = []
        for s in self._servers:
            client = self._clients.get(s.name)
            out.append({
                "name": s.name,
                "transport": s.transport,
                "enabled": s.enabled,
                "connected": client is not None and client.connected,
                "tools": len(client.tools) if client else 0,
            })
        return out
