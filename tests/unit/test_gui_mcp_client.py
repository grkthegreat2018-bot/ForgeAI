"""Unit tests for forge_gui.api.mcp_client.

Tests MCPManager config persistence, MCPServerConfig, MCPTool conversion,
and mock client behavior. Pure stdlib — no Qt/torch.
"""

from forge_gui.api.mcp_client import (
    HTTPMCPClient,
    MCPManager,
    MCPServerConfig,
    MCPTool,
)

# ── MCPServerConfig ────────────────────────────────────────────────────


class TestMCPServerConfig:
    def test_stdio_config(self):
        c = MCPServerConfig.from_dict(
            {
                "name": "github",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "xxx"},
            }
        )
        assert c.name == "github"
        assert c.transport == "stdio"
        assert c.command == "npx"
        assert c.args == ["-y", "@modelcontextprotocol/server-github"]
        assert c.env["GITHUB_TOKEN"] == "xxx"

    def test_http_config(self):
        c = MCPServerConfig.from_dict(
            {
                "name": "linear",
                "transport": "http",
                "url": "http://localhost:3001/mcp",
            }
        )
        assert c.name == "linear"
        assert c.transport == "http"
        assert c.url == "http://localhost:3001/mcp"

    def test_defaults(self):
        c = MCPServerConfig.from_dict({"name": "test"})
        assert c.transport == "stdio"
        assert c.enabled is True
        assert c.auto_start is True

    def test_roundtrip(self):
        c = MCPServerConfig(name="test", transport="http", url="http://x")
        d = c.to_dict()
        c2 = MCPServerConfig.from_dict(d)
        assert c2.name == "test"
        assert c2.url == "http://x"


# ── MCPTool ────────────────────────────────────────────────────────────


class TestMCPTool:
    def test_to_openai_def(self):
        t = MCPTool(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            server_name="web",
        )
        d = t.to_openai_def()
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"
        assert d["function"]["description"] == "Search the web"
        assert "q" in d["function"]["parameters"]["properties"]

    def test_to_openai_def_empty_schema(self):
        t = MCPTool(name="noop", server_name="test")
        d = t.to_openai_def()
        assert d["function"]["parameters"]["type"] == "object"

    def test_to_openai_def_no_description(self):
        t = MCPTool(name="x", server_name="srv")
        d = t.to_openai_def()
        assert "srv" in d["function"]["description"]


# ── MCPManager ─────────────────────────────────────────────────────────


class TestMCPManager:
    def test_empty_manager(self, tmp_path):
        m = MCPManager(root=tmp_path)
        assert m.servers == []

    def test_add_remove_server(self, tmp_path):
        m = MCPManager(root=tmp_path)
        c = MCPServerConfig(name="test", transport="http", url="http://x")
        m.add_server(c)
        assert len(m.servers) == 1
        assert m.get_server("test") is not None

        # persisted
        m2 = MCPManager(root=tmp_path)
        assert len(m2.servers) == 1
        assert m2.servers[0].name == "test"

        # remove
        assert m.remove_server("test") is True
        assert m.get_server("test") is None
        assert m.remove_server("nonexistent") is False

    def test_replace_existing(self, tmp_path):
        m = MCPManager(root=tmp_path)
        m.add_server(MCPServerConfig(name="test", url="http://old"))
        m.add_server(MCPServerConfig(name="test", url="http://new"))
        assert len(m.servers) == 1
        assert m.servers[0].url == "http://new"

    def test_status(self, tmp_path):
        m = MCPManager(root=tmp_path)
        m.add_server(MCPServerConfig(name="test", transport="http", url="http://x"))
        status = m.status()
        assert len(status) == 1
        assert status[0]["name"] == "test"
        assert status[0]["connected"] is False
        assert status[0]["tools"] == 0

    def test_all_tools_empty(self, tmp_path):
        m = MCPManager(root=tmp_path)
        assert m.all_tools() == []
        assert m.tool_defs() == []

    def test_call_tool_not_found(self, tmp_path):
        m = MCPManager(root=tmp_path)
        result = m.call_tool("nonexistent", {})
        assert "error" in result


# ── MCPClient (mocked) ─────────────────────────────────────────────────


class TestMCPClientMocked:
    def test_initialize_and_list_tools(self):
        config = MCPServerConfig(name="test", transport="http", url="http://x")
        client = HTTPMCPClient(config)

        # mock the _send_request method
        responses = [
            {"serverInfo": {"name": "test-server", "version": "1.0"}},  # initialize
            {
                "tools": [
                    {"name": "search", "description": "Search", "inputSchema": {"type": "object"}},
                ]
            },  # list_tools
        ]
        call_idx = [0]

        def mock_send(method, params=None):
            resp = responses[min(call_idx[0], len(responses) - 1)]
            call_idx[0] += 1
            return resp

        client._send_request = mock_send
        result = client.initialize()
        assert "serverInfo" in result
        assert client.connected

        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "search"
        assert tools[0].server_name == "test"

    def test_call_tool_mocked(self):
        config = MCPServerConfig(name="test", transport="http", url="http://x")
        client = HTTPMCPClient(config)

        def mock_send(method, params=None):
            if method == "tools/call":
                return {"content": [{"type": "text", "text": "result"}]}
            return {}

        client._send_request = mock_send
        result = client.call_tool("search", {"q": "test"})
        assert "content" in result
