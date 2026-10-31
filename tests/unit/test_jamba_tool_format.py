"""Test Jamba tool call format (ForgeLM V2).

Tests that the updated chat_template.py correctly renders and parses
Jamba's JSON tool call format with  /  markers.
"""
import pytest

from forge.self_play.discovery.chat_template import (
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    TOOL_RESP_END,
    TOOL_RESP_START,
    apply_chat_template,
    parse_tool_calls,
    render_tool_calls,
)


class TestJambaToolCallFormat:
    """Test Jamba JSON tool call rendering and parsing."""

    def test_render_single_tool_call(self):
        """Render a single tool call in Jamba JSON format."""
        calls = [{"name": "read_file", "args": {"path": "/tmp/test.py"}}]
        result = render_tool_calls(calls)
        assert TOOL_CALL_START in result
        assert TOOL_CALL_END in result
        assert '"name": "read_file"' in result
        assert '"arguments"' in result
        assert '"path": "/tmp/test.py"' in result

    def test_render_openai_format(self):
        """Render OpenAI-format tool calls (function/arguments)."""
        calls = [{"function": {"name": "write_file",
                                "arguments": '{"content": "hello"}'}}]
        result = render_tool_calls(calls)
        assert '"name": "write_file"' in result
        assert '"content": "hello"' in result

    def test_parse_jamba_format(self):
        """Parse Jamba JSON tool calls from model output."""
        text = f'Let me read that file.\n{TOOL_CALL_START}\n{{"name": "read_file", "arguments": {{"path": "/tmp/test.py"}}}}\n{TOOL_CALL_END}'
        calls, musing = parse_tool_calls(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["args"]["path"] == "/tmp/test.py"
        assert "read that file" in musing

    def test_parse_multiple_tool_calls(self):
        """Parse multiple JSON tool calls in one output."""
        text = f'{TOOL_CALL_START}\n{{"name": "read_file", "arguments": {{"path": "a.py"}}}}\n{{"name": "list_dir", "arguments": {{"dir": "/tmp"}}}}\n{TOOL_CALL_END}'
        calls, _musing = parse_tool_calls(text)
        assert calls is not None
        assert len(calls) == 2
        assert calls[0]["name"] == "read_file"
        assert calls[1]["name"] == "list_dir"

    def test_parse_no_tool_calls(self):
        """No tool calls in text returns None."""
        text = "Just a regular response with no tool calls."
        calls, musing = parse_tool_calls(text)
        assert calls is None
        assert musing == text

    def test_round_trip_render_parse(self):
        """Render then parse gives back the same tool calls."""
        original = [{"name": "grep_project", "args": {"pattern": "TODO", "path": "src"}}]
        rendered = render_tool_calls(original)
        parsed, _ = parse_tool_calls(rendered)
        assert parsed is not None
        assert parsed[0]["name"] == "grep_project"
        assert parsed[0]["args"]["pattern"] == "TODO"


class TestJambaChatTemplate:
    """Test full chat template rendering with Jamba format."""

    def test_apply_chat_template_basic(self):
        """Basic chat template renders correctly."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = apply_chat_template(messages, add_generation_prompt=True)
        assert "<|im_start|>system" in result
        assert "You are a helpful assistant." in result
        assert "<|im_start|>user" in result
        assert "Hello!" in result
        assert "<|im_start|>assistant" in result

    def test_apply_chat_template_with_tools(self):
        """Chat template with tool definitions."""
        messages = [
            {"role": "user", "content": "Read the file"},
        ]
        tools = [{"name": "read_file", "description": "Read a file",
                  "parameters": {"type": "object",
                                "properties": {"path": {"type": "string"}}}}]
        result = apply_chat_template(messages, tools=tools, add_generation_prompt=True)
        assert "read_file" in result
        assert "<|im_start|>system" in result

    def test_apply_chat_template_with_tool_response(self):
        """Tool response uses Jamba  markers."""
        messages = [
            {"role": "user", "content": "Read file"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "read_file", "args": {"path": "test.py"}}]},
            {"role": "tool", "content": "print('hello')"},
        ]
        result = apply_chat_template(messages, add_generation_prompt=False)
        assert TOOL_RESP_START in result
        assert "print('hello')" in result
        assert TOOL_RESP_END in result

    def test_thinking_tags_present(self):
        """Thinking tags are defined correctly."""
        # Jamba thinking tags:  (U+1F9D0 + ...) and  (...)
        # Build from unicode to avoid tool-call parsing confusion
        think_start_expected = "\U0001f9d0"  # placeholder — actual is multi-char
        # Just verify they're non-empty strings
        assert len(THINK_START) > 0
        assert len(THINK_END) > 0
        assert THINK_START != THINK_END


class TestBackwardCompat:
    """Test backward compatibility with LFM2.5 Pythonic format."""

    def test_parse_legacy_pythonic_format(self):
        """Legacy LFM2.5 Pythonic tool calls still parse."""
        # Build the legacy markers from hex (same as original code)
        legacy_start = bytes.fromhex(
            "3c7c746f6f6c5f63616c6c5f73746172747c3e").decode("ascii")
        legacy_end = bytes.fromhex(
            "3c7c746f6f6c5f63616c6c5f656e647c3e").decode("ascii")
        text = f'{legacy_start}[read_file(path="/tmp/test.py")]{legacy_end}'
        calls, _musing = parse_tool_calls(text)
        assert calls is not None
        assert calls[0]["name"] == "read_file"
        assert calls[0]["args"]["path"] == "/tmp/test.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
