"""Tests for master prompt generator and chat features."""

from forge.config import get_config
from forge_gui.api.master_prompt import (
    _detect_model_name,
    _format_arch,
    _format_params,
    generate_master_prompt,
    get_default_prompt_for_config,
)


class TestMasterPrompt:
    def test_v2_prompt(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2")
        assert "ForgeLM V2" in prompt
        assert "ForgeAI" in prompt
        assert "TEXT-ONLY" in prompt
        assert "cannot see images" in prompt
        assert "coding" in prompt.lower()

    def test_v12_jamba_prompt(self):
        cfg = get_config("forgelm_v12_jamba")
        prompt = generate_master_prompt(cfg, "forgelm_v12_jamba")
        assert "ForgeLM V12 Jamba" in prompt
        assert "MULTIMODAL" not in prompt

    def test_tools_enabled_in_prompt(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2", tools_enabled=True)
        assert "tools" in prompt.lower()
        assert "remember" in prompt.lower()

    def test_tools_disabled_no_tool_section(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2", tools_enabled=False)
        assert "load_lora" not in prompt

    def test_thinking_enabled(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2", thinking_enabled=True)
        assert "<think>" in prompt.lower()
        assert "reasoning" in prompt.lower()

    def test_blunt_brief_style_contract(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2")
        assert "blunt and brief" in prompt.lower()
        assert "lead with the answer" in prompt.lower()

    def test_arch_in_prompt(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2")
        assert "layers" in prompt
        assert "d_model" in prompt

    def test_params_in_prompt(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2")
        # should have B or M suffix
        assert any(s in prompt for s in ("B ", "M "))

    def test_donor_model(self):
        cfg = get_config("forgelm_v2")
        prompt = generate_master_prompt(cfg, "forgelm_v2")
        assert "Jamba" in prompt


class TestDetectModelName:
    def test_v2(self):
        assert _detect_model_name("forgelm_v2") == "ForgeLM V2"

    def test_v12_jamba(self):
        assert _detect_model_name("forgelm_v12_jamba") == "ForgeLM V12 Jamba"

    def test_jamba(self):
        assert _detect_model_name("jamba_x") == "ForgeLM V2"

    def test_unknown(self):
        assert _detect_model_name("some_model") == "Some Model"


class TestFormatArch:
    def test_v2_arch(self):
        cfg = get_config("forgelm_v2")
        arch = _format_arch(cfg)
        assert "28 layers" in arch
        assert "d_model=2560" in arch
        assert "GQA" in arch


class TestFormatParams:
    def test_v2_params(self):
        cfg = get_config("forgelm_v2")
        params = _format_params(cfg)
        assert "B" in params or "M" in params


class TestGetDefaultPrompt:
    def test_v2_default(self):
        prompt = get_default_prompt_for_config("forgelm_v2")
        assert "ForgeLM V2" in prompt
        assert "TEXT-ONLY" in prompt

    def test_fallback_on_bad_config(self):
        prompt = get_default_prompt_for_config("nonexistent_config")
        assert "ForgeAI" in prompt
        assert "blunt" in prompt.lower()


class TestChatStoreImage:
    def test_append_message_with_image(self):
        from forge_gui.api.chat_store import ChatStore

        store = ChatStore()
        conv = store.create()
        idx = store.append_message(conv["id"], "user", "look at this", image="/tmp/test.png")
        msg = store.get(conv["id"])["messages"][idx]
        assert msg["image"] == "/tmp/test.png"

    def test_append_message_without_image(self):
        from forge_gui.api.chat_store import ChatStore

        store = ChatStore()
        conv = store.create()
        idx = store.append_message(conv["id"], "user", "hello")
        msg = store.get(conv["id"])["messages"][idx]
        assert "image" not in msg


class TestParseTranscript:
    """chat_store.parse_transcript — paste-import for the Chat page."""

    def test_empty(self):
        from forge_gui.api.chat_store import parse_transcript

        assert parse_transcript("") == []
        assert parse_transcript("   \n\n ") == []

    def test_json_list(self):
        from forge_gui.api.chat_store import parse_transcript

        msgs = parse_transcript('[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]')
        assert [(m["role"], m["content"]) for m in msgs] == [("user", "hi"), ("assistant", "hello")]

    def test_json_messages_wrapper_and_parts(self):
        from forge_gui.api.chat_store import parse_transcript

        msgs = parse_transcript('{"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}')
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_chatml(self):
        from forge_gui.api.chat_store import parse_transcript

        msgs = parse_transcript("<|im_start|>user\nhello there<|im_end|>\n<|im_start|>assistant\nhi!<|im_end|>\n")
        assert [(m["role"], m["content"]) for m in msgs] == [("user", "hello there"), ("assistant", "hi!")]

    def test_role_prefixed_inline(self):
        from forge_gui.api.chat_store import parse_transcript

        msgs = parse_transcript("User: what's 2+2\nAssistant: 4\nUser: thanks\nAssistant: np")
        assert [(m["role"], m["content"]) for m in msgs] == [
            ("user", "what's 2+2"),
            ("assistant", "4"),
            ("user", "thanks"),
            ("assistant", "np"),
        ]

    def test_marker_lines_multiline_and_header_junk(self):
        from forge_gui.api.chat_store import parse_transcript

        msgs = parse_transcript("Exported transcript\n\nChatGPT said:\nfirst line\nsecond line\n\nYou said:\nnext")
        assert [(m["role"], m["content"]) for m in msgs] == [("assistant", "first line\nsecond line"), ("user", "next")]

    def test_markdown_markers(self):
        from forge_gui.api.chat_store import parse_transcript

        msgs = parse_transcript("**User:**\nhi\n\n**Assistant:**\nhello")
        assert [(m["role"], m["content"]) for m in msgs] == [("user", "hi"), ("assistant", "hello")]

    def test_paragraph_fallback(self):
        from forge_gui.api.chat_store import parse_transcript

        msgs = parse_transcript("hi\n\nhello there\n\nhow are you")
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]

    def test_single_blob_is_one_user_msg(self):
        from forge_gui.api.chat_store import parse_transcript

        assert parse_transcript("just a single blob") == [{"role": "user", "content": "just a single blob"}]

    def test_prose_role_word_not_triggered(self):
        from forge_gui.api.chat_store import parse_transcript

        # "user experience" mid-line must not start a turn; a bare
        # marker-less blob stays a single user message
        msgs = parse_transcript("the user experience is good")
        assert msgs == [{"role": "user", "content": "the user experience is good"}]
