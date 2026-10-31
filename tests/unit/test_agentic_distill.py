"""Tests for the agentic distillation client."""
import json

from forge.distillation.agentic_distill import (
    AgenticDistillClient,
    AgenticTrajectory,
    _is_filler_task,
    _schemas_to_openai_tools,
)
from forge.distillation.distill_client import DistillModel

# ── Schema conversion tests ──────────────────────────────────────────────

class TestSchemaConversion:
    def test_basic_conversion(self):
        schemas = [
            {"name": "run_script", "description": "Run Python code",
             "parameters": {"type": "object",
                            "properties": {"code": {"type": "string"}}}},
        ]
        tools = _schemas_to_openai_tools(schemas)
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "run_script"
        assert tools[0]["function"]["description"] == "Run Python code"
        assert tools[0]["function"]["parameters"]["type"] == "object"

    def test_missing_type_defaults_to_object(self):
        schemas = [
            {"name": "think", "description": "Record thought",
             "parameters": {"properties": {"content": {"type": "string"}}}},
        ]
        tools = _schemas_to_openai_tools(schemas)
        assert tools[0]["function"]["parameters"]["type"] == "object"

    def test_missing_properties_defaults_to_empty(self):
        schemas = [
            {"name": "finish", "description": "End session", "parameters": {}},
        ]
        tools = _schemas_to_openai_tools(schemas)
        assert tools[0]["function"]["parameters"]["properties"] == {}

    def test_empty_schemas(self):
        assert _schemas_to_openai_tools([]) == []

    def test_multiple_schemas(self):
        schemas = [
            {"name": "run_script", "description": "Run code", "parameters": {}},
            {"name": "web_search", "description": "Search web", "parameters": {}},
            {"name": "think", "description": "Record thought", "parameters": {}},
        ]
        tools = _schemas_to_openai_tools(schemas)
        assert len(tools) == 3
        names = [t["function"]["name"] for t in tools]
        assert names == ["run_script", "web_search", "think"]


# ── Task filtering tests ─────────────────────────────────────────────────

class TestTaskFiltering:
    def test_filler_too_short(self):
        assert _is_filler_task("hello") is True
        assert _is_filler_task("test") is True

    def test_filler_generic(self):
        assert _is_filler_task("what is your name?") is True
        assert _is_filler_task("tell me a joke please") is True
        assert _is_filler_task("say hello world") is True

    def test_filler_json(self):
        assert _is_filler_task('{"name": "run_script", "arguments": {}}') is True

    def test_valid_coding_task(self):
        assert _is_filler_task(
            "Write a Python function that computes fibonacci numbers"
        ) is False
        assert _is_filler_task(
            "Implement a binary search algorithm and verify correctness"
        ) is False
        assert _is_filler_task(
            "Search for python asyncio tutorial and summarize"
        ) is False

    def test_no_tool_keyword(self):
        assert _is_filler_task(
            "The weather is nice today and I feel happy"
        ) is True


# ── Tool capability filtering tests ──────────────────────────────────────

class TestToolCapabilityFilter:
    def test_groq_supports_tools(self):
        groq_model = DistillModel("groq", "openai/gpt-oss-120b", "Apache-2.0",
                                   131072, 30, 1000, True,
                                   "https://api.groq.com/openai/v1",
                                   "GROQ_API_KEY", 0, 0, True, "gpt-oss-120b")
        assert AgenticDistillClient._supports_tools(groq_model) is True

    def test_deepseek_supports_tools(self):
        ds_model = DistillModel("deepseek", "deepseek-chat", "MIT",
                                65536, 60, 0, False,
                                "https://api.deepseek.com/v1",
                                "DEEPSEEK_API_KEY", 0, 0, True, "deepseek-v3")
        assert AgenticDistillClient._supports_tools(ds_model) is True

    def test_cloudflare_no_tools(self):
        cf_model = DistillModel("cloudflare", "@cf/openai/gpt-oss-120b",
                                "Apache-2.0", 131072, 50, 0, True,
                                "https://api.cloudflare.com/...", "CLOUDFLARE_API_TOKEN",
                                0, 0, True, "gpt-oss-120b")
        assert AgenticDistillClient._supports_tools(cf_model) is False

    def test_zai_no_tools(self):
        zai_model = DistillModel("zai", "glm-4.7-flash", "MIT",
                                 200000, 100, 0, False,
                                 "https://api.z.ai/api/paas/v4/",
                                 "ZAI_API_KEY", 0, 0, True, "glm-4.7")
        assert AgenticDistillClient._supports_tools(zai_model) is False

    def test_openrouter_gpt_oss_tools(self):
        or_model = DistillModel("openrouter", "openai/gpt-oss-120b:free",
                                "Apache-2.0", 131072, 20, 1000, True,
                                "https://openrouter.ai/api/v1",
                                "OPENROUTER_API_KEY", 0, 0, True, "gpt-oss-120b")
        assert AgenticDistillClient._supports_tools(or_model) is True

    def test_openrouter_glm_no_tools(self):
        or_model = DistillModel("openrouter", "z-ai/glm-5.2:free",
                                "MIT", 131072, 20, 1000, True,
                                "https://openrouter.ai/api/v1",
                                "OPENROUTER_API_KEY", 0, 0, True, "glm-5.2")
        assert AgenticDistillClient._supports_tools(or_model) is False


# ── Trajectory dataclass tests ───────────────────────────────────────────

class TestAgenticTrajectory:
    def test_to_sft_dict(self):
        traj = AgenticTrajectory(
            task="Write a function",
            teacher_model="groq/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "test"}],
            tool_calls=[{"name": "run_script", "args": {"code": "print(1)"},
                         "result": {"stdout": "1\n"}, "success": True}],
            final_answer="def f(): pass",
            reward=0.85,
            reward_breakdown={"format_ok": 1.0, "total": 0.85},
            n_turns=3,
            stopped_after_tools=True,
            stopped_after_answer=True,
            latency_ms=1500.0,
            tokens_in=100,
            tokens_out=200,
        )
        d = traj.to_sft_dict()
        assert d["task"] == "Write a function"
        assert d["teacher_model"] == "groq/openai/gpt-oss-120b"
        assert d["reward"] == 0.85
        assert d["n_turns"] == 3
        assert len(d["messages"]) == 1
        assert len(d["tool_calls"]) == 1
        assert d["final_answer"] == "def f(): pass"

    def test_empty_trajectory(self):
        traj = AgenticTrajectory(
            task="", teacher_model="", messages=[], tool_calls=[],
            final_answer=None, reward=0.0, reward_breakdown={},
            n_turns=0, stopped_after_tools=False, stopped_after_answer=False,
            latency_ms=0, tokens_in=0, tokens_out=0,
        )
        d = traj.to_sft_dict()
        assert d["task"] == ""
        assert d["reward"] == 0.0
        assert d["final_answer"] is None


# ── Client initialization tests ──────────────────────────────────────────

class TestClientInit:
    def test_init_filters_to_tool_capable(self, monkeypatch):
        # Set a fake API key so models are "available"
        monkeypatch.setenv("GROQ_API_KEY", "fake")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "fake")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "fake")
        client = AgenticDistillClient()
        # All models should be from tool-capable providers
        for m in client.models:
            assert AgenticDistillClient._supports_tools(m), \
                f"{m.provider}/{m.model_id} should not be in agentic pool"

    def test_init_no_api_keys_falls_back_but_filters(self, monkeypatch):
        # No API keys set — parent falls back to all models, but agentic
        # filter should still remove non-tool-capable providers
        for key in ["GROQ_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY",
                     "MISTRAL_API_KEY", "NVIDIA_API_KEY", "CEREBRAS_API_KEY",
                     "SAMBANOVA_API_KEY", "HF_API_KEY", "CLOUDFLARE_API_TOKEN",
                     "ZAI_API_KEY", "SILICONFLOW_API_KEY"]:
            monkeypatch.delenv(key, raising=False)
        client = AgenticDistillClient()
        # All models should still be tool-capable (no cloudflare, zai, siliconflow)
        for m in client.models:
            assert AgenticDistillClient._supports_tools(m), \
                f"{m.provider}/{m.model_id} should not be in agentic pool"
        # No cloudflare or zai models
        providers = set(m.provider for m in client.models)
        assert "cloudflare" not in providers
        assert "zai" not in providers
        assert "siliconflow" not in providers


# ── Save trajectories tests ──────────────────────────────────────────────

class TestSaveTrajectories:
    def test_save_to_jsonl(self, tmp_path):
        client = AgenticDistillClient.__new__(AgenticDistillClient)
        client._seen_outputs = set()
        trajectories = [
            AgenticTrajectory(
                task="Task 1", teacher_model="groq/test",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=[], final_answer="answer",
                reward=0.9, reward_breakdown={"total": 0.9},
                n_turns=2, stopped_after_tools=True,
                stopped_after_answer=True,
                latency_ms=100, tokens_in=10, tokens_out=20,
            ),
            AgenticTrajectory(
                task="Task 2", teacher_model="deepseek/test",
                messages=[{"role": "user", "content": "hi2"}],
                tool_calls=[{"name": "think", "args": {}, "result": {},
                             "success": True}],
                final_answer="answer2",
                reward=0.8, reward_breakdown={"total": 0.8},
                n_turns=3, stopped_after_tools=True,
                stopped_after_answer=True,
                latency_ms=200, tokens_in=20, tokens_out=30,
            ),
        ]
        out = tmp_path / "test_traj.jsonl"
        n = client.save_trajectories(trajectories, out)
        assert n == 2
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        d0 = json.loads(lines[0])
        assert d0["task"] == "Task 1"
        assert d0["reward"] == 0.9
        d1 = json.loads(lines[1])
        assert d1["task"] == "Task 2"
        assert d1["teacher_model"] == "deepseek/test"

    def test_save_appends_to_existing(self, tmp_path):
        client = AgenticDistillClient.__new__(AgenticDistillClient)
        client._seen_outputs = set()
        out = tmp_path / "append.jsonl"
        # Write first batch
        t1 = AgenticTrajectory(
            task="T1", teacher_model="m1", messages=[], tool_calls=[],
            final_answer="a", reward=0.5, reward_breakdown={},
            n_turns=1, stopped_after_tools=False, stopped_after_answer=True,
            latency_ms=0, tokens_in=0, tokens_out=0,
        )
        client.save_trajectories([t1], out)
        # Append second batch
        t2 = AgenticTrajectory(
            task="T2", teacher_model="m2", messages=[], tool_calls=[],
            final_answer="b", reward=0.6, reward_breakdown={},
            n_turns=1, stopped_after_tools=False, stopped_after_answer=True,
            latency_ms=0, tokens_in=0, tokens_out=0,
        )
        client.save_trajectories([t2], out)
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_save_empty_list(self, tmp_path):
        client = AgenticDistillClient.__new__(AgenticDistillClient)
        client._seen_outputs = set()
        out = tmp_path / "empty.jsonl"
        n = client.save_trajectories([], out)
        assert n == 0
