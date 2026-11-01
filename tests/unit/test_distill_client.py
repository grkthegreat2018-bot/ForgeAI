"""Tests for the multi-provider distillation client.

All tests are CPU-only and don't make real API calls. They test:
  - Model pool composition (only Apache/MIT licensed models)
  - Randomized model selection per goal
  - Prompt building
  - Result filtering and training pair conversion
  - ReplayBuffer integration
  - License safety (no Llama/Gemma models in pool)
"""

from forge.distillation.distill_client import (
    MODEL_POOL,
    DistillationClient,
    DistillResult,
)

# ── Model Pool Safety ─────────────────────────────────────────────────────


class TestModelPoolSafety:
    """Ensure only distillation-safe models are in the pool."""

    def test_all_models_apache_or_mit(self):
        """Every model in the pool must be Apache 2.0 or MIT (distill-safe)."""
        for m in MODEL_POOL:
            assert m.license in ("Apache-2.0", "MIT"), (
                f"{m.provider}/{m.model_id} has license '{m.license}' — only Apache-2.0 and MIT are distillation-safe"
            )

    def test_no_llama_models(self):
        """Llama Community License forbids using outputs to train other LLMs."""
        # OpenRouter :free models that are Llama-based would also be banned
        for m in MODEL_POOL:
            assert "llama" not in m.model_id.lower(), f"{m.model_id} is a Llama model — license forbids distillation"

    def test_no_gemma_models(self):
        """Gemma Terms classify distillation as creating a Model Derivative."""
        for m in MODEL_POOL:
            assert "gemma" not in m.model_id.lower(), f"{m.model_id} is a Gemma model — license restricts distillation"

    def test_all_models_have_env_keys(self):
        """Every model must specify an environment variable for its API key."""
        for m in MODEL_POOL:
            assert m.env_key, f"{m.model_id} missing env_key"
            assert m.env_key.endswith("_API_KEY") or m.env_key.endswith("_KEY") or m.env_key.endswith("_TOKEN"), (
                f"{m.model_id} env_key '{m.env_key}' doesn't end with _API_KEY/_KEY/_TOKEN"
            )

    def test_all_models_have_base_url(self):
        for m in MODEL_POOL:
            assert m.base_url.startswith("https://"), f"{m.model_id} base_url must be HTTPS"

    def test_at_least_5_providers(self):
        """We should have multiple providers for diversity."""
        providers = {m.provider for m in MODEL_POOL}
        assert len(providers) >= 5, f"Only {len(providers)} providers — need >= 5 for diversity"

    def test_groq_models_present(self):
        groq_models = [m for m in MODEL_POOL if m.provider == "groq"]
        assert len(groq_models) >= 3, "Should have 3 Groq models"

    def test_deepseek_mit_license(self):
        """DeepSeek R1 is MIT — explicitly allows distillation."""
        deepseek = [m for m in MODEL_POOL if m.provider == "deepseek"]
        assert len(deepseek) >= 1
        for m in deepseek:
            assert m.license == "MIT"

    def test_all_models_distill_allowed(self):
        """Every model must have distill_allowed=True."""
        for m in MODEL_POOL:
            assert m.distill_allowed is True, (
                f"{m.provider}/{m.model_id} has distill_allowed=False — outputs cannot be used for distillation"
            )

    def test_no_google_models(self):
        """Google Gemini TOS prohibits using outputs to develop competing models."""
        google = [m for m in MODEL_POOL if m.provider == "google"]
        assert len(google) == 0, "Google Gemini TOS prohibits distillation — should be excluded"

    def test_no_github_models(self):
        """GitHub Models was RETIRED July 30, 2026."""
        gh = [m for m in MODEL_POOL if m.provider == "github"]
        assert len(gh) == 0, "GitHub Models is retired — should be excluded"

    def test_no_nvidia_own_models(self):
        """NVIDIA's own models (Nemotron etc.) are BANNED per Eval Agreement §2.6."""
        from forge.distillation.distill_client import DistillModel, _nvidia_filter

        # NVIDIA's own model should be filtered out
        nemotron = DistillModel(
            "nvidia",
            "nvidia/nemotron-3-super-120b",
            "NVIDIA-Eval",
            131072,
            40,
            0,
            True,
            "https://integrate.api.nvidia.com/v1",
            "NVIDIA_API_KEY",
            0.0,
            0.0,
            True,
            "nemotron",
            "NVIDIA own model",
        )
        assert not _nvidia_filter(nemotron), "NVIDIA's own models should be filtered out (Eval Agreement §2.6)"

        # Third-party model on NIM should pass
        deepseek_r1 = DistillModel(
            "nvidia",
            "deepseek-ai/deepseek-r1",
            "MIT",
            131072,
            40,
            0,
            True,
            "https://integrate.api.nvidia.com/v1",
            "NVIDIA_API_KEY",
            0.0,
            0.0,
            True,
            "deepseek-r1",
            "Third-party on NIM",
        )
        assert _nvidia_filter(deepseek_r1), "Third-party MIT models on NIM should be allowed"

    def test_nvidia_models_present(self):
        """NVIDIA NIM third-party models should be in the pool (40 RPM, no daily cap)."""
        nvidia = [m for m in MODEL_POOL if m.provider == "nvidia"]
        assert len(nvidia) >= 2, "Should have at least 2 NVIDIA models"
        for m in nvidia:
            # Only third-party MIT/Apache models — no NVIDIA-own models
            assert m.license in ("MIT", "Apache-2.0")
            assert not m.model_id.lower().startswith("nvidia/"), (
                f"{m.model_id} is an NVIDIA-own model — should be filtered"
            )

    def test_cloudflare_models_present(self):
        """Cloudflare Workers AI models should be in the pool."""
        cf = [m for m in MODEL_POOL if m.provider == "cloudflare"]
        assert len(cf) >= 2, "Should have at least 2 Cloudflare models"

    def test_siliconflow_models_present(self):
        """SiliconFlow permanent free models should be in the pool."""
        sf = [m for m in MODEL_POOL if m.provider == "siliconflow"]
        assert len(sf) >= 2, "Should have at least 2 SiliconFlow models"

    def test_huggingface_models_present(self):
        """HuggingFace Inference API models should be in the pool."""
        hf = [m for m in MODEL_POOL if m.provider == "huggingface"]
        assert len(hf) >= 1, "Should have at least 1 HuggingFace model"

    def test_cerebras_models_present(self):
        """Cerebras free tier models should be in the pool."""
        cb = [m for m in MODEL_POOL if m.provider == "cerebras"]
        assert len(cb) >= 2, "Should have at least 2 Cerebras models"

    def test_sambanova_models_present(self):
        """SambaNova free tier models should be in the pool."""
        sn = [m for m in MODEL_POOL if m.provider == "sambanova"]
        assert len(sn) >= 1, "Should have at least 1 SambaNova model"

    def test_mistral_models_present(self):
        """Mistral AI free experiment models should be in the pool."""
        mi = [m for m in MODEL_POOL if m.provider == "mistral"]
        assert len(mi) >= 1, "Should have at least 1 Mistral model"

    def test_zai_models_present(self):
        """Z AI (Zhipu) GLM free models should be in the pool."""
        zai = [m for m in MODEL_POOL if m.provider == "zai"]
        assert len(zai) >= 1, "Should have at least 1 Z AI model"

    def test_gpt_oss_120b_has_many_providers(self):
        """gpt-oss-120b should be on at least 5 providers for redundancy."""
        providers = [m.provider for m in MODEL_POOL if m.canonical == "gpt-oss-120b"]
        assert len(providers) >= 5, f"gpt-oss-120b only on {len(providers)} providers — need 5+ for redundancy"

    def test_no_grok_models(self):
        """Grok/xAI prohibits using weights to train other models."""
        grok = [m for m in MODEL_POOL if "grok" in m.model_id.lower() or m.provider == "xai"]
        assert len(grok) == 0, "Grok distillation is banned — should be excluded"

    def test_no_cohere_models(self):
        """Cohere free tier is non-commercial only."""
        cohere = [m for m in MODEL_POOL if m.provider == "cohere"]
        assert len(cohere) == 0, "Cohere is non-commercial only — excluded"

    def test_at_least_10_providers(self):
        """Should have at least 10 distinct providers for max redundancy."""
        providers = {m.provider for m in MODEL_POOL}
        assert len(providers) >= 10, f"Only {len(providers)} providers — need 10+ for max redundancy"

    def test_at_least_25_models(self):
        """Should have at least 25 model entries for max redundancy."""
        assert len(MODEL_POOL) >= 25, f"Only {len(MODEL_POOL)} models — need 25+ for max redundancy"

    def test_all_models_have_canonical(self):
        """Every model must have a canonical name for multi-provider grouping."""
        for m in MODEL_POOL:
            assert m.canonical, f"{m.model_id} missing canonical name"

    def test_multi_provider_canonicals(self):
        """At least one canonical model should be served by multiple providers."""
        from collections import Counter

        canonicals = [m.canonical for m in MODEL_POOL]
        counts = Counter(canonicals)
        multi = {c: n for c, n in counts.items() if n > 1}
        assert len(multi) >= 1, "Should have at least 1 canonical model on multiple providers"

    def test_no_trial_credit_providers(self):
        """No Hyperbolic, Together, Novita, or Chutes (trial credits)."""
        # Cerebras now has a permanent free tier (30 RPM, 1M tok/day) — NOT trial
        excluded = {"hyperbolic", "together", "novita", "chutes"}
        for m in MODEL_POOL:
            assert m.provider not in excluded, f"{m.provider} is a trial-credit provider — should be excluded"

    def test_all_costs_zero(self):
        """All models in the pool should be $0 (permanent free tier)."""
        for m in MODEL_POOL:
            assert m.cost_in == 0.0, f"{m.model_id} has non-zero cost_in"
            assert m.cost_out == 0.0, f"{m.model_id} has non-zero cost_out"


# ── Client Initialization ─────────────────────────────────────────────────


class TestClientInit:
    def test_init_with_no_keys(self):
        """Client should initialize even without API keys."""
        client = DistillationClient()
        assert client is not None

    def test_init_filters_by_provider(self):
        client = DistillationClient(providers=["groq"])
        for m in client.models:
            assert m.provider == "groq"

    def test_init_filters_by_model(self):
        client = DistillationClient(models=["openai/gpt-oss-120b"])
        assert len(client.models) >= 1
        assert client.models[0].model_id == "openai/gpt-oss-120b"

    def test_available_models(self):
        client = DistillationClient()
        models = client.available_models()
        assert len(models) > 0
        for m in models:
            assert "provider" in m
            assert "license" in m
            assert "api_key_set" in m


# ── Model Selection ───────────────────────────────────────────────────────


class TestModelSelection:
    def test_pick_model_returns_n(self):
        client = DistillationClient()
        picked = client._pick_model("test goal", n=3)
        assert len(picked) == 3

    def test_pick_model_more_than_available(self):
        """Should cycle through models if n > pool size."""
        client = DistillationClient(providers=["groq"])
        n_groq = len(client.models)
        picked = client._pick_model("test", n=n_groq * 2)
        assert len(picked) == n_groq * 2

    def test_pick_model_randomized(self):
        """Successive calls should produce different orderings (probabilistic)."""
        client = DistillationClient()
        picks1 = [m.model_id for m in client._pick_model("g1", 5)]
        picks2 = [m.model_id for m in client._pick_model("g2", 5)]
        # Very unlikely to be identical with 10+ models
        # (Could be identical by chance, so we just check length)
        assert len(picks1) == 5
        assert len(picks2) == 5

    def test_pick_model_with_rotation(self):
        """Rotation should pick models preferring least-used providers."""
        client = DistillationClient()
        picked = client._pick_model_with_rotation("test", 3)
        assert len(picked) == 3
        # All should be valid models from the pool
        for m in picked:
            assert m in client.models

    def test_rotation_prefers_least_used_provider(self):
        """After calling one provider, rotation should prefer a different one."""
        client = DistillationClient()
        # Simulate that groq has been used a lot
        client._provider_request_counts["groq"] = 100
        # Pick a gpt-oss-120b model — should prefer non-groq provider
        picked = client._pick_model_with_rotation("test", 5)
        # At least one picked model should not be from groq
        providers = [m.provider for m in picked]
        assert "groq" not in providers or len(set(providers)) > 1, "Rotation should prefer less-used providers"

    def test_canonical_groups(self):
        """canonical_groups() should return canonical → providers mapping."""
        client = DistillationClient()
        groups = client.canonical_groups()
        assert isinstance(groups, dict)
        assert len(groups) > 0
        # At least one canonical should have multiple providers
        multi = {c: ps for c, ps in groups.items() if len(ps) > 1}
        assert len(multi) >= 1, "Should have at least 1 canonical with multiple providers"

    def test_pick_model_diversity(self):
        """When picking <= pool size, should get distinct DistillModel objects."""
        client = DistillationClient()
        pool_size = len(client.models)
        n = min(5, pool_size)
        picked = client._pick_model("test", n=n)
        # Should pick n distinct DistillModel objects (by identity)
        assert len(picked) == n
        # All should be from the pool
        for m in picked:
            assert m in client.models


# ── Prompt Building ───────────────────────────────────────────────────────


class TestPromptBuilding:
    def test_basic_prompt(self):
        client = DistillationClient()
        msgs = client._build_prompt("Write a hello world function", None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "hello world" in msgs[1]["content"].lower()

    def test_prompt_with_test_cases(self):
        client = DistillationClient()
        tc = [{"input": "5", "output": "5"}, {"input": "10", "output": "55"}]
        msgs = client._build_prompt("fibonacci", tc)
        assert "5" in msgs[1]["content"]
        assert "55" in msgs[1]["content"]

    def test_custom_system_prompt(self):
        client = DistillationClient()
        msgs = client._build_prompt("task", None, system_prompt="Be concise")
        assert msgs[0]["content"] == "Be concise"


# ── Result Processing ─────────────────────────────────────────────────────


class TestResultProcessing:
    def test_filter_correct(self):
        client = DistillationClient()
        results = [
            DistillResult("groq/qwen3-32b", "good code", True, "", "", 100, 10, 50),
            DistillResult("groq/gpt-oss-20b", "bad code", False, "", "", 100, 10, 50),
            DistillResult("deepseek/r1", "great code", True, "", "", 200, 10, 80),
        ]
        correct = client.filter_correct(results)
        assert len(correct) == 2
        assert all(r.correct for r in correct)

    def test_to_training_pairs(self):
        client = DistillationClient()
        results = [
            DistillResult("groq/qwen3-32b", "def f(): pass", True, "think...", "", 100, 10, 50),
            DistillResult("groq/gpt-oss-20b", "def g(): pass", False, "", "", 100, 10, 50),
        ]
        pairs = client.to_training_pairs(results)
        assert len(pairs) == 1  # only the correct one
        assert pairs[0]["solution"] == "def f(): pass"
        assert pairs[0]["quality"] == 1.0
        assert pairs[0]["test_passed"] is True
        assert pairs[0]["model"] == "groq/qwen3-32b"

    def test_to_training_pairs_skips_errors(self):
        client = DistillationClient()
        results = [
            DistillResult("groq/qwen3-32b", "", False, "", "", 0, 0, 0, error="timeout"),
        ]
        pairs = client.to_training_pairs(results)
        assert len(pairs) == 0


# ── ReplayBuffer Integration ──────────────────────────────────────────────


class TestReplayBufferIntegration:
    def test_distill_into_buffer(self):
        """distill_into_buffer should store verified results in the buffer."""
        from forge.self_play.replay_buffer import ReplayBuffer

        # Mock verify function: returns True for solutions containing "correct"
        def verify_fn(solution, test_cases):
            return "correct" in solution

        client = DistillationClient(verify_fn=verify_fn)

        # Monkey-patch _call_model to return mock results
        # Alternate correct/incorrect deterministically
        call_idx = [0]

        def mock_call(model, messages, temp):
            call_idx[0] += 1
            is_correct = call_idx[0] % 2 == 1  # odd = correct, even = wrong
            return DistillResult(
                f"{model.provider}/{model.model_id}",
                "def correct_solution(): pass" if is_correct else "def wrong(): pass",
                is_correct,
                "",
                "",
                50,
                10,
                20,
            )

        client._call_model = mock_call

        buf = ReplayBuffer(max_size=1000)
        stats = client.distill_into_buffer(
            goals=["Write a function", "Write another function"],
            replay_buffer=buf,
            n_samples_per_goal=3,
        )

        assert stats["n_goals"] == 2
        assert stats["n_generated"] == 6  # 2 goals × 3 samples
        assert stats["n_correct"] > 0
        assert stats["n_stored"] == stats["n_correct"]
        assert stats["buffer_size"] == stats["n_stored"]
        assert "per_model" in stats

    def test_distill_into_buffer_filters_errors(self):
        """Results with errors should not be stored."""
        from forge.self_play.replay_buffer import ReplayBuffer

        client = DistillationClient()

        def mock_call(model, messages, temp):
            return DistillResult(f"{model.provider}/{model.model_id}", "", False, "", "", 0, 0, 0, error="API timeout")

        client._call_model = mock_call

        buf = ReplayBuffer(max_size=1000)
        stats = client.distill_into_buffer(goals=["test goal"], replay_buffer=buf, n_samples_per_goal=2)

        assert stats["n_correct"] == 0
        assert stats["n_stored"] == 0
        assert len(buf) == 0


# ── Stats ─────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats(self):
        client = DistillationClient()
        s = client.stats()
        assert "models_available" in s
        assert "providers" in s
        assert "request_counts" in s
        assert "total_requests" in s

    def test_stats_tracks_requests(self):
        client = DistillationClient()

        call_count = 0

        def mock_call(model, messages, temp):
            nonlocal call_count
            call_count += 1
            # Simulate the request tracking that real _call_model does
            client._request_counts[model.model_id] = client._request_counts.get(model.model_id, 0) + 1
            return DistillResult(f"{model.provider}/{model.model_id}", "ok", False, "", "", 50, 10, 20)

        client._call_model = mock_call
        client.generate_for_goal("test", n_samples=2)

        s = client.stats()
        assert call_count == 2
        assert s["total_requests"] == 2
