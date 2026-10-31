"""Multi-provider distillation client for generating verified training data.

Supports 8 free-tier API providers. Every model in the pool has
distill_allowed=True — outputs CAN be used to train/distill other models.
Randomizes model selection per goal to maximize quality diversity.
All providers are OpenAI-compatible (single SDK).

## Multi-Provider Rate-Limit Bypass

The same model (e.g. gpt-oss-120b) is served by multiple providers (Groq,
NVIDIA, Cloudflare, HuggingFace). The client groups models by `canonical`
name and rotates through providers to bypass per-provider rate limits.
When one provider's RPM/RPD is hit, the next call to the same canonical
model automatically uses a different provider.

## Supported Providers (all have free tiers as of 2026-08)

### Groq (permanent free, no expiry)
- qwen/qwen3-32b       — Apache 2.0, 60 RPM, 1000 RPD, reasoning
- qwen/qwen3.6-27b     — Apache 2.0, 60 RPM, 1000 RPD, reasoning
- openai/gpt-oss-120b  — Apache 2.0, 30 RPM, 1000 RPD
- openai/gpt-oss-20b   — Apache 2.0, 30 RPM, 1000 RPD

### NVIDIA NIM (free trial, 40 RPM, no daily token cap)
- deepseek-ai/deepseek-r1     — MIT, R1 reasoning
- deepseek-ai/deepseek-v3     — MIT, V3 chat
- qwen/qwen3.5-122b-a10b      — Apache 2.0, 122B MoE, 262K context
- openai/gpt-oss-120b         — Apache 2.0, 120B reasoning
- (NVIDIA's own models EXCLUDED — Eval Agreement §2.6 prohibits distillation)

### Cloudflare Workers AI (free, 10K neurons/day)
- @cf/openai/gpt-oss-120b     — Apache 2.0
- @cf/openai/gpt-oss-20b      — Apache 2.0
- @cf/zai-org/glm-4.7-flash   — MIT, GLM-4.7, reasoning

### SiliconFlow (permanent free models, no credit card)
- Qwen/Qwen3-8B               — Apache 2.0, 1000 RPM, 262K context, reasoning
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B — MIT, R1 distilled, reasoning

### HuggingFace Inference Providers (free, $0.10/mo credits)
- openai/gpt-oss-120b         — Apache 2.0
- openai/gpt-oss-20b          — Apache 2.0
- deepseek-ai/DeepSeek-R1     — MIT, R1 reasoning

### OpenRouter (free tier, 50 RPD without credits)
- qwen/qwen3-235b-a22b:free   — Apache 2.0, 235B MoE (largest free model)
- openai/gpt-oss-20b:free     — Apache 2.0
- deepseek/deepseek-r1:free   — MIT, R1 reasoning
- z-ai/glm-5.2:free           — MIT, GLM-5.2, reasoning

### Z AI / Zhipu (free, 1 concurrent request)
- glm-4.7-flash               — MIT, reasoning (thinking.type)

### Mistral AI (free experiment plan, ~1B tokens/month)
- mistral-small-latest        — Apache 2.0
- magistral-small-latest      — Apache 2.0, reasoning
- zai-glm-5-2                 — MIT, reasoning

## REMOVED providers (no longer free as of 2026-08)
- Cerebras    — requires $5 credit + payment method, expires 30 days
- SambaNova   — $5 starter credit expires 3 months, then pay-as-you-go
- DeepSeek direct API — one-time 5M token grant, then pay-as-you-go

## License Safety (every model verified distill_allowed=True)

- ✅ Apache-2.0  — "prepare Derivative Works" explicitly allowed
- ✅ MIT         — "distill & commercialize freely" (DeepSeek, Phi-4)
- ❌ Llama       — "will not use output to improve any other LLM" (BANNED)
- ❌ Gemma-TOS   — distilled model = "Model Derivative" (BANNED)
- ❌ Gemini-TOS  — "may not use Services to develop models that compete" (BANNED)
- ❌ NVIDIA-own  — Evaluation Agreement §2.6 prohibits (third-party on NIM = OK)
- ❌ OpenAI GPT  — "may not use Output to develop models that compete" (BANNED)
- ❌ Anthropic   — "may not use Outputs to train models that compete" (BANNED)

## BANNED providers (removed, do not re-add)
- Google AI Studio  — TOS prohibits competing model development
- GitHub Models     — RETIRED July 30, 2026
- Cerebras/Hyperbolic/Together/Novita/Chutes — trial credits, not permanent

## Usage

    from forge.distillation.distill_client import DistillationClient

    client = DistillationClient()  # auto-detects available API keys
    data = client.generate_for_goal(
        goal="Write a function to compute fibonacci numbers",
        test_cases=[{"input": "5", "output": "5"},
                    {"input": "10", "output": "55"}],
        n_samples=4,  # 4 diverse completions per goal
    )
    # data = [{"model": "qwen3-32b", "solution": "...", "correct": True}, ...]
"""
from __future__ import annotations

import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# OpenAI SDK is compatible with Groq, DeepSeek, OpenRouter, NVIDIA, Cerebras,
# SambaNova, Mistral, Z AI, Cloudflare, SiliconFlow, HuggingFace
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ─── Model Pool ─────────────────────────────────────────────────────────

@dataclass
class DistillModel:
    """A single model available for distillation."""
    provider: str          # "groq", "deepseek", "nvidia", etc.
    model_id: str          # API model string
    license: str           # "Apache-2.0", "MIT", etc.
    context: int           # max context window
    rpm: int               # requests per minute (free tier)
    rpd: int               # requests per day (free tier, 0 = unlimited)
    reasoning: bool        # supports reasoning_effort / CoT
    base_url: str          # OpenAI-compatible base URL
    env_key: str           # environment variable for API key
    cost_in: float         # $/1M input tokens (0 = free)
    cost_out: float        # $/1M output tokens (0 = free)
    distill_allowed: bool = True  # explicit: can outputs be used for distillation?
    canonical: str = ""    # canonical model name for multi-provider grouping
    notes: str = ""        # extra info


# All distillation-safe models from PERMANENT free-tier providers only.
# No trial credits, no time-limited promos. All OpenAI-compatible API.
# Every model has distill_allowed=True — outputs CAN be used to train other models.
#
# `canonical` field groups the same model across providers — the client rotates
# through providers serving the same canonical model to bypass per-provider rate
# limits. Maximum redundancy: gpt-oss-120b is on 7 providers, so if 6 burn out,
# the 7th still works.
#
# License safety:
#   ✅ Apache-2.0  — "prepare Derivative Works" explicitly allowed
#   ✅ MIT         — "distill & commercialize freely" (DeepSeek, GLM, Phi-4)
#   ❌ Llama       — "will not use output to improve any other LLM" (BANNED)
#   ❌ Gemma-TOS   — distilled model = "Model Derivative" (BANNED)
#   ❌ Gemini-TOS  — "may not use Services to develop models that compete" (BANNED)
#   ❌ NVIDIA-own  — Evaluation Agreement §2.6 prohibits (third-party on NIM = OK)
#   ❌ OpenAI GPT  — "may not use Output to develop models that compete" (BANNED)
#   ❌ Anthropic   — "may not use Outputs to train models that compete" (BANNED)
#   ❌ Grok/xAI    — "weights cannot be used to train other models" (BANNED)
#   ❌ Cohere      — non-commercial use only (BANNED for commercial distillation)
#
# BANNED providers (removed, do not re-add):
#   - Google AI Studio  — TOS prohibits competing model development
#   - GitHub Models     — RETIRED July 30, 2026
#   - Together/Hyperbolic/Novita/Chutes — trial credits, not permanent
#
# NVIDIA NIM filter: only third-party models (DeepSeek/Qwen/gpt-oss) allowed.
# NVIDIA's own models (Nemotron, etc.) are EXCLUDED per Eval Agreement §2.6.
_MODEL_POOL_RAW: list[DistillModel] = [
    # ══ gpt-oss-120b (Apache-2.0) — 4 free providers ══
    DistillModel("groq", "openai/gpt-oss-120b", "Apache-2.0", 131072,
                 30, 1000, True,
                 "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                 0.0, 0.0, True, "gpt-oss-120b", "120B reasoning, 131K ctx"),
    DistillModel("nvidia", "openai/gpt-oss-120b", "Apache-2.0", 131072,
                 40, 0, True,
                 "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
                 0.0, 0.0, True, "gpt-oss-120b", "40 RPM, no daily cap"),
    DistillModel("cloudflare", "@cf/openai/gpt-oss-120b", "Apache-2.0", 131072,
                 50, 0, True,
                 "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
                 "CLOUDFLARE_API_TOKEN",
                 0.0, 0.0, True, "gpt-oss-120b", "10K neurons/day free"),
    DistillModel("huggingface", "openai/gpt-oss-120b", "Apache-2.0", 131072,
                 10, 0, True,
                 "https://api-inference.huggingface.co/v1", "HF_API_KEY",
                 0.0, 0.0, True, "gpt-oss-120b", "HF Inference, $0.10/mo free"),

    # ══ gpt-oss-20b (Apache-2.0) — 4 free providers ══
    DistillModel("groq", "openai/gpt-oss-20b", "Apache-2.0", 131072,
                 30, 1000, True,
                 "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                 0.0, 0.0, True, "gpt-oss-20b", "Fastest Groq (1000 t/s)"),
    DistillModel("cloudflare", "@cf/openai/gpt-oss-20b", "Apache-2.0", 131072,
                 50, 0, True,
                 "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
                 "CLOUDFLARE_API_TOKEN",
                 0.0, 0.0, True, "gpt-oss-20b", "10K neurons/day free"),
    DistillModel("openrouter", "openai/gpt-oss-20b:free", "Apache-2.0", 131072,
                 20, 50, True,
                 "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                 0.0, 0.0, True, "gpt-oss-20b", "Free via OpenRouter, 50 RPD"),
    DistillModel("huggingface", "openai/gpt-oss-20b", "Apache-2.0", 131072,
                 10, 0, True,
                 "https://api-inference.huggingface.co/v1", "HF_API_KEY",
                 0.0, 0.0, True, "gpt-oss-20b", "HF Inference Providers"),

    # ══ DeepSeek R1 (MIT) — 3 free providers ══
    # NOTE: DeepSeek direct API no longer free (trial credits only).
    # Use NVIDIA NIM, HuggingFace, SiliconFlow, OpenRouter instead.
    DistillModel("nvidia", "deepseek-ai/deepseek-r1", "MIT", 131072,
                 40, 0, True,
                 "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
                 0.0, 0.0, True, "deepseek-r1", "R1 on NVIDIA, no daily cap"),
    DistillModel("huggingface", "deepseek-ai/DeepSeek-R1", "MIT", 131072,
                 10, 0, True,
                 "https://api-inference.huggingface.co/v1", "HF_API_KEY",
                 0.0, 0.0, True, "deepseek-r1", "R1 via HF Inference"),
    DistillModel("siliconflow", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "MIT", 65536,
                 1000, 0, True,
                 "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY",
                 0.0, 0.0, True, "deepseek-r1-distill-7b", "R1 distill 7B, free"),
    DistillModel("openrouter", "deepseek/deepseek-r1:free", "MIT", 131072,
                 20, 50, True,
                 "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                 0.0, 0.0, True, "deepseek-r1", "R1 free via OpenRouter, 50 RPD"),

    # ══ DeepSeek V3 (MIT) — 2 free providers ══
    # NOTE: DeepSeek direct API and SambaNova no longer free.
    DistillModel("nvidia", "deepseek-ai/deepseek-v3", "MIT", 131072,
                 40, 0, False,
                 "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
                 0.0, 0.0, True, "deepseek-v3", "V3 on NVIDIA, no daily cap"),

    # ══ Qwen3-32B (Apache-2.0) — Groq, reasoning enabled ══
    DistillModel("groq", "qwen/qwen3-32b", "Apache-2.0", 131072,
                 60, 1000, True,
                 "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                 0.0, 0.0, True, "qwen3-32b", "60 RPM, enable_thinking, reasoning"),

    # ══ Qwen3.6-27B (Apache-2.0) — Groq only ══
    DistillModel("groq", "qwen/qwen3.6-27b", "Apache-2.0", 131072,
                 60, 1000, True,
                 "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                 0.0, 0.0, True, "qwen3.6-27b", "60 RPM, thinking mode"),

    # ══ Qwen3.5-122B MoE (Apache-2.0) — NVIDIA only ══
    DistillModel("nvidia", "qwen/qwen3.5-122b-a10b", "Apache-2.0", 262144,
                 40, 0, True,
                 "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
                 0.0, 0.0, True, "qwen3.5-122b", "122B MoE, 262K ctx"),

    # ══ Qwen3-235B MoE (Apache-2.0) — OpenRouter free ══
    DistillModel("openrouter", "qwen/qwen3-235b-a22b:free", "Apache-2.0", 131072,
                 20, 50, True,
                 "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                 0.0, 0.0, True, "qwen3-235b", "235B MoE, largest free model, 50 RPD"),

    # ══ Qwen3-8B (Apache-2.0) — SiliconFlow only ══
    DistillModel("siliconflow", "Qwen/Qwen3-8B", "Apache-2.0", 262144,
                 1000, 0, True,
                 "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY",
                 0.0, 0.0, True, "qwen3-8b", "Free forever, 1000 RPM, 262K"),

    # ══ GLM-4.7 (MIT) — Z AI + Cloudflare (both free) ══
    DistillModel("zai", "glm-4.7-flash", "MIT", 200000,
                 100, 0, True,
                 "https://api.z.ai/api/paas/v4/", "ZAI_API_KEY",
                 0.0, 0.0, True, "glm-4.7", "Free, thinking.type support, 1 concurrent"),
    DistillModel("cloudflare", "@cf/zai-org/glm-4.7-flash", "MIT", 131072,
                 50, 0, True,
                 "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
                 "CLOUDFLARE_API_TOKEN",
                 0.0, 0.0, True, "glm-4.7", "GLM-4.7 Flash on Workers AI, thinking"),

    # ══ Mistral Small (Apache-2.0) — Mistral AI only ══
    DistillModel("mistral", "mistral-small-latest", "Apache-2.0", 128000,
                 2, 0, False,
                 "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                 0.0, 0.0, True, "mistral-small", "Free experiment plan, ~1B tok/mo"),

    # ══ Magistral Small (Apache-2.0, reasoning) — Mistral AI only ══
    DistillModel("mistral", "magistral-small-latest", "Apache-2.0", 128000,
                 2, 0, True,
                 "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                 0.0, 0.0, True, "magistral-small", "Reasoning model, free experiment"),

    # ══ GLM-5.2 (MIT) — OpenRouter + Mistral ══
    DistillModel("openrouter", "z-ai/glm-5.2:free", "MIT", 131072,
                 20, 50, True,
                 "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                 0.0, 0.0, True, "glm-5.2", "GLM-5.2 free via OpenRouter, 50 RPD"),
    DistillModel("mistral", "zai-glm-5-2", "MIT", 131072,
                 2, 0, True,
                 "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                 0.0, 0.0, True, "glm-5.2", "GLM-5.2 via Mistral"),

    # ══ DeepSeek direct (MIT) — DeepSeek API ══
    DistillModel("deepseek", "deepseek-chat", "MIT", 65536,
                 60, 0, False,
                 "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY",
                 0.0, 0.0, True, "deepseek-v3", "DeepSeek V3 direct API"),

    # ══ Cerebras (Apache-2.0) — free tier, fast inference ══
    DistillModel("cerebras", "openai/gpt-oss-120b", "Apache-2.0", 131072,
                 30, 0, True,
                 "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
                 0.0, 0.0, True, "gpt-oss-120b", "Cerebras free tier, fast"),
    DistillModel("cerebras", "openai/gpt-oss-20b", "Apache-2.0", 131072,
                 30, 0, True,
                 "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
                 0.0, 0.0, True, "gpt-oss-20b", "Cerebras free tier, fast"),

    # ══ SambaNova (Apache-2.0) — free tier ══
    DistillModel("sambanova", "openai/gpt-oss-120b", "Apache-2.0", 131072,
                 50, 0, True,
                 "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY",
                 0.0, 0.0, True, "gpt-oss-120b", "SambaNova free tier"),
]

# NVIDIA NIM: only third-party models allowed (MIT/Apache).
# NVIDIA's own models (Nemotron, etc.) are BANNED per Eval Agreement §2.6.
NVIDIA_ALLOWED_LICENSES = {"MIT", "Apache-2.0"}
NVIDIA_BANNED_MODEL_PREFIXES = (
    "nvidia/",       # NVIDIA's own models (Nemotron, etc.)
    "nv/",           # NVIDIA prefix variant
)


def _nvidia_filter(model: DistillModel) -> bool:
    """Return True if a NVIDIA NIM model is safe for distillation.

    NVIDIA's own models are governed by the NVIDIA Evaluation Agreement §2.6
    which prohibits using outputs to 'improve or develop any other AI model'.
    Third-party models (DeepSeek, Qwen, gpt-oss) on NIM are governed by their
    own MIT/Apache licenses, which explicitly allow distillation.
    """
    if model.provider != "nvidia":
        return True
    if model.license not in NVIDIA_ALLOWED_LICENSES:
        return False
    return all(not model.model_id.lower().startswith(prefix) for prefix in NVIDIA_BANNED_MODEL_PREFIXES)


# Pre-filtered pool: NVIDIA-own models removed, all distill_allowed=True verified
MODEL_POOL: list[DistillModel] = [
    m for m in _MODEL_POOL_RAW
    if _nvidia_filter(m) and m.distill_allowed
]


# ─── Result ──────────────────────────────────────────────────────────────

@dataclass
class DistillResult:
    """A single generated completion from a teacher model."""
    model: str           # provider/model_id
    solution: str        # generated code/solution
    correct: bool        # passed verification (if verified)
    reasoning: str       # CoT reasoning trace (if available)
    raw_response: str    # full raw response
    latency_ms: float    # generation time
    tokens_in: int       # input token count (approx)
    tokens_out: int      # output token count (approx)
    error: str = ""      # error message if failed


# ─── Client ──────────────────────────────────────────────────────────────

class DistillationClient:
    """Multi-provider distillation client with randomized model selection.

    Automatically detects which API keys are available in the environment and
    only uses models from providers with valid credentials. Randomizes model
    selection per goal to maximize quality diversity.

    Usage:
        client = DistillationClient()
        results = client.generate_for_goal(
            goal="Write fibonacci function",
            test_cases=[{"input": "5", "output": "5"}],
            n_samples=4,
        )
    """

    def __init__(self, providers: list[str] | None = None,
                 models: list[str] | None = None,
                 verify_fn: Callable[[str, list[dict]], bool] | None = None,
                 temperature_range: tuple[float, float] = (0.3, 1.0),
                 max_tokens: int = 2048,
                 timeout: float = 60.0):
        """Initialize the distillation client.

        Args:
            providers: restrict to these providers (e.g. ["groq", "deepseek"]).
                None = auto-detect from available API keys.
            models: restrict to these model IDs. None = use all available.
            verify_fn: function(solution, test_cases) -> bool for verification.
                If None, results are not verified (correct=False).
            temperature_range: random temperature range for diversity.
            max_tokens: max output tokens per completion.
            timeout: API request timeout in seconds.
        """
        self.temperature_range = temperature_range
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.verify_fn = verify_fn

        # Filter model pool to available providers
        available = []
        for m in MODEL_POOL:
            if providers and m.provider not in providers:
                continue
            if models and m.model_id not in models:
                continue
            # Check if API key is available
            if os.environ.get(m.env_key):
                available.append(m)

        if not available:
            # Fall back to all models (will error on first call if no keys)
            self.models = [m for m in MODEL_POOL
                           if (not providers or m.provider in providers)
                           and (not models or m.model_id in models)]
        else:
            self.models = available

        # Track per-model request counts for rate limiting
        self._request_counts: dict[str, int] = {m.model_id: 0 for m in self.models}
        self._daily_counts: dict[str, int] = {m.model_id: 0 for m in self.models}
        self._last_reset = time.time()

        # Lock guarding all shared mutable rate-limit state (NP3):
        # _daily_counts, _blacklisted_providers, _request_counts,
        # _provider_request_counts, _provider_cooldowns. Multi-threaded
        # distillation loops mutate these concurrently; without a lock the
        # set/dict mutations lose updates (e.g. blacklisted providers get
        # dropped, daily counts under-count).
        self._lock = threading.Lock()

        # Cache OpenAI clients per provider (different base URLs)
        self._clients: dict[str, OpenAI] = {}

        # Build canonical → [models] index for multi-provider rotation
        self._canonical_groups: dict[str, list[DistillModel]] = {}
        for m in self.models:
            canon = m.canonical or m.model_id
            self._canonical_groups.setdefault(canon, []).append(m)

        # Track per-provider request counts for rate-limit-aware rotation
        self._provider_request_counts: dict[str, int] = {}
        # Provider cooldown: skip providers that recently errored (e.g. 402)
        self._provider_cooldowns: dict[str, float] = {}
        self._cooldown_seconds = 300  # 5 min cooldown after error
        # Permanently blacklisted providers for this process (exhausted quota)
        # Once a provider returns 429 (daily limit) or 402 (payment required),
        # it is blacklisted for the rest of the process — no more useless calls.
        self._blacklisted_providers: set[str] = set()

    def _get_client(self, model: DistillModel) -> OpenAI | None:
        """Get or create an OpenAI client for the model's provider.

        Handles Cloudflare's {account_id} URL substitution.
        """
        if not _OPENAI_AVAILABLE:
            return None

        # Cloudflare needs account_id in the URL — use a per-model cache key
        cache_key = model.provider
        base_url = model.base_url
        if "{account_id}" in base_url:
            account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
            if not account_id:
                return None
            base_url = base_url.replace("{account_id}", account_id)
            cache_key = f"{model.provider}:{account_id}"

        if cache_key not in self._clients:
            api_key = os.environ.get(model.env_key)
            if not api_key:
                return None
            self._clients[cache_key] = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.timeout,
            )
        return self._clients[cache_key]

    def _pick_model(self, goal: str, n: int) -> list[DistillModel]:
        """Pick n diverse models for a goal, maximizing variety.

        Strategy: shuffle the available models and pick n, ensuring different
        providers where possible. This maximizes quality diversity for multi-
        distillation (different models have different strengths/weaknesses).

        Args:
            goal: the goal prompt (used for deterministic seeding if needed)
            n: number of models to pick

        Returns:
            list of n DistillModel instances (may repeat if n > len(models))
        """
        if not self.models:
            raise RuntimeError("No distillation models available. Set API keys "
                             "(GROQ_API_KEY, DEEPSEEK_API_KEY, etc.)")

        # Shuffle models for diversity
        shuffled = self.models.copy()
        random.shuffle(shuffled)

        # If we need more samples than models, cycle through
        if n <= len(shuffled):
            return shuffled[:n]

        # Cycle through, reshuffling each cycle
        result = []
        while len(result) < n:
            batch = shuffled.copy()
            random.shuffle(batch)
            result.extend(batch)
        return result[:n]

    def _pick_model_with_rotation(self, goal: str, n: int
                                  ) -> list[DistillModel]:
        """Pick n models, preferring canonical models with multiple providers.

        When a canonical model (e.g. gpt-oss-120b) is served by multiple
        providers, we rotate through them to bypass per-provider rate limits.
        If one provider hits its RPM/RPD limit, the next call to the same
        canonical model uses a different provider automatically.

        Strategy:
        1. Group available models by canonical name
        2. Pick n distinct canonical models (shuffled for diversity)
        3. For each, pick the provider with the lowest recent request count
        4. If a provider's rate limit is hit, skip to the next provider

        Args:
            goal: the goal prompt
            n: number of models to pick

        Returns:
            list of n DistillModel instances
        """
        if not self.models:
            raise RuntimeError("No distillation models available. Set API keys "
                             "(GROQ_API_KEY, DEEPSEEK_API_KEY, etc.)")

        # N3: reset daily counts once per 24h. _last_reset was set in __init__
        # but never checked, so RPD rate limiting silently broke after the
        # first day (counts only ever incremented).
        with self._lock:
            if time.time() - self._last_reset > 86400:
                self._daily_counts = dict.fromkeys(self._daily_counts, 0)
                self._last_reset = time.time()

        # Shuffle canonical groups for diversity
        canon_names = list(self._canonical_groups.keys())
        random.shuffle(canon_names)

        result: list[DistillModel] = []
        now = time.time()
        for canon in canon_names:
            if len(result) >= n:
                break
            candidates = self._canonical_groups[canon]
            # Sort by recent request count (least-used first) for rotation
            candidates = sorted(
                candidates,
                key=lambda m: self._provider_request_counts.get(m.provider, 0))
            # Pick the least-used provider that hasn't hit its rate limit
            for m in candidates:
                # Skip blacklisted providers (exhausted quota for this process)
                if m.provider in self._blacklisted_providers:
                    continue
                # Skip providers on cooldown (recent errors like 402)
                cooldown_until = self._provider_cooldowns.get(m.provider, 0)
                if now < cooldown_until:
                    continue
                count = self._provider_request_counts.get(m.provider, 0)
                if m.rpm > 0 and count >= m.rpm:
                    continue  # RPM limit hit, try next provider
                if m.rpd > 0 and self._daily_counts.get(m.model_id, 0) >= m.rpd:
                    continue  # RPD limit hit, try next provider
                result.append(m)
                break
            else:
                # All providers for this canonical hit limits — use least-used
                # non-blacklisted one
                avail = [m for m in candidates
                         if m.provider not in self._blacklisted_providers]
                if avail:
                    result.append(avail[0])

        # If we still need more (n > distinct canonicals), cycle
        while len(result) < n:
            for canon in canon_names:
                if len(result) >= n:
                    break
                candidates = sorted(
                    self._canonical_groups[canon],
                    key=lambda m: self._provider_request_counts.get(m.provider, 0))
                # Rotate to a different provider than last time
                last_provider = result[-1].provider if result else ""
                for m in candidates:
                    if m.provider in self._blacklisted_providers:
                        continue
                    if m.provider != last_provider:
                        result.append(m)
                        break
                else:
                    avail = [m for m in candidates
                             if m.provider not in self._blacklisted_providers]
                    if avail:
                        result.append(avail[0])

        return result[:n]

    def _build_prompt(self, goal: str, test_cases: list[dict] | None,
                      system_prompt: str | None = None) -> list[dict]:
        """Build the chat messages for a distillation request.

        Args:
            goal: the task description / goal prompt
            test_cases: list of {"input": ..., "output": ...} dicts
            system_prompt: optional custom system prompt

        Returns:
            list of {"role": ..., "content": ...} message dicts
        """
        if system_prompt is None:
            system_prompt = (
                "You are an expert Python programmer. Generate a clean, "
                "efficient solution to the given problem. The solution must "
                "pass all provided test cases. Output only the Python code, "
                "no explanations."
            )

        user_content = goal
        if test_cases:
            cases_str = "\n".join(
                f"  Input: {tc.get('input', '')} → Output: {tc.get('output', '')}"
                for tc in test_cases[:10]  # limit to 10 cases
            )
            user_content += f"\n\nTest cases:\n{cases_str}\n\nSolution:"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _call_model(self, model: DistillModel, messages: list[dict],
                    temperature: float) -> DistillResult:
        """Call a single model and return the result."""
        client = self._get_client(model)
        if client is None:
            return DistillResult(
                model=f"{model.provider}/{model.model_id}",
                solution="", correct=False, reasoning="",
                raw_response="", latency_ms=0, tokens_in=0, tokens_out=0,
                error="No API client available (missing key or openai package)",
            )

        t0 = time.time()
        try:
            kwargs: dict[str, Any] = {
                "model": model.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": self.max_tokens,
            }
            # Qwen3 on Groq supports reasoning_effort (but qwen3.6 does NOT)
            if model.reasoning and "qwen" in model.model_id.lower() and "3.6" not in model.model_id:
                kwargs["reasoning_effort"] = "high"
            # gpt-oss supports reasoning_effort
            if model.reasoning and "gpt-oss" in model.model_id.lower():
                kwargs["reasoning_effort"] = "high"

            response = client.chat.completions.create(**kwargs)

            latency = (time.time() - t0) * 1000
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning = ""

            # Extract reasoning if available (DeepSeek R1, Qwen3 thinking)
            if hasattr(choice.message, "reasoning_content"):
                reasoning = getattr(choice.message, "reasoning_content", "") or ""
            elif hasattr(choice.message, "reasoning"):
                reasoning = getattr(choice.message, "reasoning", "") or ""

            # Track token usage
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0

            # Track request count (per-model and per-provider for rotation)
            # NP3: guard shared counters with the lock so concurrent threads
            # don't lose increments.
            with self._lock:
                self._request_counts[model.model_id] = \
                    self._request_counts.get(model.model_id, 0) + 1
                self._provider_request_counts[model.provider] = \
                    self._provider_request_counts.get(model.provider, 0) + 1
                self._daily_counts[model.model_id] = \
                    self._daily_counts.get(model.model_id, 0) + 1

            return DistillResult(
                model=f"{model.provider}/{model.model_id}",
                solution=content,
                correct=False,  # verified later
                reasoning=reasoning,
                raw_response=content,
                latency_ms=latency,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        except Exception as e:
            latency = (time.time() - t0) * 1000
            err_str = str(e)
            # Blacklist provider permanently for this process on:
            # - 402 Payment required (no credits)
            # - 429 with "per-day" / "daily" / "per-month" (quota exhausted)
            # - 404 Model not found (model retired/renamed)
            should_blacklist = False
            if "402" in err_str or "Payment required" in err_str or ("429" in err_str and any(kw in err_str.lower() for kw in
                    ("per-day", "per_day", "daily", "per-month", "per_month",
                     "free-models-per-day", "rate limit exceeded"))) or ("404" in err_str and "model" in err_str.lower()):
                should_blacklist = True

            if should_blacklist:
                # NP3: guard the blacklist set with the lock — concurrent
                # threads calling .add() on a set can lose updates.
                with self._lock:
                    is_new = model.provider not in self._blacklisted_providers
                    if is_new:
                        self._blacklisted_providers.add(model.provider)
                if is_new:
                    print(f"  [BLACKLIST] {model.provider} exhausted for this "
                          f"process — {err_str[:80]}")
            elif "429" in err_str:
                # Transient rate limit (RPM) — short cooldown, not blacklist
                with self._lock:
                    self._provider_cooldowns[model.provider] = time.time() + 60
            elif "402" in err_str:
                with self._lock:
                    self._provider_cooldowns[model.provider] = time.time() + self._cooldown_seconds
            return DistillResult(
                model=f"{model.provider}/{model.model_id}",
                solution="", correct=False, reasoning="",
                raw_response="", latency_ms=latency,
                tokens_in=0, tokens_out=0,
                error=str(e),
            )

    def generate_for_goal(self, goal: str,
                          test_cases: list[dict] | None = None,
                          n_samples: int = 4,
                          system_prompt: str | None = None,
                          verify: bool = True) -> list[DistillResult]:
        """Generate n diverse completions for a single goal.

        Randomizes model selection per goal to maximize quality diversity.
        Each sample may come from a different teacher model, at a different
        temperature.

        Args:
            goal: the task description
            test_cases: list of {"input": ..., "output": ...} for verification
            n_samples: number of completions to generate
            system_prompt: optional custom system prompt
            verify: if True and verify_fn is set, verify each solution

        Returns:
            list of n DistillResult objects
        """
        models = self._pick_model_with_rotation(goal, n_samples)
        messages = self._build_prompt(goal, test_cases, system_prompt)

        results: list[DistillResult] = []
        for model in models:
            # Random temperature for diversity
            temp = random.uniform(*self.temperature_range)
            result = self._call_model(model, messages, temp)

            # Verify if we have a verify function and test cases
            if verify and self.verify_fn and test_cases and result.solution:
                try:
                    result.correct = self.verify_fn(result.solution, test_cases)
                except Exception:
                    result.correct = False

            results.append(result)

        return results

    def generate_batch(self, goals: list[str],
                       test_cases_per_goal: list[list[dict]] | None = None,
                       n_samples_per_goal: int = 4,
                       system_prompt: str | None = None,
                       verify: bool = True,
                       delay_between: float = 0.0) -> list[list[DistillResult]]:
        """Generate completions for multiple goals.

        Args:
            goals: list of task descriptions
            test_cases_per_goal: parallel list of test case lists
            n_samples_per_goal: completions per goal
            system_prompt: optional custom system prompt
            verify: verify each solution
            delay_between: delay between goals in seconds (rate limiting)

        Returns:
            list of lists of DistillResult objects
        """
        if test_cases_per_goal is None:
            test_cases_per_goal = [None] * len(goals)

        all_results: list[list[DistillResult]] = []
        for i, goal in enumerate(goals):
            tc = test_cases_per_goal[i] if i < len(test_cases_per_goal) else None
            results = self.generate_for_goal(
                goal, tc, n_samples_per_goal, system_prompt, verify)
            all_results.append(results)
            if delay_between > 0 and i < len(goals) - 1:
                time.sleep(delay_between)

        return all_results

    def filter_correct(self, results: list[DistillResult]
                       ) -> list[DistillResult]:
        """Filter to only correct (verified) results."""
        return [r for r in results if r.correct and not r.error]

    def to_training_pairs(self, results: list[DistillResult]
                          ) -> list[dict]:
        """Convert verified results to training pairs for ReplayBuffer.

        Returns list of dicts with keys: prompt, solution, quality, test_passed.
        """
        pairs = []
        for r in results:
            if not r.correct or r.error:
                continue
            pairs.append({
                "prompt": r.raw_response,  # or the original goal
                "solution": r.solution,
                "quality": 1.0,
                "test_passed": True,
                "model": r.model,
                "reasoning": r.reasoning,
            })
        return pairs

    def stats(self) -> dict:
        """Return per-model request statistics."""
        return {
            "models_available": len(self.models),
            "providers": list(set(m.provider for m in self.models)),
            "request_counts": dict(self._request_counts),
            "daily_counts": dict(self._daily_counts),
            "total_requests": sum(self._request_counts.values()),
            "blacklisted_providers": list(self._blacklisted_providers),
            "active_providers": self.active_providers(),
        }

    def active_providers(self) -> list[str]:
        """Return providers not blacklisted or on cooldown."""
        now = time.time()
        result = []
        for m in self.models:
            if m.provider in self._blacklisted_providers:
                continue
            if now < self._provider_cooldowns.get(m.provider, 0):
                continue
            if m.provider not in result:
                result.append(m.provider)
        return result

    def all_providers_exhausted(self) -> bool:
        """Return True if all providers are blacklisted or on cooldown.

        The main loop should check this and end early to avoid useless calls.
        """
        now = time.time()
        for m in self.models:
            if m.provider in self._blacklisted_providers:
                continue
            if now < self._provider_cooldowns.get(m.provider, 0):
                continue
            # At least one provider is still usable
            return False
        return True

    def available_models(self) -> list[dict]:
        """List all available models with their metadata."""
        return [
            {
                "provider": m.provider,
                "model_id": m.model_id,
                "license": m.license,
                "context": m.context,
                "rpm": m.rpm,
                "rpd": m.rpd,
                "reasoning": m.reasoning,
                "cost_in": m.cost_in,
                "cost_out": m.cost_out,
                "canonical": m.canonical,
                "distill_allowed": m.distill_allowed,
                "notes": m.notes,
                "api_key_set": bool(os.environ.get(m.env_key)),
            }
            for m in self.models
        ]

    def canonical_groups(self) -> dict[str, list[str]]:
        """List canonical model → list of providers serving it.

        Models with multiple providers can rotate between them to bypass
        per-provider rate limits.
        """
        return {
            canon: [m.provider for m in models]
            for canon, models in self._canonical_groups.items()
        }

    def distill_into_buffer(self, goals: list[str],
                            replay_buffer,
                            test_cases_per_goal: list[list[dict]] | None = None,
                            n_samples_per_goal: int = 4,
                            system_prompt: str | None = None,
                            delay_between: float = 0.0) -> dict:
        """Generate verified training data and store it in a ReplayBuffer.

        This is the main entry point for the distillation pipeline:
        1. Generate n diverse completions per goal (randomized models)
        2. Verify each solution against test cases
        3. Store verified solutions in the ReplayBuffer as golden trajectories
        4. Return statistics

        The ReplayBuffer then feeds into GRPOTrainer._inject_golden_replays()
        during training, preventing catastrophic forgetting.

        Args:
            goals: list of task descriptions
            replay_buffer: ReplayBuffer instance to store results in
            test_cases_per_goal: parallel list of test case lists
            n_samples_per_goal: completions per goal (more = more diversity)
            system_prompt: optional custom system prompt
            delay_between: delay between goals in seconds

        Returns:
            dict with statistics: n_generated, n_correct, n_stored, per_model
        """
        all_results = self.generate_batch(
            goals, test_cases_per_goal, n_samples_per_goal,
            system_prompt, verify=True, delay_between=delay_between)

        n_generated = 0
        n_correct = 0
        n_stored = 0
        per_model: dict[str, dict] = {}

        for goal, results in zip(goals, all_results):
            for r in results:
                n_generated += 1
                model_key = r.model
                if model_key not in per_model:
                    per_model[model_key] = {"generated": 0, "correct": 0}

                per_model[model_key]["generated"] += 1

                if r.error:
                    continue
                if not r.correct:
                    continue

                n_correct += 1
                per_model[model_key]["correct"] += 1

                # Store in replay buffer as a golden trajectory
                replay_buffer.add({
                    "prompt": goal,
                    "solution": r.solution,
                    "quality": 1.0,
                    "test_passed": True,
                    "source": "distill",
                    "teacher_model": r.model,
                    "reasoning": r.reasoning,
                })
                n_stored += 1

        return {
            "n_goals": len(goals),
            "n_generated": n_generated,
            "n_correct": n_correct,
            "n_stored": n_stored,
            "accuracy": n_correct / max(n_generated, 1),
            "per_model": per_model,
            "buffer_size": len(replay_buffer),
        }
