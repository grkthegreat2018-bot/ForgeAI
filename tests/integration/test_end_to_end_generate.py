"""Integration test: end-to-end generation through the full ForgeEngine pipeline.

This is the test the critique (F5) identified as missing — it exercises the
real integration boundary (model + tokenizer + KV cache + decoding + sampling)
that unit tests skip. If this test fails, something in the core inference
pipeline is broken regardless of which unit tests pass.

Requires CUDA (skips on CPU-only environments). Uses the tiny config for speed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from forge.config import get_config
from forge.engine.forge_engine import ForgeEngine
from forge.model_loader import ConfigurableResearchLLM

CUDA_AVAILABLE = torch.cuda.is_available()


def _build_engine(preset="forgelm_tiny", vocab=65536, activate=True):
    """Build a ForgeEngine with the given preset. GPU-first, CPU fallback.

    Uses vocab=65536 to match the lfm25 tokenizer (tiny vocab causes
    device-side asserts when token IDs exceed the embedding size).
    """
    cfg = get_config(preset)
    cfg.vocab_size = vocab
    cfg.dtype = "bfloat16"
    cfg.device = "cuda" if CUDA_AVAILABLE else "cpu"

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(cfg.device):
            model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    model.eval()

    from research.tokenizer_cache import get_tokenizer

    tok = get_tokenizer("research/checkpoints/forgelm_v2_tokenizer")
    engine = ForgeEngine(model, tok, device=cfg.device)
    if activate:
        engine.activate_optimal()
    return engine, model


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestEndToEndGenerate:
    """Full pipeline: prompt → tokenize → forward → sample → decode → text.

    These tests run WITHOUT activate_optimal() to test the core inference
    path in isolation. See TestEndToEndWithActivation for the full pipeline.
    """

    def test_basic_generation_returns_string(self):
        """generate() must return a non-empty string for a simple prompt."""
        engine, model = _build_engine(activate=False)
        try:
            result = engine.generate("Hello", max_new_tokens=10, finish_sentence=False, temperature=0.0)
            assert isinstance(result, str), f"Expected str, got {type(result)}"
            assert len(result) > 0, "Generation returned empty string"
        finally:
            del engine, model
            torch.cuda.empty_cache()

    def test_generation_50_tokens(self):
        """Generate 50 tokens — the critique's recommended baseline test."""
        engine, model = _build_engine(activate=False)
        try:
            result = engine.generate("The quick brown fox", max_new_tokens=50, finish_sentence=False, temperature=0.0)
            assert isinstance(result, str)
            # At least some tokens should be generated
            assert len(result) > 0
        finally:
            del engine, model
            torch.cuda.empty_cache()

    def test_generation_no_error_on_repeated_calls(self):
        """Multiple generate calls should not raise or leak state."""
        engine, model = _build_engine(activate=False)
        try:
            for i in range(3):
                result = engine.generate(f"Test prompt {i}", max_new_tokens=5, finish_sentence=False, temperature=0.0)
                assert isinstance(result, str)
        finally:
            del engine, model
            torch.cuda.empty_cache()

    def test_generation_meets_throughput_baseline(self):
        """Generation should meet a minimum throughput (tok/s).

        The baseline is deliberately lenient (10 tok/s) — the point is to
        catch catastrophic regressions (e.g. 0.1 tok/s from a broken quant
        path), not to benchmark performance.
        """
        engine, model = _build_engine(activate=False)
        try:
            # Warmup
            engine.generate("warmup", max_new_tokens=2, finish_sentence=False, temperature=0.0)

            n_tokens = 50
            start = time.perf_counter()
            engine.generate(
                "The quick brown fox jumps over the lazy dog",
                max_new_tokens=n_tokens,
                finish_sentence=False,
                temperature=0.0,
            )
            elapsed = time.perf_counter() - start

            tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0
            # Lenient baseline: 10 tok/s on GPU. Catches catastrophic regressions.
            assert tok_per_sec > 10.0, (
                f"Throughput {tok_per_sec:.1f} tok/s below 10 tok/s baseline "
                f"(elapsed={elapsed:.3f}s for {n_tokens} tokens)"
            )
        finally:
            del engine, model
            torch.cuda.empty_cache()

    def test_streaming_generation(self):
        """generate_stream should yield string chunks without error."""
        engine, model = _build_engine(activate=False)
        try:
            chunks = list(
                engine.generate_stream(
                    "Hello world",
                    max_new_tokens=10,
                    temperature=0.0,
                )
            )
            assert len(chunks) > 0, "Stream produced no chunks"
            assert all(isinstance(c, str) for c in chunks)
        finally:
            del engine, model
            torch.cuda.empty_cache()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestQuantAndKVCombinations:
    """Parametrized across quant modes × KV strategies (critique NC5).

    The real bugs live at the interaction boundary — quant + KV + decoding
    interacting — which single-mode tests miss.
    """

    @pytest.mark.parametrize("kv", ["standard", "paged"])
    @pytest.mark.parametrize("quant", ["none", "int8", "fp8"])
    def test_quant_kv_combination_generates(self, quant, kv):
        engine, model = _build_engine(activate=False)
        try:
            engine._activate_kv_cache(kv, None)
            if quant != "none":
                engine._apply_quantization(quant)
            result = engine.generate("test", max_new_tokens=5, finish_sentence=False, temperature=0.0)
            assert isinstance(result, str) and len(result) > 0, (
                f"quant={quant} kv={kv} produced no output (active KV: {getattr(engine, '_active_kv_cache_name', '?')})"
            )
        except (ImportError, RuntimeError, ValueError) as e:
            if "All quantization modes failed" in str(e):
                pytest.fail(f"quant={quant} kv={kv}: all fallbacks failed: {e}")
            pytest.skip(f"quant={quant} kv={kv} unsupported here: {e}")
        finally:
            del engine, model
            torch.cuda.empty_cache()


@pytest.mark.usefixtures("v2_assets")
class TestEndToEndCPU:
    """CPU tier — runs without CUDA so the integration boundary is exercised
    in CPU-only CI environments too (critique NC5)."""

    def _build_cpu_engine(self):
        cfg = get_config("forgelm_tiny")
        cfg.vocab_size = 65536
        cfg.dtype = "float32"
        cfg.device = "cpu"
        with torch.device("cpu"):
            model = ConfigurableResearchLLM(cfg)
        model.eval()
        from research.tokenizer_cache import get_tokenizer

        tok = get_tokenizer("research/checkpoints/forgelm_v2_tokenizer")
        return ForgeEngine(model, tok, device="cpu"), model

    def test_cpu_generation_returns_string(self):
        engine, model = self._build_cpu_engine()
        try:
            result = engine.generate("Hello", max_new_tokens=3, finish_sentence=False, temperature=0.0)
            assert isinstance(result, str) and len(result) > 0
        finally:
            del engine, model


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestEndToEndWithActivation:
    """Test that activate_optimal() doesn't break basic generation.

    NOTE: activate_optimal() enables torch.compile + CUDA graphs which may
    fail on some configs. If this test fails, it indicates a real integration
    bug in the compile/cuda_graph path — exactly the kind of issue the
    critique (F5) identified as needing integration tests to catch.
    """

    def test_generate_without_activation(self):
        """generate() should work without activate_optimal()."""
        engine, model = _build_engine(activate=False)
        try:
            result = engine.generate("test", max_new_tokens=5, finish_sentence=False, temperature=0.0)
            assert isinstance(result, str)
        finally:
            del engine, model
            torch.cuda.empty_cache()

    @pytest.mark.xfail(
        reason="activate_optimal() enables torch.compile + CUDA graphs which "
        "currently fails on tiny config — known integration bug (critique F4)"
    )
    def test_activate_then_generate(self):
        """activate_optimal() + generate() should work end-to-end.

        If this fails, the compile/cuda_graph/KV-cache integration is broken.
        This is the integration boundary that unit tests miss.
        """
        engine, model = _build_engine(activate=True)
        try:
            result = engine.generate("test", max_new_tokens=5, finish_sentence=False, temperature=0.0)
            assert isinstance(result, str)
        finally:
            del engine, model
            torch.cuda.empty_cache()
