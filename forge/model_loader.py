"""Modular model factory and inference engine for ForgeAI research."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# NOTE: no runtime configuration at import time (critique F17/NC7) —
# ``import forge.model_loader`` is side-effect-free.  Entrypoints opt in via
# ``forge.runtime.configure.configure()`` (idempotent); see
# ``forge/runtime/configure.py``.
import torch



# ── Split-module re-exports (backward compatibility) ─────────────────────────
# The implementation lives in forge/model/; this module is the stable facade.
from forge.model.attention_ops import (  # noqa: F401
    _build_block_diag_causal_mask,
    _causal_mask,
    flash_attention,
    varlen_attention,
)
from forge.model.builders import build_attention, build_ffn  # noqa: F401
from forge.model.kv_cache import *
from forge.model.kv_cache import KVCache, PreAllocatedKVCache, create_kv_cache, unpack_output_with_kv  # noqa: F401
from forge.model.layers import (  # noqa: F401
    DoubleGatedConvLayer,
    GroupedQueryAttention,
    ModularBlock,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFFN,
)
from forge.model.llm import ConfigurableResearchLLM
from forge.model.loader import ModelLoader


def load_default_model(
    config_name: str = "forgelm_v2",
    checkpoint_path: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    moe_top_k: int = 0,
    compile_mode: str | None = None,
    fast_load: bool = True,
) -> tuple["ConfigurableResearchLLM", "Any"]:
    """Load a model + tokenizer in one call.

    Centralizes the common pattern used across 10+ files:
        cfg = get_config(name, device=device)
        model = ModelLoader.build_model_fast(cfg, checkpoint_path=..., moe_top_k=..., dtype=...)
        tokenizer = get_tokenizer(...)

    Args:
        config_name: model config name (default "forgelm_v2")
        checkpoint_path: path to .safetensors checkpoint (default: config default)
        device: "cuda" or "cpu"
        dtype: torch.bfloat16 or torch.float32 (default: bf16 for cuda, fp32 for cpu)
        moe_top_k: MoE top-k routing (0 = dense_bypass)
        compile_mode: torch.compile mode if set (e.g. "default", "reduce-overhead")
        fast_load: when True (default), uses meta-init + assign=True + parallel
            tokenizer + OS prefetch for 3-6x faster cold boot. Set to False
            for the traditional build path.

    Returns:
        (model, tokenizer) tuple
    """
    from forge.config import get_config
    from research.tokenizer_cache import get_tokenizer

    cfg = get_config(config_name, device=device)
    if dtype is None:
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
    if checkpoint_path is None:
        from research.paths import V2_CHECKPOINT
        if V2_CHECKPOINT.exists():
            checkpoint_path = str(V2_CHECKPOINT)

    # Tokenizer auto-dispatch: Qwen-family checkpoints (vocab 151936) need
    # the Qwen tokenizer, not the canonical LFM tokenizer (vocab 65536).
    from forge.engine.forge_engine import _tokenizer_for_vocab
    tok_path = _tokenizer_for_vocab(cfg.vocab_size)

    # Fast load: start tokenizer in parallel with model build (hides ~2.7s)
    tok_fut = None
    _tok_ex = None
    if fast_load:
        from concurrent.futures import ThreadPoolExecutor
        _tok_ex = ThreadPoolExecutor(max_workers=1)
        tok_fut = _tok_ex.submit(get_tokenizer, tok_path)

    try:
        model = ModelLoader.build_model_fast(
            cfg, checkpoint_path=checkpoint_path,
            moe_top_k=moe_top_k, dtype=dtype, fast_load=fast_load)
        model.to(device).eval()

        if compile_mode is not None:
            try:
                model = model.compile_for_inference(mode=compile_mode)
            except Exception:
                logger.debug("compile_for_inference failed, using uncompiled model", exc_info=True)

        if tok_fut is not None:
            tokenizer = tok_fut.result()
        else:
            tokenizer = get_tokenizer(tok_path)
    finally:
        if _tok_ex is not None:
            _tok_ex.shutdown(wait=False)
    return model, tokenizer


def quantize_int4(model: torch.nn.Module, group_size: int = 32) -> torch.nn.Module:
    """Apply int4 weight-only quantization using torchao.

    Reduces model VRAM by ~58% (bf16 → int4) with minimal accuracy loss.
    Works with torch.compile and FSDP2.

    Requires MSLK (mslk-cuda>=1.0.0) for the int4 packing kernels.
    If MSLK is not available, falls back to int8 weight-only quantization
    (50% VRAM reduction instead of 58%).

    Args:
        model: model to quantize (must be on CUDA)
        group_size: quantization group size (32 = good balance, 64 = faster)

    Returns:
        The quantized model (modified in-place).

    Example:
        model = load_default_model("forgelm_v2")
        model = quantize_int4(model)  # 2.3GB → ~0.7GB VRAM (int4)
    """
    try:
        from torchao.quantization import quantize_
    except ImportError:
        print("torchao not installed — skipping quantization")
        return model

    if not torch.cuda.is_available():
        print("quantization requires CUDA — skipping")
        return model

    # Try int4 first (requires MSLK), fall back to int8
    try:
        from torchao.quantization import Int4WeightOnlyConfig
        quantize_(model, Int4WeightOnlyConfig(group_size=group_size))
        print("  [torchao] Applied int4 weight-only quantization")
    except (ImportError, RuntimeError) as e:
        print(f"  [torchao] int4 unavailable ({e}), falling back to int8")
        from torchao.quantization import Int8WeightOnlyConfig
        quantize_(model, Int8WeightOnlyConfig())
        print("  [torchao] Applied int8 weight-only quantization")
    return model


# Pre-import key modules at the bottom of the file (after all classes are
# defined) to avoid ~660ms of lazy import overhead during
# ConfigurableResearchLLM.__init__. This moves the import cost from the
# first model build to module load time. Wrapped in try/except so missing
# optional dependencies don't break the import.
try:
    import gigatoken  # noqa: F401

    # Pre-import tokenizer dependencies (avoid GIL contention when tokenizer
    # loads in a background thread during fast_load)
    import tokenizers  # noqa: F401

    from forge.keys.architecture.attn_residual_key import AttnResModule  # noqa: F401
    from forge.keys.architecture.mhc_key import MHCModule  # noqa: F401
    from forge.keys.architecture.mod_router_key import ModRouter  # noqa: F401
    from forge.keys.architecture.titan_memory_key import TitanMemory  # noqa: F401
    from forge.keys.attention.differential_attn_key import DifferentialAttention  # noqa: F401
    from forge.keys.misc.pit_key import PITEmbedding, PITLMHead  # noqa: F401
    from forge.keys.quantization.bitnet_b158_key import build_bitnet_linear  # noqa: F401
    from forge.training.bitnet_lora import convert_to_bitnet_everywhere  # noqa: F401
except ImportError:
    pass

# Pre-warm the class hierarchy by building a tiny model on meta device.
# The first ConfigurableResearchLLM() call takes ~640ms due to Python's
# first-time class instantiation overhead (nn.Module.__init__, meta tensor
# creation, etc.). Subsequent calls are ~50ms. By pre-building at import
# time, we move this cost to module load time (before the user's critical
# path), making the actual model build fast.
# Also pre-initialize the CUDA context so background threads can use CUDA
# immediately without ~500ms context creation overhead.
try:
    from forge.config import get_config as _get_config_warmup
    _warmup_cfg = _get_config_warmup("forgelm_v2", device="meta")
    with torch.device("meta"):
        _warmup_model = ConfigurableResearchLLM(_warmup_cfg)
    del _warmup_model, _warmup_cfg
    # Pre-initialize CUDA context (needed for background weight load threads)
    if torch.cuda.is_available():
        torch.cuda.init()
except Exception:
    logger.debug("CUDA context pre-initialization failed", exc_info=True)


if __name__ == "__main__":
    from forge.config import get_config

    for name in ["forgelm_tiny", "forgelm_v2"]:
        print("\n" + "=" * 50)
        cfg = get_config(name)
        cfg.device = "cpu"
        ModelLoader.build_model(cfg)
