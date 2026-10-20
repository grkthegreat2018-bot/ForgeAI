"""Activation catalog — the complete, data-driven inventory of every
ForgeEngine activation feature (mirrors ``forge/engine/activation.py``).

Pure python (no Qt / torch imports) so it can be unit-tested on any machine
and shared by the Engine page, presets, and validation. Every field of
``ActivationConfig`` appears here exactly once — a unit test enforces that
the two never drift apart.

Kinds:
    combo  — fixed set of options (string values; "none" maps to ``None``)
    bool   — feature flag
    int    — plain integer
    opt_float / opt_int — optional numeric (empty → ``None``)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Option:
    value: str            # value passed to activate(); "none" → None
    label: str            # shown in the combo
    tip: str = ""         # per-option tooltip


@dataclass(frozen=True)
class FieldSpec:
    name: str                     # ActivationConfig field name
    label: str                    # human label
    kind: str                     # combo | bool | int | opt_float | opt_int
    category: str
    tooltip: str
    options: tuple[Option, ...] = ()
    default: object = None        # mirrored from ActivationConfig
    lo: float = 0.0
    hi: float = 0.0
    decimals: int = 0
    step: float = 1.0
    suffix: str = ""


# ── categories (display order) ─────────────────────────────────────────
CATEGORIES = [
    "Core Strategies",
    "KV & Memory",
    "Caching",
    "Prefill & Attention",
    "Kernels & Graphs",
    "Speculative Decoding",
    "Scheduling",
    "MoE",
    "Position Encoding",
    "Quantization R&D",
    "Runtime",
    "I/O & Loading",
    "Architecture Overrides",
]

# ── option tables for the core combos ──────────────────────────────────

KV_CACHE_OPTIONS = (
    Option("rotorquant", "RotorQuant 4-bit", "Rotary-quantized 4-bit KV — the optimal preset default"),
    Option("paged", "Paged blocks", "vLLM-style block paging, lossless bf16 KV"),
    Option("standard", "Standard dense", "Plain contiguous KV cache (bf16)"),
    Option("spectral", "SpectralKV", "Spectral (PCA) KV compression — V10 native"),
    Option("s4r", "S4R 4-bit", "S4R quantized KV cache"),
    Option("hadamard_int4", "Hadamard INT4", "Hadamard-rotated 4-bit KV (QuaRot-style)"),
    Option("snapkv", "SnapKV", "Observation-window KV compression at prefill"),
    Option("snapkv_4bit", "SnapKV 4-bit", "SnapKV compression + 4-bit quantization"),
    Option("2bit", "2-bit KV", "Aggressive 2-bit KV cache"),
    Option("compressed", "Compressed", "Compressed KV cache backend"),
    Option("paged_eviction", "Paged + eviction", "Paged cache with LRU eviction policy"),
    Option("kvzip", "KVZip", "KVZip compression"),
    Option("hqe_kv", "HQE KV", "Hyper-precision-quality entropy KV"),
    Option("xquant", "XQuant KV", "XQuant quantized KV"),
    Option("beacon", "BeaconKV", "Beacon-centroid KV eviction — online k-means over keys, live during decode (R60)"),
    Option("anchor", "AnchorKV", "Farthest-point anchors + residual-coverage eviction, live during decode (R60)"),
    Option("metakv", "MetaKV", "Feature-based dispatcher over KV strategies (R60)"),
    Option("cpu_offload", "CPU offload", "KV lives in pinned CPU RAM (32GB) — near-zero VRAM"),
    Option("disk_offload", "Disk offload", "KV spilled to local disk"),
    Option("streaming", "Streaming", "AirLLM-style: no KV, layer-streamed inference"),
)

DECODING_OPTIONS = (
    Option("standard", "Standard", "Autoregressive token-by-token decode"),
    Option("mtp_selfspec", "MTP self-speculative", "Multi-token-prediction head drafts, k=4 (V10 MTP)"),
    Option("eagle3", "EAGLE-3", "Tree drafting via .eagle3.safetensors head"),
    Option("speculative", "Speculative", "Draft-then-verify speculative decoding"),
    Option("medusa", "Medusa", "Medusa multi-head parallel decoding"),
    Option("dspark", "DSpark", "DSpark decoding strategy"),
    Option("ngram_speculative", "N-gram speculative", "Prompt-lookup n-gram drafting — no draft model needed"),
    Option("external_draft_speculative", "External draft", "Speculative decoding with an external draft model"),
    Option("uno", "Uno (Psi-Spec)", "Lossless block diffusion decoding — n-gram drafts, AR verify (R49)"),
    Option("specbound", "SpecBound", "Adaptive bounded self-speculation — early-exit drafts, EMA-calibrated (R60)"),
    Option("batched", "Batched", "Batched continuous decoding (generate_batch)"),
)

QUANTIZE_OPTIONS = (
    Option("none", "BF16 (none)", "No weight quantization — full precision"),
    Option("forge_quant", "ForgeQuant", "INT4 dense + INT8 sparse outliers — best quality/size tradeoff on Blackwell"),
    Option("nvfp4", "NVFP4", "FP4 E2M1 block-32 — Blackwell native, 3.8x smaller"),
    Option("w8a8", "W8A8", "INT8 weights + INT8 activations"),
    Option("fp8", "FP8", "FP8 weights (Hopper+)"),
    Option("int8", "INT8", "Weight-only INT8"),
    Option("int4", "INT4", "Weight-only INT4, group size 128"),
)

ACCELERATION_OPTIONS = (
    Option("none", "None", "No decode acceleration runtime"),
    Option("cuda_graph", "CUDA graph", "Full-graph capture for decode steps"),
    Option("megakernel", "Megakernel", "Compiled fused megakernel decode"),
    Option("flex_decoding", "Flex decoding", "FlexAttention decode wrapper"),
    Option("airllm_streaming", "AirLLM streaming", "Layer streaming — weights never fully in VRAM"),
)


def _flags(names: str) -> tuple[str, ...]:
    return tuple(n for n in names.split() if n)


# ── the full field inventory (order within a category = display order) ─
FIELDS: tuple[FieldSpec, ...] = (
    # Core Strategies
    FieldSpec("kv_cache", "KV cache", "combo", "Core Strategies",
              "KV cache backend. RotorQuant/Spectral are the tuned defaults on this hardware.",
              KV_CACHE_OPTIONS, default="paged"),
    FieldSpec("decoding", "Decoding", "combo", "Core Strategies",
              "Decode strategy. mtp_selfspec uses the V10 MTP head for ~2-4x decode speed.",
              DECODING_OPTIONS, default="standard"),
    FieldSpec("quantize", "Quantization", "combo", "Core Strategies",
              "Weight quantization applied at activation. NVFP4 is native on RTX 5070 (SM120).",
              QUANTIZE_OPTIONS, default=None),
    FieldSpec("acceleration", "Acceleration", "combo", "Core Strategies",
              "Low-level decode acceleration runtime (graphs / megakernels / streaming).",
              ACCELERATION_OPTIONS, default=None),
    # KV & Memory
    FieldSpec("kv_bits", "KV bits", "int", "KV & Memory",
              "Quantization bits for the KV cache (4 or 8).", default=4,
              lo=2, hi=8, step=2, suffix=" bit"),
    FieldSpec("kv_cache_tokens", "KV token budget", "opt_int", "KV & Memory",
              "Hard cap on KV cache token capacity (empty = model max_seq_len).",
              default=None, lo=256, hi=131072, step=512, suffix=" tok"),
    FieldSpec("mrl_keep_ratio", "MRL keep ratio", "opt_float", "KV & Memory",
              "Matryoshka keep-ratio (e.g. 0.75 truncates latent dims; empty = off).",
              default=None, lo=0.1, hi=1.0, decimals=2, step=0.05),
    FieldSpec("use_v0_warm", "V0 warm start", "bool", "KV & Memory",
              "Warm-start KV cache from V0 (pre-V10) checkpoint weights — "
              "faster convergence on first tokens after load."),
    FieldSpec("use_progressive_kv", "Progressive KV", "bool", "KV & Memory",
              "Anchor tokens at 8-bit + residual quantized — quality-preserving KV compression."),
    FieldSpec("use_kara_kv", "KARA KV", "bool", "KV & Memory",
              "KARA kernel-aware rotary-approximated KV cache."),
    FieldSpec("use_moment_kv", "Moment KV", "bool", "KV & Memory",
              "Moment-based KV quantization (outlier-aware moments)."),
    FieldSpec("use_kvpop", "KVPop", "bool", "KV & Memory",
              "Popularity-weighted KV retention/eviction."),
    FieldSpec("use_conf_kv", "CONF-KV", "bool", "KV & Memory",
              "Confidence-gated KV cache compression."),
    FieldSpec("use_matryoshka_kv", "Matryoshka KV", "bool", "KV & Memory",
              "Nested-granularity KV compression — multiple resolution levels in one cache."),
    FieldSpec("use_replay_ssm", "ReplaySSM cache", "bool", "KV & Memory",
              "Input-caching ring buffer for SSM state reconstruction during speculative decode rollback."),
    # Caching
    FieldSpec("use_prefix_cache", "Prefix cache (LRU)", "bool", "Caching",
              "Cache KV for repeated prompt prefixes — big win for chat/agent system prompts."),
    FieldSpec("use_learned_prefix_cache", "Learned prefix cache", "bool", "Caching",
              "ML-eviction prefix cache (18-47% size cut vs LRU)."),
    FieldSpec("use_chunked_prefix_cache", "Chunked prefix cache", "bool", "Caching",
              "LMCache-style rolling-hash chunked prefix reuse."),
    FieldSpec("use_hotprefix", "HotPrefix", "bool", "Caching",
              "Hotness-aware prefix promotion between GPU and CPU."),
    FieldSpec("use_unified_radix", "Unified radix cache", "bool", "Caching",
              "Radix-tree KV sharing across sessions/branches."),
    FieldSpec("use_cache_blend", "CacheBlend", "bool", "Caching",
              "Non-prefix KV reuse with re-scoring (R&D14)."),
    # Prefill & Attention
    FieldSpec("use_chunked_prefill", "Chunked prefill", "bool", "Prefill & Attention",
              "512-token chunked prefill — smoother VRAM profile on long prompts."),
    FieldSpec("use_hybrid_prefill", "Hybrid prefill", "bool", "Prefill & Attention",
              "Hybrid chunked prefill scheduling."),
    FieldSpec("use_pod_attention", "PoD attention", "bool", "Prefill & Attention",
              "Prefill-decode overlap (CUDA only)."),
    FieldSpec("use_seq_split", "Seq split", "bool", "Prefill & Attention",
              "Sequence-aware attention split across SMs (CUDA only)."),
    FieldSpec("use_compact_attn", "Compact attention", "bool", "Prefill & Attention",
              "Compact attention kernel path."),
    FieldSpec("use_wavelength_pruning", "Wavelength pruning", "bool", "Prefill & Attention",
              "ATFlash wavelength-based attention pruning."),
    FieldSpec("use_cosa", "CoSA", "bool", "Prefill & Attention",
              "CoSa sparse attention."),
    FieldSpec("use_mosa", "MoSA", "bool", "Prefill & Attention",
              "Mixture-of-sparse-attention heads."),
    FieldSpec("use_fa4", "FA4 attention", "bool", "Prefill & Attention",
              "FlashAttention-4 kernels (CUDA only)."),
    FieldSpec("use_spec_attn", "Spec attention", "bool", "Prefill & Attention",
              "Reserved speculative-attention flag."),
    # Kernels & Graphs
    FieldSpec("use_compile", "torch.compile", "bool", "Kernels & Graphs",
              "Compile the model (1.3-2x decode, +2-3 min load). Broken on some triton stacks — use fast load if compile fails."),
    FieldSpec("use_compile_autotune", "Compile autotune", "bool", "Kernels & Graphs",
              "Autotune compiled kernels (CUDA only, slower first load)."),
    FieldSpec("use_triton_conv", "Triton conv fusion", "bool", "Kernels & Graphs",
              "Fuse conv layers with Triton kernels (10 layers patched on V10)."),
    FieldSpec("use_fused_qk_norm_rope_cache", "Fused QK-Norm+RoPE", "bool", "Kernels & Graphs",
              "One fused kernel for QK-norm + RoPE + KV-cache write (bit-exact with eager)."),
    FieldSpec("use_block_fusion", "Block fusion", "bool", "Kernels & Graphs",
              "Per-block CUDA graph + compile — 16 block graphs on V10."),
    FieldSpec("use_breakable_cuda_graph", "Breakable CUDA graph", "bool", "Kernels & Graphs",
              "CUDA graph capture that can break out for dynamic shapes (CUDA only)."),
    FieldSpec("use_corun", "CoRun", "bool", "Kernels & Graphs",
              "CoRun deterministic kernel scheduling (CUDA only)."),
    FieldSpec("use_foundry", "Foundry graphs", "bool", "Kernels & Graphs",
              "Foundry graph templates (CUDA only)."),
    # Speculative Decoding
    FieldSpec("use_adaptive_spec", "Adaptive speculative", "bool", "Speculative Decoding",
              "N-gram + EAGLE-3 hybrid adaptive drafting."),
    FieldSpec("use_suffix_spec", "Suffix decoding", "bool", "Speculative Decoding",
              "Suffix-automaton drafting for repetitive outputs."),
    FieldSpec("use_peagle", "P-EAGLE", "bool", "Speculative Decoding",
              "Persistent EAGLE speculative decoding (CUDA only)."),
    FieldSpec("use_lookahead_gate", "Lookahead quality gate", "bool", "Speculative Decoding",
              "Quality gate for lookahead/speculative acceptance."),
    # Scheduling
    FieldSpec("use_fastserve", "FastServe", "bool", "Scheduling",
              "FastServe iteration-level scheduling."),
    FieldSpec("use_libra", "Libra", "bool", "Scheduling",
              "Libra request scheduler."),
    FieldSpec("use_faser", "FASER", "bool", "Scheduling",
              "FASER scheduling."),
    FieldSpec("use_kairos", "Kairos", "bool", "Scheduling",
              "Kairos timing-aware scheduling."),
    FieldSpec("use_triroute", "TriRoute", "bool", "Scheduling",
              "TriRoute router/scheduler."),
    # MoE
    FieldSpec("use_elbow_moe", "Elbow MoE", "bool", "MoE",
              "Elbow load-balanced MoE routing."),
    FieldSpec("use_alloc_moe", "Alloc MoE", "bool", "MoE",
              "Allocation-aware MoE routing."),
    FieldSpec("use_lda_moe", "LDA MoE", "bool", "MoE",
              "LDA-gated MoE routing."),
    # Position Encoding
    FieldSpec("use_jet_long", "Jet-Long RoPE", "bool", "Position Encoding",
              "Jet-Long context RoPE scaling."),
    FieldSpec("use_rope_id", "RoPE-ID", "bool", "Position Encoding",
              "RoPE with ID-aware position mapping."),
    FieldSpec("use_lerope", "LeRoPE", "bool", "Position Encoding",
              "Length-extrapolatory RoPE."),
    FieldSpec("use_lampe", "LaMPE", "bool", "Position Encoding",
              "LaMPE position encoding."),
    # Quantization R&D
    FieldSpec("use_adamx", "AdaMX", "bool", "Quantization R&D",
              "AdaMX adaptive mixed-precision quantization."),
    FieldSpec("use_sharq", "SharQ", "bool", "Quantization R&D",
              "SharQ shard-aware quantization."),
    FieldSpec("use_mosaic_quant", "MosaicQuant", "bool", "Quantization R&D",
              "MosaicQuant production quantization path."),
    FieldSpec("use_aoh", "Autonomy of Heads", "bool", "Quantization R&D",
              "AoH per-head autonomy quantization."),
    FieldSpec("use_quamba2", "Quamba2 W4A8", "bool", "Quantization R&D",
              "SSM-specific W4A8 quantization — INT4 weights + INT8 activations, FP16 SSM core."),
    # Runtime
    FieldSpec("warmup", "Warmup", "bool", "Runtime",
              "Pre-run a dummy token after activation so first generation is fast."),
    # I/O & Loading (R35)
    FieldSpec("use_avmp", "AVMP paging", "bool", "I/O & Loading",
              "Asymmetric Virtual Memory Paging — tiered GPU/CPU memory for large tensors."),
    FieldSpec("use_virtual_tensor", "Virtual tensor (eLLM)", "bool", "I/O & Loading",
              "Elastic LLM virtual tensors — automatic GPU/CPU spillover for oversized layers."),
    FieldSpec("use_progressive_load", "Progressive GGUF load", "bool", "I/O & Loading",
              "Stream GGUF checkpoints layer-by-layer to reduce peak load memory."),
    # Architecture Overrides (V12+ — applied at model construction time
    # via config_overrides, NOT via activate(). These are checkpoint-safe
    # features that are identity/zero-initialized for lossless warm start.)
    FieldSpec("use_mamba3", "Mamba-3 SSM", "bool", "Architecture Overrides",
              "Complex-valued SSM states (Mamba-3). NOT lossless — complex RMSNorm changes dynamics. Requires Mamba3Key checkpoint conversion."),
    FieldSpec("use_forge_hybrid", "ForgeHybrid SSM", "bool", "Architecture Overrides",
              "Zero-init SSM path alongside attention in non-Mamba blocks. Bit-exact lossless at init (verified)."),
    FieldSpec("use_outro", "OutRo attention", "bool", "Architecture Overrides",
              "Output rotary embedding for attention blocks. Zero-strength at init = bit-exact lossless (verified)."),
    FieldSpec("use_kronecker_embed", "Kronecker embedding", "bool", "Architecture Overrides",
              "Byte-decomposed embedding. NOT checkpoint-compatible without explicit conversion (changes embed shape)."),
    FieldSpec("use_pit", "PIT tying", "bool", "Architecture Overrides",
              "Pseudo-inverse tying for embed/head. NOT checkpoint-compatible without explicit conversion."),
)

_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}


# ── presets ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PresetSpec:
    name: str
    label: str
    description: str
    config: dict = field(default_factory=dict)


PRESETS: tuple[PresetSpec, ...] = (
    PresetSpec(
        "optimal", "Optimal",
        "Engine-tuned best for RTX 5070 12GB: rotorquant 4-bit KV, Triton conv, "
        "prefix cache, fused QK-Norm+RoPE, chunked prefill, seq split. "
        "Compile follows the fast-load toggle.",
        {
            "kv_cache": "rotorquant", "kv_bits": 4, "decoding": "standard",
            "quantize": None, "acceleration": None,
            "use_triton_conv": True, "use_prefix_cache": True,
            "use_fused_qk_norm_rope_cache": True, "use_chunked_prefill": True,
            "use_seq_split": True, "warmup": True,
        }),
    PresetSpec(
        "fast", "Fast interactive",
        "Skip compile + graphs for a quick interactive session; keep the tuned "
        "KV/prefix features. Loads in ~15-45s.",
        {
            "kv_cache": "rotorquant", "kv_bits": 4, "decoding": "standard",
            "quantize": None, "acceleration": None,
            "use_compile": False, "use_triton_conv": True,
            "use_prefix_cache": True, "use_fused_qk_norm_rope_cache": True,
            "use_chunked_prefill": True, "warmup": True,
        }),
    PresetSpec(
        "min_vram", "Max memory saving",
        "Every VRAM saver on: NVFP4 weights, CPU-offloaded KV, no graphs, "
        "prefix caching for reuse.",
        {
            "kv_cache": "cpu_offload", "kv_bits": 4, "decoding": "standard",
            "quantize": "nvfp4", "acceleration": None,
            "use_compile": False, "use_prefix_cache": True, "warmup": True,
        }),
    PresetSpec(
        "speed", "Max decode speed",
        "Compile + block fusion + MTP self-speculative decoding + CUDA graphs. "
        "Longest load, fastest tokens.",
        {
            "kv_cache": "rotorquant", "kv_bits": 4,
            "decoding": "mtp_selfspec", "quantize": "nvfp4",
            "acceleration": "cuda_graph",
            "use_compile": True, "use_compile_autotune": True,
            "use_triton_conv": True, "use_prefix_cache": True,
            "use_fused_qk_norm_rope_cache": True, "use_chunked_prefill": True,
            "use_seq_split": True, "use_block_fusion": True, "warmup": True,
        }),
    PresetSpec(
        "long_context", "Long context",
        "Long-context posture: spectral KV, LeRoPE + Jet-Long scaling, "
        "chunked prefill, paged eviction.",
        {
            "kv_cache": "spectral", "kv_bits": 8, "decoding": "standard",
            "quantize": "nvfp4", "acceleration": None,
            "use_compile": False, "use_chunked_prefill": True,
            "use_prefix_cache": True, "use_lerope": True,
            "use_jet_long": True, "warmup": True,
        }),
    PresetSpec(
        "quality", "Max quality",
        "Minimal-loss config: no weight quant, 8-bit KV, dense cache, "
        "warmup only.",
        {
            "kv_cache": "standard", "kv_bits": 8, "decoding": "standard",
            "quantize": None, "acceleration": None,
            "use_compile": False, "warmup": True,
        }),
)

PRESET_BY_NAME = {p.name: p for p in PRESETS}


# ── helpers ────────────────────────────────────────────────────────────

def fields_by_category() -> list[tuple[str, list[FieldSpec]]]:
    """Ordered [(category, [fields…])] for building the UI."""
    out: list[tuple[str, list[FieldSpec]]] = []
    for cat in CATEGORIES:
        fs = [f for f in FIELDS if f.category == cat]
        if fs:
            out.append((cat, fs))
    return out


def spec(name: str) -> FieldSpec | None:
    return _BY_NAME.get(name)


def default_config() -> dict:
    """ActivationConfig defaults as a plain dict (for populating the UI)."""
    return {f.name: f.default for f in FIELDS}


def preset_config(name: str) -> dict | None:
    p = PRESET_BY_NAME.get(name)
    if p is None:
        return None
    cfg = default_config()
    cfg.update(p.config)
    return cfg


def normalize_value(f: FieldSpec, value) -> object:
    """UI value → activate() value ("none" → None, coercions)."""
    if f.kind == "combo":
        if value in (None, "", "none"):
            return None
        return str(value)
    if f.kind == "opt_float" or f.kind == "opt_int":
        if value in (None, ""):
            return None
        return float(value) if f.kind == "opt_float" else int(value)
    if f.kind == "int":
        return int(value or f.default or 0)
    if f.kind == "bool":
        return bool(value)
    return value


def validate(config: dict) -> list[str]:
    """Return a list of human-readable problems (empty = OK)."""
    errors: list[str] = []
    for name, value in config.items():
        f = _BY_NAME.get(name)
        if f is None:
            errors.append(f"unknown feature: {name}")
            continue
        if f.kind == "combo":
            valid = {o.value for o in f.options}
            if value is not None and str(value) not in valid:
                errors.append(f"{f.label}: '{value}' is not a valid option")
        elif f.kind == "int":
            if value is not None and not float(f.lo) <= float(value) <= float(f.hi):
                errors.append(f"{f.label}: {value} outside {f.lo:g}–{f.hi:g}")
    # kv_bits constraint
    kvb = config.get("kv_bits")
    if kvb is not None and int(kvb) not in (2, 4, 8):
        errors.append(f"KV bits: {kvb} (use 2, 4 or 8)")
    return errors


def active_diff(current: dict | None, desired: dict) -> list[str]:
    """Readable list of 'label: current → desired' for changed fields."""
    if not current:
        return []
    out = []
    for f in FIELDS:
        if f.name not in desired:
            continue
        cur, want = current.get(f.name), desired[f.name]
        if cur != want:
            out.append(f"{f.label}: {cur!r} → {want!r}")
    return out
