"""Declarative feature registry for ForgeEngine.

Replaces the 11 ``_activate_*`` methods (~500 lines of boilerplate) with a
single registry of ``FeatureSpec`` entries.  Each entry maps a ``use_*`` flag
to a small handler function that performs the lazy import, instantiation,
``.apply(model)`` call, and status-message return.

Adding a new inference feature is now a one-liner: append a ``FeatureSpec``
to ``_FEATURE_REGISTRY``.  No changes to ``ForgeEngine.activate()`` or
``__init__`` are needed (feature slots are initialised lazily via ``getattr``).

Handler contract
----------------
``handler(engine, feature_flags) -> str | None``
    * Receives the ``ForgeEngine`` instance and the full ``feature_flags`` dict.
    * Performs all side effects: lazy import, instantiation, ``.apply(model)``,
      ``setattr(engine, attr, instance)``.
    * Returns a status message string (printed by the caller) or ``None`` to
      stay silent.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge.engine.forge_engine import ForgeEngine

# Type alias for handler functions.
FeatureHandler = Callable[["ForgeEngine", dict], "str | None"]


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative spec for one optional ForgeEngine inference feature.

    Attributes:
        flag:      The ``use_*`` key in ``feature_flags`` that enables this
                   feature.
        handler:   Callable that activates the feature on the engine.
        cuda_only: If ``True``, the feature is skipped on non-CUDA devices.
        introduced_round: R&D round number (e.g. 66) that introduced this
                   flag — populate for NEW flags going forward so the
                   sunset policy (docs/RD_SUNSET_POLICY.md) can age out
                   flags that never ship. ``None`` = predates the policy
                   (no backfill).
    """

    flag: str
    handler: FeatureHandler
    cuda_only: bool = False
    introduced_round: int | None = None


# ── Handler helpers ────────────────────────────────────────────────────────

def _cfg(eng: ForgeEngine, name: str, default: Any) -> Any:
    """Read an attribute from the model config with a fallback."""
    return getattr(getattr(eng.model, "config", None), name, default)


def _kv_dims(eng: ForgeEngine) -> tuple[int, int]:
    return eng._kv_dimensions


# ── Attention / prefill features ────────────────────────────────────────────

def _h_wavelength_pruner(eng, _flags):
    from forge.engine.attention.atflash import WavelengthPrunedAttention
    rope_base = _cfg(eng, "rope_base", 1_000_000.0)
    eng._wavelength_pruner = WavelengthPrunedAttention(
        rope_base=rope_base, prune_factor=1.0)
    eng._wavelength_pruner.apply(eng.model)
    return "ATFlash wavelength pruning: active (37-48% attn cut)"


def _h_pod_attention(eng, _flags):
    from forge.engine.attention.pod_attention import PODAttentionScheduler
    eng._pod_attention = PODAttentionScheduler(device=str(eng.device))
    return "POD-Attention: active (prefill-decode overlap)"


def _h_adaptive_spec(eng, _flags):
    from forge.decoding.adaptive_speculative import AdaptiveSpeculativeDecoder
    eagle_head = getattr(eng.model, "eagle_head", None)
    mtp_module = getattr(eng.model, "mtp_module", None)
    eng._adaptive_spec = AdaptiveSpeculativeDecoder(
        eagle_head=eagle_head, mtp_module=mtp_module)
    return "Adaptive speculative decoding: active (n-gram + EAGLE-3)"


def _h_cosa(eng, _flags):
    from forge.engine.attention.cosa import CoSAWrapper
    eng._cosa = CoSAWrapper(block_size=16, budget_ratio=0.5, min_seq_len=2048)
    eng._cosa.apply(eng.model)
    return None


def _h_seq_split(eng, _flags):
    from forge.engine.attention.seq_aware_split import (
        SequenceAwareSplitWrapper,
    )
    eng._seq_split = SequenceAwareSplitWrapper(min_seq_len=512)
    eng._seq_split.apply(eng.model, eng.device)
    return None


def _h_compact_attn(eng, _flags):
    from forge.engine.attention.compact_attention import (
        CompactAttentionWrapper,
    )
    eng._compact_attn = CompactAttentionWrapper(
        block_size=16, budget_ratio=0.5, min_kv_len=2048)
    eng._compact_attn.apply(eng.model)
    return None


# ── Decode / graph features ──────────────────────────────────────────────────

def _h_suffix_spec(eng, _flags):
    from forge.decoding.suffix_decoding import SuffixDecoder
    eng._suffix_spec = SuffixDecoder(max_draft_len=8)
    return "Suffix decoding: active (training-free, code/RAG)"


def _h_fused_qk_cache(eng, _flags):
    from forge.engine.attention.fused_qk_norm_rope_cache import (
        FusedQKNormRopeCacheWrapper,
    )
    eng._fused_qk_cache = FusedQKNormRopeCacheWrapper(kv_quant_bits=None)
    eng._fused_qk_cache.apply(eng.model)
    return None


def _h_block_fusion(eng, _flags):
    from forge.engine.graphs.block_fusion import CompiledBlockFusion
    eng._block_fusion = CompiledBlockFusion(eng.model, device=str(eng.device))
    eng._block_fusion.capture()
    return "Block fusion: active (per-block CUDA graph + compile)"


def _h_compile_autotune(eng, _flags):
    from forge.engine.graphs.compile_autotune import auto_tune_compile
    sample_input = torch_zeros(eng, 1, 1)
    model_id = _cfg(eng, "name", "default")
    best_mode, eng.model = auto_tune_compile(
        eng.model, sample_input, model_id=model_id)
    return f"torch.compile auto-tuned: {best_mode}"


def _h_hybrid_prefill(eng, _flags):
    from forge.engine.prefill import HybridChunkedPrefiller
    eng._chunked_prefiller = HybridChunkedPrefiller(
        eng.model, chunk_size=512, device=str(eng.device))
    return "Hybrid chunked prefill: active (adaptive chunking)"


# ── Prefix / routing features ────────────────────────────────────────────────

def _h_learned_prefix_cache(eng, _flags):
    from forge.engine.kv.learned_prefix_cache import LearnedPrefixCache
    eng._learned_prefix_cache = LearnedPrefixCache(max_entries=256)
    eng._prefix_cache = eng._learned_prefix_cache  # replace LRU dict
    return "Learned prefix cache: active (ML eviction, 18-47% size cut)"


def _h_hotprefix(eng, _flags):
    from forge.engine.scheduler.hotprefix import HotPrefixManager
    eng._hotprefix = HotPrefixManager(gpu_capacity=8)
    return "HotPrefix: active (hotness-aware GPU/CPU promotion)"


def _h_mosa(eng, _flags):
    from forge.engine.kv.mosa import MoSAWrapper
    eng._mosa = MoSAWrapper(k_ratio=0.5, min_seq_len=512)
    eng._mosa.apply(eng.model)
    return None


def _h_triroute(eng, _flags):
    from forge.engine.scheduler.triroute import TriRouteWrapper
    eng._triroute = TriRouteWrapper(local_window=256, min_seq_len=512)
    eng._triroute.apply(eng.model)
    return None


# ── Graph / kernel features ──────────────────────────────────────────────────

def _h_breakable_cuda_graph(eng, _flags):
    """Static-KV decode-step graph capture.

    Builds a PreAllocatedKVCache (pos_t tensor mode) + a persistent
    attention-mask buffer so the decode step is fully tensor-driven and
    CUDA-graph capturable. The closure is reused at decode time —
    see StandardDecoding._generate_static.
    """
    import torch as _t

    from forge.engine.graphs.breakable_cuda_graph import BreakableCudaGraph
    from forge.model.kv_cache import create_kv_cache

    eng._bcg = BreakableCudaGraph(eng.model, device=str(eng.device))
    cfg = getattr(eng.model, 'config', None)
    max_len = min(getattr(cfg, 'max_seq_len', 4096) or 4096, 8192)
    try:
        cache = create_kv_cache(eng.model, max_total=max_len, batch=1,
                                device=eng.device)
        cache.enable_tensor_pos()
        mask_buf = _t.zeros(1, cache.max_seq_len, dtype=_t.long,
                            device=eng.device)

        raw_model = getattr(eng.model, '_orig_mod', eng.model)

        # KV-length buckets: each capture reads a [:b] VIEW of the same
        # cache storage (see PreAllocatedKVCache.view_len) — a full-buffer
        # read costs ~30x wasted attention at short contexts.  Switching
        # buckets at decode time is free: the views share storage.
        buckets = [b for b in (512, 1024, 2048, 4096, 8192)
                   if b <= cache.max_seq_len]
        if cache.max_seq_len not in buckets:
            buckets.append(cache.max_seq_len)

        def _step(m, mask_view, c):
            def decode_step(ids_t1):
                return m(ids_t1, attention_mask=mask_view,
                         preallocated_cache=c, use_cache=True,
                         logits_last_only=True)
            return decode_step

        def _capture_buckets(c, m_buf, B, want, prefix):
            """Capture one graph per KV bucket in ``want`` for batch size B.
            Compiled step tried twice per bucket, eager fallback once.
            Returns the number of buckets captured on the COMPILED model."""
            n_compiled = 0
            for b in want:
                c.view_len = b
                mask_view = m_buf[:, :b]
                tag = f"{prefix}kv{b}"
                if tag in eng._bcg._decode_graphs:
                    n_compiled += 1
                    continue
                for _ in range(2):
                    eng._bcg.capture_decode(
                        batch_sizes=[B],
                        decode_step=_step(eng.model, mask_view, c),
                        tag=tag)
                    if tag in eng._bcg._decode_graphs:
                        break
                if tag in eng._bcg._decode_graphs:
                    n_compiled += 1
                else:
                    eng._bcg.capture_decode(
                        batch_sizes=[B],
                        decode_step=_step(raw_model, mask_view, c),
                        tag=tag)
            c.view_len = None
            return n_compiled

        eng._decode_state = {'cache': cache, 'mask': mask_buf,
                             'buckets': buckets, 'batch': {}}
        # Decoding strategies see the model, not the engine — attach.
        eng.model._bcg = eng._bcg
        eng.model._decode_state = eng._decode_state

        def _ensure_static_batch(B, plen, need_len):
            """Lazy per-batch-size static decode state for BatchedDecoding.

            Allocates a batch-B preallocated cache + mask and captures the
            KV buckets covering positions (plen .. need_len].  Returns
            {'cache','mask','prefix','buckets'} or None (functional
            fallback).  B=1 uses the eagerly-captured state instead.
            """
            if (eng._decode_state is None or B <= 1 or B > 32
                    or not buckets):
                return None
            hi = next((b for b in buckets if b >= need_len), None)
            if hi is None:
                return None  # request outgrows every bucket
            want = [b for b in buckets if b >= plen and b <= need_len]
            if hi not in want:
                want.append(hi)
            bs_map = eng._decode_state['batch']
            bs = bs_map.get(B)
            if bs is not None and not bs:
                return None  # allocation failed before — don't retry
            if bs is not None and bs['cache'].max_seq_len < hi:
                # A previous, shorter request sized this batch's cache;
                # the new horizon needs a bigger allocation. The old
                # b{B}_ graphs replay on the OLD cache/mask storage —
                # evict them before recapture or decode reads freed
                # buffers.
                try:
                    for _tag in [t for t in eng._bcg._decode_graphs
                                 if t.startswith(f"b{B}_")]:
                        eng._bcg._decode_graphs.pop(_tag, None)
                        eng._bcg._static_buffers.pop(_tag, None)
                    c = create_kv_cache(eng.model, max_total=hi,
                                        batch=B, device=eng.device)
                    c.enable_tensor_pos()
                    m_buf = _t.zeros(B, c.max_seq_len, dtype=_t.long,
                                     device=eng.device)
                    bs = {'cache': c, 'mask': m_buf, 'prefix': f"b{B}_"}
                    bs_map[B] = bs
                except Exception:
                    return None
            if bs is None:
                try:
                    # Size the cache to the request horizon (hi), not the
                    # global max — B=16+ at full max_len wastes GBs of KV
                    # that the bucket views never touch.
                    c = create_kv_cache(eng.model, max_total=hi,
                                        batch=B, device=eng.device)
                    c.enable_tensor_pos()
                    m_buf = _t.zeros(B, c.max_seq_len, dtype=_t.long,
                                     device=eng.device)
                    bs = {'cache': c, 'mask': m_buf, 'prefix': f"b{B}_"}
                    bs_map[B] = bs
                except Exception:
                    bs_map[B] = {}
                    return None
            missing = [b for b in want
                       if f"{bs['prefix']}kv{b}"
                       not in eng._bcg._decode_graphs]
            _capture_buckets(bs['cache'], bs['mask'], B, missing,
                             bs['prefix'])
            have = [b for b in want
                    if f"{bs['prefix']}kv{b}" in eng._bcg._decode_graphs]
            if sorted(have) != sorted(want):
                return None  # partial coverage — decode would stall
            bs['buckets'] = want
            return bs

        eng.model._ensure_static_batch = _ensure_static_batch

        # Prefer capturing the COMPILED decode step — fused inductor
        # kernels + a single graph launch beats unfused eager replay.
        # Long fused-kernel names need the short TRITON_CACHE_DIR set in
        # configure.py; on compile failure fall back to eager capture
        # (still collapses ~200 per-token launches into one replay).
        compiled_graphs = _capture_buckets(cache, mask_buf, 1, buckets, "")
        eng._decode_state['compiled_step'] = compiled_graphs == len(buckets)
    except Exception as e:
        eng._decode_state = None
        return f"Breakable CUDA Graph: failed ({e})"
    return f"Breakable CUDA Graph: {eng._bcg.stats()}"


def _h_corun(eng, _flags):
    from forge.engine.scheduler.corun import CoRunScheduler
    eng._corun = CoRunScheduler(
        eng.model, max_concurrency=8,
        device=str(eng.device), dtype=eng.dtype)
    eng._corun.capture_decode_graph()
    return f"CoRun deterministic: {eng._corun.stats()}"


def _h_foundry(eng, _flags):
    from forge.engine.graphs.foundry import FoundryRunner
    eng._foundry = FoundryRunner(eng.model, device=str(eng.device))
    templates = eng._foundry.list_templates()
    if templates:
        for tname in templates[:4]:
            eng._foundry.materialize(tname)
        return f"Foundry: materialized {len(templates)} templates"
    eng._foundry.capture_and_save("decode", batch_size=1)
    eng._foundry.capture_and_save("decode", batch_size=4)
    return "Foundry: captured new templates"


def _h_fa4(eng, _flags):
    from forge.engine.attention.fa4_attention import FA4Attention
    eng._fa4 = FA4Attention(use_fp8_kv=False)
    return f"FA4: {eng._fa4.stats()}"


# ── Optional KV features ─────────────────────────────────────────────────────

def _h_kara_kv(eng, _flags):
    from forge.engine.kv.kara_kv import KARAKVCache
    n_kv, head_dim = _kv_dims(eng)
    eng._kara_kv = KARAKVCache(n_kv, head_dim, target_budget=2048)
    return "KARA KV: sliding-window compression + Token2Chunk"


def _h_moment_kv(eng, _flags):
    from forge.engine.kv.moment_kv import MomentKVCache
    n_kv, _head_dim = _kv_dims(eng)
    eng._moment_kv = MomentKVCache(n_kv, max_budget=2048)
    return "MomentKV: moment-statistics eviction (geometric regularity)"


def _h_kvpop(eng, _flags):
    from forge.engine.kv.kvpop import KVpopCache
    n_kv, _head_dim = _kv_dims(eng)
    eng._kvpop = KVpopCache(n_kv, long_range_budget=1024)
    return "KVpop: predictive online pruning (future-attention supervised)"


def _h_conf_kv(eng, _flags):
    from forge.engine.kv.conf_kv import ConfKVCache
    n_kv, _head_dim = _kv_dims(eng)
    eng._conf_kv = ConfKVCache(
        n_kv, min_budget=256, max_budget=4096)
    return "CONF-KV: confidence-aware budget + mixed FP16/INT8 storage"


def _h_matryoshka_kv(eng, _flags):
    from forge.engine.kv.matryoshka_kv import MatryoshkaKVCache
    _n_kv, _head_dim = _kv_dims(eng)
    eng._matryoshka_kv = MatryoshkaKVCache(
        n_levels=3, importance_metric="norm")
    return "MatryoshkaKV: nested-granularity KV compression (fp16→bf16→fp8)"


# ── Position features ────────────────────────────────────────────────────────

def _h_jet_long(eng, _flags):
    from forge.engine.scheduler.jet_long import JetLongAttention
    n_heads = _cfg(eng, "n_heads", 32)
    n_kv, head_dim = _kv_dims(eng)
    eng._jet_long = JetLongAttention(
        n_heads, head_dim, n_kv,
        local_window=1024, max_train_length=32768)
    return f"Jet-Long: dynamic bifocal RoPE (zero-shot 128K, {eng._jet_long.stats()})"


def _h_rope_id(eng, _flags):
    from forge.engine.position import RoPEID
    head_dim = _cfg(eng, "head_dim", 64)
    eng._rope_id = RoPEID(head_dim, rotation_fraction=0.25)
    return f"RoPE-ID: {eng._rope_id.stats()}"


def _h_lerope(eng, _flags):
    from forge.engine.position import integrate_lerope_into_model
    n_replaced = integrate_lerope_into_model(eng.model)
    return f"LeRoPE: replaced RoPE in {n_replaced} layers"


def _h_lampe(eng, _flags):
    from forge.engine.scheduler.lampe import LaMPE
    head_dim = _cfg(eng, "head_dim", 64)
    eng._lampe = LaMPE(head_dim, train_length=32768)
    return f"LaMPE: {eng._lampe.stats()}"


# ── Speculative features ─────────────────────────────────────────────────────

def _h_peagle(eng, _flags):
    from forge.decoding.peagle import PEAGLEDraftHead, PEAGLEDraftHeadTied
    d_model = _cfg(eng, "d_model", 2048)
    vocab_size = _cfg(eng, "vocab_size", 65536)
    n_draft = _cfg(eng, "mtp_n_heads", 4)  # reuse MTP n_heads or default 4
    use_tied = _cfg(eng, "use_peagle_tied", False)
    if use_tied:
        lora_rank = _cfg(eng, "peagle_lora_rank", 32)
        eng._peagle = PEAGLEDraftHeadTied(d_model, vocab_size,
                                           n_draft_tokens=n_draft,
                                           lora_rank=lora_rank)
    else:
        eng._peagle = PEAGLEDraftHead(d_model, vocab_size, n_draft_tokens=n_draft)
    eng._peagle = eng._peagle.to(eng.device)
    return "P-EAGLE: parallel speculative decoding (K=4 in 1 pass)"


def _h_lookahead_gate(eng, _flags):
    from forge.decoding.lookahead_gate import LookaheadQualityGate
    eng._lookahead_gate = LookaheadQualityGate(n_lookahead=4)
    return "Lookahead Quality Gate: geometry-based block acceptance"


# ── Scheduler features ───────────────────────────────────────────────────────

def _h_fastserve(eng, _flags):
    from forge.engine.scheduler.fastserve import FastServeScheduler
    eng._fastserve = FastServeScheduler(max_batch_size=8)
    return "FastServe: skip-join MLFQ + token-level preemption"


def _h_libra(eng, _flags):
    from forge.engine.scheduler.libra import LibraScheduler
    eng._libra = LibraScheduler(max_batch_size=8, chunk_size=512)
    return "Libra: micro-request partitioning + SLO-aware batching"


def _h_faser(eng, _flags):
    from forge.engine.attention.faser import (
        FASERController,
        FASEREarlyExit,
        FASERFrontier,
    )
    eng._faser = FASERController()
    eng._faser_exit = FASEREarlyExit()
    eng._faser_frontier = FASERFrontier()
    return "FASER: dynamic spec length + early exit + frontier overlap"


def _h_kairos(eng, _flags):
    from forge.engine.scheduler.kairos import KairosScheduler
    eng._kairos = KairosScheduler(max_batch_size=8)
    return "Kairos: urgency-based prefill + slack-guided decode"


def _h_unified_radix(eng, _flags):
    from forge.engine.scheduler.unified_radix import UnifiedRadixCache
    eng._unified_radix = UnifiedRadixCache(max_gpu_tokens=32768)
    return "Unified Radix Cache: hybrid prefix caching + HiCache"


# ── MoE features ─────────────────────────────────────────────────────────────

def _h_elbow_moe(eng, _flags):
    from forge.engine.scheduler.moe_optim import ElbowRouter
    eng._elbow_moe = ElbowRouter(min_experts=1, max_experts=4)
    return "Elbow MoE: training-free dynamic top-k routing"


def _h_alloc_moe(eng, _flags):
    from forge.engine.scheduler.moe_optim import AllocMoE
    n_layers = _cfg(eng, "n_layers", 16)
    eng._alloc_moe = AllocMoE(n_layers, n_experts=8, total_budget=0.5)
    return f"Alloc-MoE: {eng._alloc_moe.stats()}"


def _h_lda_moe(eng, _flags):
    from forge.engine.scheduler.moe_optim import LDACalibrator
    n_layers = _cfg(eng, "n_layers", 16)
    eng._lda_moe = LDACalibrator(n_layers)
    return "LDA MoE: distribution-consistent reduced routing"


# ── Adaptive quantization features ───────────────────────────────────────────

def _h_adamx(eng, _flags):
    from forge.quant.adaptive_quant import AdaMXQuantizer
    eng._adamx = AdaMXQuantizer(block_size=32, scheme="auto")
    return "AdaMX: adaptive microscaling (per-block scheme selection)"


def _h_sharq(eng, _flags):
    from forge.quant.adaptive_quant import SharQQuantizer
    eng._sharq = SharQQuantizer(n_ratio=2, m_ratio=4)
    return "SharQ: sparse-dense FP4 activation quantization"


def _h_mosaic_quant(eng, _flags):
    from forge.quant.adaptive_quant import MosaicQuantizer
    eng._mosaic = MosaicQuantizer(block_size=128)
    return "MosaicQuant: inlier-outlier disaggregation 4-bit"


def _h_quamba2(eng, _flags):
    """Quamba2 W4A8 quantization for SSM (Mamba) blocks.

    Quantizes in_proj/x_proj/out_proj/conv1d to W4A8 while keeping the SSM
    core (A_log, dt_bias, D, selective scan) in FP16.  Falls back to w8a8
    if no SSM blocks are found.
    """
    from forge.quant.quamba2 import quantize_model_quamba2
    cfg = getattr(eng.model, "config", None)
    gs = getattr(cfg, "quamba2_group_size", 128) if cfg else 128
    alpha = getattr(cfg, "quamba2_smoothquant_alpha", 0.5) if cfg else 0.5
    n_q = quantize_model_quamba2(eng.model, group_size=gs,
                                 smoothquant_alpha=alpha)
    if n_q == 0:
        return ("Quamba2: no SSM blocks found — skipping "
                "(use quantize='quamba2' for fallback to w8a8)")
    return (f"Quamba2: {n_q} SSM blocks quantized (W4A8, "
            f"group={gs}, alpha={alpha}). SSM core kept in FP16.")


# ── Final attention features ─────────────────────────────────────────────────

def _h_aoh(eng, _flags):
    from forge.engine.kv.aoh_retmask import AutonomyOfHeads
    n_heads = _cfg(eng, "n_heads", 32)
    head_dim = _cfg(eng, "head_dim", 64)
    d_model = _cfg(eng, "d_model", 2048)
    eng._aoh = AutonomyOfHeads(
        n_heads, head_dim, d_model, sparsity_ratio=0.5)
    return "AoH: data-free head classification (retrieval vs streaming)"


def _h_replay_ssm(eng, _flags):
    from forge.engine.kv.replay_ssm import ReplaySSMCache
    n_layers = _cfg(eng, "n_layers", 16)
    max_replay = _cfg(eng, "replay_ssm_max_tokens", 512)
    eng._replay_ssm = ReplaySSMCache(n_layers, max_replay_tokens=max_replay)
    return (f"ReplaySSM: input-caching for SSM state reconstruction "
            f"({n_layers} layers, max_replay={max_replay})")


# ── Utility ──────────────────────────────────────────────────────────────────

def torch_zeros(eng: ForgeEngine, *shape: int):
    """Create a long-tensor on the engine's device (used for autotune input)."""
    import torch
    return torch.zeros(*shape, dtype=torch.long, device=eng.device)


# ── The registry ─────────────────────────────────────────────────────────────
# New specs MUST set ``introduced_round`` — flags that never beat the
# incumbent within 3 R&D rounds are deleted, not left dormant
# (docs/RD_SUNSET_POLICY.md). Legacy specs keep None (no backfill).

_FEATURE_REGISTRY: list[FeatureSpec] = [
    # Attention / prefill
    FeatureSpec("use_wavelength_pruning", _h_wavelength_pruner),
    FeatureSpec("use_pod_attention", _h_pod_attention, cuda_only=True),
    FeatureSpec("use_adaptive_spec", _h_adaptive_spec),
    FeatureSpec("use_cosa", _h_cosa),
    FeatureSpec("use_seq_split", _h_seq_split, cuda_only=True),
    FeatureSpec("use_compact_attn", _h_compact_attn),
    # Decode / graph
    FeatureSpec("use_suffix_spec", _h_suffix_spec),
    FeatureSpec("use_fused_qk_norm_rope_cache", _h_fused_qk_cache),
    FeatureSpec("use_block_fusion", _h_block_fusion, cuda_only=True),
    FeatureSpec("use_compile_autotune", _h_compile_autotune, cuda_only=True),
    FeatureSpec("use_hybrid_prefill", _h_hybrid_prefill),
    # Prefix / routing
    FeatureSpec("use_learned_prefix_cache", _h_learned_prefix_cache),
    FeatureSpec("use_hotprefix", _h_hotprefix),
    FeatureSpec("use_mosa", _h_mosa),
    FeatureSpec("use_triroute", _h_triroute),
    # Graph / kernel
    FeatureSpec("use_breakable_cuda_graph", _h_breakable_cuda_graph, cuda_only=True),
    FeatureSpec("use_corun", _h_corun, cuda_only=True),
    FeatureSpec("use_foundry", _h_foundry, cuda_only=True),
    FeatureSpec("use_fa4", _h_fa4, cuda_only=True),
    # Optional KV
    FeatureSpec("use_kara_kv", _h_kara_kv),
    FeatureSpec("use_moment_kv", _h_moment_kv),
    FeatureSpec("use_kvpop", _h_kvpop),
    FeatureSpec("use_conf_kv", _h_conf_kv),
    FeatureSpec("use_matryoshka_kv", _h_matryoshka_kv),
    # Position
    FeatureSpec("use_jet_long", _h_jet_long),
    FeatureSpec("use_rope_id", _h_rope_id),
    FeatureSpec("use_lerope", _h_lerope),
    FeatureSpec("use_lampe", _h_lampe),
    # Speculative
    FeatureSpec("use_peagle", _h_peagle, cuda_only=True),
    FeatureSpec("use_lookahead_gate", _h_lookahead_gate),
    # Scheduler
    FeatureSpec("use_fastserve", _h_fastserve),
    FeatureSpec("use_libra", _h_libra),
    FeatureSpec("use_faser", _h_faser),
    FeatureSpec("use_kairos", _h_kairos),
    FeatureSpec("use_unified_radix", _h_unified_radix),
    # MoE
    FeatureSpec("use_elbow_moe", _h_elbow_moe),
    FeatureSpec("use_alloc_moe", _h_alloc_moe),
    FeatureSpec("use_lda_moe", _h_lda_moe),
    # Adaptive quantization
    FeatureSpec("use_adamx", _h_adamx),
    FeatureSpec("use_sharq", _h_sharq),
    FeatureSpec("use_mosaic_quant", _h_mosaic_quant),
    FeatureSpec("use_quamba2", _h_quamba2),
    # Final attention
    FeatureSpec("use_aoh", _h_aoh),
    # SSM / speculative
    FeatureSpec("use_replay_ssm", _h_replay_ssm),
]
