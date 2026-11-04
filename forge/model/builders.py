"""Attention/FFN factory functions and feature wrappers."""
import logging

import torch
import torch.nn as nn

from forge.config import ModelConfig

logger = logging.getLogger(__name__)

from .layers import GroupedQueryAttention, RotaryEmbedding, SwiGLUFFN


def build_attention(config: ModelConfig, layer_idx: int = 0) -> nn.Module:
    # LFM3 per-layer-type wiring: layer_types[layer_idx] tells us whether
    # this attention layer is sliding-window ("attention_swa",
    # "sliding_attention", "swa") and rope_layer_params supplies the
    # per-type theta/partial_rotary_factor (HF rope_parameters).
    ltype = "attention"
    layer_types = getattr(config, 'layer_types', None)
    if layer_types is not None and layer_idx < len(layer_types):
        ltype = layer_types[layer_idx]
    is_swa = ltype in ("attention_swa", "sliding_attention", "swa", "attn_swa")

    rope_base = config.rope_base
    rope_partial = 1.0
    rope_partial_mode = "zero_tail"
    rope_layer_params = getattr(config, 'rope_layer_params', None)
    if rope_layer_params:
        lrp = rope_layer_params.get(ltype)
        if lrp is None and is_swa:
            # fall back to any *swa*-keyed entry, then the "attention" entry
            lrp = next((v for k, v in rope_layer_params.items()
                        if 'swa' in k or 'sliding' in k), None)
        if lrp is None:
            lrp = rope_layer_params.get("attention")
        if lrp:
            rope_base = lrp.get("rope_theta", lrp.get("rope_base", rope_base))
            rope_partial = lrp.get("partial_rotary_factor", 1.0)
            rope_partial_mode = lrp.get("partial_rotary_mode", "zero_tail")

    sliding_window = getattr(config, 'sliding_window', None) if is_swa else None
    use_out_gate = getattr(config, 'use_attn_out_gate', False)

    kwargs = {"d_model": config.d_model, "n_heads": config.n_heads, "max_seq_len": config.max_seq_len, "base": rope_base, "rope_scaling": config.rope_scaling,
                  "use_qk_norm": getattr(config, 'use_qk_norm', False), "attn_scale": getattr(config, 'attn_scale', None),
                  "qk_norm_eps": getattr(config, 'norm_eps', 1e-6),
                  "use_rope": getattr(config, 'use_rope', True)}
    # LeRoPE/AdaRoPE: learnable RoPE frequencies (identity init = lossless).
    # Applied post-construction by replacing the RotaryEmbedding module.
    getattr(config, 'rope_variant', 'standard')
    if config.attn_type == "gqa":
        attn = GroupedQueryAttention(**kwargs, n_kv_heads=getattr(config, 'n_kv_heads', None),
                                     attn_bias=getattr(config, 'attn_bias', False),
                                     head_dim=getattr(config, 'head_dim', None),
                                     sliding_window=sliding_window,
                                     use_out_gate=use_out_gate,
                                     rope_partial_factor=rope_partial,
                                     rope_partial_mode=rope_partial_mode,
                                     zero_centered_norm=getattr(
                                         config, 'zero_centered_norm',
                                         False))
        attn = _maybe_bitnet_attention(config, attn)
        attn = _maybe_apply_lerope(config, attn)
        attn = _maybe_apply_learned_sink(config, attn)
        attn = _maybe_apply_csa(config, attn)
        attn = _maybe_apply_outro(config, attn)
        return _maybe_fuse_qkv(config, attn)
    if config.attn_type == "diff":
        # Differential Attention (Diff-Transformer): dual-softmax subtraction.
        from forge.keys.attention.differential_attn_key import DifferentialAttention
        attn = DifferentialAttention(
            d_model=config.d_model, n_heads=config.n_heads,
            n_kv_heads=getattr(config, 'n_kv_heads', None),
            max_seq_len=config.max_seq_len, base=config.rope_base,
            rope_scaling=config.rope_scaling,
            use_qk_norm=getattr(config, 'use_qk_norm', False),
            attn_bias=getattr(config, 'attn_bias', False),
            n_layers=config.n_layers, layer_idx=0,
            lambda_init=getattr(config, 'diff_attn_lambda_init', None))
        return _maybe_bitnet_attention(config, attn)
    if config.attn_type == "gta":
        # Grouped-Tied Attention (arXiv 2505.21487): V=K at init (lossless),
        # halves KV cache bandwidth. Training unties V from K.
        from forge.keys.attention.gta_key import GroupedTiedAttention
        attn = GroupedTiedAttention(
            d_model=config.d_model, n_heads=config.n_heads,
            n_kv_heads=getattr(config, 'n_kv_heads', None),
            max_seq_len=config.max_seq_len, base=config.rope_base,
            rope_scaling=config.rope_scaling,
            use_qk_norm=getattr(config, 'use_qk_norm', False),
            attn_bias=getattr(config, 'attn_bias', False),
            n_layers=config.n_layers, layer_idx=0)
        attn = _maybe_bitnet_attention(config, attn)
        attn = _maybe_apply_lerope(config, attn)
        attn = _maybe_apply_learned_sink(config, attn)
        attn = _maybe_apply_csa(config, attn)
        return _maybe_fuse_qkv(config, attn)
    if config.attn_type == "gla":
        # Grouped Latent Attention (arXiv 2505.21487): latent-compressed KV,
        # identity warm start (lossless). Shifts decode to compute-bound.
        from forge.keys.attention.gla_key import GroupedLatentAttention
        latent = getattr(config, 'gla_latent_dim', 0)
        attn = GroupedLatentAttention(
            d_model=config.d_model, n_heads=config.n_heads,
            n_kv_heads=getattr(config, 'n_kv_heads', None),
            max_seq_len=config.max_seq_len, base=config.rope_base,
            rope_scaling=config.rope_scaling,
            use_qk_norm=getattr(config, 'use_qk_norm', False),
            attn_bias=getattr(config, 'attn_bias', False),
            latent_dim=latent if latent > 0 else None,
            n_layers=config.n_layers, layer_idx=0)
        return _maybe_bitnet_attention(config, attn)
    raise ValueError(
        f"Unknown attention type: '{config.attn_type}'. "
        f"Valid options: 'gqa', 'diff', 'gta', 'gla'")


def _maybe_apply_lerope(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Replace RotaryEmbedding with LeRoPE/AdaRoPE (identity init = lossless).

    LeRoPE: learnable per-frequency-band scaling (dim//2 params, init=1.0).
    AdaRoPE: per-head learnable frequencies + attention scaling.
    Both init to standard RoPE → byte-identical at start.
    """
    rope_variant = getattr(config, 'rope_variant', 'standard')
    if rope_variant in ("lerope", "adarope"):
        from forge.keys.position.lerope_key import AdaRoPEEmbedding, LeRoPEEmbedding
        if hasattr(attn, 'rope') and isinstance(attn.rope, RotaryEmbedding):
            head_dim = attn.rope.dim
            max_seq = attn.rope.max_seq_len if hasattr(attn.rope, 'max_seq_len') else config.max_seq_len
            if rope_variant == "lerope":
                attn.rope = LeRoPEEmbedding(
                    dim=head_dim, max_seq_len=max_seq, base=config.rope_base,
                    rope_scaling=config.rope_scaling)
            else:  # adarope
                attn.rope = AdaRoPEEmbedding(
                    dim=head_dim, n_heads=config.n_heads,
                    max_seq_len=max_seq, base=config.rope_base,
                    rope_scaling=config.rope_scaling)
    return attn


def _maybe_apply_learned_sink(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Add learned attention sink bias to attention layers (GPT-OSS style).

    init=0 → lossless (no bias added). Training learns sink values.
    """
    if not getattr(config, 'use_learned_sink', False):
        return attn
    if hasattr(attn, 'n_heads'):
        n_heads = attn.n_heads
        init_val = getattr(config, 'learned_sink_init', 0.0)
        init_method = getattr(config, 'learned_sink_init_method', 'zero')
        if init_method == "zero":
            sinks = torch.zeros(n_heads)
        elif init_method == "constant":
            sinks = torch.full((n_heads,), init_val, dtype=torch.float32)
        elif init_method == "random":
            sinks = torch.randn(n_heads) * 0.1 + init_val
        else:
            sinks = torch.zeros(n_heads)
        attn.sinks = nn.Parameter(sinks)
    return attn


def _maybe_apply_csa(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Enable Compressed Sparse Attention (CSA) for long-context efficiency.

    CSA selects the top-k most relevant key positions per query, reducing
    attention complexity from O(S^2) to O(S*k). When the sequence is shorter
    than top_k, full attention is used (lossless for short sequences).

    Applied as a flag on the attention module — the forward checks the flag
    and uses CSA's top-k selection for long sequences.
    """
    pattern = getattr(config, 'attention_pattern', 'standard')
    if pattern not in ('csa', 'csa_hca_hybrid'):
        return attn
    top_k = getattr(config, 'csa_top_k', 256)
    attn._csa_top_k = top_k
    attn._csa_enabled = True
    return attn


def _maybe_apply_outro(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Enable OutRo (Outgoing-Rotary) sink-enhanced attention (V12 key).

    Stores the OutRo configuration on the attention module so the forward
    pass can apply non-causal masks for sink positions. Lossless at warm
    start: align_strength=0.0 → identity alignment, no modification.
    """
    if not getattr(config, 'use_outro', False):
        return attn
    from forge.keys.attention.outro_key import OutRoKey
    sink_threshold = getattr(config, 'outro_sink_threshold', 0.5)
    attn._outro_key = OutRoKey(sink_threshold=sink_threshold, align_strength=0.0)
    return attn


def _maybe_bitnet_attention(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Swap attention projections for BitNet b1.58 QAT linears (when enabled).

    Eval stays full-precision (ternary only in training) so the lossless
    checkpoint load is preserved; QAT then quantizes q/k/v/o matmuls too.
    """
    if not getattr(config, 'use_bitnet', False):
        return attn
    from forge.keys.quantization.bitnet_b158_key import build_bitnet_linear
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        lin = getattr(attn, name, None)
        if lin is None:
            continue
        setattr(attn, name, build_bitnet_linear(
            config, config.d_model, lin.out_features,
            bias=lin.bias is not None))
    return attn


def _maybe_fuse_qkv(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Fuse separate Q/K/V projections into a single GEMM (when enabled).

    Replaces q_proj, k_proj, v_proj with a single FusedQKVLinear that does
    all three projections in one matmul. Halves kernel launches per attention
    layer. The fused weights are initialized from the separate projections
    (lossless — same math, just one GEMM instead of three).

    Skipped for GLA (uses kv_down_proj instead of separate k/v) and
    DifferentialAttention (doubled q/k dims make fusion less beneficial).
    """
    if not getattr(config, 'use_fused_gemm', False):
        return attn
    # Only fuse for GQA and GTA (standard q/k/v projections)
    if type(attn).__name__ not in ("GroupedQueryAttention", "GroupedTiedAttention"):
        return attn
    from forge.keys.quantization.fused_gemm_key import FusedQKVLinear, fuse_qkv_weights
    q_proj = getattr(attn, 'q_proj', None)
    k_proj = getattr(attn, 'k_proj', None)
    v_proj = getattr(attn, 'v_proj', None)
    if q_proj is None or k_proj is None or v_proj is None:
        return attn
    # Don't fuse if already BitNet (BitNet handles its own kernel)
    if type(q_proj).__name__ == "BitNetLinear":
        return attn
    g_proj = getattr(attn, 'gate_proj', None)
    g_dim = g_proj.out_features if g_proj is not None else 0
    fused_w, fused_b = fuse_qkv_weights(
        q_proj.weight, k_proj.weight, v_proj.weight,
        q_proj.bias, k_proj.bias, v_proj.bias)
    if g_dim:
        fused_w = torch.cat([fused_w, g_proj.weight], dim=0)
        if fused_b is not None:
            fused_b = torch.cat([fused_b, g_proj.bias or
                                 torch.zeros(g_dim, device=fused_b.device,
                                             dtype=fused_b.dtype)])
    # fork_rng: the fused module's kaiming init would otherwise consume
    # global RNG state and shift every later layer's random init vs the
    # unfused build — the values are overwritten immediately anyway.
    with torch.random.fork_rng():
        fused = FusedQKVLinear(
            config.d_model, q_proj.out_features, k_proj.out_features,
            v_proj.out_features, bias=fused_b is not None, g_dim=g_dim)
    with torch.no_grad():
        fused.weight.copy_(fused_w)
        if fused_b is not None:
            fused.bias.copy_(fused_b)
    attn.qkv_proj = fused
    # Keep original projections — checkpoints load into q/k/v/gate_proj
    # keys, and loader._refresh_fused_qkv re-packs the fused matrix
    # AFTER weights land (the build-time copy is only init garbage).
    attn._fused_qkv = True
    return attn


def refresh_fused_qkv(attn: nn.Module) -> None:
    """Re-pack attn.qkv_proj from the loaded q/k/v/gate weights.

    Call after checkpoint weights land — the fused matrix built at
    construction only ever held init values.
    """
    fused = getattr(attn, 'qkv_proj', None)
    if fused is None or not getattr(attn, '_fused_qkv', False):
        return
    parts = [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]
    if fused.g_dim:
        parts.append(attn.gate_proj.weight)
    with torch.no_grad():
        fused.weight.copy_(torch.cat(parts, dim=0))
        if fused.bias is not None:
            bs = [attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias]
            if fused.g_dim:
                bs.append(attn.gate_proj.bias if attn.gate_proj.bias
                          is not None else torch.zeros(
                              fused.g_dim, device=fused.bias.device,
                              dtype=fused.bias.dtype))
            fused.bias.copy_(torch.cat(bs, dim=0))


def refresh_fused_gate_up(ffn: nn.Module) -> None:
    """Re-pack ffn.gate_up_proj from the loaded w_gate/w_up weights."""
    fused = getattr(ffn, 'gate_up_proj', None)
    if fused is None or not getattr(ffn, '_fused_gate_up', False):
        return
    with torch.no_grad():
        fused.weight.copy_(
            torch.cat([ffn.w_gate.weight, ffn.w_up.weight], dim=0))


def _build_moe_ffn(config: ModelConfig) -> nn.Module:
    """Build a MoE FFN layer with BitNet experts and shared expert.

    Uses the existing MoELayer from forge.moe, then swaps expert linears
    for BitNetLinear when use_bitnet=True. The shared expert is also BitNet.

    With dense_bypass=True (default for V5), the router is skipped at init
    and all experts run with equal weight → exact dense FFN output.
    Training gradually enables routing (disable dense_bypass after warmup).
    """
    from forge.moe.moe import MoELayer

    d_model = config.d_model
    d_ff = getattr(config, 'moe_d_ff', None) or getattr(config, 'intermediate_size', None) or d_model * 2
    n_experts = getattr(config, 'moe_n_experts', 8)
    top_k = getattr(config, 'moe_top_k', 2)
    shared = getattr(config, 'moe_shared_expert', True)
    dense_bypass = getattr(config, 'moe_dense_bypass', True)
    noisy = getattr(config, 'moe_noisy_gating', True)
    lb_weight = getattr(config, 'moe_load_balance_weight', 0.01)
    router_mode = getattr(config, 'moe_router_mode', 'switch')

    moe = MoELayer(
        d_model, n_experts=n_experts, top_k=top_k, d_ff=d_ff,
        shared_expert=shared, capacity_factor=None,
        noisy_gating=noisy, dense_bypass=dense_bypass,
        use_clamp=getattr(config, 'use_swiglu_clamp', False),
        clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
        clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0),
        router_mode=router_mode)

    # Override load balance weight
    moe.router.load_balance_loss_weight = lb_weight

    # Apply BitNet to all expert linears + shared expert
    if getattr(config, 'use_bitnet', False):
        from forge.keys.quantization.bitnet_b158_key import build_bitnet_linear
        for expert in moe.experts:
            expert.w1 = build_bitnet_linear(config, d_model, d_ff)
            expert.w3 = build_bitnet_linear(config, d_model, d_ff)
            expert.w2 = build_bitnet_linear(config, d_ff, d_model)
        if hasattr(moe, 'shared'):
            moe.shared.w1 = build_bitnet_linear(config, d_model, d_ff)
            moe.shared.w3 = build_bitnet_linear(config, d_model, d_ff)
            moe.shared.w2 = build_bitnet_linear(config, d_ff, d_model)

    return moe


def build_ffn(config: ModelConfig) -> nn.Module:
    # V5: MoE FFN — replace dense FFN with shared expert + routed experts.
    if getattr(config, 'use_moe', False):
        return _build_moe_ffn(config)
    if config.ffn_type == "geglu":
        # Spark-X2.6 GeGLU: gelu(gate_proj(x)) * up_proj(x) → down_proj.
        # Same three-matrix layout as SwiGLU — only the gate activation differs.
        return SwiGLUFFN(
            config.d_model,
            hidden_dim=getattr(config, 'intermediate_size', None),
            use_triton=getattr(config, 'use_triton_kernels', False),
            act_mode="gelu")
    if config.ffn_type == "swiglu":
        # V5.2: FFN compression (Monarch/Kronecker/TT) — replaces dense linear
        # layers with factored versions. Conversion from dense checkpoint
        # happens in build_model_fast() via _convert_ffn_compression().
        ffn_compression = getattr(config, 'ffn_compression', 'none')
        if ffn_compression == 'monarch':
            from forge.keys.compression.monarch_ffn_key import MonarchSwiGLUFFN
            ffn = MonarchSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                block_size=getattr(config, 'monarch_block_size', 32),
                use_clamp=getattr(config, 'use_swiglu_clamp', False),
                clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
                clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0))
            return ffn
        elif ffn_compression == 'kron':
            from forge.keys.compression.kron_ffn_key import KroneckerSwiGLUFFN
            # kron_a*kron_b should = intermediate, kron_c*kron_d should = d_model
            # Use config values as the (a, b) split for gate/up output factorization
            gate_kron = (getattr(config, 'kron_a', 64),
                         getattr(config, 'kron_b', 32))
            down_kron = (getattr(config, 'kron_c', 32),
                         getattr(config, 'kron_d', 256))
            ffn = KroneckerSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                gate_kron=gate_kron,
                down_kron=down_kron,
                use_clamp=getattr(config, 'use_swiglu_clamp', False),
                clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
                clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0))
            return ffn
        elif ffn_compression == 'tt':
            from forge.keys.compression.tt_ffn_key import TTSwiGLUFFN
            ffn = TTSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                tt_rank=getattr(config, 'tt_rank', 4))
            return ffn
        elif ffn_compression == 'nlrq':
            from forge.keys.compression.nlrq_ffn_key import NLRQSwiGLUFFN
            ffn = NLRQSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                rank=getattr(config, 'nlrq_rank', 256),
                factor_bits=getattr(config, 'nlrq_factor_bits', 8),
                use_residual=getattr(config, 'nlrq_use_residual', False),
                residual_group_size=getattr(config, 'nlrq_residual_group_size', 128),
                use_hadamard=getattr(config, 'nlrq_use_hadamard', False),
                use_clamp=getattr(config, 'use_swiglu_clamp', False),
                clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
                clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0))
            # HashedNLRQ — replace NLRQ low-rank factors with hashed
            # shared vectors for an additional 4-8x compression on top of
            # NLRQ's 12.8x (total ~50x).  Not lossless (different weight
            # structure), so only used when use_hashed_nlrq=True.
            if getattr(config, 'use_hashed_nlrq', False):
                from forge.training.optim.r21_cross_domain import HashedNLRQ
                _rank = getattr(config, 'nlrq_rank', 256)
                _hcomp = getattr(config, 'hashed_nlrq_compression', 8.0)
                ffn.w_gate = HashedNLRQ(
                    ffn.w_gate.out_features, ffn.w_gate.in_features,
                    rank=_rank, hash_compression=_hcomp)
                ffn.w_up = HashedNLRQ(
                    ffn.w_up.out_features, ffn.w_up.in_features,
                    rank=_rank, hash_compression=_hcomp)
                ffn.w_down = HashedNLRQ(
                    ffn.w_down.out_features, ffn.w_down.in_features,
                    rank=_rank, hash_compression=_hcomp)
            return ffn
        # Smooth-SwiGLU: per-channel RMSNorm on gate output for FP8 stability.
        # When use_smooth_swiglu=True, uses SmoothSwiGLUFFN (bounds SiLU outliers).
        # Otherwise standard SwiGLUFFN.
        if getattr(config, 'use_smooth_swiglu', False):
            from forge.training.optim.fp8_training import SmoothSwiGLUFFN
            ffn = SmoothSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None))
        else:
            ffn = SwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                use_clamp=getattr(config, 'use_swiglu_clamp', False),
                clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
                clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0),
                use_triton=getattr(config, 'use_triton_kernels', False))
        if getattr(config, 'use_bitnet', False):
            # BitNet b1.58 QAT: swap linear layers for ternary-STE versions
            # (learned per-layer scales; ternary only in training by default).
            from forge.keys.quantization.bitnet_b158_key import build_bitnet_linear
            hidden = ffn.w_gate.out_features
            ffn.w_gate = build_bitnet_linear(config, config.d_model, hidden)
            ffn.w_up = build_bitnet_linear(config, config.d_model, hidden)
            ffn.w_down = build_bitnet_linear(config, hidden, config.d_model)
        elif getattr(config, 'use_fused_gemm', False):
            # Fused Gate-Up GEMM: single matmul for w_gate + w_up.
            from forge.keys.quantization.fused_gemm_key import FusedGateUpLinear, fuse_gateup_weights
            hidden = ffn.w_gate.out_features
            fused_w = fuse_gateup_weights(ffn.w_gate.weight, ffn.w_up.weight)
            # fork_rng: the ctor's kaiming init must not perturb the global
            # stream (values are overwritten immediately anyway).
            with torch.random.fork_rng():
                fused = FusedGateUpLinear(config.d_model, hidden, bias=False)
            with torch.no_grad():
                fused.weight.copy_(fused_w)
            ffn.gate_up_proj = fused
            ffn._fused_gate_up = True
        return ffn
    raise ValueError(
        f"Unknown FFN type: '{config.ffn_type}'. "
        f"Valid options: 'swiglu'")


