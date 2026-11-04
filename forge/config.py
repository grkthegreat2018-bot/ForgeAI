"""Config-driven model specifications for ForgeAI architecture research."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """All hyperparameters needed to build a ConfigurableResearchLLM."""

    # === Architecture ===
    vocab_size: int = 65536
    d_model: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    max_seq_len: int = 32768
    attn_type: str = "gqa"            # "gqa" (only supported type now)
    ffn_type: str = "swiglu"          # "swiglu" (only supported type now)
    norm_type: str = "rmsnorm"        # "rmsnorm" (only supported type now)
    n_kv_heads: int | None = None     # KV heads for GQA (None = MHA)
    # Explicit per-head width; None derives d_model // n_heads.
    # Needed for archs that decouple the two (e.g. Qwen3-4B: 2560/32 vs
    # head_dim=128).
    head_dim: int | None = None
    intermediate_size: int | None = None  # FFN hidden dim (None = 8*d/3)
    attn_bias: bool = False
    norm_eps: float = 1e-6  # RMSNorm epsilon (LFM2.5 uses 1e-5)
    use_embed_norm: bool = False  # Norm after embedding (LFM2.5 has this, standard models don't)
    use_final_norm: bool = True  # Final norm before head (LFM2.5 doesn't have this)
    use_rope: bool = True  # Apply RoPE to attention Q/K. Jamba has no RoPE.
    rope_base: float = 1_000_000.0  # LFM2.5 base: 1M (evolution 10M was synthetic-only, reverted)
    # YaRN RoPE scaling for context extension. None = no scaling.
    rope_scaling: dict | None = None
    # RoPE variant: "standard", "lerope", or "adarope" (both learnable variants
    # init to identity = standard RoPE, lossless at start).
    rope_variant: str = "standard"
    # Share one RotaryEmbedding table set across layers with identical
    # rope params (dim, theta, scaling, partial factor). Buffer values are
    # byte-identical by construction, so sharing is function-exact; it
    # makes 262K-table models affordable (11 per-layer copies -> 2).
    share_rope: bool = False
    # Zero-centered RMSNorm reparam (Qwen3-Next): norm weights are stored
    # as gamma-1 and the forward adds 1 back — identical function, but
    # weight decay pulls toward identity scale instead of zero gain.
    # Checkpoints written at gamma are ported by subtracting 1 at load.
    zero_centered_norm: bool = False

    # === Weight sharing ===
    # Tie input embeddings to output head (default True for most small models).
    # Set False for models with separate embed/head (Qwen2.5, some larger models).
    tie_word_embeddings: bool = True

    # === Hybrid layer types ===
    # Per-layer type list. None = all attention (default).
    # Each entry: "attention" (or "attn"), "conv", "mamba", or "moe".
    layer_types: list[str] | None = None
    conv_kernel_size: int = 3
    # Mamba/SSM layer config (used when layer_types contains "mamba")
    mamba_d_state: int = 16        # SSM state expansion factor (Mamba1: 16, Mamba2: 64)
    mamba_d_conv: int = 4          # Local convolution width
    mamba_expand: int = 2          # Block expansion factor (d_inner = expand * d_model)
    mamba_dt_rank: str | int = "auto"  # "auto" = ceil(d_model/16), or explicit int
    mamba_bias: bool = False       # Bias on in_proj/out_proj
    mamba_conv_bias: bool = True   # Bias on conv1d
    # SSM type: "mamba2" (default, backward compatible) or "mamba3"
    # (complex-valued states, MIMO, exponential-trapezoidal discretization).
    # When "mamba3", layers with type "mamba"/"ssm" use Mamba3Block.
    ssm_type: str = "mamba2"
    # Mamba-3 MIMO config (n_inputs -> n_outputs per SSM unit).
    # Default 1/1 = scalar SSM (matches Mamba-2, lossless warm start).
    mamba3_n_inputs: int = 1
    mamba3_n_outputs: int = 1

    # === Our keys (all zero/identity init = lossless at start) ===
    # QK-norm: RMSNorm on Q and K before RoPE (LFM2/Gemma3/Qwen3 style).
    use_qk_norm: bool = False
    # Zero-init output projections: attn.out_proj and ffn.w_down start at zero.
    zero_init_residual: bool = False
    # PIT (Pseudo-Inverse Tying): replaces weight tying with orthonormal shared
    # memory + learned SPD transform. Identity init = lossless.
    use_pit: bool = False
    # AttnRes (Attention Residuals, Kimi K3): cross-layer retrieval, gated.
    use_attn_residual: bool = False
    attn_res_k: int = 4
    # mHC (Manifold-Constrained Hyper-Connections, DeepSeek-V4): low-rank
    # residual projection. Gate init=0 = lossless.
    use_mhc: bool = False
    mhc_rank: int = 0  # 0 = auto (d_model // 4)
    # CSA (Compressed Sparse Attention, DeepSeek-V4): top-k position selection.
    attention_pattern: str = "standard"  # "standard", "csa", "csa_hca_hybrid"
    csa_top_k: int = 256

    # === LFM3 port keys ===
    # Per-head sigmoid output gate on attention (LFM2/3 "gated attention"):
    #   out = out_proj(attn_out * sigmoid(gate_proj(x))) with one scalar
    #   gate per query head. Requires the ported gate_proj weights.
    use_attn_out_gate: bool = False
    # Sliding-window attention layers (layer_types entry "attention_swa" /
    # "sliding_attention" / "swa"): each token attends to itself + the
    # previous sliding_window-1 keys (HF LFM3/Gemma3 semantics).
    sliding_window: int | None = None
    # D2 strict-lossless: cap SWA layer KV buffers at sliding_window and
    # store position p at slot p % W (ring). Output-identical — the window
    # mask never reads older keys — but decode length is no longer bounded
    # by SWA-layer VRAM and long-context KV shrinks on those layers.
    swa_ring_kv: bool = False
    # Per-layer-type RoPE overrides, keyed by normalized layer type
    # ("attention", "attention_swa"): {"rope_theta": float,
    # "partial_rotary_factor": float}. Mirrors HF rope_parameters.
    # partial_rotary_factor < 1 uses the LFM3 zero-tail layout: the first
    # factor*head_dim//2 frequencies keep the FULL-head-dim progression and
    # the tail gets freq 0 (unrotated) — NOT a shrunk rotary dimension.
    rope_layer_params: dict | None = None
    # SandwichNorm placement: "post_residual" (existing — post_*_norm applied
    # to the stream AFTER the residual add) vs "pre_residual" (LFM2/3 —
    # post_*_norm applied to the operator/FFN OUTPUT before the add:
    #   x = x0 + post_op_norm(op(x0));  x = x + post_ffn_norm(ffn(x))).
    sandwich_norm_mode: str = "post_residual"
    # MTP (Multi-Token Prediction): shared-weight heads for speculative decode.
    use_mtp: bool = False
    mtp_n_heads: int = 4  # evolution: 4 heads (was 2), score 27.69 vs 23.6
    mtp_loss_weight: float = 0.495  # evolution: 0.495 (was 0.3), near-0.5 optimum
    # mtp_weight: CLI/train-runner alias for mtp_loss_weight. When set (not
    # None), it overrides mtp_loss_weight at MTPModule construction time.
    # mtp_weight=0.0 disables the MTP loss contribution (params become dead,
    # frozen by freeze_dead_params_ before BAdam partitioning).
    mtp_weight: float | None = None
    # V14: DeepSeek-MTP style adapter heads — each head is a small d→d
    # projection sharing the tied vocab head (~n×d² vs n×d×V params; the
    # vocab weight is read once per draft step). Requires head tying.
    mtp_adapter_heads: bool = False
    # BitNet b1.58: ternary QAT on all linear layers ({-1,0,1} weights, STE).
    # Training = ternary forward (STE, master weights fp); eval = full
    # precision unless bitnet_force_quant (deploy after QAT converges).
    use_bitnet: bool = False
    bitnet_force_quant: bool = False
    bitnet_learned_scale: bool = True  # per-layer learnable scale (QAT)
    # Differential Attention (Diff-Transformer): dual softmax maps subtracted
    # to cancel attention noise. Parameter-efficient; opt-in (not lossless).
    use_diff_attn: bool = False
    diff_attn_lambda_init: float | None = None  # None = paper formula
    # TITAN neural memory: gated long-term memory, zero-init gate = lossless.
    use_titan_memory: bool = False
    titan_memory_rank: int = 0  # 0 = d_model
    # v2 inner-loop knobs (LiveT): surprise-gated fast-weight updates.
    titan_update_lr: float = 0.5      # inner-loop lr η on fast weights
    titan_forget: float = 0.99        # per-update decay γ
    titan_momentum: float = 0.9       # inner-loop momentum μ
    titan_surprise_cap: float = 4.0   # clip on normalized surprise weight
    titan_solver: str = "gd"          # "gd" | "cg" | "solve" (ridge solve)
    titan_ridge: float = 1.0          # λ prior-strength for cg/solve
    titan_cg_iters: int = 8           # conjugate-gradient iters per chunk
    # Mixture-of-Depths: per-block token routing; keep_fraction=1.0 = lossless
    # (all tokens processed). <1.0 skips tokens in training (FLOPs ceiling).
    use_mod: bool = False
    mod_keep_fraction: float = 1.0
    mod_aux_loss_weight: float = 1e-8  # near-zero aux loss (prevents router collapse)
    mod_n_skip_layers: int = 0  # 0 = no skipping (safe default; >0 skips FFN in selected layers)
    # FFN-SkipLLM: skip FFN blocks during inference when input norm is small
    # (saturated layers contribute little). 0.0 = skip none, 0.3 = skip ~30%.
    # Only active during eval (not training). Based on EMNLP 2024 paper.
    # NOTE: Calibration on ForgeLM V3 (1.2B, 16 layers) shows NO FFN saturation
    # — cosine similarities are all low/negative (FFN actively transforms in all
    # layers). This technique requires 32+ layer models to have a non-cold region.
    # Kept in config for future larger models. See docs/FFN_RESEARCH.md.
    ffn_skip_threshold: float = 0.0  # 0.0 = disabled

    # === V4: Hardware-efficient inference keys (2026-08) ===
    # GLA (Grouped Latent Attention, arXiv 2505.21487): latent-compressed KV,
    # shifts decode from BW-bound to compute-bound. Identity warm start (lossless).
    # attn_type="gla" activates this; gla_latent_dim sets compression (0=full=lossless).
    gla_latent_dim: int = 0  # 0 = full KV dim (lossless), >0 = compressed latent
    # GTA (Grouped-Tied Attention, arXiv 2505.21487): ties V to K, halves KV cache.
    # attn_type="gta" activates this. V=K at init (lossless), training unties.
    # Fused QKV + Gate-Up GEMM: single GEMM for Q/K/V and for gate/up projections.
    # Halves kernel launches. Lossless (same math, just fused).
    use_fused_gemm: bool = False
    # BitNet int8 trainable storage (R&D round 15): stores ternary weights as
    # int8 on GPU (1 byte/param) with bf16 master weights on CPU. STE gradients
    # flow GPU→CPU, BAdam updates CPU master, re-quantizes active block to int8.
    # Enables 8B param training on 12GB VRAM (8GB weights + 2.5GB optim + 1GB act).
    bitnet_int8_training: bool = False
    # W8A8 INT8 quantization: weight + activation INT8 with tensor-core GEMM.
    # 2-3x decode speedup at batch=1. Applied at inference (not training).
    use_w8a8: bool = False
    w8a8_mode: str = "int8"  # "int8" or "fp8"
    # NVFP4 quantization (R&D 14): FP4 E2M1 weights with FP8 block scales.
    # Native on Blackwell SM120 (RTX 5070). 3.8x compression, ~99% quality.
    # Auto-selected as default quant on Blackwell by forge_engine.
    use_nvfp4: bool = False
    nvfp4_block_size: int = 32  # NVFP4 standard: 32 elements per FP8 scale
    nvfp4_w4a8: bool = False  # W4A8 mode: FP4 weights + FP8 activations

    # === FP8 Training (Blackwell/Hopper) ===
    # Smooth-SwiGLU: per-channel RMSNorm on FFN gate output to prevent FP8
    # overflow from SiLU outliers. Required for stable FP8 training.
    use_smooth_swiglu: bool = False
    # μScaling: unit-variance weight init (std=1/sqrt(d_in)) keeps all tensors
    # within FP8 range without dynamic scaling overhead. Applied at init only.
    use_mu_scaling: bool = False
    # FP8 autocast: enables torch.autocast with FP8 (E4M3) for forward pass.
    # Requires SM90+ (Hopper/Blackwell). Falls back to BF16 on older GPUs.
    use_fp8_training: bool = False

    # === V5: MoE + Factorized Embedding + Full BitNet (2026-09) ===
    # Mixture-of-Experts FFN: replace dense FFN with shared expert + routed
    # experts. Active params stay ~same; total params scale with n_experts.
    # AirMoE hotloads routed experts from disk at inference (only top-k in VRAM).
    use_moe: bool = False
    moe_n_experts: int = 8
    moe_top_k: int = 2
    moe_shared_expert: bool = True   # always-active shared expert (DeepSeek-V3 style)
    moe_dense_bypass: bool = True    # skip router at init (lossless = dense FFN)
    # Disable dense_bypass after this many training steps so the router
    # activates (experts start specializing). 0 = never disable (stays dense).
    # DeepSeek-V3 uses a warmup where the router gradually takes over.
    moe_dense_bypass_warmup_steps: int = 0
    moe_noisy_gating: bool = True    # add noise during training for exploration
    moe_load_balance_weight: float = 0.01
    # Router mode: "switch" (Switch-Transformer aux loss, backward compat) or
    # "aux_free" (DeepSeek-V3 auxiliary-loss-free load balancing with per-expert
    # bias + sequence-wise balance loss from full softmax). "aux_free" is
    # lossless at init (bias=0).
    moe_router_mode: str = "aux_free"
    moe_d_ff: int | None = None      # expert hidden dim (None = intermediate_size)
    # Expert Tying (arXiv 2606.16825): share expert weights across consecutive
    # layer groups. g=2: layers (0,1) share, (2,3) share, etc. 2x expert param
    # reduction, near-lossless. Applied as pointer aliasing after model build.
    moe_expert_tying: bool = False
    moe_tie_group_size: int = 2      # consecutive layers that share experts
    # Factorized Embedding (ALBERT pattern): decompose vocab×d_model into
    # vocab×rank × rank×d_model. 7.8x param reduction for 65536×2048.
    # Init via SVD of original embedding (lossless at start).
    use_factorized_embeddings: bool = False
    embed_factorized_rank: int = 256  # factorization rank (<< d_model)
    embed_tie_factor: float = 0.75  # evolution: high tying (was implicit 0.5)
    # BitNet on embeddings: ternary quantize embedding weights (QAT).
    # Saves ~8x on embedding VRAM. Combined with factorized = ~60x reduction.
    use_bitnet_embedding: bool = False

    # === V5.1: Additional architecture keys (2026-08) ===
    # All lossless at init (zero/identity), activated during training.

    # ValueResidual (ResFormer, arXiv 2410.17897): add V_0 residual to all
    # layers. Gate=0 at init → lossless. Training opens the gate.
    # mode: "resformer" (additive V_i + gate*V_0) or "svformer" (shared V_0).
    use_value_residual: bool = False
    value_residual_mode: str = "resformer"  # "resformer" or "svformer"
    value_residual_gate_init: float = 0.0  # 0 = lossless at start

    # SandwichNorm: post-sublayer RMSNorm (identity init = lossless).
    # Stabilizes MoE training by bounding post-FFN/post-attn activations.
    use_sandwich_norm: bool = False

    # LearnedSink (GPT-OSS): per-head attention sink bias logits.
    # init_method="zero" = lossless (no bias added). "constant" adds a
    # small positive bias to stabilize attention.
    use_learned_sink: bool = False
    learned_sink_init: float = 0.0  # 0 = lossless; 1.0 = GPT-OSS default
    learned_sink_init_method: str = "zero"  # "zero", "constant", "random"

    # SwiGLU Clamp (GPT-OSS): clamped SwiGLU activation with scaled sigmoid.
    # Prevents outlier activations from destabilizing quantized inference.
    # This is a runtime-only change (no weights), but changes the activation
    # function — NOT lossless in the strict sense, but close at init since
    # clamping only affects outliers (|x| > limit=7.0).
    use_swiglu_clamp: bool = False
    swiglu_clamp_alpha: float = 1.702  # sigmoid scale (approximates GELU)
    swiglu_clamp_limit: float = 7.0  # clamp value (prevents outliers)

    # === V5.2: Parameter-reduction keys (2026-08) ===
    # All decomposition-based keys convert dense weights on first checkpoint
    # load, then save the compressed checkpoint (no re-conversion on subsequent
    # loads). See .devin/scratchpad.md for full analysis.

    # Monarch FFN (Dao et al. 2022): W ≈ L @ R @ P with L,R block-diagonal.
    # 70% FFN param reduction, near-lossless (matches dense expressivity).
    # ffn_compression="monarch" activates this.
    # block_size: size of block-diagonal blocks (must divide d_model and intermediate).
    ffn_compression: str = "none"  # "none", "monarch", "kron", "tt", "nlrq"
    monarch_block_size: int = 32
    # NLRQ FFN (R&D winner): SVD + quantized factors + optional INT4 residual.
    # 12.8x param reduction (rank=256, 8-bit), 3x with INT4 residual at 0.15% error.
    nlrq_rank: int = 256             # SVD truncation rank
    nlrq_factor_bits: int = 8        # bits per U/V factor element
    nlrq_use_residual: bool = False  # add INT4 group-quantized residual
    nlrq_residual_group_size: int = 128
    nlrq_use_hadamard: bool = False       # HINT4: Hadamard rotation on INT4 factors
    use_peagle_tied: bool = False         # Tied PEAGLE draft head (7x param reduction)
    peagle_lora_rank: int = 32            # LoRA rank for tied PEAGLE heads
    # Kronecker FFN: W = A ⊗ B. 71% FFN param reduction, lossy-recoverable.
    # Factorization shapes for (d_model, intermediate): (a*b, c*d) = (d_model, intermediate).
    kron_a: int = 64   # A is (a, c)
    kron_b: int = 32   # B is (b, d)
    kron_c: int = 32
    kron_d: int = 256
    # Tensor-Train FFN: TT decomposition of weight matrix. 71% reduction, lossy-recoverable.
    tt_rank: int = 4   # TT rank (higher = less compression, less loss)

    # Hyperloop Transformers (arXiv 2604.21254): looped middle blocks.
    # begin_layers unique + middle shared (looped n_loop_iters times) + end_layers unique.
    # 44% layer param reduction, near-lossless (paper: BETTER quality).
    # Lossless at init: all layers unique, loop gate=0. Training opens gate.
    use_hyperloop: bool = False
    hyperloop_begin: int = 2    # unique begin layers
    hyperloop_end: int = 2      # unique end layers
    hyperloop_loop_iters: int = 3  # how many times to loop the shared middle block

    # LiSA (TACL 2026): cross-layer Q/K sharing with alignment FFN.
    # 6x Q/K compression, 19-40% throughput improvement.
    # Lossless at init: per-layer Q/K loaded from checkpoint, shared Q/K gate=0.
    # Training opens gate, then per-layer Q/K can be pruned.
    use_lisa: bool = False
    lisa_compress: int = 6     # Q/K compression factor across attention layers
    lisa_align_dim: int = 0    # alignment FFN hidden dim (0 = d_model // 4)

    # === Training ===
    dropout: float = 0.0
    device: str = "cpu"
    dtype: str = "bfloat16"
    checkpoint_dir: str = "research/checkpoints"
    data_dir: str = "research/data"
    batch_size: int = 2
    seq_len: int = 1024
    max_steps: int = 500
    warmup_steps: int = 200
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_every: int = 250
    save_every: int = 500
    eval_batches: int = 10

    # === Memory optimizations ===
    # Chunked cross-entropy: fuses head + CE without materializing full logits.
    use_chunked_ce: bool = False
    ce_chunk_size: int = 256
    # Entropy-weighted CE alpha (WeFT/VCORE 2025). When > 0, the chunked CE
    # path uses ChunkedEntropyWeightedCE instead of plain ChunkedLinearCrossEntropy.
    # 0 = disabled (plain CE), 0.5 = production default.
    entropy_alpha: float = 0.0
    # Liger-Kernel fused linear cross-entropy (requires liger-kernel).
    use_liger_ce: bool = False
    # === R&D round 14: training speedup features (2026-08-25) ===
    # Varlen attention: FlashAttention varlen path for packed sequences.
    # Eliminates cross-example attention contamination + padding-mask compute
    # waste. Requires cu_seqlens from PackedSequenceDataset (emit_cu_seqlens).
    # Community: Unsloth 2.1x faster padding-free, 50% less VRAM.
    use_varlen: bool = False
    # Triton fused training kernels: fused RMSNorm + SwiGLU as single Triton
    # kernels (Liger-Kernel-style). Reduces kernel launches and intermediate
    # materialization. Tuned for SM120 (RTX 5070, GDDR7 672 GB/s).
    # Community: Liger-Kernel 20% throughput, 60% VRAM; Unsloth 2-5x.
    use_triton_kernels: bool = False
    # Triton kernel block sizes (tuned for SM120 GDDR7 bandwidth).
    # RMSNorm: block = d_model (2048 for V10).
    # SwiGLU: block = intermediate_size (8192 for V10).
    triton_rms_block_size: int = 4096
    triton_swiglu_block_size: int = 16384
    # APOLLO optimizer: SVD-free random-projection gradient scaling.
    # SGD-like memory with AdamW-level performance. No SVD overhead vs GaLore.
    # Community: arXiv 2412.05183, ICLR 2025. APOLLO-Mini rank=1 = SGD memory.
    apollo_rank: int = 8  # auxiliary subspace rank (1 = APOLLO-Mini)
    apollo_scale: str = "tensor"  # "tensor", "channel"
    # BREAD: landscape correction for BAdam. Applies memory-efficient SGD
    # updates to inactive blocks during the same backward pass, preventing
    # optimization landscape narrowing. Microsoft BlockOptimizers.
    # Community: OpenReview zs6bRl05g8, accelerates BAdam convergence.
    bread_sgd_correction: str = "partial"  # "all", "partial", "disabled"
    bread_sgd_lr_scale: float = 5.0  # SGD lr = base_lr * this (typical 5x)
    # FlashOptim: companded 8-bit optimizer states (7 bytes/param vs 16).
    # Companding on momentum/variance with tight error bounds.
    # Community: arXiv 2602.23349, >50% per-param memory reduction.
    flashoptim_bits: int = 8  # 8 = companded uint8, 4 = companded uint4

    # === R&D Round 19: Qwen3.8-Flash-Next keys (2026-08-29) ===
    # QSA (Qwen Sparse Attention): top-k sparse position selection with
    # averaged attention scores. Reduces attention from O(n²) to O(n*k).
    # Lossless at init (k = seq_len = full attention).
    use_qsa: bool = False
    qsa_top_k: int = 256  # number of positions to keep (0 = full attention)
    # Gated Residual: per-layer learnable gate on the residual stream.
    # Gate init=1.0 = lossless (passes residual through unchanged).
    use_gated_residual: bool = False
    gated_residual_init: float = 1.0  # 1.0 = identity (lossless)
    # N-gram Embedding: host-side n-gram lookup table for knowledge
    # augmentation. 15.3 GB host RAM for 3-gram table (vocab=65536).
    use_ngram_embedding: bool = False
    ngram_n: int = 3  # n-gram order (2=bigram, 3=trigram)
    ngram_dim: int = 256  # embedding dimension per n-gram
    ngram_host: bool = True  # store on host RAM (True) or GPU (False)

    # === R&D Round 20-21: Memory-efficient training + cross-domain formats ===
    # Optimizer selection: "adamw_8bit", "adamw_4bit", "nvme_badam",
    # "muon_bitnet_4bit", "ternary", "nvme_muon_4bit" (best combo)
    optimizer_type: str = "adamw_8bit"
    # NVMe streaming: optimizer states on NVMe, 1 layer active in RAM
    nvme_streaming: bool = False
    nvme_path: str = ""  # path for NVMe optimizer state files
    # FP8 activation storage: quantize activations to FP8 during forward
    # (2x activation memory reduction). GEMMs still run in bf16.
    use_fp8_activation: bool = False
    # GradTopK: top-K% gradient sparsification with EF21 error feedback.
    # 10x fewer gradient transfers per step.
    grad_topk_ratio: float = 1.0  # 1.0 = disabled, 0.1 = top 10%
    grad_topk_ef: bool = True  # error feedback (prevents staleness)
    # HashedNLRQ: hash the NLRQ low-rank factors for additional compression.
    # NLRQ (12.8x) × hash (4-16x) = 50-100x total FFN compression.
    use_hashed_nlrq: bool = False
    hashed_nlrq_compression: float = 8.0  # hash compression factor

    # === R&D Round 24: SpectralKV + BitNetResidual (2026-08-30) ===
    # SpectralKV: Fourier-basis KV cache with O(1) memory per token.
    # Validated on real LFM2.5 weights: 63× compression at 0.095 error.
    # Selected via ActivationConfig.kv_cache="spectral" at inference time.
    use_spectral_kv: bool = False  # config flag (also selectable at inference)
    spectral_kv_max_freq: int = 64  # Fourier frequencies (more = lower error)
    spectral_kv_sink_size: int = 4  # full-precision sink tokens
    # BitNetResidual: ternary weights + element-level dense residual.
    # Validated on real LFM2.5 weights: 0.33 error at 1.8× compression.
    # Replaces pure BitNet (0.80 error) and NLRQ rank-1024 (1.4 error).
    use_bitnet_residual: bool = False
    bitnet_residual_frac: float = 0.10  # fraction of elements kept dense
    bitnet_residual_type: str = "element"  # "element" (validated best)

    # === R&D Round 26: IRI-FP4 Lossless Weight Quantization (2026-08-31) ===
    # IRI-FP4: Iterative Residual FP4. Per-block MSE-optimal scale + iterative
    # residual refinement. Validated on real LFM2.5 + Qwen 2.5 weights:
    #   - x1 (4.5 bits/w, 20.7 dB SQNR): +50.8% PPL — too lossy
    #   - x2 (9.0 bits/w, 41.6 dB SQNR): -0.4% PPL — LOSSLESS, 3.5x vs fp32
    #   - x3 (13.6 bits/w, 62.6 dB SQNR): +0.1% PPL — overkill
    # V10 uses x2: 3.5x compression, near-lossless, no QAT needed.
    use_iri_fp4: bool = False
    iri_fp4_rounds: int = 2        # 2 rounds = 9.0 bits/w, 41.6 dB SQNR, lossless
    iri_fp4_block_size: int = 32   # block size for per-block scale
    # === R&D Round 49: SubBitnet — training-free sub-bitnet quantization ===
    # Hadamard incoherence rotation + Iterative Residual Binarization (IRB) +
    # optional SVD low-rank residual. Training-free QAT substitute targeting
    # sub-bitnet bit-widths (<= 1.58 bits/w). No calibration, no gradient steps.
    #   - rounds=1, rank=0:  ~1.0 bit/w  (pure binary, sub-bitnet)  ← default
    #   - rounds=1, rank=16: ~1.02 bit/w (binary + structured residual)
    #   - rounds=2, rank=0:  ~2.0 bit/w
    # Mutually exclusive with use_bitnet / use_iri_fp4 (apply at inference).
    use_sub_bitnet: bool = False
    sub_bitnet_rounds: int = 1     # IRB binary rounds (1 = pure 1-bit)
    sub_bitnet_rank: int = 0       # SVD low-rank residual rank (0 = disabled)
    sub_bitnet_hadamard: bool = True  # Hadamard incoherence rotation
    # Gradient checkpointing: recompute forward during backward to save VRAM.
    use_gradient_checkpointing: bool = False
    # Selective checkpoint strategy: "all" (full block), "ffn" (recompute only
    # the FFN — largest activation consumer, minimal compute penalty), "attn",
    # "none". Only applies when use_gradient_checkpointing is True.
    # Evolution-discovered: "selective" with all 16 layers + block_size=512
    # gives best quality (0.92) at 50%+ VRAM savings for 32K context.
    selective_gradient_checkpointing: str = "all"
    # Checkpoint block size (evolution-discovered: 512 matches typical seq len)
    checkpoint_block_size: int = 512
    # Fixed attention scale (0.12 from NanoGPT speedrun) instead of head_dim**-0.5.
    attn_scale: float = None

    # === Training extensions ===
    # OEC (Output Embedding Centering): suppresses anisotropic common-mode shift.
    oec_mode: str = "none"
    oec_lambda: float = 1e-3
    # SC-GRPO / OM-GRPO / GVPO: RL training extensions.
    use_sc_grpo: bool = False
    use_om_grpo: bool = False
    use_gvpo: bool = False
    gvpo_lambda: float = 0.3

    # === AdaLN-zero conditioning (DiT — Diffusion Transformers) ===
    # When set, transformer blocks use AdaLNZero instead of standard LayerNorm.
    # The modulation MLP is zero-init → identity at start (lossless).
    # cond_dim=None means no conditioning (standard LayerNorm) — backward compat.
    cond_dim: int | None = None

    # === Legacy compat (kept for keys that reference these) ===
    kv_compression_dim: int = 128  # used by MLA keys (no longer in core)
    enable_draft_head: bool = False

    # === V11: Vision-Language Model (LFM2.5-VL-3B) ===
    # Multimodal extension: SigLIP2 vision encoder + projector + LM.
    # The LM is a standard ConfigurableResearchLLM; the vision tower is
    # a separate module that produces token-aligned embeddings injected
    # before the first attention layer.
    use_vision: bool = False               # enable vision tower
    vision_encoder: str = "siglip2"        # "siglip2" (only supported)
    vision_hidden_size: int = 1152         # SigLIP2-SO400M hidden dim
    vision_image_size: int = 384           # input image resolution
    vision_patch_size: int = 14            # patch size for ViT
    vision_n_layers: int = 27              # SigLIP2-SO400M layers
    vision_n_heads: int = 16               # SigLIP2 attention heads
    vision_intermediate_size: int = 4304   # SigLIP2 FFN dim
    vision_projector_dim: int = 2048       # projector output (matches d_model)
    vision_projector_type: str = "mlp"     # "mlp" or "linear"
    vision_n_queries: int = 128            # number of visual tokens after pooling
    vision_layer_idx: int = -1             # which LM layer receives visual tokens (-1 = before layer 0)

    # === V12: ForgeLM V12 — New Architecture Keys (R37) ===
    # Mamba-3: Complex-valued SSM states (R37-1). Extends Mamba-2 with
    # complex A_log, B, C. Lossless warm start (imag=0).
    use_mamba3: bool = False
    mamba3_d_state: int = 16               # complex state dimension
    # Kronecker Embeddings: byte-level factored embeddings (R37-2).
    # 91-94% input-side param reduction. Drop-in nn.Embedding replacement.
    use_kronecker_embed: bool = False
    kronecker_d_char: int = 64             # character embedding dimension
    kronecker_max_char_len: int = 8        # max characters per token
    # PIT Tying: Pseudo-Inverse Tying for stable token interface (R37-3).
    # Already has use_pit field (line 57). V12 enables it.
    # OutRo: Sink-enhanced contextual representations (R37-4).
    # Allow sink token to attend beyond causal constraint.
    use_outro: bool = False
    outro_sink_threshold: float = 0.5      # attention threshold for sink detection
    # ForgeHybrid: Sink-aware SSM+Attention routing (R37-5 NOVEL).
    # Zero-init SSM path alongside attention. Router = sink norm signal.
    use_forge_hybrid: bool = False
    forge_hybrid_d_state: int = 16         # SSM state dimension for hybrid path
    forge_hybrid_sink_threshold: float = float("inf")  # routing threshold (inf = all attention at warm start)
    forge_hybrid_n_ssm_layers: int | None = None  # None = all layers, int = subset for gradual rollout

    # === R49-2: KDA — Kimi Delta Attention (Gated DeltaNet) side-path ===
    # Adds a linear-attention branch to every block, gated to zero at init
    # (bit-exact warm start; training opens the gate). KDA layers hold a
    # fixed (H x d_k x d_v) recurrent state and NO KV cache — a future
    # 3:1 KDA:full-attention hybrid cuts KV ~75%. VRAM: ~25M params/layer
    # at d_model=2560,H=20 (bf16 ~50MB/layer); state ~1.3MB fp32/layer.
    use_kda: bool = False
    kda_n_heads: int = 0            # 0 = inherit n_heads
    kda_head_dim: int = 0           # 0 = inherit d_model // n_heads
    kda_beta_gt1: bool = False      # N3: allow beta in (0,2) — off (destabilization risk)
    kda_decay_floor: float = 0.0    # optional lower bound on per-key-dim decay

    # === R60-1: GDN2 — Gated DeltaNet-2 (arXiv:2605.22791) side-path ===
    # Strict KDA superset: decouples erase (b_t, per key-dim) from write
    # (w_t, per value-dim) — KDA ties both to scalar beta_t. gate=0 warm
    # start (bit-exact); kda.* checkpoints convert function-exactly via
    # GDN2Key (tiled scalar gate = channel-wise broadcast). Mutually
    # exclusive with use_kda (GDN2 subsumes it). VRAM: ~39M params/layer
    # at d_model=2560,H=20 (bf16 ~78MB/layer) — heavier than KDA's 25M
    # because e/w projections are channel-wise.
    use_gdn2: bool = False
    gdn2_n_heads: int = 0           # 0 = inherit n_heads
    gdn2_head_dim: int = 0          # 0 = inherit d_model // n_heads
    gdn2_decay_floor: float = 0.0   # optional lower bound on per-key-dim decay
    # V14: restrict side-paths to a layer subset (None = all blocks — legacy
    # behaviour). GDN2/Titan are param-heavy; a strided subset keeps the
    # LiveT surface at a fraction of the param cost on tight budgets.
    gdn2_layers: list[int] | None = None     # None = all blocks
    titan_layers: list[int] | None = None    # None = all blocks

    # === V14: RSI work-acceleration heads (read-only taps on hidden) ===
    # CopySpan edit head: LM emits <copy> op, pointer head emits (start,end)
    # over cached source hiddens; driver splices verbatim + chunked commit.
    use_edit_head: bool = False
    edit_head_dim: int = 0          # 0 = d_model (pointer q/k width)
    copy_op_token_id: int = -1      # vocab id of the <copy> op token (-1=unset)
    # Verifier head: scalar P(candidate passes tests) on ln_f(hidden);
    # BT-trained on evolve pairs → best-of-n rerank + early-stop gate.
    use_verifier_head: bool = False
    verifier_head_dim: int = 256

    # === Preset lineage (critique F2 — machine-enforced lineage) ===
    # parent: name of the parent preset this one is derived from (or None for
    #   root presets). When set, get_config() validates that every field in
    #   the parent is present in this preset unless listed in dropped_keys.
    parent: str | None = None
    # dropped_keys: fields intentionally NOT carried forward from parent.
    #   Each must have a reason comment in the preset definition.
    dropped_keys: tuple[str, ...] = ()

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}).")
        # GQA: n_heads must be a multiple of n_kv_heads (or n_kv_heads == n_heads).
        n_kv = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
        if n_kv <= 0 or self.n_heads % n_kv != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a positive multiple of "
                f"n_kv_heads ({n_kv}) for GQA."
            )
        # Per-layer type list must match n_layers when provided.
        # If n_layers is overridden but layer_types isn't, auto-resize by
        # truncating or cycling the existing pattern to match n_layers.
        if self.layer_types is not None:
            n_lt = len(self.layer_types)
            if n_lt != self.n_layers:
                if n_lt > self.n_layers:
                    # Truncate to first n_layers entries.
                    self.layer_types = list(self.layer_types[: self.n_layers])
                else:
                    # Cycle existing pattern to fill n_layers.
                    base = list(self.layer_types)
                    self.layer_types = [
                        base[i % n_lt] for i in range(self.n_layers)
                    ]
        # FFN intermediate size: positive when explicitly set.
        if self.intermediate_size is not None and self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) must be > 0."
            )
        # head_dim must be consistent (explicit or d_model / n_heads) and >= 1.
        head_dim = self.head_dim if self.head_dim is not None else (
            self.d_model // self.n_heads)
        if head_dim < 1:
            raise ValueError(
                f"head_dim ({head_dim}) must be >= 1."
            )
        # GDN2 strictly subsumes KDA (b=w collapse) — both side-paths on
        # one block is redundant; refuse rather than silently double-pay.
        if getattr(self, "use_gdn2", False) and getattr(self, "use_kda", False):
            raise ValueError(
                "use_gdn2 and use_kda are mutually exclusive — GDN2 is a "
                "strict superset of KDA; port kda.* checkpoints via "
                "GDN2Key (function-exact) instead of enabling both."
            )


# Pre-defined architecture targets.
MODEL_CONFIGS = {
    # Tiny hybrid (conv+attention) model for fast testing — no checkpoint.
    "forgelm_tiny": ModelConfig(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=256,
        attn_type="gqa",
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,          # LFM2.5 original: 1M (reference port)
        max_seq_len=128,
        conv_kernel_size=3,
        use_qk_norm=True,
        layer_types=["conv", "conv", "attention", "conv"],
        batch_size=2,
        seq_len=64,
        max_steps=20,
        warmup_steps=5,
    ),

    # Ultra-compact ForgeLM V7 gen model for ForgeEvolve.
    # Uses V7 architecture keys (BitNet b1.58, GTA attention, NLRQ FFN
    # compression, RoPE) at a tiny size for use as an LLM-based gen model
    # in evolutionary search. Built from scratch (random init) — no checkpoint.
    # Reuses LFM2.5 tokenizer (vocab=65536). Configurable size for grow/shrink.
    "gen_model_tiny": ModelConfig(
        vocab_size=65536,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,                   # GQA 4x (8/2)
        intermediate_size=512,
        attn_type="gta",                # Grouped-Tied Attention (V7)
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,          # LFM2.5 base: 1M
        max_seq_len=512,
        use_qk_norm=True,
        # ── V7 keys (minimal set for tiny gen model) ──
        use_bitnet=True,                # BitNet b1.58 ternary QAT
        bitnet_learned_scale=True,
        ffn_compression="nlrq",         # NLRQ FFN compression
        nlrq_rank=64,                   # small rank for tiny intermediate=512
        nlrq_factor_bits=8,
        nlrq_use_residual=False,
        # ── Training hyperparams (for fine-tuning) ──
        batch_size=1,
        seq_len=512,
        max_steps=100,
        warmup_steps=10,
        max_lr=1e-3,
        min_lr=1e-4,
    ),


    # ──────────────────────────────────────────────────────────────────────
    # ForgeLM V2 — Jamba-Reasoning-3B base (2026-09-04)
    #
    # Ported from ai21labs/AI21-Jamba-Reasoning-3B via lossless key mapping.
    # 28-layer hybrid: 26 Mamba + 2 attention (layers 7 and 21).
    # d_model=2560, vocab=65536, SwiGLU FFN, RMSNorm, RoPE base=1M.
    # Separate embed/head (not tied). Jamba-style norms on Mamba dt/B/C.
    # Checkpoint: research/checkpoints/ForgeLM_V2.safetensors (6.1 GB, bf16)
    # ──────────────────────────────────────────────────────────────────────
    "forgelm_v2": ModelConfig(
        vocab_size=65536,
        d_model=2560,
        n_layers=28,
        n_heads=20,
        n_kv_heads=1,
        intermediate_size=8192,
        attn_type="gqa",
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        norm_eps=1e-6,
        use_embed_norm=False,
        use_final_norm=True,
        rope_base=1_000_000.0,
        use_rope=False,  # Jamba attention is position-agnostic (no RoPE);
                         # Mamba layers carry positional information
        max_seq_len=262144,
        conv_kernel_size=3,
        use_qk_norm=False,
        tie_word_embeddings=False,
        layer_types=[
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
            "attention",
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
            "attention",
            "mamba", "mamba", "mamba", "mamba", "mamba", "mamba",
        ],
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=160,
        mamba_bias=False,
        mamba_conv_bias=True,
        use_iri_fp4=False,
        use_spectral_kv=False,
        use_bitnet_residual=False,
        use_bitnet=False,
        use_bitnet_embedding=False,
        ffn_compression="none",
        nlrq_rank=0,
        use_hashed_nlrq=False,
        use_factorized_embeddings=False,
        embed_factorized_rank=0,
        use_pit=False,
        zero_init_residual=False,
        batch_size=1,
        seq_len=4096,
        max_steps=50000,
        warmup_steps=2000,
        max_lr=2e-4,
        min_lr=2e-5,
    ),

}

# ──────────────────────────────────────────────────────────────────────────────
# ForgeLM V12-Jamba: V2 (Jamba-3B) + V12 architecture keys (R37).
#
# Derived from forgelm_v2 (Jamba-Reasoning-3B). Carries forward ALL V2
# architecture (28 layers, 26 Mamba + 2 attention, d_model=2560, vocab=65536).
# Adds the 5 R37 keys with lossless zero-init warm start:
#   - Mamba-3: complex state (2× expressivity), zero-init imaginary part
#   - Kronecker embeddings: 99.2% param reduction at vocab=65536
#   - PIT tying: orthonormal shared memory (identity init = lossless)
#   - OutRo: sink-aware attention (sink_threshold=0.5)
#   - ForgeHybrid: SSM+attention routing (zero-init SSM = lossless)
#
# All new keys are zero/identity-init → forward pass is identical to V2
# at warm start. The checkpoint is a converted V2 checkpoint with the
# new key weights added as zeros/identities.
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["forgelm_v12_jamba"] = ModelConfig(
    **{
        **MODEL_CONFIGS["forgelm_v2"].__dict__,
        "parent": "forgelm_v2",  # lineage: derived from V2 (critique F2)
        "dropped_keys": (  # V12-Jamba divergences from V2 (all NEW keys)
            "use_mamba3",          # R37-1: Mamba-3 complex state
            "use_kronecker_embed", # R37-2: Kronecker embeddings
            "use_pit",            # R37-3: PIT tying
            "use_outro",          # R37-4: OutRo sink-aware attention
            "use_forge_hybrid",   # R37-5: SSM+attention routing
        ),
        # ── V12 NEW: Mamba-3 (R37-1) ──
        "use_mamba3": True,
        "mamba3_d_state": 16,
        # ── V12 NEW: Kronecker Embeddings (R37-2) ──
        "use_kronecker_embed": True,
        "kronecker_d_char": 64,
        "kronecker_max_char_len": 8,
        # ── V12 NEW: PIT Tying (R37-3) ──
        "use_pit": True,
        # ── V12 NEW: OutRo (R37-4) ──
        "use_outro": True,
        "outro_sink_threshold": 0.5,
        # ── V12 NEW: ForgeHybrid (R37-5 NOVEL) ──
        "use_forge_hybrid": True,
        "forge_hybrid_d_state": 16,
        "forge_hybrid_sink_threshold": float("inf"),  # all attention at warm start
        "forge_hybrid_n_ssm_layers": None,            # all layers get SSM path
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# ForgeLM V13-GDN2: V12-Jamba + Gated DeltaNet-2 side-paths (R60-1).
#
# Derived from forgelm_v12_jamba (carries forward ALL V12 keys + V2 base).
# Adds GDN2 (arXiv:2605.22791): decoupled channel-wise erase/write gates —
# a strict KDA superset (collapses to KDA when b_t=w_t). At gate=0 the
# forward pass is bit-exact vs. V12/V2; a kda-ported checkpoint upgrades
# function-exactly via GDN2Key.convert_model_state (tiled scalar gates).
# use_kda stays False — GDN2 subsumes it.
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["forgelm_v13_gdn2"] = ModelConfig(
    **{
        **MODEL_CONFIGS["forgelm_v12_jamba"].__dict__,
        "parent": "forgelm_v12_jamba",  # lineage: derived from V12 (critique F2)
        "dropped_keys": (  # V13 divergences from V12 (all NEW keys)
            "use_gdn2",            # R60-1: GDN2 decoupled erase/write delta rule
            "gdn2_n_heads",
            "gdn2_head_dim",
            "gdn2_decay_floor",
        ),
        # ── V13 NEW: GDN2 side-paths (R60-1) ──
        "use_gdn2": True,
        "gdn2_n_heads": 0,          # inherit n_heads (20)
        "gdn2_head_dim": 0,         # inherit d_model // n_heads (128)
        "gdn2_decay_floor": 0.0,
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# ForgeLM V14-RSI: purpose-built ~2.5B RSI-loop coder (docs/V14_RSI_DESIGN.md).
#
# Derived from forgelm_v13_gdn2 (carries forward V12+V13 keys). NEW topology —
# intentionally NON-lossless vs V13 (24 layers, hybrid conv/mamba3/GLA mix,
# MoE FFN); every divergence is documented in dropped_keys below.
#
# Design targets (RTX 5070, ~672 GB/s): <3GB at max load (int4 ≈1.3GB weights),
# ~1k tok/s decode via ~1.3B active params/step + MTP self-spec + CopySpan,
# minutes-scale hone (tagged LoRA), instant ingest (GDN2/Titan/NGram live).
#
# Divergences vs V13 (all documented):
#   n_layers 28→24, layer_types: explicit conv/mamba3/GLA mix (was 26 mamba
#   +2 attn), attn_type gqa→gla (latent KV, d_c=256), use_forge_hybrid OFF
#   (explicit mamba3 layers replace per-block SSM routing), gdn2_n_heads 0→4
#   (9.2M/layer vs 39M — LiveT surface on all blocks within budget),
#   use_outro OFF (untested on GLA; latent attention supersedes it),
#   use_titan_memory ON but titan_layers subset (4 blocks ≈105M params),
#   use_moe ON (8 experts top-2 + shared, dense_bypass warm start),
#   use_mtp ON (4 heads, evolution-tuned weight), NEW use_edit_head +
#   use_verifier_head (RSI work-acceleration heads), zero_init_residual ON
#   (clean distill init), use_qk_norm + use_embed_norm ON (GLA/LFM2-style).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["forgelm_v14_rsi"] = ModelConfig(
    **{
        **MODEL_CONFIGS["forgelm_v13_gdn2"].__dict__,
        "parent": "forgelm_v13_gdn2",
        "dropped_keys": (
            "n_layers",            # 28 → 24 (new topology)
            "layer_types",         # conv/mamba3/GLA hybrid mix
            "attn_type",           # gqa → gla (latent-compressed KV)
            "gla_latent_dim",      # 0 → 256
            "max_seq_len",         # 262144 → 16384 (RSI prompts <8k; RoPE
                                   # cache buffers scale ~16× down)
            "use_forge_hybrid",    # OFF: explicit mamba3 layers replace routing
            "gdn2_n_heads",        # 0(inherit 20) → 4 (param budget)
            "gdn2_head_dim",       # 0(inherit 128) → 128 explicit (heads
                                   # otherwise derive 640-wide, ~46M/layer)
            "use_outro",           # OFF: untested on GLA; latent attn supersedes
            "use_titan_memory",    # ON: LiveT persistent store
            "titan_layers",        # subset [5,11,17,23] vs all-block default
            "titan_memory_rank",   # 0(full d×d) → 128 factorized live surface
            "use_moe",             # ON: 8 experts top-2 + shared
            "moe_d_ff",            # None → 768 (expert dim)
            "use_mtp",             # ON: 4-head self-spec draft
            "mtp_adapter_heads",   # ON: DeepSeek-style shared-head adapters
            "use_edit_head",       # NEW: CopySpan edit head
            "copy_op_token_id",    # -1 placeholder → resolved at driver
            "use_verifier_head",   # NEW: P(pass) rerank/early-stop head
            "zero_init_residual",  # ON: clean distillation init
            "use_qk_norm",         # ON: GLA stability
            "use_embed_norm",      # ON: LFM2-style embed norm
        ),
        # ── V14 topology ──
        "n_layers": 24,
        "layer_types": (
            ["conv", "conv", "mamba", "conv", "conv", "attention"] * 4
        ),  # 16 conv + 4 mamba3 + 4 GLA attention
        "attn_type": "gla",
        "gla_latent_dim": 256,
        "n_kv_heads": 1,           # MQA at GLA layer (latent dominates anyway)
        "max_seq_len": 16384,
        # ── Carried keys with budget adjustments ──
        "use_forge_hybrid": False,
        "gdn2_n_heads": 4,
        "gdn2_head_dim": 128,
        "use_outro": False,
        "use_titan_memory": True,
        "titan_layers": [5, 11, 17, 23],
        "titan_memory_rank": 128,
        # ── MoE FFN: capacity without bandwidth cost ──
        "use_moe": True,
        "moe_d_ff": 768,
        # ── Acceleration heads ──
        "use_mtp": True,
        "mtp_adapter_heads": True,
        "use_edit_head": True,
        "use_verifier_head": True,
        # ── Init norms ──
        "zero_init_residual": True,
        "use_qk_norm": True,
        "use_embed_norm": True,
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# Qwen2.5-0.5B (HF compat, R49) — native shapes for HF-style checkpoints.
#
# Matches Qwen/Qwen2.5-0.5B exactly: GQA 14Q/2KV (head_dim 64), QKV bias,
# SwiGLU intermediate 4864, RMSNorm, RoPE base 1M, 32K context, untied
# embed/head. Used by ForgeEngine._detect_config_from_header to auto-pair
# checkpoints with HF-style keys (model.layers.N.*, embed 151936x896).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["qwen25_05b"] = ModelConfig(
    vocab_size=151936,
    d_model=896,
    n_layers=24,
    n_heads=14,
    n_kv_heads=2,
    intermediate_size=4864,
    attn_type="gqa",
    attn_bias=True,                # Qwen2.5 QKV bias (q/k/v_proj.bias)
    ffn_type="swiglu",
    norm_type="rmsnorm",
    rope_base=1_000_000.0,         # Qwen2.5 rope_theta
    max_seq_len=32768,
    use_qk_norm=False,             # Qwen2.5 has no q_norm/k_norm (Qwen3 only)
    tie_word_embeddings=True,      # 0.5B ties embed/head (no lm_head.weight in ckpt)
)

# Qwen2.5-0.5B → MLA+MoE port (ForgeLM-v1 recipe, updated): GQA→latent-KV
# via SVD (d_c=256 = full joint-KV rank → bit-exact), dense SwiGLU→4 routed
# experts (dense_bypass → exact at init, router learns during SFT/evolve).
# Produced by scripts/port_qwen_mla_moe.py; ckpt carries forge_convert.json.
MODEL_CONFIGS["qwen25_05b_mla_moe"] = ModelConfig(
    vocab_size=151936,
    d_model=896,
    n_layers=24,
    n_heads=14,
    n_kv_heads=2,
    intermediate_size=4864,
    attn_type="gla",
    gla_latent_dim=256,
    attn_bias=True,                # q_proj bias; kv_down bias emitted as zeros
    ffn_type="swiglu",
    use_moe=True,
    moe_n_experts=4,
    moe_top_k=4,                   # dense_bypass ignores routing anyway
    moe_d_ff=1216,                 # 4864/4 — experts partition the dense FFN
    moe_shared_expert=False,
    moe_dense_bypass=True,         # lossless init: all experts equal weight
    moe_noisy_gating=False,
    norm_type="rmsnorm",
    norm_eps=1e-6,
    rope_base=1_000_000.0,
    max_seq_len=32768,
    tie_word_embeddings=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Qwen2.5-Coder-3B MLA+MoE port target (V14-RSI P1 teacher / proposer upgrade).
# Same recipe as qwen25_05b_mla_moe at 3B scale: GQA → GLA latent-KV via
# joint SVD (d_c=512 = 2*kv_dim, kv_dim=2*128=256 → full-rank → lossless),
# dense SwiGLU 11008 → 8 routed experts d_ff=1376 (dense_bypass = exact).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["qwen25_coder_3b_mla_moe"] = ModelConfig(
    vocab_size=151936,
    d_model=2048,
    n_layers=36,
    n_heads=16,
    n_kv_heads=2,
    intermediate_size=11008,
    attn_type="gla",
    gla_latent_dim=512,
    attn_bias=True,
    ffn_type="swiglu",
    use_moe=True,
    moe_n_experts=8,
    moe_top_k=8,                   # dense_bypass ignores routing anyway
    moe_d_ff=1376,                 # 11008/8 — experts partition the dense FFN
    moe_shared_expert=False,
    moe_dense_bypass=True,         # lossless init: all experts equal weight
    moe_noisy_gating=False,
    norm_type="rmsnorm",
    norm_eps=1e-6,
    rope_base=1_000_000.0,
    max_seq_len=32768,
    tie_word_embeddings=True,
)

# Same mla_moe recipe at Qwen2.5-Coder-7B scale: GQA 28Q/4KV -> GLA
# latent-KV (512), dense FFN 18944 -> 8 experts of 2368 (dense-bypass =
# lossless init). Differs from the 3B preset: vocab 152064 and UNTIED
# embeddings (7B has separate embed/head). Local-teacher port target:
# latent-KV shrinks per-token cache -> bigger decode batches on 12GB.
MODEL_CONFIGS["qwen25_coder_7b_mla_moe"] = ModelConfig(
    vocab_size=152064,
    d_model=3584,
    n_layers=28,
    n_heads=28,
    n_kv_heads=4,
    intermediate_size=18944,
    attn_type="gla",
    gla_latent_dim=512,
    attn_bias=True,
    ffn_type="swiglu",
    use_moe=True,
    moe_n_experts=8,
    moe_top_k=8,
    moe_d_ff=2368,                 # 18944/8
    moe_shared_expert=False,
    moe_dense_bypass=True,
    moe_noisy_gating=False,
    norm_type="rmsnorm",
    norm_eps=1e-6,
    rope_base=1_000_000.0,
    max_seq_len=32768,
    tie_word_embeddings=False,
)

# ──────────────────────────────────────────────────────────────────────────────
# Qwen3-4B-Instruct-2507 family (HF compat, R50) — native shapes for
# HF-style checkpoints.
#
# Matches Qwen/Qwen3-4B-Instruct-2507 exactly: GQA 32Q/8KV with an
# explicit head_dim=128 (d_model/n_heads=80, decoupled — the reason the
# head_dim config field exists), QK-norm on Q/K, NO QKV bias (Qwen3
# dropped it), SwiGLU intermediate 9728, RMSNorm eps 1e-6, RoPE base 5M,
# 256K native context, tied embed/head, vocab 151936.
# Used by ForgeEngine._detect_config_from_header to auto-pair checkpoints
# (embed 151936x2560, q_proj 2560x4096). First consumer:
# 0xA50C1A1/Qwen3-4B-Nymphaea-RP (uncensored RP tune).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["qwen3_4b"] = ModelConfig(
    vocab_size=151936,
    d_model=2560,
    n_layers=36,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,                  # explicit: 2560/32=80 would be wrong
    intermediate_size=9728,
    attn_type="gqa",
    attn_bias=False,               # Qwen3 has no QKV bias (unlike Qwen2.5)
    ffn_type="swiglu",
    norm_type="rmsnorm",
    norm_eps=1e-6,                 # Qwen3 rms_norm_eps
    rope_base=5_000_000.0,         # Qwen3-4B rope_theta
    max_seq_len=262144,            # 256K native context
    use_qk_norm=True,              # Qwen3 q_norm/k_norm on each head
    use_final_norm=True,
    tie_word_embeddings=True,      # tied embed/lm_head in checkpoint
)

# ──────────────────────────────────────────────────────────────────────────────
# Llama-3.1-8B family (HF compat) — native shapes for HF-style checkpoints.
#
# Matches meta-llama/Llama-3.1-8B(-Instruct) and derived tunes exactly:
# GQA 32Q/8KV, no attn/MLP bias, SwiGLU intermediate 14336, RMSNorm eps 1e-5,
# RoPE base 500K + llama3 frequency scaling (factor 8 over orig 8K ctx),
# untied embed/head, vocab 128256.
# First consumer: Hermes-3-Llama-3.1-8B-lorablated (uncensored ChatML tune;
# eos = 128040 = im_end, im_start = 128039 — both in _DEFAULT_EOS_TOKEN_IDS).
# Used by ForgeEngine._detect_config_from_header to auto-pair checkpoints
# (embed 128256x4096, q_proj 4096x4096, kv_proj 1024x4096).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["llama31_8b"] = ModelConfig(
    vocab_size=128256,
    d_model=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    intermediate_size=14336,
    attn_type="gqa",
    attn_bias=False,               # Llama has no QKV/o bias
    ffn_type="swiglu",
    norm_type="rmsnorm",
    norm_eps=1e-5,                 # Llama rms_norm_eps
    rope_base=500_000.0,           # Llama-3.1 rope_theta
    rope_scaling={                 # Llama-3.1 frequency scaling (8x over 8K base)
        "type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
    max_seq_len=32768,             # capped for 12GB VRAM KV budget (native 131072)
    use_qk_norm=False,
    use_final_norm=True,
    tie_word_embeddings=False,     # untied lm_head
)

# ──────────────────────────────────────────────────────────────────────────────
# Llama-3.2-3B family (HF compat) — native shapes for HF-style checkpoints.
#
# Matches meta-llama/Llama-3.2-3B(-Instruct) and derived tunes exactly:
# GQA 24Q/8KV (head_dim=128), no attn/MLP bias, SwiGLU intermediate 8192,
# RMSNorm eps 1e-5, RoPE base 500K + llama3 scaling (factor 32 — 3.2 scales
# 32x over the 8K base vs 3.1's 8x), TIED embed/lm_head, vocab 128256.
# First consumer: NousResearch/Hermes-3-Llama-3.2-3B (ChatML, tool calls —
# same format family as the 8B, ~3x faster decode, ~2GB at int4).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["llama32_3b"] = ModelConfig(
    vocab_size=128256,
    d_model=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    head_dim=128,
    intermediate_size=8192,
    attn_type="gqa",
    attn_bias=False,
    ffn_type="swiglu",
    norm_type="rmsnorm",
    norm_eps=1e-5,
    rope_base=500_000.0,
    rope_scaling={
        "type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
    max_seq_len=32768,             # capped for 12GB VRAM KV budget (native 131072)
    use_qk_norm=False,
    use_final_norm=True,
    tie_word_embeddings=True,      # 3.2 ties embed/lm_head (unlike 8B/70B)
)

# ──────────────────────────────────────────────────────────────────────────────
# Rocinante-X-12B (Mistral-Nemo arch, GGUF compat, R65) — native shapes for
# GGUF-converted checkpoints of Mistral-Nemo-family merges.
#
# Matches TheDrummer/Rocinante-X-12B-v1 (and any Mistral-Nemo-12B derivative)
# exactly: GQA 32Q/8KV with an explicit head_dim=128 (d_model/n_heads=160,
# decoupled — GGUF key_length/value_length=128), no biases, SwiGLU
# intermediate 14336, RMSNorm eps 1e-5, RoPE base 1M (NO scaling — Nemo is
# native 128K), untied embed/head, vocab 131072 (tekken BPE).
# First consumer: Rocinante-X-12B-v1b-Q4_K_M.gguf ported via
# scripts/convert_gguf_forge.py (dequant bf16 → HF-name safetensors).
# Used by ForgeEngine._detect_config_from_header to auto-pair checkpoints
# (embed 131072x5120, q_proj 5120x4096, kv_proj 5120x1024).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["rocinante_12b"] = ModelConfig(
    vocab_size=131072,
    d_model=5120,
    n_layers=40,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,                  # explicit: 5120/32=160 would be wrong
    intermediate_size=14336,
    attn_type="gqa",
    attn_bias=False,               # Nemo has no QKV/o bias
    ffn_type="swiglu",
    norm_type="rmsnorm",
    norm_eps=1e-5,                 # llama.attention.layer_norm_rms_epsilon
    rope_base=1_000_000.0,         # llama.rope.freq_base
    max_seq_len=131072,            # native ctx; flux KV keeps positions bounded
    use_qk_norm=False,
    use_final_norm=True,
    tie_word_embeddings=False,     # untied lm_head (output.weight in GGUF)
)


# ──────────────────────────────────────────────────────────────────────────────
# LFM3-1.7B port (2026-10-29) — LiquidAI tri-hybrid conv+attention.
#
# 24 layers: 13 gated short convs (L=7), 5 sliding-window attention (W=256,
# theta=10K, full RoPE), 6 full attention (theta=1M, partial RoPE 0.5
# zero-tail). GQA 16Q/4KV heads, head_dim=128, QK-norm, per-head sigmoid
# output gate. Pre-residual sandwich norms (post-op + post-FFN applied to
# the branch output before the residual add). Embed norm, NO final norm,
# tied embeddings, vocab 128K. No biases anywhere.
#
# Checkpoint: research/checkpoints/lfm3_1p7b_cc (mondk Claude-Code
# fine-tune — stock LFM3 arch, R1 tools + R2 bench + R3 self-RL on top of
# LiquidAI/LFM3-1.7B). HF keys remap via ModelLoader._remap_hf_keys.
# forge_convert.json in the dir names this preset; non-lossless delta vs
# the HF reference is bounded by kernel-level fp error only (verified by
# scripts/test_lfm3_port.py logit comparison, venv_train).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["lfm3_1p7b"] = ModelConfig(
    vocab_size=128_000,
    d_model=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    head_dim=128,                  # explicit (2048/16 happens to match)
    intermediate_size=7680,
    attn_type="gqa",
    attn_bias=False,
    ffn_type="swiglu",
    norm_type="rmsnorm",
    norm_eps=1e-5,                 # HF norm_eps
    use_embed_norm=False,
    use_final_norm=True,           # model.embedding_norm IS the final norm
                                   # (applied after the last layer, misnamed)
    use_rope=True,
    rope_base=1_000_000.0,         # default for full-attention layers
    # HF max_position_embeddings=262144, but RotaryEmbedding precomputes
    # cos/sin tables per attention layer (~500MB/layer at 262K → OOM on
    # 12GB). The evolve loop generates ≤~4K tokens; 16384 is lossless for
    # that workload. Raise only with a lazy/shared RoPE table.
    max_seq_len=16_384,
    conv_kernel_size=7,            # conv_L_cache
    use_qk_norm=True,
    tie_word_embeddings=True,
    layer_types=[
        "conv", "conv", "attention", "conv", "attention_swa", "conv",
        "attention", "conv", "attention_swa", "conv", "attention", "conv",
        "attention_swa", "conv", "attention", "conv", "attention_swa",
        "conv", "attention", "conv", "attention_swa", "conv", "attention",
        "conv",
    ],
    sliding_window=256,
    use_attn_out_gate=True,        # sigmoid per-head gate on attn output
    use_sandwich_norm=True,
    sandwich_norm_mode="pre_residual",
    rope_layer_params={
        "attention":       {"rope_theta": 1_000_000.0,
                            "partial_rotary_factor": 0.5},
        "attention_swa":   {"rope_theta": 10_000.0,
                            "partial_rotary_factor": 1.0},
    },
    batch_size=4,                  # batch-gen sweet spot for the evolve loop
    seq_len=2048,
    max_steps=500,
    warmup_steps=50,
    max_lr=2e-4,
    min_lr=2e-5,
)

# ──────────────────────────────────────────────────────────────────────────────
# Spark-X2.6-1.7B port (XHToken) — pure-attention hybrid SWA+full, no convs.
#
# 28 layers: sliding_attention ×3 + full_attention, repeating (full at
# 3,7,11,15,19,23,27). GQA 8Q/2KV heads, head_dim=256 (2048 = d_model).
# Fused HF q_k_v_proj (split by _remap_hf_keys), per-head sigmoid output
# gate (g_proj → attn.gate_proj, same semantics as LFM3's), GeGLU FFN
# (gelu(gate)*up). SWA window=512 theta=10K full rotary; full-attn
# theta=5M partial 0.25 in SLICE layout (rope_dim=64 on the FIRST head
# dims, tail pass-through — differs from LFM3 zero-tail pairing).
# Pre-norm RMSNorm eps=1e-6, final norm, NO embed norm, NO qk-norm,
# NO sandwich norm, NO biases. Tied embeddings, vocab 131072,
# bos=0 eos=1 pad=2. HF max_position 1M — RoPE tables clamp to 16384.
#
# Checkpoint: XHToken/Spark-X2.6-1.7B (spark2_5 arch, HF remote code).
# Keys remap via ModelLoader._remap_hf_keys (q_k_v_proj split is the only
# structural transform; everything else is a rename).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["spark_x26_1p7b"] = ModelConfig(
    vocab_size=131_072,
    d_model=2048,
    n_layers=28,
    n_heads=8,
    n_kv_heads=2,
    head_dim=256,
    intermediate_size=6656,
    attn_type="gqa",
    attn_bias=False,
    ffn_type="geglu",
    norm_type="rmsnorm",
    norm_eps=1e-6,
    use_embed_norm=False,
    use_final_norm=True,
    use_rope=True,
    rope_base=5_000_000.0,         # full-attention layers (SWA uses 10K)
    max_seq_len=16_384,
    use_qk_norm=False,
    tie_word_embeddings=True,
    layer_types=[
        "attention_swa", "attention_swa", "attention_swa", "attention",
        "attention_swa", "attention_swa", "attention_swa", "attention",
        "attention_swa", "attention_swa", "attention_swa", "attention",
        "attention_swa", "attention_swa", "attention_swa", "attention",
        "attention_swa", "attention_swa", "attention_swa", "attention",
        "attention_swa", "attention_swa", "attention_swa", "attention",
        "attention_swa", "attention_swa", "attention_swa", "attention",
    ],
    sliding_window=512,
    use_attn_out_gate=True,
    rope_layer_params={
        "attention":       {"rope_theta": 5_000_000.0,
                            "partial_rotary_factor": 0.25,
                            "partial_rotary_mode": "slice"},
        "attention_swa":   {"rope_theta": 10_000.0,
                            "partial_rotary_factor": 1.0},
    },
    batch_size=4,
    seq_len=2048,
    max_steps=500,
    warmup_steps=50,
    max_lr=2e-4,
    min_lr=2e-5,
)

# ──────────────────────────────────────────────────────────────────────────────
# LFM3-1.7B-D1: first zero-init graft round on the CC checkpoint.
#
# Derived from lfm3_1p7b (same checkpoint — lfm3_1p7b_cc). All grafts are
# gate=0 / skipped at init, so the forward pass is BIT-EXACT vs the base
# (verified: scripts/test_lfm3_d1_parity.py max|dlogit|=0.0 on the real
# ckpt, CPU fp32). 1.7288B -> 1.8029B (+73.9M dormant params).
#
# Grafts (all NEW keys, strictly additive):
#   GDN2 side-paths on conv layers i%4==1 [1,5,9,13,17,21] — each sits
#     immediately BEFORE a full-attn layer (2,6,10,14,18,22), so the
#     delta-rule global-memory consult lands right before each global read.
#     n_heads=8/head_dim=64 (budget ~30M total, vs 44M at inherit).
#   ValueResidual + LearnedSink + AttnRes on attention layers (gate=0).
#   CopySpan edit head + Verifier head (work-acceleration, gate=0).
#
# Hone plan: open gates early (~80% expand point, WSD schedule), muP-scaled
# LR on graft params. Known open items recorded in .devin/scratchpad.md
# (GDN2 pad_mask on x0 writes, AttnRes fixed k-buffer for capture, v0 donor).
# ──────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS["lfm3_1p7b_d1"] = ModelConfig(
    **{
        **MODEL_CONFIGS["lfm3_1p7b"].__dict__,
        "parent": "lfm3_1p7b",
        "dropped_keys": (  # D1 divergences from base (all NEW, gate=0)
            "use_gdn2",
            "gdn2_n_heads",
            "gdn2_head_dim",
            "gdn2_layers",
            "use_value_residual",
            "use_learned_sink",
            "use_attn_residual",
            "use_edit_head",
            "use_verifier_head",
        ),
        # ── D1 NEW: GDN2 side-paths before each full-attn layer ──
        "use_gdn2": True,
        "gdn2_n_heads": 8,
        "gdn2_head_dim": 64,
        "gdn2_layers": [1, 5, 9, 13, 17, 21],
        # ── D1 NEW: attention-side grafts (all zero-gated at init) ──
        "use_value_residual": True,
        "use_learned_sink": True,
        "use_attn_residual": True,
        # ── D1 NEW: RSI work-acceleration heads ──
        "use_edit_head": True,
        "use_verifier_head": True,
    }
)


def get_config(name: str | None = None, **overrides) -> ModelConfig:
    """Fetch a named config and apply optional overrides.

    Always returns a FRESH ModelConfig instance (never the shared preset),
    so callers can safely mutate fields (device, dtype, ...) without
    corrupting MODEL_CONFIGS for other users of the same preset.
    """
    if name is None:
        base = ModelConfig()
    else:
        if name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown config '{name}'. Available: {list(MODEL_CONFIGS)}")
        base = MODEL_CONFIGS[name]
    return ModelConfig(**{**base.__dict__, **overrides})


def validate_preset_lineage() -> list[str]:
    """Validate that every preset with a ``parent`` field documents all
    divergences from its parent in ``dropped_keys``.

    Returns a list of error messages (empty if all lineage is valid).
    Called by tests to enforce AGENTS.md directive A (build on the prior).

    A "divergence" is any field whose value in the child differs from the
    parent. Each divergence must be listed in ``dropped_keys`` (the name is
    kept for compatibility — it really means "changed/overridden keys").
    Undocumented divergences are potential silent regressions.
    """
    errors: list[str] = []
    for name, cfg in MODEL_CONFIGS.items():
        if cfg.parent is None:
            continue
        if cfg.parent not in MODEL_CONFIGS:
            errors.append(
                f"Preset '{name}' has parent='{cfg.parent}' but that preset "
                f"does not exist in MODEL_CONFIGS"
            )
            continue
        parent_cfg = MODEL_CONFIGS[cfg.parent]
        skip = {"parent", "dropped_keys"}
        # Find fields where child diverges from parent
        divergent: list[str] = []
        for key in parent_cfg.__dict__:
            if key in skip:
                continue
            parent_val = parent_cfg.__dict__[key]
            child_val = cfg.__dict__.get(key)
            if child_val != parent_val:
                divergent.append(key)
        # Keys explicitly documented as changed
        documented = set(cfg.dropped_keys)
        undocumented = set(divergent) - documented
        if undocumented:
            errors.append(
                f"Preset '{name}' (parent='{cfg.parent}') has undocumented "
                f"divergences from parent in keys: {sorted(undocumented)}. "
                f"Add them to dropped_keys, or align with parent."
            )
        # Check that dropped_keys are real field names
        valid_fields = set(parent_cfg.__dict__.keys()) - skip
        invalid_dropped = set(cfg.dropped_keys) - valid_fields
        if invalid_dropped:
            errors.append(
                f"Preset '{name}' lists dropped_keys "
                f"{sorted(invalid_dropped)} but these don't exist in parent "
                f"'{cfg.parent}'"
            )
    return errors


# Module-load enforcement (AGENTS.md directive A): any preset declaring a
# ``parent`` must document every divergence in ``dropped_keys``. This runs
# once at import so a silently-regressing preset fails loudly at import
# time instead of passing the suite unnoticed.
_LINEAGE_ERRORS = validate_preset_lineage()
if _LINEAGE_ERRORS:
    raise RuntimeError(
        "Preset lineage validation failed:\n" + "\n".join(_LINEAGE_ERRORS)
    )
