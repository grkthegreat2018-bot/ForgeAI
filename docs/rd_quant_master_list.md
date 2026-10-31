# LLM Quantization Algorithm Master List
**R44 research sweep — 2026-09-05**
**Hardware target: RTX 5070 12GB VRAM (Blackwell SM120) + 32GB system RAM**

Compiled from: 8 web surveys (arXiv 2409.16694, 2502.13178, 2507.17417, 2505.05530,
LLMC EMNLP 2024, Springer 2026 survey, LLM Quantization Gallery, awesome-llm-paper
quantization/01-ptq-algorithms.md) + ForgeAI codebase inventory.

Algorithms already implemented in ForgeAI are marked **[IMPL]**.
Algorithms that are strong candidates for novel R&D on SM120 are marked **[R&D]**.

---

## A. Weight-Only PTQ — Compensation / Hessian-based (GPTQ family)

The foundational family. All share the idea: use the Hessian of per-layer
output reconstruction MSE to decide the *order* in which weights are quantized
and to *compensate* the remaining weights for the error introduced so far.

| # | Algorithm | Year | Core idea | Bits | Status |
|---|-----------|------|-----------|------|--------|
| 1 | **RTN** (Round-to-Nearest) | 2022 | Naive uniform rounding baseline. | any | [IMPL] (int4/int8 path) |
| 2 | **OBQ / OBS** | 2022 | Optimal Brain Surgeon for LLMs — quantize the weight whose removal least increases loss, update the rest. The direct ancestor of GPTQ. | 2-4 | — |
| 3 | **GPTQ** | 2022 | OBQ + Cholesky + batched columns — fast Hessian-inverse compensation. The workhorse. W3A16/W4A16. | 3-4 | partial (referenced) |
| 4 | **GPTQ-Zero** | 2023 | GPTQ with zeroed group scales for sparser quantization. | 3-4 | — |
| 5 | **DASH-Q** | 2024 | Direction-Aware and Static-Hessian Quantization — direction-aware Hessian that accounts for activation direction. | 2-4 | — |
| 6 | **ADMM-Q** | 2024 | Alternating Direction Method of Multipliers for the quantization objective — global optimum per layer. | 2-4 | — |
| 7 | **Drop-by-Drop** | 2024 | GPTQ + dropout-style stochastic regularization during quantization to improve generalization. | 3-4 | — |
| 8 | **SLQ** (Structured Low-rank Quantization) | 2024 | Low-rank Hessian approximation → scales GPTQ to 70B+. | 2-4 | — |
| 9 | **HARP** (Hessian-Aware Rounding & Packing) | 2024 | Per-row Hessian-aware rounding + optimal bit-packing. | 2-4 | — |
| 10 | **LQER** (Low-rank Quantization Error Reduction) | 2024 | W4A6/W4A8: low-rank FP matrix absorbs the quantization residual. | 4 | — |
| 11 | **LoRaQ** | 2024 | LoRA-style low-rank adapter that *is* the quantization error correction. | 3-4 | — |
| 12 | **MR-GPTQ** (Micro-Rotated GPTQ) | 2025 | GPTQ tailored to MXFP4/NVFP4 — block-wise Hadamard + format-specific rounding. Up to 3.6x layer speedup on B200, 6x on RTX5090. | 4 (FP4) | **[R&D]** — direct fit for SM120 |
| 13 | **AutoRound** | 2024 | Auto-rounding via gradient-based optimization of rounding vars (sign SGD), no Hessian inverse. Robust across models. | 2-4 | — |
| 14 | **RAMP** | 2024 | Rounding-Aware Min-Max Programming — optimizes clipping + rounding jointly. | 2-4 | — |
| 15 | **OSAQ** (Outlier-aware Sequential) | 2024 | Sequential quantization that explicitly protects outlier channels. | 3-4 | — |

## B. Weight-Only PTQ — Salience / Activation-aware (AWQ family)

| # | Algorithm | Year | Core idea | Bits | Status |
|---|-----------|------|-----------|------|--------|
| 16 | **AWQ** (Activation-aware Weight Quantization) | 2023 | Protect salient weight channels (those that matter for large activations) via per-channel scaling. W3A16/W4A16. | 3-4 | [IMPL] (referenced in adaptive_quant) |
| 17 | **OWQ** (Outlier Weighted Quantization) | 2024 | AWQ + per-channel outlier factor — outlier channels get FP16, rest quantized. | 3-4 | — |
| 18 | **HARP-AWQ** variant | 2024 | AWQ + Hessian-aware rounding. | 3-4 | — |
| 19 | **QAM-W** | 2024 | Quantization-Aware Minimization for Weights — joint clipping+rounding optimization. | 2-4 | — |
| 20 | **NormTweaking** | 2023 | Tweaks LayerNorm parameters (not weights) to compensate for quantization error — extremely cheap. | 3-4 | — |
| 21 | **AdaDim** | 2024 | Adaptive per-dimension clipping — different bit budget per channel based on sensitivity. | 2-4 | — |
| 22 | **DGQ** (Distribution-Guided Quantization) | 2024 | Uses weight distribution shape to pick per-channel quant grid. | 2-4 | — |
| 23 | **FAIR-Calib** | 2024 | Full-precision Activation-Informed Reconstruction calibration — better than AWQ's heuristic. | 3-4 | — |

## C. Weight-Only PTQ — Rotation / Incoherence (QuIP family)

The 2024-2025 frontier. Rotations make weights+activations "incoherent"
(flatter distribution) so uniform quantization works at much lower bits.

| # | Algorithm | Year | Core idea | Bits | Status |
|---|-----------|------|-----------|------|--------|
| 24 | **QuIP** (Quantization with Incoherence Processing) | 2023 | Pre/post random rotation (incoherence) + Hessian-aware rounding. W2A16. | 2-4 | — |
| 25 | **QuIP#** (QuIP-sharp) | 2024 | QuIP + randomized Hadamard (faster, better theory) + **E8 lattice codebooks** (optimal 8D ball packing) + finetuning. SOTA at ≤4 bits. | 2-4 | **[R&D]** — lattice VQ on SM120 |
| 26 | **QuaRot** | 2024 (NeurIPS) | Hadamard rotations enable W4A4KV4 near-lossless with plain RTN. Foundation for 2025 rotation work. | 4 | [IMPL] (QuaRotKV strategy) |
| 27 | **SpinQuant** | 2024 (ICLR 2025) | Learns rotation matrices via **Cayley optimization** on the Stiefel manifold. W4A4KV4 within 2.9 pts of FP. Shipped in Meta's quantized Llama 3.2. | 4 | **[R&D]** |
| 28 | **OSTQuant** | 2025 (ICLR) | Learnable **orthogonal + scaling** transforms + KL-Top loss. Retains 99.5% FP at W4. | 4 | **[R&D]** |
| 29 | **FlatQuant** | 2025 | Reduces affine-transform overhead via **Kronecker decomposition** of the learned transform. | 4 | **[R&D]** — low-overhead rotation |
| 30 | **DuQuant** | 2024 | Dual rotation + zigzag permutation to redistribute massive outliers across blocks. | 4 | — |
| 31 | **AffineQuant** | 2024 | Full **affine** (not just diagonal/rotation) transform with gradual-mask optimization for invertibility. SOTA at W4A4. | 4 | — |
| 32 | **ReSpinQuant** | 2025 | Decomposes SpinQuant rotation into low-rank + residual → slashes rotation overhead. | 4 | **[R&D]** |
| 33 | **ParoQuant** (Pairwise Rotation) | 2025 | Pairwise (2x2) rotation blocks — cheaper than full Hadamard. | 4 | **[R&D]** |
| 34 | **DartQuant** | 2025 | Calibration-only rotation learning at SpinQuant accuracy + QuaRot compute. | 4 | **[R&D]** |
| 35 | **DiRotQ** | 2025 | Diagonal + rotation decomposition of the transform. | 4 | — |
| 36 | **SmoothRot** | 2025 | Channel-wise scaling + Hadamard — handles massive outliers QuaRot misses. 10-30% gap closure. | 4 | **[R&D]** |
| 37 | **Trainable SmoothRot** | 2025 | SmoothRot with learned scaling factors (gradient-trained). | 4 | — |
| 38 | **ResQ** (Residual Quantization + Rotation) | 2024 | Reorder + rotation + residual codebook. | 4 | — |
| 39 | **TORQ** | 2025 | Token-aware rotation for activation quantization. | 4 | — |
| 40 | **ConQuR** | 2025 | Concurrent rotation for weights and activations jointly. | 4 | — |
| 41 | **InfoQuant** | 2025 | Information-theoretic activation distribution design — defines what distribution a low-bit quantizer can represent well. | 4 | **[R&D]** — novel theory |
| 42 | **MorphoQuant** | 2025 | Morphological transformation of activation distributions. | 4 | — |
| 43 | **SplitQ** | 2025 | Splits weight matrices into sub-blocks with independent rotations. | 2-4 | — |
| 44 | **RPTQ** (Range-aware PTQ) | 2024 | Range-aware per-channel rotation. | 4 | — |

## D. Weight-Only PTQ — Vector Quantization / Codebook (extreme ≤3 bit)

The extreme-compression frontier. Stores weight *vectors* (not scalars) as
codebook indices. Pareto-optimal below 3 bits.

| # | Algorithm | Year | Core idea | Bits | Status |
|---|-----------|------|-----------|------|--------|
| 45 | **SqueezeLLM** | 2023 | Dense-and-sparse: non-uniform K-means VQ for weights + FP16 for outliers. | 3 | — |
| 46 | **AQLM** (Additive Quantization for LLMs) | 2024 (ICML) | Multi-codebook additive quantization — input-adaptive, joint block optimization. SOTA at ≤3 bits. | 2-3 | — |
| 47 | **PV-Tuning** | 2024 (NeurIPS oral) | Beyond STE — proper gradient-based finetuning of quantized weights (used with AQLM). Enables 1-bit AQLM. | 1-2 | — |
| 48 | **GPTVQ** | 2024 | VQ + GPTQ-style Hessian column updates + EM codebook init + SVD codebook compression. Fast LUT decode on CPU/NPU. | 2-4 | **[R&D]** |
| 49 | **QTIP** | 2024 | Quantization with Triangular Iterative Processing — triangular codebook structure for fast decode. | 2-4 | — |
| 50 | **LFQ** (Lookup-Free Quantization) | 2024 | No codebook — implicit VQ via hashing. Memory-tight. | 1-2 | — |
| 51 | **EXL2** (ExLlamaV2) | 2024 | Practical mixed-bitweight format — per-layer variable bits, popular in exllama serving. | 2-8 | — |
| 52 | **BTC-LLM** | 2025 | Sub-1-bit via Bayesian ternary coding. Research stage, no SM120 kernel. | <1 | parked |
| 53 | **LiftQuant** | 2025 | Continuous bit-width via dimensional lifting + projection of 1-bit lattice. Tunable 2-4 bits. Pareto-optimal. | 2-4 | **[R&D]** — novel continuous-bw |
| 54 | **CAE / ResComp** (Compositional AE) | 2024 | Compositional autoencoder residual compression. | 2-3 | — |

## E. Weight+Activation PTQ — Outlier Mitigation (SmoothQuant family)

| # | Algorithm | Year | Core idea | Precision | Status |
|---|-----------|------|-----------|-----------|--------|
| 55 | **LLM.int8()** | 2022 | Mixed-precision: 99.9% int8 + 0.1% FP16 outlier columns. The original. | W8A8 | [IMPL] (int8 path) |
| 56 | **SmoothQuant** | 2022 (ICML 2023) | Migrate quantization difficulty from activations to weights via diagonal scaling s. W8A8 near-lossless. | W8A8 | [IMPL] (w8a8_quant) |
| 57 | **Outlier Suppression** | 2022 | LayerNorm scaling to suppress outliers. | W8A8 | — |
| 58 | **Outlier Suppression+** | 2023 | Shifting + scaling to align asymmetric channel centers. | W8A8 | — |
| 59 | **ZeroQuant** | 2022 | Per-token + per-group dynamic quantization for W&A. | W4A8 | — |
| 60 | **ZeroQuant-V2** | 2023 | + Low-rank compensation (LoRC) for the residual. | W2-4A16 | — |
| 61 | **ZeroQuant-FP** | 2023 | FP8/FP4 variants of ZeroQuant. | W4A8 (FP) | — |
| 62 | **ZeroQuant(4+2)** | 2024 | Hybrid 4-bit + 2-bit per-channel. | W6 effective | — |
| 63 | **Atom** | 2024 (MLSys) | W3A3/W4A4: dynamic channel reorder + mixed-precision (outliers high-bit) + grouped quant + quantized KV. | W4A4 | **[R&D]** |
| 64 | **QServe / QoQ** | 2025 (MLSys) | W4A8KV4: progressive quantization + SmoothAttention for KV4 + register-level dequant. 1.2-3.5x throughput. | W4A8KV4 | **[R&D]** — strong fit |
| 65 | **OmniQuant** | 2023 (ICLR 2024) | Learnable clipping (LWC) + learnable weight clipping (LWC) + smoothing (LES) — optimization-based PTQ. W2-4A4-8. | W2-4A4-8 | — |
| 66 | **QLLM** | 2023 | Quality-aware LLM quantization — outlier-aware + reconstruction. | W4A4 | — |
| 67 | **OSC** (Outlier-Shift Calibration) | 2024 | Shifts outliers to reduce their quantization impact. | W4A8 | — |
| 68 | **Shift-and-Sum** | 2024 | Alternating shift + sum to balance channels. | W4A4 | — |
| 69 | **STaR-Quant** | 2024 | Synthetic-template-aware reconstruction for activation quant. | W4A4 | — |
| 70 | **HQQ** (Half-Quadratic Quantization) | 2023 | Half-quadratic splitting — alternates weight quantization + outlier optimization. Training-free, data-free. | 2-8 | — |
| 71 | **SpQR** | 2023 | Sparse-Quantized Representation — dense int3 + sparse FP16 outlier matrix. | 3 | — |
| 72 | **QUIK** | 2023 | Quantization of weights + integer activations + outlier handling. | W4A8 | — |
| 73 | **GGUF k-quants / i-quants** | 2023 | llama.cpp formats — k-quants (Q4_K, Q5_K...) with block scales; i-quants (importance-aware). | 2-8 | — |
| 74 | **FP6-LLM / Quant-LLM FP6** | 2024 | Native FP6 format with custom kernels — better than INT4 at similar size. | 6 | — |
| 75 | **WINDQuant** | 2024 | Wavelet-inspired decomposition for quantization. | 4 | — |
| 76 | **XFP** | 2024 | Cross-format FP quantization. | 4-8 | — |
| 77 | **GSQ** (Grouped Sparse Quantization) | 2024 | Group quant + sparsity. | 3-4 | — |

## F. QAT / Training-Time Quantization (BitNet family)

| # | Algorithm | Year | Core idea | Bits | Status |
|---|-----------|------|-----------|------|--------|
| 78 | **BitNet** (original 1-bit) | 2023 | Binary {-1,+1} weights via STE during pretraining. | 1 | — |
| 79 | **BitNet b1.58** | 2024 | Ternary {-1,0,+1} + absmean scale. Matches FP16 at 3B+. Add-only GEMM. | 1.58 | **[IMPL]** (bitnet_b158_key) |
| 80 | **BitNet a4.8** | 2024 | + 4-bit activations (hybrid quant+sparsify intermediate states) + 8-bit FFN out + 3-bit KV. 55% params active. | 1.58W/4A | **[IMPL]** (int8 ternary kernel) |
| 81 | **BitNet b1.58 2B4T** | 2025 | First open-source native 1.58-bit LLM at 2B scale (4T tokens). | 1.58 | — |
| 82 | **BitDistill** | 2025 | Distill FP LLM → 1.58-bit for downstream tasks. SubLN + multi-head attn distill + continual pretrain. 10x memory save. | 1.58 | **[R&D]** |
| 83 | **LLM-QAT** | 2023 | QAT with data-generated (self-generated) training data. | 4 | — |
| 84 | **QA-LoRA** | 2023 | Quantize weights to int4 + train LoRA in FP — only LoRA needs grad. | 4 | — |
| 85 | **QLoRA / NF4** | 2023 | NormalFloat 4-bit (NF4) for LoRA base weights + BF16 compute. The standard for memory-efficient finetuning. | 4 | — |
| 86 | **BitDistill-style continual pretrain** | 2025 | Warm-up continual pretrain to bridge FP→1.58 gap. | 1.58 | — |
| 87 | **BitNet + Residual** | 2025 | Ternary + element-level dense residual for the largest errors. | 1.58+ε | **[IMPL]** (bitnet_residual_key) |
| 88 | **TernaryVit / Ternary weights survey** | 2024 | Bottom-up exploration of when 1.58 bits suffice. | 1.58 | — |

## G. FP4 / FP8 / Microscaling Formats (hardware-native)

| # | Algorithm | Year | Core idea | Format | Status |
|---|-----------|------|-----------|--------|--------|
| 89 | **FP8 E4M3 / E5M2** | 2022 | Hardware-native 8-bit float (Hopper+). Near-lossless. | 8 | **[IMPL]** (fp8_infer) |
| 90 | **MXFP4** | 2024 | Microscaling FP4 — E2M1 + 32-element blocks + E8M0 power-of-2 scale. OpenAI gpt-oss format. | 4.25 | **[R&D]** |
| 91 | **NVFP4** | 2025 | NVIDIA FP4 — E2M1 + 16-element blocks + E4M3 block scale + FP32 tensor scale. 2-level scaling. Blackwell-native. 9000 TFLOPS on B200. | 4.5 | **[IMPL]** (nvfp4_quant) |
| 92 | **MR-GPTQ for FP4** | 2025 | (see #12) GPTQ adapted to MXFP4/NVFP4 block structure. | 4 | **[R&D]** |
| 93 | **AdaptScale FP4 (AS-FP4)** | ForgeAI R14 | MSE-optimal per-block FP4 scale (grid search). ~30% lower error than absmax. | 4 | **[IMPL]** (novel_quant) |
| 94 | **ResidualFP4 (R-FP4)** | ForgeAI R14 | FP4 + sparse INT8 residual for top-k errors. Near-FP8 at FP4 cost. | 4+ε | **[IMPL]** (novel_quant) |
| 95 | **IRI-FP4** | ForgeAI | Iterative Residual Refinement FP4. | 4 | **[IMPL]** (iri_fp4_key) |
| 96 | **AdaMX** | ForgeAI | Adaptive Microscaling + SharQ + MosaicQuant fusion. | 4 | **[IMPL]** (adaptive_quant) |

## H. KV Cache Quantization

| # | Algorithm | Year | Core idea | Bits | Status |
|---|-----------|------|-----------|------|--------|
| 97 | **KIVI** | 2024 (ICML) | 2-bit KV: keys per-channel, values per-token (asymmetric). Tuning-free. 2.6x memory, 4x batch. | 2 | **[IMPL]** (kv_2bit, NSNQuant) |
| 98 | **KVQuant** | 2024 (NeurIPS) | Per-channel pre-RoPE keys + Non-Uniform Quant (NUQ) + dense-and-sparse. 10M context on 8 GPUs. | 2-3 | **[IMPL]** (kv_compress) |
| 99 | **Atom KV** | 2024 | 4-bit KV cache as part of Atom W4A4. | 4 | — |
| 100 | **FlexGen KV** | 2023 | 4-bit KV + offload to CPU/disk for batch throughput. | 4 | — |
| 101 | **SmoothAttention (QServe)** | 2025 | Smooths K cache outliers for 4-bit KV quant. | 4 | **[R&D]** |
| 102 | **RotorQuant** | ForgeAI | Block-diagonal rotation for KV compression. | 4 | **[IMPL]** (rotorquant) |
| 103 | **HyQuant** | ForgeAI R32 | Pattern-aware KV quant (vertical lines + attention patterns). | 2-4 | **[IMPL]** (hyquant_kv) |
| 104 | **XQuant** | ForgeAI | KV rematerialization — cache activations, recompute K/V. | — | **[IMPL]** (xquant_kv) |
| 105 | **SnapKV / S4R** | 2024 | KV eviction (not strictly quant) — pairs with quant. | — | **[IMPL]** (kv strategies) |
| 106 | **H2O / Heavy-Hitter Oracle** | 2023 | KV eviction by attention scores. | — | **[IMPL]** (kv_compress) |
| 107 | **MatryoshkaKV** | 2025 | Matryoshka representation for KV — 60% compression, attention-only. | — | planned (scratchpad) |

## I. MoE-Specific Quantization

| # | Algorithm | Year | Core idea | Bits | Status |
|---|-----------|------|-----------|------|--------|
| 108 | **QMoE** | 2024 (MLSys) | Sub-1-bit compression for trillion-param MoEs — custom format + GPU decode kernels. 1.6T → 160GB. | <1 | — |
| 109 | **EAQuant** | 2025 | Expert-aware smoothing + router logits alignment + expert calibration balance. SOTA for MoE W4A4. | 4 | **[R&D]** |
| 110 | **MoEQuant** | 2025 (ICML) | Expert-Balanced Self-Sampling + Affinity-Guided Quant — solves inter/intra-expert imbalance. | 2-4 | **[R&D]** |
| 111 | **MxMoE** | 2025 (ICML) | Mixed-precision per expert (sensitivity + activation freq) + auto GroupGEMM kernels. 3.4x over FP. | mixed | **[R&D]** |
| 112 | **DynaExq** | 2025 | Runtime hotness-aware dynamic expert precision switching + async pipeline + fragmentation-free pool. | dynamic | **[R&D]** — fits 12GB MoE |

## J. SSM / Mamba-Specific Quantization (CRITICAL for hybrid models)

> **Rule from scratchpad**: NEVER apply attention-optimized extreme PTQ
> (AQLM/QuIP#/BitNet) to Mamba/SSM blocks. Naive W1.58 PTQ on Mamba gives
> PPL ~13M. Use SSM-specific quantizers.

| # | Algorithm | Year | Core idea | Bits | Status |
|---|-----------|------|-----------|------|--------|
| 113 | **Quamba2** | 2024 | W4A8 for SSM — handles Mamba's state recurrence (not outlier-driven). | W4A8 | **[IMPL]** (quamba2) |
| 114 | **SSDi8** | 2024 | 8-bit SSM-specific quantization. | 8 | — |
| 115 | **CAJQ** | 2025 | Mixed-precision quant for SSM+attn hybrid — 4.4x compression. | mixed | planned (scratchpad) |

## K. ForgeAI Novel / Hybrid Schemes (in-codebase R&D)

| # | Algorithm | Year | Core idea | Bits | Status |
|---|-----------|------|-----------|------|--------|
| 116 | **ForgeQuant** | R32 | INT4 dense + INT8 outlier rows (top-10% by row L2 norm, RTN, no calibration; OWQ/SpQR/GRINQH-class prior art). ~4.9 eff bits — NOT 3.6 as originally logged. | 4.9 | **[IMPL]** (forge_quant) |
| 117 | **GRINQH** | R32 | Graded Input-Based Quantization Hierarchy — dynamic per-channel precision. | dynamic | **[IMPL]** (grinqh) |
| 118 | **MixLLM** | R32 | Global mixed-precision across output features. | mixed | **[IMPL]** (mixllm) |
| 119 | **ACBQ** | R32 (ACL 2026) | Adaptive Cross-Block Quantization. | mixed | **[IMPL]** (acbq) |
| 120 | **FusedGEMM key** | ForgeAI | Fused QKV + Gate-Up GEMM for quantized paths. | — | **[IMPL]** (fused_gemm_key) |

---

## Summary statistics
- **Total algorithms catalogued: 120** (across 11 categories)
- **Already implemented in ForgeAI: ~35** (marked [IMPL])
- **Strong R&D candidates for SM120: ~20** (marked [R&D])
- **Parked / research-stage: 1** (BTC-LLM)

## Cross-cutting axes (every algorithm sits on these)
1. **Bit-width**: 1 (binary) → 1.58 (ternary) → 2-3 (extreme) → 4 (mainstream low) → 8 (safe) → 16 (baseline)
2. **Weight / Activation / KV**: W-only, W+A, W+A+KV, KV-only
3. **Granularity**: per-tensor / per-channel / per-group / per-block / per-token / per-element
4. **Symmetry**: symmetric vs asymmetric
5. **Transform**: none / scaling (SmoothQuant) / shifting (OS+) / rotation (QuaRot) / affine (AffineQuant) / codebook (AQLM)
6. **Calibration**: data-free (HQQ) / few-sample (GPTQ) / reconstruction (OmniQuant) / finetuning (PV-Tuning, SpinQuant)
7. **Hardware format**: INT / FP / microscaling (MXFP4, NVFP4) / lattice (QuIP#)
8. **Training stage**: PTQ / QAT / distillation (BitDistill) / LoRA-quant (QA-LoRA, QLoRA)

## Key 2025 trends (from the surveys)
1. **Rotation is the dominant 2024-2025 technique** — QuaRot → SpinQuant → OSTQuant → FlatQuant → ReSpinQuant → ParoQuant → DartQuant. The frontier is *cheaper/better rotations*.
2. **FP4 microscaling (NVFP4/MXFP4) is the hardware format of 2025-2026** — Blackwell-native, but needs format-specific quant (MR-GPTQ) to beat INT4.
3. **Vector quantization wins below 3 bits** — AQLM + PV-Tuning + GPTVQ + LiftQuant. Lattice codebooks (QuIP# E8) are SOTA.
4. **Mixed-precision per-expert for MoE** — MxMoE, MoEQuant, DynaExq. Hotness-aware dynamic precision is new.
5. **Continuous bit-width** (LiftQuant) is a 2025 novelty — breaks the rigid 2/3/4-bit ladder.
6. **Information-theoretic quantizer design** (InfoQuant) — defines the target distribution, not just the transform.
7. **SM120 (RTX 5070) is NOT SM100** — uses SM80-era mma.sync, no tcgen05, no TMEM, 99KB SMEM. INT4 is fast; FP4 needs the blackwell-geforce patches. ForgeQuant already exploits this.
