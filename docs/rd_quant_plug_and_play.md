# Plug-and-Play LLM Quantization with Maximum Memory Savings
**R44 research sweep (b) — 2026-09-05**
**Target: RTX 5070 12GB VRAM (Blackwell SM120) + 32GB system RAM**

Focus: PTQ methods that are **drop-in** (no retraining, minimal/no calibration) and
deliver the **lowest effective bit-width** (max memory savings). Ranked by
plug-and-play ease × memory savings × deployment readiness.

---

## Plug-and-Play Tiers (ease of deployment)

| Tier | Criteria | Methods |
|------|----------|---------|
| **T0 — Instant, data-free** | No calibration data at all; quantize in minutes | HQQ, bitsandbytes/NF4 |
| **T1 — Minimal calibration** | 128 samples, PTQ, HF-native, ready kernels | GPTQ, AWQ, AutoRound |
| **T2 — Extreme PTQ (<3 bit)** | More involved quantization but still PTQ, no retraining | VPTQ, AQLM, QuIP#, GPTVQ, SpQR, SqueezeLLM, LiftQuant |
| **T3 — Sub-1-bit** | Research/practical frontier | IQ1_S/M (llama.cpp), BTC-LLM, QMoE |

---

## T0 — Instant, Data-Free, Drop-In

### 1. HQQ (Half-Quadratic Quantization)
- **Bits**: 8, 4, 3, 2, 1 — full range
- **Calibration**: **NONE** — data-free, uses weights only
- **Speed to quantize**: <5 min for Llama-2-70B (50x faster than GPTQ)
- **Deployment**: HF Transformers native (`HqqConfig`), `torch.compile` compatible, Marlin/torchao fused kernels (200 tok/s on 4090 at 4-bit)
- **Quality**: Competitive with calibration-based methods at 4-bit; degrades gracefully at 2-bit
- **Recommended settings**: `nbits=4, group_size=64, axis=1` (fast inference) or `axis=0` (better quality, no fused kernel)
- **Why it's #1 for plug-and-play**: Zero calibration, any model (LLM/vision/audio), any bit-width, HF-native, PEFT/QLoRA compatible
- **Memory (7B model)**: 4-bit ≈ 4.3GB, 2-bit ≈ 2.2GB, 1-bit ≈ 1.2GB
- **Repo**: github.com/dropbox/hqq

### 2. bitsandbytes / NF4 (QLoRA)
- **Bits**: 8, 4 (NF4 = NormalFloat 4-bit)
- **Calibration**: NONE — data-free
- **Deployment**: HF Transformers native, the standard for QLoRA finetuning
- **Quality**: NF4 is near-lossless for finetuning (designed for LoRA base weights)
- **Memory (7B)**: NF4 ≈ 4GB
- **Limitation**: 4-bit only (no 2-bit/1-bit), not optimized for inference speed (designed for training)
- **Repo**: github.com/bitsandeth-org/bitsandbytes

---

## T1 — Minimal Calibration (128 samples), PTQ, HF-Native

### 3. GPTQ
- **Bits**: 2, 3, 4 (weight-only)
- **Calibration**: 128 samples from C4/wikitext (~5 min for 7B, ~4 hrs for 175B)
- **Deployment**: HF Transformers native (`GPTQConfig`), Marlin kernels, `gptqmodel` package
- **Quality**: SOTA at 4-bit among scalar methods; usable at 3-bit; 2-bit needs group_size tuning
- **Memory (7B)**: 4-bit ≈ 4.1GB, 3-bit ≈ 3.2GB, 2-bit ≈ 2.5GB
- **Key variants**: `--act-order` (quantize by decreasing activation magnitude), `--true-sequential`
- **Repo**: github.com/IST-DASLab/gptq (original), github.com/ModelCloud/gptqmodel (modern)

### 4. AWQ (Activation-aware Weight Quantization)
- **Bits**: 3, 4 (weight-only)
- **Calibration**: 128 samples
- **Deployment**: HF Transformers native, Best Paper MLSys 2024, TinyChat CUDA kernels (1.5-1.7x faster)
- **Quality**: Better than GPTQ at same bit-width for instruction-tuned models; protects salient channels
- **Memory (7B)**: 4-bit ≈ 4.3GB, 3-bit ≈ 3.3GB
- **Pre-quantized model zoo**: Llama-1/2/3, OPT, CodeLlama, StarCoder, Vicuna, VILA, LLaVA, DeepSeek-R1-Distill
- **Repo**: github.com/mit-han-lab/llm-awq

### 5. AutoRound
- **Bits**: 2, 3, 4
- **Calibration**: 128 samples, but uses sign-SGD to optimize rounding (not Hessian)
- **Deployment**: Part of gptqmodel, robust across model architectures
- **Quality**: More robust than GPTQ across diverse models; less sensitive to calibration data
- **Memory (7B)**: same as GPTQ
- **Repo**: github.com/intel/auto-round

---

## T2 — Extreme PTQ (<3 bit), Still No Retraining

### 6. VPTQ (Vector Post-Training Quantization) ⭐ TOP PICK for max savings + plug-and-play
- **Bits**: **1-2 bit** (extreme low-bit via vector quantization)
- **Calibration**: PTQ, no retraining (uses second-order optimization + channel-independent VQ)
- **Deployment**: **HF Transformers native** (since v4.48.0), `pip install vptq`, VPTQ-community model zoo on HF
- **Quality**: Reduces PPL by 0.01-0.34 (LLaMA-2), 0.38-0.68 (Mistral), 4.41-7.34 (LLaMA-3) vs SOTA at 2-bit. 1.6-1.8x inference throughput vs SOTA.
- **Scale**: 70B @ 2-bit, **405B @ <2-bit** (quantizes in ~17 hrs)
- **Memory (70B)**: 2-bit ≈ 18GB, 1-bit ≈ 9GB
- **Memory (7B)**: 2-bit ≈ 2.0GB, 1-bit ≈ 1.1GB
- **Why top pick**: Only sub-2-bit method that is (a) HF-native, (b) pip-installable, (c) has a model zoo, (d) no retraining. EMNLP 2024.
- **Repo**: github.com/microsoft/VPTQ

### 7. AQLM (Additive Quantization for LLMs)
- **Bits**: 2-3 (and **1-bit** with PV-Tuning finetuning)
- **Calibration**: PTQ + optional PV-Tuning finetuning (beyond STE)
- **Deployment**: HF Transformers native, pre-quantized models on HF (Llama-2-7b, Llama-3-8b, Mistral)
- **Quality**: Pareto-optimal below 3 bits. 1-bit AQLM (1x8 codebook, 256 entries) achieves WikiText PPL 7.85 on Llama-2-7b
- **Memory (7B)**: 2-bit ≈ 2.2GB, 1-bit ≈ 1.3GB
- **Limitation**: Quantization is slower than VPTQ (joint block optimization); decoding needs codebook lookup
- **Repo**: github.com/Vahe1994/AQLM

### 8. QuIP# (QuIP-sharp)
- **Bits**: 2-4 (SOTA at ≤4 bits)
- **Calibration**: PTQ + optional finetuning
- **Core innovation**: Randomized Hadamard transform (incoherence) + **E8 lattice codebooks** (optimal 8D ball packing) + finetuning
- **Quality**: SOTA weight-only PTQ at 2-4 bits. Llama-2-70B at 2-bit < 20GB.
- **Memory (7B)**: 2-bit ≈ 2.0GB, 3-bit ≈ 2.8GB
- **Limitation**: Codebook decode is slower than scalar on GPU (designed for quality, not max speed)
- **Repo**: github.com/Cornell-RelaxML/quip-sharp

### 9. GPTVQ
- **Bits**: 2-4
- **Calibration**: PTQ (GPTQ-style Hessian column updates + EM codebook init + SVD compression)
- **Core innovation**: VQ with **small per-block LUTs** — fast decode via CPU LUT instructions. 19% footprint reduction + 10% token rate vs INT4 on mobile.
- **Quality**: SOTA size-accuracy on Llama-v2/Mistral
- **Memory (7B)**: 2-bit ≈ 2.1GB
- **Repo**: github.com/Qualcomm-AI-research/gptvq

### 10. SpQR (Sparse-Quantized Representation)
- **Bits**: 3-4 (dense) + FP16 (sparse outliers)
- **Calibration**: PTQ (GPTQ-based, 128 samples)
- **Core innovation**: Isolates outlier weights → FP16 sparse; rest → 3-bit dense with group_size=16. **Near-lossless** (<1% PPL).
- **Quality**: 33B on single 24GB GPU with no degradation, 15% speedup vs FP16
- **Memory (7B)**: 3-bit ≈ 2.8GB (with outlier overhead)
- **Deployment**: HF Transformers native
- **Repo**: github.com/Vahe1994/spqr

### 11. SqueezeLLM
- **Bits**: 3 (dense+sparse)
- **Calibration**: PTQ
- **Core innovation**: Non-uniform K-means VQ for weights + FP16 for outliers (dense-and-sparse)
- **Memory (7B)**: 3-bit ≈ 2.9GB
- **Repo**: github.com/SqueezeAILab/SqueezeLLM

### 12. LiftQuant ⭐ NOVEL — Continuous Bit-Width
- **Bits**: **continuous 2.0-4.0** (not restricted to integers!)
- **Calibration**: PTQ (lift-then-project, no retraining needed for deployment; optional finetuning for benchmarks)
- **Core innovation**: "Lift-then-project" — represent d-dim weight vectors by projecting 1-bit lattice from D-dim lifted space. Effective bit-width = D/d (tunable). Decoding = linear transform + 1-bit quantizer (hardware-friendly).
- **Quality**: 70B @ 2.4-bit fits 24GB GPU, **outperforms all 2-bit baselines** on same device. 6.7x faster decode than FP16.
- **Memory (70B)**: 2.4-bit ≈ 21GB (fits 24GB), 3-bit ≈ 26GB
- **Why novel**: Breaks the rigid 2/3/4-bit ladder. Tune bit-width to *exactly* fit your VRAM budget. ICML 2026 Spotlight.
- **Deployment recommendation**: 30B models → 2.5-bit + block correction only; 7B/14B → 3-bit + block correction. (Finetuning helps benchmarks but may hurt chat quality.)
- **Repo**: github.com/Heliulu/LiftQuant

---

## T3 — Sub-1-Bit (Extreme Frontier)

### 13. IQ1_S / IQ1_M (llama.cpp) — Practical sub-2-bit
- **Bits**: IQ1_S = 1.56 bpw, IQ1_M = 1.75 bpw
- **Calibration**: Optional imatrix (importance matrix) — improves quality significantly
- **Deployment**: llama.cpp native, CPU/GPU/Metal, single self-contained GGUF file
- **Quality (Llama-2-7B)**: IQ1_S PPL 11.86, IQ1_M PPL 9.34 (vs FP16 ~5.47). Usable but degraded.
- **Quality (Llama-2-70B)**: IQ1_S PPL 5.21, IQ1_M PPL 4.83 (vs FP16 ~3.32). Much better at scale.
- **Memory (7B)**: IQ1_S ≈ 1.5GB, IQ1_M ≈ 1.7GB
- **Speed**: IQ1_S at 79.7 tok/s on Llama-3.1-8B (vs FP16 29.2 tok/s) — faster because less memory bandwidth
- **Why it matters**: The most practical sub-2-bit option today. Universal (CPU/GPU/Apple Silicon).

### 14. BTC-LLM — Sub-1-bit via Binary Codebook
- **Bits**: 0.7-1.11 bit
- **Calibration**: PTQ with learnable transformation (no full retraining)
- **Core innovation**: Binary codebook clusters recurring weight vectors + learnable transform to reduce outliers. Eliminates sparse masks (hardware-friendly).
- **Quality**: 0.8-bit on LLaMA-2-13B → only 3.1% accuracy drop in zero-shot. 1.6x speedup over FP16.
- **Status**: ACL 2026 paper, research stage — no SM120 kernel yet
- **Memory (13B)**: 0.8-bit ≈ 1.3GB

### 15. QMoE — Sub-1-bit for MoE
- **Bits**: <1 bit (0.8 bpw demonstrated)
- **Calibration**: PTQ, custom format + GPU decode kernels
- **Scale**: 1.6T SwitchTransformer → 160GB (20x compression)
- **Status**: MLSys 2024, practical but MoE-specific
- **Memory (1.6T MoE)**: 0.8-bit ≈ 160GB

---

## Practical Ecosystem Formats (not algorithms, but deployment-ready)

### GGUF k-quants / i-quants (llama.cpp)
The most universally deployed quant format. Single self-contained file, CPU/GPU/Metal/ROCm.

| Format | bpw | Size (7B) | PPL delta | Notes |
|--------|-----|-----------|-----------|-------|
| IQ1_S | 1.56 | 1.5 GB | very high | ternary, experimental |
| IQ1_M | 1.75 | 1.7 GB | high | improved 1-bit |
| IQ2_XXS | 2.06 | 2.0 GB | high | ultra-compressed |
| IQ2_S | 2.50 | 2.4 GB | moderate | 2-bit small |
| Q2_K | 2.96 | 2.8 GB | +3.52 | 2-bit k-quant |
| IQ3_XXS | 3.06 | 2.9 GB | moderate | 3-bit ultra-small |
| Q3_K_M | 3.74 | 3.5 GB | +0.66 | 3-bit balanced |
| IQ4_XS | 4.25 | 4.0 GB | low | 4-bit extra-small |
| Q4_K_M | 4.58 | 4.3 GB | +0.18 | **best balance** ⭐ |
| Q5_K_M | 5.33 | 5.0 GB | +0.06 | 5-bit balanced |
| Q6_K | 6.14 | 5.8 GB | +0.02 | 6-bit, near-lossless |
| Q8_0 | 8.50 | 8.0 GB | +0.003 | 8-bit, virtually lossless |

- **i-quants** (IQ2-IQ4): codebook + sign tricks, importance-matrix calibrated, better than k-quants at same bpw
- **k-quants** (Q2_K-Q6_K): super-block structure (256 elements), quantized scales, mixed precision across sub-blocks
- **RSF** (Refined Scale Fit): modern imatrix calibration applied to legacy k-quants → approaches AWQ quality
- **Deployment**: `llama-quantize` CLI, universal compatibility

### EXL2 (ExLlamaV2)
- **Bits**: non-integer 2.5, 3.0, 3.5, 4.0, 5.0 (per-layer variable bit allocation)
- **Deployment**: ExLlamaV2, fastest on NVIDIA consumer GPUs (15-30% edge over GPTQ/GGUF)
- **Quality**: Best quality-per-bit (per-layer bit allocation optimizes the global budget)
- **Calibration**: Required
- **Limitation**: NVIDIA CUDA only, no CPU/Apple Silicon

---

## Decision Matrix for RTX 5070 12GB

| Goal | Best choice | Effective bits | 7B memory | 13B memory | 70B memory |
|------|-------------|---------------|-----------|------------|------------|
| **Instant, no calibration, 4-bit** | HQQ | 4 | 4.3 GB | 7.5 GB | 38 GB (offload) |
| **Instant, no calibration, 2-bit** | HQQ | 2 | 2.2 GB | 3.9 GB | 19 GB (offload) |
| **Best 4-bit quality, minimal cal** | AWQ | 4 | 4.3 GB | 7.5 GB | 38 GB (offload) |
| **Best 3-bit quality, minimal cal** | GPTQ | 3 | 3.2 GB | 5.6 GB | 28 GB (offload) |
| **Max savings, HF-native, <2-bit** | **VPTQ** ⭐ | 2 | 2.0 GB | 3.5 GB | 18 GB |
| **Max savings, extreme 1-bit** | VPTQ / AQLM | 1 | 1.1 GB | 1.9 GB | 9 GB |
| **Continuous bit-width, fit exact VRAM** | **LiftQuant** ⭐ | 2.4-3.0 | 2.5 GB | 4.3 GB | 21 GB |
| **Near-lossless 3-bit** | SpQR | 3 | 2.8 GB | 4.9 GB | 25 GB |
| **SOTA 2-bit quality** | QuIP# | 2 | 2.0 GB | 3.5 GB | 18 GB |
| **Universal, CPU+GPU, sub-2-bit** | IQ1_M (GGUF) | 1.75 | 1.7 GB | 3.0 GB | 15 GB |
| **Fastest on NVIDIA, mixed bits** | EXL2 | 2.5-4.0 | 2.5-4.3 GB | 4.4-7.5 GB | 22-38 GB |
| **Sub-1-bit (research)** | BTC-LLM | 0.8 | 0.9 GB | 1.3 GB | 7 GB |

---

## Key Findings

### For maximum memory savings + plug-and-play on ForgeAI (RTX 5070 12GB):

1. **VPTQ is the clear winner for <2-bit plug-and-play**: HF-native, pip-installable, model zoo, no retraining, 70B@2bit=18GB, 405B@<2bit. EMNLP 2024. This is the method that makes 70B models fit on a single 24GB card at 2-bit with usable quality.

2. **LiftQuant is the novel breakthrough**: continuous bit-width means you can tune to *exactly* fit 12GB. 70B@2.4bit=21GB (fits 24GB with buffer). ICML 2026 Spotlight. Decoding is linear + 1-bit quantizer (SM120-friendly).

3. **HQQ is the zero-friction default**: no calibration at all, any bit-width 1-8, <5 min for 70B, torch.compile + Marlin kernels. Start here if you want instant results.

4. **IQ1_M (llama.cpp) is the practical sub-2-bit**: 1.75 bpw, universal (CPU/GPU/Metal), single GGUF file. Quality is degraded but usable at 70B+ scale. Fastest decode (less bandwidth).

5. **For near-lossless**: SpQR at 3-bit (<1% PPL) or Q4_K_M at 4.58 bpw (+0.18 PPL) are the safest high-compression options.

### What's NOT plug-and-play (excluded from this report):
- BitNet b1.58 / a4.8 — requires QAT (training from scratch or distillation)
- SpinQuant / OSTQuant / FlatQuant — rotation methods need learned transforms (calibration + optimization, not instant)
- OmniQuant — optimization-based PTQ (not instant, but no retraining)
- BitDistill — requires distillation finetuning

### SM120 (RTX 5070) specific notes:
- INT4 GEMM is fast via `mma.sync.aligned.m16n8k16.s32` (SM80 instruction set)
- FP4 (NVFP4/MXFP4) needs the blackwell-geforge patches — not as fast as on SM100
- ForgeQuant (already in ForgeAI) stores dense at INT4 + INT8 outlier rows — benefits via bandwidth (~0.6 B/weight GEMV read); its Triton kernel does NOT use mma.sync, so the tensor-core point applies to Marlin-class INT4 kernels, not FQ's current kernel
- For extreme <2-bit: VPTQ/LiftQuant/AQLM decode is codebook LUT lookup (not tensor-core GEMM), so SM120's lack of tcgen05 doesn't hurt them
- HQQ's Marlin/torchao fused kernels work on SM120 (SM80-compatible)
