# Evolve-loop research digest

Source: free API (keys from repo .env). One row per candidate technique; applicability notes tie each hit to the real forge/ function it could improve. Regenerate:
`python -m forge.evolution.research`

## top_p_sort — `research.sampling_utils.top_p_filter_logits / auto top-p path`
*query (free): fast top-p nucleus sampling kernel CUDA torch partial sort*

- **CUDA Toolkit Documentation** — https://docs.nvidia.com/cuda/
  Build, tune, and deploy accelerated applications with CUDA. Find installation instructions, launch highlights, programming guides, compiler documentation, API references, CUDA libraries, profiling tools, samples, and release notes for the CUDA Toolkit.
- **PyTorch Foundation - PyTorch** — https://pytorch.org/
  Scalable distributed training and performance optimization in research and production is enabled by the torch.distributed backend. Robust Ecosystem.CUDA 12.8. ROCm 10.0. CPU.
- **gpu - Multiple processes launching CUDA kernels in... - Stack Overflow** — https://stackoverflow.com/questions/14895034/multiple-processes-launching-cuda-kernels-in-parallel
  I am confused because the CUDA C programming guide says: "A kernel from one CUDA context cannot execute concurrently with a kernel from another CUDA context." This brings me to my second question, what are CUDA "contexts"?
- **Failed to build CUDA kernels for upfirdn2d. UserWarning: Failed to...** — https://github.com/NVlabs/stylegan2-ada-pytorch/issues/155
  Describe the bug Error message Failed to build CUDA kernels for upfirdn2d.

## min_k — `forge.engine.engine_common._min_k_filter`
*query (free): efficient top-k threshold filtering logits vocab GPU without full sort*

- **Efficient Knowledge Distillation for LLMs:Offline Top-K ...** — https://arxiv.org/html/2608.03796v1
  Aug 4, 2026 · Two systems choices carry most of the benefit: distil offline from cached top- K $K$ teacher logits to match online quality at lower memory and higher throughput, and use a fused chunked KL loss so that peak memory is linear in sequence length and long-context healing fits on a single 
- **[2608.03796] Efficient Knowledge Distillation for LLMs ...** — https://arxiv.org/abs/2608.03796
  Aug 4, 2026 · This recovery step largely decides the final quality, yet it is expensive. We present a practitioner's study of how to make distillation training efficient, organised around two systems contributions.
- **Sorting-Free GPU Kernels for LLM Sampling | FlashInfer** — https://flashinfer.ai/2025/03/10/sampling.html
  Mar 10, 2025 · In practice, the combination of Top-K and Top-P filtering is popular and used as the standard setting for LLM sampling. This allows for finer-grained control over the generation process.
- **flashinfer.sampling.top_k_top_p_sampling_from_logits** — https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_logits.html
  Fused GPU kernel for top-k and top-p sampling from pre-softmax logits, this operator implements GPU-based rejection sampling without explicit sorting. Check the blog post for more details.

## block_diag_mask — `forge.model.attention_ops._build_block_diag_causal_mask`
*query (free): vectorized block diagonal causal attention mask document packing varlen*

- **Packing, Masking, and Tokenization as a Model Interface** — https://thegustafson.com/blog/packing-and-masking
  Build a block-diagonal causal attention mask for packed sequences."Loss masking and packing masks are the same thing." They solve different problems. Packing masks (block-diagonal attention masks) prevent cross-contamination between documents during attention.
- **Attention/SDPA with both varlen and custom block-diagonal mask...** — https://discuss.pytorch.org/t/attention-sdpa-with-both-varlen-and-custom-block-diagonal-mask-support/224054
  Does any backend of SDPA or FlexAttention support both features at once: varlen packed inputs/outputs. custom block-diagonal mask: some blocks full, some blocks causal.
- **Attention Mechanisms | axolotl-ai-cloud/axolotl | DeepWiki** — https://deepwiki.com/axolotl-ai-cloud/axolotl/5.3-memory-optimizations
  This page documents the attention mechanism implementations supported in Axolotl, including Flash Attention (FA2, FA3, FA4), Flex Attention, SDPA, xFormers, SageAttention, and standard eager attention.
- **GitHub - thu-ml/SageAttention: [ICLR2025, ICML2025, NeurIPS2025...]** — https://github.com/thu-ml/SageAttention
  is_causal determines the use of a causal mask. Available APIssageattn_varlen: INT8 quantization for $QK^\top$ and FP16 for $PV$ using Triton backend. Support for varying sequence lengths within the same batch.

## causal_mask — `forge.model.attention_ops._causal_mask`
*query (free): causal attention mask construction fastest triu vs arange compare pytorch*

- **Should Transformer's causal attention mask be... - PyTorch Forums** — https://discuss.pytorch.org/t/should-transformers-causal-attention-mask-be-upper-triangular-or-lower-triangular/205396
  Hi, I am a bit confused about the way to implement a causal attention mask in Transformer, specifically, should “-inf” be placed in the upper or lower triangular part of the attention matrix. In PyTorch’s recommanded wa…
- **MARBLE: exact sequence parallelism for packed multimodal masks** — https://lumalabs.ai/news/packed-multimodal-attention
  We compare our fastest exact configuration against each baseline's fastest across 48 (model, layout, scale) cases with a FlashMask + Ulysses result. We are faster in 47 and equal in one, and FlashMask CP is slower than us in all 43 cases it can run.
- **range() vs. arange() in Python** — https://www.stechies.com/range-vs-arangein-python/
  arange() is fast and efficient if used with the NumPy data structure. Conclusion: When dealing with large data sets, range() function will be less efficient as compared to arange(). This is because the arange() takes much lesser memory than that of the built-in range() function.
- **GitHub - pytorch/pytorch: Tensors and Dynamic neural networks in...** — https://github.com/pytorch/pytorch
  Hence, PyTorch is quite fast — whether you run small or large neural networks. The memory usage in PyTorch is extremely efficient compared to Torch or some of the alternatives.Currently, VS 2017 / 2019, and Ninja are supported as the generator of CMake.

## w8a8_act — `forge.engine.quant.w8a8_quant.w8a8_quantize_activation`
*query (free): per-token dynamic INT8 activation quantization kernel absmax torch._int_mm*

- **ComfyUI-INT8-Fast/int8_quant.py at main...** — https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/int8_quant.py
  Custom node to load Flux2 in INT8 for 2X Speed gains on 30 series cards. - ComfyUI-INT8-Fast/int8_quant.py at main · BobJohnson24/ComfyUI-INT8-Fast.
- **INT8 Quantization: Absmax - Interactive** — https://mbrenndoerfer.com/writing/int8-quantization-absmax-smooth-quantization-implementation
  Covers INT8 weight quantization with absmax and smooth quantization techniques. Solve the outlier problem in large language models.
- **JongYeop/Llama-3.1-8B-Instruct-INT8-W8A8-Dynamic-Per-Token...** — https://huggingface.co/JongYeop/Llama-3.1-8B-Instruct-INT8-W8A8-Dynamic-Per-Token
  Activation Precision: INT8 (8-bit integer), dynamic per-token quantization. Scale computed at runtime: absmax / 127.0 per token (row). No activation scales stored in checkpoint.
- **Realizing Native INT8 Compute for Diffusion Transformers...** — https://arxiv.org/html/2606.14598
  int32 accumulation is bit-exact against torch._int_mm and the dequantized kernel output matches the reference at cosine similarity. 1.01.0. with no NaNs, with a per-GEMM speedup of.Against the torch._int_mm integer reference it matches exactly: cosine similarity. 1.01.0.

## exp_discretize — `forge.engine.mamba3.exp_trapezoidal_discretize`
*query (free): mamba SSM discretization numerical stability expm1 fused kernel complex*

- **Linux Kernel TCP Flaw With 9.8 CVSS Score Affects Ubuntu, Red Hat...** — https://thehackernews.com/2026/10/linux-kernel-tcp-flaw-with-98-cvss.html
  The kernel's own CVE team rates it 9.8 out of 10 on the CVSS scale. Anthropic said on October 27 that its AI model wrote an exploit that uses the flaw to let a remote attacker with no credentials run commands as root on Linux servers with a multipath TCP listener.
- **Mamba je besplatan vebsajt za zabavljanje i najveći sajt za besplatno...** — https://www.mamba.ru/sr/rating
  Mamba je najpopularniji besplatni vebsajt za zabavljanja u Rusiji i CIS. Povežite se sa interesantnim ljudima za ćaskanje, igranje i flertovanje.
- **torch.Tensor.expm1_ — PyTorch 2.6 documentation** — https://pytorch.org/docs/2.6/generated/torch.Tensor.expm1_.html
  Multiprocessing best practices. Numerical accuracy.Tensor.expm1_() → Tensor¶. In-place version of expm1().
- **kernels-community/mamba-ssm · Hugging Face** — https://huggingface.co/kernels-community/mamba-ssm
  This is the repository card of kernels-community/mamba-ssm that has been pushed on the Hub. It was built to be used with the kernels library. This card was automatically generated. How to use. # make sure `kernels` is installed: `pip install -U kernels` from kernels import get_kernel.

## complex_norm — `forge.engine.mamba3.complex_rmsnorm`
*query (free): complex valued RMSNorm implementation view_as_real optimization*

- **RMSNorm | Sebastian Raschka, PhD** — https://sebastianraschka.com/llms-from-scratch/ch04/09_rmsnorm/
  RMSNorm is one of the least flashy changes in modern LLMs, but it is also one of the most widespread. In practice, it largely displaced LayerNorm in current decoder stacks because it solves the same optimization problem with a slightly cheaper computation.
- **Implementation — RMSNorm: Efficient Normalization for Modern LLMs** — https://mbrenndoerfer.com/writing/rmsnorm-efficient-normalization-modern-llms/implementation
  Jun 5, 2025 · With the mathematical foundation established, let's translate our understanding into working code. We'll build RMSNorm from first principles, compare it with…
- **Normalization (RMSNorm) | pszemraj/decoder-pytorch-template ...** — https://deepwiki.com/pszemraj/decoder-pytorch-template/3.4.4-normalization-(rmsnorm)
  Oct 20, 2025 · This page documents the Root Mean Square Layer Normalization (RMSNorm) implementation used throughout the Llama transformer architecture. RMSNorm serves as the normalization layer applied before attention and feedforward sub-layers in the pre-normalization architecture pattern.
- **Understanding RMSNorm: My Notes on Faster Layer Normalization** — https://neuraforge.substack.com/p/understanding-rmsnorm-my-notes-on
  Jan 3, 2026 · This post will dive into the mathematical differences, implementation details, and invariance properties that allow RMSNorm to reduce latency while maintaining model performance.

## ngram_draft — `forge.engine.decoding._ngram_lookup`
*query (free): n-gram speculative decoding draft lookup suffix automaton prompt lookup decoding*

- **GitHub - apoorvumang/prompt-lookup-decoding: Simple speculative...** — https://github.com/apoorvumang/prompt-lookup-decoding
  Prompt Lookup Decoding. UPDATE 2: This method is now available in vLLM as well by setting speculative_model="[ngram]". UPDATE: This has been added to the transformers library. Prompt lookup exploits this pattern to speed up autoregressive d...
- **Speculative Decoding - vLLM** — https://docs.vllm.ai/en/latest/features/speculative_decoding/
  N-gram. Suffix decoding. Cross-Vocabulary Draft Models (TLI).To train your own draft models for optimized speculative decoding, see vllm-project/speculators for seamless training and integration with vLLM. vLLM Speculation Methods¶.
- **Accelerating LLM inference with speculative decoding: Lessons from...** — https://www.linkedin.com/blog/engineering/ai/accelerating-llm-inference-with-speculative-decoding-lessons-from-linkedins-hiring-assistant
  N‑gram speculative decoding: First introduced as prompt lookup decoding, this is a model-agnostic and purely statistical approach that uses patterns from the existing input to predict the next few tokens. Its drafting cost is low, and works best when outputs contain rephrasings or structured text.
- **Speculative Decoding Explained 2026 | packet.ai** — https://packet.ai/blog/speculative-decoding-explained
  N-gram speculative decoding (prompt lookup).Speculative decoding is a technique that speeds up LLM text generation by pairing a small, fast draft model with the large target model.

## jsd — `forge.engine.decoding._jsd (DoLa layer contrast)`
*query (free): vectorized Jensen Shannon divergence pytorch single pass softmax*

- **Jensen Shannon Divergence - PyTorch Forums** — https://discuss.pytorch.org/t/jensen-shannon-divergence/2626
  I am trying to implement Jensen Shannon Divergence (JSD) in Pytorch: class JSD(torch.nn.Module).Are you able to implement Jensen Shannon Divergence? If so kindly share the code. It will be really helpful.
- **Jensen-Shannon Divergence — PyTorch-Metrics 1.9.0 documentation** — https://lightning.ai/docs/torchmetrics/stable/regression/js_divergence.html
  js_divergence (Tensor): A tensor with the Jensen-Shannon divergence.val¶ (Union[Tensor, Sequence[Tensor], None]) – Either a single result from calling metric.forward or metric.compute or a list of these results.
- **Understanding Jensen-Shannon Divergence and Cross-Entropy Loss...** — https://www.educative.io/courses/getting-started-with-image-classification-with-pytorch/jensen-shannon-divergence-and-cross-entropy-loss
  Explore how PyTorch's JsdCrossEntropy loss function integrates Jensen-Shannon divergence with cross-entropy to measure differences in probability distributions. Understand its parameters and role in enhancing model learnability and robustness in image classification tasks.
- **JS Divergence in PyTorch: A Comprehensive Guide — codegenes.net** — https://www.codegenes.net/blog/js-divergence-pytorch/
  Jensen - Shannon (JS) divergence is one such measure that quantifies the similarity between two probability distributions. It is a symmetric and smoothed version of the Kullback - Leibler (KL) divergence.

## ece — `forge.engine.decision_head.expected_calibration_error`
*query (free): expected calibration error vectorized histogram bucketize pytorch*

- **Expected Calibration Error (ECE): A Step-by-Step Visual ...Calibration Error — PyTorch-Metrics 1.3.0 documentationCalibration Error — PyTorch-Metrics 0.8.0 documentationGitHub - Jonathan-Pearce/calibration-toolbox: NumPy library ...Pytorch model calibration - Uncategorized - PyTorch ForumsUnderstanding Model Calibration - A gentle introduction and ...torch.bucketize — PyTorch 2.14 documentation** — https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/
  Jul 12, 2023 · There are a few options, but the purpose and focus of this article is to explain and run through only one simple yet relatively sufficient measure for assessing model calibration: the Expected Calibration Error (ECE). The expected calibration error can be used to quantify how well a g
- **Calibration Error — PyTorch-Metrics 1.3.0 documentation** — https://torchmetrics.readthedocs.io/en/v1.3.0/classification/calibration_error.html
  The expected calibration error can be used to quantify how well a given model is calibrated e.g. how well the predicted output probabilities of the model matches the actual probabilities of the ground truth distribution.
- **Calibration Error — PyTorch-Metrics 0.8.0 documentation** — https://torchmetrics.readthedocs.io/en/v0.8.0/classification/calibration_error.html
  Computes the Top-label Calibration Error Three different norms are implemented, each corresponding to variations on the calibration error metric. L1 norm (Expected Calibration Error)
- **GitHub - Jonathan-Pearce/calibration-toolbox: NumPy library ...** — https://github.com/Jonathan-Pearce/calibration-toolbox
  Calibration Toolbox is a Python library for evaluating machine learning model calibration using binning-based metrics. The package provides a comprehensive collection of calibration metrics and visualization tools for research in deep learning and uncertainty quantification.

## kv_slice — `forge.engine.prefix_cache._slice_past_kv`
*query (free): KV cache slice clone memory efficiency prefix cache reuse LMCache*

- **LMCache – Building the foundation of AI memory tensor with KV...** — https://lmcache.ai/
  LMCache pioneers KV cache infrastructure for LLM inference, turning KV cache into AI-native memory that can be stored, compressed, searched, and reused across your entire cluster.Find reusable KV cache chunks beyond exact prefix matches with CacheBlend.
- **GitHub - LMCache/LMCache: LMCache: Supercharge Your LLM with...** — https://github.com/LMCache/LMCache
  Non-prefix KV reuse: Extend KV reuse beyond prefix caching by reusing cached KV blocks at any position in the prompt.LMCache builds on research in KV cache management, including cache reuse, offloading, compression, and serving optimization.
- **Meet LMCache: Supercharging vLLM with Lightning-Fast Inference** — https://www.linkedin.com/pulse/meet-lmcache-supercharging-vllm-lightning-fast-celia-lozano-grijalba-dbswf
  Traditional KV cache reuse only works with prefixes, and combining multiple KV caches naïvely breaks attention quality. This diagram shows how LMCache solves that through: KV Cache Store Stores cached representations (KV) and the original text chunks.
- **LMCache | NVIDIA Dynamo Documentation** — https://docs.nvidia.com/dynamo/v1.0.1/integrations/lm-cache
  LMCache is a high-performance KV cache layer that supercharges LLM serving by enabling prefill-once, reuse-everywhere semantics.This document describes how LMCache is integrated into Dynamo’s vLLM backend to provide enhanced performance and memory efficiency.

## stop_trunc — `forge.engine.engine_common._truncate_at_stop`
*query (free): streaming stop sequence detection token generation early truncation efficient*

- **Token indices sequence length is longer than the specified maximum...** — https://discuss.huggingface.co/t/token-indices-sequence-length-is-longer-than-the-specified-maximum-sequence-length-for-this-model/47083
  I am trying to use the pre-trained model - “openai/clip-vit-large-patch14” for generating the text embeddings (reference code attached below) and running to the following error: Token indices sequence length is longer …
- **ChromaDB Stop Sequence Trigger Failure In Development Sandbox...** — https://iposobo.com/chromadb-stop-sequence-trigger-development-sandbox/
  WARN: Truncation disabled – possible OOM; set CHROMA_TRUNCATE_ENABLED=true to enforce stop sequences. sequence_truncation_failed: token_id 12345 not in vocabulary. API responses that continue past the intended stop token, e.g.
- **truncation - Efficient random generation from truncated Laplace...** — https://stats.stackexchange.com/questions/357597/efficient-random-generation-from-truncated-laplace-distribution/357598
  We have several ways of drawing random samples from Laplace distribution. Is there any efficient way of sampling from left truncated Laplace distribution? Inverse transform sampling is an obvious.
- **Fixing the 'Token indices sequence length' Error in... | Error Notes** — https://errornotes.dev/en/errors/ai-tools/fixing-the-token-indices-sequence-length-error-in-huggingface
  Error Message. Token indices sequence length is longer than the specified maximum sequence length for this model (1523 > 512). Running this sequence through the model will result in indexing errors.
