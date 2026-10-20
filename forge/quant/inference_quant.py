"""Inference weight-only quantization (INT8/INT4) for speed.

Weight-only quant: weights in INT8/INT4, activations in BF16.
Speedup comes from reduced memory bandwidth (small models are memory-bound).
2x speedup with INT8, ~3x with INT4 (on memory-bound workloads).

Usage:
    from forge.quant.inference_quant import quantize_model_int8, quantize_model_int4

    # INT8 weight-only quantization (2x speedup, <1% quality loss)
    quantize_model_int8(model)

    # INT4 weight-only quantization (3x speedup, ~1-2% quality loss)
    quantize_model_int4(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.quant.protocol import QuantizedLinearMixin


class QuantizedLinear(QuantizedLinearMixin):
    """Weight-only quantized Linear layer for inference.
    
    Weights are stored as int8/int4 with per-channel scales.
    Forward dequantizes on-the-fly: y = x @ (scale * dequant(W)).T + b
    
    Args:
        original: the nn.Linear to quantize
        bits: 8 or 4
        group_size: quantization group size (smaller = more accurate, more overhead)
                    None = per-channel (one scale per output row)
    """

    def __init__(self, original: nn.Linear, bits: int = 8, group_size: int | None = None):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bits = bits
        self.group_size = group_size
        self.bias = original.bias

        # Quantize weights.
        W = original.weight.data.float()  # (out, in)

        if group_size is None:
            # Per-channel quantization (one scale per output row).
            scales, q_w = self._quantize_per_channel(W, bits)
        else:
            # Group-wise quantization.
            scales, q_w = self._quantize_grouped(W, bits, group_size)

        # Store as registered buffers (move with model.to()).
        self.register_buffer('q_weight', q_w)
        self.register_buffer('scales', scales)
        if self.bias is not None:
            self.register_buffer('bias_weight', self.bias.data)

    def _quantize_per_channel(self, W: torch.Tensor, bits: int):
        """Per-channel quantization: one scale per output row."""
        max_val = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        qmax = (2 ** (bits - 1)) - 1  # symmetric
        scales = max_val / qmax
        q_w = torch.round(W / scales).clamp(-qmax, qmax).to(torch.int8 if bits == 8 else torch.float16)
        return scales.squeeze(1), q_w

    def _quantize_grouped(self, W: torch.Tensor, bits: int, group_size: int):
        """Group-wise quantization along input dimension."""
        out_f, in_f = W.shape
        # Pad to multiple of group_size.
        pad = (group_size - in_f % group_size) % group_size
        if pad > 0:
            W = F.pad(W, (0, pad))
        n_groups = W.shape[1] // group_size

        W_grouped = W.view(out_f, n_groups, group_size)
        max_val = W_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
        qmax = (2 ** (bits - 1)) - 1
        scales = max_val / qmax  # (out, n_groups, 1)
        q_w = torch.round(W_grouped / scales).clamp(-qmax, qmax)

        return scales.view(out_f, n_groups), q_w.view(out_f, -1).to(torch.float16)

    def _dequantize(self) -> torch.Tensor:
        """Dequantize weights to float16/bfloat16 (not float32 to save memory)."""
        if self.group_size is None:
            # Per-channel: dequantize directly to bf16.
            return self.q_weight.to(torch.bfloat16) * self.scales.unsqueeze(1).to(torch.bfloat16)
        else:
            # Group-wise.
            out_f = self.out_features
            in_f_padded = self.q_weight.shape[1]
            n_groups = self.scales.shape[1]
            group_size = in_f_padded // n_groups
            q = self.q_weight.view(out_f, n_groups, group_size).to(torch.bfloat16)
            return (q * self.scales.unsqueeze(2).to(torch.bfloat16)).view(out_f, in_f_padded)[:, :self.in_features]

    def forward(self, x):
        # Dequantize weights directly to input dtype (avoid float32 intermediate).
        # Cache the dequantized weight for inference (weights don't change).
        # After caching, we can free the quantized weights to save VRAM.
        if not self.training:
            if not hasattr(self, '_cached_weight') or self._cached_weight is None:
                self._cached_weight = self._dequantize().to(x.dtype)
                # Free quantized weights to save VRAM (cache replaces them).
                self.q_weight = None
                self.scales = None
            weight = self._cached_weight
        else:
            weight = self._dequantize().to(x.dtype)
        return F.linear(x, weight, self.bias_weight.to(x.dtype) if self.bias is not None else None)

    def __repr__(self):
        return f"QuantizedLinear(in={self.in_features}, out={self.out_features}, bits={self.bits})"


class FastINT8Linear(QuantizedLinearMixin):
    """FP8 weight-only Linear using torch._scaled_mm (Blackwell-native FP8 matmul).

    Stores weights as FP8 (e4m3fn) + per-tensor fp32 scale. Forward quantizes
    the bf16 activation to FP8 dynamically and calls torch._scaled_mm, which
    performs the matmul directly in FP8 on Tensor Cores — NO dequantization
    step. This avoids the dequant+matmul overhead of QuantizedLinear that
    makes INT8 slower than bf16 on RTX 5070 (Blackwell, fast bf16 matmul).

    Note: torch._scaled_mm supports FP8 (not INT8) on Blackwell.
    Falls back to dequant + F.linear on CPU / when _scaled_mm is unavailable.
    """

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bias = original.bias

        # Per-tensor symmetric quantization of weight [out, in] to FP8 e4m3fn.
        W = original.weight.data.float()
        max_val = W.abs().amax().clamp(min=1e-8)
        scale = (max_val / 448.0).to(torch.float32)  # e4m3fn max = 448
        # Quantize to FP8 e4m3fn
        q_w = (W / scale).to(torch.float8_e4m3fn)
        self.register_buffer('q_weight', q_w, persistent=False)
        self.register_buffer('w_scale', scale, persistent=False)
        if self.bias is not None:
            self.register_buffer('bias_weight', self.bias.data.clone(), persistent=False)

        self._use_scaled_mm = (
            hasattr(torch, '_scaled_mm')
            and torch.cuda.is_available()
        )

    def _get_dequantized_weight(self):
        """Get BF16 weight from FP8 for INT4 requantization."""
        return self.q_weight.to(torch.float32) * self.w_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        bias = self.bias_weight if self.bias is not None else None

        if self._use_scaled_mm and x2d.is_cuda and self.q_weight.is_cuda:
            # Dynamic per-tensor activation scale for FP8 e4m3fn.
            x_scale_val = (x2d.abs().amax().float() / 448.0).clamp(min=1e-12)
            x_fp8 = (x2d.float() / x_scale_val).to(torch.float8_e4m3fn)
            # _scaled_mm(a, b): a [M, K] row-major fp8, b [K, N] col-major fp8.
            out = torch._scaled_mm(
                x_fp8, self.q_weight.T,
                scale_a=x_scale_val.to(torch.float32),
                scale_b=self.w_scale,
                out_dtype=torch.bfloat16,
            )
            if bias is not None:
                out = out + bias.to(out.dtype)
            return out.reshape(*orig_shape[:-1], self.out_features)

        # Fallback: dequant weight to bf16, standard matmul (CPU / no _scaled_mm).
        w_bf16 = self.q_weight.to(torch.float32) * self.w_scale.to(torch.float32)
        w_out = w_bf16.to(x.dtype)
        bias_out = bias.to(x.dtype) if bias is not None else None
        return F.linear(x, w_out, bias_out).reshape(*orig_shape[:-1], self.out_features)

    def __repr__(self):
        return f"FastINT8Linear(in={self.in_features}, out={self.out_features}, fp8_scaled_mm)"


def quantize_model_int8(model, target_modules=None, verbose=True, fast=False):
    """Replace Linear layers with INT8 weight-only quantized versions.
    
    Args:
        model: the model
        target_modules: list of module name substrings to target.
                       Default: all Linear layers except embeddings and lm_head.
        verbose: print stats
        fast: if True, use FastINT8Linear (torch._scaled_mm, no dequant overhead).
              Recommended on Blackwell (RTX 5070) where bf16 matmul is fast and
              dequant overhead makes standard INT8 slower than bf16.
    
    Returns:
        number of layers quantized
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj',
                         'w1', 'w2', 'w3', 'w_gate', 'w_up', 'w_down',
                         'fc1', 'fc2', 'qkv_proj',
                         'kv_down_proj', 'k_up_proj', 'v_up_proj',
                         'in_proj']

    n_quantized = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Check if name matches any target.
            if any(t in name for t in target_modules):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                if fast:
                    quantized = FastINT8Linear(module)
                else:
                    quantized = QuantizedLinear(module, bits=8)
                setattr(parent, child_name, quantized)
                n_quantized += 1

    if verbose:
        # Estimate memory savings.
        total_params = 0
        quantized_bytes = 0
        for m in model.modules():
            if isinstance(m, (QuantizedLinear, FastINT8Linear)):
                total_params += m.in_features * m.out_features
                quantized_bytes += m.in_features * m.out_features * 1  # 1 byte int8
            elif isinstance(m, nn.Linear):
                total_params += m.weight.numel()
                quantized_bytes += m.weight.numel() * 2  # BF16
        mode = "fast (_scaled_mm)" if fast else "dequant"
        print(f"  [InferQuant] INT8 ({mode}): {n_quantized} layers quantized")
        if total_params > 0:
            print(f"  [InferQuant] weight memory: {quantized_bytes/1024**2:.1f} MB (was {total_params*2/1024**2:.1f} MB, {quantized_bytes/(total_params*2)*100:.0f}%)")
    return n_quantized


def quantize_model_int4(model, target_modules=None, group_size=128, verbose=True):
    """Replace Linear layers with INT4 weight-only quantized versions.

    INT4 with group_size=128 gives best speed/quality tradeoff.
    Handles both nn.Linear and already-quantized QuantizedLinear/FastINT8Linear.

    Args:
        model: the model
        target_modules: modules to target (default: all projections)
        group_size: quantization group size (128 is standard)
        verbose: print stats

    Returns:
        number of layers quantized
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj',
                         'w1', 'w2', 'w3', 'w_gate', 'w_up', 'w_down',
                         'fc1', 'fc2', 'qkv_proj',
                         'kv_down_proj', 'k_up_proj', 'v_up_proj',
                         'in_proj']

    n_quantized = 0
    for name, module in list(model.named_modules()):
        # Handle nn.Linear directly
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                quantized = QuantizedLinear(module, bits=4, group_size=group_size)
                setattr(parent, child_name, quantized)
                n_quantized += 1
        # Handle already-quantized layers (dequantize first, then requantize)
        elif isinstance(module, (QuantizedLinear, FastINT8Linear)):
            if any(t in name for t in target_modules):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                # Dequantize to a temporary nn.Linear
                if hasattr(module, '_get_dequantized_weight'):
                    w = module._get_dequantized_weight()
                elif hasattr(module, '_cached_weight') and module._cached_weight is not None:
                    w = module._cached_weight
                elif hasattr(module, '_dequantize'):
                    w = module._dequantize()
                else:
                    continue
                tmp = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
                tmp.weight.data.copy_(w.to(tmp.weight.dtype))
                if module.bias is not None:
                    tmp.bias.data.copy_(module.bias_weight.to(tmp.bias.dtype))
                tmp = tmp.to(w.device)
                # Requantize to INT4
                quantized = QuantizedLinear(tmp, bits=4, group_size=group_size)
                setattr(parent, child_name, quantized)
                n_quantized += 1

    if verbose:
        print(f"  [InferQuant] INT4 (group={group_size}): {n_quantized} layers quantized")
    return n_quantized


_W4_TARGET_SUFFIXES = (
    'q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj',
    'w1', 'w2', 'w3', 'w_gate', 'w_up', 'w_down',
    'fc1', 'fc2', 'qkv_proj', 'in_proj',
)


def quantize_model_w4(model, target_modules=None, group_size=128,
                      device=None, verbose=True):
    """Pack Linear weights to true INT4 via torchao's tile-packed W4A16 kernel.

    Unlike :class:`QuantizedLinear` (which caches a dequantized bf16 weight at
    first forward, restoring full VRAM cost), this path keeps weights packed
    and runs the fused ``torch._weight_int4pack_mm`` tinygemm kernel — storage
    stays ~0.5 byte/param for the whole session. On the RTX 5070 this is the
    only weight-quant path that lets an 8B model stay GPU-resident
    (~5.6GB total vs ~16GB bf16) *and* decode faster than bf16.

    The ``nn.Linear`` modules are kept in place; only ``weight`` is swapped to
    a packed ``Int4TilePackedTo4dTensor``, so the rest of the stack sees
    ordinary modules. Requires ``torchao`` — already in the ForgeAI venv.

    Note: ``aten::_convert_weight_to_int4pack`` is CUDA-only in torch 2.11,
    so when ``device`` is CUDA and matched weights are still on CPU this
    function moves *one Linear at a time* to the GPU, quantizes there, and
    frees the bf16 source — peak VRAM is packed total + a single layer's
    bf16 weight (~150MB), never the full model.

    Args:
        model: the model (typically CPU-resident when pre-GPU-placement).
        target_modules: module-name substrings to quantize. Default: all
            attention/FFN projections (lm_head and embeddings stay bf16).
        group_size: int4 quantization group size (128 = standard W4G128).
        device: quantize on this device ("cuda" enables the incremental
            GPU path; None/CPU quantizes in place — requires an int4pack
            CPU kernel, absent in torch 2.11).
        verbose: print stats.

    Returns:
        number of layers quantized
    """
    from torchao.quantization import quantize_
    from torchao.quantization.quant_api import (
        Int4WeightOnlyConfig, Int4PackingFormat)

    targets = target_modules or _W4_TARGET_SUFFIXES

    def _filter(module, fqn):
        return (isinstance(module, nn.Linear)
                and any(t in fqn for t in targets))

    cfg = Int4WeightOnlyConfig(
        group_size=group_size,
        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
        set_inductor_config=False)

    # Int4TilePackedTo4dTensor only accepts bf16 source weights — cast fp32
    # sources in place first (also halves peak host RAM mid-conversion).
    matched = [m for n, m in model.named_modules() if _filter(m, n)]
    for m in matched:
        if m.weight.dtype != torch.bfloat16:
            m.weight.data = m.weight.data.to(torch.bfloat16)

    on_cuda = (device is not None
               and torch.device(device).type == "cuda"
               and any(m.weight.device.type != "cuda" for m in matched))

    if on_cuda:
        # Incremental path: one Linear's bf16 weight on GPU at a time.
        for m in matched:
            m.to(device)
            quantize_(m, cfg)  # root module is the Linear itself
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        quantize_(model, cfg, filter_fn=_filter, device=device)

    if verbose:
        packed = sum(
            m.weight.numel() for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and any(t in n for t in targets))
        print(f"  [InferQuant] W4-int4pack (group={group_size}): "
              f"{len(matched)} layers -> ~{packed * 0.5 / 1024**3:.2f} GB packed")
    return len(matched)


def benchmark_quantization(model, tokenizer, prompts, device="cuda"):
    """Benchmark inference speed before and after INT8 quantization.
    
    Returns dict with tokens/sec for both configurations.
    """
    import time

    # Baseline (BF16).
    model.eval()
    total_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            for _ in range(50):
                out = model(ids)
                logits = out[0] if isinstance(out, tuple) else out
                next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_tok], dim=1)
                total_tokens += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    baseline_tps = total_tokens / (time.time() - t0)

    # INT8.
    quantize_model_int8(model, verbose=False)
    total_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            for _ in range(50):
                out = model(ids)
                logits = out[0] if isinstance(out, tuple) else out
                next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_tok], dim=1)
                total_tokens += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    int8_tps = total_tokens / (time.time() - t0)

    speedup = int8_tps / baseline_tps
    print(f"  [InferQuant] BF16: {baseline_tps:.0f} tok/s | INT8: {int8_tps:.0f} tok/s | {speedup:.2f}x speedup")
    return {"baseline_tps": baseline_tps, "int8_tps": int8_tps, "speedup": speedup}


def dequantize_gptq(weight: torch.Tensor, qweight: torch.Tensor,
                    qzeros: torch.Tensor, scales: torch.Tensor,
                    bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """Dequantize GPTQ-packed weights to full precision.

    GPTQ packs quantized weights into int32 tensors where each int32 holds
    8/bits quantized values. Scales and zeros are per-group.

    Args:
        weight: output tensor [out_features, in_features] (to be filled)
        qweight: packed quantized weights [out_features, in_features // pack_factor]
        qzeros: zero points [num_groups, out_features // pack_factor]
        scales: scales [num_groups, out_features]
        bits: quantization bits (4 or 8)
        group_size: number of elements per quantization group
    """
    pack_factor = 32 // bits
    out_features, packed_in = qweight.shape
    in_features = packed_in * pack_factor
    num_groups = (in_features + group_size - 1) // group_size

    # Unpack quantized weights
    weight = torch.zeros(out_features, in_features, dtype=scales.dtype, device=qweight.device)
    for i in range(pack_factor):
        col_start = i * (in_features // pack_factor)
        col_end = (i + 1) * (in_features // pack_factor)
        # Extract bits at position i
        mask = (1 << bits) - 1
        qw_unpacked = (qweight[:, col_start:col_end] >> (i * bits)) & mask
        # Subtract zero point and scale
        for g in range(num_groups):
            g_start = g * group_size
            g_end = min((g + 1) * group_size, in_features)
            g_len = g_end - g_start
            if g_len <= 0:
                break
            # Get zero point for this group (subtract 1 from packed zero)
            if g < qzeros.shape[0]:
                zp = (qzeros[g, :out_features].float() - 1) * scales[g, :out_features]
            else:
                zp = torch.zeros(out_features, dtype=scales.dtype, device=qweight.device)
            # Apply scale and zero point
            weight[:, g_start:g_end] = (
                qw_unpacked[:, g_start:col_start + g_len].float() *
                scales[g, :out_features].unsqueeze(1).float()
            ) - zp.unsqueeze(1)
    return weight


def dequantize_awq(weight: torch.Tensor, qweight: torch.Tensor,
                   qzeros: torch.Tensor, scales: torch.Tensor,
                   bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """Dequantize AWQ-packed weights to full precision.

    AWQ uses a similar packing scheme to GPTQ but with different
    zero-point handling (zeros are stored as actual values, not offset by 1).

    Args:
        weight: output tensor [out_features, in_features] (to be filled)
        qweight: packed quantized weights [in_features // pack_factor, out_features]
        qzeros: zero points [num_groups, out_features]
        scales: scales [num_groups, out_features]
        bits: quantization bits (4 or 8)
        group_size: number of elements per quantization group
    """
    pack_factor = 32 // bits
    packed_in, out_features = qweight.shape
    in_features = packed_in * pack_factor
    num_groups = (in_features + group_size - 1) // group_size

    # AWQ stores qweight transposed vs GPTQ
    weight = torch.zeros(out_features, in_features, dtype=scales.dtype, device=qweight.device)
    for i in range(pack_factor):
        mask = (1 << bits) - 1
        qw_unpacked = (qweight[i::pack_factor, :].T >> 0) & mask  # AWQ interleaves differently
        for g in range(num_groups):
            g_start = g * group_size
            g_end = min((g + 1) * group_size, in_features)
            g_len = g_end - g_start
            if g_len <= 0:
                break
            if g < qzeros.shape[0]:
                zp = qzeros[g, :out_features].float()
            else:
                zp = torch.zeros(out_features, dtype=scales.dtype, device=qweight.device)
            scale = scales[g, :out_features].float() if g < scales.shape[0] else torch.ones(out_features)
            weight[:, g_start:g_end] = (
                qw_unpacked[:out_features, :g_len].float() - zp.unsqueeze(1)
            ) * scale.unsqueeze(1)
    return weight


def load_gptq_linear(layer: torch.nn.Linear, qweight: torch.Tensor,
                     qzeros: torch.Tensor, scales: torch.Tensor,
                     bits: int = 4, group_size: int = 128) -> torch.nn.Linear:
    """Load GPTQ-quantized weights into a standard nn.Linear layer.

    Dequantizes the GPTQ weights and loads them into the layer for
    inference with standard matmul. This is the simple path — for
    fast inference, a fused GPTQ kernel would be needed.
    """
    out_features, in_features = layer.weight.shape
    dequant = dequantize_gptq(
        layer.weight.data, qweight, qzeros, scales, bits, group_size)
    layer.weight.data.copy_(dequant.to(layer.weight.dtype))
    return layer


def load_awq_linear(layer: torch.nn.Linear, qweight: torch.Tensor,
                    qzeros: torch.Tensor, scales: torch.Tensor,
                    bits: int = 4, group_size: int = 128) -> torch.nn.Linear:
    """Load AWQ-quantized weights into a standard nn.Linear layer."""
    out_features, in_features = layer.weight.shape
    dequant = dequantize_awq(
        layer.weight.data, qweight, qzeros, scales, bits, group_size)
    layer.weight.data.copy_(dequant.to(layer.weight.dtype))
    return layer
