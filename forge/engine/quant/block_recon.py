"""Block-level reconstruction for post-training quantization.

General-purpose infrastructure that improves any quantized model by optimizing
quantized layer parameters to minimize block-level output error, not just
weight-level error. This is the key missing piece for extreme low-bit
quantization (NanoQuant, BTC-LLM, TernaryPTQ) to achieve usable quality.

How it works:
1. Run calibration data through the original (unquantized) model, capturing
   each transformer block's input and output hidden states.
2. Replace the original model with the quantized model.
3. For each block, freeze the input activations and optimize the quantized
   layer parameters (via STE) to minimize ||block_orig(x) - block_q(x)||².
4. This "block-level" objective is strictly better than weight-level MSE
   because it accounts for how quantization errors propagate through the
   block's non-linearities (attention, MLP, norms).

This is model-architecture-agnostic: it works with any HuggingFace model
that exposes `model.model.layers` (Qwen, Llama, Mistral, etc.) or
`model.transformer.h` (GPT-style).

Usage:
    from forge.engine.quant.block_recon import BlockReconstructor

    # After quantizing the model
    recon = BlockReconstructor(
        model_orig=original_model,
        model_quant=quantized_model,
        calibration_data=calib_input_ids,
        device='cuda',
    )
    recon.reconstruct(n_iters=50, lr=0.01, loss='mse')
    # model_quant is now optimized in-place

References:
    - NanoQuant (ICML 2026): block + model reconstruction pipeline
    - GPTQ: layer-wise error compensation (related but different)
    - OmniQuant: block-level reconstruction with learnable scales
"""
from __future__ import annotations

import gc
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Block extraction — find transformer blocks in any model architecture
# ──────────────────────────────────────────────────────────────────────────

def _find_block_list(model: nn.Module) -> tuple[nn.Module, str, nn.ModuleList]:
    """Find the transformer block list in a HuggingFace model.

    Supports:
        - Qwen2/Llama/Mistral: model.model.layers
        - GPT-2/GPT-Neo: model.transformer.h
        - Generic: search for the first nn.ModuleList with >4 children

    Returns:
        (parent_module, attr_name, block_list)
    """
    # Try common patterns
    candidates = [
        ("model", "layers"),       # Qwen2, Llama, Mistral
        ("transformer", "h"),      # GPT-2, GPT-Neo
        ("transformer", "layers"), # Some models
        ("gpt_neox", "layers"),    # GPT-NeoX
        ("model", "decoder_layers"),  # Some encoder-decoder
    ]

    for parent_attr, list_attr in candidates:
        parent = getattr(model, parent_attr, None)
        if parent is not None:
            blocks = getattr(parent, list_attr, None)
            if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
                return parent, list_attr, blocks

    # Fallback: search for any nn.ModuleList with >4 children
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 4:
            # Find parent
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            return parent, parts[-1], module

    raise ValueError(
        "Could not find transformer block list. "
        "Expected model.model.layers or model.transformer.h")


def _find_embed(model: nn.Module) -> nn.Module | None:
    """Find the token embedding module."""
    # Common attribute names across architectures
    names = ["embed_tokens", "wte", "embeddings", "embed", "embedding"]
    for name in names:
        # Check model.model.embed_tokens (Qwen/Llama)
        inner = getattr(model, "model", None)
        if inner is not None:
            emb = getattr(inner, name, None)
            if emb is not None and isinstance(emb, (nn.Embedding, nn.Module)):
                return emb
        # Check model.transformer.wte (GPT)
        inner = getattr(model, "transformer", None)
        if inner is not None:
            emb = getattr(inner, name, None)
            if emb is not None and isinstance(emb, (nn.Embedding, nn.Module)):
                return emb
        # Direct on model
        emb = getattr(model, name, None)
        if emb is not None and isinstance(emb, (nn.Embedding, nn.Module)):
            # Make sure it's actually an embedding-like module
            if isinstance(emb, nn.Embedding) or hasattr(emb, 'weight'):
                return emb
    return None


# ──────────────────────────────────────────────────────────────────────────
# Activation capture — record block inputs/outputs
# ──────────────────────────────────────────────────────────────────────────

class _BlockIOCapture:
    """Context manager that captures block inputs, kwargs, and outputs via hooks."""

    def __init__(self, blocks: nn.ModuleList, device: str):
        self.blocks = blocks
        self.device = device
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.kwargs_list: list[dict] = []
        self._hooks = []

    def __enter__(self):
        for i, block in enumerate(self.blocks):
            def make_hook(idx):
                def hook(module, args, kwargs, output):
                    # args[0] is hidden_states
                    inp = args[0] if args else kwargs.get("hidden_states")
                    if inp is not None:
                        self.inputs.append(inp.detach().cpu())
                    # Capture only safe-to-replay kwargs.
                    # Skip mutable state: past_key_values, use_cache, cache_position
                    # (these are modified during forward and can't be replayed)
                    safe_keys = {'position_embeddings', 'attention_mask', 'position_ids'}
                    kw_copy = {}
                    for k, v in (kwargs or {}).items():
                        if k not in safe_keys:
                            continue
                        if isinstance(v, torch.Tensor):
                            kw_copy[k] = v.detach().cpu()
                        elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                            kw_copy[k] = tuple(x.detach().cpu() for x in v)
                        else:
                            kw_copy[k] = v
                    self.kwargs_list.append(kw_copy)
                    # output is a tuple; first element is hidden_states
                    out = output[0] if isinstance(output, tuple) else output
                    if out is not None:
                        self.outputs.append(out.detach().cpu())
                return hook
            h = block.register_forward_hook(make_hook(i), with_kwargs=True)
            self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ──────────────────────────────────────────────────────────────────────────
# STE (Straight-Through Estimator) for binary/ternary weights
# ──────────────────────────────────────────────────────────────────────────

def _ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Sign with STE: forward = sign, backward = identity."""
    return x + (torch.sign(x) - x).detach()


def _ste_ternary(x: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """Ternarize with STE: forward = sign(x)*(|x|>threshold), backward = identity."""
    ternary = torch.sign(x) * (x.abs() > threshold).float()
    return x + (ternary - x).detach()


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with STE for integer quantization."""
    return x + (torch.round(x) - x).detach()


# ──────────────────────────────────────────────────────────────────────────
# Quantized parameter extractor — find what to optimize in each block
# ──────────────────────────────────────────────────────────────────────────

def _get_optimizable_params(block: nn.Module) -> dict[str, torch.Tensor]:
    """Find quantized layer parameters that can be optimized via STE.

    Returns a dict mapping parameter names to their current values.
    These are the "soft" versions of the quantized parameters that will
    be optimized and then re-quantized.
    """
    params = {}
    for name, module in block.named_modules():
        cls_name = type(module).__name__

        # NanoQuant: optimize U, V (binary factors) and s1, s2 (scales)
        if cls_name == 'NanoQuantLinear':
            if module.U_packed.numel() > 0:
                params[f"{name}.U_soft"] = _unpack_to_soft(module, 'U')
                params[f"{name}.V_soft"] = _unpack_to_soft(module, 'V')
                params[f"{name}.s1"] = module.s1.clone()
                params[f"{name}.s2"] = module.s2.clone()

        # BTC: optimize scales (codebook is discrete, hard to optimize)
        elif cls_name == 'BTCQuantLinear' or cls_name == 'TernaryPTQLinear':
            if module.scales.numel() > 0:
                params[f"{name}.scales"] = module.scales.clone()

        # R46 FP4 methods: optimize scales
        elif cls_name in ('AWQFP4Linear', 'GPTQFP4Linear', 'HadamardRotatedFP4Linear',
                          'OptimalGridFP4Linear', 'HadamardGPTQFP4Linear',
                          'HadamardAWQFP4Linear'):
            if hasattr(module, 'weight_global_scale') and module.weight_global_scale.numel() > 0:
                params[f"{name}.global_scale"] = module.weight_global_scale.clone()
            if hasattr(module, 'weight_scales') and module.weight_scales.numel() > 0:
                params[f"{name}.block_scales"] = module.weight_scales.clone()

        # Generic: any module with a _cached_weight (dequantized) can optimize scales
        elif hasattr(module, '_cached_weight') and hasattr(module, 'weight_packed'):
            if hasattr(module, 'weight_scales') and module.weight_scales.numel() > 0:
                params[f"{name}.scales"] = module.weight_scales.clone()

    return params


def _unpack_to_soft(module, which: str) -> torch.Tensor:
    """Unpack binary packed weights to a soft (float) tensor for optimization."""
    from forge.engine.quant.novel_quant_r48 import _unpack_binary_bits
    if which == 'U':
        shape = module.U_shape
        packed = module.U_packed
    else:
        shape = module.V_shape
        packed = module.V_packed
    n = shape[0] * shape[1]
    binary = _unpack_binary_bits(packed, n).view(shape).float()
    return binary


def _apply_optimized_params(block: nn.Module, params: dict[str, torch.Tensor]):
    """Apply optimized parameters back to the quantized layers."""
    from forge.engine.quant.novel_quant_r48 import _pack_binary_bits

    for name, module in block.named_modules():
        cls_name = type(module).__name__

        # Skip if any param has NaN
        relevant = [v for k, v in params.items() if k.startswith(f"{name}.")]
        if any(torch.isnan(v).any() or torch.isinf(v).any() for v in relevant):
            logger.warning(f"  Skipping {name}: NaN/Inf in params")
            continue

        if cls_name == 'NanoQuantLinear':
            u_key = f"{name}.U_soft"
            v_key = f"{name}.V_soft"
            if u_key in params:
                U = torch.sign(params[u_key])
                U[U == 0] = 1
                module.U_packed = _pack_binary_bits(U.to(torch.int8)).to(torch.uint8)
                module._invalidate_cache()
            if v_key in params:
                V = torch.sign(params[v_key])
                V[V == 0] = 1
                module.V_packed = _pack_binary_bits(V.to(torch.int8)).to(torch.uint8)
                module._invalidate_cache()
            if f"{name}.s1" in params:
                module.s1 = params[f"{name}.s1"].to(torch.float16)
                module._invalidate_cache()
            if f"{name}.s2" in params:
                module.s2 = params[f"{name}.s2"].to(torch.float16)
                module._invalidate_cache()

        elif cls_name == 'BTCQuantLinear' or cls_name == 'TernaryPTQLinear':
            if f"{name}.scales" in params:
                module.scales = params[f"{name}.scales"].to(torch.float16)
                module._invalidate_cache()

        elif cls_name in ('AWQFP4Linear', 'GPTQFP4Linear', 'HadamardRotatedFP4Linear',
                          'OptimalGridFP4Linear', 'HadamardGPTQFP4Linear',
                          'HadamardAWQFP4Linear'):
            if f"{name}.global_scale" in params:
                module.weight_global_scale = params[f"{name}.global_scale"].to(torch.float32)
                module._invalidate_cache()
            if f"{name}.block_scales" in params:
                module.weight_scales = params[f"{name}.block_scales"].to(torch.float16)
                module._invalidate_cache()

        elif hasattr(module, '_cached_weight') and hasattr(module, 'weight_packed'):
            if f"{name}.scales" in params and hasattr(module, 'weight_scales'):
                module.weight_scales = params[f"{name}.scales"].to(torch.float16)
                if hasattr(module, '_invalidate_cache'):
                    module._invalidate_cache()
                else:
                    module._cached_weight = None


# ──────────────────────────────────────────────────────────────────────────
# Soft-forward patching — make quantized layers use STE'd soft params
# ──────────────────────────────────────────────────────────────────────────

def _install_soft_forward(block: nn.Module, opt_params: dict[str, torch.Tensor]) -> dict:
    """Patch quantized layers in the block to use soft (STE'd) params.

    Returns a dict of patch info for later removal.
    """

    patches = {}
    for name, module in block.named_modules():
        cls_name = type(module).__name__
        prefix = f"{name}."

        if cls_name == 'NanoQuantLinear':
            # Patch forward to use soft scales (binary factors stay frozen from packed)
            s1_key = prefix + "s1"
            s2_key = prefix + "s2"
            u_key = prefix + "U_soft"
            v_key = prefix + "V_soft"
            has_soft_binary = u_key in opt_params
            has_soft_scales = s1_key in opt_params or s2_key in opt_params
            if has_soft_scales or has_soft_binary:
                orig_forward = module.forward
                patches[id(module)] = {
                    'module': module,
                    'orig_forward': orig_forward,
                    'u_key': u_key, 'v_key': v_key,
                    's1_key': s1_key, 's2_key': s2_key,
                    'type': 'nanoquant',
                    'has_soft_binary': has_soft_binary,
                }

                def make_soft_forward(mod, info):
                    def soft_forward(x):
                        from forge.engine.quant.novel_quant_r48 import _unpack_binary_bits
                        # Get binary factors: soft if available, else from packed
                        if info['has_soft_binary'] and info['u_key'] in opt_params:
                            U = _ste_sign(opt_params[info['u_key']]).to(x.dtype)
                            V = _ste_sign(opt_params[info['v_key']]).to(x.dtype)
                        else:
                            # Use frozen packed binary factors
                            U = _unpack_binary_bits(mod.U_packed,
                                                    mod.U_shape[0] * mod.U_shape[1])
                            U = U.view(mod.U_shape).to(x.dtype)
                            V = _unpack_binary_bits(mod.V_packed,
                                                    mod.V_shape[0] * mod.V_shape[1])
                            V = V.view(mod.V_shape).to(x.dtype)
                        # Get scales: soft if available, else from frozen
                        if info['s1_key'] in opt_params:
                            s1 = opt_params[info['s1_key']].to(x.dtype)
                        else:
                            s1 = mod.s1.to(x.dtype)
                        if info['s2_key'] in opt_params:
                            s2 = opt_params[info['s2_key']].to(x.dtype)
                        else:
                            s2 = mod.s2.to(x.dtype)
                        W = s1.unsqueeze(1) * (U @ V.T) * s2.unsqueeze(0)
                        bias = mod.bias.to(x.dtype) if mod.bias is not None else None
                        return F.linear(x, W, bias)
                    return soft_forward

                module.forward = make_soft_forward(module, patches[id(module)])

        elif cls_name in ('TernaryPTQLinear', 'BTCQuantLinear'):
            s_key = prefix + "scales"
            if s_key in opt_params:
                orig_forward = module.forward
                patches[id(module)] = {
                    'module': module,
                    'orig_forward': orig_forward,
                    's_key': s_key,
                    'type': cls_name.lower(),
                }

                if cls_name == 'TernaryPTQLinear':
                    def make_soft_forward_tern(mod, info):
                        def soft_forward(x):
                            from forge.engine.quant.novel_quant import (
                                base3_packed_to_ternary,
                            )
                            n = mod.n_weights.item()
                            W_t = base3_packed_to_ternary(mod.weight_packed, n)
                            W_t = W_t[:n].view(mod.out_features, mod.in_features).to(x.dtype)
                            scales = opt_params[info['s_key']].to(x.dtype)
                            W = W_t * scales.unsqueeze(1)
                            bias = mod.bias.to(x.dtype) if mod.bias is not None else None
                            return F.linear(x, W, bias)
                        return soft_forward
                    module.forward = make_soft_forward_tern(module, patches[id(module)])

                elif cls_name == 'BTCQuantLinear':
                    def make_soft_forward_btc(mod, info):
                        def soft_forward(x):
                            from forge.engine.quant.novel_quant_r48 import (
                                _hadamard_matrix,
                                _unpack_binary_bits,
                            )
                            cb = _unpack_binary_bits(mod.codebook_packed,
                                                     mod.codebook_d_in_eff)
                            cb = cb.to(x.dtype)
                            idx = mod.row_indices.long()
                            scales = opt_params[info['s_key']].to(x.dtype)
                            W_rot = scales.unsqueeze(1) * cb[idx]
                            if mod.use_rotation and mod.hadamard_order > 0:
                                h_size = 2 ** mod.hadamard_order
                                H = _hadamard_matrix(h_size, x.device, x.dtype)
                                W = W_rot @ H.T
                                W = W[:, :mod.in_features]
                            else:
                                W = W_rot[:, :mod.in_features]
                            bias = mod.bias.to(x.dtype) if mod.bias is not None else None
                            return F.linear(x, W, bias)
                        return soft_forward
                    module.forward = make_soft_forward_btc(module, patches[id(module)])

        elif cls_name in ('AWQFP4Linear', 'GPTQFP4Linear', 'HadamardRotatedFP4Linear',
                          'OptimalGridFP4Linear', 'HadamardGPTQFP4Linear',
                          'HadamardAWQFP4Linear'):
            gs_key = prefix + "global_scale"
            bs_key = prefix + "block_scales"
            if gs_key in opt_params or bs_key in opt_params:
                orig_forward = module.forward
                patches[id(module)] = {
                    'module': module,
                    'orig_forward': orig_forward,
                    'gs_key': gs_key, 'bs_key': bs_key,
                    'type': 'fp4',
                }

                def make_soft_forward_fp4(mod, info):
                    def soft_forward(x):
                        # Use the module's _dequantize_weight but with soft scales
                        # We need to temporarily replace the scales
                        orig_gs = mod.weight_global_scale
                        orig_bs = mod.weight_scales
                        if info['gs_key'] in opt_params:
                            mod.weight_global_scale = opt_params[info['gs_key']].detach().to(torch.float32)
                        if info['bs_key'] in opt_params:
                            mod.weight_scales = opt_params[info['bs_key']].detach().to(torch.float16)
                        mod._invalidate_cache()
                        W = mod._dequantize_weight(x.dtype)
                        # Restore (not needed since we'll invalidate again)
                        mod.weight_global_scale = orig_gs
                        mod.weight_scales = orig_bs
                        bias = mod.bias.to(x.dtype) if mod.bias is not None else None
                        return F.linear(x, W, bias)
                    return soft_forward
                module.forward = make_soft_forward_fp4(module, patches[id(module)])

    return patches


def _update_soft_forward(patches: dict, opt_params: dict[str, torch.Tensor]):
    """Update the soft params reference in patches (no-op since we reference opt_params directly)."""
    pass  # The closures reference opt_params directly, so updates are automatic


def _uninstall_soft_forward(patches: dict):
    """Remove soft-forward patches and restore original forward methods."""
    for _pid, info in patches.items():
        info['module'].forward = info['orig_forward']


# ──────────────────────────────────────────────────────────────────────────
# Manual block forward — call sub-modules directly to avoid signature issues
# ──────────────────────────────────────────────────────────────────────────

def _manual_block_forward(block: nn.Module, hidden_states: torch.Tensor,
                          model: nn.Module, block_idx: int) -> torch.Tensor:
    """Run a transformer block forward by calling sub-modules directly.

    This avoids needing to know the exact block.forward() signature, which
    varies across architectures and transformers versions.

    Supports Qwen2/Llama/Mistral-style blocks:
        hidden → input_layernorm → self_attn → residual →
        post_attention_layernorm → mlp → residual

    For attention, we use a simplified causal self-attention that calls
    the quantized q/k/v/o projections directly.
    """
    # Step 1: input layernorm
    norm1 = getattr(block, 'input_layernorm', None) or getattr(block, 'ln_1', None)
    if norm1 is None:
        # Can't find norm, just return hidden (no transformation)
        return hidden_states

    normed = norm1(hidden_states)

    # Step 2: self-attention (simplified — call projections directly)
    attn = getattr(block, 'self_attn', None) or getattr(block, 'attn', None)
    if attn is not None:
        # Get dimensions
        cfg = getattr(model, 'config', None)
        n_heads = getattr(cfg, 'num_attention_heads', 14) if cfg else 14
        n_kv_heads = getattr(cfg, 'num_key_value_heads', 2) if cfg else 2
        head_dim = getattr(cfg, 'head_dim', None) or (
            getattr(cfg, 'hidden_size', 896) // n_heads if cfg else 64)

        normed.shape[-1]

        # QKV projections (these are the quantized layers with soft forward)
        q = attn.q_proj(normed)  # (batch, seq, n_heads * head_dim)
        k = attn.k_proj(normed)
        v = attn.v_proj(normed)

        # Reshape to (batch, n_heads, seq, head_dim)
        q = q.view(q.shape[0], q.shape[1], n_heads, head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], n_kv_heads, head_dim).transpose(1, 2)

        # Repeat KV heads to match Q heads (GQA)
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Simplified attention (no RoPE, no position embeddings)
        # This is an approximation — RoPE would require position_ids
        scale = head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch, heads, seq, seq)

        # Causal mask
        seq_len = scores.shape[-1]
        causal = torch.triu(torch.ones(seq_len, seq_len, device=scores.device,
                                       dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float('-inf'))

        attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            attn_out.shape[0], attn_out.shape[2], -1)

        # Output projection
        attn_out = attn.o_proj(attn_out)
    else:
        attn_out = torch.zeros_like(hidden_states)

    # Residual
    hidden_states = hidden_states + attn_out

    # Step 3: post-attention layernorm
    norm2 = getattr(block, 'post_attention_layernorm', None) or getattr(block, 'ln_2', None)
    if norm2 is not None:
        normed = norm2(hidden_states)
    else:
        normed = hidden_states

    # Step 4: MLP
    mlp = getattr(block, 'mlp', None)
    if mlp is not None:
        if hasattr(mlp, 'gate_proj'):
            # SwiGLU-style (Qwen/Llama)
            gate = mlp.gate_proj(normed)
            up = mlp.up_proj(normed)
            act_fn = getattr(mlp, 'act_fn', None)
            if act_fn is not None:
                gate = act_fn(gate)
            else:
                gate = F.silu(gate)
            mlp_out = mlp.down_proj(gate * up)
        else:
            # Simple MLP
            mlp_out = mlp(normed)
    else:
        mlp_out = torch.zeros_like(hidden_states)

    # Residual
    output = hidden_states + mlp_out

    return output


# ──────────────────────────────────────────────────────────────────────────
# BlockReconstructor — the main class
# ──────────────────────────────────────────────────────────────────────────

class BlockReconstructor:
    """Block-level reconstruction for post-training quantization.

    Optimizes quantized layer parameters to minimize block-level output error,
    using calibration data captured from the original (unquantized) model.

    Args:
        model_orig: the original unquantized model (for capturing targets)
        model_quant: the quantized model (to be optimized in-place)
        calibration_data: (batch, seq_len) tensor of input_ids for calibration
        device: 'cuda' or 'cpu'
        max_blocks: if set, only reconstruct the first N blocks (for testing)
        block_batch_size: number of blocks to reconstruct simultaneously
                          (higher = more parallelism, more memory)

    Example:
        >>> # After quantizing
        >>> recon = BlockReconstructor(model_fp, model_quant, calib_ids, 'cuda')
        >>> recon.reconstruct(n_iters=50, lr=0.01)
        >>> # model_quant is now optimized
    """

    def __init__(
        self,
        model_orig: nn.Module,
        model_quant: nn.Module,
        calibration_data: torch.Tensor,
        device: str = "cuda",
        max_blocks: int | None = None,
    ):
        self.device = torch.device(device)
        self.calib_data = calibration_data.to(device)

        # Find block lists in both models
        _, _, self.blocks_orig = _find_block_list(model_orig)
        _, _, self.blocks_quant = _find_block_list(model_quant)

        if len(self.blocks_orig) != len(self.blocks_quant):
            raise ValueError(
                f"Block count mismatch: orig={len(self.blocks_orig)}, "
                f"quant={len(self.blocks_quant)}")

        self.n_blocks = len(self.blocks_quant)
        if max_blocks is not None:
            self.n_blocks = min(self.n_blocks, max_blocks)

        self.model_orig = model_orig
        self.model_quant = model_quant

        # Find embedding layer for generating hidden states
        self.embed = _find_embed(model_orig)
        if self.embed is None:
            raise ValueError("Could not find token embedding layer")

        # Captured data (filled by capture_block_io)
        self._block_inputs: list[torch.Tensor] = []
        self._block_outputs: list[torch.Tensor] = []
        self._block_kwargs: list[dict] = []

        logger.info(f"BlockReconstructor: {self.n_blocks} blocks, "
                     f"device={device}")

    @torch.no_grad()
    def capture_block_io(self) -> None:
        """Run calibration data through the original model and capture
        each block's input, kwargs, and output hidden states."""
        self._block_inputs = []
        self._block_outputs = []
        self._block_kwargs = []

        self.model_orig.eval()
        device = self.device

        with _BlockIOCapture(self.blocks_orig, str(device)) as capture, torch.no_grad():
            # Disable cache so blocks don't accumulate KV state
            cfg = getattr(self.model_orig, 'config', None)
            if cfg is not None and hasattr(cfg, 'use_cache'):
                old_use_cache = cfg.use_cache
                cfg.use_cache = False
            self.model_orig(self.calib_data)
            if cfg is not None and hasattr(cfg, 'use_cache'):
                cfg.use_cache = old_use_cache

        self._block_inputs = capture.inputs
        self._block_outputs = capture.outputs
        self._block_kwargs = capture.kwargs_list

        logger.info(f"Captured I/O for {len(self._block_inputs)} blocks "
                     f"(input shape: {self._block_inputs[0].shape})")

    def reconstruct_block(
        self,
        block_idx: int,
        n_iters: int = 50,
        lr: float = 0.01,
        loss_fn: str = "mse",
        optimize_binary: bool = True,
    ) -> float:
        """Reconstruct a single block using isolated block forward.

        Runs the block in isolation using the original model's captured
        input for this block. This avoids numerical explosion from running
        the full quantized model (where bad early blocks corrupt later inputs).

        The block forward is computed manually by calling sub-modules
        (norm, attention, MLP) directly, avoiding signature mismatches.

        Args:
            block_idx: which block to reconstruct
            n_iters: optimization iterations
            lr: learning rate
            loss_fn: 'mse' or 'kl'

        Returns:
            final loss value
        """
        if not self._block_outputs:
            raise RuntimeError("Must call capture_block_io() first")

        block = self.blocks_quant[block_idx]
        target = self._block_outputs[block_idx].to(self.device)
        block_input = self._block_inputs[block_idx].to(self.device)
        block_kwargs = self._block_kwargs[block_idx] if block_idx < len(self._block_kwargs) else {}

        # Move kwargs tensors to device
        kw = {}
        for k, v in block_kwargs.items():
            if isinstance(v, torch.Tensor):
                kw[k] = v.to(self.device)
            elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                kw[k] = tuple(x.to(self.device) for x in v)
            else:
                kw[k] = v

        # Get optimizable parameters — include latent matrices (U_soft, V_soft)
        # for STE-based optimization when optimize_binary=True.
        # The soft forward uses STE so gradients flow through the continuous
        # latent values even though the forward sees sign(latent).
        all_params = _get_optimizable_params(block)
        if optimize_binary:
            params_dict = all_params  # optimize everything: U_soft, V_soft, s1, s2
        else:
            # Only optimize continuous scales — safer but less powerful
            params_dict = {k: v for k, v in all_params.items()
                           if not (k.endswith('.U_soft') or k.endswith('.V_soft'))}
            if not params_dict:
                params_dict = all_params  # fallback
        if not params_dict:
            logger.warning(f"Block {block_idx}: no optimizable parameters found")
            return 0.0

        # Create optimization tensors with separate LRs for binary vs scale params
        # Binary latent matrices need higher LR (STE gradients are small but meaningful)
        # Scales need lower LR to prevent divergence
        opt_params = {}
        binary_params = []
        scale_params = []
        for name, val in params_dict.items():
            p = val.clone().to(self.device).requires_grad_(True)
            opt_params[name] = p
            if name.endswith('.U_soft') or name.endswith('.V_soft'):
                binary_params.append(p)
            else:
                scale_params.append(p)

        # Use SGD with momentum — Adam produces NaN with tiny STE gradients
        # due to sqrt(v) division. SGD is stable.
        param_groups = []
        if binary_params:
            param_groups.append({'params': binary_params, 'lr': lr * 10})
        if scale_params:
            param_groups.append({'params': scale_params, 'lr': lr})
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

        # Install soft-forward patches on quantized layers in this block
        # Even though we only optimize scales, we need the soft forward to
        # use the soft scales instead of the frozen packed scales
        patches = _install_soft_forward(block, opt_params)

        best_loss = float('inf')
        best_params = {k: v.detach().clone() for k, v in opt_params.items()}

        try:
            for iter in range(n_iters):
                optimizer.zero_grad()

                # Clear caches in this block
                for m in block.modules():
                    if hasattr(m, '_invalidate_cache'):
                        m._invalidate_cache()

                # Run block forward using the REAL block forward with captured kwargs
                # This ensures RoPE, attention masks, etc. are applied correctly
                with torch.enable_grad():
                    try:
                        output = block(block_input, **kw)
                    except Exception as e:
                        logger.warning(f"Block {block_idx} iter {iter}: forward failed: {e}")
                        break

                if output is None:
                    break

                # Compute loss
                if loss_fn == "mse":
                    loss = F.mse_loss(output, target)
                elif loss_fn == "kl":
                    p = F.log_softmax(output, dim=-1)
                    q = F.softmax(target, dim=-1)
                    loss = F.kl_div(p, q, reduction='batchmean')
                else:
                    loss = F.mse_loss(output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(opt_params.values(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    logger.warning(f"  Block {block_idx} iter {iter}: NaN/Inf loss, stopping")
                    break

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_params = {k: v.detach().clone() for k, v in opt_params.items()}

                if iter % 5 == 0 or iter == n_iters - 1:
                    logger.info(f"  Block {block_idx} iter {iter}: loss={loss_val:.6f}")
                    # Also print for visibility when logger isn't configured
                    if not logger.handlers:
                        logger.info(f"    iter {iter}: loss={loss_val:.6f}")
        finally:
            _uninstall_soft_forward(patches)

        # Apply best params to the actual buffers
        _apply_optimized_params(block, best_params)

        # Clean up
        for m in block.modules():
            if hasattr(m, '_invalidate_cache'):
                m._invalidate_cache()

        del opt_params, optimizer
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return best_loss

    def _run_block_forward(self, block: nn.Module, hidden_states: torch.Tensor,
                           block_idx: int) -> torch.Tensor:
        """Run a single block forward pass.

        Handles different model architectures by trying common signatures.
        """
        # Get position embeddings from the model's model
        inner = getattr(self.model_quant, 'model', None) or getattr(self.model_quant, 'transformer', None)
        if inner is not None:
            # Try to get position embeddings
            if hasattr(inner, 'rotary_emb'):
                # Qwen2/Llama style
                position_ids = torch.arange(hidden_states.shape[1],
                                            device=hidden_states.device).unsqueeze(0)
                try:
                    pos_emb = inner.rotary_emb(hidden_states, position_ids)
                    output = block(hidden_states, position_embeddings=pos_emb)
                    return output
                except Exception:
                    logger.debug("Block forward with position_embeddings failed", exc_info=True)

            # Fallback: try without position embeddings
            try:
                output = block(hidden_states)
                return output
            except Exception:
                logger.debug("Block forward without position_embeddings failed", exc_info=True)

        # Last resort: just run the block
        output = block(hidden_states)
        return output

    def reconstruct(
        self,
        n_iters: int = 50,
        lr: float = 0.01,
        loss_fn: str = "mse",
        blocks_to_reconstruct: list[int] | None = None,
        verbose: bool = True,
    ) -> dict[int, float]:
        """Reconstruct all (or selected) blocks.

        Args:
            n_iters: optimization iterations per block
            lr: learning rate
            loss_fn: 'mse' or 'kl'
            blocks_to_reconstruct: list of block indices, or None for all
            verbose: print progress

        Returns:
            dict mapping block_idx → final loss
        """
        if verbose:
            logger.info("BlockReconstructor: capturing block I/O from original model...")

        self.capture_block_io()

        if blocks_to_reconstruct is None:
            blocks_to_reconstruct = list(range(self.n_blocks))

        results = {}
        for i, idx in enumerate(blocks_to_reconstruct):
            if verbose:
                logger.info(f"  [{i+1}/{len(blocks_to_reconstruct)}] Reconstructing block {idx}...")

            try:
                loss = self.reconstruct_block(idx, n_iters=n_iters, lr=lr,
                                              loss_fn=loss_fn)
                results[idx] = loss
                if verbose:
                    logger.info(f"    Final loss: {loss:.6f}")
            except Exception as e:
                if verbose:
                    logger.info(f"    FAILED: {e}")
                results[idx] = float('inf')

            # Clean up between blocks
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if verbose:
            avg_loss = sum(results.values()) / max(len(results), 1)
            logger.info(f"BlockReconstructor complete: avg loss={avg_loss:.6f}")

        return results

    def reconstruct_progressive(
        self,
        n_iters: int = 30,
        lr: float = 0.01,
        loss_fn: str = "mse",
        verbose: bool = True,
    ) -> dict[int, float]:
        """Progressive reconstruction: reconstruct blocks in order,
        using each reconstructed block's output as input for the next.

        This is more accurate than parallel reconstruction because it
        accounts for how quantization errors in early blocks affect
        later blocks' inputs.

        Args:
            n_iters: optimization iterations per block
            lr: learning rate
            loss_fn: 'mse' or 'kl'
            verbose: print progress

        Returns:
            dict mapping block_idx → final loss
        """
        if verbose:
            logger.info("BlockReconstructor (progressive): capturing initial I/O...")

        self.capture_block_io()

        results = {}
        current_input = self._block_inputs[0].to(self.device)

        for idx in range(self.n_blocks):
            if verbose:
                logger.info(f"  [{idx+1}/{self.n_blocks}] Progressive block {idx}...")

            # Update this block's target to use the current (propagated) input
            # but keep the original target output
            target = self._block_outputs[idx].to(self.device)

            # Temporarily set the block's captured input to current_input
            original_input = self._block_inputs[idx]
            self._block_inputs[idx] = current_input.cpu()

            try:
                loss = self.reconstruct_block(idx, n_iters=n_iters, lr=lr,
                                              loss_fn=loss_fn)
                results[idx] = loss
                if verbose:
                    logger.info(f"    Final loss: {loss:.6f}")
            except Exception as e:
                if verbose:
                    logger.info(f"    FAILED: {e}")
                results[idx] = float('inf')

            # Restore original input
            self._block_inputs[idx] = original_input

            # Propagate current input through the reconstructed block
            # to get the input for the next block
            with torch.no_grad():
                block = self.blocks_quant[idx]
                for m in block.modules():
                    if hasattr(m, '_invalidate_cache'):
                        m._invalidate_cache()
                try:
                    output = self._run_block_forward(block, current_input, idx)
                    if isinstance(output, tuple):
                        output = output[0]
                    current_input = output.detach()
                except Exception:
                    # If forward fails, use original output as fallback
                    current_input = target

            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if verbose:
            avg_loss = sum(results.values()) / max(len(results), 1)
            logger.info(f"Progressive reconstruction complete: avg loss={avg_loss:.6f}")

        return results

    def calibrate_kl(
        self,
        n_iters: int = 20,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> float:
        """Model-level KL divergence calibration (NanoQuant Step 3.3).

        After block reconstruction, run the full quantized model and optimize
        all scale parameters to minimize KL divergence between the original
        and quantized output distributions. This aligns the global output
        statistics that block-level MSE cannot capture.

        Args:
            n_iters: optimization iterations
            lr: learning rate for scale optimization
            verbose: print progress

        Returns:
            final KL loss
        """
        if verbose:
            logger.info("KL calibration: capturing original model output...")

        # Capture original model output
        self.model_orig.eval()
        with torch.no_grad():
            orig_output = self.model_orig(self.calib_data)
            if hasattr(orig_output, 'logits'):
                orig_output = orig_output.logits
            elif isinstance(orig_output, tuple):
                orig_output = orig_output[0]
            orig_logits = orig_output.detach()

        # Collect all scale parameters across all blocks
        all_scale_params = {}
        for i, block in enumerate(self.blocks_quant):
            block_params = _get_optimizable_params(block)
            for name, val in block_params.items():
                # Only optimize scales, not binary factors (those are done in block recon)
                if not (name.endswith('.U_soft') or name.endswith('.V_soft')):
                    key = f"block{i}.{name}"
                    p = val.clone().to(self.device).requires_grad_(True)
                    all_scale_params[key] = p

        if not all_scale_params:
            if verbose:
                logger.info("  No scale parameters found for KL calibration")
            return 0.0

        # Install soft forward patches on all blocks
        all_patches = {}
        for i, block in enumerate(self.blocks_quant):
            block_opt = {k.replace(f"block{i}.", ""): v
                         for k, v in all_scale_params.items()
                         if k.startswith(f"block{i}.")}
            if block_opt:
                patches = _install_soft_forward(block, block_opt)
                all_patches[i] = (block, patches)

        optimizer = torch.optim.SGD(all_scale_params.values(), lr=lr, momentum=0.9)

        best_loss = float('inf')
        best_params = {k: v.detach().clone() for k, v in all_scale_params.items()}

        try:
            for iter in range(n_iters):
                optimizer.zero_grad()

                # Invalidate all caches
                for i, (block, _) in all_patches.items():
                    for m in block.modules():
                        if hasattr(m, '_invalidate_cache'):
                            m._invalidate_cache()

                # Run full quantized model
                with torch.enable_grad():
                    try:
                        quant_output = self.model_quant(self.calib_data)
                        if hasattr(quant_output, 'logits'):
                            quant_output = quant_output.logits
                        elif isinstance(quant_output, tuple):
                            quant_output = quant_output[0]
                    except Exception as e:
                        logger.warning(f"KL iter {iter}: forward failed: {e}")
                        break

                # KL divergence: D(quant || orig)
                # Use log_softmax for numerical stability
                p = F.log_softmax(quant_output.float(), dim=-1)
                q = F.softmax(orig_logits.float(), dim=-1)
                loss = F.kl_div(p, q, reduction='batchmean')

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_scale_params.values(), max_norm=1.0)
                optimizer.step()

                loss_val = loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    logger.warning(f"KL iter {iter}: NaN/Inf loss, stopping")
                    break

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_params = {k: v.detach().clone()
                                   for k, v in all_scale_params.items()}

                if verbose and (iter % 5 == 0 or iter == n_iters - 1):
                    logger.info(f"  KL iter {iter}: loss={loss_val:.6f}")
        finally:
            # Remove patches
            for i, (block, patches) in all_patches.items():
                _uninstall_soft_forward(patches)

        # Apply best params back to blocks
        for i, block in enumerate(self.blocks_quant):
            block_best = {k.replace(f"block{i}.", ""): v
                          for k, v in best_params.items()
                          if k.startswith(f"block{i}.")}
            if block_best:
                _apply_optimized_params(block, block_best)
                for m in block.modules():
                    if hasattr(m, '_invalidate_cache'):
                        m._invalidate_cache()

        if verbose:
            logger.info(f"KL calibration complete: best loss={best_loss:.6f}")

        del all_scale_params, optimizer
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return best_loss

    def reconstruct_with_error_mitigation(
        self,
        n_iters: int = 50,
        lr: float = 0.05,
        loss_fn: str = "mse",
        optimize_binary: bool = True,
        verbose: bool = True,
    ) -> dict[int, float]:
        """Sequential reconstruction with error propagation mitigation.

        NanoQuant Step 1: Before quantizing each block, adjust the FP weights
        to compensate for errors accumulated from prior blocks. This is done
        by computing the difference between the original block output and the
        actual quantized model output (which includes errors from all prior
        blocks), and using that to adjust the target for the current block.

        The key insight: instead of matching block_orig(x_orig), we match
        block_orig(x_corrected) where x_corrected is the actual (corrupted)
        input arriving at this block from the quantized model.

        Args:
            n_iters: optimization iterations per block
            lr: learning rate
            loss_fn: 'mse' or 'kl'
            verbose: print progress

        Returns:
            dict mapping block_idx → final loss
        """
        if verbose:
            logger.info("Error-mitigated reconstruction: capturing block I/O...")

        self.capture_block_io()

        results = {}
        current_input = self._block_inputs[0].to(self.device)

        for idx in range(self.n_blocks):
            if verbose:
                logger.info(f"  [{idx+1}/{self.n_blocks}] Error-mitigated block {idx}...")

            # Original target: what the original block produced from original input
            target_orig = self._block_outputs[idx].to(self.device)

            # Error mitigation: compute what the original block WOULD produce
            # from the current (corrupted) input. This is the error-corrected
            # target. We approximate it by running the original block with
            # the current input.
            with torch.no_grad():
                orig_block = self.blocks_orig[idx]
                kw = {}
                if idx < len(self._block_kwargs):
                    for k, v in self._block_kwargs[idx].items():
                        if isinstance(v, torch.Tensor):
                            kw[k] = v.to(self.device)
                        elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                            kw[k] = tuple(x.to(self.device) for x in v)
                        else:
                            kw[k] = v
                try:
                    cfg = getattr(self.model_orig, 'config', None)
                    if cfg is not None and hasattr(cfg, 'use_cache'):
                        cfg.use_cache = False
                    corrected_target = orig_block(current_input, **kw)
                    if cfg is not None and hasattr(cfg, 'use_cache'):
                        cfg.use_cache = True
                    if isinstance(corrected_target, tuple):
                        corrected_target = corrected_target[0]
                    corrected_target = corrected_target.detach()
                except Exception:
                    # Fallback: use original target
                    corrected_target = target_orig

            # Temporarily set the block's target to the corrected target
            # and input to the current (propagated) input
            original_input = self._block_inputs[idx]
            original_output = self._block_outputs[idx]
            self._block_inputs[idx] = current_input.cpu()
            self._block_outputs[idx] = corrected_target.cpu()

            try:
                loss = self.reconstruct_block(idx, n_iters=n_iters, lr=lr,
                                              loss_fn=loss_fn,
                                              optimize_binary=optimize_binary)
                results[idx] = loss
                if verbose:
                    logger.info(f"    Final loss: {loss:.6f}")
            except Exception as e:
                if verbose:
                    logger.info(f"    FAILED: {e}")
                results[idx] = float('inf')

            # Restore original I/O
            self._block_inputs[idx] = original_input
            self._block_outputs[idx] = original_output

            # Propagate current input through the reconstructed block
            with torch.no_grad():
                block = self.blocks_quant[idx]
                for m in block.modules():
                    if hasattr(m, '_invalidate_cache'):
                        m._invalidate_cache()
                try:
                    output = self._run_block_forward(block, current_input, idx)
                    if isinstance(output, tuple):
                        output = output[0]
                    current_input = output.detach()
                except Exception:
                    current_input = corrected_target

            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        if verbose:
            avg_loss = sum(results.values()) / max(len(results), 1)
            logger.info(f"Error-mitigated reconstruction complete: "
                  f"avg loss={avg_loss:.6f}")

        return results
