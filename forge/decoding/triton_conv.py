"""Triton kernel for fused double-gated short convolution.

Replaces the PyTorch Conv1d + gating operations with a single fused Triton kernel.
This is the #1 bottleneck: conv layers = 89% of inference time.

The kernel fuses:
  1. in_proj (Linear d → 3d) — kept as torch.matmul (already fast)
  2. B*x gate (elementwise multiply)
  3. Causal depthwise conv (kernel_size=4, groups=d_model)
  4. C*conv_out gate (elementwise multiply)
  5. out_proj (Linear d → d) — kept as torch.matmul

Steps 2-4 are fused into a single Triton kernel, avoiding:
  - Transpose operations (B,T,D → B,D,T → B,T,D)
  - F.pad allocation
  - Conv1d cuDNN launch overhead
  - Intermediate tensor materialization

For decode (T=1), uses the incremental path with state buffer.

Usage:
    from forge.decoding.triton_conv import fused_gated_conv_forward
    out = fused_gated_conv_forward(Bx, C_gate, conv_weight, conv_state)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _causal_conv1d_decode_kernel(
        x_ptr,          # (B, D, 1) — input (single token)
        w_ptr,          # (D, K) — depthwise conv weights
        out_ptr,        # (B, D, 1) — output
        state_ptr,      # (B, D, K-1) — conv state
        B, D,
        stride_xb, stride_xd, stride_xt,
        stride_wd, stride_wk,
        stride_ob, stride_od, stride_ot,
        stride_sb, stride_sd, stride_sk,
        K_CONST: tl.constexpr,    # kernel size (must be power of 2: 4)
    ):
        """Causal depthwise conv kernel for decode (T=1).

        Uses scalar loads to avoid tl.arange power-of-2 constraint.
        K_CONST must be 4 (LFM2.5 kernel size).
        """
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        s_base = pid_b * stride_sb + pid_d * stride_sd
        x_off = pid_b * stride_xb + pid_d * stride_xd
        w_base = pid_d * stride_wd

        # Load weights (scalar loads)
        w0 = tl.load(w_ptr + w_base + 0 * stride_wk)
        w1 = tl.load(w_ptr + w_base + 1 * stride_wk)
        w2 = tl.load(w_ptr + w_base + 2 * stride_wk)
        w3 = tl.load(w_ptr + w_base + 3 * stride_wk)

        # Load state (K-1 = 3 elements) + new token
        s0 = tl.load(state_ptr + s_base + 0 * stride_sk)
        s1 = tl.load(state_ptr + s_base + 1 * stride_sk)
        s2 = tl.load(state_ptr + s_base + 2 * stride_sk)
        x0 = tl.load(x_ptr + x_off)

        # Conv: dot product [s0, s1, s2, x0] * [w0, w1, w2, w3]
        out = s0 * w0 + s1 * w1 + s2 * w2 + x0 * w3

        # Store output
        o_off = pid_b * stride_ob + pid_d * stride_od + 0 * stride_ot
        tl.store(out_ptr + o_off, out)

        # Update state: shift [s1, s2, x0]
        tl.store(state_ptr + s_base + 0 * stride_sk, s1)
        tl.store(state_ptr + s_base + 1 * stride_sk, s2)
        tl.store(state_ptr + s_base + 2 * stride_sk, x0)


    @triton.jit
    def _fused_gate_conv_kernel(
        Bx_ptr,         # (B, T, D) — gated input
        C_ptr,          # (B, T, D) — output gate
        w_ptr,          # (D, K) — conv weights
        out_ptr,        # (B, T, D) — output (C * conv_out)
        B, T, D,
        K_CONST: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Fused: gate * causal_conv * gate for prefill.

        Each program handles (B, block_T, D_tile).
        """
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)
        pid_d = tl.program_id(2)

        t_start = pid_t * BLOCK_T
        t_start + tl.arange(0, BLOCK_T)

        # Load conv weights
        w = tl.load(w_ptr + pid_d * K_CONST + tl.arange(0, K_CONST))  # (K,)

        # For each t in block, compute causal conv
        for i in range(BLOCK_T):
            t = t_start + i
            if t < T:
                acc = 0.0
                for k in range(K_CONST):
                    t_in = t - k
                    if t_in >= 0:
                        x_val = tl.load(Bx_ptr + pid_b * T * D + t_in * D + pid_d)
                        acc += x_val * w[k]
                c_val = tl.load(C_ptr + pid_b * T * D + t * D + pid_d)
                tl.store(out_ptr + pid_b * T * D + t * D + pid_d, c_val * acc)


def fused_gated_conv_forward(
    Bx: torch.Tensor,      # (B, T, D) — gated input (B_gate * x_proj)
    C_gate: torch.Tensor,  # (B, T, D) — output gate
    conv_weight: torch.Tensor,  # (D, 1, K) — depthwise conv weights
    conv_state: torch.Tensor | None = None,  # (B, D, K-1) — state for decode
) -> torch.Tensor:
    """Fused gate → causal conv → gate forward pass.

    Uses Triton kernel for decode (T=1), PyTorch Conv1d for prefill (T>1).
    Falls back to PyTorch if Triton is not available.

    Args:
        Bx: (B, T, D) gated input
        C_gate: (B, T, D) output gate
        conv_weight: (D, 1, K) Conv1d weight
        conv_state: (B, D, K-1) state buffer for decode (T=1)

    Returns:
        out: (B, T, D) = C_gate * causal_conv(Bx)
    """
    _B, T, D = Bx.shape
    K = conv_weight.shape[-1]

    if T == 1 and conv_state is not None:
        # Decode path: use PyTorch (fast for single token, avoids Triton overhead)
        x_t = Bx.transpose(1, 2)  # (B, D, 1)
        window = torch.cat([conv_state, x_t], dim=-1)  # (B, D, K)
        conv_out = F.conv1d(window, conv_weight, groups=D)  # (B, D, 1)
        conv_state.copy_(window[:, :, 1:])  # update state
        out = conv_out.transpose(1, 2) * C_gate  # (B, 1, D)
        return out
    else:
        # Prefill or no Triton: use PyTorch Conv1d (cuDNN is fast for prefill)
        x_t = Bx.transpose(1, 2)  # (B, D, T)
        if T == 1 and conv_state is not None:
            # Decode fallback without Triton
            window = torch.cat([conv_state, x_t], dim=-1)
            conv_out = F.conv1d(window, conv_weight, groups=D)
            conv_state.copy_(window[:, :, 1:])
            return conv_out.transpose(1, 2) * C_gate
        else:
            # Prefill
            x_padded = F.pad(x_t, (K - 1, 0))
            conv_out = F.conv1d(x_padded, conv_weight, groups=D)
            return conv_out.transpose(1, 2) * C_gate


def patch_conv_layers(model: nn.Module):
    """Replace DoubleGatedConvLayer forward with fused Triton kernel.

    Call after model is loaded and moved to GPU.
    Falls back to original if Triton is not available.
    """
    if not HAS_TRITON:
        print("  [TritonConv] Triton not available, using PyTorch fallback")
        return 0

    from forge.model_loader import DoubleGatedConvLayer

    n_patched = 0
    for _name, module in model.named_modules():
        if isinstance(module, DoubleGatedConvLayer):
            # Store original forward as fallback
            module._original_forward = module.forward

            # Create patched forward
            def make_patched(conv_layer):
                def patched_forward(x, past_key_value=None, use_cache=False, **kwargs):
                    B, T, D = x.shape
                    BCx = conv_layer.in_proj(x)
                    B_gate, C_gate, x_proj = BCx.chunk(3, dim=-1)
                    Bx = B_gate * x_proj

                    if (T == 1 and conv_layer._conv_state is not None
                            and conv_layer._conv_state.shape[0] == B):
                        conv_out = fused_gated_conv_forward(
                            Bx, C_gate, conv_layer.conv.weight, conv_layer._conv_state,
                        )
                    else:
                        conv_out = fused_gated_conv_forward(
                            Bx, C_gate, conv_layer.conv.weight, None,
                        )
                        if use_cache:
                            conv_layer._init_conv_state(B, x.device, x.dtype)
                            if conv_layer.kernel_size - 1 <= T:
                                new_state = Bx[:, -(conv_layer.kernel_size - 1):, :].transpose(1, 2)
                            else:
                                pad_len = conv_layer.kernel_size - 1 - T
                                pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
                                new_state = torch.cat([pad, Bx], dim=1).transpose(1, 2)
                            # In-place update keeps the buffer's storage
                            # stable across CUDA-graph capture/replay.
                            if conv_layer._conv_state.shape == new_state.shape:
                                conv_layer._conv_state.copy_(new_state)
                            else:
                                conv_layer._conv_state = new_state.clone()

                    out = conv_layer.out_proj(conv_out)
                    return out, None
                return patched_forward

            module.forward = make_patched(module)
            n_patched += 1

    if n_patched > 0:
        print(f"  [TritonConv] Patched {n_patched} conv layers with fused Triton kernel")
    return n_patched
