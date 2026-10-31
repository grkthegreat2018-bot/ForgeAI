"""Attention kernels (flash/SDPA wrappers, varlen, causal masks)."""
import logging
import math
import weakref

import torch
import torch.nn.functional as F

from forge.evolution.auto_targets import self_improve

logger = logging.getLogger(__name__)

# Packed-sequence metadata cache (cf. Unsloth/NVIDIA packed-metadata caching):
# cu_seqlens → (on-device int32 cu, max_seqlen, block bounds) is identical for
# every layer of a forward pass; recomputing per layer costs a D2H sync per
# layer (~N syncs per batch).  Keyed on the cu_seqlens object via a weakref +
# version counter so in-place mutation or buffer reuse recomputes safely.
_VARLEN_CACHE_MAX = 32
_varlen_meta_cache: dict = {}


def _varlen_meta(cu_seqlens: torch.Tensor, device: torch.device):
    """Return (cu_i32_list, max_list, blocks_list) for cu_seqlens, cached.

    Lists are per batch row: ``cu_i32_list[b]`` is the on-device int32
    cumulative-lengths tensor, ``max_list[b]`` the max example length, and
    ``blocks_list[b]`` the ``[(start, end), ...]`` non-empty example bounds.
    One host read (``tolist``) per *batch*, not per layer.
    """
    key = id(cu_seqlens)
    ver = cu_seqlens._version
    hit = _varlen_meta_cache.get(key)
    if hit is not None:
        ref, cached_ver, meta = hit
        if ref() is cu_seqlens and cached_ver == ver:
            return meta
    rows = cu_seqlens if cu_seqlens.dim() > 1 else cu_seqlens.unsqueeze(0)
    cu_list = [r.to(device, dtype=torch.int32) for r in rows]
    bounds = rows.tolist()
    max_list = []
    blocks_list = []
    for row in bounds:
        blocks = [(int(row[i]), int(row[i + 1])) for i in range(len(row) - 1)]
        max_list.append(max((e - s for s, e in blocks), default=0))
        blocks_list.append([b for b in blocks if b[1] > b[0]])
    meta = (cu_list, max_list, blocks_list)
    _varlen_meta_cache[key] = (weakref.ref(cu_seqlens), ver, meta)
    while len(_varlen_meta_cache) > _VARLEN_CACHE_MAX:
        _varlen_meta_cache.pop(next(iter(_varlen_meta_cache)))
    return meta

def flash_attention(q, k, v, is_causal=True) -> torch.Tensor:
    """Use FlashAttention-2 via PyTorch's SDPA when available.

    PyTorch 2.x automatically dispatches to FlashAttention-2 (FA2) on CUDA
    when using F.scaled_dot_product_attention with is_causal=True.
    This is ~2x faster than manual attention and uses O(1) memory.

    Falls back to manual computation on CPU.
    """
    if q.is_cuda:
        # FA2 is automatically used by SDPA on CUDA in PyTorch 2.x
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    else:
        # Manual attention for CPU
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            T, S = scores.size(-2), scores.size(-1)
            # Bottom-right-anchored causal mask — query row i may attend to
            # keys 0..(S-T+i), matching SDPA is_causal semantics when past
            # KV makes S > T (delta prefill).  The old tril(T,T) mask both
            # mis-anchored and failed to broadcast when S != T.
            q_pos = torch.arange(S - T, S, device=scores.device)
            k_pos = torch.arange(S, device=scores.device)
            mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # (T, S)
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


def varlen_attention(q, k, v, cu_seqlens, max_seqlen=None) -> torch.Tensor:
    """Variable-length attention for packed sequences (R&D round 14).

    Uses FlashAttention varlen API to attend WITHIN each packed example,
    preventing cross-example attention contamination. This is the correct
    way to handle packed sequences: instead of a global causal mask over
    the entire packed sequence, each example gets its own causal mask.

    Community: Unsloth 2.1x faster padding-free, 50% less VRAM.
    Requires flash_attn package (flash_attn_varlen_func).

    Args:
        q, k, v: (B, n_heads, T, head_dim) — standard attention tensors.
            For varlen, B should be 1 (all examples packed into one
            sequence) or we process each batch element separately.
        cu_seqlens: (B, n_examples+1) or (n_examples+1,) cumulative
            sequence lengths. If 2D, each batch element has its own
            set of examples. If 1D, all examples are in one sequence.
        max_seqlen: max sequence length (for FA varlen API). Auto-computed
            if None.

    Returns:
        (B, n_heads, T, head_dim) attention output.
    """
    # Try flash_attn varlen (fastest path, O(1) memory per example).
    try:
        from flash_attn import flash_attn_varlen_func
        # flash_attn expects (total_seq, n_heads, head_dim) without batch dim.
        # If B=1, squeeze the batch dim. If B>1, process per-batch.
        B, n_heads, T, head_dim = q.shape
        cu_list, max_list, _ = _varlen_meta(cu_seqlens, q.device)

        if B == 1:
            # Single packed sequence: squeeze batch, use cu_seqlens directly.
            q_vl = q.squeeze(0).transpose(0, 1)  # (T, n_heads, head_dim)
            k_vl = k.squeeze(0).transpose(0, 1)
            v_vl = v.squeeze(0).transpose(0, 1)
            cu = cu_list[0]
            ms = max_seqlen if max_seqlen is not None else max_list[0]
            out = flash_attn_varlen_func(
                q_vl, k_vl, v_vl, cu, cu, ms, ms,
                softmax_scale=1.0 / math.sqrt(head_dim), causal=True,
            )
            # (T, n_heads, head_dim) → (1, n_heads, T, head_dim)
            return out.transpose(0, 1).unsqueeze(0)
        else:
            # B > 1: process each batch element separately and concatenate.
            outs = []
            for b in range(B):
                q_vl = q[b].transpose(0, 1)
                k_vl = k[b].transpose(0, 1)
                v_vl = v[b].transpose(0, 1)
                i = b if len(cu_list) > 1 else 0
                ms = max_seqlen if max_seqlen is not None else max_list[i]
                o = flash_attn_varlen_func(
                    q_vl, k_vl, v_vl, cu_list[i], cu_list[i], ms, ms,
                    softmax_scale=1.0 / math.sqrt(head_dim), causal=True,
                )
                outs.append(o.transpose(0, 1))
            return torch.stack(outs, dim=0)
    except ImportError:
        pass

    # Fallback: block-diagonal causal mask via SDPA.
    # Build a mask where position (i, j) is valid only if i and j are in
    # the same example AND i >= j (causal). This is correct but slower
    # than flash_attn varlen (materializes the full T×T mask).
    B, n_heads, T, head_dim = q.shape
    _, _, blocks_list = _varlen_meta(cu_seqlens, q.device)
    if cu_seqlens.dim() > 1:
        # Process per-batch with block-diagonal masks.
        outs = []
        for b in range(B):
            mask = _build_block_diag_causal_mask(
                cu_seqlens[b], T, q.device, q.dtype, blocks=blocks_list[b])
            # SDPA expects (B, n_heads, T, T) mask → (1, 1, T, T) broadcast.
            out = F.scaled_dot_product_attention(
                q[b:b+1], k[b:b+1], v[b:b+1], attn_mask=mask.unsqueeze(0).unsqueeze(0))
            outs.append(out)
        return torch.cat(outs, dim=0)
    else:
        mask = _build_block_diag_causal_mask(
            cu_seqlens, T, q.device, q.dtype, blocks=blocks_list[0])
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0).expand(B, n_heads, T, T))


def _evolve_inputs_block_diag_mask():
    """Representative cases for _build_block_diag_causal_mask: packed-
    sequence cu_seqlens at real batch sizes, blocks precomputed (the
    hot path) and None (exercises _varlen_meta)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cases = []
    for segs in ((0, 5, 9, 14), (0, 128, 300, 512), (0, 8),
                 (0, 32, 64, 96, 128, 200)):
        T = segs[-1]
        cu = torch.tensor(list(segs), dtype=torch.int32, device=dev)
        blocks = [(int(segs[i]), int(segs[i + 1]))
                  for i in range(len(segs) - 1)]
        cases.append((cu, T, torch.device(dev), torch.float32, blocks))
        cases.append((cu, T, torch.device(dev), torch.float16, blocks))
    # blocks=None -> _varlen_meta path (int64 cpu cu_seqlens like callers)
    cu = torch.tensor([0, 40, 90, 160], dtype=torch.int64)
    cases.append((cu, 160, torch.device(dev), torch.float32, None))
    return cases


@self_improve(gen_input=_evolve_inputs_block_diag_mask, evals=80,
              deps=("_varlen_meta",),
              imports=("import weakref",),
              globals={"_VARLEN_CACHE_MAX": 32, "_varlen_meta_cache": {}},
              tags=("attention", "mask", "varlen", "hot-path"),
              prompt=("Faster _build_block_diag_causal_mask(cu_seqlens, "
                      "T, device, dtype, blocks): additive (T,T) mask, 0 "
                      "inside each block's causal tril, -inf elsewhere. "
                      "Seed allocates a full -inf (T,T) then per-block "
                      "builds tril+where and copies it in. Ideas: one "
                      "vectorized segment-id compare (no per-block "
                      "loop), tril of a zeros matrix + scatter -inf "
                      "outside blocks, preallocated buffers. Same mask "
                      "values."))
def _build_block_diag_causal_mask(cu_seqlens, T, device, dtype, blocks=None):
    """Build a block-diagonal causal mask for varlen attention fallback.

    Each block (defined by consecutive cu_seqlens entries) gets its own
    causal mask. Positions outside the block are masked to -inf.

    Args:
        cu_seqlens: (n_examples+1,) cumulative sequence lengths.
        T: total sequence length.
        device, dtype: for the mask tensor.
        blocks: optional precomputed [(start, end), ...] from _varlen_meta —
            skips the per-call host read.

    Returns:
        (T, T) additive mask: 0 for valid positions, -inf for invalid.
    """
    if blocks is None:
        blocks = _varlen_meta(cu_seqlens, device)[2][0]
    mask = torch.full((T, T), float('-inf'), device=device, dtype=dtype)
    for start, end in blocks:
        # Causal mask within this block
        block_len = end - start
        causal = torch.tril(torch.ones(block_len, block_len, device=device, dtype=dtype))
        mask[start:end, start:end] = torch.where(
            causal > 0, torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float('-inf'), device=device, dtype=dtype))
    return mask


def _evolve_inputs_causal_mask():
    """Representative cases for _causal_mask: prefill (past=0, T==total),
    chunked prefill (past>0) and single-token decode (seq_len=1)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cases = []
    for seq, total, past in ((1, 1, 0), (512, 512, 0), (128, 384, 256),
                             (1, 1024, 1023), (64, 64, 0),
                             (2048, 2048, 0)):
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            cases.append((seq, total, past, torch.device(dev), dt))
    cases.append((16, 48, 32, torch.device(dev), None))   # dtype default
    return cases


@self_improve(gen_input=_evolve_inputs_causal_mask, evals=80,
              tags=("attention", "mask", "hot-path"),
              prompt=("Faster _causal_mask(seq_len, total_len, past_len,"
                      " device, dtype): additive causal mask, -inf above "
                      "the (past_len+1)-shifted diagonal. Seed allocates "
                      "a full -inf matrix then triu's it. Ideas: arange "
                      "compare -> bool/select (one kernel), "
                      "masked_fill on zeros, new_full + where. Same "
                      "values/dtype/shape."))
def _causal_mask(seq_len: int, total_len: int, past_len: int, device: torch.device,
                 dtype: torch.dtype = None) -> torch.Tensor:
    """Create a causal mask for a query of length `seq_len` attending to `total_len` keys."""
    # For standard prefill (past_len == 0, seq_len == total_len) this is the usual upper-triangular mask.
    if dtype is None:
        dtype = torch.float32
    return torch.triu(torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
                      diagonal=past_len + 1)


