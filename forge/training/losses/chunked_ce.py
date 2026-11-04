"""Pure-PyTorch Fused Linear Cross-Entropy (FLCE) — no Triton dependency.

Achieves the same memory savings as Liger FLCE by chunking the token dimension
and never materializing the full [B*T, V] logits tensor. Works on any GPU,
including consumer Blackwell (sm_120) where Liger's Triton kernels crash.

Trade-off: ~2-3x slower than Liger's fused Triton kernel, but same memory profile
and exact numerical parity with F.cross_entropy.

Also provides ChunkedEntropyWeightedCE for token-entropy-weighted loss
(WeFT/VCORE 2025) without materializing full logits — 20x+ faster than the
naive full-logits path on large vocabularies.
"""
import torch
import torch.nn.functional as F
from torch.autograd import Function


class ChunkedLinearCrossEntropy(Function):
    """Fused linear + cross-entropy with chunked computation.

    Computes: loss = cross_entropy(x @ weight.t(), target)
    Without ever materializing the full [N, V] logits tensor.

    Args (forward):
        x:       [N, H]  hidden states (requires_grad)
        weight:  [V, H]  LM head weight (requires_grad)
        target:  [N]     token ids
        chunk_size: number of tokens per chunk (tune for memory/speed)

    Returns:
        loss scalar
    """

    @staticmethod
    def forward(ctx, x, weight, target, chunk_size=512):
        N, _H = x.shape
        weight.shape[0]
        device = x.device
        ignore_index = -100
        # Cast weight to x's dtype for the matmul (fixes bf16/fp32 mismatch).
        weight_cast = weight.to(x.dtype)

        grad_input = torch.zeros_like(x) if x.requires_grad else None
        grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
        loss_total = torch.zeros((), device=device, dtype=torch.float32)
        n_valid = 0  # count of non-ignored tokens for mean reduction

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x[start:end]                          # [cs, H]
            target_chunk = target[start:end]                # [cs]

            # Mask for non-ignored tokens (target != ignore_index)
            valid_mask = target_chunk != ignore_index       # [cs]
            n_valid_chunk = valid_mask.sum().item()
            n_valid += n_valid_chunk

            # Materialize logits only for this chunk: [cs, V]
            logits_chunk = x_chunk @ weight_cast.t()
            # F.cross_entropy handles ignore_index=-100 by default
            loss_chunk = F.cross_entropy(
                logits_chunk.float(), target_chunk, reduction="sum"
            )
            loss_total = loss_total + loss_chunk

            if x.requires_grad:
                # grad_logits = softmax(logits) - one_hot(target), then / n_valid for mean
                with torch.no_grad():
                    probs = F.softmax(logits_chunk.float(), dim=-1)  # [cs, V]
                    # Subtract 1 at target positions (gradient of CE w.r.t. logits)
                    # Only for non-ignored tokens; use clamped index to avoid OOB.
                    safe_target = target_chunk.clamp(min=0)  # -100 → 0 (will be masked)
                    probs.scatter_add_(
                        1, safe_target.unsqueeze(1),
                        torch.full_like(probs[:, :1], -1.0),
                    )
                    # Zero out gradients for ignored positions
                    probs = probs * valid_mask.unsqueeze(1).to(probs.dtype)

                # grad_input_chunk = grad_logits @ weight  → [cs, H]
                grad_input[start:end] = probs.to(x.dtype) @ weight_cast
                # grad_weight += grad_logits.t() @ x_chunk → [V, H]
                if grad_weight is not None:
                    grad_weight += probs.to(x.dtype).t() @ x_chunk

            # logits_chunk and probs go out of scope here → freed

        # Mean over non-ignored tokens (not total N)
        loss = loss_total / max(n_valid, 1)

        ctx.save_for_backward(grad_input, grad_weight)
        ctx.n_valid = max(n_valid, 1)
        return loss.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        # Forward accumulated d(loss_total)/dx (sum reduction); loss = loss_total / n_valid.
        # Scale by 1/n_valid to get d(loss)/dx, then by grad_output for chain rule.
        scale = grad_output / ctx.n_valid
        grad_input = grad_input * scale
        if grad_weight is not None:
            grad_weight = grad_weight * scale
        return grad_input, grad_weight, None, None


class ChunkedEntropyWeightedCE(Function):
    """Chunked linear + cross-entropy with token-entropy weighting (WeFT/VCORE 2025).

    Computes: loss = mean( (1 + alpha * norm_entropy) * ce_per_token )
    Without ever materializing the full [N, V] logits tensor.

    The entropy weight is computed under no_grad (treated as constant in backward),
    so the gradient is simply: weight * (softmax - one_hot) / n_valid.

    Args (forward):
        x:       [N, H]  hidden states (requires_grad)
        weight:  [V, H]  LM head weight (requires_grad)
        target:  [N]     token ids
        chunk_size: tokens per chunk
        entropy_alpha: weighting strength (0=disabled, 0.5=production default)

    Returns:
        loss scalar
    """

    @staticmethod
    def forward(ctx, x, weight, target, chunk_size=512, entropy_alpha=0.5):
        N, _H = x.shape
        V = weight.shape[0]
        device = x.device
        ignore_index = -100
        weight_cast = weight.to(x.dtype)
        max_entropy = torch.log(torch.tensor(float(V), device=device, dtype=torch.float32))

        grad_input = torch.zeros_like(x) if x.requires_grad else None
        grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
        loss_total = torch.zeros((), device=device, dtype=torch.float32)
        n_valid = 0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x[start:end]
            target_chunk = target[start:end]
            valid_mask = target_chunk != ignore_index
            n_valid_chunk = valid_mask.sum().item()
            n_valid += n_valid_chunk

            # Logits for this chunk only: [cs, V]
            logits_chunk = x_chunk @ weight_cast.t()  # [cs, V] bf16
            logits_f = logits_chunk.float()  # [cs, V] fp32 for stability

            # Per-token CE (sum reduction, we'll divide later)
            ce_chunk = F.cross_entropy(
                logits_f, target_chunk, reduction="none"
            )  # [cs]

            # Entropy weighting (no_grad — constant in backward)
            with torch.no_grad():
                probs = F.softmax(logits_f, dim=-1)  # [cs, V]
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [cs]
                norm_entropy = (entropy / max_entropy.clamp(min=1e-8)) * 2.0  # [0, 2]
                ent_weight = 1.0 + entropy_alpha * norm_entropy  # [cs]

            # Weighted CE
            weighted_ce = ce_chunk * ent_weight * valid_mask.to(ce_chunk.dtype)
            loss_total = loss_total + weighted_ce.sum()

            # Gradient: weight * (softmax - one_hot) / n_valid
            if x.requires_grad:
                with torch.no_grad():
                    # Reuse probs from entropy computation
                    safe_target = target_chunk.clamp(min=0)
                    probs.scatter_add_(
                        1, safe_target.unsqueeze(1),
                        torch.full_like(probs[:, :1], -1.0),
                    )
                    # Apply entropy weight and valid mask
                    probs = probs * ent_weight.unsqueeze(1)
                    probs = probs * valid_mask.unsqueeze(1).to(probs.dtype)

                grad_input[start:end] = probs.to(x.dtype) @ weight_cast
                if grad_weight is not None:
                    grad_weight += probs.to(x.dtype).t() @ x_chunk

            # logits_chunk, probs, etc. freed here

        loss = loss_total / max(n_valid, 1)
        ctx.save_for_backward(grad_input, grad_weight)
        ctx.n_valid = max(n_valid, 1)
        return loss.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        scale = grad_output / ctx.n_valid
        grad_input = grad_input * scale
        if grad_weight is not None:
            grad_weight = grad_weight * scale
        return grad_input, grad_weight, None, None, None


class ChunkedDistillCE(Function):
    """Chunked linear + advantage-weighted CE for offline on-policy distillation.

    Implements the Thinking Machines / Agarwal et al. on-policy distillation
    update as a weighted cross-entropy, against *cached* per-token teacher
    log-probs (sequence-level, so teacher and student may use different
    tokenizers — the scorer detokenizes student text and re-scores it in the
    teacher's own vocab).

    Per response token t:
        A_t = stopgrad(teacher_logp_t - student_logp_t)
        w_t = floor + (1 - floor) * clip(alpha * A_t, -clip, +clip)
        loss = sum(w_t * ce_t) / n_valid

    A_t > 0 (teacher likes the token more than the student) reinforces it;
    A_t < 0 suppresses it. floor keeps a plain-CE weight on every token so the
    loss cannot collapse to zero signal when teacher and student already
    agree; floor=1.0 reduces exactly to chunked_linear_cross_entropy.

    Tokens with teacher_logp <= TEACHER_LOGP_MISSING are treated as unscored
    and get weight 1.0 (neutral CE). Prompt/pad positions are masked by
    target == -100 as usual.

    Never materializes the full [N, V] logits tensor.
    """

    @staticmethod
    def forward(ctx, x, weight, target, teacher_logp, chunk_size=512,
                alpha=1.0, clip=5.0, floor=0.3):
        N, _H = x.shape
        device = x.device
        ignore_index = -100
        missing = ChunkedDistillCE.TEACHER_LOGP_MISSING
        weight_cast = weight.to(x.dtype)
        teacher_logp = teacher_logp.to(device=device, dtype=torch.float32)

        grad_input = torch.zeros_like(x) if x.requires_grad else None
        grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
        loss_total = torch.zeros((), device=device, dtype=torch.float32)
        n_valid = 0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x[start:end]
            target_chunk = target[start:end]
            tlp_chunk = teacher_logp[start:end]
            valid_mask = target_chunk != ignore_index
            n_valid += valid_mask.sum().item()

            logits_chunk = x_chunk @ weight_cast.t()
            logits_f = logits_chunk.float()

            # Per-token CE (= -student_logp) and softmax for the gradient.
            ce_chunk = F.cross_entropy(
                logits_f, target_chunk, reduction="none"
            )  # [cs]
            with torch.no_grad():
                probs = F.softmax(logits_f, dim=-1)
                student_logp = -ce_chunk
                adv = (tlp_chunk - student_logp) * alpha
                w = floor + (1.0 - floor) * torch.clamp(adv, -clip, clip)
                # Unscored tokens (or prompt/pad) fall back to neutral CE.
                w = torch.where(tlp_chunk <= missing,
                                torch.ones_like(w), w)

            weighted_ce = ce_chunk * w * valid_mask.to(ce_chunk.dtype)
            loss_total = loss_total + weighted_ce.sum()

            if x.requires_grad:
                with torch.no_grad():
                    safe_target = target_chunk.clamp(min=0)
                    probs.scatter_add_(
                        1, safe_target.unsqueeze(1),
                        torch.full_like(probs[:, :1], -1.0),
                    )
                    probs = probs * w.unsqueeze(1)
                    probs = probs * valid_mask.unsqueeze(1).to(probs.dtype)

                grad_input[start:end] = probs.to(x.dtype) @ weight_cast
                if grad_weight is not None:
                    grad_weight += probs.to(x.dtype).t() @ x_chunk

        loss = loss_total / max(n_valid, 1)
        ctx.save_for_backward(grad_input, grad_weight)
        ctx.n_valid = max(n_valid, 1)
        return loss.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        scale = grad_output / ctx.n_valid
        grad_input = grad_input * scale
        if grad_weight is not None:
            grad_weight = grad_weight * scale
        return grad_input, grad_weight, None, None, None, None, None, None


# Sentinel written by score_teacher_logprobs.py for student tokens it could
# not map to teacher logprobs. Treated as "neutral CE weight" by ChunkedDistillCE.
ChunkedDistillCE.TEACHER_LOGP_MISSING = -99.0


def chunked_linear_cross_entropy(x, weight, target, chunk_size=512):
    """Functional interface for ChunkedLinearCrossEntropy."""
    return ChunkedLinearCrossEntropy.apply(x, weight, target, chunk_size)


def chunked_distill_ce(x, weight, target, teacher_logp, chunk_size=512,
                       alpha=1.0, clip=5.0, floor=0.3):
    """Functional interface for ChunkedDistillCE (advantage-weighted CE).

    teacher_logp: [N] float tensor aligned with target (flat). Entries <=
    TEACHER_LOGP_MISSING are treated as unscored → neutral CE weight.
    """
    return ChunkedDistillCE.apply(x, weight, target, teacher_logp,
                                  chunk_size, alpha, clip, floor)


def chunked_entropy_weighted_ce(x, weight, target, chunk_size=512, entropy_alpha=0.5):
    """Functional interface for ChunkedEntropyWeightedCE.

    Computes entropy-weighted CE without materializing full [N, V] logits.
    ~20x faster than the naive full-logits path on vocab=65536.
    """
    return ChunkedEntropyWeightedCE.apply(x, weight, target, chunk_size, entropy_alpha)
