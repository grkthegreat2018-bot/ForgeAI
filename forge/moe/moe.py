"""Sparse Mixture-of-Experts (MoE) FFN layer.

Replaces dense FFN with top-k routed experts. For small models, we use
2-4 experts with top-1 or top-2 routing (vs large models using 8-64).

Key features:
- Router with load balancing loss (auxiliary loss)
- Top-k gating with noisy top-k (for exploration during training)
- Expert capacity factor (drop tokens if expert is overloaded)
- Shared expert (always-active, like DeepSeek-V3) for better quality

Memory: with 4 experts at 2x expansion, FFN params = 4x dense.
But only 1-2 experts are active per token, so FLOPs stay similar.

Usage:
    from forge.moe import MoELayer, replace_ffn_with_moe

    # Replace FFN in existing model
    replace_ffn_with_moe(model, n_experts=4, top_k=2, d_model=1024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Top-k router for MoE. Outputs expert assignments + gating weights.

    Two load-balancing modes:
      - mode="switch" (default for backward compat): Switch-Transformer aux
        loss = n_experts * sum(fraction * mean_gate), computed from top-k
        dispatched weights (non-differentiable fraction term).
      - mode="aux_free" (DeepSeek-V3): adds a learnable per-expert bias to
        routing logits AND uses a sequence-wise balance loss computed from
        the FULL softmax probabilities (differentiable through all experts,
        no top-k truncation in the loss). The bias is updated by the balance
        loss only (decoupled from the main gate via stop-gradient on the
        gate's contribution to the bias update path). This is the
        "auxiliary-loss-free load balancing" from DeepSeek-V3 §3.3.

    Args:
        d_model: input dimension
        n_experts: number of experts
        top_k: number of experts to route each token to
        noisy_gating: if True, add noise during training (exploration)
        load_balance_loss_weight: weight of auxiliary load balancing loss
        mode: "switch" or "aux_free"
        gating: "softmax" (default) or "sigmoid" (MiniMax-M2 style: selection
            on sigmoid scores, weights = renormalized selected scores)
    """

    def __init__(self, d_model, n_experts, top_k=2, noisy_gating=True,
                 load_balance_loss_weight=0.01, mode="switch", gating="softmax"):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.noisy_gating = noisy_gating
        self.load_balance_loss_weight = load_balance_loss_weight
        self.mode = mode
        self.gating = gating

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        if noisy_gating:
            self.noise = nn.Linear(d_model, n_experts, bias=False)
            self.noise_scale = nn.Parameter(torch.ones(1) * 0.1)
        # DeepSeek-V3 aux-loss-free: per-expert bias added to routing logits.
        # This is a BUFFER (not a Parameter) — updated by a direct load-
        # statistic rule (update_bias()), NOT by backprop. The bias steers
        # top-k selection toward underused experts; the gate is trained by
        # the sequence-wise balance loss (aux_loss) for quality + balance.
        # Init to zero (lossless — bias has no effect at start).
        if mode == "aux_free":
            self.register_buffer("expert_bias", torch.zeros(n_experts))
            self.bias_update_rate = 1e-3  # gamma in DeepSeek-V3 §3.3

    def forward(self, x):
        """Route tokens to experts.

        Args:
            x: (B, T, d_model) or (N, d_model) flattened tokens

        Returns:
            dispatch_mask: (N, n_experts) — 1.0 if token routed to expert
            gating_weights: (N, n_experts) — gating weights (0 for non-routed)
            aux_loss: scalar load balancing loss
        """
        if x.dim() == 3:
            x = x.view(-1, self.d_model)  # (N, d_model)
        N = x.shape[0]

        # Compute routing logits.
        clean_logits = self.gate(x)  # (N, n_experts)

        # DeepSeek-V3: add learnable per-expert bias to logits for routing
        # decisions (bias steers traffic toward underused experts), but the
        # bias is NOT used in the gating weight softmax (so expert output
        # scaling is unaffected by the balance mechanism).
        if self.mode == "aux_free":
            routing_logits = clean_logits + self.expert_bias  # used for top-k
            gating_logits = clean_logits                      # used for weights
        else:
            routing_logits = clean_logits
            gating_logits = clean_logits

        if self.noisy_gating and self.training:
            noise = torch.randn_like(clean_logits) * self.noise_scale
            routing_logits = routing_logits + noise * F.softplus(self.noise(x))

        # Top-k selection on routing logits (with bias).
        if self.gating == "sigmoid":
            # MiniMax-M2: selection on sigmoid scores (bias included via
            # routing_logits), weights = renormalized selected sigmoid scores.
            top_k_scores, top_k_indices = torch.sigmoid(routing_logits).topk(
                self.top_k, dim=-1)
            top_k_weights = top_k_scores / top_k_scores.sum(
                dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            top_k_logits, top_k_indices = routing_logits.topk(self.top_k, dim=-1)
            # Gating weights from gating logits (without bias) — recompute softmax
            # over the selected top-k experts using the unbiased logits.
            if self.mode == "aux_free":
                top_k_gating_logits = gating_logits.gather(1, top_k_indices)
                top_k_weights = F.softmax(top_k_gating_logits, dim=-1)
            else:
                top_k_weights = F.softmax(top_k_logits, dim=-1)  # (N, top_k)

        # Build dispatch mask and gating weights.
        dispatch_mask = torch.zeros(N, self.n_experts, device=x.device, dtype=x.dtype)
        gating_weights = torch.zeros(N, self.n_experts, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # (N,)
            weight = top_k_weights[:, k]  # (N,)
            dispatch_mask.scatter_(1, expert_idx.unsqueeze(1), 1.0)
            gating_weights.scatter_(1, expert_idx.unsqueeze(1), weight.unsqueeze(1).to(gating_weights.dtype))

        if self.mode == "aux_free":
            # DeepSeek-V3 auxiliary-loss-free balance loss.
            # Uses FULL softmax probabilities (differentiable through all
            # experts, not truncated to top-k). Computed per-sequence then
            # averaged, matching DeepSeek-V3's sequence-wise formulation.
            #   f_i = fraction of tokens dispatched to expert i (detached)
            #   P_i = softmax(gating_logits)_i   (full, not top-k)
            #   loss = n_experts * sum(f_i * P_i)
            # This loss trains the GATE (via P_i) for quality-aware balance.
            # The expert_bias is updated separately by update_bias() using a
            # direct load-statistic rule (not backprop) — DeepSeek-V3 §3.3.
            probs_full = (torch.sigmoid(gating_logits) if self.gating == "sigmoid"
                          else F.softmax(gating_logits, dim=-1))  # (N, n_experts)
            f_per_expert = dispatch_mask.detach().float().mean(dim=0)  # (n_experts,)
            mean_prob = probs_full.mean(dim=0)  # (n_experts,)
            aux_loss = self.n_experts * (f_per_expert * mean_prob).sum()
            # Store load stats for update_bias() (called by trainer after step)
            self._last_load = f_per_expert.detach()
        else:
            # Switch Transformer aux loss (original path, backward compat).
            with torch.no_grad():
                tokens_per_expert = dispatch_mask.sum(dim=0)  # (n_experts,)
                fraction_per_expert = tokens_per_expert / (N * self.top_k)
                mean_gate = gating_weights.mean(dim=0)  # (n_experts,)
            aux_loss = self.n_experts * (fraction_per_expert * mean_gate).sum()

        return dispatch_mask, gating_weights, aux_loss * self.load_balance_loss_weight

    @torch.no_grad()
    def update_bias(self):
        """DeepSeek-V3 aux-loss-free bias update rule.

        After each forward, adjust expert_bias so underused experts get a
        positive bias (more likely to be selected) and overloaded experts get
        a negative bias. The update is:

            b_i += gamma * (target_fraction - actual_fraction_i)

        where target_fraction = top_k / n_experts (uniform ideal). This is
        called by the trainer after each optimizer step (not by backprop).

        No-op for mode="switch" (no bias buffer).
        """
        if self.mode != "aux_free" or not hasattr(self, "_last_load"):
            return
        target = self.top_k / self.n_experts
        # _last_load is on the forward device; bias is on the same device.
        delta = self.bias_update_rate * (target - self._last_load)
        self.expert_bias.add_(delta.to(self.expert_bias.dtype))


class Expert(nn.Module):
    """A single FFN expert (SwiGLU-style).

    With use_clamp=True, uses GPT-OSS clamped SwiGLU (outlier prevention).

    With intra_sparsity > 0, exploits intra-expert activation sparsity
    (arXiv 2605.08575): SwiGLU FFNs have naturally sparse activations
    (up to 90% near-zero values in pretrained MoE models). Thresholds
    the intermediate activation to skip computation in w2 for inactive
    neurons, achieving up to 2.5x MoE layer speedup with no training.
    """

    def __init__(self, d_model, d_ff=None, activation="swiglu",
                 use_clamp=False, clamp_alpha=1.702, clamp_limit=7.0,
                 intra_sparsity: float = 0.0):
        super().__init__()
        d_ff = d_ff or d_model * 2  # smaller than dense (4x) since we have multiple experts
        if activation == "swiglu":
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = activation
        self.use_clamp = use_clamp
        self.clamp_alpha = clamp_alpha
        self.clamp_limit = clamp_limit
        # Intra-expert activation sparsity (R3-24, arXiv 2605.08575).
        # 0 = disabled (dense). >0 = threshold fraction of max activation
        # below which neurons are zeroed before w2 projection.
        self.intra_sparsity = intra_sparsity

    def forward(self, x):
        if self.activation == "swiglu":
            gate = self.w1(x)
            up = self.w3(x)
            if self.use_clamp:
                gate = gate.clamp(min=None, max=self.clamp_limit)
                up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
                glu = gate * torch.sigmoid(self.clamp_alpha * gate)
                h = (up + 1) * glu
            else:
                h = F.silu(gate) * up
            # Intra-expert activation sparsity: zero out inactive neurons
            # before the w2 projection. This skips computation for neurons
            # whose activation magnitude is below a fraction of the max.
            # Training-free, no quality loss at moderate sparsity (up to 90%
            # of neurons are already near-zero in pretrained MoE models).
            if self.intra_sparsity > 0.0 and not self.training:
                threshold = h.abs().max(dim=-1, keepdim=True).values * self.intra_sparsity
                h = torch.where(h.abs() >= threshold, h, torch.zeros_like(h))
            return self.w2(h)
        else:
            h = F.gelu(self.fc1(x))
            if self.intra_sparsity > 0.0 and not self.training:
                threshold = h.abs().max(dim=-1, keepdim=True).values * self.intra_sparsity
                h = torch.where(h.abs() >= threshold, h, torch.zeros_like(h))
            return self.fc2(h)


class MoELayer(nn.Module):
    """Mixture-of-Experts FFN layer with optional shared expert.

    Args:
        d_model: model dimension
        n_experts: number of routed experts
        top_k: experts activated per token
        d_ff: hidden dim per expert (default 2*d_model)
        shared_expert: if True, add an always-active shared expert (DeepSeek-V3 style)
        capacity_factor: max tokens per expert (drop excess). None = no limit.
        noisy_gating: add noise during training for exploration
    """

    def __init__(self, d_model, n_experts=4, top_k=2, d_ff=None,
                 shared_expert=True, capacity_factor=None, noisy_gating=True,
                 dense_bypass=False, use_clamp=False, clamp_alpha=1.702,
                 clamp_limit=7.0, router_mode="switch", gating="softmax",
                 intra_sparsity: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.has_shared = shared_expert
        # dense_bypass: skip router, run ALL experts with equal weight.
        # Used when router is untrained (uniform init) to reproduce dense FFN.
        # With w2 scaled by n_experts, equal weight 1/n gives exact dense output.
        self.dense_bypass = dense_bypass
        self.use_clamp = use_clamp
        self.clamp_alpha = clamp_alpha
        self.clamp_limit = clamp_limit
        self.intra_sparsity = intra_sparsity

        self.router = Router(d_model, n_experts, top_k, noisy_gating,
                             mode=router_mode, gating=gating)
        # Cached stacked expert weights for the graph-safe gather path
        # (built lazily on first eval-mode routed forward).
        self._w1s = None
        self._w2s = None
        self._w3s = None
        # Lazy: True when all expert weights are bit-identical — lets the
        # dense_bypass path run ONE expert instead of N (mean of identical
        # outputs == that output).  Checked on first eval forward; works on
        # quantized modules by comparing state_dict tensors, not .weight.
        self._experts_identical: bool | None = None
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, use_clamp=use_clamp,
                   clamp_alpha=clamp_alpha, clamp_limit=clamp_limit,
                   intra_sparsity=intra_sparsity)
            for _ in range(n_experts)
        ])
        if shared_expert:
            self.shared = Expert(d_model, d_ff, use_clamp=use_clamp,
                                 clamp_alpha=clamp_alpha, clamp_limit=clamp_limit,
                                 intra_sparsity=intra_sparsity)

    def train(self, mode: bool = True):
        # Weight stacks go stale whenever params may update — invalidate.
        if mode:
            self._w1s = self._w2s = self._w3s = None
            self._experts_identical = None
        return super().train(mode)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, T, d_model)

        Returns:
            output: (B, T, d_model)
            aux_loss: load balancing loss
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        # Dense bypass: skip router, run all experts with equal weight.
        # This reproduces the original dense FFN when the router is untrained.
        if self.dense_bypass:
            # Bit-exact fast path: experts port in as N identical copies of
            # the dense FFN, so mean over them equals a single expert's
            # output.  Divergence (LoRA/training) flips the flag off once
            # and this branch disappears — checked lazily on first forward.
            if self._experts_identical is None:
                # state_dict() detach() fails on torchao packed
                # "inference tensors" — compare parameter tensors
                # directly and treat incomparable (quantized) experts
                # as not-identical: the quantized path below is the
                # correct fallback anyway.
                try:
                    ref = dict(self.experts[0].named_parameters())
                    ref.update(self.experts[0].named_buffers())
                    identical = True
                    for e in self.experts[1:]:
                        sd = dict(e.named_parameters())
                        sd.update(e.named_buffers())
                        if sd.keys() != ref.keys():
                            identical = False
                            break
                        if not all(torch.equal(ref[k], sd[k]) for k in ref):
                            identical = False
                            break
                    self._experts_identical = identical
                except Exception:
                    self._experts_identical = False
            if self._experts_identical:
                output = self.experts[0](x_flat)
                if self.has_shared:
                    output = output + self.shared(x_flat)
                aux_loss = x.new_zeros(())
                self._last_aux_loss = aux_loss
                return output.view(B, T, D), aux_loss
            # Check if experts expose stackable dense weights. Quantized
            # modules (BitNetLinear, QuantizedLinear, NanoQuantLinear, ...)
            # store packed buffers instead of .weight/.base_weight — they
            # must run through their own forward (dequant on the fly).
            first_w1 = self.experts[0].w1
            is_bitnet_int8 = (type(first_w1).__name__ == "BitNetLinear"
                              and getattr(first_w1, '_prequantized', False))

            def _get_weight(mod):
                # Only plain torch Tensors are stackable — quantized
                # packs (torchao Int4TilePackedTo4dTensor etc.) are
                # Tensor subclasses whose ops are dispatched/limited;
                # treat them as unstackable so the per-expert forward
                # path handles them.
                w = getattr(mod, 'base_weight',
                            getattr(mod, 'weight', None))
                if w is not None and type(w) not in (torch.Tensor,
                                                     torch.nn.Parameter):
                    return None
                return w

            if is_bitnet_int8 or _get_weight(first_w1) is None:
                # Quantized path: call each expert's forward individually
                expert_outs = []
                for expert in self.experts:
                    h1 = expert.w1(x_flat)
                    h3 = expert.w3(x_flat)
                    h = F.silu(h1) * h3
                    out = expert.w2(h)
                    expert_outs.append(out)
                expert_out = torch.stack(expert_outs)  # (n_experts, N, D)
                output = expert_out.mean(dim=0)
            else:
                # Standard path: batched matmul with stacked weights
                w1 = torch.stack([_get_weight(e.w1) for e in self.experts])
                w3 = torch.stack([_get_weight(e.w3) for e in self.experts])
                w2 = torch.stack([_get_weight(e.w2) for e in self.experts])
                # x_flat: (N, D) -> (n_experts, N, d_ff) via batched matmul
                x_exp = x_flat.unsqueeze(0).expand(self.n_experts, -1, -1)
                h1 = torch.bmm(x_exp, w1.transpose(1, 2))
                h3 = torch.bmm(x_exp, w3.transpose(1, 2))
                if self.use_clamp:
                    # GPT-OSS clamped SwiGLU in batched path
                    h1 = h1.clamp(min=None, max=self.clamp_limit)
                    h3 = h3.clamp(min=-self.clamp_limit, max=self.clamp_limit)
                    glu = h1 * torch.sigmoid(self.clamp_alpha * h1)
                    expert_out = torch.bmm((h3 + 1) * glu, w2.transpose(1, 2))
                else:
                    expert_out = torch.bmm(F.silu(h1) * h3, w2.transpose(1, 2))
                output = expert_out.mean(dim=0)
            if self.has_shared:
                output = output + self.shared(x_flat)
            # new_zeros: direct device alloc — torch.tensor(0.0, device=cuda)
            # is a pageable H2D copy that breaks CUDA-graph capture (and
            # costs a sync per layer per token).
            aux_loss = x.new_zeros(())
            self._last_aux_loss = aux_loss
            return output.view(B, T, D), aux_loss

        # Normal routed MoE path.
        dispatch_mask, gating_weights, aux_loss = self.router(x_flat)

        # Graph-safe fast path (inference, small N — decode + spec verify):
        # the per-expert nonzero()/scatter loop below is NOT CUDA-graph
        # capturable (data-dependent shapes + host syncs). Here we gather
        # top-k expert WEIGHTS per token — static shapes, dynamic values —
        # so only the selected experts' weights are read (bandwidth win
        # preserved) while every op stays tensor-driven and replayable.
        if (not self.training and getattr(self, '_graph_moe_max_tokens', 64) >= N
                and all(type(getattr(e.w1, 'weight', None))
                        in (torch.Tensor, torch.nn.Parameter)
                        for e in self.experts)):
            # gating_weights is sparse (only top-k entries nonzero) — topk
            # recovers static-shape (N, k) indices + scores.
            scores, idx = gating_weights.topk(self.top_k, dim=-1)  # (N,k)
            if (self._w1s is None or self._w1s.device != x.device
                    or self._w1s.dtype != x.dtype):
                self._w1s = torch.stack([e.w1.weight for e in self.experts])
                self._w2s = torch.stack([e.w2.weight for e in self.experts])
                self._w3s = torch.stack([e.w3.weight for e in self.experts])
            # Gather selected expert weights: (N, k, out, in)
            W1 = self._w1s[idx]  # (N,k,d_ff,D)
            W3 = self._w3s[idx]
            W2 = self._w2s[idx]  # (N,k,D,d_ff)
            h1 = torch.einsum('nd,nkfd->nkf', x_flat, W1)
            h3 = torch.einsum('nd,nkfd->nkf', x_flat, W3)
            if self.use_clamp:
                h1 = h1.clamp(min=None, max=self.clamp_limit)
                h3 = h3.clamp(min=-self.clamp_limit, max=self.clamp_limit)
                glu = h1 * torch.sigmoid(self.clamp_alpha * h1)
                out_e = torch.einsum('nkf,nkdf->nkd', (h3 + 1) * glu, W2)
            else:
                out_e = torch.einsum('nkf,nkdf->nkd', F.silu(h1) * h3, W2)
            output = (out_e * scores.unsqueeze(-1)).sum(1)  # (N, D)
            if self.has_shared:
                output = output + self.shared(x_flat)
            self._last_aux_loss = aux_loss
            return output.view(B, T, D), aux_loss

        # Compute expert outputs.
        # For efficiency, process each expert's assigned tokens.
        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)

        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert.
            token_indices = dispatch_mask[:, i].nonzero(as_tuple=True)[0]
            if len(token_indices) == 0:
                continue

            # Capacity check (drop excess tokens).
            if self.capacity_factor is not None:
                capacity = int(self.capacity_factor * N / self.n_experts)
                if len(token_indices) > capacity:
                    token_indices = token_indices[:capacity]

            # Process tokens.
            expert_input = x_flat[token_indices]
            expert_output = expert(expert_input)
            weights = gating_weights[token_indices, i].unsqueeze(-1)
            output[token_indices] += expert_output * weights

        # Add shared expert output (always active, weight=1).
        if self.has_shared:
            shared_output = self.shared(x_flat)
            output = output + shared_output

        # Store aux loss for collection by collect_aux_loss().
        self._last_aux_loss = aux_loss

        return output.view(B, T, D), aux_loss


def replace_ffn_with_moe(model, n_experts=4, top_k=2, d_model=None,
                         shared_expert=True, capacity_factor=None, d_ff=None,
                         dense_bypass=False, gating="softmax"):
    """Replace FFN layers in a model with MoE layers.

    Args:
        model: the model (must have .blocks with .ffn)
        n_experts: number of experts per MoE layer
        top_k: experts activated per token
        d_model: model dimension (auto-detected if None)
        shared_expert: add shared expert
        capacity_factor: expert capacity limit
        d_ff: hidden dim per expert (default: d_model*2)
        dense_bypass: if True, skip router and run all experts (for untrained router)

    Returns:
        number of layers replaced
    """
    n_replaced = 0
    for block in model.blocks:
        d = d_model or block.ffn[-1].out_features if hasattr(block, 'ffn') else d_model
        if d is None:
            # Try to infer from the block.
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'w2'):
                d = block.ffn.w2.out_features
            elif hasattr(block, 'ffn') and hasattr(block.ffn, 'fc2'):
                d = block.ffn.fc2.out_features

        if d is None:
            print("  [MoE] WARNING: could not infer d_model for block, skipping")
            continue

        moe = MoELayer(d, n_experts=n_experts, top_k=top_k, d_ff=d_ff,
                       shared_expert=shared_expert, capacity_factor=capacity_factor,
                       dense_bypass=dense_bypass, gating=gating)
        block.ffn = moe
        n_replaced += 1

    mode = "dense-bypass" if dense_bypass else f"top-{top_k}"
    print(f"  [MoE] replaced {n_replaced} FFN layers with {n_experts}-expert MoE ({mode})")
    return n_replaced


def collect_aux_loss(model):
    """Collect auxiliary (load balancing) losses from all MoE layers.

    Call this after forward pass to get the total aux loss for backprop.
    """
    total_aux = torch.tensor(0.0, device=next(model.parameters()).device)
    for block in model.blocks:
        if hasattr(block, 'ffn') and isinstance(block.ffn, MoELayer):
            # Store last aux loss on the layer for collection.
            if hasattr(block.ffn, '_last_aux_loss'):
                total_aux = total_aux + block.ffn._last_aux_loss
    return total_aux


def update_moe_biases(model):
    """Update DeepSeek-V3 aux-loss-free expert biases on all MoE layers.

    Call after each optimizer step. No-op for layers using "switch" mode
    (no expert_bias buffer). This is the bias-update half of DeepSeek-V3's
    auxiliary-loss-free load balancing (§3.3).
    """
    for block in model.blocks:
        if hasattr(block, 'ffn') and isinstance(block.ffn, MoELayer):
            block.ffn.router.update_bias()


def disable_dense_bypass(model):
    """Disable dense_bypass on all MoE layers so the router activates.

    After warmup, call this to switch from "all experts run equally" (dense
    FFN behavior, lossless init) to actual top-k routing (experts specialize).
    The router weights were trained during warmup via the aux loss path even
    in dense_bypass mode (the router forward still runs, just its output is
    ignored for the FFN computation). After disabling, routing takes effect.
    """
    n_disabled = 0
    for block in model.blocks:
        if hasattr(block, 'ffn') and isinstance(block.ffn, MoELayer) and block.ffn.dense_bypass:
            block.ffn.dense_bypass = False
            n_disabled += 1
    if n_disabled:
        print(f"  [MoE] Disabled dense_bypass on {n_disabled} layers — router active")
    return n_disabled


def set_intra_expert_sparsity(model, sparsity: float):
    """Enable/disable intra-expert activation sparsity at runtime (R3-24).

    Intra-expert sparsity exploits the natural activation sparsity in
    pretrained MoE models (up to 90% of intermediate neurons are near-zero).
    Thresholds the SwiGLU intermediate activation to skip computation in w2
    for inactive neurons. Training-free, no quality loss at moderate sparsity.

    Call with sparsity=0 to disable (exact dense computation).
    Call with sparsity=0.1-0.5 to enable (typical sweet spot).

    Args:
        model: model with MoE layers
        sparsity: fraction of max activation magnitude below which neurons
            are zeroed. 0 = disabled, 0.5 = aggressive.

    Returns:
        number of experts updated
    """
    n_updated = 0
    for block in model.blocks:
        if hasattr(block, 'ffn') and isinstance(block.ffn, MoELayer):
            for expert in block.ffn.experts:
                expert.intra_sparsity = sparsity
                n_updated += 1
            if hasattr(block.ffn, 'shared'):
                block.ffn.shared.intra_sparsity = sparsity
                n_updated += 1
    return n_updated
