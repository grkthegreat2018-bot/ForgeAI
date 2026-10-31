"""Medusa: Parallel speculative decoding with multiple prediction heads.

Unlike EAGLE (autoregressive draft model), Medusa adds multiple prediction
heads to the main model that predict tokens at different positions in parallel.
No draft model needed — just extra heads on the main model.

Key advantage: all heads run in parallel (one forward pass), vs EAGLE's
sequential draft generation. Simpler + faster for small models.

Structure:
- Head 0: predicts token t+1 (same as main model, used for verification)
- Head 1: predicts token t+2
- Head 2: predicts token t+3
- Head 3: predicts token t+4

At inference, generate k candidates per position, verify with main model,
accept longest matching prefix. Typical speedup: 2-3x.

Paper: "Medusa: Simple LLM Inference Acceleration Framework" (2024)

Usage:
    from forge.decoding.medusa import MedusaHeads, train_medusa, medusa_generate

    # Add Medusa heads to model
    heads = MedusaHeads(model, d_model=1024, vocab_size=151665, n_heads=4)

    # Train heads (short fine-tuning)
    train_medusa(model, heads, dataset, steps=1000)

    # Inference with speculative decoding
    output = medusa_generate(model, heads, tokenizer, prompt, max_new_tokens=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedusaHeads(nn.Module):
    """Medusa multi-head prediction for parallel speculative decoding.

    Args:
        d_model: hidden dimension of the main model
        vocab_size: vocabulary size
        n_heads: number of prediction heads (each predicts 1 token ahead)
        hidden_dim: hidden dimension of each Medusa head (default = d_model)
        share_embedding: if True, share lm_head weight with prediction heads
    """

    def __init__(self, d_model, vocab_size, n_heads=4, hidden_dim=None,
                 share_embedding=None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        hidden_dim = hidden_dim or d_model

        # Each head: MLP → vocab projection.
        # Head k predicts token at position t + k + 1.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, vocab_size, bias=False),
            )
            for _ in range(n_heads)
        ])

        # Optionally share embedding weights with main model's lm_head.
        if share_embedding is not None:
            for head in self.heads:
                head[-1].weight = share_embedding.weight

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """Predict next n_heads tokens from hidden states.

        Args:
            hidden_states: (B, T, d_model) from main model

        Returns:
            list of (B, T, vocab_size) logits, one per head
        """
        return [head(hidden_states) for head in self.heads]

    def predict_candidates(self, hidden_states: torch.Tensor,
                           top_k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict top-k candidate tokens for each position.

        Args:
            hidden_states: (B, T, d_model)
            top_k: number of candidates per head per position

        Returns:
            candidate_tokens: (B, n_heads, top_k) — top-k token ids per head
            candidate_probs: (B, n_heads, top_k) — corresponding probabilities
        """
        all_logits = self.forward(hidden_states[:, -1:, :])  # list of (B, 1, vocab)

        candidate_tokens = []
        candidate_probs = []

        for logits in all_logits:
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (B, vocab)
            topk_probs, topk_tokens = probs.topk(top_k, dim=-1)
            candidate_tokens.append(topk_tokens)
            candidate_probs.append(topk_probs)

        # Stack: (B, n_heads, top_k)
        return torch.stack(candidate_tokens, dim=1), torch.stack(candidate_probs, dim=1)


class MedusaTrainer:
    """Trainer for Medusa heads (main model frozen).

    Args:
        model: the main model (frozen during Medusa training)
        medusa: the MedusaHeads module
        lr: learning rate for Medusa heads
        tree_size: top-k candidates per head (for training with tree attention)
    """

    def __init__(self, model, medusa, lr=1e-4, tree_size=5):
        self.model = model
        self.medusa = medusa
        self.tree_size = tree_size
        self.optimizer = torch.optim.AdamW(medusa.parameters(), lr=lr)

        # Freeze main model.
        for param in self.model.parameters():
            param.requires_grad = False

    def compute_loss(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute Medusa training loss.

        Head k predicts token at position t + k + 1.
        Loss = sum of CE losses for all heads.

        Args:
            input_ids: (B, T) token ids

        Returns:
            (total_loss, stats_dict)
        """
        _B, T = input_ids.shape

        # Get hidden states from frozen main model.
        with torch.no_grad():
            out = self.model(input_ids)
            hidden = out[0] if isinstance(out, tuple) else out

        # Handle case where model returns logits (not hidden states).
        if hidden.size(-1) == self.medusa.vocab_size:
            # Need to get hidden states — project back.
            if not hasattr(self, "_proj"):
                self._proj = nn.Linear(self.medusa.vocab_size,
                                      self.medusa.d_model, bias=False).to(hidden.device)
            hidden = self._proj(hidden)

        # Medusa predictions.
        all_logits = self.medusa(hidden)  # list of (B, T, vocab)

        # Compute loss for each head.
        total_loss = 0.0
        head_losses = []

        for k, logits in enumerate(all_logits):
            # Head k predicts token at position t + k + 1.
            offset = k + 1
            if offset >= T:
                continue
            pred = logits[:, :-offset, :].contiguous()
            target = input_ids[:, offset:].contiguous()
            loss = F.cross_entropy(
                pred.view(-1, pred.size(-1)),
                target.view(-1),
                ignore_index=-100,
            )
            head_losses.append(loss.item())
            total_loss = total_loss + loss

        # Average across heads.
        n_heads = len(head_losses)
        if n_heads > 0:
            total_loss = total_loss / n_heads

        return total_loss, {"head_losses": head_losses, "avg_loss": total_loss.item()}

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Single training step.

        Args:
            input_ids: (B, T) token ids

        Returns:
            stats dict
        """
        self.medusa.train()
        loss, stats = self.compute_loss(input_ids)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.medusa.parameters(), 1.0)
        self.optimizer.step()
        return stats


def medusa_generate(model, medusa, input_ids: torch.Tensor,
                    max_new_tokens: int = 100, temperature: float = 0.0,
                    top_k_candidates: int = 5, device: str = "cuda") -> torch.Tensor:
    """Generate tokens using Medusa speculative decoding.

    Algorithm:
    1. Forward pass through main model → hidden states
    2. Medusa heads predict k candidate tokens in parallel
    3. Build candidate tree (top-k per head)
    4. Verify candidates with main model
    5. Accept longest matching prefix
    6. Repeat

    Args:
        model: the main LLM
        medusa: the MedusaHeads module
        input_ids: (1, T) prompt token ids
        max_new_tokens: max tokens to generate
        temperature: 0 for greedy, >0 for sampling
        top_k_candidates: candidates per Medusa head
        device: cuda or cpu

    Returns:
        (1, T + generated) token ids
    """
    model.eval()
    medusa.eval()
    device = torch.device(device)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        prompt_len = input_ids.shape[1]
        while input_ids.shape[1] - prompt_len < max_new_tokens:
            # 1. Forward pass through main model.
            out = model(input_ids)
            hidden = out[0] if isinstance(out, tuple) else out
            main_logits = hidden if hidden.size(-1) == medusa.vocab_size else None

            # Get main model's prediction (token t+1).
            if main_logits is None and hasattr(model, "lm_head"):
                main_logits = model.lm_head(hidden)
            elif main_logits is None and hasattr(model, "head"):
                main_logits = model.head(hidden)

            if main_logits is None:
                break

            # Main model's next token (greedy or sampled).
            if temperature == 0:
                main_token = main_logits[:, -1, :].argmax(-1, keepdim=True)
            else:
                probs = F.softmax(main_logits[:, -1, :] / temperature, dim=-1)
                main_token = torch.multinomial(probs, num_samples=1)

            # 2. Medusa predicts candidates for t+2, t+3, ...
            # Get hidden states for Medusa (need d_model dim).
            medusa_input = hidden
            if hidden.size(-1) == medusa.vocab_size:
                # Hidden is actually logits — skip Medusa for this step.
                input_ids = torch.cat([input_ids, main_token], dim=1)
                continue

            # Get full Medusa logits for rejection sampling (when temperature > 0).
            # Each head produces (1, 1, V) logits from the last position.
            medusa_logits_list = medusa(medusa_input[:, -1:, :])  # list of (1, 1, V)

            candidate_tokens, _candidate_probs = medusa.predict_candidates(
                medusa_input, top_k=top_k_candidates
            )  # (1, n_heads, top_k)

            # 3. Build candidate sequence: main_token + medusa candidates.
            # Simple greedy: take top-1 from each head.
            medusa_tokens = candidate_tokens[0, :, 0]  # (n_heads,) top-1 per head

            # 4. Verify: forward with [main_token + medusa_tokens] and check.
            candidate_seq = torch.cat([
                main_token,  # (1, 1)
                medusa_tokens.unsqueeze(0)  # (1, n_heads)
            ], dim=1)  # (1, 1 + n_heads)

            # Append to input and verify.
            verify_input = torch.cat([input_ids, candidate_seq], dim=1)
            verify_out = model(verify_input)
            verify_hidden = verify_out[0] if isinstance(verify_out, tuple) else verify_out

            if verify_hidden.size(-1) != medusa.vocab_size:
                if hasattr(model, "lm_head"):
                    verify_logits = model.lm_head(verify_hidden)
                elif hasattr(model, "head"):
                    verify_logits = model.head(verify_hidden)
                else:
                    verify_logits = verify_hidden
            else:
                verify_logits = verify_hidden

            # 5. Accept longest matching prefix.
            # Compare Medusa predictions with actual model predictions.
            n_heads = medusa.n_heads
            start_pos = input_ids.shape[1]
            n_heads = min(n_heads, verify_logits.shape[1] - start_pos)
            if n_heads > 0:
                if temperature == 0:
                    # Greedy: argmax comparison (correct for temperature == 0)
                    target_preds = verify_logits[0, start_pos:start_pos + n_heads, :].argmax(-1)
                    matches = (medusa_tokens[:n_heads] == target_preds)
                    n_accepted = matches.cumprod(dim=-1).sum().item()
                else:
                    # Rejection sampling (Leviathan 2023) for temperature > 0
                    # Target probs: (n_heads, V)
                    target_probs = F.softmax(
                        verify_logits[0, start_pos:start_pos + n_heads, :] / temperature, dim=-1)
                    # Draft probs from Medusa heads: (n_heads, V)
                    draft_probs = torch.stack(
                        [F.softmax(ml[:, -1, :].squeeze(0) / temperature, dim=-1)
                         for ml in medusa_logits_list[:n_heads]], dim=0)  # (n_heads, V)
                    # q(x) and p(x) at the draft token positions
                    draft_tok = medusa_tokens[:n_heads]  # (n_heads,)
                    q_x = draft_probs.gather(1, draft_tok.unsqueeze(-1)).squeeze(-1)  # (n_heads,)
                    p_x = target_probs.gather(1, draft_tok.unsqueeze(-1)).squeeze(-1)  # (n_heads,)
                    # Acceptance ratios: min(1, p/q)
                    ratios = (p_x / q_x.clamp(min=1e-8)).clamp(max=1.0)  # (n_heads,)
                    # Sample randoms on GPU, find first rejection
                    rand = torch.rand_like(ratios)
                    accepted_mask = rand < ratios
                    rejected = ~accepted_mask
                    if rejected.any().item():
                        n_accepted = rejected.int().argmax().item()
                    else:
                        n_accepted = n_heads
            else:
                n_accepted = 0

            # 6. Append accepted tokens.
            accepted = candidate_seq[:, :n_accepted + 1]  # +1 for main_token
            input_ids = torch.cat([input_ids, accepted], dim=1)

            # For rejection sampling: append resampled/bonus token when temperature > 0.
            if temperature > 0 and n_heads > 0:
                if n_accepted < n_heads:
                    # Rejection: resample from residual distribution
                    p_at_rej = target_probs[n_accepted]  # (V,)
                    q_at_rej = draft_probs[n_accepted]  # (V,)
                    residual = (p_at_rej - q_at_rej).clamp(min=0.0)
                    residual = residual / residual.sum().clamp(min=1e-8)
                    resampled = torch.multinomial(
                        residual, num_samples=1).unsqueeze(0)  # (1, 1)
                    input_ids = torch.cat([input_ids, resampled], dim=1)
                else:
                    # All accepted: bonus token from target
                    bonus_pos = start_pos + n_heads
                    if bonus_pos < verify_logits.shape[1]:
                        bonus_probs = F.softmax(
                            verify_logits[0, bonus_pos, :] / temperature, dim=-1)
                        bonus = torch.multinomial(
                            bonus_probs, num_samples=1).unsqueeze(0)  # (1, 1)
                        input_ids = torch.cat([input_ids, bonus], dim=1)

            # Check if we hit max tokens.
            if input_ids.shape[1] - prompt_len >= max_new_tokens:
                break

            # Check for EOS (simplified).
            if hasattr(model, "config") and hasattr(model.config, "eos_token_id"):
                eos_id = model.config.eos_token_id
                if (main_token == eos_id).reshape(-1).any().item():
                    break

    return input_ids


def add_medusa_to_model(model, d_model=None, vocab_size=None, n_heads=4):
    """Add Medusa heads to a model and return them.

    Args:
        model: the main LLM
        d_model: hidden dimension (auto-detected if None)
        vocab_size: vocab size (auto-detected if None)
        n_heads: number of Medusa heads

    Returns:
        MedusaHeads module
    """
    # Auto-detect dimensions.
    if d_model is None:
        if hasattr(model, "config"):
            d_model = model.config.d_model
        else:
            # Infer from first Linear.
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    d_model = m.in_features
                    break
    if vocab_size is None:
        if hasattr(model, "config"):
            vocab_size = model.config.vocab_size
        elif hasattr(model, "head"):
            vocab_size = model.head.out_features
        elif hasattr(model, "lm_head"):
            vocab_size = model.lm_head.out_features

    # Try to share embedding weights.
    share_emb = None
    if hasattr(model, "head"):
        share_emb = model.head
    elif hasattr(model, "lm_head"):
        share_emb = model.lm_head

    medusa = MedusaHeads(d_model, vocab_size, n_heads=n_heads, share_embedding=share_emb)
    n_params = sum(p.numel() for p in medusa.parameters())
    print(f"  [Medusa] {n_heads} heads added ({n_params:,} params, "
          f"{'shared' if share_emb else 'separate'} embeddings)")
    return medusa
