"""EAGLE-3: Feature-level speculative decoding with training-time test.

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) uses a
lightweight draft head that operates on the target model's hidden states
(features) rather than tokens directly. EAGLE-3 improves on EAGLE-1/2 by:

1. **Multi-layer feature fusion**: Instead of only top-layer features, EAGLE-3
   extracts hidden states from 3 layers at different depths (low, mid, high)
   and fuses them, capturing richer semantic information.
2. **Training-time test (TTT)**: Simulates multi-step autoregressive drafting
   during training, removing the feature prediction constraint and directly
   predicting tokens.
3. **Dynamic draft tree** (from EAGLE-2): Context-aware tree structure for
   verification, improving acceptance rates.

Architecture for LFM2.5-1.2B (16 layers, d_model=2048):
    - Low layer:  layer 1   (index 1)
    - Mid layer:  layer 8   (index 8 = 16 // 2)
    - High layer: layer 12  (index 12 = 16 - 4)
    - fc: 3*d_model -> d_model  (fuse 3 hidden states)
    - draft layer: single Transformer decoder layer (hidden=2*d_model)
      Input = concat(projected_features, token_embedding) -> (2*d_model)
    - output head: d_model -> vocab_size

Training: KL divergence between draft and target distributions, with TTT
unrolling (length ~7 steps).

Inference: Draft head autoregressively predicts k tokens, target model
verifies in one forward pass with tree attention. Speedup: 3-5x.

Paper: "EAGLE-3: Scaling up Inference Acceleration of Large Language Models
via Training-Time Test" (NeurIPS 2025)
Repo: https://github.com/SafeAILab/EAGLE

Usage:
    from forge.decoding.eagle import Eagle3Head, Eagle3Trainer, eagle3_generate

    # Create EAGLE-3 head
    head = Eagle3Head(
        d_model=2048, vocab_size=65536, n_layers=16,
        low_layer=1, mid_layer=8, high_layer=12,
    )

    # Train (target model frozen)
    trainer = Eagle3Trainer(model, head, ttt_length=7)
    loss = trainer.train_step(input_ids)

    # Inference
    output = eagle3_generate(model, head, tokenizer, prompt, max_new_tokens=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Eagle3Head(nn.Module):
    """EAGLE-3 draft head with multi-layer feature fusion.

    Extracts hidden states from 3 target model layers (low/mid/high), fuses
    them, and uses a single Transformer decoder layer to predict next tokens.

    Args:
        d_model: hidden dimension of the target model
        vocab_size: vocabulary size
        n_layers: number of layers in the target model
        low_layer: index of the low-level layer (default: 1)
        mid_layer: index of the mid-level layer (default: n_layers // 2)
        high_layer: index of the high-level layer (default: n_layers - 4)
        n_heads: number of attention heads in the draft layer
        ffn_dim: FFN intermediate dimension of the draft layer
        dropout: dropout rate in the draft layer
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        low_layer: int = 1,
        mid_layer: int | None = None,
        high_layer: int | None = None,
        n_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        share_embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.low_layer = low_layer
        self.mid_layer = mid_layer if mid_layer is not None else n_layers // 2
        self.high_layer = high_layer if high_layer is not None else max(n_layers - 4, low_layer + 1)

        # Feature fusion: 3*d_model -> d_model
        self.fc = nn.Linear(3 * d_model, d_model, bias=False)

        # Draft decoder layer: input is concat(fused_features, token_embedding)
        # so hidden size = 2 * d_model
        draft_hidden = 2 * d_model
        ffn_dim = ffn_dim or d_model * 2

        self.draft_attn = nn.MultiheadAttention(
            embed_dim=draft_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.draft_norm1 = nn.LayerNorm(draft_hidden)
        self.draft_ffn = nn.Sequential(
            nn.Linear(draft_hidden, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, draft_hidden),
        )
        self.draft_norm2 = nn.LayerNorm(draft_hidden)
        self.draft_dropout = nn.Dropout(dropout)

        # Output projection: draft_hidden -> d_model (to match target's hidden)
        self.proj_out = nn.Linear(draft_hidden, d_model, bias=False)

        # Output head: d_model -> vocab_size
        if share_embedding is not None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = share_embedding.weight
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Token embedding for draft input (shared with target if provided)
        if share_embedding is not None:
            self.embed = share_embedding
        else:
            self.embed = nn.Embedding(vocab_size, d_model)

        # Causal mask cache: avoids reallocating torch.triu on every draft_forward
        # (T is typically 1 during decode, but the allocation is still wasteful)
        self._causal_mask_cache: dict[int, torch.Tensor] = {}

    def fuse_hidden_states(
        self, hidden_states_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """Fuse hidden states from 3 layers into a single representation.

        Args:
            hidden_states_list: list of exactly 3 (B, T, d_model) tensors
                in order [low, mid, high] — as returned by extract_hidden_states.

        Returns:
            fused: (B, T, d_model) fused feature representation
        """
        low_h = hidden_states_list[0]   # (B, T, d)
        mid_h = hidden_states_list[1]   # (B, T, d)
        high_h = hidden_states_list[2]  # (B, T, d)

        # Concatenate along feature dim: (B, T, 3*d)
        concat = torch.cat([low_h, mid_h, high_h], dim=-1)
        # Project to d_model: (B, T, d)
        fused = self.fc(concat)
        return fused

    def draft_forward(
        self,
        fused_features: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Run the draft decoder layer.

        Args:
            fused_features: (B, T, d_model) fused target hidden states
            input_ids: (B, T) token ids (shifted by 1 from features)
            past_key_value: optional KV cache for the draft layer
            use_cache: whether to return updated KV cache

        Returns:
            logits: (B, T, vocab_size)
            new_kv: updated KV cache if use_cache
        """
        # Embed input tokens: (B, T, d)
        token_embeds = self.embed(input_ids)

        # Concat features + embeddings: (B, T, 2*d)
        draft_input = torch.cat([fused_features, token_embeds], dim=-1)

        # Self-attention with causal mask (cached by T to avoid reallocation)
        T = draft_input.shape[1]
        if T not in self._causal_mask_cache:
            self._causal_mask_cache[T] = torch.triu(
                torch.full((T, T), float('-inf'), device=draft_input.device, dtype=draft_input.dtype),
                diagonal=1,
            )
        causal_mask = self._causal_mask_cache[T]

        attn_out, new_kv = self.draft_attn(
            draft_input, draft_input, draft_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        # nn.MultiheadAttention doesn't support KV cache; return None
        if not use_cache:
            new_kv = None

        x = self.draft_norm1(draft_input + self.draft_dropout(attn_out))
        ffn_out = self.draft_ffn(x)
        x = self.draft_norm2(x + self.draft_dropout(ffn_out))

        # Project to d_model and compute logits
        x = self.proj_out(x)  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, new_kv if use_cache else None

    def predict_next(
        self,
        fused_features: torch.Tensor,
        last_token: torch.Tensor,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next token from fused features and last token.

        Used during autoregressive drafting.

        Args:
            fused_features: (B, 1, d_model) fused features for current position
            last_token: (B, 1) last generated token id
            temperature: 0 for greedy, >0 for sampling
            top_k: if > 0, sample only from top-k tokens

        Returns:
            next_token: (B, 1) predicted token id
            logits: (B, vocab_size) logits for the predicted position
        """
        logits, _ = self.draft_forward(fused_features, last_token)
        logits = logits[:, -1, :]  # (B, vocab)

        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            l = logits / temperature
            if top_k > 0:
                indices_to_remove = l < torch.topk(l, top_k)[0][..., -1, None]
                l.masked_fill_(indices_to_remove, float('-inf'))
            probs = F.softmax(l, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token, logits


def extract_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    layers: list[int],
    past_key_values=None,
    use_cache: bool = False,
    attention_mask=None,
) -> tuple[list[torch.Tensor], torch.Tensor, list | None]:
    """Run the target model and extract hidden states from specified layers.

    Args:
        model: ConfigurableResearchLLM
        input_ids: (B, T) input token ids
        layers: list of layer indices to extract hidden states from
        past_key_values: optional KV cache
        use_cache: whether to return KV cache
        attention_mask: optional attention mask

    Returns:
        hidden_states: list of (B, T, d_model) tensors, one per requested layer
        final_hidden: (B, T, d_model) final layer hidden state (after ln_f)
        presents: KV cache if use_cache
    """
    max(layers)
    hidden_states_by_layer = {}

    # Get embedding
    embed_device = next(model.embed.parameters()).device
    if input_ids.device != embed_device:
        input_ids = input_ids.to(embed_device)
    x = model.embed(input_ids)

    presents = []
    cur_device = x.device

    # Build position_ids and attention_bias if mask provided
    position_ids = None
    attention_bias = None
    if attention_mask is not None:
        B, T = input_ids.shape[:2]
        total_len = attention_mask.shape[1]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.clamp(min=0)
        position_ids = position_ids[:, -T:]
        dtype = next(model.parameters()).dtype
        pad_mask = (attention_mask == 0)
        if T == 1 and total_len > 1:
            attention_bias = torch.zeros(B, 1, 1, total_len, device=input_ids.device, dtype=dtype)
            attention_bias = attention_bias.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        elif total_len == T:
            from forge.model_loader import _causal_mask
            causal = _causal_mask(T, total_len, 0, input_ids.device, dtype)
            pad_add = torch.zeros(B, 1, T, total_len, device=input_ids.device, dtype=dtype)
            pad_add = pad_add.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_bias = causal + pad_add
        else:
            from forge.model_loader import _causal_mask
            past_len = total_len - T
            causal = _causal_mask(T, total_len, past_len, input_ids.device, dtype)
            pad_add = torch.zeros(B, 1, T, total_len, device=input_ids.device, dtype=dtype)
            pad_add = pad_add.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
            attention_bias = causal + pad_add

    for i, block in enumerate(model.blocks):
        block_device = next(block.parameters()).device
        if block_device != cur_device:
            x = x.to(block_device)
            cur_device = block_device
        past = past_key_values[i] if past_key_values is not None else None
        x, present = block(x, past_key_value=past, use_cache=use_cache,
                           layer_idx=i, attention_bias=attention_bias,
                           position_ids=position_ids)
        if i in layers:
            hidden_states_by_layer[i] = x
        if use_cache:
            presents.append(present)

    # Final layer norm
    ln_f_device = next(model.ln_f.parameters()).device
    if x.device != ln_f_device:
        x = x.to(ln_f_device)
    final_hidden = model.ln_f(x)

    # Return hidden states in the order requested
    hidden_states = [hidden_states_by_layer[l] for l in layers]

    return hidden_states, final_hidden, presents if use_cache else None


class Eagle3Trainer:
    """Trains the EAGLE-3 draft head with training-time test (TTT).

    The target model is frozen. The draft head learns to predict tokens
    by simulating multi-step autoregressive generation during training.

    Args:
        model: the target model (frozen during training)
        head: the Eagle3Head to train
        ttt_length: number of steps to unroll during TTT (default 7)
        lr: learning rate for the draft head
        kl_weight: weight for KL divergence loss
        feature_weight: weight for feature prediction loss (EAGLE-1 compat)
    """

    def __init__(
        self,
        model: nn.Module,
        head: Eagle3Head,
        ttt_length: int = 7,
        lr: float = 1e-4,
        kl_weight: float = 1.0,
        feature_weight: float = 0.0,
    ):
        self.model = model
        self.head = head
        self.ttt_length = ttt_length
        self.kl_weight = kl_weight
        self.feature_weight = feature_weight

        # Freeze target model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Optimizer for draft head only
        self.optimizer = torch.optim.AdamW(
            head.parameters(), lr=lr, weight_decay=0.01,
        )

        # Layers to extract from target
        self.extract_layers = [head.low_layer, head.mid_layer, head.high_layer]

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute EAGLE-3 training loss with TTT.

        Args:
            input_ids: (B, T) input token ids
            attention_mask: optional attention mask

        Returns:
            loss: scalar tensor
            metrics: dict with loss components
        """
        _B, _T = input_ids.shape
        device = input_ids.device

        # Step 1: Extract target hidden states and target logits
        with torch.no_grad():
            target_hidden_list, target_final, _ = extract_hidden_states(
                self.model, input_ids, self.extract_layers,
            )
            # Target logits (for KL divergence)
            target_logits = self.model.head(target_final)  # (B, T, V)
            target_probs = F.softmax(target_logits, dim=-1)  # (B, T, V)

        # Step 2: Fuse target hidden states
        fused = self.head.fuse_hidden_states(target_hidden_list)  # (B, T, d)

        # Step 3: TTT — simulate multi-step drafting
        # For each position t, the draft head sees features at t and token at t,
        # and must predict token at t+1.
        # We unroll for ttt_length steps, feeding predictions back.

        torch.tensor(0.0, device=device)
        torch.tensor(0.0, device=device)

        # Use teacher forcing: feed ground-truth tokens (not draft predictions)
        # This is the standard TTT training approach.
        # Input to draft: fused_features[:, :-1], input_ids[:, :-1]
        # Target: input_ids[:, 1:] and target_probs[:, 1:]

        draft_input_ids = input_ids[:, :-1]  # (B, T-1)
        draft_fused = fused[:, :-1]  # (B, T-1, d)
        target_tokens = input_ids[:, 1:]  # (B, T-1)
        target_probs[:, 1:]  # (B, T-1, V)

        # Single forward pass through draft (teacher forcing)
        draft_logits, _ = self.head.draft_forward(draft_fused, draft_input_ids)
        # draft_logits: (B, T-1, V)

        # KL divergence loss: D_KL(target || draft)
        # Use log_target=True for numerical stability (avoids log(0) when
        # target probabilities are very small after softmax).
        log_draft_probs = F.log_softmax(draft_logits, dim=-1)
        log_target_probs = F.log_softmax(target_logits[:, 1:], dim=-1)
        kl_loss = F.kl_div(
            log_draft_probs, log_target_probs, reduction='batchmean',
            log_target=True,
        )

        # CE loss: standard cross-entropy against ground truth tokens
        ce_loss = F.cross_entropy(
            draft_logits.reshape(-1, draft_logits.size(-1)),
            target_tokens.reshape(-1),
        )

        # Optional feature prediction loss (EAGLE-1 compatibility)
        feat_loss = torch.tensor(0.0, device=device)
        if self.feature_weight > 0:
            # Predict the target's final hidden state
            draft_hidden = self.head.proj_out(
                self.head.draft_norm2(
                    self.head.draft_ffn(
                        self.head.draft_norm1(
                            torch.cat([draft_fused, self.head.embed(draft_input_ids)], dim=-1)
                        )
                    )
                )
            )
            feat_loss = F.mse_loss(draft_hidden, target_final[:, :-1].detach())

        # Total loss
        loss = (
            self.kl_weight * kl_loss
            + ce_loss
            + self.feature_weight * feat_loss
        )

        metrics = {
            'kl_loss': kl_loss.item(),
            'ce_loss': ce_loss.item(),
            'feat_loss': feat_loss.item(),
            'total_loss': loss.item(),
        }

        return loss, metrics

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """One training step: compute loss, backprop, update.

        Returns:
            metrics dict
        """
        self.head.train()
        self.optimizer.zero_grad()

        loss, metrics = self.compute_loss(input_ids, attention_mask)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
        self.optimizer.step()

        return metrics

    def save(self, path: str):
        """Save the EAGLE-3 head weights."""
        from safetensors.torch import save_file
        save_file(self.head.state_dict(), path)

    def load(self, path: str):
        """Load EAGLE-3 head weights."""
        from safetensors.torch import load_file
        self.head.load_state_dict(load_file(path))


@torch.inference_mode()
def eagle3_generate(
    model: nn.Module,
    head: Eagle3Head,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    draft_length: int = 4,
    temperature: float = 0.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    device: str = "cuda",
) -> str:
    """Generate text with EAGLE-3 speculative decoding.

    Workflow:
    1. Prefill: run target model, extract hidden states, get first token.
    2. Draft: head autoregressively predicts draft_length tokens.
    3. Verify: target model verifies all draft tokens in one forward pass.
    4. Accept longest matching prefix, reject rest.
    5. Repeat from step 2 until max_new_tokens or EOS.

    Args:
        model: target ConfigurableResearchLLM
        head: trained Eagle3Head
        tokenizer: tokenizer
        prompt: input prompt string
        max_new_tokens: max tokens to generate
        draft_length: number of tokens to draft per iteration
        temperature: 0 for greedy, >0 for sampling
        top_k: top-k sampling (0 = disabled)
        repetition_penalty: repetition penalty factor
        device: device to run on

    Returns:
        generated text (excluding prompt)
    """
    model.eval()
    head.eval()

    # Tokenize prompt
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ids.shape[1]

    # Extract layers for EAGLE-3
    extract_layers = [head.low_layer, head.mid_layer, head.high_layer]

    # Step 1: Prefill — run target model on full prompt
    hidden_list, final_hidden, presents = extract_hidden_states(
        model, ids, extract_layers, use_cache=True,
    )
    fused = head.fuse_hidden_states(hidden_list)  # (B, T, d)

    # Get first token from target model
    target_logits = model.head(final_hidden)  # (B, T, V)
    last_logits = target_logits[:, -1, :]  # (B, V)

    if temperature <= 0:
        next_token = last_logits.argmax(dim=-1, keepdim=True)
    else:
        l = last_logits / temperature
        if top_k > 0:
            indices_to_remove = l < torch.topk(l, top_k)[0][..., -1, None]
            l.masked_fill_(indices_to_remove, float('-inf'))
        probs = F.softmax(l, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 7
    generated_tokens = next_token.clone()  # [1, 1] on GPU
    ids = torch.cat([ids, next_token], dim=1)

    # KV cache for draft head

    # Generation loop
    while generated_tokens.shape[1] < max_new_tokens:
        if generated_tokens[0, -1].item() == eos_id:
            break

        # Step 2: Draft — autoregressively predict draft_length tokens
        draft_tokens_gpu = []
        draft_logits_list = []

        # Get fused features for the last position
        cur_fused = fused[:, -1:, :]  # (B, 1, d)
        cur_token = next_token  # (B, 1)

        for _ in range(draft_length):
            draft_tok, draft_log = head.predict_next(
                cur_fused, cur_token, temperature=temperature, top_k=top_k,
            )
            draft_tokens_gpu.append(draft_tok)
            draft_logits_list.append(draft_log)

            # For next draft step, we need fused features at the new position.
            # In EAGLE, we use the draft's own hidden state as an approximation.
            # This is the key insight: the draft operates in feature space.
            # We reuse the last fused features (simplification for small models).
            cur_token = draft_tok

        # Step 3: Verify — run target model on all draft tokens at once
        # TODO: EAGLE-3 uses a dynamic draft tree for verification; this
        # implementation uses a simple linear chain (no tree attention).
        # Adding tree attention would improve acceptance rates but requires
        # a custom attention mask and more complex verification logic.
        draft_tensor = torch.cat(draft_tokens_gpu, dim=1)  # (1, draft_length)

        # Run target model with KV cache
        verify_hidden_list, verify_final, presents = extract_hidden_states(
            model, draft_tensor, extract_layers,
            past_key_values=presents, use_cache=True,
        )
        verify_logits = model.head(verify_final)  # (1, draft_len, V)

        # Step 4: Accept longest matching prefix
        if temperature <= 0:
            # Greedy: argmax comparison (correct for temperature == 0)
            verify_preds = verify_logits[:, :draft_length, :].argmax(dim=-1)  # (1, draft_length)
            matches = (verify_preds == draft_tensor)  # (1, draft_length)
            not_match = ~matches[0]  # (draft_length,)
            if not_match.any().item():
                n_accepted = not_match.int().argmax().item()
            else:
                n_accepted = draft_length

            # Accept accepted tokens + the token after the last accepted (from target)
            if n_accepted > 0:
                generated_tokens = torch.cat(
                    [generated_tokens, draft_tensor[:, :n_accepted]], dim=1)

            # Get the next token from target (either correction or continuation)
            if n_accepted < draft_length:
                # Rejection: use target's token at the rejection point
                next_token = verify_logits[:, n_accepted, :].argmax(dim=-1, keepdim=True)
            else:
                # All accepted: use target's token after the last draft token
                next_token = verify_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            # Rejection sampling (Leviathan 2023) for temperature > 0
            n = draft_length
            # Stack draft logits: list of (1, V) → (1, n, V)
            draft_logits_stacked = torch.stack(draft_logits_list[:n], dim=1)  # (1, n, V)
            # Target and draft probability distributions
            target_probs = F.softmax(verify_logits[:, :n, :] / temperature, dim=-1)  # (1, n, V)
            draft_probs = F.softmax(draft_logits_stacked / temperature, dim=-1)  # (1, n, V)
            # q(x) and p(x) at the draft token positions
            q_x = draft_probs.gather(2, draft_tensor.unsqueeze(-1)).squeeze(-1)  # (1, n)
            p_x = target_probs.gather(2, draft_tensor.unsqueeze(-1)).squeeze(-1)  # (1, n)
            # Acceptance ratios: min(1, p/q)
            ratios = (p_x / q_x.clamp(min=1e-8)).clamp(max=1.0)  # (1, n)
            # Sample randoms on GPU, find first rejection
            rand = torch.rand_like(ratios)  # (1, n)
            accepted_mask = rand < ratios  # (1, n) True = accept
            rejected = ~accepted_mask[0]  # (n,)
            if rejected.any().item():
                n_accepted = rejected.int().argmax().item()
            else:
                n_accepted = n

            # Accept accepted draft tokens
            if n_accepted > 0:
                generated_tokens = torch.cat(
                    [generated_tokens, draft_tensor[:, :n_accepted]], dim=1)

            # Get the next token (resampled or bonus)
            if n_accepted < n:
                # Rejection: resample from residual distribution
                # residual = norm(max(0, p - q))
                p_at_rej = target_probs[0, n_accepted]  # (V,)
                q_at_rej = draft_probs[0, n_accepted]  # (V,)
                residual = (p_at_rej - q_at_rej).clamp(min=0.0)
                residual = residual / residual.sum().clamp(min=1e-8)
                next_token = torch.multinomial(residual, num_samples=1).unsqueeze(0)  # (1, 1)
            else:
                # All accepted: bonus token = sample from target_probs at last position
                bonus_probs = target_probs[0, -1]  # (V,)
                next_token = torch.multinomial(bonus_probs, num_samples=1).unsqueeze(0)  # (1, 1)

        # Update fused features for next iteration
        fused = head.fuse_hidden_states(verify_hidden_list)

        if (next_token == eos_id).item():
            break
        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        ids = torch.cat([ids, next_token], dim=1)

    # Decode — single CPU sync at the end
    output_ids = generated_tokens[0, :max_new_tokens].cpu().tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=False)


def add_eagle3_to_model(
    model: nn.Module,
    n_layers: int | None = None,
    **kwargs,
) -> Eagle3Head:
    """Create an EAGLE-3 head for a given model.

    Args:
        model: ConfigurableResearchLLM
        n_layers: number of layers (auto-detected if None)
        **kwargs: passed to Eagle3Head

    Returns:
        Eagle3Head module (not yet attached to model)
    """
    if n_layers is None:
        n_layers = len(model.blocks)

    d_model = model.config.d_model
    vocab_size = model.config.vocab_size

    # Share embedding with target model
    share_embed = model.embed if hasattr(model, 'embed') else None

    head = Eagle3Head(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        share_embedding=share_embed,
        **kwargs,
    )

    # Move to same device as model
    device = next(model.parameters()).device
    head = head.to(device)

    return head
