"""DSpark — Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation.

Paper: https://arxiv.org/html/2607.05147 (Cheng et al., DeepSeek-AI, 2026)

DSpark unifies high-throughput parallel generation with adaptive, load-aware
verification via two complementary mechanisms:

1. **Semi-autoregressive generation** (Section 3.1): a parallel backbone (like
   Medusa/MTP) predicts N tokens in one forward pass, then a lightweight sequential
   module injects intra-block dependency modeling to mitigate suffix decay.
   - Parallel stage: base logits U_1..U_gamma from n_predict independent heads.
   - Sequential stage: transition bias B_k conditioned on previously sampled tokens
     within the block, via an RNN head with GRU-like gated update (Eq. 6).
   - Final distribution: p_k(v) = softmax(U_k(v) + B_k(x_0, x_{<k}, v))  (Eq. 4)

2. **Confidence-scheduled verification** (Section 3.2): a confidence head estimates
   per-position prefix survival probabilities c_k (Eq. 7), and a hardware-aware
   scheduler dynamically chooses the verification length per request based on
   throughput profiles (Algorithm 1).

Architecture:
    - Parallel backbone: n_predict independent prediction heads (Medusa-style MLPs)
    - Sequential module: RNN head with GRU-like gated update + low-rank transition
      (W1 embedding + W2 projection, r=256 by default)
    - Confidence head: linear projection + sigmoid per position

Usage:
    from forge.decoding.dspark import DSparkHead, dspark_generate, DSparkTrainer

    head = DSparkHead(d_model=1024, vocab_size=151665, n_predict=4, n_layers=2)
    output = dspark_generate(model, head, input_ids, max_new_tokens=100)
"""
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.decoding.acceptance_losses import eal_loss, wtv_loss


class DSparkHead(nn.Module):
    """DSpark semi-autoregressive speculative decoding head.

    Combines a parallel backbone (n_predict Medusa-style heads) with a lightweight
    sequential RNN module that injects intra-block dependencies, plus a confidence
    head for scheduled verification.

    Args:
        d_model: hidden dimension of the main model
        vocab_size: vocabulary size
        n_predict: number of tokens to predict in parallel (block size gamma)
        n_layers: depth of each parallel prediction head's MLP
        seq_rank: rank r for the low-rank transition factorization (default 256)
        seq_mode: 'rnn' for RNN head (full prefix history) or 'markov' for
            first-order Markov head (memoryless beyond one step)
        share_embedding: optional nn.Embedding to tie W1 weights with
    """

    def __init__(self, d_model: int, vocab_size: int, n_predict: int = 4,
                 n_layers: int = 2, seq_rank: int = 256, seq_mode: str = "rnn",
                 share_embedding: nn.Embedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_predict = n_predict
        self.n_layers = n_layers
        self.seq_rank = seq_rank
        self.seq_mode = seq_mode

        # ── Parallel backbone: n_predict independent prediction heads ──
        # Head k predicts token at position t + k + 1 (Medusa-style).
        # Each head is an n_layers-deep MLP: Linear → GELU → ... → Linear(vocab).
        self.parallel_heads = nn.ModuleList()
        for _ in range(n_predict):
            layers: list[nn.Module] = []
            in_dim = d_model
            for li in range(n_layers):
                out_dim = d_model if li < n_layers - 1 else vocab_size
                layers.append(nn.Linear(in_dim, out_dim, bias=(li == n_layers - 1)))
                if li < n_layers - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.LayerNorm(out_dim))
                in_dim = out_dim
            self.parallel_heads.append(nn.Sequential(*layers))

        # ── Sequential module: low-rank transition bias ──
        # W1: token embedding lookup (V × r), W2: logit projection (r → V).
        # Shared with confidence head's Markov embedding.
        if share_embedding is not None and share_embedding.embedding_dim == seq_rank:
            self.W1 = share_embedding
        else:
            self.W1 = nn.Embedding(vocab_size, seq_rank)
        self.W2 = nn.Linear(seq_rank, vocab_size, bias=False)

        if seq_mode == "rnn":
            # RNN head: GRU-like gated update over full prefix history.
            # z_k = [s_{k-1}; W1[x_{k-1}]; h_k] ∈ R^{2r + d}
            # s_k = σ(W_g z_k) ⊙ s_{k-1} + (1 - σ(W_g z_k)) ⊙ tanh(W_c z_k)
            # B_k = W2^T tanh(W_o z_k)
            # Single projection split into gate, candidate, output (3 × r).
            self.seq_proj = nn.Linear(2 * seq_rank + d_model, 3 * seq_rank, bias=True)
        elif seq_mode == "markov":
            # Markov head: B(x_{k-1}, ·) = W1[x_{k-1}] W2 — no recurrent state.
            self.seq_proj = None
        else:
            raise ValueError(f"Unknown seq_mode: {seq_mode!r} (expected 'rnn' or 'markov')")

        # ── Confidence head: c_k = σ(w^T [h_k; W1[x_{k-1}]])  (Eq. 7) ──
        self.conf_proj = nn.Linear(d_model + seq_rank, 1, bias=True)

        # Post-hoc calibration temperatures (Sequential Temperature Scaling).
        # One per position; initialized to 1.0 (no scaling).
        self.register_buffer("calib_temps", torch.ones(n_predict))

    # ──────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _parallel_backbone(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """Run parallel backbone: produce base logits U_1..U_gamma.

        Args:
            hidden_states: (B, T, d_model)

        Returns:
            list of (B, T, vocab_size) base logits, one per prediction position
        """
        return [head(hidden_states) for head in self.parallel_heads]

    def _seq_step(self, k: int, hidden_k: torch.Tensor,
                  prev_token: torch.Tensor,
                  state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of the sequential module at position k.

        Computes transition bias B_k and updates recurrent state.

        Args:
            k: position index (0-based within block)
            hidden_k: (B, d_model) backbone hidden state at position k
            prev_token: (B,) previously sampled token id (x_{k-1})
            state: (B, r) recurrent state s_{k-1}, or None for k=0

        Returns:
            bias: (B, vocab_size) transition bias B_k
            new_state: (B, r) updated recurrent state s_k
        """
        B = hidden_k.size(0)
        r = self.seq_rank
        prev_emb = self.W1(prev_token)  # (B, r)

        if self.seq_mode == "markov":
            # B(x_{k-1}, ·) = W1[x_{k-1}] W2  (Eq. 5)
            bias = self.W2(prev_emb)  # (B, vocab)
            return bias, prev_emb  # state not used, return emb as placeholder

        # RNN head (Eq. 6)
        if state is None:
            state = torch.zeros(B, r, device=hidden_k.device, dtype=hidden_k.dtype)

        z = torch.cat([state, prev_emb, hidden_k], dim=-1)  # (B, 2r + d)
        proj = self.seq_proj(z)  # (B, 3r)
        gate, candidate, output = proj.chunk(3, dim=-1)  # each (B, r)

        gate_s = torch.sigmoid(gate)
        new_state = gate_s * state + (1.0 - gate_s) * torch.tanh(candidate)
        bias = self.W2(torch.tanh(output))  # (B, vocab)

        return bias, new_state

    def _confidence(self, hidden_k: torch.Tensor,
                    prev_token: torch.Tensor) -> torch.Tensor:
        """Compute confidence score c_k = σ(w^T [h_k; W1[x_{k-1}]])  (Eq. 7).

        Args:
            hidden_k: (B, d_model)
            prev_token: (B,) previous token id

        Returns:
            conf: (B,) confidence score in (0, 1)
        """
        prev_emb = self.W1(prev_token)  # (B, r)
        feat = torch.cat([hidden_k, prev_emb], dim=-1)  # (B, d + r)
        return torch.sigmoid(self.conf_proj(feat).squeeze(-1))  # (B,)

    # ──────────────────────────────────────────────────────────────────
    #  Forward (training)
    # ──────────────────────────────────────────────────────────────────

    def forward(self, hidden_states: torch.Tensor,
                input_ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass: parallel backbone + sequential refinement.

        During training, uses ground-truth input_ids as the "previous predictions"
        to compute transition biases. During inference, use generate_block() instead.

        Args:
            hidden_states: (B, T, d_model) from the main model
            input_ids: (B, T) token ids (used for sequential conditioning)

        Returns:
            logits_list: list of n_predict tensors, each (B, T, vocab_size).
                logits_list[k] = U_k + B_k (base logits + transition bias)
            confidences: (B, T, n_predict) per-position confidence scores
        """
        B, T, _ = hidden_states.shape

        # Parallel stage: base logits U_1..U_gamma.
        base_logits = self._parallel_backbone(hidden_states)  # list of (B, T, V)

        # Sequential stage: add transition bias B_k conditioned on prev tokens.
        logits_list = []
        conf_list = []

        for k in range(self.n_predict):
            # Hidden state at position k (for sequential + confidence).
            h_k = hidden_states  # (B, T, d)

            # Previous token for sequential conditioning.
            # During training, each position t uses the ground-truth token at t
            # as x_{k-1} (teacher forcing). Head k predicts token at t+k+1,
            # so the "previous" token within the block is at offset k from t.
            if k == 0:
                # Position 0 conditions on the anchor (current position t).
                prev_tok = input_ids  # (B, T)
            else:
                # Position k conditions on token at t+k (the k-th ground-truth
                # token ahead, which is what head k-1 would have predicted).
                prev_tok = input_ids  # teacher forcing: use ground truth at t+k
                # We shift by k so that position t sees token at t+k.
                if k < T:
                    prev_tok = torch.cat(
                        [input_ids[:, k:], input_ids[:, :k]], dim=1
                    )

            # Compute transition bias and confidence for all positions at once.
            # Reshape for batch processing: (B*T, ...) per position k.
            h_flat = h_k.reshape(B * T, self.d_model)
            prev_flat = prev_tok.reshape(B * T)

            if self.seq_mode == "rnn":
                # For training, we process all timesteps with the RNN.
                # We need to run the RNN across the block dimension (k), not time.
                # Since each k is independent across time positions, we can
                # use a simplified approach: treat each (b, t) independently.
                # For efficiency, we approximate by using zero state for training.
                state = torch.zeros(B * T, self.seq_rank,
                                    device=hidden_states.device, dtype=hidden_states.dtype)
                bias, _ = self._seq_step(k, h_flat, prev_flat, state)
            else:
                bias, _ = self._seq_step(k, h_flat, prev_flat, None)

            bias = bias.reshape(B, T, self.vocab_size)
            refined = base_logits[k] + bias  # U_k + B_k  (Eq. 4)
            logits_list.append(refined)

            # Confidence: c_k = σ(w^T [h_k; W1[x_{k-1}]])
            conf = self._confidence(h_flat, prev_flat)  # (B*T,)
            conf_list.append(conf.reshape(B, T))

        confidences = torch.stack(conf_list, dim=-1)  # (B, T, n_predict)
        return logits_list, confidences

    # ──────────────────────────────────────────────────────────────────
    #  Block generation (inference)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_block(self, model: nn.Module, input_ids: torch.Tensor,
                       max_block_size: int = 4,
                       temperature: float = 0.0,
                       return_probs: bool = False) -> list[tuple[int, float]] | tuple:
        """Generate a block of up to max_block_size tokens using semi-AR generation.

        Algorithm:
        1. Forward pass through main model → hidden states (anchor).
        2. Parallel backbone produces base logits U_1..U_gamma.
        3. Sequential module samples left-to-right: at each position k,
           p_k = softmax(U_k + B_k(x_0, x_{<k})), sample x_k.
        4. Confidence head estimates c_k at each position.

        Args:
            model: the main LLM
            input_ids: (1, T) current token sequence
            max_block_size: maximum tokens to draft (capped by n_predict)
            temperature: 0 for greedy, >0 for sampling
            return_probs: if True, also return the refined probability
                distributions (list of (1, V) tensors) for rejection sampling

        Returns:
            list of (token_id, confidence) pairs, length ≤ max_block_size.
            If return_probs=True, returns (results, refined_probs_list).
        """
        model.eval()
        self.eval()
        block_size = min(max_block_size, self.n_predict)

        # 1. Forward pass through main model to get hidden states.
        try:
            out = model(input_ids, return_hidden=True)
            # (logits, loss, hidden) when return_hidden=True
            hidden = out[2] if len(out) > 2 else out[0]
        except TypeError:
            # Model doesn't support return_hidden
            out = model(input_ids)
            hidden = out[0] if isinstance(out, tuple) else out
        # Handle model returning logits instead of hidden states.
        if hidden.size(-1) == self.vocab_size:
            if hasattr(model, "embed"):
                hidden = model.embed(hidden.argmax(-1))
            elif hasattr(model, "wte"):
                hidden = model.wte(hidden.argmax(-1))
            elif hasattr(model, "embed_tokens"):
                hidden = model.embed_tokens(hidden.argmax(-1))
            else:
                return ([], []) if return_probs else []

        # Use the last position's hidden state as the anchor context.
        anchor_hidden = hidden[:, -1:, :]  # (1, 1, d_model)

        # 2. Parallel backbone: base logits for all positions.
        base_logits = self._parallel_backbone(anchor_hidden)  # list of (1, 1, V)
        base_logits = [bl[:, -1, :] for bl in base_logits]  # list of (1, V)

        # 3. Sequential sampling: left-to-right within the block.
        anchor_token = input_ids[:, -1]  # (1,) last token = anchor x_0
        results: list[tuple[int, float]] = []
        refined_probs_list: list[torch.Tensor] = []
        state = None  # RNN state
        prev_token = anchor_token  # x_{k-1}, starts with anchor

        for k in range(block_size):
            h_k = anchor_hidden[:, 0, :]  # (1, d_model) — same anchor for all k

            # Sequential step: transition bias + state update.
            bias, state = self._seq_step(k, h_k, prev_token, state)  # (1, V), (1, r)

            # Refined logits: U_k + B_k  (Eq. 4)
            refined = base_logits[k] + bias  # (1, V)

            # Apply calibration temperature (post-hoc STS).
            calib_temp = self.calib_temps[k]
            refined = refined / torch.clamp(calib_temp, min=1e-6)

            # Sample token.
            if temperature == 0:
                token = refined.argmax(dim=-1)  # (1,)
                if return_probs:
                    refined_probs_list.append(F.softmax(refined, dim=-1))
            else:
                probs = F.softmax(refined / temperature, dim=-1)
                if return_probs:
                    refined_probs_list.append(probs)
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (1,)

            # Confidence score c_k.
            conf = self._confidence(h_k, prev_token)  # (1,)

            # Batch sync: single .tolist() for token + confidence.
            tok_conf = torch.stack([token, conf]).tolist()
            results.append((tok_conf[0], tok_conf[1]))
            prev_token = token  # x_k becomes x_{k-1} for next step

        if return_probs:
            return results, refined_probs_list
        return results

    # ──────────────────────────────────────────────────────────────────
    #  Confidence-scheduled verification (Section 3.2)
    # ──────────────────────────────────────────────────────────────────

    def confidence_schedule(self, confidences: list[float],
                            throughput_profile: Callable[[int], float] | None = None,
                            threshold: float = 0.5) -> int:
        """Decide how many tokens to verify based on confidence scores.

        Implements Algorithm 1 (Hardware-Aware Prefix Scheduler) when a throughput
        profile is provided, or a simple cumulative-threshold heuristic otherwise.

        Args:
            confidences: list of per-position confidence scores c_1..c_gamma
            throughput_profile: optional function SPS(B) → steps/sec for batch
                size B. If provided, uses the greedy throughput maximization.
            threshold: minimum cumulative survival probability for the simple
                heuristic (used when no throughput_profile is given).

        Returns:
            n_verify: number of tokens to verify (0 to len(confidences))
        """
        gamma = len(confidences)
        if gamma == 0:
            return 0

        # Compute prefix survival probabilities: a_j = ∏_{i≤j} c_i.
        survival = []
        cum = 1.0
        for c in confidences:
            cum *= c
            survival.append(cum)

        if throughput_profile is not None:
            # ── Algorithm 1: Hardware-Aware Prefix Scheduler ──
            # Greedy admission maximizing Θ = τ · SPS(B).
            # Single-request version: B = 1 + ℓ, τ = 1 + Σ a_j.
            # Start with ℓ=0 (verify only the anchor/bonus token).
            best_theta = 1.0 * throughput_profile(1)  # τ=R=1, B=1
            best_ell = 0

            tau = 1.0  # expected accepts (starts with 1 for the bonus token)
            B = 1  # batch size

            for j in range(gamma):
                B += 1
                tau += survival[j]
                theta = tau * throughput_profile(B)
                if theta > best_theta:
                    best_theta = theta
                    best_ell = j + 1
                else:
                    # Early stopping (non-anticipating property).
                    break

            return best_ell

        # ── Simple heuristic: cumulative survival probability threshold ──
        # Verify the longest prefix where cumulative survival ≥ threshold.
        n_verify = 0
        for j in range(gamma):
            if survival[j] >= threshold:
                n_verify = j + 1
            else:
                break

        return n_verify


# ──────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────

class DSparkTrainer:
    """Trains a DSpark head with the three-term objective from Section 3.3.

    Loss = L_pred + L_tv + L_conf, position-weighted by w_k = exp(-(k-1)/gamma).

    Args:
        model: the main LLM (frozen during DSpark training)
        dspark_head: DSparkHead instance
        lr: learning rate
        tv_weight: weight for distribution-matching loss L_tv
        conf_weight: weight for confidence loss L_conf
        loss_type: draft-prediction term — "ce" (cross-entropy, default,
            Eq. 9), "eal" (Expected Accepted Length for greedy
            verification, arXiv 2609.24150), or "wtv" (Window
            Total-Variation for sampling verification).
        temperature: softmax temperature for "wtv" (should match the
            decode-time sampling temperature).
    """

    def __init__(self, model, dspark_head, lr=1e-4,
                 tv_weight=0.2, conf_weight=0.5,
                 loss_type: str = "ce", temperature: float = 1.0):
        if loss_type not in ("ce", "eal", "wtv"):
            raise ValueError(f"unknown loss_type {loss_type!r} "
                             "(expected 'ce', 'eal', or 'wtv')")
        self.model = model
        self.dspark_head = dspark_head
        self.tv_weight = tv_weight
        self.conf_weight = conf_weight
        self.loss_type = loss_type
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(dspark_head.parameters(), lr=lr)

        # Freeze main model.
        for param in self.model.parameters():
            param.requires_grad = False

    def _position_weights(self, gamma: int) -> torch.Tensor:
        """Exponential decay weights: w_k = exp(-(k-1)/gamma)."""
        k = torch.arange(1, gamma + 1, dtype=torch.float)
        return torch.exp(-(k - 1) / gamma)

    def compute_loss(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute the three-term DSpark training loss.

        L_pred: draft token prediction — CE (Eq. 9) when loss_type="ce",
            eal_loss / wtv_loss (arXiv 2609.24150) when "eal"/"wtv"
        L_tv: total variation distance between draft and target distributions (Eq. 10)
        L_conf: BCE between predicted confidence and analytical acceptance rate (Eq. 8)

        Args:
            input_ids: (B, T) token ids

        Returns:
            (total_loss, stats_dict)
        """
        _B, T = input_ids.shape
        gamma = self.dspark_head.n_predict
        device = input_ids.device

        # Forward through frozen main model — get hidden states (not logits).
        with torch.no_grad():
            try:
                out = self.model(input_ids, return_hidden=True)
                # (logits, loss, hidden) or (logits, loss, presents, hidden)
                hidden = out[-1] if len(out) > 2 else out[0]
            except (TypeError, AttributeError):
                out = self.model(input_ids)
                hidden = out[0] if isinstance(out, tuple) else out

        # Handle model returning logits instead of hidden states.
        if hidden.size(-1) == self.dspark_head.vocab_size:
            if hasattr(self.model, "embed"):
                hidden = self.model.embed(hidden.argmax(-1))
            elif hasattr(self.model, "wte"):
                hidden = self.model.wte(hidden.argmax(-1))
            elif hasattr(self.model, "embed_tokens"):
                hidden = self.model.embed_tokens(hidden.argmax(-1))

        # DSpark forward: refined logits + confidences.
        logits_list, confidences = self.dspark_head(hidden, input_ids)
        # logits_list: list of (B, T, V), confidences: (B, T, gamma)

        # Get target distributions from the main model (for L_tv and L_conf).
        with torch.no_grad():
            target_logits_list = []
            for k in range(gamma):
                offset = k + 1
                if offset >= T:
                    break
                # Target distribution at position t+k+1.
                input_ids[:, offset:]
                tgt_hidden = hidden[:, :-offset, :]
                if hasattr(self.model, "head"):
                    tgt_logits = self.model.head(tgt_hidden)
                elif hasattr(self.model, "lm_head"):
                    tgt_logits = self.model.lm_head(tgt_hidden)
                else:
                    tgt_logits = tgt_hidden
                target_logits_list.append(tgt_logits)

        weights = self._position_weights(gamma).to(device)  # (gamma,)

        # ── L_pred: draft-prediction loss ──
        # "ce" (default): cross-entropy on ground-truth tokens (Eq. 9).
        # "eal"/"wtv": acceptance-aware losses (arXiv 2609.24150) on the
        # (B, T', V) draft/target logits — sequence positions act as the
        # verification window. See acceptance_losses.py for formulas.
        pred_loss = 0.0
        for k in range(min(gamma, len(target_logits_list))):
            offset = k + 1
            if offset >= T:
                break
            pred = logits_list[k][:, :-offset, :].contiguous()
            if self.loss_type == "eal":
                loss_k = eal_loss(pred, target_logits_list[k])
            elif self.loss_type == "wtv":
                loss_k = wtv_loss(pred, target_logits_list[k],
                                  temperature=self.temperature)
            else:
                target = input_ids[:, offset:].contiguous()
                loss_k = F.cross_entropy(
                    pred.view(-1, pred.size(-1)),
                    target.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).mean()
            pred_loss = pred_loss + loss_k * weights[k]
        ce_loss = pred_loss

        # ── L_tv + L_conf: combined loop (avoids redundant softmax computation) ──
        # Both losses need draft_probs and target_probs — compute once, reuse.
        tv_loss = 0.0
        conf_loss = 0.0
        for k in range(min(gamma, len(target_logits_list))):
            offset = k + 1
            if offset >= T:
                break
            draft_logits = logits_list[k][:, :-offset, :].contiguous()
            target_logits = target_logits_list[k].contiguous()

            draft_probs = F.softmax(draft_logits, dim=-1)
            target_probs = F.softmax(target_logits, dim=-1)
            prob_diff = (draft_probs - target_probs).abs()  # (B, T', V)

            # L_tv: TV distance = 0.5 * ||p_d - p_t||_1
            tv_dist = 0.5 * prob_diff.sum(dim=-1).mean()
            tv_loss = tv_loss + tv_dist * weights[k]

            # L_conf: c_k* = 1 - 0.5 * ||p_k^d - p_k^t||_1  (analytical acceptance rate)
            c_star = 1.0 - 0.5 * prob_diff.sum(dim=-1)  # (B, T')
            c_pred = confidences[:, :-offset, k].contiguous()  # (B, T')

            conf_loss = conf_loss + F.binary_cross_entropy(
                c_pred.clamp(1e-6, 1 - 1e-6),
                c_star.clamp(1e-6, 1 - 1e-6),
                reduction="mean",
            ) * weights[k]

        total_loss = ce_loss + self.tv_weight * tv_loss + self.conf_weight * conf_loss

        return total_loss, {
            "ce_loss": float(ce_loss) if isinstance(ce_loss, float) else ce_loss.item(),
            "tv_loss": float(tv_loss) if isinstance(tv_loss, float) else tv_loss.item(),
            "conf_loss": float(conf_loss) if isinstance(conf_loss, float) else conf_loss.item(),
            "total_loss": total_loss.item(),
            "loss_type": self.loss_type,
        }

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Single training step.

        Args:
            input_ids: (B, T) token ids

        Returns:
            stats dict
        """
        self.dspark_head.train()
        loss, stats = self.compute_loss(input_ids)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dspark_head.parameters(), 1.0)
        self.optimizer.step()
        return stats

    def calibrate(self, validation_data: list[torch.Tensor],
                  n_grid: int = 50) -> dict:
        """Post-hoc Sequential Temperature Scaling (STS) calibration.

        Calibrates confidence scores left-to-right, finding optimal temperature
        per position via 1D grid search to minimize ECE of cumulative products.

        Args:
            validation_data: list of (B, T) validation token id batches
            n_grid: grid search resolution

        Returns:
            dict with calibration results
        """
        self.dspark_head.eval()
        gamma = self.dspark_head.n_predict
        device = next(self.dspark_head.parameters()).device

        # Collect predicted confidences and empirical acceptance rates.
        all_preds = [[] for _ in range(gamma)]  # predicted c_k per position
        all_labels = [[] for _ in range(gamma)]  # empirical accept (0/1) per position

        with torch.no_grad():
            for input_ids in validation_data:
                input_ids = input_ids.to(device)
                try:
                    out = self.model(input_ids, return_hidden=True)
                    hidden = out[-1] if len(out) > 2 else out[0]
                except (TypeError, AttributeError):
                    out = self.model(input_ids)
                    hidden = out[0] if isinstance(out, tuple) else out
                if hidden.size(-1) == self.dspark_head.vocab_size:
                    if hasattr(self.model, "embed"):
                        hidden = self.model.embed(hidden.argmax(-1))
                    elif hasattr(self.model, "wte"):
                        hidden = self.model.wte(hidden.argmax(-1))
                    elif hasattr(self.model, "embed_tokens"):
                        hidden = self.model.embed_tokens(hidden.argmax(-1))

                logits_list, confidences = self.dspark_head(hidden, input_ids)

                for k in range(gamma):
                    offset = k + 1
                    if offset >= input_ids.shape[1]:
                        continue
                    # Empirical acceptance: does draft top-1 match target top-1?
                    draft_tok = logits_list[k][:, :-offset, :].argmax(-1)
                    target_tok = input_ids[:, offset:]
                    accept = (draft_tok == target_tok).float().view(-1)

                    c_pred = confidences[:, :-offset, k].contiguous().view(-1)
                    all_preds[k].append(c_pred)
                    all_labels[k].append(accept)

        # Sequential temperature scaling: left-to-right.
        # Temperature scaling is applied in logit space: σ(logit/T) not σ(logit)/T.
        # Confidence c = σ(z) → logit z = log(c / (1-c)) → scaled c' = σ(z / T).
        temps = []
        for k in range(gamma):
            if not all_preds[k]:
                temps.append(1.0)
                continue

            preds = torch.cat(all_preds[k]).clamp(1e-6, 1 - 1e-6)
            labels = torch.cat(all_labels[k])

            # Convert confidence to logit space for already-calibrated positions.
            # σ(z/T_j) = c' → z/T_j = log(c'/(1-c')) → z = T_j * log(c'/(1-c'))
            # For position k, apply cumulative temperature: z_eff = z / (∏_{j<k} T_j)
            logit_preds = torch.log(preds / (1 - preds))
            for j in range(k):
                logit_preds = logit_preds / max(temps[j], 1e-6)

            # Grid search for optimal temperature at position k.
            best_temp, best_ece = 1.0, float("inf")
            for t_val in torch.linspace(0.1, 5.0, n_grid):
                scaled = torch.sigmoid(logit_preds / t_val)
                # Binned ECE: group predictions into bins, compare avg confidence
                # to empirical accuracy per bin.
                n_bins = 10
                bin_bounds = torch.linspace(0, 1, n_bins + 1)
                ece = 0.0
                for bi in range(n_bins):
                    mask = (scaled >= bin_bounds[bi]) & (scaled < bin_bounds[bi + 1])
                    if bi == n_bins - 1:
                        mask = (scaled >= bin_bounds[bi]) & (scaled <= bin_bounds[bi + 1])
                    if mask.any():
                        bin_conf = scaled[mask].mean()
                        bin_acc = labels[mask].mean()
                        bin_frac = mask.float().mean()
                        ece += bin_frac * (bin_conf - bin_acc).abs().item()
                if ece < best_ece:
                    best_ece = ece
                    best_temp = t_val.item()

            temps.append(best_temp)

        self.dspark_head.calib_temps.copy_(torch.tensor(temps, device=device))
        return {"calib_temps": temps, "method": "STS"}


# ──────────────────────────────────────────────────────────────────────
#  Generation (inference)
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def dspark_generate(model: nn.Module, dspark_head: DSparkHead,
                    input_ids: torch.Tensor, max_new_tokens: int = 100,
                    temperature: float = 0.0, max_block_size: int = 4,
                    throughput_profile: Callable[[int], float] | None = None,
                    conf_threshold: float = 0.5,
                    device: str = "cuda") -> torch.Tensor:
    """Generate tokens using DSpark speculative decoding.

    Algorithm (per cycle):
    1. Main model forward pass → anchor token + hidden states.
    2. DSparkHead.generate_block() → draft block of (token, confidence) pairs.
    3. confidence_schedule() → decide how many tokens to verify.
    4. Main model verifies the scheduled prefix in a single forward pass.
    5. Accept longest matching prefix; append bonus token from target.
    6. On rejection (0 accepted), fall back to regular greedy generation.

    Args:
        model: the main LLM
        dspark_head: DSparkHead instance
        input_ids: (1, T) prompt token ids
        max_new_tokens: maximum tokens to generate
        temperature: 0 for greedy, >0 for sampling
        max_block_size: maximum draft block size per cycle
        throughput_profile: optional SPS(B) function for hardware-aware scheduling
        conf_threshold: confidence threshold for simple scheduling (no profile)
        device: cuda or cpu

    Returns:
        (1, T + generated) token ids
    """
    model.eval()
    dspark_head.eval()
    dev = torch.device(device)
    input_ids = input_ids.to(dev)
    prompt_len = input_ids.shape[1]

    # EOS detection: Qwen uses <|im_end|> (151645) and <|endoftext|> (151643)
    eos_ids = set()
    if hasattr(model, "config") and hasattr(model.config, "eos_token_id"):
        eid = model.config.eos_token_id
        if eid is not None:
            eos_ids.add(eid)
    # Qwen2.5 special tokens
    eos_ids.update({151643, 151645})
    eos_ids_tensor = torch.tensor(list(eos_ids), device=next(model.parameters()).device)  # <|endoftext|>, <|im_end|>

    def _get_lm_head_logits(hidden: torch.Tensor) -> torch.Tensor:
        """Get logits from hidden states using the model's LM head."""
        if hidden.size(-1) == dspark_head.vocab_size:
            return hidden  # already logits
        if hasattr(model, "lm_head"):
            return model.lm_head(hidden)
        if hasattr(model, "head"):
            return model.head(hidden)
        return hidden

    while input_ids.shape[1] - prompt_len < max_new_tokens:
        # ── Step 1: Main model forward → anchor token ──
        out = model(input_ids, return_hidden=True)
        # (logits, loss, hidden) when return_hidden=True, no use_cache
        hidden = out[2] if len(out) > 2 else out[0]
        main_logits = _get_lm_head_logits(hidden)

        # Anchor token: the main model's prediction for the next position.
        if temperature == 0:
            anchor_token = main_logits[:, -1, :].argmax(-1, keepdim=True)  # (1, 1)
        else:
            probs = F.softmax(main_logits[:, -1, :] / temperature, dim=-1)
            anchor_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Check for EOS on anchor token — GPU-side check avoids .item() sync.
        if eos_ids_tensor is not None and (anchor_token == eos_ids_tensor).any().item():
            input_ids = torch.cat([input_ids, anchor_token], dim=1)
            break

        # ── Step 2: DSpark generates a draft block ──
        # Temp append anchor to get hidden states for drafting.
        draft_input = torch.cat([input_ids, anchor_token], dim=1)
        block_result = dspark_head.generate_block(
            model, draft_input, max_block_size=max_block_size,
            temperature=temperature, return_probs=(temperature > 0),
        )
        if temperature > 0:
            block, draft_probs_list = block_result
        else:
            block = block_result

        if not block:
            # No draft possible — just accept the anchor token.
            input_ids = draft_input
            continue

        draft_tokens = [tok for tok, _ in block]
        draft_confs = [conf for _, conf in block]

        # Check if any draft token is EOS — stop early
        if any(t in eos_ids for t in draft_tokens):
            # Find first EOS in draft, accept up to and including it
            for _i, t in enumerate(draft_tokens):
                if t in eos_ids:
                    eos_draft = torch.tensor([[t]], dtype=input_ids.dtype, device=dev)
                    input_ids = torch.cat([draft_input, eos_draft], dim=1)
                    return input_ids

        # ── Step 3: Confidence-scheduled verification ──
        n_verify = dspark_head.confidence_schedule(
            draft_confs, throughput_profile=throughput_profile,
            threshold=conf_threshold,
        )

        if n_verify == 0:
            # No tokens worth verifying — accept just the anchor.
            input_ids = draft_input
            continue

        # ── Step 4: Verify scheduled prefix with main model ──
        verify_tokens = draft_tokens[:n_verify]
        verify_seq = torch.tensor(
            [verify_tokens], dtype=input_ids.dtype, device=dev,
        )  # (1, n_verify)
        verify_input = torch.cat([draft_input, verify_seq], dim=1)
        # verify_input = [prompt ... anchor_token draft_1 ... draft_n_verify]

        verify_out = model(verify_input, return_hidden=True)
        verify_hidden = verify_out[2] if len(verify_out) > 2 else verify_out[0]
        verify_logits = _get_lm_head_logits(verify_hidden)

        # ── Step 5: Accept longest matching prefix ──
        # The anchor token is at position `input_ids.shape[1]` in verify_input.
        # Draft token k is verified against the model's prediction at position
        # `input_ids.shape[1] + k` (i.e., the model predicts what comes after
        # the anchor + accepted drafts).
        anchor_pos = input_ids.shape[1]  # position of anchor in verify_input
        n_accepted = 0
        bonus_token = None

        # GPU-side: compare all draft tokens with model predictions at once.
        n_check = min(n_verify, verify_logits.shape[1] - 1 - anchor_pos)
        if n_check > 0:
            if temperature == 0:
                # Greedy: argmax comparison (correct for temperature == 0)
                target_preds = verify_logits[0, anchor_pos:anchor_pos + n_check, :].argmax(-1)
                verify_tensor = torch.tensor(verify_tokens[:n_check], device=dev)
                matches = (verify_tensor == target_preds)
                n_accepted = matches.cumprod(dim=-1).sum().item()

                if n_accepted < n_check:
                    # Rejection at position n_accepted — bonus is model's prediction.
                    bonus_token = target_preds[n_accepted:n_accepted + 1].unsqueeze(-1)
                else:
                    # All verified tokens accepted — bonus = model's prediction after last.
                    pos = anchor_pos + n_verify
                    if pos < verify_logits.shape[1]:
                        bonus_token = verify_logits[:, pos, :].argmax(-1, keepdim=True)
                    else:
                        bonus_token = anchor_token  # fallback
            else:
                # Rejection sampling (Leviathan 2023) for temperature > 0
                verify_tensor = torch.tensor(verify_tokens[:n_check], device=dev)
                # Target probs: (n_check, V)
                target_probs = F.softmax(
                    verify_logits[0, anchor_pos:anchor_pos + n_check, :] / temperature, dim=-1)
                # Draft probs: stack refined_probs → (n_check, V)
                draft_probs = torch.stack(
                    [p.squeeze(0) for p in draft_probs_list[:n_check]], dim=0)  # (n_check, V)
                # q(x) and p(x) at the draft token positions
                q_x = draft_probs.gather(1, verify_tensor.unsqueeze(-1)).squeeze(-1)  # (n_check,)
                p_x = target_probs.gather(1, verify_tensor.unsqueeze(-1)).squeeze(-1)  # (n_check,)
                # Acceptance ratios: min(1, p/q)
                ratios = (p_x / q_x.clamp(min=1e-8)).clamp(max=1.0)  # (n_check,)
                # Sample randoms on GPU, find first rejection
                rand = torch.rand_like(ratios)
                accepted_mask = rand < ratios
                rejected = ~accepted_mask
                if rejected.any().item():
                    n_accepted = rejected.int().argmax().item()
                else:
                    n_accepted = n_check

                if n_accepted < n_check:
                    # Rejection: resample from residual distribution
                    p_at_rej = target_probs[n_accepted]  # (V,)
                    q_at_rej = draft_probs[n_accepted]  # (V,)
                    residual = (p_at_rej - q_at_rej).clamp(min=0.0)
                    residual = residual / residual.sum().clamp(min=1e-8)
                    bonus_token = torch.multinomial(
                        residual, num_samples=1).unsqueeze(0)  # (1, 1)
                else:
                    # All accepted: bonus = sample from target at last verified position
                    pos = anchor_pos + n_verify
                    if pos < verify_logits.shape[1]:
                        bonus_probs = F.softmax(
                            verify_logits[0, pos, :] / temperature, dim=-1)
                        bonus_token = torch.multinomial(
                            bonus_probs, num_samples=1).unsqueeze(0)
                    else:
                        bonus_token = anchor_token  # fallback
        else:
            n_accepted = 0
            bonus_token = anchor_token  # fallback

        # ── Step 6: Assemble accepted tokens ──
        # Always accept the anchor token + accepted draft prefix + bonus.
        accepted_drafts = torch.tensor(
            [verify_tokens[:n_accepted]], dtype=input_ids.dtype, device=dev,
        ) if n_accepted > 0 else None

        new_tokens = [anchor_token]  # anchor always accepted
        if accepted_drafts is not None:
            new_tokens.append(accepted_drafts)
        if bonus_token is not None:
            new_tokens.append(bonus_token)

        new_tokens = torch.cat(new_tokens, dim=1)  # (1, 1 + n_accepted + 1)
        input_ids = torch.cat([input_ids, new_tokens], dim=1)

        # Check EOS on bonus token and accepted drafts — GPU-side.
        all_new = new_tokens[0]  # (n,) or (1, n)
        if all_new.dim() == 0:
            all_new = all_new.unsqueeze(0)
        n_new = all_new.shape[-1]
        # GPU-side: check which tokens are EOS, find first EOS index.
        eos_mask = torch.isin(all_new, eos_ids_tensor)  # (n_new,) bool GPU
        if eos_mask.any().item():  # single sync
            first_eos = eos_mask.nonzero()[0].item()  # second sync (rare path)
            input_ids = input_ids[:, :input_ids.shape[1] - (n_new - first_eos - 1)]
            break
        # No EOS — continue loop.

    # Trim to max_new_tokens if exceeded.
    excess = input_ids.shape[1] - prompt_len - max_new_tokens
    if excess > 0:
        input_ids = input_ids[:, :-excess]

    return input_ids


def add_dspark_to_model(model: nn.Module, d_model: int | None = None,
                        vocab_size: int | None = None, n_predict: int = 4,
                        n_layers: int = 2) -> DSparkHead:
    """Create a DSparkHead and attach it to a model.

    Auto-detects d_model and vocab_size from the model if not specified.

    Args:
        model: the main LLM
        d_model: hidden dimension (auto-detected if None)
        vocab_size: vocab size (auto-detected if None)
        n_predict: number of tokens to predict in parallel
        n_layers: depth of each parallel prediction head

    Returns:
        DSparkHead module
    """
    # Auto-detect dimensions.
    if d_model is None:
        if hasattr(model, "config"):
            d_model = getattr(model.config, "d_model", None)
            if d_model is None:
                d_model = getattr(model.config, "hidden_size", None)
        if d_model is None:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    d_model = m.in_features
                    break
    if vocab_size is None:
        if hasattr(model, "config"):
            vocab_size = getattr(model.config, "vocab_size", None)
        if vocab_size is None and hasattr(model, "lm_head"):
            vocab_size = model.lm_head.out_features
        if vocab_size is None and hasattr(model, "head"):
            vocab_size = model.head.out_features

    # Try to share embedding weights.
    share_emb = None
    if hasattr(model, "wte"):
        share_emb = model.wte
    elif hasattr(model, "embed_tokens"):
        share_emb = model.embed_tokens

    head = DSparkHead(
        d_model=d_model, vocab_size=vocab_size,
        n_predict=n_predict, n_layers=n_layers,
        share_embedding=share_emb,
    )
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  [DSpark] {n_predict} predict heads, {n_layers} layers "
          f"({n_params:,} params, "
          f"{'shared' if share_emb is not None else 'separate'} embeddings)")
    return head
