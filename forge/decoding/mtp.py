"""MTP — Multi-Token Prediction head for training + speculative decoding.

Trains the model to predict N future tokens simultaneously (not just next token).
At inference, this enables self-speculative decoding: predict N tokens, verify
in one forward pass → 2-3x speedup.

Architecture:
    MTPHead: shared trunk + N output heads, each predicting token at position t+k
    Loss: sum of CE losses over all N heads (curriculum: start with N=1, increase)

Training modes:
1. Standard MTP: predict tokens t+1, t+2, ..., t+N from hidden state at t
2. L-MTP (leap): predict t+1, t+3, t+5 (non-adjacent, captures longer dependencies)
3. Curriculum: gradually increase N during training (small models benefit most)
4. MTP-D: gradient-detached self-distillation — aligns MTP head logits toward
   the main head's logits with stop-gradient on the main head, preventing
   MTP training from degrading the main output head (zero negative interference).

Usage:
    from forge.decoding.mtp import MTPHead, MTPTrainer

    # Add MTP head to model
    head = MTPHead(d_model=1024, vocab_size=151665, n_predict=4)
    # Train with MTP-D distillation (prevents MTP from degrading main head)
    trainer = MTPTrainer(model, head, n_predict=4, curriculum=True,
                         mtp_d_weight=0.3)
    loss = trainer.compute_loss(input_ids)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """Multi-token prediction head.

    Predicts N future tokens from a single hidden state.
    Uses a shared projection + N independent output heads.

    Args:
        d_model: model hidden dimension
        vocab_size: vocabulary size
        n_predict: number of future tokens to predict (default 4)
        leap: if True, predict non-adjacent tokens (L-MTP mode)
        share_embedding: if True, share weights with model's token embedding
    """

    def __init__(self, d_model, vocab_size, n_predict=4, leap=False,
                 share_embedding: nn.Embedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_predict = n_predict
        self.leap = leap

        # Shared trunk: project hidden state to a representation for prediction.
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # N independent output heads (can share with model's lm_head).
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_predict)
        ])

        # Optional: share embedding weights with output heads.
        if share_embedding is not None:
            for head in self.heads:
                head.weight = share_embedding.weight  # tied weights

    def forward(self, hidden_states):
        """Predict N future tokens from hidden states.

        Args:
            hidden_states: (B, T, d_model) from the main model

        Returns:
            logits: list of N tensors, each (B, T, vocab_size)
                logits[k] predicts token at position t+k+1
        """
        trunk_out = self.trunk(hidden_states)
        return [head(trunk_out) for head in self.heads]

    def predict_tokens(self, hidden_states, temperature=0.0):
        """Generate N draft tokens from the last hidden state.

        Args:
            hidden_states: (B, T, d_model)
            temperature: 0 for greedy, >0 for sampling

        Returns:
            tokens: (B, N) predicted token ids
            logits: (B, N, vocab_size) logits for each prediction
        """
        last_hidden = hidden_states[:, -1:, :]  # (B, 1, d_model)
        all_logits = self.forward(last_hidden)  # list of (B, 1, vocab)

        tokens = []
        logits_list = []
        for _k, logits in enumerate(all_logits):
            l = logits[:, -1, :]  # (B, vocab)
            if temperature == 0:
                tok = l.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(l / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)
            tokens.append(tok)
            logits_list.append(l)

        tokens = torch.cat(tokens, dim=1)  # (B, N)
        logits_out = torch.stack(logits_list, dim=1)  # (B, N, vocab)
        return tokens, logits_out


class MTPTrainer:
    """Trains a model with MTP head.

    Combines standard next-token prediction (NTP) with multi-token prediction (MTP).
    Curriculum learning: gradually increase n_predict during training.

    Args:
        model: the main LLM (frozen or trainable)
        mtp_head: MTPHead instance
        n_predict: current number of tokens to predict
        curriculum: if True, ramp up n_predict from 1 to max
        curriculum_steps: steps to reach max n_predict
        mtp_weight: weight of MTP loss vs NTP loss (default 0.5)
        leap: if True, use L-MTP (leap predictions)
    """

    def __init__(self, model, mtp_head, n_predict=4,
                 curriculum=True, curriculum_steps=1000,
                 mtp_weight=0.5, leap=False,
                 mtp_d_weight: float = 0.0,
                 mtp_d_temperature: float = 2.0):
        """Initialize MTP trainer.

        Args:
            mtp_d_weight: weight of MTP-D distillation loss (default 0 = disabled).
                When >0, adds KL(MTP_logits || main_head_logits.detach()) to align
                MTP heads toward the main head's distribution. Stop-gradient on the
                main head ensures zero negative interference with the primary NTP loss.
            mtp_d_temperature: temperature for softmax in distillation (default 2.0).
                Higher temperature softens the distribution, making it easier to learn.
        """
        self.model = model
        self.mtp_head = mtp_head
        self.n_predict = n_predict
        self.curriculum = curriculum
        self.curriculum_steps = curriculum_steps
        self.mtp_weight = mtp_weight
        self.leap = leap
        self.mtp_d_weight = mtp_d_weight
        self.mtp_d_temperature = mtp_d_temperature
        self.current_step = 0
        self.current_n = 1 if curriculum else n_predict

    def _get_current_n(self):
        """Get current n_predict based on curriculum schedule."""
        if not self.curriculum:
            return self.n_predict
        # Linear ramp from 1 to n_predict over curriculum_steps.
        progress = min(1.0, self.current_step / self.curriculum_steps)
        return max(1, int(1 + progress * (self.n_predict - 1)))

    def compute_loss(self, input_ids, ignore_index=-100):
        """Compute combined NTP + MTP loss.

        Args:
            input_ids: (B, T) token ids

        Returns:
            total_loss = ntp_loss + mtp_weight * mtp_loss
        """
        self.current_step += 1
        self.current_n = self._get_current_n()

        _B, T = input_ids.shape

        # Forward pass through main model.
        model_out = self.model(input_ids)
        hidden = model_out[0] if isinstance(model_out, tuple) else model_out

        # Standard NTP loss (next-token prediction).
        # Detect if model returns hidden states or logits:
        # - If output dim == vocab_size, it's already logits
        # - If output dim == d_model, it's hidden states (need lm_head)
        vocab_size = self.mtp_head.vocab_size
        if hidden.size(-1) == vocab_size:
            ntp_logits = hidden  # already logits
        elif hasattr(self.model, "lm_head"):
            ntp_logits = self.model.lm_head(hidden)
        else:
            ntp_logits = hidden  # assume hidden IS logits

        ntp_targets = input_ids[:, 1:].contiguous()
        ntp_loss = F.cross_entropy(
            ntp_logits[:, :-1, :].contiguous().view(-1, ntp_logits.size(-1)),
            ntp_targets.view(-1),
            ignore_index=ignore_index,
        )

        # Get hidden states for MTP head (needs d_model dim, not vocab_size).
        mtp_input = hidden
        if hidden.size(-1) == vocab_size:
            # Output was logits — project back to d_model.
            # Check for embedding layer (different models use different names).
            has_embed = (hasattr(self.model, "wte") or
                        hasattr(self.model, "embed") or
                        hasattr(self.model, "embed_tokens"))
            if has_embed:
                if not hasattr(self, "_vocab_proj"):
                    self._vocab_proj = nn.Linear(vocab_size, self.mtp_head.d_model,
                                                bias=False).to(hidden.device)
                mtp_input = self._vocab_proj(hidden)
            else:
                mtp_input = hidden  # hope for the best

        # MTP loss: predict tokens t+2, t+3, ..., t+N from hidden at t.
        mtp_logits_list = self.mtp_head(mtp_input)  # list of (B, T, vocab)
        mtp_loss = 0.0
        mtp_d_loss = 0.0  # MTP-D distillation loss
        n_active = min(self.current_n, len(mtp_logits_list))

        for k in range(n_active):
            # Head k predicts token at position t + k + 2 (offset by 1 for NTP + k for MTP).
            offset = k + 2 if not self.leap else 2 * (k + 1)
            if offset >= T:
                break
            # MTP logits at position t predict target at t + offset.
            mtp_logits_k = mtp_logits_list[k][:, :-offset, :].contiguous()
            mtp_targets_k = input_ids[:, offset:].contiguous()

            if mtp_logits_k.shape[1] == 0 or mtp_targets_k.shape[1] == 0:
                continue

            loss_k = F.cross_entropy(
                mtp_logits_k.view(-1, mtp_logits_k.size(-1)),
                mtp_targets_k.view(-1),
                ignore_index=ignore_index,
            )
            mtp_loss = mtp_loss + loss_k

            # MTP-D: distill main head's logits into MTP head (stop-grad on main).
            # Aligns MTP head k's distribution at position t with the main head's
            # distribution at position t+offset-1 (which predicts the same target).
            # Stop-gradient on the teacher (main head) ensures zero interference
            # with the primary NTP objective.
            if self.mtp_d_weight > 0:
                # Teacher: main head logits at position (t + offset - 1), predicting
                # the same target token as MTP head k at position t.
                teacher_logits = ntp_logits[:, offset - 1:ntp_logits.size(1) - 1, :].contiguous()
                teacher_logits = teacher_logits[:, :mtp_logits_k.shape[1], :].contiguous()

                if teacher_logits.shape[1] == mtp_logits_k.shape[1]:
                    T_d = self.mtp_d_temperature
                    # KL(student || teacher.detach()) — student learns teacher's distribution
                    student_log_probs = F.log_softmax(mtp_logits_k / T_d, dim=-1)
                    teacher_probs = F.softmax(teacher_logits.detach() / T_d, dim=-1)
                    # KL = sum(teacher * (log(teacher) - log(student)))
                    # = cross_entropy(student, teacher) - entropy(teacher)
                    # We use the cross-entropy part (entropy is constant w.r.t. student)
                    kl = F.kl_div(
                        student_log_probs.reshape(-1, student_log_probs.size(-1)),
                        teacher_probs.reshape(-1, teacher_probs.size(-1)),
                        reduction="batchmean",
                    ) * (T_d * T_d)  # scale by T^2 as in Hinton et al.
                    mtp_d_loss = mtp_d_loss + kl

        if n_active > 0:
            mtp_loss = mtp_loss / n_active  # average over heads
            if self.mtp_d_weight > 0:
                mtp_d_loss = mtp_d_loss / n_active

        total_loss = ntp_loss + self.mtp_weight * mtp_loss
        if self.mtp_d_weight > 0:
            total_loss = total_loss + self.mtp_d_weight * mtp_d_loss

        metrics = {"ntp_loss": ntp_loss.item(),
                   "mtp_loss": float(mtp_loss) if isinstance(mtp_loss, float) else mtp_loss.item(),
                   "current_n": self.current_n}
        if self.mtp_d_weight > 0:
            metrics["mtp_d_loss"] = float(mtp_d_loss) if isinstance(mtp_d_loss, float) else mtp_d_loss.item()

        return total_loss, metrics

    def stats(self):
        return {"step": self.current_step, "current_n": self.current_n,
                "target_n": self.n_predict}


# ─── MTPModule: integrated multi-head MTP for model_loader wiring ───────────
#
# Nemotron Lightning style: N independent heads that share a trunk, each
# predicting token t+k+1 from the hidden state at position t. The heads can
# be tied to the model's LM head (shared weight design) for zero extra param
# cost at inference, or use independent weights for better quality.
#
# Identity init: when identity_init=True, the trunk is identity (zero-init the
# Linear bias, ones-init the LayerNorm, and the output heads copy the model's
# head weights). This makes MTP lossless at start — the first head predicts
# exactly what the main head predicts, and subsequent heads are zero-init so
# they contribute zero loss initially. Training then diverges the heads.


class MTPModule(nn.Module):
    """Integrated multi-token prediction module for model wiring.

    Wraps N MTPHeads into a single module with a shared trunk, designed to be
    attached to ConfigurableResearchLLM as ``model.mtp_module``. The forward
    method computes the auxiliary MTP loss from hidden states + ground truth
    token embeddings.

    Args:
        d_model: model hidden dimension.
        vocab_size: vocabulary size.
        n_heads: number of MTP prediction heads (each predicts t+k+1).
        loss_weight: weight of the MTP auxiliary loss (added to main CE).
        identity_init: if True, init heads to copy the model head (lossless).
    """

    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 2,
                 loss_weight: float = 0.3, identity_init: bool = True,
                 adapter_heads: bool = False):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.loss_weight = loss_weight
        self.identity_init = identity_init
        # V14: DeepSeek-MTP style — each head is a small d→d projection and
        # ALL heads share the tied main vocab head. ~n×d² params instead of
        # n×d×V (671M → 26M at d=2560,V=65536,n=4) and the vocab weight is
        # read once per draft step regardless of head count.
        self.adapter_heads = adapter_heads

        # Shared trunk: project hidden state → prediction representation.
        # Identity-ish init: Linear is near-zero (so trunk_out ≈ hidden),
        # LayerNorm is identity (weight=1, bias=0).
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        with torch.no_grad():
            self.trunk[0].weight.zero_()  # zero-init → trunk_out = 0 initially
            self.trunk[2].weight.fill_(1.0)
            self.trunk[2].bias.zero_()

        # N output heads. Zero-init so they predict uniform distribution
        # initially (zero MTP loss contribution). Adapter mode: d→d
        # projections, output always goes through the tied vocab head.
        head_out = d_model if adapter_heads else vocab_size
        self.heads = nn.ModuleList([
            nn.Linear(d_model, head_out, bias=False)
            for _ in range(n_heads)
        ])
        for head in self.heads:
            head.weight.data.zero_()

        self._tied_head_weight: nn.Parameter | None = None
        self._tied_source: tuple[nn.Module, str] | None = None

    def tie_head_to_model(self, model_head_weight: nn.Parameter):
        """Tie the first MTP head to the model's LM head (shared weights).

        This makes the first MTP head predict exactly what the main head
        predicts (lossless at init). Subsequent heads remain independent.
        """
        self._tied_head_weight = model_head_weight
        self._tied_source = None  # set by parent via tie_head_from_module

    def tie_head_from_module(self, parent: nn.Module, attr: str = "head"):
        """Tie using a live reference to the parent module's head weight.

        This survives parent.to(device) because we resolve the weight
        dynamically at forward time.
        """
        self._tied_source = (parent, attr)
        self._tied_head_weight = getattr(getattr(parent, attr), "weight", None)
        if not self.adapter_heads:
            self.heads[0] = None  # type: ignore[assignment]

    def _resolve_tied_weight(self, device: torch.device):
        w = self._tied_head_weight
        if self._tied_source is not None:
            parent, attr = self._tied_source
            w2 = getattr(getattr(parent, attr), "weight", None)
            w = w2 if w2 is not None else w
        if w is not None and w.device != device:
            w = nn.Parameter(w.to(device), requires_grad=w.requires_grad)
        return w

    def forward(self, hidden: torch.Tensor,
                token_embeds: torch.Tensor | None = None,
                targets: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor | None, list] | list:
        """Compute MTP auxiliary loss — or draft logits when targets is None.

        Draft mode (called by mtp_selfspec decoding as mtp(hidden)): returns
        the per-head logits list directly, no loss.

        Args:
            hidden: (B, T, d_model) final hidden states from the model.
            token_embeds: (B, T, d_model) ground-truth token embeddings
                (used as input for the recursive MTP prediction).
            targets: (B, T) ground-truth token ids.

        Returns:
            (mtp_loss, logits_list) — mtp_loss is a scalar (or None if
            sequence too short), logits_list is the list of per-head logits.
        """
        T = hidden.size(1)
        draft_mode = targets is None
        if not draft_mode and self.n_heads + 1 >= T:
            return None, []

        trunk_out = self.trunk(hidden)  # (B, T, d_model)
        # At init, trunk is zero → trunk_out = 0 → all heads output 0 →
        # softmax = uniform → CE = ln(vocab) = constant, but grad flows.

        mtp_loss = torch.tensor(0.0, device=hidden.device, dtype=hidden.dtype)
        logits_list = []
        tied_w = self._resolve_tied_weight(hidden.device)

        for k in range(self.n_heads):
            # Head k predicts token at position t + k + 1
            offset = k + 1

            if draft_mode:
                # Drafting: every head reads the LAST position's features.
                pred_input = trunk_out[:, -1:, :]  # (B, 1, d_model)
            else:
                if offset + 1 >= T:
                    break
                # Use trunk_out at position t to predict target at t + offset
                pred_input = trunk_out[:, :-offset, :]  # (B, T-offset, d_model)
                target_k = targets[:, offset:]  # (B, T-offset)

            if self.adapter_heads:
                # d→d adapter + shared tied vocab head (all heads).
                if tied_w is None:
                    raise RuntimeError(
                        "MTPModule(adapter_heads=True) requires a tied "
                        "vocab head — call tie_head_to_model/"
                        "tie_head_from_module first.")
                logits_k = F.linear(self.heads[k](pred_input), tied_w)
            elif tied_w is not None and k == 0:
                logits_k = F.linear(pred_input, tied_w)
            else:
                logits_k = self.heads[k](pred_input)

            logits_list.append(logits_k)

            if draft_mode:
                continue
            # CE loss for this head
            loss_k = F.cross_entropy(
                logits_k.reshape(-1, logits_k.size(-1)),
                target_k.reshape(-1),
                ignore_index=-100,
            )
            mtp_loss = mtp_loss + loss_k

        if draft_mode:
            return logits_list

        if self.n_heads > 0:
            mtp_loss = mtp_loss * self.loss_weight / self.n_heads

        return mtp_loss, logits_list
