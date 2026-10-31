"""PIT (Pseudo-Inverse Tying) — replace weight tying with orthonormal
shared memory + learned SPD transform.

PIT (arXiv:2602.04556) addresses the bias problem of standard weight tying:
weight tying biases the shared matrix toward the OUTPUT space (unembedding),
compromising input representation quality — harmful at small scale where
embeddings dominate parameter count.

PIT architecture:
  - Shared memory M ∈ R^{V×D} (orthonormal via thin polar decomposition
    for teacher init, or standard embedding init for from-scratch).
  - Hidden transform T = L·L^T (L lower-triangular, learned, init=I).
    T is symmetric positive definite (SPD).
  - Output (LM head):  logits = (T · h) · M^T
    = h @ T @ M^T  (T transforms hidden state before projecting to vocab)
  - Input (embedding): embed = solve(L, M[token]^T) then solve(L^T, ·)
    = solve(L^T, solve(L, M[token]^T))
    This is the inverse of T applied to the token's shared memory row,
    recovered via two stable triangular solves (no matrix inversion).

Identity init: L = I → T = I → logits = h @ M^T, embed = M[token].
This is EXACTLY standard weight tying — PIT is lossless at start.

Benefits over weight tying:
  1. T decouples the input/output spaces: the model can learn a transform
     that makes the output projection better without degrading input quality.
  2. The SPD constraint on T ensures the transform is stable and invertible
     throughout training (no singularities).
  3. No vocabulary-sized auxiliary parameters — T is D×D (small).
  4. Better training stability, layerwise semantic consistency, and reduced
     side effects of post-training edits (critical for live_learn.py and
     self-play knowledge injection).

Classification: PARTIAL (forward direction: data -> weights with learned T).
The reverse direction requires extracting T from a trained model, which
needs a separate probe.

Usage:
    from forge.keys.misc.pit_key import PITKey, PITEmbedding, PITLMHead

    # In model construction:
    pit_embed = PITEmbedding(vocab_size, d_model)
    pit_head = PITLMHead.from_embedding(pit_embed)  # shares M and L
    # logits = pit_head(hidden_states)
    # embed = pit_head(input_ids)  # or pit_embed(input_ids)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.keys.misc.base import Key, KeyClass, KeyResult


class PITEmbedding(nn.Module):
    """PIT input embedding: embed = solve(L^T, solve(L, M[token]^T)).

    The shared memory M is orthonormal (for teacher init) or standard
    embedding (for from-scratch). The transform T = L·L^T is applied
    inversely at embedding lookup via stable triangular solves.

    Args:
        vocab_size: vocabulary size V.
        d_model: model dimension D.
        init: "standard" (normal embedding init) or "orthonormal"
            (thin polar decomposition for teacher init).
        padding_idx: optional padding token index (zeroed in output).
    """

    def __init__(self, vocab_size: int, d_model: int,
                 init: str = "standard", padding_idx: int | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        # Shared memory M: [V, D]
        if init == "orthonormal":
            # Thin polar decomposition: random matrix -> orthonormal via SVD.
            M = torch.randn(vocab_size, d_model)
            U, _S, Vh = torch.linalg.svd(M, full_matrices=False)
            M = U @ Vh  # orthonormal rows
        else:
            # Standard embedding init (normal, scaled by 1/sqrt(d_model)).
            M = torch.randn(vocab_size, d_model) * (1.0 / d_model ** 0.5)

        self.memory = nn.Parameter(M)

        # L: lower-triangular, init=I (identity → T=I → standard tying).
        L = torch.eye(d_model)
        # Make L explicitly lower-triangular with unit diagonal (Cholesky-like).
        # The off-diagonal lower entries are learnable; diagonal is fixed at 1
        # to ensure T = L·L^T is SPD with determinant >= 1.
        # Actually, for full expressiveness, allow diagonal to be learnable
        # but initialized to 1. Use register_tril for the parameter.
        self.L = nn.Parameter(L)

        # Mask for lower-triangular (applied in forward to ensure structure).
        self.register_buffer("tril_mask", torch.tril(torch.ones(d_model, d_model)),
                             persistent=False)

    def get_T(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Compute T = L · L^T (SPD transform).

        Args:
            dtype: if given, cast L and tril_mask to this dtype before
                computing T. This prevents dtype mismatch when L is bf16
                but tril_mask is float32 (type promotion would make T
                float32, breaking bf16 matmuls downstream).
        """
        L = self.L
        mask = self.tril_mask
        if dtype is not None:
            L = L.to(dtype)
            mask = mask.to(dtype)
        L = L * mask  # enforce lower-triangular
        return L @ L.transpose(-1, -2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input_ids: solve(L^T, solve(L, M[token]^T)).

        Args:
            input_ids: (B, seq_len) token indices.

        Returns:
            (B, seq_len, d_model) embedded vectors.
        """
        # M[token]: (B, seq_len, D)
        embed = F.embedding(input_ids, self.memory, padding_idx=self.padding_idx)
        # embed: (B, seq_len, D)

        # Apply T^{-1} = (L·L^T)^{-1} = L^{-T} · L^{-1}
        # via two triangular solves:
        #   step 1: solve L · y = M[token]^T  → y = L^{-1} · M[token]^T
        #   step 2: solve L^T · z = y         → z = L^{-T} · y
        # z = T^{-1} · M[token]^T

        # Reshape for batched solve: (B*seq_len, D)
        B, S, D = embed.shape
        x = embed.reshape(-1, D).unsqueeze(-1)  # (B*S, D, 1)

        # triangular_solve_cuda not implemented for BFloat16 on CUDA —
        # upcast to float32 for the solve, then cast back.
        orig_dtype = x.dtype
        L = self.L * self.tril_mask  # enforce lower-triangular
        L_f = L.float() if L.dtype != torch.float32 else L
        x_f = x.float() if x.dtype != torch.float32 else x

        # solve(L, x): solves L · y = x
        y = torch.linalg.solve_triangular(L_f, x_f, upper=False)
        # solve(L^T, y): solves L^T · z = y
        z = torch.linalg.solve_triangular(L_f.transpose(-1, -2), y, upper=True)

        z = z.squeeze(-1).reshape(B, S, D).to(orig_dtype)
        return z


class PITLMHead(nn.Module):
    """PIT LM head: logits = (T · h) · M^T.

    Shares the memory M and transform L with PITEmbedding.

    Args:
        memory: shared memory parameter [V, D] (from PITEmbedding).
        L: lower-triangular transform parameter [D, D] (from PITEmbedding).
        tril_mask: lower-triangular mask (from PITEmbedding).
        bias: optional bias for the head (default None).
    """

    def __init__(self, memory: nn.Parameter, L: nn.Parameter,
                 tril_mask: torch.Tensor, bias: bool = False):
        super().__init__()
        # Share parameters (not copy) — modifications affect both.
        self.memory = memory
        self.L = L
        self.register_buffer("tril_mask", tril_mask, persistent=False)
        self.bias = nn.Parameter(torch.zeros(memory.shape[0])) if bias else None

    @classmethod
    def from_embedding(cls, embed: PITEmbedding, bias: bool = False) -> PITLMHead:
        """Create a PITLMHead sharing parameters with a PITEmbedding."""
        return cls(embed.memory, embed.L, embed.tril_mask, bias=bias)

    def get_T(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Compute T = L · L^T.

        Args:
            dtype: if given, cast L and tril_mask to this dtype before
                computing T. Prevents dtype mismatch when tril_mask is
                float32 but L/model is bf16.
        """
        L = self.L
        mask = self.tril_mask
        if dtype is not None:
            L = L.to(dtype)
            mask = mask.to(dtype)
        L = L * mask
        return L @ L.transpose(-1, -2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute logits: (T · h) · M^T.

        Args:
            hidden: (B, seq_len, D) hidden states.

        Returns:
            (B, seq_len, V) logits.
        """
        T = self.get_T(dtype=hidden.dtype)  # [D, D], matches hidden dtype
        # Apply T to hidden: h_T = h @ T  (T is symmetric, so T·h = h·T^T = h·T)
        h_transformed = hidden @ T  # (B, seq_len, D)
        # Project to vocab: logits = h_T @ M^T
        logits = h_transformed @ self.memory.transpose(-1, -2)  # (B, seq_len, V)
        if self.bias is not None:
            logits = logits + self.bias
        return logits


class PITKey(Key):
    """PIT (Pseudo-Inverse Tying) key.

    Converts between standard tied embedding weights and PIT parameters
    (shared memory M + lower-triangular transform L).

    forward(data): Given a tied weight (embed_weight = head_weight),
        initialize M = embed_weight and L = I (identity).
        This is lossless: PIT with L=I is exactly standard tying.

    reverse(weights): Given M and L, extract the equivalent tied weight.
        With L=I, this is just M. With L≠I, the "effective" tied weight
        is M @ T (for output) and M @ T^{-1} (for input) — these differ,
        which is the whole point of PIT (decoupling input/output).
        We return M as the shared weight and L separately.
    """

    @property
    def name(self) -> str:
        return "pit"

    @property
    def description(self) -> str:
        return ("PIT (Pseudo-Inverse Tying): orthonormal shared memory + "
                "learned SPD transform. Replaces weight tying.")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """data -> weights. Expects 'embed_weight' (tied weight).

        Initializes M = embed_weight, L = I (identity = standard tying).
        """
        try:
            embed_weight = data["embed_weight"]
            V, D = embed_weight.shape
            memory = embed_weight.clone()
            L = torch.eye(D)
            return KeyResult(
                success=True,
                weights={"memory": memory, "L": L},
                metadata={"vocab_size": V, "d_model": D, "init": "from_tied"},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """weights -> data. Extract embed_weight from PIT parameters.

        With L=I, embed_weight = M. With L≠I, the input embedding is
        M @ T^{-1} and the output head is M @ T — we return the input
        embedding (M @ T^{-1}) as the "data" representation.
        """
        try:
            memory = weights.get("memory")
            L = weights.get("L")
            if memory is None or L is None:
                return KeyResult(success=False,
                                 error="Missing 'memory' or 'L' in weights")
            D = memory.shape[1]
            tril_mask = torch.tril(torch.ones(D, D, device=memory.device,
                                              dtype=memory.dtype))
            L_lt = L * tril_mask
            T = L_lt @ L_lt.transpose(-1, -2)
            # Input embedding = M @ T^{-1}
            T_inv = torch.linalg.inv(T)
            embed_weight = memory @ T_inv
            return KeyResult(
                success=True,
                data={"embed_weight": embed_weight},
                metadata={"pit": True, "had_transform": not torch.allclose(L, torch.eye(D))},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def apply_pit_to_model(model: nn.Module, config=None,
                       test_input=None, safe: bool = True) -> nn.Module:
    """Replace tied embedding/head in a model with PIT.

    Uses safety validation to ensure the model is not corrupted. PIT with
    L=I (identity init) is exactly standard weight tying, so the forward
    output should be numerically identical after application.

    Args:
        model: the model with a tied embedding (model.embed_tokens.weight
            is shared with model.lm_head.weight).
        config: optional ModelConfig with vocab_size and d_model.
        test_input: optional input tensor for forward pass validation.
        safe: if True, use safe_apply with rollback on corruption.

    Returns:
        The model with PIT components (in-place modification).
    """
    def _apply(m):
        embed = getattr(m, "embed_tokens", None) or getattr(m, "embedding", None)
        if embed is None:
            raise ValueError("Cannot find embedding layer (looked for embed_tokens, embedding)")

        V, D = embed.weight.shape

        pit_embed = PITEmbedding(V, D, init="standard",
                                 padding_idx=getattr(embed, "padding_idx", None))
        pit_embed.memory.data.copy_(embed.weight.data)

        pit_head = PITLMHead.from_embedding(pit_embed)

        if hasattr(m, "embed_tokens"):
            m.embed_tokens = pit_embed
        else:
            m.embedding = pit_embed

        if hasattr(m, "lm_head"):
            m.lm_head = pit_head
        elif hasattr(m, "head"):
            m.head = pit_head
        return m

    if safe:
        from forge.keys.safety import safe_apply
        return safe_apply(model, _apply, identity_init=True,
                          test_input=test_input, atol=1e-4, rtol=1e-3)
    return _apply(model)
