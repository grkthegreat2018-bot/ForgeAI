"""SIGReg — Spectral Implicit Geometry Regularization (LeWM / LeCun et al.).

Prevents hidden-state collapse in deep models by penalizing spectral norms
that fall below a threshold. The regularization encourages the largest
singular value of each layer's hidden-state matrix to stay above ``threshold``,
preventing all hidden states from converging to a low-rank manifold.

Loss formula::

    L_sigreg = sum_l max(0, threshold - sigma_1(H_l))^2

where ``sigma_1(H_l)`` is the largest singular value of the hidden-state
matrix at layer *l*.

Two spectral-norm backends are provided:
  - ``"svd"``: exact via ``torch.linalg.svdvals`` (accurate, differentiable).
  - ``"power"``: power-iteration estimate (faster for large matrices,
    differentiable through the final Rayleigh quotient).

The power-iteration backend runs a fixed number of iterations (default 3)
which is sufficient for a close upper-bound on the largest singular value
while keeping compute low.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SIGRegLoss(nn.Module):
    """Spectral Implicit Geometry Regularization loss.

    Computes a hinge-like penalty on the largest singular value of each
    layer's hidden-state matrix, encouraging spectral norms to stay above
    a threshold to prevent hidden-state collapse.

    Args:
        threshold: minimum desired spectral norm per layer.
        backend: ``"svd"`` (exact) or ``"power"`` (power-iteration estimate).
        power_iters: number of power iterations when ``backend="power"``.
        reduction: ``"sum"`` or ``"mean"`` over layers.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        backend: str = "svd",
        power_iters: int = 3,
        reduction: str = "sum",
    ):
        super().__init__()
        self.threshold = threshold
        self.backend = backend
        self.power_iters = power_iters
        self.reduction = reduction

    @staticmethod
    def _largest_singular_value_svd(h: torch.Tensor) -> torch.Tensor:
        """Exact largest singular value via ``torch.linalg.svdvals``.

        ``h`` is reshaped to 2-D ``(M, N)`` if it has more than 2 dims
        (the spectral norm of a 2-D reshape is an upper bound on the
        true spectral norm of the higher-order tensor, which is a
        conservative choice for the regularizer).
        """
        if h.dim() > 2:
            h = h.reshape(h.shape[0] * h.shape[1], -1) if h.dim() == 3 else h.reshape(-1, h.shape[-1])
        # svdvals returns singular values in descending order.
        s = torch.linalg.svdvals(h.float())
        return s[0]

    @staticmethod
    def _largest_singular_value_power(
        h: torch.Tensor, n_iters: int = 3
    ) -> torch.Tensor:
        """Largest singular value estimate via power iteration.

        Estimates ``sigma_1(H) = sqrt(lambda_max(H^T H))`` by iterating
        ``v <- H^T (H v)`` and normalizing.  The final Rayleigh quotient
        ``||H v|| / ||v||`` gives the singular value estimate.
        """
        if h.dim() > 2:
            h = h.reshape(h.shape[0] * h.shape[1], -1) if h.dim() == 3 else h.reshape(-1, h.shape[-1])
        h_f = h.float()
        _M, N = h_f.shape
        # Random init vector in the column space.
        v = torch.randn(N, device=h_f.device, dtype=h_f.dtype)
        v = v / v.norm().clamp(min=1e-8)
        for _ in range(n_iters):
            # u = H v  (M,)
            u = h_f @ v
            u_norm = u.norm().clamp(min=1e-8)
            u = u / u_norm
            # v = H^T u  (N,)
            v = h_f.t() @ u
            v_norm = v.norm().clamp(min=1e-8)
            v = v / v_norm
        # sigma = ||H v||
        sigma = (h_f @ v).norm()
        return sigma

    def _largest_singular_value(self, h: torch.Tensor) -> torch.Tensor:
        if self.backend == "power":
            return self._largest_singular_value_power(h, self.power_iters)
        return self._largest_singular_value_svd(h)

    def forward(
        self,
        hidden_states_list: list[torch.Tensor],
        threshold: float | None = None,
    ) -> torch.Tensor:
        """Compute the SIGReg penalty.

        Args:
            hidden_states_list: one hidden-state tensor per layer, each of
                shape ``(B, T, D)`` or ``(M, N)``.
            threshold: optional per-call override of the threshold.

        Returns:
            Scalar loss tensor (sum or mean of per-layer penalties).
        """
        thr = threshold if threshold is not None else self.threshold
        total = torch.zeros((), device=hidden_states_list[0].device,
                            dtype=torch.float32)
        n_layers = 0
        for h in hidden_states_list:
            sigma = self._largest_singular_value(h)
            # Hinge: max(0, threshold - sigma)^2
            penalty = torch.clamp(thr - sigma, min=0.0).pow(2)
            total = total + penalty
            n_layers += 1
        if self.reduction == "mean" and n_layers > 0:
            total = total / n_layers
        return total
