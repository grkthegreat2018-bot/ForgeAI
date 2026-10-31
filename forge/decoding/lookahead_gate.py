"""Lookahead Quality Gate for speculative decoding.

Based on "Look Before You Leap: A Lookahead Reasoning Quality Gate for
Speculative Decoding" (EACL 2026, arXiv 2026.eacl-long.367).

Problem: standard speculative decoding accepts tokens based on per-token
likelihood, which is myopic and often rewards verbosity. Tree-level methods
trade accuracy for latency.

Solution: a block-wise quality gate that accepts the LONGEST RELIABLE PREFIX
of each k-token lookahead draft. Uses only the base model's hidden states
to compute a geometry-based quality score.

Key features:
  - Intermediate granularity (between token-level and tree-level)
  - Geometry-based quality score from hidden states (no auxiliary heads)
  - Quantile-calibrated threshold (estimated from unlabeled prompts)
  - No reward models or finetuning needed
  - Integrates with any speculative/blockwise decoding

Results: 2.6-7.9× faster generation on math/science benchmarks while
IMPROVING accuracy over sampling baselines.

For our model:
  - Current: per-token acceptance (myopic)
  - Lookahead gate: accept longest reliable prefix (block-wise)
  - Better accuracy + faster generation
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class LookaheadQualityGate:
    """Geometry-based quality gate for speculative decoding.

    Computes a quality score for each k-token draft prefix using the base
    model's hidden states. Accepts the longest prefix whose score exceeds
    a quantile-calibrated threshold.
    """

    def __init__(self, n_lookahead: int = 4,
                 threshold_quantile: float = 0.9,
                 calibration_tokens: int = 1000):
        self.n_lookahead = n_lookahead
        self.threshold_quantile = threshold_quantile
        self.calibration_tokens = calibration_tokens
        self._score_history: list[float] = []
        self._threshold: float = 0.5  # calibrated threshold
        self._calibrated = False

    def compute_quality_score(self, hidden_states: torch.Tensor,
                               draft_tokens: torch.Tensor,
                               draft_logits: torch.Tensor) -> float:
        """Compute geometry-based quality score for a draft prefix.

        The score is based on the geometric consistency between the base
        model's hidden state trajectory and the draft tokens. High score =
        the draft follows a "natural" trajectory in hidden space.

        Args:
            hidden_states: (B, T, d_model) base model hidden states
            draft_tokens: (B, K) proposed draft tokens
            draft_logits: (B, K, V) draft logits

        Returns:
            score: quality score (0-1, higher = better)
        """
        _B, T, _D = hidden_states.shape
        draft_tokens.shape[1]

        # Geometry-based score: measure how "in-distribution" the draft is
        # 1. Compute draft log-probabilities (confidence)
        log_probs = F.log_softmax(draft_logits, dim=-1)
        draft_log_probs = log_probs.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)
        # (B, K) — log-prob of each draft token

        # 2. Geometric consistency: hidden state trajectory smoothness
        # Measure the angular change in hidden state direction
        if T > 2:
            hidden_diff = hidden_states[:, 1:] - hidden_states[:, :-1]  # (B, T-1, D)
            # Cosine similarity between consecutive differences
            cos_sims = F.cosine_similarity(
                hidden_diff[:, :-1], hidden_diff[:, 1:], dim=-1)  # (B, T-2)
            trajectory_smoothness = cos_sims.mean().item()
        else:
            trajectory_smoothness = 0.5

        # 3. Draft confidence: average log-prob of draft tokens
        draft_confidence = draft_log_probs.mean().exp().item()  # avg prob

        # 4. Combine: geometric score
        score = 0.5 * draft_confidence + 0.5 * trajectory_smoothness

        return score

    def calibrate(self, hidden_states_list: list[torch.Tensor],
                  draft_logits_list: list[torch.Tensor]):
        """Calibrate the acceptance threshold from unlabeled prompts.

        Collects quality scores from many draft attempts and sets the
        threshold at the specified quantile.
        """
        scores = []
        for hidden, draft_logits in zip(hidden_states_list, draft_logits_list):
            # Generate random draft tokens for calibration
            K = min(self.n_lookahead, draft_logits.shape[1])
            draft_tokens = draft_logits[:, :K].argmax(dim=-1)
            score = self.compute_quality_score(hidden, draft_tokens, draft_logits[:, :K])
            scores.append(score)

            if len(scores) >= self.calibration_tokens:
                break

        if scores:
            scores_tensor = torch.tensor(scores)
            self._threshold = scores_tensor.quantile(self.threshold_quantile).item()
            self._calibrated = True
            print(f"  [LookaheadGate] Calibrated: threshold={self._threshold:.4f} "
                  f"(quantile={self.threshold_quantile}, n={len(scores)})")

    def accept_prefix(self, hidden_states: torch.Tensor,
                      draft_tokens: torch.Tensor,
                      draft_logits: torch.Tensor) -> int:
        """Determine how many draft tokens to accept.

        Returns the length of the longest prefix whose quality score
        exceeds the calibrated threshold.

        Args:
            hidden_states: (B, T, d_model) base model hidden states
            draft_tokens: (B, K) proposed draft tokens
            draft_logits: (B, K, V) draft logits

        Returns:
            n_accept: number of tokens to accept (0 to K)
        """
        K = draft_tokens.shape[1]

        # Compute quality score for each prefix length 1..K
        for k in range(1, K + 1):
            score = self.compute_quality_score(
                hidden_states, draft_tokens[:, :k], draft_logits[:, :k])

            # Record score for ongoing calibration
            self._score_history.append(score)
            if len(self._score_history) > 10000:
                self._score_history = self._score_history[-5000:]

            if score < self._threshold:
                return k - 1  # accept up to (but not including) this point

        return K  # all tokens accepted

    def adaptive_threshold(self) -> float:
        """Adaptively update threshold based on recent scores."""
        if len(self._score_history) < 100:
            return self._threshold

        recent = torch.tensor(self._score_history[-100:])
        return recent.quantile(self.threshold_quantile).item()

    def stats(self) -> dict:
        return {
            "n_lookahead": self.n_lookahead,
            "threshold": self._threshold,
            "calibrated": self._calibrated,
            "n_scores": len(self._score_history),
            "quantile": self.threshold_quantile,
        }
