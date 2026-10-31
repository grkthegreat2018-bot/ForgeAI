"""MoE routing: keyword-matching and semantic embedding-based routers.

Merged from keyword_router.py (2.9KB) + semantic_router.py (6.4KB).

KeywordRouter: keyword-matching topic router — scores topics by keyword
substring hits, returns the highest-scoring topic.

SemanticRouter: embedding-based router — uses the model's hidden states
(or logits as fallback) to compute mean-pooled embeddings of topic
descriptions and queries, then routes by cosine similarity.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

try:
    from loguru import logger as _log
except ImportError:
    import logging
    _log = logging.getLogger(__name__)


# ── KeywordRouter ──────────────────────────────────────────────────────

class KeywordRouter:
    """Base class for keyword-matching topic routers.

    Args:
        keywords: mapping of topic name -> list of keyword strings.
        fallback: topic name returned when nothing scores above threshold.
        min_score: confidence threshold — minimum keyword hits required for
            a topic to be considered a match (default 1, i.e. any hit).
    """

    def __init__(self, keywords: dict[str, list[str]] | None = None,
                 fallback: str = "general", min_score: int = 1):
        self.keywords: dict[str, list[str]] = dict(keywords) if keywords else {}
        self.fallback = fallback
        self.min_score = max(1, min_score)

    def _iter_keywords(self) -> Iterable[tuple[str, list[str]]]:
        """Yield (topic, keywords) pairs. Subclasses may override to source
        keywords dynamically (e.g. from a mutable manifest or index)."""
        return self.keywords.items()

    def _topic_names(self) -> Iterable[str]:
        """Topic names for list_topics(). Subclasses may override."""
        return self.keywords.keys()

    def _score(self, query_lower: str) -> dict[str, int]:
        """Score all topics against an already-lowercased query.

        Only topics meeting the confidence threshold are included.
        """
        scores = {}
        for topic, kws in self._iter_keywords():
            score = sum(1 for kw in kws if kw in query_lower)
            if score >= self.min_score:
                scores[topic] = score
        return scores

    def classify(self, query: str) -> str:
        """Classify a query into the best-matching topic.

        Returns the topic with the highest keyword match score, or the
        fallback topic if nothing meets the confidence threshold.
        """
        scores = self._score(query.lower())
        if scores:
            return max(scores, key=scores.get)
        return self.fallback

    def classify_multi(self, query: str, top_n: int = 2) -> list[str]:
        """Classify a query into up to top_n topics, sorted by match score."""
        scores = self._score(query.lower())
        if not scores:
            return [self.fallback]
        sorted_topics = sorted(scores, key=scores.get, reverse=True)
        return sorted_topics[:top_n]

    def list_topics(self) -> list[str]:
        """List all known topic names (sorted)."""
        return sorted(self._topic_names())


# ── SemanticRouter ─────────────────────────────────────────────────────

DEFAULT_TOPIC_DESCRIPTIONS = {
    "python_algorithms": "Python programming algorithms: sorting, searching, recursion, dynamic programming, data structures, fibonacci, factorial, prime numbers, graph traversal",
    "math_arithmetic": "Mathematical arithmetic: calculations, number theory, factorials, GCD, LCM, prime factorization, modular arithmetic, basic algebra",
    "python_strings": "Python string manipulation: reversing, parsing, pattern matching, regex, text processing, character counting, palindromes",
    "python_general": "General Python programming: functions, loops, conditionals, lists, dictionaries, file I/O, error handling, object-oriented programming",
    "python_file_io": "Python file input/output: reading files, writing files, CSV processing, JSON handling, file parsing",
    "python_math": "Python mathematical computing: numpy, calculations, numerical methods, statistics, linear algebra",
    "python_oop": "Python object-oriented programming: classes, inheritance, polymorphism, encapsulation, design patterns",
    "coding": "Software development: code generation, debugging, refactoring, testing, algorithms, data structures, programming languages",
    "math": "Mathematics: algebra, calculus, geometry, statistics, probability, number theory, proofs, equations",
    "algorithms": "Algorithm design and analysis: complexity, sorting, searching, graph algorithms, dynamic programming, greedy algorithms",
    "theory": "Theoretical reasoning: explanation, analysis, logic, proofs, conceptual understanding, critical thinking",
    "creativity": "Creative writing: stories, poems, brainstorming, imaginative text, narrative generation",
    "tool_use": "Tool usage and function calling: API calls, command execution, tool selection, automation, scripting",
    "token_efficiency": "Concise efficient responses: brief answers, minimal tokens, clear and direct communication",
    "general": "General knowledge and assistance: questions, explanations, help, advice, information",
}

MAX_LEN = 128
LOW_CONF_THRESHOLD = 0.3


class SemanticRouter:
    def __init__(self, model, tokenizer, topic_descriptions: dict[str, str], device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.topic_embeddings: dict[str, torch.Tensor] = {}
        self._supports_hidden = self._check_hidden_support()
        self._register_batch(topic_descriptions)

    def _check_hidden_support(self) -> bool:
        """Probe whether forward returns hidden states alongside logits."""
        try:
            self.model.eval()
            dummy = torch.zeros((1, 2), dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = self.model(dummy)
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                return True
        except Exception as e:
            _log.warning(f"[SemanticRouter] hidden-state probe failed ({e}); using logits fallback")
        return False

    def _embed(self, texts: list[str]) -> torch.Tensor:
        """Compute mean-pooled embeddings for a batch of texts."""
        self.model.eval()
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
        )
        input_ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids)
        if self._supports_hidden and isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            hidden = out[1]
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[-1]
            mask = enc.get("attention_mask")
            if mask is not None:
                mask = mask.to(self.device).unsqueeze(-1).float()
                hidden = hidden * mask
                emb = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                emb = hidden.mean(dim=1)
        else:
            logits = out[0] if isinstance(out, tuple) else out
            emb = logits.float().mean(dim=1)
        return emb.float()

    def _register_batch(self, topic_descriptions: dict[str, str]) -> None:
        if not topic_descriptions:
            return
        topics = list(topic_descriptions.keys())
        descs = [topic_descriptions[t] for t in topics]
        embs = self._embed(descs)
        for t, e in zip(topics, embs):
            self.topic_embeddings[t] = e.detach()

    def register_topic(self, topic: str, description: str) -> None:
        self.topic_embeddings[topic] = self._embed([description])[0].detach()

    def _similarities(self, query: str) -> dict[str, float]:
        q = self._embed([query])[0]
        sims = {}
        for t, e in self.topic_embeddings.items():
            sims[t] = F.cosine_similarity(q.unsqueeze(0), e.unsqueeze(0)).item()
        return sims

    def classify(self, query: str) -> str:
        sims = self._similarities(query)
        if not sims:
            return "general"
        best_topic, best_sim = max(sims.items(), key=lambda kv: kv[1])
        return best_topic if best_sim >= LOW_CONF_THRESHOLD else "general"


# ── R39-7: LASER — Plug-and-Play Expert Routing ──────────────────────────

class LASERRouter:
    """LASER: Layer-Selective Expert Routing.

    Plug-and-play routing algorithm that modifies expert selection at
    inference time without retraining. The key insight: not all layers
    need the same number of active experts. Early layers benefit from
    more experts (feature extraction), later layers need fewer (reasoning).

    Args:
        n_experts: Total number of experts per layer.
        n_layers: Number of MoE layers.
        default_top_k: Default number of experts to activate.
        layer_k_overrides: Dict of {layer_idx: top_k} to override defaults.
    """

    def __init__(self, n_experts: int = 8, n_layers: int = 32,
                 default_top_k: int = 2,
                 layer_k_overrides: dict[int, int] | None = None):
        self.n_experts = n_experts
        self.n_layers = n_layers
        self.default_top_k = default_top_k
        self.layer_k_overrides = layer_k_overrides or {}
        # Default: early layers get more experts, later layers fewer
        if not self.layer_k_overrides:
            for i in range(n_layers):
                if i < n_layers // 3:
                    self.layer_k_overrides[i] = min(default_top_k + 1, n_experts)
                elif i > 2 * n_layers // 3:
                    self.layer_k_overrides[i] = max(default_top_k - 1, 1)
                else:
                    self.layer_k_overrides[i] = default_top_k

    def get_top_k(self, layer_idx: int) -> int:
        """Get the number of experts to activate for a given layer."""
        return self.layer_k_overrides.get(layer_idx, self.default_top_k)

    def route(self, router_logits: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts based on layer-specific top-k.

        Args:
            router_logits: (batch * seq_len, n_experts) raw router logits.
            layer_idx: Index of the MoE layer.

        Returns:
            (expert_indices, expert_weights) — top-k experts per token.
        """
        top_k = self.get_top_k(layer_idx)
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = probs.topk(top_k, dim=-1)
        # Normalize selected expert weights
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return topk_indices, topk_probs

    def get_config(self) -> dict:
        """Return the routing configuration for logging."""
        return {
            "algorithm": "LASER",
            "n_experts": self.n_experts,
            "n_layers": self.n_layers,
            "default_top_k": self.default_top_k,
            "layer_overrides": dict(self.layer_k_overrides),
        }


# ── R39-7: METRO — Memory-Efficient Expert Routing ───────────────────────

class METRORouter:
    """METRO: Memory-Efficient Throughput-Routing.

    Balances activated experts at inference time to reduce decode latency
    in the memory-bound regime (which is the case for 12GB VRAM).

    Key insight: in the memory-bound regime, decode latency is dominated
    by expert weight loading, not compute. Balancing expert activation
    reduces the number of unique experts loaded per batch, improving
    cache hit rate.

    Args:
        n_experts: Total number of experts.
        top_k: Number of experts to activate per token.
        balance_threshold: Maximum allowed load imbalance ratio.
            If max_expert_load / mean_expert_load > threshold, rebalance.
        history_window: Number of recent tokens to track for load balancing.
    """

    def __init__(self, n_experts: int = 8, top_k: int = 2,
                 balance_threshold: float = 1.5,
                 history_window: int = 256):
        self.n_experts = n_experts
        self.top_k = top_k
        self.balance_threshold = balance_threshold
        self.history_window = history_window
        # Track expert activation counts
        self.expert_counts = torch.zeros(n_experts)
        self.total_tokens = 0

    def route(self, router_logits: torch.Tensor, layer_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts with load balancing.

        If the current expert load is imbalanced, applies a correction
        factor to the router logits to encourage underutilized experts.

        Args:
            router_logits: (batch * seq_len, n_experts) raw router logits.
            layer_idx: Layer index (unused, for API compatibility).

        Returns:
            (expert_indices, expert_weights) — top-k experts per token.
        """
        # Compute current load ratios
        if self.total_tokens > 0:
            mean_load = self.expert_counts.mean().clamp(min=1.0)
            max_load = self.expert_counts.max().clamp(min=1.0)
            imbalance = max_load / mean_load
        else:
            imbalance = 1.0

        # Apply correction if imbalanced
        if imbalance > self.balance_threshold:
            # Boost underutilized experts by adding a bonus to their logits
            mean_load_val = mean_load.item()
            load_ratio = self.expert_counts / max(mean_load_val, 1.0)
            # Bonus for underutilized experts (load_ratio < 1), penalty for overused
            # Scale by imbalance factor for stronger correction when very imbalanced
            correction_strength = (imbalance - 1.0) * 3.0
            correction = torch.clamp(
                (1.0 - load_ratio) * correction_strength, min=-5.0, max=10.0
            ).to(router_logits.dtype)
            corrected_logits = router_logits + correction.unsqueeze(0)
        else:
            corrected_logits = router_logits

        probs = F.softmax(corrected_logits, dim=-1)
        topk_probs, topk_indices = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Update expert counts
        with torch.no_grad():
            for idx in topk_indices.flatten():
                self.expert_counts[idx] += 1
            self.total_tokens += topk_indices.shape[0]
            # Trim history
            if self.total_tokens > self.history_window:
                decay = self.history_window / self.total_tokens
                self.expert_counts *= decay
                self.total_tokens = int(self.total_tokens * decay)

        return topk_indices, topk_probs

    def get_load_balance(self) -> float:
        """Return the current load balance ratio (1.0 = perfectly balanced)."""
        if self.total_tokens == 0:
            return 1.0
        mean_load = self.expert_counts.mean().clamp(min=1.0)
        max_load = self.expert_counts.max().clamp(min=1.0)
        self.expert_counts.min()
        return (max_load / mean_load).item()

    def reset_stats(self) -> None:
        """Reset expert load tracking."""
        self.expert_counts.zero_()
        self.total_tokens = 0

    def get_config(self) -> dict:
        """Return the routing configuration for logging."""
        return {
            "algorithm": "METRO",
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "balance_threshold": self.balance_threshold,
            "history_window": self.history_window,
            "current_balance": self.get_load_balance(),
        }

    def classify_multi(self, query: str, top_n: int = 2) -> list[str]:
        sims = self._similarities(query)
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        return [t for t, _ in ranked[:top_n]]

    def list_topics(self) -> list[str]:
        return sorted(self.topic_embeddings.keys())

    def save(self, path: str) -> None:
        torch.save(
            {"topic_embeddings": {t: e.cpu() for t, e in self.topic_embeddings.items()}},
            path,
        )

    @classmethod
    def load(cls, path: str, model, tokenizer, device: str = "cuda") -> SemanticRouter:
        try:
            data = torch.load(path, map_location=device, weights_only=True)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Could not load router state {path} with weights_only=True "
                f"({e}). Falling back to weights_only=False (UNSAFE — "
                f"enables arbitrary code execution from the file). Only "
                f"proceed if you trust this checkpoint's source.",
                RuntimeWarning, stacklevel=2)
            data = torch.load(path, map_location=device, weights_only=False)
        inst = cls(model, tokenizer, {}, device=device)
        for t, e in data["topic_embeddings"].items():
            inst.topic_embeddings[t] = e.to(device)
        return inst
