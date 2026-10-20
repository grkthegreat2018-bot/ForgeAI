"""Tiny neural generators — produce candidate configs from noise + context.

Each generator is a small MLP (4 layers, ~200K params with hidden=128).
Input: noise vector (diversity) + context vector (current best config).
Output: candidate config (continuous params, discretized by domain).

Population of N generators explores different regions of config space.
REINFORCE updates push generators toward producing high-scoring configs.
Bottom 20% die each generation; top 20% spawn mutated children.

Adaptive features (self-improving loop):
  - mutation_rate decays over generations (exploit more as we converge)
  - diversity burst triggers when population variance drops (anti-plateau)
  - LayerNorm in generators for training stability (deeper nets train better)
  - 4-layer network (was 3) for more expressive config generation

CUDA: All generators are batched into a single (N, ...) weight tensor.
Forward pass for all 500 generators runs as ONE batched matmul on GPU.
This turns 500 sequential CPU forward passes into 1 GPU batched op.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GeneratorConfig:
    noise_dim: int = 16           # randomness input
    context_dim: int = 32         # current best config encoding
    hidden_dim: int = 128         # MLP hidden size (was 64; deeper nets explore better)
    output_dim: int = 8           # number of parameters to generate
    n_generators: int = 1000      # population size
    mutation_rate: float = 0.1    # weight noise when cloning (adaptive: decays over gens)
    init_scale: float = 0.3       # weight init scale
    n_layers: int = 4             # network depth (was 3; 4 gives more expressive configs)
    adaptive_mutation: bool = True  # decay mutation_rate as population matures
    diversity_threshold: float = 0.01  # trigger diversity burst if param variance < this


class BatchedGenerator(nn.Module):
    """All N generators batched into a single module for GPU efficiency.

    Instead of N separate nn.Linear layers, we store weights as:
      W0: (N, in_dim, hidden_dim)
      b0: (N, hidden_dim)
      W1: (N, hidden_dim, hidden_dim)
      b1: (N, hidden_dim)
      W2: (N, hidden_dim, hidden_dim)   [4-layer: extra depth for expressivity]
      b2: (N, hidden_dim)
      W3: (N, hidden_dim, out_dim)
      b3: (N, out_dim)
      ln_gamma/beta: (N, hidden_dim)  [LayerNorm for training stability]

    Forward pass for all N generators with different noise:
      input: (N, noise_dim + context_dim) — each generator gets its own noise
      output: (N, out_dim) via batched matmul

    This is ~100x faster than looping over N separate modules on CPU.
    LayerNorm (batched per-generator) stabilizes deeper network training.
    """

    def __init__(self, cfg: GeneratorConfig, device: torch.device = None):
        super().__init__()
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        n = cfg.n_generators
        in_dim = cfg.noise_dim + cfg.context_dim
        h = cfg.hidden_dim
        out = cfg.output_dim
        scale = cfg.init_scale

        # Batched weights: (N, in, out) for each layer
        self.W0 = nn.Parameter(torch.randn(n, in_dim, h, device=self.device) * scale)
        self.b0 = nn.Parameter(torch.zeros(n, h, device=self.device))
        self.W1 = nn.Parameter(torch.randn(n, h, h, device=self.device) * scale)
        self.b1 = nn.Parameter(torch.zeros(n, h, device=self.device))
        # 4th layer (extra depth for more expressive config generation)
        self.W2 = nn.Parameter(torch.randn(n, h, h, device=self.device) * scale)
        self.b2 = nn.Parameter(torch.zeros(n, h, device=self.device))
        self.W3 = nn.Parameter(torch.randn(n, h, out, device=self.device) * scale)
        self.b3 = nn.Parameter(torch.zeros(n, out, device=self.device))

        # Per-generator LayerNorm (stabilizes deeper net training)
        self.ln_gamma = nn.Parameter(torch.ones(n, h, device=self.device))
        self.ln_beta = nn.Parameter(torch.zeros(n, h, device=self.device))

        # Per-generator fitness tracking (not a parameter)
        self.register_buffer("fitness_ema", torch.zeros(n, device=self.device))
        self.register_buffer("age", torch.zeros(n, device=self.device))

    def _layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Per-generator LayerNorm: x: (N, h) or (N, k, h) → normalized."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + 1e-6).sqrt()
        if x.dim() == 3:
            return x_norm * self.ln_gamma.unsqueeze(1) + \
                self.ln_beta.unsqueeze(1)
        return x_norm * self.ln_gamma + self.ln_beta

    def forward(self, noise: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Batched forward pass for all N generators.

        Args:
            noise: (M, noise_dim) — M = n_generators * n_per_gen; rows are
                   grouped by generator (row i -> generator i // k)
            context: (context_dim,) — shared context (current best)

        Returns:
            (M, output_dim) outputs in [0, 1] (sigmoid)
        """
        # Context shapes: (ctx,) shared — expand; (N, ctx) per-generator —
        # repeat per sample; (M, ctx) per-row — use as-is.
        m = noise.shape[0]
        n = self.cfg.n_generators
        if context.dim() == 1:
            ctx = context.unsqueeze(0).expand(m, -1)
        elif context.dim() == 2 and context.shape[0] == n and m != n:
            ctx = context.repeat_interleave(m // n, dim=0)
        else:
            ctx = context
        x = torch.cat([noise, ctx], dim=-1)        # (M, in_dim)

        if m != n:
            # Multi-sample per generator: group rows as (N, k, in_dim) so
            # per-generator weights broadcast without materializing them.
            k = m // n
            xg = x.view(n, k, -1)
            h0 = torch.tanh(self._layer_norm(
                torch.einsum("nki,nih->nkh", xg, self.W0)
                + self.b0.unsqueeze(1)))
            h1 = torch.tanh(self._layer_norm(
                torch.einsum("nki,nih->nkh", h0, self.W1)
                + self.b1.unsqueeze(1)))
            h2 = torch.tanh(self._layer_norm(
                torch.einsum("nki,nih->nkh", h1, self.W2)
                + self.b2.unsqueeze(1)))
            out = torch.einsum("nki,nih->nkh", h2, self.W3) \
                + self.b3.unsqueeze(1)
            return torch.sigmoid(out.reshape(m, -1))

        # 4-layer batched MLP with LayerNorm + tanh
        h0 = torch.bmm(x.unsqueeze(1), self.W0).squeeze(1) + self.b0
        h0 = torch.tanh(self._layer_norm(h0))
        h1 = torch.bmm(h0.unsqueeze(1), self.W1).squeeze(1) + self.b1
        h1 = torch.tanh(self._layer_norm(h1))
        h2 = torch.bmm(h1.unsqueeze(1), self.W2).squeeze(1) + self.b2
        h2 = torch.tanh(self._layer_norm(h2))
        out = torch.bmm(h2.unsqueeze(1), self.W3).squeeze(1) + self.b3
        return torch.sigmoid(out)  # (N, output_dim) in [0, 1]

    def forward_single(self, noise: torch.Tensor, context: torch.Tensor,
                       gen_idx: int) -> torch.Tensor:
        """Forward pass for a single generator (for REINFORCE updates)."""
        ctx = context.expand(1, -1)  # (1, context_dim)
        x = torch.cat([noise.unsqueeze(0), ctx], dim=-1)  # (1, in_dim)

        w0 = self.W0[gen_idx:gen_idx+1]  # (1, in, h)
        b0 = self.b0[gen_idx:gen_idx+1]
        w1 = self.W1[gen_idx:gen_idx+1]
        b1 = self.b1[gen_idx:gen_idx+1]
        w2 = self.W2[gen_idx:gen_idx+1]
        b2 = self.b2[gen_idx:gen_idx+1]
        w3 = self.W3[gen_idx:gen_idx+1]
        b3 = self.b3[gen_idx:gen_idx+1]
        ln_g = self.ln_gamma[gen_idx:gen_idx+1]
        ln_b = self.ln_beta[gen_idx:gen_idx+1]

        h0 = torch.bmm(x.unsqueeze(1), w0).squeeze(1) + b0
        h0 = torch.tanh(self._ln_single(h0, ln_g, ln_b))
        h1 = torch.bmm(h0.unsqueeze(1), w1).squeeze(1) + b1
        h1 = torch.tanh(self._ln_single(h1, ln_g, ln_b))
        h2 = torch.bmm(h1.unsqueeze(1), w2).squeeze(1) + b2
        h2 = torch.tanh(self._ln_single(h2, ln_g, ln_b))
        out = torch.bmm(h2.unsqueeze(1), w3).squeeze(1) + b3
        return torch.sigmoid(out).squeeze(0)  # (output_dim,)

    @staticmethod
    def _ln_single(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """LayerNorm for single generator (1, h)."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (var + 1e-6).sqrt() * gamma + beta

    def mutate_generator(self, idx: int, parent_idx: int, rate: float | None = None):
        """Copy parent's weights into slot idx, then add mutation noise."""
        rate = rate or self.cfg.mutation_rate
        with torch.no_grad():
            self.W0[idx] = self.W0[parent_idx] + torch.randn_like(self.W0[idx]) * rate
            self.b0[idx] = self.b0[parent_idx] + torch.randn_like(self.b0[idx]) * rate
            self.W1[idx] = self.W1[parent_idx] + torch.randn_like(self.W1[idx]) * rate
            self.b1[idx] = self.b1[parent_idx] + torch.randn_like(self.b1[idx]) * rate
            self.W2[idx] = self.W2[parent_idx] + torch.randn_like(self.W2[idx]) * rate
            self.b2[idx] = self.b2[parent_idx] + torch.randn_like(self.b2[idx]) * rate
            self.W3[idx] = self.W3[parent_idx] + torch.randn_like(self.W3[idx]) * rate
            self.b3[idx] = self.b3[parent_idx] + torch.randn_like(self.b3[idx]) * rate
            self.ln_gamma[idx] = self.ln_gamma[parent_idx]
            self.ln_beta[idx] = self.ln_beta[parent_idx]
            self.fitness_ema[idx] = self.fitness_ema[parent_idx] * 0.5
            self.age[idx] = 0

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TemplateGenerator:
    """Non-neural generator: exhaustive/random sampling of discrete params.

    Used alongside neural generators to ensure full coverage of small
    discrete search spaces (e.g., block_size in {16, 32, 64, 128}).
    No learning — just systematic or random enumeration.
    """

    def __init__(self, choices: dict[str, list], seed: int = 42):
        self.choices = choices
        self.rng = np.random.RandomState(seed)
        self._keys = list(choices.keys())
        self._indices = dict.fromkeys(self._keys, 0)
        self._exhausted = False
        self._score = 0.0
        self._fitness_ema = 0.0

    def sample(self, n: int) -> list[dict]:
        """Sample n configs. Mixes exhaustive (first) with random (rest)."""
        results = []
        for _ in range(n):
            if not self._exhausted:
                config = {}
                for k in self._keys:
                    idx = self._indices[k]
                    config[k] = self.choices[k][idx]
                results.append(config)
                for k in reversed(self._keys):
                    self._indices[k] += 1
                    if self._indices[k] < len(self.choices[k]):
                        break
                    self._indices[k] = 0
                else:
                    self._exhausted = True
            else:
                config = {k: self.rng.choice(v) for k, v in self.choices.items()}
                results.append(config)
        return results

    def reset(self):
        self._indices = dict.fromkeys(self._keys, 0)
        self._exhausted = False


class GeneratorPopulation:
    """Manages batched generators + optional template generator.

    All N neural generators are stored in a single BatchedGenerator module
    for GPU-efficient batched forward passes.
    """

    def __init__(self, cfg: GeneratorConfig, template: TemplateGenerator | None = None,
                 device: torch.device = None, compile: bool = True):
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.batched_gen = BatchedGenerator(cfg, device=self.device).to(self.device)
        self.template = template
        self.context = torch.zeros(cfg.context_dim, device=self.device)
        self.generation = 0
        self.noise_buffer = None  # stored for REINFORCE

        # torch.compile the generator forward pass for kernel fusion.
        # This fuses the 4-layer MLP + LayerNorm + sigmoid into fewer CUDA kernels,
        # reducing Python→GPU dispatch overhead by ~3-5x.
        # Use "default" mode (not "reduce-overhead") because CUDA graphs
        # break with dynamic batch sizes from adaptive population sizing.
        # Increase recompile limit since adaptive pop changes batch size.
        self._compiled = False
        # torch.compile DISABLED: corrupts inductor cache when multiple threads
        # compile simultaneously during concurrent domain execution.
        # The JSONDecodeError "Extra data" is a corrupted inductor cache file.
        # Raw forward pass is fast enough for our generator sizes.

        total_params = self.batched_gen.n_params()
        mem_mb = total_params * 4 / 1e6
        compile_tag = " [compiled]" if self._compiled else ""
        print(f"[Generators] {cfg.n_generators} batched generators on {self.device}, "
              f"{total_params/1e6:.1f}M total params, {mem_mb:.1f} MB{compile_tag}")

    def generate(self, n_per_gen: int = 1,
                 contexts: torch.Tensor | None = None) -> list[dict[str, Any]]:
        """Generate candidates from all generators in one batched forward pass.

        Args:
            n_per_gen: samples per generator (row i -> generator i // n_per_gen)
            contexts: optional (n_generators, context_dim) per-generator
                      conditioning — archive-sampled parents per generator
                      instead of one shared elite (emitter-style QD).
        """
        n = self.cfg.n_generators
        total = n * n_per_gen

        # Single batched noise tensor for all generators
        noise = torch.randn(total, self.cfg.noise_dim, device=self.device)
        # Per-generator contexts if provided, else shared context
        ctx = contexts if contexts is not None else self.context

        # One batched forward pass — all generators at once
        with torch.no_grad():
            outputs = self.batched_gen(noise, ctx)  # (total, output_dim)

        candidates = []
        for i in range(total):
            gen_idx = i // n_per_gen
            candidates.append({
                "params": outputs[i].detach(),
                "generator_idx": gen_idx,
                "noise": noise[i].detach(),
            })

        # Store noise for REINFORCE
        self.noise_buffer = noise.detach()

        # Add template samples if present
        if self.template and not self.template._exhausted:
            template_configs = self.template.sample(200)
            for tc in template_configs:
                candidates.append({"params": None, "template_config": tc,
                                   "generator_idx": -1})

        return candidates

    def evolve(self, scores: list[float], generator_indices: list[int],
               novelties: list[float] | None = None,
               explore_frac: float = 0.2,
               extra_explore: set[int] | None = None) -> list[int]:
        """Evolve population: kill worst, clone+mutate best, update fitness.
        Returns list of generator indices that were mutated (for optimizer reset).

        Adaptive features (self-improving):
        - mutation_rate decays over generations (exploit more as we converge)
        - diversity burst triggers when population variance drops below threshold
          (anti-plateau: injects noise to escape local optima)
        - Every 10 generations: standard diversity burst on 30% of mid-tier
        - explore emitters (first explore_frac of generators) are credited on
          proposal NOVELTY not score (MEMES-style explore emitters — a
          generator exploring a low-scoring niche isn't culled for it)
        """
        self.generation += 1

        # Update fitness EMA per generator. Explore emitters accumulate
        # (rank-shaped) novelty; everyone else accumulates (rank-shaped) score.
        n_explore = int(len(self.batched_gen.fitness_ema) * explore_frac)
        extra = extra_explore or set()
        nov_iter = novelties if novelties is not None else scores
        gen_scores: dict[int, list[float]] = {}
        for score, nov, idx in zip(scores, nov_iter, generator_indices):
            if idx >= 0:
                # Explore emitters (first explore_frac) and explicit
                # extra_explore slots (wild gens) accrue novelty/local-
                # quality fitness — never raw score.
                val = nov if (idx < n_explore or idx in extra) else score
                gen_scores.setdefault(idx, []).append(val)

        with torch.no_grad():
            for idx, sc_list in gen_scores.items():
                avg = sum(sc_list) / len(sc_list)
                self.batched_gen.fitness_ema[idx] = \
                    0.7 * self.batched_gen.fitness_ema[idx] + 0.3 * avg
                self.batched_gen.age[idx] += 1

        # Sort by fitness
        fitness = self.batched_gen.fitness_ema.cpu().numpy()
        ranked = np.argsort(fitness)[::-1]  # descending
        n_kill = len(ranked) // 5
        n_clone = len(ranked) // 5

        bottom_indices = ranked[-n_kill:]
        top_indices = ranked[:n_clone]

        # Adaptive mutation rate: decay over generations (exploit more as we mature)
        if self.cfg.adaptive_mutation:
            current_rate = self.cfg.mutation_rate * max(
                0.02, 1.0 - self.generation * 0.01)
        else:
            current_rate = self.cfg.mutation_rate

        mutated = []
        for i, bad_idx in enumerate(bottom_indices):
            parent_idx = top_indices[i % n_clone]
            self.batched_gen.mutate_generator(int(bad_idx), int(parent_idx),
                                               rate=current_rate)
            mutated.append(int(bad_idx))

        # Diversity check: if population variance is too low, inject noise
        # to prevent overfitting/plateau (anti-convergence mechanism)
        with torch.no_grad():
            w_var = self.batched_gen.W0.var(dim=0).mean().item()
        if w_var < self.cfg.diversity_threshold:
            n_burst = len(ranked) // 2  # aggressive: 50% get noise
            mid_indices = ranked[n_clone:n_clone + n_burst]
            burst_rate = max(0.05, current_rate * 2)  # double the mutation
            with torch.no_grad():
                for idx in mid_indices:
                    self.batched_gen.W0[idx] += torch.randn_like(
                        self.batched_gen.W0[idx]) * burst_rate
                    self.batched_gen.W1[idx] += torch.randn_like(
                        self.batched_gen.W1[idx]) * burst_rate
                    self.batched_gen.W2[idx] += torch.randn_like(
                        self.batched_gen.W2[idx]) * burst_rate
                    self.batched_gen.W3[idx] += torch.randn_like(
                        self.batched_gen.W3[idx]) * burst_rate
                    mutated.append(int(idx))

        # Standard diversity burst every 10 generations
        elif self.generation % 10 == 0:
            n_burst = len(ranked) // 3
            mid_indices = ranked[n_clone:n_clone + n_burst]
            with torch.no_grad():
                for idx in mid_indices:
                    self.batched_gen.W0[idx] += torch.randn_like(
                        self.batched_gen.W0[idx]) * 0.15
                    self.batched_gen.W1[idx] += torch.randn_like(
                        self.batched_gen.W1[idx]) * 0.15
                    self.batched_gen.W2[idx] += torch.randn_like(
                        self.batched_gen.W2[idx]) * 0.15
                    self.batched_gen.W3[idx] += torch.randn_like(
                        self.batched_gen.W3[idx]) * 0.15
                    mutated.append(int(idx))

        return mutated

    def update_context(self, best_params: torch.Tensor):
        """Update context vector with current best config."""
        ctx = torch.zeros(self.cfg.context_dim, device=self.device)
        n = min(best_params.shape[0], self.cfg.context_dim)
        ctx[:n] = best_params[:n].to(self.device)
        self.context = ctx
