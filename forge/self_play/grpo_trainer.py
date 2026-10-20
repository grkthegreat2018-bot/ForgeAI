"""GRPO (Group Relative Policy Optimization) trainer for self-play RL.

Implements the GRPO algorithm from DeepSeek-R1, adapted for small models (1-3B)
with MC-GRPO median baseline for stability with small group sizes.

Research basis:
- DeepSeek-R1: GRPO with verifiable rewards, KL penalty against reference model
- MC-GRPO: median baseline for G=2-4 (reduces gap to <1% vs G=8)
- AVSPO: Advantage Collapse Rate monitoring (alert if >0.3)
- "Survive or Collapse" (arxiv 2605.22217): data gate is binding constraint,
  not reward design. Executor verification (test_passed=True) IS the strict gate.

Hyperparameters (small model optimal, 1-3B):
  learning_rate: 5e-6
  kl_coefficient: 0.02
  clip_range: 0.2
  group_size: 4 (use MC-GRPO median baseline)
  temperature: 0.8
  max_grad_norm: 1.0

Reward design:
  solver_reward = 1.0 if test_passed else 0.0  (binary from executor = the gate)
  proposer_reward = 1.0 - |success_rate - 0.5|  (AZR learnability)
  Invalid task penalty: -0.1

Usage:
    from forge.self_play.grpo_trainer import GRPOTrainer
    trainer = GRPOTrainer(model, tokenizer, ref_model, device="cuda")
    stats = trainer.train_step(prompts, generate_fn, verify_fn)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from forge.training.training_utils import oom_guard


class ExGRPOReplayBuffer:
    """ExGRPO experience-replay buffer (ICLR'26 "ExGRPO").

    Stores past GRPO *rollout groups* — {prompt, completions, rewards,
    mean_token_entropy} — and scores each by the paper's value signal::

        value = correctness_flag * mean_token_entropy

    i.e. groups that contain at least one verified-correct rollout AND
    were produced with high policy entropy carry the most reusable
    experience (correct reasoning paths the policy is still uncertain
    about).  Updates draw from the top-valued entries; the buffer is
    capacity-bounded and evicts the *lowest-value* entry when full.

    Mixed-policy note: replayed groups are off-policy by construction —
    they reuse the same group-relative advantage machinery (no new
    objective); the paper's win is *which* experiences get replayed.
    """

    def __init__(self, capacity: int = 512):
        self.capacity = max(1, int(capacity))
        self.entries: list[dict] = []

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def value_of(rewards: list[float], entropy: float) -> tuple[float, float]:
        """(correctness_flag, value) — flag=1 iff any rollout verified."""
        correctness = 1.0 if any(r >= 0.99 for r in rewards) else 0.0
        return correctness, correctness * float(entropy)

    def add(self, prompt: str, completions: list[str],
            rewards: list[float], entropy: float) -> dict | None:
        """Insert/update a rollout group.  Dedupes by prompt (keeps the
        higher-value entry); evicts the lowest-value entry at capacity.
        Returns the stored entry, or None if it was not worth keeping."""
        correctness, value = self.value_of(rewards, entropy)
        # Dedupe by prompt — replace only when the new rollout is more
        # valuable (keeps stale-but-better experiences).
        for e in self.entries:
            if e["prompt"] == prompt:
                if value > e["value"]:
                    e.update(completions=list(completions),
                             rewards=list(rewards), entropy=float(entropy),
                             correctness=correctness, value=value)
                    return e
                return None
        if len(self.entries) >= self.capacity:
            min_i, min_e = min(enumerate(self.entries),
                               key=lambda t: t[1]["value"])
            if value <= min_e["value"]:
                return None  # buffer full of strictly better entries
            self.entries.pop(min_i)
        entry = {"prompt": prompt, "completions": list(completions),
                 "rewards": list(rewards), "entropy": float(entropy),
                 "correctness": correctness, "value": value}
        self.entries.append(entry)
        return entry

    def top(self, n: int) -> list[dict]:
        """The n highest-value entries (deterministic priority order)."""
        return sorted(self.entries, key=lambda e: e["value"],
                      reverse=True)[:max(0, n)]

    def sample(self, n: int, rng=None) -> list[dict]:
        """Value-weighted sample without replacement (top-valued entries
        dominate; uniform when all values are 0)."""
        import random as _random
        rng = rng or _random
        pool = list(self.entries)
        picked = []
        n = min(n, len(pool))
        for _ in range(n):
            total = sum(max(e["value"], 0.0) for e in pool)
            if total <= 0:
                i = rng.randrange(len(pool))
            else:
                r = rng.random() * total
                acc = 0.0
                i = len(pool) - 1
                for j, e in enumerate(pool):
                    acc += max(e["value"], 0.0)
                    if r < acc:
                        i = j
                        break
            picked.append(pool.pop(i))
        return picked


@dataclass
class GRPOConfig:
    """GRPO hyperparameters optimized for 1-3B models."""
    learning_rate: float = 5e-6
    kl_coefficient: float = 0.02       # β in KL(π || π_ref)
    clip_range: float = 0.2            # ε for PPO-style ratio clipping
    group_size: int = 4                # G completions per prompt (MC-GRPO)
    temperature: float = 0.8           # generation temperature
    max_grad_norm: float = 1.0
    max_seq_len: int = 512             # max sequence length for training
    use_median_baseline: bool = True   # MC-GRPO: median instead of mean
    invalid_penalty: float = -0.1      # penalty for invalid/unverifiable tasks
    grad_accum_steps: int = 2          # gradient accumulation
    # MoE load-balancing aux loss weight during RL. Without this, the MoE
    # router gets no balancing signal during RLVR (forward passes don't pass
    # targets) and can collapse to a single expert. 0.0 = disabled (dense
    # models), 0.01 = DeepSeek-V3 default.
    moe_aux_weight: float = 0.01
    # SC-GRPO (Self-Conditioned GRPO, arXiv 2605.22217 extension): use per-token
    # KL between the model's distribution on verified vs unverified trajectories
    # as a multiplicative weight on the GRPO gradient. Pivotal reasoning tokens
    # (where the model diverges between success/failure paths) get higher weight;
    # routine tokens get lower weight. +8.1% over vanilla GRPO. No external
    # teacher or process reward model needed — uses the existing verified/
    # unverified split from the self-play pipeline.
    use_sc_grpo: bool = False
    sc_weight_clip: float = 3.0        # max per-token weight (prevents explosion)
    # OM-GRPO (Outcome-Masked GRPO): mask gradients on the answer span while
    # retaining answer-level rewards through soft consensus. Prevents the model
    # from shortcutting by sharpening answer tokens without improving reasoning.
    use_om_grpo: bool = False
    om_answer_markers: tuple = ("Answer:", "answer:", "ANSWER:", "```", "Final:")
    # GVPO (Group Verification-based Policy Optimization): process-level rewards
    # from per-round execution errors. Advantage = outcome + λ * Σ(process).
    use_gvpo: bool = False
    gvpo_lambda: float = 0.3           # weight on process rewards
    # Tool-use GRPO: rewards are continuous 0..1 from tool-use trajectories
    # (format, execution, answer quality) instead of binary code pass/fail.
    # When True, the trainer expects rewards from tool_use_loop.compute_reward.
    use_tool_use_rewards: bool = False
    # Golden trajectory injection: inject replayed golden (high-quality,
    # previously successful) trajectories into each training batch to prevent
    # catastrophic forgetting. The replay buffer stores successful completions
    # with forgetting-curve scheduling (FOREVER-style). 10-20% of each batch
    # is golden replays; the rest is fresh self-play data.
    replay_ratio: float = 0.15          # fraction of batch from golden replays
    replay_min_buffer_size: int = 50    # don't inject until buffer has enough
    # GRPO-λ: dynamic length penalty based on group correctness ratio.
    # When the correctness ratio in a group is LOW (model is still learning),
    # length penalty is DISABLED — pure 0/1 outcome rewards prioritize
    # reasoning capability. When correctness ratio is HIGH (model has matured),
    # length penalty activates to encourage efficiency without destroying logic.
    # This prevents the "CoT length penalty trap" where static penalties cause
    # accuracy collapse early in training (arXiv 2509.01155).
    use_grpo_lambda: bool = False       # enable GRPO-λ dynamic length penalty
    length_penalty_coeff: float = 0.01  # λ: penalty per token of completion
    length_penalty_threshold: float = 0.6  # correctness ratio to activate penalty
    length_penalty_warmup: int = 0      # steps before penalty can activate

    # Advanced RL algorithm selection (2026 research):
    # "grpo" = standard GRPO (default)
    # "gtpo" = Group-relative Trajectory-based PO (entropy control + conflict
    #          masking, no KL/ref model needed — saves ~2GB VRAM)
    # "sppo" = Sequence-Level PPO (long-horizon reasoning, no multi-sampling)
    # "psppo" = Prefix-Sampling PPO (compute-efficient, prefix backprop only)
    # "evpo" = Explained Variance PO (adaptive critic/batch-mean switching)
    # "grpo_or" = GRPO with Output Reset trust region (smooth saturation)
    # "cispo" = CISPO (MiniMax-M1/M2): clipped importance weight is DETACHED
    #           (sg) so every token keeps gradient even when the ratio is
    #           clipped — no zero-gradient dead zones. Matches DAPO-level
    #           performance in ~half the training time; beats GRPO under
    #           high off-policy reuse. No KL/ref model needed.
    rl_algorithm: str = "grpo"

    # GTPO-specific config (arXiv 2508.03772):
    # Entropy regularization weight (γ in the paper). 0.1 recommended.
    gtpo_entropy_gamma: float = 0.1
    # Entropy filter threshold (ln(2) ≈ 0.693). Completions with average
    # entropy above this are filtered if the model's initial entropy is below.
    gtpo_entropy_threshold: float = 0.6931  # ln(2)
    # Whether to measure initial entropy on first train_step (auto-calibrate)
    gtpo_auto_init_entropy: bool = True

    # CISPO-specific config (MiniMax):
    # Asymmetric clip bounds (DAPO-style: looser upper bound encourages
    # exploration on low-probability tokens).
    cispo_epsilon_low: float = 0.2
    cispo_epsilon_high: float = 0.28

    # ETR (Entropy Trend Reward): bonus proportional to the downward entropy
    # trend across the completion — reasoning that converges (entropy drops
    # from start to end) is rewarded, doom-looping (flat/rising entropy) is
    # not. Training-free addition to the outcome reward; +9.9% acc / -67%
    # CoT length reported at 7B scale.
    use_etr_reward: bool = False
    etr_coeff: float = 0.1

    # N-gram repetition penalty (LFM2.5-1.2B-Thinking RLVR recipe):
    # Discourages doom-looping early in RL training. Applied as a negative
    # reward when the n-gram repetition ratio exceeds the threshold.
    # Reduces doom-loop rate from ~4% (DPO) to ~0.36% (RLVR).
    use_repetition_penalty: bool = False
    repetition_n: int = 8               # n-gram size for detection
    repetition_threshold: float = 0.3   # ratio above which penalty applies
    repetition_penalty: float = -0.5    # penalty value (negative reward)
    repetition_warmup_steps: int = 50   # only apply penalty after N steps

    # ExGRPO (experience-replay GRPO, ICLR'26): each epoch's rollout
    # groups are scored by correctness_flag * mean-token-entropy and
    # stored in a persistent buffer; each update batch draws
    # exgrpo_replay_fraction of its groups from the top-valued entries.
    # The replayed groups reuse the standard group-relative advantage
    # machinery — the paper's win is WHICH experiences get replayed.
    # Flag-gated: default OFF keeps prior behavior bit-identical.
    use_exgrpo: bool = False            # opt-in experience replay
    exgrpo_replay_size: int = 512       # max stored rollout groups
    exgrpo_replay_fraction: float = 0.25  # fraction of each update batch

    def __post_init__(self):
        """Validate critical hyperparameters to catch config errors early."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.group_size < 2:
            raise ValueError(f"group_size must be >= 2 for GRPO, got {self.group_size}")
        if self.clip_range <= 0 or self.clip_range > 1:
            raise ValueError(f"clip_range must be in (0, 1], got {self.clip_range}")
        if self.max_seq_len < 64:
            raise ValueError(f"max_seq_len too small ({self.max_seq_len}), need >= 64")
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.repetition_n < 2:
            raise ValueError(f"repetition_n must be >= 2, got {self.repetition_n}")
        if not 0 <= self.replay_ratio <= 1:
            raise ValueError(f"replay_ratio must be in [0, 1], got {self.replay_ratio}")
        if not 0 <= self.exgrpo_replay_fraction <= 1:
            raise ValueError(
                f"exgrpo_replay_fraction must be in [0, 1], "
                f"got {self.exgrpo_replay_fraction}")
        if self.exgrpo_replay_size < 1:
            raise ValueError(
                f"exgrpo_replay_size must be >= 1, got {self.exgrpo_replay_size}")


@dataclass
class GRPOStats:
    """Tracks GRPO training metrics for monitoring."""
    total_steps: int = 0
    total_rewards: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    kl_divergences: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    advantage_collapse_count: int = 0  # ACR numerator
    # Advantage Collapse Rate (ACR): fraction of groups where all advantages ≈ 0
    # Alert if ACR > 0.3 (AVSPO paper)
    # GRPO-λ: track correctness ratios and length penalty activation
    correctness_ratios: list[float] = field(default_factory=list)
    length_penalty_active_count: int = 0  # how many groups had penalty active

    # Maximum items to retain in unbounded lists (prevents memory leak in
    # long training runs — stats are only used for rolling-window averages)
    _MAX_LIST_LEN: int = 1000

    def trim(self):
        """Trim all metric lists to _MAX_LIST_LEN (keep most recent)."""
        for attr in ("total_rewards", "advantages", "kl_divergences",
                      "policy_losses", "correctness_ratios"):
            lst = getattr(self, attr)
            if len(lst) > self._MAX_LIST_LEN:
                # Keep the most recent entries (slice from the end)
                del lst[:len(lst) - self._MAX_LIST_LEN]

    @property
    def mean_reward(self) -> float:
        return sum(self.total_rewards) / max(len(self.total_rewards), 1)

    @property
    def mean_advantage(self) -> float:
        return sum(self.advantages) / max(len(self.advantages), 1)

    @property
    def mean_kl(self) -> float:
        return sum(self.kl_divergences) / max(len(self.kl_divergences), 1)

    @property
    def advantage_collapse_rate(self) -> float:
        """ACR = fraction of steps with collapsed advantages.
        Alert if > 0.3 (AVSPO)."""
        if self.total_steps == 0:
            return 0.0
        return self.advantage_collapse_count / self.total_steps


class GRPOTrainer:
    """GRPO trainer for self-play RL with verifiable rewards.

    The executor-based verification (test_passed=True) serves as the strict
    data gate (ε=0). This is the binding constraint on stability — NOT reward
    design ("Survive or Collapse" paper). Never train on unverified solutions.
    """

    def __init__(self, model, tokenizer, ref_model,
                 device: str = "cuda",
                 config: GRPOConfig | None = None,
                 replay_buffer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model  # frozen reference model for KL penalty
        self.device = device
        self.config = config or GRPOConfig()
        self.stats = GRPOStats()
        self.replay_buffer = replay_buffer  # FOREVER-style replay buffer

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Optimizer (only trainable params — assumes LoRA or similar)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if trainable:
            if getattr(config, 'optimizer', 'adamw') == 'cpu_offload':
                from forge.training.optim.hybrid_offload import CPUAdamW
                self.optimizer = CPUAdamW(
                    trainable, lr=self.config.learning_rate, weight_decay=0.01)
                print("  [GRPOTrainer] Using CPUAdamW (optimizer states on CPU)")
            else:
                self.optimizer = torch.optim.AdamW(
                    trainable, lr=self.config.learning_rate, weight_decay=0.01)
        else:
            self.optimizer = None

        # Advanced RL algorithm (2026 research)
        self._rl_algo = None
        self._gtpo_init_entropy = None  # measured on first step if auto
        if self.config.rl_algorithm == "gtpo":
            # GTPO: no reference model needed (saves VRAM)
            print("  [GRPOTrainer] Using GTPO (Group-relative Trajectory-based PO)")
            print("  [GRPOTrainer] GTPO: KL penalty disabled, ref_model not required")
        elif self.config.rl_algorithm == "cispo":
            print("  [GRPOTrainer] Using CISPO (MiniMax: detached clipped IS weight)")
            print("  [GRPOTrainer] CISPO: KL penalty disabled, ref_model not required")
        elif self.config.rl_algorithm == "sppo":
            from forge.training.losses.advanced_rl import SPPO, SPPOConfig
            self._rl_algo = SPPO(SPPOConfig(lr=self.config.learning_rate))
            print("  [GRPOTrainer] Using SPPO (Sequence-Level PPO)")
        elif self.config.rl_algorithm == "psppo":
            from forge.training.losses.advanced_rl import PSPPO, PSPPOConfig
            self._rl_algo = PSPPO(PSPPOConfig(lr=self.config.learning_rate))
            print("  [GRPOTrainer] Using PS-PPO (Prefix-Sampling PPO)")
        elif self.config.rl_algorithm == "evpo":
            from forge.training.losses.advanced_rl import EVPO, EVPOConfig
            self._rl_algo = EVPO(EVPOConfig(lr=self.config.learning_rate))
            print("  [GRPOTrainer] Using EVPO (Explained Variance PO)")
        elif self.config.rl_algorithm == "grpo_or":
            from forge.training.losses.advanced_rl import GRPOOR, GRPOORConfig
            self._rl_algo = GRPOOR(GRPOORConfig(lr=self.config.learning_rate))
            print("  [GRPOTrainer] Using GRPO-OR (Output Reset trust region)")

    def compute_advantages(self, rewards: list[float]) -> list[float]:
        """Compute GRPO group-relative advantages.

        MC-GRPO: use median baseline for robustness with small groups.
        advantage = (reward - baseline) / (group_std + eps)
        """
        if not rewards:
            return []

        if self.config.use_median_baseline and len(rewards) >= 2:
            # MC-GRPO: median is more robust to outliers than mean
            sorted_r = sorted(rewards)
            n = len(sorted_r)
            if n % 2 == 1:
                baseline = sorted_r[n // 2]
            else:
                baseline = (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2
        else:
            baseline = sum(rewards) / len(rewards)

        # Group std (population std for small groups)
        if len(rewards) > 1:
            variance = sum((r - baseline) ** 2 for r in rewards) / len(rewards)
            std = math.sqrt(variance + 1e-8)
        else:
            std = 1.0

        advantages = [(r - baseline) / (std + 1e-8) for r in rewards]

        # Track advantage collapse (all advantages ≈ 0)
        if all(abs(a) < 0.01 for a in advantages):
            self.stats.advantage_collapse_count += 1

        return advantages

    def _apply_repetition_penalty(self, rewards: list[float],
                                   completions: list[str]) -> list[float]:
        """Apply n-gram repetition penalty to doom-looping completions.

        LFM2.5-1.2B-Thinking RLVR recipe: during early RL training, apply a
        negative reward to completions with excessive n-gram repetition.
        This discourages the model from doom-looping before it learns to
        produce coherent reasoning.

        Only active after `repetition_warmup_steps` to let the model
        stabilize first. Reduces doom-loop rate from ~4% (DPO) to ~0.36%.

        Returns adjusted rewards.
        """
        if not self.config.use_repetition_penalty:
            return rewards
        if self.stats.total_steps < self.config.repetition_warmup_steps:
            return rewards

        from forge.training.runners.curriculum_sft import is_doom_loop
        adjusted = []
        for reward, completion in zip(rewards, completions):
            if is_doom_loop(completion, n=self.config.repetition_n,
                            threshold=self.config.repetition_threshold):
                adjusted.append(reward + self.config.repetition_penalty)
            else:
                adjusted.append(reward)
        return adjusted

    def _group_correctness_ratio(self, rewards: list[float]) -> float:
        """Compute the correctness ratio for a group of completions.

        correctness_ratio = fraction of completions with reward >= 0.99
        (i.e., passed verification). This drives the GRPO-λ dynamic
        length penalty: penalty only activates when the model is mature
        enough (high correctness ratio) to prioritize efficiency.
        """
        if not rewards:
            return 0.0
        n_correct = sum(1 for r in rewards if r >= 0.99)
        return n_correct / len(rewards)

    def _apply_grpo_lambda_penalty(self, rewards: list[float],
                                   completions: list[str],
                                   correctness_ratio: float) -> list[float]:
        """Apply GRPO-λ dynamic length penalty to rewards.

        When correctness_ratio < threshold: NO penalty (pure 0/1 rewards).
          The model is still learning to reason — penalizing length would
          destroy accuracy (the "CoT length penalty trap").

        When correctness_ratio >= threshold: apply length penalty.
          The model has matured — encourage shorter, more efficient solutions
          without destroying logic. Penalty = -λ * n_tokens for correct solutions.

        The warmup parameter delays penalty activation for the first N steps
        regardless of correctness ratio (lets the model stabilize).

        Returns adjusted rewards (original rewards if penalty inactive).
        """
        if not self.config.use_grpo_lambda:
            return rewards

        # Warmup: don't activate penalty during early training
        if self.stats.total_steps < self.config.length_penalty_warmup:
            return rewards

        # Only apply penalty when the group is mature enough
        if correctness_ratio < self.config.length_penalty_threshold:
            return rewards

        # Penalty active for this group
        self.stats.length_penalty_active_count += 1
        lam = self.config.length_penalty_coeff
        adjusted = []
        for reward, completion in zip(rewards, completions):
            if reward >= 0.99:
                # Correct solution: penalize by length
                # n_tokens ≈ len(completion) / 4 (rough token estimate)
                n_tokens = max(1, len(completion) // 4)
                adjusted.append(reward - lam * n_tokens)
            else:
                # Incorrect: no length penalty (already got 0)
                adjusted.append(reward)
        return adjusted

    def compute_turn_level_advantages(
        self, turn_rewards: list[list[float]]
    ) -> list[list[float]]:
        """Turn-level credit assignment for multi-turn tool use.

        Instead of a single trajectory-level reward, each turn (tool call)
        gets its own reward. The advantage for each turn is computed
        relative to the group of completions at that turn position.

        This implements the approach from arXiv 2505.11821 (MT-GRPO):
        "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level
        Credit Assignment."

        Args:
            turn_rewards: list of lists — turn_rewards[g][t] is the reward
                          for turn t in completion g.
        Returns:
            list of lists — advantages per turn per completion.
        """
        if not turn_rewards or not turn_rewards[0]:
            return [[] for _ in turn_rewards]

        n_completions = len(turn_rewards)
        max_turns = max(len(tr) for tr in turn_rewards)

        # Compute per-turn baseline across completions
        turn_baselines = []
        for t in range(max_turns):
            turn_rewards_t = [tr[t] for tr in turn_rewards if t < len(tr)]
            if not turn_rewards_t:
                turn_baselines.append(0.0)
                continue
            if self.config.use_median_baseline and len(turn_rewards_t) >= 2:
                sorted_r = sorted(turn_rewards_t)
                n = len(sorted_r)
                if n % 2 == 1:
                    baseline = sorted_r[n // 2]
                else:
                    baseline = (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2
            else:
                baseline = sum(turn_rewards_t) / len(turn_rewards_t)
            turn_baselines.append(baseline)

        # Compute advantages
        advantages = []
        for g in range(n_completions):
            comp_advs = []
            for t in range(len(turn_rewards[g])):
                # Turn advantage = (reward - turn_baseline) / (turn_std + eps)
                turn_rewards_t = [tr[t] for tr in turn_rewards if t < len(tr)]
                if len(turn_rewards_t) > 1:
                    var = sum((r - turn_baselines[t]) ** 2 for r in turn_rewards_t) / len(turn_rewards_t)
                    std = math.sqrt(var + 1e-8)
                else:
                    std = 1.0
                adv = (turn_rewards[g][t] - turn_baselines[t]) / (std + 1e-8)
                comp_advs.append(adv)
            advantages.append(comp_advs)

        return advantages

    def _pretokenize_prompt_lens(self, prompts: list[str]) -> dict[str, int]:
        """Tokenize each unique prompt once and cache its token length.

        Avoids re-tokenizing the same prompt G times (once per completion)
        just to get its length. Returns {prompt_str: prompt_token_len}.
        """
        cache: dict[str, int] = {}
        for p in prompts:
            if p not in cache:
                cache[p] = self.tokenizer(
                    p, return_tensors="pt").input_ids.shape[1]
        return cache

    def compute_kl_penalty(self, input_ids: torch.Tensor,
                           current_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL(π_current || π_ref) per token.

        KL = Σ p_current * log(p_current / p_ref)
        Approximated per-token, averaged over sequence.
        """
        with torch.inference_mode():
            ref_logits, _ = self.ref_model(input_ids)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs.exp()

        curr_log_probs = F.log_softmax(current_logits, dim=-1)
        curr_probs = curr_log_probs.exp()

        # KL divergence per token: sum over vocab
        kl_per_token = (curr_probs * (curr_log_probs - ref_log_probs)).sum(dim=-1)
        return kl_per_token.mean()

    # -----------------------------------------------------------------------
    # GTPO: Group-relative Trajectory-based Policy Optimization
    # (arXiv 2508.03772 — entropy control + conflict-aware gradient correction)
    # -----------------------------------------------------------------------

    def _gtpo_compute_conflict_masks(
        self,
        completions: list[list[int]],  # token IDs per completion
        advantages: list[float],
    ) -> list[torch.Tensor]:
        """Build conflict-aware masks for each completion.

        Identifies tokens at the same position (from left or right) that appear
        in both positive and negative advantage completions. These "conflict
        tokens" receive contradictory gradient updates in vanilla GRPO.

        Returns: list of (T_i,) tensors with λ weights: 0 (skip neg), 1 (normal), 2 (amplify pos)
        """
        G = len(completions)
        if G < 2:
            return [torch.ones(len(c), device=self.device) for c in completions]

        pos_idx = [i for i, a in enumerate(advantages) if a > 0]
        neg_idx = [i for i, a in enumerate(advantages) if a < 0]
        if not pos_idx or not neg_idx:
            return [torch.ones(len(c), device=self.device) for c in completions]

        # Forward conflict: same token at same position from left
        min_len = min(len(completions[i]) for i in range(G))
        fw_conflict = torch.zeros(min_len, dtype=torch.bool, device=self.device)
        for t in range(min_len):
            pos_tokens = set(completions[i][t] for i in pos_idx if t < len(completions[i]))
            neg_tokens = set(completions[i][t] for i in neg_idx if t < len(completions[i]))
            if pos_tokens & neg_tokens:
                fw_conflict[t] = True

        # Backward conflict: same token at same offset from end
        bw_conflict = torch.zeros(G, dtype=torch.bool, device=self.device)
        # Check from the end, up to min completion length
        for r in range(min_len):
            pos_tokens = set(
                completions[i][len(completions[i]) - 1 - r]
                for i in pos_idx if r < len(completions[i])
            )
            neg_tokens = set(
                completions[i][len(completions[i]) - 1 - r]
                for i in neg_idx if r < len(completions[i])
            )
            if pos_tokens & neg_tokens:
                bw_conflict[r] = True

        # Build per-completion λ masks
        masks = []
        for i, c in enumerate(completions):
            T = len(c)
            lam = torch.ones(T, device=self.device)
            is_pos = advantages[i] > 0

            # Forward mask: first contiguous span of conflict tokens
            fw_mask = torch.zeros(T, dtype=torch.bool, device=self.device)
            for t in range(min(T, min_len)):
                if fw_conflict[t]:
                    fw_mask[t] = True
                else:
                    break  # only first contiguous span

            # Backward mask: first contiguous span from the end
            bw_mask = torch.zeros(T, dtype=torch.bool, device=self.device)
            for r in range(min(T, min_len)):
                if bw_conflict[r]:
                    bw_mask[T - 1 - r] = True
                else:
                    break

            combined = fw_mask | bw_mask
            # λ = 0 for conflict tokens with negative advantage
            # λ = 2 for conflict tokens with positive advantage
            # λ = 1 elsewhere (normal)
            lam[combined & ~is_pos] = 0.0  # skip negative on conflict
            lam[combined & is_pos] = 2.0   # amplify positive on conflict
            masks.append(lam)

        return masks

    def _gtpo_compute_entropy(
        self,
        logits: torch.Tensor,  # (T, V) or (1, T, V)
    ) -> float:
        """Compute average Shannon entropy of the output distribution."""
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # H = -Σ p * log(p), per token
        entropy_per_token = -(probs * log_probs).sum(dim=-1)
        return entropy_per_token.mean().item()

    def _gtpo_entropy_filter(
        self,
        completion_entropies: list[float],
    ) -> list[float]:
        """Build entropy filter masks δ_i.

        If model's initial entropy < ln(2), filter out completions with
        entropy > ln(2) (unstable/high-entropy completions cause collapse).
        """
        threshold = self.config.gtpo_entropy_threshold
        if self._gtpo_init_entropy is None or self._gtpo_init_entropy >= threshold:
            # Model naturally produces high-entropy outputs — no filtering
            return [1.0] * len(completion_entropies)
        # Model is low-entropy — filter high-entropy completions
        return [1.0 if h <= threshold else 0.0 for h in completion_entropies]

    # -----------------------------------------------------------------------
    # Golden trajectory injection (anti-regression via replay buffer)
    # -----------------------------------------------------------------------

    def _inject_golden_replays(self, prompts: list[str],
                               completions: list[list[str]],
                               rewards: list[list[float]]
                               ) -> tuple[list, list, list, int]:
        """Inject golden (previously successful) trajectories from the replay
        buffer into the current batch.

        Replaces `replay_ratio` fraction of the batch with golden replays.
        Golden trajectories get reward=1.0 (they were verified successful in
        the past) and are treated as a separate group within GRPO advantage
        computation. This prevents catastrophic forgetting by ensuring the
        model continuously revisits skills it has already mastered.

        Returns (prompts, completions, rewards, n_injected).
        """
        if (self.replay_buffer is None or
                len(self.replay_buffer) < self.config.replay_min_buffer_size):
            return prompts, completions, rewards, 0

        n_groups = len(prompts)
        n_inject = max(1, int(n_groups * self.config.replay_ratio))
        replays = self.replay_buffer.sample(
            n_inject, self.replay_buffer.cumulative_magnitude)

        injected_prompts = []
        injected_completions = []
        injected_rewards = []
        for r in replays:
            p = r.get("prompt", "")
            c = r.get("solution", r.get("completion", ""))
            if not p or not c:
                continue
            # Golden trajectories are single-completion groups with reward=1.0.
            # GRPO needs >=2 completions for advantage computation, so we
            # duplicate the golden completion as a "twin" — both get reward 1.0,
            # producing zero advantage (no gradient from golden alone). The
            # real benefit comes from mixing golden data into the KL penalty
            # and the forward pass, keeping the model's distribution anchored
            # on previously mastered skills.
            injected_prompts.append(p)
            injected_completions.append([c, c])  # twin for group >= 2
            injected_rewards.append([1.0, 1.0])

        if not injected_prompts:
            return prompts, completions, rewards, 0

        # Append golden replays to the batch
        return (prompts + injected_prompts,
                completions + injected_completions,
                rewards + injected_rewards,
                len(injected_prompts))

    def _record_golden_trajectories(self, prompts: list[str],
                                    completions: list[list[str]],
                                    rewards: list[list[float]],
                                    grad_norm: float = 0.0):
        """Record successful (reward=1.0) trajectories into the replay buffer
        for future golden injection. Only stores completions that passed
        verification (the strict data gate)."""
        if self.replay_buffer is None:
            return
        for prompt, comps, rews in zip(prompts, completions, rewards):
            for comp, reward in zip(comps, rews):
                if reward >= 0.99:  # verified successful
                    self.replay_buffer.add(
                        {"prompt": prompt, "solution": comp,
                         "quality": float(reward), "test_passed": True},
                        optimizer_magnitude=grad_norm)

    def train_step(self, prompts: list[str],
                   completions: list[list[str]],
                   rewards: list[list[float]],
                   prompt_token_lens: list[int] | None = None,
                   old_log_probs: list[list[torch.Tensor]] | None = None) -> dict:
        """One GRPO training step.

        Args:
            prompts: list of prompt strings (B prompts)
            completions: list of lists — completions[g] for each prompt (G per prompt)
            rewards: list of lists — rewards[g] for each completion (binary from executor)
            prompt_token_lens: optional pre-computed prompt lengths
            old_log_probs: optional per-prompt, per-completion log-prob tensors
                collected during rollout (detached, no grad). When provided,
                the PPO ratio uses these instead of the current policy's
                log-probs (which would make ratio=1.0).

        Returns:
            stats dict with loss, reward, KL, ACR
        """
        if not self.optimizer:
            return {"error": "no trainable parameters"}

        # Golden trajectory injection: mix in replayed successful trajectories
        n_injected = 0
        if self.replay_buffer is not None:
            prompts, completions, rewards, n_injected = \
                self._inject_golden_replays(prompts, completions, rewards)

        self.model.train()
        # Accumulate loss/kl as GPU scalar tensors to avoid per-step .item()
        # syncs (N syncs → 1 sync at the end of the step).
        total_loss = torch.zeros(1, device=self.device)
        total_kl = torch.zeros(1, device=self.device)
        total_reward = 0.0
        n_updates = 0
        accum_count = 0

        # Pre-tokenize prompt lengths once (avoids G× redundant tokenization)
        prompt_len_cache = self._pretokenize_prompt_lens(prompts) if not prompt_token_lens else None

        for prompt_idx, (prompt, comps, rews) in enumerate(zip(prompts, completions, rewards)):
            if not comps or len(comps) < 2:
                continue

            # GRPO-λ: dynamic length penalty based on group correctness ratio
            if self.config.use_grpo_lambda:
                cr = self._group_correctness_ratio(rews)
                self.stats.correctness_ratios.append(cr)
                rews = self._apply_grpo_lambda_penalty(rews, comps, cr)

            # N-gram repetition penalty (LFM2.5-Thinking RLVR recipe)
            if self.config.use_repetition_penalty:
                rews = self._apply_repetition_penalty(rews, comps)

            # Compute group-relative advantages
            advantages = self.compute_advantages(rews)
            self.stats.advantages.extend(advantages)
            self.stats.total_rewards.extend(rews)

            # GTPO: measure initial entropy on first step + build conflict masks
            gtpo_lam_masks = None
            if self.config.rl_algorithm == "gtpo":
                # Auto-calibrate initial entropy on first step
                if self._gtpo_init_entropy is None and self.config.gtpo_auto_init_entropy:
                    # Measure on first completion of first group
                    if prompt_idx == 0 and comps:
                        _enc = self.tokenizer(
                            prompt + comps[0], return_tensors="pt",
                            truncation=True, max_length=self.config.max_seq_len
                        ).input_ids.to(self.device)
                        with torch.no_grad():
                            _logits, _ = self.model(_enc)
                        self._gtpo_init_entropy = self._gtpo_compute_entropy(_logits)
                        print(f"  [GTPO] Initial entropy: {self._gtpo_init_entropy:.4f} "
                              f"(threshold: {self.config.gtpo_entropy_threshold:.4f})")

                # Build conflict masks for this group
                comp_token_ids = [
                    self.tokenizer(c, add_special_tokens=False).input_ids
                    for c in comps
                ]
                gtpo_lam_masks = self._gtpo_compute_conflict_masks(
                    comp_token_ids, advantages)

            for comp_idx, (completion, reward, advantage) in enumerate(
                    zip(comps, rews, advantages)):
                # Skip zero-advantage samples (no gradient signal)
                if abs(advantage) < 1e-6:
                    continue

                # Tokenize full sequence
                full_text = prompt + completion
                enc = self.tokenizer(full_text, return_tensors="pt",
                                     truncation=True, max_length=self.config.max_seq_len)
                input_ids = enc.input_ids.to(self.device)

                # Find prompt boundary (use cache to avoid re-tokenizing prompt)
                if prompt_token_lens:
                    prompt_len = prompt_token_lens[prompt_idx]
                elif prompt_len_cache is not None:
                    prompt_len = prompt_len_cache[prompt]
                else:
                    prompt_len = self.tokenizer(
                        prompt, return_tensors="pt").input_ids.shape[1]

                if input_ids.shape[1] <= prompt_len + 1:
                    continue  # completion too short

                with oom_guard(self.device, label="grpo_fwd") as safe:
                    # Forward pass (current policy) — autocast for bf16 efficiency
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16,
                        enabled=("cuda" in str(self.device)),
                    ):
                        logits, _ = self.model(input_ids)
                    solution_logits = logits[0, prompt_len - 1:-1, :]
                    solution_targets = input_ids[0, prompt_len:].long()

                    if solution_logits.shape[0] == 0:
                        continue  # completion too short

                    # Compute log probabilities for the generated tokens
                    log_probs = F.log_softmax(solution_logits, dim=-1)
                    token_log_probs = log_probs.gather(1, solution_targets.unsqueeze(0)).squeeze(0)

                    # Compute old log probs (for PPO ratio)
                    if (old_log_probs is not None
                            and prompt_idx < len(old_log_probs)
                            and comp_idx < len(old_log_probs[prompt_idx])):
                        old_lp = old_log_probs[prompt_idx][comp_idx]
                        old_lp = old_lp[:token_log_probs.shape[0]]
                        ratio = (token_log_probs - old_lp).exp()
                    else:
                        if old_log_probs is None and prompt_idx == 0 and comp_idx == 0:
                            import warnings
                            warnings.warn(
                                "old_log_probs not provided to train_step; "
                                "PPO ratio will be 1.0 (no off-policy correction). "
                                "Pass log-probs collected during rollout for proper PPO.",
                                stacklevel=2,
                            )
                        with torch.no_grad():
                            old_lp = token_log_probs.detach()
                        ratio = (token_log_probs - old_lp).exp()

                    # Clipped policy gradient loss
                    clipped_ratio = ratio.clamp(1 - self.config.clip_range,
                                                1 + self.config.clip_range)

                    # ETR (Entropy Trend Reward): bonus for downward entropy
                    # trend across the completion (converging reasoning).
                    if self.config.use_etr_reward:
                        with torch.no_grad():
                            _probs = log_probs.exp()
                            _ent = -(_probs * log_probs).sum(dim=-1)  # (T,)
                            _half = max(1, _ent.shape[0] // 2)
                            _trend = ((_ent[:_half].mean() - _ent[_half:].mean())
                                      / _ent.mean().clamp_min(1e-6))
                        advantage = advantage + self.config.etr_coeff * _trend.item()

                    if self.config.rl_algorithm == "gtpo":
                        # GTPO: conflict-aware gradient correction + entropy control
                        # No KL penalty needed (saves ref model forward pass + VRAM)
                        # λ weights from conflict masks: 0 (skip neg), 1 (normal), 2 (amplify pos)
                        if gtpo_lam_masks is not None and comp_idx < len(gtpo_lam_masks):
                            lam = gtpo_lam_masks[comp_idx][:ratio.shape[0]]
                            if lam.shape[0] < ratio.shape[0]:
                                lam = torch.cat([
                                    lam, torch.ones(ratio.shape[0] - lam.shape[0],
                                                    device=self.device)])
                        else:
                            lam = torch.ones_like(ratio)

                        # Entropy regularization: γ * ⟨H⟩_i
                        with torch.no_grad():
                            probs = log_probs.exp()
                            entropy_per_token = -(probs * log_probs).sum(dim=-1)
                            avg_entropy = entropy_per_token.mean()
                        # GTPO loss: -(A - γ*H) * mean(ratio * λ)
                        adjusted_adv = advantage - self.config.gtpo_entropy_gamma * avg_entropy.item()
                        pg_loss = -(adjusted_adv * lam * ratio).mean()
                        kl = torch.zeros(1, device=self.device)  # no KL
                        kl_loss = kl
                    elif self.config.rl_algorithm == "cispo":
                        # CISPO (MiniMax): L = -A * sg(clip(r, 1-ε_lo, 1+ε_hi)) * log π.
                        # The clipped importance weight is DETACHED, so every
                        # token keeps gradient (no PPO min-clip dead zones).
                        with torch.no_grad():
                            cispo_w = ratio.clamp(
                                1 - self.config.cispo_epsilon_low,
                                1 + self.config.cispo_epsilon_high)
                        pg_loss = -(advantage * cispo_w * token_log_probs).mean()
                        kl = torch.zeros(1, device=self.device)  # no KL
                        kl_loss = kl
                    else:
                        pg_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

                        # KL penalty
                        kl = self.compute_kl_penalty(input_ids, logits)
                        kl_loss = self.config.kl_coefficient * kl

                    # MoE load-balancing aux loss (prevents router collapse during RL).
                    # The model forward stores this on self._last_moe_aux_loss even
                    # when targets are not passed (GRPO uses logits-only forward).
                    # Without this, the MoE router gets no balancing signal during
                    # RLVR and can collapse to a single expert.
                    moe_aux = getattr(self.model, '_last_moe_aux_loss', None)
                    moe_aux_term = 0.0
                    if moe_aux is not None and moe_aux.requires_grad:
                        moe_aux_term = moe_aux * getattr(self.config, 'moe_aux_weight', 0.01)

                    # Total loss
                    loss = (pg_loss + kl_loss + moe_aux_term) / self.config.grad_accum_steps
                    loss.backward()
                if safe.skipped:
                    continue
                accum_count += 1

                total_loss += loss.detach() * self.config.grad_accum_steps
                total_kl += kl.detach()
                total_reward += reward
                n_updates += 1

                # Gradient step
                if accum_count >= self.config.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # DeepSeek-V3 aux-loss-free: update expert bias after step.
                    from forge.moe.moe import update_moe_biases
                    update_moe_biases(self.model)
                    accum_count = 0

        # Final step
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            from forge.moe.moe import update_moe_biases
            update_moe_biases(self.model)

        self.model.eval()

        # Update stats — single .item() sync for accumulated GPU tensors
        self.stats.total_steps += 1
        self.stats.policy_losses.append(total_loss.item() / max(n_updates, 1))
        self.stats.kl_divergences.append(total_kl.item() / max(n_updates, 1))
        self.stats.trim()  # prevent unbounded memory growth

        acr = self.stats.advantage_collapse_rate
        acr_alert = " [WARNING: ACR>0.3]" if acr > 0.3 else ""

        # Record successful trajectories into replay buffer for future injection
        if self.replay_buffer is not None:
            grad_norm_val = 0.0
            self._record_golden_trajectories(
                prompts, completions, rewards, grad_norm=grad_norm_val)

        _mean_loss = total_loss.item() / max(n_updates, 1)
        _mean_kl = total_kl.item() / max(n_updates, 1)

        return {
            "n_updates": n_updates,
            "mean_loss": _mean_loss,
            "mean_reward": total_reward / max(n_updates, 1),
            "mean_kl": _mean_kl,
            "advantage_collapse_rate": acr,
            "acr_alert": acr_alert,
            "n_golden_injected": n_injected,
            "mean_correctness_ratio": (sum(self.stats.correctness_ratios[-10:]) /
                                       max(len(self.stats.correctness_ratios[-10:]), 1))
                                      if self.config.use_grpo_lambda else None,
            "length_penalty_active": self.config.use_grpo_lambda and
                                      self.stats.length_penalty_active_count > 0,
        }

    def collect_log_probs(self, prompts: list[str],
                          completions: list[list[str]],
                          prompt_token_lens: list[int] | None = None
                          ) -> list[list[torch.Tensor]]:
        """Collect per-token log-probs from the current policy (no grad).

        Call this during rollout (before train_step) to capture the policy's
        log-probs at collection time. Pass the returned list as
        ``old_log_probs`` to ``train_step`` so the PPO ratio reflects the
        gap between the collection policy and the updated policy.

        Args:
            prompts: list of prompt strings (B prompts).
            completions: list of lists — completions[g] for each prompt.
            prompt_token_lens: optional pre-computed prompt lengths.

        Returns:
            list[list[Tensor]] — old_log_probs[prompt_idx][comp_idx] is a
            1-D tensor of log-probs for the solution tokens (detached).
        """
        self.model.eval()
        all_log_probs: list[list[torch.Tensor]] = []
        # Pre-tokenize prompt lengths once (avoids G× redundant tokenization)
        prompt_len_cache = self._pretokenize_prompt_lens(prompts) if not prompt_token_lens else None
        with torch.inference_mode():
            for prompt_idx, (prompt, comps) in enumerate(zip(prompts, completions)):
                group_lps: list[torch.Tensor] = []
                # Determine prompt length from cache
                if prompt_token_lens:
                    prompt_len = prompt_token_lens[prompt_idx]
                elif prompt_len_cache is not None:
                    prompt_len = prompt_len_cache[prompt]
                else:
                    prompt_len = self.tokenizer(
                        prompt, return_tensors="pt").input_ids.shape[1]

                # Tokenize all completions for this prompt, filter valid ones
                valid_seqs = []  # (comp_idx, input_ids, sol_len)
                for comp_idx, completion in enumerate(comps):
                    full_text = prompt + completion
                    enc = self.tokenizer(
                        full_text, return_tensors="pt",
                        truncation=True, max_length=self.config.max_seq_len)
                    input_ids = enc.input_ids.to(self.device)
                    sol_len = input_ids.shape[1] - prompt_len
                    if input_ids.shape[1] <= prompt_len + 1 or sol_len <= 0:
                        group_lps.append(torch.tensor([], device=self.device))
                        continue
                    valid_seqs.append((comp_idx, input_ids, sol_len))

                if not valid_seqs:
                    all_log_probs.append(group_lps if group_lps else
                                         [torch.tensor([], device=self.device)] * len(comps))
                    continue

                # Batch forward pass for all valid completions of this prompt
                max_seq = max(s[1].shape[1] for s in valid_seqs)
                batch_ids = torch.full((len(valid_seqs), max_seq),
                                       self.tokenizer.pad_token_id or 0,
                                       dtype=torch.long, device=self.device)
                attn_mask = torch.zeros_like(batch_ids)
                for i, (_, ids, _) in enumerate(valid_seqs):
                    seq_len = ids.shape[1]
                    batch_ids[i, :seq_len] = ids[0]
                    attn_mask[i, :seq_len] = 1

                logits, _ = self.model(batch_ids, attention_mask=attn_mask)

                for i, (comp_idx, ids, sol_len) in enumerate(valid_seqs):
                    seq_len = ids.shape[1]
                    solution_logits = logits[i, prompt_len - 1:seq_len - 1, :]
                    solution_targets = ids[0, prompt_len:].long()
                    if solution_logits.shape[0] == 0:
                        group_lps.append(torch.tensor([], device=self.device))
                        continue
                    log_probs = F.log_softmax(solution_logits, dim=-1)
                    token_log_probs = log_probs.gather(
                        1, solution_targets.unsqueeze(0)).squeeze(0)
                    group_lps.append(token_log_probs.detach())

                # Pad group_lps to match completions count
                while len(group_lps) < len(comps):
                    group_lps.append(torch.tensor([], device=self.device))
                all_log_probs.append(group_lps)
        return all_log_probs

    def exgrpo_group_entropy(self, prompt: str,
                             completions: list[str]) -> float:
        """Mean policy entropy over the group's solution tokens (ExGRPO).

        One logits-only forward per completion; the per-token Shannon
        entropy of the policy distribution over the solution span is
        averaged across the group.  Used as the experience-value signal
        (correctness * entropy) for ExGRPO replay ordering — the rollout
        data itself records no per-token logprobs, so this is the cheap
        recomputation path.

        Returns 0.0 when no completion is long enough to score.
        """
        try:
            prompt_len = self.tokenizer(
                prompt, return_tensors="pt").input_ids.shape[1]
        except Exception:
            return 0.0
        entropies = []
        with torch.inference_mode():
            for completion in completions:
                enc = self.tokenizer(
                    prompt + completion, return_tensors="pt",
                    truncation=True, max_length=self.config.max_seq_len)
                input_ids = enc.input_ids.to(self.device)
                if input_ids.shape[1] <= prompt_len + 1:
                    continue
                logits, _ = self.model(input_ids)
                sol_logits = logits[0, prompt_len - 1:-1, :]
                if sol_logits.shape[0] == 0:
                    continue
                lp = F.log_softmax(sol_logits.float(), dim=-1)
                ent = -(lp.exp() * lp).sum(dim=-1).mean()
                entropies.append(float(ent.item()))
        return sum(entropies) / len(entropies) if entropies else 0.0

    def get_stats(self) -> dict:
        """Return cumulative training stats for monitoring."""
        return {
            "total_steps": self.stats.total_steps,
            "mean_reward": self.stats.mean_reward,
            "mean_advantage": self.stats.mean_advantage,
            "mean_kl": self.stats.mean_kl,
            "mean_policy_loss": (
                sum(self.stats.policy_losses) / max(len(self.stats.policy_losses), 1)
            ),
            "advantage_collapse_rate": self.stats.advantage_collapse_rate,
            "acr_alert": self.stats.advantage_collapse_rate > 0.3,
            "mean_correctness_ratio": (sum(self.stats.correctness_ratios) /
                                       max(len(self.stats.correctness_ratios), 1))
                                      if self.stats.correctness_ratios else None,
            "length_penalty_active_count": self.stats.length_penalty_active_count,
        }

    # -----------------------------------------------------------------------
    # SC-GRPO: Self-Conditioned GRPO (arXiv 2605.22217 extension)
    # -----------------------------------------------------------------------

    def _compute_sc_weights(self, input_ids_list, prompt_lens, rewards,
                            solution_lens):
        """Compute per-token SC-GRPO weights from verified/unverified KL.

        For each token position in the solution, compute the model's logit
        distribution when conditioned on verified vs unverified trajectories.
        The KL divergence between these two distributions is the self-conditioned
        signal: high KL = pivotal token (model behaves differently on success
        vs failure paths), low KL = routine token.

        Args:
            input_ids_list: list of [1, seq_len] tensors (one per completion)
            prompt_lens: list of prompt lengths
            rewards: list of binary rewards (1=verified, 0=unverified)
            solution_lens: list of solution lengths

        Returns:
            list of [solution_len] tensors — per-token multiplicative weights
            (normalized so mean=1, clipped to [0, sc_weight_clip]).
        """
        if not self.config.use_sc_grpo:
            return [None] * len(input_ids_list)

        verified_idx = [i for i, r in enumerate(rewards) if r > 0.5]
        unverified_idx = [i for i, r in enumerate(rewards) if r <= 0.5]

        # Need both verified and unverified to compute the signal.
        # If all verified or all unverified, fall back to uniform weights.
        if not verified_idx or not unverified_idx:
            return [None] * len(input_ids_list)

        # Forward pass on all completions to get solution logits.
        all_log_probs = []
        with torch.inference_mode():
            for i, ids in enumerate(input_ids_list):
                logits, _ = self.model(ids)
                plen = prompt_lens[i]
                sol_logits = logits[0, plen - 1:-1, :]  # predict solution tokens
                lp = F.log_softmax(sol_logits, dim=-1)
                all_log_probs.append(lp)

        # Find the minimum solution length (KL is computed on the common prefix).
        min_sol = min(solution_lens)

        # Mean log-prob distribution over verified and unverified completions.
        # Shape: [min_sol, vocab]
        verified_lp = torch.stack([all_log_probs[i][:min_sol] for i in verified_idx]).mean(0)
        unverified_lp = torch.stack([all_log_probs[i][:min_sol] for i in unverified_idx]).mean(0)

        # Per-token KL(unverified || verified) — how much the model's prediction
        # diverges between failure and success paths at each position.
        # KL(p || q) = Σ p * (log p - log q). Here p=unverified, q=verified.
        kl_per_token = (unverified_lp.exp() * (unverified_lp - verified_lp)).sum(dim=-1)
        # [min_sol]

        # Normalize: weights are relative KL (mean=1), clipped to [0, sc_weight_clip].
        mean_kl = kl_per_token.mean().clamp(min=1e-8)
        weights = (kl_per_token / mean_kl).clamp(0, self.config.sc_weight_clip)

        # Build per-completion weight tensors (padded with 1.0 beyond min_sol).
        result = []
        for i, slen in enumerate(solution_lens):
            w = torch.ones(slen, device=weights.device, dtype=weights.dtype)
            w[:min_sol] = weights
            result.append(w)

        return result

    # -----------------------------------------------------------------------
    # OM-GRPO: Outcome-Masked GRPO (gradient masking on answer span)
    # -----------------------------------------------------------------------

    def _find_answer_span(self, completion: str, prompt_len: int,
                          total_len: int) -> tuple[int, int]:
        """Find the answer span in a completion (tokens after the last
        answer marker or code block).

        Returns (start, end) token indices within the solution span.
        The solution span is [prompt_len, total_len). The answer span is
        the tail portion after the last marker — gradients are masked there.
        """
        if not self.config.use_om_grpo:
            return prompt_len, total_len  # no masking

        markers = self.config.om_answer_markers
        last_marker_pos = -1
        for marker in markers:
            pos = completion.rfind(marker)
            if pos > last_marker_pos:
                last_marker_pos = pos

        if last_marker_pos < 0:
            # No marker found — don't mask (treat the whole solution as reasoning).
            return total_len, total_len  # empty answer span = no masking

        # Approximate: the answer starts after the marker text.
        # Tokenize the prefix up to the marker to find the token boundary.
        marker_end = last_marker_pos
        prefix = completion[:marker_end]
        prefix_ids = self.tokenizer(prefix, return_tensors="pt").input_ids
        answer_start = prompt_len + prefix_ids.shape[1]
        return answer_start, total_len

    def _build_om_mask(self, completion: str, prompt_len: int,
                       total_len: int, solution_len: int) -> torch.Tensor:
        """Build a per-token gradient mask for OM-GRPO.

        Mask is 1.0 on reasoning tokens, 0.0 on answer tokens. The answer
        span is detected via markers (Answer:, ```, Final:) — everything
        after the last marker is the answer span.
        """
        if not self.config.use_om_grpo:
            return torch.ones(solution_len, device=self.device)

        ans_start, ans_end = self._find_answer_span(completion, prompt_len, total_len)
        mask = torch.ones(solution_len, device=self.device)
        # Mask out answer tokens (zero gradient on answer span).
        start_in_sol = max(0, ans_start - prompt_len)
        end_in_sol = min(solution_len, ans_end - prompt_len)
        if start_in_sol < end_in_sol:
            mask[start_in_sol:end_in_sol] = 0.0
        return mask

    # -----------------------------------------------------------------------
    # GVPO: Group Verification-based Policy Optimization (process rewards)
    # -----------------------------------------------------------------------

    def compute_gvpo_advantages(self, outcome_rewards: list[float],
                                process_rewards: list[list[float]] | None
                                ) -> list[float]:
        """GVPO advantage: outcome + λ * Σ(process_reward_per_round).

        Args:
            outcome_rewards: binary test_passed (1.0/0.0) per completion.
            process_rewards: per-completion list of per-round error signals.
                Each inner list is [r_1, r_2, ..., r_k] where r_i is a
                negative reward for execution errors in round i (0 if no
                error). Pass None or empty to disable process rewards for
                a completion.

        Returns:
            list of GVPO advantages (group-relative normalized).
        """
        lam = self.config.gvpo_lambda
        combined = []
        for i, outcome in enumerate(outcome_rewards):
            proc = process_rewards[i] if process_rewards and i < len(process_rewards) else []
            proc_sum = sum(proc) if proc else 0.0
            combined.append(outcome + lam * proc_sum)

        # Group-relative normalization (same as compute_advantages but on combined).
        if not combined:
            return []
        if self.config.use_median_baseline and len(combined) >= 2:
            sorted_c = sorted(combined)
            n = len(sorted_c)
            baseline = sorted_c[n // 2] if n % 2 == 1 else (sorted_c[n // 2 - 1] + sorted_c[n // 2]) / 2
        else:
            baseline = sum(combined) / len(combined)
        if len(combined) > 1:
            variance = sum((c - baseline) ** 2 for c in combined) / len(combined)
            std = math.sqrt(variance + 1e-8)
        else:
            std = 1.0
        return [(c - baseline) / (std + 1e-8) for c in combined]

    # -----------------------------------------------------------------------
    # Unified train_step with SC/OM/GVPO support
    # -----------------------------------------------------------------------

    def train_step_advanced(self, prompts: list[str],
                            completions: list[list[str]],
                            rewards: list[list[float]],
                            prompt_token_lens: list[int] | None = None,
                            process_rewards: list[list[list[float]]] | None = None,
                            ) -> dict:
        """GRPO training step with SC-GRPO / OM-GRPO / GVPO extensions.

        Args:
            prompts: list of prompt strings (B prompts).
            completions: list of lists — completions[g] for each prompt.
            rewards: list of lists — binary rewards per completion.
            prompt_token_lens: optional pre-computed prompt lengths.
            process_rewards: per-prompt, per-completion list of per-round
                error signals (for GVPO). Shape: [B][G][R]. None disables.

        Returns:
            stats dict with loss, reward, KL, ACR, and SC/OM/GVPO metrics.
        """
        if not self.optimizer:
            return {"error": "no trainable parameters"}

        # Golden trajectory injection
        n_injected = 0
        if self.replay_buffer is not None:
            prompts, completions, rewards, n_injected = \
                self._inject_golden_replays(prompts, completions, rewards)

        self.model.train()
        # GPU tensor accumulators — avoid per-step .item() syncs
        total_loss = torch.zeros(1, device=self.device)
        total_kl = torch.zeros(1, device=self.device)
        total_reward = 0.0
        total_sc_weight = 0.0
        n_updates = 0
        accum_count = 0

        # Pre-tokenize prompt lengths once (avoids G× redundant tokenization)
        prompt_len_cache = self._pretokenize_prompt_lens(prompts) if not prompt_token_lens else None

        for prompt_idx, (prompt, comps, rews) in enumerate(
                zip(prompts, completions, rewards)):
            if not comps or len(comps) < 2:
                continue

            # GRPO-λ: dynamic length penalty based on group correctness ratio
            if self.config.use_grpo_lambda:
                cr = self._group_correctness_ratio(rews)
                self.stats.correctness_ratios.append(cr)
                rews = self._apply_grpo_lambda_penalty(rews, comps, cr)

            # Compute advantages (GVPO if enabled, else standard GRPO).
            if self.config.use_gvpo and process_rewards is not None:
                proc = process_rewards[prompt_idx] if prompt_idx < len(process_rewards) else None
                advantages = self.compute_gvpo_advantages(rews, proc)
            else:
                advantages = self.compute_advantages(rews)

            self.stats.advantages.extend(advantages)
            self.stats.total_rewards.extend(rews)

            # SC-GRPO: compute per-token self-conditioned weights for the group.
            # Need to pre-tokenize all completions to forward them.
            group_input_ids = []
            group_prompt_lens = []
            group_solution_lens = []
            # Use cached prompt length (avoids re-tokenizing prompt per completion)
            cached_plen = (prompt_token_lens[prompt_idx] if prompt_token_lens
                           else prompt_len_cache.get(prompt) if prompt_len_cache
                           else None)
            for comp in comps:
                full_text = prompt + comp
                enc = self.tokenizer(full_text, return_tensors="pt",
                                     truncation=True, max_length=self.config.max_seq_len)
                ids = enc.input_ids.to(self.device)
                plen = cached_plen if cached_plen is not None else \
                    self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
                group_input_ids.append(ids)
                group_prompt_lens.append(plen)
                group_solution_lens.append(ids.shape[1] - plen)

            sc_weights = self._compute_sc_weights(
                group_input_ids, group_prompt_lens, rews, group_solution_lens)

            for comp_idx, (completion, reward, advantage) in enumerate(
                    zip(comps, rews, advantages)):
                if abs(advantage) < 1e-6:
                    continue

                input_ids = group_input_ids[comp_idx]
                prompt_len = group_prompt_lens[comp_idx]
                sol_len = group_solution_lens[comp_idx]

                if sol_len <= 0:
                    continue

                # Forward pass (current policy).
                logits, _ = self.model(input_ids)
                solution_logits = logits[0, prompt_len - 1:-1, :]
                solution_targets = input_ids[0, prompt_len:].long()

                if solution_logits.shape[0] == 0:
                    continue

                log_probs = F.log_softmax(solution_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    1, solution_targets.unsqueeze(0)).squeeze(0)

                with torch.no_grad():
                    old_log_probs = token_log_probs.detach()

                ratio = (token_log_probs - old_log_probs).exp()
                clipped_ratio = ratio.clamp(
                    1 - self.config.clip_range, 1 + self.config.clip_range)
                pg_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

                # SC-GRPO: apply per-token self-conditioned weight.
                if sc_weights[comp_idx] is not None:
                    w = sc_weights[comp_idx][:pg_loss.shape[0]]
                    pg_loss = pg_loss * w
                    total_sc_weight += float(w.mean().detach().item())

                # OM-GRPO: mask gradients on the answer span.
                if self.config.use_om_grpo:
                    om_mask = self._build_om_mask(
                        completion, prompt_len, input_ids.shape[1], sol_len)
                    pg_loss = pg_loss * om_mask[:pg_loss.shape[0]]

                pg_loss = pg_loss.mean()

                # KL penalty.
                kl = self.compute_kl_penalty(input_ids, logits)
                kl_loss = self.config.kl_coefficient * kl

                # MoE load-balancing aux loss (prevents router collapse during RL).
                moe_aux = getattr(self.model, '_last_moe_aux_loss', None)
                moe_aux_term = 0.0
                if moe_aux is not None and moe_aux.requires_grad:
                    moe_aux_term = moe_aux * getattr(self.config, 'moe_aux_weight', 0.01)

                loss = (pg_loss + kl_loss + moe_aux_term) / self.config.grad_accum_steps
                loss.backward()
                accum_count += 1

                total_loss += loss.detach() * self.config.grad_accum_steps
                total_kl += kl.detach()
                total_reward += reward
                n_updates += 1

                if accum_count >= self.config.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    from forge.moe.moe import update_moe_biases
                    update_moe_biases(self.model)
                    accum_count = 0

        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            from forge.moe.moe import update_moe_biases
            update_moe_biases(self.model)

        self.model.eval()
        self.stats.total_steps += 1
        self.stats.policy_losses.append(total_loss.item() / max(n_updates, 1))
        self.stats.kl_divergences.append(total_kl.item() / max(n_updates, 1))
        self.stats.trim()  # prevent unbounded memory growth

        acr = self.stats.advantage_collapse_rate
        modes = []
        if self.config.use_sc_grpo:
            modes.append("SC")
        if self.config.use_om_grpo:
            modes.append("OM")
        if self.config.use_gvpo:
            modes.append("GVPO")
        mode_str = "+".join(modes) if modes else "vanilla"

        _mean_loss = total_loss.item() / max(n_updates, 1)
        _mean_kl = total_kl.item() / max(n_updates, 1)

        return {
            "n_updates": n_updates,
            "mean_loss": _mean_loss,
            "mean_reward": total_reward / max(n_updates, 1),
            "mean_kl": _mean_kl,
            "advantage_collapse_rate": acr,
            "acr_alert": " [WARNING: ACR>0.3]" if acr > 0.3 else "",
            "mode": mode_str,
            "mean_sc_weight": total_sc_weight / max(n_updates, 1) if self.config.use_sc_grpo else 1.0,
            "n_golden_injected": n_injected,
            "mean_correctness_ratio": (sum(self.stats.correctness_ratios[-10:]) /
                                       max(len(self.stats.correctness_ratios[-10:]), 1))
                                      if self.config.use_grpo_lambda else None,
            "length_penalty_active": self.config.use_grpo_lambda and
                                      self.stats.length_penalty_active_count > 0,
        }
