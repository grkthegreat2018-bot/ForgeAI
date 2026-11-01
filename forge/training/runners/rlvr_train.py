"""RLVR (Reinforcement Learning with Verifiable Rewards) runner.

Implements the RLVR stage of the ForgeLM V10-Thinking recipe:
  - GRPO-style RL on verifiable tasks (math, code, reasoning)
  - Binary reward: 1.0 if verified correct, 0.0 otherwise (the binding gate)
  - N-gram repetition penalty early in training (doom-loop mitigation)
  - KL penalty against DPO checkpoint (reference model)
  - SPPO/PS-PPO for long CoT (configurable via --rl-algorithm)

Uses the existing GRPOTrainer with:
  - use_repetition_penalty=True (ForgeLM V10-Thinking recipe)
  - Binary verified rewards from task executors
  - Reference model = DPO checkpoint

## Verifiable task sources

  - Math (gsm8k, orca_math, metamath): extract final answer, compare to gold
  - Code (code_alpaca, codeforces): execute solution, check test output
  - Reasoning (bbh, folio): compare final answer to gold answer

## Usage

  python -m research.training.runners.rlvr_train \\
      --tasks data/hf_datasets/gsm8k.jsonl \\
      --checkpoint research/checkpoints/forgelm_v2_DPO.safetensors \\
      --save research/checkpoints/forgelm_v2_RLVR.safetensors \\
      --config forgelm_v2 \\
      --max-steps 500 \\
      --group-size 4 \\
      --rl-algorithm grpo \\
      --use-repetition-penalty \\
      --optimizer cpu_offload
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

from forge.config import get_config
from forge.model_loader import ModelLoader
from forge.training.training_utils import (
    oom_guard,
    patch_triton_cache_for_windows,
    write_heartbeat,
    write_status_json,
)
from research.tokenizer_cache import get_tokenizer

# ── Verifiable reward functions ─────────────────────────────────────────────

def extract_math_answer(text: str) -> str | None:
    """Extract the final numerical answer from a math solution.

    Handles common formats:
    - "#### 42" (GSM8K format)
    - "The answer is 42"
    - "\\boxed{42}"
    - "Final answer: 42"
    """
    # GSM8K format: #### <number>
    m = re.search(r'####\s*([\d,\.\-]+)', text)
    if m:
        return m.group(1).replace(",", "").strip()

    # \boxed{...}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()

    # "The answer is X" / "Final answer: X"
    m = re.search(r'(?:answer is|Final answer[:\s]+)\s*([\d,\.\-]+)', text, re.I)
    if m:
        ans = m.group(1).replace(",", "").strip()
        # Strip trailing period (not part of the number)
        if ans.endswith(".") and not ans[:-1].endswith("."):
            ans = ans[:-1]
        return ans

    # Last number in the text (fallback)
    numbers = re.findall(r'[\d,\.\-]+', text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison."""
    ans = ans.strip().lower()
    ans = ans.replace(",", "")
    ans = ans.replace("$", "")
    ans = ans.replace("\\", "")
    ans = ans.replace(" ", "")
    # Try to convert to float for numeric comparison
    try:
        return str(float(ans))
    except (ValueError, TypeError):
        return ans


def math_verify(completion: str, gold_answer: str) -> bool:
    """Verify a math completion against the gold answer."""
    extracted = extract_math_answer(completion)
    if extracted is None:
        return False
    return normalize_answer(extracted) == normalize_answer(gold_answer)


def extract_gold_answer(solution: str, task_type: str = "math") -> str | None:
    """Extract the gold answer from a solution text."""
    if task_type == "math":
        return extract_math_answer(solution)
    return None


# ── Task loading ────────────────────────────────────────────────────────────

@dataclass
class VerifiableTask:
    """A single verifiable task for RLVR."""
    prompt: str
    gold_answer: str
    task_type: str  # "math", "code", "reasoning"
    verify_fn: Callable[[str, str], bool]


def load_math_tasks(path: str, max_tasks: int | None = None) -> list[VerifiableTask]:
    """Load math tasks from a JSONL file with {prompt, solution} format.

    The gold answer is extracted from the solution using extract_math_answer.
    """
    tasks = []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = obj.get("prompt", "")
        solution = obj.get("solution", obj.get("response", ""))
        if not prompt or not solution:
            continue
        gold = extract_gold_answer(solution, "math")
        if gold is None:
            continue
        tasks.append(VerifiableTask(
            prompt=prompt, gold_answer=gold,
            task_type="math", verify_fn=math_verify,
        ))
        if max_tasks and len(tasks) >= max_tasks:
            break
    return tasks


# ── Candidate generation for GRPO ───────────────────────────────────────────

def generate_completions(
    model, tokenizer, prompt: str, group_size: int = 4,
    max_new_tokens: int = 512, temperature: float = 0.8,
    device: str = "cuda", top_k: int = 50, top_p: float = 0.95,
) -> list[str]:
    """Generate G completions for a single prompt."""
    prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids["input_ids"].to(device)

    completions = []
    with torch.inference_mode():
        for _ in range(group_size):
            if hasattr(model, "generate"):
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id or 0,
                )
            else:
                out_ids = _manual_generate(
                    model, input_ids, max_new_tokens, temperature, top_k, top_p, device
                )
            text = tokenizer.decode(out_ids[0, input_ids.shape[1]:],
                                     skip_special_tokens=True)
            completions.append(text)
    return completions


def _manual_generate(model, input_ids, max_new_tokens, temperature=0.8,
                     top_k=None, top_p=None, device="cuda"):
    """Manual autoregressive generation fallback."""
    B = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    # Pre-allocate output buffer (O(1) per token vs O(n²) torch.cat)
    buf = torch.zeros(B, prompt_len + max_new_tokens, dtype=input_ids.dtype,
                      device=input_ids.device)
    buf[:, :prompt_len] = input_ids
    pos = prompt_len
    eos_id = 7  # <|im_end|>

    for _ in range(max_new_tokens):
        out = model(buf[:, :pos])
        logits = out[0] if isinstance(out, tuple) else out
        next_logits = logits[:, -1, :].float() / max(temperature, 1e-8)
        if top_k:
            v, _ = next_logits.topk(top_k)
            next_logits[next_logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        buf[:, pos:pos+1] = next_token
        pos += 1
        if next_token.item() == eos_id:
            break
    return buf[:, :pos]


# ── Main RLVR training loop ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RLVR with GRPO + n-gram repetition penalty")
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="JSONL files with verifiable tasks (math/code/reasoning)")
    parser.add_argument("--task-type", default="math", choices=["math", "code", "reasoning"],
                        help="Type of verification to use")
    parser.add_argument("--checkpoint", required=True,
                        help="DPO checkpoint to start RLVR from")
    parser.add_argument("--save", required=True,
                        help="Output checkpoint path")
    parser.add_argument("--config", default="forgelm_v2")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=4,
                        help="GRPO group size (G completions per prompt)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl-coefficient", type=float, default=0.02)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--rl-algorithm", default="grpo",
                        choices=["grpo", "sppo", "psppo", "evpo", "grpo_or"],
                        help="RL algorithm (grpo=default, sppo=long CoT, psppo=compute-efficient)")
    parser.add_argument("--use-repetition-penalty", action="store_true", default=True,
                        help="Enable n-gram repetition penalty (ForgeLM V10-Thinking recipe)")
    parser.add_argument("--repetition-penalty", type=float, default=-0.5)
    parser.add_argument("--repetition-warmup-steps", type=int, default=50)
    parser.add_argument("--optimizer", default="cpu_offload",
                        help="Optimizer: cpu_offload (full precision), adamw, bnb")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks to load (for testing)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    patch_triton_cache_for_windows()
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("  RLVR: REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS")
    print(f"{'='*70}")
    print(f"Tasks: {args.tasks}")
    print(f"Task type: {args.task_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Save: {args.save}")
    print(f"RL algorithm: {args.rl_algorithm}")
    print(f"Group size: {args.group_size} | LR: {args.lr} | Steps: {args.max_steps}")
    print(f"Repetition penalty: {args.use_repetition_penalty}")

    # ── Load tasks ──
    print(f"\nLoading {args.task_type} tasks...")
    tasks = []
    for path in args.tasks:
        if args.task_type == "math":
            t = load_math_tasks(path, args.max_tasks)
        else:
            print(f"  Warning: task_type '{args.task_type}' not yet implemented, using math")
            t = load_math_tasks(path, args.max_tasks)
        tasks.extend(t)
        print(f"  {path}: {len(t)} tasks loaded")
    print(f"Total tasks: {len(tasks)}")

    if not tasks:
        print("No tasks loaded. Check --tasks paths.")
        return

    # ── Load model + reference model ──
    print(f"\nLoading model ({args.config})...")
    config = get_config(args.config)
    model = ModelLoader.build_model(config, checkpoint_path=args.checkpoint)
    model = model.to(device).to(torch.bfloat16)

    # Reference model (frozen, for KL penalty)
    ref_model = ModelLoader.build_model(config, checkpoint_path=args.checkpoint)
    ref_model = ref_model.to(device).to(torch.bfloat16)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    tokenizer = get_tokenizer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    # ── Configure GRPO trainer ──
    from forge.self_play.grpo_trainer import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        learning_rate=args.lr,
        kl_coefficient=args.kl_coefficient,
        clip_range=args.clip_range,
        group_size=args.group_size,
        temperature=args.temperature,
        max_grad_norm=1.0,
        max_seq_len=args.max_seq_len,
        grad_accum_steps=args.grad_accum,
        rl_algorithm=args.rl_algorithm,
        use_repetition_penalty=args.use_repetition_penalty,
        repetition_penalty=args.repetition_penalty,
        repetition_warmup_steps=args.repetition_warmup_steps,
    )

    # Set optimizer choice on config for GRPOTrainer
    grpo_config.optimizer = args.optimizer

    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, ref_model=ref_model,
        device=str(device), config=grpo_config,
    )

    # ── RLVR training loop ──
    print(f"\nTraining {args.max_steps} RLVR steps...")
    t0 = time.time()
    step = 0
    total_reward = 0.0
    total_correct = 0
    total_completions = 0
    doom_loops = 0

    from forge.training.runners.curriculum_sft import is_doom_loop

    while step < args.max_steps:
        # Sample a batch of tasks
        batch_size = 4  # prompts per step
        batch_tasks = random.sample(tasks, min(batch_size, len(tasks)))

        prompts = []
        completions = []
        rewards = []

        for task in batch_tasks:
            # Generate G completions (OOM-safe — skip task on OOM)
            with oom_guard(str(device), label="rlvr_gen") as gen_safe:
                comps = generate_completions(
                    model, tokenizer, task.prompt,
                    group_size=args.group_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=str(device), top_k=args.top_k, top_p=args.top_p,
                )
            if gen_safe.skipped:
                continue
            # Verify each completion
            rews = []
            for comp in comps:
                correct = task.verify_fn(comp, task.gold_answer)
                rews.append(1.0 if correct else 0.0)
                total_completions += 1
                if correct:
                    total_correct += 1
                if is_doom_loop(comp):
                    doom_loops += 1

            prompts.append(task.prompt)
            completions.append(comps)
            rewards.append(rews)

        # GRPO training step (OOM-safe — skip step on OOM)
        if not prompts:
            continue
        with oom_guard(str(device), label="rlvr_step") as step_safe:
            stats = trainer.train_step(prompts, completions, rewards)
        if step_safe.skipped:
            continue
        step += 1

        # Track metrics
        batch_reward = sum(sum(r) for r in rewards) / max(total_completions, 1)
        total_reward += batch_reward

        # Logging
        if step % 5 == 0 or step == 1:
            elapsed = time.time() - t0
            accuracy = total_correct / max(total_completions, 1)
            doom_rate = doom_loops / max(total_completions, 1)
            print(f"  step {step}/{args.max_steps} | "
                  f"acc {accuracy:.2%} | "
                  f"doom {doom_rate:.2%} | "
                  f"loss {stats.get('loss', 0):.4f} | "
                  f"kl {stats.get('kl', 0):.4f} | "
                  f"{elapsed:.0f}s")
            write_heartbeat("rlvr")
            write_status_json("rlvr", step, args.max_steps,
                              stats.get('loss', 0), args.lr)

        # Periodic save
        if step % args.save_every == 0 and step < args.max_steps:
            from forge.checkpoint_io import save_training_checkpoint
            save_training_checkpoint(model, args.save.replace(".safetensors",
                              f"_step{step}.safetensors"), step)

    # ── Final save ──
    print(f"\nSaving final checkpoint: {args.save}")
    from forge.checkpoint_io import save_training_checkpoint
    save_training_checkpoint(model, args.save, step)

    accuracy = total_correct / max(total_completions, 1)
    doom_rate = doom_loops / max(total_completions, 1)
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("  RLVR COMPLETE")
    print(f"{'='*70}")
    print(f"Steps: {step}")
    print(f"Accuracy: {accuracy:.2%} ({total_correct}/{total_completions})")
    print(f"Doom-loop rate: {doom_rate:.2%} (target: <1%)")
    print(f"Time: {elapsed:.0f}s")
    print(f"Checkpoint: {args.save}")
    print("\nNext: Phase 5 — wire into self-play loop")


if __name__ == "__main__":
    main()
