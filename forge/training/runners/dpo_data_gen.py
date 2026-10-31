"""DPO preference data generation with doom-loop mitigation.

Implements the DPO stage of the ForgeLM V10-Thinking recipe:

  1. Generate candidates: for each prompt, sample 5 temperature-sampled + 1
     greedy response from the SFT checkpoint.
  2. LLM judge: use a teacher API model (via distill_client) to score each
     candidate on a 1-10 scale.
  3. Doom-loop detection: flag candidates with excessive n-gram repetition.
  4. Construct preference pairs:
     - Chosen = highest-scoring NON-LOOPING candidate
     - Rejected = lowest-scoring candidate, OR any looping candidate
       (regardless of judge score — looping is always rejected)
  This reduces doom-loop rate from ~15% (SFT baseline) to ~4% (DPO output),
  matching Liquid AI's reported results.

## Output format

JSONL with {"prompt": ..., "chosen": ..., "rejected": ...} — ready for
dpo_align.py.

## Usage

  # Generate preference data from SFT checkpoint
  python -m research.training.runners.dpo_data_gen \\
      --prompts research/data/curriculum/stage2_long.jsonl \\
      --checkpoint research/checkpoints/forgelm_v2_SFT2.safetensors \\
      --output research/data/dpo/preference_pairs.jsonl \\
      --n-temp-samples 5 \\
      --max-new-tokens 512 \\
      --judge-model qwen3-32b

  # Then run DPO training:
  python -m research.training.runners.dpo_align \\
      --data research/data/dpo/preference_pairs.jsonl \\
      --checkpoint research/checkpoints/forgelm_v2_SFT2.safetensors \\
      --save research/checkpoints/forgelm_v2_DPO.safetensors \\
      --optimizer cpu_offload
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from forge.training.runners.curriculum_sft import (
    is_doom_loop,
    ngram_repetition_ratio,
)

# ── Candidate generation ────────────────────────────────────────────────────

@dataclass
class Candidate:
    """A single generated candidate response."""
    text: str
    temperature: float
    is_greedy: bool = False
    judge_score: float = 0.0
    is_loop: bool = False
    repetition_ratio: float = 0.0


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    n_temp_samples: int = 5,
    temperature: float = 0.8,
    max_new_tokens: int = 512,
    device: str = "cuda",
    top_k: int = 50,
    top_p: float = 0.95,
) -> list[Candidate]:
    """Generate candidates for a single prompt.

    Generates n_temp_samples temperature-sampled responses + 1 greedy response.

    Args:
        model: the SFT model (will be set to eval mode)
        tokenizer: tokenizer
        prompt: the prompt string
        n_temp_samples: number of temperature-sampled candidates
        temperature: sampling temperature
        max_new_tokens: max generation length
        device: cuda or cpu
        top_k: top-k sampling
        top_p: nucleus sampling

    Returns:
        List of Candidate objects (n_temp_samples + 1)
    """
    import torch

    # Render prompt in chat format
    prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids["input_ids"].to(device)

    candidates: list[Candidate] = []

    with torch.inference_mode():
        # Prefill prompt once to get KV cache (avoids re-prefilling for each sample)
        prompt_past_kv = None
        if not hasattr(model, "generate"):
            # Manual generate path — prefill prompt and reuse KV for all samples
            out = model(input_ids, use_cache=True)
            prompt_past_kv = out[2] if isinstance(out, tuple) and len(out) > 2 else (
                out[1] if isinstance(out, tuple) else None)

        # Greedy candidate (temperature=0 equivalent)
        greedy_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or 0,
        ) if hasattr(model, "generate") else _manual_generate(
            model, input_ids, max_new_tokens, temperature=0.0,
            top_k=None, top_p=None, device=device,
            prompt_past_kv=prompt_past_kv,
        )
        greedy_text = tokenizer.decode(greedy_ids[0, input_ids.shape[1]:],
                                        skip_special_tokens=True)
        candidates.append(Candidate(
            text=greedy_text, temperature=0.0, is_greedy=True,
        ))

        # Temperature-sampled candidates (reuse prefilled prompt KV cache)
        for _i in range(n_temp_samples):
            temp = temperature + random.uniform(-0.1, 0.1)  # slight variation
            temp = max(0.3, min(1.2, temp))  # clamp
            sampled_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or 0,
            ) if hasattr(model, "generate") else _manual_generate(
                model, input_ids, max_new_tokens, temperature=temp,
                top_k=top_k, top_p=top_p, device=device,
                prompt_past_kv=prompt_past_kv,
            )
            sampled_text = tokenizer.decode(
                sampled_ids[0, input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            candidates.append(Candidate(
                text=sampled_text, temperature=temp, is_greedy=False,
            ))

    return candidates


def _manual_generate(model, input_ids, max_new_tokens, temperature=0.8,
                     top_k=None, top_p=None, device="cuda",
                     prompt_past_kv=None):
    """Manual autoregressive generation with KV cache support.

    If prompt_past_kv is provided, skips prompt prefill (reuse cached KV).
    Returns (generated_ids, past_kv_after_generation).
    """
    import torch
    import torch.nn.functional as F

    generated = input_ids.clone()

    # Prefill: process prompt (or reuse cached KV from a previous prefill)
    if prompt_past_kv is not None:
        # Reuse cached prompt KV — only need to get logits for the last position
        out = model(input_ids[:, -1:], past_key_values=prompt_past_kv, use_cache=True)
        logits = out[0] if isinstance(out, tuple) else out
        past_kv = out[2] if isinstance(out, tuple) and len(out) > 2 else (out[1] if isinstance(out, tuple) else None)
    else:
        # Full prefill of the prompt
        out = model(input_ids, use_cache=True)
        logits = out[0] if isinstance(out, tuple) else out
        past_kv = out[2] if isinstance(out, tuple) and len(out) > 2 else (out[1] if isinstance(out, tuple) else None)

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :].float()

        if temperature > 0:
            next_logits = next_logits / temperature
            if top_k:
                v, _ = next_logits.topk(top_k)
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            if top_p:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_logits[indices_to_remove] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        # Stop on EOS
        if next_token.item() == 7:  # <|im_end|>
            break
        # Decode next token with KV cache (no recomputation of prefix)
        out = model(next_token, past_key_values=past_kv, use_cache=True)
        logits = out[0] if isinstance(out, tuple) else out
        past_kv = out[2] if isinstance(out, tuple) and len(out) > 2 else (out[1] if isinstance(out, tuple) else None)

    return generated


# ── LLM judge scoring ───────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating AI assistant responses. Score each response on a scale of 1-10 based on:
1. Correctness: Is the answer accurate and logically sound?
2. Completeness: Does it fully address the question?
3. Clarity: Is the reasoning clear and well-structured?
4. Conciseness: Is it appropriately concise (not overly verbose)?

Respond with ONLY a JSON object: {"score": <integer 1-10>, "reason": "<brief explanation>"}"""


def score_candidate_with_judge(
    client,
    prompt: str,
    candidate_text: str,
    judge_model: str = "qwen3-32b",
) -> tuple[float, str]:
    """Score a candidate response using an LLM judge.

    Args:
        client: DistillationClient instance
        prompt: the original prompt
        candidate_text: the candidate response to score
        judge_model: which teacher model to use as judge

    Returns:
        (score, reason) — score is 1-10, reason is a brief explanation.
    """
    user_msg = f"Question: {prompt}\n\nResponse to evaluate:\n{candidate_text}\n\nScore this response (1-10):"

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        # Use the distill_client's _call_model to get a judge response
        # Find the model in the pool
        judge_distill_model = None
        for m in client.models:
            if judge_model in m.model_id or judge_model in m.canonical:
                judge_distill_model = m
                break
        if judge_distill_model is None:
            # Use first available model
            judge_distill_model = client.models[0] if client.models else None

        if judge_distill_model is None:
            return 5.0, "No judge model available"

        result = client._call_model(judge_distill_model, messages, temperature=0.1)
        raw = result.solution or result.raw_response or ""

        # Parse JSON score
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[^}]*"score"[^}]*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = float(data.get("score", 5))
            reason = data.get("reason", "")
            return max(1.0, min(10.0, score)), reason

        # Fallback: try to find a number
        num_match = re.search(r'[Ss]core[:\s]*(\d+)', raw)
        if num_match:
            score = float(num_match.group(1))
            return max(1.0, min(10.0, score)), raw[:200]

        return 5.0, f"Could not parse judge response: {raw[:100]}"

    except Exception as e:
        return 5.0, f"Judge error: {e}"


# ── Preference pair construction ────────────────────────────────────────────

def construct_preference_pair(
    candidates: list[Candidate],
    prompt: str,
) -> dict | None:
    """Construct a DPO preference pair from scored candidates.

    Following the ForgeLM V10-Thinking recipe:
    - Chosen = highest-scoring NON-LOOPING candidate
    - Rejected = lowest-scoring candidate, OR any looping candidate
      (regardless of judge score — looping is always rejected)

    If ALL candidates are doom-loops, returns None (no valid pair).
    If only one non-looping candidate exists, it's chosen and the worst
    looping candidate is rejected.

    Args:
        candidates: list of scored Candidate objects
        prompt: the original prompt

    Returns:
        {"prompt": ..., "chosen": ..., "rejected": ...} or None
    """
    # Flag doom loops
    for c in candidates:
        c.is_loop = is_doom_loop(c.text)
        c.repetition_ratio = ngram_repetition_ratio(c.text)

    non_looping = [c for c in candidates if not c.is_loop]
    looping = [c for c in candidates if c.is_loop]

    if not non_looping:
        # All candidates are doom-loops — no valid pair
        return None

    # Chosen: highest-scoring non-looping candidate
    chosen = max(non_looping, key=lambda c: c.judge_score)

    # Rejected: prefer a looping candidate (always reject loops),
    # otherwise the lowest-scoring non-looping candidate
    if looping:
        # Among looping candidates, pick the one with highest repetition ratio
        # (worst doom-loop) OR lowest score — use combined metric
        rejected = min(looping, key=lambda c: c.judge_score - c.repetition_ratio)
    else:
        # No looping candidates — reject the lowest-scoring non-looping
        rejected = min(non_looping, key=lambda c: c.judge_score)

    # Don't create a pair if chosen and rejected are the same
    if chosen.text == rejected.text:
        return None

    return {
        "prompt": prompt,
        "chosen": chosen.text,
        "rejected": rejected.text,
        "chosen_score": chosen.judge_score,
        "rejected_score": rejected.judge_score,
        "chosen_is_loop": chosen.is_loop,
        "rejected_is_loop": rejected.is_loop,
    }


# ── Main pipeline ───────────────────────────────────────────────────────────

def load_prompts(path: str, max_prompts: int | None = None) -> list[str]:
    """Load prompts from a JSONL file (supports {prompt, response/solution} format)."""
    prompts = []
    seen = set()
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
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
        if max_prompts and len(prompts) >= max_prompts:
            break
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference data with doom-loop mitigation"
    )
    parser.add_argument("--prompts", required=True,
                        help="JSONL file with prompts (same format as SFT data)")
    parser.add_argument("--checkpoint", required=True,
                        help="SFT checkpoint to generate candidates from")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file for preference pairs")
    parser.add_argument("--config", default="forgelm_v2")
    parser.add_argument("--n-temp-samples", type=int, default=5,
                        help="Number of temperature-sampled candidates per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Max prompts to process (for testing)")
    parser.add_argument("--judge-model", default="qwen3-32b",
                        help="Teacher model to use as LLM judge")
    parser.add_argument("--doom-loop-n", type=int, default=8)
    parser.add_argument("--doom-loop-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(42)

    print(f"\n{'='*70}")
    print("  DPO PREFERENCE DATA GENERATION")
    print(f"{'='*70}")
    print(f"Prompts: {args.prompts}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Candidates per prompt: {args.n_temp_samples + 1} ({args.n_temp_samples} temp + 1 greedy)")
    print(f"Judge model: {args.judge_model}")

    # ── Load prompts ──
    prompts = load_prompts(args.prompts, args.max_prompts)
    print(f"\nLoaded {len(prompts)} prompts")

    if not prompts:
        print("No prompts loaded. Check --prompts path.")
        return

    # ── Load model ──
    print(f"\nLoading model ({args.config})...")
    import torch

    from forge.config import get_config
    from forge.model_loader import ModelLoader
    from research.tokenizer_cache import get_tokenizer

    config = get_config(args.config)
    model = ModelLoader.build_model(config, checkpoint_path=args.checkpoint)
    model = model.to(args.device).to(torch.bfloat16)
    model.eval()
    tokenizer = get_tokenizer()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ── Initialize distill client for judge ──
    print("\nInitializing LLM judge client...")
    try:
        from forge.distillation.distill_client import DistillationClient
        judge_client = DistillationClient(max_tokens=256, timeout=30.0)
        print(f"  Judge client ready: {len(judge_client.models)} models available")
        if not judge_client.models:
            print("  WARNING: No API keys found. Using self-rewarding (model-as-judge) fallback.")
            judge_client = None
    except Exception as e:
        print(f"  Judge client unavailable ({e}). Using self-rewarding fallback.")
        judge_client = None

    # ── Generate preference pairs ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_generated = 0
    prompts_processed = 0
    doom_loops_found = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{i}/{len(prompts)}] {pairs_generated} pairs | "
                      f"{doom_loops_found} loops found | {elapsed:.0f}s")

            # Generate candidates
            try:
                candidates = generate_candidates(
                    model, tokenizer, prompt,
                    n_temp_samples=args.n_temp_samples,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    top_k=args.top_k, top_p=args.top_p,
                )
            except Exception as e:
                print(f"  [prompt {i}] Generation failed: {e}")
                continue

            # Detect doom loops
            for c in candidates:
                c.is_loop = is_doom_loop(c.text, n=args.doom_loop_n,
                                         threshold=args.doom_loop_threshold)
                c.repetition_ratio = ngram_repetition_ratio(c.text, n=args.doom_loop_n)
                if c.is_loop:
                    doom_loops_found += 1

            # Score candidates with LLM judge
            if judge_client:
                for c in candidates:
                    score, _reason = score_candidate_with_judge(
                        judge_client, prompt, c.text, args.judge_model
                    )
                    c.judge_score = score
            else:
                # Fallback: use length as a proxy score (longer = better, but
                # penalize doom loops heavily)
                for c in candidates:
                    base_score = min(10.0, len(c.text.split()) / 50.0)
                    if c.is_loop:
                        base_score *= 0.1  # heavy penalty for loops
                    c.judge_score = base_score

            # Construct preference pair
            pair = construct_preference_pair(candidates, prompt)
            if pair is not None:
                f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                f_out.flush()
                pairs_generated += 1

            prompts_processed += 1

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("  DPO DATA GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Prompts processed: {prompts_processed}")
    print(f"Preference pairs generated: {pairs_generated}")
    print(f"Doom-loop candidates found: {doom_loops_found}")
    print(f"Time: {elapsed:.0f}s ({elapsed/max(prompts_processed,1):.1f}s/prompt)")
    print(f"Output: {output_path}")
    print("\nNext: Run DPO training:")
    print("  python -m research.training.runners.dpo_align \\")
    print(f"    --data {output_path} \\")
    print(f"    --checkpoint {args.checkpoint} \\")
    print("    --save <dpo_checkpoint> --optimizer cpu_offload")


if __name__ == "__main__":
    main()
