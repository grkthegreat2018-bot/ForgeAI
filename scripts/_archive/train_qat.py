"""QAT training with self-distillation, using ForgeAI's training utilities.

Reuses ForgeAI's:
  - configure_optimizer (fused AdamW, Muon, CPU offload)
  - get_lr (cosine warmup schedule)
  - oom_guard (OOM recovery)
  - grad_accum_for_effective_batch
  - BatchedDecoding (mass data generation with per-sequence temp/seed)

Pipeline:
  1. FP16 teacher generates diverse data via BatchedDecoding
  2. Student model converted to QAT (NanoQuant STE or BitNet ternary)
  3. Training: KL distillation + CE loss, batched for GPU utilization
  4. Bake to inference mode, evaluate PPL

Usage:
  python scripts/train_qat.py --model Qwen/Qwen2.5-0.5B --method nanoquant \
      --rank 128 --qat-steps 500 --lr 1e-4 --seq-len 256
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from forge.engine.batched_decoding import BatchedDecoding
from forge.engine.quant.novel_quant_r48 import (
    bake_qat_model,
    convert_model_to_nanoquant_qat,
    estimate_r48_memory,
)
from forge.training.training_utils import (
    configure_optimizer,
    get_lr,
    oom_guard,
)


def compute_ppl(model, input_ids):
    """Compute perplexity on a sequence."""
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1))
        return torch.exp(loss).item()


@torch.no_grad()
def generate_training_data(teacher_model, tokenizer, n_samples=256,
                           max_new_tokens=128, min_prompt_len=16,
                           device="cuda"):
    """Generate training data using FP16 teacher + ForgeEngine BatchedDecoding.

    Uses diverse per-sequence (temperature, seed) for maximum output diversity.
    Auto-detects max batch size from available VRAM.
    """
    prompts = [
        "The future of artificial intelligence depends on",
        "In machine learning, gradient descent is used to",
        "Quantization reduces model size by",
        "The transformer architecture revolutionized",
        "Natural language processing has evolved through",
        "Deep learning models require large amounts of",
        "Memory efficiency in neural networks is achieved by",
        "The key insight behind attention mechanisms is",
        "Model compression techniques include",
        "Transfer learning allows models to",
        "The development of language models has",
        "Optimization algorithms for neural networks",
        "Hardware acceleration for inference requires",
        "Knowledge distillation transfers knowledge from",
        "Sparse attention reduces computational cost by",
        "The training of large language models involves",
        "Efficient inference on edge devices requires",
        "Model pruning removes less important",
        "Low-rank decomposition approximates weight matrices by",
        "The balance between model quality and size",
        "Recent advances in quantization have shown",
        "Binary neural networks use weights constrained to",
        "The straight-through estimator enables training of",
        "Post-training quantization can degrade quality when",
        "Quantization-aware training addresses the limitations of",
        "The information bottleneck in compression theory",
        "Neural network weights often follow a distribution that",
        "Hessian-based preconditioning improves optimization by",
        "Block-level reconstruction refines quantized models by",
        "The tradeoff between compression ratio and quality",
        "A neural network learns by adjusting its parameters through",
        "The gradient of a loss function indicates the direction",
        "Backpropagation computes gradients efficiently by",
        "Stochastic gradient descent updates weights using",
        "The learning rate controls how much the model adjusts",
        "Regularization techniques prevent overfitting by",
        "Dropout randomly deactivates neurons during training to",
        "Batch normalization stabilizes training by",
        "The vanishing gradient problem occurs when",
        "Residual connections help mitigate the vanishing gradient by",
        "Attention mechanisms allow models to focus on",
        "Multi-head attention captures different representation",
        "Positional encoding provides sequence order information by",
        "The feed-forward network in each transformer block",
        "Layer normalization normalizes activations across",
        "The softmax function converts logits into",
        "Cross-entropy loss measures the difference between",
        "Perplexity is defined as the exponential of",
        "Beam search explores multiple hypotheses by",
        "Sampling with temperature controls the sharpness of",
        "Top-k sampling restricts the candidate pool to",
        "Nucleus sampling selects tokens from the smallest set",
        "Repetition penalty discourages generating the same",
        "The context window limits how many tokens the model",
        "KV caching stores key-value pairs from previous",
        "Flash attention optimizes the attention computation by",
        "Mixture of experts routes tokens to specialized",
        "LoRA adds low-rank trainable adapters to frozen",
        "Quantization-aware training simulates integer arithmetic",
        "The straight-through estimator approximates the gradient of",
    ]

    # Tokenize prompts, cycle to reach n_samples
    tokenized = []
    while len(tokenized) < n_samples:
        for prompt in prompts:
            if len(tokenized) >= n_samples:
                break
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            if ids.shape[1] >= min_prompt_len:
                tokenized.append(ids)

    # Diverse per-sequence settings
    temp_cycle = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    temperatures = [temp_cycle[i % len(temp_cycle)] for i in range(n_samples)]
    seeds = [42 + i for i in range(n_samples)]
    top_ks = [80] * n_samples

    # Auto-detect batch size from VRAM
    if device == "cuda":
        free_vram = torch.cuda.mem_get_info()[0] / 1e9
        batch_size = int(free_vram * 0.7 / 0.003)  # ~3MB per seq KV
        batch_size = max(batch_size, 16)
        batch_size = min(batch_size, n_samples)
    else:
        batch_size = 16

    print(f"  Batch size: {batch_size} (auto from VRAM)")

    decoder = BatchedDecoding(eos_token_id=tokenizer.eos_token_id)
    teacher_model.eval()
    all_outputs = []

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_prompts = tokenized[batch_start:batch_end]
        n_batches = (n_samples + batch_size - 1) // batch_size
        print(f"  Generating batch {batch_start//batch_size + 1}/{n_batches} "
              f"({len(batch_prompts)} seqs)...")

        outputs = decoder.generate_batch(
            teacher_model, batch_prompts,
            max_tokens_list=[max_new_tokens] * len(batch_prompts),
            temperatures=temperatures[batch_start:batch_end],
            top_ps=[0.9] * len(batch_prompts),
            top_k_list=top_ks[batch_start:batch_end],
            seed_list=seeds[batch_start:batch_end],
            tokenizer=tokenizer,
        )
        all_outputs.extend(outputs)

    return all_outputs


def pack_batches(samples, seq_len, train_batch_size, device):
    """Pack samples into left-padded batches for training."""
    samples = [s[:, :seq_len] for s in samples if s.shape[1] >= 16]
    train_batches = []
    for i in range(0, len(samples), train_batch_size):
        chunk = samples[i:i+train_batch_size]
        max_len = max(s.shape[1] for s in chunk)
        B = len(chunk)
        padded = torch.zeros(B, max_len, dtype=chunk[0].dtype, device=device)
        attn_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for j, s in enumerate(chunk):
            L = s.shape[1]
            padded[j, -L:] = s[0]
            attn_mask[j, -L:] = True
        train_batches.append((padded, attn_mask))
    return train_batches


def train_qat_step(student, teacher, batch_ids, attn_mask, optimizer,
                   lr, distill_weight, ce_weight, temperature):
    """One QAT training step using ForgeAI-style loss computation.

    Loss = ce_weight * CE(student, tokens) + distill_weight * KL(student || teacher)
    """
    student.train()
    teacher.eval()

    # Teacher logits (no grad)
    with torch.no_grad():
        teacher_out = teacher(batch_ids, attention_mask=attn_mask)
        teacher_logits = teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out[0]
        teacher_logits = teacher_logits.detach()

    # Student forward
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=batch_ids.is_cuda):
        student_out = student(batch_ids, attention_mask=attn_mask)
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]

    # Shift for next-token prediction
    shift_student = student_logits[..., :-1, :].contiguous().float()
    shift_teacher = teacher_logits[..., :-1, :].contiguous().float()
    shift_labels = batch_ids[..., 1:].contiguous()

    # CE loss (standard LM)
    ce_loss = F.cross_entropy(
        shift_student.view(-1, shift_student.size(-1)),
        shift_labels.view(-1))

    # KL distillation loss (soft targets)
    student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    kl_loss = kl_loss * (temperature ** 2)

    total_loss = ce_weight * ce_loss + distill_weight * kl_loss

    # Set LR (ForgeAI-style manual schedule)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), ce_loss.item(), kl_loss.item()


def main():
    parser = argparse.ArgumentParser(description="QAT with ForgeAI training utilities")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="nanoquant",
                        choices=["nanoquant", "bitnet"])
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--admm-iters", type=int, default=50,
                        help="ADMM iters (only if quick_init=False)")
    parser.add_argument("--quick-init", type=int, default=1,
                        help="1=SVD fast init (default), 0=ADMM init")
    parser.add_argument("--qat-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--optimizer", default="fused",
                        choices=["fused", "bnb", "lion", "muon_sf", "cpu_offload"])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    device = args.device
    model_id = args.model

    print(f"\n{'='*70}")
    print(f"QAT Training — {model_id}")
    print(f"Method: {args.method}, Optimizer: {args.optimizer}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load teacher (FP16)
    print("\nLoading FP16 teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    test_text = ("Quantization is a technique used to reduce the memory footprint "
                 "of large language models by representing weights with lower "
                 "precision numbers.")
    test_ids = tokenizer(test_text, return_tensors="pt", truncation=True,
                         max_length=128)["input_ids"].to(device)

    fp16_ppl = compute_ppl(teacher, test_ids)
    print(f"FP16 teacher PPL: {fp16_ppl:.4f}")

    # Load + convert student
    print(f"\nLoading student for {args.method} QAT...")
    student = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True).to(device).eval()

    if args.method == "nanoquant":
        n = convert_model_to_nanoquant_qat(
            student, rank=args.rank, admm_iters=args.admm_iters,
            quick_init=bool(args.quick_init), verbose=True)
        mem = estimate_r48_memory(student)
        print(f"Converted: {n} layers, ~{mem['total_mb']:.1f} MB "
              f"({mem['avg_eff_bits']:.3f} bits)")
    elif args.method == "bitnet":
        from forge.keys.quantization.bitnet_b158_key import BitNetLinear
        n = 0
        for name, module in list(student.named_modules()):
            if isinstance(module, nn.Linear) and type(module).__name__ == "nn.Linear":
                if any(s in name for s in ["lm_head", "embed"]):
                    continue
                parent = student
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                bnl = BitNetLinear(module.in_features, module.out_features,
                                   bias=module.bias is not None,
                                   quantize=True, force_quant=True,
                                   learned_scale=True)
                bnl.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    bnl.bias.data = module.bias.data.clone()
                with torch.no_grad():
                    bnl.qscale.data = module.weight.data.abs().mean().clamp(min=1e-8) / 0.7
                setattr(parent, parts[-1], bnl)
                n += 1
        print(f"  [BitNetQAT] {n} layers converted")

    student.eval()
    ppl_before = compute_ppl(student, test_ids)
    print(f"PPL before QAT: {ppl_before:.4f}")

    # Generate self-distillation data
    print(f"\nGenerating {args.n_samples} self-distillation samples...")
    t0 = time.time()
    samples = generate_training_data(
        teacher, tokenizer, n_samples=args.n_samples,
        max_new_tokens=args.gen_tokens, device=device)
    print(f"Generated {len(samples)} samples in {time.time()-t0:.1f}s")

    # Pack into training batches (auto-size from VRAM)
    if device == "cuda":
        free_vram = torch.cuda.mem_get_info()[0] / 1e9
        train_batch_size = int(free_vram * 0.4 / 0.05)  # 40% free, ~50MB/seq
        train_batch_size = max(train_batch_size, 8)
        train_batch_size = min(train_batch_size, len(samples))
    else:
        train_batch_size = 8
    print(f"Training batch size: {train_batch_size}")

    train_batches = pack_batches(samples, args.seq_len, train_batch_size, device)
    print(f"Packed into {len(train_batches)} training batches")

    # Configure optimizer using ForgeAI's utility
    optimizer = configure_optimizer(
        student, max_lr=args.lr, weight_decay=args.weight_decay,
        optimizer_name=args.optimizer)

    # Training loop with ForgeAI's LR schedule + OOM guard
    print(f"\n{'='*70}")
    print(f"QAT training ({args.qat_steps} steps, lr={args.lr}, "
          f"opt={args.optimizer})")
    print(f"{'='*70}")

    best_ppl = ppl_before
    step = 0
    t0 = time.time()

    while step < args.qat_steps:
        for batch_ids, attn_mask in train_batches:
            if step >= args.qat_steps:
                break

            lr = get_lr(step, args.qat_steps, args.lr, args.min_lr,
                        args.warmup_steps)

            with oom_guard(device, skip=True, label=f"step {step}"):
                total_loss, ce_loss, kl_loss = train_qat_step(
                    student, teacher, batch_ids, attn_mask, optimizer,
                    lr, args.distill_weight, args.ce_weight, args.temperature)

            if hasattr(oom_guard, 'skipped') and oom_guard.skipped:
                continue

            step += 1

            if step % 10 == 0 or step == 1:
                elapsed = time.time() - t0
                print(f"  Step {step:4d}/{args.qat_steps}: "
                      f"loss={total_loss:.4f} (ce={ce_loss:.4f}, kl={kl_loss:.4f}) "
                      f"lr={lr:.2e} [{elapsed:.1f}s, {step/elapsed:.1f} step/s]")

            if step % args.eval_interval == 0 or step == args.qat_steps:
                student.eval()
                ppl = compute_ppl(student, test_ids)
                student.train()
                improved = ppl < best_ppl
                if improved:
                    best_ppl = ppl
                marker = " *** BEST" if improved else ""
                print(f"  Step {step}: PPL={ppl:.4f} "
                      f"(best={best_ppl:.4f}, fp16={fp16_ppl:.4f}){marker}")

    elapsed = time.time() - t0
    print(f"\nQAT complete: {step} steps in {elapsed:.1f}s "
          f"({elapsed/max(step,1):.2f}s/step)")

    # Final eval
    student.eval()
    ppl_after = compute_ppl(student, test_ids)
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"FP16 teacher PPL:    {fp16_ppl:.4f}")
    print(f"PPL before QAT:      {ppl_before:.4f} (delta={ppl_before-fp16_ppl:+.4f})")
    print(f"PPL after QAT:       {ppl_after:.4f} (delta={ppl_after-fp16_ppl:+.4f})")
    print(f"Best PPL during QAT: {best_ppl:.4f}")
    print(f"Improvement:         {ppl_before - ppl_after:.4f} PPL points")
    print(f"QAT steps:           {step}")
    print(f"Training time:       {elapsed:.1f}s")

    # Bake (NanoQuant only)
    if args.method == "nanoquant" and args.save:
        print("\nBaking QAT model to inference mode...")
        n_baked = bake_qat_model(student, verbose=True)
        ppl_baked = compute_ppl(student, test_ids)
        print(f"PPL after baking:    {ppl_baked:.4f}")
        mem = estimate_r48_memory(student)
        print(f"Memory: {mem['total_mb']:.1f} MB ({mem['avg_eff_bits']:.3f} bits)")
        torch.save(student.state_dict(), args.save)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
