"""Generate self-distillation data from FP16 teacher, save as JSONL for sft_train."""
import json
import sys
import time

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from forge.engine.batched_decoding import BatchedDecoding

device = "cuda"
model_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

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

n_samples = 256
tokenized = []
while len(tokenized) < n_samples:
    for prompt in prompts:
        if len(tokenized) >= n_samples:
            break
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        tokenized.append(ids)

temp_cycle = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
temperatures = [temp_cycle[i % 8] for i in range(n_samples)]
seeds = [42 + i for i in range(n_samples)]

free_vram = torch.cuda.mem_get_info()[0] / 1e9
batch_size = min(int(free_vram * 0.7 / 0.003), n_samples)
print(f"Batch size: {batch_size}")

decoder = BatchedDecoding(eos_token_id=tok.eos_token_id)
all_outputs = []
t0 = time.time()

for batch_start in range(0, n_samples, batch_size):
    batch_end = min(batch_start + batch_size, n_samples)
    batch_prompts = tokenized[batch_start:batch_end]
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(f"  Batch {batch_start//batch_size + 1}/{n_batches} ({len(batch_prompts)} seqs)...")

    outputs = decoder.generate_batch(
        model, batch_prompts,
        max_tokens_list=[96] * len(batch_prompts),
        temperatures=temperatures[batch_start:batch_end],
        top_ps=[0.9] * len(batch_prompts),
        top_k_list=[80] * len(batch_prompts),
        seed_list=seeds[batch_start:batch_end],
        tokenizer=tok,
    )
    all_outputs.extend(outputs)

print(f"Generated {len(all_outputs)} samples in {time.time()-t0:.1f}s")

# Save as JSONL in sft_train format: {"messages": [...]}
out_path = "data/qat_self_distill.jsonl"
import os

os.makedirs("data", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for i, out in enumerate(all_outputs):
        text = tok.decode(out[0], skip_special_tokens=True)
        prompt = tok.decode(tokenized[i][0], skip_special_tokens=True)
        response = text[len(prompt):].strip()
        if not response:
            continue
        entry = {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Saved to {out_path}")
