"""Tune KL calibration: more iterations, higher LR, multiple rounds."""
import sys
import time

import torch

sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer

from forge.engine.quant.block_recon import BlockReconstructor
from forge.engine.quant.novel_quant_r48 import estimate_r48_memory, quantize_model_nanoquant

device = 'cuda'
model_id = 'Qwen/Qwen2.5-0.5B'
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dtype = torch.bfloat16

model_orig = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=dtype, trust_remote_code=True).to(device).eval()
test_text = ("Quantization is a technique used to reduce the memory footprint "
             "of large language models by representing weights with lower "
             "precision numbers.")
input_ids = tok(test_text, return_tensors="pt", truncation=True,
                max_length=128)["input_ids"].to(device)

calib_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can be compressed using quantization.",
    "Neural networks process information through layers of neurons.",
    "Transformers use attention mechanisms to weigh input importance.",
    "Gradient descent optimizes model parameters iteratively.",
    "Deep learning has revolutionized natural language processing.",
    "Model compression enables deployment on resource-constrained devices.",
    "Post-training quantization reduces model size without retraining.",
]
calib_ids = tok(calib_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=128)["input_ids"].to(device)


def compute_ppl(model, input_ids):
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1))
        return torch.exp(loss).item()


fp16_ppl = compute_ppl(model_orig, input_ids)
print(f"FP16 PPL: {fp16_ppl:.4f}")

# Test rank 512 with multiple KL rounds
rank = 512
print(f"\n{'='*60}")
print(f"Rank {rank}, ADMM 200 iters, multiple KL rounds")
print(f"{'='*60}")

model_quant = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=dtype, trust_remote_code=True).to(device).eval()
n = quantize_model_nanoquant(model_quant, rank=rank, admm_iters=200,
                             verbose=False)
mem = estimate_r48_memory(model_quant)
ppl_base = compute_ppl(model_quant, input_ids)
print(f"  Base: {mem['total_mb']:.1f} MB ({mem['avg_eff_bits']:.3f} bits), "
      f"PPL={ppl_base:.2f}")

recon = BlockReconstructor(
    model_orig=model_orig, model_quant=model_quant,
    calibration_data=calib_ids, device=device)

# Multiple KL rounds with increasing LR
for round_i in range(3):
    lr = 0.001 * (10 ** round_i)  # 0.001, 0.01, 0.1
    t0 = time.time()
    kl_loss = recon.calibrate_kl(n_iters=50, lr=lr, verbose=False)
    ppl = compute_ppl(model_quant, input_ids)
    print(f"  KL round {round_i+1} (lr={lr}): PPL={ppl:.2f}, "
          f"KL={kl_loss:.4f}, time={time.time()-t0:.1f}s")

# Also try standard block recon + KL
print("\n  Now trying standard block recon + KL...")
recon2 = BlockReconstructor(
    model_orig=model_orig, model_quant=model_quant,
    calibration_data=calib_ids, device=device)
t0 = time.time()
results = recon2.reconstruct(n_iters=30, lr=0.01, verbose=False)
ppl_after_recon = compute_ppl(model_quant, input_ids)
print(f"  After standard recon: PPL={ppl_after_recon:.2f}, "
      f"time={time.time()-t0:.1f}s")

kl_loss = recon2.calibrate_kl(n_iters=50, lr=0.01, verbose=False)
ppl_after_kl = compute_ppl(model_quant, input_ids)
print(f"  After KL: PPL={ppl_after_kl:.2f}, KL={kl_loss:.4f}")

print(f"\nSummary: fp16={fp16_ppl:.2f}, base={ppl_base:.2f}, "
      f"best_KL={ppl:.2f}, recon+KL={ppl_after_kl:.2f}")
