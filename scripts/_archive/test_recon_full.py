"""Full reconstruction validation: error mitigation + KL calibration."""
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

model_quant = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=dtype, trust_remote_code=True).to(device).eval()
n = quantize_model_nanoquant(model_quant, rank=128, admm_iters=200, verbose=True)
mem = estimate_r48_memory(model_quant)
print(f"Quantized: {n} layers, {mem['total_mb']:.1f} MB "
      f"({mem['avg_eff_bits']:.3f} bits)")

ppl_before = compute_ppl(model_quant, input_ids)
print(f"PPL before recon: {ppl_before:.4f}")

recon = BlockReconstructor(
    model_orig=model_orig, model_quant=model_quant,
    calibration_data=calib_ids, device=device)

# Test 1: error mitigation with scales only
t0 = time.time()
results = recon.reconstruct_with_error_mitigation(
    n_iters=50, lr=0.05, optimize_binary=False, verbose=False)
t1 = time.time()
ppl_after_em = compute_ppl(model_quant, input_ids)
print(f"After error_mitigation (scales only): PPL={ppl_after_em:.4f}, "
      f"time={t1-t0:.1f}s")

# Test 2: KL calibration
t0 = time.time()
kl_loss = recon.calibrate_kl(n_iters=20, lr=0.005, verbose=False)
t1 = time.time()
ppl_after_kl = compute_ppl(model_quant, input_ids)
print(f"After KL calibration: PPL={ppl_after_kl:.4f}, "
      f"KL loss={kl_loss:.6f}, time={t1-t0:.1f}s")

print(f"\nSummary: fp16={fp16_ppl:.2f}, before={ppl_before:.2f}, "
      f"after_EM={ppl_after_em:.2f}, after_KL={ppl_after_kl:.2f}")
