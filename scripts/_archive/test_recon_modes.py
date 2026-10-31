"""Test different reconstruction modes on Qwen 2.5 0.5B."""
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

# Test different ranks
for rank in [128, 256, 512]:
    print(f"\n{'='*60}")
    print(f"Rank {rank}, ADMM 200 iters")
    print(f"{'='*60}")

    model_quant = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True).to(device).eval()
    n = quantize_model_nanoquant(model_quant, rank=rank, admm_iters=200,
                                 verbose=False)
    mem = estimate_r48_memory(model_quant)
    ppl_base = compute_ppl(model_quant, input_ids)
    print(f"  Base: {n} layers, {mem['total_mb']:.1f} MB "
          f"({mem['avg_eff_bits']:.3f} bits), PPL={ppl_base:.2f}")

    # KL calibration only (no block recon)
    recon = BlockReconstructor(
        model_orig=model_orig, model_quant=model_quant,
        calibration_data=calib_ids, device=device)

    t0 = time.time()
    try:
        kl_loss = recon.calibrate_kl(n_iters=20, lr=0.001, verbose=False)
        ppl_kl = compute_ppl(model_quant, input_ids)
        print(f"  After KL only: PPL={ppl_kl:.2f}, KL={kl_loss:.4f}, "
              f"time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  KL failed: {e}")

    print(f"  Delta: base={ppl_base-fp16_ppl:+.2f}, kl={ppl_kl-fp16_ppl:+.2f}")

    del model_quant, recon
    if device == 'cuda':
        torch.cuda.empty_cache()
