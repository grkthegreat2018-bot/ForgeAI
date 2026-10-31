"""Verify per-sequence temp/seed produces different outputs in BatchedDecoding."""
import sys

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from forge.engine.batched_decoding import BatchedDecoding

device = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()

# SAME prompt, 8 different (temp, seed) combos
prompt = "The future of artificial intelligence depends on"
ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
prompts = [ids] * 8

temps = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
seeds = [42, 43, 44, 45, 46, 47, 48, 49]

decoder = BatchedDecoding(eos_token_id=tok.eos_token_id)
outputs = decoder.generate_batch(
    model, prompts,
    max_tokens_list=[50] * 8,
    temperatures=temps,
    top_ps=[0.9] * 8,
    top_k_list=[80] * 8,
    seed_list=seeds,
    tokenizer=tok,
)

print(f'Same prompt: "{prompt}"')
print("8 different (temp, seed) combos:\n")
texts = []
for i, out in enumerate(outputs):
    text = tok.decode(out[0], skip_special_tokens=True)
    texts.append(text)
    print(f"  [{i}] temp={temps[i]}, seed={seeds[i]}:")
    print(f"      {text}")
    print()

unique = len(set(texts))
print(f"Unique outputs: {unique}/8")
if unique == 8:
    print("PASS: All 8 outputs different — per-sequence temp/seed works!")
elif unique > 1:
    print(f"PARTIAL: {unique} unique outputs (some overlap)")
else:
    print("FAIL: All outputs identical — per-sequence settings not working!")
