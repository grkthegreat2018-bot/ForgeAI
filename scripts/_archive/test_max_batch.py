"""Find max batch size for BatchedDecoding on Qwen 0.5B with 12GB VRAM."""
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

prompt = "The future of artificial intelligence depends on"
ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)

decoder = BatchedDecoding(eos_token_id=tok.eos_token_id)

# Model takes ~1GB. 12GB total. Test increasing batch sizes.
for batch_size in [256, 512, 1024, 2048]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    prompts = [ids] * batch_size
    try:
        outputs = decoder.generate_batch(
            model, prompts,
            max_tokens_list=[64] * batch_size,
            temperatures=[0.8] * batch_size,
            top_ps=[0.9] * batch_size,
            top_k_list=[80] * batch_size,
            seed_list=[42 + i for i in range(batch_size)],
            tokenizer=tok,
        )
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"Batch {batch_size:4d}: OK, peak VRAM={peak:.2f} GB, "
              f"outputs={len(outputs)}")
        del outputs
    except torch.cuda.OutOfMemoryError:
        print(f"Batch {batch_size:4d}: OOM")
        torch.cuda.empty_cache()
        break
    except Exception as e:
        print(f"Batch {batch_size:4d}: FAILED: {e}")
        break
