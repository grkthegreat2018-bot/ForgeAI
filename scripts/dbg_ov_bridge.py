"""Bridge smoke test — OvEngine through LLMProposer's engine dispatch."""
import sys, time
sys.path.insert(0, r"D:\windsurf\ForgeAI")
IM_END = "<" + "|im_end|" + ">"
from forge.evolution.code_evolve import (CodeEvolveConfig, LLMProposer,
                                         OvEngine)

CKPT = r"research\checkpoints\lfm3_1p7b_cc_fq"

cfg = CodeEvolveConfig(
    target="suffix_mem", model_id=CKPT, lora_root=None,
    proposer_backend="ov", proposer_ov_url="http://127.0.0.1:8790")
p = LLMProposer(cfg, db=None)

kind, eng, lock = p._engine()
print("kind:", kind, "| engine:", type(eng).__name__, "| dev:", eng.device)
print("eos:", getattr(eng.tokenizer, "eos_token_id", None))

PROMPTS = [
    ("Improve this function for speed. Reply with ONLY a ```python fenced "
     "block containing the complete improved function. Keep the same name "
     "and signature.\n\n```python\n"
     "def top_k_mask(logits, k):\n"
     "    top = logits.topk(k, dim=-1).values[..., -1:]\n"
     "    return logits < top\n```"),
    ("Write a Python function `squares(ss)` returning x**2 for each x. "
     "Reply with ONLY a ```python fenced block."),
]
wrapped = [p._wrap(x) for x in PROMPTS]
print("--- wrapped[0][:200]:", wrapped[0][:200].replace("\n", "\\n"))

t0 = time.time()
raws = eng.generate_batch(
    wrapped, max_new_tokens=160,
    temperature=0.8, top_p=0.95, top_k=80,
    temperatures=[0.9, 0.8], repetition_penalties=[1.05, 1.1],
    seeds=[1, 2],
    stops=[[IM_END], [IM_END]], skip_special_tokens=True)
dt = time.time() - t0
for i, r in enumerate(raws):
    print(f"--- raw[{i}] ({dt:.1f}s total): {r[:220]!r}")