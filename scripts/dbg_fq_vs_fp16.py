"""Decisive fp16-vs-_fq proposer probe."""
import sys, torch
sys.path.insert(0, ".")
from forge.engine.forge_engine import ForgeEngine

SEED = "def top_k_mask(logits, k):\n    top = logits.topk(k, dim=-1).values[..., -1:]\n    return logits < top\n"
USER = ("Improve this function for speed. Reply with ONLY a ```python fenced block "
        "containing the complete improved function. Keep the same name and signature.\n\n"
        "```python\n" + SEED + "```")
IM = chr(60) + "|im_start|" + chr(62)
IE = chr(60) + "|im_end|" + chr(62)
CHATML = IM + "user\n" + USER + IE + "\n" + IM + "assistant\n"

def probe(ckpt, tag):
    print("=== " + tag + ": " + ckpt)
    eng = ForgeEngine.load(ckpt, device="cuda")
    cases = [("greedy", dict(temperature=0.0)),
             ("t08", dict(temperature=0.8, top_p=0.9, seed=7)),
             ("t10", dict(temperature=1.0, top_p=0.95, seed=11))]
    for name, kw in cases:
        out = eng.generate(CHATML, max_tokens=160, **kw)
        txt = out if isinstance(out, str) else out[0]
        print("--" + name + ": " + repr(txt[:220]))
    del eng
    torch.cuda.empty_cache()

probe(sys.argv[1], "A")
probe(sys.argv[2], "B")
