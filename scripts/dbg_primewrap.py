"""Verify the primed+ChatML wrap: prompt ends mid-signature inside a fence,
assistant turn gets the prime, completion should continue as code."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "research" / "checkpoints" / "lfm3_1p7b_cc_fq"

from forge.engine.forge_engine import ForgeEngine

IMS = chr(60) + "|im_start|>"
IME = chr(60) + "|im_end|>"

# simulated primed evolve prompt: instructions + seed + primed fence/sig
prompt = (
    "# Rewrite this Python program to be much better.\n"
    "# Direction: ms_per_call MINIMIZE\n"
    "SEED:\n```python\n"
    "def square(xs):\n    return [x * x for x in xs]\n```\n"
    "Reply with ONLY one ```python code block.\n"
    "```python\ndef square("
)

# new primed wrap: instruction in user, prime in assistant turn
i = prompt.rfind("```python")
wrapped = (IMS + "user\n" + prompt[:i] + IME + "\n" + IMS
           + "assistant\n" + prompt[i:])

eng = ForgeEngine.from_checkpoint(
    str(CKPT), config_name="lfm3_1p7b",
    kv_cache="standard", use_compile=False,
    use_breakable_cuda_graph=False, use_block_fusion=False,
    use_adaptive_spec=False, use_suffix_spec=False,
    use_faser=False, use_lookahead_gate=False,
    use_hotprefix=False, use_pod_attention=False,
    use_hybrid_prefill=False)

for name, kw in [("greedy", dict(temperature=0.0, top_k=1)),
                 ("t=0.8/p=.95", dict(temperature=0.8, top_p=0.95, top_k=80))]:
    out = eng.generate(wrapped, max_new_tokens=80,
                       repetition_penalty=1.0, finish_sentence=False, **kw)
    t = out if isinstance(out, str) else out.get("text", "")
    print(f"=== {name} ===")
    print(repr(t[:260]))
    print()
sys.exit(0)
