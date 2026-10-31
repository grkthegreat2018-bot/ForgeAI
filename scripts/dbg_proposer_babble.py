"""Is the proposer babbling? Probes on _fq vs fp16, raw vs ChatML prompts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "research" / "checkpoints" / (
    "lfm3_1p7b_cc_fq" if len(sys.argv) < 2 or sys.argv[1] != "fp16"
    else "lfm3_1p7b_cc")

from forge.engine.forge_engine import ForgeEngine

eng = ForgeEngine.from_checkpoint(
    str(CKPT), config_name="lfm3_1p7b",
    kv_cache="standard", use_compile=False,
    use_breakable_cuda_graph=False, use_block_fusion=False,
    use_adaptive_spec=False, use_suffix_spec=False,
    use_faser=False, use_lookahead_gate=False,
    use_hotprefix=False, use_pod_attention=False,
    use_hybrid_prefill=False)

IM_START = chr(60) + "|im_start|>"
IM_END = chr(60) + "|im_end|>"
prompts = {
    "raw": "def add(a, b):\n    ",
    "chatml": (IM_START + "user\nWrite Python: def add(a, b) returns a+b."
               + IM_END + "\n" + IM_START + "assistant\n"),
}
for pname, prompt in prompts.items():
    for name, kw in [
        ("greedy", dict(temperature=0.0, top_k=1)),
        ("t=0.8/p=.95/k=80", dict(temperature=0.8, top_p=0.95, top_k=80)),
    ]:
        out = eng.generate(prompt, max_new_tokens=48,
                           repetition_penalty=1.0, finish_sentence=False,
                           **kw)
        text = out if isinstance(out, str) else out.get("text", "")
        print(f"=== {pname} {name} ===")
        print(repr(text[:220]))
        print()
sys.exit(0)
