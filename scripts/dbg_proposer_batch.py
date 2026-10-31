"""Reproduce the evolve proposer path: real DB prompt through generate_batch
with the exact evolve sampling params, on the _fq engine."""
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "research" / "checkpoints" / "lfm3_1p7b_cc_fq"

from forge.engine.forge_engine import ForgeEngine

cur = sqlite3.connect(str(ROOT / "forge_evolve.db")).cursor()
prompt = cur.execute(
    "SELECT prompt FROM code_programs WHERE prompt IS NOT NULL "
    "AND LENGTH(prompt) > 500 ORDER BY id DESC LIMIT 1").fetchone()[0]

IM_START = chr(60) + "|im_start|>"
IM_END = chr(60) + "|im_end|>"
wrapped = IM_START + "user\n" + prompt + IM_END + "\n" + IM_START + "assistant\n"
print("prompt tail:", repr(wrapped[-120:]))

eng = ForgeEngine.from_checkpoint(
    str(CKPT), config_name="lfm3_1p7b",
    kv_cache="standard", use_compile=False,
    use_breakable_cuda_graph=False, use_block_fusion=False,
    use_adaptive_spec=False, use_suffix_spec=False,
    use_faser=False, use_lookahead_gate=False,
    use_hotprefix=False, use_pod_attention=False,
    use_hybrid_prefill=False)

rnd = random.Random(7)
n = 4
temps = [0.6 + 0.4 * i / max(1, n - 1) for i in range(n)]
reps = [1.0 + 0.15 * rnd.random() for _ in range(n)]
seeds = [rnd.randrange(1 << 60) for _ in range(n)]
outs = eng.generate_batch(
    [wrapped] * n, max_new_tokens=96,
    temperature=0.8, top_p=0.95, top_k=80,
    temperatures=temps, repetition_penalties=reps, seeds=seeds,
    stops=[[IM_END]] * n, skip_special_tokens=True)
for i, o in enumerate(outs):
    t = o if isinstance(o, str) else o.get("text", "")
    print(f"--- row {i} (t={temps[i]:.2f} rp={reps[i]:.2f}) ---")
    print(repr(t[:200]))
    print()
sys.exit(0)
