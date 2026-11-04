"""E2E: activate(decoding='suffix_draft') on the real LFM3 checkpoint.

Spec decode is lossless for greedy — output must match standard decode
bit-for-bit. The memory learns the emitted stream, so a repetitive
prompt should produce real draft proposals.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.engine.forge_engine import ForgeEngine

CKPT = "research/checkpoints/lfm3_1p7b_cc"
PROMPT = ("def a0(x): return x + 0\n"
          "def a1(x): return x + 1\n"
          "def a2(x): return x + 2\n"
          "def a3(x): return x + 3\n"
          "def a4(x): return x + 4\n"
          "def a5(x):")

eng = ForgeEngine.from_checkpoint(CKPT, config_name="lfm3_1p7b",
                                  device="cuda", auto_activate=False)

# baseline: standard greedy
t0 = time.time()
out_std = eng.generate(PROMPT, max_new_tokens=48, temperature=0.0)
t_std = time.time() - t0
print("STD :", repr(out_std)[:200])

# suffix_draft: memory sidecar as the draft model
mem = eng.enable_suffix_memory()
eng.activate(decoding="suffix_draft")
print("decoding:", eng.decoding.__class__.__name__,
      "draft:", type(eng.decoding.draft_model).__name__)
t0 = time.time()
out_spec = eng.generate(PROMPT, max_new_tokens=48, temperature=0.0)
t_spec = time.time() - t0
print("SPEC:", repr(out_spec)[:200])

print(f"\nstd {t_std:.2f}s | spec {t_spec:.2f}s | "
      f"match={'YES' if out_spec == out_std else 'NO'}")
print("spec:", eng.decoding.spec_report())
print("mem:", mem.info())
