"""Wrapper to run sft_train QAT with proper output flushing."""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONPATH"] = "."
os.environ["PYTHONIOENCODING"] = "utf-8"
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

cmd = [
    sys.executable, "-u", "-m", "forge.training.runners.sft_train",
    "--hf-model", "Qwen/Qwen2.5-0.5B",
    "--data", "data/qat_self_distill.jsonl",
    "--nanoquant-qat",
    "--no-bitnet-everywhere",
    "--nanoquant-rank", "128",
    "--nanoquant-quick-init", "1",
    "--max-steps", "500",
    "--lr", "1e-3",
    "--batch-size", "16",
    "--seq-len", "256",
    "--warmup-steps", "20",
    "--optimizer", "fused",
    "--no-lora",
    "--no-grad-checkpoint",
    "--no-use-forge-engine",
    "--no-async-prefetch",
    "--no-pack-sequences",
    "--no-disk-cache",
]
print(f"Running: {' '.join(cmd)}", flush=True)
proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                      errors="replace", cwd=os.getcwd())
print("=== STDOUT ===", flush=True)
print(proc.stdout, flush=True)
print("=== STDERR ===", flush=True)
print(proc.stderr[-3000:] if len(proc.stderr) > 3000 else proc.stderr, flush=True)
print(f"Exit code: {proc.returncode}", flush=True)
