"""Quick QLoRA fine-tune of ForgeLM V2 on Glaive tool-use data.

Usage: python scripts/train_tooluse_qlora.py [--max-steps N] [--lr F]
"""
import os
import sys

os.environ["FORGE_NO_COMPILE"] = "1"

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

# Build default args for tool-use QLoRA training
defaults = [
    "sft_train",
    "--config", "forgelm_v2",
    "--checkpoint", "research/checkpoints/ForgeLM_V2.safetensors",
    "--data",
    "data/sft/glaive_fc_pythonic.jsonl",
    "data/sft/nontool_general.jsonl",
    "--forge-quant",
    "--lora",
    "--no-bitnet-everywhere",
    "--lora-r", "32",
    "--lora-alpha", "64",
    "--seq-len", "512",
    "--batch-size", "1",
    "--grad-checkpoint",
    "--grad-accum", "8",
    "--max-steps", "500",
    "--lr", "1e-4",
    "--min-lr", "1e-5",
    "--save", "research/checkpoints/forgelm_v2_tooluse.safetensors",
    "--save-lora-adapter",
    "--no-disk-cache",
    "--val-every", "20",
    "--val-size", "0.05",
    "--early-stop-patience", "3",
    "--min-train-loss", "0.05",
    "--config-overrides",
    '{"use_forge_hybrid":true,"use_outro":true}',
    "--use-forge-engine",
]

# If user passed extra args, append them (they override defaults via argparse)
sys.argv = defaults + sys.argv[1:]

from forge.training.runners.sft_train import main

main()
