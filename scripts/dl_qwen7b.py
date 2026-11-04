"""Download Qwen2.5-Coder-7B-Instruct via hf_transfer (multipart)."""
import os
import sys

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download  # noqa: E402

p = snapshot_download(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
print("DONE", p)
sys.stdout.flush()
