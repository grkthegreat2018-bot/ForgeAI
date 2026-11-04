"""Quick chat-completion latency test on CF/OpenRouter free lanes."""
import json
import subprocess
import sys
import time

KEYS = {}
for line in open(".env", encoding="utf-8", errors="replace"):
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        KEYS[k.strip()] = v.strip().strip('"').strip("'")

PROMPT = ("Output ONLY one ```python code block implementing a function "
          "sum_pair(a, b) that returns a + b. No prose.")

CF = "https://api.cloudflare.com/client/v4/accounts/" + \
    KEYS.get("CLOUDFLARE_ACCOUNT_ID", "") + "/ai/v1/chat/completions"
CFK = KEYS.get("CLOUDFLARE_API_TOKEN")
OR = "https://openrouter.ai/api/v1/chat/completions"
ORK = KEYS.get("OPENROUTER_API_KEY")

TESTS = [
    ("cf-qwen3.8-27b", CF, CFK, "@cf/qwen/qwen3.8-27b"),
    ("cf-qwen3-30b-a3b", CF, CFK, "@cf/qwen/qwen3-30b-a3b-fp8"),
    ("cf-nemotron-120b", CF, CFK, "@cf/nvidia/nemotron-3-120b-a12b"),
    ("cf-kimi-k2.6", CF, CFK, "@cf/moonshotai/kimi-k2.6"),
    ("cf-gemma4-26b", CF, CFK, "@cf/google/gemma-4-26b-a4b-it"),
    ("cf-llama4-scout", CF, CFK,
     "@cf/meta/llama-4-scout-17b-16e-instruct"),
]

for name, url, key, model in TESTS:
    if not key:
        print(f"{name}: NO KEY"); continue
    body = json.dumps({"model": model, "max_tokens": 200,
                       "messages": [{"role": "user", "content": PROMPT}]})
    t0 = time.time()
    r = subprocess.run(
        ["curl.exe", "-s", "-m", "60", url,
         "-H", f"Authorization: Bearer {key}",
         "-H", "Content-Type: application/json",
         "--data-binary", "@-"],
        input=body.encode(), capture_output=True, timeout=70)
    dt = time.time() - t0
    try:
        d = json.loads(r.stdout.decode("utf-8", "replace"))
        msg = d["choices"][0]["message"].get("content", "")
        ok = "```python" in msg and "sum_pair" in msg
        print(f"{name}: {dt:.1f}s ok={ok} len={len(msg)}")
    except Exception:
        print(f"{name}: {dt:.1f}s FAIL {r.stdout[:140]!r}")
