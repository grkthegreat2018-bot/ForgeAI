"""Probe every OpenRouter :free model for fenced-code output + key quota."""
import json
import subprocess

KEYS = {}
for line in open(".env", encoding="utf-8", errors="replace"):
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        KEYS[k.strip()] = v.strip().strip('"').strip("'")
KEY = KEYS.get("OPENROUTER_API_KEY", "")

# key quota
r = subprocess.run(["curl.exe", "-s", "-m", "20",
                    "https://openrouter.ai/api/v1/key",
                    "-H", f"Authorization: Bearer {KEY}"],
                   capture_output=True, timeout=30)
d = json.loads(r.stdout.decode("utf-8", "replace"))
print("KEY:", json.dumps(d.get("data", d), indent=0)[:400])

r = subprocess.run(["curl.exe", "-s", "-m", "30",
                    "https://openrouter.ai/api/v1/models",
                    "-H", f"Authorization: Bearer {KEY}"],
                   capture_output=True, timeout=40)
models = [m["id"] for m in json.loads(
    r.stdout.decode())["data"] if m["id"].endswith(":free")]
print(f"\n{len(models)} :free models")

PROMPT = ("Output ONLY one ```python code block implementing "
          "sum_pair(a, b) that returns a + b. No prose.")
for mid in models:
    body = json.dumps({"model": mid, "max_tokens": 150,
                       "messages": [{"role": "user",
                                     "content": PROMPT}]})
    r = subprocess.run(
        ["curl.exe", "-s", "-m", "60",
         "https://openrouter.ai/api/v1/chat/completions",
         "-H", f"Authorization: Bearer {KEY}",
         "-H", "Content-Type: application/json",
         "--data-binary", "@-"],
        input=body.encode(), capture_output=True, timeout=70)
    try:
        dd = json.loads(r.stdout.decode("utf-8", "replace"))
        msg = dd["choices"][0]["message"].get("content", "")
        fenced = "```python" in msg or "```" in msg
        print(f"{'OK ' if fenced else 'no '}{mid} len={len(msg)}")
    except Exception:
        print(f"ERR {mid} {r.stdout[:100]!r}")
