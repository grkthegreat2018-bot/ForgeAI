"""Probe keyless OpenAI-compatible endpoints for fenced-code output."""
import json
import subprocess

PROMPT = ("Output ONLY one ```python code block implementing "
          "sum_pair(a, b) that returns a + b. No prose.")

TESTS = [
    ("pollinations-openai-fast",
     "https://text.pollinations.ai/openai", "openai-fast"),
    ("llm7-codestral",
     "https://api.llm7.io/v1/chat/completions", "codestral-latest"),
    ("llm7-llama31-turbo",
     "https://api.llm7.io/v1/chat/completions",
     "meta-Llama-3.1-8B-Instruct-Turbo"),
    ("llm7-gpt-oss-20b",
     "https://api.llm7.io/v1/chat/completions", "gpt-oss:20b"),
    ("ovh-gpt-oss-120b",
     "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions",
     "gpt-oss-120b"),
    ("ovh-qwen3.5-397b",
     "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions",
     "Qwen3.5-397B-A17B"),
    ("ovh-llama33-70b",
     "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions",
     "Meta-Llama-3_3-70B-Instruct"),
]

for name, url, model in TESTS:
    body = json.dumps({"model": model, "max_tokens": 150,
                       "messages": [{"role": "user",
                                     "content": PROMPT}]})
    r = subprocess.run(
        ["curl.exe", "-s", "-m", "45", url,
         "-H", "Content-Type: application/json",
         "--data-binary", "@-"],
        input=body.encode(), capture_output=True, timeout=50)
    try:
        d = json.loads(r.stdout.decode("utf-8", "replace"))
        msg = d["choices"][0]["message"].get("content", "")
        print(f"{'OK ' if '```' in msg else 'no '}{name} "
              f"len={len(msg)}")
    except Exception:
        print(f"ERR {name} {r.stdout[:120]!r}")
