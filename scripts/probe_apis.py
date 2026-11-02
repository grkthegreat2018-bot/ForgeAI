"""Probe API reachability/models with keys from .env (values never printed)."""
import json
import subprocess
import sys

KEYS = {}
for line in open(".env", encoding="utf-8", errors="replace"):
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        KEYS[k.strip()] = v.strip().strip('"').strip("'")


def curl_json(url, key=None, extra=None):
    cmd = ["curl.exe", "-s", "-m", "20", url]
    if key:
        cmd += ["-H", f"Authorization: Bearer {key}"]
    cmd += extra or []
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    try:
        return json.loads(r.stdout.decode("utf-8", "replace"))
    except Exception:
        return {"_raw": r.stdout.decode("utf-8", "replace")[:300]}


which = sys.argv[1] if len(sys.argv) > 1 else "all"

if which in ("all", "groq"):
    d = curl_json("https://api.groq.com/openai/v1/models",
                  KEYS.get("GROQ_API_KEY"))
    ids = [m.get("id") for m in d.get("data", [])]
    print("GROQ models:", len(ids))
    for i in ids:
        print("  ", i)
    if "error" in d:
        print("  err:", str(d["error"])[:160])

if which in ("all", "or"):
    d = curl_json("https://openrouter.ai/api/v1/models",
                  KEYS.get("OPENROUTER_API_KEY"))
    free = [m["id"] for m in d.get("data", [])
            if m["id"].endswith(":free")]
    print("OPENROUTER :free models:", len(free))
    for i in free:
        print("  ", i)

if which in ("all", "cf"):
    acct, tok = (KEYS.get("CLOUDFLARE_ACCOUNT_ID"),
                 KEYS.get("CLOUDFLARE_API_TOKEN"))
    if acct and tok:
        d = curl_json(
            f"https://api.cloudflare.com/client/v4/accounts/{acct}"
            "/ai/models/search?task=Text%20Generation&per_page=50", tok)
        models = [m.get("name") for m in
                  (d.get("result") or [])]
        print("CLOUDFLARE models:", len(models))
        for i in models[:40]:
            print("  ", i)
        if not d.get("success"):
            print("  err:", str(d.get("errors"))[:160])
    else:
        print("CLOUDFLARE: creds missing")

if which in ("all", "codestral"):
    d = curl_json("https://codestral.mistral.ai/v1/models",
                  KEYS.get("MISTRAL_API_KEY"))
    ids = [m.get("id") for m in d.get("data", [])]
    print("CODESTRAL endpoint models:", ids or str(d)[:160])
