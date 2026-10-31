"""Post-fix verification: real Chub path (Kobold -> completions) +
memory-growth check across repeated requests."""
import json
import subprocess as sp
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def post(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vram():
    out = sp.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"])
    return int(out.decode().strip())


LT, GT = chr(60) + chr(124), chr(124) + chr(62)
def mk(n):
    return LT + n + GT

CARD = ("Seraphina is a sharp-witted elven ranger traveling with the user. "
        "She speaks plainly, teases often, and hates formal titles.\n\n")
HIST = "".join(
    f"User: turn {i} done.\nSeraphina: noted, keep walking.\n\n"
    for i in range(30))
BLOB = CARD + HIST + "User: What's that smell?\nSeraphina:"

# What kobold_to_completions now forwards (raw prompt + stops)
payload = {
    "model": "hermes3-8b",
    "prompt": BLOB,
    "max_tokens": 150,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 41,
    "stop": [mk("im_end"), mk("eot_id"), mk("end_of_text"), "</s>",
             mk("start_header_id"), "\n" + mk("im_start")],
}

print(f"VRAM at start: {vram()} MB")
for i in range(6):
    r = post("http://localhost:8000/v1/completions", payload)
    text = r["choices"][0]["text"]
    print(f"req{i+1}: vram={vram()}MB  text={text[:140]!r}")

# Kobold-format temp=2 (what Chub actually sends) for comparison
payload["temperature"] = 2.0
payload["top_p"] = 0.8
r = post("http://localhost:8000/v1/completions", payload)
print(f"\ntemp2.0: {r['choices'][0]['text'][:200]!r}")
print(f"VRAM final: {vram()} MB")
print("DONE")
