"""bench_serving-style benchmark for the ForgeAI inference server.

Measures TTFT (time-to-first-token chunk) + TPOT-ish inter-chunk stats
and end-to-end latency over N chat-completions requests; prints p50/p95.

Live server (stdlib-only client, no deps):

    python scripts/bench_serving.py --url http://127.0.0.1:8000 -n 16

In-process self-test — spins up ForgeServer against a stub
registry/engine via TestClient (no GPU, no checkpoint):

    python scripts/bench_serving.py --self-test
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time

CHUNK_DELAY_S = 0.003  # stub engine per-chunk sleep — keeps TTFT/TPOT real


def _pct(vals, p):
    """Nearest-rank percentile."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, round(p / 100.0 * (len(s) - 1))))
    return s[k]


def _summ(name, vals):
    if not vals:
        print(f"  {name:<28} (no samples)")
        return
    print(f"  {name:<28} n={len(vals):<3} "
          f"p50={_pct(vals, 50) * 1e3:8.1f}ms  "
          f"p95={_pct(vals, 95) * 1e3:8.1f}ms  "
          f"mean={statistics.fmean(vals) * 1e3:8.1f}ms")


def _is_token_line(line: str) -> bool:
    """True when an SSE ``data:`` line carries generated content
    (skips the leading empty-role chunk and [DONE])."""
    if not line.startswith("data:"):
        return False
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return False
    try:
        j = json.loads(payload)
    except ValueError:
        return False
    choice = (j.get("choices") or [{}])[0]
    delta = choice.get("delta")
    if isinstance(delta, dict):
        return bool(delta.get("content"))
    return bool(choice.get("text"))


def _payload(args) -> dict:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": not args.no_stream,
    }


def _report(tag, samples):
    """samples: list of dicts {ttft, inter[], total, chunks, status}."""
    ok = [s for s in samples if s["status"] == 200]
    print(f"== {tag}: {len(ok)}/{len(samples)} requests ok ==")
    if not ok:
        for s in samples[:3]:
            print(f"   failed: status={s['status']} err={s.get('error')}")
        return
    ttfts = [s["ttft"] for s in ok if s["ttft"] is not None]
    tpots = [statistics.fmean(s["inter"]) for s in ok if s["inter"]]
    totals = [s["total"] for s in ok]
    tps = [s["chunks"] / s["total"] for s in ok
           if s["chunks"] and s["total"] > 0]
    _summ("TTFT (first token chunk)", ttfts)
    _summ("TPOT (mean inter-chunk)", tpots)
    _summ("E2E latency", totals)
    if tps:
        print(f"  {'tok/s per request':<28} "
              f"p50={_pct(tps, 50):8.1f}  p95={_pct(tps, 95):8.1f}  "
              f"mean={statistics.fmean(tps):8.1f}")


# ── Live-URL client (urllib — zero deps) ──────────────────────────────────

def _make_url_sender(args):
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    url = args.url.rstrip("/") + "/v1/chat/completions"

    def send():
        req = urllib.request.Request(
            url, data=json.dumps(_payload(args)).encode(),
            headers=headers)
        t0 = time.perf_counter()
        ttft = None
        inter = []
        n_chunks = 0
        last = t0
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                status = r.status
                if args.no_stream:
                    r.read()
                    return {"ttft": None, "inter": [], "chunks": 0,
                            "total": time.perf_counter() - t0,
                            "status": status}
                buf = b""
                while True:
                    piece = r.read(4096)
                    if not piece:
                        break
                    buf += piece
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = raw.decode("utf-8", "replace").strip()
                        if not _is_token_line(line):
                            continue
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        else:
                            inter.append(now - last)
                        last = now
                        n_chunks += 1
        except Exception as e:
            return {"ttft": None, "inter": [], "chunks": n_chunks,
                    "total": time.perf_counter() - t0,
                    "status": getattr(e, "code", 0) or 0, "error": str(e)}
        return {"ttft": ttft, "inter": inter, "chunks": n_chunks,
                "total": time.perf_counter() - t0, "status": status}

    return send


# ── In-process self-test (ForgeServer + duck-typed stub registry) ─────────

class _StubTokenizer:
    eos_token_id = 7

    def encode(self, text):
        return [ord(c) % 128 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class _StubEngine:
    def __init__(self, chunks, delay=CHUNK_DELAY_S):
        self.tokenizer = _StubTokenizer()
        self.device = type("Dev", (), {"type": "cpu"})()
        self._gen_lock = threading.RLock()
        self.total_tokens_generated = 0
        self._chunks = list(chunks)
        self._delay = delay

    def generate(self, prompt, **kw):
        time.sleep(self._delay)
        self.total_tokens_generated += len(self._chunks)
        return "".join(self._chunks)

    def generate_stream(self, prompt, **kw):
        for c in self._chunks:
            time.sleep(self._delay)
            yield c
        self.total_tokens_generated += len(self._chunks)


class _StubRegistry:
    """Duck-typed stand-in for ModelRegistry — the exact surface the
    server's chat path + stream drain touch (no ModelRegistry import:
    bench must run even when the registry contract is mid-flight)."""

    def __init__(self, model_id="stub-model"):
        self._id = model_id
        self._engine = _StubEngine(
            ["The", " quick", " brown", " fox", " jumps", " over",
             " the", " lazy", " dog", "."] * 4)
        self._inflight_loads = {}

    def generate(self, model_id, prompt, **kw):
        return self._engine.generate(prompt, **kw)

    def generate_stream(self, model_id, prompt, **kw):
        return self._engine.generate_stream(prompt, **kw)

    def get_engine(self, model_id):
        return self._engine if model_id == self._id else None

    def config_name_for(self, model_id):
        return "forgelm_v2"

    def list_models(self):
        return [{"id": self._id, "awake": True, "vram_budget_gb": 0.01,
                 "config": "forgelm_v2", "last_used_ago_s": 0.0,
                 "generations": 0, "total_tokens": 0, "in_flight": 0}]

    def stats(self):
        return {"total_models": 1, "awake": 1, "asleep": 0,
                "awake_vram_gb": 0.01, "total_vram_gb": 32.0,
                "free_vram_gb": 31.0, "models": self.list_models()}


def _self_test(args):
    """Build ForgeServer on the stub and bench through TestClient."""
    from fastapi.testclient import TestClient
    from forge.engine.forge_server import ForgeServer

    server = ForgeServer(registry=_StubRegistry(model_id=args.model))
    client = TestClient(server.app)

    def send():
        t0 = time.perf_counter()
        ttft = None
        inter = []
        n_chunks = 0
        last = t0
        try:
            if args.no_stream:
                r = client.post("/v1/chat/completions",
                                json=_payload(args))
                return {"ttft": None, "inter": [], "chunks": 0,
                        "total": time.perf_counter() - t0,
                        "status": r.status_code,
                        "error": r.text if r.status_code != 200 else None}
            with client.stream("POST", "/v1/chat/completions",
                               json=_payload(args)) as r:
                status = r.status_code
                for line in r.iter_lines():
                    if not _is_token_line(line):
                        continue
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        inter.append(now - last)
                    last = now
                    n_chunks += 1
        except Exception as e:
            return {"ttft": None, "inter": [], "chunks": n_chunks,
                    "total": time.perf_counter() - t0,
                    "status": 0, "error": str(e)}
        return {"ttft": ttft, "inter": inter, "chunks": n_chunks,
                "total": time.perf_counter() - t0,
                "status": status}

    samples = [send() for _ in range(args.requests)]
    _report("self-test (in-process stub)", samples)
    return all(s["status"] == 200 for s in samples)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="bench_serving for ForgeAI — TTFT/TPOT p50/p95")
    ap.add_argument("--url", default=None,
                    help="Live server base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--self-test", action="store_true",
                    help="Run against an in-process stub server")
    ap.add_argument("--model", default="forgelm-v2-jamba")
    ap.add_argument("-n", "--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--prompt", default="Write one short sentence.")
    ap.add_argument("--no-stream", action="store_true",
                    help="Non-streaming requests (E2E latency only)")
    ap.add_argument("--api-key", default=None,
                    help="Bearer token for FORGE_API_KEY-gated servers")
    args = ap.parse_args(argv)

    if args.self_test:
        ok = _self_test(args)
        return 0 if ok else 1
    if not args.url:
        ap.error("pass --url or --self-test")
    send = _make_url_sender(args)
    samples = [send() for _ in range(args.requests)]
    _report(f"live {args.url}", samples)
    return 0 if all(s["status"] == 200 for s in samples) else 1


if __name__ == "__main__":
    sys.exit(main())
