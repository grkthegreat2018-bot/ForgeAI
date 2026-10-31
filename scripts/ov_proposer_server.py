"""OpenVINO GenAI proposer server — runs in venv_ov (transformers-free).

Serves the OpenVINO-exported LFM3 checkpoint on NPU/GPU/CPU behind a
tiny JSON HTTP API so the evolve loop's LLMProposer can offload
generation from the RTX 5070 (freeing all 12GB VRAM for eval workers).

POST /generate {"prompts": [...], "configs": [{max_new_tokens,
    temperature, top_p, top_k, repetition_penalty, do_sample}]}
  -> {"texts": [...], "gen_seconds": float}
GET /health -> {"ok": true, "device": ...}

Run:  venv_ov\\Scripts\\python.exe scripts\\ov_proposer_server.py ^
        --model <ov_model_dir> --device NPU --port 8790
"""
import argparse, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openvino_genai as ov_genai

_PIPE = None
_DEVICE = "NPU"


def _gen_cfg(c: dict) -> ov_genai.GenerationConfig:
    g = ov_genai.GenerationConfig()
    g.max_new_tokens = int(c.get("max_new_tokens", 448))
    t = float(c.get("temperature", 0.0) or 0.0)
    g.do_sample = t > 0
    g.temperature = max(t, 1e-5)
    g.top_p = float(c.get("top_p", 0.95))
    g.top_k = int(c.get("top_k", 80))
    rp = float(c.get("repetition_penalty", 1.0) or 1.0)
    if rp != 1.0:
        try:
            g.repetition_penalty = rp
        except AttributeError:
            pass
    return g


def _generate(prompts, configs):
    texts = []
    # GenAI LLMPipeline: batch generate isn't supported on NPU — run
    # per-prompt. Prefill is the fast part on NPU; sequential decode
    # still beats GPU-contended batches at our batch sizes.
    for p, c in zip(prompts, configs):
        try:
            r = _PIPE.generate(p, _gen_cfg(c))
            texts.append(r if isinstance(r, str) else str(r))
        except Exception as e:
            texts.append(f"")
            print(f"[ov] generate error: {e}", flush=True)
    return texts


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "device": _DEVICE})
        else:
            self._send(404, {})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {})
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        prompts = req.get("prompts", [])
        cfgs = req.get("configs") or [{}] * len(prompts)
        t0 = time.time()
        texts = _generate(prompts, cfgs)
        self._send(200, {"texts": texts,
                         "gen_seconds": round(time.time() - t0, 3)})

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global _PIPE, _DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    _DEVICE = args.device
    t0 = time.time()
    try:
        _PIPE = ov_genai.LLMPipeline(args.model, args.device)
    except Exception as e:
        print(f"[ov] {args.device} load failed ({e}); falling back CPU",
              flush=True)
        _DEVICE = "CPU"
        _PIPE = ov_genai.LLMPipeline(args.model, "CPU")
    print(f"[ov] ready device={_DEVICE} load={time.time()-t0:.1f}s "
          f"port={args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), _H).serve_forever()


if __name__ == "__main__":
    main()
