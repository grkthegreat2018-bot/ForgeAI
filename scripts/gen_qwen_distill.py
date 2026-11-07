"""Teacher distill set for the LFM3 proposer — HF or Groq backend.

Draws the REAL prompt distribution — logged proposal prompts from
code_programs (stratified per target) plus template prompts over every
seed in evolution/targets — and asks a teacher model for the completion.

SEMANTIC GATE (the fix after round-1 poison): each completion is exec'd
and run through the matched evolution target's own test(ns) — compile-
valid is not enough; the code must produce identical outputs. Semantic
failures get ONE feedback retry (the teacher sees the failure detail)
before the row is dropped. Unmatched-target rows are kept as
unverifiable (they still teach format).

Groq backend: GROQ_API_KEY env, OpenAI-compatible /chat/completions.
Model choice matters legally — avoid Llama-family ids (Llama Community
License bars using outputs to improve other models). gpt-oss-120b
(Apache-2.0) is the default teacher.

Usage:
    GROQ_API_KEY=... venv\\Scripts\\python.exe scripts/gen_qwen_distill.py \
        --backend groq --rows 300 --out results/qwen_distill_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from forge.evolution.code_evolve import _extract_code  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"


def _cf_url():
    """Workers AI OpenAI-compat endpoint — acct id from env/.env."""
    import os
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    if not acct:
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("CLOUDFLARE_ACCOUNT_ID="):
                    acct = line.split("=", 1)[1].strip()
    return ("https://api.cloudflare.com/client/v4/accounts/"
            f"{acct}/ai/v1/chat/completions")


BACKENDS = {
    "groq": (GROQ_URL, "GROQ_API_KEY", "openai/gpt-oss-120b",
             {"reasoning_effort": "low"}),
    "deepseek": (DEEPSEEK_URL, "DEEPSEEK_API_KEY", "deepseek-flash",
                 {}),
    "mistral": (MISTRAL_URL, "MISTRAL_API_KEY", "codestral-latest",
                {}),
    "cf": (None, "CLOUDFLARE_API_TOKEN",
           "@cf/qwen/qwen2.5-coder-32b-instruct", {}),
    "or": (OR_URL, "OPENROUTER_API_KEY",
           "nvidia/nemotron-3-ultra-550b-a55b:free", {}),
    "poll": ("https://text.pollinations.ai/openai", "",
             "openai-fast", {}),
    "llm7": ("https://api.llm7.io/v1/chat/completions", "",
             "codestral-latest", {}),
    "ovh": ("https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat"
            "/completions", "", "gpt-oss-120b", {}),
}

_print_lock = threading.Lock()


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


# ── prompt sources ────────────────────────────────────────────────────

def _real_prompts(db: str, per_target: int, total: int) -> list[str]:
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT target, prompt FROM code_programs "
        "WHERE prompt IS NOT NULL AND length(prompt) > 200 "
        "AND prompt LIKE '# Rewrite this Python program%' "
        "ORDER BY RANDOM()").fetchall()
    seen: set[str] = set()
    by_target: dict[str, int] = {}
    out: list[str] = []
    for tgt, p in rows:
        if len(out) >= total:
            break
        if by_target.get(tgt, 0) >= per_target:
            continue
        h = p[:300]
        if h in seen:
            continue
        seen.add(h)
        by_target[tgt] = by_target.get(tgt, 0) + 1
        out.append(p)
    return out


def _target_prompts(targets_dir: Path) -> list[str]:
    import hashlib
    out = []
    for f in sorted(targets_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        spec = {}
        src = f.read_text(encoding="utf-8", errors="replace")
        head = src[:2500]
        for k in ("NAME", "GOAL", "METRIC", "REQUIRED", "BUDGETS"):
            m = re.search(rf"^{k}\s*=\s*(.+)$", head, re.M)
            if m:
                try:
                    spec[k] = eval(m.group(1))
                except Exception:
                    pass
        m = re.search(r"SEED_CODE\s*=\s*'((?:[^'\\]|\\.)*)'", src, re.S)
        seed = m.group(1).encode().decode('unicode_escape') if m else ""
        if not seed.strip():
            m = re.search(r'SEED_CODE\s*=\s*"((?:[^"\\]|\\.)*)"', src, re.S)
            seed = m.group(1).encode().decode('unicode_escape') if m else ""
        if not seed.strip():
            continue
        required = spec.get("REQUIRED", [])
        tag = hashlib.sha1(seed.encode()).hexdigest()[:8]
        metric = spec.get("METRIC", "score")
        goal = spec.get("GOAL", "min").upper() + "IMIZE"
        goal = "MINIMIZE" if "MIN" in goal else "MAXIMIZE"
        out.append(
            "# Rewrite this Python program to be much better.\n"
            f"# Candidate: {tag}\n"
            f"# Direction: {metric} {goal}\n"
            f"# Hard budgets (must satisfy): {spec.get('BUDGETS', {})}\n"
            f"# Rules: keep public names ({', '.join(required)}); "
            "stdlib+numpy+torch only;\n"
            "# output ONLY one python code block; identical semantics.\n"
            "#\n# Current best:\n"
            f"```python\n{seed.strip()}\n```\n"
            "Reply with ONLY one ```python code block containing the "
            "complete improved implementation.")
    return out


# ── semantic oracle ───────────────────────────────────────────────────

def _load_oracles(targets_dir: Path):
    table = {}
    for f in sorted(targets_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        mod = types.ModuleType(f.stem)
        try:
            exec(compile(f.read_text(encoding="utf-8", errors="replace"),
                         f.name, "exec"), mod.__dict__)
        except Exception:
            continue
        req = tuple(sorted(getattr(mod, "REQUIRED", []) or ()))
        test = getattr(mod, "test", None)
        if req and callable(test):
            table[frozenset(req)] = (f.stem, test)
    return table


def _required_from_prompt(prompt: str) -> frozenset:
    m = re.search(r"keep public names \(([^)]*)\)", prompt)
    if not m:
        return frozenset()
    return frozenset(s.strip() for s in m.group(1).split(",")
                     if s.strip())


def _semantic_check(code: str, prompt: str, table) -> tuple[str, str]:
    """-> ('ok'|'semantic_fail'|'compile_fail'|'exec_fail'|
          'no_target', detail)"""
    try:
        compile(code, "<c>", "exec")
    except SyntaxError:
        return "compile_fail", "syntax"
    req = _required_from_prompt(prompt)
    entry = table.get(req)
    if entry is None:
        cands = [v for k, v in table.items() if k <= req]
        entry = cands[0] if len(cands) == 1 else None
    if entry is None:
        return "no_target", ""
    tname, test = entry
    ns = {}
    try:
        exec(compile(code, "<c>", "exec"), ns)
    except Exception as e:
        return "exec_fail", f"{type(e).__name__}: {str(e)[:80]}"
    missing = [n for n in req if n not in ns]
    if missing:
        return "semantic_fail", f"missing names {missing}"
    try:
        res = test(ns)
    except Exception as e:
        return "exec_fail", f"test_raise {type(e).__name__}: {str(e)[:80]}"
    if res.get("ok"):
        return "ok", tname
    return "semantic_fail", f"{tname}: {res.get('detail', '?')[:150]}"


# ── generation backends ───────────────────────────────────────────────

def _hf_gen(model, tok, prompt: str, max_new: int) -> str:
    import torch
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True) + "```python\n"
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def _hf_gen_batch(model, tok, prompts: list, max_new: int,
                  temperature: float = 0.7, seed: int | None = None,
                  top_p: float = 0.95, top_k: int = 80) -> list:
    """Batched decode — left-pad so new tokens align right; one
    generate() per N prompts. HF shares one seed/temp per call, so the
    caller passes a different seed + jittered temp per chunk — every
    row still samples independently inside the call."""
    import random
    import torch
    if seed is None:
        seed = random.randrange(1 << 60)
    torch.manual_seed(seed)
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False,
        add_generation_prompt=True) + "```python\n" for p in prompts]
    prev = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids = tok(texts, return_tensors="pt", padding=True).to(model.device)
    do_sample = temperature > 0.05
    out = model.generate(
        **ids, max_new_tokens=max_new,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        pad_token_id=tok.eos_token_id)
    tok.padding_side = prev
    n = ids["input_ids"].shape[1]
    return [tok.decode(o[n:], skip_special_tokens=True) for o in out]


def _api_gen(url: str, api_key: str, model_name: str, prompt: str,
             max_new: int, temperature: float = 0.0,
             extra_user: tuple | None = None,
             extra_body: dict | None = None) -> str:
    """curl.exe transport — Python urllib gets Cloudflare-1010'd on this
    route (browser-signature ban), while curl reaches the API fine."""
    import subprocess
    msgs = [{"role": "user", "content": prompt}]
    if extra_user:
        prev, feedback = extra_user
        msgs.append({"role": "assistant", "content": _fence(prev)})
        msgs.append({"role": "user", "content": feedback})
    body = {"model": model_name, "messages": msgs,
            "temperature": temperature, "max_tokens": max_new}
    body.update(extra_body or {})
    data = json.dumps(body).encode()
    auth = (["-H", f"Authorization: Bearer {api_key}"]
            if api_key else [])
    for attempt in range(8):
        try:
            r = subprocess.run(
                ["curl.exe", "-s", "-X", "POST", url] + auth + [
                 "-H", "Content-Type: application/json",
                 "--data-binary", "@-", "--max-time", "300",
                 "-w", "\nHTTPSTATUS:%{http_code}"],
                input=data, capture_output=True, timeout=320)
            txt = r.stdout.decode("utf-8", errors="replace")
            body_txt, _, status = txt.rpartition("HTTPSTATUS:")
            status = status.strip()
            if status == "429":
                wait = 60 + 15 * attempt
                _log(f"[api] 429 rate-limit, waiting {wait}s")
                time.sleep(wait)
                continue
            d = json.loads(body_txt)
            if "error" in d:
                raise RuntimeError(str(d["error"])[:150])
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            # reasoning models: fall back to reasoning if content empty
            if not content.strip() and reasoning:
                content = reasoning
                reasoning = ""
            return content, reasoning
        except json.JSONDecodeError:
            wait = 20 * (attempt + 1)
            _log(f"[api] bad response (status={status}), retry {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = 20 * (attempt + 1)
            _log(f"[api] {type(e).__name__} retry in {wait}s "
                 f"({str(e)[:100]})")
            time.sleep(wait)
    raise RuntimeError("api retries exhausted")


def _mistral_batch(url_base: str, api_key: str, model_name: str,
                   prompts: list, max_new: int, oracles,
                   stats: dict, out_f, poll_s: int = 60,
                   timeout_h: float = 8.0) -> None:
    """Bulk path: one batch job for the whole prompt set.

    Mistral batch jobs run server-side (up to 1M requests, NOT counted
    against real-time rate limits) — the right lane for overnight bulk
    corpus work. Verify + write rows exactly like the live path."""
    import tempfile
    base = url_base.rsplit("/chat/completions", 1)[0]
    req_file = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    with req_file.open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({
                "custom_id": str(i),
                "body": {"max_tokens": max_new,
                         "messages": [{"role": "user",
                                       "content": p}]}}) + "\n")
    def _curl(args2, data=None):
        cmd = ["curl.exe", "-s", "-X", "POST"] + args2 + [
            "-H", f"Authorization: Bearer {api_key}"]
        r = subprocess.run(
            cmd + (["--data-binary", "@-"] if data is not None else []),
            input=data, capture_output=True, timeout=300)
        return json.loads(r.stdout.decode("utf-8", errors="replace"))

    up = _curl([f"{base}/files",
                "-F", "purpose=batch", "-F",
                f"file=@{req_file};type=application/jsonl"])
    fid = up.get("id")
    if not fid:
        raise RuntimeError(f"batch upload failed: {str(up)[:200]}")
    job = _curl([f"{base}/batch/jobs",
                 "-H", "Content-Type: application/json"],
                json.dumps({"input_files": [fid],
                            "endpoint": "/v1/chat/completions",
                            "model": model_name,
                            "timeout_hours": timeout_h}).encode())
    jid = job.get("id")
    if not jid:
        raise RuntimeError(f"batch create failed: {str(job)[:200]}")
    _log(f"[batch] job {jid} submitted ({len(prompts)} requests)")
    t0 = time.time()
    while True:
        time.sleep(poll_s)
        r = subprocess.run(
            ["curl.exe", "-s", f"{base}/batch/jobs/{jid}",
             "-H", f"Authorization: Bearer {api_key}"],
            capture_output=True, timeout=60)
        st = json.loads(r.stdout.decode())
        status = st.get("status", "?")
        _log(f"[batch] {status} "
             f"({st.get('succeeded_requests', 0)}/"
             f"{st.get('total_requests', '?')})")
        if status in ("SUCCESS", "FAILED", "CANCELLED",
                      "TIMEOUT_EXCEEDED"):
            break
        if time.time() - t0 > timeout_h * 3600:
            raise RuntimeError("batch poll exceeded timeout")
    ofid = st.get("output_file")
    if not ofid:
        _log(f"[batch] no output file ({status})")
        return
    r = subprocess.run(
        ["curl.exe", "-s", f"{base}/files/{ofid}/content",
         "-H", f"Authorization: Bearer {api_key}"],
        capture_output=True, timeout=300)
    for line in r.stdout.decode("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
            i = int(row["custom_id"])
            msg = row["response"]["body"]["choices"][0]["message"]
            raw = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
        except Exception:
            stats["gen_err"] += 1
            continue
        code = _extract_code(raw)
        if not code.strip():
            stats["compile_fail"] += 1
            continue
        status, _ = _semantic_check(code, prompts[i], oracles)
        if status in ("ok", "no_target"):
            think = reasoning or re.sub(
                r"```.*?```", "", raw, flags=re.S).strip()
            resp = ((f"<think>\n{think}\n</think>\n" if think else "")
                    + "```python\n" + code.rstrip() + "\n```\n")
            out_f.write(json.dumps({
                "prompt": prompts[i], "response": resp,
                "reasoning": think, "target": "qwen_distill",
                "verified": status == "ok",
                "semantic": status}) + "\n")
            out_f.flush()
        stats["ok" if status == "ok" else status] = \
            stats.get(status, 0) + (1 if status != "ok" else 0) \
            + (1 if status == "ok" else 0)


def _fence(resp: str) -> str:
    if not resp.lstrip().startswith("```"):
        resp = "```python\n" + resp
    if "```" not in resp[10:]:
        resp = resp.rstrip() + "\n```\n"
    return resp


# ── main ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="forge_evolve.db")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--per-target", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=900)
    ap.add_argument("--backend", choices=["hf", "groq", "deepseek",
                                          "mistral", "cf", "or",
                                          "poll", "llm7", "ovh"],
                    default="hf")
    ap.add_argument("--hf-model", default=None,
                    help="HF id for --backend hf (default "
                         "Qwen/Qwen2.5-Coder-3B-Instruct)")
    ap.add_argument("--hf-batch", type=int, default=1,
                    help="batched prompts per generate call on --backend hf")
    ap.add_argument("--hf-quant", default=None, choices=["int4"],
                    help="bnb quantization for --backend hf (int4 lets a "
                         "7B fit alongside a big KV batch)")
    ap.add_argument("--api-model", default=None,
                    help="override the backend's default model id")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=1,
                    help="semantic-fail feedback retries per prompt")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch", action="store_true",
                    help="mistral backend only: submit all prompts as one "
                         "server-side batch job (up to 1M requests, off "
                         "real-time rate limits) instead of live calls")
    ap.add_argument("--batch-timeout-h", type=float, default=8.0)
    args = ap.parse_args()

    prompts = _real_prompts(args.db, args.per_target, args.rows)
    seed_prompts = _target_prompts(REPO / "forge" / "evolution" / "targets")
    prompts += [p for p in seed_prompts
                if p[:300] not in {q[:300] for q in prompts}]
    _log(f"[gen] {len(prompts)} prompts "
         f"({len(seed_prompts)} target-template) backend={args.backend}")
    oracles = _load_oracles(REPO / "forge" / "evolution" / "targets")
    _log(f"[gen] {len(oracles)} semantic oracles loaded")

    model = tok = None
    api_key = api_url = model_name = extra_body = None
    if args.backend == "hf":
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_id = args.hf_model or MODEL_ID
        kw = {}
        cache_dir = None
        if args.hf_quant == "int4":
            # Quantize once, then load the serialized 4-bit weights —
            # bnb quantization on load is ~10min here; the cache makes
            # subsequent loads seconds.
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", hf_id)
            cache_dir = REPO / "research" / "hf_quant_cache" / safe
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
            if cache_dir.exists():
                hf_id = str(cache_dir)
        else:
            kw["dtype"] = torch.bfloat16
        tok = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, device_map="cuda", **kw)
        if cache_dir is not None and not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(cache_dir)
            tok.save_pretrained(cache_dir)
            _log(f"[gen] quant cache saved -> {cache_dir}")
        model.eval()
        _log(f"[gen] hf: {hf_id} batch={args.hf_batch} "
             f"quant={args.hf_quant}")
    else:
        import os
        api_url, key_env, model_name, extra_body = BACKENDS[args.backend]
        api_url = api_url or _cf_url()
        model_name = args.api_model or model_name
        api_key = ""
        if key_env:
            api_key = os.environ.get(key_env, "")
            if not api_key:
                # fall back to the repo .env
                env = REPO / ".env"
                if env.exists():
                    for line in env.read_text().splitlines():
                        if line.startswith(key_env + "="):
                            api_key = line.split("=", 1)[1].strip()
            if not api_key:
                print(f"{key_env} not set", file=sys.stderr)
                return 2
        _log(f"[gen] {args.backend}: {model_name} @ {api_url}")

    stats = {"ok": 0, "semantic_fail": 0, "compile_fail": 0,
             "exec_fail": 0, "no_target": 0, "gen_err": 0}
    done = [0]

    def work(item, raw0=None):
        i, p = item
        raw = raw0
        reasoning = ""
        # bound before the loop: the retry-feedback path references it and
        # retries=-1 would otherwise leave it (and status) unbound (F823).
        status, last_detail = "gen_err", "no generation attempt"
        try:
            for attempt in range(args.retries + 1):
                if raw is not None:
                    pass                # pre-generated (hf batch)
                elif args.backend == "hf":
                    raw = _hf_gen(model, tok, p, args.max_new)
                else:
                    fb = None
                    if attempt and raw is not None:
                        fb = (raw,
                              "Your previous code failed verification: "
                              f"{last_detail}. Reply with ONLY one "
                              "```python code block containing the "
                              "corrected complete implementation.")
                    raw, reasoning = _api_gen(
                        api_url, api_key, model_name, p, args.max_new,
                        temperature=(args.temperature
                                     if attempt == 0
                                     else max(0.3, args.temperature)),
                        extra_user=fb, extra_body=extra_body)
                code = _extract_code(raw)
                if not code.strip():
                    last_detail = "no code block in output"
                    status = "compile_fail"
                else:
                    status, last_detail = _semantic_check(code, p, oracles)
                if status in ("ok", "no_target"):
                    break
            if status in ("ok", "no_target"):
                think = reasoning
                if not think:
                    # models that reason in-content: prose outside the
                    # code fence is the reasoning channel
                    think = re.sub(r"```.*?```", "", raw, flags=re.S).strip()
                resp = ((f"<think>\n{think}\n</think>\n" if think else "")
                        + "```python\n" + code.rstrip() + "\n```\n")
                row = {"prompt": p, "response": resp,
                       "reasoning": think,
                       "target": "qwen_distill",
                       "verified": status == "ok",
                       "semantic": status}
                with _print_lock:
                    out_f.write(json.dumps(row) + "\n")
                    out_f.flush()
                stats["ok" if status == "ok" else status] += 1
            else:
                stats[status] = stats.get(status, 0) + 1
                _log(f"[gen] {i} drop {status}: {last_detail[:80]}")
        except Exception as e:
            stats["gen_err"] += 1
            _log(f"[gen] {i} gen error: {str(e)[:120]}")
        done[0] += 1
        if done[0] % 20 == 0:
            _log(f"[gen] {done[0]}/{len(prompts)} {stats}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as out_f:
        items = list(enumerate(prompts))
        if args.batch and args.backend == "mistral":
            _mistral_batch(api_url, api_key, model_name, prompts,
                           args.max_new, oracles, stats, out_f,
                           timeout_h=args.batch_timeout_h)
        elif args.backend in ("groq", "deepseek", "mistral",
                              "cf", "or", "poll", "llm7", "ovh"):
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(work, items))
        elif args.hf_batch > 1:
            import random as _rnd
            for j in range(0, len(items), args.hf_batch):
                grp = items[j:j + args.hf_batch]
                raws = _hf_gen_batch(
                    model, tok, [p for _, p in grp], args.max_new,
                    temperature=max(0.2, args.temperature or 0.7)
                    + _rnd.uniform(-0.1, 0.1),
                    seed=_rnd.randrange(1 << 60))
                for (i, p), raw in zip(grp, raws):
                    work((i, p), raw0=raw)
        else:
            for item in items:
                work(item)

    total = sum(stats.values())
    _log(f"[gen] done: {stats} kept={stats['ok'] + stats['no_target']}"
         f"/{total} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
