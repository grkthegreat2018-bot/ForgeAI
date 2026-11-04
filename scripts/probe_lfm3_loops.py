"""Probe: does zero-shot looping (Dory-style latent recurrence) make the
LFM3-cc finetune smarter, and is the speed cost worth it?

Context: NVIDIA Dory1.0-4B is TRAINED looped (I-R-O split, input
reinjection, runtime num_loops, last-loop KV). LFM3-cc is NOT — this
probe re-runs blocks[begin:end) K times per forward via the runtime
dial added to ConfigurableResearchLLM.forward (_loop_iters/_loop_begin/
_loop_end/_loop_reinject) and measures:

  1. perplexity on held-out-ish text at loops {1,2,4} x reinject {0,1}
  2. greedy generations on checkable prompts at loops {1,2,4}
  3. engine decode tok/s (functional KV path) per loop count
  4. analytic layer-pass multiplier (speed upper bound)

Run:
    venv\\Scripts\\python.exe scripts\\probe_lfm3_loops.py
    venv\\Scripts\\python.exe scripts\\probe_lfm3_loops.py --loops 1 2 4 --fq
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "research" / "checkpoints" / "lfm3_1p7b_cc"
CKPT_FQ = ROOT / "research" / "checkpoints" / "lfm3_1p7b_cc_fq"

IM_END = chr(60) + "|im_end" + chr(62) + ">"

# ── PPL corpus: code (cc use-case) + prose + structured text ────────────
_PPL_TEXTS = [
    (
        "def attention(query, key, value, mask=None):\n"
        "    scores = torch.matmul(query, key.transpose(-2, -1))\n"
        "    scores = scores / math.sqrt(query.size(-1))\n"
        "    if mask is not None:\n"
        "        scores = scores.masked_fill(mask == 0, -1e9)\n"
        "    weights = torch.softmax(scores, dim=-1)\n"
        "    return torch.matmul(weights, value)\n\n"
        "class KVCache:\n"
        "    def __init__(self, max_len, n_heads, head_dim):\n"
        "        self.k = torch.zeros(1, n_heads, max_len, head_dim)\n"
        "        self.v = torch.zeros(1, n_heads, max_len, head_dim)\n"
        "        self.pos = 0\n"
    ),
    (
        "The inference engine schedules requests in two phases. During "
        "prefill, the prompt tokens are processed in parallel and the "
        "key-value cache is populated for every layer. Decode then runs "
        "one token at a time; each step appends the new key and value "
        "vectors to the cache and samples the next token from the final "
        "logits. Because decode is memory-bound, throughput scales with "
        "batch size until the GPU saturates its memory bandwidth."
    ),
    (
        "import torch.nn as nn\n\n"
        "class RMSNorm(nn.Module):\n"
        "    def __init__(self, dim, eps=1e-6):\n"
        "        super().__init__()\n"
        "        self.weight = nn.Parameter(torch.ones(dim))\n"
        "        self.eps = eps\n\n"
        "    def forward(self, x):\n"
        "        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()\n"
        "        return x / rms * self.weight\n"
    ),
]

# ── checkable generation prompts (name, prompt, must-contain list) ──────
_TASKS = [
    ("math",  "What is 17 * 24? Answer with only the number.", ["408"]),
    ("rate",  "A train travels 60 km in 45 minutes. What is its speed "
              "in km/h? Answer with only the number.", ["80"]),
    ("capital", "What is the capital of France? One word.", ["Paris"]),
    ("fib",   "Write a Python function that returns the nth Fibonacci "
              "number. Respond with only a ```python code block.",
              ["def fib", "return"]),
    ("bugfix", "This function has a bug:\n```python\ndef add(a, b):\n"
               "    return a - b\n```\nReturn the corrected function in "
               "a ```python block.", ["a + b"]),
    ("kernel", "Improve this kernel — same signature, faster, vectorized "
               "torch only. Respond with only a ```python block.\n"
               "```python\ndef top_k_sum(logits, k):\n"
               "    total = 0\n"
               "    for i in range(k):\n"
               "        total += logits[i]\n"
               "    return total\n```", ["def top_k_sum"]),
]


def _lfm3_wrap(prompt: str, prime_code: bool = False) -> str:
    """Official LFM3 chat shape (code_evolve._wrap): startoftext BOS,
    im_start/im_end user turn, assistant turn with the EMPTY think pair
    (enable_thinking=False render). prime_code opens a ```python fence
    inside the assistant turn so the model continues code."""
    out = ("<|startoftext|><|im_start|>user\n" + prompt
           + "<|im_end|>\n<|im_start|>assistant\n<think></think>")
    if prime_code:
        out += "```python\n"
    return out


def set_loops(model, iters: int, b0: int, b1: int, reinject: float) -> None:
    model._loop_iters = iters
    model._loop_begin = b0
    model._loop_end = b1
    model._loop_reinject = reinject


def clear_loops(model) -> None:
    set_loops(model, 1, 0, 0, 0.0)


def load_engine(ckpt: Path):
    from forge.engine.forge_engine import ForgeEngine
    t0 = time.time()
    eng = ForgeEngine.from_checkpoint(
        str(ckpt), config_name="lfm3_1p7b", device="cuda",
        quantize=None, load_quantize=None,
        kv_cache="standard", use_compile=False,
        use_breakable_cuda_graph=False, use_block_fusion=False,
        use_chunked_prefix_cache=False, use_learned_prefix_cache=False,
        use_adaptive_spec=False, use_suffix_spec=False,
        use_faser=False, use_lookahead_gate=False,
        use_hotprefix=False, use_pod_attention=False,
        use_hybrid_prefill=False,
        use_fused_qk_norm_rope_cache=False, use_seq_split=False)
    print(f"[load] {ckpt.name} {time.time() - t0:.1f}s  "
          f"vram={torch.cuda.memory_allocated() / 2**30:.2f}GB")
    return eng


@torch.no_grad()
def ppl(eng, texts, chunk: int = 1024) -> float:
    nll = ntok = 0
    tok = eng.tokenizer
    for t in texts:
        ids = tok(t, add_special_tokens=True).input_ids
        x = torch.tensor(ids[:chunk + 1], dtype=torch.long,
                         device="cuda").unsqueeze(0)
        if x.shape[1] < 8:
            continue
        logits, _ = eng.model(x[:, :-1])
        l = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1),
            reduction="sum")
        nll += l.item()
        ntok += x.shape[1] - 1
    return math.exp(nll / max(ntok, 1))


def gen_at(eng, iters, b0, b1, reinject, tasks, max_new):
    set_loops(eng.model, iters, b0, b1, reinject)
    torch.cuda.synchronize()
    outs = eng.generate_batch(
        [_lfm3_wrap(p, prime_code=n in ("fib", "bugfix", "kernel"))
         for n, p, _ in tasks], max_new_tokens=max_new,
        temperature=0.0, top_k=1,
        stops=[[IM_END]] * len(tasks), skip_special_tokens=True)
    torch.cuda.synchronize()
    return outs


@torch.no_grad()
def decode_step_ms(eng, iters, b0, b1, steps: int = 40) -> float:
    """Fixed-step decode timing — avoids early-EOS bias in generate_batch.
    Prefills a real prompt once, then times raw model forward calls."""
    set_loops(eng.model, iters, b0, b1, 0.0)
    tok = eng.tokenizer
    ids = torch.tensor(tok(_lfm3_wrap(_TASKS[0][1]),
                           add_special_tokens=False).input_ids,
                       dtype=torch.long, device="cuda").unsqueeze(0)
    logits, _, past = eng.model(ids, use_cache=True)
    x = logits[:, -1:].argmax(-1)
    for _ in range(4):  # warmup
        logits, _, past = eng.model(x, past_key_values=past, use_cache=True)
        x = logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        logits, _, past = eng.model(x, past_key_values=past, use_cache=True)
        x = logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / steps


def score(name, out, needles) -> bool:
    return all(n in out for n in needles)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--reinject", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--begin", type=int, default=4)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--bench-tokens", type=int, default=128)
    ap.add_argument("--bench-b", type=int, default=8)
    ap.add_argument("--fq", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    args = ap.parse_args()

    b0, b1 = args.begin, args.end
    ckpt = CKPT_FQ if args.fq else CKPT
    eng = load_engine(ckpt)
    n_layers = len(eng.model.blocks)
    n_mid = b1 - b0
    print(f"[cfg] {n_layers} layers, loop region [{b0},{b1}) = {n_mid} mid")

    # ── 1. PPL sweep ────────────────────────────────────────────────────
    print("\n== perplexity ==")
    for reinj in args.reinject:
        row = []
        for k in args.loops:
            set_loops(eng.model, k, b0, b1, reinj)
            row.append(f"K{k}={ppl(eng, _PPL_TEXTS):.2f}")
        print(f"  reinject={reinj}: " + "  ".join(row))

    # ── 2. greedy generation A/B ────────────────────────────────────────
    print("\n== greedy generations ==")
    gen_results = {}
    for k in args.loops:
        for reinj in ([0.0] if k == 1 else args.reinject):
            outs = gen_at(eng, k, b0, b1, reinj, _TASKS, args.tokens)
            npass = sum(score(n, o, nd) for (n, _, nd), o in zip(_TASKS, outs))
            gen_results[(k, reinj)] = npass
            print(f"  K={k} reinject={reinj}: {npass}/{len(_TASKS)} checks pass")
            for (n, _, nd), o in zip(_TASKS, outs):
                ok = score(n, o, nd)
                tag = "PASS" if ok else "FAIL"
                snippet = (o.replace("\n", " ")[:110]
                           .encode("ascii", "replace").decode())
                print(f"    [{tag}] {n}: {snippet!r}")

    # ── 3. per-step decode cost (fixed steps, B=1 — no EOS bias) ────────
    if not args.skip_speed:
        print("\n== decode step cost (B=1, fixed steps, functional KV) ==")
        for k in args.loops:
            ms = decode_step_ms(eng, k, b0, b1)
            passes = b0 + k * n_mid + (n_layers - b1)
            print(f"  K={k}: {ms:.1f} ms/step ({1000 / ms:.0f} tok/s B=1)  "
                  f"layer-passes={passes}/{n_layers} "
                  f"({passes / n_layers:.2f}x compute)  "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB")

    clear_loops(eng.model)
    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
