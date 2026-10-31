"""LFM3 evolve-proposer throughput + graph-equality harness.

Loads the LFM3 checkpoint the way LLMProposer._load_engine_obj does
(ForgeQuant _fq dir, cuBLAS dequant-cached linears) and measures
generate_batch decode throughput in both modes:

  functional  — eager forward per step (the pre-fix path)
  static      — bucketed static-KV CUDA graphs (use_breakable_cuda_graph)

--equality runs a greedy batch through BOTH modes and compares decoded
text row-by-row — the token-equality check the B>8 OOV bug needed before
static graphs could ship to the evolve loop.

Run:
    venv\\Scripts\\python.exe scripts\\bench_lfm3_evolve.py
    venv\\Scripts\\python.exe scripts\\bench_lfm3_evolve.py --b 4 8 16 --tokens 256
    venv\\Scripts\\python.exe scripts\\bench_lfm3_evolve.py --equality
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "research" / "checkpoints" / "lfm3_1p7b_cc_fq"


def _set_ckpt(p: Path) -> None:
    global CKPT
    CKPT = p

# Basin-like prompt: long shared instruction prefix + per-row parent code
# + a divergence hint (mirrors _build_prompt/_with_hint conditioning).
_SHARED = (
    "You are improving a hot-path kernel inside the ForgeAI inference "
    "engine. The function below is the incumbent. Produce ONE improved "
    "implementation — same signature, same semantics, faster. Only output "
    "code inside a single ```python fence.\n\n"
    "Constraints: pure torch, no python loops over the token dimension, "
    "must handle the general case exactly. The eval harness benchmarks "
    "your candidate against the incumbent on randomized inputs.\n\n"
    "Parent implementation:\n```python\n"
)

_PARENT = (
    "def {name}(logits, values, k):\n"
    "    \"\"\"Reference implementation to beat.\"\"\"\n"
    "    B, V = logits.shape\n"
    "    out = torch.zeros_like(values)\n"
    "    for i in range(k):\n"
    "        w = logits[:, i].softmax(-1)\n"
    "        out = out + w.unsqueeze(-1) * values[:, i]\n"
    "    return out\n"
)

_HINT = ("\n```\n# Try this approach: {hint}\n"
         "Respond with only the improved ```python block.\n")


def _prompts(n: int) -> list[str]:
    hints = [
        "vectorize with pure torch tensor ops, no python loops",
        "precompute and cache repeated work (LUTs, masks, cumsums)",
        "fuse sequential passes into one traversal",
        "use bit-level tricks / integer arithmetic where legal",
        "use torch.sort/topk/scatter/einsum primitives cleverly",
        "restructure for early-exit / short-circuit on common cases",
        "eliminate temporaries — in-place and fused kernels",
        "reformulate the math (log-space, deferred normalize, approx)",
        "batch dimensions differently for better parallelism",
        "hybrid: keep exact semantics but reorder expensive steps",
    ]
    out = []
    for i in range(n):
        p = _SHARED + _PARENT.format(name=f"kernel_v{i}") + \
            _HINT.format(hint=hints[i % len(hints)])
        out.append(p)
    return out


def _load(static: bool, compile_: bool, cache_w: bool,
          wrappers: bool = True):
    from forge.engine.forge_engine import ForgeEngine
    t0 = time.time()
    eng = ForgeEngine.from_checkpoint(
        str(CKPT), config_name="lfm3_1p7b", device="cuda",
        quantize=None, load_quantize=None,
        kv_cache="standard", use_compile=compile_,
        use_breakable_cuda_graph=static,
        use_block_fusion=False,
        use_chunked_prefix_cache=True,
        use_learned_prefix_cache=False,
        use_adaptive_spec=False, use_suffix_spec=False,
        use_faser=False, use_lookahead_gate=False,
        use_hotprefix=False, use_pod_attention=False,
        use_hybrid_prefill=False,
        use_fused_qk_norm_rope_cache=wrappers,
        use_seq_split=wrappers)
    n_cu = 0
    if cache_w:
        for m in eng.model.modules():
            if (hasattr(m, "_fused_gemv")
                    and hasattr(m, "_dequantize_weight")):
                m._fused_gemv = False
                m._cache_w = True
                n_cu += 1
    print(f"[load] static={static} compile={compile_} cache_w={cache_w} "
          f"({n_cu} dequant-cached) {time.time() - t0:.1f}s  "
          f"vram={torch.cuda.memory_allocated() / 2**30:.2f}GB")
    return eng


def _pad_to_b(prompts, b):
    if len(prompts) >= b:
        return prompts[:b]
    return prompts + [prompts[i % len(prompts)]
                      for i in range(b - len(prompts))]


def _bench(eng, prompts, max_new, label):
    B = len(prompts)
    # warmup — also triggers lazy per-batch graph capture at THIS B
    eng.generate_batch(prompts, max_new_tokens=8, temperature=0.0,
                       top_k=1)
    torch.cuda.synchronize()
    t0 = time.time()
    outs = eng.generate_batch(
        prompts, max_new_tokens=max_new, temperature=0.8, top_p=0.95,
        top_k=80,
        temperatures=[0.8 + 0.05 * (i % 8) for i in range(B)],
        repetition_penalties=[1.0 + 0.05 * (i % 4) for i in range(B)],
        seeds=[1234 + i for i in range(B)],
        stops=[[IM_END]] * B, skip_special_tokens=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_tok = sum(len(eng.tokenizer(o, add_special_tokens=False).input_ids)
                for o in outs)
    tps = n_tok / dt
    print(f"[{label}] B={B} new={max_new} wall={dt:.1f}s "
          f"tokens={n_tok} aggregate={tps:.0f} tok/s "
          f"per-row={tps / B:.1f} tok/s  "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GB")
    return outs, tps


def _equality(eng_func, prompts, max_new, b):
    """Greedy outputs: functional first (engine then slept to free VRAM
    for the static load), static second. Returns (func_outs, static_eng)
    — comparison happens per-B in main."""
    outs_f = eng_func.generate_batch(
        prompts, max_new_tokens=max_new, temperature=0.0, top_k=1)
    return outs_f


def _compare(outs_s, outs_f, prompts, max_new):
    bad = 0
    for i, (a, b_) in enumerate(zip(outs_s, outs_f)):
        if a != b_:
            bad += 1
            print(f"  row {i}: DIVERGES\n    static: {a[:160]!r}\n"
                  f"    func:   {b_[:160]!r}")
    print(f"[equality] {len(prompts) - bad}/{len(prompts)} rows "
          f"identical (max_new={max_new})")
    return bad == 0


IM_END = chr(60) + "|im_end" + chr(62) + "|"  # chat-turn terminator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--no-func", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-cache-w", action="store_true")
    ap.add_argument("--equality", action="store_true")
    ap.add_argument("--no-wrappers", action="store_true",
                    help="disable fused_qk_norm_rope_cache + seq_split "
                         "on the functional reference (kernel-parity "
                         "with the static path)")
    args = ap.parse_args()

    _set_ckpt(Path(args.ckpt))
    max_b = max(args.b)
    prompts = _pad_to_b(_prompts(max_b), max_b)
    plens = [len(p) for p in prompts]
    print(f"[setup] prompts B<={max_b}, chars {min(plens)}-{max(plens)}")

    eng_f = eng_s = None
    if not args.no_func:
        eng_f = _load(static=False, compile_=False,
                      cache_w=not args.no_cache_w,
                      wrappers=not args.no_wrappers)
    if not args.no_static:
        eng_s = _load(static=True, compile_=not args.no_compile,
                      cache_w=not args.no_cache_w)

    if args.equality:
        if eng_f is None:
            eng_f = _load(static=False, compile_=False,
                          cache_w=not args.no_cache_w,
                          wrappers=not args.no_wrappers)
        # Functional outputs per-B first, then sleep the engine so the
        # static load doesn't fight it for VRAM.
        ref = {}
        for b in args.b:
            ref[b] = _equality(eng_f, _pad_to_b(prompts, b),
                               min(args.tokens, 96), b)
        try:
            eng_f.sleep(2)
        except Exception:
            eng_f = None
        eng_s = _load(static=True, compile_=not args.no_compile,
                      cache_w=not args.no_cache_w)
        for b in args.b:
            outs_s = eng_s.generate_batch(
                _pad_to_b(prompts, b), max_new_tokens=min(args.tokens, 96),
                temperature=0.0, top_k=1)
            ok = _compare(outs_s, ref[b], _pad_to_b(prompts, b),
                          min(args.tokens, 96))
            print(f"  B={b}: {'PASS' if ok else 'FAIL'}")
        return 0

    for b in args.b:
        ps = _pad_to_b(prompts, b)
        if eng_f is not None:
            _bench(eng_f, ps, args.tokens, f"functional B={b}")
        if eng_s is not None:
            _bench(eng_s, ps, args.tokens, f"static     B={b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
