"""Static-vs-functional decode divergence probe for LFM3.

Three passes over the SAME reference token stream, logits compared
per step:
  (1) functional   — past_key_values cache, the proven-correct path
  (2) eager-static — model called with (mask_view, preallocated_cache)
                     signature, same inputs the graph sees, no replay
  (3) replay       — captured CUDA graph replay

If (2)==(1) but (3)!=(2): capture baked wrong ops. If (2)!=(1): the
static-cache eager path itself is wrong (pos_t/mask/conv-state bug).

Run: venv\\Scripts\\python.exe scripts\\dbg_lfm3_static.py
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forge.model.kv_cache import unpack_output_with_kv  # noqa: E402

CKPT = ROOT / "research" / "checkpoints" / "lfm3_1p7b_cc_fq"
PROMPT = ("Write a Python function fizzbuzz(n) that prints Fizz for "
          "multiples of 3, Buzz for 5, FizzBuzz for 15.\n\n```python\n"
          "def fizzbuzz(n):\n")


def load():
    from forge.engine.forge_engine import ForgeEngine
    t0 = time.time()
    eng = ForgeEngine.from_checkpoint(
        str(CKPT), config_name="lfm3_1p7b", device="cuda",
        quantize=None, load_quantize=None,
        kv_cache="standard", use_compile=False,
        use_breakable_cuda_graph=True,
        use_block_fusion=False,
        use_chunked_prefix_cache=False,
        use_learned_prefix_cache=False,
        use_adaptive_spec=False, use_suffix_spec=False,
        use_faser=False, use_lookahead_gate=False,
        use_hotprefix=False, use_pod_attention=False,
        use_hybrid_prefill=False)
    for m in eng.model.modules():
        if (hasattr(m, "_fused_gemv")
                and hasattr(m, "_dequantize_weight")):
            m._fused_gemv = False
            m._cache_w = True
    print(f"[load] {time.time() - t0:.1f}s vram="
          f"{torch.cuda.memory_allocated() / 2**30:.2f}GB")
    return eng


def functional_pass(fwd, ids, n, with_mask=False):
    """past_kv greedy; returns (tokens, per-step last logits).
    with_mask=True routes through the attention_bias/position_ids
    path (explicit additive mask) while keeping functional past_kv —
    isolates 'mask path' from 'preallocated-cache path'."""
    toks, lgs = [], []
    dev = ids.device
    with torch.inference_mode():
        pm = (torch.ones(1, ids.shape[1], dtype=torch.long, device=dev)
              if with_mask else None)
        out = fwd(ids, attention_mask=pm, use_cache=True,
                  logits_last_only=True)
        logits, past = unpack_output_with_kv(out)
        plen = ids.shape[1]
        for i in range(n):
            lgs.append(logits[:, -1, :].float().clone())
            tok = logits[:, -1, :].argmax(-1, keepdim=True)
            toks.append(int(tok.item()))
            dm = (torch.ones(1, plen + i + 1, dtype=torch.long,
                             device=dev) if with_mask else None)
            out = fwd(tok, attention_mask=dm, past_key_values=past,
                      use_cache=True, logits_last_only=True)
            logits, past = unpack_output_with_kv(out)
    return toks, lgs


def _prefill_static(fwd, ids, cache, mask_buf):
    cache.reset()
    mask_buf.zero_()
    plen = ids.shape[1]
    with torch.inference_mode():
        pmask = torch.ones(1, plen, dtype=torch.long, device=ids.device)
        out = fwd(ids, attention_mask=pmask, preallocated_cache=cache,
                  use_cache=True, logits_last_only=True)
        logits, _ = unpack_output_with_kv(out)
    cache.position = plen
    mask_buf[:, :plen] = 1
    return logits


def _bucket(buckets, graphs, need):
    for b in buckets:
        if b >= need and f"kv{b}" in graphs:
            return b, f"kv{b}"
    return None, None


def eager_static_pass(fwd, ids, ref_toks, cache, mask_buf, buckets,
                      graphs):
    """Same static-cache signature as the captured step, run eagerly.
    Feeds ref tokens (not own argmax) so logits are comparable."""
    _prefill_static(fwd, ids, cache, mask_buf)
    lgs = []
    with torch.inference_mode():
        for t in ref_toks:
            pos = cache.position
            b, tag = _bucket(buckets, graphs, pos + 1)
            assert tag, f"no bucket for pos {pos}"
            mask_buf[:, pos] = 1
            cache.pos_t.fill_(pos)
            cache.view_len = b  # capture sets this per bucket; eager
            # reads it live
            inp = torch.tensor([[t]], dtype=torch.long,
                               device=ids.device)
            out = fwd(inp, attention_mask=mask_buf[:, :b],
                      preallocated_cache=cache, use_cache=True,
                      logits_last_only=True)
            logits, _ = unpack_output_with_kv(out)
            # NOTE: no cache.position bump — the model's forward calls
            # preallocated_cache.advance(T) itself (llm.py ~647). Replay
            # doesn't run python-side advance, so replay_pass DOES bump.
            lgs.append(logits[:, -1, :].float().clone())
    return lgs


def replay_pass(ids, ref_toks, cache, mask_buf, buckets, graphs, bufs):
    """Captured graph replay, feeding ref tokens via the static input
    buffer."""
    _prefill_static(fwd_global, ids, cache, mask_buf)
    lgs = []
    with torch.inference_mode():
        for t in ref_toks:
            pos = cache.position
            b, tag = _bucket(buckets, graphs, pos + 1)
            assert tag, f"no bucket for pos {pos}"
            mask_buf[:, pos] = 1
            cache.pos_t.fill_(pos)
            bufs[tag]["input_ids"].copy_(
                torch.tensor([[t]], dtype=torch.long,
                             device=ids.device))
            graphs[tag].replay()
            logits, _ = unpack_output_with_kv(bufs[tag]["output"])
            cache.position += 1
            lgs.append(logits[:, -1, :].float().clone())
    return lgs


fwd_global = None


def main():
    global fwd_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--blocks", action="store_true")
    ap.add_argument("--step-bisect", action="store_true")
    args = ap.parse_args()

    eng = load()
    fwd_global = getattr(eng.model, "_orig_mod", eng.model)
    ids = eng.tokenizer(PROMPT, return_tensors="pt").input_ids.to(
        eng.device)
    state = eng._decode_state
    cache, mask_buf, buckets = (state["cache"], state["mask"],
                                state["buckets"])
    graphs = eng._bcg._decode_graphs
    bufs = eng._bcg._static_buffers
    print(f"[prompt] {ids.shape[1]} tok | buckets={buckets}")

    toks, lg_f = functional_pass(fwd_global, ids, args.n)
    print(f"[func tokens] {toks}")
    print(f"  {eng.tokenizer.decode(toks)!r}")

    # Prefill-vs-prefill: does the static-cache signature change the
    # prefill's last logits? (bias-path vs flash-path / KV layout)
    cache.reset()
    mask_buf.zero_()
    with torch.inference_mode():
        out1 = fwd_global(ids, use_cache=True, logits_last_only=True)
        l1, _ = unpack_output_with_kv(out1)
        pmask = torch.ones(1, ids.shape[1], dtype=torch.long,
                           device=ids.device)
        out2 = fwd_global(ids, attention_mask=pmask,
                          preallocated_cache=cache, use_cache=True,
                          logits_last_only=True)
        l2, _ = unpack_output_with_kv(out2)
    cache.position = ids.shape[1]
    mask_buf[:, :ids.shape[1]] = 1
    d = (l1[:, -1, :].float() - l2[:, -1, :].float()).abs()
    print(f"[prefill diff] max={d.max().item():.4f} "
          f"argmax f={int(l1[:, -1, :].argmax())} "
          f"s={int(l2[:, -1, :].argmax())}")

    if args.blocks and d.max().item() > 0.1:
        # Per-block bisection: record each block's output under both
        # signatures, find first divergence.
        origs = [b.forward for b in fwd_global.blocks]
        def mk(orig, sink):
            def f(*a, **k):
                r = orig(*a, **k)
                sink.append(r[0].detach().clone())
                return r
            return f
        for label, kw in (("func", dict(use_cache=True)),
                          ("stat", dict(attention_mask=pmask,
                                        preallocated_cache=cache,
                                        use_cache=True))):
            taps = []
            for b, o in zip(fwd_global.blocks, origs):
                b.forward = mk(o, taps)
            cache.reset()
            mask_buf.zero_()
            with torch.inference_mode():
                fwd_global(ids, **kw, logits_last_only=True)
            for b, o in zip(fwd_global.blocks, origs):
                b.forward = o
            if label == "func":
                taps_f = taps
            else:
                taps_s = taps
        print("[block bisect] idx kind   max-diff")
        first = True
        for i, (a, b) in enumerate(zip(taps_f, taps_s)):
            dd = (a.float() - b.float()).abs().max().item()
            blk = fwd_global.blocks[i]
            kind = "conv" if getattr(blk, "_is_conv", False) else \
                type(getattr(blk, "attn", blk)).__name__
            if dd > 0.05 or first:
                first = False
                print(f"  {i:>3} {kind:<28} {dd:>10.4f}"
                      f"{'   <-- FIRST' if dd > 0.05 else ''}")

    toks_m, lg_m = functional_pass(fwd_global, ids, args.n,
                                   with_mask=True)
    print(f"[func+mask] {toks_m}")
    md = max((a.float() - b.float()).abs().max().item()
             for a, b in zip(lg_f, lg_m))
    print(f"  masked-func logits max diff vs func: {md:.4f}")

    if args.step_bisect:
        # Single decode step (feed toks[0]) tapped per block under both
        # signatures — first divergent block = the off-by-one culprit.
        t0 = torch.tensor([[toks[0]]], dtype=torch.long,
                          device=ids.device)
        taps = {}
        origs = [b.forward for b in fwd_global.blocks]
        def mk(orig, tag):
            def f(*a, **k):
                r = orig(*a, **k)
                taps[tag].append(r[0].detach().clone())
                return r
            return f
        # functional: prefill then one token
        with torch.inference_mode():
            out = fwd_global(ids, use_cache=True, logits_last_only=True)
            _, past = unpack_output_with_kv(out)
            taps['f'] = []
            for b, o in zip(fwd_global.blocks, origs):
                b.forward = mk(o, 'f')
            fwd_global(t0, past_key_values=past, use_cache=True,
                       logits_last_only=True)
            for b, o in zip(fwd_global.blocks, origs):
                b.forward = o
            # static: prefill into cache then one token
            _prefill_static(fwd_global, ids, cache, mask_buf)
            pos = cache.position
            b_, tag = _bucket(buckets, graphs, pos + 1)
            mask_buf[:, pos] = 1
            cache.pos_t.fill_(pos)
            cache.view_len = b_
            taps['s'] = []
            for b, o in zip(fwd_global.blocks, origs):
                b.forward = mk(o, 's')
            fwd_global(t0, attention_mask=mask_buf[:, :b_],
                       preallocated_cache=cache, use_cache=True,
                       logits_last_only=True)
            for b, o in zip(fwd_global.blocks, origs):
                b.forward = o
        print("[step bisect] idx kind            max-diff")
        shown = 0
        for i, (a, b) in enumerate(zip(taps['f'], taps['s'])):
            dd = (a.float() - b.float()).abs().max().item()
            blk = fwd_global.blocks[i]
            kind = "conv" if getattr(blk, "_is_conv", False) else \
                ("swa" if getattr(getattr(blk, 'attn', blk),
                                  'sliding_window', None) else "gqa")
            if dd > 0.05 or shown < 2:
                print(f"  {i:>3} {kind:<16} {dd:>10.4f}"
                      f"{'   <-- FIRST' if dd > 0.05 and shown < 20 else ''}")
                shown += 1

    lg_e = eager_static_pass(fwd_global, ids, toks, cache, mask_buf,
                             buckets, graphs)
    lg_r = replay_pass(ids, toks, cache, mask_buf, buckets, graphs,
                       bufs)

    # lg_f[i] predicts toks[i] (after consuming toks[..i-1]);
    # lg_e/lg_r[i] are taken AFTER consuming toks[i] → compare them to
    # lg_f[i+1] (same position), not lg_f[i].
    print(f"{'step':>4} {'func-arg':>8} {'eag-arg':>8} {'rep-arg':>8}"
          f" {'dF-E':>9} {'dF-R':>9} {'dE-R':>9}")
    for i in range(len(toks) - 1):
        a_f = int(lg_f[i + 1].argmax())
        a_e = int(lg_e[i].argmax())
        a_r = int(lg_r[i].argmax())
        dfe = (lg_f[i + 1] - lg_e[i]).abs().max().item()
        dfr = (lg_f[i + 1] - lg_r[i]).abs().max().item()
        der = (lg_e[i] - lg_r[i]).abs().max().item()
        mark = " <-- DIVERGE" if not (a_f == a_e == a_r) else ""
        print(f"{i:>4} {a_f:>8} {a_e:>8} {a_r:>8} {dfe:>9.4f} "
              f"{dfr:>9.4f} {der:>9.4f}{mark}")


if __name__ == "__main__":
    main()
