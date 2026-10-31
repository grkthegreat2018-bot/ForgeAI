"""R50 + R49-2 GPU smoke test — real ForgeLM V2 (3.2B bf16).

Verifies on the actual checkpoint:
1. DRY penalty fires on real tokenized context + changes generation.
2. DoLa decodes on the real hybrid (dynamic premature-layer selection).
3. KDA port is bit-exact on the real V2 checkpoint (strict=True load of
   a convert_model_state'd state dict, gate=0 → max|dlogit| == 0).

Run: venv/Scripts/python.exe scripts/smoke_r50_gpu.py
"""
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = []


def check(name, ok, extra=""):
    PASS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          f"{(' — ' + extra) if extra else ''}")


def show(text):
    print("    >", " ".join(text.split())[:120])


def main():
    from forge.engine.decoding import DoLaDecoding, StandardDecoding
    from forge.engine.engine_common import _dry_penalties
    from forge.model_loader import load_default_model

    torch.manual_seed(0)
    print("Loading ForgeLM V2 (bf16, cuda)...")
    model, tok = load_default_model("forgelm_v2", device="cuda")
    print(f"  loaded — {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    # ── 1. DRY on real context + real generation ──────────────────────
    print("\n== DRY on real checkpoint ==")
    rep_prompt = ("The cat sat on the mat. The cat sat on the mat. "
                  "The cat sat on the mat. The cat sat")
    ids = tok(rep_prompt, return_tensors="pt").input_ids.cuda()
    ctx = ids[0].tolist()
    pen = _dry_penalties(ctx, last_n=512, allowed_length=2,
                         multiplier=1.0, base=1.75)
    check("DRY penalizes real repeated suffix", len(pen) > 0,
          f"{len(pen)} tokens penalized")
    top_pen = sorted(pen.items(), key=lambda kv: -kv[1])[:5]
    for tid, p in top_pen:
        print(f"    token {tid} ({tok.decode([tid])!r}): -{p:.2f} logits")

    std = StandardDecoding()
    torch.manual_seed(0)
    out_off = std.generate(model, ids, max_new_tokens=24,
                           temperature=0.8, top_p=0.9, top_k=64,
                           repetition_penalty=1.0)
    torch.manual_seed(0)
    out_on = std.generate(model, ids, max_new_tokens=24,
                          temperature=0.8, top_p=0.9, top_k=64,
                          repetition_penalty=1.0,
                          dry_multiplier=1.0, dry_base=1.75,
                          dry_allowed_length=2, dry_penalty_last_n=512)
    t_off = tok.decode(out_off[0, ids.shape[1]:])
    t_on = tok.decode(out_on[0, ids.shape[1]:])
    check("generate(dry) runs on V2", out_on.shape[1] > ids.shape[1])
    show(f"dry=off: {t_off}")
    show(f"dry=on:  {t_on}")
    check("DRY changes sampled continuation", t_off != t_on)

    # ── 2. DoLa on the real hybrid ────────────────────────────────────
    print("\n== DoLa on real checkpoint ==")
    prompt = "The capital of France is"
    ids2 = tok(prompt, return_tensors="pt").input_ids.cuda()
    dola = DoLaDecoding(candidate_top_k=64)  # dynamic layer selection
    out_d = dola.generate(model, ids2, max_new_tokens=12, temperature=0.0)
    t_dola = tok.decode(out_d[0, ids2.shape[1]:])
    out_g = std.generate(model, ids2, max_new_tokens=12, temperature=0.0)
    t_greedy = tok.decode(out_g[0, ids2.shape[1]:])
    check("DoLa generates on V2", out_d.shape[1] > ids2.shape[1])
    show(f"greedy: {t_greedy}")
    show(f"dola:   {t_dola}")
    # logits contract: contrasted distribution must be a valid logits row
    check("DoLa output decodes", len(t_dola.strip()) > 0)

    # ── 3. KDA bit-exact port on the REAL checkpoint ──────────────────
    print("\n== KDA bit-exact port on real V2 ==")
    ids3 = tok("The capital of France is", return_tensors="pt") \
        .input_ids.cuda()
    with torch.no_grad():
        ref = model(ids3)[0].float().cpu()  # baseline logits saved to CPU
    del model
    gc.collect(); torch.cuda.empty_cache()

    # Convert the real checkpoint state dict via KDAKey.convert_model_state
    import dataclasses

    from safetensors.torch import load_file

    from forge.config import get_config
    from forge.keys.attention.kda_key import KDAKey
    from forge.model.llm import ConfigurableResearchLLM
    from research.paths import V2_CHECKPOINT

    cfg_kda = dataclasses.replace(get_config("forgelm_v2"), use_kda=True)
    key = KDAKey(cfg_kda.d_model, cfg_kda.n_heads,
                 head_dim=cfg_kda.d_model // cfg_kda.n_heads)
    sd = load_file(str(V2_CHECKPOINT))
    res = key.convert_model_state(sd)
    check("convert_model_state on real checkpoint",
          res.success and len(res.metadata["blocks"]) == 28,
          f"{len(res.metadata['blocks'])} blocks ported")
    gates = [k for k in res.weights if k.endswith("_kda.gate")]
    check("all 28 kda gates zero",
          len(gates) == 28 and
          all(res.weights[g].abs().max() == 0 for g in gates))

    torch.manual_seed(0)
    m_kda = ConfigurableResearchLLM(cfg_kda)
    missing, unexpected = m_kda.load_state_dict(res.weights, strict=True)
    check("converted checkpoint loads strict=True",
          len(missing) == 0 and len(unexpected) == 0)
    m_kda = m_kda.to("cuda", torch.bfloat16).eval()
    with torch.no_grad():
        kda_logits = m_kda(ids3)[0].float().cpu()
    diff = (kda_logits - ref).abs().max().item()
    check("use_kda V2 bit-exact vs baseline checkpoint",
          diff == 0.0, f"max|dlogit|={diff}")
    del m_kda, res, sd
    gc.collect(); torch.cuda.empty_cache()

    n_ok = sum(1 for _, ok in PASS if ok)
    print(f"\n{n_ok}/{len(PASS)} checks passed")
    sys.exit(0 if n_ok == len(PASS) else 1)


if __name__ == "__main__":
    main()
