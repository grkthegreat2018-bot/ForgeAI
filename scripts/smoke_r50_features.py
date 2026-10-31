"""R50 + R49-2 feature smoke test — CPU end-to-end.

Exercises every new path on a real (tiny) model instance and the real
ForgeLM tokenizer where applicable. Random weights → output is gibberish;
what's verified is that each path runs, shapes are right, and the
feature actually changes behavior vs. its disabled baseline.

Run: venv/Scripts/python.exe scripts/smoke_r50_features.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.config import get_config
from forge.engine.decision_head import fit_prm, score_steps
from forge.engine.decoding import DoLaDecoding, StandardDecoding, build_decoding
from forge.engine.engine_common import _dry_penalties
from forge.engine.kv.filler_kv import FillerKVCache, filler_token_ids
from forge.engine.kv_backend import build_kv_cache
from forge.model.llm import ConfigurableResearchLLM
from research.merge_models import depth_upscale, parse_layer_map

PASS = []


def check(name, ok, extra=""):
    PASS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")


def tiny_model(use_kda=False, n_layers=4):
    cfg = get_config("forgelm_tiny")
    if use_kda:
        import dataclasses
        cfg = dataclasses.replace(cfg, use_kda=True)
    torch.manual_seed(0)
    m = ConfigurableResearchLLM(cfg).eval()
    return m, cfg


# ── 1. DRY on a real generation path ─────────────────────────────────────────

def test_dry_generate():
    print("\n== DRY penalty (StandardDecoding, real model) ==")
    model, _ = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4]])
    out_off = StandardDecoding().generate(
        model, ids, max_new_tokens=8, temperature=0.8,
        repetition_penalty=1.0)
    out_on = StandardDecoding().generate(
        model, ids, max_new_tokens=8, temperature=0.8,
        repetition_penalty=1.0, dry_multiplier=1.0, dry_base=1.75,
        dry_allowed_length=2, dry_penalty_last_n=64)
    check("generate with dry_multiplier>0 runs",
          out_on.shape[1] > ids.shape[1])
    check("output shape sane", out_off.shape[1] > ids.shape[1]
          and out_off.shape[1] <= ids.shape[1] + 8)
    # penalty math on the real prompt context
    pen = _dry_penalties([1, 2, 3, 4, 1, 2, 3, 4], 64, 2, 1.0, 1.75)
    check("repeated suffix penalized", pen.get(1, 0) > 0,
          f"token 1 penalty={pen.get(1):.3f}")


# ── 2. DoLa on a real generation path ────────────────────────────────────────

def test_dola_generate():
    print("\n== DoLa decoding (real model, hidden-state path) ==")
    model, _ = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    d = build_decoding("dola", early_layer=1, candidate_top_k=32)
    out = d.generate(model, ids, max_new_tokens=5, temperature=0.0)
    check("DoLa generate runs", out.shape[1] >= ids.shape[1])
    # auto layer-selection path (JSD over candidates)
    d2 = DoLaDecoding()
    out2 = d2.generate(model, ids, max_new_tokens=3, temperature=0.0)
    check("DoLa dynamic premature-layer path runs",
          out2.shape[1] >= ids.shape[1])


# ── 3. Filler KV on the real tokenizer ───────────────────────────────────────

def test_filler_kv():
    print("\n== Filler KV (real tokenizer-derived filler set) ==")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "research/checkpoints/forgelm_v2_tokenizer")
    ids = filler_token_ids(tok)
    check("filler id set built", len(ids) > 20, f"{len(ids)} ids")
    the_ids = tok.encode(" the", add_special_tokens=False)
    check("' the' is a filler", all(i in ids for i in the_ids))

    cache = FillerKVCache(observation_window=8, budget=16, n_kv_heads=2,
                          head_dim=8, n_sink=4, filler_ids=ids,
                          device="cpu", dtype=torch.float32)
    # ~60 tokens of function-word-dense text → seq_len > max_capacity(24)
    # so eviction actually fires; fillers ("the","and","of",",") sit in
    # the unprotected middle region.
    text = ("the cat and the dog ran in the park and the boy and the "
            "girl sat on the bench in the shade of the tree and the "
            "wind blew over the field of grass and the sun set")
    toks = tok.encode(text, add_special_tokens=False)
    k = torch.randn(1, 2, len(toks), 8)
    v = torch.randn(1, 2, len(toks), 8)
    before_fillers = len(toks)
    cache.append(k, v, 0, token_ids=toks)
    check("cache evicted to capacity",
          cache.seq_len <= cache.max_capacity,
          f"{before_fillers} -> {cache.seq_len}")
    check("fillers evicted first", cache.filler_evicted > 0,
          f"{cache.filler_evicted} filler evictions")
    strat = build_kv_cache("filler")
    check("factory returns filler strategy",
          type(strat).__name__ == "FillerKVCacheStrategy")


# ── 4. PRM on the real tokenizer + tiny model ────────────────────────────────

def test_prm():
    print("\n== PRM head (real tokenizer, tiny model) ==")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "research/checkpoints/forgelm_v2_tokenizer")
    # Tiny model needs the real tokenizer's vocab range (65536).
    import dataclasses
    cfg = dataclasses.replace(get_config("forgelm_tiny"),
                              vocab_size=65536)
    torch.manual_seed(0)
    model = ConfigurableResearchLLM(cfg).eval()
    dataset = [
        ("What is 2+2?", ["Compute 2+2.", "The answer is 4."], [1, 1]),
        ("What is 2+2?", ["Guess wildly.", "The answer is 9."], [0, 0]),
        ("Capital of France?", ["Think of France.", "It is Paris."], 1),
        ("Capital of France?", ["Random guess.", "It is Mars."], 0),
        ("3*3?", ["Multiply.", "Nine."], [1, 1]),
        ("3*3?", ["Add.", "Six."], [0, 0]),
    ]
    head, metrics = fit_prm(model, tok, "cpu", dataset,
                            epochs=6, val_frac=0.2, progress=False, seed=0)
    check("fit_prm returns head + metrics",
          head is not None and metrics["steps"] == 12,
          str({k: metrics[k] for k in ("n_examples", "steps")}))
    probs = score_steps(head, model, tok, "Q?", ["step a", "step b"], "cpu")
    check("score_steps returns per-step probs",
          len(probs) == 2 and all(0.0 <= p <= 1.0 for p in probs),
          f"probs={probs}")


# ── 5. Depth upscale on a real state dict ────────────────────────────────────

def test_depth_upscale():
    print("\n== Depth upscale (real tiny model state dict) ==")
    model, cfg = tiny_model()
    state = model.state_dict()
    # layer_types = [conv,conv,attn,conv]; new slots 4,5 are conv →
    # duplicate conv sources 0,1 (type-consistent map).
    layer_map = parse_layer_map("0-3,0-1")  # 4 -> 6 blocks
    new_types = cfg.layer_types + ["conv", "conv"]
    out, _meta = depth_upscale(state, layer_map, layer_types=new_types)
    check("new state has 6 blocks",
          any(k.startswith("blocks.5.") for k in out))
    check("duplicated weights identical",
          torch.equal(out["blocks.4.ln1.weight"],
                      state["blocks.0.ln1.weight"]))
    check("non-block keys preserved",
          torch.equal(out["embed.weight"], state["embed.weight"]))
    # typed-layer consistency: derived types must match the target config
    from research.merge_models import depth_layer_types
    derived = depth_layer_types(state, layer_map, cfg.layer_types)
    check("derived types match target schedule", derived == new_types,
          str(derived))
    # reload into a 6-layer config and forward
    import dataclasses
    cfg6 = dataclasses.replace(
        cfg, n_layers=6, layer_types=new_types)
    m6 = ConfigurableResearchLLM(cfg6).eval()
    missing, unexpected = m6.load_state_dict(out, strict=False)
    check("upscaled state loads into 6-layer model",
          len(unexpected) == 0 and len(missing) == 0,
          f"missing={missing}, unexpected={len(unexpected)}")
    # negative: mapping a conv slot to an attention source block breaks
    # the load (why the typed-layer caveat exists)
    bad_out, _ = depth_upscale(state, [0, 1, 2, 3, 1, 2])
    bad_missing, bad_unexp = m6.load_state_dict(bad_out, strict=False)
    check("type-mismatched map detected via load errors",
          len(bad_missing) > 0 and len(bad_unexp) > 0)
    x = torch.randint(0, 256, (1, 6))
    with torch.no_grad():
        out6 = m6(x)
    logits = out6[0] if isinstance(out6, tuple) else out6
    check("upscaled model forward finite",
          torch.isfinite(logits).all().item())


# ── 6. KDA: bit-exact on the model + stateful decode ─────────────────────────

def test_kda():
    print("\n== KDA key (bit-exact on real model + decode path) ==")
    m_base, cfg = tiny_model()
    import dataclasses
    cfg_kda = dataclasses.replace(cfg, use_kda=True)
    m_kda = ConfigurableResearchLLM(cfg_kda).eval()
    ref_sd = m_base.state_dict()
    m_kda.load_state_dict(
        {k: v for k, v in ref_sd.items() if k in m_kda.state_dict()},
        strict=False)
    x = torch.randint(0, 256, (1, 7))
    with torch.no_grad():
        lo_base = m_base(x)[0]
        lo_kda = m_kda(x)[0]
    diff = (lo_kda - lo_base).abs().max().item()
    check("use_kda model bit-exact vs baseline", diff == 0.0,
          f"max|dlogit|={diff}")
    # open the gate on one block -> branch contributes
    blk = m_kda.blocks[0]
    blk._kda.gate.data.fill_(0.5)
    blk._kda_gate_zero = None
    with torch.no_grad():
        lo_open = m_kda(x)[0]
    check("opened gate changes output", not torch.equal(lo_open, lo_base))
    check("opened output finite", torch.isfinite(lo_open).all().item())
    # KV-cache decode path (use_cache) — prefill + one-step decode.
    # Open the gate so the KDA branch actually exercises its state.
    blk._kda.gate.data.fill_(0.5)
    blk._kda_gate_zero = None
    with torch.no_grad():
        out = m_kda(x[:, :5], use_cache=True)
        kv = out[2] if isinstance(out, tuple) and len(out) > 2 else out[1]
        out2 = m_kda(x[:, 5:], past_key_values=kv, use_cache=True)
    logits2 = out2[0] if isinstance(out2, tuple) else out2
    check("use_cache decode path runs + finite",
          torch.isfinite(logits2).all().item())


def main():
    torch.manual_seed(0)
    print("R50/R49-2 feature smoke test (CPU)")
    test_dry_generate()
    test_dola_generate()
    test_filler_kv()
    test_prm()
    test_depth_upscale()
    test_kda()
    n_ok = sum(1 for _, ok in PASS if ok)
    print(f"\n{n_ok}/{len(PASS)} checks passed")
    sys.exit(0 if n_ok == len(PASS) else 1)


if __name__ == "__main__":
    main()
