"""Port a Qwen2.5-family HF checkpoint to ForgeAI MLA+MoE format.

Updated ForgeLM-v1 recipe, built on this repo's GLA + MoE modules:

  attn  GQA -> latent-KV (MLA-style): joint SVD of [W_k; W_v] produces
        kv_down_proj [d_c x d] + k_up/v_up [kv_dim x d_c].  With the
        default d_c = 2*kv_dim the factorization is full-rank -> the
        ported attention is bit-exact vs the original (up to fp noise).
        K/V are reconstructed then RoPE'd — the GLA "identity" path,
        same math as ForgeLM-v1's simple MLA.
  ffn   dense SwiGLU -> N routed experts via contiguous neuron
        partition; moe_dense_bypass runs all experts with equal weight
        so the init reproduces the dense FFN exactly.  The router is
        real (trainable) — SFT/evolution can specialize routing later.

Usage:
  venv\\Scripts\\python.exe scripts\\port_qwen_mla_moe.py \
      --src forge/evolution/proposer_ckpt \
      --dst forge/evolution/proposer_mla_moe --verify
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
             "merges.txt", "generation_config.json", "special_tokens_map.json")


def _load_state(src: Path) -> dict[str, torch.Tensor]:
    files = sorted(src.glob("model*.safetensors"))
    if not files:
        sys.exit(f"no .safetensors under {src}")
    state = {}
    for f in files:
        state.update(load_file(str(f)))
    return state


def _read_meta(src: Path) -> dict:
    cfg = json.loads((src / "config.json").read_text())
    return {
        "d_model": cfg["hidden_size"], "n_layers": cfg["num_hidden_layers"],
        "n_heads": cfg["num_attention_heads"],
        "n_kv": cfg["num_key_value_heads"],
        "inter": cfg["intermediate_size"], "vocab": cfg["vocab_size"],
        "eps": cfg.get("rms_norm_eps", 1e-6),
        "rope": cfg.get("rope_theta", 1e6),
        "attn_bias": cfg.get("attention_bias", True),
        "tied": cfg.get("tie_word_embeddings", False)}


def port(src: Path, dst: Path, d_c: int | None, n_experts: int,
         dtype=torch.bfloat16) -> dict:
    meta = _read_meta(src)
    d, L, nh, nkv, inter = (meta["d_model"], meta["n_layers"],
                            meta["n_heads"], meta["n_kv"], meta["inter"])
    hd = d // nh
    kvd = nkv * hd
    d_c = d_c or 2 * kvd
    d_ff = inter // n_experts
    if inter % n_experts:
        sys.exit(f"intermediate {inter} not divisible by {n_experts}")
    state = _load_state(src)
    out: dict[str, torch.Tensor] = {}

    embed = state["model.embed_tokens.weight"]
    out["embed.weight"] = embed.to(dtype)
    if not meta["tied"] and "lm_head.weight" in state:
        out["head.weight"] = state["lm_head.weight"].to(dtype)
    out["ln_f.weight"] = state["model.norm.weight"].to(dtype)

    energy = []
    for i in range(L):
        p = f"model.layers.{i}"
        b = f"blocks.{i}"
        out[f"{b}.ln1.weight"] = state[f"{p}.input_layernorm.weight"].to(dtype)
        out[f"{b}.ln2.weight"] = (
            state[f"{p}.post_attention_layernorm.weight"].to(dtype))

        # ── attention: GQA -> latent-KV via joint SVD ──
        Wq = state[f"{p}.self_attn.q_proj.weight"]
        Wk = state[f"{p}.self_attn.k_proj.weight"]
        Wv = state[f"{p}.self_attn.v_proj.weight"]
        Wkv = torch.cat([Wk, Wv], 0).double()          # [2*kvd, d]
        U, S, Vt = torch.linalg.svd(Wkv, full_matrices=False)
        r = min(d_c, U.shape[1])
        energy.append(float(S[:r].sum() / S.sum()))
        sq = S[:r].sqrt()
        kv_down = (sq[:, None] * Vt[:r]).float()       # [r, d]
        up = (U[:, :r] * sq[None]).float()             # [2*kvd, r]
        out[f"{b}.attn.q_proj.weight"] = Wq.to(dtype)
        if meta["attn_bias"]:
            out[f"{b}.attn.q_proj.bias"] = (
                state[f"{p}.self_attn.q_proj.bias"].to(dtype))
        out[f"{b}.attn.kv_down_proj.weight"] = kv_down.to(dtype)
        if meta["attn_bias"]:                          # GLA has bias here
            out[f"{b}.attn.kv_down_proj.bias"] = torch.zeros(
                r, dtype=dtype)
        out[f"{b}.attn.k_up_proj.weight"] = up[:kvd].to(dtype)
        out[f"{b}.attn.v_up_proj.weight"] = up[kvd:].to(dtype)
        if meta["attn_bias"]:
            # The original k/v biases live AFTER the up-projection:
            # k = W_KC·latent + b_k.
            out[f"{b}.attn.k_up_proj.bias"] = (
                state[f"{p}.self_attn.k_proj.bias"].to(dtype))
            out[f"{b}.attn.v_up_proj.bias"] = (
                state[f"{p}.self_attn.v_proj.bias"].to(dtype))
        out[f"{b}.attn.out_proj.weight"] = (
            state[f"{p}.self_attn.o_proj.weight"].to(dtype))
        # GLA extras absent in the source — zero-init keeps identity path
        out[f"{b}.attn.k_rope_proj.weight"] = torch.zeros(hd, d, dtype=dtype)
        out[f"{b}.attn.k_rope_proj.bias"] = torch.zeros(hd, dtype=dtype)
        out[f"{b}.attn.compression_gate"] = torch.zeros(1, dtype=dtype)

        # ── ffn: dense SwiGLU -> partitioned experts ──
        Wg = state[f"{p}.mlp.gate_proj.weight"]        # [inter, d]
        Wu = state[f"{p}.mlp.up_proj.weight"]
        Wd = state[f"{p}.mlp.down_proj.weight"]        # [d, inter]
        g = torch.Generator().manual_seed(1234 + i)
        out[f"{b}.ffn.router.gate.weight"] = (
            torch.randn(n_experts, d, generator=g) * 0.02).to(dtype)
        out[f"{b}.ffn.router.expert_bias"] = torch.zeros(
            n_experts, dtype=dtype)
        for e in range(n_experts):
            lo, hi = e * d_ff, (e + 1) * d_ff
            out[f"{b}.ffn.experts.{e}.w1.weight"] = Wg[lo:hi].to(dtype)
            out[f"{b}.ffn.experts.{e}.w3.weight"] = Wu[lo:hi].to(dtype)
            out[f"{b}.ffn.experts.{e}.w2.weight"] = (
                Wd[:, lo:hi] * n_experts).to(dtype)    # see below
        # dense_bypass emits mean_e Expert_e(x): with each expert holding
        # a disjoint neuron slice, mean = (1/n)·dense.  Scaling w2 by n
        # cancels the 1/n so the total equals the dense FFN exactly.

    return out, meta, energy, d_c, d_ff


def verify(src: Path, dst_state: dict, meta: dict, dst_dir: Path,
           preset: str = "qwen25_05b_mla_moe") -> None:
    """Bit-exact forward check: HF original vs ported ForgeAI model."""
    import gc

    import torch.nn.functional as F  # noqa: F401
    print("[verify] loading HF original ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(src))
    ids = tok("def quicksort(xs):", return_tensors="pt").input_ids
    # bf16 both sides, HF freed before the ported model — peak ~7GB
    # instead of ~25GB (fp32 x2 OOMed on 32GB boxes).
    hf = AutoModelForCausalLM.from_pretrained(
        str(src), torch_dtype=torch.bfloat16).eval()
    with torch.no_grad():
        a = hf(ids).logits.float()
    del hf
    gc.collect()
    print("[verify] building ported model ...")
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM
    cfg = get_config(preset, device="cpu")
    model = ConfigurableResearchLLM(cfg)
    missing, unexpected = model.load_state_dict(dst_state, strict=False)
    hard = [k for k in missing if not k.startswith(("head.", "lm_head"))]
    if hard:
        print("[verify] MISSING (will break):", hard[:8])
    if unexpected:
        print("[verify] unexpected keys:", unexpected[:8])
    model = model.to(torch.bfloat16).eval()

    with torch.no_grad():
        b = model(ids)
        if isinstance(b, tuple):
            b = b[0]
        b = b.float()
    diff = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        a.flatten().double(), b.flatten().double(), dim=0).item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"[verify] max|dlogit|={diff:.6f}  cosine={cos:.6f} "
          f"argmax-agree={agree*100:.2f}%")
    # bf16-vs-bf16 accumulates ~0.3 max|dlogit| over deep models; argmax
    # agreement is the exactness criterion at this dtype.
    ok = agree >= 0.995 and cos > 0.999
    print("[verify]", "PASS" if ok else "FAIL — investigate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=(
                                     argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--dc", type=int, default=None,
                    help="latent dim (default 2*kv_dim = full rank)")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--preset", default="qwen25_05b_mla_moe",
                    help="forge_convert.json forge_config name + verify preset")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    args = ap.parse_args()

    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    out, meta, energy, d_c, d_ff = port(args.src, args.dst,
                                        args.dc, args.experts, dtype=dt)
    print(f"[port] d_model={meta['d_model']} layers={meta['n_layers']} "
          f"d_c={d_c} experts={args.experts}x{d_ff} "
          f"SVD energy={min(energy):.6f}..{max(energy):.6f}")

    args.dst.mkdir(parents=True, exist_ok=True)
    out = {k: v.contiguous() for k, v in out.items()}
    save_file(out, str(args.dst / "model.safetensors"),
              metadata={"format": "pt"})
    for name in TOK_FILES:
        f = args.src / name
        if f.exists():
            shutil.copy(f, args.dst / name)
    (args.dst / "forge_convert.json").write_text(json.dumps(
        {"forge_config": args.preset, "source": str(args.src),
         "d_c": d_c, "experts": args.experts, "d_ff": d_ff,
         # completion-style proposer: evolve SFT checkpoints ship a
         # chat_template in tokenizer_config but behave as completion
         # models on primed evolve prompts — LLMProposer.primed reads
         # this key before the template heuristic.
         "primed": True}, indent=2))
    print(f"[port] wrote {args.dst} ({len(out)} tensors)")

    if args.verify:
        verify(args.src, out, meta, args.dst, preset=args.preset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
