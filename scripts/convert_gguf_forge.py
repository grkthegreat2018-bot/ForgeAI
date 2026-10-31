"""Convert a llama-arch GGUF (Mistral-Nemo family incl. Rocinante-X-12B) to a
ForgeAI HF-style safetensors checkpoint.

Why a converter and not the in-engine GGUF path (``_load_gguf_checkpoint``):
  - that path hardcodes rope_base=10000, ignores head_dim, never un-permutes
    the interleaved q/k row order, and materializes fp16 weights — a 12B
    model can never fit 12GB VRAM that way.
  - a one-time conversion produces a canonical bf16 master that loads via
    the fast pipelined path and can be packed to int4 on load
    (``load_quantize="int4"``).

Details:
  - dequantizes every tensor with the reference ``gguf`` package (correct for
    all K-quants — the hand-rolled engine dequantizer's Q4_K unpacker is
    approximate),
  - un-permutes ``attn_q``/``attn_k`` rows: llama.cpp's convert script
    interleaves RoPE pairs (reshape(h, hd//2, 2, d).swapaxes(1,2)); ForgeAI's
    ``RotaryEmbedding`` uses the HF half-split ``rotate_half`` convention, so
    the inverse permutation is applied here,
  - emits HF-format tensor names (``model.layers.N.self_attn.q_proj.weight``)
    which ``ModelLoader._remap_hf_keys`` maps to Forge names on load,
  - streams tensors one at a time into a single-file safetensors blob so
    peak host RAM stays ~one tensor (not 24GB).

Usage:
    python scripts/convert_gguf_forge.py --gguf <input.gguf> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── GGUF → HF tensor name map (llama arch) ────────────────────────────────────
_BLOCK_MAP = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
}
_TOP_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def gguf_to_hf_name(name: str) -> str | None:
    if name in _TOP_MAP:
        return _TOP_MAP[name]
    if name.startswith("blk."):
        _, idx, rest = name.split(".", 2)
        base = rest.rsplit(".weight", 1)[0]
        mapped = _BLOCK_MAP.get(base)
        if mapped is not None:
            return f"model.layers.{idx}.{mapped}.weight"
    return None


def unpermute_rope_rows(w: torch.Tensor, n_heads: int,
                        head_dim: int) -> torch.Tensor:
    """Inverse of llama.cpp's interleave permute on Q/K rows.

    ``w`` is HF-layout ``[out=n_heads*head_dim, in]``.  HF→GGUF applied
    ``w.reshape(h, hd//2, 2, d).swapaxes(1,2)`` so llama.cpp's adjacent-pair
    RoPE sees the right pairing; ForgeAI (like HF) uses rotate_half, so undo
    it: reshape(h, 2, hd//2, d).swapaxes(1,2).  ``head_dim`` is explicit
    because Nemo's d_model/n_heads (160) differs from head_dim (128).
    """
    h, hd, d = n_heads, head_dim, w.shape[1]
    assert w.shape[0] == h * hd, (w.shape, h, hd)
    return (w.reshape(h, 2, hd // 2, d)
             .transpose(1, 2)
             .reshape(w.shape))


def convert(gguf_path: str, out_dir: str, n_heads: int, n_kv_heads: int,
            dtype: torch.dtype = torch.bfloat16, permute: bool = True,
            head_dim: int = 128):
    import gguf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = gguf.GGUFReader(gguf_path)

    # Pass 1: build the safetensors header (name -> dtype/shape/offsets).
    entries: list[tuple[str, str, tuple[int, ...]]] = []
    for t in reader.tensors:
        hf = gguf_to_hf_name(t.name)
        if hf is None:
            print(f"  [skip] unmapped GGUF tensor {t.name}")
            continue
        # gguf.quants.dequantize returns the REVERSED t.shape — i.e.
        # numpy/HF layout [out, in] for mats, [vocab, d] for embeds.
        shape = tuple(int(d) for d in reversed(t.shape))
        entries.append((hf, "BF16" if dtype == torch.bfloat16 else "F16", shape))

    st_dtype = "BF16" if dtype == torch.bfloat16 else "F16"
    esize = 2
    header: dict[str, dict] = {}
    off = 0
    for name, dt, shape in entries:
        nbytes = esize * int(np.prod(shape))
        header[name] = {"dtype": dt, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes

    header_bytes = json.dumps(header).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad

    out_file = out_dir / "model.safetensors"
    t0 = time.time()
    total_gb = off / 1024**3
    print(f"[convert] {len(entries)} tensors -> {out_file} ({total_gb:.1f} GB)")

    with open(out_file, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)

        # Pass 2: stream tensors in header order.
        order = {t.name: i for i, t in enumerate(reader.tensors)}
        for hf_name in header:
            # find the gguf tensor mapped to this hf name (linear scan of
            # order map is fine — 363 entries)
            for gname, hf in ((t.name, gguf_to_hf_name(t.name))
                              for t in reader.tensors):
                if hf == hf_name:
                    t = reader.tensors[order[gname]]
                    arr = gguf.quants.dequantize(t.data, t.tensor_type)
                    w = torch.from_numpy(np.ascontiguousarray(arr))
                    if permute and hf_name.endswith("q_proj.weight"):
                        w = unpermute_rope_rows(w, n_heads, head_dim)
                    elif permute and hf_name.endswith("k_proj.weight"):
                        w = unpermute_rope_rows(w, n_kv_heads, head_dim)
                    w = w.to(dtype)
                    # bf16 isn't a numpy dtype — write the raw 2-byte
                    # payload through a uint16 view (fp16 takes the same
                    # path, keeping the code branch-free).
                    f.write(w.view(torch.uint16).numpy().tobytes())
                    break
            pct = 100.0 * f.tell() / (off + len(header_bytes) + 8)
            print(f"  {hf_name:60s} {pct:5.1f}%  "
                  f"{time.time() - t0:6.1f}s", flush=True)

    # Sidecar: detected arch + tokenizer provenance for the engine.
    sidecar = {
        "forge_config": "rocinante_12b",
        "source_gguf": str(gguf_path),
        "arch": str(reader.fields.get("general.architecture", "")),
        "tokenizer": "gguf-tekken",  # use GGUFTekkenTokenizer via gguf:path
    }
    (out_dir / "forge_convert.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[convert] done in {time.time() - t0:.1f}s -> {out_file}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--n-kv-heads", type=int, default=8)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--head-dim", type=int, default=128,
                    help="RoPE head_dim — Nemo decouples it from "
                         "d_model/n_heads (5120/32=160 != 128)")
    ap.add_argument("--no-permute", action="store_true",
                    help="skip the RoPE un-permutation (if conversion was "
                         "verified wrong for this file)")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    convert(args.gguf, args.out, args.n_heads, args.n_kv_heads,
            dtype=dt, permute=not args.no_permute, head_dim=args.head_dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
