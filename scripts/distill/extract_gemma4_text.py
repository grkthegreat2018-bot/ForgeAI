"""Extract the text LM from the multimodal Gemma4Unified checkpoint.

The OBLITERATUS repo ships Gemma4UnifiedForConditionalGeneration weights
(model.language_model.* + small vision/audio embedders). This rewrites the
safetensors as a text-only checkpoint for Gemma4UnifiedForCausalLM /
AutoModelForCausalLM:

    model.language_model.X  ->  model.X
    (drops model.embed_vision.*, model.embed_audio.*)

Zero-copy: patches the JSON header and streams kept tensor byte ranges
(peak RAM ~ header size, not 24GB).

Usage (training venv):
    venv_train\\Scripts\\python.exe scripts\\distill\\extract_gemma4_text.py ^
        --src <hf-snapshot-dir> --dst checkpoints\\gemma4_oblit_text
"""
from __future__ import annotations

import argparse, json, os, shutil, struct

from huggingface_hub import snapshot_download

REPO = "OBLITERATUS/Gemma-4-12B-OBLITERATED"
CHUNK = 64 * 1024 * 1024


def stream_rewrite(src_path, dst_path):
    with open(src_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header.pop("__metadata__", None)
        # kept tensors, renamed, in original data order
        keep = {}
        for k, v in header.items():
            if k.startswith("model.language_model."):
                keep["model." + k[len("model.language_model."):]] = v
        order = sorted(keep, key=lambda k: keep[k]["data_offsets"][0])
        # new header with re-packed offsets
        new_header, off = {}, 0
        for k in order:
            v = dict(keep[k])
            lo, hi = v["data_offsets"]
            v["data_offsets"] = [off, off + hi - lo]
            off += hi - lo
            new_header[k] = v
        if meta:
            new_header["__metadata__"] = meta
        hb = json.dumps(new_header).encode()
        pad = (8 - (len(hb) % 8)) % 8
        hb += b" " * pad
        # copy file ranges: data section starts at 8 + n in src
        with open(dst_path, "wb") as o:
            o.write(struct.pack("<Q", len(hb)))
            o.write(hb)
            for k in order:
                lo, hi = header_order = keep[k]["data_offsets"]
                f.seek(8 + n + lo)
                left = hi - lo
                while left:
                    b = f.read(min(CHUNK, left))
                    if not b:
                        raise IOError(f"short read at {k}")
                    o.write(b)
                    left -= len(b)
    return len(new_header) - (1 if meta else 0), len(header) - len(new_header)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=r"checkpoints\gemma4_oblit_text")
    ap.add_argument("--token", default=None,
                    help="Hugging Face token (default: HF_TOKEN env var)")
    args = ap.parse_args()

    # EC-050: token comes from env/CLI — never hardcode credentials here.
    token = args.token or os.environ.get("HF_TOKEN")
    if args.src is None and not token:
        ap.error("HF_TOKEN env var or --token required to download REPO")

    src = args.src or snapshot_download(
        REPO,
        allow_patterns=["model.safetensors", "config.json",
                        "tokenizer.json", "tokenizer_config.json",
                        "chat_template.jinja", "generation_config.json"],
        token=token)
    print("src:", src, flush=True)

    os.makedirs(args.dst, exist_ok=True)
    kept, dropped = stream_rewrite(
        os.path.join(src, "model.safetensors"),
        os.path.join(args.dst, "model.safetensors"))
    print(f"{kept} text tensors kept, {dropped} non-text/header dropped",
          flush=True)

    cfg = json.load(open(os.path.join(src, "config.json")))
    text_cfg = cfg["text_config"]
    text_cfg["architectures"] = ["Gemma4UnifiedForCausalLM"]
    text_cfg["transformers_version"] = cfg.get("transformers_version", "5.10.0")
    json.dump(text_cfg, open(os.path.join(args.dst, "config.json"), "w"),
              indent=2)
    for fn in ("tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "generation_config.json"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.dst, fn))
    print("wrote text-only checkpoint ->", args.dst)


if __name__ == "__main__":
    main()
