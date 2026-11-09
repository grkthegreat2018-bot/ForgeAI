"""Predict tok/s for layer-streaming inference under different overlap schemes.

Current AirLLMStreamer (forge/engine/airllm_streamer.py): synchronous — every
layer's params are copied CPU->GPU per forward, no prefetch of layer N+1.

This bench measures real pinned H2D bandwidth on this box, sizes each layer
from the safetensors index, then models:
  serial    : t = sum(load_i + compute_i)                (today's airllm)
  prefetch  : t = sum(max(load_i, compute_i))            (double-buffer N+1)
  hot-res   : attention/norms resident, FFN-only streamed (prefetch)
  int4      : same but weights stored/streamed as int4 g32 (+gpu dequant)
Also reports compressed-transfer (entropy-coded int4 over the bus, decode on GPU)
assuming entropy ~3.4 bits/param (measured by bench_lossless_weights.py).

Usage: venv/Scripts/python.exe scripts/bench_airllm_prefetch.py [ckpt_dir]
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

GBPS_REF = 12.9  # measured load_safetensors_pipelined on this box (AGENTS.md)


def measure_h2d_gbps(mb=256, pinned=True) -> float:
    """CPU -> GPU copy bandwidth, GB/s."""
    src = torch.empty(mb * 1024 * 1024, dtype=torch.uint8, pin_memory=pinned)
    dst = torch.empty_like(src, device="cuda")
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return (mb * 1024 * 1024 * 10) / dt / 1e9


def layer_bytes(ckpt: Path):
    idx = json.loads((ckpt / "model.safetensors.index.json").read_text())
    per_layer = {}
    misc = 0
    for name, shard in idx["weight_map"].items():
        # dtype assumed bf16 (2B); read shape lazily via header would be better,
        # but index has no shapes -> quick parse of shard header
        pass
    # parse shard headers once
    sizes = {}
    for sh in ckpt.glob("model-*.safetensors"):
        with open(sh, "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hlen))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            b0, b1 = meta["data_offsets"]
            sizes[name] = b1 - b0
    per_layer = {}
    misc = 0
    for name, nbytes in sizes.items():
        if ".layers." in name:
            ln = int(name.split(".layers.")[1].split(".")[0])
            per_layer.setdefault(ln, {})[name] = nbytes
        else:
            misc += nbytes
    return per_layer, misc, sizes


def main():
    ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else "research/checkpoints/spark_x26_1p7b")
    per_layer, misc, sizes = layer_bytes(ckpt)
    nlayers = len(per_layer)
    total_w = sum(sizes.values())

    bw = measure_h2d_gbps(pinned=True)
    bw_page = measure_h2d_gbps(pinned=False)
    print(f"H2D pinned bandwidth: {bw:.1f} GB/s | pageable: {bw_page:.1f} GB/s"
          f" (pipelined disk-load ref {GBPS_REF} GB/s)")
    print(f"layers: {nlayers}, weights total {total_w/1e9:.2f} GB, misc/embed {misc/1e9:.3f} GB")

    # split each layer into attn-ish (small) vs ffn/ssm-ish (large) by name heuristic
    def split(names_bytes):
        attn = sum(b for n, b in names_bytes.items()
                   if any(t in n for t in ("self_attn", "attn.", "in_proj", "qkv",
                                           "norm", "mixer.in_proj", "mixer.out_proj")))
        return attn, sum(names_bytes.values()) - attn

    layer_load_ms = {}
    ffn_frac = []
    for ln, nb in sorted(per_layer.items()):
        attn_b, big_b = split(nb)
        ffn_frac.append(big_b / sum(nb.values()))
        layer_load_ms[ln] = (sum(nb.values()), attn_b, big_b)
    mean_layer = np.mean([v[0] for v in layer_load_ms.values()])
    mean_ffn_frac = np.mean(ffn_frac)
    print(f"mean layer {mean_layer/1e6:.1f} MB, mean non-attn frac {mean_ffn_frac*100:.0f}%")

    # resident compute estimate: assume resident int4 decode ~ R tok/s on this class
    # of 1.7B model. Use engine-measured analog: spark bf16 ~= 60-90 tok/s; take 75.
    # Per-layer compute time is then split evenly for the model (rough but honest).
    for label, resid_tps in [("bf16-resident 75 tok/s", 75.0)]:
        t_resid = 1.0 / resid_tps
        t_compute_layer = t_resid / nlayers

        def report(name, bytes_per_layer_scale, transfer_bytes_frac=1.0):
            rows = []
            for ln, (tot_b, attn_b, big_b) in sorted(layer_load_ms.items()):
                stream_b = (tot_b * bytes_per_layer_scale if transfer_bytes_frac == 1.0
                            else (attn_b * 0 + big_b * bytes_per_layer_scale))
                rows.append((stream_b, t_compute_layer))
            serial = sum(b / (bw * 1e9) + c for b, c in rows)
            pipe = sum(max(b / (bw * 1e9), c) for b, c in rows)
            print(f"{name:<38} serial {1/serial:>6.2f} tok/s   prefetched {1/pipe:>6.2f} tok/s")
            return 1 / serial, 1 / pipe

        print(f"\n== streaming {nlayers} layers/token, assume {label} compute ==")
        report("bf16 full-stream", 1.0)
        report("bf16 ffn-only stream", 1.0, transfer_bytes_frac=0)
        report("int4 g32 full-stream (x0.28 bytes)", 0.28)
        report("int4 ffn-only stream", 0.28, transfer_bytes_frac=0)
        report("int4+entropy (~3.4b/p) ffn-only", 3.4 / 16, transfer_bytes_frac=0)

        # what airllm ACTUALLY does today: pageable (or worse, disk) sync copies
        bw_real = bw_page
        serial_pg = sum(tot_b / (bw_real * 1e9) + t_compute_layer
                        for tot_b, _, _ in layer_load_ms.values())
        print(f"\n>> TODAY's airllm equivalent (pageable H2D {bw_real:.1f} GB/s,"
              f" sync): {1/serial_pg:.2f} tok/s")
        bw_disk = 1.0  # SMR D: small-file regime ~<1-2GB/s best case
        serial_disk = sum(tot_b / (bw_disk * 1e9) + t_compute_layer
                          for tot_b, _, _ in layer_load_ms.values())
        print(f">> airllm from disk (~{bw_disk:.0f} GB/s, sync): {1/serial_disk:.2f} tok/s")


if __name__ == "__main__":
    main()
