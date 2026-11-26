# Quant/Decoding Rules — forge/quant/, forge/engine/quant/, forge/decoding/

## Hard rules
- **Quantization is first-class** (directive D) — BitNet b1.58, W8A8,
  NVFP4, OffQ, AAAC, SharQ, MosaicQuant are production paths on this
  12GB box. Prefer them over "use a smaller model".
- **Every quant mode reports effective state** — the applied mode in
  diagnostics, never the requested one (engine invariant).
- New quant methods register through `feature_registry.py` +
  `quantize="<name>"` in config; check `forge/engine/quant/` for the
  existing pattern (latest: `littlebit2.py` — sidecar save/load pattern
  via `save_/load_littlebit2_state` for expensive conversions).

## Gotchas (measured on this box)
- Sub-1-bit PTQ-only quality is limited — QAT-first methods
  (`SmoothSign`/`STEBinary` STEs keep params learnable). R45: binary PTQ
  without training is catastrophic.
- Batched decode prefers cached-bf16 cuBLAS GEMM over fused int4 Triton
  (`_fused_gemv=False`, `_cache_w=True`) — ~3.3x at B>1.
- Weight lossless ceiling ≈10.7 b/param (bf16) — see
  `scripts/bench_lossless_weights.py` before claiming a codec win.

## Decoding
- `forge/decoding/` — CopySpan/edit-mode (`edit_decode.py`,
  `model/edit_head.py`, V14-RSI heads — docs/design/V14_RSI_DESIGN.md).
- Spec strategies: reuse `_spec_snapshot/_spec_truncate/_spec_repair`.
- Lookahead: `lookahead_decode.py` (FORGE_LOOKAHEAD=k,w; greedy-only,
  ~2.3x code / ~0.9x open text).
