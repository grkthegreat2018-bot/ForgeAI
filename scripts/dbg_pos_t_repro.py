"""Repro the BCG warmup failure with a full traceback."""
import sys, torch, traceback
sys.path.insert(0, ".")
from forge.engine.forge_engine import ForgeEngine

eng = ForgeEngine.from_checkpoint(
    "research/checkpoints/lfm3_1p7b_cc_fq", config_name="lfm3_1p7b",
    device="cuda", kv_cache="standard", use_compile=False,
    use_breakable_cuda_graph=True,
    use_block_fusion=False, use_chunked_prefix_cache=False,
    use_learned_prefix_cache=False, use_adaptive_spec=False,
    use_suffix_spec=False, use_faser=False, use_lookahead_gate=False,
    use_hotprefix=False, use_pod_attention=False, use_hybrid_prefill=False)

from forge.model.kv_cache import create_kv_cache
m = eng.model
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
mode = sys.argv[2] if len(sys.argv) > 2 else "plain"
if mode == "ensure":
    # Replicate the _ensure_static_batch + decode_step contract exactly.
    bs = m._ensure_static_batch(B, 1, 400)
    print("ensure ->", "None" if bs is None else "bs ok")
    raise SystemExit(0)
if mode == "capture":
    # Drive the exact _capture_buckets warmup path with a live traceback.
    try:
        m._ensure_static_batch(B, 1, 400)
    except Exception:
        traceback.print_exc()
    bs = m._decode_state['batch'].get(B)
    print("bs cache:", bs is not None and bool(bs))
    c, m_buf = bs['cache'], bs['mask']
    c.view_len = 512
    mask_view = m_buf[:, :512]
    ids = torch.zeros(B, 1, dtype=torch.long, device="cuda")
    try:
        with torch.inference_mode():
            m(ids, attention_mask=mask_view, preallocated_cache=c,
              use_cache=True, logits_last_only=True)
        print("CAPTURE-WARMUP OK")
    except Exception:
        traceback.print_exc()
    raise SystemExit(0)
c = create_kv_cache(m, max_total=512, batch=B, device="cuda")
c.enable_tensor_pos()
c.view_len = 512
mask = torch.zeros(B, 512, dtype=torch.long, device="cuda")
ids = torch.zeros(B, 1, dtype=torch.long, device="cuda")
try:
    with torch.inference_mode():
        out = m(ids, attention_mask=mask, preallocated_cache=c,
                use_cache=True, logits_last_only=True)
    print("OK")
except Exception:
    traceback.print_exc()
