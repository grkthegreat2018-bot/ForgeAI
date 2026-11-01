"""LFM3-CC -> lfm3_1p7b_d1 graft bit-exactness check (lineage gate, directive A).

Loads the real lfm3_1p7b_cc checkpoint twice on CPU fp32:
  A) stock lfm3_1p7b preset
  B) same preset + zero-init grafts (GDN2 side-paths on conv blocks,
     ValueResidual, LearnedSink, AttnRes, CopySpan + Verifier heads)
and compares logits. All grafts are gate=0 / skipped at init, so the
expected result is max|dlogit| == 0.0 (lossless), per AGENTS.md preset-
lineage rule.

Run: venv\\Scripts\\python.exe scripts\\test_lfm3_d1_parity.py [--seq 48]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

CKPT = ROOT / "research" / "checkpoints" / "lfm3_1p7b_cc"

GRAFT_LAYERS = [1, 5, 9, 13, 17, 21]  # i%4==1 convs: before each full-attn


def _load(grafted: bool, seq: int):
    from forge.config import get_config
    from forge.model.loader import ModelLoader
    cfg = get_config("lfm3_1p7b_d1" if grafted else "lfm3_1p7b",
                     device="cpu")
    t0 = time.time()
    m = ModelLoader.build_model_fast(
        cfg, str(CKPT), dtype=torch.float32, fast_load=False)
    print(f"  [{'d1' if grafted else 'base'}] loaded in "
          f"{time.time()-t0:.0f}s, params={sum(p.numel() for p in m.parameters())/1e9:.4f}B")
    m.eval()
    return m


def main() -> int:
    seq = 48
    if "--seq" in sys.argv:
        seq = int(sys.argv[sys.argv.index("--seq") + 1])
    torch.manual_seed(0)
    idx = torch.randint(0, 120000, (1, seq))

    m_base = _load(False, seq)
    m_d1 = _load(True, seq)

    with torch.no_grad():
        out_a = m_base(idx)
        out_b = m_d1(idx)
    la, lb = out_a[0], out_b[0]
    diff = (la - lb).abs()
    print(f"logits {tuple(la.shape)}")
    print(f"max|dlogit| = {diff.max().item():.3e}")
    print(f"argmax agreement = {(la.argmax(-1) == lb.argmax(-1)).float().mean().item()*100:.2f}%")
    ok = diff.max().item() == 0.0
    print("LINEAGE:", "LOSSLESS (bit-exact)" if ok else "NON-LOSSLESS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
