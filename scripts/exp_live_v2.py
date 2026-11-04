"""FluxLM v2 direction-finding: which live surfaces actually learn on the fly.

Runs the probes MicroLiveLM does NOT measure — the failure modes that
killed FluxLM v1 (memorize-only, no generalization, state corruption):

  1. verbatim:     baseline fact recall after ingest (replicates domain)
  2. distractor:   gain after N noise tokens (durability vs recency)
  3. overwrite:    same keys -> new values (correct update semantics?)
  4. alt-context:  facts probed via a DIFFERENT separator (fuzzy
                   generalization — TitanMemory should transfer where
                   exact-suffix NGramMemory cannot)
  5. false-memory: never-trained keys — confident-wrong or uncertain?
                   (memory must not poison the substrate)

Each config builds ONE model/session and runs all probes on it.
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F

from forge.model.live import LiveConfig, LiveSession, livet_config
from forge.model.llm import ConfigurableResearchLLM
from forge.model.ngram_mem import NGramMemory

V = 256
SEP = 200          # trained separator token
SEP2 = 201         # held-out separator (alt-context probe)
N_PAIRS = 8
NOISE_LO, NOISE_HI = 150, 195   # filler-token range (never in facts)


def _stream(pairs: dict[int, int], reps: int, sep: int = SEP) -> torch.Tensor:
    seq = []
    for _ in range(reps):
        for k, v in pairs.items():
            seq += [sep, k, v]
    return torch.tensor(seq)


def _noise(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(NOISE_LO, NOISE_HI, (n,), generator=g)


def build(cfg: dict, dev: str):
    torch.manual_seed(0)
    lc = livet_config(
        vocab_size=V, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg.get("n_heads", 4),
        intermediate_size=cfg["d_model"] * cfg.get("ffn_mult", 3),
        layer_types=["attention"] * cfg["n_layers"],
        use_gdn2=cfg.get("use_gdn2", True),
        use_titan_memory=cfg.get("use_titan", True),
        device=dev, dtype="float32")
    model = ConfigurableResearchLLM(lc).to(dev).eval()
    s = LiveSession(model, LiveConfig(
        chunk=cfg.get("chunk", 32),
        solver=cfg.get("solver"),
        mem_lr=cfg.get("mem_lr", 1.0),
        ridge=cfg.get("ridge", 0.1),
        forget=cfg.get("forget"),
        momentum=cfg.get("momentum", 0.9),
        open_gates=True))
    if cfg.get("use_ngram"):
        s.attach_sparse(NGramMemory(
            V, orders=tuple(cfg.get("orders", (4, 8))),
            gate=cfg.get("ngram_gate", 1.0)))
    s.freeze_substrate()
    return model, s


def fact_ce(s, probe: torch.Tensor, dev: str) -> float:
    """Mean CE on every 3rd position (the VALUE slot of SEP,k,v)."""
    p = probe.to(dev)
    with torch.no_grad():
        logits = s.model(p[:-1].unsqueeze(0))[0]
        if s.sparse is not None:
            logits = logits + s.sparse.lift(p[:-1].unsqueeze(0))
        ce = F.cross_entropy(logits[0].float(), p[1:], reduction="none")
    # value slots are probe positions 2,5,8,... -> predicted by ce[1,4,7,...]
    return float(ce[torch.arange(ce.numel()) % 3 == 1].mean())


def probes(cfg: dict, dev: str) -> dict:
    model, s = build(cfg, dev)
    map1 = {10 + i: 100 + i for i in range(N_PAIRS)}
    probe1 = _stream(map1, 1)
    probe_alt = _stream(map1, 1, sep=SEP2)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # -- 1. verbatim ----------------------------------------------------
    base = fact_ce(s, probe1, dev)
    t0 = time.time()
    s.ingest(_stream(map1, 5).to(dev))
    ingest_tps = 5 * 3 * N_PAIRS / max(time.time() - t0, 1e-9)
    s.reset()
    v_after = fact_ce(s, probe1, dev)

    # -- 2. distractor durability ---------------------------------------
    s.ingest(_noise(512, 7).to(dev))
    d_after = fact_ce(s, probe1, dev)

    # -- 3. overwrite semantics ------------------------------------------
    map2 = {k: NOISE_LO + i for i, k in enumerate(map1)}
    s.ingest(_stream(map2, 5).to(dev))
    s.reset()
    probe2 = _stream(map2, 1)
    ow = fact_ce(s, probe2, dev)          # new values recalled?
    ow_stale = fact_ce(s, probe1, dev)    # old values suppressed?

    # -- 4. alt-context generalization ------------------------------------
    alt = fact_ce(s, probe_alt, dev)      # same pairs, unseen separator

    # -- 5. false-memory on never-trained keys ----------------------------
    mapX = {30 + i: 120 + i for i in range(4)}
    fx = fact_ce(s, _stream(mapX, 1), dev)

    vram = (torch.cuda.max_memory_allocated() / 1e9
            if dev == "cuda" else 0.0)
    del model, s
    if dev == "cuda":
        torch.cuda.empty_cache()
    return {
        "verbatim_gain": base - v_after,
        "post_distract_gain": base - d_after,
        "overwrite_ce": ow, "stale_ce": ow_stale,
        "alt_ctx_ce": alt,
        "unseen_ce": fx,
        "base_ce": base,
        "ingest_tps": ingest_tps,
        "vram_gb": vram,
    }


CONFIGS = {
    "full (titan+gdn2+ngram)": dict(
        d_model=128, n_layers=4, use_gdn2=True, use_titan=True,
        use_ngram=True, orders=(4, 8), solver="solve", mem_lr=4.0,
        chunk=32),
    "titan+gdn2 (fuzzy only)": dict(
        d_model=128, n_layers=4, use_gdn2=True, use_titan=True,
        use_ngram=False, solver="solve", mem_lr=4.0, chunk=32),
    "ngram only (exact only)": dict(
        d_model=128, n_layers=4, use_gdn2=False, use_titan=False,
        use_ngram=True, orders=(4, 8), solver="gd", mem_lr=1.0,
        chunk=32),
    "titan only (gd)": dict(
        d_model=128, n_layers=4, use_gdn2=False, use_titan=True,
        use_ngram=False, solver="gd", mem_lr=1.0, chunk=32),
    "scaled (d384 L6)": dict(
        d_model=384, n_layers=6, use_gdn2=True, use_titan=True,
        use_ngram=True, orders=(4, 8), solver="solve", mem_lr=4.0,
        chunk=32),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dev={dev}  V={V}  pairs={N_PAIRS}  probes=5")
    hdr = (f"{'config':<28} {'verb':>5} {'dist':>5} {'ow':>5} "
           f"{'stale':>5} {'alt':>5} {'unse':>5} {'tps':>7} {'vram':>5}")
    print(hdr)
    for name in args.configs.split(","):
        name = name.strip()
        cfg = CONFIGS[name]
        t0 = time.time()
        r = probes(cfg, dev)
        print(f"{name:<28} {r['verbatim_gain']:>5.2f} "
              f"{r['post_distract_gain']:>5.2f} {r['overwrite_ce']:>5.2f} "
              f"{r['stale_ce']:>5.2f} {r['alt_ctx_ce']:>5.2f} "
              f"{r['unseen_ce']:>5.2f} {r['ingest_tps']:>7.0f} "
              f"{r['vram_gb']:>5.2f}   [{time.time()-t0:.0f}s]")
    print("base_ce reference: log(256)=%.2f" % math.log(256))


if __name__ == "__main__":
    main()
