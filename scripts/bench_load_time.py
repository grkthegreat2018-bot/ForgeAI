"""Phase-level benchmark for ForgeEngine checkpoint loading.

Times every stage of ForgeEngine.from_checkpoint() on the real ForgeLM V2
checkpoint (RTX 5070): imports, tokenizer, weight I/O, meta init, assign,
activation phases, and each feature-registry handler.

Run: venv/Scripts/python.exe scripts/bench_load_time.py [--second]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.runtime.configure import configure

configure()

_t0 = time.perf_counter()
_marks: list[tuple[str, float]] = []


def mark(name: str, t0: float) -> float:
    dt = time.perf_counter() - t0
    _marks.append((name, dt))
    print(f"    [{dt:6.2f}s] {name}")
    return time.perf_counter()


def main():
    import torch

    t = time.perf_counter()
    from forge.engine.forge_engine import ForgeEngine
    mark("import ForgeEngine", t)

    t = time.perf_counter()
    from research.tokenizer_cache import get_tokenizer
    tok = get_tokenizer()
    mark("get_tokenizer", t)

    # ── instrument loader internals ──────────────────────────────────
    from forge.model import loader as ml

    orig_mmap = ml.ModelLoader._load_safetensors_mmap

    def _timed_mmap(path, model, device=None):
        t = time.perf_counter()
        s = orig_mmap(path, model, device=device)
        mark(f"weight load ({len(s)} tensors, "
             f"{sum(t_.numel() * t_.element_size() for t_ in s.values()) / 1e9:.2f} GB)",
             t)
        return s

    ml.ModelLoader._load_safetensors_mmap = staticmethod(_timed_mmap)

    orig_lsd = torch.nn.Module.load_state_dict

    def _timed_lsd(self, *a, **kw):
        t = time.perf_counter()
        r = orig_lsd(self, *a, **kw)
        mark("load_state_dict(assign)", t)
        return r

    torch.nn.Module.load_state_dict = _timed_lsd

    # ── instrument activation phases ─────────────────────────────────
    from forge.engine import engine_activation as ea

    for phase in ("_activate_core_innovations", "_activate_kv_cache",
                  "_activate_decoding", "_activate_acceleration",
                  "_activate_compile_runtime", "_apply_feature_registry",
                  "_finalize_activation", "_detect_keystack_features",
                  "_apply_quantization"):
        fn = getattr(ea._ActivationMixin, phase, None)
        if fn is None:
            continue

        def _wrap(f, name):
            def _inner(*a, **kw):
                t = time.perf_counter()
                r = f(*a, **kw)
                mark(name, t)
                return r
            return _inner

        setattr(ea._ActivationMixin, phase, _wrap(fn, phase))

    # ── per-feature handler timing ───────────────────────────────────
    from forge.engine import feature_registry as fr

    timed_specs = []
    for spec in fr._FEATURE_REGISTRY:
        h = spec.handler

        def _wrap_h(f, name):
            def _inner(*a, **kw):
                t = time.perf_counter()
                r = f(*a, **kw)
                mark(f"  feature:{name}", t)
                return r
            return _inner

        timed_specs.append(fr.FeatureSpec(spec.flag, _wrap_h(h, spec.flag),
                                          spec.cuda_only))
    fr._FEATURE_REGISTRY.clear()
    fr._FEATURE_REGISTRY.extend(timed_specs)

    # ── the real load ────────────────────────────────────────────────
    from research.paths import V2_CHECKPOINT

    t = time.perf_counter()
    engine = ForgeEngine.from_checkpoint(
        checkpoint=str(V2_CHECKPOINT),
        config_name="forgelm_v2",
        auto_activate=True,
    )
    total = mark("TOTAL from_checkpoint", t)

    free, tot = torch.cuda.mem_get_info()
    print(f"\n  VRAM: {(tot - free) / 1e9:.2f} GB used")

    # optional: second load in-process (warm page cache + warm imports)
    if "--second" in sys.argv:
        del engine
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        _marks.clear()
        t = time.perf_counter()
        engine = ForgeEngine.from_checkpoint(
            checkpoint=str(V2_CHECKPOINT),
            config_name="forgelm_v2",
            auto_activate=True,
        )
        mark("TOTAL second load", t)

    print("\n== Summary (sorted) ==")
    for name, dt in sorted(_marks, key=lambda x: -x[1]):
        print(f"  {dt:7.2f}s  {name}")


if __name__ == "__main__":
    main()
