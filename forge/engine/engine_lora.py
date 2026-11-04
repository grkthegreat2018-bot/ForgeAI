"""LoRA adapter management mixin for ForgeEngine.

Residency model (audit A8): single-adapter serialize-and-restore.
``load_lora`` unloads whatever is resident before attaching — there is
NO concurrent multi-adapter residency.  Per-request selection
(``ChatCompletionRequest.lora_adapter``) swaps the requested adapter in
for one request (streaming and non-streaming alike) and restores the
previous one afterwards; requests asking for different adapters
serialize through ``_gen_lock``.  True multi-residency needs named
adapter slots in ``forge.training.bitnet_lora`` (per-request bind
instead of the single ``module.lora_adapter`` attribute) — out of
scope; documented here so the contract is not misread as concurrent.
"""
from .engine_common import (  # noqa: F401
    _CKPT_CACHE_MAX,
    _DEFAULT_CPU_MEMORY_BYTES,
    _DEFAULT_EOS_TOKEN_IDS,
    _QWEN_TOKENIZER_PATH,
    _QWEN_VOCAB,
    _checkpoint_metadata_cache,
    _checkpoint_size_cache,
    _ckpt_cache_lock,
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    logger,
)


class _LoRAMixin:
    # ── LoRA hot-loading ───────────────────────────────────────────────────

    def load_lora(self, lora_path: str, rank: int = 32, alpha: int | None = None,
                  target_modules: list[str] | None = None) -> int:
        """Hot-load a LoRA adapter onto the model.

        Attaches LoRA adapters to the specified modules and loads weights from
        a safetensors checkpoint. The base model weights are frozen — only the
        LoRA adapters are trainable. This enables dynamic skill injection at
        runtime (e.g. loading a tool-calling LoRA for self-play, then swapping
        it for a code-generation LoRA).

        Args:
            lora_path: Path to LoRA adapter safetensors file.
            rank: LoRA rank (must match the checkpoint).
            alpha: LoRA alpha (scale = alpha / rank). If None, uses rank * 2.
            target_modules: Module name substrings to attach LoRA to.
                If None, defaults to FFN+attention modules:
                ["w_gate", "w_up", "w_down", "q_proj", "v_proj",
                 "out_proj", "in_proj"]
                Pass ["w_gate", "w_up", "w_down"] for FFN-only.

        Returns:
            Number of LoRA parameters loaded.

        Raises:
            FileNotFoundError: If lora_path doesn't exist.
            ValueError: If LoRA checkpoint doesn't match model structure.
        """
        # Serialise against in-flight generation: adapter attachment
        # monkey-patches module forwards — unsafe mid-forward.
        with self._gen_lock:
            return self._load_lora_impl(
                lora_path, rank, alpha, target_modules)

    def _load_lora_impl(self, lora_path: str, rank: int,
                        alpha: int | None,
                        target_modules: list[str] | None) -> int:
        from pathlib import Path

        from safetensors.torch import load_file as _load

        from forge.training.bitnet_lora import add_lora_adapters

        if target_modules is None:
            target_modules = ["w_gate", "w_up", "w_down", "q_proj", "v_proj",
                              "out_proj", "in_proj"]

        lora_file = Path(lora_path)
        if not lora_file.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

        # Remove any existing LoRA adapters first
        self.unload_lora()

        # Rank from the FILE, not the manifest/caller — lora_A is (r, in)
        # so safetensors shapes are ground truth; a stale manifest or a
        # default-32 caller crashed on param.data.copy_ (16-vs-32).
        state = _load(str(lora_file))
        a_key = next((k for k in state if k.endswith("lora_A")), None)
        if a_key is not None and int(state[a_key].shape[0]) != rank:
            file_rank = int(state[a_key].shape[0])
            self._log(f"LoRA rank mismatch: requested {rank}, file has "
                      f"{file_rank} — using file rank", level="warn")
            rank = file_rank
        if alpha is None:
            alpha = rank * 2

        self._log(f"Loading LoRA adapter: {lora_file.name} (rank={rank})")
        n_adapters, _ = add_lora_adapters(
            self.model, rank=rank, alpha=alpha,
            target_modules=target_modules)

        loaded = 0
        for name, param in self.model.named_parameters():
            if "lora_" in name and name in state:
                param.data.copy_(state[name])
                loaded += 1

        if loaded < len(state):
            missing = set(state.keys()) - {
                n for n, _ in self.model.named_parameters() if "lora_" in n}
            self._log(f"WARNING: {len(state) - loaded} LoRA tensors not found "
                      f"in model (checkpoint has {len(state)}, loaded {loaded}). "
                      f"Missing: {list(missing)[:5]}...",
                      level="warning")

        n_params = sum(v.numel() for v in state.values())
        self._lora_config = {
            "path": str(lora_file), "rank": rank, "alpha": alpha,
            "target_modules": target_modules, "n_adapters": n_adapters,
            "n_params": n_params,
        }
        self._log(f"LoRA loaded: {loaded}/{len(state)} tensors, "
                  f"{n_adapters} adapters, {n_params / 1e6:.1f}M params")
        return loaded

    def unload_lora(self) -> bool:
        """Remove LoRA adapters from the model.

        Detaches all LoRA modules and restores the original forward functions.
        The base model weights are unaffected.

        Returns:
            True if any LoRA adapters were removed, False if none were attached.
        """
        with self._gen_lock:
            return self._unload_lora_impl()

    def _unload_lora_impl(self) -> bool:
        removed = 0
        for _name, module in self.model.named_modules():
            # Quantized linears keep lora_adapter as an attribute their
            # forward checks (`is not None`) — deleting it outright leaves
            # subsequent forwards raising AttributeError. Set None.
            if hasattr(module, 'lora_adapter') and module.lora_adapter is not None:
                module.lora_adapter = None
                removed += 1
            # nn.Linear with monkey-patched forward: restore original
            if hasattr(module, '_lora_orig_forward'):
                module.forward = module._lora_orig_forward
                del module._lora_orig_forward
                if hasattr(module, 'lora_adapter'):
                    module.lora_adapter = None
                removed += 1

        if removed > 0:
            self._log(f"LoRA unloaded: {removed} adapters removed")
            self._lora_config = None
            # code_evolve._ensure_lora skips re-attach when its tag marker
            # still matches the last adapter — clear both markers so an
            # external unload doesn't leave the engine silently unadapted.
            for marker in ("_evolve_lora_tag", "_evolve_lora_badbase"):
                if hasattr(self, marker):
                    delattr(self, marker)
        return removed > 0

    def consolidate_pairs(self, pairs: list[tuple[str, str]],
                          replay: list[tuple[str, str]] | None = None,
                          lr: float = 5e-4, steps: int = 100,
                          rank: int = 8, alpha: int | None = None,
                          target_modules: list[str] | None = None,
                          mem_filter: bool = True,
                          merge: bool = False,
                          control: str | None = None,
                          save_adapter: str | None = None) -> dict:
        """Fold verified (prompt, response) pairs into weights via a
        short in-engine LoRA fit — the L2 consolidation step of the
        live-learning loop (L1 sidecar → L2 weights).

        ``pairs`` is the candidate set (e.g. rows ingested this
        session). With ``mem_filter`` and a SuffixMemory sidecar
        attached, pairs whose answer the memory cannot replay are
        dropped — the sidecar never landed them, so fitting them
        teaches noise the L1 store doesn't actually carry.

        ``replay`` pairs are mixed into every optimization step — the
        measured brake on catastrophic forgetting (exp_l2_consolidate:
        lr 3e-3 without replay → +3.86 ctrl-CE drift; lr 5e-4 + replay
        → +0.31). Keep ``replay`` non-empty.

        ``merge`` folds the trained delta into the base weights and
        removes the adapters; otherwise the adapter stays resident as
        a normal single-residency adapter (``unload_lora`` drops it).
        ``control`` is a held-out text whose next-token CE measures
        catastrophic forgetting — it is NEVER in the fit set (replay
        rows are trained, so they cannot serve as the control).
        ``save_adapter`` writes the trained A/B tensors as a
        load_lora-compatible safetensors file.

        Returns a metrics dict: pre/post answer CE on ``pairs``, control
        CE drift on ``replay[0]``, fitted/dropped counts, n_params.
        """
        import torch
        import torch.nn.functional as F
        from forge.training.bitnet_lora import (
            add_lora_adapters, merge_lora_adapters)

        self._require_awake()
        with self._gen_lock:
            model = self.model
            dev = next(model.parameters()).device

            def _enc(t: str):
                enc = getattr(self.tokenizer, "encode", None)
                ids = enc(t) if callable(enc) else self.tokenizer(
                    t, add_special_tokens=False)["input_ids"]
                return torch.tensor(list(ids), device=dev)

            def _answer_ce(q_ids, a_ids) -> float:
                ids = torch.cat([q_ids, a_ids])
                with torch.no_grad():
                    z = model(ids.unsqueeze(0))[0]
                    z = z[0] if z.dim() == 3 else z
                lp = F.cross_entropy(
                    z[q_ids.numel() - 1:-1].float(),
                    ids[q_ids.numel():], reduction="none")
                return float(lp.mean())

            fitted = list(pairs)
            dropped = 0
            mem = getattr(self, "_suffix_mem", None)
            if mem_filter and mem is not None:
                kept = []
                for p, a in fitted:
                    p_ids, a_ids = _enc(p), _enc(a)
                    mem._tail = p_ids[-mem.max_order:].tolist()
                    drafted = mem.draft(min(4, a_ids.numel()))
                    if drafted and drafted[0] == int(a_ids[0]):
                        kept.append((p, a))
                    else:
                        dropped += 1
                fitted = kept

            rows = [(p, _enc(p), _enc(a)) for p, a in fitted]
            replay_rows = [(p, _enc(p), _enc(a))
                           for p, a in (replay or [])]
            if not rows:
                return {"fitted": 0, "dropped": dropped,
                        "error": "no pairs survived the memory filter"}

            def _text_ce(ids) -> float:
                with torch.no_grad():
                    z = model(ids.unsqueeze(0))[0]
                    z = z[0] if z.dim() == 3 else z
                return float(F.cross_entropy(
                    z[:-1].float(), ids[1:], reduction="mean"))

            # Held-out control — never enters the fit set, so its CE
            # drift is a real catastrophic-forgetting measurement
            # (unlike replay rows, which are trained).
            if control is None:
                control = ("The quick brown fox jumps over the lazy dog. "
                           "Pack my box with five dozen liquor jugs.")
            ctrl_ids = _enc(control)

            pre = sum(_answer_ce(q, a) for q, a in
                      (r[1:] for r in rows)) / len(rows)
            ctrl_pre = _text_ce(ctrl_ids)

            n_ad, lora_params = add_lora_adapters(
                model, rank=rank, alpha=alpha or rank * 2,
                target_modules=target_modules)
            # Only the LoRA params should carry grads — freezing the rest
            # skips ~2B params of dead grad allocation each backward.
            lora_set = set(id(p) for p in lora_params)
            frozen = [p for p in model.parameters()
                      if p.requires_grad and id(p) not in lora_set]
            for p in frozen:
                p.requires_grad = False
            opt = torch.optim.AdamW(lora_params, lr=lr)
            model.train()
            fit_rows = rows + replay_rows
            for _step in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = 0.0
                for _p, q_ids, a_ids in fit_rows:
                    ids = torch.cat([q_ids, a_ids])
                    z = model(ids.unsqueeze(0))[0]
                    z = z[0] if z.dim() == 3 else z
                    loss = loss + F.cross_entropy(
                        z[q_ids.numel() - 1:-1].float(),
                        ids[q_ids.numel():])
                (loss / len(fit_rows)).backward()
                opt.step()
            model.eval()
            for p in frozen:
                p.requires_grad = True

            post = sum(_answer_ce(q, a) for q, a in
                       (r[1:] for r in rows)) / len(rows)
            ctrl_post = _text_ce(ctrl_ids)

            n_params = sum(p.numel() for p in lora_params)
            if save_adapter is not None:
                from safetensors.torch import save_file
                save_file({n: p.detach().cpu() for n, p in
                           model.named_parameters() if "lora_" in n},
                          save_adapter)
            if merge:
                merge_lora_adapters(model)
                self._lora_config = None
            else:
                self._lora_config = {
                    "path": save_adapter, "rank": rank,
                    "alpha": alpha or rank * 2,
                    "target_modules": target_modules,
                    "n_adapters": n_ad, "n_params": n_params,
                    "source": "consolidate_pairs"}
            result = {
                "fitted": len(rows), "dropped": dropped,
                "n_adapters": n_ad, "n_params": n_params,
                "pre_ce": round(pre, 4), "post_ce": round(post, 4),
                "ctrl_drift": round(ctrl_post - ctrl_pre, 4),
                "peak_vram_gb": round(
                    torch.cuda.max_memory_allocated() / 1e9, 2)
                if dev.type == "cuda" else None,
            }
            self._log(f"consolidate_pairs: {result}")
            return result

    def has_lora(self) -> bool:
        """Check if LoRA adapters are currently loaded."""
        return getattr(self, '_lora_config', None) is not None

    def lora_info(self) -> dict | None:
        """Return info about the currently loaded LoRA, or None.

        Adds ``residency: "single"`` — adapters are serialize-and-restore,
        not concurrent multi-adapter slots (module docstring, audit A8).
        Returns a copy so callers cannot mutate ``_lora_config``.
        """
        cfg = getattr(self, '_lora_config', None)
        return dict(cfg, residency="single") if cfg is not None else None

