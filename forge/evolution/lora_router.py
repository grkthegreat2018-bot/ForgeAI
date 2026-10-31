"""Decision-model LoRA router — the model decides which adapter a prompt
needs, instead of name-matched selection.

Features come from the *proposer itself*: ``ForgeEngine.embed`` mean-pools
the model's last hidden state (<=512 tok, L2-normed), so routing happens in
the base model's own representation space. A tiny aux MLP
(``forge.aux_train.train_mlp`` / ``forge.aux_models.load_aux_model``)
classifies embedding -> adapter tag. Stored inside the LoRA registry:

    <lora_root>/router.safetensors   aux MLP weights
    <lora_root>/router.json          {arch, tags, dim, trained_rows, ...}

Training signal: per-tag prompt corpora (each adapter's own SFT rows at
first; later, db rows tagged by adapter + outcome once provenance accrues —
``code_programs.adapter`` column). Falls back to ``LoRARegistry.resolve``
when no router is trained or routing is uncertain (max logit below
``min_conf``).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class LoRARouter:
    NET = "router.safetensors"
    META = "router.json"

    def __init__(self, root: str | Path, min_conf: float = 0.15):
        self.root = Path(root)
        self.min_conf = min_conf
        self._loaded = False
        self._net = None
        self._tags: list[str] = []
        self._meta: dict = {}

    @property
    def tags(self) -> list[str]:
        self._load()
        return list(self._tags)

    def exists(self) -> bool:
        return (self.root / self.NET).exists() and \
            (self.root / self.META).exists()

    def _load(self) -> bool:
        if self._loaded:
            return self._net is not None
        self._loaded = True
        try:
            meta = json.loads((self.root / self.META).read_text(
                encoding="utf-8"))
            from forge.aux_models import load_aux_model
            self._net = load_aux_model(meta, str(self.root / self.NET),
                                       device="auto")
            self._tags = list(meta["tags"])
            self._meta = meta
            return True
        except Exception:
            self._net = None
            return False

    def route(self, emb) -> str | None:
        """embedding -> tag, or None when uncertain/unavailable."""
        if not self._load() or not self._tags:
            return None
        import torch
        x = np.asarray(emb, dtype=np.float32)
        # _net is an AuxModelRunner, NOT an nn.Module — parameters()/
        # direct call raised AttributeError on every invocation and the
        # caller's try/except silently routed every prompt to the
        # fallback. infer() takes numpy and already unbatches 1-D input.
        logits = torch.as_tensor(self._net.infer(x), dtype=torch.float32)
        i = int(logits.argmax())
        if float(logits[i]) < self.min_conf and len(self._tags) > 1:
            return None
        return self._tags[i]

    # ── training ────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, root: str | Path, embed_fn, tag_prompts: dict[str, list],
            *, max_per_tag: int = 64, epochs: int = 80,
            device: str = "auto") -> dict:
        """Train the router on per-tag prompt corpora.

        ``embed_fn(text) -> list[float]`` — the proposer engine's own
        embed() so features match deployment exactly.
        """
        from forge.aux_train import export_aux_checkpoint, train_mlp

        tags = sorted(tag_prompts)
        X, Y = [], []
        for ti, tag in enumerate(tags):
            prompts = tag_prompts[tag][:max_per_tag]
            for p in prompts:
                e = np.asarray(embed_fn(str(p)), dtype=np.float32)
                if e.size == 0 or not np.isfinite(e).all():
                    continue
                X.append(e)
                y = np.zeros(len(tags), dtype=np.float32)
                y[ti] = 1.0
                Y.append(y)
        X = np.stack(X)
        Y = np.stack(Y)
        if len(tags) < 1 or X.shape[0] < 8:
            raise ValueError(
                f"router needs >=8 embeds across >=1 tags "
                f"(got {X.shape[0]})")

        net, metrics = train_mlp(
            X, Y, layers=[X.shape[1], 64, 64, len(tags)],
            # binary_cross_entropy on raw logits (out_activation=None)
            # NaNs on the first out-of-[0,1] output — sigmoid heads are
            # required for bce, and make min_conf a probability gate.
            out_activation="sigmoid", loss="bce", epochs=epochs,
            batch_size=512, device=device)
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        export_aux_checkpoint(net, str(root / cls.NET))
        meta = {
            "arch": metrics["arch"], "tags": tags,
            "dim": int(X.shape[1]), "trained_rows": int(X.shape[0]),
            "val_loss": metrics["val_loss"],
            "created": datetime.now(timezone.utc)
            .isoformat(timespec="seconds"),
        }
        (root / cls.META).write_text(json.dumps(meta, indent=2),
                                    encoding="utf-8")
        return meta
