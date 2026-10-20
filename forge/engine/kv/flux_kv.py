"""FluxKV — FluxLM-style context independence for KV-cache models (R65).

The transformer half of the FluxLM port: a strictly bounded KV window
whose eviction is safe because the companion ``SuffixMemory`` sidecar
(forge/model/ngram_mem.py) keeps exact verbatim recall of the whole
stream.  Attention sees sink ∪ beacon-scored middle ∪ recent window —
all positions the RoPE table was trained for — while the sidecar holds
the verbatim tail-to-successor map over arbitrarily long sessions.

Semantic vs. plain ``beacon``: same mechanism, RP-tuned geometry
(big recent window for dialogue continuity, more beacons for
long-session character/fact retention) and a distinct name so the
GUI / server config can select it without guessing ctor defaults.

VRAM is O(budget), attention compute per token is O(budget), RoPE
positions compress at eviction so they never leave the trained range —
generation cost and memory are independent of context length.
"""
from __future__ import annotations

from forge.engine.kv.beacon_kv import BeaconKVCache


class FluxKVCache(BeaconKVCache):
    """Bounded beacon+window KV for infinite-context sessions.

    Default geometry is sized for a 12B on a 12GB card:
      budget 6144 tokens of KV — ~1.0GB at 40L×8KV×128hd fp16 —
      with a generous 2048-token recent window (RP dialogue continuity)
      and 8 sink slots (prompt anchor).
    """

    def __init__(self, budget: int = 6144, sink: int = 8,
                 window: int = 2048, n_beacons: int = 24, ema: float = 0.9,
                 recency_weight: float = 0.5, slack: int | None = None):
        super().__init__(budget=budget, sink=sink, window=window,
                         n_beacons=n_beacons, ema=ema,
                         recency_weight=recency_weight, slack=slack)

    def info(self):
        d = super().info()
        d["type"] = "flux"
        return d
