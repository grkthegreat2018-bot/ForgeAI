"""STRATA - cortex-over-database LM substrate (R60).

Three separated systems:
  WEIGHTS (.strata)  - labeled CellBlocks of packed info cells.
                       Info's only job: hold facts.  Sealed/active
                       lifecycle + per-block undo arenas + tag-scoped
                       revert + bounded hot/cold budget.
  PARAMS  (.iq)      - the tiny compute machinery (IQ).  Mixer
                       weights, fingerprint table, optional dense
                       channel.  Decides HOW info is used.
  KNOBS   (.cond)    - conditioning scalars (gates, lifts, sampling).
                       Ingest never touches them, so weights can be
                       pumped with info without changing behavior.

M0 (this package): the storage substrate + artifact IO.  The model
dynamics (channels, context sketch, fused decode) build on top.
"""
from .blocks import BlockMeta, BlockStore, CellBlock
from .cells import (
    EMPTY_FOLL,
    cell_bytes,
    count_lut,
    decode_counts,
    foll_aux,
    foll_cntq,
    foll_tok,
    foll_valid,
    meta_flags,
    meta_nfoll,
    meta_wtag,
    pack_follower,
    pack_meta,
    quantize_counts,
)
from .config import StrataConfig
from .io import MAGIC, load_container, load_store, save_container, save_store
from .knobs import DEFAULT_KNOBS, KnobSet
from .model import StrataLM
from .params import ParamSet, dequantize_param, quantize_param

__all__ = [
    "DEFAULT_KNOBS",
    "EMPTY_FOLL",
    "MAGIC",
    "BlockMeta",
    "BlockStore",
    "CellBlock",
    "KnobSet",
    "ParamSet",
    "StrataConfig",
    "StrataLM",
    "cell_bytes",
    "count_lut",
    "decode_counts",
    "dequantize_param",
    "foll_aux",
    "foll_cntq",
    "foll_tok",
    "foll_valid",
    "load_container",
    "load_store",
    "meta_flags",
    "meta_nfoll",
    "meta_wtag",
    "pack_follower",
    "pack_meta",
    "quantize_counts",
    "quantize_param",
    "save_container",
    "save_store",
]
