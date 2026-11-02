"""Sequential API-proposer evolve sweep — overnight harvest.

Runs fn_evolve per target (one at a time) with the remote teacher so
the GPU stays free. Every run feeds forge_evolve_api.db (verified
wins for export_sft) and results/api_proposer_traces.jsonl
(reasoning traces).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / "venv" / "Scripts" / "python.exe"

TARGETS = [
    "auto_build_block_diag_causal_mask", "auto_causal_mask",
    "auto_clone_state_tree", "auto_complex_rmsnorm",
    "auto_expected_calibration_error", "auto_exp_trapezoidal_discretize",
    "auto_jsd", "auto_ngram_lookup", "auto_slice_past_kv",
    "auto_spec_slice_kv", "auto_truncate_at_stop",
    "auto_w8a8_quantize_activation", "canon_key", "decode_step",
    "dequantize_4bit", "dry_penalties", "fast_softmax", "min_k_filter",
    "prefix_hash", "quantize_4bit", "radix_prefix", "sample_pipeline",
    "stop_matcher", "suffix_mem", "top_p_filter", "top_p_probs",
]


def main() -> int:
    import os
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    model = os.environ.get("EVOLVE_API_MODEL", "deepseek-v4-pro")
    url = os.environ.get("EVOLVE_API_URL", "")
    keyenv = os.environ.get("EVOLVE_API_KEY_ENV", "DEEPSEEK_API_KEY")
    mine = [t for i, t in enumerate(TARGETS) if i % shards == shard]
    log = (REPO / "results" / f"evolve_sweep_{shard}.log").open(
        "a", encoding="utf-8")
    extra = (["--api-url", url] if url else []) + [
        "--api-key-env", keyenv]
    for t in mine:
        log.write(f"\n=== {t} ===\n")
        log.flush()
        r = subprocess.run(
            [str(VENV_PY), str(REPO / "scripts" / "fn_evolve.py"),
             "--target", t, "--proposer", "api",
             "--api-model", model, "--api-workers", "3",
             "--evals", "60", "--workers", "3", "--patience", "15",
             "--db", "forge_evolve_api.db", "--log-gens", "--fresh",
             "--quiet"] + extra,
            stdout=log, stderr=subprocess.STDOUT)
        log.write(f"=== {t} rc={r.returncode} ===\n")
        log.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
