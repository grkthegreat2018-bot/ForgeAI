"""AirMoE — Infinite Expert Library with VRAM-limited hotswap.

Architecture:
  Base model (always in VRAM):
    - Embedding + LM Head
    - Attention layers (MLA)
    - Shared FFN expert per layer (always active, handles general cases)
    - Router (classifies query → topic)

  Expert library (unlimited, on disk):
    - Each expert is a standalone FFN (w1, w2, w3) for one layer
    - Named by topic: expert_l{layer}_{topic}.safetensors
    - Topics can be anything: python_algorithms, math_geometry, science_physics, ...
    - New experts added anytime without retraining
    - SVD + int4 compressed (~2 MB each)

  At inference:
    1. Router classifies query → topic (e.g., "python_algorithms")
    2. AirMoE cache loads expert_l{0..27}_python_algorithms.safetensors from disk
    3. Expert weights injected into the "routed expert" slot for each layer
    4. Forward pass: shared expert + routed expert (topic-specific)
    5. LRU evicts least-recently-used topic when VRAM is full

  No hard expert limit. VRAM is the only constraint.
  With int4 + SVD: each expert ~2 MB, can cache 100+ experts in 2 GB VRAM.

File layout:
  forgelm_v2/
    base_model_int4.safetensors     # base weights (no routed experts)
    manifest.json                    # expert library index
    rotorquant_rotations.pt          # KV cache compression
    experts/                         # infinite expert library
      expert_l0_python_algorithms.safetensors
      expert_l0_math_geometry.safetensors
      expert_l0_science_physics.safetensors
      ... (unlimited, add new topics anytime)

Usage:
    from forge.moe.airmoe_infinite import InfiniteAirMoE
    airmoe = InfiniteAirMoE(model, tokenizer, "forgelm_v2/")
    airmoe.load_expert_topic("python_algorithms")  # hotswap
    result = thinker.generate_with_thinking("Sort a list using merge sort")
"""
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from forge.moe.routers import KeywordRouter

try:
    from loguru import logger as _log
except ImportError:
    import logging
    _log = logging.getLogger(__name__)


class ExpertRouter(KeywordRouter):
    """Routes queries to expert topics using keyword matching.

    The router has no hard limit on topics — new topics can be registered
    at any time. It maps a user query to the best-matching expert topic
    in the library.

    Keyword scoring/threshold logic lives in the shared KeywordRouter base;
    this class sources keywords from the expert library manifest.
    """

    # Module-level manifest cache: {manifest_path: manifest_dict}
    # Avoids re-reading manifest.json from disk on every ExpertRouter creation.
    _manifest_cache: dict[str, dict] = {}

    def __init__(self, manifest_path: str):
        if manifest_path not in self._manifest_cache:
            with open(manifest_path, encoding="utf-8") as f:
                self._manifest_cache[manifest_path] = json.load(f)
        self.manifest = self._manifest_cache[manifest_path]

        self.topics: dict[str, dict] = self.manifest.get("topics", {})
        self.n_layers = self.manifest.get("n_layers", 28)
        super().__init__(fallback="general", min_score=1)

    def _iter_keywords(self):
        # Keywords come from the (mutable) manifest topics, not a static dict.
        for topic_name, info in self.topics.items():
            yield topic_name, info.get("keywords", [])

    def _topic_names(self):
        return self.topics.keys()

    def register_topic(self, topic: str, keywords: list[str],
                        label: str = "", subtopics: list[str] | None = None):
        """Register a new expert topic at runtime.

        This allows adding new experts to the library without restarting.
        """
        self.topics[topic] = {
            "label": label or topic,
            "keywords": keywords,
            "subtopics": subtopics or [],
        }


class InfiniteAirMoE:
    """Infinite expert library with VRAM-limited hotswap.

    The model has a shared FFN (always in VRAM) + a routed expert slot
    that gets filled dynamically from the expert library on disk.

    Only VRAM limits how many experts can be cached. With SVD+int4
    compression (~2 MB per expert), 100+ experts fit in 2 GB.

    Hotswap flow:
      1. Router classifies query → topic
      2. Check if topic's experts are in VRAM cache
      3. If not: load from disk (28 files, one per layer)
      4. Inject expert weights into model's expert slots
      5. If VRAM full: evict LRU topic
      6. Forward pass with shared + routed expert
    """

    def __init__(self, model, tokenizer, module_dir: str,
                 device: str = "cuda",
                 vram_budget_gb: float = 2.0,
                 max_cached_topics: int = 10):
        """
        Args:
            model: ForgeLM model with shared FFN + expert slots
            tokenizer: tokenizer
            module_dir: path to forgelm_v2/ directory
            device: target device
            vram_budget_gb: VRAM budget for expert cache
            max_cached_topics: max topics cached simultaneously
        """
        self.model = model
        self.tokenizer = tokenizer
        self.module_dir = Path(module_dir)
        self.device = device
        self.vram_budget_gb = vram_budget_gb
        self.max_cached_topics = max_cached_topics

        # Load router from manifest — prefer SemanticRouter (embedding-based)
        # over keyword-based ExpertRouter for better query routing.
        manifest_path = self.module_dir / "manifest.json"
        # Initialize before try so it's always defined for n_layers below.
        manifest_data: dict = {}
        try:
            from forge.moe.routers import DEFAULT_TOPIC_DESCRIPTIONS, SemanticRouter
            # Build topic descriptions from manifest topics (use cached manifest)
            manifest_data = ExpertRouter._manifest_cache.get(str(manifest_path))
            if manifest_data is None:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest_data = json.load(f)
                ExpertRouter._manifest_cache[str(manifest_path)] = manifest_data
            topic_descs = {}
            for tname, tinfo in manifest_data.get("topics", {}).items():
                kws = tinfo.get("keywords", [])
                topic_descs[tname] = tinfo.get("label", tname) + ": " + ", ".join(kws)
            # Fill in defaults for any missing topics
            for k, v in DEFAULT_TOPIC_DESCRIPTIONS.items():
                if k not in topic_descs:
                    topic_descs[k] = v
            self.router = SemanticRouter(model, tokenizer, topic_descs, device=device)
            print(f"  [AirMoE] Using SemanticRouter ({len(topic_descs)} topics)")
        except Exception as e:
            _log.warning(f"[AirMoE] SemanticRouter failed ({e}), falling back to keyword router")
            print(f"  [AirMoE] SemanticRouter failed ({e}), falling back to keyword router")
            self.router = ExpertRouter(str(manifest_path))
            # manifest_data may be empty here — recover n_layers from the
            # keyword router's manifest (already loaded by ExpertRouter).
            if not manifest_data:
                manifest_data = getattr(self.router, "manifest", {}) or {}
        self.n_layers = manifest_data.get("n_layers", 28)

        # Expert file integrity: optional sha256 verification against hashes
        # published in manifest.json ("expert_hashes": {filename: sha256}).
        # Off by default (hashing 28 files adds disk I/O); corrupt files are
        # always caught at safetensors parse time regardless (warn + skip).
        self.verify_hashes: bool = False
        self._expert_hashes: dict[str, str] = manifest_data.get("expert_hashes", {}) or {}

        # LRU cache: topic → {layer: expert_weights}
        self.cache: OrderedDict[str, dict[int, dict[str, torch.Tensor]]] = OrderedDict()
        self.current_topic: str | None = None
        self.current_experts: dict[int, dict[str, torch.Tensor]] | None = None

        # Stats
        self.stats = {
            "topic_loads": 0,
            "cache_hits": 0,
            "evictions": 0,
            "total_load_time_ms": 0,
            "bytes_loaded": 0,
        }

    def _get_expert_path(self, layer: int, topic: str) -> Path:
        """Get the file path for a topic expert at a given layer."""
        # Expert files are directly in module_dir (flat layout)
        path = self.module_dir / f"expert_l{layer}_{topic}.safetensors"
        if path.exists():
            return path
        # Fallback: experts/ subdir (legacy layout)
        return self.module_dir / "experts" / f"expert_l{layer}_{topic}.safetensors"

    def _verify_sha256(self, path: Path):
        """Verify an expert file's sha256 against the manifest hash (if any).

        Warn-only: a mismatch is logged but does not block loading.
        Files without a manifest hash entry are skipped (no-op).
        """
        expected = self._expert_hashes.get(path.name)
        if not expected:
            return
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected:
            _log.warning(f"[AirMoE] sha256 mismatch for {path.name}: "
                         f"expected {expected[:16]}…, got {actual[:16]}…")

    def _load_topic_from_disk(self, topic: str) -> dict[int, dict[str, torch.Tensor]]:
        """Load all layer experts for a topic from disk.

        Uses ThreadPoolExecutor for parallel disk reads (10-20x faster on
        NVMe/SSD when loading 28 separate expert files).

        Returns: {layer_idx: {"w1": tensor, "w2": tensor, "w3": tensor}}
        """
        from concurrent.futures import ThreadPoolExecutor

        from safetensors.torch import load_file

        # Collect valid paths first
        paths = {}
        for layer in range(self.n_layers):
            path = self._get_expert_path(layer, topic)
            if path.exists():
                paths[layer] = path

        if not paths:
            return {}

        def _load_one_disk(layer_path_tuple):
            """Thread-safe: only disk I/O, no GPU ops.

            Validates the expert file before returning it: optional sha256
            check against the manifest, then a real safetensors parse. A
            corrupt/unparseable file is skipped (warn-only) instead of
            crashing the whole topic load.
            """
            layer, path = layer_path_tuple
            try:
                if self.verify_hashes:
                    self._verify_sha256(path)
                state = load_file(str(path))
                return layer, state, path.stat().st_size
            except Exception as e:
                _log.warning(f"[AirMoE] Skipping corrupt expert file {path}: {e}")
                return layer, None, 0

        experts = {}
        total_bytes = 0

        # Phase 1: Parallel disk reads (I/O bound, thread-safe)
        max_workers = min(8, len(paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            raw_results = list(executor.map(_load_one_disk, paths.items()))

        # Phase 2: GPU decompression (sequential — CUDA is single-threaded)
        for layer, state, size in raw_results:
            if state is None:
                continue  # corrupt file — already logged in _load_one_disk
            # Decompress: check which format
            if any(k.endswith("_U_q") for k in state):
                expert = self._decompress_expert(state)
            elif any(k.endswith("_U") and "_U_q" not in k and "_U_shape" not in k and "_U_scale" not in k for k in state):
                expert = self._decompress_expert_svd_only(state)
            else:
                expert = {}
                # Legacy SwiGLU format: w1, w2, w3
                for part in ["w1", "w2", "w3"]:
                    k = f"{part}.weight"
                    if k in state:
                        expert[part] = state[k].to(self.device)
                # LatentMoE format: up, down
                for part in ["up", "down"]:
                    k = f"{part}.weight"
                    if k in state:
                        expert[part] = state[k].to(self.device)
            experts[layer] = expert
            total_bytes += size

        self.stats["bytes_loaded"] += total_bytes
        return experts

    def _decompress_expert(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decompress SVD + int4 expert to full weight matrices on GPU."""
        expert = {}
        dev = self.device
        # Support both legacy (w1/w2/w3) and LatentMoE (up/down) formats
        for part in ["w1", "w2", "w3", "up", "down"]:
            U_q = state.get(f"{part}_U_q")
            U_scale = state.get(f"{part}_U_scale")
            S = state.get(f"{part}_S")
            Vh_q = state.get(f"{part}_Vh_q")
            Vh_scale = state.get(f"{part}_Vh_scale")
            U_shape = state.get(f"{part}_U_shape")
            Vh_shape = state.get(f"{part}_Vh_shape")

            if U_q is None:
                continue

            # Dequantize U on GPU: q is [n_groups, gs], scale is [n_groups]
            U_q_d = U_q.to(dev)
            U_scale_d = U_scale.to(dev)
            U = (U_q_d.float() * U_scale_d.float().unsqueeze(-1)).reshape(-1)
            if U_shape is not None:
                U = U[:U_shape[0].item() * U_shape[1].item()].reshape(
                    U_shape[0].item(), U_shape[1].item())

            # Dequantize Vh on GPU
            Vh_q_d = Vh_q.to(dev)
            Vh_scale_d = Vh_scale.to(dev)
            Vh = (Vh_q_d.float() * Vh_scale_d.float().unsqueeze(-1)).reshape(-1)
            if Vh_shape is not None:
                Vh = Vh[:Vh_shape[0].item() * Vh_shape[1].item()].reshape(
                    Vh_shape[0].item(), Vh_shape[1].item())

            # Reconstruct on GPU: W = U @ diag(S) @ Vh
            S_d = S.to(dev)
            W = (U * S_d.float().unsqueeze(0)) @ Vh
            expert[part] = W.to(torch.bfloat16)
            del U_q_d, U_scale_d, Vh_q_d, Vh_scale_d, S_d

        return expert

    def _decompress_expert_svd_only(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decompress SVD-only expert (v2 format) on GPU for speed."""
        expert = {}
        dev = self.device
        # Support both legacy (w1/w2/w3) and LatentMoE (up/down) formats
        for part in ["w1", "w2", "w3", "up", "down"]:
            U = state.get(f"{part}_U")
            S = state.get(f"{part}_S")
            Vh = state.get(f"{part}_Vh")
            if U is None:
                continue
            # Reconstruct on GPU: W = U @ diag(S) @ Vh
            U_d = U.to(dev)
            S_d = S.to(dev)
            Vh_d = Vh.to(dev)
            W = (U_d.float() * S_d.float().unsqueeze(0)) @ Vh_d.float()
            expert[part] = W.to(torch.bfloat16)
            del U_d, S_d, Vh_d
        return expert

    def _inject_expert(self, layer: int, expert: dict[str, torch.Tensor]):
        """Inject expert weights into the model's expert slot for a layer.

        Handles two expert formats:
        1. Legacy SwiGLU experts (MoELayer.Expert): w1, w2, w3 attributes
        2. LatentMoE experts (nn.Sequential): [0].weight (up), [2].weight (down)
           operating in latent space (ℓ → hidden → ℓ)
        """
        try:
            block = self.model.blocks[layer]
            ffn = block.ffn

            # If MoE, inject into the first routed expert slot
            if hasattr(ffn, 'experts') and len(ffn.experts) > 0:
                expert_module = ffn.experts[0]
                model_dtype = next(self.model.parameters()).dtype

                # Legacy SwiGLU format: w1 (gate), w2 (down), w3 (up)
                if hasattr(expert_module, 'w1'):
                    if "w1" in expert:
                        expert_module.w1.weight.data = expert["w1"].to(self.device, model_dtype)
                    if "w2" in expert:
                        expert_module.w2.weight.data = expert["w2"].to(self.device, model_dtype)
                    if "w3" in expert:
                        expert_module.w3.weight.data = expert["w3"].to(self.device, model_dtype)

                # LatentMoE format: nn.Sequential([Linear, SquaredReLU, Linear])
                # Expert operates in latent space: ℓ → hidden → ℓ
                # Keys from disk: "up.weight" (first linear), "down.weight" (last linear)
                # Or from decompression: "up", "down"
                elif isinstance(expert_module, nn.Sequential) and len(expert_module) >= 3:
                    up_w = expert.get("up") or expert.get("up.weight")
                    down_w = expert.get("down") or expert.get("down.weight")
                    # Backward compat: map w1→up, w2→down
                    if up_w is None:
                        up_w = expert.get("w1") or expert.get("w1.weight")
                    if down_w is None:
                        down_w = expert.get("w2") or expert.get("w2.weight")
                    if up_w is not None:
                        expert_module[0].weight.data = up_w.to(self.device, model_dtype)
                    if down_w is not None:
                        expert_module[2].weight.data = down_w.to(self.device, model_dtype)
                else:
                    _log.warning(f"[AirMoE] Unknown expert format at layer {layer}, "
                                 f"type={type(expert_module).__name__}")
        except (IndexError, AttributeError) as e:
            # Model might not have expert slots at this layer
            _log.warning(f"[AirMoE] Expert injection skipped at layer {layer}: {e}")

    def _evict_lru(self):
        """Evict the least recently used topic from the cache."""
        if not self.cache:
            return

        evicted_topic, evicted_experts = self.cache.popitem(last=False)
        self.stats["evictions"] += 1

        # Free VRAM
        del evicted_experts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"    [AirMoE] Evicted topic: {evicted_topic}")

    def _estimate_cache_vram_gb(self) -> float:
        """Estimate current cache VRAM usage."""
        total_params = 0
        for topic_experts in self.cache.values():
            for _layer, expert in topic_experts.items():
                for w in expert.values():
                    total_params += w.numel()
        return total_params * 2 / 1e9  # bf16

    def load_topic(self, topic: str) -> bool:
        """Load a topic's experts into VRAM (hotswap).

        If the topic is already cached, just move to MRU.
        Otherwise, load from disk and evict LRU if needed.

        Returns True if successful, False if topic not found.
        """
        # Cache hit
        if topic in self.cache:
            self.cache.move_to_end(topic)
            self.current_topic = topic
            self.current_experts = self.cache[topic]
            self.stats["cache_hits"] += 1
            return True

        # Check if topic exists on disk (any layer, not just layer 0)
        found = False
        for layer in range(self.n_layers):
            if self._get_expert_path(layer, topic).exists():
                found = True
                break
        if not found:
            print(f"    [AirMoE] Topic '{topic}' not found in expert library")
            return False

        # Load from disk
        t0 = time.time()
        experts = self._load_topic_from_disk(topic)
        load_ms = (time.time() - t0) * 1000
        self.stats["topic_loads"] += 1
        self.stats["total_load_time_ms"] += load_ms

        # Evict LRU if over budget
        while (len(self.cache) >= self.max_cached_topics or
               self._estimate_cache_vram_gb() > self.vram_budget_gb):
            if not self.cache:
                break
            self._evict_lru()

        # Add to cache
        self.cache[topic] = experts
        self.current_topic = topic
        self.current_experts = experts

        # Inject into model
        for layer, expert in experts.items():
            self._inject_expert(layer, expert)

        n_layers_loaded = len(experts)
        print(f"    [AirMoE] Loaded topic: {topic} "
              f"({n_layers_loaded} layers, {load_ms:.0f}ms, "
              f"cache: {len(self.cache)}/{self.max_cached_topics})")

        return True

    def route_and_load(self, query: str) -> str:
        """Route a query to a topic and load the relevant expert.

        This is the main entry point for inference:
          1. Router classifies query → topic
          2. Load topic expert (hotswap if needed)
          3. Return the topic name

        The model is now ready to generate with the topic-specific expert.
        """
        topic = self.router.classify(query)
        self.load_topic(topic)
        return topic

    def route_and_load_multi(self, query: str, top_n: int = 2) -> list[str]:
        """Route a query to multiple topics and load them all.

        For complex queries that span multiple domains (e.g., "write a
        Python function to solve a math equation"), this loads both
        the code and math experts.
        """
        topics = self.router.classify_multi(query, top_n=top_n)
        for topic in topics:
            self.load_topic(topic)
        return topics

    def get_stats(self) -> dict:
        """Get hotswap statistics."""
        return {
            **self.stats,
            "cached_topics": len(self.cache),
            "current_topic": self.current_topic,
            "cache_vram_gb": self._estimate_cache_vram_gb(),
            "vram_budget_gb": self.vram_budget_gb,
            "available_topics": len(self.router.list_topics()),
        }

    def print_stats(self):
        """Print AirMoE statistics."""
        s = self.get_stats()
        print(f"\n{'='*70}")
        print("Infinite AirMoE Statistics")
        print(f"{'='*70}")
        print(f"  Available topics:    {s['available_topics']} (unlimited library)")
        print(f"  Cached topics:       {s['cached_topics']}/{self.max_cached_topics}")
        print(f"  Current topic:       {s['current_topic']}")
        print(f"  Topic loads:         {s['topic_loads']}")
        print(f"  Cache hits:          {s['cache_hits']}")
        print(f"  Evictions:           {s['evictions']}")
        print(f"  Cache VRAM:          {s['cache_vram_gb']:.3f} / {s['vram_budget_gb']:.1f} GB")
        print(f"  Total load time:     {s['total_load_time_ms']:.0f}ms")
        print(f"  Bytes loaded:        {s['bytes_loaded']/1e6:.1f} MB")
        print(f"{'='*70}")


def main():
    """Test the infinite AirMoE system."""
    from forge.config import get_config
    from forge.model_loader import ModelLoader
    from research.tokenizer_cache import get_tokenizer

    print("=" * 70)
    print("Infinite AirMoE Test")
    print("=" * 70)

    from research.paths import EXPERTS_DIR, as_str
    module_dir = as_str(EXPERTS_DIR)

    if not os.path.exists(os.path.join(module_dir, "manifest.json")):
        print(f"\n  ERROR: No V2 expert library at {module_dir}")
        print("  Run: python scripts/train_expert.py --topic <name> --data <data.json>")
        return

    # Load model
    print("\n[1] Loading ForgeLM V10 base...")
    cfg = get_config("forgelm_v2", device="cuda")
    model = ModelLoader.build_model_fast(cfg,
        checkpoint_path="research/checkpoints/ForgeLM_V2.safetensors")
    model.to("cuda").eval()
    tokenizer = get_tokenizer("research/checkpoints/qwen_hf")

    # Create infinite AirMoE
    print("\n[2] Creating infinite AirMoE...")
    airmoe = InfiniteAirMoE(model, tokenizer, module_dir,
                             device="cuda",
                             vram_budget_gb=2.0,
                             max_cached_topics=10)

    print(f"  Available topics: {airmoe.router.list_topics()}")

    # Test routing + hotswap
    test_queries = [
        "Write a Python function to sort a list",
        "Solve the quadratic equation x^2 + 5x + 6 = 0",
        "Prove that the sum of two even numbers is even",
        "Explain the scientific method",
    ]

    print("\n[3] Testing routing + hotswap...")
    for query in test_queries:
        topic = airmoe.route_and_load(query)
        print(f"  Query: {query[:50]}")
        print(f"  → Topic: {topic}")

    airmoe.print_stats()


if __name__ == "__main__":
    main()
