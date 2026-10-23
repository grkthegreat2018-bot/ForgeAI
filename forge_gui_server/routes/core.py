"""Core routes: status snapshot, engine lifecycle + ops, models, LoRA,
activation catalog."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..deps import services

router = APIRouter()

# Sane bounds for LoRA rank/alpha — unclamped values can blow up VRAM.
LORA_RANK_MIN, LORA_RANK_MAX = 1, 256
LORA_ALPHA_MIN, LORA_ALPHA_MAX = 1, 4096


def _clamp(value: int | None, lo: int, hi: int) -> int | None:
    if value is None:
        return None
    return max(lo, min(int(value), hi))


def _confined_path(path: str, allowed_dirs: list[Path],
                   must_exist: bool = False) -> str | None:
    """Resolve ``path`` (relative → under project root) and require it to
    live inside one of ``allowed_dirs``. Returns the resolved string, or
    None when it escapes or fails the existence check."""
    from forge_gui.api.status_reader import project_root
    if not path:
        return None
    root = project_root().resolve()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    try:
        p = p.resolve()
    except OSError:
        return None
    for d in allowed_dirs:
        d = d.resolve()
        if p == d or d in p.parents:
            if must_exist and not p.is_file():
                return None
            return str(p)
    return None


def _checkpoint_dirs() -> list[Path]:
    """Dirs merge/LoRA/model paths are confined to."""
    from forge_gui.api.status_reader import project_root
    root = project_root()
    return [root / "research" / "checkpoints"]


def _writable_dirs() -> list[Path]:
    """Dirs a route may write new artifacts into."""
    from forge_gui.api.status_reader import project_root
    root = project_root()
    return [root / "research" / "checkpoints", root / "data" / "merged"]


def _session_dict(s: Any) -> dict:
    """Serialize a SessionState dataclass (or plain value) to JSON-safe."""
    if s is None:
        return {}
    if isinstance(s, dict):
        return s
    d = {
        "session_id": getattr(s, "session_id", None),
        "created_at": getattr(s, "created_at", None),
        "last_access": getattr(s, "last_access", None),
        "ttl": getattr(s, "ttl", None),
        "ttl_expires_at": getattr(s, "ttl_expires_at", None),
        "pinned": getattr(s, "pinned", None),
    }
    return {k: v for k, v in d.items() if v is not None}


def _run_dict(r) -> dict:
    return {"id": r.id, "name": r.name, "status_file": r.status_file,
            "status": r.status, "step": r.step, "max_steps": r.max_steps,
            "loss": r.loss, "lr": r.lr, "vram_gb": r.vram_gb,
            "method": r.method, "updated_at": r.updated_at,
            "heartbeat_age_s": round(r.heartbeat_age_s, 1),
            "extra": r.extra, "progress_pct": round(r.progress_pct, 1),
            "is_live": r.is_live}


# ── status ────────────────────────────────────────────────────────────

@router.get("/status")
async def status():
    runs = await asyncio.get_running_loop().run_in_executor(
        None, services.status_reader.snapshot)
    return {
        "engine": services.engine.snapshot(),
        "gpu": services.gpu.snapshot(),
        "lora": services.lora.status(),
        "runs": [_run_dict(r) for r in runs],
        "tasks": services.procs.all_tasks(),
        "agent_runs": services.agent.list_runs(),
        "ws_clients": services.hub.subscriber_count,
    }


@router.get("/runs")
async def runs():
    snaps = await asyncio.get_running_loop().run_in_executor(
        None, services.status_reader.snapshot)
    return {"runs": [_run_dict(r) for r in snaps]}


class StopRun(BaseModel):
    status_file: str


@router.post("/runs/stop")
async def stop_run(body: StopRun):
    loop = asyncio.get_running_loop()
    snaps = await loop.run_in_executor(
        None, services.status_reader.snapshot)
    target = next((r for r in snaps if r.status_file == body.status_file),
                  None)
    if target is None:
        return {"ok": False, "error": "run not found"}
    return {"ok": await loop.run_in_executor(
        None, services.status_reader.request_stop, target)}


@router.get("/runs/log_tail")
async def run_log_tail(status_file: str, lines: int = 200):
    lines = max(1, min(int(lines), 5000))
    return {"lines": await asyncio.get_running_loop().run_in_executor(
        None, services.status_reader.read_log_tail, status_file, lines)}


# ── engine lifecycle ──────────────────────────────────────────────────

class LoadRequest(BaseModel):
    checkpoint: str = ""
    config_name: str = "forgelm_v2"
    use_compile: bool | None = None
    activation: dict | None = None
    config_overrides: dict | None = None


@router.post("/engine/load")
async def engine_load(body: LoadRequest):
    services.engine.load(
        checkpoint=body.checkpoint, config_name=body.config_name,
        use_compile=body.use_compile, activation=body.activation,
        config_overrides=body.config_overrides)
    return {"ok": True, "state": services.engine.state}


@router.post("/engine/unload")
async def engine_unload():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, services.engine.unload)
    return {"ok": True}


class LearnRequest(BaseModel):
    path: str
    tag: str | None = None
    max_tokens: int = 0


@router.post("/engine/learn")
async def engine_learn(body: LearnRequest):
    """Live-learn a corpus file/dir into the resident FluxLM engine."""
    services.engine.learn(body.path, tag=body.tag,
                          max_tokens=body.max_tokens)
    return {"ok": True}


@router.post("/engine/learn/cancel")
async def engine_learn_cancel():
    services.engine.learn_cancel()
    return {"ok": True}


@router.post("/engine/reactivate")
async def engine_reactivate(body: dict):
    # reactivate() returns {"ok": bool, ...} — refusals (no engine / busy)
    # and ignored activation keys are surfaced instead of hidden behind ok.
    return services.engine.reactivate(body.get("activation") or body)


class PowerRequest(BaseModel):
    level: int = 1


def _engine_call(fn, *args, timeout_s: float = 120.0, **kw):
    """Run a blocking engine method under the lease in an executor."""
    def work():
        with services.engine.acquire(timeout_s=timeout_s) as eng:
            return fn(eng, *args, **kw)
    return asyncio.get_running_loop().run_in_executor(None, work)


@router.post("/engine/sleep")
async def engine_sleep(body: PowerRequest):
    eng = services.engine.try_engine()
    if eng is None:
        return {"ok": False, "error": "no resident engine"}
    await _engine_call(lambda e: e.sleep(body.level))
    return {"ok": True}


@router.post("/engine/wake")
async def engine_wake():
    eng = services.engine.try_engine()
    if eng is None:
        return {"ok": False, "error": "no resident engine"}
    await _engine_call(lambda e: e.wake())
    return {"ok": True}


@router.get("/engine/stats")
async def engine_stats():
    eng = services.engine.try_engine()
    if eng is None:
        return {"ready": False}
    try:
        stats = await _engine_call(lambda e: e.stats())
    except Exception as ex:
        stats = {"error": str(ex)}
    try:
        vram = await _engine_call(lambda e: e.vram_usage())
    except Exception:
        vram = None
    try:
        lora = await _engine_call(lambda e: e.lora_info())
    except Exception:
        lora = None
    try:
        awake = await _engine_call(lambda e: e.is_awake)
    except Exception:
        awake = None
    return {"ready": True, "stats": stats, "vram": vram,
            "lora": lora, "awake": awake,
            "info": services.engine.info}


# ── engine maintenance / tools ────────────────────────────────────────

class MaintRequest(BaseModel):
    prompt: str = "The quick brown fox"
    max_tokens: int = 64
    runs: int = 3


@router.post("/engine/benchmark")
async def engine_benchmark(body: MaintRequest):
    try:
        out = await _engine_call(
            lambda e: e.benchmark(prompt=body.prompt,
                                  max_tokens=body.max_tokens,
                                  runs=body.runs), timeout_s=600)
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/bottleneck")
async def engine_bottleneck():
    try:
        out = await _engine_call(lambda e: e.bottleneck(), timeout_s=300)
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/diagnose")
async def engine_diagnose():
    try:
        out = await _engine_call(lambda e: e.diagnose(), timeout_s=300)
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.get("/engine/log")
async def engine_log(lines: int = 200):
    eng = services.engine.try_engine()
    if eng is None:
        return {"lines": []}
    try:
        out = await _engine_call(lambda e: e.read_log(lines))
        return {"lines": out if isinstance(out, list) else [str(out)]}
    except Exception as ex:
        return {"lines": [f"error: {ex}"]}


@router.get("/engine/outputs")
async def engine_outputs(n: int = 10):
    eng = services.engine.try_engine()
    if eng is None:
        return {"outputs": []}
    try:
        out = await _engine_call(lambda e: e.read_output(n))
        return {"outputs": out}
    except Exception as ex:
        return {"outputs": [], "error": str(ex)}


@router.post("/engine/recover")
async def engine_recover():
    try:
        out = await _engine_call(lambda e: e.recover(), timeout_s=300)
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/clear_recovery")
async def engine_clear_recovery():
    try:
        out = await _engine_call(lambda e: e.clear_recovery())
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/reset_stats")
async def engine_reset_stats():
    try:
        out = await _engine_call(lambda e: e.reset_stats())
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


class CacheBlendRequest(BaseModel):
    chunk_size: int = 512
    max_chunks: int = 16
    text: str = ""


@router.post("/engine/cache_blend/enable")
async def cache_blend_enable(body: CacheBlendRequest):
    try:
        out = await _engine_call(
            lambda e: e.enable_cache_blend(
                chunk_size=body.chunk_size, max_chunks=body.max_chunks))
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/cache_blend/register")
async def cache_blend_register(body: CacheBlendRequest):
    try:
        out = await _engine_call(
            lambda e: e.register_blend_chunk(body.text))
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


# ── sessions ──────────────────────────────────────────────────────────

class SessionRequest(BaseModel):
    session_id: str = ""
    ttl_s: float = 300
    prompt: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


@router.get("/engine/sessions")
async def session_stats():
    try:
        out = await _engine_call(lambda e: e.session_stats())
        return {"ok": True, "stats": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/sessions/begin")
async def session_begin(body: SessionRequest):
    try:
        sid = body.session_id or f"gui_{id(body) & 0xFFFFFFFF:08x}"
        out = await _engine_call(
            lambda e: e.begin_session(sid, ttl=body.ttl_s))
        # begin_session returns a SessionState dataclass — serialize it.
        return {"ok": True, "result": _session_dict(out)}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/sessions/continue")
async def session_continue(body: SessionRequest):
    try:
        out = await _engine_call(
            lambda e: e.continue_session(
                body.session_id, body.prompt,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature, top_p=body.top_p),
            timeout_s=300)
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/sessions/pin")
async def session_pin(body: SessionRequest):
    try:
        out = await _engine_call(
            lambda e: e.pin_session(body.session_id, ttl=body.ttl_s))
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/sessions/unpin")
async def session_unpin(body: SessionRequest):
    try:
        out = await _engine_call(
            lambda e: e.unpin_session(body.session_id))
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/sessions/end")
async def session_end(body: SessionRequest):
    try:
        out = await _engine_call(
            lambda e: e.end_session(body.session_id))
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


# ── engine library ────────────────────────────────────────────────────

class LibEntryRequest(BaseModel):
    content: str = ""
    category: str = "general"
    priority: int = 5
    tags: list[str] = []
    description: str = ""
    triggers: list[str] = []


@router.get("/engine/library/stats")
async def library_stats():
    try:
        out = await _engine_call(lambda e: e.library_stats())
        return {"ok": True, "stats": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.get("/engine/library/list")
async def library_list():
    try:
        out = await _engine_call(lambda e: e.library.list_entries())
        return {"ok": True, "entries": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/library/save")
async def library_save(body: LibEntryRequest):
    try:
        out = await _engine_call(
            lambda e: e.library_save(
                content=body.content, category=body.category,
                priority=body.priority, tags=body.tags,
                description=body.description, triggers=body.triggers))
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/library/search")
async def library_search(body: dict):
    try:
        out = await _engine_call(
            lambda e: e.library_search(body.get("query", "")))
        return {"ok": True, "results": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/library/lookup")
async def library_lookup(body: dict):
    try:
        out = await _engine_call(
            lambda e: e.library_lookup(
                category=body.get("category"),
                tags=body.get("tags")))
        return {"ok": True, "results": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/library/enabled")
async def library_enabled(body: dict):
    try:
        out = await _engine_call(
            lambda e: e.library_set_enabled(bool(body.get("enabled"))))
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/library/budget")
async def library_budget(body: dict):
    try:
        out = await _engine_call(
            lambda e: e.library_set_budget(int(body.get("budget", 256))))
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/library/optimize")
async def library_optimize():
    try:
        out = await _engine_call(lambda e: e.library_optimize(),
                                 timeout_s=300)
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


# ── merge ─────────────────────────────────────────────────────────────

class MergeRequest(BaseModel):
    parents: list[str] = []
    method: str = "blockwise_crossover"
    out_path: str = ""
    hot_swap: bool = False
    generations: int = 4
    population: int = 6
    elitism: int = 2
    crossover: float = 0.5
    mutation: float = 0.1
    mut_rate: float = 0.05
    bench_prompt: str = "def sort_list(x):"
    bench_tokens: int = 32


@router.post("/engine/merge")
async def engine_merge(body: MergeRequest):
    # Parents are read paths, out_path is a write path — all confined.
    parents = [_confined_path(p, _checkpoint_dirs(), must_exist=True)
               for p in body.parents]
    if not parents or any(p is None for p in parents):
        return {"ok": False, "error":
                "parents must be existing files under research/checkpoints"}
    out_path = None
    if body.out_path:
        out_path = _confined_path(body.out_path, _writable_dirs())
        if out_path is None:
            return {"ok": False, "error":
                    "out_path must live under research/checkpoints or data/merged"}
    try:
        out = await _engine_call(
            lambda e: e.merge_checkpoints(
                parents, method=body.method,
                out_path=out_path,
                # engine kwarg is load_result — 'hot_swap' was silently
                # swallowed by **kwargs and the swap never happened.
                load_result=body.hot_swap), timeout_s=1800)
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@router.post("/engine/evolve_merge")
async def engine_evolve_merge(body: MergeRequest):
    parents = [_confined_path(p, _checkpoint_dirs(), must_exist=True)
               for p in body.parents]
    if not parents or any(p is None for p in parents):
        return {"ok": False, "error":
                "parents must be existing files under research/checkpoints"}
    out_dir = None
    if body.out_path:
        out_dir = _confined_path(body.out_path, _writable_dirs())
        if out_dir is None:
            return {"ok": False, "error":
                    "out_path must live under research/checkpoints or data/merged"}
    try:
        out = await _engine_call(
            lambda e: e.evolve_merge(
                parents,
                # Real signature: n_generations=, population_size=,
                # mutation_rate=, out_dir=, benchmark_prompt=/_tokens —
                # the old call (out_path=/generations=/population=/
                # mut_rate=) TypeError'd on every request.
                out_dir=out_dir,
                n_generations=body.generations,
                population_size=body.population,
                elitism=body.elitism,
                mutation_rate=body.mut_rate,
                benchmark_prompt=body.bench_prompt,
                benchmark_tokens=body.bench_tokens), timeout_s=3600)
        return {"ok": True, "result": out}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


# ── activation catalog ────────────────────────────────────────────────

@router.get("/activation/catalog")
async def activation_catalog():
    from forge_gui.api import activation_catalog as ac
    fields = ac.fields_by_category()
    return {
        "categories": [
            {"name": cat, "fields": [
                {"name": f.name, "kind": f.kind, "label": f.label,
                 "tooltip": f.tooltip, "default": f.default,
                 "options": [{"value": o.value, "label": o.label,
                              "tip": o.tip}
                             for o in (f.options or ())],
                 "lo": f.lo, "hi": f.hi, "step": f.step,
                 "decimals": f.decimals, "suffix": f.suffix}
                for f in fl]}
            for cat, fl in fields],
        "presets": [{"name": p.name, "desc": p.description}
                    for p in ac.PRESETS],
        "default": ac.default_config(),
    }


@router.get("/activation/preset/{name}")
async def activation_preset(name: str):
    from forge_gui.api import activation_catalog as ac
    cfg = ac.preset_config(name)
    if cfg is None:
        return {"ok": False, "error": "unknown preset"}
    return {"ok": True, "config": cfg}


@router.post("/activation/validate")
async def activation_validate(body: dict):
    from forge_gui.api import activation_catalog as ac
    return {"errors": ac.validate(body.get("config") or body)}


@router.post("/activation/diff")
async def activation_diff(body: dict):
    from forge_gui.api import activation_catalog as ac
    current = services.engine.info.get("activation")
    return {"diff": ac.active_diff(current, body.get("config") or body)}


# ── models index ──────────────────────────────────────────────────────

@router.get("/models")
async def models():
    loop = asyncio.get_running_loop()
    entries = await loop.run_in_executor(
        None, services.models_index.models)
    return {"models": [
        {"name": m.name, "path": m.path, "size_bytes": m.size_bytes,
         "size_label": m.size_label, "ext": m.ext,
         "config_name": m.config_name, "config": m.config,
         "meta": m.meta, "modified": m.modified,
         "is_safetensors": m.is_safetensors,
         "is_flux": m.ext == ".flux",
         "is_lora": "lora" in m.name.lower()}
        for m in entries]}


@router.get("/models/configs")
async def model_configs():
    loop = asyncio.get_running_loop()
    cfgs = await loop.run_in_executor(
        None, services.models_index.configs)
    return {"configs": [vars(c) for c in cfgs]}


@router.delete("/models")
async def delete_model(path: str):
    from forge_gui.api.status_reader import project_root
    root = project_root().resolve()
    ckpt_dir = (root / "research" / "checkpoints").resolve()
    p = (root / path).resolve()
    # resolve() FIRST, then containment — a lexical relative_to check on
    # an unresolved path is bypassable via '..' segments.
    if not (p == ckpt_dir or ckpt_dir in p.parents):
        return {"ok": False, "error": "outside checkpoints dir"}
    if not p.is_file() or p.suffix not in (".safetensors", ".gguf",
                                           ".pt", ".bin", ".flux"):
        return {"ok": False, "error": "not a model file"}
    p.unlink()
    # Real sidecars are '<filename>.meta.json' (e.g. m.safetensors.meta.json)
    # — with_suffix('.meta.json') would look for 'm.meta.json' instead.
    for side in (Path(str(p) + ".meta.json"), p.with_suffix(".json"),
                 p.with_suffix(".train.pt")):
        try:
            if side.is_file():
                side.unlink()
        except OSError:
            pass
    return {"ok": True}


class BootRequest(BaseModel):
    checkpoint: str = ""
    config_name: str = "forgelm_v2"
    prompt: str = "def fibonacci(n):"
    max_tokens: int = 64


@router.post("/models/boot")
async def models_boot(body: BootRequest):
    return await services.boot.boot_and_test(
        body.checkpoint, body.config_name, body.prompt, body.max_tokens)


@router.post("/models/boot/cancel")
async def models_boot_cancel():
    services.boot.cancel()
    return {"ok": True}


# ── LoRA ──────────────────────────────────────────────────────────────

@router.get("/lora")
async def lora_index():
    loop = asyncio.get_running_loop()
    adapters = await loop.run_in_executor(None, services.lora.scan)
    return {"adapters": adapters, "status": services.lora.status()}


class LoraLoadRequest(BaseModel):
    path: str = ""
    rank: int = 32
    alpha: int | None = None
    target_key: str = "default"


@router.post("/lora/load")
async def lora_load(body: LoraLoadRequest):
    return await services.lora.load_on_engine(
        body.path, _clamp(body.rank, LORA_RANK_MIN, LORA_RANK_MAX),
        _clamp(body.alpha, LORA_ALPHA_MIN, LORA_ALPHA_MAX),
        body.target_key)


@router.post("/lora/unload")
async def lora_unload():
    return await services.lora.unload_from_engine()


@router.get("/lora/info")
async def lora_info():
    return await services.lora.refresh_info()


class LoraMergeRequest(BaseModel):
    base: str = ""
    config_name: str = "forgelm_v2"
    adapter: str = ""
    rank: int = 32
    alpha: int | None = None
    out: str = ""


@router.post("/lora/merge")
async def lora_merge(body: LoraMergeRequest):
    return await services.lora.merge(
        body.base, body.config_name, body.adapter,
        _clamp(body.rank, LORA_RANK_MIN, LORA_RANK_MAX),
        _clamp(body.alpha, LORA_ALPHA_MIN, LORA_ALPHA_MAX),
        body.out)


@router.post("/lora/mode")
async def lora_mode(body: dict):
    return await services.lora.set_mode(body.get("mode", "chat"))


@router.post("/lora/pin")
async def lora_pin(body: dict):
    return await services.lora.pin_adapter(body.get("path", ""))


@router.post("/lora/unpin")
async def lora_unpin():
    return await services.lora.unpin()


@router.get("/lora/recommend")
async def lora_recommend(mode: str = "chat"):
    loop = asyncio.get_running_loop()
    return {"adapter": await loop.run_in_executor(
        None, services.lora.recommend_for_mode, mode)}
