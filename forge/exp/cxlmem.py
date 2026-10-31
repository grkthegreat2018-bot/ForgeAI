"""E-cxl: CXL-style pooled/tiered memory prototype for KV cache (+ FFN weights).

No CXL hardware exists on this box, so the "CXL tier" is *pinned host RAM*
reached over PCIe — the same transport a CXL Type-3 memory expander sits
behind (load/store semantics approximated by explicit page copies; the
measured cost is the honest PCIe transfer, which is the dominant term).

Tier model
----------
    GPU tier  (fast, small) — the most recent ``gpu_window_tokens`` K/V of
        each layer stay resident on-device: a fixed VRAM budget.
    CXL tier  (large, slow) — older K/V pages migrate to pinned host RAM
        (D2H, ``non_blocking``). On attention reads, pages prefetch back
        over the link (H2D) — either wholesale via ``materialize()`` or
        streamed in bursts inside an online-softmax merge via ``attend()``,
        so the transient GPU working set is ~one burst, not the whole tail.

API
---
    tier = TieredKV(n_layers=24, gpu_window_tokens=1024, page_tokens=128,
                    cxl_capacity_tokens=1 << 16)
    tier.append(li, k, v)            # post-RoPE K/V, (B, n_kv, T, hd)
    k, v = tier.materialize(li)      # full history materialized on GPU
    o = tier.attend(li, q, k, v)     # streamed full-context attention
    tier.prefetch_async(li)          # async read-ahead on the link stream
    tier.stats()

Model integration: ``attach_tiered_kv(model, tier, mode="stream"|"fetch")``
patches every GQA block's ``attn.forward``.  ``"stream"`` runs the
page-streaming merge for T==1 decode (bounded transient VRAM); ``"fetch"``
materializes the whole cold tail every step (the naive CXL reader).  While
attached, the tier owns ALL KV state — incoming ``past_key_value`` is
ignored and ``present`` returns None.  Call ``tier.reset()`` between
sequences.

Optional weight tier: ``wrap_ffn_tiered(model, block_ids, link)`` moves a
block's SwiGLU weights into pinned host RAM (dropped from the GPU module)
and streams them back per forward — weight pooling over CXL.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int,
                       elt: int = 2) -> int:
    """K+V bytes for one token across all layers (bf16/fp16 -> elt=2)."""
    return n_layers * 2 * n_kv_heads * head_dim * elt


class CxLLink:
    """Simulated CXL/PCIe link: owns the async migration stream + stats.

    ``h2d``/``d2h`` account bytes; every contiguous H2D transaction records
    a CUDA-event pair (GPU) or a perf_counter delta (CPU) so per-miss
    latency is measured, never assumed.
    """

    def __init__(self, device: torch.device | str = "cuda"):
        self.device = torch.device(device)
        self.is_cuda = (self.device.type == "cuda"
                        and torch.cuda.is_available())
        self.stream = (torch.cuda.Stream(device=self.device)
                       if self.is_cuda else None)
        self.stats = {
            "h2d_bytes": 0,        # CXL -> GPU reads/prefetches
            "d2h_bytes": 0,        # GPU -> CXL spills
            "misses": 0,           # cold pages fetched H2D
            "fetch_ops": 0,        # contiguous H2D transactions
            "hits": 0,             # reads served fully by the GPU tier
            "spill_events": 0,     # hot->cold page migrations
            "dropped_tokens": 0,   # evicted past CXL capacity (data lost)
            "materialize_calls": 0,
            "attend_calls": 0,
            "w_h2d_bytes": 0,      # weight-tier H2D traffic
            "w_fetch_ops": 0,
        }
        self._fetch_ev: list = []  # (e0, e1, nbytes, npages) on cuda
        self._fetch_s: list = []   # (dt_s, nbytes, npages) on cpu

    # ── transfers ────────────────────────────────────────────────────
    def h2d(self, dst: torch.Tensor, src: torch.Tensor, n_pages: int,
            timed: bool = True):
        """Host(CXL) -> device copy with honest transfer accounting.

        ``src`` should be contiguous pinned memory for true async DMA;
        works with any src (sync fallback) on CPU.
        """
        n = src.numel() * src.element_size()
        self.stats["h2d_bytes"] += n
        # `misses` counts KV pages (a k+v pair is one page) — callers pass
        # n_pages=0 for the v copy of a pair.
        self.stats["misses"] += n_pages
        self.stats["fetch_ops"] += 1
        if self.is_cuda:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            dst.copy_(src, non_blocking=True)
            e1.record()
            if timed:
                self._fetch_ev.append((e0, e1, n, n_pages))
        else:
            t0 = time.perf_counter()
            dst.copy_(src)
            self._fetch_s.append((time.perf_counter() - t0, n, n_pages))

    def d2h(self, dst: torch.Tensor, src: torch.Tensor):
        """Device -> pinned host(CXL) spill."""
        n = src.numel() * src.element_size()
        self.stats["d2h_bytes"] += n
        dst.copy_(src, non_blocking=self.is_cuda)

    def w_h2d(self, nbytes: int):
        self.stats["w_h2d_bytes"] += nbytes
        self.stats["w_fetch_ops"] += 1

    # ── latency reporting ────────────────────────────────────────────
    def fetch_latencies_ms(self) -> list[float]:
        """Per-transaction H2D transfer time, ms."""
        if self.is_cuda:
            out = []
            for e0, e1, _n, _p in self._fetch_ev:
                e1.synchronize()
                out.append(e0.elapsed_time(e1))
            return out
        return [d * 1e3 for d, _n, _p in self._fetch_s]

    def mean_miss_ms(self) -> float:
        """Mean transfer ms per fetched page (miss)."""
        if self.is_cuda:
            tot_ms, tot_p = 0.0, 0
            for e0, e1, _n, p in self._fetch_ev:
                e1.synchronize()
                tot_ms += e0.elapsed_time(e1)
                tot_p += p
            return tot_ms / max(tot_p, 1)
        tot_ms = sum(d * 1e3 for d, _n, _p in self._fetch_s)
        tot_p = sum(p for _d, _n, p in self._fetch_s)
        return tot_ms / max(tot_p, 1)

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0
        self._fetch_ev.clear()
        self._fetch_s.clear()


class _TierState:
    """Per-layer buffers.

    ``hot_k/hot_v``: (B, H, W, hd) on device — the resident GPU window.
    ``cold_k/cold_v``: (cap, B, H, hd) pinned host — token-major so any
        token range [c0:c0+n) is ONE contiguous DMA block.
    """

    __slots__ = (
        "cold_k",
        "cold_len",
        "cold_v",
        "head_dim",
        "hot_k",
        "hot_len",
        "hot_v",
        "n_kv",
        "stage_ev",
        "stage_k",
        "stage_v",
        "staged",
    )

    def __init__(self):
        self.hot_k = self.hot_v = None
        self.hot_len = 0
        self.cold_k = self.cold_v = None
        self.cold_len = 0
        self.n_kv = 0
        self.head_dim = 0
        # persistent link-stream staging (prefetch read-ahead target);
        # token-major like the cold pool, grown geometrically
        self.stage_k = self.stage_v = None
        self.stage_ev = None
        self.staged = 0       # cold_len covered by the staged copy


class TieredKV:
    """Two-tier KV cache: GPU hot window + CXL (pinned host) cold pool.

    Args:
        n_layers: transformer block count.
        gpu_window_tokens: resident GPU budget W per layer (the "fast tier").
        page_tokens: migration granularity — hot->cold spills and cold->GPU
            fetches happen in whole pages.
        cxl_capacity_tokens: pinned-host pool size per layer.  Overflow
            drops the OLDEST tokens (counted in stats['dropped_tokens']).
        burst_pages: pages per H2D burst in streamed ``attend``.
        prefetch_ahead: when True, ``attend`` consumes bursts staged by
            ``prefetch_async`` on the link stream (read-ahead overlap).
        device/dtype: GPU tier placement; dtype is inferred on first append.
    """

    def __init__(self, n_layers: int, gpu_window_tokens: int = 1024,
                 page_tokens: int = 128, cxl_capacity_tokens: int = 1 << 16,
                 burst_pages: int = 8, prefetch_ahead: bool = False,
                 device: torch.device | str = "cuda",
                 link: CxLLink | None = None):
        self.n_layers = n_layers
        self.W = int(gpu_window_tokens)
        self.page = int(page_tokens)
        self.cxl_cap = int(cxl_capacity_tokens)
        self.burst_pages = int(burst_pages)
        self.prefetch_ahead = prefetch_ahead
        self.device = torch.device(device)
        self.link = link if link is not None else CxLLink(self.device)
        self._st = [_TierState() for _ in range(n_layers)]

    # ── state helpers ────────────────────────────────────────────────
    def _state(self, li: int, k: torch.Tensor | None = None) -> _TierState:
        st = self._st[li]
        if st.hot_k is None:
            assert k is not None
            B, H, _T, hd = k.shape
            pin = self.link.is_cuda
            st.n_kv, st.head_dim = H, hd
            st.hot_k = torch.zeros(B, H, self.W, hd, dtype=k.dtype,
                                   device=self.device)
            st.hot_v = torch.zeros_like(st.hot_k)
            st.cold_k = torch.zeros(self.cxl_cap, B, H, hd, dtype=k.dtype,
                                    device="cpu", pin_memory=pin)
            st.cold_v = torch.zeros_like(st.cold_k)
        return st

    def seq_len(self, li: int) -> int:
        st = self._st[li]
        return st.cold_len + st.hot_len

    @property
    def total_len(self) -> int:
        return self.seq_len(0)

    def reset(self):
        """Drop all tier state (new sequence). Frees the pinned pool."""
        self._st = [_TierState() for _ in range(self.n_layers)]

    def stats(self) -> dict:
        s = dict(self.link.stats)
        s["seq_len"] = self.total_len
        s["cold_len"] = self._st[0].cold_len
        s["hot_len"] = self._st[0].hot_len
        s["stage_bytes"] = sum(
            2 * st.stage_k.numel() * st.stage_k.element_size()
            for st in self._st if st.stage_k is not None)
        s["fetch_latencies_ms"] = [round(x, 4)
                                   for x in self.link.fetch_latencies_ms()]
        s["miss_ms_mean"] = round(self.link.mean_miss_ms(), 4)
        return s

    # ── memory accounting ────────────────────────────────────────────
    def elt_bytes(self) -> int:
        return self._st[0].hot_k.element_size() if self._st[0].hot_k is not None else 2

    def persistent_gpu_bytes(self) -> int:
        """Allocated resident GPU footprint (hot windows, all layers)."""
        st = self._st[0]
        if st.hot_k is None:
            return 0
        per = 2 * st.n_kv * self.W * st.head_dim * st.hot_k.element_size()
        return per * self.n_layers

    def cpu_pool_bytes(self) -> int:
        """Allocated pinned host pool (cold buffers, all layers)."""
        st = self._st[0]
        if st.cold_k is None:
            return 0
        return (2 * self.cxl_cap * st.n_kv * st.head_dim
                * st.cold_k.element_size() * self.n_layers)

    def logical_kv_bytes(self) -> int:
        st = self._st[0]
        if st.hot_k is None:
            return 0
        return (2 * st.n_kv * st.head_dim * st.hot_k.element_size()
                * self.total_len * self.n_layers)

    # ── tier management ──────────────────────────────────────────────
    def _cold_write(self, st: _TierState, k: torch.Tensor,
                    v: torch.Tensor):
        """Append n tokens to the cold (CXL) pool.

        k/v: (B, H, n, hd).  Writes token-major [c0:c0+n).  If the pool
        overflows, the OLDEST tokens fall off the front (and, if n > cap,
        the oldest incoming tokens are dropped too) — all counted in
        ``stats['dropped_tokens']``.
        """
        n = k.shape[2]
        if n <= 0:
            return
        total = st.cold_len + n
        if total > self.cxl_cap:
            drop = total - self.cxl_cap
            drop_old = min(drop, st.cold_len)
            keep = st.cold_len - drop_old
            if keep > 0:
                # overlapping left-shift — clone the source first
                st.cold_k[:keep].copy_(st.cold_k[drop_old:st.cold_len].clone())
                st.cold_v[:keep].copy_(st.cold_v[drop_old:st.cold_len].clone())
            st.cold_len = keep
            skip = drop - drop_old        # oldest incoming tokens lost too
            if skip:
                k = k[:, :, skip:]
                v = v[:, :, skip:]
                n -= skip
            self.link.stats["dropped_tokens"] += drop
        c0 = st.cold_len
        # (B,H,n,hd) -> (n,B,H,hd) token-major contiguous dst
        self.link.d2h(st.cold_k[c0:c0 + n], k.permute(2, 0, 1, 3))
        self.link.d2h(st.cold_v[c0:c0 + n], v.permute(2, 0, 1, 3))
        st.cold_len += n

    def _spill(self, st: _TierState, n_tokens: int):
        """Migrate the oldest ``n_tokens`` of the hot window to CXL."""
        if n_tokens <= 0:
            return
        self._cold_write(st, st.hot_k[:, :, :n_tokens],
                         st.hot_v[:, :, :n_tokens])
        remain = st.hot_len - n_tokens
        if remain > 0:
            # overlapping left-shift: clone the source first — an in-place
            # overlapping copy_ is UB on CUDA (parallel elementwise order)
            # and nondeterministically corrupted decode KV here.
            st.hot_k[:, :, :remain] = st.hot_k[:, :, n_tokens:st.hot_len].clone()
            st.hot_v[:, :, :remain] = st.hot_v[:, :, n_tokens:st.hot_len].clone()
        st.hot_len = remain
        self.link.stats["spill_events"] += 1

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append post-RoPE K/V for ``layer_idx``: (B, n_kv, T, hd)."""
        st = self._state(layer_idx, k)
        T = k.shape[2]
        k = k.to(device=self.device)
        v = v.to(device=self.device)
        if T >= self.W:
            # bulk ingest (prefill): everything older than the newest W
            # tokens belongs to the CXL pool.
            self._spill(st, st.hot_len)
            self._cold_write(st, k[:, :, :T - self.W],
                             v[:, :, :T - self.W])
            st.hot_k[:, :, :self.W] = k[:, :, T - self.W:]
            st.hot_v[:, :, :self.W] = v[:, :, T - self.W:]
            st.hot_len = self.W
            return
        if st.hot_len + T > self.W:
            # evict oldest hot tokens, page-granular
            ovf = st.hot_len + T - self.W
            n_spill = min(st.hot_len,
                          ((ovf + self.page - 1) // self.page) * self.page)
            self._spill(st, n_spill)
        st.hot_k[:, :, st.hot_len:st.hot_len + T] = k
        st.hot_v[:, :, st.hot_len:st.hot_len + T] = v
        st.hot_len += T

    # ── reads ────────────────────────────────────────────────────────
    def materialize(self, layer_idx: int):
        """Prefetch the full history of ``layer_idx`` back to the GPU tier.

        Returns (k, v): (B, n_kv, seq_len, hd) device tensors.  The cold
        tail crosses the link in ONE contiguous DMA (token-major layout),
        then a small device-side permute restores (B,H,T,hd).
        """
        st = self._st[layer_idx]
        self.link.stats["materialize_calls"] += 1
        if st.hot_k is None:
            return None, None
        cl, hl = st.cold_len, st.hot_len
        if cl == 0:
            self.link.stats["hits"] += 1
            return (st.hot_k[:, :, :hl], st.hot_v[:, :, :hl])
        B, H, hd = st.hot_k.shape[0], st.n_kv, st.head_dim
        dev = self.device
        out_k = torch.empty(B, H, cl + hl, hd, dtype=st.hot_k.dtype,
                            device=dev)
        out_v = torch.empty_like(out_k)
        # CXL -> GPU: one contiguous transaction of `cl` tokens
        tk = torch.empty(cl, B, H, hd, dtype=st.cold_k.dtype, device=dev)
        tv = torch.empty_like(tk)
        npages = (cl + self.page - 1) // self.page
        self.link.h2d(tk, st.cold_k[:cl], npages)
        self.link.h2d(tv, st.cold_v[:cl], 0)
        out_k[:, :, :cl] = tk.permute(1, 2, 0, 3)
        out_v[:, :, :cl] = tv.permute(1, 2, 0, 3)
        if hl:
            out_k[:, :, cl:] = st.hot_k[:, :, :hl]
            out_v[:, :, cl:] = st.hot_v[:, :, :hl]
        return out_k, out_v

    # ── prefetch (read-ahead over the link) ──────────────────────────
    def prefetch_async(self, layer_idx: int):
        """Stage a layer's cold tail into a persistent per-layer GPU
        staging buffer on the link stream; the next ``attend`` consumes
        it with one event wait — transfer hides under intervening layers'
        compute.  The staging buffer is *extra* resident GPU memory and
        is reported in stats (``stage_bytes``) — an honest read-ahead
        cost, like a CXL device's DRAM staging area."""
        if not self.link.is_cuda or layer_idx >= self.n_layers:
            return
        st = self._st[layer_idx]
        cl = st.cold_len
        if cl == 0:
            return
        if st.staged == cl and st.stage_ev is not None:
            return                              # already covering
        s = self.link.stream
        if st.stage_k is None or st.stage_k.shape[0] < cl:
            # grow staging (rare); drain the link before freeing old
            s.synchronize()
            cap = max(cl, 4096 if st.stage_k is None
                      else st.stage_k.shape[0] * 2)
            st.stage_k = torch.empty(cap, *st.cold_k.shape[1:],
                                     dtype=st.cold_k.dtype,
                                     device=self.device)
            st.stage_v = torch.empty_like(st.stage_k)
            st.staged = 0
        with torch.cuda.stream(s):
            self.link.h2d(st.stage_k[:cl], st.cold_k[:cl],
                          (cl + self.page - 1) // self.page)
            self.link.h2d(st.stage_v[:cl], st.cold_v[:cl], 0)
            st.stage_ev = torch.cuda.Event()
            st.stage_ev.record(s)
        st.staged = cl

    # ── streamed attention over the tiers ────────────────────────────
    def attend(self, layer_idx: int, q: torch.Tensor,
               k_new: torch.Tensor, v_new: torch.Tensor,
               append: bool = True) -> torch.Tensor:
        """Full-context attention with a bounded GPU working set.

        q: (B, H, T, hd) — H query heads.  k_new/v_new: (B, G, T, hd).
        Merges [cold bursts | hot window | new tokens] with an online
        softmax in fp32; cold bursts cross the link one at a time (or are
        consumed from a ``prefetch_async`` stage).  Appends k_new/v_new.

        Returns (B, H, T, hd) in q.dtype.
        """
        st = self._state(layer_idx, k_new)
        self.link.stats["attend_calls"] += 1
        B, H, T, hd = q.shape
        G = st.n_kv
        R = H // G
        dev = q.device
        scale = 1.0 / math.sqrt(hd)
        q5 = q.reshape(B, G, R, T, hd).float()
        m = torch.full((B, G, R, T, 1), float("-inf"),
                       device=dev, dtype=torch.float32)
        l = torch.zeros_like(m)
        acc = torch.zeros(B, G, R, T, hd, device=dev,
                          dtype=torch.float32)

        causal = None
        if T > 1:
            causal = torch.full((T, T), float("-inf"), device=dev)
            causal = causal.triu(1)[None, None, None]

        def absorb(kc, vc, mask=None):
            # kc/vc: (B, G, C, hd)
            s = torch.matmul(q5, kc.unsqueeze(2).transpose(-1, -2)
                             .float()) * scale          # (B,G,R,T,C)
            if mask is not None:
                s = s + mask
            mc = s.amax(-1, keepdim=True)
            p = (s - mc).exp()
            oc = torch.matmul(p, vc.unsqueeze(2).float())
            m_new = torch.maximum(m, mc)
            a = torch.exp(m - m_new)
            b = torch.exp(mc - m_new)
            acc.mul_(a).add_(oc * b)
            l.mul_(a).add_(p.sum(-1, keepdim=True) * b)
            m.copy_(m_new)

        # ── cold tier: burst fetches (or a staged prefetch) ──
        cl = st.cold_len
        if cl:
            burst_tok = self.burst_pages * self.page
            use_staged = (self.link.is_cuda and st.staged == cl
                          and st.stage_ev is not None)
            if use_staged:
                torch.cuda.current_stream(self.device).wait_event(
                    st.stage_ev)
                src_k, src_v = st.stage_k, st.stage_v
            else:
                src_k, src_v = st.cold_k, st.cold_v
            for off in range(0, cl, burst_tok):
                n = min(burst_tok, cl - off)
                if use_staged:
                    kb = src_k[off:off + n]          # already on-device
                    vb = src_v[off:off + n]
                elif self.link.is_cuda:
                    kb = torch.empty(n, *st.cold_k.shape[1:],
                                     dtype=st.cold_k.dtype, device=dev)
                    vb = torch.empty_like(kb)
                    self.link.h2d(kb, st.cold_k[off:off + n],
                                  (n + self.page - 1) // self.page)
                    self.link.h2d(vb, st.cold_v[off:off + n], 0)
                else:
                    kb = st.cold_k[off:off + n]
                    vb = st.cold_v[off:off + n]
                absorb(kb.permute(1, 2, 0, 3), vb.permute(1, 2, 0, 3))
        else:
            self.link.stats["hits"] += 1

        # ── GPU tier + newest tokens ──
        if st.hot_len:
            absorb(st.hot_k[:, :, :st.hot_len],
                   st.hot_v[:, :, :st.hot_len])
        absorb(k_new, v_new, causal)
        if append:
            self.append(layer_idx, k_new, v_new)
        return (acc / l).to(q.dtype).reshape(B, H, T, hd)


# ── model integration ────────────────────────────────────────────────

def _is_gqa(attn) -> bool:
    return (attn is not None
            and hasattr(attn, "q_proj") and hasattr(attn, "k_proj")
            and hasattr(attn, "v_proj") and hasattr(attn, "out_proj")
            and hasattr(attn, "n_kv_heads") and hasattr(attn, "rope"))


def _make_tiered_fwd(attn, li: int, tier: TieredKV, mode: str):
    orig = attn._cxl_orig

    def fwd(x, past_key_value=None, use_cache=False, **kw):
        B, T, _ = x.shape
        # Anything unusual (padding mask, packed seqs, a real preallocated
        # cache, training, CSA) bypasses the tier entirely.
        if (not use_cache or attn.training
                or kw.get("attention_bias") is not None
                or kw.get("position_ids") is not None
                or kw.get("preallocated_cache") is not None
                or getattr(attn, "_csa_enabled", False)):
            return orig(x, past_key_value=past_key_value,
                        use_cache=use_cache, **kw)
        if T == 1 and mode == "stream":
            q = attn.q_proj(x).view(B, T, attn.n_heads,
                                    attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(B, T, attn.n_kv_heads,
                                    attn.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(B, T, attn.n_kv_heads,
                                    attn.head_dim).transpose(1, 2)
            if attn.use_qk_norm and not attn._qk_norm_identity:
                q = attn.q_norm(q)
                k = attn.k_norm(k)
            off = tier.seq_len(li)
            if getattr(attn, "use_rope", False) and attn.rope is not None:
                q = attn.rope(q, offset=off)
                k = attn.rope(k, offset=off)
            # LearnedSink active -> degrade to fetch path (correct, slower)
            if attn.sinks is not None and attn.sinks.abs().max().item() != 0.0:
                past = tier.materialize(li)
                out, present = orig(x, past_key_value=past,
                                    use_cache=True, **kw)
                tier.append(li, present[0][:, :, -T:],
                            present[1][:, :, -T:])
                return out, None
            if tier.prefetch_ahead:
                tier.prefetch_async(li + 1)
            o = tier.attend(li, q, k, v)          # appends k/v internally
            o = o.transpose(1, 2).reshape(B, T, attn.n_heads * attn.head_dim)
            return attn.out_proj(o), None
        # fetch mode for T==1, and all T>1 chunk forwards in either mode
        if tier.seq_len(li) == 0:
            out, present = orig(x, past_key_value=None, use_cache=True, **kw)
            if present is not None:
                tier.append(li, present[0], present[1])
            return out, None
        past = tier.materialize(li)
        out, present = orig(x, past_key_value=past, use_cache=True, **kw)
        if present is not None:
            tier.append(li, present[0][:, :, -T:],
                        present[1][:, :, -T:])
        return out, None

    return fwd


def attach_tiered_kv(model, tier: TieredKV, mode: str = "stream") -> int:
    """Patch every GQA block's attn.forward to route K/V through ``tier``.

    mode: "stream" (page-streaming decode, bounded transient VRAM) or
    "fetch" (materialize the whole cold tail per step).  While attached,
    the tier owns all KV state — caller ``past_key_value`` is ignored and
    ``present`` is None.  Call ``tier.reset()`` before each new sequence.
    Returns the number of wrapped attention modules.
    """
    assert mode in ("stream", "fetch")
    wrapped = 0
    for i, blk in enumerate(model.blocks):
        attn = getattr(blk, "attn", None)
        if not _is_gqa(attn):
            continue
        if getattr(attn, "_cxl_orig", None) is None:
            attn._cxl_orig = attn.forward
        attn.forward = _make_tiered_fwd(attn, i, tier, mode)
        attn._cxl_tier = tier
        wrapped += 1
    model._cxl_tier = tier
    model._cxl_mode = mode
    return wrapped


def detach_tiered_kv(model) -> int:
    """Restore original attention forwards."""
    n = 0
    for blk in getattr(model, "blocks", []):
        attn = getattr(blk, "attn", None)
        if attn is not None and getattr(attn, "_cxl_orig", None) is not None:
            attn.forward = attn._cxl_orig
            attn._cxl_orig = None
            attn._cxl_tier = None
            n += 1
    if hasattr(model, "_cxl_tier"):
        model._cxl_tier = None
    return n


# ── weight tier (CXL-pooled FFN weights) ─────────────────────────────

class TieredFFN(nn.Module):
    """SwiGLU FFN whose weights live in pinned host RAM (the CXL pool).

    The original nn.Linear weights are dropped from the GPU module — the
    pinned tensors are plain attributes so ``.parameters()``/``.to()``
    never see them (they're *not* VRAM-resident).  Each forward stages the
    three matrices back over the link (transient, allocator-cached).
    ``prefetch_async`` lets the previous tiered block's FFN completion
    trigger this block's fetch on the link stream — the transfer hides
    under this block's attention compute.
    """

    def __init__(self, ffn, link: CxLLink):
        super().__init__()
        self.link = link
        self.d_model = ffn.w_gate.in_features
        self.hidden_dim = ffn.w_gate.out_features
        self.use_clamp = getattr(ffn, "use_clamp", False)
        self.clamp_alpha = getattr(ffn, "clamp_alpha", 1.702)
        self.clamp_limit = getattr(ffn, "clamp_limit", 7.0)
        pin = link.is_cuda
        self._wg = self._pin(ffn.w_gate.weight, pin)
        self._wu = self._pin(ffn.w_up.weight, pin)
        self._wd = self._pin(ffn.w_down.weight, pin)
        self._staged = None   # (wg, wu, wd, event) on GPU
        self._nxt: TieredFFN | None = None
        self.ffn_bytes = (self._wg.numel() + self._wu.numel()
                          + self._wd.numel()) * self._wg.element_size()

    @staticmethod
    def _pin(w: torch.Tensor, pin: bool) -> torch.Tensor:
        t = w.detach().to("cpu", copy=True)
        return t.pin_memory() if pin else t

    def prefetch_async(self):
        """Stage weights on the link stream (call one block early)."""
        if not self.link.is_cuda or self._staged is not None:
            return
        s = self.link.stream
        with torch.cuda.stream(s):
            wg = self._wg.to(self.link.device, non_blocking=True)
            wu = self._wu.to(self.link.device, non_blocking=True)
            wd = self._wd.to(self.link.device, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(s)
        self._staged = (wg, wu, wd, ev)
        self.link.w_h2d(self.ffn_bytes)

    def _weights(self):
        if self._staged is not None:
            wg, wu, wd, ev = self._staged
            self._staged = None
            cur = torch.cuda.current_stream(self.link.device)
            cur.wait_event(ev)
            for t in (wg, wu, wd):
                t.record_stream(cur)
            return wg, wu, wd
        dev = self.link.device
        if self.link.is_cuda:
            wg = torch.empty_like(self._wg, device=dev)
            wu = torch.empty_like(self._wu, device=dev)
            wd = torch.empty_like(self._wd, device=dev)
            wg.copy_(self._wg, non_blocking=True)
            wu.copy_(self._wu, non_blocking=True)
            wd.copy_(self._wd, non_blocking=True)
        else:
            wg, wu, wd = self._wg, self._wu, self._wd
        self.link.w_h2d(self.ffn_bytes)
        return wg, wu, wd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wg, wu, wd = self._weights()
        if self._nxt is not None:
            self._nxt.prefetch_async()
        gate = F.linear(x, wg)
        up = F.linear(x, wu)
        if self.use_clamp:
            gate = gate.clamp(min=None, max=self.clamp_limit)
            up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
            glu = gate * torch.sigmoid(self.clamp_alpha * gate)
            return F.linear((up + 1) * glu, wd)
        return F.linear(F.silu(gate) * up, wd)


def wrap_ffn_tiered(model, block_ids: list[int], link: CxLLink) -> dict:
    """Move SwiGLU FFN weights of ``block_ids`` into the CXL pool.

    Returns {"freed_gpu_bytes": int, "blocks": [...]}.  Only blocks whose
    ``ffn`` has w_gate/w_up/w_down are converted.  The original GPU weight
    storage is released; restore with ``unwrap_ffn_tiered``.
    """
    made = {}
    for i in block_ids:
        blk = model.blocks[i]
        ffn = blk.ffn
        if not (hasattr(ffn, "w_gate") and hasattr(ffn, "w_up")
                and hasattr(ffn, "w_down")) or isinstance(ffn, TieredFFN):
            continue
        tf = TieredFFN(ffn, link)
        blk.ffn = tf           # releases the GPU-resident weights
        made[i] = tf
    # chain prefetch: each tiered FFN arms the NEXT tiered block
    ids = sorted(made)
    for a, b in zip(ids, ids[1:]):
        made[a]._nxt = made[b]
    return {"freed_gpu_bytes": sum(t.ffn_bytes for t in made.values()),
            "blocks": ids}


def unwrap_ffn_tiered(model) -> int:
    """Restore SwiGLUFFN modules, streaming weights back to the GPU."""
    from forge.model.layers import SwiGLUFFN
    n = 0
    for blk in getattr(model, "blocks", []):
        ffn = getattr(blk, "ffn", None)
        if isinstance(ffn, TieredFFN):
            new = SwiGLUFFN(ffn.d_model, ffn.hidden_dim,
                            use_clamp=ffn.use_clamp,
                            clamp_alpha=ffn.clamp_alpha,
                            clamp_limit=ffn.clamp_limit)
            dev = ffn.link.device
            new = new.to(device=dev, dtype=ffn._wg.dtype)
            new.w_gate.weight.data.copy_(ffn._wg)
            new.w_up.weight.data.copy_(ffn._wu)
            new.w_down.weight.data.copy_(ffn._wd)
            blk.ffn = new
            n += 1
    return n
