"""Batched decoding strategy — processes multiple requests together.

Shifts matmul from GEMV (memory-bound at batch=1) to GEMM (compute-bound
at batch>1), delivering 3-5x throughput on RTX 5070 for 1.2B models.

Supports per-sequence generation settings:
  - temperature, top_p, top_k (sampling)
  - repetition_penalty
  - seed (per-sequence torch.Generator for reproducible generation)
  - stop sequences (string-based, checked after decode)
"""
import os

import torch
import torch.nn.functional as F

from forge.engine.decoding import DecodingStrategy
from forge.model_loader import unpack_output_with_kv


class BatchedDecoding(DecodingStrategy):
    """Decode multiple prompts in a single batched forward pass.

    Finished sequences stay in the batch (masked) so KV cache dimensions
    remain stable. Sampling happens on the full batch at once.

    Per-sequence settings (temperature, top_k, repetition_penalty, seed,
    stop) are applied independently — each sequence in the batch can have
    completely different generation parameters.
    """

    def __init__(self, eos_token_id: int = 7,
                 eos_token_ids: set[int] | None = None):
        self.eos_token_id = eos_token_id
        # Callers should pass the engine's resolved EOS set — on Jamba V2 a
        # chat turn ends on <|im_end|> (519) / <|endoftext|> (2), which the
        # legacy LFM/Qwen-only default would never stop on (batch runs to
        # max_tokens every time).
        self.eos_set = (set(eos_token_ids) if eos_token_ids
                        else {7, 151643, 151645})

    @property
    def name(self) -> str:
        return "BatchedDecoding"

    def generate(self, model, input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 top_k: int = 0, repetition_penalty: float = 1.0,
                 stop=None, tokenizer=None, **_ignored) -> torch.Tensor:
        # top_k/stop/tokenizer are real generate_batch params — dropping
        # them silently mis-sampled, and **_ignored swallows the rest of
        # the uniform engine kwarg surface (kv_compact, suffix_mem, ...).
        return self.generate_batch(
            model,
            [input_ids],
            [max_new_tokens],
            [temperature],
            [top_p],
            top_k_list=[top_k],
            repetition_penalty_list=[repetition_penalty],
            stop_list=[stop],
            tokenizer=tokenizer,
        )[0]

    def generate_batch(
        self,
        model,
        prompts: list[torch.Tensor],  # each [1, prompt_len]
        max_tokens_list: list[int],
        temperatures: list[float],
        top_ps: list[float],
        top_k_list: list[int] | None = None,
        repetition_penalty_list: list[float] | None = None,
        seed_list: list[int | None] | None = None,
        stop_list: list[list[str] | None] | None = None,
        tokenizer=None,
        processor_list: list | None = None,
        suppress_ids: list[int] | None = None,
    ) -> list[torch.Tensor]:
        N = len(prompts)
        if N == 0:
            return []
        device = prompts[0].device

        # Default per-sequence settings
        if top_k_list is None:
            top_k_list = [0] * N
        if repetition_penalty_list is None:
            repetition_penalty_list = [1.0] * N
        if seed_list is None:
            seed_list = [None] * N
        if stop_list is None:
            stop_list = [None] * N
        if processor_list is None:
            processor_list = [None] * N

        all_temperatures = list(temperatures)
        all_top_ps = list(top_ps)
        all_top_k = list(top_k_list)
        all_rp = list(repetition_penalty_list)
        all_seeds = list(seed_list)
        all_stops = list(stop_list)
        all_procs = list(processor_list)

        max_tokens = max(max_tokens_list)
        max_prompt_len = max(p.shape[1] for p in prompts)
        prompt_lens_all = [p.shape[1] for p in prompts]

        # Static-graph decode is lazily captured per batch size up to
        # WIDE_CAP. Prefer ONE wide graph over windowing: N <= WIDE_CAP
        # decodes every row in a single pass (no refills at all); only
        # N > WIDE_CAP windows at WIDE_CAP and REFILLs freed slots from
        # the pending queue — continuous batching: the old row's cache
        # slot gets a scratch-prefilled new prompt spliced in, so every
        # decode step keeps producing real tokens instead of idling on
        # early finishers' tails. The 32-slot window is the low-VRAM
        # fallback when a WIDE_CAP cache won't fit.
        SLOT_CAP = 32
        WIDE_CAP = 64
        slotted = False
        bstate = None
        ensure = getattr(model, '_ensure_static_batch', None)
        slot_n = N
        if N > SLOT_CAP and ensure is not None:
            cands = ([N] if N <= WIDE_CAP else [WIDE_CAP, SLOT_CAP])
            for cap in cands:
                hard_est = min(
                    max_tokens * ((N + cap - 1) // cap) + 1,
                    max(8192 - max_prompt_len - 1, max_tokens))
                try:
                    bstate = ensure(cap, max_prompt_len,
                                    max_prompt_len + hard_est)
                except Exception:
                    bstate = None
                # Conv/recurrent layers carry per-batch state the
                # splice can't reset — refill needs pure-attn caches.
                if bstate is not None and any(
                        kc is None
                        for kc in bstate['cache'].k_caches):
                    bstate = None
                if bstate is not None:
                    slot_n = cap
                    break
            # Refills only exist when prompts outnumber the window.
            slotted = bstate is not None and N > slot_n

        if bstate is None and N > SLOT_CAP:
            # No graph path — an N-wide eager batch OOMs on 12GB.
            # Sequential ≤32-wide chunks keep decode batched.
            outs = []
            for off in range(0, N, SLOT_CAP):
                hi = min(off + SLOT_CAP, N)
                outs += self.generate_batch(
                    model, prompts[off:hi],
                    max_tokens_list[off:hi], temperatures[off:hi],
                    top_ps[off:hi], top_k_list=top_k_list[off:hi],
                    repetition_penalty_list=repetition_penalty_list[off:hi],
                    seed_list=seed_list[off:hi],
                    stop_list=stop_list[off:hi], tokenizer=tokenizer,
                    processor_list=processor_list[off:hi],
                    suppress_ids=suppress_ids)
            return outs

        B = slot_n
        # Per-SLOT working copies — a refill overwrites its slot's row.
        temperatures = list(all_temperatures[:slot_n])
        top_ps = list(all_top_ps[:slot_n])
        top_k_list = list(all_top_k[:slot_n])
        repetition_penalty_list = list(all_rp[:slot_n])
        seed_list = list(all_seeds[:slot_n])
        stop_list = list(all_stops[:slot_n])
        processor_list = list(all_procs[:slot_n])

        # Slot bookkeeping: which prompt occupies each slot and where its
        # generated tokens start inside the shared gen_only buffer.
        slot_prompt = list(range(slot_n))
        slot_start = [0] * slot_n
        slot_cap = [max_tokens_list[i] for i in range(slot_n)]
        # First decode step index whose EOS flag may finish the slot.
        # Originals sample from step 0; a stream refilled at step R is
        # only responsible for flags recorded at steps > R — masks from
        # earlier steps describe the dead predecessor's filler tokens.
        eos_floor = [0] * slot_n
        pending = list(range(slot_n, N)) if slotted else []
        results: list = [None] * N

        # Per-sequence generators for reproducible sampling
        generators = []
        for seed in seed_list:
            if seed is not None:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                generators.append(gen)
            else:
                generators.append(None)
        # Batched-sampling generator: when ALL rows are seeded, one
        # generator seeded deterministically from the row-seed tuple
        # keeps the run reproducible while allowing a single (B,V)
        # multinomial per step instead of B per-row calls. Its stream
        # differs from the per-row streams — only determinism-given-
        # seed_list is the contract, not per-row stream identity.
        batch_gen = None
        if B > 0 and all(g is not None for g in generators):
            batch_gen = torch.Generator(device=device)
            comb = 5381
            for s in seed_list:
                comb = ((comb * 1000003) ^ int(s)) & ((1 << 63) - 1)
            batch_gen.manual_seed(comb)

        # Pad prompts to same length (LEFT-pad: zeros on left, prompt on right)
        # Left-padding is correct for causal LM generation — the last position
        # is the generation start, consistent across all sequences in the batch.
        padded_ids = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
        prompt_lens = []
        for i in range(B):
            L = prompt_lens_all[i]
            padded_ids[i, -L:] = prompts[i][0]  # right-align (pad left)
            prompt_lens.append(L)

        # Attention mask — attend to the rightmost L positions (the prompt)
        attn_mask = torch.zeros(B, max_prompt_len, dtype=torch.bool, device=device)
        for i, L in enumerate(prompt_lens):
            attn_mask[i, -L:] = True

        # Output rows are reassembled at the end from a shared
        # [B, hard_cap] buffer (gen_only below) — one column write per
        # step replaces the old per-row in-place appends, which cost B
        # python-indexed GPU writes every step.  Slotted mode needs more
        # columns than max_tokens: global steps span every refill.
        cache_cap = (bstate['cache'].max_seq_len - max_prompt_len - 1
                     if bstate is not None else max_tokens)
        hard_cap = (min(max_tokens * ((N + slot_n - 1) // slot_n) + 1,
                        cache_cap)
                    if slotted else max_tokens)

        # Static-KV graph path: lazily captures per-(batch, KV-bucket)
        # decode graphs at activation request time — one replay produces
        # B tokens, so the batch shares weight reads AND launch overhead.
        # Any failure falls through to the functional loop.
        if bstate is None and ensure is not None:
            try:
                bstate = ensure(B, max_prompt_len,
                                max_prompt_len + max_tokens)
            except Exception:
                bstate = None

        if bstate is not None:
            bcg = model._bcg
            graphs, bufs_map = bcg._decode_graphs, bcg._static_buffers
            cache, m_buf = bstate['cache'], bstate['mask']
            prefix, bbuckets = bstate['prefix'], bstate['buckets']
            cache.reset()
            m_buf.zero_()
            fwd_raw = getattr(model, '_orig_mod', model)
            with torch.inference_mode():
                out = fwd_raw(padded_ids, attention_mask=attn_mask,
                              preallocated_cache=cache, use_cache=True,
                              logits_last_only=True)
                logits, _ = unpack_output_with_kv(out)
            cache.position = max_prompt_len
            m_buf[:, :max_prompt_len] = attn_mask.to(m_buf.dtype)
            # Per-row ring validity: mark only each row's real prompt
            # slots — left-pad KV would otherwise stay attended on SWA
            # ring layers (the broadcast slot<=pos mask can't see
            # per-row prompt lengths).
            for l in range(cache.n_layers):
                rv = cache.ring_valid[l]
                if rv is None:
                    continue
                W = rv.shape[1]
                for i in range(B):
                    idx = torch.arange(
                        max_prompt_len - prompt_lens[i],
                        max_prompt_len, device=device) % W
                    rv[i].zero_()
                    rv[i].index_fill_(0, idx, True)
            past_kv = None

            def _step_fn(nids):
                pos = cache.position
                if pos >= cache.max_seq_len:
                    return None
                tag = next((f"{prefix}kv{b}" for b in bbuckets
                            if b >= pos + 1
                            and f"{prefix}kv{b}" in graphs), None)
                if tag is None:
                    return None
                m_buf[:, pos] = 1
                cache.pos_t.fill_(pos)
                # After a compaction the captured graph is wider than
                # the live row set — pad the input and drop dead
                # logits rows (their KV is written but never read).
                buf = bufs_map[tag]['input_ids']
                n_live = nids.shape[0]
                if buf.shape[0] == n_live:
                    buf.copy_(nids)
                else:
                    buf[:n_live].copy_(nids)
                    buf[n_live:].zero_()
                with torch.inference_mode():
                    graphs[tag].replay()
                cache.position += 1
                lg, _ = unpack_output_with_kv(bufs_map[tag]['output'])
                return lg if lg.shape[0] == n_live else lg[:n_live]

            # Pre-capture the compaction halving widths (B -> B/2 ->
            # B/4) at THIS run's horizon — _compact would otherwise
            # lazily capture + allocate a fresh cache the first time
            # the batch drains (multi-second stall mid-decode). ensure
            # is a no-op for already-captured widths; a failed
            # allocation just leaves that width lazy (compact then
            # falls back to the wider graph, same as before).
            if ensure is not None:
                _w = B // 2
                for _ in range(2):
                    if _w < 2:
                        break
                    try:
                        ensure(_w, cache.position,
                               min(cache.position + hard_cap, 8192))
                    except Exception:
                        break
                    _w //= 2
        else:
            # Prefill
            with torch.inference_mode():
                out = model(padded_ids, attention_mask=attn_mask,
                            use_cache=True, logits_last_only=True)
                logits, past_kv = unpack_output_with_kv(out)

            # Decode-time mask: grows by one (all-ones) column per step so
            # pad positions in the cached KV stay masked. Without this,
            # every decode step attends to pad-token KV — silent pollution
            # for any batch with unequal prompt lengths.
            cur_mask = attn_mask
            ones_col = torch.ones(B, 1, dtype=torch.bool, device=device)

            def _step_fn(nids):
                nonlocal past_kv, cur_mask
                cur_mask = torch.cat([cur_mask, ones_col], dim=1)
                with torch.inference_mode():
                    out = model(nids, past_key_values=past_kv,
                                attention_mask=cur_mask, use_cache=True)
                    lg, past_kv = unpack_output_with_kv(out)
                return lg

        # Decode — one batched step per token
        next_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
        temps_cpu = [max(t, 1e-5) for t in temperatures]
        active_cpu = torch.ones(B, dtype=torch.bool)
        active = torch.ones(B, dtype=torch.bool, device=device)
        eos_mask_gpu = torch.zeros(B, dtype=torch.bool, device=device)
        finished_gpu = torch.zeros(B, dtype=torch.bool, device=device)
        # Pipelined EOS check (CUDA): each step's flag + row mask are
        # copied to pinned host buffers ASYNC and consumed one step
        # LATER — the host sync that used to drain the pipeline every
        # step (eos_mask.any().item() forced the host to wait for the
        # in-flight replay) is replaced by an event sync on a copy that
        # was enqueued BEFORE the current replay → returns instantly.
        # Cost: <=1 overrun token per finished row (its slice is
        # trimmed by _finish's end index anyway).
        _pipe_eos = device.type == 'cuda'
        eos_bufs = [eos_mask_gpu,
                    torch.zeros(B, dtype=torch.bool, device=device)]
        if _pipe_eos:
            mask_host = [torch.zeros(B, dtype=torch.bool,
                                     pin_memory=True)
                         for _ in range(2)]
            flag_host = [torch.zeros((), dtype=torch.bool,
                                     pin_memory=True)
                         for _ in range(2)]
            eos_evs = [torch.cuda.Event(), torch.cuda.Event()]
            eos_live = [False, False]
        had_inactive = False
        V = logits.shape[-1]

        # Repetition-penalty state: [B,V] 'token seen' mask + per-row
        # penalty. One scatter + one where/div per step replaces B
        # per-row unique()+index_copy_ kernel launches.  Seeded with the
        # prompt ids — the original unique() covered prompt+generated.
        rp_any = any(rp != 1.0 for rp in repetition_penalty_list)
        if rp_any:
            seen = torch.zeros(B, V, dtype=torch.bool, device=device)
            for i in range(B):
                seen[i, padded_ids[i, -prompt_lens[i]:]] = True
            rp_pen = torch.tensor(
                repetition_penalty_list, device=device,
                dtype=logits.dtype).unsqueeze(1)          # [B,1]
        else:
            seen = rp_pen = None

        # Stop-string ids → [B,V] bool mask: one gather per step
        # replaces the per-step next_tokens.tolist() D2H round-trip.
        stop_mask = None
        if tokenizer is not None:
            _sm = torch.zeros(B, V, dtype=torch.bool, device=device)
            for i, stops in enumerate(stop_list):
                if stops:
                    for s in stops:
                        ids = tokenizer(
                            s, add_special_tokens=False).input_ids
                        if ids:
                            _sm[i, ids[-1]] = True
            if bool(_sm.any()):
                stop_mask = _sm

        # GPU-side structural ban — one index_fill per step covers every
        # row, replacing per-row Python processors for static ban sets.
        sup_t = (torch.tensor(sorted(set(suppress_ids)), dtype=torch.long,
                              device=device) if suppress_ids else None)

        # Temperatures are loop-invariant — build the divisor once.
        temp_t = torch.tensor(
            temps_cpu, device=device, dtype=logits.dtype).unsqueeze(1)

        # Generated tokens land in one [B, hard_cap] buffer: a single
        # column write per step replaces B python-indexed GPU writes.
        gen_only = torch.zeros(B, hard_cap, dtype=torch.long,
                               device=device)
        steps_done = 0
        proc_rows = [i for i in range(B) if processor_list[i] is not None]
        uniform_topk = len(set(top_k_list)) == 1 and top_k_list[0] > 0
        uniform_topp = len(set(top_ps)) == 1 and top_ps[0] < 1.0

        graph_B = B              # captured graph width (>= live rows
                                 # once _compact shrinks the batch)
        scratch = None           # lazy B=RB prefill cache for refills
        # Refills stall the whole decode window on an eager scratch
        # prefill (~50ms at B=1 — ~2k kernel launches). Deferring until
        # >= _REFILL_MIN slots are free and preflilling up to
        # _REFILL_CAP pending prompts in ONE batched call amortizes the
        # stall to a few ms per refilled stream.
        _REFILL_MIN = 6
        _REFILL_CAP = 16
        _dbg_rows: set = {0, 1} if os.environ.get(
            "FORGE_DBG_STEPS") else set()  # row 0 vs healthy row 1

        def _finish(i, e):
            """Record the finished prompt's [prompt|gen] output."""
            j = slot_prompt[i]
            plen = prompt_lens_all[j]
            s = slot_start[i]
            n = max(0, e - s)
            g = torch.zeros(1, plen + n, dtype=torch.long, device=device)
            g[0, :plen] = prompts[j][0]
            if n:
                g[0, plen:] = gen_only[i, s:e]
            results[j] = g

        def _drain_eos(q, e_end):
            """Consume the EOS flag enqueued for step ``e_end - 1``.

            ``eos_evs[q]`` was recorded right after that step's async
            flag+mask D2H copies — BEFORE the next ``_step_fn`` replay
            was enqueued — so synchronize() waits only for the copy,
            not for any model compute (instant in practice). Fired
            rows finish with outputs ending at ``e_end`` (their EOS
            token is the last included column)."""
            nonlocal had_inactive
            if not eos_live[q]:
                return
            eos_live[q] = False
            eos_evs[q].synchronize()
            if not bool(flag_host[q]):
                return
            mb = mask_host[q]
            for i in range(B):
                # A slot refilled at step R owes its flags only to
                # tokens sampled at steps > R — a mask recorded at
                # step e_end-1 <= R still describes the DEAD stream's
                # filler token, so it must not finish the new stream.
                if (bool(mb[i]) and active_cpu[i]
                        and eos_floor[i] <= e_end - 1):
                    active_cpu[i] = False
                    had_inactive = True
                    _finish(i, e_end)

        def _refill_batch(rows, js):
            """Splice pending prompts js into freed slots rows at the
            CURRENT decode position, sharing ONE batched scratch
            prefill. Prompts are left-padded to a common length and
            prefilled at position 0 (clean causal prefill — a nonzero
            scratch position leaves a zero-KV prefix that attention
            still scores, corrupting every refilled row), then each
            layer's KV is copied VERBATIM into the slot: position_ids
            are derived from mask-cumsum, so the refilled stream is
            numbered 0,1,2,... in mask space — K must keep R(src), any
            absolute-position rebase detunes RoPE.

            Full-attn layers: copy the last min(Lj,pos) real tokens to
            absolute slots [pos-n, pos); m_buf hides the stale prefix.

            Ring layers (swa_ring_kv): per-row ring_valid marks exactly
            the slots this stream holds — stale ring KV is excluded by
            the mask, not overwritten, so no fabricated context is
            needed. Real prompt tail goes to slots (pos-n..pos-1)%W;
            everything else stays masked."""
            nonlocal scratch, uniform_topk, uniform_topp
            k = len(rows)
            Lj_s = [prompt_lens_all[j] for j in js]
            Lmax = max(Lj_s)
            pos = cache.position
            if os.environ.get("FORGE_DBG_REFILL"):
                print(f"[refill] slots {rows} <- prompts {js} pos={pos} "
                      f"Lj={Lj_s} steps={steps_done}")
            # Left-pad the k real prompts to [_REFILL_CAP, Lmax]; rows
            # >= k are 1-token dummies — the scratch cache batch is
            # fixed at _REFILL_CAP and their KV is never read.
            padded = torch.zeros(_REFILL_CAP, Lmax, dtype=torch.long,
                                 device=device)
            amask = torch.zeros(_REFILL_CAP, Lmax, dtype=torch.bool,
                                device=device)
            for r, Lj in enumerate(Lj_s):
                padded[r, -Lj:] = prompts[js[r]][0]
                amask[r, -Lj:] = True
            for r in range(k, _REFILL_CAP):
                amask[r, -1] = True
            if scratch is None:
                from forge.model.kv_cache import create_kv_cache
                scratch = create_kv_cache(
                    model, max_total=cache.max_seq_len,
                    batch=_REFILL_CAP, device=device)
            scratch.position = 0
            for l in range(scratch.n_layers):
                skc, svc = scratch.k_caches[l], scratch.v_caches[l]
                if skc is not None:
                    skc.zero_()
                    svc.zero_()
            with torch.inference_mode():
                out = fwd_raw(padded, attention_mask=amask,
                              preallocated_cache=scratch, use_cache=True,
                              logits_last_only=True)
                lg, _ = unpack_output_with_kv(out)
            for r, i, j, Lj in zip(range(k), rows, js, Lj_s):
                p0 = Lmax - Lj    # left-pad offset inside scratch
                n_fill = min(Lj, pos)
                base = pos - n_fill
                # Per-slot metadata FIRST — the new stream's first
                # token must be sampled under ITS params (temp/top-k/
                # top-p/seed/rp), not the dead stream's.
                slot_prompt[i] = j
                slot_cap[i] = max_tokens_list[j]
                temperatures[i] = all_temperatures[j]
                temps_cpu[i] = max(all_temperatures[j], 1e-5)
                temp_t[i, 0] = temps_cpu[i]
                top_ps[i] = all_top_ps[j]
                top_k_list[i] = all_top_k[j]
                repetition_penalty_list[i] = all_rp[j]
                if rp_any:
                    rp_pen[i, 0] = all_rp[j]
                    seen[i].zero_()
                    seen[i, prompts[j][0]] = True
                seed_list[i] = all_seeds[j]
                if all_seeds[j] is not None:
                    g = torch.Generator(device=device)
                    g.manual_seed(all_seeds[j])
                    generators[i] = g
                else:
                    generators[i] = None
                stop_list[i] = all_stops[j]
                if stop_mask is not None:
                    stop_mask[i].zero_()
                    if all_stops[j] and tokenizer is not None:
                        for s in all_stops[j]:
                            ids = tokenizer(
                                s, add_special_tokens=False).input_ids
                            if ids:
                                stop_mask[i, ids[-1]] = True
                processor_list[i] = all_procs[j]
                # Pre-sample the new stream's first token from the
                # prefill logits and feed it as next_ids[i] for the
                # upcoming step: its KV lands at `pos`, keeping the
                # stream contiguous (prompt [pos-Lj,pos) + token@pos +
                # gen@pos+1..). The old design let the dead stream's
                # EOS occupy pos and masked it — a permanent +1
                # position hole that shifts every prompt→gen distance
                # and measurably corrupts generation.
                lgi = lg[r, -1].float()
                if rp_any:
                    lgi = lgi / torch.where(
                        seen[i], rp_pen[i],
                        torch.ones((), device=device, dtype=lgi.dtype))
                lgi = lgi / temp_t[i, 0]
                if top_k_list[i] > 0:
                    thr = lgi.topk(min(top_k_list[i], V)).values[-1]
                    lgi = torch.where(
                        lgi < thr,
                        torch.full_like(lgi, float('-inf')), lgi)
                if top_ps[i] < 1.0:
                    lgi = self._top_p(
                        lgi.unsqueeze(0), top_ps[i]).squeeze(0)
                if sup_t is not None:
                    lgi = lgi.index_fill(-1, sup_t, float('-inf'))
                if temps_cpu[i] == 0:
                    tok1 = int(lgi.argmax().item())
                else:
                    p1 = F.softmax(lgi, dim=-1).clamp(min=1e-10)
                    tok1 = int(torch.multinomial(
                        p1, num_samples=1,
                        generator=generators[i]).item())
                gen_only[i, _step] = tok1
                next_ids[i, 0] = tok1
                slot_start[i] = _step
                if gen_mirror is not None:
                    # New stream owns this slot from here — drop the
                    # previous stream's ids and record tok1 directly
                    # (the pending pipelined copy still holds the OLD
                    # row's token for this step → skip it for _step).
                    gen_mirror[i] = [int(tok1)]
                    _skip_mirror[i] = _step
                eos_floor[i] = _step + 1
                if rp_any:
                    seen[i, tok1] = True
                for l in range(cache.n_layers):
                    skc, svc = scratch.k_caches[l], scratch.v_caches[l]
                    kc, vc = cache.k_caches[l], cache.v_caches[l]
                    if kc is None or skc is None:
                        continue
                    rc = cache.ring_caps[l]
                    src_rc = (scratch.ring_caps[l]
                              if rc is not None else None)
                    kc[i].zero_()
                    vc[i].zero_()
                    if rc is not None:
                        # Real prompt tail: last min(Lj,pos,rc) real
                        # tokens → slots (pos-n..pos-1)%W. Stale slots
                        # keep dead KV but ring_valid excludes them. In
                        # the padded scratch the real prompt occupies
                        # seq indices [p0, Lmax); its last-W window
                        # starts at max(p0, Lmax-src_rc).
                        avail_lo = (max(p0, Lmax - src_rc)
                                    if src_rc is not None else 0)
                        n_real = min(Lj, pos, rc)
                        lp = torch.arange(
                            pos - n_real, pos, device=device)
                        src = (Lmax - n_real
                               + torch.arange(n_real, device=device))
                        src = src.clamp(min=avail_lo)
                        src_idx = (src % src_rc
                                   if src_rc is not None else src)
                        dst = lp % rc
                        # NO rebase: position_ids come from mask-cumsum
                        # — the refilled row's mask {base..pos} numbers
                        # its stream 0,1,2,... so K keeps R(src)
                        # verbatim.
                        kc[i].index_copy_(
                            1, dst, skc[r].index_select(1, src_idx))
                        vc[i].index_copy_(
                            1, dst, svc[r].index_select(1, src_idx))
                        rv = cache.ring_valid[l]
                        if rv is not None:
                            rv[i].zero_()
                            rv[i].index_fill_(0, dst, True)
                    else:
                        src = torch.arange(
                            Lmax - n_fill, Lmax, device=device)
                        kc[i, :, base:pos] = skc[r].index_select(1, src)
                        vc[i, :, base:pos] = svc[r].index_select(1, src)
                m_buf[i].zero_()
                m_buf[i, base:pos] = 1
                if tok1 in self.eos_set or (
                        stop_mask is not None
                        and bool(stop_mask[i, tok1])):
                    _finish(i, _step + 1)
                    continue  # first sampled token ends the stream
                active_cpu[i] = True
                finished_gpu[i] = False
                _dbg_rows.add(i)
            proc_rows.clear()
            proc_rows.extend(
                i for i in range(B) if processor_list[i] is not None)
            uniform_topk = (len(set(top_k_list)) == 1
                            and top_k_list[0] > 0)
            uniform_topp = (len(set(top_ps)) == 1 and top_ps[0] < 1.0)

        def _compact():
            """Shrink the decode batch onto a smaller captured graph.
            Once the pending queue is empty, finished slots still cost a
            full-width graph replay per step while emitting garbage —
            the dominant tail loss after the refill drain. Gathering the
            survivors into the next-smaller captured batch keeps the
            per-step cost proportional to live rows. B tracks LIVE rows;
            graph_B tracks the captured graph width (>= B)."""
            nonlocal B, graph_B, cache, m_buf, prefix, bbuckets
            nonlocal gen_only, next_ids, temp_t, temps_cpu
            nonlocal temperatures, top_ps, top_k_list
            nonlocal repetition_penalty_list, seed_list, generators
            nonlocal stop_list, processor_list, stop_mask
            nonlocal slot_prompt, slot_start, slot_cap, eos_floor
            nonlocal seen, rp_pen
            nonlocal active, active_cpu, finished_gpu, eos_mask_gpu
            nonlocal uniform_topk, uniform_topp, _dbg_rows
            nonlocal eos_bufs, mask_host, flag_host, eos_evs, eos_live
            nonlocal gen_mirror, tok_host, tok_live
            # Drain any unconsumed EOS flag BEFORE rows are remapped —
            # its mask indexes pre-compaction rows, so it must finish
            # them while slot_prompt still maps 1:1.
            if _pipe_eos:
                for _q in (0, 1):
                    _drain_eos(_q, steps_done)
            # Same for the token mirror's pipelined copy — it indexes
            # pre-compaction rows and must land before the gather.
            if tok_live:
                tok_ev.synchronize()
                col = tok_host.tolist()
                for i in range(B):
                    if _skip_mirror.get(i) != tok_copy_step:
                        gen_mirror[i].append(col[i])
                _skip_mirror.clear()
                tok_live = False
            idx_cpu = active_cpu.nonzero(
                as_tuple=False).squeeze(1).tolist()
            n_active = len(idx_cpu)
            pos = cache.position
            if n_active == 1:
                # Last row: the activation-time B=1 graphs (prefix "")
                # decode a lone stream at serial speed — cheaper than
                # replaying a padded batch-2 graph.
                ds1 = getattr(model, '_decode_state', None)
                if (ds1 is None or graph_B <= 1
                        or not any(f"kv{b}" in graphs
                                   for b in ds1['buckets']
                                   if b >= pos + 1)):
                    return False
                new_gB = 1
                bs = {'cache': ds1['cache'], 'mask': ds1['mask'],
                      'prefix': '', 'buckets': ds1['buckets']}
            else:
                new_gB = max(2, graph_B // 2)
                if (not (0 < n_active <= graph_B // 2)
                        or new_gB >= graph_B or new_gB < n_active):
                    return False
                bs = ensure(new_gB, pos,
                            min(pos + hard_cap - _step, 8192))
                if bs is None:
                    return False
            idx_t = torch.tensor(idx_cpu, dtype=torch.long,
                                 device=device)
            cn, mn = bs['cache'], bs['mask']
            for l in range(cn.n_layers):
                kn, vn = cn.k_caches[l], cn.v_caches[l]
                ko, vo = cache.k_caches[l], cache.v_caches[l]
                if kn is None or ko is None:
                    continue
                n_cp = min(pos, kn.shape[2])
                kn.zero_()
                vn.zero_()
                # Slice before gather — the old cache's full-seq
                # index_select would materialize GBs transiently. Live
                # rows land in [0, n_active); rows above stay dead.
                kn[:n_active, :, :n_cp].copy_(
                    ko[:, :, :n_cp].index_select(0, idx_t))
                vn[:n_active, :, :n_cp].copy_(
                    vo[:, :, :n_cp].index_select(0, idx_t))
                rvn, rvo = cn.ring_valid[l], cache.ring_valid[l]
                if rvn is not None and rvo is not None:
                    rvn[:n_active].copy_(rvo.index_select(0, idx_t))
            cn.position = pos
            mn.zero_()
            mn[:n_active, :pos] = m_buf.index_select(0, idx_t)[:, :pos]
            cache, m_buf = cn, mn
            prefix, bbuckets = bs['prefix'], bs['buckets']
            B = n_active
            graph_B = new_gB

            def g(x):
                return [x[i] for i in idx_cpu]
            temperatures, top_ps = g(temperatures), g(top_ps)
            top_k_list = g(top_k_list)
            repetition_penalty_list = g(repetition_penalty_list)
            seed_list, generators = g(seed_list), g(generators)
            stop_list, processor_list = g(stop_list), g(processor_list)
            slot_prompt = g(slot_prompt)
            slot_start, slot_cap = g(slot_start), g(slot_cap)
            eos_floor = g(eos_floor)
            temps_cpu = g(temps_cpu)
            gen_only = gen_only.index_select(0, idx_t)
            next_ids = next_ids.index_select(0, idx_t)
            temp_t = temp_t.index_select(0, idx_t)
            if gen_mirror is not None:
                gen_mirror = [gen_mirror[i] for i in idx_cpu]
                tok_host = torch.zeros(B, dtype=torch.long,
                                       pin_memory=True)
            if rp_any:
                seen = seen.index_select(0, idx_t)
                rp_pen = rp_pen.index_select(0, idx_t)
            if stop_mask is not None:
                stop_mask = stop_mask.index_select(0, idx_t)
            active_cpu = torch.ones(B, dtype=torch.bool)
            active = torch.ones(B, dtype=torch.bool, device=device)
            finished_gpu = torch.zeros(B, dtype=torch.bool,
                                       device=device)
            eos_mask_gpu = torch.zeros(B, dtype=torch.bool,
                                       device=device)
            if _pipe_eos:
                eos_bufs = [eos_mask_gpu,
                            torch.zeros(B, dtype=torch.bool,
                                        device=device)]
                mask_host = [torch.zeros(B, dtype=torch.bool,
                                         pin_memory=True)
                             for _ in range(2)]
                flag_host = [torch.zeros((), dtype=torch.bool,
                                         pin_memory=True)
                             for _ in range(2)]
                eos_evs = [torch.cuda.Event(), torch.cuda.Event()]
                eos_live = [False, False]
            _dbg_rows = {r for r, o in enumerate(idx_cpu)
                         if o in _dbg_rows}
            proc_rows.clear()
            proc_rows.extend(i for i in range(B)
                             if processor_list[i] is not None)
            uniform_topk = (len(set(top_k_list)) == 1
                            and top_k_list[0] > 0)
            uniform_topp = (len(set(top_ps)) == 1
                            and top_ps[0] < 1.0)
            if os.environ.get("FORGE_DBG_REFILL"):
                print(f"[compact] graph {graph_B} rows {idx_cpu} "
                      f"pos={pos}")
            return True

        # Incremental CPU mirror of gen_only for logits processors —
        # appends each step's [B] token column via a one-step-pipelined
        # pinned D2H instead of re-copying the whole [B, steps] matrix
        # every step (the old gen_only[:, :steps_done].tolist() was
        # O(B·steps²) host traffic per decode). Lazily allocated on the
        # first step that actually has a processor — the suppress_ids /
        # stop-string path never pays for it.
        gen_mirror = None
        tok_host = None
        tok_ev = None
        tok_live = False
        tok_copy_step = -1          # _step the pending tok_host copy holds
        _skip_mirror: dict = {}     # row -> refill step; skip only when it
        # matches the pending copy's step (a stale skip would drop a
        # legit new-stream token if the drain is deferred a step)

        for _step in range(hard_cap):
            if not active_cpu.any() and not pending:
                break

            # Sample all B tokens at once from batched logits [B, 1, vocab]
            next_logits_all = logits[:, -1, :]  # [B, vocab]

            # Repetition penalty BEFORE temperature (HF ordering; also
            # applies to the all-greedy path).
            if rp_any:
                pen = torch.where(seen, rp_pen,
                                  torch.ones((), device=device,
                                             dtype=next_logits_all.dtype))
                next_logits_all = next_logits_all / pen

            # k-subset sampling: with uniform top-k and no per-row
            # processors/mixed seeds, softmax + multinomial can run
            # over the k survivors instead of [B,V] — identical
            # distribution, ~3ms cheaper per step at V=131k.
            mixed_seed = any(
                (generators[i] is None) != (generators[0] is None)
                for i in range(B) if active_cpu[i])
            use_kset = (uniform_topk and not proc_rows
                        and not mixed_seed
                        and not all(t == 0 for t in temperatures))
            if uniform_topk and not rp_any:
                # top-k INDICES are invariant to positive temperature
                # scaling — select on raw logits, scale only the k
                # survivors. Saves the [B,V] division every step.
                k = min(top_k_list[0], V)
                top_v, top_i = next_logits_all.topk(k, dim=-1)
                top_v = top_v / temp_t
            else:
                # Temperature scaling [B, vocab]
                next_logits_all = next_logits_all / temp_t
            if uniform_topk:
                # Top-k (one batched topk) + optional top-p over the k
                # survivors — sort [B,k] instead of [B,V]. Softmax over
                # the k kept entries equals the full-width softmax with
                # -inf elsewhere, so results are identical.
                if rp_any:
                    k = min(top_k_list[0], V)
                    top_v, top_i = next_logits_all.topk(k, dim=-1)
                if uniform_topp:
                    sv, si = top_v.sort(dim=-1, descending=True)
                    probs = sv.softmax(-1)
                    keep = (probs.cumsum(-1) - probs) <= top_ps[0]
                    sv = sv.masked_fill(~keep, float('-inf'))
                    top_v = top_v.scatter(-1, si, sv)
                if not use_kset:
                    next_logits_all = torch.full_like(
                        next_logits_all, float('-inf')).scatter_(
                            -1, top_i, top_v)
            else:
                for i in range(B):
                    if top_k_list[i] > 0 and active_cpu[i]:
                        k = min(top_k_list[i], V)
                        thr = next_logits_all[i].topk(k).values[-1]
                        next_logits_all[i] = torch.where(
                            next_logits_all[i] < thr,
                            torch.full_like(
                                next_logits_all[i], float('-inf')),
                            next_logits_all[i])
                for i in range(B):
                    if top_ps[i] < 1.0 and active_cpu[i]:
                        next_logits_all[i] = self._top_p(
                            next_logits_all[i].unsqueeze(0),
                            top_ps[i]).squeeze(0)

            # Per-sequence logits processors (e.g. chat_loop's think-cap /
            # structural-token bans). Each gets generated-only ids —
            # same contract as the serial path's logits_processor.
            # gen_mirror[i] already contains exactly this stream's ids
            # (reset at each refill → equivalent to the old
            # gen_cpu[i][slot_start[i]:] slice).
            if any(processor_list[i] is not None and active_cpu[i]
                   for i in proc_rows):
                if gen_mirror is None:
                    # Mid-decode alloc (a processor row became active
                    # late): seed from gen_only so earlier tokens aren't
                    # missed — one-time copy, still O(1) per step after.
                    _gc = gen_only[:, :steps_done].tolist()
                    gen_mirror = [_gc[i][slot_start[i]:]
                                  for i in range(B)]
                    tok_host = torch.zeros(B, dtype=torch.long,
                                           pin_memory=True)
                    tok_ev = torch.cuda.Event()
                if tok_live:
                    # Consume last step's pipelined [B] token copy —
                    # almost always complete already (issued one step
                    # ago, overlapped with this step's forward).
                    tok_ev.synchronize()
                    col = tok_host.tolist()
                    for i in range(B):
                        if _skip_mirror.get(i) != tok_copy_step:
                            gen_mirror[i].append(col[i])
                    _skip_mirror.clear()
                    tok_live = False
                for i in proc_rows:
                    if active_cpu[i]:
                        next_logits_all[i] = processor_list[i](
                            next_logits_all[i].unsqueeze(0),
                            gen_mirror[i]).squeeze(0)

            if sup_t is not None and not use_kset:
                next_logits_all = next_logits_all.index_fill(
                    -1, sup_t, float("-inf"))

            # Mask finished sequences (skip when none have finished —
            # saves a [B,V] fill every step; the k-subset path masks
            # top_v instead)
            if had_inactive and not use_kset:
                next_logits_all[finished_gpu] = float('-inf')

            if os.environ.get("FORGE_DBG_STEPS") and _dbg_rows:
                for i in _dbg_rows:
                    if active_cpu[i]:
                        v, t_ = next_logits_all[i].topk(5)
                        print(f"[dbg] step {_step} row {i} top5 "
                              f"{t_.tolist()} {v.tolist()}")

            # Greedy or sampling
            if all(t == 0 for t in temperatures):
                next_tokens = next_logits_all.argmax(-1)  # [B]
            elif use_kset:
                # Sample over the k survivors: ban/mask via top_v,
                # softmax+multinomial on [B,k], gather token ids.
                if sup_t is not None:
                    top_v = top_v.masked_fill(
                        torch.isin(top_i, sup_t), float('-inf'))
                if had_inactive:
                    top_v = top_v.masked_fill(
                        finished_gpu.unsqueeze(1), float('-inf'))
                probs = top_v.softmax(-1).clamp(min=1e-10)
                probs = torch.nan_to_num(
                    probs, nan=1.0 / probs.shape[-1])
                sel = torch.multinomial(
                    probs, num_samples=1,
                    generator=batch_gen).squeeze(-1)
                next_tokens = top_i.gather(
                    1, sel.unsqueeze(1)).squeeze(1)
                if had_inactive:
                    next_tokens = torch.where(
                        ~finished_gpu, next_tokens,
                        torch.zeros_like(next_tokens))
                for i in range(B):
                    if temps_cpu[i] == 0 and active_cpu[i]:
                        next_tokens[i] = top_i[i, top_v[i].argmax()]
            else:
                probs = F.softmax(next_logits_all, dim=-1).clamp(min=1e-10)
                probs = torch.nan_to_num(probs, nan=1.0 / probs.shape[-1])
                # One batched multinomial when seeds are uniform (all
                # rows seeded → batch_gen; none seeded → default gen).
                # temp==0 rows get argmax overwrite after — cheaper
                # than per-row multinomial for the whole batch. Slow
                # path survives only for genuinely mixed seed configs.
                if not mixed_seed:
                    next_tokens = torch.multinomial(
                        probs, num_samples=1,
                        generator=batch_gen).squeeze(-1)
                    if had_inactive:
                        next_tokens = torch.where(
                            ~finished_gpu, next_tokens,
                            torch.zeros_like(next_tokens))
                    for i in range(B):
                        if temps_cpu[i] == 0 and active_cpu[i]:
                            next_tokens[i] = next_logits_all[i].argmax()
                else:
                    next_tokens = torch.zeros(B, dtype=torch.long, device=device)
                    for i in range(B):
                        if not active_cpu[i]:
                            next_tokens[i] = 0
                            continue
                        if temps_cpu[i] == 0:
                            next_tokens[i] = next_logits_all[i].argmax()
                        elif generators[i] is not None:
                            next_tokens[i] = torch.multinomial(
                                probs[i], num_samples=1,
                                generator=generators[i])[0]
                        else:
                            next_tokens[i] = torch.multinomial(
                                probs[i], num_samples=1)[0]

            # Check EOS + stop ids (all GPU ops; stop ids come from the
            # [B,V] mask gather — no per-step .tolist() round-trip).
            p = _step & 1
            cur = eos_bufs[p]
            cur.zero_()
            for eid in self.eos_set:
                cur |= (next_tokens == eid)
            if stop_mask is not None:
                cur |= stop_mask.gather(
                    1, next_tokens.unsqueeze(1)).squeeze(1)
            finished_gpu |= cur
            active = ~finished_gpu
            is_eos_cpu = None
            if _pipe_eos:
                # Async flag+mask D2H consumed NEXT step (see
                # _drain_eos): no host-side pipeline drain here.
                flag_host[p].copy_(cur.any(), non_blocking=True)
                mask_host[p].copy_(cur, non_blocking=True)
                eos_evs[p].record()
                eos_live[p] = True
            elif cur.any().item():
                # CPU fallback: synchronous check (no events/pinned).
                is_eos_cpu = cur.cpu().clone()

            # Append this step's tokens for every row (one batched
            # write) BEFORE _finish slices — the EOS token itself
            # belongs to the finished row's output.
            next_ids[:, 0] = next_tokens
            gen_only[:, _step] = next_tokens
            steps_done = _step + 1
            if gen_mirror is not None:
                # Pipeline this step's token column: the pinned D2H
                # overlaps next step's forward and is consumed by the
                # processor block one step later.
                tok_host.copy_(next_tokens, non_blocking=True)
                tok_ev.record()
                tok_live = True
                tok_copy_step = _step

            # Update CPU active state + record finished rows' outputs.
            # Pipelined: consume the PREVIOUS step's flag — its EOS'd
            # rows end at _step (token _step-1 inclusive), matching
            # the old synchronous _finish(i, _step+1) slice.
            if _pipe_eos:
                _drain_eos(1 - p, _step)
            elif is_eos_cpu is not None:
                had_inactive = True
                for i in range(B):
                    if bool(is_eos_cpu[i]) and active_cpu[i]:
                        active_cpu[i] = False
                        _finish(i, steps_done)

            # Per-slot token cap (independent of EOS): rows past their
            # own max_tokens finish without an EOS token.
            for i in range(B):
                if (active_cpu[i]
                        and _step + 1 - slot_start[i] >= slot_cap[i]):
                    active_cpu[i] = False
                    finished_gpu[i] = True
                    had_inactive = True
                    _finish(i, _step + 1)

            if rp_any:
                seen.scatter_(1, next_tokens.unsqueeze(1), True)

            # Refill freed slots from the pending queue — continuous
            # batching keeps the decode window full. Refills defer until
            # >= _REFILL_MIN slots are free (or the queue drains) so the
            # scratch prefill runs batched, amortizing its eager-launch
            # stall across several streams.
            if pending:
                free = [i for i in range(B) if not active_cpu[i]]
                take = min(len(free), len(pending), _REFILL_CAP)
                if take and (take >= _REFILL_MIN
                             or take == len(pending)):
                    _refill_batch(free[:take], pending[:take])
                    del pending[:take]

            # Queue drained → finished slots still cost a full-B replay
            # each step. Compact survivors onto a smaller captured
            # batch so the tail decodes at ~live-row cost.
            if (bstate is not None and not pending
                    and int(active_cpu.sum()) <= graph_B // 2):
                _compact()

            if not active_cpu.any() and not pending:
                break

            logits = _step_fn(next_ids)
            if logits is None:
                break  # static path exhausted (bucket/cache bound)

        for i in range(B):
            if active_cpu[i]:
                _finish(i, steps_done)
        for j in range(N):
            g = results[j]
            if g is None:  # never ran (horizon exhausted)
                g = torch.zeros(1, prompt_lens_all[j], dtype=torch.long,
                                device=device)
                g[0] = prompts[j][0]
                results[j] = g
        return results

    def _top_p(self, logits, top_p):
        from research.sampling_utils import top_p_filter_logits
        return top_p_filter_logits(logits, top_p)
