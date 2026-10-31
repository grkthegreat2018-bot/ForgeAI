"""Batched decoding strategy — processes multiple requests together.

Shifts matmul from GEMV (memory-bound at batch=1) to GEMM (compute-bound
at batch>1), delivering 3-5x throughput on RTX 5070 for 1.2B models.

Supports per-sequence generation settings:
  - temperature, top_p, top_k (sampling)
  - repetition_penalty
  - seed (per-sequence torch.Generator for reproducible generation)
  - stop sequences (string-based, checked after decode)
"""
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
    ) -> list[torch.Tensor]:
        B = len(prompts)
        if B == 0:
            return []
        device = prompts[0].device

        # Default per-sequence settings
        if top_k_list is None:
            top_k_list = [0] * B
        if repetition_penalty_list is None:
            repetition_penalty_list = [1.0] * B
        if seed_list is None:
            seed_list = [None] * B
        if stop_list is None:
            stop_list = [None] * B
        if processor_list is None:
            processor_list = [None] * B

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
        max_prompt_len = max(p.shape[1] for p in prompts)
        padded_ids = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
        prompt_lens = []
        for i, p in enumerate(prompts):
            L = p.shape[1]
            padded_ids[i, -L:] = p[0]  # right-align prompt (pad on left)
            prompt_lens.append(L)

        # Attention mask — attend to the rightmost L positions (the prompt)
        attn_mask = torch.zeros(B, max_prompt_len, dtype=torch.bool, device=device)
        for i, L in enumerate(prompt_lens):
            attn_mask[i, -L:] = True

        # Output rows are reassembled at the end from a shared
        # [B, max_tokens] buffer (gen_only below) — one column write per
        # step replaces the old per-row in-place appends, which cost B
        # python-indexed GPU writes every step.
        max_tokens = max(max_tokens_list)

        # Static-KV graph path: lazily captures per-(batch, KV-bucket)
        # decode graphs at activation request time — one replay produces
        # B tokens, so the batch shares weight reads AND launch overhead.
        # Any failure falls through to the functional loop.
        bstate = None
        ensure = getattr(model, '_ensure_static_batch', None)
        if ensure is not None:
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
                bufs_map[tag]['input_ids'].copy_(nids)
                with torch.inference_mode():
                    graphs[tag].replay()
                cache.position += 1
                lg, _ = unpack_output_with_kv(bufs_map[tag]['output'])
                return lg
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

        # Temperatures are loop-invariant — build the divisor once.
        temp_t = torch.tensor(
            temps_cpu, device=device, dtype=logits.dtype).unsqueeze(1)

        # Generated tokens land in one [B, max_tokens] buffer: a single
        # column write per step replaces B python-indexed GPU writes.
        # end_step[i] = inclusive stop index +1 for rows that finished.
        gen_only = torch.zeros(B, max_tokens, dtype=torch.long,
                               device=device)
        end_step = [max_tokens] * B
        steps_done = 0
        uniform_topk = len(set(top_k_list)) == 1 and top_k_list[0] > 0
        uniform_topp = len(set(top_ps)) == 1 and top_ps[0] < 1.0

        for _step in range(max_tokens):
            if not active_cpu.any():
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

            # Temperature scaling [B, vocab]
            next_logits_all = next_logits_all / temp_t

            if uniform_topk:
                # Top-k (one batched topk) + optional top-p over the k
                # survivors — sort [B,k] instead of [B,V]. Softmax over
                # the k kept entries equals the full-width softmax with
                # -inf elsewhere, so results are identical.
                k = min(top_k_list[0], V)
                top_v, top_i = next_logits_all.topk(k, dim=-1)
                if uniform_topp:
                    sv, si = top_v.sort(dim=-1, descending=True)
                    probs = sv.softmax(-1)
                    keep = (probs.cumsum(-1) - probs) <= top_ps[0]
                    sv = sv.masked_fill(~keep, float('-inf'))
                    top_v = top_v.scatter(-1, si, sv)
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
            for i in range(B):
                if processor_list[i] is not None and active_cpu[i]:
                    row_gen = gen_only[i, :min(end_step[i], steps_done)]
                    next_logits_all[i] = processor_list[i](
                        next_logits_all[i].unsqueeze(0),
                        row_gen.tolist()).squeeze(0)

            # Mask finished sequences (skip when none have finished —
            # saves a [B,V] fill every step)
            if had_inactive:
                next_logits_all[finished_gpu] = float('-inf')

            # Greedy or sampling
            if all(t == 0 for t in temperatures):
                next_tokens = next_logits_all.argmax(-1)  # [B]
            else:
                probs = F.softmax(next_logits_all, dim=-1).clamp(min=1e-10)
                probs = torch.nan_to_num(probs, nan=1.0 / probs.shape[-1])
                # One batched multinomial when seeds are uniform (all
                # rows seeded → batch_gen; none seeded → default gen).
                # temp==0 rows get argmax overwrite after — cheaper
                # than per-row multinomial for the whole batch. Slow
                # path survives only for genuinely mixed seed configs.
                mixed_seed = any(
                    (generators[i] is None) != (generators[0] is None)
                    for i in range(B) if active_cpu[i])
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
            for eid in self.eos_set:
                eos_mask_gpu |= (next_tokens == eid)
            if stop_mask is not None:
                eos_mask_gpu |= stop_mask.gather(
                    1, next_tokens.unsqueeze(1)).squeeze(1)
            # Single scalar sync — only transfer full tensor when EOS
            # occurs. .cpu() is a no-op returning the SAME tensor when
            # already on CPU — clone() so the zero_() can't wipe it.
            if eos_mask_gpu.any().item():
                is_eos_cpu = eos_mask_gpu.cpu().clone()
            else:
                is_eos_cpu = None
            finished_gpu |= eos_mask_gpu
            active = ~finished_gpu
            eos_mask_gpu.zero_()

            # Update CPU active state + mark finished rows' end index
            if is_eos_cpu is not None:
                had_inactive = True
                for i in range(B):
                    if bool(is_eos_cpu[i]):
                        active_cpu[i] = False
                        if end_step[i] == max_tokens:
                            end_step[i] = _step + 1

            # Append this step's tokens for every row (one batched
            # write); finished rows' later columns are sliced off by
            # end_step when the outputs are reassembled.
            next_ids[:, 0] = next_tokens
            gen_only[:, _step] = next_tokens
            steps_done = _step + 1

            if rp_any:
                seen.scatter_(1, next_tokens.unsqueeze(1), True)

            if not active_cpu.any():
                break

            logits = _step_fn(next_ids)
            if logits is None:
                break  # static path exhausted (bucket/cache bound)

        out = []
        for i in range(B):
            n_gen = min(end_step[i], steps_done)
            g = torch.zeros(1, prompt_lens[i] + n_gen,
                            dtype=torch.long, device=device)
            g[0, :prompt_lens[i]] = padded_ids[i, -prompt_lens[i]:]
            if n_gen:
                g[0, prompt_lens[i]:] = gen_only[i, :n_gen]
            out.append(g)
        return out

    def _top_p(self, logits, top_p):
        from research.sampling_utils import top_p_filter_logits
        return top_p_filter_logits(logits, top_p)
