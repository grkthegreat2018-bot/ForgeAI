"""CopySpan edit-mode decoding driver (V14-RSI, docs/V14_RSI_DESIGN.md §3a).

When the LM emits the <copy> op token, the pointer head emits (start, end)
over the source span's prefill-cached hiddens; the driver splices
source[start:end+1] verbatim into the output and commits <copy> + span with ONE
chunked T>1 forward — preserving the committed conv/SSM left context per
the R60-4 invariant (same mechanism as chunked prefill / spec verify).
The committed-stream convention is `<copy>` followed by the verbatim span
tokens, matching the SFT format; the <copy> op is NOT counted in the
returned output ids.

Lossless fallback: a model that never emits <copy> decodes exactly like
standard autoregressive generation. batch=1 only (RSI proposer path is
single-sequence or uses generate_batch for parallel candidates).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def generate_with_copy_spans(
    model,
    input_ids: torch.Tensor,
    *,
    src_span: tuple[int, int],
    copy_token_id: int,
    max_new_tokens: int = 512,
    max_copy_len: int = 4096,
    eos_token_id: int | None = None,
    greedy: bool = True,
    temperature: float = 1.0,
    on_copy=None,
) -> torch.Tensor:
    """Generate tokens; <copy> ops splice source spans at ~O(1)/span cost.

    Args:
        model: ConfigurableResearchLLM with .edit_head (use_edit_head).
        input_ids: (1, T) prompt ids.
        src_span: (lo, hi) token index range in the prompt that pointer
            positions refer to (exclusive hi). Pointer index i maps to
            prompt token input_ids[0, lo + i].
        copy_token_id: vocab id of the <copy> op token.
        max_new_tokens: cap on EMITTED output tokens (copied tokens count
            once each against the cap — they are real output tokens).
        max_copy_len: clamp on a single copy span length.
        eos_token_id: stop token (optional).
        greedy: argmax decode; else temperature sampling.
        on_copy: optional callback(start, end, n_copied) per copy op.

    Returns:
        (1, N) generated token ids (excluding prompt, excluding <copy> ops).
    """
    assert model.edit_head is not None, "model built without use_edit_head"
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "batch=1 only"
    lo, hi = src_span
    assert 0 <= lo < hi <= input_ids.size(1), "src_span out of range"

    device = input_ids.device
    # Prefill: full hidden for src key cache, last-position logits.
    out = model(
        input_ids, use_cache=True, return_hidden=True, logits_last_only=True)
    logits, _, past, hidden = out  # (1,1,V), past, (1,T,d)
    src_h = hidden[0, lo:hi]                              # (S, d)
    src_k = model.edit_head.key_cache(src_h)              # (S, ptr)
    h_t = hidden[:, -1:, :]                               # (1,1,d)

    generated: list[int] = []
    while len(generated) < max_new_tokens:
        step_logits = logits[:, -1, :]                    # (1, V)
        if greedy:
            tok = int(step_logits.argmax(-1).item())
        else:
            probs = torch.softmax(step_logits / max(temperature, 1e-5), -1)
            tok = int(torch.multinomial(probs, 1).item())

        if tok == copy_token_id:
            # Pointer decode: start argmax, end argmax constrained >= start.
            start_l, end_l = model.edit_head(h_t, src_k)  # (1,S) each
            start = int(start_l.argmax(-1).item())
            end_l = end_l.clone()
            end_l[:, :start] = float("-inf")
            end = int(end_l.argmax(-1).item())            # inclusive
            end = min(end, start + max_copy_len - 1, hi - lo - 1)
            copied = input_ids[0, lo + start: lo + end + 1]
            if on_copy is not None:
                on_copy(start, end, int(copied.numel()))
            generated.extend(copied.tolist())
            # Commit <copy> + the whole span in ONE chunked forward —
            # <copy> IS part of the committed stream (the SFT convention:
            # op token, then verbatim span). Conv/SSM/GDN2 states advance
            # exactly as sequential decode would (R60-4 invariant).
            chunk = torch.cat([
                torch.tensor([copy_token_id], device=device,
                             dtype=input_ids.dtype),
                copied.to(device)])
            out = model(
                chunk.unsqueeze(0),
                past_key_values=past, use_cache=True,
                return_hidden=True, logits_last_only=True)
            logits, _, past, hidden = out
            h_t = hidden[:, -1:, :]
        else:
            if eos_token_id is not None and tok == eos_token_id:
                break
            generated.append(tok)
            out = model(
                torch.tensor([[tok]], device=device, dtype=input_ids.dtype),
                past_key_values=past, use_cache=True,
                return_hidden=True, logits_last_only=True)
            logits, _, past, hidden = out
            h_t = hidden[:, -1:, :]
            if len(generated) >= max_new_tokens:
                break

    return torch.tensor([generated], device=device, dtype=input_ids.dtype)
