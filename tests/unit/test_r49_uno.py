"""R49 Uno decoder tests: Psi-Spec lossless block decoding (arXiv:2609.04010)."""

import torch

from forge.decoding.uno import NgramProposer, UnoDecoding

VOCAB = 32
D = 16


class TinyAR(torch.nn.Module):
    """Deterministic per-position AR model with tuple KV cache."""

    def __init__(self, vocab=VOCAB, d=D):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab, bias=False)

    def forward(self, ids, use_cache=False, past_key_value=None, **kw):
        x = self.emb(ids)
        logits = self.head(x) * 50.0
        past = (x.detach(), x.detach()) if use_cache else None
        return logits, None, past


def _make(seed=0, prompt_len=5):
    torch.manual_seed(seed)
    model = TinyAR()
    prompt = torch.randint(0, VOCAB, (1, prompt_len))
    return model, prompt


def _reference(model, prompt, n):
    ids = list(prompt[0].tolist())
    for _ in range(n):
        logits = model.head(model.emb(torch.tensor([ids[-1]])))
        ids.append(int(logits.argmax()))
    return ids


class TestNgramProposer:
    def test_shapes(self):
        p = NgramProposer(n=3)
        p.update([1, 2, 3, 4, 1, 2, 3, 5])
        draft, lp = p.propose([1, 2], 4)
        assert draft is not None
        assert draft.dtype == torch.long
        assert draft.shape == lp.shape
        assert 1 <= draft.shape[0] <= 4

    def test_unknown_context_returns_none(self):
        p = NgramProposer(n=3)
        assert p.propose([9, 9, 9], 4) == (None, None)


class TestUnoLossless:
    def test_fallback_matches_ar(self):
        model, prompt = _make()
        ref = _reference(model, prompt, 12)
        dec = UnoDecoding(proposer=None)
        out = dec.generate(model, prompt, max_new_tokens=12)
        assert out[0].tolist() == ref

    def test_corrupt_drafts_still_lossless(self):
        model, prompt = _make()
        ref = _reference(model, prompt, 12)

        def bad_proposer(ids, k):
            return torch.randint(0, VOCAB, (k,)), None

        dec = UnoDecoding(proposer=bad_proposer, block_size=4)
        out = dec.generate(model, prompt, max_new_tokens=12)
        assert out[0].tolist() == ref

    def test_good_drafts_accepted(self):
        model, prompt = _make()
        ref = _reference(model, prompt, 12)

        def oracle_proposer(ids, k):
            draft, cur = [], list(ids)
            for _ in range(k):
                logits = model.head(model.emb(torch.tensor([cur[-1]])))
                draft.append(int(logits.argmax()))
                cur.append(draft[-1])
            return torch.tensor(draft, dtype=torch.long), None

        dec = UnoDecoding(proposer=oracle_proposer, block_size=4)
        out = dec.generate(model, prompt, max_new_tokens=12)
        assert out[0].tolist() == ref

    def test_entropy_stop_disables_drafting(self):
        model, prompt = _make()

        def oracle_proposer(ids, k):
            draft, cur = [], list(ids)
            for _ in range(k):
                logits = model.head(model.emb(torch.tensor([cur[-1]])))
                draft.append(int(logits.argmax()))
                cur.append(draft[-1])
            return torch.tensor(draft, dtype=torch.long), None

        dec = UnoDecoding(proposer=oracle_proposer, block_size=4, entropy_stop=0.5)
        out = dec.generate(model, prompt, max_new_tokens=10)
        assert out.shape[1] == prompt.shape[1] + 10
        assert dec._drafting_enabled is False

    def test_output_length_exact(self):
        model, prompt = _make()

        def bad_proposer(ids, k):
            return torch.randint(0, VOCAB, (k,)), None

        dec = UnoDecoding(proposer=bad_proposer, block_size=3)
        out = dec.generate(model, prompt, max_new_tokens=9)
        assert out.shape == (1, prompt.shape[1] + 9)
