"""EXP-XTAL tests — grammar correctness (the foundation all measurements
sit on), probe sanity, and a CPU overfit smoke test of the training path."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forge.exp.xtal_grammar import (HOLDOUT_MEMBERS, ITEM0, MEMBERS, TAG_A,
                                    TAG_AH, TAG_B, TAG_C, VOCAB, XtalGrammar,
                                    class_of)


def member_of(tok: int) -> int:
    return (tok - ITEM0) % MEMBERS


def is_item(tok: int) -> bool:
    return ITEM0 <= tok < ITEM0 + 64


def test_agree_chains_obey_permutation():
    g = XtalGrammar(seed=0, device="cpu")
    toks, tags = g.gen_batch(8)
    B, T1 = toks.shape
    for b in range(B):
        for i in range(1, T1):
            if tags[b, i].item() == TAG_A:
                prev_cls = class_of(toks[b, i - 1].item())
                assert class_of(toks[b, i].item()) == g.perm_f[prev_cls]


def test_copy_positions_exact_repeat():
    g = XtalGrammar(seed=1, device="cpu")
    toks, tags = g.gen_batch(8)
    n_copy = 0
    for b in range(toks.shape[0]):
        for i in range(1, toks.shape[1]):
            if tags[b, i].item() == TAG_C:
                assert toks[b, i].item() == toks[b, i - 3].item()
                n_copy += 1
    assert n_copy > 0, "no copy positions generated — mixing broken?"


def test_ruleb_positions_obey_perm_g():
    g = XtalGrammar(seed=2, device="cpu")
    # B is rare (5%) — generate enough sequences to catch some
    found = 0
    for _ in range(20):
        toks, tags = g.gen_batch(8)
        for b in range(toks.shape[0]):
            for i in range(1, toks.shape[1]):
                if tags[b, i].item() == TAG_B:
                    x = toks[b, i - 2].item()
                    assert class_of(toks[b, i].item()) == g.perm_g[class_of(x)]
                    found += 1
    assert found > 0, "no ruleB positions in 160 sequences — mixing broken?"


def test_holdout_members_never_in_trigger_roles_during_training():
    """Redesign v2 invariant: holdout members MAY appear as chain-terminating
    successors (the class-revealing leak) but must NEVER appear in a trigger
    role — the single invariant capturing this: no TAG_A position is ever
    preceded by a holdout member."""
    g = XtalGrammar(seed=3, device="cpu")
    toks, tags = g.gen_batch(32)
    n_holdout_succ = 0
    for b in range(toks.shape[0]):
        for i in range(1, toks.shape[1]):
            if tags[b, i].item() == TAG_A:
                succ, trig = toks[b, i].item(), toks[b, i - 1].item()
                if is_item(trig):
                    assert member_of(trig) not in HOLDOUT_MEMBERS
                if is_item(succ) and member_of(succ) in HOLDOUT_MEMBERS:
                    n_holdout_succ += 1
    assert n_holdout_succ > 0, "controlled leak missing — no holdout successors"


def test_holdout_eval_batches_use_heldout_triggers():
    g = XtalGrammar(seed=4, device="cpu")
    toks, tags = g.gen_eval_batch("AH", batch_size=8)
    n_ah = 0
    for b in range(toks.shape[0]):
        for i in range(1, toks.shape[1]):
            if tags[b, i].item() == TAG_AH:
                trig = toks[b, i - 1].item()
                assert member_of(trig) in HOLDOUT_MEMBERS
                # successor itself obeys the class rule
                assert class_of(toks[b, i].item()) == g.perm_f[class_of(trig)]
                n_ah += 1
    assert n_ah > 0


def test_ruled_positions_are_class_members_of_x():
    """Rule-D (class-copy): target at TAG_D must be a member of the same
    class as x (the token 3 positions back, after marker_D)."""
    g = XtalGrammar(seed=9, device="cpu",
                    mix={"A": 0.2, "B": 0.05, "C": 0.2, "D": 0.4, "N": 0.15})
    found = 0
    for _ in range(10):
        toks, tags = g.gen_batch(8)
        for b in range(toks.shape[0]):
            for i in range(1, toks.shape[1]):
                if tags[b, i].item() == 5:  # TAG_D
                    x = toks[b, i - 3].item()
                    assert class_of(toks[b, i].item()) == class_of(x)
                    found += 1
    assert found > 0, "no ruleD positions generated"


def test_ruler_positions_are_modular_class_sums():
    """Rule-R (relational): target class must equal (class(x1)+class(x2))
    mod n_classes — x1,x2 at fixed offsets in the chunk."""
    g = XtalGrammar(seed=10, device="cpu",
                    mix={"A": 0.2, "B": 0.05, "C": 0.2, "R": 0.4, "N": 0.15})
    found = 0
    for _ in range(10):
        toks, tags = g.gen_batch(8)
        for b in range(toks.shape[0]):
            for i in range(1, toks.shape[1]):
                if tags[b, i].item() == 6:  # TAG_R
                    x1 = toks[b, i - 2].item()
                    x2 = toks[b, i - 1].item()
                    expect = (class_of(x1) + class_of(x2)) % g.n_classes
                    assert class_of(toks[b, i].item()) == expect
                    found += 1
    assert found > 0, "no ruleR positions generated"


def test_ruler_holdout_eval_uses_holdout_members():
    g = XtalGrammar(seed=11, device="cpu")
    toks, tags = g.gen_eval_batch("RH", batch_size=8)
    n_holdout_x = 0
    for b in range(toks.shape[0]):
        for i in range(1, toks.shape[1]):
            if tags[b, i].item() == 6:
                x1, x2 = toks[b, i - 2].item(), toks[b, i - 1].item()
                if member_of(x1) in HOLDOUT_MEMBERS or member_of(x2) in HOLDOUT_MEMBERS:
                    n_holdout_x += 1
    assert n_holdout_x > 0


def test_eval_batches_deterministic():
    g = XtalGrammar(seed=5, device="cpu")
    t1, _ = g.gen_eval_batch("A", batch_size=4)
    t2, _ = g.gen_eval_batch("A", batch_size=4)
    assert torch.equal(t1, t2)


def test_batch_shapes_and_vocab_range():
    g = XtalGrammar(seed=6, device="cpu")
    toks, tags = g.gen_batch(4)
    assert toks.shape == (4, g.seq_len + 1)
    assert tags.shape == toks.shape
    assert toks.dtype == torch.long
    assert int(toks.min()) >= 0 and int(toks.max()) < VOCAB


def test_probe_machinery_cpu():
    from forge.exp.xtal_probe import (batch_loss, embedding_geometry,
                                      eval_fixed_batches, flat_grad,
                                      grad_alignment)
    from scripts.exp_xtal import build_model

    class Args:
        d_model, n_layers, n_heads, device = 64, 2, 4, "cpu"
    model = build_model(Args())
    g = XtalGrammar(seed=7, device="cpu")
    eval_sets = {k: g.gen_eval_batch(k, batch_size=4) for k in ("A", "C", "N")}
    res = eval_fixed_batches(model, eval_sets)
    assert "A/ruleA" in res and "C/ruleC" in res
    ga = grad_alignment(model, eval_sets)
    for k, v in ga.items():
        if k.startswith("align/"):
            assert -1.0001 <= v <= 1.0001
    eg = embedding_geometry(model)
    assert "emb/class_gap" in eg
    # flat_grad nonzero on a real batch
    toks, tags = g.gen_batch(4)
    loss = batch_loss(model, toks, tags)
    gv = flat_grad(model, loss)
    assert gv.norm() > 0


def test_overfit_single_batch_cpu():
    """The whole training path must be able to crush one fixed batch —
    the minimal sanity gate for any training harness."""
    from forge.exp.xtal_probe import batch_loss
    from scripts.exp_xtal import build_model

    class Args:
        d_model, n_layers, n_heads, device = 64, 2, 4, "cpu"
    torch.manual_seed(0)
    model = build_model(Args())
    g = XtalGrammar(seed=8, device="cpu")
    toks, tags = g.gen_batch(8)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = last = None
    for step in range(150):
        loss = batch_loss(model, toks, tags)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
        last = float(loss)
    assert last < 0.6 * first, f"overfit failed: {first:.3f} -> {last:.3f}"
