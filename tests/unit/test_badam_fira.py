"""CPU tests for BAdam and FiraNLRQ optimizers (V7-8B training stack)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from forge.training.optim.badam import BAdam


class TinyLayer(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return self.fc(x)


class TinyModel(nn.Module):
    """Mimics ForgeAI layout: blocks.N containers + embeddings/head."""

    def __init__(self, d=8, n_layers=3):
        super().__init__()
        self.embed = nn.Embedding(16, d)
        self.blocks = nn.ModuleList([TinyLayer(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, 16)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _loss(model):
    ids = torch.randint(0, 16, (2, 4))
    return model(ids).float().pow(2).mean()


def test_badam_partition_covers_all_params():
    model = TinyModel()
    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    covered = {id(p) for b in opt._blocks for p in b["params"]}
    all_params = {id(p) for p in model.parameters()}
    assert covered == all_params, "every param must belong to exactly one block"
    assert opt._n_blocks >= 4  # 3 layers + embeddings_head (may be chunked further)


def test_badam_chunks_oversized_head_block():
    """embeddings_head must not stay one giant block (fp32 optimizer spike).

    Note: chunking splits by parameter boundaries — a single monolithic
    param (e.g. one huge embedding) can't be split. Real models have many
    medium tensors in the head block, which is what this tests.
    """
    model = TinyModel(d=8, n_layers=2)
    # many small extras → embeddings_head >> layer blocks
    model.extras = nn.ModuleList([nn.Linear(8, 8) for _ in range(50)])
    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    layer_sizes = [sum(p.numel() for p in b["params"])
                   for b in opt._blocks if b["name"].startswith("blocks.")]
    head_sizes = [sum(p.numel() for p in b["params"])
                  for b in opt._blocks if not b["name"].startswith("blocks.")]
    target = sorted(layer_sizes)[len(layer_sizes) // 2]
    assert max(head_sizes) <= target * 3, (
        f"head chunks {head_sizes} vs layer target {target} — chunking failed")


def test_badam_descending_order():
    model = TinyModel()
    opt = BAdam(model, lr=1e-3, switch_every=1, switch_mode="descending",
                verbose=False)
    assert opt._blocks[0]["name"] == "blocks.2"  # output layer first


def test_badam_only_active_block_has_grads():
    model = TinyModel()
    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    _loss(model).backward()
    opt.step()
    opt.zero_grad()
    for i, block in enumerate(opt._blocks):
        for p in block["params"]:
            assert p.requires_grad == (i == opt._block_idx)
            assert p.grad is None  # zero_grad clears ALL blocks (incl. frozen)


def test_badam_state_dict_roundtrip_restores_schedule():
    model = TinyModel()
    opt = BAdam(model, lr=1e-3, switch_every=100, verbose=False)
    for _ in range(7):
        opt.zero_grad()
        _loss(model).backward()
        opt.step()
    saved_idx, saved_steps = opt._block_idx, opt._steps_in_block
    state = opt.state_dict()

    # fresh optimizer on the SAME model, restore schedule
    opt2 = BAdam(model, lr=1e-3, switch_every=100, verbose=False)
    opt2.load_state_dict(state)
    assert opt2._block_idx == saved_idx
    # _activate_block resets this; load_state_dict must restore it AFTER
    assert opt2._steps_in_block == saved_steps


def test_fira_nlrq_steps_on_nlrq_model():
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    from forge.training.optim.fira_nlrq import FiraNLRQ

    model = nn.Sequential(NLRQLinear(8, 8, rank=4), nn.Linear(8, 4))
    opt = FiraNLRQ(model, lr=1e-3, verbose=False)
    x = torch.randn(2, 8)
    loss = model(x).float().pow(2).mean()
    loss.backward()
    s_before = model[0].S.detach().clone()
    opt.step()
    assert not torch.equal(s_before, model[0].S.detach())  # S got updated
    # U_q/V_q stay int8 buffers (not trainable)
    assert model[0].U_q.dtype == torch.int8
    assert not any(p.requires_grad for p in [model[0].U_q.float()])


def test_fira_nlrq_reduces_loss():
    torch.manual_seed(0)
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    from forge.training.optim.fira_nlrq import FiraNLRQ

    layer = NLRQLinear(16, 16, rank=8)
    opt = FiraNLRQ(layer, lr=1e-2, verbose=False)
    x = torch.randn(4, 16)
    target = torch.randn(4, 16)
    first = last = None
    for _ in range(50):
        opt.zero_grad()
        loss = (layer(x) - target).float().pow(2).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < first, f"loss should decrease: {first} → {last}"


def test_badam_fp32_states_and_no_decay():
    model = TinyModel()
    opt = BAdam(model, lr=1e-3, weight_decay=0.5, switch_every=100, verbose=False)
    _loss(model).backward()
    # Weight (2D) in active block gets decayed; bias (1D) must not be.
    block = opt._blocks[opt._block_idx]
    w = next(p for p in block["params"] if p.ndim == 2)
    b = [p for p in block["params"] if p.ndim == 1 and p.dim() != 0
         and p.numel() == w.shape[0]]
    assert w.requires_grad
    w_before = w.detach().clone()
    if b:
        b_before = b[0].detach().clone()
    opt.step()
    # decay: p *= (1 - lr*wd) plus adam update; both change w.
    # bias: only adam update, no decay — verify via state dtype instead:
    st = opt.state[w]
    assert st["exp_avg"].dtype == torch.float32
    assert st["exp_avg_sq"].dtype == torch.float32
    if b:
        assert opt.state[b[0]]["exp_avg"].dtype == torch.float32


def test_badam_no_decay_actually_skips_1d():
    torch.manual_seed(11)  # deterministic grads (assumes decay dominates adam)
    model = TinyModel()
    opt = BAdam(model, lr=1e-1, weight_decay=1.0, switch_every=100, verbose=False)
    _loss(model).backward()
    # embed block: contains only 2D embedding weight; find a block with a bias.
    # Activate block 0 (embeddings_head) then a layer block with bias.
    opt._activate_block(opt._blocks.index(
        next(b for b in opt._blocks if any(p.ndim == 1 for p in b["params"]))))
    _loss(model).backward()
    bias = next(p for b in opt._blocks for p in b["params"]
                if p.ndim == 1 and p.requires_grad)
    before = bias.detach().clone()
    opt.step()
    after = bias.detach()
    # With wd=1.0, lr=0.1 → decay factor 0.9 would dominate any adam update.
    ratio = (after / before).abs().mean().item()
    assert ratio > 0.95, f"1D param decayed: ratio={ratio}"


def test_badam_slim_state_roundtrip():
    model = TinyModel()
    opt = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    for _ in range(5):
        opt.zero_grad()
        _loss(model).backward()
        opt.step()
    from forge.training.runners.train_8b_all import slim_badam_state
    slim = slim_badam_state(opt)
    keep_blocks = {opt._block_idx, (opt._block_idx - 1) % opt._n_blocks}
    keep_ids = {id(p) for i in keep_blocks for p in opt._blocks[i]["params"]}
    keep_pos = {i for i, p in enumerate(opt.param_groups[0]["params"])
                if id(p) in keep_ids}
    saved_pos = {int(k) for k in slim["optimizer_state"]["state"]}
    assert saved_pos <= keep_pos, "slim state must only cover kept blocks"
    assert saved_pos, "slim state must include the just-updated block"
    # param_groups stay FULL-size so torch load validation passes
    assert len(slim["optimizer_state"]["param_groups"][0]["params"]) == \
        len(opt.param_groups[0]["params"])

    opt2 = BAdam(model, lr=1e-3, switch_every=1, verbose=False)
    opt2.load_state_dict(slim)
    assert opt2._block_idx == opt._block_idx
    assert opt2._steps_in_block == opt._steps_in_block


# ── NLRQ factor training (STE) ───────────────────────────────────────────

def _tiny_nlrq():
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    torch.manual_seed(3)
    return NLRQLinear(12, 10, rank=4)


def test_nlrq_ste_forward_matches_quantized():
    layer = _tiny_nlrq()
    x = torch.randn(3, 12)
    out_ref = layer(x).clone()
    layer.enable_factor_training_()
    # STE path starts exactly at the quantized solution
    out_ste = layer(x)
    assert torch.allclose(out_ref, out_ste, atol=1e-5)


def test_nlrq_ste_grads_reach_masters():
    layer = _tiny_nlrq()
    layer.enable_factor_training_()
    assert layer.U_m.requires_grad and layer.V_m.requires_grad
    x = torch.randn(3, 12)
    layer(x).sum().backward()
    assert layer.U_m.grad is not None and layer.U_m.grad.abs().sum() > 0
    assert layer.V_m.grad is not None and layer.V_m.grad.abs().sum() > 0


def test_nlrq_export_roundtrip():
    layer = _tiny_nlrq()
    layer.enable_factor_training_()
    with torch.no_grad():
        layer.U_m += 0.01  # perturb masters
    layer.export_quantized_()
    # export is idempotent: quantize(dequant(buffers)) == buffers
    assert torch.equal(layer.U_q,
                       torch.round(layer.U_m.float().cpu()
                                   / layer.U_scale.float()).clamp(-127, 127)
                       .to(torch.int8))
    # disabling drops masters but keeps progress in buffers
    layer.disable_factor_training_(export=False)
    assert layer.U_m is None


def test_nlrq_state_dict_strips_masters_via_snapshot():
    from forge.training.runners.train_8b_all import snapshot_state
    model = nn.Sequential(_tiny_nlrq(), nn.Linear(10, 4))
    model[0].enable_factor_training_()
    state = snapshot_state(model, step=7)
    assert not any(k.endswith((".U_m", ".V_m")) for k in state)
    assert any(k.endswith("U_q") for k in state)
    assert state["step"] == 7


# ── trainer utilities ────────────────────────────────────────────────────

def test_lr_schedules():
    from forge.training.runners.train_8b_all import lr_at
    assert lr_at(0, 1.0, 10, 100) == 0.0
    assert abs(lr_at(5, 1.0, 10, 100) - 0.5) < 1e-9
    assert lr_at(10, 1.0, 10, 100) == 1.0
    assert lr_at(100, 1.0, 10, 100) == 0.0
    # wsd: stable until decay_frac, then linear to zero
    assert lr_at(50, 1.0, 10, 100, "wsd", 0.3) == 1.0
    assert lr_at(73, 1.0, 10, 100, "wsd", 0.3) == 1.0   # decay starts at prog=0.7
    assert lr_at(100, 1.0, 10, 100, "wsd", 0.3) == 0.0
    # cosine: midpoint of decay → half lr
    assert abs(lr_at(55, 1.0, 10, 100, "cosine") - 0.5) < 1e-9


def test_chunked_ce_matches_full_ce():
    from forge.training.runners.train_8b_all import chunked_next_token_ce
    torch.manual_seed(1)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(50, 16)
            self.head = nn.Linear(16, 50)

        def forward(self, ids, return_hidden=False):
            h = self.embed(ids)
            if return_hidden:
                return None, None, h
            return self.head(h)

    m = M()
    ids = torch.randint(0, 50, (2, 13))
    with torch.no_grad():
        logits = m(ids)
        shift = logits[:, :-1, :].contiguous()
        tgt = ids[:, 1:].contiguous()
        ref = torch.nn.functional.cross_entropy(
            shift.view(-1, 50), tgt.view(-1))
    got = chunked_next_token_ce(m, ids, chunk=5)
    assert torch.allclose(ref, got, atol=1e-5), f"{ref} vs {got}"


def test_epoch_sampler_uniform_covers_all():
    from forge.training.runners.train_8b_all import EpochBatchSampler, PackedDataset
    ds = [PackedDataset("a", torch.zeros(10, 4, dtype=torch.long),
                        torch.zeros(1, 4, dtype=torch.long)),
          PackedDataset("b", torch.zeros(6, 4, dtype=torch.long),
                        torch.zeros(1, 4, dtype=torch.long))]
    s = EpochBatchSampler(ds, batch_size=4, weights=None, seed=0)
    seen = []
    for _ in range(4):  # 4 batches × 4 = 16 = exactly one epoch
        seen += s.next().tolist()
    assert sorted(seen) == list(range(16))


def test_epoch_sampler_stratified_upweights():
    from forge.training.runners.train_8b_all import EpochBatchSampler, PackedDataset
    ds = [PackedDataset("v7", torch.zeros(10, 4, dtype=torch.long),
                        torch.zeros(1, 4, dtype=torch.long)),
          PackedDataset("lfm", torch.zeros(30, 4, dtype=torch.long),
                        torch.zeros(1, 4, dtype=torch.long))]
    s = EpochBatchSampler(ds, batch_size=8, weights={"v7": 3.0}, seed=0)
    seen = []
    for _ in range(5):  # 5 batches × 8 = exactly one 40-seq epoch
        seen += s.next().tolist()
    # share ∝ weight × size: v7 3*10 vs lfm 1*30 → 50/50 of the 40-seq epoch
    n_v7 = sum(i < 10 for i in seen)
    assert n_v7 == 20 and len(seen) == 40, (n_v7, len(seen))


def test_freeze_dead_params_finds_unused_module():
    from forge.training.runners.train_8b_all import freeze_dead_params_

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(20, 8)
            self.body = nn.Linear(8, 8)
            self.head = nn.Linear(8, 20)
            self.unused = nn.Linear(8, 8)  # never called in forward

        def forward(self, ids, return_hidden=False):
            h = self.body(self.embed(ids))
            if return_hidden:
                return None, None, h
            return self.head(h)

    m = M()
    n_dead = freeze_dead_params_(m, torch.device("cpu"), use_flce=False)
    assert n_dead == 2  # unused.weight + unused.bias
    assert not m.unused.weight.requires_grad
    assert m.embed.weight.requires_grad
    assert m.body.weight.requires_grad

    # no-dead-block crash: a BAdam over only-dead params is now impossible
    # because dead params are excluded from the partition
    opt = BAdam(m, lr=1e-3, switch_every=1, verbose=False)
    covered = {id(p) for b in opt._blocks for p in b["params"]}
    assert id(m.unused.weight) not in covered


def test_normalize_logit_scale_fixes_confidently_wrong_init():
    import math as _math
    from types import SimpleNamespace

    import torch.nn.functional as F

    from forge.training.runners.train_8b_all import forward_model, normalize_logit_scale_

    torch.manual_seed(5)

    class M(nn.Module):
        def __init__(self, V=64, d=16):
            super().__init__()
            self.embed = nn.Embedding(V, d)
            self.body = nn.Linear(d, d)
            self.head = nn.Linear(d, V)

        def forward(self, ids):
            return self.head(self.body(self.embed(ids)))

    m = M()
    with torch.no_grad():  # blow up the head: confidently-wrong init
        m.head.weight.mul_(14.0)
    cfg = SimpleNamespace(vocab_size=64)
    ids = torch.randint(0, 64, (2, 9))

    with torch.no_grad():
        logits = forward_model(m, ids)
        ce_before = F.cross_entropy(
            logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1)).item()
    scale = normalize_logit_scale_(m, torch.device("cpu"), cfg)
    with torch.no_grad():
        logits = forward_model(m, ids)
        ce_after = F.cross_entropy(
            logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1)).item()
    assert ce_before > _math.log(64) + 4, f"setup failed: {ce_before}"
    # std=1 logits ⇒ CE ≈ ln(V) + ½ (Gaussian entropy of the logit spread)
    assert _math.log(64) < ce_after < _math.log(64) + 0.8, f"CE {ce_after}"
    assert 0.02 < scale < 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"All {len(fns)} tests passed.")
