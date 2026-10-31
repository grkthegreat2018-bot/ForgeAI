"""Tests for R&D Round 29: MinimalLoRA knowledge injection invariants.

Covers the guarantees the R29 experiment relies on:
  1. Zero-init LoRA forward is a bit-exact no-op vs the parent Linear.
  2. bypass() context reproduces the exact parent-only forward (trained state).
  3. Zero-delta merge is bit-exact (weight and forward).
  4. Trained-delta merge reproduces the adapter forward within fp32 GEMM
     reassociation tolerance, and delta_W is exactly scale * B @ A.
  5. fresh_trio() restores the pristine parent FFN trio after a contaminated
     merge (the parent-contamination guard for sequential conditions).
  6. Trainable param count is rank * (in + out) per module (minimal growth).
  7. ce_loss shifts logits/labels by one position (off-by-one regression guard
     for the batched training path).
"""
import sys

sys.path.insert(0, r"D:\windsurf\ForgeAI")
sys.path.insert(0, r"D:\windsurf\ForgeAI\scripts")

import test_r29_injection as r29
import torch
import torch.nn.functional as F
from torch import nn


def _parent():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


def _x():
    torch.manual_seed(1)
    return torch.randn(4, 64)


class _Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(48, 24)
                self.up_proj = nn.Linear(48, 24)
                self.down_proj = nn.Linear(24, 48)
        class L(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = M()
        self.layers = nn.ModuleList([L()])


class _MockHF(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Trunk()

    @property
    def device(self):
        return next(self.parameters()).device


def test_zero_init_noop_bit_exact():
    parent = _parent()
    lora = r29.MinimalLoRA(parent, rank=4, scale=2.0)
    x = _x()
    assert torch.equal(parent(x), lora(x))


def test_bypass_exact_parent_forward():
    parent = _parent()
    lora = r29.MinimalLoRA(parent, rank=4, scale=2.0)
    with torch.no_grad():
        lora.lora_A.normal_(0, 0.1)
        lora.lora_B.normal_(0, 0.1)
    x = _x()
    with lora.bypass():
        assert torch.equal(parent(x), lora(x))
    # sanity: without bypass the trained delta must actually change the output
    assert not torch.equal(parent(x), lora(x))


def test_zero_delta_merge_bit_exact():
    parent = _parent()
    lora = r29.MinimalLoRA(parent, rank=4, scale=2.0)
    merged = lora.merge_into_base()
    x = _x()
    assert torch.equal(merged.weight.data, parent.weight.data)
    assert torch.equal(merged(x), lora(x))
    assert torch.equal(merged(x), parent(x))


def test_trained_delta_merge_tolerance():
    parent = _parent()
    lora = r29.MinimalLoRA(parent, rank=4, scale=2.0)
    with torch.no_grad():
        lora.lora_A.normal_(0, 0.05)
        lora.lora_B.normal_(0, 0.05)
    merged = lora.merge_into_base()
    x = _x()
    # delta_W reconstruction is exact
    assert torch.equal(lora.delta_W(), (lora.lora_B @ lora.lora_A) * lora.scale)
    # merged weight = W0 + delta exactly
    assert torch.equal(merged.weight.data, parent.weight.data + lora.delta_W())
    # merged forward == adapter forward up to fp32 GEMM reassociation
    assert torch.allclose(merged(x), lora(x), atol=1e-5, rtol=1e-5)


def test_param_count_minimal():
    parent = _parent()
    rank = 4
    lora = r29.MinimalLoRA(parent, rank=rank, scale=2.0)
    assert lora.trainable_params() == rank * (64 + 32)
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert set(trainable) == {"lora_A", "lora_B"}


def test_fresh_trio_restores_pristine_parent():
    hf = _MockHF()
    torch.manual_seed(2)
    xm = torch.randn(4, 48)
    orig_layer_idx = r29.LAYER_IDX
    r29.LAYER_IDX = 0
    r29.ORIG.clear()
    try:
        r29.capture_original(hf)
        mlp = hf.model.layers[0].mlp
        orig_g = mlp.gate_proj(xm).clone()

        # Condition 1: install, contaminate, merge into parent
        loras1 = r29.fresh_trio(hf, rank=2)
        with torch.no_grad():
            for lm in loras1:
                lm.lora_A.normal_(0, 0.1)
                lm.lora_B.normal_(0, 0.1)
        r29.merge_trio(hf)
        assert isinstance(mlp.gate_proj, nn.Linear)
        contaminated = mlp.gate_proj(xm)
        assert not torch.allclose(contaminated, orig_g, atol=1e-4)

        # Condition 2: fresh_trio must restore the pristine parent exactly
        loras2 = r29.fresh_trio(hf, rank=4)
        assert len(loras2) == 3
        assert all(isinstance(lm, r29.MinimalLoRA) for lm in loras2)
        restored = mlp.gate_proj(xm)
        assert torch.equal(restored, orig_g), "parent contamination leaked"
        # and the fresh LoRAs are exact no-ops
        assert torch.equal(mlp.gate_proj(xm), orig_g)
    finally:
        r29.LAYER_IDX = orig_layer_idx
        r29.ORIG.clear()


def test_ce_loss_shifts_by_one_position():
    """ce_loss must align logits[i] with labels[i+1] (off-by-one guard)."""
    V = 10

    class Stub(nn.Module):
        """Returns logits where position i predicts token (i+1)%V perfectly."""
        def __init__(self):
            super().__init__()
            self.eye = nn.Parameter(torch.eye(V), requires_grad=False)

        def forward(self, ids, attention_mask=None):
            nxt = ids.roll(-1, dims=1)  # [t1,t2,t3, t0]
            logits = self.eye[nxt].unsqueeze(0) * 20.0
            return type("Out", (), {"logits": logits})()

    stub = Stub()
    ids = torch.tensor([[1, 2, 3]])
    attn = torch.ones_like(ids)
    labels = ids.clone()
    loss = r29.ce_loss(stub, ids, attn, labels)
    # correct shift: predictions are perfect -> near-zero loss
    assert loss.item() < 1e-4, f"ce_loss misaligned: {loss.item()}"
    # sanity: if we DIDN'T shift (align logits[i] with labels[i]), the same
    # stub would give a large loss — proves the test can detect the bug
    logits = stub(ids, attn).logits
    unshifted = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
    assert unshifted.item() > 1.0
