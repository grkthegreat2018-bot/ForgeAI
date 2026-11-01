"""zero_centered_norm (d2): RMSNorm stored as gamma-1, forward adds 1 back.

Identical function by construction; the point is weight decay in hone
pulls toward IDENTITY scale instead of zeroing channel gains. Verifies:
  - zc RMSNorm output equals standard RMSNorm given gamma vs gamma-1
  - _warm_start_zero_center shifts stored weights -1 for zc modules only
  - a checkpoint written at gamma ports losslessly into a zc model
  - _scan_qk_norm_identity treats w==0 as identity under zero-centering
"""
import torch

from forge.config import get_config
from forge.model.layers import RMSNorm, GroupedQueryAttention
from forge.model.loader import ModelLoader
from forge.model_loader import ConfigurableResearchLLM


def test_rmsnorm_zero_centered_equivalent():
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    std = RMSNorm(64).eval()
    with torch.no_grad():
        std.weight.mul_(3.0)            # non-identity gamma
    zc = RMSNorm(64, zero_centered=True).eval()
    with torch.no_grad():
        zc.weight.copy_(std.weight - 1.0)   # store gamma-1
    assert torch.equal(std(x), zc(x))


def test_rmsnorm_zc_init_is_identity():
    zc = RMSNorm(64, zero_centered=True).eval()
    x = torch.randn(2, 64)
    ref = torch.nn.functional.rms_norm(x, [64], torch.ones(64), 1e-6)
    assert torch.equal(zc(x), ref)


def test_warm_start_shifts_only_zc_norm_keys():
    model = ConfigurableResearchLLM(
        get_config("forgelm_tiny", device="cpu",
                   zero_centered_norm=True, use_qk_norm=True))
    zc_keys = {f"{n}.weight" for n, mod in model.named_modules()
               if type(mod) is RMSNorm and mod._zero_centered}
    assert zc_keys
    state = {k: v.clone() for k, v in model.state_dict().items()}
    # simulate a standard-gamma checkpoint: every norm weight = gamma
    for k in zc_keys:
        state[k] = torch.ones_like(state[k])
    shifted = ModelLoader._warm_start_zero_center(dict(state), model)
    for k in shifted:
        if k in zc_keys:
            assert torch.equal(shifted[k], torch.zeros_like(shifted[k]))
        else:
            assert torch.equal(shifted[k], state[k])


def test_qk_norm_identity_scan_zero_centered():
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        d_model=64, n_heads=4, n_kv_heads=2, use_qk_norm=True,
        zero_centered_norm=True)
    cfg = get_config("forgelm_tiny", device="cpu", use_qk_norm=True)

    import types
    m = types.SimpleNamespace(
        blocks=[types.SimpleNamespace(attn=attn)])
    ModelLoader._scan_qk_norm_identity(m, cfg)
    # zero-centered init (w=0) must register as identity, not as a
    # non-identity norm that would engage the qk-norm fast path.
    assert attn._qk_norm_identity is True
