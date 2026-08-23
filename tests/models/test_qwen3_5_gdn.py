"""Wire up the GDN correctness oracle that ``gdn_reference`` was written to be.

``models/qwen3_5_moe/gdn_reference.py`` is a verbatim port of HF's
``torch_recurrent_gated_delta_rule`` and calls itself the oracle for the kernel-backed
op -- but nothing imported it, so the vendored fla path had never been checked against
it. Ornith-1.5-35B is 30 GDN layers out of 40, and a wrong-but-same-magnitude GDN is
exactly the failure that produces fluent-looking token soup with a numerically healthy
residual stream.

These tests hit the fla kernels directly (no engine context, no checkpoint), so they run
anywhere there's a GPU:

    .venv/bin/python -m pytest tests/models/test_qwen3_5_gdn.py -v -s

The GQA test is the pointed one. Ornith runs 16 key heads against 32 value heads, so
every k/v head pair shares a key head, and there are two incompatible ways to say that:

    repeat_interleave  ->  v-head j reads k-head j // 2      (HF, and the reference)
    repeat/tile        ->  v-head j reads k-head j % 16

Both preserve magnitudes, so nothing downstream notices the difference -- the model just
attends to the wrong history. ``test_prefill_gqa_convention`` reports which one the
kernel implements instead of only asserting, so a failure names the fix.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN kernels are CUDA-only")

# Ornith-1.5-35B-A3B geometry (qwen35moe.ssm.*): 16 key heads, 32 value heads, 128/128.
NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 16, 32, 128, 128
SCALE = HEAD_K_DIM**-0.5


def _rand_inputs(total: int, seed: int = 0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    q = torch.randn(1, total, NUM_K_HEADS, HEAD_K_DIM, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, total, NUM_K_HEADS, HEAD_K_DIM, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, total, NUM_V_HEADS, HEAD_V_DIM, device=dev, dtype=torch.bfloat16)
    # g is a log-decay: <= 0, and near 0 so the recurrence actually retains history
    # (a strongly-forgetting state would hide a wrong head pairing).
    g = -torch.rand(1, total, NUM_V_HEADS, device=dev, dtype=torch.float32) * 0.1
    beta = torch.rand(1, total, NUM_V_HEADS, device=dev, dtype=torch.float32)
    return q, k, v, g, beta


def _expand(t: torch.Tensor, how: str) -> torch.Tensor:
    """[1, T, num_k_heads, D] -> [1, T, num_v_heads, D] under one GQA convention."""
    rep = NUM_V_HEADS // NUM_K_HEADS
    if how == "repeat_interleave":
        return t.repeat_interleave(rep, dim=2)
    if how == "repeat":
        return t.repeat(1, 1, rep, 1)
    raise ValueError(how)


def _reference(q, k, v, g, beta, how="repeat_interleave", initial_state=None):
    from freetoken.models.qwen3_5_moe.gdn_reference import recurrent_gated_delta_rule

    out, state = recurrent_gated_delta_rule(
        _expand(q, how), _expand(k, how), v, g, beta,
        initial_state=initial_state, use_qk_l2norm=True,
    )
    return out[0].float(), state


def _kernel_prefill(q, k, v, g, beta, state_source=None):
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

    total = q.shape[1]
    dev = q.device
    if state_source is None:
        state_source = torch.zeros(
            1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device=dev, dtype=torch.float32
        )
    out = gdn_prefill_chunk_fla(
        q, k, v, g, beta,
        state_source=state_source,
        indices=torch.tensor([0], device=dev, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, total], device=dev, dtype=torch.int64),
        scale=SCALE,
    )
    return out.float(), state_source


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


# bf16 inputs through a long fp32 recurrence: agreement to ~1% is the honest bar. A wrong
# head pairing or a dropped l2norm lands at O(1), not near this.
TOL = 2e-2


@pytest.mark.parametrize("total", [16, 64, 200])
def test_prefill_matches_reference(total):
    """The chunked fla prefill kernel must reproduce the HF recurrence."""
    q, k, v, g, beta = _rand_inputs(total)
    got, _ = _kernel_prefill(q, k, v, g, beta)
    ref, _ = _reference(q, k, v, g, beta)
    err = _rel(got, ref)
    print(f"\n  prefill total={total:<4d} max_rel_err={err:.4e}")
    assert err < TOL, f"fla prefill diverges from the HF reference: {err:.3e}"


def test_prefill_gqa_convention():
    """Report which key-head mapping the kernel implements, and require HF's."""
    q, k, v, g, beta = _rand_inputs(128, seed=3)
    got, _ = _kernel_prefill(q, k, v, g, beta)
    errs = {how: _rel(got, _reference(q, k, v, g, beta, how=how)[0])
            for how in ("repeat_interleave", "repeat")}
    best = min(errs, key=errs.get)
    print("\n  GQA convention match (lower = what the kernel does):")
    for how, err in errs.items():
        print(f"    {how:<18s} max_rel_err={err:.4e}{'   <-- kernel matches this' if how == best else ''}")
    assert errs["repeat_interleave"] < TOL, (
        f"kernel does not implement HF's repeat_interleave GQA (err {errs['repeat_interleave']:.3e}); "
        f"closest convention was {best!r}"
    )


def test_prefill_final_state_matches_reference():
    """The state written back to ``state_source`` seeds every later decode step, so a
    correct output with a wrong final state still corrupts generation."""
    q, k, v, g, beta = _rand_inputs(96, seed=7)
    _, state = _kernel_prefill(q, k, v, g, beta)
    _, ref_state = _reference(q, k, v, g, beta)
    err = _rel(state[0].float(), ref_state[0].float())
    print(f"\n  prefill final state max_rel_err={err:.4e}")
    assert err < TOL, f"fla prefill final state diverges: {err:.3e}"


def test_decode_matches_reference():
    """One fused decode step against the reference, from a non-zero prior state --
    this is the path every generated token after the first goes through."""
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

    dev = torch.device("cuda")
    torch.manual_seed(11)
    q, k, v, _, _ = _rand_inputs(1, seed=11)
    a = torch.randn(1, NUM_V_HEADS, device=dev, dtype=torch.float32)
    b = torch.randn(1, NUM_V_HEADS, device=dev, dtype=torch.float32)
    A_log = torch.randn(NUM_V_HEADS, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(NUM_V_HEADS, device=dev, dtype=torch.float32)
    prior = torch.randn(
        1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, device=dev, dtype=torch.float32
    ) * 0.1

    state_source = prior.clone()
    got = gdn_decode_fla(
        q, k, v, a, b, A_log=A_log, dt_bias=dt_bias,
        state_source=state_source,
        indices=torch.tensor([0], device=dev, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 1], device=dev, dtype=torch.int64),
        scale=SCALE,
    ).float()

    # The kernel folds the gating in; the reference takes g/beta directly.
    g = (-A_log.exp() * torch.nn.functional.softplus(a + dt_bias)).reshape(1, 1, NUM_V_HEADS)
    beta = b.sigmoid().reshape(1, 1, NUM_V_HEADS)
    ref, _ = _reference(q, k, v, g, beta, initial_state=prior)
    err = _rel(got, ref)
    print(f"\n  decode max_rel_err={err:.4e}")
    assert err < TOL, f"fla decode diverges from the HF reference: {err:.3e}"
