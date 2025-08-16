import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.wave_ops.fused_moe import (
    moe_split_w1_wave,
)
from wave_lang.kernel.lang import DataType

num_experts = [8, 64]
top_ks = [2]
m_values = [1, 33]
n_values = [128, 1024]
k_values = [511]
dtypes = [torch.float16, torch.bfloat16]

torch.manual_seed(0)


@torch.compile(  # Requires PyTorch 2.0+
    fullgraph=False,  # Allow dynamic control flow
    mode="max-autotune",  # Aggressive optimizations
)
def torch_ref_moe_split_w1(a, w1_gate, w1_up, w2, score, topk):
    m, k = a.shape
    a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)  # [m * topk, k]
    out = torch.zeros(m * topk, w2.shape[1], dtype=a.dtype, device=a.device)  # [m * topk, k]
    score = torch.softmax(score, dim=-1, dtype=torch.float32)  # [m, e]
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)  # [m * topk]
    topk_ids = topk_ids.view(-1)  # [m * topk]
    for i in range(w1_gate.shape[0]):
        mask = (
            topk_ids == i
        )  # num_selected (which of the m * topk tokens selected this expert)
        if mask.sum():
            # Split into gate and up projections
            gate = a[mask] @ w1_gate[i].transpose(0, 1) # [num_selected, n]
            up = a[mask] @ w1_up[i].transpose(0, 1)  # [num_selected, n]
            lhs = torch.zeros(m, w2.shape[-1], dtype=a.dtype, device=a.device)
            lhs = F.silu(gate) * up
            out[mask] = lhs @ w2[i].transpose(0, 1)  # [num_selected, k]
    return (
        out.view(m, -1, w2.shape[1]) * topk_weight.view(m, -1, 1).to(out.dtype)
    ).sum(
        dim=1
    )  # [m, k]


@pytest.mark.parametrize("m", m_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("e", num_experts)
@pytest.mark.parametrize("topk", top_ks)
@pytest.mark.parametrize("dtype", dtypes)
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: DataType,
):
    """
    Tests the fused_moe function using Pytest parameterization.
    """
    device = "cuda"
    rtol, atol = 1e-1, 1e-2

    if dtype == torch.float16 and k == 1024:
        pytest.skip("This combination generates NaNs and INFs")

    # TODO: investigate why using torch.randn would have precision issue in silu computation
    a = torch.rand((m, k), dtype=dtype, device=device)
    w1 = torch.rand((e, 2 * n, k), dtype=dtype, device=device)
    w2 = torch.rand((e, k, n), dtype=dtype, device=device)
    score = torch.rand((m, e), dtype=dtype, device=device)

    # TODO: remove manual splitting
    # We need to manually split w1 into 2 halves, since this is
    # required by `silu_and_mul` kernel, and currently we can't
    # do this in Wave.
    w1_gate = w1[:, :n, :]  # First half for gate
    w1_up = w1[:, n:, :]  # Second half for up projection

    # Make sure the algorithm with w1 splitting works in PyTorch.
    ref_output = torch_ref_moe_split_w1(a, w1_gate, w1_up, w2, score, topk)

    # The implementation in Wave should also work.
    tkw_output = moe_split_w1_wave(a, w1_gate, w1_up, w2, score, topk)
    torch.testing.assert_close(tkw_output, ref_output, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
