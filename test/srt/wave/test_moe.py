import unittest

import torch
import torch.nn.functional as F
from tqdm import tqdm
from wave_lang.kernel.lang import DataType

from sglang.srt.layers.moe.wave_ops.fused_moe import moe_split_w1_wave
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

_is_hip = is_hip()
dtypes = [torch.float16, torch.bfloat16]


class TestWaveFusedMOE(CustomTestCase):
    NUM_EXPERTS = [8, 64]
    TOP_KS = [2, 6]

    def get_tolerance(self, dtype):
        """Get tolerance values for different data types

        Args:
            dtype: Data type

        Returns:
            tuple: (relative tolerance, absolute tolerance)
        """
        if dtype == torch.float32:
            return 1e-3, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-1, 1e-2
        else:
            return 1e-2, 1e-2  # Default values for other types

    @torch.compile(  # Requires PyTorch 2.0+
        fullgraph=False,  # Allow dynamic control flow
        mode="max-autotune",  # Aggressive optimizations
    )
    def torch_moe_split_w1(self, a, w1_gate, w1_up, w2, score, topk):
        m, k = a.shape
        a = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)  # [m * topk, k]
        out = torch.zeros(
            m * topk, w2.shape[1], dtype=a.dtype, device=a.device
        )  # [m * topk, k]
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
                gate = a[mask] @ w1_gate[i].transpose(0, 1)  # [num_selected, n]
                up = a[mask] @ w1_up[i].transpose(0, 1)  # [num_selected, n]
                lhs = torch.zeros(m, w2.shape[-1], dtype=a.dtype, device=a.device)
                lhs = F.silu(gate) * up
                out[mask] = lhs @ w2[i].transpose(0, 1)  # [num_selected, k]
        return (
            out.view(m, -1, w2.shape[1]) * topk_weight.view(m, -1, 1).to(out.dtype)
        ).sum(
            dim=1
        )  # [m, k]

    def _test_case(self, m, n, k, e, topk, dtype):
        device = "cuda"
        rtol, atol = self.get_tolerance(dtype)

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
        ref_output = self.torch_moe_split_w1(a, w1_gate, w1_up, w2, score, topk)

        # The implementation in Wave should also work.
        tkw_output = moe_split_w1_wave(a, w1_gate, w1_up, w2, score, topk)
        torch.testing.assert_close(tkw_output, ref_output, rtol=rtol, atol=atol)

    def test_various_configurations(self):
        m_values = [1, 33, 64, 222]
        n_values = [128, 1024]
        k_values = [256, 511]
        dtypes = [torch.float16, torch.bfloat16]

        # Calculate total number of tests
        total_tests = (
            len(m_values)
            * len(n_values)
            * len(k_values)
            * len(self.NUM_EXPERTS)
            * len(self.TOP_KS)
            * len(dtypes)
        )

        torch.manual_seed(0)

        # Create progress bar
        with tqdm(total=total_tests, desc="Running Wave MoE tests") as pbar:
            for m in m_values:
                for n in n_values:
                    for k in k_values:
                        for e in self.NUM_EXPERTS:
                            for topk in self.TOP_KS:
                                for dtype in dtypes:
                                    with self.subTest(
                                        m=m,
                                        n=n,
                                        k=k,
                                        e=e,
                                        topk=topk,
                                        dtype=dtype,
                                    ):
                                        self._test_case(
                                            m,
                                            n,
                                            k,
                                            e,
                                            topk,
                                            dtype,
                                        )
                                        torch.cuda.empty_cache()
                                    pbar.update(1)


if __name__ == "__main__":
    unittest.main()
