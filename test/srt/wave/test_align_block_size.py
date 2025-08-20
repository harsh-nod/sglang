import unittest

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sglang.srt.layers.moe.wave_ops.fused_moe import (
    moe_split_w1_wave,
)
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase
from wave_lang.kernel.lang import DataType

_is_hip = is_hip()
dtypes = [torch.float16, torch.bfloat16]

if _is_hip:
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

class TestMoeAlignBlockSize(CustomTestCase):
    NUM_EXPERTS = [4, 8, 64]
    TOP_KS = [2]

    def moe_align_block_size_pytorch(
        self,
        topk_ids: torch.Tensor,
        num_experts: int,
        block_size: int,
        sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
    ):
        """
        PyTorch implementation matching moe_align_block_size behavior.
        All output tensors are pre-allocated by the caller.

        Args:
            topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
            num_experts: Total number of experts
            block_size: Block size for expert processing
            sorted_ids: Pre-allocated output tensor for sorted token indices
            expert_ids: Pre-allocated output tensor for expert block assignments
            num_tokens_post_pad: Pre-allocated output tensor for total padded tokens
        """
        device = topk_ids.device
        num_tokens = topk_ids.numel()
        padding_value = num_tokens  # Value for padding tokens

        # Initialize output buffers
        sorted_ids.fill_(padding_value)
        num_tokens_post_pad.zero_()

        # Flatten the input and get expert counts
        flat_topk = topk_ids.view(-1).to(torch.int32)
        expert_counts = torch.bincount(flat_topk, minlength=num_experts)

        # Calculate padding needed per expert
        blocks_per_expert = (expert_counts + block_size - 1) // block_size
        padded_counts = blocks_per_expert * block_size
        total_size_with_padding = padded_counts.sum().item()
        num_tokens_post_pad.fill_(total_size_with_padding)

        # Calculate exclusive cumsum for expert offsets
        cumsum = torch.cumsum(padded_counts, dim=0) - padded_counts

        # Assign expert IDs to blocks
        expert_starts = torch.cumsum(padded_counts, dim=0) - padded_counts
        num_blocks = total_size_with_padding // block_size
        expert_ids[:num_blocks] = torch.repeat_interleave(
            torch.arange(num_experts, device=device), blocks_per_expert
        )

        if num_tokens == 0:
            return

        # Sort tokens by expert and fill valid positions

        # Get sorted order of tokens by expert: tokens are first sorted by the id
        # of their assigned expert, and if two tokens have the same expert id,
        # they'll be sorted by their original position in the flatten tensor.
        # I.e., first comes all indices of tokens assigned to expert 0, then all
        # indices of tokens assigned to expert 1, and so on.
        sorted_indices = torch.argsort(
            flat_topk * (num_tokens + 1) + torch.arange(num_tokens, device=device)
        )
        sorted_values = flat_topk[sorted_indices]

        # Calculate destination positions for each token
        token_positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        current_offsets = cumsum.to(torch.int64)

        # 2. Get per-expert offsets
        offsets = torch.cat(
            [torch.zeros(1, device=device), expert_counts.cumsum(0)[:-1]]
        ).long()

        # 3. Calculate local positions (0,1,2,... within each expert)
        local_positions = torch.arange(
            num_tokens, device=device
        ) - offsets.repeat_interleave(expert_counts)

        # 4. Calculate final positions
        token_positions = expert_starts[sorted_values] + local_positions

        # Scatter the original token indices
        original_indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
        sorted_ids[token_positions.long()] = original_indices[sorted_indices]

    def _test_case(self, num_tokens, block_size, num_experts, topk):
        device = "cuda"

        scores = torch.rand(num_tokens, num_experts, device=device)

        # Get topk expert indices for each token
        _, topk_ids = torch.topk(scores, k=topk, dim=1)

        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
        sorted_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
        )
        max_num_m_blocks = -(max_num_tokens_padded // -block_size)
        expert_ids = torch.empty(
            (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
        )
        num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

        # In EP, expert_ids for filtered experts are -1. We have num_experts + 1 ids in total.
        cumsum_buffer = torch.empty(
            (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
        )

        wave_sorted_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
        )
        wave_expert_ids = torch.empty(
            (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
        )
        wave_num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

        # Threshold based on benchmark results
        fuse_sorted_ids_padding = sorted_ids.shape[0] <= 4096
        if not fuse_sorted_ids_padding:
            sorted_ids.fill_(topk_ids.numel())
            wave_sorted_ids.fill_(topk_ids.numel())

        self.moe_align_block_size_pytorch(
            topk_ids, num_experts, block_size, wave_sorted_ids, wave_expert_ids, wave_num_tokens_post_pad
        )

        sgl_moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            fuse_sorted_ids_padding,
        )

        torch.equal(sorted_ids, wave_sorted_ids)
        torch.equal(expert_ids, wave_expert_ids)
        torch.equal(num_tokens_post_pad, wave_num_tokens_post_pad)

    def test_various_configurations(self):
        num_tokens_values = [1, 33, 64, 222]
        block_size_values = [16, 32, 64]

        # Calculate total number of tests
        total_tests = (
            len(num_tokens_values)
            * len(block_size_values)
            * len(self.NUM_EXPERTS)
            * len(self.TOP_KS)
        )

        torch.manual_seed(0)

        # Create progress bar
        with tqdm(total=total_tests, desc="Running Wave moe_align_block_size tests") as pbar:
            for num_tokens in num_tokens_values:
                for block_size in block_size_values:
                    for e in self.NUM_EXPERTS:
                        for topk in self.TOP_KS:
                            with self.subTest(
                                num_tokens=num_tokens,
                                block_size=block_size,
                                num_experts=e,
                                topk=topk,
                            ):
                                self._test_case(
                                    num_tokens,
                                    block_size,
                                    e,
                                    topk,
                                )
                                torch.cuda.empty_cache()
                            pbar.update(1)


if __name__ == "__main__":
    unittest.main()
