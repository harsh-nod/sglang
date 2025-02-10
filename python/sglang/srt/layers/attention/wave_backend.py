from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import logging

from sglang.srt.layers.attention import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo


class WaveAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.wave_ops.extend_attention import (
            extend_attention_wave,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )
        super().__init__()
        self.decode_attention_fwd = decode_attention_fwd
        self.extend_attention_fwd = extend_attention_wave  # Wave
        self.extend_attention_triton_fwd = extend_attention_fwd  # Wave

        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )

        self.num_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.forward_metadata = None

        self.cuda_graph_max_seq_len = model_runner.model_config.context_len

        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for wave attention backend."""

        if forward_batch.forward_mode.is_decode():
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_head,
                    self.num_kv_splits,
                    self.v_head_dim + 1,
                ),
                dtype=torch.float32,
                device=self.device,
            )

            max_extend_len = None
        else:
            attn_logits = None
            # attn_logits_max = None
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()

        # self.forward_metadata = attn_logits, attn_logits_max, max_extend_len
        self.forward_metadata = attn_logits, max_extend_len

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_max_total_num_tokens = max_bs * self.cuda_graph_max_seq_len

        self.cuda_graph_start_loc = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )

        # self.cuda_graph_attn_logits = torch.empty(
        #     (self.num_kv_splits, max_bs, self.num_head, self.v_head_dim),
        #     dtype=torch.float32,
        #     device="cuda",
        # )
        self.cuda_graph_attn_logits = torch.empty(
            (max_bs, self.num_head, self.num_kv_splits, self.v_head_dim + 1),
            dtype=torch.float32,
            device="cuda",
        )

        self.cuda_graph_attn_logits_max = torch.empty(
            (self.num_kv_splits, max_bs, self.num_head),
            dtype=torch.float32,
            device="cuda",
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        assert encoder_lens is None, "Not supported"
        assert forward_mode.is_decode(), "Not supported"
        assert spec_info is None, "Not supported"

        self.forward_metadata = (
            self.cuda_graph_attn_logits,
            self.cuda_graph_attn_logits_max,
            # None,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        # NOTE: encoder_lens expected to be zeros or None
        self.cuda_graph_start_loc.zero_()
        self.cuda_graph_start_loc[1:bs] = torch.cumsum(seq_lens[: bs - 1], dim=0)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # Wave forward_extend
    # """
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
            o_triton = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
            o_triton = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
        _, max_extend_len = self.forward_metadata


        computed_max_ext_seq_len = torch.max(forward_batch.extend_seq_lens)
        if computed_max_ext_seq_len != max_extend_len:
            logging.info(f"shape of Q: {q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).shape}")
            logging.info(f"original ext seq len: {forward_batch.extend_seq_lens}")
            logging.info(f"original seq lens: {forward_batch.seq_lens}")
            logging.info("\n\n")
            assert len(forward_batch.extend_seq_lens) == 1
            forward_batch.extend_seq_lens[0] = max_extend_len
            forward_batch.seq_lens = max_extend_len

        # print("forward_batch.seq_lens", forward_batch.seq_lens)
        # print("forward_batch.extend_seq_lens", forward_batch.extend_seq_lens)
        # logging.info(f"Scaling:{layer.scaling}")
        # logging.info(f"Logit cap:{layer.logit_cap}")
        # tmp_logit_cap = 0
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            # TODO: Add additional parameters for logit_cap and scaling.
            layer_scaling=layer.scaling,
            logit_cap=layer.logit_cap,
        )

        run_triton = False
        if run_triton:
            _, max_extend_len = self.forward_metadata
            self.extend_attention_triton_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k.contiguous(),
                v.contiguous(),
                o_triton.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.extend_seq_lens,
                forward_batch.extend_start_loc,
                max_extend_len,
                layer.scaling,
                logit_cap=tmp_logit_cap,
            )
            if not torch.allclose(o_triton, o, atol=1e-2):
                import threading
                save_lock = threading.Lock()
                with save_lock:
                    dir = '/home/amd/jacky/jacky/sglang/python/client_artifacts/'
                    torch.save(q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),dir+"qextend_client.pt")
                    torch.save(k.contiguous(), dir+"kextend_client.pt")
                    torch.save(v.contiguous(), dir+"vextend_client.pt")
                    torch.save(o_triton.view(-1, layer.tp_q_head_num, layer.v_head_dim), dir+"otriton_client.pt")
                    torch.save(forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id), dir+"kbuffer_client.pt")
                    torch.save(forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id), dir+"vbuffer_client.pt")
                    torch.save(forward_batch.req_to_token_pool.req_to_token, dir+"req_to_token_client.pt")
                    torch.save(forward_batch.req_pool_indices, dir+"req_pool_indices_client.pt")
                    torch.save(forward_batch.seq_lens, dir+"seq_lens_client.pt")
                    torch.save(forward_batch.extend_seq_lens, dir+"extend_seq_lens_client.pt")
                    torch.save(forward_batch.extend_start_loc, dir+"extend_start_loc_client.pt")
                    torch.save(max_extend_len, dir+"max_extend_len_client.pt")
                    torch.save(layer.scaling, dir+"sm_scale_client.pt")
                    torch.save(tmp_logit_cap, dir+"tmp_logit_cap_client.pt")
                    max_error = torch.max(torch.abs(o_triton - o))
                    logging.info(f"Max error:{max_error}")
                    logging.info(f"O_triton:{o_triton}")
                    logging.info(f"O:{o}")
                raise ValueError(f"Not good. max error:{torch.max(torch.abs(o_triton - o))}")
        return o
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        attn_logits, _ = self.forward_metadata

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            attn_logits,
            #attn_logits_max,   # wave-attn only
            self.num_kv_splits,
            layer.scaling,
            layer.logit_cap,
        )
        return o
