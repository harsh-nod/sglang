# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1

import torch
import math

import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.constraints import MMAType

from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.templates.extend_attention import (
    get_extend_attention_kernel,
)

import os
import logging

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))

import inspect, re

def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)


def extend_attention_wave(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    req_to_tokens,
    b_req_idx,
    b_seq_len,
    b_seq_len_extend,
    b_start_loc_extend,
    max_seq_len,
    output,
    layer_scaling=None,
    is_causal=True, # TODO: check this
    logit_cap = 0.  # TODO: check this
    # is_causal=False,
):
    # arguments = [
    #         ("q_extend", q_extend),
    #         ("k_extend", k_extend),
    #         ("v_extend", v_extend),
    #         ("output", output),
    #         ("req_to_tokens", req_to_tokens),
    #         ("k_buffer", k_buffer),
    #         ("v_buffer", v_buffer),
    #     ]

    # local_arg = locals()
    # for arg in local_arg:
    #     if hasattr(arg, "shape"):
    #         logging.info(f"Argument {varname(arg)}: shape = {arg.shape}")
    #     else:
    #         logging.info(f"Argument {varname(arg)}: value = {arg}")
    # logging.info(f"K cache: {k_buffer.shape}")
    # logging.info(f"V cache: {v_buffer.shape}")
    # logging.info(f"req to token: {req_to_tokens.shape}")
    # logging.info(f"k_stride = {k_buffer.stride()}")
    # logging.info(f"v_stride = {v_buffer.stride()}")
    # logging.info(f"q_extend_shape = {q_extend.shape}")
    # logging.info(f"k_extend_shape = {k_extend.shape}")
    # logging.info(f"v_extend_shape = {v_extend.shape}")
    # logging.info(f"req_to_tokens.stride = {req_to_tokens.stride()}")
    # logging.info(f"output.stride = {output.stride()}")
    # logging.info(f"b_seq_len = {b_seq_len}")
    # logging.info(f"b_seq_len_extend = {b_seq_len_extend}")
    # logging.info(f"b_req_idx = {b_req_idx}")
    # logging.info(f"req_to_tokens[0, 0:32] = {req_to_tokens[0, 0:32]}")

    shape = AttentionShape(
        num_query_heads=q_extend.shape[1],
        num_kv_heads=k_extend.shape[1],
        head_size=q_extend.shape[2],
        head_size_kv=k_extend.shape[2],
        num_seqs=b_seq_len.shape[0],
        max_seq_len=max_seq_len,
        block_size=64,
    )

    assert shape.num_query_heads % shape.num_kv_heads == 0
    # Run the wave kernel.
    mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_extend.shape,
        k_extend.shape,
        v_extend.shape,
        req_to_tokens.shape,
        k_buffer.shape,
        v_buffer.shape,
        output.shape,
        q_extend.dtype,
        output.dtype,
        b_seq_len.dtype,
        is_causal,
        logit_cap,
    )

    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()

    # log2e = 1.44269504089
    # dk_sqrt = layer_scaling or math.sqrt(1.0 / shape.head_size)
    # scaling = math.log(math.e, 2) * dk_sqrt
    # dk_sqrt = math.sqrt(1.0 / shape.head_size)
    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        schedule=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        # TODO: Add scaling of QK as part of kernel.
        mb = extend_attention(
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            output,
        )

        if dump_generated_mlir:
            shape_list = [
                q_extend.shape[0],
                q_extend.shape[1],
                k_extend.shape[1],
                q_extend.shape[2],
                k_extend.shape[2],
            ]
            filename = f"wave_prefill_attention_{'x'.join(map(str, shape_list))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

