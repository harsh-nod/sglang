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

import functools
import os

import torch
import torch.nn.functional as F
import wave_lang.kernel as tk
import wave_lang.kernel.lang as tkl
from wave_lang.kernel.lang import DataType
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.templates.moe import get_gemm_kernel, get_silu_and_mul_kernel
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))
enable_scheduling_barriers = int(os.environ.get("WAVE_USE_SCHED_BARRIERS", 0))
check_individual_kernels = int(os.environ.get("WAVE_CHECK_INDIV_KERNS", 0))


rtol, atol = 1e-1, 1e-2


@functools.lru_cache
def get_wave_gemm_kernel(
    m: int,
    k: int,
    n: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    gemm, symbols = get_gemm_kernel(
        m,
        k,
        n,
        mfma_variant,
        datatype,
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)
    return gemm


@functools.lru_cache
def get_wave_silu_and_mul_kernel(
    m: int,
    n: int,
    datatype: DataType,
):
    assert datatype in [
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported datatype: {datatype}"
    silu_and_mul_kernel, symbols = get_silu_and_mul_kernel(
        m, n, tkl.f16 if datatype == torch.float16 else tkl.bf16
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        wave_runtime=False,
    )
    options = set_default_run_config(options)
    silu_and_mul = wave_compile(options, silu_and_mul_kernel)
    return silu_and_mul


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    assert len(gate.shape) == len(up.shape) == 2
    assert gate.shape[0] == up.shape[0] and gate.shape[1] == up.shape[1]
    wave_kernel = get_wave_silu_and_mul_kernel(gate.shape[0], gate.shape[1], gate.dtype)

    out = torch.zeros(gate.shape, dtype=gate.dtype, device=gate.device)
    wave_kernel(gate, up, out)
    ref = F.silu(gate) * up
    if check_individual_kernels:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol, check_device=False)
    return out


def moe_split_w1_wave(a, w1_gate, w1_up, w2, score, topk):
    num_tokens, k = a.shape
    a = a.view(num_tokens, -1, k).repeat(1, topk, 1).reshape(-1, k)
    out = torch.zeros(num_tokens * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    dtype = tkl.f16 if a.dtype == torch.float16 else tkl.bf16
    assert w1_gate.shape[0] == w1_up.shape[0] == w2.shape[0]
    e = w1_gate.shape[0]
    for i in range(e):
        mask = topk_ids == i
        if mask.sum():
            m = int(mask.sum())
            gate = torch.zeros(
                m, w1_gate[i].shape[0], dtype=torch.float32, device=a.device
            )
            up = torch.zeros(
                m, w1_gate[i].shape[0], dtype=torch.float32, device=a.device
            )
            assert w1_gate[i].shape == w1_up[i].shape
            gemm_kernel_gate_up = get_wave_gemm_kernel(
                m,  # M
                w1_gate[i].shape[-1],  # K
                w1_gate[i].shape[0],  # N
                MMAType.F32_16x16x16_F16,
                dtype,
            )
            gemm_kernel_gate_up(a[mask], w1_gate[i], gate)
            gemm_kernel_gate_up(a[mask], w1_up[i], up)
            gate = gate.to(dtype=a.dtype)
            up = up.to(dtype=a.dtype)
            if check_individual_kernels:
                torch.testing.assert_close(
                    gate,
                    a[mask] @ w1_gate[i].transpose(0, 1),
                    rtol=rtol,
                    atol=atol,
                    check_device=False,
                )
                torch.testing.assert_close(
                    up,
                    a[mask] @ w1_up[i].transpose(0, 1),
                    rtol=rtol,
                    atol=atol,
                    check_device=False,
                )
            lhs = torch.zeros(m, w2.shape[-1], dtype=a.dtype, device=a.device)
            lhs = silu_and_mul(gate, up)
            rhs = w2[i]
            partial_out = torch.zeros(
                m, w2.shape[1], dtype=torch.float32, device=a.device
            )
            gemm_kernel_out = get_wave_gemm_kernel(
                m,  # M
                w2[i].shape[-1],  # K
                w2[i].shape[0],  # N
                MMAType.F32_16x16x16_F16,
                dtype,
            )
            gemm_kernel_out(lhs, rhs, partial_out)
            partial_out = partial_out.to(dtype=a.dtype)
            if check_individual_kernels:
                torch.testing.assert_close(
                    partial_out,
                    lhs @ rhs.transpose(0, 1),
                    rtol=rtol,
                    atol=atol,
                    check_device=False,
                )
            out[mask] = partial_out
    return (
        out.view(num_tokens, -1, w2.shape[1])
        * topk_weight.view(num_tokens, -1, 1).to(out.dtype)
    ).sum(dim=1)
