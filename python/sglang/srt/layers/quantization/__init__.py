# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py

from typing import Dict, Type

from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.deepspeedfp import DeepSpeedFPConfig
from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
from vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8Config
from vllm.model_executor.layers.quantization.gguf import GGUFConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.gptq_marlin_24 import GPTQMarlin24Config
from vllm.model_executor.layers.quantization.marlin import MarlinConfig
from vllm.model_executor.layers.quantization.qqq import QQQConfig
from vllm.model_executor.layers.quantization.tpu_int8 import Int8TpuConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config

QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "tpu_int8": Int8TpuConfig,
    "fp8": Fp8Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    "marlin": MarlinConfig,
    "modelopt": ModelOptFp8Config,
    "gguf": GGUFConfig,
    "gptq_marlin_24": GPTQMarlin24Config,
    "gptq_marlin": GPTQMarlinConfig,
    "awq_marlin": AWQMarlinConfig,
    "gptq": GPTQConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "qqq": QQQConfig,
    "experts_int8": ExpertsInt8Config,
    "w8a8_int8": W8A8Int8Config,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )
    return QUANTIZATION_METHODS[quantization]


def gptq_get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinLinearMethod,
        GPTQMarlinMoEMethod,
    )

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if isinstance(layer, LinearBase):
        return GPTQMarlinLinearMethod(self)
    elif isinstance(layer, FusedMoE):
        return GPTQMarlinMoEMethod(self)
    return None


def awq_get_quant_method(self, layer, prefix):
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinLinearMethod,
        AWQMoEMethod,
    )

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if isinstance(layer, LinearBase):
        return AWQMarlinLinearMethod(self)
    elif isinstance(layer, FusedMoE):
        return AWQMoEMethod(self)
    return None


def patch_vllm_linear_base_isinstance():
    import builtins

    from vllm.model_executor.layers.linear import LinearBase

    from sglang.srt.layers.linear import LinearBase as PatchedLinearBase

    original_isinstance = builtins.isinstance

    def patched_isinstance(obj, classinfo):
        if classinfo is LinearBase:
            return original_isinstance(obj, PatchedLinearBase)
        return original_isinstance(obj, classinfo)

    builtins.isinstance = patched_isinstance


def apply_monkey_patches():
    """Apply all monkey patches in one place."""
    setattr(GPTQMarlinConfig, "get_quant_method", gptq_get_quant_method)
    setattr(AWQMarlinConfig, "get_quant_method", awq_get_quant_method)


patch_vllm_linear_base_isinstance()
# Apply patches when module is imported
apply_monkey_patches()


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
