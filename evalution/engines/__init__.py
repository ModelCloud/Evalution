# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from .base import (
    BaseEngine,
    BaseEngineDeviceConfig,
    BaseEnginePagedBatchingConfig,
    BaseEngineQuantizationConfig,
    BaseEngineTokenizerModeConfig,
    BaseEngineTransformersRuntimeConfig,
    BaseInferenceSession,
    SharedEngineConfig,
)
from .gptqmodel_engine import GPTQModel
from .llama_cpp_engine import LlamaCpp
from .openai_engine import OpenAICompatible
from .openai_server import OpenAICompatibleServer, build_openai_compatible_server
from .openvino_engine import OpenVINO
from .sglang_engine import SGLang
from .tensorrt_llm_engine import TensorRTLLM
from .tinygrad_engine import Tinygrad
from .transformers import Transformers
from .transformers_compat import TransformersCompat
from .vllm_engine import VLLM

# Keep engine defaults and compatibility flags explicit at module scope.
__all__ = [
    "BaseEngine",
    "BaseEngineDeviceConfig",
    "BaseEnginePagedBatchingConfig",
    "BaseEngineQuantizationConfig",
    "BaseEngineTokenizerModeConfig",
    "BaseEngineTransformersRuntimeConfig",
    "BaseInferenceSession",
    "GPTQModel",
    "LlamaCpp",
    "OpenAICompatible",
    "OpenAICompatibleServer",
    "OpenVINO",
    "SGLang",
    "SharedEngineConfig",
    "TensorRTLLM",
    "Tinygrad",
    "Transformers",
    "TransformersCompat",
    "VLLM",
    "build_openai_compatible_server",
]
