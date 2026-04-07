# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any

from evalution.config import Model
from evalution.engines.base import BaseEngineDeviceConfig, SharedEngineConfig
from evalution.engines.transformers_common import (
    BaseTransformerSession,
    _clone_prepare_tokenizer,
    _load_tokenizer_from_model,
    _normalize_tokenizer_special_tokens,
    _resolve_tokenizer_source,
    _seed_transformer_runtime,
    _seed_with_internal_apis,
)
from evalution.logbar import get_logger


@dataclass(slots=True)
class _LoadedOpenVINORuntime:
    # Bundle the OpenVINO runtime objects so session construction stays deterministic.
    model: Any
    tokenizer: Any
    prepare_tokenizer: Any | None
    input_device: Any


@dataclass(slots=True)
class _OpenVINOConfig(BaseEngineDeviceConfig, SharedEngineConfig):
    # Hold the shared OpenVINO engine controls and keep engine serialization stable.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OpenVINO(_OpenVINOConfig):
    # Load decoder-only models through Optimum Intel's OpenVINO backend.
    compile: bool | None = None
    dynamic_shapes: bool | None = None
    ov_config: dict[str, Any] | None = None

    def build(self, model: Model) -> BaseTransformerSession:
        self.resolved_engine = "OpenVINO"
        return OpenVINOSession.from_config(self, model)


@dataclass(slots=True)
class OpenVINOSession(BaseTransformerSession):
    # Reuse the shared transformer session logic while tracking OpenVINO runtime settings.

    @classmethod
    def from_config(cls, config: OpenVINO, model_config: Model) -> OpenVINOSession:
        # Build the OpenVINO runtime first so the session always receives fully initialized state.
        runtime = load_openvino_runtime(config, model_config)
        return cls(
            config=config,
            model_config=model_config,
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            prepare_tokenizer=runtime.prepare_tokenizer,
            input_device=runtime.input_device,
            generation_backend="openvino_generate",
        )


def _import_openvino_optimum() -> Any:
    """Import the optional Optimum Intel module and raise one stable install hint when absent."""

    try:
        return importlib.import_module("optimum.intel.openvino")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenVINO engine requires the optional `optimum[openvino]` and `transformers>=4.45,<4.58` dependency"
        ) from exc


def load_openvino_runtime(
    config: OpenVINO,
    model_config: Model,
) -> _LoadedOpenVINORuntime:
    """Load the OpenVINO model and tokenizer pair using Evalution's shared transformer setup."""

    import torch

    _seed_transformer_runtime(config.seed)

    trust_remote_code = (
        config.trust_remote_code
        if config.trust_remote_code is not None
        else model_config.trust_remote_code
    )
    tokenizer = _load_tokenizer_from_model(
        _resolve_tokenizer_source(model_config),
        revision=model_config.revision,
        trust_remote_code=trust_remote_code,
        **model_config.tokenizer_kwargs,
    )
    tokenizer.padding_side = config.padding_side

    openvino_module = _import_openvino_optimum()
    device_str = config.device or "cpu"
    input_device = torch.device(device_str)
    load_kwargs = {
        **model_config.model_kwargs,
        "revision": model_config.revision,
        "trust_remote_code": trust_remote_code,
        "device": device_str,
    }
    if config.compile is not None:
        load_kwargs["compile"] = config.compile
    if config.dynamic_shapes is not None:
        load_kwargs["dynamic_shapes"] = config.dynamic_shapes
    if config.ov_config is not None:
        load_kwargs["ov_config"] = dict(config.ov_config)

    model = openvino_module.OVModelForCausalLM.from_pretrained(
        model_config.path,
        **load_kwargs,
    )
    _seed_with_internal_apis(model, config.seed)
    freeze = getattr(model, "requires_grad_", None)
    if callable(freeze):
        freeze(False)
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        model = eval_method()
    _normalize_tokenizer_special_tokens(tokenizer=tokenizer, model=model)

    return _LoadedOpenVINORuntime(
        model=model,
        tokenizer=tokenizer,
        prepare_tokenizer=_clone_prepare_tokenizer(
            tokenizer=tokenizer,
            model_config=model_config,
            trust_remote_code=trust_remote_code,
            model=model,
        ),
        input_device=input_device,
    )
