# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import os
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalution.config import Model
from evalution.engines.base import BaseEnginePagedBatchingConfig
from evalution.engines.memory import resolve_dtype
from evalution.engines.transformers import (
    TransformersSession,
    _effective_attn_implementation,
    _resolve_paged_attention,
    _warn_pending_nogil_transformers_pr_once,
)
from evalution.engines.transformers_common import (
    _TransformersCommonConfig,
    _base_attn_implementation,
    _clone_prepare_tokenizer,
    _normalize_tokenizer_special_tokens,
    _load_tokenizer_from_model,
    _resolve_tokenizer_source,
    _requests_paged_attention,
    _resolve_input_device,
    _seed_with_internal_apis,
    _seed_transformer_runtime,
    transformers_continuous_batching_support,
)
from evalution.logbar import get_logger

_UNSUPPORTED_GPTQMODEL_BACKENDS = frozenset({"mlx", "sglang", "vllm"})


def _default_gptqmodel_path() -> str | None:
    for env_name in ("EVALUTION_GPTQMODEL_PATH", "GPTQMODEL_PATH"):
        configured = os.environ.get(env_name)
        if configured:
            return configured

    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (
            repo_root / "gptqmodel",
            repo_root.parent / "gptqmodel",
            Path.cwd() / "gptqmodel",
    ):
        if candidate.exists():
            return str(candidate)

    return None


@dataclass(slots=True)
class _LoadedGPTQModelRuntime:
    # Bundle the GPTQModel runtime objects so session construction can stay deterministic.
    model_wrapper: Any
    model: Any
    tokenizer: Any
    prepare_tokenizer: Any | None
    input_device: Any
    requested_attn_implementation: str | None
    resolved_backend: str | None
    quant_method: str | None
    runtime_format: str | None


@dataclass(slots=True)
class GPTQModel(BaseEnginePagedBatchingConfig, _TransformersCommonConfig):
    # Load quantized Hugging Face-compatible checkpoints through GPTQModel.
    backend: str = "auto"
    gptqmodel_path: str | None = field(default_factory=_default_gptqmodel_path)

    # Reuse the same paged-attention feature gating as the native transformer engine.
    def build(self, model: Model) -> TransformersSession:
        supports_continuous_batching, reason = transformers_continuous_batching_support()
        if not supports_continuous_batching and _requests_paged_attention(self.attn_implementation):
            raise ValueError(
                "paged attn_implementation requires a transformers build with continuous batching support"
            )

        if supports_continuous_batching:
            _warn_pending_nogil_transformers_pr_once()

        self.resolved_engine = "GPTQModel"
        return GPTQModelSession.from_config(
            self,
            model,
            supports_continuous_batching=supports_continuous_batching,
        )


@dataclass(slots=True)
class GPTQModelSession(TransformersSession):
    # Keep the GPTQModel wrapper alive so quantized kernels and metadata outlive the inner HF
    # model, while inheriting the shared queued request/result path from TransformersSession.
    model_wrapper: Any | None = field(default=None, repr=False)
    resolved_backend: str | None = None
    quant_method: str | None = None
    runtime_format: str | None = None

    @classmethod
    def from_config(
        cls,
        config: GPTQModel,
        model_config: Model,
        *,
        supports_continuous_batching: bool,
    ) -> GPTQModelSession:
        runtime = load_gptqmodel_runtime(config, model_config)
        raw_attn_implementation = config.attn_implementation or runtime.requested_attn_implementation
        paged_attention_enabled = supports_continuous_batching and _resolve_paged_attention(
            attn_implementation=raw_attn_implementation,
            model=runtime.model,
            input_device=runtime.input_device,
        )
        effective_attn_implementation = _effective_attn_implementation(
            raw_attn_implementation,
            paged_attention_enabled=paged_attention_enabled,
        )
        generation_backend = "continuous_batching" if paged_attention_enabled else "generate"
        get_logger().info(
            "gptqmodel attention requested=%s effective=%s backend=%s paged_attention=%s quant_backend=%s",
            raw_attn_implementation,
            effective_attn_implementation,
            generation_backend,
            paged_attention_enabled,
            runtime.resolved_backend,
        )

        return cls(
            config=config,
            model_config=model_config,
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            prepare_tokenizer=runtime.prepare_tokenizer,
            input_device=runtime.input_device,
            requested_attn_implementation=raw_attn_implementation,
            effective_attn_implementation=effective_attn_implementation,
            paged_attention_enabled=paged_attention_enabled,
            generation_backend=generation_backend,
            model_wrapper=runtime.model_wrapper,
            resolved_backend=runtime.resolved_backend,
            quant_method=runtime.quant_method,
            runtime_format=runtime.runtime_format,
        )

    # Surface GPTQModel-specific runtime details in addition to the shared transformer execution data.
    def describe_execution(self) -> dict[str, Any]:
        execution = super(GPTQModelSession, self).describe_execution()
        execution.update(
            {
                "quantized_backend": self.resolved_backend,
                "quant_method": self.quant_method,
                "runtime_format": self.runtime_format,
            }
        )
        return execution

    # Release the outer GPTQModel wrapper along with the inherited paged-manager and HF resources.
    def close(self) -> None:
        with self._generation_lock:
            with self._prepare_tokenizer_lock:
                with self._tokenizer_lock:
                    with self._state_lock:
                        with suppress(Exception):
                            del self.model_wrapper
        super(GPTQModelSession, self).close()


# Load tokenizer metadata through Tokenicer, then attach the already-initialized inner HF model.
def load_gptqmodel_runtime(
    config: GPTQModel,
    model_config: Model,
) -> _LoadedGPTQModelRuntime:
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

    backend = _normalize_gptqmodel_backend(config.backend)
    _validate_gptqmodel_backend(backend)
    gptqmodel_module = _import_gptqmodel(config.gptqmodel_path)

    load_kwargs = {
        **model_config.model_kwargs,
        "revision": model_config.revision,
    }
    resolved_dtype = resolve_dtype(config.dtype)
    if resolved_dtype is not None and (
        config.dtype != "auto" or ("dtype" not in load_kwargs and "torch_dtype" not in load_kwargs)
    ):
        load_kwargs["dtype"] = resolved_dtype
    if config.dtype != "auto" and "dtype" in load_kwargs:
        load_kwargs.pop("torch_dtype", None)

    raw_attn_implementation = config.attn_implementation
    attn_implementation = _base_attn_implementation(raw_attn_implementation)
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wrapper = gptqmodel_module.GPTQModel.load(
        model_id_or_path=model_config.path,
        device_map=config.device_map,
        device=device if config.device_map is None or config.device is not None else None,
        backend=backend,
        trust_remote_code=trust_remote_code,
        **load_kwargs,
    )
    freeze = getattr(wrapper, "requires_grad_", None)
    if callable(freeze):
        freeze(False)
    wrapper.eval()

    model = getattr(wrapper, "model", wrapper)
    _seed_with_internal_apis(model, config.seed)
    _seed_with_internal_apis(wrapper, config.seed)
    if config.device_map is None:
        input_device = torch.device(device)
    else:
        input_device = _resolve_input_device(model, prefer=config.device)
    _normalize_tokenizer_special_tokens(tokenizer=tokenizer, model=model)

    requested_attn_implementation = (
        raw_attn_implementation
        or getattr(model.config, "_attn_implementation", None)
        or getattr(model.config, "attn_implementation", None)
    )
    quantize_config = getattr(wrapper, "quantize_config", None)

    return _LoadedGPTQModelRuntime(
        model_wrapper=wrapper,
        model=model,
        tokenizer=tokenizer,
        prepare_tokenizer=_clone_prepare_tokenizer(
            tokenizer=tokenizer,
            model_config=model_config,
            trust_remote_code=trust_remote_code,
            model=model,
        ),
        input_device=input_device,
        requested_attn_implementation=requested_attn_implementation,
        resolved_backend=_detect_loaded_backend(wrapper),
        quant_method=_enum_value(getattr(quantize_config, "method", None)),
        runtime_format=_enum_value(
            getattr(quantize_config, "runtime_format", None)
            or getattr(quantize_config, "format", None)
        ),
    )


# Import the local checkout only as a fallback so installed environments keep working unchanged.
def _import_gptqmodel(gptqmodel_path: str | None) -> Any:
    try:
        return importlib.import_module("gptqmodel")
    except ModuleNotFoundError as exc:
        if not gptqmodel_path:
            raise ModuleNotFoundError(
                "gptqmodel is not importable; install it or configure `gptqmodel_path`"
            ) from exc

        checkout_path = Path(gptqmodel_path)
        if not checkout_path.exists():
            raise ModuleNotFoundError(
                f"gptqmodel is not importable and the configured checkout path does not exist: {checkout_path}"
            ) from exc

        checkout_root = str(checkout_path)
        if checkout_root not in sys.path:
            sys.path.insert(0, checkout_root)

        try:
            return importlib.import_module("gptqmodel")
        except ModuleNotFoundError as nested_exc:
            raise ModuleNotFoundError(
                "failed to import gptqmodel from the configured checkout; install GPTQModel runtime "
                "dependencies for that checkout first"
            ) from nested_exc


def _normalize_gptqmodel_backend(backend: Any) -> str:
    value = getattr(backend, "value", backend)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("backend must be a non-empty string")
    return value.strip().lower()


# External runtimes bypass the direct HF model API that Evalution uses for token scoring and paged
# continuous batching, so keep this engine on GPTQModel's local kernels only.
def _validate_gptqmodel_backend(backend: str) -> None:
    if backend in _UNSUPPORTED_GPTQMODEL_BACKENDS:
        raise ValueError(
            "GPTQModel only supports native GPTQModel/HF-style backends; "
            "vllm, sglang, and mlx are incompatible with Evalution scoring"
        )


def _detect_loaded_backend(wrapper: Any) -> str | None:
    for module in getattr(getattr(wrapper, "model", None), "modules", lambda: [])():
        backend = getattr(module, "backend", None)
        if backend is not None:
            return _enum_value(backend)
    return None


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))
