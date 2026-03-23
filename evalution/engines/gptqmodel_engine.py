# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalution.config import Model
from evalution.engines.memory import resolve_dtype
from evalution.engines.transformer import (
    TransformerSession,
    _AUTO_PAGED_ATTENTION,
    _effective_attn_implementation,
    _resolve_paged_attention,
    _warn_pending_nogil_transformers_pr_once,
)
from evalution.engines.transformers_common import (
    _TransformersCommonConfig,
    _base_attn_implementation,
    _clone_prepare_tokenizer,
    _resolve_input_device,
    transformers_continuous_batching_support,
)
from evalution.logbar import get_logger

_DEFAULT_GPTQMODEL_PATH = "/root/gptqmodel"
_UNSUPPORTED_GPTQMODEL_BACKENDS = frozenset({"mlx", "sglang", "vllm"})


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
class GPTQModelEngine(_TransformersCommonConfig):
    # Load quantized Hugging Face-compatible checkpoints through GPTQModel.
    backend: str = "auto"
    gptqmodel_path: str | None = _DEFAULT_GPTQMODEL_PATH
    paged_attention: bool | str = _AUTO_PAGED_ATTENTION
    manual_eviction: bool = False
    allow_block_sharing: bool = True
    use_async_batching: bool | None = None
    q_padding_interval_size: int = 0
    kv_padding_interval_size: int = 0
    max_cached_graphs: int = 0

    # Reuse the same paged-attention feature gating as the native transformer engine.
    def build(self, model: Model) -> TransformerSession:
        supports_continuous_batching, reason = transformers_continuous_batching_support()
        if not supports_continuous_batching and self.paged_attention not in {False, _AUTO_PAGED_ATTENTION}:
            get_logger().warning(
                "gptqmodel continuous batching is unavailable: %s; falling back to standard generate()",
                reason,
            )

        if supports_continuous_batching:
            _warn_pending_nogil_transformers_pr_once()

        self.resolved_engine = "GPTQModelEngine"
        return GPTQModelSession.from_config(
            self,
            model,
            supports_continuous_batching=supports_continuous_batching,
        )


@dataclass(slots=True)
class GPTQModelSession(TransformerSession):
    # Keep the GPTQModel wrapper alive so quantized kernels and metadata outlive the inner HF model.
    model_wrapper: Any | None = field(default=None, repr=False)
    resolved_backend: str | None = None
    quant_method: str | None = None
    runtime_format: str | None = None

    @classmethod
    def from_config(
        cls,
        config: GPTQModelEngine,
        model_config: Model,
        *,
        supports_continuous_batching: bool,
    ) -> GPTQModelSession:
        runtime = load_gptqmodel_runtime(config, model_config)
        raw_attn_implementation = runtime.requested_attn_implementation
        paged_attention_enabled = supports_continuous_batching and _resolve_paged_attention(
            paged_attention=config.paged_attention,
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
        execution = super().describe_execution()
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
        super().close()


# Load the tokenizer through transformers for Evalution's shared session logic, then attach the
# already-initialized inner HF model returned by GPTQModel.
def load_gptqmodel_runtime(
    config: GPTQModelEngine,
    model_config: Model,
) -> _LoadedGPTQModelRuntime:
    import torch
    from transformers import AutoTokenizer

    trust_remote_code = (
        config.trust_remote_code
        if config.trust_remote_code is not None
        else model_config.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_path or model_config.path,
        revision=model_config.revision,
        trust_remote_code=trust_remote_code,
        **model_config.tokenizer_kwargs,
    )
    tokenizer.padding_side = config.padding_side
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("tokenizer must define either a pad_token, eos_token, or unk_token")

    backend = _normalize_gptqmodel_backend(config.backend)
    _validate_gptqmodel_backend(backend)
    gptqmodel_module = _import_gptqmodel(config.gptqmodel_path)

    load_kwargs = {
        **model_config.model_kwargs,
        **config.load_kwargs,
        "revision": model_config.revision,
    }
    if "dtype" not in load_kwargs and "torch_dtype" in load_kwargs:
        load_kwargs["dtype"] = load_kwargs["torch_dtype"]
    resolved_dtype = resolve_dtype(config.dtype)
    if resolved_dtype is not None:
        load_kwargs["dtype"] = resolved_dtype
    if "dtype" in load_kwargs:
        load_kwargs.pop("torch_dtype", None)

    raw_attn_implementation = config.attention_impl or config.attn_implementation
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
    if config.device_map is None:
        input_device = torch.device(device)
    else:
        input_device = _resolve_input_device(model, prefer=config.device)

    requested_attn_implementation = (
        attn_implementation
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
            "GPTQModelEngine only supports native GPTQModel/HF-style backends; "
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
