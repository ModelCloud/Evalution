# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryProfile:
    """Define the memory profile helper class."""
    # Keep the class-level state explicit for this helper.
    dtype_name: str
    dtype_bytes: int
    total_vram_gib: float
    free_vram_gib: float
    parameter_count_billions: float
    kv_cache_bytes_per_token: int | None


def build_memory_profile(
    model: Any,
    *,
    input_device: Any,
    configured_dtype: str | None,
) -> MemoryProfile:
    """Build memory profile."""
    import torch

    dtype_name, dtype_bytes = dtype_metadata(getattr(model, "dtype", None), configured=configured_dtype)
    total_vram_gib = 0.0
    free_vram_gib = 0.0
    if getattr(input_device, "type", None) == "cuda" and torch.cuda.is_available():
        device = input_device.index if input_device.index is not None else 0
        total_vram_gib = bytes_to_gib(torch.cuda.get_device_properties(device).total_memory)
        with suppress(Exception):
            free_bytes, _ = torch.cuda.mem_get_info(device)
            free_vram_gib = bytes_to_gib(free_bytes)

    return MemoryProfile(
        dtype_name=dtype_name,
        dtype_bytes=dtype_bytes,
        total_vram_gib=total_vram_gib,
        free_vram_gib=free_vram_gib,
        parameter_count_billions=parameter_count_billions(model),
        kv_cache_bytes_per_token=kv_cache_bytes_per_token(model, dtype_bytes=dtype_bytes),
    )


def dtype_metadata(dtype: Any, *, configured: str | None) -> tuple[str, int]:
    """Implement dtype metadata for this module."""
    import torch

    resolved_dtype = dtype
    if resolved_dtype is None or resolved_dtype == "auto":
        resolved_dtype = resolve_dtype(configured)

    mapping = {
        torch.float16: ("float16", 2),
        torch.bfloat16: ("bfloat16", 2),
        torch.float32: ("float32", 4),
        torch.float64: ("float64", 8),
    }
    if resolved_dtype in mapping:
        return mapping[resolved_dtype]
    if isinstance(resolved_dtype, str):
        aliases = {
            "fp16": ("float16", 2),
            "float16": ("float16", 2),
            "bf16": ("bfloat16", 2),
            "bfloat16": ("bfloat16", 2),
            "fp32": ("float32", 4),
            "float32": ("float32", 4),
            "auto": ("auto", 2),
        }
        return aliases.get(resolved_dtype, (resolved_dtype, 2))
    return (str(resolved_dtype), 2)


def resolve_dtype(dtype: str | None) -> Any:
    """Resolve dtype."""
    if dtype is None:
        return None
    if dtype == "auto":
        return "auto"

    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype}") from exc


def parameter_count_billions(model: Any) -> float:
    """Implement parameter count billions for this module."""
    with suppress(Exception):
        parameter_count = model.num_parameters()
        if parameter_count:
            return float(parameter_count) / 1_000_000_000.0
    return 1.0


def kv_cache_bytes_per_token(model: Any, *, dtype_bytes: int) -> int | None:
    """Implement kv cache bytes per token for this module. Preserve the fallback order expected by the surrounding caller."""
    config = getattr(model, "config", None)
    if config is None:
        return None

    num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    num_heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None)
    num_kv_heads = (
        getattr(config, "num_key_value_heads", None)
        or getattr(config, "num_kv_heads", None)
        or num_heads
    )
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if hidden_size is not None and num_heads is not None:
            head_dim = int(hidden_size) // int(num_heads)

    if not all(value is not None for value in (num_layers, num_kv_heads, head_dim)):
        return None

    return int(2 * int(num_layers) * int(num_kv_heads) * int(head_dim) * dtype_bytes)


def bytes_to_gib(num_bytes: int) -> float:
    """Implement bytes to gib for this module."""
    return num_bytes / (1024**3)


def gib_to_bytes(num_gib: float) -> float:
    """Implement gib to bytes for this module."""
    return num_gib * (1024**3)
