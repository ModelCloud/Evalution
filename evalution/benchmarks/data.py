# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import random
from itertools import islice
from time import perf_counter
from typing import Any, Callable, TypeVar

import pcre

from evalution.logbar import get_logger, spinner

T = TypeVar("T")
_DEFAULT_SHUFFLE_SEED = 7
_UNEXPECTED_LOADER_KWARG_PATTERN = pcre.compile(r"unexpected keyword argument '([^']+)'")
_UNEXPECTED_BUILDER_CONFIG_KEY_PATTERN = pcre.compile(r"doesn't have a '([^']+)' key")


def _unexpected_loader_kwargs(exc: TypeError) -> set[str]:
    return set(_UNEXPECTED_LOADER_KWARG_PATTERN.findall(str(exc)))


def _unexpected_loader_config_keys(exc: ValueError) -> set[str]:
    return set(_UNEXPECTED_BUILDER_CONFIG_KEY_PATTERN.findall(str(exc)))


def _cached_stream_config_miss(exc: ValueError) -> bool:
    message = str(exc)
    return "Couldn't find cache for" in message and "-stream=" in message


def _invoke_dataset_loader(loader: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return loader(*args, **kwargs)
    except TypeError as exc:
        unexpected = _unexpected_loader_kwargs(exc)
        if "stream" in unexpected and "stream" in kwargs:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["streaming"] = fallback_kwargs.pop("stream")
            return loader(*args, **fallback_kwargs)
        raise
    except ValueError as exc:
        unexpected = _unexpected_loader_config_keys(exc)
        if "stream" in unexpected and "stream" in kwargs:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["streaming"] = fallback_kwargs.pop("stream")
            return loader(*args, **fallback_kwargs)
        if "stream" in kwargs and _cached_stream_config_miss(exc):
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["streaming"] = fallback_kwargs.pop("stream")
            return loader(*args, **fallback_kwargs)
        raise


def normalize_order(order: str) -> str:
    normalized = order.strip().lower()
    if normalized in {"native", "length|asc", "length|desc"}:
        return normalized
    if normalized == "shuffle":
        return f"shuffle|{_DEFAULT_SHUFFLE_SEED}"
    if normalized.startswith("shuffle|"):
        _, seed_text = normalized.split("|", maxsplit=1)
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"unsupported benchmark order seed: {order!r}") from exc
        return f"shuffle|{seed}"
    raise ValueError(
        f"unsupported benchmark order: {order!r}; expected one of native, shuffle, "
        "shuffle|<seed>, length|asc, length|desc"
    )


def apply_order(
    items: list[T],
    *,
    order: str,
    length_key: Callable[[T], int],
) -> list[T]:
    normalized_order = normalize_order(order)
    ordered = list(items)
    if normalized_order == "native":
        return ordered
    if normalized_order.startswith("shuffle|"):
        seed = int(normalized_order.split("|", maxsplit=1)[1])
        random.Random(seed).shuffle(ordered)
        return ordered
    reverse = normalized_order == "length|desc"
    return sorted(ordered, key=length_key, reverse=reverse)


# Load the suite dataset and return both the rows object and wall-clock load time.
def load_suite_dataset(
    loader: Any,
    *,
    task_name: str,
    dataset_path: str,
    dataset_name: str | None,
    split: str,
    cache_dir: str | None,
    stream: bool,
    purpose: str | None = None,
) -> tuple[Any, float]:
    logger = get_logger()
    dataset_ref = dataset_path if dataset_name is None else f"{dataset_path}/{dataset_name}"
    purpose_prefix = f"{purpose.strip()} " if isinstance(purpose, str) and purpose.strip() else ""
    logger.info("loading %sdataset %s split=%s for %s", purpose_prefix, dataset_ref, split, task_name)
    kwargs = {
        "split": split,
        "cache_dir": cache_dir,
        # Keep `stream=` as the single internal suite contract. Dataset loaders that only
        # understand Hugging Face's `streaming=` kwarg must reject `stream` so the fallback
        # path in `_invoke_dataset_loader` can translate the call at the final boundary.
        "stream": stream,
    }
    dataset_load_started = perf_counter()
    with spinner(f"{task_name}: loading dataset"):
        if dataset_name is None:
            loaded_docs = _invoke_dataset_loader(loader, dataset_path, **kwargs)
        else:
            loaded_docs = _invoke_dataset_loader(loader, dataset_path, dataset_name, **kwargs)
    return loaded_docs, perf_counter() - dataset_load_started


# Select an explicit row subset before applying the usual max_rows cap.
def select_docs(
    docs: Any,
    *,
    row_indices: tuple[int, ...] | None,
    max_rows: int | None,
) -> Any:
    if row_indices is None:
        return limit_docs(docs, max_rows)

    normalized_indices = tuple(int(index) for index in row_indices)
    if any(index < 0 for index in normalized_indices):
        raise ValueError("benchmark row_indices must be non-negative")

    if hasattr(docs, "select") and hasattr(docs, "__len__"):
        doc_count_value = len(docs)
        if any(index >= doc_count_value for index in normalized_indices):
            raise IndexError("benchmark row_indices exceeded the available dataset rows")
        selected_docs = docs.select(list(normalized_indices))
        return limit_docs(selected_docs, max_rows)

    selected_rows: list[Any] = []
    wanted_positions = {
        index: position
        for position, index in enumerate(normalized_indices)
    }
    max_index = max(normalized_indices, default=-1)
    for index, row in enumerate(docs):
        if index > max_index and len(selected_rows) == len(normalized_indices):
            break
        position = wanted_positions.get(index)
        if position is None:
            continue
        selected_rows.append((position, row))
        if len(selected_rows) == len(normalized_indices):
            break
    if len(selected_rows) != len(normalized_indices):
        raise IndexError("benchmark row_indices exceeded the available dataset rows")
    ordered_rows = [row for _position, row in sorted(selected_rows, key=lambda item: item[0])]
    return limit_docs(ordered_rows, max_rows)


# Apply an optional row cap while preserving streaming datasets.
def limit_docs(docs: Any, max_rows: int | None) -> Any:
    if max_rows is None:
        return docs
    if hasattr(docs, "select") and hasattr(docs, "__len__"):
        return docs.select(range(min(max_rows, len(docs))))
    return islice(docs, max_rows)


# Resolve the row count for both materialized and streaming datasets.
def doc_count(
    docs: Any,
    *,
    loaded_docs: Any,
    max_rows: int | None,
    split: str,
) -> int:
    if hasattr(docs, "__len__"):
        count = len(docs)
        return min(max_rows, count) if max_rows is not None else count

    split_metadata = getattr(getattr(loaded_docs, "info", None), "splits", None)
    split_info = split_metadata.get(split) if hasattr(split_metadata, "get") else None
    if split_info is not None and getattr(split_info, "num_examples", None) is not None:
        count = int(split_info.num_examples)
        return min(max_rows, count) if max_rows is not None else count

    if max_rows is not None:
        return int(max_rows)

    raise ValueError(
        "streaming dataset row count is unavailable; set `max_rows` or use a dataset split with known num_examples"
    )
