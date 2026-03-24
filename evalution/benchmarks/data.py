# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import random
from itertools import islice
from time import perf_counter
from typing import Any, Callable, TypeVar

from evalution.logbar import get_logger, spinner

T = TypeVar("T")
_DEFAULT_SHUFFLE_SEED = 7

def normalize_order(order: str) -> str:
    normalized = order.strip().lower()
    if normalized in {"native", "length|asc", "length|desc"}:
        return normalized
    if normalized in {"shuffle", "random"}:
        return f"shuffle|{_DEFAULT_SHUFFLE_SEED}"
    if normalized.startswith("shuffle|") or normalized.startswith("random|"):
        mode, seed_text = normalized.split("|", maxsplit=1)
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"unsupported benchmark order seed: {order!r}") from exc
        del mode
        return f"shuffle|{seed}"
    raise ValueError(
        f"unsupported benchmark order: {order!r}; expected one of native, shuffle, random, "
        "shuffle|<seed>, random|<seed>, length|asc, length|desc"
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
    streaming: bool,
) -> tuple[Any, float]:
    logger = get_logger()
    dataset_ref = dataset_path if dataset_name is None else f"{dataset_path}/{dataset_name}"
    logger.info("loading dataset %s split=%s for %s", dataset_ref, split, task_name)

    kwargs = {
        "split": split,
        "cache_dir": cache_dir,
        "streaming": streaming,
    }
    dataset_load_started = perf_counter()
    with spinner(f"{task_name}: loading dataset"):
        if dataset_name is None:
            loaded_docs = loader(dataset_path, **kwargs)
        else:
            loaded_docs = loader(dataset_path, dataset_name, **kwargs)
    return loaded_docs, perf_counter() - dataset_load_started


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

    split_info = getattr(getattr(loaded_docs, "info", None), "splits", {}).get(split)
    if split_info is not None and getattr(split_info, "num_examples", None) is not None:
        count = int(split_info.num_examples)
        return min(max_rows, count) if max_rows is not None else count

    if max_rows is not None:
        return int(max_rows)

    raise ValueError(
        "streaming dataset row count is unavailable; set `max_rows` or use a dataset split with known num_examples"
    )
