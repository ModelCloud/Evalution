from __future__ import annotations

from itertools import islice
from time import perf_counter
from typing import Any

from evalution.logbar import get_logger, spinner


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


# Apply an optional row limit while preserving streaming datasets.
def limit_docs(docs: Any, limit: int | None) -> Any:
    if limit is None:
        return docs
    if hasattr(docs, "select") and hasattr(docs, "__len__"):
        return docs.select(range(min(limit, len(docs))))
    return islice(docs, limit)


# Resolve the row count for both materialized and streaming datasets.
def doc_count(
    docs: Any,
    *,
    loaded_docs: Any,
    limit: int | None,
    split: str,
) -> int:
    if hasattr(docs, "__len__"):
        count = len(docs)
        return min(limit, count) if limit is not None else count

    split_info = getattr(getattr(loaded_docs, "info", None), "splits", {}).get(split)
    if split_info is not None and getattr(split_info, "num_examples", None) is not None:
        count = int(split_info.num_examples)
        return min(limit, count) if limit is not None else count

    if limit is not None:
        return int(limit)

    raise ValueError(
        "streaming dataset row count is unavailable; set `limit` or use a dataset split with known num_examples"
    )
