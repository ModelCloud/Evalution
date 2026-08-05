# GPU=-1
from __future__ import annotations

from typing import Any

from datasets import Dataset, IterableDataset

from evalution.benchmarks.data import load_suite_dataset, select_docs


def test_load_suite_dataset_uses_local_cache_for_streaming() -> None:
    """Verify load suite dataset uses local cache instead of streaming from the Hub."""
    # Use the cached local dataset instead of streaming from the Hub.
    captured: dict[str, Any] = {}

    def loader(dataset_path: str, dataset_name: str | None = None, **kwargs: Any) -> list[str]:
        """Support the surrounding tests with loader."""
        captured["path"] = dataset_path
        captured["name"] = dataset_name
        captured["kwargs"] = kwargs
        return ["row"]

    rows, _ = load_suite_dataset(
        loader,
        task_name="foo",
        dataset_path="path",
        dataset_name="name",
        split="test",
        cache_dir="cache",
        stream=True,
    )

    assert rows == ["row"]
    assert captured["path"] == "path"
    assert captured["name"] == "name"
    assert captured["kwargs"]["split"] == "test"
    assert captured["kwargs"]["cache_dir"] == "cache"
    assert captured["kwargs"]["stream"] is False


def test_load_suite_dataset_returns_cached_iterable_for_streaming() -> None:
    """Verify stream=True caches the dataset locally and returns a lazy IterableDataset."""
    captured: dict[str, Any] = {}
    cached = Dataset.from_list([{"value": "cached"}])

    def loader(dataset_path: str, dataset_name: str | None = None, **kwargs: Any) -> Dataset:
        """Support the surrounding tests with loader."""
        captured["path"] = dataset_path
        captured["name"] = dataset_name
        captured["kwargs"] = kwargs
        return cached

    rows, _ = load_suite_dataset(
        loader,
        task_name="foo",
        dataset_path="path",
        dataset_name="name",
        split="test",
        cache_dir="cache",
        stream=True,
    )

    assert isinstance(rows, IterableDataset)
    assert [row["value"] for row in rows] == ["cached"]
    assert captured["kwargs"]["stream"] is False


def test_load_suite_dataset_falls_back_to_streaming_kwarg() -> None:
    """Verify load suite dataset falls back to streaming kwarg."""
    captured: dict[str, Any] = {}

    def loader(dataset_path: str, dataset_name: str | None = None, **kwargs: Any) -> list[str]:
        """Support the surrounding tests with loader."""
        if "stream" in kwargs:
            raise TypeError("unexpected keyword argument 'stream'")
        captured["path"] = dataset_path
        captured["name"] = dataset_name
        captured["kwargs"] = kwargs
        return ["row"]

    rows, _ = load_suite_dataset(
        loader,
        task_name="foo",
        dataset_path="path",
        dataset_name="name",
        split="test",
        cache_dir="cache",
        stream=True,
    )

    assert rows == ["row"]
    assert captured["kwargs"]["streaming"] is False


def test_load_suite_dataset_falls_back_to_streaming_after_builder_config_error() -> None:
    """Verify load suite dataset falls back to streaming after builder config error."""
    captured: dict[str, Any] = {}

    def loader(dataset_path: str, dataset_name: str | None = None, **kwargs: Any) -> list[str]:
        """Support the surrounding tests with loader."""
        if "stream" in kwargs:
            raise ValueError("BuilderConfig Foo doesn't have a 'stream' key.")
        captured["path"] = dataset_path
        captured["name"] = dataset_name
        captured["kwargs"] = kwargs
        return ["row"]

    rows, _ = load_suite_dataset(
        loader,
        task_name="foo",
        dataset_path="path",
        dataset_name="name",
        split="test",
        cache_dir="cache",
        stream=True,
    )

    assert rows == ["row"]
    assert captured["kwargs"]["streaming"] is False


def test_select_docs_can_pick_explicit_dataset_rows_before_capping() -> None:
    docs = Dataset.from_list(
        [
            {"value": "zero"},
            {"value": "one"},
            {"value": "two"},
            {"value": "three"},
        ]
    )

    selected = select_docs(
        docs,
        row_indices=(3, 1, 2),
        max_rows=2,
    )

    assert [row["value"] for row in selected] == ["three", "one"]


def test_select_docs_can_pick_explicit_streaming_rows() -> None:
    selected = select_docs(
        ({"value": value} for value in ("zero", "one", "two", "three")),
        row_indices=(2, 0),
        max_rows=None,
    )

    assert [row["value"] for row in selected] == ["two", "zero"]
