# GPU=-1
from __future__ import annotations

from typing import Any

from datasets import Dataset

from evalution.benchmarks.data import load_suite_dataset, select_docs


def test_load_suite_dataset_forwards_streaming_flag() -> None:
    """Verify load suite dataset forwards streaming flag."""
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
    assert captured["kwargs"]["stream"] is True


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
    assert captured["kwargs"]["streaming"] is True


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
    assert captured["kwargs"]["streaming"] is True


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
