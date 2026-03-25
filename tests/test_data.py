from __future__ import annotations

from typing import Any

from evalution.benchmarks.data import load_suite_dataset


def test_load_suite_dataset_forwards_streaming_flag() -> None:
    captured: dict[str, Any] = {}

    def loader(dataset_path: str, dataset_name: str | None = None, **kwargs: Any) -> list[str]:
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
    # We call the loader with our internal `stream` flag, but downstream APIs expect `streaming`.
    assert captured["kwargs"]["streaming"] is True
