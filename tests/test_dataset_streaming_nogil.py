# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import IterableDataset, load_dataset


def _is_runtime_nogil() -> bool:
    if hasattr(sys, "_is_gil_enabled"):
        return not sys._is_gil_enabled()
    if "PYTHON_GIL" in os.environ:
        return os.environ["PYTHON_GIL"] == "0"
    raise AssertionError(
        "Cannot verify GIL state on this runtime because _is_gil_enabled() is unavailable."
    )


def test_streaming_dataset_supports_many_threads_in_nogil_mode(tmp_path) -> None:
    if not _is_runtime_nogil():
        raise AssertionError(
            "This test validates free-threaded datasets behavior and must run with "
            "PYTHON_GIL=0 / no-GIL runtime."
        )

    source_path = tmp_path / "rows.jsonl"
    with source_path.open("w", encoding="utf-8") as out:
        for index in range(1_000):
            out.write(json.dumps({"id": index, "text": "payload"}) + "\n")

    workers = 12
    rows_to_fetch = 128

    def worker(worker_index: int) -> tuple[int, int]:
        dataset = load_dataset(
            "json",
            data_files=str(source_path),
            split="train",
            streaming=True,
            cache_dir=str(tmp_path / f"hf_cache_{worker_index}"),
        )
        assert isinstance(dataset, IterableDataset)

        iterator = iter(dataset)
        first = next(iterator)
        last = first["id"]
        for _ in range(rows_to_fetch - 1):
            last = next(iterator)["id"]
        return first["id"], last

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, worker_index) for worker_index in range(workers)]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == workers
    assert all(first == 0 for first, _ in results)
    assert all(last == rows_to_fetch - 1 for _, last in results)
