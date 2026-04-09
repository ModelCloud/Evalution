# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

# Pin the audited Spanish WNLI translation files so evaluation never executes the remote dataset script.
WNLI_ES_DATASET_PATH = "PlanTL-GOB-ES/wnli-es"
WNLI_ES_FILE_SPECS = {
    "train": (
        "wnli-train-es.csv",
        "87a0f9466e2f54293217c89cec7418074aff0829d965b9bf459bd132ffbbb5c2",
    ),
    "validation": (
        "wnli-dev-es.csv",
        "12b2dc74ac4b0e5d418160631b05f52e41c37592ad1ea9e5dcfa4aee8875a756",
    ),
    "test": (
        "wnli-test-shuffled-es.csv",
        "c150e76a9cb9b697b4ccddb3c6464e274325c266c959983fff7ffabfe630f8b9",
    ),
}
_WNLI_ES_SPLIT_ALIASES = {
    "train": "train",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolved_split(split: str) -> str:
    normalized = split.strip().lower()
    resolved = _WNLI_ES_SPLIT_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            f"unsupported wnli_es split: {split!r}; expected one of {tuple(_WNLI_ES_SPLIT_ALIASES)}"
        )
    return resolved


def _download_wnli_es_file(
    dataset_path: str,
    *,
    split: str,
    cache_dir: str | None,
) -> Path:
    if dataset_path != WNLI_ES_DATASET_PATH:
        raise ValueError(
            f"wnli_es dataset_path must be {WNLI_ES_DATASET_PATH!r}, got {dataset_path!r}"
        )
    filename, expected_sha256 = WNLI_ES_FILE_SPECS[split]
    local_path = Path(
        hf_hub_download(
            repo_id=dataset_path,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )
    actual_sha256 = _sha256_file(local_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "wnli_es checksum mismatch for "
            f"{filename}: expected {expected_sha256}, got {actual_sha256}"
        )
    return local_path


@lru_cache(maxsize=None)
def _load_rows_cached(
    *,
    split: str,
    cache_root_key: str,
) -> tuple[dict[str, Any], ...]:
    csv_path = _download_wnli_es_file(
        WNLI_ES_DATASET_PATH,
        split=split,
        cache_dir=cache_root_key or None,
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["index", "sentence1", "sentence2", "label"]:
            raise ValueError(f"wnli_es file has unexpected columns: {reader.fieldnames!r}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "index": int(str(row["index"]).strip()),
                    "sentence1": str(row["sentence1"]).strip(),
                    "sentence2": str(row["sentence2"]).strip(),
                    "label": int(str(row["label"]).strip()),
                }
            )
    return tuple(rows)


def load_wnli_es_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> list[dict[str, Any]]:
    if dataset_name is not None:
        raise ValueError("wnli_es dataset_name must be None")
    if stream:
        raise ValueError("wnli_es vendored loader does not support stream=True")
    resolved_split = _resolved_split(split)
    return list(
        _load_rows_cached(
            split=resolved_split,
            cache_root_key=cache_dir or "",
        )
    )
