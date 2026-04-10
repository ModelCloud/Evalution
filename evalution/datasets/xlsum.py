# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from functools import lru_cache
from pathlib import Path, PurePosixPath
from shutil import copyfileobj, move, rmtree
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from huggingface_hub import hf_hub_download

# Pin the XLSum archive we audit locally so evaluation never executes the remote dataset script.
XLSUM_DATASET_PATH = "csebuetnlp/xlsum"
XLSUM_ARCHIVES = {
    "spanish": {
        "filename": "data/spanish_XLSum_v2.0.tar.bz2",
        "sha256": "70499154fe1d1c8df3b4667921d2c8c7b508da5473aa9387c4330b3b22288360",
        "split_files": {
            "train": "spanish_train.jsonl",
            "validation": "spanish_val.jsonl",
            "test": "spanish_test.jsonl",
        },
    }
}
_XLSUM_SPLIT_ALIASES = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "test": "test",
}
_XLSUM_EXTRACT_LOCK = Lock()


def _default_cache_root() -> Path:
    return Path.home() / ".cache" / "evalution" / "datasets"


def _cache_root(cache_dir: str | None) -> Path:
    if cache_dir is None:
        return _default_cache_root()
    return Path(cache_dir) / "evalution-vendored-datasets"


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
    resolved = _XLSUM_SPLIT_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            f"unsupported xlsum split: {split!r}; expected one of {tuple(_XLSUM_SPLIT_ALIASES)}"
        )
    return resolved


def _archive_spec(dataset_name: str) -> dict[str, Any]:
    spec = XLSUM_ARCHIVES.get(dataset_name)
    if spec is None:
        raise ValueError(f"unsupported xlsum dataset_name: {dataset_name!r}")
    return spec


def _download_archive(
    dataset_path: str,
    dataset_name: str,
    *,
    cache_dir: str | None,
) -> Path:
    if dataset_path != XLSUM_DATASET_PATH:
        raise ValueError(f"xlsum dataset_path must be {XLSUM_DATASET_PATH!r}, got {dataset_path!r}")
    spec = _archive_spec(dataset_name)
    archive_path = Path(
        hf_hub_download(
            repo_id=dataset_path,
            filename=str(spec["filename"]),
            repo_type="dataset",
            cache_dir=str(_cache_root(cache_dir)),
        )
    )
    actual_sha256 = _sha256_file(archive_path)
    expected_sha256 = str(spec["sha256"])
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "xlsum archive checksum mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return archive_path


def _safe_extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        for member in archive.getmembers():
            normalized_name = member.name[2:] if member.name.startswith("./") else member.name
            member_path = PurePosixPath(normalized_name)
            if not member_path.parts:
                continue
            if member_path.is_absolute():
                raise ValueError(f"xlsum archive contains absolute path: {member.name!r}")
            if any(part == ".." for part in member_path.parts):
                raise ValueError(f"xlsum archive contains unsafe path traversal: {member.name!r}")
            if member.islnk() or member.issym() or member.isdev():
                raise ValueError(f"xlsum archive contains unsupported link/device entry: {member.name!r}")

            target_path = destination.joinpath(*member_path.parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"xlsum archive contains unsupported tar entry: {member.name!r}")

            extracted_file = archive.extractfile(member)
            if extracted_file is None:
                raise ValueError(f"xlsum archive could not extract file: {member.name!r}")
            with target_path.open("wb") as handle:
                copyfileobj(extracted_file, handle)


def _extracted_root(dataset_name: str, cache_dir: str | None) -> Path:
    spec = _archive_spec(dataset_name)
    return _cache_root(cache_dir) / f"xlsum-{dataset_name}-{str(spec['sha256'])[:12]}"


def ensure_local_xlsum_archive(dataset_name: str, cache_dir: str | None = None) -> Path:
    extracted_root = _extracted_root(dataset_name, cache_dir)
    marker_path = extracted_root / ".complete"
    split_file = extracted_root / str(_archive_spec(dataset_name)["split_files"]["test"])
    if marker_path.exists() and split_file.exists():
        return extracted_root

    with _XLSUM_EXTRACT_LOCK:
        if marker_path.exists() and split_file.exists():
            return extracted_root

        archive_path = _download_archive(XLSUM_DATASET_PATH, dataset_name, cache_dir=cache_dir)
        extracted_root.parent.mkdir(parents=True, exist_ok=True)
        if extracted_root.exists():
            rmtree(extracted_root)

        with TemporaryDirectory(dir=extracted_root.parent) as temporary_dir:
            temporary_root = Path(temporary_dir) / extracted_root.name
            _safe_extract_archive(archive_path, temporary_root)
            (temporary_root / ".complete").write_text("ok\n", encoding="utf-8")
            move(str(temporary_root), str(extracted_root))
        return extracted_root


def _normalize_inline_spaces(text: str) -> str:
    return re.sub(r" +", " ", text).strip()


@lru_cache(maxsize=None)
def _load_rows_cached(
    *,
    dataset_name: str,
    split: str,
    cache_root_key: str,
) -> tuple[dict[str, Any], ...]:
    dataset_root = ensure_local_xlsum_archive(dataset_name, cache_root_key or None)
    split_file = dataset_root / str(_archive_spec(dataset_name)["split_files"][split])
    rows: list[dict[str, Any]] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            rows.append(
                {
                    "id": str(doc["id"]),
                    "url": str(doc["url"]),
                    "title": _normalize_inline_spaces(str(doc["title"])),
                    "summary": _normalize_inline_spaces(str(doc["summary"])),
                    "text": _normalize_inline_spaces(str(doc["text"])),
                }
            )
    return tuple(rows)


def load_xlsum_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> list[dict[str, Any]]:
    if dataset_name is None:
        raise ValueError("xlsum dataset_name is required")
    if stream:
        raise ValueError("xlsum vendored loader does not support stream=True")
    resolved_split = _resolved_split(split)
    return list(
        _load_rows_cached(
            dataset_name=dataset_name,
            split=resolved_split,
            cache_root_key=cache_dir or "",
        )
    )
