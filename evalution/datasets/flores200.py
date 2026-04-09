# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import csv
import hashlib
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath
from shutil import copyfileobj, move, rmtree
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Lock
from typing import Any
from urllib.request import urlopen

# Pin the resolved FLORES-200 archive instead of trusting the historic tinyurl redirect used by
# the original dataset script. This keeps the data path explicit and auditable inside the repo.
FLORES200_ARCHIVE_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES200_ARCHIVE_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
FLORES200_LANGUAGE_CODES = frozenset(
    {
        "cat_Latn",
        "deu_Latn",
        "eng_Latn",
        "eus_Latn",
        "fra_Latn",
        "glg_Latn",
        "ita_Latn",
        "por_Latn",
        "spa_Latn",
    }
)
_FLORES200_DATASET_ROOT = "flores200_dataset"
_FLORES200_ARCHIVE_NAME = "flores200_dataset.tar.gz"
_FLORES200_SPLIT_ALIASES = {
    "dev": "dev",
    "validation": "dev",
    "devtest": "devtest",
    "test": "devtest",
}
# Serialize archive extraction so concurrent suites do not race on the same cache path.
_FLORES200_EXTRACT_LOCK = Lock()


def _default_cache_root() -> Path:
    return Path.home() / ".cache" / "evalution" / "datasets"


def _cache_root(cache_dir: str | None) -> Path:
    if cache_dir is None:
        return _default_cache_root()
    return Path(cache_dir) / "evalution-vendored-datasets"


def _archive_path(cache_dir: str | None) -> Path:
    return _cache_root(cache_dir) / _FLORES200_ARCHIVE_NAME


def _extracted_root(cache_dir: str | None) -> Path:
    return _cache_root(cache_dir) / f"flores200-extracted-{FLORES200_ARCHIVE_SHA256[:12]}"


def _dataset_root(cache_dir: str | None) -> Path:
    return _extracted_root(cache_dir) / _FLORES200_DATASET_ROOT


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_archive(cache_dir: str | None) -> Path:
    archive_path = _archive_path(cache_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and _sha256_file(archive_path) == FLORES200_ARCHIVE_SHA256:
        return archive_path

    with NamedTemporaryFile(dir=archive_path.parent, delete=False) as temporary_handle:
        temporary_path = Path(temporary_handle.name)
        with urlopen(FLORES200_ARCHIVE_URL, timeout=120) as response:
            copyfileobj(response, temporary_handle)

    archive_digest = _sha256_file(temporary_path)
    if archive_digest != FLORES200_ARCHIVE_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            "flores200 archive checksum mismatch: "
            f"expected {FLORES200_ARCHIVE_SHA256}, got {archive_digest}"
        )

    temporary_path.replace(archive_path)
    return archive_path


def _safe_extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            normalized_name = member.name[2:] if member.name.startswith("./") else member.name
            member_path = PurePosixPath(normalized_name)
            if not member_path.parts:
                continue
            if member_path.is_absolute():
                raise ValueError(f"flores200 archive contains absolute path: {member.name!r}")
            if any(part == ".." for part in member_path.parts):
                raise ValueError(f"flores200 archive contains unsafe path traversal: {member.name!r}")
            if member.islnk() or member.issym() or member.isdev():
                raise ValueError(f"flores200 archive contains unsupported link/device entry: {member.name!r}")

            target_path = destination.joinpath(*member_path.parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"flores200 archive contains unsupported tar entry: {member.name!r}")

            extracted_file = archive.extractfile(member)
            if extracted_file is None:
                raise ValueError(f"flores200 archive could not extract file: {member.name!r}")
            with target_path.open("wb") as handle:
                copyfileobj(extracted_file, handle)


def ensure_local_flores200(cache_dir: str | None = None) -> Path:
    dataset_root = _dataset_root(cache_dir)
    marker_path = _extracted_root(cache_dir) / ".complete"
    if marker_path.exists() and dataset_root.exists():
        return dataset_root

    with _FLORES200_EXTRACT_LOCK:
        if marker_path.exists() and dataset_root.exists():
            return dataset_root

        archive_path = _download_archive(cache_dir)
        final_root = _extracted_root(cache_dir)
        final_root.parent.mkdir(parents=True, exist_ok=True)
        if final_root.exists():
            rmtree(final_root)

        with TemporaryDirectory(dir=final_root.parent) as temporary_dir:
            temporary_root = Path(temporary_dir) / final_root.name
            _safe_extract_archive(archive_path, temporary_root)
            if not (temporary_root / _FLORES200_DATASET_ROOT).exists():
                raise ValueError("flores200 archive did not contain the expected dataset root")
            (temporary_root / ".complete").write_text("ok\n", encoding="utf-8")
            move(str(temporary_root), str(final_root))
        return dataset_root


def _resolved_split(split: str) -> str:
    normalized = split.strip().lower()
    resolved = _FLORES200_SPLIT_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            f"unsupported flores200 split: {split!r}; expected one of {tuple(_FLORES200_SPLIT_ALIASES)}"
        )
    return resolved


@lru_cache(maxsize=None)
def _load_pair_docs_cached(
    *,
    split: str,
    source_language: str,
    target_language: str,
    cache_root_key: str,
) -> tuple[dict[str, Any], ...]:
    dataset_root = ensure_local_flores200(cache_root_key or None)
    metadata_path = dataset_root / f"metadata_{split}.tsv"
    source_path = dataset_root / split / f"{source_language}.{split}"
    target_path = dataset_root / split / f"{target_language}.{split}"

    with metadata_path.open("r", encoding="utf-8") as metadata_handle:
        metadata_rows = list(csv.DictReader(metadata_handle, delimiter="\t"))
    with source_path.open("r", encoding="utf-8") as source_handle:
        source_sentences = [line.rstrip("\n") for line in source_handle]
    with target_path.open("r", encoding="utf-8") as target_handle:
        target_sentences = [line.rstrip("\n") for line in target_handle]

    if not (len(metadata_rows) == len(source_sentences) == len(target_sentences)):
        raise ValueError(
            "flores200 row-count mismatch: "
            f"metadata={len(metadata_rows)} source={len(source_sentences)} target={len(target_sentences)}"
        )

    docs: list[dict[str, Any]] = []
    for index, (metadata, source_sentence, target_sentence) in enumerate(
        zip(metadata_rows, source_sentences, target_sentences, strict=True),
        start=1,
    ):
        docs.append(
            {
                "id": index,
                "URL": metadata["URL"],
                "domain": metadata["domain"],
                "topic": metadata["topic"],
                "has_image": metadata["has_image"].strip().lower() == "yes",
                "has_hyperlink": metadata["has_hyperlink"].strip().lower() == "yes",
                f"sentence_{source_language}": source_sentence,
                f"sentence_{target_language}": target_sentence,
            }
        )
    return tuple(docs)


def load_flores200_pair(
    dataset_path: str,
    dataset_name: str | None = None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
    source_language: str,
    target_language: str,
) -> list[dict[str, Any]]:
    if dataset_path != "facebook/flores":
        raise ValueError(f"flores200 dataset_path must be 'facebook/flores', got {dataset_path!r}")
    if dataset_name not in {None, "all"}:
        raise ValueError("flores200 dataset_name must be None or 'all'")
    if stream:
        raise ValueError("flores200 vendored loader does not support stream=True")
    if source_language not in FLORES200_LANGUAGE_CODES:
        raise ValueError(f"unsupported flores200 source language: {source_language!r}")
    if target_language not in FLORES200_LANGUAGE_CODES:
        raise ValueError(f"unsupported flores200 target language: {target_language!r}")

    resolved_split = _resolved_split(split)
    return list(
        _load_pair_docs_cached(
            split=resolved_split,
            source_language=source_language,
            target_language=target_language,
            cache_root_key=cache_dir or "",
        )
    )
