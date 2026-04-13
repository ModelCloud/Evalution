# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from shutil import move
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile

# Pin the public MEDIQA 2019 zip so evaluation never executes the remote BigBio dataset script.
MEDIQA_QA_DATASET_PATH = "bigbio/mediqa_qa"
MEDIQA_QA_DATASET_NAME = "mediqa_qa_source"
MEDIQA_QA_SOURCE_URL = "https://github.com/abachaa/MEDIQA2019/archive/refs/heads/master.zip"
MEDIQA_QA_SOURCE_SHA256 = "c078ff1fe9132cf5eb89234233b1c61edeb39af21444cf5f8d04b737df52c11f"
MEDIQA_QA_SPLIT_FILES = {
    "train_live_qa_med": "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TrainingSet1-LiveQAMed.xml",
    "train_alexa": "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TrainingSet2-Alexa.xml",
    "validation": "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-ValidationSet.xml",
    "test": "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TestSet-wLabels.xml",
}
_MEDIQA_QA_ROW_LOCK = Lock()


def _default_cache_root() -> Path:
    """Implement default cache root for this module."""
    return Path.home() / ".cache" / "evalution" / "datasets"


def _cache_root(cache_dir: str | None) -> Path:
    """Implement cache root for this module."""
    if cache_dir is None:
        return _default_cache_root()
    return Path(cache_dir) / "evalution-vendored-datasets"


def _sha256_file(path: Path) -> str:
    """Implement sha256 file for this module."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _archive_member_for_split(split: str) -> str:
    """Implement archive member for split for this module."""
    try:
        return MEDIQA_QA_SPLIT_FILES[str(split).strip()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported mediqa_qa split: {split!r}; expected one of {tuple(MEDIQA_QA_SPLIT_FILES)}"
        ) from exc


def _download_source_zip(cache_dir: str | None) -> Path:
    """Implement download source zip for this module."""
    cache_root = _cache_root(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    target_path = cache_root / f"mediqa-qa-{MEDIQA_QA_SOURCE_SHA256[:12]}.zip"
    if target_path.exists():
        return target_path

    with NamedTemporaryFile(dir=cache_root, suffix=".zip", delete=False) as handle:
        temporary_path = Path(handle.name)
        with urlopen(MEDIQA_QA_SOURCE_URL) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    move(str(temporary_path), str(target_path))
    return target_path


def _verified_source_zip(cache_dir: str | None) -> Path:
    """Implement verified source zip for this module."""
    archive_path = _download_source_zip(cache_dir)
    actual_sha256 = _sha256_file(archive_path)
    if actual_sha256 != MEDIQA_QA_SOURCE_SHA256:
        raise ValueError(
            "mediqa_qa source checksum mismatch: "
            f"expected {MEDIQA_QA_SOURCE_SHA256}, got {actual_sha256}"
        )
    return archive_path


def _answer_blob(answer: ET.Element) -> dict[str, Any]:
    # Preserve the source answer metadata because downstream scoring may want rank provenance later.
    """Implement answer blob for this module."""
    return {
        "Answer": {
            "AID": str(answer.attrib.get("AID", "")),
            "SystemRank": int(answer.attrib.get("SystemRank", "0")),
            "ReferenceRank": int(answer.attrib.get("ReferenceRank", "0")),
            "ReferenceScore": int(answer.attrib.get("ReferenceScore", "0")),
            "AnswerURL": str(answer.findtext("AnswerURL") or ""),
            "AnswerText": str(answer.findtext("AnswerText") or ""),
        }
    }


def _load_source_rows(archive_path: Path, split: str) -> list[dict[str, Any]]:
    """Load source rows."""
    member_name = _archive_member_for_split(split)
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as member:
            root = ET.parse(member).getroot()

    rows: list[dict[str, Any]] = []
    for question in root.findall("Question"):
        answers = question.find("AnswerList")
        if answers is None:
            continue
        rows.append(
            {
                "QUESTION": {
                    "QID": str(question.attrib.get("QID", "")),
                    "QuestionText": str(question.findtext("QuestionText") or ""),
                    "AnswerList": [_answer_blob(answer) for answer in answers.findall("Answer")],
                }
            }
        )
    return rows


@lru_cache(maxsize=None)
def _load_rows_cached(*, split: str, cache_root_key: str) -> tuple[dict[str, Any], ...]:
    """Load rows cached."""
    with _MEDIQA_QA_ROW_LOCK:
        rows = _load_source_rows(_verified_source_zip(cache_root_key or None), split)
    return tuple(rows)


def load_mediqa_qa_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> list[dict[str, Any]]:
    """Load mediqa QA dataset."""
    if dataset_path != MEDIQA_QA_DATASET_PATH:
        raise ValueError(
            f"mediqa_qa dataset_path must be {MEDIQA_QA_DATASET_PATH!r}, got {dataset_path!r}"
        )
    if dataset_name not in {None, MEDIQA_QA_DATASET_NAME}:
        raise ValueError(
            f"mediqa_qa dataset_name must be {MEDIQA_QA_DATASET_NAME!r}, got {dataset_name!r}"
        )
    if stream:
        raise ValueError("mediqa_qa vendored loader does not support stream=True")
    resolved_split = str(split).strip()
    _archive_member_for_split(resolved_split)
    return list(
        _load_rows_cached(
            split=resolved_split,
            cache_root_key=cache_dir or "",
        )
    )
