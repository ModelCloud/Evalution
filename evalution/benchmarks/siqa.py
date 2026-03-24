# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
from zipfile import ZipFile

from datasets import Dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_SOCIAL_IQA_ARCHIVE_URL = (
    "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
)
_SOCIAL_IQA_ARCHIVE_NAME = "socialiqa-train-dev.zip"
_SOCIAL_IQA_SPLIT_FILES = {
    "train": (
        "socialiqa-train-dev/train.jsonl",
        "socialiqa-train-dev/train-labels.lst",
    ),
    "validation": (
        "socialiqa-train-dev/dev.jsonl",
        "socialiqa-train-dev/dev-labels.lst",
    ),
}


def _social_iqa_prompt(context: str, question: str) -> str:
    return f"Q: {context.strip()} {question.strip()}\nA:"


def _social_iqa_cache_path(cache_dir: str | None) -> Path:
    if cache_dir is not None:
        base_dir = Path(cache_dir)
    else:
        base_dir = Path.home() / ".cache" / "evalution" / "downloads"
    return base_dir / _SOCIAL_IQA_ARCHIVE_NAME


def _ensure_social_iqa_archive(*, cache_dir: str | None) -> Path:
    archive_path = _social_iqa_cache_path(cache_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        urlretrieve(_SOCIAL_IQA_ARCHIVE_URL, archive_path)
    return archive_path


def _load_social_iqa_dataset(
    dataset_path: str,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> Dataset:
    del stream
    if dataset_path != "allenai/social_i_qa":
        raise ValueError(f"unsupported Social IQA dataset path: {dataset_path!r}")

    split_files = _SOCIAL_IQA_SPLIT_FILES.get(split)
    if split_files is None:
        raise ValueError(f"unsupported Social IQA split: {split!r}")
    jsonl_member, labels_member = split_files

    archive_path = _ensure_social_iqa_archive(cache_dir=cache_dir)
    with ZipFile(archive_path) as archive:
        with archive.open(labels_member) as labels_file:
            labels = [line.decode("utf-8").strip() for line in labels_file if line.strip()]
        rows: list[dict[str, str]] = []
        with archive.open(jsonl_member) as rows_file:
            for index, line in enumerate(rows_file):
                row = json.loads(line)
                row["label"] = labels[index]
                rows.append(row)
    return Dataset.from_list(rows)


@dataclass(slots=True)
class SIQA(BaseMultipleChoiceSuite):
    dataset_path: str = "allenai/social_i_qa"
    split: str = "validation"
    stream: bool = False

    def dataset_loader(self) -> Any:
        return _load_social_iqa_dataset

    def task_name(self) -> str:
        return "siqa"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
        return MultipleChoiceSample(
            index=index,
            prompt=_social_iqa_prompt(str(doc["context"]), str(doc["question"])),
            choices=choices,
            gold_index=int(doc["label"]) - 1,
            metadata={
                "context": str(doc["context"]),
                "question": str(doc["question"]),
                "choice_texts": [str(choice) for choice in choices],
            },
        )


def siqa(**kwargs: Any) -> SIQA:
    return SIQA(**kwargs)
