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

_PUBMEDQA_ARCHIVE_URL = "https://huggingface.co/datasets/bigbio/pubmed_qa/resolve/main/pqal.zip"
_PUBMEDQA_ARCHIVE_NAME = "pqal.zip"
_PUBMEDQA_SPLIT_FILES = {
    "train": "pqal_fold0/train_set.json",
    "validation": "pqal_fold0/dev_set.json",
    "test": "pqal_test_set.json",
}
_PUBMEDQA_CHOICES = ["yes", "no", "maybe"]


def _pubmedqa_prompt(question: str, contexts: list[str]) -> str:
    abstract = "\n".join(str(context).strip() for context in contexts if str(context).strip())
    return f"Abstract: {abstract}\nQuestion: {question.strip()}\nAnswer:"


def _pubmedqa_cache_path(cache_dir: str | None) -> Path:
    if cache_dir is not None:
        base_dir = Path(cache_dir)
    else:
        base_dir = Path.home() / ".cache" / "evalution" / "downloads"
    return base_dir / _PUBMEDQA_ARCHIVE_NAME


def _ensure_pubmedqa_archive(*, cache_dir: str | None) -> Path:
    archive_path = _pubmedqa_cache_path(cache_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        urlretrieve(_PUBMEDQA_ARCHIVE_URL, archive_path)
    return archive_path


def _load_pubmedqa_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = True,
) -> Dataset:
    del stream
    if dataset_path != "bigbio/pubmed_qa":
        raise ValueError(f"unsupported PubMedQA dataset path: {dataset_path!r}")
    if dataset_name != "pubmed_qa_labeled_fold0_source":
        raise ValueError(f"unsupported PubMedQA dataset name: {dataset_name!r}")

    split_member = _PUBMEDQA_SPLIT_FILES.get(split)
    if split_member is None:
        raise ValueError(f"unsupported PubMedQA split: {split!r}")

    archive_path = _ensure_pubmedqa_archive(cache_dir=cache_dir)
    with ZipFile(archive_path) as archive:
        with archive.open(split_member) as split_file:
            rows = json.load(split_file)
    normalized_rows = []
    for row_id, row in rows.items():
        normalized_row = dict(row)
        normalized_row["PUBID"] = str(row_id)
        normalized_rows.append(normalized_row)
    return Dataset.from_list(normalized_rows)


@dataclass(slots=True)
class PubMedQA(BaseMultipleChoiceSuite):
    dataset_path: str = "bigbio/pubmed_qa"
    dataset_name: str | None = "pubmed_qa_labeled_fold0_source"
    split: str = "test"
    stream: bool = True

    def dataset_loader(self) -> Any:
        return _load_pubmedqa_dataset

    def task_name(self) -> str:
        return "pubmedqa"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        contexts = [str(item) for item in doc["CONTEXTS"]]
        question = str(doc["QUESTION"]).strip()
        answer = str(doc["final_decision"]).strip().lower()
        return MultipleChoiceSample(
            index=index,
            prompt=_pubmedqa_prompt(question, contexts),
            choices=list(_PUBMEDQA_CHOICES),
            gold_index=_PUBMEDQA_CHOICES.index(answer),
            metadata={
                "pubid": str(doc.get("PUBID", "")),
                "question": question,
                "context_labels": [str(label) for label in doc["LABELS"]],
                "meshes": [str(mesh) for mesh in doc["MESHES"]],
                "reasoning_required_pred": str(doc["reasoning_required_pred"]),
                "reasoning_free_pred": str(doc["reasoning_free_pred"]),
                "long_answer": str(doc["LONG_ANSWER"]).strip(),
                "choice_labels": list(_PUBMEDQA_CHOICES),
                "choice_texts": list(_PUBMEDQA_CHOICES),
            },
        )


def pubmedqa(**kwargs: Any) -> PubMedQA:
    return PubMedQA(**kwargs)
