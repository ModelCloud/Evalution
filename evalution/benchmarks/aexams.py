# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from zipfile import ZipFile
from typing import Any

from datasets import Dataset
from huggingface_hub import hf_hub_download

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.benchmarks.multiple_choice_utils import choice_index_from_labels

AEXAMS_SUBJECTS = (
    "biology",
    "islamic_studies",
    "physics",
    "science",
    "social",
)

_AEXAMS_LABELS = ("A", "B", "C", "D")
_AEXAMS_SUBJECT_SPECS = {
    "biology": {
        "dataset_name": "Biology",
        "description": "قم بالإجابة على مايلي في مجال العلوم الحيوية",
    },
    "islamic_studies": {
        "dataset_name": "IslamicStudies",
        "description": "قم بالإجابة على مايلي في مجال العلوم الإسلامية",
    },
    "physics": {
        "dataset_name": "Physics",
        "description": "قم بالإجابة على مايلي في مجال الفيزياء",
    },
    "science": {
        "dataset_name": "Science",
        "description": "قم بالإجابة على مايلي في مجال العلوم",
    },
    "social": {
        "dataset_name": "Social",
        "description": "قم بالإجابة على مايلي في مجال العلوم الإجتماعية",
    },
}


@lru_cache(maxsize=1)
def _aexams_archive_path() -> str:
    return hf_hub_download(
        repo_id="Hennara/aexams",
        filename="aexams_v0.zip",
        repo_type="dataset",
    )


@lru_cache(maxsize=None)
def _aexams_rows(dataset_name: str, split: str) -> tuple[dict[str, Any], ...]:
    member_name = f"{split}/{dataset_name}.jsonl"
    rows: list[dict[str, Any]] = []
    with ZipFile(_aexams_archive_path()) as archive:
        with archive.open(member_name) as rows_file:
            for raw_line in rows_file:
                rows.append(json.loads(raw_line))
    return tuple(rows)


def _load_aexams_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> Dataset:
    del cache_dir
    if dataset_path != "Hennara/aexams":
        raise ValueError(f"unsupported AEXAMS dataset path: {dataset_path!r}")
    if dataset_name is None:
        raise ValueError("AEXAMS dataset_name is required")
    if split not in {"test", "dev"}:
        raise ValueError(f"unsupported AEXAMS split: {split!r}")
    if stream:
        raise ValueError("AEXAMS raw zip loader requires non-stream dataset materialization")
    return Dataset.from_list(list(_aexams_rows(dataset_name, split)))


def _aexams_prompt(description: str, question: str, option_texts: list[str]) -> str:
    lines = [description.strip(), "", question.strip()]
    for label, option_text in zip(_AEXAMS_LABELS, option_texts, strict=True):
        lines.append(f"{label}. {option_text.strip()}")
    lines.append("الجواب:")
    return "\n".join(lines)


@dataclass(slots=True)
class AEXAMS(BaseMultipleChoiceSuite):
    dataset_path: str = "Hennara/aexams"
    dataset_name: str | None = None
    split: str = "test"
    stream: bool = False
    subject: str = ""

    def __post_init__(self) -> None:
        if self.subject not in AEXAMS_SUBJECTS:
            raise ValueError(f"unsupported aexams subject: {self.subject!r}")
        expected_dataset_name = _AEXAMS_SUBJECT_SPECS[self.subject]["dataset_name"]
        if self.dataset_name in {None, expected_dataset_name}:
            self.dataset_name = expected_dataset_name
            return
        raise ValueError("aexams dataset_name must match the configured subject")

    def dataset_loader(self) -> Any:
        return _load_aexams_dataset

    def task_name(self) -> str:
        return f"aexams_{self.subject}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choice_texts = [str(doc[label]).strip() for label in _AEXAMS_LABELS]
        question = str(doc["question"]).strip()
        description = str(_AEXAMS_SUBJECT_SPECS[self.subject]["description"])
        answer_label = str(doc["answer"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_aexams_prompt(description, question, choice_texts),
            choices=list(_AEXAMS_LABELS),
            gold_index=choice_index_from_labels(list(_AEXAMS_LABELS), answer_label),
            metadata={
                "subject": self.subject,
                "question": question,
                "choice_labels": list(_AEXAMS_LABELS),
                "choice_texts": choice_texts,
                "answer_label": answer_label,
            },
        )

    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        choice_texts = list(sample.metadata["choice_texts"])
        description = str(_AEXAMS_SUBJECT_SPECS[self.subject]["description"])
        lines = [description, "", str(sample.metadata["question"])]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {choice_texts[choice_index]}")
        lines.append("الجواب:")
        return "\n".join(lines)


def aexams(*, subject: str, **kwargs: Any) -> AEXAMS:
    return AEXAMS(subject=subject, **kwargs)


def aexams_biology(**kwargs: Any) -> AEXAMS:
    return aexams(subject="biology", **kwargs)


def aexams_islamic_studies(**kwargs: Any) -> AEXAMS:
    return aexams(subject="islamic_studies", **kwargs)


def aexams_physics(**kwargs: Any) -> AEXAMS:
    return aexams(subject="physics", **kwargs)


def aexams_science(**kwargs: Any) -> AEXAMS:
    return aexams(subject="science", **kwargs)


def aexams_social(**kwargs: Any) -> AEXAMS:
    return aexams(subject="social", **kwargs)
