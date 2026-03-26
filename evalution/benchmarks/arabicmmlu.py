# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Frozen upstream config snapshot for import safety. Refresh deliberately if the dataset adds
# or removes subsets.
ARABICMMLU_SUBSETS = (
    "All",
    "Islamic Studies",
    "Islamic Studies (Middle School)",
    "Islamic Studies (Primary School)",
    "Islamic Studies (High School)",
    "Driving Test",
    "Natural Science (Middle School)",
    "Natural Science (Primary School)",
    "History (Middle School)",
    "History (Primary School)",
    "History (High School)",
    "General Knowledge",
    "General Knowledge (Middle School)",
    "General Knowledge (Primary School)",
    "Law (Professional)",
    "Physics (High School)",
    "Social Science (Middle School)",
    "Social Science (Primary School)",
    "Management (University)",
    "Arabic Language (Middle School)",
    "Arabic Language (Primary School)",
    "Arabic Language (High School)",
    "Political Science (University)",
    "Philosophy (High School)",
    "Accounting (University)",
    "Computer Science (Middle School)",
    "Computer Science (Primary School)",
    "Computer Science (High School)",
    "Computer Science (University)",
    "Geography (Middle School)",
    "Geography (Primary School)",
    "Geography (High School)",
    "Math (Primary School)",
    "Biology (High School)",
    "Economics (Middle School)",
    "Economics (High School)",
    "Economics (University)",
    "Arabic Language (General)",
    "Arabic Language (Grammar)",
    "Civics (Middle School)",
    "Civics (High School)",
)
_OPTION_FIELDS = ("Option 1", "Option 2", "Option 3", "Option 4", "Option 5")
_LEVEL_TEXT = {
    "Primary": "primary school",
    "Middle": "middle school",
    "High": "high school",
    "Univ": "university",
    "Prof": "professional",
}


def _slugify_subset_name(subset: str) -> str:
    slug = subset.lower()
    for old, new in (
        ("(", ""),
        (")", ""),
        ("/", " "),
        ("-", " "),
        ("&", " and "),
    ):
        slug = slug.replace(old, new)
    slug = "_".join(part for part in slug.replace(",", " ").split())
    return slug


ARABICMMLU_TASKS = tuple(f"arabicmmlu_{_slugify_subset_name(subset)}" for subset in ARABICMMLU_SUBSETS)
_SUBSET_TO_TASK = dict(zip(ARABICMMLU_SUBSETS, ARABICMMLU_TASKS, strict=True))


def _arabicmmlu_prompt(doc: dict[str, Any]) -> str:
    level = ""
    if doc["Level"]:
        level = f" for {_LEVEL_TEXT[str(doc['Level'])]}"
    country = ""
    if doc["Country"]:
        country = f" in {str(doc['Country'])}"
    metadata = f"{str(doc['Subject'])} question{level}{country}"
    question = str(doc["Question"]).strip()
    if doc["Context"]:
        question = f"{str(doc['Context']).strip()}\n\n{question}"
    options = []
    labels = []
    for index, field in enumerate(_OPTION_FIELDS):
        value = doc[field]
        if value is None or str(value).strip() == "":
            break
        label = chr(ord("A") + index)
        labels.append(label)
        options.append(f"{label}. {str(value).strip()}")
    return (
        f"This is a {metadata}. Select the correct answer!\n\n"
        f"Question: {question}\n"
        f"{chr(10).join(options)}\n\n"
        "Answer:"
    )


@dataclass(slots=True)
class ArabicMMLU(BaseMultipleChoiceSuite):
    """ArabicMMLU suite backed by a frozen subset registry to keep imports offline-safe."""

    dataset_path: str = "MBZUAI/ArabicMMLU"
    dataset_name: str | None = None
    split: str = "test"
    subset: str = ""

    def __post_init__(self) -> None:
        if self.subset not in ARABICMMLU_SUBSETS:
            raise ValueError(f"unsupported arabicmmlu subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("arabicmmlu dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return _SUBSET_TO_TASK[self.subset]

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        raw_choices = []
        choice_labels = []
        for option_index, field in enumerate(_OPTION_FIELDS):
            value = doc[field]
            if value is None or str(value).strip() == "":
                break
            raw_choices.append(str(value).strip())
            choice_labels.append(chr(ord("A") + option_index))
        answer_label = str(doc["Answer Key"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_arabicmmlu_prompt(doc),
            choices=choice_labels,
            gold_index=choice_labels.index(answer_label),
            metadata={
                "id": int(doc["ID"]),
                "subset": self.subset,
                "group": str(doc["Group"]),
                "subject": str(doc["Subject"]),
                "level": None if doc["Level"] is None else str(doc["Level"]),
                "country": None if doc["Country"] is None else str(doc["Country"]),
                "question": str(doc["Question"]).strip(),
                "context": None if doc["Context"] is None else str(doc["Context"]).strip(),
                "answer_label": answer_label,
                "choice_labels": choice_labels,
                "raw_choices": raw_choices,
            },
        )


def arabicmmlu(*, subset: str, **kwargs: Any) -> ArabicMMLU:
    return ArabicMMLU(subset=subset, dataset_name=subset, **kwargs)


def _make_arabicmmlu_factory(subset: str) -> Any:
    def factory(**kwargs: Any) -> ArabicMMLU:
        return arabicmmlu(subset=subset, **kwargs)

    factory.__name__ = _SUBSET_TO_TASK[subset]
    return factory


for _subset in ARABICMMLU_SUBSETS:
    globals()[_SUBSET_TO_TASK[_subset]] = _make_arabicmmlu_factory(_subset)

del _subset
