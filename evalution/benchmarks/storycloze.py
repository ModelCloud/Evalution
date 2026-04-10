# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep the public StoryCloze surface small and explicit because upstream only ships two yearly releases.
STORYCLOZE_YEARS = ("2016", "2018")
STORYCLOZE_TASKS = tuple(f"storycloze_{year}" for year in STORYCLOZE_YEARS)


def _storycloze_prompt(doc: dict[str, Any]) -> str:
    # Concatenate the observed story sentences once so each ending is scored against identical context.
    return " ".join(
        str(doc[field]).strip()
        for field in (
            "input_sentence_1",
            "input_sentence_2",
            "input_sentence_3",
            "input_sentence_4",
        )
    )


@dataclass(slots=True)
class StoryCloze(BaseMultipleChoiceSuite):
    # Mirror the public yearly StoryCloze releases while keeping dataset_name and year locked together.
    dataset_path: str = "LSDSem/story_cloze"
    dataset_name: str | None = "2016"
    split: str = "validation"
    year: str = "2016"

    def __post_init__(self) -> None:
        if self.year not in STORYCLOZE_YEARS:
            raise ValueError(f"unsupported storycloze year: {self.year!r}")
        if self.dataset_name in {None, self.year}:
            self.dataset_name = self.year
            return
        raise ValueError("storycloze dataset_name must match the configured year")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"storycloze_{self.year}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [
            str(doc["sentence_quiz1"]).strip(),
            str(doc["sentence_quiz2"]).strip(),
        ]
        return MultipleChoiceSample(
            index=index,
            prompt=_storycloze_prompt(doc),
            choices=choices,
            gold_index=int(doc["answer_right_ending"]) - 1,
            metadata={
                "year": self.year,
                "story_id": str(doc.get("story_id", "")),
                "input_sentences": [
                    str(doc["input_sentence_1"]).strip(),
                    str(doc["input_sentence_2"]).strip(),
                    str(doc["input_sentence_3"]).strip(),
                    str(doc["input_sentence_4"]).strip(),
                ],
                "choice_texts": choices,
            },
        )


def storycloze(*, year: str = "2016", **kwargs: Any) -> StoryCloze:
    # Default the dataset config to the requested public release year.
    kwargs.setdefault("dataset_name", year)
    return StoryCloze(year=year, **kwargs)


def _make_storycloze_factory(year: str) -> Any:
    # Publish one import-stable zero-argument factory per yearly release.
    def factory(**kwargs: Any) -> StoryCloze:
        return storycloze(year=year, **kwargs)

    factory.__name__ = f"storycloze_{year}"
    return factory


for _year in STORYCLOZE_YEARS:
    globals()[f"storycloze_{_year}"] = _make_storycloze_factory(_year)

del _year
