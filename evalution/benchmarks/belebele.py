# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_BELEBELE_LABELS = ("A", "B", "C", "D")
# Add the explicit regional-suite aliases that upstream family bundles reference directly.
BELEBELE_LANGUAGE_TASKS = (
    "belebele_por_Latn",
    "belebele_spa_Latn",
)


def _belebele_prompt(doc: dict[str, Any]) -> str:
    return (
        f"P: {str(doc['flores_passage']).strip()}\n"
        f"Q: {str(doc['question']).strip()}\n"
        f"A: {str(doc['mc_answer1']).strip()}\n"
        f"B: {str(doc['mc_answer2']).strip()}\n"
        f"C: {str(doc['mc_answer3']).strip()}\n"
        f"D: {str(doc['mc_answer4']).strip()}\n"
        "Answer:"
    )


@dataclass(slots=True)
class Belebele(BaseMultipleChoiceSuite):
    dataset_path: str = "facebook/belebele"
    dataset_name: str | None = "eng_Latn"
    split: str = "test"
    language: str = "eng_Latn"

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise ValueError("belebele language must be a non-empty dataset config")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
            return
        raise ValueError("belebele dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"belebele_{self.language}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        raw_choices = [str(doc[f"mc_answer{i}"]).strip() for i in range(1, 5)]
        correct_answer_num = str(doc["correct_answer_num"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_belebele_prompt(doc),
            choices=list(_BELEBELE_LABELS),
            gold_index=int(correct_answer_num) - 1,
            metadata={
                "language": self.language,
                "dialect": str(doc["dialect"]).strip(),
                "question_number": int(doc["question_number"]),
                "link": str(doc["link"]).strip(),
                "passage": str(doc["flores_passage"]).strip(),
                "question": str(doc["question"]).strip(),
                "raw_choices": raw_choices,
                "correct_answer_num": correct_answer_num,
            },
        )


def belebele(*, language: str, **kwargs: Any) -> Belebele:
    kwargs.setdefault("dataset_name", language)
    return Belebele(language=language, **kwargs)


def _make_belebele_language_factory(language: str) -> Any:
    def factory(**kwargs: Any) -> Belebele:
        return belebele(language=language, **kwargs)

    factory.__name__ = f"belebele_{language}"
    return factory


for _language in ("por_Latn", "spa_Latn"):
    globals()[f"belebele_{_language}"] = _make_belebele_language_factory(_language)

del _language
