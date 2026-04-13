# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _arabic_question_answer_prompt(question: str) -> str:
    """Implement arabic question answer prompt for this module."""
    return f"السؤال: {question.strip()}\nالجواب:"


@dataclass(slots=True)
class COPAArabic(BaseMultipleChoiceSuite):
    """Implement the copaarabic benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "Hennara/copa_ar"
    split: str = "test"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "copa_ar"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        return MultipleChoiceSample(
            index=index,
            prompt=_arabic_question_answer_prompt(doc["query"]),
            choices=[doc["sol1"].strip(), doc["sol2"].strip()],
            gold_index=int(doc["label"]),
            metadata={
                "source_benchmark": "copa",
                "query": doc["query"].strip(),
            },
        )


@dataclass(slots=True)
class PIQAArabic(BaseMultipleChoiceSuite):
    """Implement the piqaarabic benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "Hennara/pica_ar"
    split: str = "test"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "piqa_ar"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        return MultipleChoiceSample(
            index=index,
            prompt=_arabic_question_answer_prompt(doc["goal"]),
            choices=[doc["sol1"].strip(), doc["sol2"].strip()],
            gold_index=int(doc["label"]),
            metadata={
                "source_benchmark": "piqa",
                "goal": doc["goal"].strip(),
            },
        )


def copa_ar(**kwargs: Any) -> COPAArabic:
    """Implement COPA ar for this module."""
    return COPAArabic(**kwargs)


def piqa_ar(**kwargs: Any) -> PIQAArabic:
    """Implement PIQA ar for this module."""
    return PIQAArabic(**kwargs)
