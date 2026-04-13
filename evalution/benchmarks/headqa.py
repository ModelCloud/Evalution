# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _headqa_prompt(question: str) -> str:
    """Implement headqa prompt for this module."""
    return f"Question: {question.strip()}\nAnswer:"


@dataclass(slots=True)
class HEADQA(BaseMultipleChoiceSuite):
    # Evaluate healthcare-domain multiple-choice reasoning in English and Spanish.
    """Implement the headqa benchmark suite."""
    dataset_path: str = "EleutherAI/headqa"
    dataset_name: str | None = "en"
    split: str = "test"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        if self.dataset_name is None:
            return "headqa"
        return f"headqa_{self.dataset_name}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        answers = sorted(doc["answers"], key=lambda answer: int(answer["aid"]))
        choices = [answer["atext"].strip() for answer in answers]
        return MultipleChoiceSample(
            index=index,
            prompt=_headqa_prompt(doc["qtext"]),
            choices=choices,
            gold_index=int(doc["ra"]) - 1,
            metadata={
                "name": doc["name"],
                "year": doc["year"],
                "category": doc["category"],
                "qid": doc["qid"],
                "choice_labels": ["A", "B", "C", "D"],
                "choice_ids": [str(answer["aid"]) for answer in answers],
                "choice_texts": choices,
            },
        )


def _headqa_variant(dataset_name: str, **kwargs: Any) -> HEADQA:
    """Implement headqa variant for this module."""
    return HEADQA(dataset_name=dataset_name, **kwargs)


def headqa_en(**kwargs: Any) -> HEADQA:
    """Implement headqa en for this module."""
    return _headqa_variant("en", **kwargs)


def headqa_es(**kwargs: Any) -> HEADQA:
    """Implement headqa es for this module."""
    return _headqa_variant("es", **kwargs)
