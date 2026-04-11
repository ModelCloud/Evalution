# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep benchmark defaults and public task ids explicit at module scope.
_MEDMCQA_LABELS = ["A", "B", "C", "D"]


def _medmcqa_prompt(question: str, option_texts: list[str]) -> str:
    """Implement medmcqa prompt for this module."""
    lines = [f"Question: {question.strip()}", "Choices:"]
    for label, option_text in zip(_MEDMCQA_LABELS, option_texts, strict=True):
        lines.append(f"{label}. {option_text.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class MedMCQA(BaseMultipleChoiceSuite):
    # Evaluate medical entrance-exam questions by ranking answer labels after showing all option texts.
    """Implement the med mcqa benchmark suite."""
    dataset_path: str = "openlifescienceai/medmcqa"
    split: str = "validation"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "medmcqa"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        choice_texts = [
            str(doc["opa"]).strip(),
            str(doc["opb"]).strip(),
            str(doc["opc"]).strip(),
            str(doc["opd"]).strip(),
        ]
        question = str(doc["question"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_medmcqa_prompt(question, choice_texts),
            choices=list(_MEDMCQA_LABELS),
            gold_index=int(doc["cop"]),
            metadata={
                "id": str(doc["id"]),
                "question": question,
                "choice_type": str(doc["choice_type"]),
                "subject_name": str(doc["subject_name"]),
                "topic_name": None if doc["topic_name"] is None else str(doc["topic_name"]),
                "explanation": None if doc["exp"] is None else str(doc["exp"]),
                "choice_labels": list(_MEDMCQA_LABELS),
                "choice_texts": choice_texts,
            },
        )

    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        """Implement label prompt for med mcqa."""
        choice_texts = list(sample.metadata["choice_texts"])
        lines = [f"Question: {sample.metadata['question']}", "Choices:"]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {choice_texts[choice_index]}")
        lines.append("Answer:")
        return "\n".join(lines)


def medmcqa(**kwargs: Any) -> MedMCQA:
    """Implement medmcqa for this module."""
    return MedMCQA(**kwargs)
