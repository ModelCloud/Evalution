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
_MEDQA_LABELS = ["A", "B", "C", "D"]


def _medqa_prompt(question: str, option_texts: list[str]) -> str:
    """Implement medqa prompt for this module."""
    lines = [f"Question: {question.strip()}"]
    for label, option_text in zip(_MEDQA_LABELS, option_texts, strict=True):
        lines.append(f"{label}. {option_text.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class MedQA(BaseMultipleChoiceSuite):
    # Evaluate USMLE-style medical questions by ranking answer labels after showing option texts.
    """Implement the med QA benchmark suite."""
    dataset_path: str = "GBaker/MedQA-USMLE-4-options-hf"
    split: str = "test"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "medqa_4options"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        choice_texts = [
            str(doc["ending0"]).strip(),
            str(doc["ending1"]).strip(),
            str(doc["ending2"]).strip(),
            str(doc["ending3"]).strip(),
        ]
        question = str(doc["sent1"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_medqa_prompt(question, choice_texts),
            choices=list(_MEDQA_LABELS),
            gold_index=int(doc["label"]),
            metadata={
                "id": str(doc["id"]),
                "question": question,
                "question_suffix": str(doc["sent2"]).strip(),
                "choice_labels": list(_MEDQA_LABELS),
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
        """Implement label prompt for med QA."""
        choice_texts = list(sample.metadata["choice_texts"])
        lines = [f"Question: {sample.metadata['question']}"]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {choice_texts[choice_index]}")
        lines.append("Answer:")
        return "\n".join(lines)


def medqa_4options(**kwargs: Any) -> MedQA:
    """Implement medqa 4options for this module."""
    return MedQA(**kwargs)
