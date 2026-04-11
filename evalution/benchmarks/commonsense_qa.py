# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.benchmarks.multiple_choice_utils import choice_index_from_labels


def _commonsense_qa_prompt(question: str, labels: list[str], texts: list[str]) -> str:
    """Implement commonsense QA prompt for this module."""
    lines = [f"Question: {question.strip()}"]
    for label, text in zip(labels, texts, strict=True):
        lines.append(f"{label}. {text.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class CommonsenseQA(BaseMultipleChoiceSuite):
    # Evaluate commonsense reasoning by ranking answer labels after showing all labeled options.
    """Implement the commonsense QA benchmark suite."""
    dataset_path: str = "tau/commonsense_qa"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical CommonsenseQA benchmark.
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    # Return the stable suite name used in logs, results, and YAML specs.
    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "commonsense_qa"

    # Match the upstream harness prompt shape by listing the labeled options inside the prompt and
    # scoring the answer label continuation directly.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        labels = list(doc["choices"]["label"])
        texts = [text.strip() for text in doc["choices"]["text"]]
        question = doc["question"].strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_commonsense_qa_prompt(question, labels, texts),
            choices=labels,
            gold_index=choice_index_from_labels(labels, doc["answerKey"]),
            metadata={
                "id": doc["id"],
                "question": question,
                "question_concept": doc["question_concept"],
                "choice_labels": labels,
                "choice_texts": texts,
            },
        )

    # Keep the optional label-permutation scorer meaningful by permuting the displayed option texts
    # while still asking the model to emit only the answer label.
    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        """Implement label prompt for commonsense QA."""
        choice_texts = list(sample.metadata["choice_texts"])
        lines = [f"Question: {sample.metadata['question']}"]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {choice_texts[choice_index]}")
        lines.append("Answer:")
        return "\n".join(lines)


# Mirror the public suite factory style used by the rest of the package.
def commonsense_qa(**kwargs: Any) -> CommonsenseQA:
    """Implement commonsense QA for this module."""
    return CommonsenseQA(**kwargs)
