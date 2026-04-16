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
SUPERGPQA_DATASET_PATH = "m-a-p/SuperGPQA"
_CHOICE_LABELS = tuple(chr(ord("A") + index) for index in range(26))


def _supergpqa_prompt(question: str, options: list[str]) -> str:
    """Implement supergpqa prompt for this module."""
    lines = [f"Question: {question.strip()}"]
    for label, option in zip(_CHOICE_LABELS[: len(options)], options, strict=True):
        lines.append(f"{label}. {option.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class SuperGPQA(BaseMultipleChoiceSuite):
    """Implement the SuperGPQA benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = SUPERGPQA_DATASET_PATH
    dataset_name: str | None = None
    split: str = "train"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "supergpqa"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        raw_options = [str(option).strip() for option in doc["options"]]
        answer_label = str(doc["answer_letter"]).strip().upper()
        choice_labels = list(_CHOICE_LABELS[: len(raw_options)])
        if answer_label not in choice_labels:
            raise ValueError(f"unsupported supergpqa answer label: {answer_label!r}")
        return MultipleChoiceSample(
            index=index,
            prompt=_supergpqa_prompt(str(doc["question"]), raw_options),
            choices=choice_labels,
            gold_index=choice_labels.index(answer_label),
            metadata={
                "uuid": str(doc["uuid"]),
                "raw_choices": raw_options,
                "answer_label": answer_label,
                "answer_text": str(doc["answer"]).strip(),
                "discipline": str(doc["discipline"]).strip(),
                "field": str(doc["field"]).strip(),
                "subfield": str(doc["subfield"]).strip(),
                "difficulty": str(doc["difficulty"]).strip(),
                "is_calculation": bool(doc["is_calculation"]),
            },
        )


def supergpqa(**kwargs: Any) -> SuperGPQA:
    """Implement supergpqa for this module."""
    return SuperGPQA(**kwargs)
