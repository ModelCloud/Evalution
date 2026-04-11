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
_CHOICE_LABELS = ("A", "B", "C", "D")
_ANSWER_TO_INDEX = {label: idx for idx, label in enumerate(_CHOICE_LABELS)}


def _mutual_prompt(article: str, choices: list[str]) -> str:
    """Implement mutual prompt for this module."""
    lines = [f"Dialogue: {article.strip()}", "Reply options:"]
    lines.extend(
        f"{label}. {choice.strip()}"
        for label, choice in zip(_CHOICE_LABELS, choices, strict=True)
    )
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class MuTual(BaseMultipleChoiceSuite):
    """Implement the mu tual benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "tasksource/mutual"
    dataset_name: str | None = None
    split: str = "validation"
    stream: bool = (False)

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "mutual"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        answer_key = str(doc["answers"]).strip()
        choices = [str(choice).strip() for choice in doc["options"]]
        return MultipleChoiceSample(
            index=index,
            prompt=_mutual_prompt(str(doc["article"]), choices),
            choices=choices,
            gold_index=_ANSWER_TO_INDEX[answer_key],
            metadata={
                "dialogue_id": str(doc["id"]),
                "answer_key": answer_key,
                "choice_labels": list(_CHOICE_LABELS),
                "choice_texts": choices,
            },
        )


def mutual(**kwargs: Any) -> MuTual:
    """Implement mutual for this module."""
    return MuTual(**kwargs)
