# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# EusTrivia uses fixed answer labels in the dataset, so we score label continuations instead of option text.
_EUS_TRIVIA_LABELS = ("A", "B", "C", "D")


def _eus_trivia_prompt(doc: dict[str, Any]) -> str:
    """Implement eus trivia prompt for this module."""
    candidates = [str(candidate).strip() for candidate in doc["candidates"]]
    if len(candidates) < 2:
        raise ValueError("eus_trivia requires at least two candidates")
    if len(candidates) > len(_EUS_TRIVIA_LABELS):
        raise ValueError("eus_trivia supports at most four candidates")
    formatted_choices = "\n".join(
        f"{label}: {candidate}"
        for label, candidate in zip(_EUS_TRIVIA_LABELS, candidates, strict=False)
    )
    return f"Galdera: {str(doc['question']).strip()}\n{formatted_choices}\nErantzuna:"


@dataclass(slots=True)
class EusTrivia(BaseMultipleChoiceSuite):
    # The public EusTrivia evaluation split is a four-way multiple-choice test set.
    """Implement the eus trivia benchmark suite."""
    dataset_path: str = "HiTZ/EusTrivia"
    dataset_name: str | None = "default"
    split: str = "test"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "eus_trivia"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        candidates = [str(candidate).strip() for candidate in doc["candidates"]]
        choice_labels = list(_EUS_TRIVIA_LABELS[: len(candidates)])
        gold_index = int(doc["answer"])
        if gold_index < 0 or gold_index >= len(choice_labels):
            raise ValueError("eus_trivia answer index is out of range")
        return MultipleChoiceSample(
            index=index,
            prompt=_eus_trivia_prompt(doc),
            choices=choice_labels,
            gold_index=gold_index,
            metadata={
                "id": int(doc["id"]),
                "category": str(doc["category"]),
                "difficulty": str(doc["difficulty"]),
                "question": str(doc["question"]).strip(),
                "raw_choices": candidates,
                "choice_labels": choice_labels,
            },
        )


def eus_trivia(**kwargs: Any) -> EusTrivia:
    """Implement eus trivia for this module."""
    return EusTrivia(**kwargs)
