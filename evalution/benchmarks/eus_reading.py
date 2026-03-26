# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# EusReading uses answer-label scoring over up to four candidates.
_EUS_READING_LABELS = ("A", "B", "C", "D")


def _eus_reading_prompt(doc: dict[str, Any]) -> str:
    candidates = [str(candidate).strip() for candidate in doc["candidates"]]
    if len(candidates) < 2:
        raise ValueError("eus_reading requires at least two candidates")
    if len(candidates) > len(_EUS_READING_LABELS):
        raise ValueError("eus_reading supports at most four candidates")
    formatted_choices = "\n".join(
        f"{label}: {candidate}"
        for label, candidate in zip(_EUS_READING_LABELS, candidates, strict=False)
    )
    return (
        f"Pasartea: {str(doc['context']).strip()}\n\n"
        f"Galdera: {str(doc['question']).strip()}\n"
        f"{formatted_choices}\n"
        "Erantzuna:"
    )


@dataclass(slots=True)
class EusReading(BaseMultipleChoiceSuite):
    # The benchmarked public evaluation split is the test set.
    dataset_path: str = "HiTZ/EusReading"
    dataset_name: str | None = "default"
    split: str = "test"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "eus_reading"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        candidates = [str(candidate).strip() for candidate in doc["candidates"]]
        choice_labels = list(_EUS_READING_LABELS[: len(candidates)])
        gold_index = int(doc["answer"])
        if gold_index < 0 or gold_index >= len(choice_labels):
            raise ValueError("eus_reading answer index is out of range")
        return MultipleChoiceSample(
            index=index,
            prompt=_eus_reading_prompt(doc),
            choices=choice_labels,
            gold_index=gold_index,
            metadata={
                "id": int(doc["id"]),
                "context": str(doc["context"]).strip(),
                "question": str(doc["question"]).strip(),
                "raw_choices": candidates,
                "choice_labels": choice_labels,
            },
        )


def eus_reading(**kwargs: Any) -> EusReading:
    return EusReading(**kwargs)
