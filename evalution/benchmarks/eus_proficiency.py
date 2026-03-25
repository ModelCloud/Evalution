# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# EusProficiency is published as a fixed four-way multiple-choice test over answer labels.
_EUS_PROFICIENCY_LABELS = ("A", "B", "C", "D")


def _eus_proficiency_prompt(doc: dict[str, Any]) -> str:
    candidates = [str(candidate).strip() for candidate in doc["candidates"]]
    if len(candidates) != len(_EUS_PROFICIENCY_LABELS):
        raise ValueError("eus_proficiency requires exactly four candidates")
    return (
        f"Galdera: {str(doc['question']).strip()}\n"
        f"A: {candidates[0]}\n"
        f"B: {candidates[1]}\n"
        f"C: {candidates[2]}\n"
        f"D: {candidates[3]}\n"
        "Erantzuna:"
    )


@dataclass(slots=True)
class EusProficiency(BaseMultipleChoiceSuite):
    # The public test split is the benchmarked evaluation set in lm-eval.
    dataset_path: str = "HiTZ/EusProficiency"
    dataset_name: str | None = "default"
    split: str = "test"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "eus_proficiency"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        gold_index = int(doc["answer"])
        if gold_index < 0 or gold_index >= len(_EUS_PROFICIENCY_LABELS):
            raise ValueError("eus_proficiency answer index is out of range")
        return MultipleChoiceSample(
            index=index,
            prompt=_eus_proficiency_prompt(doc),
            choices=list(_EUS_PROFICIENCY_LABELS),
            gold_index=gold_index,
            metadata={
                "id": int(doc["id"]),
                "question": str(doc["question"]).strip(),
                "raw_choices": [str(candidate).strip() for candidate in doc["candidates"]],
                "choice_labels": list(_EUS_PROFICIENCY_LABELS),
            },
        )


def eus_proficiency(**kwargs: Any) -> EusProficiency:
    return EusProficiency(**kwargs)
