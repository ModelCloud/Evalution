# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# GroundCocoa always ranks these five option labels after the flight-selection prompt.
_GROUNDCOCOA_OPTION_LABELS = ("A", "B", "C", "D", "E")
_GROUNDCOCOA_CHOICES = tuple(f"The answer is Option {label}" for label in _GROUNDCOCOA_OPTION_LABELS)
_GROUNDCOCOA_GOLD_INDICES = {
    label: index for index, label in enumerate(_GROUNDCOCOA_OPTION_LABELS)
}


def _groundcocoa_prompt(doc: dict[str, Any]) -> str:
    return (
        "A user has specified certain criteria for booking a flight. Below are five "
        "different flight options labeled 'A', 'B', 'C', 'D', and 'E'. Review these "
        "options and select the one that best matches the user requirements. Respond "
        "with a single option and the phrase 'The answer is Option ' followed by the "
        "correct letter - 'A', 'B', 'C', 'D', or 'E'\n\n"
        f"User Criteria: {str(doc['query'])}\n\n"
        f" Option A: {str(doc['Option A'])}\n"
        f"\n Option B: {str(doc['Option B'])}\n"
        f"\n Option C: {str(doc['Option C'])}\n"
        f"\n Option D: {str(doc['Option D'])}\n"
        f"\n Option E: {str(doc['Option E'])}\n"
    )


def _groundcocoa_gold_index(answer: str) -> int:
    try:
        return _GROUNDCOCOA_GOLD_INDICES[str(answer).strip()]
    except KeyError as exc:
        raise ValueError(f"unsupported GroundCocoa answer label: {answer!r}") from exc


@dataclass(slots=True)
class GroundCocoa(BaseMultipleChoiceSuite):
    # GroundCocoa scores whether the model can map natural-language travel constraints to the best flight option.
    dataset_path: str = "harsh147/GroundCocoa"
    dataset_name: str | None = None
    split: str = "test"
    stream: bool = True

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "groundcocoa"

    def result_metadata(self) -> dict[str, Any]:
        metadata = super().result_metadata()
        metadata["prompt_variant"] = "flight_criteria_with_option_labels"
        return metadata

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_groundcocoa_prompt(doc),
            choices=list(_GROUNDCOCOA_CHOICES),
            gold_index=_groundcocoa_gold_index(str(doc["Answer"])),
            metadata={
                "id": str(doc["id"]),
                "query_pos": str(doc["query_pos"]),
                "is_typical": bool(doc["is_typical"]),
            },
        )


def groundcocoa(**kwargs: Any) -> GroundCocoa:
    return GroundCocoa(**kwargs)
