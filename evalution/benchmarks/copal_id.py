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
_VARIANT_SPLITS = {
    "standard": "test",
    "colloquial": "test_colloquial",
}


def _copal_id_connector(question: str) -> str:
    """Implement copal id connector for this module."""
    return {
        "cause": "karena",
        "effect": "maka",
    }[question]


def _copal_id_choice_text(choice: str) -> str:
    """Implement copal id choice text for this module."""
    return choice[:1].lower() + choice[1:]


def _copal_id_prompt(premise: str, question: str) -> str:
    """Implement copal id prompt for this module."""
    stripped = premise.strip()
    if stripped.endswith((".", "!", "?")):
        stripped = stripped[:-1]
    return f"{stripped} {_copal_id_connector(question)}"


@dataclass(slots=True)
class COPALID(BaseMultipleChoiceSuite):
    """Implement the copalid benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "haryoaw/COPAL"
    dataset_name: str | None = "id"
    split: str = "test"
    variant: str = "standard"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.variant not in _VARIANT_SPLITS:
            raise ValueError(f"unsupported copal_id variant: {self.variant!r}")
        expected_split = _VARIANT_SPLITS[self.variant]
        if self.split != expected_split:
            raise ValueError("copal_id split must match the configured variant")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"copal_id_{self.variant}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        return MultipleChoiceSample(
            index=index,
            prompt=_copal_id_prompt(str(doc["premise"]), str(doc["question"])),
            choices=[
                _copal_id_choice_text(str(doc["choice1"])),
                _copal_id_choice_text(str(doc["choice2"])),
            ],
            gold_index=int(doc["label"]),
            metadata={
                "idx": int(doc["idx"]),
                "question": str(doc["question"]),
                "premise": str(doc["premise"]).strip(),
                "raw_choices": [
                    str(doc["choice1"]).strip(),
                    str(doc["choice2"]).strip(),
                ],
                "variant": self.variant,
                "terminology": bool(doc["Terminology"]),
                "culture": bool(doc["Culture"]),
                "language": bool(doc["Language"]),
            },
        )

    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        """Implement label prompt for copalid."""
        relation = {
            "cause": "sebab",
            "effect": "akibat",
        }[sample.metadata["question"]]
        lines = [
            f"Premis: {sample.metadata['premise']}",
            f"Pertanyaan: Opsi mana yang lebih mungkin menjadi {relation}?",
        ]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {sample.metadata['raw_choices'][choice_index]}")
        lines.append("Jawaban:")
        return "\n".join(lines)


def copal_id(*, variant: str, **kwargs: Any) -> COPALID:
    """Implement copal id for this module."""
    if variant not in _VARIANT_SPLITS:
        raise ValueError(f"unsupported copal_id variant: {variant!r}")
    return COPALID(variant=variant, split=_VARIANT_SPLITS[variant], **kwargs)


def copal_id_standard(**kwargs: Any) -> COPALID:
    """Implement copal id standard for this module."""
    return copal_id(variant="standard", **kwargs)


def copal_id_colloquial(**kwargs: Any) -> COPALID:
    """Implement copal id colloquial for this module."""
    return copal_id(variant="colloquial", **kwargs)
