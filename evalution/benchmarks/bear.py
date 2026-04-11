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
_BEAR_VARIANTS = {
    "bear": "BEAR",
    "bear_big": "BEAR_big",
}


@dataclass(slots=True)
class BEAR(BaseMultipleChoiceSuite):
    """Implement the bear benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "lm-pub-quiz/BEAR"
    dataset_name: str | None = "BEAR"
    split: str = "test"
    variant: str = "bear"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.variant not in _BEAR_VARIANTS:
            raise ValueError(f"unsupported bear variant: {self.variant!r}")
        expected_name = _BEAR_VARIANTS[self.variant]
        if self.dataset_name in {None, expected_name}:
            self.dataset_name = expected_name
            return
        raise ValueError("bear dataset_name must match the configured variant")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return self.variant

    def result_metadata(self) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        metadata = super().result_metadata()
        metadata["prompt_variant"] = "empty_context_full_statement"
        return metadata

    def continuation_for_choice(self, choice: str) -> str:
        """Implement continuation for choice for bear."""
        return choice

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        raw_choices = [str(choice) for choice in doc["text_options"]]
        answer_options = [str(choice) for choice in doc["answer_options"]]
        return MultipleChoiceSample(
            index=index,
            prompt="",
            choices=raw_choices,
            gold_index=int(doc["correct"]),
            metadata={
                "variant": self.variant,
                "composite_id": str(doc["composite_id"]),
                "relation": str(doc["relation"]),
                "item": int(doc["item"]),
                "template_index": int(doc["template_index"]),
                "template": str(doc["template"]),
                "subject": str(doc["subject"]),
                "answer_options": answer_options,
                "choice_count": len(raw_choices),
            },
        )


def bear(**kwargs: Any) -> BEAR:
    """Implement bear for this module."""
    return BEAR(variant="bear", dataset_name="BEAR", **kwargs)


def bear_big(**kwargs: Any) -> BEAR:
    """Implement bear big for this module."""
    return BEAR(variant="bear_big", dataset_name="BEAR_big", **kwargs)
