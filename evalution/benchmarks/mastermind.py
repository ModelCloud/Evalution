# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

MASTERMIND_VARIANTS = (
    "mastermind_24_easy",
    "mastermind_24_hard",
    "mastermind_35_easy",
    "mastermind_35_hard",
    "mastermind_46_easy",
    "mastermind_46_hard",
)


@dataclass(frozen=True, slots=True)
class _MastermindConfig:
    dataset_path: str
    code_shape: str
    difficulty: str


_MASTERMIND_CONFIGS = {
    "mastermind_24_easy": _MastermindConfig("flair/mastermind_24_mcq_random", "24", "easy"),
    "mastermind_24_hard": _MastermindConfig("flair/mastermind_24_mcq_close", "24", "hard"),
    "mastermind_35_easy": _MastermindConfig("flair/mastermind_35_mcq_random", "35", "easy"),
    "mastermind_35_hard": _MastermindConfig("flair/mastermind_35_mcq_close", "35", "hard"),
    "mastermind_46_easy": _MastermindConfig("flair/mastermind_46_mcq_random", "46", "easy"),
    "mastermind_46_hard": _MastermindConfig("flair/mastermind_46_mcq_close", "46", "hard"),
}


def _mastermind_prompt(instruction: str) -> str:
    return f"{instruction.strip()}\n\nThe secret code is:"


@dataclass(slots=True)
class Mastermind(BaseMultipleChoiceSuite):
    # Mastermind ranks candidate secret codes via multiple-choice loglikelihood.
    dataset_path: str = "flair/mastermind_24_mcq_random"
    dataset_name: str | None = None
    split: str = "test"
    stream: bool = False
    variant: str = "mastermind_24_easy"

    def __post_init__(self) -> None:
        if self.variant not in MASTERMIND_VARIANTS:
            raise ValueError(f"unsupported mastermind variant: {self.variant!r}")
        expected_path = _MASTERMIND_CONFIGS[self.variant].dataset_path
        if self.dataset_path in {"", expected_path}:
            self.dataset_path = expected_path
        else:
            raise ValueError("mastermind dataset_path must match the configured variant")
        if self.dataset_name is not None:
            raise ValueError("mastermind does not use a dataset_name")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return self.variant

    def result_metadata(self) -> dict[str, Any]:
        config = _MASTERMIND_CONFIGS[self.variant]
        return {
            **super().result_metadata(),
            "variant": self.variant,
            "code_shape": config.code_shape,
            "difficulty": config.difficulty,
        }

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        config = _MASTERMIND_CONFIGS[self.variant]
        option_labels = [str(label) for label in doc["options"]["label"]]
        option_texts = [str(text).strip() for text in doc["options"]["text"]]
        gold_index = option_labels.index(str(doc["answerKey"]))
        return MultipleChoiceSample(
            index=index,
            prompt=_mastermind_prompt(str(doc["instruction"])),
            choices=option_texts,
            gold_index=gold_index,
            metadata={
                "id": int(doc["id"]),
                "variant": self.variant,
                "code_shape": config.code_shape,
                "difficulty": config.difficulty,
                "option_labels": option_labels,
                "choice_texts": option_texts,
            },
        )


def mastermind(*, variant: str, **kwargs: Any) -> Mastermind:
    if variant not in MASTERMIND_VARIANTS:
        raise ValueError(f"unsupported mastermind variant: {variant!r}")
    return Mastermind(variant=variant, dataset_path=_MASTERMIND_CONFIGS[variant].dataset_path, **kwargs)


def mastermind_24_easy(**kwargs: Any) -> Mastermind:
    return mastermind(variant="mastermind_24_easy", **kwargs)


def mastermind_24_hard(**kwargs: Any) -> Mastermind:
    return mastermind(variant="mastermind_24_hard", **kwargs)


def mastermind_35_easy(**kwargs: Any) -> Mastermind:
    return mastermind(variant="mastermind_35_easy", **kwargs)


def mastermind_35_hard(**kwargs: Any) -> Mastermind:
    return mastermind(variant="mastermind_35_hard", **kwargs)


def mastermind_46_easy(**kwargs: Any) -> Mastermind:
    return mastermind(variant="mastermind_46_easy", **kwargs)


def mastermind_46_hard(**kwargs: Any) -> Mastermind:
    return mastermind(variant="mastermind_46_hard", **kwargs)
