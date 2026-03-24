# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

INVERSE_SCALING_SUBSETS = (
    "hindsight-neglect",
    "into-the-unknown",
    "memo-trap",
    "modus-tollens",
    "neqa",
    "pattern-matching-suppression",
    "prompt-injection",
    "redefine",
    "repetitive-algebra",
    "resisting-correction",
    "sig-figs",
)
INVERSE_SCALING_TASKS = tuple(
    f"inverse_scaling_{subset.replace('-', '_')}"
    for subset in INVERSE_SCALING_SUBSETS
)
_SUBSET_TO_TASK = dict(zip(INVERSE_SCALING_SUBSETS, INVERSE_SCALING_TASKS, strict=True))


def _parse_choice_list(raw_classes: Any) -> list[str]:
    if isinstance(raw_classes, list):
        return [str(choice).strip() for choice in raw_classes]
    if not isinstance(raw_classes, str):
        raise TypeError("inverse_scaling classes must be a list or a string literal list")
    parsed = ast.literal_eval(raw_classes)
    if not isinstance(parsed, list):
        raise TypeError("inverse_scaling classes literal must decode to a list")
    return [str(choice).strip() for choice in parsed]


@dataclass(slots=True)
class InverseScaling(BaseMultipleChoiceSuite):
    dataset_path: str = "pminervini/inverse-scaling"
    dataset_name: str | None = "hindsight-neglect"
    split: str = "data"
    subset: str = "hindsight-neglect"
    streaming: bool = False

    def __post_init__(self) -> None:
        if self.subset not in INVERSE_SCALING_SUBSETS:
            raise ValueError(f"unsupported inverse_scaling subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("inverse_scaling dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return _SUBSET_TO_TASK[self.subset]

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = _parse_choice_list(doc["classes"])
        gold_index = int(doc["answer_index"])
        return MultipleChoiceSample(
            index=index,
            prompt=str(doc["prompt"]).rstrip(),
            choices=choices,
            gold_index=gold_index,
            metadata={
                "subset": self.subset,
                "round": int(doc["round"]),
                "part": int(doc["part"]),
                "choice_labels": [chr(ord("A") + i) for i in range(len(choices))],
                "choice_texts": choices,
            },
        )


def inverse_scaling(*, subset: str, **kwargs: Any) -> InverseScaling:
    return InverseScaling(subset=subset, dataset_name=subset, **kwargs)


def _make_inverse_scaling_factory(subset: str) -> Any:
    def factory(**kwargs: Any) -> InverseScaling:
        return inverse_scaling(subset=subset, **kwargs)

    factory.__name__ = _SUBSET_TO_TASK[subset]
    return factory


for _subset in INVERSE_SCALING_SUBSETS:
    globals()[_SUBSET_TO_TASK[_subset]] = _make_inverse_scaling_factory(_subset)

del _subset
