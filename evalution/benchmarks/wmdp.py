# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

WMDP_SUBSETS = ("bio", "chem", "cyber")
WMDP_TASKS = tuple(f"wmdp_{subset}" for subset in WMDP_SUBSETS)
_SUBSET_TO_TASK = dict(zip(WMDP_SUBSETS, WMDP_TASKS, strict=True))
_CHOICE_LABELS = ("A", "B", "C", "D")


def _wmdp_prompt(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    for label, choice in zip(_CHOICE_LABELS, choices, strict=True):
        lines.append(f"{label}. {choice.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class WMDP(BaseMultipleChoiceSuite):
    dataset_path: str = "walledai/WMDP"
    dataset_name: str | None = None
    split: str = "bio"
    subset: str = "bio"
    stream: bool = False

    def __post_init__(self) -> None:
        if self.subset not in WMDP_SUBSETS:
            raise ValueError(f"unsupported wmdp subset: {self.subset!r}")
        if self.split != self.subset:
            raise ValueError("wmdp split must match the configured subset")
        if self.dataset_name is not None:
            raise ValueError("wmdp does not use dataset_name")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return _SUBSET_TO_TASK[self.subset]

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [str(choice).strip() for choice in doc["choices"]]
        return MultipleChoiceSample(
            index=index,
            prompt=_wmdp_prompt(str(doc["question"]), choices),
            choices=choices,
            gold_index=int(doc["answer"]),
            metadata={
                "subset": self.subset,
                "choice_labels": list(_CHOICE_LABELS),
                "choice_texts": choices,
            },
        )


def wmdp(*, subset: str, **kwargs: Any) -> WMDP:
    return WMDP(subset=subset, split=subset, **kwargs)


def _make_wmdp_factory(subset: str) -> Any:
    def factory(**kwargs: Any) -> WMDP:
        return wmdp(subset=subset, **kwargs)

    factory.__name__ = _SUBSET_TO_TASK[subset]
    return factory


for _subset in WMDP_SUBSETS:
    globals()[_SUBSET_TO_TASK[_subset]] = _make_wmdp_factory(_subset)

del _subset
