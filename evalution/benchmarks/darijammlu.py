# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import get_dataset_config_names, load_dataset

from evalution.benchmarks.arabic_subject_mmlu import (
    CHOICE_LABELS,
    slugify_subset_name,
    subject_mmlu_prompt,
)
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

DARIJAMMLU_SUBSETS = tuple(get_dataset_config_names("MBZUAI-Paris/DarijaMMLU"))
DARIJAMMLU_TASKS = tuple(
    f"darijammlu_{slugify_subset_name(subset)}" for subset in DARIJAMMLU_SUBSETS
)
_SUBSET_TO_TASK = dict(zip(DARIJAMMLU_SUBSETS, DARIJAMMLU_TASKS, strict=True))


@dataclass(slots=True)
class DarijaMMLU(BaseMultipleChoiceSuite):
    dataset_path: str = "MBZUAI-Paris/DarijaMMLU"
    dataset_name: str | None = None
    split: str = "test"
    subset: str = "accounting"

    def __post_init__(self) -> None:
        if self.subset not in DARIJAMMLU_SUBSETS:
            raise ValueError(f"unsupported darijammlu subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("darijammlu dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return _SUBSET_TO_TASK[self.subset]

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [str(choice).strip() for choice in doc["choices"]]
        answer_index = int(doc["answer"])
        return MultipleChoiceSample(
            index=index,
            prompt=subject_mmlu_prompt(
                benchmark_name="DarijaMMLU",
                subject_native=str(doc["subject_darija"]),
                question=str(doc["question"]),
                choices=choices,
                context=None if doc["context"] is None else str(doc["context"]),
            ),
            choices=list(CHOICE_LABELS[: len(choices)]),
            gold_index=answer_index,
            metadata={
                "subset": self.subset,
                "subject": str(doc["subject"]).strip(),
                "subject_darija": str(doc["subject_darija"]).strip(),
                "question": str(doc["question"]).strip(),
                "context": None if doc["context"] is None else str(doc["context"]).strip(),
                "answer_index": answer_index,
                "raw_choices": choices,
                "source": str(doc["source"]).strip(),
            },
        )


def darijammlu(*, subset: str, **kwargs: Any) -> DarijaMMLU:
    return DarijaMMLU(subset=subset, dataset_name=subset, **kwargs)


def _make_darijammlu_factory(subset: str) -> Any:
    def factory(**kwargs: Any) -> DarijaMMLU:
        return darijammlu(subset=subset, **kwargs)

    factory.__name__ = _SUBSET_TO_TASK[subset]
    return factory


for _subset in DARIJAMMLU_SUBSETS:
    globals()[_SUBSET_TO_TASK[_subset]] = _make_darijammlu_factory(_subset)

del _subset
