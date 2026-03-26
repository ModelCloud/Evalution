# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.localized_bbq import CHOICE_LABELS, bbq_prompt, slugify_config_name
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Frozen upstream config snapshot for import safety. Refresh deliberately if the dataset adds
# or removes categories.
CABBQ_CATEGORIES = (
    "Age",
    "DisabilityStatus",
    "Gender",
    "LGBTQIA",
    "Nationality",
    "PhysicalAppearance",
    "RaceEthnicity",
    "Religion",
    "SES",
    "SpanishRegion",
)
CABBQ_TASKS = tuple(f"cabbq_{slugify_config_name(category)}" for category in CABBQ_CATEGORIES)
_CATEGORY_TO_TASK = dict(zip(CABBQ_CATEGORIES, CABBQ_TASKS, strict=True))


@dataclass(slots=True)
class CaBBQ(BaseMultipleChoiceSuite):
    """CaBBQ suite backed by a frozen category registry to keep imports offline-safe."""

    dataset_path: str = "BSC-LT/CaBBQ"
    dataset_name: str | None = "Age"
    split: str = "test"
    category: str = "Age"

    def __post_init__(self) -> None:
        if self.category not in CABBQ_CATEGORIES:
            raise ValueError(f"unsupported cabbq category: {self.category!r}")
        if self.dataset_name in {None, self.category}:
            self.dataset_name = self.category
            return
        raise ValueError("cabbq dataset_name must match the configured category")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return _CATEGORY_TO_TASK[self.category]

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [str(doc["ans0"]).strip(), str(doc["ans1"]).strip(), str(doc["ans2"]).strip()]
        gold_index = int(doc["label"])
        return MultipleChoiceSample(
            index=index,
            prompt=bbq_prompt(str(doc["context"]), str(doc["question"]), choices),
            choices=list(CHOICE_LABELS),
            gold_index=gold_index,
            metadata={
                "category": self.category,
                "question_polarity": str(doc["question_polarity"]),
                "context_condition": str(doc["context_condition"]),
                "question_type": str(doc["question_type"]),
                "target_label": int(doc["label"]),
                "raw_choices": choices,
            },
        )


def cabbq(*, category: str, **kwargs: Any) -> CaBBQ:
    return CaBBQ(category=category, dataset_name=category, **kwargs)


def _make_cabbq_factory(category: str) -> Any:
    def factory(**kwargs: Any) -> CaBBQ:
        return cabbq(category=category, **kwargs)

    factory.__name__ = _CATEGORY_TO_TASK[category]
    return factory


for _category in CABBQ_CATEGORIES:
    globals()[_CATEGORY_TO_TASK[_category]] = _make_cabbq_factory(_category)

del _category
