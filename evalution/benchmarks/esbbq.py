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
ESBBQ_CATEGORIES = (
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
ESBBQ_TASKS = tuple(f"esbbq_{slugify_config_name(category)}" for category in ESBBQ_CATEGORIES)
_CATEGORY_TO_TASK = dict(zip(ESBBQ_CATEGORIES, ESBBQ_TASKS, strict=True))


def _load_esbbq_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = True,
) -> Any:
    return load_dataset(
        dataset_path,
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        streaming=stream,
        verification_mode="no_checks",
    )


@dataclass(slots=True)
class EsBBQ(BaseMultipleChoiceSuite):
    """EsBBQ suite backed by a frozen category registry to keep imports offline-safe."""

    dataset_path: str = "BSC-LT/EsBBQ"
    dataset_name: str | None = "Age"
    split: str = "test"
    category: str = "Age"

    def __post_init__(self) -> None:
        if self.category not in ESBBQ_CATEGORIES:
            raise ValueError(f"unsupported esbbq category: {self.category!r}")
        if self.dataset_name in {None, self.category}:
            self.dataset_name = self.category
            return
        raise ValueError("esbbq dataset_name must match the configured category")

    def dataset_loader(self) -> Any:
        return _load_esbbq_dataset

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


def esbbq(*, category: str, **kwargs: Any) -> EsBBQ:
    return EsBBQ(category=category, dataset_name=category, **kwargs)


def _make_esbbq_factory(category: str) -> Any:
    def factory(**kwargs: Any) -> EsBBQ:
        return esbbq(category=category, **kwargs)

    factory.__name__ = _CATEGORY_TO_TASK[category]
    return factory


for _category in ESBBQ_CATEGORIES:
    globals()[_CATEGORY_TO_TASK[_category]] = _make_esbbq_factory(_category)

del _category
