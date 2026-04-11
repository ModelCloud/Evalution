# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.arabic_subject_mmlu import (
    CHOICE_LABELS,
    slugify_subset_name,
    subject_mmlu_prompt,
)
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Frozen upstream config snapshot for import safety. Refresh deliberately if the dataset adds
# or removes subsets.
DARIJAMMLU_SUBSETS = (
    "accounting",
    "arabic_language",
    "arabic_language_(general)",
    "arabic_language_(grammar)",
    "biology",
    "civics",
    "computer_science",
    "driving_test",
    "economics",
    "general_knowledge",
    "geography",
    "global_facts",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_world_history",
    "history",
    "human_aging",
    "international_law",
    "islamic_studies",
    "jurisprudence",
    "law",
    "logical_fallacies",
    "management",
    "management_ar",
    "marketing",
    "math",
    "moral_disputes",
    "moral_scenarios",
    "natural_science",
    "nutrition",
    "philosophy",
    "philosophy_ar",
    "physics",
    "political_science",
    "professional_law",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "social_science",
    "sociology",
    "world_religions",
)
DARIJAMMLU_TASKS = tuple(
    f"darijammlu_{slugify_subset_name(subset)}" for subset in DARIJAMMLU_SUBSETS
)
_SUBSET_TO_TASK = dict(zip(DARIJAMMLU_SUBSETS, DARIJAMMLU_TASKS, strict=True))


@dataclass(slots=True)
class DarijaMMLU(BaseMultipleChoiceSuite):
    """DarijaMMLU suite backed by a frozen subset registry to keep imports offline-safe."""

    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "MBZUAI-Paris/DarijaMMLU"
    dataset_name: str | None = None
    split: str = "test"
    subset: str = "accounting"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.subset not in DARIJAMMLU_SUBSETS:
            raise ValueError(f"unsupported darijammlu subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("darijammlu dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return _SUBSET_TO_TASK[self.subset]

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
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
    """Implement darijammlu for this module."""
    return DarijaMMLU(subset=subset, dataset_name=subset, **kwargs)


def _make_darijammlu_factory(subset: str) -> Any:
    """Make darijammlu factory."""
    def factory(**kwargs: Any) -> DarijaMMLU:
        """Implement factory for this module."""
        return darijammlu(subset=subset, **kwargs)

    factory.__name__ = _SUBSET_TO_TASK[subset]
    return factory


for _subset in DARIJAMMLU_SUBSETS:
    globals()[_SUBSET_TO_TASK[_subset]] = _make_darijammlu_factory(_subset)

del _subset
