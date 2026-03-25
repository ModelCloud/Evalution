# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from evalution.benchmarks.localized_bbq import CHOICE_LABELS, bbq_prompt, slugify_config_name
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

BBQ_CATEGORIES = (
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
)
BBQ_TASKS = tuple(f"bbq_{slugify_config_name(category)}" for category in BBQ_CATEGORIES)
_CATEGORY_TO_TASK = dict(zip(BBQ_CATEGORIES, BBQ_TASKS, strict=True))


def _load_bbq_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> Any:
    if dataset_path != "heegyu/bbq":
        raise ValueError(f"unsupported BBQ dataset path: {dataset_path!r}")
    if split != "test":
        raise ValueError(f"unsupported BBQ split: {split!r}")
    if dataset_name not in BBQ_CATEGORIES:
        raise ValueError(f"unsupported BBQ category: {dataset_name!r}")

    data_file = hf_hub_download(
        repo_id=dataset_path,
        filename=f"data/{dataset_name}.jsonl",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return load_dataset(
        "json",
        data_files={split: data_file},
        split=split,
        cache_dir=cache_dir,
        streaming=stream,
    )


@dataclass(slots=True)
class BBQ(BaseMultipleChoiceSuite):
    dataset_path: str = "heegyu/bbq"
    dataset_name: str | None = "Age"
    split: str = "test"
    category: str = "Age"

    def __post_init__(self) -> None:
        if self.category not in BBQ_CATEGORIES:
            raise ValueError(f"unsupported bbq category: {self.category!r}")
        if self.dataset_name in {None, self.category}:
            self.dataset_name = self.category
            return
        raise ValueError("bbq dataset_name must match the configured category")

    def dataset_loader(self) -> Any:
        return _load_bbq_dataset

    def task_name(self) -> str:
        return _CATEGORY_TO_TASK[self.category]

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [str(doc["ans0"]).strip(), str(doc["ans1"]).strip(), str(doc["ans2"]).strip()]
        additional_metadata = dict(doc.get("additional_metadata", {}))
        return MultipleChoiceSample(
            index=index,
            prompt=bbq_prompt(str(doc["context"]), str(doc["question"]), choices),
            choices=list(CHOICE_LABELS),
            gold_index=int(doc["label"]),
            metadata={
                "category": self.category,
                "question_index": str(doc["question_index"]),
                "question_polarity": str(doc["question_polarity"]),
                "context_condition": str(doc["context_condition"]),
                "example_id": int(doc["example_id"]),
                "subcategory": str(additional_metadata.get("subcategory", "")),
                "stereotyped_groups": [str(item) for item in additional_metadata.get("stereotyped_groups", [])],
                "raw_choices": choices,
            },
        )


def bbq(*, category: str, **kwargs: Any) -> BBQ:
    return BBQ(category=category, dataset_name=category, **kwargs)


def _make_bbq_factory(category: str) -> Any:
    def factory(**kwargs: Any) -> BBQ:
        return bbq(category=category, **kwargs)

    factory.__name__ = _CATEGORY_TO_TASK[category]
    return factory


for _category in BBQ_CATEGORIES:
    globals()[_CATEGORY_TO_TASK[_category]] = _make_bbq_factory(_category)

del _category
