# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _multirc_prompt(doc: dict[str, Any]) -> str:
    return f"{str(doc['paragraph']).strip()}\nQuestion: {str(doc['question']).strip()}\nAnswer:"


@dataclass(slots=True)
class MultiRC(BaseMultipleChoiceSuite):
    dataset_path: str = "super_glue"
    dataset_name: str | None = "multirc"
    split: str = "validation"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "multirc"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        metadata: dict[str, Any] = {
            "paragraph": str(doc["paragraph"]),
            "question": str(doc["question"]),
            "answer": str(doc["answer"]),
        }
        if "idx" in doc:
            metadata["idx"] = doc["idx"]
        return MultipleChoiceSample(
            index=index,
            prompt=_multirc_prompt(doc),
            choices=[
                f"{str(doc['answer']).strip()}\nIs the answer correct? yes",
                f"{str(doc['answer']).strip()}\nIs the answer correct? no",
            ],
            gold_index=int(doc["label"]),
            metadata=metadata,
        )


def multirc(**kwargs: Any) -> MultiRC:
    return MultiRC(**kwargs)
