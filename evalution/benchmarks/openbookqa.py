# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.benchmarks.multiple_choice_utils import choice_index_from_labels, question_answer_prompt


@dataclass(slots=True)
class OpenBookQA(BaseMultipleChoiceSuite):
    # Evaluate elementary science questions by ranking four answer choices with token log-likelihood.
    dataset_path: str = "allenai/openbookqa"
    dataset_name: str | None = "main"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the public OpenBookQA benchmark.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "openbookqa"

    # Convert one OpenBookQA row into the shared prompt and four-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        labels = list(doc["choices"]["label"])
        texts = list(doc["choices"]["text"])
        return MultipleChoiceSample(
            index=index,
            prompt=question_answer_prompt(doc["question_stem"]),
            choices=texts,
            gold_index=choice_index_from_labels(labels, doc["answerKey"]),
            metadata={"id": doc["id"], "choice_labels": labels},
        )


# Mirror the public suite factory style used by the rest of the package.
def openbookqa(**kwargs: Any) -> OpenBookQA:
    return OpenBookQA(**kwargs)
