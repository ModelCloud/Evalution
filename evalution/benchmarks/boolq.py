# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _format_boolq_question(question: str) -> str:
    # Normalize trailing punctuation so prompts stay readable even if the source row already ends with a question mark.
    stripped = question.strip()
    return stripped if stripped.endswith("?") else f"{stripped}?"


@dataclass(slots=True)
class BoolQ(BaseMultipleChoiceSuite):
    # Evaluate yes-or-no reading comprehension by ranking boolean answers with token log-likelihood.
    dataset_path: str = "super_glue"
    dataset_name: str | None = "boolq"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical BoolQ task packaged inside SuperGLUE.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "boolq"

    # Convert one BoolQ row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=f"{doc['passage']}\nQuestion: {_format_boolq_question(doc['question'])}\nAnswer:",
            choices=["no", "yes"],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )


# Mirror the public suite factory style used by the rest of the package.
def boolq(**kwargs: Any) -> BoolQ:
    return BoolQ(**kwargs)
