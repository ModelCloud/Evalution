# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _qnli_prompt(question: str, sentence: str) -> str:
    # Present question answering as an explicit yes-or-no judgment over the candidate sentence.
    return (
        f"{question.strip()}\n"
        f"{sentence.strip()}\n"
        "Question: Does this response answer the question?\n"
        "Answer:"
    )


@dataclass(slots=True)
class QNLI(BaseMultipleChoiceSuite):
    # Evaluate question-answer sentence relevance with yes versus no label ranking.
    dataset_path: str = "nyu-mll/glue"
    dataset_name: str | None = "qnli"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical QNLI task inside GLUE.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "qnli"

    # Convert one QNLI row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_qnli_prompt(doc["question"], doc["sentence"]),
            choices=["yes", "no"],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )


# Mirror the public suite factory style used by the rest of the package.
def qnli(**kwargs: Any) -> QNLI:
    return QNLI(**kwargs)
