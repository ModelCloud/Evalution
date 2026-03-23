# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _sciq_prompt(*, support: str, question: str) -> str:
    lines: list[str] = []
    support_text = support.strip()
    if support_text:
        lines.append(support_text)
    lines.append(f"Question: {question.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class SciQ(BaseMultipleChoiceSuite):
    # Evaluate science question answering by ranking four answer strings with token log-likelihood.
    dataset_path: str = "allenai/sciq"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the public SciQ benchmark.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used in logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "sciq"

    # Match the upstream task ordering where the three distractors precede the correct answer.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [
            doc["distractor1"].strip(),
            doc["distractor2"].strip(),
            doc["distractor3"].strip(),
            doc["correct_answer"].strip(),
        ]
        return MultipleChoiceSample(
            index=index,
            prompt=_sciq_prompt(support=doc["support"], question=doc["question"]),
            choices=choices,
            gold_index=3,
            metadata={
                "support": doc["support"].strip(),
                "question": doc["question"].strip(),
                "choice_labels": ["A", "B", "C", "D"],
                "choice_texts": choices,
            },
        )


# Mirror the public suite factory style used by the rest of the package.
def sciq(**kwargs: Any) -> SciQ:
    return SciQ(**kwargs)
