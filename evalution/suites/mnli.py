# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_MNLI_CHOICES = ["True", "Neither", "False"]


def _mnli_prompt(premise: str, hypothesis: str) -> str:
    # Preserve the benchmark's three-way entailment wording while normalizing sentence-final punctuation.
    hypothesis_text = hypothesis.strip()
    if hypothesis_text and not hypothesis_text.endswith("."):
        hypothesis_text = f"{hypothesis_text}."
    return f"{premise.strip()}\nQuestion: {hypothesis_text} True, False or Neither?\nAnswer:"


@dataclass(slots=True)
class MNLI(BaseMultipleChoiceSuite):
    # Evaluate matched-domain MNLI with the canonical three-label ranking setup.
    dataset_path: str = "nyu-mll/glue"
    dataset_name: str | None = "mnli"
    split: str = "validation_matched"

    # Use the canonical Hugging Face datasets loader for the GLUE MNLI task.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite identifier used by logs, YAML specs, and serialized results.
    def task_name(self) -> str:
        return "mnli"

    # Convert one MNLI row into the shared multiple-choice sample structure.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_mnli_prompt(doc["premise"], doc["hypothesis"]),
            choices=list(_MNLI_CHOICES),
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )


# Mirror the public suite factory style used across the package.
def mnli(**kwargs: Any) -> MNLI:
    return MNLI(**kwargs)
