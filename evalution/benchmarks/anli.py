# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_ANLI_CHOICES = ["True", "Neither", "False"]
_ANLI_DEFAULT_SPLITS = {
    "r1": "test_r1",
    "r2": "test_r2",
    "r3": "test_r3",
}


def _anli_prompt(premise: str, hypothesis: str) -> str:
    # Match the upstream ANLI prompt format and label wording so scores stay comparable.
    return f"{premise.strip()}\nQuestion: {hypothesis.strip()} True, False, or Neither?\nAnswer:"


@dataclass(slots=True)
class ANLI(BaseMultipleChoiceSuite):
    # Evaluate adversarial NLI by ranking entailment, neutral, and contradiction labels.
    dataset_path: str = "facebook/anli"
    dataset_name: str | None = None
    split: str = "test_r1"
    round_name: str = "r1"

    # Use the Hugging Face datasets loader for the public ANLI benchmark.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return f"anli_{self.round_name}"

    # Convert one ANLI row into the shared prompt and three-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_anli_prompt(doc["premise"], doc["hypothesis"]),
            choices=list(_ANLI_CHOICES),
            gold_index=int(doc["label"]),
            metadata={
                "uid": doc["uid"],
                "reason": doc["reason"],
                "round": self.round_name,
                "choice_labels": ["A", "B", "C"],
                "choice_texts": list(_ANLI_CHOICES),
            },
        )


def _anli_round(round_name: str, **kwargs: Any) -> ANLI:
    if "split" not in kwargs:
        kwargs["split"] = _ANLI_DEFAULT_SPLITS[round_name]
    return ANLI(round_name=round_name, **kwargs)


# Mirror the public suite factory style used by the rest of the package.
def anli_r1(**kwargs: Any) -> ANLI:
    return _anli_round("r1", **kwargs)


def anli_r2(**kwargs: Any) -> ANLI:
    return _anli_round("r2", **kwargs)


def anli_r3(**kwargs: Any) -> ANLI:
    return _anli_round("r3", **kwargs)
