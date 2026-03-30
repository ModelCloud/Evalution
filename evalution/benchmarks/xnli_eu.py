# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_XNLI_EU_CHOICES = ["Bai", "Gainera", "Ez"]


def _xnli_eu_prompt(premise: str, hypothesis: str) -> str:
    return f"{premise.strip()}, ezta? {hypothesis.strip()}"


@dataclass(slots=True)
class XNLIEU(BaseMultipleChoiceSuite):
    # The Basque XNLI benchmark uses the public test split for the `eu` translation.
    dataset_path: str = "HiTZ/xnli-eu"
    dataset_name: str | None = "eu"
    split: str = "test"
    stream: bool = True

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "xnli_eu"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        premise = str(doc["premise"]).strip()
        hypothesis = str(doc["hypothesis"]).strip()
        choices = [f"{choice}, {hypothesis}" for choice in _XNLI_EU_CHOICES]
        return MultipleChoiceSample(
            index=index,
            prompt=_xnli_eu_prompt(premise, hypothesis),
            choices=choices,
            gold_index=int(doc["label"]),
            metadata={
                "language": "eu",
                "premise": premise,
                "hypothesis": hypothesis,
                "choice_texts": list(_XNLI_EU_CHOICES),
                "raw_choices": choices,
            },
        )


def xnli_eu(**kwargs: Any) -> XNLIEU:
    return XNLIEU(**kwargs)
