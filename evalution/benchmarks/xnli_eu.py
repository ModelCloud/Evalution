# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep benchmark defaults and public task ids explicit at module scope.
_XNLI_EU_CHOICES = ["Bai", "Gainera", "Ez"]


def _xnli_eu_prompt(premise: str, hypothesis: str) -> str:
    """Implement XNLI eu prompt for this module."""
    return f"{premise.strip()}, ezta? {hypothesis.strip()}"


@dataclass(slots=True)
class XNLIEU(BaseMultipleChoiceSuite):
    # The Basque XNLI benchmark uses the public test split for the `eu` translation.
    """Implement the xnlieu benchmark suite."""
    dataset_path: str = "HiTZ/xnli-eu"
    dataset_name: str | None = "eu"
    split: str = "test"
    stream: bool = True

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "xnli_eu"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
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
    """Implement XNLI eu for this module."""
    return XNLIEU(**kwargs)
