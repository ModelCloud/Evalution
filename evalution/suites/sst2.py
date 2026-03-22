# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _sst2_prompt(sentence: str) -> str:
    # Phrase sentiment analysis as a two-choice question so the model scores label tokens instead of free-form text.
    return f"{sentence.strip()}\nQuestion: Is this sentence positive or negative?\nAnswer:"


@dataclass(slots=True)
class SST2(BaseMultipleChoiceSuite):
    # Evaluate SST-2 sentiment classification with positive versus negative label ranking.
    dataset_path: str = "nyu-mll/glue"
    dataset_name: str | None = "sst2"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical SST-2 task inside GLUE.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "sst2"

    # Convert one SST-2 row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_sst2_prompt(doc["sentence"]),
            choices=["negative", "positive"],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )


# Mirror the public suite factory style used by the rest of the package.
def sst2(**kwargs: Any) -> SST2:
    return SST2(**kwargs)
