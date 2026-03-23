# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


@dataclass(slots=True)
class SWAG(BaseMultipleChoiceSuite):
    # Evaluate grounded commonsense inference by ranking four candidate next-event completions.
    dataset_path: str = "swag"
    dataset_name: str | None = "regular"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the public SWAG benchmark.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used in logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "swag"

    # Match the upstream harness prompt shape where the model scores each candidate continuation
    # directly after the shared start phrase.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [
            doc["ending0"].strip(),
            doc["ending1"].strip(),
            doc["ending2"].strip(),
            doc["ending3"].strip(),
        ]
        return MultipleChoiceSample(
            index=index,
            prompt=doc["startphrase"].strip(),
            choices=choices,
            gold_index=int(doc["label"]),
            metadata={
                "video_id": doc["video-id"],
                "fold_index": doc["fold-ind"],
                "gold_source": doc["gold-source"],
                "sent1": doc["sent1"].strip(),
                "sent2": doc["sent2"].strip(),
                "choice_labels": ["A", "B", "C", "D"],
                "choice_texts": choices,
            },
        )


# Mirror the public suite factory style used by the rest of the package.
def swag(**kwargs: Any) -> SWAG:
    return SWAG(**kwargs)
