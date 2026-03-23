# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.scorers.classification import matthews_corrcoef
from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _cola_prompt(sentence: str) -> str:
    # Match the GLUE CoLA prompt framing so acceptability is scored as a binary choice instead of free-form text.
    return f"{sentence.strip()}\nQuestion: Does this sentence make sense?\nAnswer:"


@dataclass(slots=True)
class CoLA(BaseMultipleChoiceSuite):
    # Evaluate linguistic acceptability with yes versus no label ranking over the CoLA validation split.
    dataset_path: str = "nyu-mll/glue"
    dataset_name: str | None = "cola"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical CoLA task inside GLUE.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "cola"

    # Convert one CoLA row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_cola_prompt(doc["sentence"]),
            choices=["no", "yes"],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )

    # Report the GLUE primary metric for CoLA alongside raw and length-normalized accuracy.
    def extra_metrics(
        self,
        *,
        samples: list[MultipleChoiceSample],
        raw_predictions: list[int],
        normalized_predictions: list[int],
    ) -> dict[str, float]:
        gold_labels = [sample.gold_index for sample in samples]
        return {
            "mcc,loglikelihood": matthews_corrcoef(gold_labels, raw_predictions),
            "mcc,loglikelihood_norm": matthews_corrcoef(gold_labels, normalized_predictions),
        }


# Mirror the public suite factory style used by the rest of the package.
def cola(**kwargs: Any) -> CoLA:
    return CoLA(**kwargs)
