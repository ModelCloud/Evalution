# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.scorers.classification import f1_for_label
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _mrpc_prompt(sentence1: str, sentence2: str) -> str:
    # Frame paraphrase detection as a direct yes-or-no question over the sentence pair.
    """Implement mrpc prompt for this module."""
    return (
        f"Sentence 1: {sentence1.strip()}\n"
        f"Sentence 2: {sentence2.strip()}\n"
        "Question: Do both sentences mean the same thing?\n"
        "Answer:"
    )


@dataclass(slots=True)
class MRPC(BaseMultipleChoiceSuite):
    # Evaluate Microsoft Research Paraphrase Corpus sentence-pair classification with yes/no label ranking.
    """Implement the mrpc benchmark suite."""
    dataset_path: str = "nyu-mll/glue"
    dataset_name: str | None = "mrpc"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical MRPC task inside GLUE.
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "mrpc"

    # Convert one MRPC row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        return MultipleChoiceSample(
            index=index,
            prompt=_mrpc_prompt(doc["sentence1"], doc["sentence2"]),
            choices=["no", "yes"],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )

    # Report positive-class F1 because MRPC is commonly tracked with both accuracy and paraphrase F1.
    def extra_metrics(
        self,
        *,
        samples: list[MultipleChoiceSample],
        raw_predictions: list[int],
        normalized_predictions: list[int],
    ) -> dict[str, float]:
        """Compute extra metrics from the collected predictions. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        gold_labels = [sample.gold_index for sample in samples]
        return {
            "f1,ll_boolean": f1_for_label(gold_labels, raw_predictions, label=1),
            "f1,ll_avg_boolean": f1_for_label(
                gold_labels,
                normalized_predictions,
                label=1,
            ),
        }


# Mirror the public suite factory style used by the rest of the package.
def mrpc(**kwargs: Any) -> MRPC:
    """Implement mrpc for this module."""
    return MRPC(**kwargs)
