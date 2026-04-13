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


def _qqp_prompt(question1: str, question2: str) -> str:
    # Frame duplicate-question detection as a direct yes-or-no decision over the pair.
    """Implement qqp prompt for this module."""
    return (
        f"Question 1: {question1.strip()}\n"
        f"Question 2: {question2.strip()}\n"
        "Question: Do both questions ask the same thing?\n"
        "Answer:"
    )


@dataclass(slots=True)
class QQP(BaseMultipleChoiceSuite):
    # Evaluate Quora Question Pairs with yes versus no label ranking on the GLUE validation split.
    """Implement the qqp benchmark suite."""
    dataset_path: str = "nyu-mll/glue"
    dataset_name: str | None = "qqp"
    split: str = "validation"

    # Use the canonical Hugging Face datasets loader for the GLUE QQP task.
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    # Return the stable suite identifier used by logs, YAML specs, and serialized results.
    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "qqp"

    # Convert one QQP row into the shared binary-choice sample structure.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        return MultipleChoiceSample(
            index=index,
            prompt=_qqp_prompt(doc["question1"], doc["question2"]),
            choices=["no", "yes"],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )

    # Report positive-class F1 because QQP is commonly tracked with both accuracy and duplicate F1.
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


# Mirror the public suite factory style used across the package.
def qqp(**kwargs: Any) -> QQP:
    """Implement qqp for this module."""
    return QQP(**kwargs)
