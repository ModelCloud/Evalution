from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.classification_metrics import macro_f1
from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_CB_CHOICES = ["True", "False", "Neither"]


def _cb_prompt(premise: str, hypothesis: str) -> str:
    # Match the benchmark's textual entailment framing with an explicit three-way answer slot.
    return f"{premise.strip()}\nQuestion: {hypothesis.strip()}. True, False, or Neither?\nAnswer:"


@dataclass(slots=True)
class CB(BaseMultipleChoiceSuite):
    # Evaluate commitment bank entailment as a three-choice log-likelihood ranking task.
    dataset_path: str = "super_glue"
    dataset_name: str | None = "cb"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical CommitmentBank task.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "cb"

    # Convert one CB row into the shared prompt and three-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_cb_prompt(doc["premise"], doc["hypothesis"]),
            choices=list(_CB_CHOICES),
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )

    # Report macro F1 for both raw and length-normalized predictions because CB is commonly tracked with both.
    def extra_metrics(
        self,
        *,
        samples: list[MultipleChoiceSample],
        raw_predictions: list[int],
        normalized_predictions: list[int],
    ) -> dict[str, float]:
        gold_labels = [sample.gold_index for sample in samples]
        label_indices = list(range(len(_CB_CHOICES)))
        return {
            "f1,loglikelihood_macro": macro_f1(
                gold_labels,
                raw_predictions,
                labels=label_indices,
            ),
            "f1,loglikelihood_norm_macro": macro_f1(
                gold_labels,
                normalized_predictions,
                labels=label_indices,
            ),
        }


# Mirror the public suite factory style used by the rest of the package.
def cb(**kwargs: Any) -> CB:
    return CB(**kwargs)
