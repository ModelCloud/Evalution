from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.classification_metrics import f1_for_label
from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _mrpc_prompt(sentence1: str, sentence2: str) -> str:
    # Frame paraphrase detection as a direct yes-or-no question over the sentence pair.
    return (
        f"Sentence 1: {sentence1.strip()}\n"
        f"Sentence 2: {sentence2.strip()}\n"
        "Question: Do both sentences mean the same thing?\n"
        "Answer:"
    )


@dataclass(slots=True)
class MRPC(BaseMultipleChoiceSuite):
    # Evaluate Microsoft Research Paraphrase Corpus sentence-pair classification with yes/no label ranking.
    dataset_path: str = "nyu-mll/glue"
    dataset_name: str | None = "mrpc"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical MRPC task inside GLUE.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "mrpc"

    # Convert one MRPC row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
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
        gold_labels = [sample.gold_index for sample in samples]
        return {
            "f1,loglikelihood_yes": f1_for_label(gold_labels, raw_predictions, label=1),
            "f1,loglikelihood_norm_yes": f1_for_label(
                gold_labels,
                normalized_predictions,
                label=1,
            ),
        }


# Mirror the public suite factory style used by the rest of the package.
def mrpc(**kwargs: Any) -> MRPC:
    return MRPC(**kwargs)
