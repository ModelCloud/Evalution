from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _wnli_prompt(sentence1: str, sentence2: str) -> str:
    # Phrase the Winograd NLI pair as a direct true-or-false question over the second sentence.
    return f"{sentence1.strip()}\nQuestion: {sentence2.strip()} True or False?\nAnswer:"


@dataclass(slots=True)
class WNLI(BaseMultipleChoiceSuite):
    # Evaluate Winograd NLI sentence-pair inference with False versus True label ranking.
    dataset_path: str = "nyu-mll/glue"
    dataset_name: str | None = "wnli"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical WNLI task inside GLUE.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "wnli"

    # Convert one WNLI row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_wnli_prompt(doc["sentence1"], doc["sentence2"]),
            choices=["False", "True"],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"])},
        )


# Mirror the public suite factory style used by the rest of the package.
def wnli(**kwargs: Any) -> WNLI:
    return WNLI(**kwargs)
