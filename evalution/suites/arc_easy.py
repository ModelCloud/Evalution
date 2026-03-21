from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.suites.multiple_choice_utils import choice_index_from_labels, question_answer_prompt


@dataclass(slots=True)
class ARCEasy(BaseMultipleChoiceSuite):
    # Evaluate the ARC easy split by ranking answer options with token log-likelihood instead of generation parsing.
    dataset_path: str = "allenai/ai2_arc"
    dataset_name: str | None = "ARC-Easy"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the ARC easy benchmark split.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "arc_easy"

    # Convert one ARC easy row into the shared prompt and multiple-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        labels = list(doc["choices"]["label"])
        texts = list(doc["choices"]["text"])
        return MultipleChoiceSample(
            index=index,
            prompt=question_answer_prompt(doc["question"]),
            choices=texts,
            gold_index=choice_index_from_labels(labels, doc["answerKey"]),
            metadata={"id": doc["id"], "choice_labels": labels},
        )


# Mirror the public suite factory style used by the rest of the package.
def arc_easy(**kwargs: Any) -> ARCEasy:
    return ARCEasy(**kwargs)
