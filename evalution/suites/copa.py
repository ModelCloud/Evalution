from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _copa_connector(question: str) -> str:
    # Map the COPA relation type to the textual connector used by the original benchmark prompt.
    return {
        "cause": "because",
        "effect": "therefore",
    }[question]


def _copa_choice_text(choice: str) -> str:
    # Lowercase the first character so the continuation joins naturally after the prompt connector.
    return choice[:1].lower() + choice[1:]


@dataclass(slots=True)
class COPA(BaseMultipleChoiceSuite):
    # Evaluate causal commonsense reasoning by ranking the two candidate sentence completions.
    dataset_path: str = "super_glue"
    dataset_name: str | None = "copa"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the COPA task packaged inside SuperGLUE.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "copa"

    # Convert one COPA row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        premise = doc["premise"].strip()
        prompt = premise[:-1] if premise.endswith(".") else premise
        prompt = f"{prompt} {_copa_connector(doc['question'])}"
        return MultipleChoiceSample(
            index=index,
            prompt=prompt,
            choices=[
                _copa_choice_text(doc["choice1"]),
                _copa_choice_text(doc["choice2"]),
            ],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"]), "question": doc["question"]},
        )


# Mirror the public suite factory style used by the rest of the package.
def copa(**kwargs: Any) -> COPA:
    return COPA(**kwargs)
