from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _wic_target_word(doc: dict[str, Any]) -> str:
    # Extract the focus word from the first sentence using the dataset-provided character offsets.
    return str(doc["sentence1"])[int(doc["start1"]) : int(doc["end1"])]


def _wic_prompt(doc: dict[str, Any]) -> str:
    # Format the two-sentence word-sense question using the canonical same-meaning prompt wording.
    target_word = _wic_target_word(doc)
    return (
        f"Sentence 1: {str(doc['sentence1']).strip()}\n"
        f"Sentence 2: {str(doc['sentence2']).strip()}\n"
        f"Question: Is the word '{target_word}' used in the same way in the two sentences above?\n"
        "Answer:"
    )


@dataclass(slots=True)
class WiC(BaseMultipleChoiceSuite):
    # Evaluate word-in-context disambiguation by ranking yes/no continuations for the paired sentences.
    dataset_path: str = "super_glue"
    dataset_name: str | None = "wic"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the WiC task packaged inside SuperGLUE.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "wic"

    # Convert one WiC row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_wic_prompt(doc),
            choices=["no", "yes"],
            gold_index=int(doc["label"]),
            metadata={"idx": int(doc["idx"]), "word": str(doc["word"])},
        )


# Mirror the public suite factory style used by the rest of the package.
def wic(**kwargs: Any) -> WiC:
    return WiC(**kwargs)
