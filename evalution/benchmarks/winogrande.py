# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _split_winogrande_sentence(sentence: str) -> tuple[str, str]:
    # Split the cloze template once so each answer choice can reuse the shared trailing suffix.
    prefix, suffix = sentence.split("_", maxsplit=1)
    return prefix.rstrip(), suffix.strip()


@dataclass(slots=True)
class WinoGrande(BaseMultipleChoiceSuite):
    # Evaluate commonsense pronoun resolution by ranking the two blank-filled sentence variants.
    dataset_path: str = "winogrande"
    dataset_name: str | None = "winogrande_xl"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the public WinoGrande benchmark.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "winogrande"

    # Convert one WinoGrande row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        prefix, suffix = _split_winogrande_sentence(doc["sentence"])
        choices = [
            f"{doc['option1']} {suffix}",
            f"{doc['option2']} {suffix}",
        ]
        return MultipleChoiceSample(
            index=index,
            prompt=prefix,
            choices=choices,
            gold_index=int(doc["answer"]) - 1,
            metadata={
                "sentence": doc["sentence"],
                "option_texts": [doc["option1"], doc["option2"]],
            },
        )

    # The optional label-permutation scorer rewrites WinoGrande back into an explicit blank-filling
    # question so label scoring compares the candidate fillers rather than the raw continuation length.
    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        lines = [
            f"Sentence: {sample.metadata['sentence']}",
            "Question: Which option best fills the blank?",
        ]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {sample.metadata['option_texts'][choice_index]}")
        lines.append("Answer:")
        return "\n".join(lines)


# Mirror the public suite factory style used by the rest of the package.
def winogrande(**kwargs: Any) -> WinoGrande:
    return WinoGrande(**kwargs)
