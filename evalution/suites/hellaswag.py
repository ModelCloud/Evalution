# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pcre
from datasets import load_dataset

from evalution.suites.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_BRACKET_ARTIFACT_PATTERN = pcre.compile(r"\[.*?\]")


def _clean_hellaswag_text(text: str) -> str:
    # Strip WikiHow artifacts and collapse spacing so scored prompts stay comparable across rows.
    cleaned = text.strip().replace(" [title]", ". ")
    cleaned = _BRACKET_ARTIFACT_PATTERN.sub("", cleaned)
    return " ".join(cleaned.split())


@dataclass(slots=True)
class HellaSwag(BaseMultipleChoiceSuite):
    # Evaluate commonsense completion via log-likelihood ranking over four candidate endings.
    dataset_path: str = "Rowan/hellaswag"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the canonical HellaSwag benchmark.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the public suite name used in logs, results, and factory wiring.
    def task_name(self) -> str:
        return "hellaswag"

    # Normalize one raw row into the prompt and choice structure shared by multiple-choice suites.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        context = f"{doc['ctx_a']} {doc['ctx_b'].capitalize()}"
        prompt = _clean_hellaswag_text(f"{doc['activity_label']}: {context}")
        choices = [_clean_hellaswag_text(choice) for choice in doc["endings"]]
        return MultipleChoiceSample(
            index=index,
            prompt=prompt,
            choices=choices,
            gold_index=int(doc["label"]),
            metadata={
                "activity_label": doc["activity_label"],
                "source_id": doc["source_id"],
                "split_type": doc["split_type"],
            },
        )

    # The optional label-permutation scorer keeps HellaSwag as an explicit completion choice task
    # instead of scoring raw ending text directly, which helps separate label priors from ending length.
    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        lines = [
            f"Context: {sample.prompt}",
            "Question: Which ending best continues the context?",
        ]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {sample.choices[choice_index]}")
        lines.append("Answer:")
        return "\n".join(lines)


# Mirror the public suite factory style used by the rest of the package.
def hellaswag(**kwargs: Any) -> HellaSwag:
    return HellaSwag(**kwargs)
