# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample


def _copa_connector(question: str) -> str:
    # Map the COPA relation type to the textual connector used by the original benchmark prompt.
    """Implement COPA connector for this module."""
    return {
        "cause": "because",
        "effect": "therefore",
    }[question]


def _copa_choice_text(choice: str) -> str:
    # Lowercase the first character so the continuation joins naturally after the prompt connector.
    """Implement COPA choice text for this module."""
    return choice[:1].lower() + choice[1:]


@dataclass(slots=True)
class COPA(BaseMultipleChoiceSuite):
    # Evaluate causal commonsense reasoning by ranking the two candidate sentence completions.
    """Implement the COPA benchmark suite."""
    dataset_path: str = "super_glue"
    dataset_name: str | None = "copa"
    split: str = "validation"

    # Use the Hugging Face datasets loader for the COPA task packaged inside SuperGLUE.
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "copa"

    # Convert one COPA row into the shared prompt and binary-choice structure used by the helper.
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
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
            metadata={
                "idx": int(doc["idx"]),
                "question": doc["question"],
                "premise": doc["premise"].strip(),
                "raw_choices": [doc["choice1"].strip(), doc["choice2"].strip()],
            },
        )

    # The optional label-permutation scorer rewrites COPA as an explicit labeled-choice question
    # so it can score `A/B` labels while preserving cause-versus-effect semantics.
    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        """Implement label prompt for COPA."""
        relation = "cause" if sample.metadata["question"] == "cause" else "effect"
        lines = [
            f"Premise: {sample.metadata['premise']}",
            f"Question: Which option is the more likely {relation}?",
        ]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {sample.metadata['raw_choices'][choice_index]}")
        lines.append("Answer:")
        return "\n".join(lines)


# Mirror the public suite factory style used by the rest of the package.
def copa(**kwargs: Any) -> COPA:
    """Implement COPA for this module."""
    return COPA(**kwargs)
