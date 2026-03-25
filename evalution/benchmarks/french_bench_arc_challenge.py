# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.benchmarks.multiple_choice_utils import choice_index_from_labels


def _french_bench_arc_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nRéponse:"


@dataclass(slots=True)
class FrenchBenchARCChallenge(BaseMultipleChoiceSuite):
    # Score the French ARC-Challenge subset with the same raw and length-normalized choice ranking
    # used by lm-eval for this benchmark family.
    dataset_path: str = "manu/french_bench_arc_challenge"
    dataset_name: str | None = None
    split: str = "test"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "french_bench_arc_challenge"

    def continuation_for_choice(self, choice: str) -> str:
        return choice

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choice_labels = ["A", "B", "C", "D"]
        return MultipleChoiceSample(
            index=index,
            prompt=_french_bench_arc_prompt(str(doc["question"])),
            choices=list(doc["choices"]),
            gold_index=choice_index_from_labels(choice_labels, str(doc["answerKey"])),
            metadata={
                "id": str(doc["id"]),
                "choice_labels": choice_labels,
            },
        )


def french_bench_arc_challenge(**kwargs: Any) -> FrenchBenchARCChallenge:
    return FrenchBenchARCChallenge(**kwargs)
