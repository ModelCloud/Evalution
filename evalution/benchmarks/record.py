# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pcre
from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.scorers.qa_text import best_qa_scores

_RECORD_HIGHLIGHT_AFTER_PUNCT_RE = pcre.compile(r"(\.|\?|\!|\"|')\n@highlight\n")
_RECORD_HIGHLIGHT_RE = pcre.compile(r"\n@highlight\n")


def _record_passage(passage: str) -> str:
    passage = _RECORD_HIGHLIGHT_AFTER_PUNCT_RE.sub(r"\1 ", passage)
    return _RECORD_HIGHLIGHT_RE.sub(". ", passage)


def _record_prompt(doc: dict[str, Any]) -> str:
    return " ".join(
        [
            "record query:",
            str(doc["query"]),
            "entities:",
            ", ".join(str(entity) for entity in doc["entities"]),
            "passage:",
            _record_passage(str(doc["passage"])),
        ]
    )


def _record_answers(doc: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for answer in doc["answers"]:
        text = str(answer).strip()
        if text and text not in answers:
            answers.append(text)
    if not answers:
        raise ValueError("record requires at least one gold answer")
    return answers


@dataclass(slots=True)
class ReCoRD(BaseMultipleChoiceSuite):
    dataset_path: str = "super_glue"
    dataset_name: str | None = "record"
    split: str = "validation"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "record"

    def result_metadata(self) -> dict[str, Any]:
        return {
            **super().result_metadata(),
            "primary_metric": "f1",
        }

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        answers = _record_answers(doc)
        metadata: dict[str, Any] = {
            "query": str(doc["query"]),
            "answers": answers,
            "entities": [str(entity) for entity in doc["entities"]],
        }
        if "idx" in doc:
            metadata["idx"] = doc["idx"]
        return MultipleChoiceSample(
            index=index,
            prompt=_record_prompt(doc),
            choices=list(metadata["entities"]),
            gold_index=metadata["entities"].index(answers[0]) if answers[0] in metadata["entities"] else 0,
            metadata=metadata,
        )

    def extra_metrics(
        self,
        *,
        samples: list[MultipleChoiceSample],
        raw_predictions: list[int],
        normalized_predictions: list[int],
    ) -> dict[str, float]:
        del normalized_predictions
        total_exact = 0.0
        total_f1 = 0.0
        for sample, raw_prediction_index in zip(samples, raw_predictions, strict=True):
            prediction = sample.choices[raw_prediction_index]
            answers = list(sample.metadata["answers"])
            exact, f1_score, _best_index = best_qa_scores(prediction, answers)
            total_exact += exact
            total_f1 += f1_score
        denominator = max(len(samples), 1)
        return {
            "em": total_exact / denominator,
            "f1": total_f1 / denominator,
        }


def record(**kwargs: Any) -> ReCoRD:
    return ReCoRD(**kwargs)
