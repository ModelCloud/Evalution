# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.qa_text import best_qa_scores, canonicalize_no_answer

_STOP_STRINGS = ("\n", "\nQuestion:", "\nPassage:")


def _drop_prompt(*, passage: str, question: str) -> str:
    return f"Passage: {passage.strip()}\nQuestion: {question.strip()}\nAnswer:"


def _answer_spans(doc: dict[str, Any]) -> list[str]:
    deduped: list[str] = []
    for span in doc["answers_spans"]["spans"]:
        text = str(span).strip()
        if text and text not in deduped:
            deduped.append(text)
    if not deduped:
        raise ValueError("drop requires at least one answer span")
    return deduped


@dataclass(slots=True)
class DROP(BaseTestSuite):
    dataset_path: str = "drop"
    dataset_name: str | None = None
    split: str = "validation"
    max_rows: int | None = None
    max_new_tokens: int = 32
    batch_size: int | None = None
    cache_dir: str | None = None
    stream: bool = False
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "drop"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            answers = _answer_spans(doc)
            yield PreparedSample(
                index=index,
                doc=doc,
                target=answers[0],
                request=GenerationRequest(
                    prompt=_drop_prompt(
                        passage=str(doc["passage"]),
                        question=str(doc["question"]),
                    ),
                    stop=list(_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        answers = _answer_spans(prepared_sample.doc)
        exact, f1_score, best_index = best_qa_scores(output.text, answers)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-normalized": canonicalize_no_answer(output.text),
                "best_answer_index": str(best_index),
                "best_answer": answers[best_index],
            },
            scores={
                "em": exact,
                "f1": f1_score,
            },
            metadata={
                "section_id": str(prepared_sample.doc["section_id"]),
                "query_id": str(prepared_sample.doc["query_id"]),
                "question": str(prepared_sample.doc["question"]),
                "answer_spans": answers,
                "answer_types": [str(kind) for kind in prepared_sample.doc["answers_spans"]["types"]],
            },
        )


def drop(**kwargs: Any) -> DROP:
    return DROP(**kwargs)
