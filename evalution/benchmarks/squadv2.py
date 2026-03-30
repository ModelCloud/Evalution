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

_NO_ANSWER_TOKEN = "unanswerable"
_STOP_STRINGS = ("\n", "\nQuestion:", "\nContext:")


def _squadv2_prompt(*, title: str, context: str, question: str) -> str:
    title_line = f"Title: {title.strip()}\n" if title.strip() else ""
    return (
        f"{title_line}Context: {context.strip()}\n"
        f"Question: {question.strip()}\n"
        f"If the question cannot be answered from the context, answer with "
        f"'{_NO_ANSWER_TOKEN}'.\n"
        "Answer:"
    )


def _answer_texts(doc: dict[str, Any]) -> list[str]:
    answers = doc["answers"]["text"]
    deduped: list[str] = []
    for answer in answers:
        text = str(answer).strip()
        if text and text not in deduped:
            deduped.append(text)
    if deduped:
        return deduped
    return [_NO_ANSWER_TOKEN]


@dataclass(slots=True)
class SQuADV2(BaseTestSuite):
    dataset_path: str = "squad_v2"
    dataset_name: str | None = "squad_v2"
    split: str = "validation"
    max_rows: int | None = None
    max_new_tokens: int = 32
    batch_size: int | None = None
    cache_dir: str | None = None
    stream: bool = (False)
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "squadv2"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
            "no_answer_token": _NO_ANSWER_TOKEN,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            answers = _answer_texts(doc)
            yield PreparedSample(
                index=index,
                doc=doc,
                target=answers[0],
                request=GenerationRequest(
                    prompt=_squadv2_prompt(
                        title=str(doc["title"]),
                        context=str(doc["context"]),
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
        answers = _answer_texts(prepared_sample.doc)
        exact, f1_score, best_index = best_qa_scores(output.text, answers)
        canonical_prediction = canonicalize_no_answer(output.text)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-normalized": canonical_prediction,
                "best_answer_index": str(best_index),
                "best_answer": answers[best_index],
            },
            scores={
                "em": exact,
                "f1": f1_score,
            },
            metadata={
                "id": str(prepared_sample.doc["id"]),
                "title": str(prepared_sample.doc["title"]),
                "question": str(prepared_sample.doc["question"]),
                "answer_texts": answers,
                "has_answer": answers != [_NO_ANSWER_TOKEN],
            },
        )


def squadv2(**kwargs: Any) -> SQuADV2:
    return SQuADV2(**kwargs)
