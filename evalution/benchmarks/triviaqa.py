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

_STOP_STRINGS = ("\n", "\nQuestion:")


def _triviaqa_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def _answer_aliases(doc: dict[str, Any]) -> list[str]:
    answer = doc["answer"]
    aliases = []
    for alias in answer["aliases"]:
        text = str(alias).strip()
        if text and text not in aliases:
            aliases.append(text)
    if aliases:
        return aliases

    value = str(answer.get("value", "")).strip()
    if value:
        return [value]
    raise ValueError("triviaqa requires at least one answer alias")


@dataclass(slots=True)
class TriviaQA(BaseTestSuite):
    dataset_path: str = "trivia_qa"
    dataset_name: str | None = "rc.nocontext"
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
        return "triviaqa"

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
            aliases = _answer_aliases(doc)
            yield PreparedSample(
                index=index,
                doc=doc,
                target=aliases[0],
                request=GenerationRequest(
                    prompt=_triviaqa_prompt(str(doc["question"])),
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
        aliases = _answer_aliases(prepared_sample.doc)
        exact, f1_score, best_index = best_qa_scores(output.text, aliases)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-normalized": canonicalize_no_answer(output.text),
                "best_answer_index": str(best_index),
                "best_answer": aliases[best_index],
            },
            scores={
                "em": exact,
                "f1": f1_score,
            },
            metadata={
                "question_id": str(prepared_sample.doc["question_id"]),
                "question_source": str(prepared_sample.doc["question_source"]),
                "question": str(prepared_sample.doc["question"]),
                "answer_aliases": aliases,
                "answer_type": str(prepared_sample.doc["answer"]["type"]),
                "answer_value": str(prepared_sample.doc["answer"]["value"]),
            },
        )


def triviaqa(**kwargs: Any) -> TriviaQA:
    return TriviaQA(**kwargs)
