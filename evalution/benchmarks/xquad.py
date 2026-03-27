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

XQUAD_LANGUAGES = ("ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh")
XQUAD_TASKS = tuple(f"xquad_{language}" for language in XQUAD_LANGUAGES)
_STOP_STRINGS = ("\n", "\nQuestion:", "\nContext:")


def _xquad_prompt(*, context: str, question: str) -> str:
    return f"Context: {context.strip()}\n\nQuestion: {question.strip()}\n\nAnswer:"


def _xquad_answer_texts(doc: dict[str, Any]) -> list[str]:
    answers = doc["answers"]["text"]
    deduped: list[str] = []
    for answer in answers:
        text = str(answer).strip()
        if text and text not in deduped:
            deduped.append(text)
    if not deduped:
        raise ValueError("xquad requires at least one non-empty answer")
    return deduped


@dataclass(slots=True)
class XQuAD(BaseTestSuite):
    dataset_path: str = "google/xquad"
    dataset_name: str | None = "xquad.en"
    split: str = "validation"
    language: str = "en"
    max_rows: int | None = None
    max_new_tokens: int = 32
    batch_size: int | None = None
    cache_dir: str | None = None
    stream: bool = True
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.language not in XQUAD_LANGUAGES:
            raise ValueError(f"unsupported xquad language: {self.language!r}")
        expected_dataset_name = f"xquad.{self.language}"
        if self.dataset_name in {None, expected_dataset_name}:
            self.dataset_name = expected_dataset_name
            return
        raise ValueError("xquad dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"xquad_{self.language}"

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
            answers = _xquad_answer_texts(doc)
            yield PreparedSample(
                index=index,
                doc=doc,
                target=answers[0],
                request=GenerationRequest(
                    prompt=_xquad_prompt(
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
        answers = _xquad_answer_texts(prepared_sample.doc)
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
                "id": str(prepared_sample.doc["id"]),
                "language": self.language,
                "question": str(prepared_sample.doc["question"]),
                "context": str(prepared_sample.doc["context"]),
                "answer_texts": answers,
            },
        )


def xquad(*, language: str, **kwargs: Any) -> XQuAD:
    kwargs.setdefault("dataset_name", f"xquad.{language}")
    return XQuAD(language=language, **kwargs)


def _make_xquad_factory(language: str) -> Any:
    def factory(**kwargs: Any) -> XQuAD:
        return xquad(language=language, **kwargs)

    factory.__name__ = f"xquad_{language}"
    return factory


for _language in XQUAD_LANGUAGES:
    globals()[f"xquad_{_language}"] = _make_xquad_factory(_language)

del _language
