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
from evalution.scorers.math_exact_match import extract_math_answer, math_exact_match, normalize_math_string

_STOP_STRINGS = ("Question:", "</s>", "<|im_end|>", "<|eot_id|>")
_AIME_VARIANTS = {
    "aime": {
        "dataset_path": "gneubig/aime-1983-2024",
        "split": "train",
        "question_field": "Question",
        "answer_field": "Answer",
        "id_field": "ID",
        "metadata_fields": ("Year", "Problem Number", "Part"),
    },
    "aime24": {
        "dataset_path": "Maxwell-Jia/AIME_2024",
        "split": "train",
        "question_field": "Problem",
        "answer_field": "Answer",
        "id_field": "ID",
        "metadata_fields": ("Solution",),
    },
    "aime25": {
        "dataset_path": "math-ai/aime25",
        "split": "test",
        "question_field": "problem",
        "answer_field": "answer",
        "id_field": "id",
        "metadata_fields": (),
    },
    "aime26": {
        "dataset_path": "math-ai/aime26",
        "split": "test",
        "question_field": "problem",
        "answer_field": "answer",
        "id_field": "id",
        "metadata_fields": (),
    },
}


def _aime_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


@dataclass(slots=True)
class AIME(BaseTestSuite):
    dataset_path: str = "gneubig/aime-1983-2024"
    dataset_name: str | None = None
    split: str = "train"
    variant_name: str = "aime"
    question_field: str = "Question"
    answer_field: str = "Answer"
    id_field: str = "ID"
    metadata_fields: tuple[str, ...] = ("Year", "Problem Number", "Part")
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return self.variant_name

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_math_exact_match",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=str(doc[self.answer_field]),
                request=GenerationRequest(
                    prompt=_aime_prompt(str(doc[self.question_field])),
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
        target = prepared_sample.target
        extracted_answer = extract_math_answer(output.text)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=target,
            prediction=output.text,
            extracted={
                "prediction-stripped": output.text.strip(),
                "answer-extract": extracted_answer,
                "prediction-normalized": normalize_math_string(extracted_answer),
                "target-normalized": normalize_math_string(target),
            },
            scores={"em": math_exact_match(output.text, target)},
            metadata={
                "problem_id": str(prepared_sample.doc[self.id_field]),
                **{
                    field.lower().replace(" ", "_"): prepared_sample.doc[field]
                    for field in self.metadata_fields
                    if field in prepared_sample.doc
                },
            },
        )


def _aime_variant(variant_name: str, **kwargs: Any) -> AIME:
    variant = _AIME_VARIANTS[variant_name]
    return AIME(
        dataset_path=variant["dataset_path"],
        split=variant["split"],
        variant_name=variant_name,
        question_field=variant["question_field"],
        answer_field=variant["answer_field"],
        id_field=variant["id_field"],
        metadata_fields=variant["metadata_fields"],
        **kwargs,
    )


def aime(**kwargs: Any) -> AIME:
    return _aime_variant("aime", **kwargs)


def aime24(**kwargs: Any) -> AIME:
    return _aime_variant("aime24", **kwargs)


def aime25(**kwargs: Any) -> AIME:
    return _aime_variant("aime25", **kwargs)


def aime26(**kwargs: Any) -> AIME:
    # Expose the standalone AIME 2026 dataset variant shipped in the recent lm-eval PR.
    return _aime_variant("aime26", **kwargs)
