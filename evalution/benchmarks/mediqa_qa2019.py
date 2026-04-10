# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rouge_score import rouge_scorer
import sacrebleu

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.datasets import (
    MEDIQA_QA_DATASET_NAME,
    MEDIQA_QA_DATASET_PATH,
    MEDIQA_QA_SOURCE_SHA256,
    MEDIQA_QA_SOURCE_URL,
    load_mediqa_qa_dataset,
)
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult

# Keep the task prompt aligned with the upstream task description while using the vendored XML loader.
_MEDIQA_QA2019_INSTRUCTION = (
    "Instructions: The following text is a question asked by a patient. "
    "Answer how a doctor would, while trying to be as informative and helpful as possible."
)
_MEDIQA_QA2019_STOP_STRINGS = ("\n\n",)


def _mediqa_qa2019_prompt(question: str) -> str:
    return f"{_MEDIQA_QA2019_INSTRUCTION}\n\nQuestion: {question.strip()}\n\nAnswer:"


@lru_cache(maxsize=1)
def _mediqa_qa2019_rouge_scorer() -> rouge_scorer.RougeScorer:
    # The upstream task uses non-stemmed overlap metrics, so keep the local scorer equally literal.
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def _zero_summary_scores() -> dict[str, float]:
    # Empty predictions or references should count as zero overlap instead of poisoning the suite aggregate.
    return {
        "bleu": 0.0,
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
    }


def _mediqa_qa2019_answer_scores(prediction: str, reference: str) -> dict[str, float]:
    prediction_text = prediction.strip()
    reference_text = reference.strip()
    if not prediction_text or not reference_text:
        return _zero_summary_scores()
    rouge_scores = _mediqa_qa2019_rouge_scorer().score(reference_text, prediction_text)
    bleu_score = sacrebleu.sentence_bleu(prediction_text, [reference_text]).score / 100.0
    return {
        "bleu": float(bleu_score),
        "rouge1": float(rouge_scores["rouge1"].fmeasure),
        "rouge2": float(rouge_scores["rouge2"].fmeasure),
        "rougeL": float(rouge_scores["rougeL"].fmeasure),
    }


@dataclass(slots=True)
class MediqaQA2019(BaseTestSuite):
    # MEDIQA 2019 QA evaluates whether a model can answer patient questions in a clinically helpful style.
    dataset_path: str = MEDIQA_QA_DATASET_PATH
    dataset_name: str | None = MEDIQA_QA_DATASET_NAME
    split: str = "test"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        return load_mediqa_qa_dataset

    def task_name(self) -> str:
        return "mediqa_qa2019"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_medical_answer_quality",
            "primary_metric": "rouge1",
            "prompt_variant": "instruction_plus_patient_question",
            "source_url": MEDIQA_QA_SOURCE_URL,
            "source_sha256": MEDIQA_QA_SOURCE_SHA256,
            "omitted_upstream_metrics": ["bert_score", "bleurt"],
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            question_blob = dict(doc["QUESTION"])
            answers = list(question_blob["AnswerList"])
            target = str(answers[0]["Answer"]["AnswerText"]).strip()
            question_text = str(question_blob["QuestionText"]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=_mediqa_qa2019_prompt(question_text),
                    stop=list(_MEDIQA_QA2019_STOP_STRINGS),
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
        prediction = output.text.strip()
        reference = prepared_sample.target.strip()
        first_answer = prepared_sample.doc["QUESTION"]["AnswerList"][0]["Answer"]
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": prediction,
                "reference-stripped": reference,
            },
            scores=_mediqa_qa2019_answer_scores(prediction, reference),
            metadata={
                "qid": str(prepared_sample.doc["QUESTION"]["QID"]),
                "answer_count": len(prepared_sample.doc["QUESTION"]["AnswerList"]),
                "first_answer_aid": str(first_answer["AID"]),
                "first_answer_reference_rank": int(first_answer["ReferenceRank"]),
                "first_answer_reference_score": int(first_answer["ReferenceScore"]),
            },
        )


def mediqa_qa2019(**kwargs: Any) -> MediqaQA2019:
    return MediqaQA2019(**kwargs)
