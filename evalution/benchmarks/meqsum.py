# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import sacrebleu

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.datasets import (
    MEQSUM_DATASET_NAME,
    MEQSUM_DATASET_PATH,
    MEQSUM_SOURCE_SHA256,
    MEQSUM_SOURCE_URL,
    load_meqsum_dataset,
)
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.rouge import RougeScorer

# Preserve the upstream instruction string and stop behavior while keeping the loader local and auditable.
_MEQSUM_INSTRUCTION = (
    "Instructions: The following text is contains a medical question. "
    "Extract and summarize the question."
)
_MEQSUM_STOP_STRINGS = ("\n\n",)


def _meqsum_question_text(text: str) -> str:
    """Implement meqsum question text for this module."""
    marker = "MESSAGE"
    index = text.find(marker)
    if index == -1:
        return text.strip()
    return text[index + len(marker) + 1 :].strip()


def _meqsum_prompt(question_text: str) -> str:
    """Implement meqsum prompt for this module."""
    return f"{_MEQSUM_INSTRUCTION}\n\n{question_text.strip()}"


@lru_cache(maxsize=1)
def _meqsum_rouge_scorer() -> RougeScorer:
    # Match the upstream MeQSum task's non-stemmed ROUGE scoring for short question summaries.
    """Implement meqsum ROUGE scorer for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def _meqsum_summary_scores(prediction: str, reference: str) -> dict[str, float]:
    """Implement meqsum summary scores for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    rouge_scores = _meqsum_rouge_scorer().score(reference, prediction)
    bleu = sacrebleu.sentence_bleu(prediction, [reference]).score / 100.0
    return {
        "bleu": float(bleu),
        "rouge1": float(rouge_scores["rouge1"].fmeasure),
        "rouge2": float(rouge_scores["rouge2"].fmeasure),
        "rougeL": float(rouge_scores["rougeL"].fmeasure),
    }


@dataclass(slots=True)
class MeQSum(BaseTestSuite):
    # MeQSum summarizes long consumer-health questions into the minimal medical information need.
    """Implement the me qsum benchmark suite."""
    dataset_path: str = MEQSUM_DATASET_PATH
    dataset_name: str | None = MEQSUM_DATASET_NAME
    split: str = "train"
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_meqsum_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "meqsum"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_medical_question_summary",
            "primary_metric": "rouge1",
            "prompt_variant": "instruction_plus_source_question",
            "source_url": MEQSUM_SOURCE_URL,
            "source_sha256": MEQSUM_SOURCE_SHA256,
            "omitted_upstream_metrics": ["bert_score", "bleurt"],
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            question_text = _meqsum_question_text(str(doc["CHQ"]))
            reference = str(doc["Summary"]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=reference,
                request=GenerationRequest(
                    prompt=_meqsum_prompt(question_text),
                    stop=list(_MEQSUM_STOP_STRINGS),
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
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        prediction = output.text.strip()
        reference = prepared_sample.target.strip()
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": prediction,
                "reference-stripped": reference,
            },
            scores=_meqsum_summary_scores(prediction, reference),
            metadata={
                "file": str(prepared_sample.doc["File"]),
                "question_chars": len(str(prepared_sample.doc["CHQ"])),
                "summary_words": len(reference.split()),
            },
        )


def meqsum(**kwargs: Any) -> MeQSum:
    """Implement meqsum for this module."""
    return MeQSum(**kwargs)
