# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pcre
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult

# Based-SQuAD stops at the first newline because the task asks for a short answer-bearing continuation.
_SQUAD_COMPLETION_STOP_STRINGS = ("\n",)


def _contains_target_prediction(prediction: str, target: str) -> bool:
    """Implement contains target prediction for this module."""
    pattern = pcre.compile(pcre.escape(str(target)), pcre.IGNORECASE)
    return bool(pattern.search(prediction))


@dataclass(slots=True)
class SQuADCompletion(BaseTestSuite):
    # Based-SQuAD checks whether a completion of the truncated passage still contains the held-out answer span.
    """Implement the squ adcompletion benchmark suite."""
    dataset_path: str = "hazyresearch/based-squad"
    dataset_name: str | None = "default"
    split: str = "validation"
    stream: bool = True
    max_new_tokens: int = 48
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "squad_completion"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_contains_match",
            "primary_metric": "contains",
            "prompt_variant": "truncated_context_completion",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            target = str(doc["value"])
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=str(doc["text"]),
                    stop=list(_SQUAD_COMPLETION_STOP_STRINGS),
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
        contained = _contains_target_prediction(output.text, prepared_sample.target)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "contains-target": str(int(contained)),
                "target": prepared_sample.target,
                "target-matched": str(int(contained)),
            },
            scores={"contains": float(contained)},
            metadata={
                "doc_id": str(prepared_sample.doc["doc_id"]),
                "title": str(prepared_sample.doc["title"]),
                "question": str(prepared_sample.doc["question"]),
            },
        )


def squad_completion(**kwargs: Any) -> SQuADCompletion:
    """Implement SQuAD completion for this module."""
    return SQuADCompletion(**kwargs)
