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

# Base FDA dataset and prompting constants from the lm-eval implementation.
FDA_DATASET_PATH = "hazyresearch/based-fda"
FDA_DATASET_NAME = "default"
FDA_PROMPT_STOP_STRINGS = ("\n",)


def _contains_target_prediction(prediction: str, target: str) -> bool:
    pattern = pcre.compile(pcre.escape(str(target)), pcre.IGNORECASE)
    return bool(pattern.search(prediction))


@dataclass(slots=True)
class FDA(BaseTestSuite):
    # FDA is a generated completion task where the target must be contained in output text.
    dataset_path: str = FDA_DATASET_PATH
    dataset_name: str | None = FDA_DATASET_NAME
    split: str = "validation"
    stream: bool = True
    max_new_tokens: int = 48
    do_sample: bool = False
    temperature: float = 0.0

    def task_name(self) -> str:
        return "fda"

    def dataset_loader(self) -> Any:
        return load_dataset

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_contains_match",
            "primary_metric": "contains",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            target = str(doc["value"])
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=str(doc["text"]),
                    stop=list(FDA_PROMPT_STOP_STRINGS),
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
        contained = _contains_target_prediction(output.text, prepared_sample.target)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "contains-target": str(int(contained)),
                "target": prepared_sample.target,
                "target-matched": str(int(bool(contained))),
            },
            scores={"contains": float(contained)},
            metadata={
                "doc_id": str(prepared_sample.doc["doc_id"]),
                "file_name": str(prepared_sample.doc["file_name"]),
                "key": str(prepared_sample.doc["key"]),
            },
        )

def fda(**kwargs: Any) -> FDA:
    return FDA(**kwargs)
