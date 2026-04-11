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
from evalution.scorers.choice_label import exact_match

# Keep benchmark defaults and public task ids explicit at module scope.
_STOP_STRINGS = ("\n", "Passage:")


def _babi_prompt(passage: str, question: str) -> str:
    # Match the upstream bAbI formatting so generated continuations remain directly comparable.
    """Implement babi prompt for this module."""
    return f"Passage: {passage}Question: {question}\nAnswer:"


@dataclass(slots=True)
class BABI(BaseTestSuite):
    # Evaluate short-answer reading comprehension with raw generated exact match.
    """Implement the babi benchmark suite."""
    dataset_path: str = "Muennighoff/babi"
    dataset_name: str | None = None
    split: str = "test"
    max_new_tokens: int = 16
    do_sample: bool = False
    temperature: float = 0.0

    # Use the Hugging Face datasets loader for the public bAbI benchmark.
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "babi"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
        }

    # Convert dataset rows into plain generation requests and exact-match targets.
    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            prompt = _babi_prompt(str(doc["passage"]), str(doc["question"]).strip())
            yield PreparedSample(
                index=index,
                doc=doc,
                target=f" {str(doc['answer']).strip()}",
                request=GenerationRequest(
                    prompt=prompt,
                    stop=list(_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    # Score exact raw answer-string match against the generated continuation.
    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        score = exact_match(output.text, prepared_sample.target)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": output.text.strip(),
                "target-stripped": prepared_sample.target.strip(),
            },
            scores={"em": score},
            metadata={"task": int(prepared_sample.doc["task"])},
        )


# Mirror the public suite factory style used by the rest of the package.
def babi(**kwargs: Any) -> BABI:
    """Implement babi for this module."""
    return BABI(**kwargs)
