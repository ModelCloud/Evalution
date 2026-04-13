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
from evalution.scorers.summary_rouge import summary_rouge_scores


def _cnn_dailymail_prompt(article: str) -> str:
    """Implement CNN dailymail prompt for this module."""
    return (
        "Summarize the following news article.\n\n"
        f"Article:\n{article.strip()}\n\n"
        "Summary:"
    )


@dataclass(slots=True)
class CNNDailyMail(BaseTestSuite):
    """Implement the cnndaily mail benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "cnn_dailymail"
    dataset_name: str | None = "3.0.0"
    split: str = "validation"
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "cnn_dailymail"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_summary_rouge",
            "primary_metric": "rougeLsum",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            article = str(doc["article"]).strip()
            reference = str(doc["highlights"]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=reference,
                request=GenerationRequest(
                    prompt=_cnn_dailymail_prompt(article),
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
            scores=summary_rouge_scores(prediction, reference),
            metadata={
                "id": str(prepared_sample.doc["id"]),
                "article_chars": len(str(prepared_sample.doc["article"])),
                "reference_lines": len(reference.splitlines()),
            },
        )


def cnn_dailymail(**kwargs: Any) -> CNNDailyMail:
    """Implement CNN dailymail for this module."""
    return CNNDailyMail(**kwargs)
