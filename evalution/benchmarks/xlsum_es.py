# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.datasets.xlsum import XLSUM_ARCHIVES, load_xlsum_dataset
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.summary_rouge import summary_rouge_scores


def _xlsum_es_prompt(text: str) -> str:
    """Implement XLSum es prompt for this module."""
    return f"Texto: {text.strip()}\n\nResumen:"


@dataclass(slots=True)
class XLSUMES(BaseTestSuite):
    # Evaluate the Spanish XLSum split with the local audited archive loader and summary ROUGE scoring.
    """Implement the xlsumes benchmark suite."""
    dataset_path: str = "csebuetnlp/xlsum"
    dataset_name: str | None = "spanish"
    split: str = "test"
    max_new_tokens: int = 128
    stop: tuple[str, ...] = ("\n",)

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.dataset_name not in {None, "spanish"}:
            raise ValueError("xlsum_es dataset_name must be None or 'spanish'")
        if self.dataset_name is None:
            self.dataset_name = "spanish"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_xlsum_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "xlsum_es"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        archive_spec = XLSUM_ARCHIVES["spanish"]
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_summary_rouge",
            "primary_metric": "rougeLsum",
            "archive_filename": str(archive_spec["filename"]),
            "archive_sha256": str(archive_spec["sha256"]),
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            text = str(doc["text"]).strip()
            reference = str(doc["summary"]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=reference,
                request=GenerationRequest(
                    prompt=_xlsum_es_prompt(text),
                    stop=list(self.stop),
                    max_new_tokens=self.max_new_tokens,
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
                "url": str(prepared_sample.doc["url"]),
                "title": str(prepared_sample.doc["title"]),
                "article_chars": len(str(prepared_sample.doc["text"])),
                "reference_lines": len(reference.splitlines()),
            },
        )


def xlsum_es(**kwargs: Any) -> XLSUMES:
    """Implement XLSum es for this module."""
    return XLSUMES(**kwargs)
