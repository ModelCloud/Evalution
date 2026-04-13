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

# Keep benchmark defaults and public task ids explicit at module scope.
_STOP_STRINGS = ("\n", "\nQuestion:")


def _nq_open_prompt(question: str) -> str:
    """Implement nq open prompt for this module."""
    return f"Question: {question.strip()}\nAnswer:"


def _answer_aliases(doc: dict[str, Any]) -> list[str]:
    """Implement answer aliases for this module."""
    aliases: list[str] = []
    for answer in doc["answer"]:
        text = str(answer).strip()
        if text and text not in aliases:
            aliases.append(text)
    if not aliases:
        raise ValueError("nq_open requires at least one answer alias")
    return aliases


@dataclass(slots=True)
class NQOpen(BaseTestSuite):
    """Implement the nqopen benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "nq_open"
    dataset_name: str | None = "nq_open"
    split: str = "validation"
    max_rows: int | None = None
    max_new_tokens: int = 32
    batch_size: int | None = None
    cache_dir: str | None = None
    stream: bool = (False)
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "nq_open"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            aliases = _answer_aliases(doc)
            yield PreparedSample(
                index=index,
                doc=doc,
                target=aliases[0],
                request=GenerationRequest(
                    prompt=_nq_open_prompt(str(doc["question"])),
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
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
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
                "question": str(prepared_sample.doc["question"]),
                "answer_aliases": aliases,
            },
        )


def nq_open(**kwargs: Any) -> NQOpen:
    """Implement nq open for this module."""
    return NQOpen(**kwargs)
