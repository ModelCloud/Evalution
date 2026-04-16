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

# Keep benchmark defaults and public task ids explicit at module scope.
_STOP_STRINGS = ("Question:", "</s>", "<|im_end|>", "<|eot_id|>")
HMMT_TASKS = ("hmmt_feb25", "hmmt_nov25", "hmmt_feb26")
_HMMT_VARIANTS = {
    "hmmt_feb25": {
        "dataset_path": "MathArena/hmmt_feb_2025",
        "split": "train",
    },
    "hmmt_nov25": {
        "dataset_path": "MathArena/hmmt_nov_2025",
        "split": "train",
    },
    "hmmt_feb26": {
        "dataset_path": "MathArena/hmmt_feb_2026",
        "split": "train",
    },
}


def _hmmt_prompt(problem: str) -> str:
    """Implement hmmt prompt for this module."""
    return f"Question: {problem}\nAnswer:"


@dataclass(slots=True)
class HMMT(BaseTestSuite):
    """Implement the HMMT benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "MathArena/hmmt_feb_2025"
    dataset_name: str | None = None
    split: str = "train"
    variant_name: str = "hmmt_feb25"
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return self.variant_name

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_math_exact_match",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=str(doc["answer"]),
                request=GenerationRequest(
                    prompt=_hmmt_prompt(str(doc["problem"])),
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
        target = prepared_sample.target
        extracted_answer = extract_math_answer(output.text)
        metadata = {
            "problem_id": int(prepared_sample.doc["problem_idx"]),
        }
        if "problem_type" in prepared_sample.doc:
            metadata["problem_type"] = list(prepared_sample.doc["problem_type"])
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
            metadata=metadata,
        )


def _hmmt_variant(variant_name: str, **kwargs: Any) -> HMMT:
    """Implement hmmt variant for this module."""
    variant = _HMMT_VARIANTS[variant_name]
    return HMMT(
        dataset_path=variant["dataset_path"],
        split=variant["split"],
        variant_name=variant_name,
        **kwargs,
    )


def hmmt_feb25(**kwargs: Any) -> HMMT:
    """Implement hmmt_feb25 for this module."""
    return _hmmt_variant("hmmt_feb25", **kwargs)


def hmmt_nov25(**kwargs: Any) -> HMMT:
    """Implement hmmt_nov25 for this module."""
    return _hmmt_variant("hmmt_nov25", **kwargs)


def hmmt_feb26(**kwargs: Any) -> HMMT:
    """Implement hmmt_feb26 for this module."""
    return _hmmt_variant("hmmt_feb26", **kwargs)
