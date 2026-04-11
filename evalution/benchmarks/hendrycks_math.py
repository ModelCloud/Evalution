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
_STOP_STRINGS = ("Problem:", "</s>", "<|im_end|>", "<|eot_id|>")
HENDRYCKS_MATH_SUBSETS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
HENDRYCKS_MATH_TASKS = tuple(f"hendrycks_math_{subset}" for subset in HENDRYCKS_MATH_SUBSETS)
_SUBSET_TO_TASK = dict(zip(HENDRYCKS_MATH_SUBSETS, HENDRYCKS_MATH_TASKS, strict=True))


def _hendrycks_math_prompt(problem: str) -> str:
    """Implement hendrycks math prompt for this module."""
    return f"Problem: {problem}\nAnswer:"


def _safe_normalize_math_string(text: str) -> str:
    """Implement safe normalize math string for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    try:
        return normalize_math_string(text)
    except Exception:
        return text


@dataclass(slots=True)
class HendrycksMath(BaseTestSuite):
    """Implement the hendrycks math benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "EleutherAI/hendrycks_math"
    dataset_name: str | None = None
    split: str = "test"
    subset: str = "algebra"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.subset not in HENDRYCKS_MATH_SUBSETS:
            raise ValueError(f"unsupported hendrycks_math subset: {self.subset!r}")
        if self.split != "test":
            raise ValueError("hendrycks_math uses the test split for evaluation")
        if self.dataset_name not in {None, self.subset}:
            raise ValueError("hendrycks_math dataset_name must match subset")
        self.dataset_name = self.subset

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return _SUBSET_TO_TASK[self.subset]

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
                target=extract_math_answer(str(doc["solution"])),
                request=GenerationRequest(
                    prompt=_hendrycks_math_prompt(str(doc["problem"])),
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
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=target,
            prediction=output.text,
            extracted={
                "prediction-stripped": output.text.strip(),
                "answer-extract": extracted_answer,
                "prediction-normalized": _safe_normalize_math_string(extracted_answer),
                "target-normalized": _safe_normalize_math_string(target),
            },
            scores={"em": math_exact_match(output.text, target)},
            metadata={
                "subset": self.subset,
                "level": prepared_sample.doc["level"],
                "problem_type": prepared_sample.doc["type"],
            },
        )


def hendrycks_math(*, subset: str, **kwargs: Any) -> HendrycksMath:
    """Implement hendrycks math for this module."""
    return HendrycksMath(subset=subset, dataset_name=subset, **kwargs)


def _make_hendrycks_math_factory(subset: str) -> Any:
    """Make hendrycks math factory."""
    def factory(**kwargs: Any) -> HendrycksMath:
        """Implement factory for this module."""
        return hendrycks_math(subset=subset, **kwargs)

    factory.__name__ = _SUBSET_TO_TASK[subset]
    return factory


for _subset in HENDRYCKS_MATH_SUBSETS:
    globals()[_SUBSET_TO_TASK[_subset]] = _make_hendrycks_math_factory(_subset)

del _subset
