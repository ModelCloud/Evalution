# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.math_exact_match import extract_math_answer, math_exact_match, normalize_math_string

# Keep benchmark defaults and public task ids explicit at module scope.
IMOANSWERBENCH_DATASET_PATH = "google-deepmind/superhuman/imobench"
_IMOANSWERBENCH_DATA_URL = (
    "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench_v2.csv"
)
_STOP_STRINGS = ("Question:", "</s>", "<|im_end|>", "<|eot_id|>")


def _imoanswerbench_dataset_loader() -> Callable[..., Any]:
    # The official release lives in the DeepMind repo as a CSV rather than a Hugging Face dataset script.
    """Implement imoanswerbench dataset loader for this module."""

    def _loader(
        dataset_path: str,
        *,
        split: str,
        cache_dir: str | None = None,
        streaming: bool = False,
    ) -> Any:
        """Implement loader for this module."""
        del dataset_path
        return load_dataset(
            "csv",
            data_files={split: _IMOANSWERBENCH_DATA_URL},
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

    return _loader


def _imoanswerbench_prompt(problem: str) -> str:
    """Implement imoanswerbench prompt for this module."""
    return f"Question: {problem}\nAnswer:"


@dataclass(slots=True)
class IMOAnswerBench(BaseTestSuite):
    """Implement the IMO-AnswerBench benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = IMOANSWERBENCH_DATASET_PATH
    dataset_name: str | None = None
    split: str = "test"
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _imoanswerbench_dataset_loader()

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "imoanswerbench"

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
                target=str(doc["Short Answer"]),
                request=GenerationRequest(
                    prompt=_imoanswerbench_prompt(str(doc["Problem"])),
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
                "prediction-normalized": normalize_math_string(extracted_answer),
                "target-normalized": normalize_math_string(target),
            },
            scores={"em": math_exact_match(output.text, target)},
            metadata={
                "problem_id": str(prepared_sample.doc["Problem ID"]),
                "category": str(prepared_sample.doc["Category"]),
                "subcategory": str(prepared_sample.doc["Subcategory"]),
                "source": str(prepared_sample.doc["Source"]),
            },
        )


def imoanswerbench(**kwargs: Any) -> IMOAnswerBench:
    """Implement imoanswerbench for this module."""
    return IMOAnswerBench(**kwargs)
