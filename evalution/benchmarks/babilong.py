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
BABILONG_CONTEXT_LENGTHS = (
    "0k",
    "1k",
    "2k",
    "4k",
    "8k",
    "16k",
    "32k",
    "64k",
    "128k",
    "256k",
    "512k",
    "1M",
    "10M",
)
BABILONG_TASK_SPLITS = tuple(f"qa{i}" for i in range(1, 21))
BABILONG_TASKS = tuple(f"babilong_{qa_split}" for qa_split in BABILONG_TASK_SPLITS)
_STOP_STRINGS = ("\n",)


def _babilong_prompt(context: str, question: str) -> str:
    """Implement babilong prompt for this module."""
    return (
        "Context:\n"
        f"{context.strip()}\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Answer:"
    )


def _normalize_babilong_answer(text: str) -> str:
    """Normalize babilong answer. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return " ".join(first_line.rstrip(".").lower().split())


@dataclass(slots=True)
class BABILong(BaseTestSuite):
    """Implement the babilong benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "RMT-team/babilong"
    dataset_name: str | None = "0k"
    split: str = "qa1"
    context_length: str = "0k"
    qa_split: str = "qa1"
    max_new_tokens: int = 16
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization. Preserve the fallback order expected by the surrounding caller."""
        if self.context_length not in BABILONG_CONTEXT_LENGTHS:
            raise ValueError(f"unsupported babilong context length: {self.context_length!r}")
        if self.qa_split not in BABILONG_TASK_SPLITS:
            raise ValueError(f"unsupported babilong split: {self.qa_split!r}")
        if self.dataset_name in {None, self.context_length}:
            self.dataset_name = self.context_length
        else:
            raise ValueError("babilong dataset_name must match the configured context length")
        if self.split != self.qa_split:
            raise ValueError("babilong split must match the configured qa_split")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"babilong_{self.qa_split}"

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
            "context_length": self.context_length,
            "qa_split": self.qa_split,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            target = _normalize_babilong_answer(str(doc["target"]))
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=_babilong_prompt(str(doc["input"]), str(doc["question"])),
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
        normalized_prediction = _normalize_babilong_answer(output.text)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-stripped": normalized_prediction,
                "target-stripped": prepared_sample.target,
            },
            scores={"em": exact_match(normalized_prediction, prepared_sample.target)},
            metadata={
                "context_length": self.context_length,
                "qa_split": self.qa_split,
                "question": str(prepared_sample.doc["question"]).strip(),
            },
        )


def babilong(*, qa_split: str, context_length: str = "0k", **kwargs: Any) -> BABILong:
    """Implement babilong for this module."""
    return BABILong(
        qa_split=qa_split,
        split=qa_split,
        context_length=context_length,
        dataset_name=context_length,
        **kwargs,
    )


def _make_babilong_factory(qa_split: str) -> Any:
    """Make babilong factory."""
    def factory(**kwargs: Any) -> BABILong:
        """Implement factory for this module."""
        return babilong(qa_split=qa_split, **kwargs)

    factory.__name__ = f"babilong_{qa_split}"
    return factory


for _qa_split in BABILONG_TASK_SPLITS:
    globals()[f"babilong_{_qa_split}"] = _make_babilong_factory(_qa_split)

del _qa_split
