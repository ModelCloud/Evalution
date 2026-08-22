# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Agentic benchmark suites.

The five most popular agentic evaluation sets are scaffolded as text-generation
benchmarks: SWE-bench, WebArena, GAIA, OSWorld, and AgentBench.  They use the
standard ``BaseTestSuite`` pipeline and PyPcre for all text normalization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pcre
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult

# Keep benchmark defaults and public task ids explicit at module scope.
_STOP_STRINGS = (
    "Task:",
    "Question:",
    "</s>",
    "<|eot_id|>",
    "<|endoftext|>",
    "\n\n",
)

_WS_PATTERN = pcre.compile(r"\s+")

AGENTIC_TASKS = (
    "agentbench",
    "gaia",
    "gaia_level1",
    "gaia_level2",
    "gaia_level3",
    "osworld",
    "swe_bench",
    "webarena",
    "webarena_hard",
)


def _normalize(text: str) -> str:
    """Collapse whitespace and lower-case a string for comparison."""
    return _WS_PATTERN.sub(" ", text.strip()).lower()


def _exact_score(prediction: str, target: str) -> float:
    """Return 1.0 when the normalized prediction equals the normalized target."""
    return 1.0 if _normalize(prediction) == _normalize(target) else 0.0


def _task_prompt(instruction: str) -> str:
    """Wrap an agentic instruction in a simple text-generation prompt."""
    return f"Task: {instruction}\n\nAnswer:"


def _json_loads(value: Any) -> Any:
    """Parse a JSON string or pass through a value that is already decoded."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _webarena_target(doc: dict[str, Any]) -> str:
    """Extract the reference answer strings from a WebArena ``eval`` JSON column."""
    answers: list[str] = []
    for item in _json_loads(doc.get("eval", "[]")):
        expected = item.get("expected") if isinstance(item, dict) else None
        if not isinstance(expected, dict):
            continue
        retrieved = expected.get("retrieved_data")
        if isinstance(retrieved, list):
            answers.extend(str(entry) for entry in retrieved)
        elif isinstance(retrieved, str):
            answers.append(retrieved)
    return "; ".join(answers)


def _swe_bench_target(doc: dict[str, Any]) -> str:
    """Return the gold patch for a SWE-bench instance as the reference target."""
    return str(doc.get("patch", ""))


def _gaia_target(doc: dict[str, Any]) -> str:
    """Return the GAIA ``Final answer`` column as the reference target."""
    return str(doc.get("Final answer", ""))


def _agentbench_target(doc: dict[str, Any]) -> str:
    """Return the ground-truth answer for an AgentBench instance."""
    ground_truth = doc.get("ground_truth")
    if ground_truth is not None:
        return str(ground_truth)
    return ""


@dataclass(slots=True)
class SWEBench(BaseTestSuite):
    """SWE-bench text-generation scaffold."""

    dataset_path: str = "princeton-nlp/SWE-bench"
    dataset_name: str | None = None
    split: str = "test"
    variant_name: str = "swe_bench"
    max_new_tokens: int = 256
    batch_size: int = 1
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
            "scoring_mode": "patch_exact_match",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=_swe_bench_target(doc),
                request=GenerationRequest(
                    prompt=_task_prompt(str(doc.get("problem_statement", ""))),
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
        """Score one sample against its expected outputs."""
        target = prepared_sample.target
        prediction = output.text
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=target,
            prediction=prediction,
            extracted={
                "prediction-normalized": _normalize(prediction),
                "target-normalized": _normalize(target),
            },
            scores={"em": _exact_score(prediction, target)},
            metadata={
                "instance_id": str(prepared_sample.doc.get("instance_id", "")),
                "repo": str(prepared_sample.doc.get("repo", "")),
            },
        )


@dataclass(slots=True)
class WebArena(BaseTestSuite):
    """WebArena text-generation scaffold."""

    dataset_path: str = "AmineHA/WebArena-Verified"
    dataset_name: str | None = None
    split: str = "full"
    variant_name: str = "webarena"
    max_new_tokens: int = 256
    batch_size: int = 1
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
            "scoring_mode": "expected_answer_exact_match",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=_webarena_target(doc),
                request=GenerationRequest(
                    prompt=_task_prompt(str(doc.get("intent", ""))),
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
        """Score one sample against its expected outputs."""
        target = prepared_sample.target
        prediction = output.text
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=target,
            prediction=prediction,
            extracted={
                "prediction-normalized": _normalize(prediction),
                "target-normalized": _normalize(target),
            },
            scores={"em": _exact_score(prediction, target)},
            metadata={
                "task_id": str(prepared_sample.doc.get("task_id", "")),
                "sites": str(prepared_sample.doc.get("sites", "")),
            },
        )


@dataclass(slots=True)
class GAIA(BaseTestSuite):
    """GAIA text-generation scaffold."""

    dataset_path: str = "gaia-benchmark/GAIA"
    dataset_name: str = "2023_level1"
    split: str = "validation"
    variant_name: str = "gaia"
    max_new_tokens: int = 256
    batch_size: int = 1
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
            "scoring_mode": "final_answer_exact_match",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=_gaia_target(doc),
                request=GenerationRequest(
                    prompt=_task_prompt(str(doc.get("Question", ""))),
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
        """Score one sample against its expected outputs."""
        target = prepared_sample.target
        prediction = output.text
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=target,
            prediction=prediction,
            extracted={
                "prediction-normalized": _normalize(prediction),
                "target-normalized": _normalize(target),
            },
            scores={"em": _exact_score(prediction, target)},
            metadata={
                "task_id": str(prepared_sample.doc.get("task_id", "")),
                "level": str(prepared_sample.doc.get("Level", "")),
                "file_name": str(prepared_sample.doc.get("file_name", "")),
            },
        )


@dataclass(slots=True)
class OSWorld(BaseTestSuite):
    """OSWorld text-generation scaffold using the public text-only gold set."""

    dataset_path: str = "hud-evals/OSWorld-Gold"
    dataset_name: str | None = None
    split: str = "train"
    variant_name: str = "osworld"
    max_new_tokens: int = 256
    batch_size: int = 1
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
            "scoring_mode": "none",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target="",
                request=GenerationRequest(
                    prompt=_task_prompt(str(doc.get("prompt", ""))),
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
        """Score one sample; OSWorld has no text reference target."""
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target="",
            prediction=output.text,
            extracted={
                "prediction-normalized": _normalize(output.text),
                "target-normalized": "",
            },
            scores={"em": 0.0},
            metadata={
                "task_id": str(prepared_sample.doc.get("id", "")),
                "tags": str(prepared_sample.doc.get("metadata", "")),
            },
        )


@dataclass(slots=True)
class AgentBench(BaseTestSuite):
    """AgentBench text-generation scaffold (OSBench split)."""

    dataset_path: str = "iFurySt/AgentBench"
    dataset_name: str = "default"
    split: str = "osbench"
    variant_name: str = "agentbench"
    max_new_tokens: int = 256
    batch_size: int = 1
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
            "scoring_mode": "ground_truth_exact_match",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=_agentbench_target(doc),
                request=GenerationRequest(
                    prompt=_task_prompt(str(doc.get("description", ""))),
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
        """Score one sample against its expected outputs."""
        target = prepared_sample.target
        prediction = output.text
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=target,
            prediction=prediction,
            extracted={
                "prediction-normalized": _normalize(prediction),
                "target-normalized": _normalize(target),
            },
            scores={"em": _exact_score(prediction, target)},
            metadata={
                "instance_id": str(prepared_sample.doc.get("instance_id", "")),
                "comparison_method": str(prepared_sample.doc.get("comparison_method", "")),
            },
        )


def _swe_bench_variant(variant_name: str, **kwargs: Any) -> SWEBench:
    """Build a SWE-bench variant."""
    return SWEBench(variant_name=variant_name, **kwargs)


def swe_bench(**kwargs: Any) -> SWEBench:
    """Factory for the SWE-bench test split."""
    return _swe_bench_variant("swe_bench", **kwargs)


def _webarena_variant(variant_name: str, split: str, **kwargs: Any) -> WebArena:
    """Build a WebArena variant."""
    return WebArena(variant_name=variant_name, split=split, **kwargs)


def webarena(**kwargs: Any) -> WebArena:
    """Factory for the full WebArena split."""
    return _webarena_variant("webarena", "full", **kwargs)


def webarena_hard(**kwargs: Any) -> WebArena:
    """Factory for the hard WebArena split."""
    return _webarena_variant("webarena_hard", "hard", **kwargs)


def _gaia_variant(variant_name: str, dataset_name: str, **kwargs: Any) -> GAIA:
    """Build a GAIA variant."""
    return GAIA(variant_name=variant_name, dataset_name=dataset_name, **kwargs)


def gaia(**kwargs: Any) -> GAIA:
    """Factory for the GAIA 2023 all-levels validation split."""
    return _gaia_variant("gaia", "2023_all", **kwargs)


def gaia_level1(**kwargs: Any) -> GAIA:
    """Factory for GAIA 2023 level 1 validation."""
    return _gaia_variant("gaia_level1", "2023_level1", **kwargs)


def gaia_level2(**kwargs: Any) -> GAIA:
    """Factory for GAIA 2023 level 2 validation."""
    return _gaia_variant("gaia_level2", "2023_level2", **kwargs)


def gaia_level3(**kwargs: Any) -> GAIA:
    """Factory for GAIA 2023 level 3 validation."""
    return _gaia_variant("gaia_level3", "2023_level3", **kwargs)


def osworld(**kwargs: Any) -> OSWorld:
    """Factory for the OSWorld text-only gold set."""
    return OSWorld(**kwargs)


def agentbench(**kwargs: Any) -> AgentBench:
    """Factory for the AgentBench OSBench split."""
    return AgentBench(**kwargs)
