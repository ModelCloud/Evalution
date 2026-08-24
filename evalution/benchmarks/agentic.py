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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pcre
from datasets import Dataset, load_dataset

from evalution.benchmarks.agentic_docker import DockerSandbox, extract_command
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

_TASK_TOML_DOCKER_IMAGE_RE = pcre.compile(r'^docker_image\s*=\s*"([^"]+)"', pcre.MULTILINE)
_TASK_TOML_NAME_RE = pcre.compile(r'^name\s*=\s*"([^"]+)"', pcre.MULTILINE)

AGENTIC_TASKS = (
    "agentbench",
    "deep_swe",
    "gaia",
    "gaia_level1",
    "gaia_level2",
    "gaia_level3",
    "osworld",
    "swe_atlas_qna",
    "swe_bench",
    "swe_bench_multilingual",
    "swe_bench_pro",
    "terminal_bench_21",
    "toolathlon_verified",
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


def _swe_atlas_target(doc: dict[str, Any]) -> str:
    """Return the SWE-Atlas-QnA reference answer as the target."""
    return str(doc.get("reference_answer", ""))


def _read_task_instruction(task_dir: Path) -> str:
    """Read the human-readable instruction for a local task directory."""
    instruction_path = task_dir / "instruction.md"
    if instruction_path.is_file():
        return instruction_path.read_text(encoding="utf-8", errors="ignore").strip()

    config_path = task_dir / "task_config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        parts = [
            str(config.get("id", "")),
            str(config.get("meta", {}).get("description", "")),
        ]
        requirements = config.get("meta", {}).get("requirements", [])
        if requirements:
            parts.append("\n".join(f"- {req}" for req in requirements))
        return "\n".join(part for part in parts if part).strip()

    for name in ("README.md", "readme.md"):
        readme_path = task_dir / name
        if readme_path.is_file():
            return readme_path.read_text(encoding="utf-8", errors="ignore").strip()

    return ""


def _read_task_solution(task_dir: Path) -> str:
    """Read a reference solution artifact (patch or script) from a task directory."""
    candidates = [
        task_dir / "solution" / "solution.patch",
        task_dir / "solution" / "solve.sh",
        task_dir / "solution" / "solve.py",
        task_dir / "solve.sh",
        task_dir / "solve.py",
        task_dir / "solution.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="ignore").strip()

    groundtruth_dir = task_dir / "groundtruth_workspace"
    if groundtruth_dir.is_dir():
        parts = []
        text_suffixes = {".json", ".txt", ".md", ".py", ".sh", ".yaml", ".yml", ".toml"}
        for file_path in sorted(groundtruth_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in text_suffixes:
                rel = file_path.relative_to(task_dir)
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    parts.append(f"--- {rel} ---\n{text}")
        return "\n\n".join(parts).strip()

    return ""


def _read_task_test_command(task_dir: Path) -> str:
    """Read the verifier/test script for a local task directory."""
    for name in (
        "tests/test.sh",
        "tests/run.sh",
        "test.sh",
        "run.sh",
    ):
        path = task_dir / name
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _read_task_docker_image(task_dir: Path) -> str:
    """Parse the ``environment.docker_image`` value from a ``task.toml``."""
    toml_path = task_dir / "task.toml"
    if toml_path.is_file():
        text = toml_path.read_text(encoding="utf-8", errors="ignore")
        match = _TASK_TOML_DOCKER_IMAGE_RE.search(text)
        if match:
            return match.group(1)
    return ""


def _read_task_id(task_dir: Path) -> str:
    """Return a stable task id from task metadata or the directory name."""
    toml_path = task_dir / "task.toml"
    if toml_path.is_file():
        text = toml_path.read_text(encoding="utf-8", errors="ignore")
        match = _TASK_TOML_NAME_RE.search(text)
        if match:
            return match.group(1)

    config_path = task_dir / "task_config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        task_id = config.get("id")
        if task_id:
            return str(task_id)

    return task_dir.name


def _load_local_tasks_dataset(path: str, *args: Any, **kwargs: Any) -> Dataset:
    """Load a Harbor-style local task directory into an in-memory dataset."""
    del args, kwargs
    root = Path(os.path.expanduser(path))
    if not root.exists():
        raise FileNotFoundError(
            f"Local task directory not found: {root}. Clone the benchmark repository "
            "and set `dataset_path` to the tasks directory."
        )

    if (root / "tasks").is_dir() and not (root / "task.toml").is_file():
        tasks_dir = root / "tasks"
    else:
        tasks_dir = root

    rows: list[dict[str, Any]] = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        instruction = _read_task_instruction(task_dir)
        if not instruction:
            continue

        solution = _read_task_solution(task_dir)
        test_command = _read_task_test_command(task_dir)
        docker_image = _read_task_docker_image(task_dir)
        instance_id = _read_task_id(task_dir)

        rows.append(
            {
                "instance_id": instance_id,
                "problem_statement": instruction,
                "patch": solution,
                "test_command": test_command,
                "docker_image": docker_image,
            }
        )

    if not rows:
        raise ValueError(f"No task directories found under {tasks_dir}")

    return Dataset.from_dict(
        {
            key: [row[key] for row in rows]
            for key in rows[0]
        }
    )


@dataclass(slots=True)
class SWEBenchMultilingual(SWEBench):
    """SWE-bench Multilingual text-generation scaffold."""

    dataset_path: str = "SWE-bench/SWE-bench_Multilingual"
    dataset_name: str | None = None
    split: str = "test"
    variant_name: str = "swe_bench_multilingual"


@dataclass(slots=True)
class SWEBenchPro(SWEBench):
    """SWE-bench Pro text-generation scaffold."""

    dataset_path: str = "ScaleAI/SWE-bench_Pro"
    dataset_name: str | None = None
    split: str = "test"
    variant_name: str = "swe_bench_pro"


@dataclass(slots=True)
class SWEAtlasQnA(BaseTestSuite):
    """SWE Atlas (Codebase QnA) text-generation scaffold."""

    dataset_path: str = "ScaleAI/SWE-Atlas-QnA"
    dataset_name: str | None = None
    split: str = "test"
    variant_name: str = "swe_atlas_qna"
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
            "scoring_mode": "reference_answer_exact_match",
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=_swe_atlas_target(doc),
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
        """Score one sample against its expected outputs."""
        target = prepared_sample.target
        prediction = output.text
        doc = prepared_sample.doc
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
                "task_id": str(doc.get("task_id", "")),
                "language": str(doc.get("language", "")),
                "category": str(doc.get("category", "")),
                "repository_url": str(doc.get("repository_url", "")),
            },
        )


@dataclass(slots=True)
class _LocalAgenticBenchmark(BaseTestSuite):
    """Base class for agentic benchmarks that ship as local Harbor task directories."""

    dataset_name: str | None = None
    split: str = "test"
    variant_name: str = "local_agentic"
    max_new_tokens: int = 1024
    batch_size: int = 1
    do_sample: bool = False
    temperature: float = 0.0
    use_docker: bool = False
    docker_image: str = "alpine:latest"
    docker_timeout: float = 60.0

    def dataset_loader(self) -> Any:
        """Return the local task directory loader bound to this suite."""
        return _load_local_tasks_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return self.variant_name

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        scoring_mode = (
            "docker_stdout_exact_match"
            if self.use_docker
            else "patch_exact_match"
        )
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": scoring_mode,
            "primary_metric": "em",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=str(doc.get("patch", "")),
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
        doc = prepared_sample.doc
        target = prepared_sample.target
        prediction = output.text

        if self.use_docker:
            command = extract_command(prediction)
            image = str(doc.get("docker_image", "")) or self.docker_image
            sandbox = DockerSandbox(
                image=image,
                timeout=self.docker_timeout,
                pull="never",
            )
            run_result = sandbox.run(command)
            score = (
                1.0
                if _normalize(run_result.stdout) == _normalize(target)
                else 0.0
            )
            return SampleResult(
                index=prepared_sample.index,
                prompt=output.prompt,
                target=target,
                prediction=prediction,
                extracted={
                    "command": command,
                    "stdout": run_result.stdout,
                    "prediction-normalized": _normalize(prediction),
                    "target-normalized": _normalize(target),
                },
                scores={"em": score},
                metadata={
                    "instance_id": str(doc.get("instance_id", "")),
                    "docker_image": image,
                    "docker_exit_code": run_result.exit_code,
                },
            )

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
                "instance_id": str(doc.get("instance_id", "")),
                "docker_image": str(doc.get("docker_image", "")),
            },
        )


@dataclass(slots=True)
class TerminalBench21(_LocalAgenticBenchmark):
    """Terminal-Bench 2.1 text-generation scaffold."""

    dataset_path: str = "~/.cache/evalution/terminal-bench-2-1/tasks"
    variant_name: str = "terminal_bench_21"
    docker_image: str = "alpine:latest"


@dataclass(slots=True)
class DeepSWE(_LocalAgenticBenchmark):
    """DeepSWE text-generation scaffold."""

    dataset_path: str = "~/.cache/evalution/deep-swe/tasks"
    variant_name: str = "deep_swe"
    docker_image: str = "alpine:latest"


@dataclass(slots=True)
class ToolathlonVerified(_LocalAgenticBenchmark):
    """Toolathlon-Verified text-generation scaffold."""

    dataset_path: str = "~/.cache/evalution/toolathlon/tasks/finalpool"
    variant_name: str = "toolathlon_verified"
    docker_image: str = "alpine:latest"


def swe_bench_multilingual(**kwargs: Any) -> SWEBenchMultilingual:
    """Factory for the SWE-bench Multilingual test split."""
    return SWEBenchMultilingual(**kwargs)


def swe_bench_pro(**kwargs: Any) -> SWEBenchPro:
    """Factory for the SWE-bench Pro test split."""
    return SWEBenchPro(**kwargs)


def swe_atlas_qna(**kwargs: Any) -> SWEAtlasQnA:
    """Factory for the SWE Atlas (Codebase QnA) test split."""
    return SWEAtlasQnA(**kwargs)


def deep_swe(**kwargs: Any) -> DeepSWE:
    """Factory for the DeepSWE task directory benchmark."""
    return DeepSWE(**kwargs)


def terminal_bench_21(**kwargs: Any) -> TerminalBench21:
    """Factory for the Terminal-Bench 2.1 task directory benchmark."""
    return TerminalBench21(**kwargs)


def toolathlon_verified(**kwargs: Any) -> ToolathlonVerified:
    """Factory for the Toolathlon-Verified task directory benchmark."""
    return ToolathlonVerified(**kwargs)
