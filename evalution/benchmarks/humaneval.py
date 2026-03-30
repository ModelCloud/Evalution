# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

import pcre
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult

# HumanEval ships benchmark-owned Python prompts and tests for local execution.
HUMANEVAL_DATASET_PATH = "openai/openai_humaneval"
HUMANEVAL_DATASET_NAME = "openai_humaneval"
HUMANEVAL_TEST_SPLIT = "test"
HUMANEVAL_STOP_STRINGS = ("\nclass ", "\ndef ", "\nif __name__")


def _extract_code(text: str) -> str:
    # Prefer fenced code and otherwise strip the most common chat preamble.
    fence_pattern = pcre.compile(r"```(?:python)?\n?(.*?)\n?```", pcre.DOTALL)
    match = fence_pattern.search("```" + text)
    if match:
        return match.group(1).strip()
    preamble_pattern = pcre.compile(r"^(?:Here is .*?:\s*)", pcre.MULTILINE)
    return preamble_pattern.sub("", text).strip()


def _build_python_script(*, prompt: str, completion: str, test_code: str, entry_point: str) -> str:
    # Join the candidate solution and benchmark test harness into one process.
    return f"{prompt}{completion}\n{test_code}\ncheck({entry_point})\n"


def _run_script(script: str, *, timeout: int = 10) -> bool:
    try:
        completed = subprocess.run(
            ["python3", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


@dataclass(slots=True)
class HumanEval(BaseTestSuite):
    # HumanEval reports pass@1 by executing the generated function completion.
    dataset_path: str = HUMANEVAL_DATASET_PATH
    dataset_name: str | None = HUMANEVAL_DATASET_NAME
    split: str = HUMANEVAL_TEST_SPLIT
    stream: bool = (False)
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0

    def task_name(self) -> str:
        return "humaneval"

    def dataset_loader(self) -> Any:
        return load_dataset

    def requires_full_doc_materialization(self) -> bool:
        return True

    def result_metadata(self, *, generation_submission_mode: str) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_code_execution",
            "primary_metric": "pass@1",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            prompt = (
                "Complete the following Python function. "
                "Return only the valid Python continuation.\n\n"
                f"{doc['prompt']}"
            )
            yield PreparedSample(
                index=index,
                doc=doc,
                target=doc["entry_point"],
                request=GenerationRequest(
                    prompt=prompt,
                    stop=list(HUMANEVAL_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def _score_prediction(self, prediction: str, doc: dict[str, Any]) -> bool:
        script = _build_python_script(
            prompt=doc["prompt"],
            completion=_extract_code(prediction),
            test_code=doc["test"],
            entry_point=doc["entry_point"],
        )
        return _run_script(script)

    def score_sample(self, prepared_sample: PreparedSample, output: GenerationOutput) -> SampleResult:
        passed = self._score_prediction(output.text, prepared_sample.doc)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "passed": "1" if passed else "0",
                "code": _extract_code(output.text),
            },
            scores={"pass@1": float(passed)},
            metadata={
                "task_id": str(prepared_sample.doc["task_id"]),
                "entry_point": str(prepared_sample.doc["entry_point"]),
            },
        )


def humaneval(**kwargs: Any) -> HumanEval:
    return HumanEval(**kwargs)
