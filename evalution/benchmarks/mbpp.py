# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pcre
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult

# MBPP sanitized split uses self-contained unit tests per task.
MBPP_DATASET_PATH = "mbpp"
MBPP_DATASET_NAME = "sanitized"
MBPP_TEST_SPLIT = "test"
MBPP_STOP_STRINGS = ("[DONE]",)


def _extract_code(text: str) -> str:
    # Prefer fenced code blocks, fall back to everything before [DONE].
    """Extract code."""
    fence_pattern = pcre.compile(r"```(?:python)?\\n?(.*?)\\n?```", pcre.DOTALL)
    match = fence_pattern.search("```" + text)
    if match:
        return match.group(1)
    return text.split("[DONE]", 1)[0].strip()


def _build_python_script(
    *,
    code: str,
    test_imports: list[str],
    test_list: list[str],
) -> str:
    """Build python script."""
    lines: list[str] = []
    lines.extend(test_imports)
    lines.append(code)
    lines.extend(test_list)
    return "\n".join(lines) + "\n"


def _run_script(script: str) -> bool:
    """Run script."""
    try:
        completed = subprocess.run(
            ["python3", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


@dataclass(slots=True)
class MBPP(BaseTestSuite):
    # MBPP evaluates pass@1 by executing the generated solution against provided assertions.
    """Implement the MBPP benchmark suite."""
    dataset_path: str = MBPP_DATASET_PATH
    dataset_name: str | None = MBPP_DATASET_NAME
    split: str = MBPP_TEST_SPLIT
    stream: bool = (False)
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "mbpp"

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def requires_full_doc_materialization(self) -> bool:
        """Implement requires full doc materialization for MBPP."""
        return True

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_code_execution",
            "primary_metric": "pass@1",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            prompt = (
                "You are an expert Python programmer, and here is your task: "
                f"{doc['prompt']} Your code should pass these tests:\n\n"
                f"{doc['test_list'][0]}\n{doc['test_list'][1]}\n{doc['test_list'][2]}\n[BEGIN]\n"
            )
            yield PreparedSample(
                index=index,
                doc=doc,
                target="\n".join(doc["test_list"]),
                request=GenerationRequest(
                    prompt=prompt,
                    stop=list(MBPP_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def _score_prediction(self, code_text: str, doc: dict[str, Any]) -> bool:
        """Score prediction. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        code_body = _extract_code(code_text)
        script = _build_python_script(
            code=code_body,
            test_imports=list(doc.get("test_imports", [])),
            test_list=list(doc["test_list"]),
        )
        return _run_script(script)

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
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
                "source_file": str(prepared_sample.doc["source_file"]),
                "test_import_count": len(prepared_sample.doc.get("test_imports", [])),
            },
        )


def mbpp(**kwargs: Any) -> MBPP:
    """Implement MBPP for this module."""
    return MBPP(**kwargs)
