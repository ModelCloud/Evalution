# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import ast
import base64
import json
import pickle
import subprocess
import zlib
from dataclasses import dataclass
from typing import Any, Callable

import pcre
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult

# Keep benchmark defaults and public task ids explicit at module scope.
LIVECODEBENCH_DATASET_PATH = "livecodebench/code_generation_lite"
LIVECODEBENCH_TASKS = ("livecodebench_v6",)
_LIVECODEBENCH_URL_ROOT = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main"
)
_VERSION_FILES = {
    "release_v6": (
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ),
}
_CODE_FENCE_PATTERN = pcre.compile(r"(?is)```(?:python)?\s*(.*?)\s*```")
_LEADING_CHAT_PREFIX_PATTERN = pcre.compile(r"(?im)^(?:here is .*?:\s*)")
_CLASS_SOLUTION_PATTERN = pcre.compile(r"(?m)^\s*class\s+Solution\b")


def _livecodebench_dataset_loader(version_tag: str) -> Callable[..., Any]:
    # Hugging Face still serves LiveCodeBench v6 through a dataset script, so Evalution loads the raw JSONL shards directly.
    """Implement livecodebench dataset loader for this module."""

    def _loader(
        dataset_path: str,
        *,
        split: str,
        cache_dir: str | None = None,
        streaming: bool = False,
    ) -> Any:
        """Implement loader for this module."""
        del dataset_path
        urls = [f"{_LIVECODEBENCH_URL_ROOT}/{file_name}" for file_name in _VERSION_FILES[version_tag]]
        return load_dataset(
            "json",
            data_files={split: urls},
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

    return _loader


def _livecodebench_prompt(doc: dict[str, Any]) -> str:
    """Implement livecodebench prompt for this module."""
    lines = [
        f"Title: {str(doc['question_title']).strip()}",
        "",
        str(doc["question_content"]).strip(),
    ]
    starter_code = str(doc.get("starter_code") or "").rstrip()
    if starter_code:
        lines.extend(
            [
                "",
                "Starter code:",
                starter_code,
            ]
        )
        lines.append("")
        lines.append("Return only the full final Python solution that completes the starter code.")
    else:
        lines.append("")
        lines.append("Return only the full final Python solution.")
    return "\n".join(lines)


def _extract_code(text: str) -> str:
    """Extract Python code from a model response."""
    fence_match = _CODE_FENCE_PATTERN.search(text)
    if fence_match:
        return str(fence_match.group(1)).strip()
    return _LEADING_CHAT_PREFIX_PATTERN.sub("", text).strip()


def _decode_private_test_cases(encoded_private_cases: str) -> list[dict[str, str]]:
    """Decode the official pickled/zlib private-test payload for one LiveCodeBench task."""
    payload = pickle.loads(zlib.decompress(base64.b64decode(encoded_private_cases)))
    return list(json.loads(payload))


def _all_test_cases(doc: dict[str, Any]) -> list[dict[str, str]]:
    """Collect the public and private test cases for one LiveCodeBench task."""
    return list(json.loads(str(doc["public_test_cases"]))) + _decode_private_test_cases(
        str(doc["private_test_cases"])
    )


def _arg_count_from_starter_code(starter_code: str, func_name: str) -> int:
    """Infer the callable arity from the shipped starter code when a task uses functional tests."""
    try:
        tree = ast.parse(starter_code)
    except SyntaxError:
        return 1

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == func_name:
                    return max(0, len(child.args.args) - 1)
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return len(node.args.args)
    return 1


def _merged_candidate_code(doc: dict[str, Any], prediction: str) -> str:
    """Merge method-body style completions back into the provided starter code when needed."""
    code = _extract_code(prediction)
    starter_code = str(doc.get("starter_code") or "").rstrip()
    if not starter_code:
        return code

    metadata = json.loads(str(doc.get("metadata") or "{}"))
    func_name = str(metadata.get("func_name") or "").strip()
    if not func_name:
        return code

    func_pattern = pcre.compile(rf"(?m)^\s*def\s+{pcre.escape(func_name)}\b")
    if _CLASS_SOLUTION_PATTERN.search(code) or func_pattern.search(code):
        return code
    return f"{starter_code}\n{code}".rstrip()


def _build_stdin_script(*, code: str, cases: list[dict[str, str]]) -> str:
    """Build the Python harness used for stdin-style LiveCodeBench tasks."""
    return f"""from __future__ import annotations
import contextlib
import io
import sys
from typing import *

CANDIDATE_CODE = {code!r}
CASES = {json.dumps(cases)}

for case in CASES:
    namespace = {{"__name__": "__main__"}}
    stdin_buffer = io.StringIO(case["input"])
    stdout_buffer = io.StringIO()
    previous_stdin = sys.stdin
    try:
        sys.stdin = stdin_buffer
        with contextlib.redirect_stdout(stdout_buffer):
            exec(CANDIDATE_CODE, namespace)
    finally:
        sys.stdin = previous_stdin
    if stdout_buffer.getvalue().strip() != case["output"].strip():
        raise AssertionError("stdin test case mismatch")
"""


def _build_functional_script(
    *,
    code: str,
    func_name: str,
    arg_count: int,
    cases: list[dict[str, str]],
) -> str:
    """Build the Python harness used for functional LiveCodeBench tasks."""
    return f"""from __future__ import annotations
import ast
import json
from typing import *

CANDIDATE_CODE = {code!r}
FUNC_NAME = {func_name!r}
ARG_COUNT = {arg_count}
CASES = {json.dumps(cases)}


def _parse_case_value(text):
    stripped = text.strip()
    try:
        return ast.literal_eval(stripped)
    except Exception:
        try:
            return json.loads(stripped)
        except Exception:
            return stripped


def _normalize(value):
    if isinstance(value, dict):
        return {{str(key): _normalize(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}}
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


namespace = {{}}
exec(CANDIDATE_CODE, namespace)
if "Solution" in namespace:
    candidate = getattr(namespace["Solution"](), FUNC_NAME)
else:
    candidate = namespace[FUNC_NAME]

for case in CASES:
    raw_args = _parse_case_value(case["input"])
    if ARG_COUNT != 1 and isinstance(raw_args, (list, tuple)):
        args = tuple(raw_args)
    else:
        args = (raw_args,)
    actual = _normalize(candidate(*args))
    expected = _normalize(_parse_case_value(case["output"]))
    if actual != expected:
        raise AssertionError("functional test case mismatch")
"""


def _run_script(script: str, *, timeout: int = 10) -> bool:
    """Run one generated LiveCodeBench harness script."""
    try:
        completed = subprocess.run(
            ["python3", "-"],
            input=script,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


@dataclass(slots=True)
class LiveCodeBench(BaseTestSuite):
    """Implement the LiveCodeBench v6 benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = LIVECODEBENCH_DATASET_PATH
    dataset_name: str | None = None
    split: str = "test"
    # Stream JSONL shards by default so small regression slices do not force multi-GB downloads.
    stream: bool = True
    version_tag: str = "release_v6"
    max_new_tokens: int = 768
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _livecodebench_dataset_loader(self.version_tag)

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "livecodebench_v6"

    def requires_full_doc_materialization(self) -> bool:
        """Implement requires full doc materialization for LiveCodeBench."""
        return True

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "version_tag": self.version_tag,
            "scoring_mode": "generated_code_execution",
            "primary_metric": "pass@1",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=str(doc["question_id"]),
                request=GenerationRequest(
                    prompt=_livecodebench_prompt(doc),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def _score_prediction(self, prediction: str, doc: dict[str, Any]) -> bool:
        """Score prediction. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        cases = _all_test_cases(doc)
        candidate_code = _merged_candidate_code(doc, prediction)
        test_types = {str(case["testtype"]).strip() for case in cases}
        if test_types == {"stdin"}:
            return _run_script(_build_stdin_script(code=candidate_code, cases=cases))
        if test_types == {"functional"}:
            metadata = json.loads(str(doc.get("metadata") or "{}"))
            func_name = str(metadata["func_name"]).strip()
            arg_count = _arg_count_from_starter_code(str(doc.get("starter_code") or ""), func_name)
            return _run_script(
                _build_functional_script(
                    code=candidate_code,
                    func_name=func_name,
                    arg_count=arg_count,
                    cases=cases,
                )
            )
        raise ValueError(f"unsupported livecodebench test mix: {sorted(test_types)!r}")

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        passed = self._score_prediction(output.text, prepared_sample.doc)
        metadata = json.loads(str(prepared_sample.doc.get("metadata") or "{}"))
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "passed": "1" if passed else "0",
                "code": _merged_candidate_code(prepared_sample.doc, output.text),
            },
            scores={"pass@1": float(passed)},
            metadata={
                "question_id": str(prepared_sample.doc["question_id"]),
                "platform": str(prepared_sample.doc["platform"]),
                "contest_id": str(prepared_sample.doc["contest_id"]),
                "difficulty": str(prepared_sample.doc["difficulty"]),
                "test_mode": "functional" if str(prepared_sample.doc.get("starter_code") or "").strip() else "stdin",
                **(
                    {"func_name": str(metadata["func_name"]).strip()}
                    if "func_name" in metadata
                    else {}
                ),
            },
        )


def livecodebench_v6(**kwargs: Any) -> LiveCodeBench:
    """Implement livecodebench_v6 for this module."""
    return LiveCodeBench(**kwargs)
