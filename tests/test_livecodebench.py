# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import base64
import importlib
import json
import pickle
import zlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
livecodebench_module = importlib.import_module("evalution.benchmarks.livecodebench")


def _encode_private_cases(cases: list[dict[str, str]]) -> str:
    """Encode private cases with the same format used by the official dataset."""
    payload = pickle.dumps(json.dumps(cases))
    return base64.b64encode(zlib.compress(payload)).decode("ascii")


class FakeGenerationSession:
    """Provide the fake generation session helper used by the surrounding tests."""

    def __init__(self, responses: list[str]) -> None:
        """Initialize this object."""
        self.responses = responses

    def generate(self, requests, *, batch_size=None):
        """Implement generate for the fake LiveCodeBench session."""
        assert batch_size == len(requests)
        return [
            GenerationOutput(prompt=request.prompt or "", text=response)
            for request, response in zip(requests, self.responses, strict=True)
        ]


def test_livecodebench_scores_stdin_tasks(monkeypatch) -> None:
    """Verify LiveCodeBench executes stdin-style tasks against public and private cases."""
    dataset = Dataset.from_list(
        [
            {
                "question_title": "Add One",
                "question_content": "Read an integer and print the next integer.",
                "platform": "atcoder",
                "question_id": "stdin-1",
                "contest_id": "abc000",
                "contest_date": "2025-01-01T00:00:00",
                "starter_code": "",
                "difficulty": "easy",
                "public_test_cases": json.dumps(
                    [{"input": "1\n", "output": "2\n", "testtype": "stdin"}]
                ),
                "private_test_cases": _encode_private_cases(
                    [{"input": "41\n", "output": "42\n", "testtype": "stdin"}]
                ),
                "metadata": "{}",
            }
        ]
    )
    monkeypatch.setattr(livecodebench_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.livecodebench_v6(max_rows=1, batch_size=1).evaluate(
        FakeGenerationSession(["x = int(input())\nprint(x + 1)"])
    )

    assert result.name == "livecodebench_v6"
    assert result.metrics == {"pass@1": 1.0}
    assert result.samples[0].metadata["test_mode"] == "stdin"
    assert result.samples[0].metadata["platform"] == "atcoder"


def test_livecodebench_scores_functional_tasks(monkeypatch) -> None:
    """Verify LiveCodeBench executes functional tasks with the shipped starter-code signature."""
    dataset = Dataset.from_list(
        [
            {
                "question_title": "Add One",
                "question_content": "Implement addOne.",
                "platform": "leetcode",
                "question_id": "functional-1",
                "contest_id": "weekly-1",
                "contest_date": "2025-01-02T00:00:00",
                "starter_code": "class Solution:\n    def addOne(self, value: int) -> int:\n        ",
                "difficulty": "easy",
                "public_test_cases": json.dumps(
                    [{"input": "1", "output": "2", "testtype": "functional"}]
                ),
                "private_test_cases": _encode_private_cases(
                    [{"input": "41", "output": "42", "testtype": "functional"}]
                ),
                "metadata": json.dumps({"func_name": "addOne"}),
            }
        ]
    )
    monkeypatch.setattr(livecodebench_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.livecodebench_v6(max_rows=1, batch_size=1).evaluate(
        FakeGenerationSession(
            [
                "class Solution:\n"
                "    def addOne(self, value: int) -> int:\n"
                "        return value + 1\n"
            ]
        )
    )

    assert result.name == "livecodebench_v6"
    assert result.metrics == {"pass@1": 1.0}
    assert result.samples[0].metadata["test_mode"] == "functional"
    assert result.samples[0].metadata["func_name"] == "addOne"
    assert result.samples[0].metadata["platform"] == "leetcode"
