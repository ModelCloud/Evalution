# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution

mbpp_module = importlib.import_module("evalution.benchmarks.mbpp")


def test_mbpp_scores_pass_at_1(monkeypatch):
    dataset = Dataset.from_list(
        [
            {
                "source_file": "demo",
                "task_id": 1,
                "prompt": "Write a function add that returns a+b",
                "code": "def add(a,b):\n    return a+b",
                "test_imports": [],
                "test_list": [
                    "assert add(2, 3) == 5",
                    "assert add(-1, 1) == 0",
                    "assert add(10, 5) == 15",
                ],
            }
        ]
    )

    monkeypatch.setattr(mbpp_module, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(mbpp_module, "_run_script", lambda script: True)

    class FakeSession:
        def generate(self, requests, *, batch_size):
            assert batch_size == 1
            assert len(requests) == 1
            return [
                evalution.engines.base.GenerationOutput(
                    prompt=requests[0].prompt,
                    text="def add(a,b):\n    return a+b\n[DONE]",
                )
            ]

    result = evalution.benchmarks.mbpp(max_rows=1, batch_size=1).evaluate(FakeSession())

    assert result.name == "mbpp"
    assert result.metrics == {"pass@1": 1.0}
    assert result.metadata["dataset_path"] == "mbpp"
    assert result.metadata["dataset_name"] == "sanitized"
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "generated_code_execution"
    sample = result.samples[0]
    assert sample.extracted["passed"] == "1"
    assert "add" in sample.extracted["code"]
