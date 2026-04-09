# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution

humaneval_module = importlib.import_module("evalution.benchmarks.humaneval")


def test_humaneval_scores_pass_at_1(monkeypatch):
    dataset = Dataset.from_list(
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "canonical_solution": "    return a + b\n",
                "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
                "entry_point": "add",
            }
        ]
    )

    monkeypatch.setattr(humaneval_module, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(humaneval_module, "_run_script", lambda script, timeout=10: True)

    class FakeSession:
        def generate(self, requests, *, batch_size):
            assert batch_size == 1
            assert len(requests) == 1
            return [
                evalution.engines.base.GenerationOutput(
                    prompt=requests[0].prompt,
                    text="    return a + b\n",
                )
            ]

    result = evalution.benchmarks.humaneval(max_rows=1, batch_size=1).evaluate(FakeSession())

    assert result.name == "humaneval"
    assert result.metrics == {"pass@1": 1.0}
    assert result.metadata["dataset_path"] == "openai/openai_humaneval"
    assert result.metadata["dataset_name"] == "openai_humaneval"
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "generated_code_execution"
    sample = result.samples[0]
    assert sample.extracted["passed"] == "1"
    assert "return a + b" in sample.extracted["code"]
