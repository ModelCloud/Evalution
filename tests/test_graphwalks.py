# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from datasets import Dataset
import pytest

import evalution
import evalution.benchmarks.graphwalks as graphwalks_module
from evalution.engines.base import GenerationOutput


class FakeSession:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, requests, *, batch_size=None):
        return [GenerationOutput(prompt=request.prompt, text=self.text, metadata={}) for request in requests]


@pytest.mark.parametrize(
    "generation_text, expected_f1, expected_flexible",
    [
        ("Reasoning\nFinal Answer: [a, b]\n", 1.0, 1.0),
        (
            "Final Answer: []\nthoughts\nFinal Answer: [a, b]\nclosing note",
            0.0,
            1.0,
        ),
    ],
)
def test_graphwalks_parses_node_sets(monkeypatch, generation_text, expected_f1, expected_flexible) -> None:
    dataset = Dataset.from_list(
        [
            {
                "prompt": "Find parents of node X.",
                "answer_nodes": ["a", "b"],
                "problem_type": "parents",
                "prompt_chars": 1024,
            }
        ]
    )
    monkeypatch.setattr(
        graphwalks_module,
        "load_dataset",
        lambda *args, **kwargs: dataset,
    )

    session = FakeSession(generation_text)
    result = evalution.benchmarks.graphwalks_128k(max_rows=1, batch_size=1).evaluate(session)

    assert result.metrics["f1"] == pytest.approx(expected_f1)
    assert result.metrics["flexible_f1"] == pytest.approx(expected_flexible)
    assert result.metadata["data_file"] == "graphwalks_128k_and_shorter.parquet"
    sample = result.samples[0]
    assert sample.metadata["problem_type"] == "parents"
    assert sample.metadata["prompt_chars"] == 1024
