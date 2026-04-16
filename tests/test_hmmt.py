# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
hmmt_module = importlib.import_module("evalution.benchmarks.hmmt")


class FakeGenerationSession:
    """Provide the fake generation session helper used by the surrounding tests."""

    def __init__(self, responses: list[str]) -> None:
        """Initialize this object."""
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        """Implement generate for the fake HMMT session."""
        assert batch_size == 1
        self.requests.extend(requests)
        return [
            GenerationOutput(prompt=request.prompt or "", text=response)
            for request, response in zip(requests, self.responses, strict=True)
        ]


def test_hmmt_scores_math_exact_match(monkeypatch) -> None:
    """Verify HMMT scores math exact match across its olympiad-style answer format."""
    dataset = Dataset.from_list(
        [
            {
                "problem_idx": 7,
                "problem": "Compute the value.",
                "answer": "\\frac{1}{3}",
                "problem_type": ["Algebra"],
            }
        ]
    )
    monkeypatch.setattr(hmmt_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.hmmt_feb25(max_rows=1, batch_size=1).evaluate(
        FakeGenerationSession(["Solution... \\boxed{\\frac{1}{3}}"])
    )

    assert result.name == "hmmt_feb25"
    assert result.metrics == {"em": 1.0}
    assert result.metadata == {
        "dataset_path": "MathArena/hmmt_feb_2025",
        "dataset_name": None,
        "split": "train",
        "order": "native",
        "stream": False,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_math_exact_match",
        "primary_metric": "em",
    }
    sample = result.samples[0]
    assert sample.prompt == "Question: Compute the value.\nAnswer:"
    assert sample.metadata == {
        "problem_id": 7,
        "problem_type": ["Algebra"],
    }


def test_hmmt_variant_factories_expose_the_expected_datasets() -> None:
    """Verify the HMMT variant factories expose the expected public datasets."""
    assert evalution.benchmarks.hmmt_feb25().dataset_path == "MathArena/hmmt_feb_2025"
    assert evalution.benchmarks.hmmt_nov25().dataset_path == "MathArena/hmmt_nov_2025"
    assert evalution.benchmarks.hmmt_feb26().dataset_path == "MathArena/hmmt_feb_2026"
