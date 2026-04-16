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
imoanswerbench_module = importlib.import_module("evalution.benchmarks.imoanswerbench")


class FakeGenerationSession:
    """Provide the fake generation session helper used by the surrounding tests."""

    def generate(self, requests, *, batch_size=None):
        """Implement generate for the fake IMOAnswerBench session."""
        assert batch_size == 1
        assert len(requests) == 1
        return [GenerationOutput(prompt=requests[0].prompt or "", text="Reasoning... \\boxed{3}")]


def test_imoanswerbench_scores_math_exact_match(monkeypatch) -> None:
    """Verify IMOAnswerBench scores the official short-answer field with math exact match."""
    dataset = Dataset.from_list(
        [
            {
                "Problem ID": "imo-bench-algebra-001",
                "Problem": "Find the value.",
                "Short Answer": "3",
                "Category": "Algebra",
                "Subcategory": "Operation",
                "Source": "IMO Shortlist 2021",
            }
        ]
    )
    monkeypatch.setattr(imoanswerbench_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.imoanswerbench(max_rows=1, batch_size=1).evaluate(
        FakeGenerationSession()
    )

    assert result.name == "imoanswerbench"
    assert result.metrics == {"em": 1.0}
    assert result.metadata == {
        "dataset_path": "google-deepmind/superhuman/imobench",
        "dataset_name": None,
        "split": "test",
        "order": "native",
        "stream": False,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_math_exact_match",
        "primary_metric": "em",
    }
    sample = result.samples[0]
    assert sample.prompt == "Question: Find the value.\nAnswer:"
    assert sample.metadata == {
        "problem_id": "imo-bench-algebra-001",
        "category": "Algebra",
        "subcategory": "Operation",
        "source": "IMO Shortlist 2021",
    }
