# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

hendrycks_math_module = importlib.import_module("evalution.benchmarks.hendrycks_math")


class FakeGenerationSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        assert batch_size in {1, 3}
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]


def test_hendrycks_math_scores_boxed_answer_exact_match(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "problem": "How many factors does 6 have?",
                "level": "Level 1",
                "type": "Number Theory",
                "solution": "The answer is \\boxed{4}.",
            }
        ]
    )
    monkeypatch.setattr(hendrycks_math_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = FakeGenerationSession(["Reasoning... \\boxed{4}"])
    result = evalution.benchmarks.hendrycks_math_algebra(max_rows=1, batch_size=3).evaluate(session)

    assert result.name == "hendrycks_math_algebra"
    assert result.metrics == {"em": 1.0}
    assert result.metadata == {
        "dataset_path": "EleutherAI/hendrycks_math",
        "dataset_name": "algebra",
        "split": "test",
        "streaming": False,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_math_exact_match",
        "primary_metric": "em",
    }
    assert session.requests[0].prompt == "Problem: How many factors does 6 have?\nAnswer:"
    assert session.requests[0].stop == ["Problem:", "</s>", "<|im_end|>", "<|eot_id|>"]

    sample = result.samples[0]
    assert sample.target == "4"
    assert sample.metadata == {
        "subset": "algebra",
        "level": "Level 1",
        "problem_type": "Number Theory",
    }


def test_hendrycks_math_factory_uses_test_split() -> None:
    suite = evalution.benchmarks.hendrycks_math(subset="geometry")

    assert suite.dataset_path == "EleutherAI/hendrycks_math"
    assert suite.dataset_name == "geometry"
    assert suite.split == "test"
    assert suite.task_name() == "hendrycks_math_geometry"


def test_hendrycks_math_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported hendrycks_math subset"):
        evalution.benchmarks.hendrycks_math(subset="unknown")
