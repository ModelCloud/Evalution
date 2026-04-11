# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
babilong_module = importlib.import_module("evalution.benchmarks.babilong")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, responses: list[str]) -> None:
        """Initialize this object."""
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        assert batch_size == 2
        assert len(requests) == 2
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def test_babilong_scores_normalized_generated_exact_match(monkeypatch) -> None:
    """Verify babilong scores normalized generated exact match. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "input": "John travelled to the hallway. Mary journeyed to the bathroom.",
                "question": "Where is Mary? ",
                "target": "bathroom",
            },
            {
                "input": "John went to the hallway. Daniel moved to the office.",
                "question": "Is Daniel in the office? ",
                "target": "yes",
            },
        ]
    )
    monkeypatch.setattr(babilong_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = FakeSession(["Bathroom.", " yes "])
    result = evalution.benchmarks.babilong(qa_split="qa1", max_rows=2, batch_size=2).evaluate(session)

    assert result.name == "babilong_qa1"
    assert result.metrics == {"em": 1.0}
    assert result.metadata["dataset_path"] == "RMT-team/babilong"
    assert result.metadata["dataset_name"] == "0k"
    assert result.metadata["split"] == "qa1"
    assert result.metadata["context_length"] == "0k"
    assert result.metadata["qa_split"] == "qa1"
    assert result.metadata["scoring_mode"] == "generated_exact_match"
    assert result.metadata["primary_metric"] == "em"

    request = session.requests[0]
    assert request.prompt == (
        "Context:\n"
        "John travelled to the hallway. Mary journeyed to the bathroom.\n\n"
        "Question:\n"
        "Where is Mary?\n\n"
        "Answer:"
    )
    assert request.stop == ["\n"]
    assert request.max_new_tokens == 16

    sample = result.samples[0]
    assert sample.target == "bathroom"
    assert sample.prediction == "Bathroom."
    assert sample.extracted == {
        "prediction-stripped": "bathroom",
        "target-stripped": "bathroom",
    }
    assert sample.metadata == {
        "context_length": "0k",
        "qa_split": "qa1",
        "question": "Where is Mary?",
    }


def test_babilong_normalizer_trims_case_and_final_period() -> None:
    """Verify babilong normalizer trims case and final period."""
    assert babilong_module._normalize_babilong_answer("Bathroom.") == "bathroom"
    assert babilong_module._normalize_babilong_answer("  s,w  ") == "s,w"
    assert babilong_module._normalize_babilong_answer("Line one\nline two") == "line one"


def test_babilong_rejects_unknown_context_length() -> None:
    """Verify babilong rejects unknown context length."""
    with pytest.raises(ValueError, match="unsupported babilong context length"):
        evalution.benchmarks.babilong(qa_split="qa1", context_length="3k")


def test_babilong_rejects_unknown_split() -> None:
    """Verify babilong rejects unknown split."""
    with pytest.raises(ValueError, match="unsupported babilong split"):
        evalution.benchmarks.babilong(qa_split="qa21")
