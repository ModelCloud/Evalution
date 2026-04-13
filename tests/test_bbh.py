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
bbh_module = importlib.import_module("evalution.benchmarks.bbh")


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


def test_bbh_scores_generated_exact_match(monkeypatch) -> None:
    """Verify BBH scores generated exact match. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "input": "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937",
                "target": "(B)",
            },
            {
                "input": "not ( True ) and ( True ) is",
                "target": "False",
            },
        ]
    )
    monkeypatch.setattr(bbh_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = FakeSession([" (B)", "The answer is False."])
    result = evalution.benchmarks.bbh(subset="date_understanding", max_rows=2, batch_size=3).evaluate(session)

    assert result.name == "bbh_date_understanding"
    assert result.metrics == {"em": 1.0}
    assert result.metadata["dataset_path"] == "lukaemon/bbh"
    assert result.metadata["dataset_name"] == "date_understanding"
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "generated_exact_match"
    assert result.metadata["primary_metric"] == "em"
    assert len(result.samples) == 2

    request = session.requests[0]
    assert request.prompt == (
        "Q: Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\n"
        "Options:\n"
        "(A) 12/11/1937\n"
        "(B) 12/25/1937\n"
        "A:"
    )
    assert request.stop == ["\n"]
    assert request.max_new_tokens == 64

    sample = result.samples[0]
    assert sample.prompt == request.prompt
    assert sample.target == "(B)"
    assert sample.prediction == " (B)"
    assert sample.extracted == {
        "prediction-stripped": "(B)",
        "target-stripped": "(B)",
    }
    assert sample.metadata == {
        "subset": "date_understanding",
        "input": "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937",
        "target_text": "(B)",
    }


def test_bbh_normalizer_handles_common_answer_formats() -> None:
    """Verify BBH normalizer handles common answer formats."""
    assert bbh_module._normalize_bbh_prediction("The answer is False.", target="False") == "False"
    assert bbh_module._normalize_bbh_prediction("Answer: (C)", target="(C)") == "(C)"
    assert bbh_module._normalize_bbh_prediction("A: sort, these, words", target="sort, these, words") == "sort, these, words"


def test_bbh_rejects_unknown_subset() -> None:
    """Verify BBH rejects unknown subset."""
    with pytest.raises(ValueError, match="unsupported bbh subset"):
        evalution.benchmarks.bbh(subset="unknown")
