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
afrimgsm_module = importlib.import_module("evalution.benchmarks.afrimgsm")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self) -> None:
        """Initialize this object."""
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        assert batch_size == 1
        self.requests.extend(requests)
        assert len(requests) == 1
        assert requests[0].prompt == (
            "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins "
            "for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per "
            "fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:"
        )
        return [
            GenerationOutput(
                prompt=requests[0].prompt,
                text="The answer is 18.",
            )
        ]

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def test_afrimgsm_scores_numeric_generation(monkeypatch) -> None:
    """Verify afrimgsm scores numeric generation. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": (
                    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins "
                    "for her friends every day with four. She sells the remainder at the farmers' market daily for "
                    "$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
                ),
                "answer": None,
                "answer_number": 18,
                "equation_solution": None,
            }
        ]
    )
    monkeypatch.setattr(afrimgsm_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.afrimgsm_eng(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "afrimgsm_eng"
    assert result.metrics == {"acc,num": 1.0}
    assert result.metadata["dataset_path"] == "masakhane/afrimgsm"
    assert result.metadata["dataset_name"] == "eng"
    assert result.metadata["split"] == "test"
    assert result.metadata["variant"] == "base"
    assert result.metadata["language"] == "eng"
    sample = result.samples[0]
    assert sample.target == "18"
    assert sample.extracted["numeric-extract"] == "18"
    assert sample.metadata["answer_number"] == "18"


def test_afrimgsm_rejects_unknown_language() -> None:
    """Verify afrimgsm rejects unknown language."""
    with pytest.raises(ValueError, match="unsupported afrimgsm language"):
        evalution.benchmarks.afrimgsm(language="xyz")


def test_afrimgsm_rejects_non_base_variants() -> None:
    """Verify afrimgsm rejects non base variants."""
    with pytest.raises(ValueError, match="only supports the direct base variant"):
        evalution.benchmarks.afrimgsm(language="eng", variant="cot")
