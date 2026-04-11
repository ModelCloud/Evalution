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
mgsm_module = importlib.import_module("evalution.benchmarks.mgsm")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self) -> None:
        """Initialize this object."""
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        assert batch_size in {1, 8}
        self.requests.extend(requests)
        assert len(requests) == 1
        assert requests[0].prompt == (
            "Question: If Maya has 7 marbles and buys 5 more, how many marbles does she have?\n"
            "Answer:"
        )
        return [
            GenerationOutput(
                prompt=requests[0].prompt,
                text="The answer is 12.",
            )
        ]

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def test_mgsm_scores_direct_numeric_generation(monkeypatch) -> None:
    """Verify mgsm scores direct numeric generation. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "If Maya has 7 marbles and buys 5 more, how many marbles does she have?",
                "answer_number": 12,
            }
        ]
    )
    monkeypatch.setattr(mgsm_module, "_load_mgsm_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.mgsm_direct_en(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "mgsm_direct_en"
    assert result.metrics == {"acc,num": 1.0}
    assert result.metadata["dataset_path"] == "juletxara/mgsm"
    assert result.metadata["dataset_name"] == "en"
    assert result.metadata["split"] == "test"
    assert result.metadata["variant"] == "base"
    assert result.metadata["language"] == "en"
    sample = result.samples[0]
    assert sample.target == "12"
    assert sample.extracted["numeric-extract"] == "12"
    assert sample.metadata["language"] == "en"
    assert sample.metadata["answer_number"] == "12"


def test_mgsm_rejects_unknown_language() -> None:
    """Verify mgsm rejects unknown language."""
    with pytest.raises(ValueError, match="unsupported MGSM language"):
        evalution.benchmarks.mgsm(language="xx")


def test_mgsm_rejects_non_base_variants() -> None:
    """Verify mgsm rejects non base variants."""
    with pytest.raises(ValueError, match="only supports the direct base variant"):
        evalution.benchmarks.mgsm(language="en", variant="cot")
