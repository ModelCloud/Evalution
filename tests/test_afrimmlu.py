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
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
afrimmlu_module = importlib.import_module("evalution.benchmarks.afrimmlu")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "Question: What is the value of p in 24 = 2p?\n"
            "A. p = 4\n"
            "B. p = 8\n"
            "C. p = 12\n"
            "D. p = 24\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
        ]


def test_afrimmlu_scores_four_way_multiple_choice(monkeypatch) -> None:
    """Verify afrimmlu scores four way multiple choice. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "subject": "elementary_mathematics",
                "question": "What is the value of p in 24 = 2p?",
                "choices": "['p = 4', 'p = 8', 'p = 12', 'p = 24']",
                "answer": "C",
            }
        ]
    )
    monkeypatch.setattr(afrimmlu_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.afrimmlu_eng(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "afrimmlu_eng"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "masakhane/afrimmlu"
    assert result.metadata["dataset_name"] == "eng"
    assert result.metadata["split"] == "test"
    assert result.metadata["language"] == "eng"
    sample = result.samples[0]
    assert sample.target == "C"
    assert sample.prediction == "C"
    assert sample.metadata["subject"] == "elementary_mathematics"
    assert sample.metadata["raw_choices"] == ["p = 4", "p = 8", "p = 12", "p = 24"]


def test_afrimmlu_rejects_unknown_language() -> None:
    """Verify afrimmlu rejects unknown language."""
    with pytest.raises(ValueError, match="unsupported afrimmlu language"):
        evalution.benchmarks.afrimmlu(language="xyz")
