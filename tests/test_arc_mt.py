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

arc_mt_module = importlib.import_module("evalution.benchmarks.arc_mt")


class FakeSession:
    def __init__(self, outputs: list[LoglikelihoodOutput]) -> None:
        self.outputs = outputs
        self.requests = []

    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 7
        assert len(requests) == 4
        self.requests.extend(requests)
        return self.outputs


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": "Mercury_7175875",
                "question": (
                    "En astronom observerer, at en planet roterer hurtigere efter et meteoritnedslag. "
                    "Hvad er den mest sandsynlige effekt af denne øgede rotation?"
                ),
                "choices": {
                    "text": [
                        "Planeternes tæthed vil falde.",
                        "Planetår vil blive længere.",
                        "Planetens dage vil blive kortere.",
                        "Planetens tyngdekraft vil blive stærkere.",
                    ],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "C",
            }
        ]
    )


def test_arc_mt_scores_translated_arc_exam_score(monkeypatch) -> None:
    monkeypatch.setattr(arc_mt_module, "load_dataset", lambda *args, **kwargs: _dataset())

    suite = evalution.benchmarks.arc_mt_da(max_rows=1, batch_size=7)
    session = FakeSession(
        [
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=4),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=4),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=4),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=4),
        ]
    )
    result = suite.evaluate(session)

    assert result.name == "arc_mt_da"
    assert result.metrics == {"acc,exam": 1.0}
    assert result.metadata == {
        "dataset_path": "LumiOpen/arc_challenge_mt",
        "dataset_name": "da",
        "split": "test",
        "stream": False,
        "order": "native",
        "scoring_mode": "multiple_choice_exam_score",
        "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
        "language": "da",
    }
    sample = result.samples[0]
    assert sample.prompt == (
        "Question: En astronom observerer, at en planet roterer hurtigere efter et meteoritnedslag. "
        "Hvad er den mest sandsynlige effekt af denne øgede rotation?\nAnswer:"
    )
    assert sample.target == "Planetens dage vil blive kortere."
    assert sample.prediction == "Planetens dage vil blive kortere."
    assert sample.metadata["language"] == "da"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert session.requests[2].continuation == " Planetens dage vil blive kortere."


def test_arc_mt_uses_icelandic_dataset_path() -> None:
    suite = evalution.benchmarks.arc_mt_is()
    assert suite.dataset_path == "mideind/icelandic-arc-challenge"
    assert suite.dataset_name is None
    assert suite.task_name() == "arc_mt_is"


def test_arc_mt_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported arc_mt language"):
        evalution.benchmarks.arc_mt(language="fr")


def test_arc_mt_rejects_dataset_path_mismatch() -> None:
    with pytest.raises(ValueError, match="dataset_path must match"):
        evalution.benchmarks.arc_mt(language="da", dataset_path="elsewhere/arc_mt")
