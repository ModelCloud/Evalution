# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

arc_challenge_module = importlib.import_module("evalution.suites.arc_challenge")


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
                    "An astronomer observes that a planet rotates faster after a meteorite "
                    "impact. Which is the most likely effect of this increase in rotation?"
                ),
                "choices": {
                    "text": [
                        "Planetary density will decrease.",
                        "Planetary years will become longer.",
                        "Planetary days will become shorter.",
                        "Planetary gravity will become stronger.",
                    ],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "C",
            }
        ]
    )


def test_arc_challenge_scores_original_style_exam_score(monkeypatch) -> None:
    monkeypatch.setattr(
        arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: _dataset(),
    )

    suite = evalution.arc_challenge(max_rows=1, batch_size=7)
    session = FakeSession(
        [
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=6),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=6),
        ]
    )
    result = suite.evaluate(session)

    assert result.name == "arc_challenge"
    assert result.metrics == {"accuracy,exam_score": 1.0}
    assert result.metadata == {
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Challenge",
        "split": "test",
        "streaming": False,
        "scoring_mode": "multiple_choice_exam_score",
        "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
    }

    sample = result.samples[0]
    assert sample.prompt == (
        "Question: An astronomer observes that a planet rotates faster after a meteorite "
        "impact. Which is the most likely effect of this increase in rotation?\nAnswer:"
    )
    assert sample.target == "Planetary days will become shorter."
    assert sample.prediction == "Planetary days will become shorter."
    assert sample.extracted == {
        "gold_index": "2",
        "selected_indices": "2",
        "selected_labels": "C",
    }
    assert sample.metadata["id"] == "Mercury_7175875"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_logprobs"] == [-1.3, -1.1, -0.2, -1.0]
    assert sample.metadata["selected_count"] == 1
    assert session.requests[0].context.endswith("\nAnswer:")
    assert session.requests[0].continuation == " Planetary density will decrease."
    assert session.requests[2].continuation == " Planetary days will become shorter."


def test_arc_challenge_awards_partial_credit_for_tied_top_choices(monkeypatch) -> None:
    monkeypatch.setattr(
        arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: _dataset(),
    )

    suite = evalution.arc_challenge(max_rows=1, batch_size=7)
    session = FakeSession(
        [
            LoglikelihoodOutput(logprob=-0.4, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.4, is_greedy=True, token_count=6),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=6),
        ]
    )
    result = suite.evaluate(session)

    assert result.metrics == {"accuracy,exam_score": 0.5}
    assert result.samples[0].prediction == (
        "Planetary density will decrease. | Planetary days will become shorter."
    )
    assert result.samples[0].extracted["selected_indices"] == "0,2"
    assert result.samples[0].extracted["selected_labels"] == "A,C"
    assert result.samples[0].metadata["selected_count"] == 2


def test_arc_challenge_passes_streaming_flag_to_load_dataset(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_dataset(*args, **kwargs):
        del args
        calls.append(kwargs)
        return _dataset()

    monkeypatch.setattr(arc_challenge_module, "load_dataset", fake_load_dataset)

    suite = evalution.arc_challenge(
        max_rows=1,
        batch_size=7,
        streaming=True,
    )
    session = FakeSession(
        [
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=6),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=6),
        ]
    )
    result = suite.evaluate(session)

    assert result.metadata["streaming"] is True
    assert calls
    assert calls[0]["streaming"] is True
