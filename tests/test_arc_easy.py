# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

arc_easy_module = importlib.import_module("evalution.benchmarks.arc_easy")


class FakeSession:
    def __init__(self, outputs: list[LoglikelihoodOutput]) -> None:
        self.outputs = outputs
        self.requests = []

    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 9
        assert len(requests) == 4
        self.requests.extend(requests)
        return self.outputs


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "id": "MCAS_2000_4_6",
                "question": "Which technology was developed most recently?",
                "choices": {
                    "text": ["cellular telephone", "television", "refrigerator", "airplane"],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "A",
            }
        ]
    )


def test_arc_easy_scores_original_style_exam_score(monkeypatch) -> None:
    monkeypatch.setattr(arc_easy_module, "load_dataset", lambda *args, **kwargs: _dataset())

    result = evalution.benchmarks.arc_easy(max_rows=1, batch_size=9).evaluate(
        FakeSession(
            [
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                LoglikelihoodOutput(logprob=-0.2, is_greedy=False, token_count=10),
                LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
                LoglikelihoodOutput(logprob=-0.9, is_greedy=False, token_count=1),
            ]
        )
    )

    assert result.name == "arc_easy"
    assert result.metrics == {"acc,exam": 1.0}
    assert result.metadata == {
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Easy",
        "split": "test",
        "order": "native",
        "stream": False,
        "scoring_mode": "multiple_choice_exam_score",
        "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
    }

    sample = result.samples[0]
    assert sample.prompt == "Question: Which technology was developed most recently?\nAnswer:"
    assert sample.target == "cellular telephone"
    assert sample.prediction == "cellular telephone"
    assert sample.extracted == {
        "gold_index": "0",
        "selected_indices": "0",
        "selected_labels": "A",
    }
    assert sample.metadata["id"] == "MCAS_2000_4_6"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_logprobs"] == [-0.1, -0.2, -1.3, -0.9]
    assert sample.metadata["selected_count"] == 1


def test_arc_easy_awards_partial_credit_for_tied_top_choices(monkeypatch) -> None:
    monkeypatch.setattr(arc_easy_module, "load_dataset", lambda *args, **kwargs: _dataset())

    result = evalution.benchmarks.arc_easy(max_rows=1, batch_size=9).evaluate(
        FakeSession(
            [
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                LoglikelihoodOutput(logprob=-0.1, is_greedy=False, token_count=10),
                LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
                LoglikelihoodOutput(logprob=-0.9, is_greedy=False, token_count=1),
            ]
        )
    )

    assert result.metrics == {"acc,exam": 0.5}
    assert result.samples[0].prediction == "cellular telephone | television"
    assert result.samples[0].extracted["selected_indices"] == "0,1"
    assert result.samples[0].extracted["selected_labels"] == "A,B"
    assert result.samples[0].metadata["selected_count"] == 2
