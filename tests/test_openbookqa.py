# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

openbookqa_module = importlib.import_module("evalution.suites.openbookqa")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 7
        assert len(requests) == 4
        assert requests[0].context == "Question: Which material conducts electricity best?\nAnswer:"
        assert requests[0].continuation == " wool"
        assert requests[2].continuation == " copper"
        return [
            LoglikelihoodOutput(logprob=-2.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.9, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
        ]


def test_openbookqa_scores_four_way_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "obqa-1",
                "question_stem": "Which material conducts electricity best?",
                "choices": {
                    "text": ["wool", "plastic", "copper", "wood"],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "C",
            }
        ]
    )
    monkeypatch.setattr(openbookqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.openbookqa(max_rows=1, batch_size=7).evaluate(FakeSession())

    assert result.name == "openbookqa"
    assert result.metrics == {
        "accuracy,loglikelihood": 1.0,
        "accuracy,loglikelihood_norm": 1.0,
    }
    assert result.metadata["dataset_path"] == "allenai/openbookqa"
    assert result.metadata["dataset_name"] == "main"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Question: Which material conducts electricity best?\nAnswer:"
    assert sample.target == "copper"
    assert sample.prediction == "copper"
    assert sample.extracted == {
        "gold_index": "2",
        "predicted_index": "2",
        "predicted_index_norm": "2",
    }
    assert sample.metadata["id"] == "obqa-1"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
