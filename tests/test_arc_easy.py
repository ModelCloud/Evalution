# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

arc_easy_module = importlib.import_module("evalution.suites.arc_easy")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 9
        assert len(requests) == 4
        assert requests[0].context == "Question: Which technology was developed most recently?\nAnswer:"
        assert requests[0].continuation == " cellular telephone"
        assert requests[3].continuation == " airplane"
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=False, token_count=10),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.9, is_greedy=False, token_count=1),
        ]


def test_arc_easy_scores_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
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
    monkeypatch.setattr(arc_easy_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.arc_easy(max_rows=1, batch_size=9).evaluate(FakeSession())

    assert result.name == "arc_easy"
    assert result.metrics == {
        "accuracy,loglikelihood": 1.0,
        "accuracy,loglikelihood_norm": 0.0,
    }
    assert result.metadata["dataset_path"] == "allenai/ai2_arc"
    assert result.metadata["dataset_name"] == "ARC-Easy"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Question: Which technology was developed most recently?\nAnswer:"
    assert sample.target == "cellular telephone"
    assert sample.prediction == "television"
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["id"] == "MCAS_2000_4_6"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
