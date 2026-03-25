# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

french_bench_arc_challenge_module = importlib.import_module(
    "evalution.benchmarks.french_bench_arc_challenge"
)


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 7
        assert len(requests) == 4
        assert requests[0].context == (
            "Question: Anna tient un glaçon. Pourquoi le glaçon fond-il dans sa main ?\nRéponse:"
        )
        assert requests[0].continuation == "La chaleur se déplace de sa main vers le glaçon."
        assert requests[3].continuation == "Le froid se déplace du glaçon vers sa main."
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=11),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=11),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=11),
            LoglikelihoodOutput(logprob=-1.9, is_greedy=False, token_count=11),
        ]


def test_french_bench_arc_challenge_scores_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "ARC-Challenge/test/Mercury_SC_415719",
                "question": "Anna tient un glaçon. Pourquoi le glaçon fond-il dans sa main ?",
                "choices": [
                    "La chaleur se déplace de sa main vers le glaçon.",
                    "Le froid se déplace de sa main vers le glaçon.",
                    "La chaleur se déplace du glaçon vers sa main.",
                    "Le froid se déplace du glaçon vers sa main.",
                ],
                "answerKey": "A",
            }
        ]
    )
    monkeypatch.setattr(
        french_bench_arc_challenge_module,
        "load_dataset",
        lambda *args, **kwargs: dataset,
    )

    result = evalution.benchmarks.french_bench_arc_challenge(
        max_rows=1,
        batch_size=7,
    ).evaluate(FakeSession())

    assert result.name == "french_bench_arc_challenge"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata == {
        "dataset_path": "manu/french_bench_arc_challenge",
        "dataset_name": None,
        "split": "test",
        "order": "native",
        "stream": True,
        "scoring_mode": "multiple_choice_loglikelihood",
    }

    sample = result.samples[0]
    assert sample.prompt == (
        "Question: Anna tient un glaçon. Pourquoi le glaçon fond-il dans sa main ?\nRéponse:"
    )
    assert sample.target == "La chaleur se déplace de sa main vers le glaçon."
    assert sample.prediction == "La chaleur se déplace de sa main vers le glaçon."
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.metadata["id"] == "ARC-Challenge/test/Mercury_SC_415719"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
