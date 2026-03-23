# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

sciq_module = importlib.import_module("evalution.benchmarks.sciq")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "Darwin studied finches in the Galapagos Islands.\n"
            "Question: Who proposed the theory of evolution by natural selection?\n"
            "Answer:"
        )
        assert requests[0].continuation == " Linnaeus"
        assert requests[3].continuation == " darwin"
        return [
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.9, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
        ]


def test_sciq_scores_four_way_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "Who proposed the theory of evolution by natural selection?",
                "distractor1": "Linnaeus",
                "distractor2": "shaw",
                "distractor3": "Scopes",
                "correct_answer": "darwin",
                "support": "Darwin studied finches in the Galapagos Islands.",
            }
        ]
    )
    monkeypatch.setattr(sciq_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.sciq(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "sciq"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "allenai/sciq"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "validation"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == (
        "Darwin studied finches in the Galapagos Islands.\n"
        "Question: Who proposed the theory of evolution by natural selection?\n"
        "Answer:"
    )
    assert sample.target == "darwin"
    assert sample.prediction == "darwin"
    assert sample.extracted == {
        "gold_index": "3",
        "predicted_index": "3",
        "predicted_index_norm": "3",
    }
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_texts"] == ["Linnaeus", "shaw", "Scopes", "darwin"]


def test_sciq_can_emit_label_permutation_metric(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "Who proposed the theory of evolution by natural selection?",
                "distractor1": "Linnaeus",
                "distractor2": "shaw",
                "distractor3": "Scopes",
                "correct_answer": "darwin",
                "support": "Darwin studied finches in the Galapagos Islands.",
            }
        ]
    )
    monkeypatch.setattr(sciq_module, "load_dataset", lambda *args, **kwargs: dataset)

    class LabelPermutationSession:
        def __init__(self) -> None:
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            assert batch_size == 6
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 4
                return [
                    LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                    LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.9, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=1),
                ]

            assert len(requests) == 24
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. darwin" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.4,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.sciq(
        max_rows=1,
        batch_size=6,
        label_permutations=0.25,
    ).evaluate(LabelPermutationSession())

    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 0.0,
        "acc,label_perm:0.25": 1.0,
    }
    assert result.metadata["label_permutations"] == 0.25
    assert result.metadata["label_permutation_metric"] == "acc,label_perm:0.25"
    assert result.samples[0].extracted["predicted_index_label_perm:0.25"] == "3"
    assert result.samples[0].metadata["label_permutation_count"] == 6
