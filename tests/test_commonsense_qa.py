# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

commonsense_qa_module = importlib.import_module("evalution.benchmarks.commonsense_qa")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 9
        assert len(requests) == 5
        assert requests[0].context == (
            "Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?\n"
            "A. bank\n"
            "B. library\n"
            "C. department store\n"
            "D. mall\n"
            "E. new york\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D", " E"]
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.6, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=1),
        ]


def test_commonsense_qa_scores_five_way_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "csqa-1",
                "question": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
                "question_concept": "revolving door",
                "choices": {
                    "text": ["bank", "library", "department store", "mall", "new york"],
                    "label": ["A", "B", "C", "D", "E"],
                },
                "answerKey": "A",
            }
        ]
    )
    monkeypatch.setattr(commonsense_qa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.commonsense_qa(max_rows=1, batch_size=9).evaluate(FakeSession())

    assert result.name == "commonsense_qa"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "tau/commonsense_qa"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "validation"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == (
        "Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?\n"
        "A. bank\n"
        "B. library\n"
        "C. department store\n"
        "D. mall\n"
        "E. new york\n"
        "Answer:"
    )
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.metadata["id"] == "csqa-1"
    assert sample.metadata["question_concept"] == "revolving door"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D", "E"]
    assert sample.metadata["choice_texts"] == [
        "bank",
        "library",
        "department store",
        "mall",
        "new york",
    ]


def test_commonsense_qa_can_emit_label_permutation_metric(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "csqa-1",
                "question": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
                "question_concept": "revolving door",
                "choices": {
                    "text": ["bank", "library", "department store", "mall", "new york"],
                    "label": ["A", "B", "C", "D", "E"],
                },
                "answerKey": "A",
            }
        ]
    )
    monkeypatch.setattr(commonsense_qa_module, "load_dataset", lambda *args, **kwargs: dataset)

    class LabelPermutationSession:
        def __init__(self) -> None:
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            assert batch_size == 9
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 5
                return [
                    LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                    LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
                ]

            assert len(requests) == 25
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. bank" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.6,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.commonsense_qa(
        max_rows=1,
        batch_size=9,
        label_permutations=0.04,
    ).evaluate(LabelPermutationSession())

    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 0.0,
        "acc,label_perm:0.04": 1.0,
    }
    assert result.metadata["label_permutations"] == 0.04
    assert result.metadata["label_permutation_metric"] == "acc,label_perm:0.04"
    assert result.samples[0].extracted["predicted_index_label_perm:0.04"] == "0"
    assert result.samples[0].metadata["label_permutation_count"] == 5
