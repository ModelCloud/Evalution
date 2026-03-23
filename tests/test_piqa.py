# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

piqa_module = importlib.import_module("evalution.suites.piqa")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 2
        assert requests[0].context == "Question: Chill a drink quickly\nAnswer:"
        assert requests[0].continuation == " Put the bottle in the freezer for a short time."
        assert requests[1].continuation == " Leave the bottle near a warm oven."
        return [
            LoglikelihoodOutput(logprob=-0.4, is_greedy=True, token_count=11),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=8),
        ]


def test_piqa_scores_binary_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "goal": "Chill a drink quickly",
                "sol1": "Put the bottle in the freezer for a short time.",
                "sol2": "Leave the bottle near a warm oven.",
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(piqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.piqa(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "piqa"
    assert result.metrics == {
        "accuracy,loglikelihood": 1.0,
        "accuracy,loglikelihood_norm": 1.0,
    }
    assert result.metadata["dataset_path"] == "baber/piqa"
    assert result.metadata["split"] == "validation"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Question: Chill a drink quickly\nAnswer:"
    assert sample.target == "Put the bottle in the freezer for a short time."
    assert sample.prediction == "Put the bottle in the freezer for a short time."
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.scores == {
        "accuracy,loglikelihood": 1.0,
        "accuracy,loglikelihood_norm": 1.0,
    }
    assert sample.metadata["choice_logprobs"] == [-0.4, -1.4]
    assert len(sample.metadata["choice_logprobs_norm"]) == 2


def test_piqa_can_emit_label_permutation_metric(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "goal": "Chill a drink quickly",
                "sol1": "Put the bottle in the freezer for a short time.",
                "sol2": "Leave the bottle near a warm oven.",
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(piqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    class LabelPermutationSession:
        def __init__(self) -> None:
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            assert batch_size == 4
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 2
                return [
                    LoglikelihoodOutput(logprob=-0.8, is_greedy=False, token_count=6),
                    LoglikelihoodOutput(logprob=-0.4, is_greedy=True, token_count=6),
                ]

            assert len(requests) == 4
            assert requests[0].context == (
                "Question: Chill a drink quickly\n"
                "Options:\n"
                "A. Put the bottle in the freezer for a short time.\n"
                "B. Leave the bottle near a warm oven.\n"
                "Answer:"
            )
            gold_text = "Put the bottle in the freezer for a short time."
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. {gold_text}" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.2,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.piqa(
        max_rows=1,
        batch_size=4,
        label_permutations=0.5,
    ).evaluate(LabelPermutationSession())

    assert result.metrics == {
        "accuracy,loglikelihood": 0.0,
        "accuracy,loglikelihood_norm": 0.0,
        "accuracy,label_perm_0.5": 1.0,
    }
    assert result.metadata["label_permutations"] == 0.5
    assert result.metadata["label_permutation_metric"] == "accuracy,label_perm_0.5"
    assert result.samples[0].extracted["predicted_index_label_perm_0.5"] == "0"
    assert result.samples[0].metadata["label_permutation_count"] == 2
