# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

swag_module = importlib.import_module("evalution.benchmarks.swag")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == "Students lower their eyes nervously. She"
        assert requests[0].continuation == " pats her shoulder, then saunters toward someone."
        assert requests[2].continuation == " walks slowly towards someone."
        return [
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=8),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=4),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=4),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=7),
        ]


def test_swag_scores_four_way_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "video-id": "vid-1",
                "fold-ind": "18313",
                "startphrase": "Students lower their eyes nervously. She",
                "sent1": "Students lower their eyes nervously.",
                "sent2": "She",
                "gold-source": "gold",
                "ending0": "pats her shoulder, then saunters toward someone.",
                "ending1": "turns with two students.",
                "ending2": "walks slowly towards someone.",
                "ending3": "wheels around as her dog thunders out.",
                "label": 2,
            }
        ]
    )
    monkeypatch.setattr(swag_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.swag(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "swag"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "swag"
    assert result.metadata["dataset_name"] == "regular"
    assert result.metadata["split"] == "validation"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Students lower their eyes nervously. She"
    assert sample.target == "walks slowly towards someone."
    assert sample.prediction == "walks slowly towards someone."
    assert sample.extracted == {
        "gold_index": "2",
        "predicted_index": "2",
        "predicted_index_norm": "2",
    }
    assert sample.metadata["video_id"] == "vid-1"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_texts"] == [
        "pats her shoulder, then saunters toward someone.",
        "turns with two students.",
        "walks slowly towards someone.",
        "wheels around as her dog thunders out.",
    ]


def test_swag_can_emit_label_permutation_metric(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "video-id": "vid-1",
                "fold-ind": "18313",
                "startphrase": "Students lower their eyes nervously. She",
                "sent1": "Students lower their eyes nervously.",
                "sent2": "She",
                "gold-source": "gold",
                "ending0": "pats her shoulder, then saunters toward someone.",
                "ending1": "turns with two students.",
                "ending2": "walks slowly towards someone.",
                "ending3": "wheels around as her dog thunders out.",
                "label": 2,
            }
        ]
    )
    monkeypatch.setattr(swag_module, "load_dataset", lambda *args, **kwargs: dataset)

    class LabelPermutationSession:
        def __init__(self) -> None:
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            assert batch_size == 6
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 4
                return [
                    LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=8),
                    LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=4),
                    LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=4),
                    LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=7),
                ]

            assert len(requests) == 24
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. walks slowly towards someone." in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.6,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.swag(
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
    assert result.samples[0].extracted["predicted_index_label_perm:0.25"] == "2"
    assert result.samples[0].metadata["label_permutation_count"] == 6
