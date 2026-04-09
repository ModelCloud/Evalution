# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

hellaswag_module = importlib.import_module("evalution.benchmarks.hellaswag")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == "Roof shingle removal: A man is sitting on a roof. He"
        assert requests[0].continuation == " is using wrap to wrap a pair of skis."
        assert requests[3].continuation == " starts pulling up roofing on a roof."
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=True, token_count=8),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=2),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=True, token_count=6),
            LoglikelihoodOutput(logprob=-0.6, is_greedy=True, token_count=10),
        ]


def test_hellaswag_scores_raw_and_normalized_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "ind": 24,
                "activity_label": "Roof shingle removal",
                "ctx_a": "A man is sitting on a roof.",
                "ctx_b": "he",
                "ctx": "A man is sitting on a roof. he",
                "endings": [
                    "is using wrap to wrap a pair of skis.",
                    "is ripping level tiles off.",
                    "is holding a rubik's cube.",
                    "starts pulling up roofing on a roof.",
                ],
                "source_id": "activitynet~v_-JhWjGDPHMY",
                "split": "val",
                "split_type": "indomain",
                "label": "3",
            }
        ]
    )
    monkeypatch.setattr(hellaswag_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.hellaswag(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "hellaswag"
    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "Rowan/hellaswag"
    assert result.metadata["split"] == "validation"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Roof shingle removal: A man is sitting on a roof. He"
    assert sample.target == "starts pulling up roofing on a roof."
    assert sample.prediction == "starts pulling up roofing on a roof."
    assert sample.extracted == {
        "gold_index": "3",
        "predicted_index": "1",
        "predicted_index_norm": "3",
    }
    assert sample.scores == {
        "acc,ll": 0.0,
        "acc,ll_avg": 1.0,
    }
    assert sample.metadata["activity_label"] == "Roof shingle removal"
    assert sample.metadata["split_type"] == "indomain"
    assert len(sample.metadata["choice_logprobs"]) == 4
    assert len(sample.metadata["choice_logprobs_norm"]) == 4


def test_hellaswag_text_cleaner_strips_artifacts() -> None:
    cleaned = hellaswag_module._clean_hellaswag_text("  Intro [title] [noise] ending  ")

    assert cleaned == "Intro. ending"
