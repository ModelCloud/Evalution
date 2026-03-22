# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

sst2_module = importlib.import_module("evalution.suites.sst2")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 4
        assert requests[0].context == (
            "it 's a charming and often affecting journey .\n"
            "Question: Is this sentence positive or negative?\n"
            "Answer:"
        )
        assert requests[0].continuation == " negative"
        assert requests[1].continuation == " positive"
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
        ]


def test_sst2_scores_sentiment_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "sentence": "it 's a charming and often affecting journey .",
                "label": 1,
                "idx": 0,
            },
            {
                "sentence": "unflinchingly bleak and desperate",
                "label": 0,
                "idx": 1,
            },
        ]
    )
    monkeypatch.setattr(sst2_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.sst2(max_rows=2, batch_size=4).evaluate(FakeSession())

    assert result.name == "sst2"
    assert result.metrics == {
        "accuracy,loglikelihood": 1.0,
        "accuracy,loglikelihood_norm": 1.0,
    }
    assert result.metadata["dataset_path"] == "nyu-mll/glue"
    assert result.metadata["dataset_name"] == "sst2"
    assert len(result.samples) == 2

    first_sample = result.samples[0]
    assert first_sample.target == "positive"
    assert first_sample.prediction == "positive"
    assert first_sample.metadata["idx"] == 0

    second_sample = result.samples[1]
    assert second_sample.target == "negative"
    assert second_sample.prediction == "negative"
    assert second_sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }


def test_sst2_prompt_helper_formats_sentiment_question() -> None:
    assert (
        sst2_module._sst2_prompt("A great film.")
        == "A great film.\nQuestion: Is this sentence positive or negative?\nAnswer:"
    )
