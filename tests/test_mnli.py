# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

mnli_module = importlib.import_module("evalution.suites.mnli")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 5
        assert len(requests) == 9
        assert requests[0].context == (
            "The new rights are nice enough\n"
            "Question: Everyone really likes the newest benefits. True, False or Neither?\n"
            "Answer:"
        )
        assert requests[0].continuation == " True"
        assert requests[1].continuation == " Neither"
        assert requests[2].continuation == " False"
        return [
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.3, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=False, token_count=1),
        ]


def test_mnli_scores_three_way_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "premise": "The new rights are nice enough",
                "hypothesis": "Everyone really likes the newest benefits ",
                "label": 1,
                "idx": 0,
            },
            {
                "premise": "This church choir sings to the masses as they sing joyous songs from the book at a church.",
                "hypothesis": "The church is filled with song.",
                "label": 0,
                "idx": 1,
            },
            {
                "premise": "A person on a horse jumps over a broken down airplane.",
                "hypothesis": "A person is training his horse for a competition.",
                "label": 1,
                "idx": 2,
            },
        ]
    )
    monkeypatch.setattr(mnli_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.mnli(max_rows=3, batch_size=5).evaluate(FakeSession())

    assert result.name == "mnli"
    assert result.metrics == {
        "accuracy,loglikelihood": pytest.approx(2 / 3),
        "accuracy,loglikelihood_norm": pytest.approx(2 / 3),
    }
    assert result.metadata["dataset_path"] == "nyu-mll/glue"
    assert result.metadata["dataset_name"] == "mnli"
    assert result.metadata["split"] == "validation_matched"
    assert len(result.samples) == 3

    first_sample = result.samples[0]
    assert first_sample.target == "Neither"
    assert first_sample.prediction == "Neither"
    assert first_sample.metadata["idx"] == 0

    second_sample = result.samples[1]
    assert second_sample.target == "True"
    assert second_sample.prediction == "True"

    third_sample = result.samples[2]
    assert third_sample.target == "Neither"
    assert third_sample.prediction == "False"
    assert third_sample.extracted == {
        "gold_index": "1",
        "predicted_index": "2",
        "predicted_index_norm": "2",
    }


def test_mnli_prompt_helper_formats_three_way_entailment_prompt() -> None:
    assert (
        mnli_module._mnli_prompt("Premise text", "Hypothesis text")
        == "Premise text\nQuestion: Hypothesis text. True, False or Neither?\nAnswer:"
    )
