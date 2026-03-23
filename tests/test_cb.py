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

cb_module = importlib.import_module("evalution.benchmarks.cb")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 7
        assert len(requests) == 9
        assert requests[0].context == (
            "The child saw the dog by the gate.\n"
            "Question: The child saw an animal. True, False, or Neither?\n"
            "Answer:"
        )
        assert requests[0].continuation == " True"
        assert requests[1].continuation == " False"
        assert requests[2].continuation == " Neither"
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=4),
            LoglikelihoodOutput(logprob=-4.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.3, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-0.4, is_greedy=False, token_count=1),
        ]


def test_cb_scores_accuracy_and_macro_f1(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "premise": "The child saw the dog by the gate.",
                "hypothesis": "The child saw an animal",
                "idx": 0,
                "label": 0,
            },
            {
                "premise": "The scientist closed the empty notebook.",
                "hypothesis": "The notebook contained fresh results",
                "idx": 1,
                "label": 1,
            },
            {
                "premise": "A musician packed a violin before the train arrived.",
                "hypothesis": "The musician boarded the train",
                "idx": 2,
                "label": 2,
            },
        ]
    )
    monkeypatch.setattr(cb_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.cb(max_rows=3, batch_size=7).evaluate(FakeSession())

    assert result.name == "cb"
    assert result.metrics["acc,ll"] == pytest.approx(2 / 3)
    assert result.metrics["acc,ll_avg"] == pytest.approx(2 / 3)
    assert result.metrics["f1,ll_macro"] == pytest.approx(5 / 9)
    assert result.metrics["f1,ll_avg_macro"] == pytest.approx(5 / 9)
    assert result.metadata["dataset_path"] == "super_glue"
    assert result.metadata["dataset_name"] == "cb"
    assert len(result.samples) == 3

    first_sample = result.samples[0]
    assert first_sample.target == "True"
    assert first_sample.prediction == "True"
    assert first_sample.metadata["idx"] == 0

    second_sample = result.samples[1]
    assert second_sample.target == "False"
    assert second_sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }

    third_sample = result.samples[2]
    assert third_sample.target == "Neither"
    assert third_sample.extracted == {
        "gold_index": "2",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }


def test_cb_prompt_helper_formats_three_way_entailment_question() -> None:
    assert (
        cb_module._cb_prompt("Premise.", "Hypothesis")
        == "Premise.\nQuestion: Hypothesis. True, False, or Neither?\nAnswer:"
    )
