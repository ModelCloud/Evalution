# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

cola_module = importlib.import_module("evalution.benchmarks.cola")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 6
        assert requests[0].context == (
            "Our friends won't buy this analysis, let alone the next one we propose.\n"
            "Question: Does this sentence make sense?\n"
            "Answer:"
        )
        assert requests[0].continuation == " no"
        assert requests[1].continuation == " yes"
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=False, token_count=1),
        ]


def test_cola_scores_accuracy_and_mcc(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "sentence": "Our friends won't buy this analysis, let alone the next one we propose.",
                "label": 1,
                "idx": 0,
            },
            {
                "sentence": "One more pseudo generalization and I'm giving up.",
                "label": 0,
                "idx": 1,
            },
            {
                "sentence": "The author laughed the editor.",
                "label": 1,
                "idx": 2,
            },
        ]
    )
    monkeypatch.setattr(cola_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.cola(max_rows=3, batch_size=4).evaluate(FakeSession())

    assert result.name == "cola"
    assert result.metrics["acc,ll"] == pytest.approx(2 / 3)
    assert result.metrics["acc,ll_avg"] == pytest.approx(2 / 3)
    assert result.metrics["mcc,ll"] == pytest.approx(0.5)
    assert result.metrics["mcc,ll_avg"] == pytest.approx(0.5)
    assert result.metadata["dataset_path"] == "nyu-mll/glue"
    assert result.metadata["dataset_name"] == "cola"
    assert len(result.samples) == 3

    first_sample = result.samples[0]
    assert first_sample.target == "yes"
    assert first_sample.prediction == "yes"
    assert first_sample.metadata["idx"] == 0

    third_sample = result.samples[2]
    assert third_sample.target == "yes"
    assert third_sample.prediction == "no"
    assert third_sample.extracted == {
        "gold_index": "1",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }


def test_cola_prompt_helper_formats_acceptability_question() -> None:
    assert (
        cola_module._cola_prompt("Colorless green ideas sleep furiously.")
        == "Colorless green ideas sleep furiously.\nQuestion: Does this sentence make sense?\nAnswer:"
    )
