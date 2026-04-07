# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import pytest

import evalution
from evalution.engines.base import LoglikelihoodOutput

mc_taco_module = importlib.import_module("evalution.benchmarks.mc_taco")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 2
        assert requests[0].context == (
            "The store opens at 9 a.m. every weekday.\n"
            "Question: Is the store open at 10 a.m.?\n"
            "Answer: yes\n"
            "Plausible:"
        )
        assert requests[0].continuation == " no"
        assert requests[1].continuation == " yes"
        return [
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
        ]


def test_mc_taco_scores_binary_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "sentence": "The store opens at 9 a.m. every weekday.",
                "question": "Is the store open at 10 a.m.?",
                "answer": "yes",
                "label": "yes",
                "category": "Typical Time",
            }
        ]
    )
    monkeypatch.setattr(mc_taco_module, "_load_mc_taco_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.mc_taco(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "mc_taco"
    assert result.metrics["acc,ll"] == 1.0
    assert result.metrics["acc,ll_avg"] == 1.0
    assert result.metrics["f1,ll_boolean"] == pytest.approx(1.0)
    assert result.metrics["f1,ll_avg_boolean"] == pytest.approx(1.0)
    assert result.metadata["dataset_path"] == "CogComp/mc_taco"
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "yes"
    assert sample.prediction == "yes"
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["category"] == "Typical Time"


def test_mc_taco_loader_reads_raw_tsv_via_csv_builder() -> None:
    dataset = mc_taco_module._load_mc_taco_dataset(
        "CogComp/mc_taco",
        split="validation",
        stream=False,
    )

    first = dataset[0]
    assert set(first) == {"sentence", "question", "answer", "label", "category"}
    assert isinstance(first["sentence"], str)
    assert isinstance(first["label"], str)
