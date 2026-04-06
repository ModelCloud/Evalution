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

mrpc_module = importlib.import_module("evalution.benchmarks.mrpc")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 5
        assert len(requests) == 6
        assert requests[0].context == (
            "Sentence 1: He said the foodservice pie business doesn't fit the company long-term growth strategy.\n"
            "Sentence 2: The foodservice pie business does not fit our long-term growth strategy.\n"
            "Question: Do both sentences mean the same thing?\n"
            "Answer:"
        )
        assert requests[0].continuation == " no"
        assert requests[1].continuation == " yes"
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.3, is_greedy=True, token_count=1),
        ]


def test_mrpc_scores_accuracy_and_positive_class_f1(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "sentence1": "He said the foodservice pie business doesn't fit the company long-term growth strategy.",
                "sentence2": "The foodservice pie business does not fit our long-term growth strategy.",
                "label": 1,
                "idx": 9,
            },
            {
                "sentence1": "Magnarelli said Racicot hated the Iraqi regime and looked forward to using his long years of training in the war.",
                "sentence2": "His wife said he was 100 percent behind George Bush and looked forward to using his years of training in the war.",
                "label": 0,
                "idx": 18,
            },
            {
                "sentence1": "The dollar was at 116.92 yen against the yen, flat on the session.",
                "sentence2": "The dollar was at 116.78 yen, virtually flat on the session.",
                "label": 1,
                "idx": 25,
            },
        ]
    )
    monkeypatch.setattr(mrpc_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.mrpc(max_rows=3, batch_size=5).evaluate(FakeSession())

    assert result.name == "mrpc"
    assert result.metrics["acc,ll"] == pytest.approx(2 / 3)
    assert result.metrics["acc,ll_avg"] == pytest.approx(2 / 3)
    assert result.metrics["f1,ll_boolean"] == pytest.approx(0.8)
    assert result.metrics["f1,ll_avg_boolean"] == pytest.approx(0.8)
    assert result.metadata["dataset_path"] == "nyu-mll/glue"
    assert result.metadata["dataset_name"] == "mrpc"
    assert len(result.samples) == 3

    first_sample = result.samples[0]
    assert first_sample.target == "yes"
    assert first_sample.prediction == "yes"
    assert first_sample.metadata["idx"] == 9

    second_sample = result.samples[1]
    assert second_sample.target == "no"
    assert second_sample.prediction == "yes"
    assert second_sample.extracted == {
        "gold_index": "0",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }


def test_mrpc_prompt_helper_formats_paraphrase_question() -> None:
    assert (
        mrpc_module._mrpc_prompt("Sentence A.", "Sentence B.")
        == "Sentence 1: Sentence A.\nSentence 2: Sentence B.\nQuestion: Do both sentences mean the same thing?\nAnswer:"
    )
