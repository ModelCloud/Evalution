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

qqp_module = importlib.import_module("evalution.benchmarks.qqp")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 6
        assert requests[0].context == (
            "Question 1: How do I learn machine learning?\n"
            "Question 2: What is the best way to study machine learning?\n"
            "Question: Do both questions ask the same thing?\n"
            "Answer:"
        )
        assert requests[0].continuation == " no"
        assert requests[1].continuation == " yes"
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
        ]


def test_qqp_scores_accuracy_and_positive_class_f1(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question1": "How do I learn machine learning?",
                "question2": "What is the best way to study machine learning?",
                "label": 1,
                "idx": 10,
            },
            {
                "question1": "How can I improve my English speaking skill?",
                "question2": "Why is my cat afraid of water?",
                "label": 0,
                "idx": 11,
            },
            {
                "question1": "How do I become a data scientist?",
                "question2": "What should I eat for dinner tonight?",
                "label": 0,
                "idx": 12,
            },
        ]
    )
    monkeypatch.setattr(qqp_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.qqp(max_rows=3, batch_size=4).evaluate(FakeSession())

    assert result.name == "qqp"
    assert result.metrics["acc,ll"] == pytest.approx(2 / 3)
    assert result.metrics["acc,ll_avg"] == pytest.approx(2 / 3)
    assert result.metrics["f1,ll_boolean"] == pytest.approx(2 / 3)
    assert result.metrics["f1,ll_avg_boolean"] == pytest.approx(2 / 3)
    assert result.metadata["dataset_path"] == "nyu-mll/glue"
    assert result.metadata["dataset_name"] == "qqp"
    assert len(result.samples) == 3

    first_sample = result.samples[0]
    assert first_sample.target == "yes"
    assert first_sample.prediction == "yes"
    assert first_sample.metadata["idx"] == 10

    third_sample = result.samples[2]
    assert third_sample.target == "no"
    assert third_sample.prediction == "yes"
    assert third_sample.extracted == {
        "gold_index": "0",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }


def test_qqp_prompt_helper_formats_duplicate_question_prompt() -> None:
    assert (
        qqp_module._qqp_prompt("Question A?", "Question B?")
        == "Question 1: Question A?\nQuestion 2: Question B?\nQuestion: Do both questions ask the same thing?\nAnswer:"
    )
