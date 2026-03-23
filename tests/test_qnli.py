# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

qnli_module = importlib.import_module("evalution.benchmarks.qnli")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 3
        assert len(requests) == 4
        assert requests[0].context == (
            "What came into force after the new constitution was herald?\n"
            "As of that day, the new constitution heralding the Second Republic came into force.\n"
            "Question: Does this response answer the question?\n"
            "Answer:"
        )
        assert requests[0].continuation == " yes"
        assert requests[1].continuation == " no"
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
        ]


def test_qnli_scores_answer_relevance_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "What came into force after the new constitution was herald?",
                "sentence": "As of that day, the new constitution heralding the Second Republic came into force.",
                "label": 0,
                "idx": 0,
            },
            {
                "question": "What is the first major city in the stream of the Rhine?",
                "sentence": "The most important tributaries in this area are the Ill below of Strasbourg, the Neckar in Mannheim and the Main across from Mainz.",
                "label": 1,
                "idx": 1,
            },
        ]
    )
    monkeypatch.setattr(qnli_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.qnli(max_rows=2, batch_size=3).evaluate(FakeSession())

    assert result.name == "qnli"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "nyu-mll/glue"
    assert result.metadata["dataset_name"] == "qnli"
    assert len(result.samples) == 2

    first_sample = result.samples[0]
    assert first_sample.target == "yes"
    assert first_sample.prediction == "yes"
    assert first_sample.metadata["idx"] == 0

    second_sample = result.samples[1]
    assert second_sample.target == "no"
    assert second_sample.prediction == "no"
    assert second_sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }


def test_qnli_prompt_helper_formats_question_answer_judgment() -> None:
    assert (
        qnli_module._qnli_prompt("Who won?", "The home team won.")
        == "Who won?\nThe home team won.\nQuestion: Does this response answer the question?\nAnswer:"
    )
