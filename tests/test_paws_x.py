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

paws_x_module = importlib.import_module("evalution.benchmarks.paws_x")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 5
        assert len(requests) == 6
        assert requests[0].context == (
            "Sentence 1: Alpha moved to Paris.\n"
            "Sentence 2: Alpha relocated to Paris.\n"
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
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.3, is_greedy=True, token_count=1),
        ]


def test_paws_x_scores_accuracy_and_positive_class_f1(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": 4,
                "sentence1": "Alpha moved to Paris.",
                "sentence2": "Alpha relocated to Paris.",
                "label": 1,
            },
            {
                "id": 5,
                "sentence1": "The train left early.",
                "sentence2": "The train arrived late.",
                "label": 0,
            },
            {
                "id": 6,
                "sentence1": "The chef cooked dinner.",
                "sentence2": "Dinner was prepared by the chef.",
                "label": 1,
            },
        ]
    )
    monkeypatch.setattr(paws_x_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.paws_x_en(max_rows=3, batch_size=5).evaluate(FakeSession())

    assert result.name == "paws_x_en"
    assert result.metrics["acc,ll"] == pytest.approx(2 / 3)
    assert result.metrics["acc,ll_avg"] == pytest.approx(2 / 3)
    assert result.metrics["f1,ll_boolean"] == pytest.approx(0.8)
    assert result.metrics["f1,ll_avg_boolean"] == pytest.approx(0.8)
    assert result.metadata["dataset_path"] == "paws-x"
    assert result.metadata["dataset_name"] == "en"
    assert len(result.samples) == 3

    first_sample = result.samples[0]
    assert first_sample.target == "yes"
    assert first_sample.prediction == "yes"
    assert first_sample.metadata["id"] == 4
    assert first_sample.metadata["language"] == "en"


def test_paws_x_prompt_helper_formats_paraphrase_question() -> None:
    assert (
        paws_x_module._paws_x_prompt("Sentence A.", "Sentence B.")
        == "Sentence 1: Sentence A.\nSentence 2: Sentence B.\nQuestion: Do both sentences mean the same thing?\nAnswer:"
    )


def test_paws_x_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported paws-x language"):
        paws_x_module.paws_x(language="it")
