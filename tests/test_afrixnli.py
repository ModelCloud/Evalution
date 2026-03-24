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

afrixnli_module = importlib.import_module("evalution.benchmarks.afrixnli")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 6
        assert requests[0].context == (
            "Premise: Well, I wasn't even thinking about that, but I was so frustrated, and, I ended up talking to him again.\n"
            "Hypothesis: I havent spoken to him again.\n"
            "Question: What is the relationship between the premise and hypothesis: entailment, neutral, or contradiction?\n"
            "Answer:"
        )
        assert [request.continuation for request in requests[:3]] == [
            " entailment",
            " neutral",
            " contradiction",
        ]
        return [
            LoglikelihoodOutput(logprob=-2.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
        ]


def test_afrixnli_scores_three_way_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "premise": "Well, I wasn't even thinking about that, but I was so frustrated, and, I ended up talking to him again.",
                "hypothesis": "I havent spoken to him again.",
                "label": 2,
            },
            {
                "premise": "Well, I wasn't even thinking about that, but I was so frustrated, and, I ended up talking to him again.",
                "hypothesis": "I was so upset that I just started talking to him again.",
                "label": 0,
            },
        ]
    )
    monkeypatch.setattr(afrixnli_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.afrixnli(language="eng", max_rows=2, batch_size=4).evaluate(FakeSession())

    assert result.name == "afrixnli_eng"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "masakhane/afrixnli"
    assert result.metadata["dataset_name"] == "eng"
    assert result.metadata["split"] == "test"

    first_sample = result.samples[0]
    assert first_sample.target == "contradiction"
    assert first_sample.prediction == "contradiction"
    assert first_sample.metadata["language"] == "eng"
    assert first_sample.metadata["choice_texts"] == ["entailment", "neutral", "contradiction"]

    second_sample = result.samples[1]
    assert second_sample.target == "entailment"
    assert second_sample.prediction == "entailment"


def test_afrixnli_prompt_helper_formats_nli_prompt() -> None:
    assert (
        afrixnli_module._afrixnli_prompt("Premise text", "Hypothesis text")
        == "Premise: Premise text\nHypothesis: Hypothesis text\nQuestion: What is the relationship between the premise and hypothesis: entailment, neutral, or contradiction?\nAnswer:"
    )


def test_afrixnli_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported afrixnli language"):
        evalution.benchmarks.afrixnli(language="zzz")


def test_afrixnli_rejects_dataset_name_mismatch() -> None:
    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.afrixnli(language="eng", dataset_name="fra")
