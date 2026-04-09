# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

wnli_module = importlib.import_module("evalution.benchmarks.wnli")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 4
        assert requests[0].context == (
            "The drain is clogged with hair. It has to be cleaned.\n"
            "Question: The hair has to be cleaned. True or False?\n"
            "Answer:"
        )
        assert requests[0].continuation == " False"
        assert requests[1].continuation == " True"
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
        ]


def test_wnli_scores_true_false_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "sentence1": "The drain is clogged with hair. It has to be cleaned.",
                "sentence2": "The hair has to be cleaned.",
                "label": 0,
                "idx": 0,
            },
            {
                "sentence1": "Jane knocked on Susan's door but she did not answer.",
                "sentence2": "Susan did not answer.",
                "label": 1,
                "idx": 1,
            },
        ]
    )
    monkeypatch.setattr(wnli_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.wnli(max_rows=2, batch_size=4).evaluate(FakeSession())

    assert result.name == "wnli"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "nyu-mll/glue"
    assert result.metadata["dataset_name"] == "wnli"
    assert len(result.samples) == 2

    first_sample = result.samples[0]
    assert first_sample.target == "False"
    assert first_sample.prediction == "False"
    assert first_sample.metadata["idx"] == 0

    second_sample = result.samples[1]
    assert second_sample.target == "True"
    assert second_sample.prediction == "True"
    assert second_sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }


def test_wnli_prompt_helper_formats_true_false_question() -> None:
    assert (
        wnli_module._wnli_prompt("Sentence one.", "Sentence two.")
        == "Sentence one.\nQuestion: Sentence two. True or False?\nAnswer:"
    )
