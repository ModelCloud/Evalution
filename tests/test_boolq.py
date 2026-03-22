# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

boolq_module = importlib.import_module("evalution.suites.boolq")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 2
        assert requests[0].context == "Water freezes at 0 C.\nQuestion: Does water freeze at 0 C?\nAnswer:"
        assert requests[0].continuation == " no"
        assert requests[1].continuation == " yes"
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


def test_boolq_scores_boolean_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "Does water freeze at 0 C",
                "passage": "Water freezes at 0 C.",
                "idx": 7,
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(boolq_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.boolq(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "boolq"
    assert result.metrics == {
        "accuracy,loglikelihood": 1.0,
        "accuracy,loglikelihood_norm": 1.0,
    }
    assert result.metadata["dataset_path"] == "super_glue"
    assert result.metadata["dataset_name"] == "boolq"
    assert result.metadata["split"] == "validation"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Water freezes at 0 C.\nQuestion: Does water freeze at 0 C?\nAnswer:"
    assert sample.target == "yes"
    assert sample.prediction == "yes"
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["idx"] == 7


def test_boolq_question_formatter_adds_question_mark_once() -> None:
    assert boolq_module._format_boolq_question("Is snow cold") == "Is snow cold?"
    assert boolq_module._format_boolq_question("Is snow cold?") == "Is snow cold?"
