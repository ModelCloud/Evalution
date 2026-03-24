# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

multirc_module = importlib.import_module("evalution.benchmarks.multirc")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 5
        assert len(requests) == 2
        assert requests[0].context == (
            "Tom planted tomatoes in April.\n"
            "Question: What did Tom plant?\n"
            "Answer:"
        )
        assert requests[0].continuation == " tomatoes\nIs the answer correct? yes"
        assert requests[1].continuation == " tomatoes\nIs the answer correct? no"
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=6),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=6),
        ]


def test_multirc_scores_binary_answer_validation(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "paragraph": "Tom planted tomatoes in April.",
                "question": "What did Tom plant?",
                "answer": "tomatoes",
                "label": 0,
                "idx": {"paragraph": 1, "question": 2, "answer": 3},
            }
        ]
    )
    monkeypatch.setattr(multirc_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.multirc(max_rows=1, batch_size=5).evaluate(FakeSession())

    assert result.name == "multirc"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "super_glue"
    assert result.metadata["dataset_name"] == "multirc"
    assert result.metadata["split"] == "validation"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "tomatoes\nIs the answer correct? yes"
    assert sample.prediction == "tomatoes\nIs the answer correct? yes"
    assert sample.metadata["paragraph"] == "Tom planted tomatoes in April."
    assert sample.metadata["question"] == "What did Tom plant?"
    assert sample.metadata["answer"] == "tomatoes"
    assert sample.metadata["idx"] == {"paragraph": 1, "question": 2, "answer": 3}


def test_multirc_prompt_matches_upstream_shape() -> None:
    doc = {
        "paragraph": "A short passage.",
        "question": "Is this short?",
    }
    assert multirc_module._multirc_prompt(doc) == "A short passage.\nQuestion: Is this short?\nAnswer:"
