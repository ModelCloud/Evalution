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

agieval_module = importlib.import_module("evalution.benchmarks.agieval")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 5
        assert requests[0].context == (
            "Question: A car is being driven toward a tower. The angle changes from 45 degrees to 60 degrees in 10 minutes. "
            "How much more time until the car reaches the base?\n"
            "A. 5(√3 + 1)\n"
            "B. 6(√3 + √2)\n"
            "C. 7(√3 – 1)\n"
            "D. 8(√3 – 2)\n"
            "E. None of these\n\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D", " E"]
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
        ]


def test_agieval_scores_single_answer_multiple_choice_rows(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "passage": None,
                "question": (
                    "A car is being driven toward a tower. The angle changes from 45 degrees to 60 degrees in 10 "
                    "minutes. How much more time until the car reaches the base?"
                ),
                "options": [
                    "(A)5(√3 + 1)",
                    "(B)6(√3 + √2)",
                    "(C)7(√3 – 1)",
                    "(D)8(√3 – 2)",
                    "(E)None of these",
                ],
                "label": "A",
                "other": {"solution": "worked solution"},
                "explanation": None,
            }
        ]
    )
    monkeypatch.setattr(agieval_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.agieval(subset="aqua-rat", max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "agieval_aqua_rat"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "RUCAIBox/AGIEval"
    assert result.metadata["dataset_name"] == "aqua-rat"
    assert result.metadata["split"] == "test"
    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.metadata["subset"] == "aqua-rat"
    assert sample.metadata["question"].startswith("A car is being driven")
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D", "E"]
    assert sample.metadata["raw_choices"][-1] == "None of these"


def test_agieval_slugify_subset_name() -> None:
    assert agieval_module._slugify_subset_name("aqua-rat") == "aqua_rat"
    assert agieval_module._slugify_subset_name("logiqa-en") == "logiqa_en"


def test_agieval_rejects_unsupported_subset() -> None:
    with pytest.raises(ValueError, match="unsupported agieval subset"):
        evalution.benchmarks.agieval(subset="math")
