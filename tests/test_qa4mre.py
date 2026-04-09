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

qa4mre_module = importlib.import_module("evalution.benchmarks.qa4mre")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 5
        assert requests[0].context.startswith("Annie Lennox Why I am an HIVAIDS activist")
        assert requests[0].continuation == " the imprisonment of Nelson Mandela at Robben Island"
        assert requests[4].continuation == " Nelson Mandela's conference to the world press"
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=8),
            LoglikelihoodOutput(logprob=-2.5, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=4),
            LoglikelihoodOutput(logprob=-2.2, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=6),
        ]


def test_qa4mre_scores_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "topic_id": "1",
                "topic_name": "AIDS",
                "test_id": "1",
                "document_id": "1",
                "document_str": "Annie Lennox Why I am an HIVAIDS activist",
                "question_id": "1",
                "question_str": "What event caused Annie Lennox to commit herself to the fight against AIDS?",
                "answer_options": {
                    "answer_id": ["1", "2", "3", "4", "5"],
                    "answer_str": [
                        "the imprisonment of Nelson Mandela at Robben Island",
                        "the closing ceremony of Nelson Mandela's Foundation",
                        "the meeting with Youssou N'Dour",
                        "the racial segregation in South Africa",
                        "Nelson Mandela's conference to the world press",
                    ],
                },
                "correct_answer_id": "5",
            }
        ]
    )
    monkeypatch.setattr(qa4mre_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.qa4mre_2011(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "qa4mre_2011"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "qa4mre"
    assert result.metadata["dataset_name"] == "2011.main.EN"
    assert result.metadata["split"] == "train"
    sample = result.samples[0]
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target == "Nelson Mandela's conference to the world press"
    assert sample.prediction == sample.target
    assert sample.metadata["year"] == "2011"
    assert sample.metadata["question_id"] == "1"
    assert len(sample.metadata["choice_texts"]) == 5


def test_qa4mre_rejects_unknown_year() -> None:
    with pytest.raises(ValueError, match="unsupported qa4mre year"):
        qa4mre_module.QA4MRE(year="2014")
