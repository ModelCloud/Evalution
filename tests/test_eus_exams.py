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

eus_exams_module = importlib.import_module("evalution.benchmarks.eus_exams")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "Question: De acuerdo con la Constitución española, es un valor superior del ordenamiento jurídico español:\n"
            "A. La soberanía nacional.\n"
            "B. El estado social.\n"
            "C. La igualdad.\n"
            "D. La democracia.\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=1),
        ]


def test_eus_exams_scores_exam_subset(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "1",
                "question": "De acuerdo con la Constitución española, es un valor superior del ordenamiento jurídico español:",
                "candidates": [
                    "La soberanía nacional.",
                    "El estado social.",
                    "La igualdad.",
                    "La democracia.",
                ],
                "answer": 2,
                "link": "https://example.test/question/1",
            }
        ]
    )
    monkeypatch.setattr(eus_exams_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.eus_exams_es_ejadministrativo(
        max_rows=1,
        batch_size=8,
    ).evaluate(FakeSession())

    assert result.name == "eus_exams_es_ejadministrativo"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "HiTZ/EusExams"
    assert result.metadata["dataset_name"] == "es_ejadministrativo"
    sample = result.samples[0]
    assert sample.target == "C"
    assert sample.prediction == "C"
    assert sample.metadata["language"] == "es"
    assert sample.metadata["question_id"] == "1"


def test_eus_exams_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported eus_exams subset"):
        evalution.benchmarks.eus_exams(subset="unknown_subset")
