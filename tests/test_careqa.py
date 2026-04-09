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

careqa_module = importlib.import_module("evalution.benchmarks.careqa")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "Question: In relation to iron metabolism and its control mediated by hepcidin, it is true that:\n"
            "A. HIF increases the expression of hepcidin.\n"
            "B. Increased serum iron stimulates hepcidin synthesis in the liver and downregulates ferroportin.\n"
            "C. Hepcidin inactivates DMT1.\n"
            "D. HFE mutations increase hepcidin production.\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
        ]


def test_careqa_scores_closed_ended_question(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "exam_id": 26,
                "question": "In relation to iron metabolism and its control mediated by hepcidin, it is true that:",
                "op1": "HIF increases the expression of hepcidin.",
                "op2": "Increased serum iron stimulates hepcidin synthesis in the liver and downregulates ferroportin.",
                "op3": "Hepcidin inactivates DMT1.",
                "op4": "HFE mutations increase hepcidin production.",
                "cop": 2,
                "year": 2024,
                "category": "Medicine",
                "unique_id": "sample-1",
            }
        ]
    )
    monkeypatch.setattr(careqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.careqa_en(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "careqa_en"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "HPAI-BSC/CareQA"
    assert result.metadata["dataset_name"] == "CareQA_en"
    sample = result.samples[0]
    assert sample.target == "B"
    assert sample.prediction == "B"
    assert sample.metadata["language"] == "en"
    assert sample.metadata["category"] == "Medicine"


def test_careqa_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported careqa language"):
        evalution.benchmarks.careqa(language="fr")
