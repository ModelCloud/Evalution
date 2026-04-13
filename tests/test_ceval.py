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

# Keep shared test fixtures and expectations explicit at module scope.
ceval_module = importlib.import_module("evalution.benchmarks.ceval")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "下列关于税法基本原则的表述中，不正确的是____。\n"
            "A. 税收法定原则包括税收要件法定原则和税务合法性原则\n"
            "B. 税收公平原则源于法律上的平等性原则\n"
            "C. 税收效率原则包含经济效率和行政效率两个方面\n"
            "D. 税务机关按法定程序依法征税，可以自由做出减征、停征或免征税款的决定\n"
            "答案："
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


def test_ceval_scores_four_way_multiple_choice(monkeypatch) -> None:
    """Verify ceval scores four way multiple choice. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": 0,
                "question": "下列关于税法基本原则的表述中，不正确的是____。",
                "A": "税收法定原则包括税收要件法定原则和税务合法性原则",
                "B": "税收公平原则源于法律上的平等性原则",
                "C": "税收效率原则包含经济效率和行政效率两个方面",
                "D": "税务机关按法定程序依法征税，可以自由做出减征、停征或免征税款的决定",
                "answer": "D",
                "explanation": "",
            }
        ]
    )
    monkeypatch.setattr(ceval_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.ceval(subset="accountant", max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "ceval_accountant"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "ceval/ceval-exam"
    assert result.metadata["dataset_name"] == "accountant"
    assert result.metadata["split"] == "val"
    sample = result.samples[0]
    assert sample.target == "D"
    assert sample.prediction == "D"
    assert sample.metadata["subset"] == "accountant"
    assert sample.metadata["answer_label"] == "D"
    assert sample.metadata["raw_choices"][0].startswith("税收法定原则")


def test_ceval_rejects_unknown_subset() -> None:
    """Verify ceval rejects unknown subset."""
    with pytest.raises(ValueError, match="unsupported ceval subset"):
        evalution.benchmarks.ceval(subset="unknown_subset")
