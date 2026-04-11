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
egymmlu_module = importlib.import_module("evalution.benchmarks.egymmlu")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "This is a EgyMMLU multiple-choice question about المحاسبة.\n\n"
            "Question: فرع محاسبي بيهدف للتحقق من صحة وسلامة المعلومات المالية.\n"
            "A. المحاسبة الحكومية\n"
            "B. المحاسبة الدولية\n"
            "C. المراجعة\n"
            "D. المحاسبة الاجتماعية\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
        ]


def test_egymmlu_scores_multiple_choice_subject_subset(monkeypatch) -> None:
    """Verify egymmlu scores multiple choice subject subset. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "فرع محاسبي بيهدف للتحقق من صحة وسلامة المعلومات المالية.",
                "context": "",
                "choices": [
                    "المحاسبة الحكومية",
                    "المحاسبة الدولية",
                    "المراجعة",
                    "المحاسبة الاجتماعية",
                ],
                "subject": "accounting",
                "egy_subject": "المحاسبة",
                "answer": 2,
                "split": "test",
                "source": "ar_mmlu",
                "__index_level_0__": 0,
            }
        ]
    )
    monkeypatch.setattr(egymmlu_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.egymmlu_accounting(max_rows=1, batch_size=8).evaluate(
        FakeSession()
    )

    assert result.name == "egymmlu_accounting"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "UBC-NLP/EgyMMLU"
    assert result.metadata["dataset_name"] == "accounting"
    sample = result.samples[0]
    assert sample.target == "C"
    assert sample.prediction == "C"
    assert sample.metadata["subset"] == "accounting"
    assert sample.metadata["egy_subject"] == "المحاسبة"


def test_egymmlu_rejects_unknown_subset() -> None:
    """Verify egymmlu rejects unknown subset."""
    with pytest.raises(ValueError, match="unsupported egymmlu subset"):
        evalution.benchmarks.egymmlu(subset="unknown_subject")
