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
darijammlu_module = importlib.import_module("evalution.benchmarks.darijammlu")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "This is a DarijaMMLU multiple-choice question about المحاسبة.\n\n"
            "Question: فرض محاسبي كيقول بلي الشركة عندها كيان قانوني مستقل.\n"
            "A. الاستمرارية\n"
            "B. الشخصية المعنوية\n"
            "C. الفترية\n"
            "D. القياس النقدي\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=1),
        ]


def test_darijammlu_scores_multiple_choice_subject_subset(monkeypatch) -> None:
    """Verify darijammlu scores multiple choice subject subset. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "فرض محاسبي كيقول بلي الشركة عندها كيان قانوني مستقل.",
                "context": "",
                "choices": ["الاستمرارية", "الشخصية المعنوية", "الفترية", "القياس النقدي"],
                "answer": 1,
                "subject": "accounting",
                "subject_darija": "المحاسبة",
                "source": "arabic_mmlu",
            }
        ]
    )
    monkeypatch.setattr(darijammlu_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.darijammlu_accounting(max_rows=1, batch_size=8).evaluate(
        FakeSession()
    )

    assert result.name == "darijammlu_accounting"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "MBZUAI-Paris/DarijaMMLU"
    assert result.metadata["dataset_name"] == "accounting"
    assert result.metadata["split"] == "test"
    sample = result.samples[0]
    assert sample.target == "B"
    assert sample.prediction == "B"
    assert sample.metadata["subset"] == "accounting"
    assert sample.metadata["subject_darija"] == "المحاسبة"
    assert sample.metadata["raw_choices"] == [
        "الاستمرارية",
        "الشخصية المعنوية",
        "الفترية",
        "القياس النقدي",
    ]


def test_darijammlu_rejects_unknown_subset() -> None:
    """Verify darijammlu rejects unknown subset."""
    with pytest.raises(ValueError, match="unsupported darijammlu subset"):
        evalution.benchmarks.darijammlu(subset="unknown_subject")
