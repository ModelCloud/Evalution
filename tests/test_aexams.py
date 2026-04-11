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

# Keep shared test fixtures and expectations explicit at module scope.
aexams_module = importlib.import_module("evalution.benchmarks.aexams")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "قم بالإجابة على مايلي في مجال العلوم الحيوية\n\n"
            "ما الوحدة الأساسية للحياة؟\n"
            "A. الخلية\n"
            "B. النسيج\n"
            "C. العضو\n"
            "D. الجهاز\n"
            "الجواب:"
        )
        assert requests[0].continuation == " A"
        assert requests[3].continuation == " D"
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
        ]


def test_aexams_scores_labeled_subject_multiple_choice(monkeypatch) -> None:
    """Verify aexams scores labeled subject multiple choice. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "ما الوحدة الأساسية للحياة؟",
                "A": "الخلية",
                "B": "النسيج",
                "C": "العضو",
                "D": "الجهاز",
                "answer": "A",
            }
        ]
    )
    monkeypatch.setattr(aexams_module, "_load_aexams_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.aexams_biology(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "aexams_biology"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "Hennara/aexams"
    assert result.metadata["dataset_name"] == "Biology"
    assert result.metadata["split"] == "test"

    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.metadata["subject"] == "biology"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_texts"] == ["الخلية", "النسيج", "العضو", "الجهاز"]


def test_aexams_validates_subject_and_prompt_helper() -> None:
    """Verify aexams validates subject and prompt helper."""
    assert aexams_module._aexams_prompt("مقدمة", "سؤال؟", ["أ", "ب", "ج", "د"]).endswith("الجواب:")
    try:
        aexams_module.AEXAMS(subject="history")
    except ValueError as exc:
        assert "unsupported aexams subject" in str(exc)
    else:
        raise AssertionError("expected unsupported subject to raise")
