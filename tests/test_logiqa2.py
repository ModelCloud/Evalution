# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib
import json

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
logiqa2_module = importlib.import_module("evalution.benchmarks.logiqa2")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "Passage: Rain fell all night and the roads were flooded by sunrise.\n"
            "Question: Why did commuters arrive late?\n"
            "Choices:\n"
            "A. The roads were closed by flooding.\n"
            "B. Everyone woke up early.\n"
            "C. The buses ran ahead of schedule.\n"
            "D. The sun dried the streets instantly.\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [
            " The roads were closed by flooding.",
            " Everyone woke up early.",
            " The buses ran ahead of schedule.",
            " The sun dried the streets instantly.",
        ]
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=7),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=7),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=7),
        ]


def test_logiqa2_scores_multiple_choice_reasoning(monkeypatch) -> None:
    """Verify logiqa2 scores multiple choice reasoning. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "text": json.dumps(
                    {
                        "id": 7,
                        "answer": 0,
                        "text": "Rain fell all night and the roads were flooded by sunrise.",
                        "question": "Why did commuters arrive late?",
                        "options": [
                            "The roads were closed by flooding.",
                            "Everyone woke up early.",
                            "The buses ran ahead of schedule.",
                            "The sun dried the streets instantly.",
                        ],
                        "type": {"Causal Reasoning": True},
                    }
                )
            }
        ]
    )
    monkeypatch.setattr(logiqa2_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.logiqa2(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "logiqa2"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "datatune/LogiQA2.0"
    assert result.metadata["dataset_name"] is None
    sample = result.samples[0]
    assert sample.target == "The roads were closed by flooding."
    assert sample.prediction == "The roads were closed by flooding."
    assert sample.metadata["question_id"] == 7
    assert sample.metadata["question_type"] == ["Causal Reasoning"]
