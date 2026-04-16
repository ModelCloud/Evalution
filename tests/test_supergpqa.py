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
supergpqa_module = importlib.import_module("evalution.benchmarks.supergpqa")


class FakeLoglikelihoodSession:
    """Provide the fake loglikelihood session helper used by the surrounding tests."""

    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for the fake SuperGPQA session."""
        assert batch_size == 9
        return [
            LoglikelihoodOutput(
                logprob=0.0 if request.continuation.strip() == "I" else -10.0,
                is_greedy=True,
                token_count=1,
            )
            for request in requests
        ]


def test_supergpqa_scores_label_selected_by_loglikelihood(monkeypatch) -> None:
    """Verify SuperGPQA scores the labeled choice selected by loglikelihood."""
    dataset = Dataset.from_list(
        [
            {
                "uuid": "supergpqa-1",
                "question": "Which option is correct?",
                "options": [
                    "Alpha",
                    "Bravo",
                    "Charlie",
                    "Delta",
                    "Echo",
                    "Foxtrot",
                    "Golf",
                    "Hotel",
                    "India",
                ],
                "answer": "India",
                "answer_letter": "I",
                "discipline": "Science",
                "field": "Physics",
                "subfield": "Mechanics",
                "difficulty": "hard",
                "is_calculation": False,
            }
        ]
    )
    monkeypatch.setattr(supergpqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.supergpqa(max_rows=1, batch_size=9).evaluate(FakeLoglikelihoodSession())

    assert result.name == "supergpqa"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata == {
        "dataset_path": "m-a-p/SuperGPQA",
        "dataset_name": None,
        "split": "train",
        "order": "native",
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
    }
    sample = result.samples[0]
    assert sample.prompt.startswith("Question: Which option is correct?")
    assert "\nI. India\nAnswer:" in sample.prompt
    assert sample.target == "I"
    assert sample.prediction == "I"
    assert sample.metadata == {
        "uuid": "supergpqa-1",
        "raw_choices": [
            "Alpha",
            "Bravo",
            "Charlie",
            "Delta",
            "Echo",
            "Foxtrot",
            "Golf",
            "Hotel",
            "India",
        ],
        "answer_label": "I",
        "answer_text": "India",
        "discipline": "Science",
        "field": "Physics",
        "subfield": "Mechanics",
        "difficulty": "hard",
        "is_calculation": False,
        "choice_logprobs": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0],
        "choice_logprobs_norm": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0],
    }
